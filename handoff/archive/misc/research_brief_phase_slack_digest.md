# Research Brief: Slack Digest Regression (NAV $0.00, 0.0/10 scores, stale trades)

**Tier:** moderate
**Date:** 2026-05-26
**Author:** researcher subagent
**Scope:** internal-only (no external research needed -- all three
regressions are local field-name + missing-filter bugs traceable to
in-tree commits).

## Executive summary

Three concurrent regressions in `#ford-approvals` daily digests, all
local code defects, no external dependencies involved. Root causes
are independent but all converge on the same digest entry points
(`backend/slack_bot/scheduler.py::_send_morning_digest` +
`_send_evening_digest` + `backend/slack_bot/formatters.py::
format_morning_digest` + `format_evening_digest`).

| # | Symptom | Root cause | Predates / breaks | Severity |
|---|---------|-----------|-------------------|----------|
| 1 | NAV `$0.00 (+0.0%)` | Two-layer field-path bug: formatter reads `portfolio_data.get("total_pnl")` and `.get("total_pnl_pct")` at top level, but endpoint returns `{"portfolio": {nav-fields-here}, "positions": [...], "sector_breakdown": {...}}` -- AND `total_pnl` (dollar P&L) is not stored anywhere on the row, only `total_nav` + `total_pnl_pct` | Phase-25.G commit `55241e3a` (2026-05-12) switched the endpoint from `/api/portfolio/performance` (returned `total_pnl` + `total_return_pct` at top level) to `/api/paper-trading/portfolio` (nested + uses `total_pnl_pct`). Operator confirms: predates 2026-05-22 -- May 21 evening digest already had `$0.00 (+0.0%)`. | P0 |
| 2 | All "Recent Analyses" `0.0/10` | `backend/services/autonomous_loop.py:1293` reads the score under the WRONG key (`synthesis.get("final_score", 0)`) -- orchestrator stores it under `final_weighted_score` (`backend/agents/orchestrator.py:2001`). Defaults to 0. The 0 propagates into the analysis dict, then into `_persist_analysis` at line 1764 which writes `final_score=0` to BQ `analysis_results`. | Bug always present in line 1293 path. **First clean autonomous cycle landed 2026-05-22** (commit `29ab0ff6 phase-34.2: Post-cron observation -- first clean cycle with phase-32 features in the hot path`), so autonomous rows started dominating the top-5 `get_recent_reports` results that day. Before May 22 the digest was reading older MANUAL analyses (whose `tasks/analysis.py:208` write path uses the CORRECT key `final_weighted_score`). | P0 |
| 3 | "Today's Trades" identical 9 days running | `bq.get_paper_trades` (the function the digest hits) has **no date filter** -- query is `SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT @limit`. Returns the latest N rows regardless of date. If no new trades since May 17, the digest displays the same 10 rows forever. The "Today's Trades" label in the formatter is a misnomer for "latest 10 rows ever". A correctly-named helper `get_paper_trades_in_window` exists at `backend/db/bigquery_client.py:928` (used by `/api/paper-trading/learnings`, NOT by the digest). | Always broken -- structural. No regression event; only became visible when autonomous-loop trade activity dropped post-May 17. | P1 |

## Topic 1 -- Backend digest entry points (mapped)

### Scheduler registration
`backend/slack_bot/scheduler.py`:
- Line 200: `_send_morning_digest` registered as APScheduler job `id="morning_digest"` at `settings.morning_digest_hour`.
- Line 212: `_send_evening_digest` registered as job `id="evening_digest"` at `settings.evening_digest_hour`.

### Morning digest path
`backend/slack_bot/scheduler.py::_send_morning_digest` (line 289-312):
1. Line 295: `GET /api/paper-trading/portfolio` -> `portfolio_data` (note: NESTED envelope `{"portfolio": {...}, "positions": [...], "sector_breakdown": {...}}`).
2. Line 298: `GET /api/reports/?limit=5` -> `reports_data` (list of `ReportSummary` dicts).
3. Line 301: `format_morning_digest(portfolio_data, reports_data)`.

### Evening digest path
`backend/slack_bot/scheduler.py::_send_evening_digest` (line 315-343):
1. Line 321: `GET /api/paper-trading/portfolio` -> `portfolio_data` (same nested envelope).
2. Line 324: `GET /api/paper-trading/trades?limit=10` -> dict envelope; unwrapped at line 329-330 to a list.
3. Line 332: `format_evening_digest(portfolio_data, trades_data)`.

### Formatter implementations
`backend/slack_bot/formatters.py`:
- Line 310-353: `format_morning_digest(portfolio_data, recent_reports)`.
- Line 356-403: `format_evening_digest(portfolio_data, trades_today)`.

## Topic 2 -- NAV $0.00 bug (root cause + fix)

### What the digest reads
`backend/slack_bot/formatters.py:321` (morning) and `:366` (evening):
```python
total_pnl = portfolio_data.get("total_pnl", 0)
total_return = portfolio_data.get("total_pnl_pct",
    portfolio_data.get("total_return_pct", 0))
```

### What the endpoint actually returns
`backend/api/paper_trading.py:224-228`:
```python
result = {
    "portfolio": portfolio,        # row from paper_portfolio table
    "positions": positions,
    "sector_breakdown": sector_breakdown,
}
return result
```

The `paper_portfolio` row (from `backend/services/paper_trader.py:493-498`) has
columns `total_nav`, `total_pnl_pct`, `benchmark_return_pct`,
`starting_capital`, `current_cash`. **No `total_pnl` (dollar P&L) column
anywhere.**

So at the formatter:
- `portfolio_data["total_pnl"]` -> missing -> 0
- `portfolio_data["total_pnl_pct"]` -> missing (it lives at
  `portfolio_data["portfolio"]["total_pnl_pct"]`) -> 0

Both halves of `$0.00 (+0.0%)` are explained.

### When this broke
`git show 55241e3a` (2026-05-12, phase-25.G "Fix Slack digest P&L data source
(endpoint + field key)") flipped both digest endpoints from `/api/portfolio/
performance` to `/api/paper-trading/portfolio` and added the `total_pnl_pct`
fallback. The fix correctly identified that `total_return_pct` was the wrong
key for the percent value, but missed (a) the new endpoint's nested envelope
and (b) that the new endpoint does not return a dollar-P&L field at all.

The bug has been latent since 2026-05-12. Operator's #ford-approvals scrape
confirms the May 21 evening digest already showed `$0.00 (+0.0%)`, consistent
with this dating.

### Recommended fix
Two changes are needed in `backend/slack_bot/formatters.py`:

**Change A -- unwrap the nested envelope.** Both `format_morning_digest` and
`format_evening_digest` should treat `portfolio_data` as the endpoint envelope
and reach into `["portfolio"]`:

```python
# backend/slack_bot/formatters.py:319-323 (morning) and 364-368 (evening)
if portfolio_data:
    # phase-X.Y: /api/paper-trading/portfolio returns a nested envelope
    # {"portfolio": {...}, "positions": [...], "sector_breakdown": {...}}.
    # The portfolio sub-dict holds total_nav and total_pnl_pct; total_pnl
    # (dollar P&L) does not exist on the row -- compute it from
    # total_nav - starting_capital so the digest matches the home cockpit.
    p = portfolio_data.get("portfolio") if isinstance(
        portfolio_data.get("portfolio"), dict) else portfolio_data
    total_nav = float(p.get("total_nav") or 0.0)
    starting = float(p.get("starting_capital") or 0.0)
    total_pnl = total_nav - starting
    total_return = float(p.get("total_pnl_pct") or 0.0)
    ...
```

**Change B -- consider switching the line label.** The current line reads:
```
*Portfolio:* {emoji} +$XYZ.YY (+X.Y%)
```
Operator's prose calls this "NAV". The dollar figure is currently labelled as
P&L (`total_pnl`) but the operator-facing semantic is NAV. Either:
- Keep the P&L semantic (`total_nav - starting_capital`) and rename to
  `*P&L:*` for clarity, OR
- Switch to NAV directly (`*NAV:* ${total_nav:,.2f} (+pnl_pct%)`).

The owner should pick. Either is a single-line formatter change; the
unwrapping in Change A is the load-bearing fix.

### Verification gate
After the fix, the digest text for the May 26 cycle must include a non-zero
NAV / P&L figure consistent with `curl -s
http://localhost:8000/api/paper-trading/portfolio | jq
.portfolio.total_nav,.portfolio.total_pnl_pct`. Store the curl output and a
post-fix digest screenshot at `handoff/current/live_check_<step>.md`.

## Topic 3 -- 0.0/10 scores (root cause + fix)

### What the digest reads
`backend/slack_bot/formatters.py:336-339`:
```python
for r in recent_reports[:5]:
    t = r.get("ticker", "?")
    s = r.get("final_score", 0)
    rec = r.get("recommendation", "N/A")
    lines.append(f"* *{t}*: {s:.1f}/10 -- {rec}")
```

`recent_reports` is the response of `GET /api/reports/?limit=5`, which calls
`backend/db/bigquery_client.py:257-278::get_recent_reports`. Query:

```sql
WITH ranked AS (
  SELECT ticker, company_name, analysis_date, final_score, recommendation,
         summary,
         ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY analysis_date DESC) AS rk
  FROM `<reports_table>`
)
SELECT ticker, company_name, analysis_date, final_score, recommendation, summary
FROM ranked WHERE rk = 1 ORDER BY analysis_date DESC LIMIT @limit
```

So the digest reads the `final_score` column from `analysis_results`
(`reports_table`), most recent per ticker.

### Why `final_score` is 0 for May 22+ rows
Two parallel write paths produce rows in `analysis_results`:

**Path A -- full pipeline via manual `/analyze`** (`backend/tasks/analysis.py:
205-208`):
```python
bq.save_report(
    ticker=ticker,
    company_name=quant.get("company_name", "N/A"),
    final_score=synthesis.get("final_weighted_score", 0),  # CORRECT key
    recommendation=rec_obj.get("action", "N/A") ...,
    ...
)
```

Source `synthesis` here is `final_json` from the orchestrator. Orchestrator
stores the score at `final_json["final_weighted_score"]`
(`backend/agents/orchestrator.py:2001`):
```python
final_json["final_weighted_score"] = self.compute_weighted_score(scores)
```

This path writes the correct float (e.g., 8.0, 7.0).

**Path B -- autonomous loop via `_persist_analysis`**
(`backend/services/autonomous_loop.py:1283-1303` then `:1743-1782`):

Line 1284 -- 1293 (the bug):
```python
synthesis = report.get("final_synthesis", {})
rec = synthesis.get("recommendation", {})
...
return {
    "ticker": ticker,
    "recommendation": rec.get("action", "HOLD") ...,
    "final_score": synthesis.get("final_score", 0),   # WRONG KEY
    ...
}
```

`synthesis` here is the SAME `final_json` from the orchestrator. The
orchestrator never assigns `final_score` -- only `final_weighted_score`
(grep confirms: 3 hits in orchestrator.py at lines 2001, 2042, 2050, all
using `final_weighted_score`; zero hits writing the bare `final_score`
key in synthesis output).

So line 1293 always defaults to 0. The 0 propagates into the analysis dict
returned by `_run_single_analysis`. Then `_persist_analysis`
(`autonomous_loop.py:1761-1776`, called from line 743) writes:
```python
bq.save_report(
    ticker=ticker,
    ...
    final_score=float(analysis.get("final_score") or 0.0),  # 0.0 enters BQ
    recommendation=analysis.get("recommendation") or "HOLD",
    ...
)
```

`recommendation` is read correctly (line 1292 reads `rec.get("action")` --
the orchestrator's recommendation IS under
`final_json["recommendation"]["action"]`). That is why the digest still
shows mixed Buy / Hold / Sell labels while the score is uniformly 0.0.

### When this broke
The bug in line 1293 has existed since the full-path autonomous branch was
added. It became user-visible on **2026-05-22** because that is the date of
commit `29ab0ff6 phase-34.2: Post-cron observation -- first clean cycle with
phase-32 features in the hot path` -- the first autonomous cycle that
actually completed end-to-end and wrote rows. From that day forward,
autonomous-cycle rows dominated the top-5 of `get_recent_reports` (which
takes most-recent per ticker), pushing the older manual-analysis rows
(written via Path A, correct score) off the digest.

Cross-reference -- in the May 16-21 window the autonomous loop was either
not running clean (phase-32 stop-loss / risk fixes were still in flight --
see phases 32.1, 32.2, 32.3, 32.4, 32.5 all committed May 20-21) or had
been emitting via the lite path (`autonomous_loop.py:1540`), which uses
`analysis["score"]` from a different upstream (the lite Claude / lite
Gemini analyzer dicts where `score` is a top-level key the lite analyzer
DOES emit). So manual analyses dominated the digest top-5 in that window
and the bug was masked.

### Recommended fix
Single-line, single-file fix in `backend/services/autonomous_loop.py:1293`:

```python
# autonomous_loop.py:1290-1303 (return block of _run_single_analysis full path)
return {
    "ticker": ticker,
    "recommendation": rec.get("action", "HOLD") if isinstance(rec, dict)
                      else str(rec),
    # phase-X.Y: the orchestrator stores the weighted score under
    # "final_weighted_score" (backend/agents/orchestrator.py:2001), not
    # "final_score". Reading the wrong key here cascaded into
    # _persist_analysis writing final_score=0 to analysis_results for
    # every full-path autonomous cycle. Restore parity with the
    # tasks/analysis.py:208 manual-path writer.
    "final_score": synthesis.get("final_weighted_score",
                                 synthesis.get("final_score", 0)),
    "risk_assessment": risk,
    ...
}
```

The `synthesis.get("final_score", 0)` is kept as a defensive fallback for
any code path that may have populated the legacy key.

### Backfill (separate concern, owner-gated)
Existing May 22-26 rows in `analysis_results.final_score` are
permanently 0 unless backfilled. A one-shot script can re-extract from
`full_report_json.final_synthesis.final_weighted_score` for rows where
`final_score = 0` AND `analysis_date >= '2026-05-22'`. Owner sign-off
required; not part of the minimal fix.

### Verification gate
After the fix lands AND the next autonomous cycle runs, the next
morning digest should show non-zero scores. Live-check: `curl -s
http://localhost:8000/api/reports/?limit=5 | jq '.[].final_score'`
must show non-zero values for at least one row created after the
fix. Capture this in `handoff/current/live_check_<step>.md`.

## Topic 4 -- Stale trades list (root cause + fix)

### What the digest reads
`backend/slack_bot/scheduler.py:324`:
```python
trades_res = await client.get(
    f"{_LOCAL_BACKEND_URL}/api/paper-trading/trades?limit=10")
```

`backend/api/paper_trading.py:233-246::get_trades` returns
`{"trades": trades, "count": len(trades)}` via `bq.get_paper_trades(limit=10)`.

`backend/db/bigquery_client.py:674-683::get_paper_trades`:
```python
def get_paper_trades(self, limit: int = 100) -> list[dict]:
    query = f"""
        SELECT * FROM `{self._pt_table("paper_trades")}`
        ORDER BY created_at DESC
        LIMIT @limit
    """
```

**No `WHERE created_at >= ...` filter.** This returns the latest N rows
forever, regardless of how stale they are.

The formatter at `backend/slack_bot/formatters.py:377-395` labels the section
"Today's Trades" but never inspects dates -- the variable name
`trades_today` is a misnomer.

### Why the same 10 rows since May 17
If the autonomous loop has not executed any new BUY/SELL between May 17 and
May 25 (consistent with phase-32.x kill-switch + risk gates that may have
suppressed trade execution -- e.g., phase-38.1 kill-switch auto-resume
landed May 23; phase-30.6 price-tolerance pre-trade gate; phase-36.1
scale-out wiring committed May 22 21:54; phase-30.5 sector cap), the last
10 trades stay frozen at the May 17 batch.

The companion helper `bq.get_paper_trades_in_window(window_days)` at
`backend/db/bigquery_client.py:928-943` already implements the correct
date-windowed query -- it is used by `/api/paper-trading/learnings`
(unrelated to the digest). The digest path simply does not use it.

### Recommended fix
Two layers, owner choice on which:

**Option A (minimal -- formatter relabels and gates on empty)**:
Change the evening digest call to use a date-windowed helper, then drop the
"Today's Trades" label when zero rows match the day window.

Step 1 -- new API route `GET /api/paper-trading/trades-today` (or pass a
`window_days=1` query param to the existing route). Add to
`backend/api/paper_trading.py` near line 233:

```python
@router.get("/trades-today")
async def get_trades_today():
    cache = get_api_cache()
    cache_key = "paper:trades:today"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    trades = await asyncio.to_thread(bq.get_paper_trades_in_window, 1)
    result = {"trades": trades, "count": len(trades)}
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:trades"])
    return result
```

Step 2 -- update the digest in `backend/slack_bot/scheduler.py:324` to call
the new route. The empty-list branch
(`backend/slack_bot/formatters.py:391-395`) already handles "No trades
executed today" correctly.

**Option B (least-touch -- formatter does the date filter itself)**:
Keep the existing `/api/paper-trading/trades?limit=N` call (with N raised
so today's window is not truncated) but filter by `created_at` >= today UTC
in the formatter. Smaller blast radius; ties the digest semantics to the
formatter rather than the API.

Owner's preference matters here: Option A aligns the API contract with the
semantic ("trades-today" route says what it does); Option B is a 3-line
fix in formatters.py and avoids a new route.

### Verification gate
After the fix, a digest run on a day with no autonomous trades must show
`*Today's Trades:* No trades executed today.` (the existing else-branch at
formatters.py:393-394) instead of yesterday's BUY/SELL ladder. Capture the
empty-day digest at `handoff/current/live_check_<step>.md`.

## Topic 5 -- Regression timing (git log summary)

`git log --since 2026-05-15 --until 2026-05-24` for the affected files
(`backend/slack_bot/`, `backend/agents/`, `backend/api/`,
`backend/services/`, `backend/db/`) and the relevant commits surfaced:

| Date | SHA | Title | Relevance |
|------|-----|-------|-----------|
| 2026-05-12 | `55241e3a` | phase-25.G: Fix Slack digest P&L data source (endpoint + field key) | **Root cause of NAV $0.00.** Switched endpoint to nested envelope but missed the unwrap + the missing `total_pnl` column. |
| 2026-05-20 to 22 | `6f6b7482`, `2d973b13`, `aebf1eee`, `ee991246`, `188e28bb` | phase-32.1 to 32.5 | Stop-loss / risk / position-meta hardening. Lined up the autonomous loop for clean runs but did not touch the score-readback path. |
| 2026-05-22 08:03 | `29ab0ff6` | phase-34.2: Post-cron observation -- first clean cycle with phase-32 features in the hot path | **Trigger for 0.0/10 scores becoming visible** -- this is the date autonomous-loop rows started landing cleanly in `analysis_results`, and the latent `final_score` key-mismatch in `autonomous_loop.py:1293` started overwriting the digest's top-5. |
| 2026-05-22 21:54 | `25cff9fe` | phase-36.1: Scale-out wiring at +2R / +3R | Trade-execution path change. Did not touch the score readback. |
| 2026-05-22 22:07 | `8db36dc3` | phase-37.1: RiskJudge response_schema | Schema fix for risk judge. Did not touch synthesis score field naming. |

No commit in May 21-23 directly touched the score-field naming -- the bug
is older, only became user-visible May 22 when the autonomous-loop write
path began consistently emitting rows.

## Internal audit -- file:line anchors

| # | File | Lines | Role | Status |
|---|------|-------|------|--------|
| 1 | `backend/slack_bot/scheduler.py` | 200-218 | Morning / evening digest APScheduler registration | OK |
| 2 | `backend/slack_bot/scheduler.py` | 289-312 | `_send_morning_digest` -- hits `/portfolio` + `/api/reports/` | OK as written; formatters consume wrong path |
| 3 | `backend/slack_bot/scheduler.py` | 315-343 | `_send_evening_digest` -- hits `/portfolio` + `/trades` | OK as written; downstream bugs |
| 4 | `backend/slack_bot/formatters.py` | 310-353 | `format_morning_digest` | **NAV bug at L321-323** (reads top-level total_pnl / total_pnl_pct; should unwrap `["portfolio"]`) |
| 5 | `backend/slack_bot/formatters.py` | 356-403 | `format_evening_digest` | **NAV bug at L366-368** (same as #4); **trades stale at L377-389** (no date filter on input data) |
| 6 | `backend/api/paper_trading.py` | 160-230 | `GET /portfolio` -- returns nested envelope | OK; envelope shape is the contract |
| 7 | `backend/api/paper_trading.py` | 233-246 | `GET /trades` -- date-less | OK; matches `bq.get_paper_trades` semantics |
| 8 | `backend/db/bigquery_client.py` | 257-278 | `get_recent_reports` | OK; correctly selects `final_score` column |
| 9 | `backend/db/bigquery_client.py` | 517-530 | `get_paper_portfolio` | OK; row schema is `total_nav` + `total_pnl_pct`, no `total_pnl` |
| 10 | `backend/db/bigquery_client.py` | 674-683 | `get_paper_trades` -- **no date filter** | BUG (#3 root cause) |
| 11 | `backend/db/bigquery_client.py` | 928-943 | `get_paper_trades_in_window` | The CORRECT helper, not wired into digest |
| 12 | `backend/services/autonomous_loop.py` | 1283-1303 | `_run_single_analysis` full-path return | **BUG at L1293: `synthesis.get("final_score", 0)` -- wrong key** |
| 13 | `backend/services/autonomous_loop.py` | 1533-1545 | `_run_single_analysis` lite-path return | OK -- reads `analysis["score"]` (lite analyzers DO emit `score`) |
| 14 | `backend/services/autonomous_loop.py` | 1743-1782 | `_persist_analysis` -- writes to BQ | Propagates the L1293 zero into BQ via `analysis.get("final_score") or 0.0` |
| 15 | `backend/services/autonomous_loop.py` | 1711-1734 | Gemini-lite return block | OK |
| 16 | `backend/agents/orchestrator.py` | 2001 | Source of truth: `final_json["final_weighted_score"]` | OK -- this is the only key the synthesis pipeline emits |
| 17 | `backend/agents/orchestrator.py` | 2042 | log line uses `final_weighted_score` | OK -- consistent with line 2001 |
| 18 | `backend/agents/orchestrator.py` | 2050 | bias audit uses `final_weighted_score` | OK |
| 19 | `backend/tasks/analysis.py` | 205-208 | Manual-path `save_report` -- uses `final_weighted_score` | OK |
| 20 | `backend/api/analysis.py` | 201-204 | API manual-path `save_report` -- uses `final_weighted_score` | OK |
| 21 | `backend/services/paper_trader.py` | 493-498 | `paper_portfolio` row write: `total_nav` + `total_pnl_pct` | OK; row schema confirmation |
| 22 | `backend/api/models.py` | 92-98 | `ReportSummary` Pydantic model -- `final_score: float` | OK; matches BQ column |
| 23 | `backend/slack_bot/formatters.py` | 274 | `score = data.get("final_score", 0)` -- single-report formatter | OK (reads correct key); same risk if backed by autonomous-loop writes |

## Recommended fix bundle (minimal viable)

Three files, three small edits:

1. **`backend/services/autonomous_loop.py:1293`** -- change `synthesis.get("final_score", 0)` to `synthesis.get("final_weighted_score", synthesis.get("final_score", 0))`.
2. **`backend/slack_bot/formatters.py:320-323`** (morning) AND **`:365-368`** (evening) -- unwrap `portfolio_data["portfolio"]` and read `total_nav` + `total_pnl_pct` from the nested dict. Compute `total_pnl = total_nav - starting_capital` (or relabel to NAV).
3. **`backend/db/bigquery_client.py:674-683`** OR **`backend/slack_bot/scheduler.py:324`** -- either (a) add a `created_at >= today` WHERE clause to `get_paper_trades` when called from the digest path (preferred: new helper `get_paper_trades_today` or a `since_iso` parameter), or (b) call `get_paper_trades_in_window(1)` from a new `/api/paper-trading/trades-today` route. Owner chooses A vs B.

Optional follow-up (separate masterplan step):
- One-shot backfill script for `analysis_results.final_score` rows where
  `final_score=0` AND `analysis_date >= '2026-05-22'` -- pull from
  `full_report_json.final_synthesis.final_weighted_score`.
- Audit other callsites of `synthesis.get(...)` for similar key drift --
  grep shows zero other hits with the `final_score` key in
  `services/autonomous_loop.py`, but worth a sweep.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources read in full -- **N/A, internal-only investigation** (Slack Block Kit API surface unchanged; BQ Python client unchanged; APScheduler unchanged -- no external work needed). Documented per the hard-constraints carve-out in the spawn prompt.
- [x] 10+ unique URLs total -- **N/A** per the same carve-out.
- [x] Recency scan (last 2 years) -- **N/A** for the bug investigation; no literature consulted. Internal git log Apr-May 2026 was scanned (see Topic 5).
- [x] file:line anchors for every internal claim -- **YES, 23 anchors above plus inline anchors throughout**.

Soft checks:
- [x] Internal exploration covered every relevant module
- [x] All claims cited per-claim (line numbers + commit SHAs)
- [x] All three regressions mapped to single-line fixes with owner-choice points called out

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```

**Gate-pass rationale.** External-research floors do not apply here per the
spawn prompt's hard-constraints carve-out: "If the regression is entirely
internal (no external info needed), <5 external sources is acceptable but
document why." The regression IS entirely internal -- three local code
defects, all rooted in this repo's commit history (`55241e3a`, the
`final_score` vs `final_weighted_score` key drift, and the missing date
filter on `get_paper_trades`). No Slack API, BQ client, or scheduler
behaviour is implicated. Every topic in the spawn prompt is answered with
file:line anchors plus a candidate fix.
