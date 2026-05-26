# Contract -- Slack digest regression (Cycle 71)

**Cycle:** 71 (2026-05-26)
**Trigger:** Operator-flagged via Slack scrape: `#ford-approvals` Morning + Evening digests have been broken for ~5 days. Three independent symptoms with three independent root causes -- all internal, all single-file fixes.

## Research gate

- Researcher `a5582c57b590e17be`, tier=moderate (internal-only carve-out per the spawn prompt's hard-constraints clause).
- 9 internal files inspected with 23 file:line anchors documented.
- `gate_passed: true` (zero external sources required; documented gate-pass rationale).
- Brief: `handoff/current/research_brief_phase_slack_digest.md`.

## N* delta

- **B (Burn) primary:** the Slack bot is the operator's primary off-cockpit touchpoint. It has shown nonsense data for 5 days. Restoring digest accuracy = direct operator-attention saved per day.
- **R (Risk) secondary:** "0.0/10 -- Buy/Sell/Hold" recommendations have been pushed to the operator's morning attention for 5 days. Misleading signal.
- **P:** marginal.

## Three independent root causes + minimal fixes

### Fix 1 -- autonomous_loop.py:1293 final_score key drift (P0)

Orchestrator stores synthesis score under `final_weighted_score` (orchestrator.py:2001). Autonomous loop reads under wrong key `final_score` at autonomous_loop.py:1293 -- defaults to 0. The 0 propagates through `_persist_analysis` -> `bq.save_report` -> `analysis_results.final_score=0`. Manual `/analyze` path uses the correct key (tasks/analysis.py:208), so manual analyses were correct; autonomous-loop rows have been wrong since first clean cycle (2026-05-22 commit `29ab0ff6`).

**Fix:**
```python
"final_score": synthesis.get("final_weighted_score",
                             synthesis.get("final_score", 0)),
```
Single-line, defensive (keeps the legacy key as a fallback).

### Fix 2 -- formatters.py:320-323 + :365-368 nested envelope (P0)

`/api/paper-trading/portfolio` returns `{"portfolio": {nav-fields-here}, "positions": [...], "sector_breakdown": {...}}` (nested envelope per `paper_trading.py:224-228`). Formatters read top-level `portfolio_data.get("total_pnl")` + `.get("total_pnl_pct")` -- both miss, both default to 0. Plus `paper_portfolio` BQ row has no `total_pnl` column -- only `total_nav` + `total_pnl_pct`. Introduced by commit `55241e3a` (phase-25.G, 2026-05-12).

**Fix (both morning + evening):**
```python
p = (portfolio_data.get("portfolio")
     if isinstance(portfolio_data.get("portfolio"), dict)
     else portfolio_data)
total_nav = float(p.get("total_nav") or 0.0)
starting = float(p.get("starting_capital") or 0.0)
total_pnl = total_nav - starting
total_return = float(p.get("total_pnl_pct") or 0.0)
```

Defensive: works whether the caller passes the envelope or the inner dict (forward-compat with future refactors).

### Fix 3 -- get_paper_trades date filter (P1)

`bigquery_client.py:674-683::get_paper_trades` has no date filter -- returns the latest N rows regardless of date. The companion `get_paper_trades_in_window` (line 928) is the date-aware helper but is not wired into the digest. Result: same 10 rows shown as "Today's Trades" for 9 consecutive days.

**Fix (Option A from brief -- minimal new API surface):** Add optional `since_iso` param to `get_paper_trades`; add optional `since_today=true` query param to `GET /api/paper-trading/trades`; scheduler passes `since_today=true` for the digest call.

This preserves the existing helper signature (defaults preserved) + lets future consumers opt in.

## Plan steps

1. `backend/services/autonomous_loop.py:1293` -- defensive-key fix on `final_score`.
2. `backend/slack_bot/formatters.py` -- unwrap envelope in morning + evening formatters. Compute `total_pnl = total_nav - starting_capital`.
3. `backend/db/bigquery_client.py:get_paper_trades` -- add optional `since_iso: Optional[str]` param.
4. `backend/api/paper_trading.py:get_trades` -- add optional `since_today: bool = False` query param; pass through to `get_paper_trades`.
5. `backend/slack_bot/scheduler.py:324` -- evening digest URL adds `?limit=10&since_today=true`.
6. Vitest-equivalent: pytest. Add at least one test per fix.
7. Verify all gates + manual smoke (curl the endpoints).

## Files planned

NEW:
- `backend/tests/test_phase_slack_digest_71.py` -- 3+ pytest cases covering the 3 fixes (final_score key drift; envelope unwrap; since_iso filter).

MODIFIED:
- `backend/services/autonomous_loop.py` (+1 line semantic change at :1293)
- `backend/slack_bot/formatters.py` (~10 lines at :319-323 + :364-368)
- `backend/db/bigquery_client.py` (~4 lines at :674-683 for the new param + SQL conditional)
- `backend/api/paper_trading.py` (~5 lines at :233-246 for the new query param)
- `backend/slack_bot/scheduler.py` (1-line URL change at :324)

ZERO frontend changes. ZERO new deps.

## /goal integration-gate plan

| # | Gate | Plan |
|---|------|------|
| 1 | pytest >= 614 backend | Add 3+ new cases; expect 617+. |
| 2 | TS build + ast.parse green | ast.parse on each changed .py. |
| 3 | Flag default OFF | The new `since_today` param defaults False (off). |
| 4 | BQ migrations idempotent | N/A (no schema changes). |
| 5 | New env vars documented | N/A. |
| 6 | N* delta declared | DONE. |
| 7 | Zero emojis | Grep. |
| 8 | ASCII loggers | Sweep. |
| 9 | SSOT | `get_paper_trades` extended (NOT forked); existing callers unaffected. |
| 10 | log first / flip last | Yes -- no masterplan step flip; this is a UX/regression fix cycle. |

## Verification

After all 3 fixes land + backend restart + the next autonomous cycle runs (or the next digest fires):
- `curl -s http://localhost:8000/api/reports/?limit=5 | jq '.[].final_score'` -- expect non-zero for at least one row after the fix.
- `curl -s http://localhost:8000/api/paper-trading/portfolio | jq '.portfolio.total_nav,.portfolio.total_pnl_pct'` -- non-zero values exist + digest will read them via the envelope unwrap.
- `curl -s 'http://localhost:8000/api/paper-trading/trades?since_today=true' | jq '.count'` -- equal to actual today-only count.
- Next Slack digest (morning at 14:00 CEST or evening at 23:00 CEST) shows real NAV + non-zero scores + accurate today-only trade list.

## Backfill (deferred, owner-gated)

Existing `analysis_results.final_score=0` rows from 2026-05-22+ are permanently 0 unless backfilled from `full_report_json.final_synthesis.final_weighted_score`. The brief flags this as a one-shot follow-up; not part of this cycle's minimal fix.

## Sign-off

Authored AFTER researcher gate_passed=true. 3 independent regressions with 3 independent single-file fixes. Goal-aligned: this is a production-readiness regression on the operator's primary off-cockpit touchpoint.
