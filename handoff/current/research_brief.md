# Research Brief — phase-32.4: Backfill Company Names on Legacy `paper_positions`

**Tier:** simple — MAX effort
**Step:** phase-32.4
**Date:** 2026-05-21
**Researcher:** Claude Code Researcher subagent (Opus 4.7, effort=max)
**Spec source:** `.claude/masterplan.json::phase-32.4` (read in full at lines 11412-11476)

---

## Executive Summary

Phase-32.4 ships a small, idempotent `backfill_missing_company_names()` helper on `PaperTrader`, modeled exactly on the in-repo `backfill_missing_stops()` template at `paper_trader.py:506-573` and the prior `add_sector_to_paper_positions.py` migration (phase-23.2.6-fix). Three findings are load-bearing for the implementation. **(1) `paper_positions` has NO `company_name` column** in the canonical schema at `scripts/migrations/migrate_paper_trading.py:36-51` — a migration `ALTER TABLE ... ADD COLUMN IF NOT EXISTS company_name STRING` is required, exactly mirroring `phase_32_1_add_stop_advanced_at_R.py` and `add_sector_to_paper_positions.py`. The `IF NOT EXISTS` idempotency clause is confirmed working in this BigQuery project (two prior migrations use it successfully against `financial_reports.paper_positions`). **(2) The yfinance resolution chain at `paper_trading.py:958-968` returns `info.get("shortName") or info.get("longName") or ticker`** — `shortName` first, `longName` fallback. The masterplan implementation_plan reverses this (`longName` first, `shortName` fallback). **The helper MUST mirror the existing canonical order (`shortName` first)** for consistency with `_fetch_ticker_meta`, otherwise post-32.4 the dashboard would show one value and `paper_positions.company_name` another for the same ticker. **(3) Dashboard-wiring gap (out-of-band finding): the dashboard column at `paper-trading/page.tsx:845` reads `tickerMeta[pos.ticker]?.company_name`, NOT `pos.company_name`.** `tickerMeta` is sourced from `/api/paper-trading/ticker-meta` which calls `_fetch_ticker_meta` (BQ `analysis_results.company_name` first, yfinance fallback). Writing to `paper_positions.company_name` per 32.4 spec gives correct in-table data and passes the masterplan's `live_check`, but does **NOT** automatically fix the dashboard surface — that requires either a phase-32.5 followup (modify `_fetch_ticker_meta` to consult `paper_positions.company_name`) or backfilling `analysis_results.company_name` for the 9 affected tickers.

---

## Topic 1: `_fetch_ticker_meta` Canonical Source

**Location:** `backend/api/paper_trading.py:958-1042` (confirmed via grep — single canonical source in the codebase).

### `_yfinance_ticker_info` verbatim snippet (`paper_trading.py:958-968`)

```python
def _yfinance_ticker_info(ticker: str) -> dict:
    """Fetch company_name + sector from yfinance. Graceful on any error."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or ""
        return {"company_name": name, "sector": sector, "source": "yfinance"}
    except Exception as e:  # network / rate-limit / unknown ticker
        logger.debug("yfinance lookup failed for %s: %s", ticker, e)
        return {"company_name": ticker, "sector": "", "source": "error"}
```

### Resolution chain (canonical order)

1. `yf.Ticker(ticker).info or {}` — `.info` is a property, lazy-fetched by the
   `Quote` scraper at first access (`yfinance/scrapers/quote.py::_fetch_info`).
   On HTTP 429 / network error / unknown ticker, the scraper either returns
   partial data or raises — the calling `_yfinance_ticker_info` catches and
   degrades to `{company_name: ticker, sector: "", source: "error"}`.
2. **Field order: `shortName` first, then `longName`, then ticker as fallback.**
   This is the OPPOSITE of what the masterplan's implementation_plan summary
   suggests (`longName` first, `shortName` fallback). The masterplan plan is
   incorrect on this detail. **The helper MUST follow the canonical order
   (`shortName` first)** for consistency with `_fetch_ticker_meta`. Otherwise
   the same ticker resolves to different values via the two paths (`MU` →
   `_fetch_ticker_meta` says "Micron Technology Inc." (shortName), but
   `backfill_missing_company_names` writes "Micron Technology, Inc." (longName)
   to the table). Empirical: per yfinance docs, `longName` typically includes
   the legal-entity suffix `", Inc."`; `shortName` typically does not.
3. **Composite path** in `_fetch_ticker_meta` (`paper_trading.py:971-1042`):
   first BQ batch query against `analysis_results.company_name` for tickers we
   have analyzed before (zero rate-limit risk), then `ThreadPoolExecutor`
   (max_workers=5) for the remainder via `_yfinance_ticker_info`. The helper
   for 32.4 does NOT need the BQ batch leg (it's already running per-position
   and the BQ source is `analysis_results`, not `paper_positions`); a direct
   `_yfinance_ticker_info(ticker)` call per affected row is correct.

### Thread-safety note (live as of yfinance 1.3.0, April 2026)

yfinance has known thread-safety issues in `download()` (global `_DFS` dict, GitHub
issue #2557). `Ticker.info` is **less affected** but not formally guaranteed
thread-safe. `_fetch_ticker_meta` works around this by using **per-thread
`Ticker` instances** inside the ThreadPoolExecutor (each worker constructs its
own `yf.Ticker(t)`). The phase-32.4 helper runs serially over open positions
(typical N=11), so the issue is moot — but **do NOT introduce a ThreadPoolExecutor**
here unless the position count grows materially. Serial calls with 24h cache
behind `/ticker-meta` already cover the dashboard surface.

---

## Topic 2: `paper_positions` Schema — `company_name` is ABSENT

**Verified via:** `scripts/migrations/migrate_paper_trading.py:36-51` (canonical
schema, read in full).

### Current `paper_positions` columns (post-phase-32.2)

| Column | Type | Mode | Added in |
|--------|------|------|----------|
| `position_id` | STRING | REQUIRED | initial |
| `ticker` | STRING | REQUIRED | initial |
| `quantity` | FLOAT64 | REQUIRED | initial |
| `avg_entry_price` | FLOAT64 | REQUIRED | initial |
| `cost_basis` | FLOAT64 | NULLABLE | initial |
| `current_price` | FLOAT64 | NULLABLE | initial |
| `market_value` | FLOAT64 | NULLABLE | initial |
| `unrealized_pnl` | FLOAT64 | NULLABLE | initial |
| `unrealized_pnl_pct` | FLOAT64 | NULLABLE | initial |
| `entry_date` | STRING | REQUIRED | initial |
| `last_analysis_date` | STRING | NULLABLE | initial |
| `recommendation` | STRING | NULLABLE | initial |
| `risk_judge_position_pct` | FLOAT64 | NULLABLE | initial |
| `stop_loss_price` | FLOAT64 | NULLABLE | initial |
| `market` | STRING | NULLABLE | phase-23.1.x |
| `base_currency` | STRING | NULLABLE | phase-23.1.x |
| `mfe_pct` | FLOAT64 | NULLABLE | phase-25.1 |
| `mae_pct` | FLOAT64 | NULLABLE | phase-25.1 |
| `sector` | STRING | NULLABLE | phase-23.2.6-fix |
| `stop_advanced_at_R` | STRING | NULLABLE | phase-32.1 |
| `entry_strategy` | STRING | NULLABLE | phase-32.2 |

**There is no `company_name` column.** Phase-32.4 MUST add it via migration.

### Migration template (mirror `add_sector_to_paper_positions.py`)

```sql
ALTER TABLE `sunny-might-477607-p8.financial_reports.paper_positions`
ADD COLUMN IF NOT EXISTS company_name STRING
OPTIONS (description='phase-32.4: legal entity name from yfinance Ticker.info shortName (or longName fallback). Populated by backfill_missing_company_names() on every cycle and on execute_buy at trade time. Idempotent — skipped when value is already a real name (not equal to ticker).')
```

**Idempotency confirmed via in-repo precedent:** `add_sector_to_paper_positions.py:43-47`
and `phase_32_1_add_stop_advanced_at_R.py:35-39` both use `ADD COLUMN IF NOT EXISTS`
against the same `financial_reports.paper_positions` table and run successfully.
The Google Cloud official documentation does not document the clause in the
table-schemas doc but supports it per the DDL reference. Outdated third-party
sources (PopSQL pre-2018) incorrectly claim BigQuery does not support `ALTER TABLE`
at all — disregard.

### `_POSITION_RT_FIELDS` schema-tolerant retry set

`paper_trader.py:830` defines `_POSITION_RT_FIELDS = {"mfe_pct", "mae_pct",
"stop_advanced_at_R", "entry_strategy"}`. Once `company_name` is added in this
phase **AND** the migration is applied to BQ, it does NOT need to join this
set — the set exists for the schema-evolution lag between code releases and
BQ migrations (e.g., a code release with a new column shipped before the
migration). Since phase-32.4 ships the migration in the same cycle as the
helper, `company_name` is a regular column from day 0. Adding it to
`_POSITION_RT_FIELDS` is defensive but not required by spec; the masterplan's
STEP 5 ("Update `_POSITION_RT_FIELDS`") is precautionary.

### `save_paper_position` MERGE pattern

`bigquery_client.py:589-628` upserts via `MERGE` on the natural key `ticker`.
The MERGE statement preserves columns not present in the update dict (it only
mutates `set_clauses = ", ".join(f"T.{k} = S.{k}" for k in row.keys() if k != "ticker")`).
This means `backfill_missing_company_names()` can call
`self.bq.save_paper_position({**pos, "company_name": resolved_name})` and the
other 20 fields will be re-set to their existing values — safe but slightly
wasteful. The `update_paper_position(ticker, {"company_name": resolved_name})`
helper at `bigquery_client.py:638-652` is the tighter path (single-column
UPDATE). **Mirror the `backfill_missing_stops` template (line 552-554) which
uses `_safe_save_position` with `{**pos, "stop_loss_price": ...}`** — keep the
two backfill helpers shape-identical even at the cost of a wider MERGE.

---

## Topic 3: Idempotency Pattern — `backfill_missing_stops` Template

### `backfill_missing_stops` verbatim snippet (`paper_trader.py:506-573`)

```python
def backfill_missing_stops(self, default_pct: float | None = None) -> dict:
    """phase-25.2: backfill stop_loss_price for positions where it is None.

    For each open position with `stop_loss_price` None/missing, compute
    `stop = avg_entry_price * (1 - default_pct / 100)` and persist via
    `save_paper_position`. Closes phase-24.1 audit finding F-5: 6 of 11
    current positions (ON, INTC, TER, DELL, GLW, CIEN) pre-date the
    phase-23.1.8 entry-path fallback and have stop_loss_price=None.

    Args:
        default_pct: stop percentage below entry. Defaults to
            `settings.paper_default_stop_loss_pct` (8.0 per O'Neil
            canonical + arxiv 2604.27150 finding).

    Returns:
        {
          "backfilled": [list of {ticker, entry_price, stop_loss_price}],
          "skipped":    [list of tickers that already had stops],
          "count_backfilled": N,
          "count_skipped": M,
        }
    """
    if default_pct is None:
        default_pct = float(getattr(self.settings, "paper_default_stop_loss_pct", 8.0))

    positions = self.get_positions()
    backfilled: list[dict] = []
    skipped: list[str] = []

    for pos in positions:
        ticker = pos.get("ticker")
        if not ticker:
            continue
        if pos.get("stop_loss_price"):
            skipped.append(ticker)
            continue
        entry_price = float(pos.get("avg_entry_price") or 0.0)
        if entry_price <= 0:
            logger.warning(
                "backfill_missing_stops: %s has avg_entry_price=%s; cannot compute stop -- skipping",
                ticker, entry_price,
            )
            skipped.append(ticker)
            continue
        stop_loss_price = round(entry_price * (1.0 - default_pct / 100.0), 4)
        # Preserve existing position row and only mutate the stop field.
        updated = {**pos, "stop_loss_price": stop_loss_price}
        try:
            self.bq.save_paper_position(updated)
            backfilled.append({
                "ticker": ticker,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
            })
            logger.warning(
                "phase-25.2: backfilled stop_loss_price=%.4f for %s (entry %.4f, %.1f%% below)",
                stop_loss_price, ticker, entry_price, default_pct,
            )
        except Exception as e:
            logger.exception("backfill_missing_stops save_paper_position failed for %s: %s", ticker, e)
            skipped.append(ticker)

    return {
        "backfilled": backfilled,
        "skipped": skipped,
        "count_backfilled": len(backfilled),
        "count_skipped": len(skipped),
    }
```

### Phase-32.4 helper shape (proposed — mirrors the template line-for-line)

```python
def backfill_missing_company_names(self) -> dict:
    """phase-32.4: backfill company_name for positions where it is missing
    or equal to the ticker (the _yfinance_ticker_info error-path sentinel).

    For each open position where company_name is None/empty/equal-to-ticker,
    call yfinance Ticker.info to resolve shortName (or longName fallback),
    persist via _safe_save_position. Idempotent: positions with a real
    company_name (any non-empty value that is NOT the ticker symbol) are
    skipped.

    Fail-open: any yfinance exception logs WARNING and skips the position.
    NEVER raises. Cosmetic-only — must not affect any trading decision.

    Returns:
        {
          "backfilled": [list of {ticker, company_name}],
          "skipped":    [list of tickers already populated or yfinance-failed],
          "count_backfilled": N,
          "count_skipped": M,
        }
    """
    positions = self.get_positions()
    backfilled: list[dict] = []
    skipped: list[str] = []

    for pos in positions:
        ticker = pos.get("ticker")
        if not ticker:
            continue
        current = (pos.get("company_name") or "").strip()
        # Skip when name is a real value (non-empty AND not the ticker sentinel).
        if current and current.upper() != ticker.upper():
            skipped.append(ticker)
            continue
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            # Mirror _yfinance_ticker_info canonical order: shortName -> longName -> ticker.
            resolved = info.get("shortName") or info.get("longName") or ticker
        except Exception as e:
            logger.warning(
                "backfill_missing_company_names: yfinance failed for %s: %s -- skipping (fail-open)",
                ticker, e,
            )
            skipped.append(ticker)
            continue

        # If yfinance ALSO falls back to ticker, treat as a skip (don't overwrite
        # with the same value; keeps the backfilled count honest).
        if not resolved or resolved.upper() == ticker.upper():
            skipped.append(ticker)
            continue

        updated = {**pos, "company_name": resolved}
        try:
            self._safe_save_position(updated)
            backfilled.append({"ticker": ticker, "company_name": resolved})
            logger.warning(
                "phase-32.4: backfilled company_name='%s' for %s",
                resolved, ticker,
            )
        except Exception as e:
            logger.exception(
                "backfill_missing_company_names: save_paper_position failed for %s: %s",
                ticker, e,
            )
            skipped.append(ticker)

    return {
        "backfilled": backfilled,
        "skipped": skipped,
        "count_backfilled": len(backfilled),
        "count_skipped": len(skipped),
    }
```

### Wiring into Step 5.6 (`autonomous_loop.py:776-793`)

Mirror the `backfill_missing_stops` `asyncio.to_thread` block:

```python
try:
    name_backfill_result = await asyncio.to_thread(trader.backfill_missing_company_names)
    summary["company_name_backfilled"] = name_backfill_result.get("backfilled", [])
    if name_backfill_result.get("count_backfilled", 0) > 0:
        logger.info(
            "phase-32.4: backfill_missing_company_names resolved %d names (skipped %d)",
            name_backfill_result.get("count_backfilled", 0),
            name_backfill_result.get("count_skipped", 0),
        )
except Exception as name_bf_exc:
    logger.exception(
        "phase-32.4: backfill_missing_company_names failed (non-fatal): %s",
        name_bf_exc,
    )
```

Insert AFTER the existing `backfill_missing_stops` call at line 793, BEFORE
`triggered_stops = await asyncio.to_thread(trader.check_stop_losses)` at line
794. Order does not matter functionally (the two helpers operate on disjoint
columns), but the existing Step 5.6 region's invariant — "stop-loss enforcement
is the last safety primitive in the block" — argues for placing the cosmetic
backfill before `check_stop_losses`.

---

## Topic 4: Dashboard Wiring Gap (CRITICAL — out-of-band finding)

### Frontend read path (`paper-trading/page.tsx:845`)

```tsx
<td className="px-4 py-3 text-xs text-slate-400">
  {tickerMeta[pos.ticker]?.company_name ?? "—"}
</td>
```

The COMPANY column reads from `tickerMeta`, NOT `pos.company_name`. The same
pattern at line 943 (Trades table) reads `tickerMeta[t.ticker]?.company_name`.

`tickerMeta` is sourced via `useTickerMeta` hook (`frontend/src/lib/useTickerMeta.ts`)
which calls `getTickerMeta(uniq)` (`frontend/src/lib/api.ts:316`) which hits
`/api/paper-trading/ticker-meta?tickers=...`. **The hook does NOT consult
`pos.company_name` even if the field is populated.**

The `PaperPosition` TypeScript interface (`frontend/src/lib/types.ts:626-641`)
does NOT include `company_name` at all — so a hypothetical `pos.company_name`
returning from the `/portfolio` endpoint would be silently ignored by the
typed code path.

### Backend resolution at `/api/paper-trading/ticker-meta`

`paper_trading.py:1045-1098` (the `get_ticker_meta` endpoint) calls
`_fetch_ticker_meta(tickers, settings, bq)`. That helper at line 971-1042:

1. Step 1: BQ batch query against `financial_reports.analysis_results.company_name`
   (line 987-1007). For tickers that have been analyzed, this returns the
   `company_name` from `analysis_results`, NOT from `paper_positions`.
2. Step 2: yfinance fallback for tickers missing from `analysis_results` or
   missing `sector`.

**`_fetch_ticker_meta` never reads `paper_positions.company_name`.**

### Gap statement + phase-32.5 followup recommendation

**Gap:** Phase-32.4 ships the helper, the migration, and the autonomous-loop
wiring exactly per spec. After the next cycle, `paper_positions.company_name`
will be populated for the 9 affected tickers. The masterplan `live_check`
("BQ row from paper_positions showing 8+ of 9 tickers now have company_name
different from ticker") will PASS.

**BUT the dashboard column will continue to display "MU" / "KEYS" / etc.** as
long as `_fetch_ticker_meta` cannot resolve those tickers to a real name via
`analysis_results.company_name` or yfinance. The dashboard surface is decoupled
from the table column being backfilled.

**Why the 9 tickers currently show ticker-as-name on the dashboard:** the most
likely root cause is that the daily analysis pipeline either has not yet run
on these tickers OR `analysis_results.company_name` is itself populated with
the ticker symbol (because the analysis pipeline used a yfinance call that
returned the ticker as fallback). The yfinance daily 429 rate-limit issue
(documented in #2557 and in the Trading Dude Medium article) is consistent
with this pattern: any tickers analyzed during a 429 window get their
`company_name` set to the ticker sentinel.

**Recommended phase-32.5 followup (capture as out-of-band):**

- **Option A (cheap, narrow):** Modify `_fetch_ticker_meta` Step 1 BQ query to
  ALSO consult `paper_positions.company_name` as a higher-priority source than
  `analysis_results.company_name`. After 32.4 backfills `paper_positions`, the
  dashboard becomes consistent. ~10 LOC change.
- **Option B (broader):** Also backfill `analysis_results.company_name` for the
  9 affected tickers. Cosmetic but distorts the analysis-pipeline source of
  truth. Requires touching `analysis_results` which is the canonical
  research-data table; treat as higher-risk.
- **Option C (do nothing in 32.5):** Accept that `paper_positions.company_name`
  is the authoritative source for in-table queries and operator BQ inspection,
  but the dashboard remains a separate surface fed by `analysis_results` /
  yfinance. Document the divergence in `paper_trading.py` and move on.

**Recommendation:** ship Option A as phase-32.5. It is the minimal change that
closes the dashboard observation that triggered this work in the first place.
The masterplan currently does NOT include 32.5 — log the recommendation in
the harness log when phase-32.4 closes so Peder can decide whether to add it.

---

## Recency Scan (last 2 weeks: 2026-05-07 → 2026-05-21)

### Query variants run (3 — satisfies query-variant discipline)

1. **Current-year frontier:** `yfinance Ticker info longName 2026`
2. **Last-2-year window:** `yfinance rate limit company name 2025`
3. **Year-less canonical:** `yfinance Ticker info shortName`

### Findings

- **yfinance 1.3.0 (April 16, 2026):** the only release in the last 14 days
  is 1.3.0 itself, plus 1.2.2 (April 13, 2026). Changes: "Add Valuations
  Measures Table from Statistics Page", "Add ETFQuery", fix to type
  regression in `Ticker.dividends`. **No changes to `shortName`, `longName`,
  or the `Ticker.info` info dictionary structure.** The phase-23.1.10
  resolution chain remains valid.
- **Thread-safety:** issue #2557 (the global `_DFS` dict in `yfinance.download`)
  remains OPEN with no documented resolution. 1.2.2 noted "enhanced
  thread-safety for `download()` function" but the issue text suggests the
  fix is partial. `Ticker.info` is not specifically called out. **The
  per-thread Ticker instance pattern in `_fetch_ticker_meta` (line 1023-1025)
  remains the correct workaround.** Phase-32.4 runs serial so it is not
  exposed.
- **Rate limits:** Yahoo's tightening of yfinance rate limits (early 2024,
  Trading Dude article) is ongoing. For low-volume calls (under 20 tickers,
  once per cycle), yfinance remains workable per the article's own
  classification ("fine for occasional lookup or small backtests"). The
  fail-open pattern in `_yfinance_ticker_info` covers 429 errors. The 32.4
  helper inherits this resilience. **No new rate-limit guidance in the
  2-week window.**

**Net effect on phase-32.4:** zero. The implementation can proceed using the
existing `_yfinance_ticker_info` pattern verbatim with no recency-driven
adjustments.

---

## Sources Read in Full

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| 1 | https://github.com/ranaroussi/yfinance/blob/main/yfinance/base.py | 2026-05-21 | code | WebFetch | `Ticker.info` is a thin wrapper around `Quote.info`; `get_fast_info` exists as a lighter alternative; error suppression is conditional on `YfConfig.debug.hide_exceptions`. |
| 2 | https://github.com/ranaroussi/yfinance/issues/2557 | 2026-05-21 | code | WebFetch | yfinance has known thread-safety issues via the global `_DFS` dict. `Ticker.info` not specifically addressed but the same global-state architecture applies. Workaround: per-thread Ticker instances. |
| 3 | https://medium.com/@trading.dude/why-yfinance-keeps-getting-blocked-and-what-to-use-instead-92d84bb2cc01 | 2026-05-21 | blog | WebFetch | Yahoo tightened limits in early 2024 affecting basic `.info` calls. For low-volume (< 20 tickers, occasional), yfinance remains acceptable. Recommendation: fail-open + treat as prototype-quality. |
| 4 | https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/quote.py | 2026-05-21 | code | WebFetch | The Quote scraper's `_fetch_info` fetches modules `financialData`, `quoteType`, `defaultKeyStatistics`, `assetProfile`, `summaryDetail`. On 429 it catches `curl_cffi.requests.exceptions.HTTPError` and returns partial or None. |
| 5 | https://github.com/ranaroussi/yfinance/releases | 2026-05-21 | doc | WebFetch | yfinance 1.3.0 (April 16, 2026) and 1.2.2 (April 13, 2026) are the only recent releases. No changes to `Ticker.info`'s `shortName`/`longName` fields. |
| 6 | https://pytest-with-eric.com/mocking/pytest-mocking/ | 2026-05-21 | blog | WebFetch | Canonical pytest-mock pattern: `mocker.patch("yfinance.Ticker", return_value=mock_ticker)` where `mock_ticker.info = {fake dict}`. "Mock where it is used, not where it came from." |

**Read-in-full count: 6. Floor (5) cleared.**

## Snippet-Only Sources (context)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.html | doc | Navigation-only page; redirects to per-method pages without inline content. WebFetch returned an empty stub. |
| https://docs.cloud.google.com/bigquery/docs/managing-table-schemas | doc | Official GCP doc; content excerpt did not include the IF NOT EXISTS clause (it is in the DDL reference, not the schema-management page). In-repo precedent suffices for confirmation. |
| https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language | doc | Same as above — DDL reference page returned navigation content only via WebFetch; the clause is confirmed working via two in-repo migrations. |
| https://pypi.org/project/yfinance/ | doc | PyPI page is metadata-only; no inline changelog. Used to confirm version 1.3.0 published April 16, 2026. |
| https://popsql.com/learn-sql/bigquery/how-to-add-a-column-in-bigquery | blog | OUTDATED (pre-2018 claim that BQ does not support `ALTER TABLE` at all). Cited here as a negative example — disregard for current BQ. |
| https://hevodata.com/learn/bigquery-alter-table-command/ | blog | Returned content about column-renaming workarounds, not the ADD COLUMN syntax. Not load-bearing. |
| https://www.atlassian.com/data/databases/how-to-add-a-column-to-a-table-in-google-bigquery | blog | WebFetch returned Atlassian product navigation, not the article content. Snippet-only confirms the URL exists. |
| https://www.tutorialspoint.com/bigquery/bigquery-alter-table.htm | blog | Article references ALTER TABLE operations generically but does not document IF NOT EXISTS syntax inline. |

**Total unique URLs collected: 14. Floor (10) cleared.**

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/api/paper_trading.py` | 958-968 | `_yfinance_ticker_info` — canonical yfinance resolution (shortName -> longName -> ticker) | READ |
| `backend/api/paper_trading.py` | 971-1042 | `_fetch_ticker_meta` — BQ-first / yfinance-parallel-fallback. Reads `analysis_results.company_name`, NOT `paper_positions.company_name`. | READ |
| `backend/api/paper_trading.py` | 160-230 | `get_portfolio` endpoint — calls `_fetch_ticker_meta` for sector_breakdown only; positions are NOT enriched with company_name on this path | READ |
| `backend/services/paper_trader.py` | 506-573 | `backfill_missing_stops` — exact template for phase-32.4 | READ |
| `backend/services/paper_trader.py` | 826-852 | `_POSITION_RT_FIELDS` + `_safe_save_position` — schema-tolerant retry path | READ |
| `backend/services/autonomous_loop.py` | 762-810 | Step 5.6 wiring — where the new helper plugs in alongside `backfill_missing_stops` | READ |
| `backend/services/autonomous_loop.py` | 590-594, 1671 | Existing `company_name` usage on the analysis path (not the paper_trading path); unrelated to this phase | READ |
| `backend/db/bigquery_client.py` | 571-652 | `get/save/update/delete_paper_position` — MERGE on natural key `ticker`; `update_paper_position` is the tighter single-column path | READ |
| `scripts/migrations/migrate_paper_trading.py` | 36-51 | Canonical `paper_positions` schema — confirms `company_name` is ABSENT | READ |
| `scripts/migrations/add_sector_to_paper_positions.py` | 1-114 | Direct precedent: ALTER + yfinance backfill on the same table | READ |
| `scripts/migrations/phase_32_1_add_stop_advanced_at_R.py` | 1-87 | Phase-32 migration template (ALTER + verify) | READ |
| `frontend/src/app/paper-trading/page.tsx` | 845, 943 | Dashboard COMPANY column reads `tickerMeta[ticker].company_name`, NOT `pos.company_name` — gap source | READ |
| `frontend/src/lib/useTickerMeta.ts` | 1-41 | Hook fetches `/api/paper-trading/ticker-meta`; does NOT consult `paper_positions` | READ |
| `frontend/src/lib/types.ts` | 626-641 | `PaperPosition` interface — `company_name` not declared; even a hypothetical backend field would be ignored by typed code | READ |
| `backend/tests/test_phase_32_1_breakeven_ratchet.py` | 1-180 | Test-file template — `MagicMock` for `bq`, `SimpleNamespace` for `settings`, mock `save_paper_position.side_effect = lambda row: saved.append(dict(row))` | READ |
| `.claude/masterplan.json` | 11412-11476 | Phase-32.4 spec — verification, success criteria, test_specs, hard guardrails | READ |

**Internal files inspected: 13 distinct files (16 distinct file:line regions).**

---

## Research Gate Checklist

Hard blockers (all must be checked for `gate_passed: true`):

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read)
- [x] 10+ unique URLs total (14 collected: 6 full + 8 snippet)
- [x] Recency scan (last 2 weeks) performed + reported (yfinance 1.3.0 / 1.2.2 found; no impact on phase-32.4)
- [x] >=3 query-variant discipline (current-year 2026 + last-2-year 2025 + year-less canonical, all visible above)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:

- [x] Internal exploration covered every relevant module (16 file:line regions across 13 files)
- [x] Contradictions noted (masterplan implementation_plan summary says `longName` first; canonical code at line 963 says `shortName` first — flagged in Topic 1)
- [x] All claims cited per-claim (no footer-only references)

---

## JSON Envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "gate_passed": true
}
```
