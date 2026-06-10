# Experiment Results — phase-53.3 (Data-stack elevation)

**Date:** 2026-06-10. **Status:** complete. Column-pruned the two hot
`historical_fundamentals` `SELECT *` reads → **−21.2% bytes** (dry-run measured,
results byte-identical); freshness/lineage recorded; partition/cluster + Sortino-lineage
documented as operator-gated. 30s timeout preserved; no schema mutation. $0.

## What was done

- **Opt-1/Opt-2:** `backend/backtest/cache.py:153` (`preload_fundamentals`) + `:351`
  (`cached_fundamentals` fallback): `SELECT *` → explicit 12 consumed columns. Drops 4
  never-read columns (0 call-sites, re-grep-proven). −21.2% bytes (655,079 → 515,937).
- **Freshness/lineage check** recorded (signal/price tables green; macro red + the
  `sortino.py` dataset-mismatch lineage discrepancy documented).
- **Did NOT** add date filters (proven cargo-cult on the non-partitioned tables), repoint
  Sortino (result change), or mutate schema (operator-gated).

## Files changed

| File | Change |
|------|--------|
| `backend/backtest/cache.py` | 2× `SELECT *` → 12-column projection on `historical_fundamentals` (preload + fallback); 30s timeout + WHERE/ORDER/LIMIT unchanged. |
| `handoff/current/live_check_53.3.md` | Audit + before/after bytes + freshness/lineage + operator-gated recs. |

## Verification output (verbatim)

```
ast.parse cache.py -> parses
$0 dry-run (historical_fundamentals):
  OLD SELECT *       : 655,079 bytes
  NEW 12-col project : 515,937 bytes
  DELTA              : -139,142 bytes (-21.2%)
DO-NO-HARM grep (backend/backtest/historical_data.py + data_server):
  dropped filing_date/ingested_at/market/currency -> 0 call-sites each
  consumed total_revenue(3) net_income(2) total_debt(2) total_equity(2) total_assets(2)
    operating_cash_flow(1) shares_outstanding(1) sector(2) industry(2) dividends_per_share(1)
  backend-wide sweep: the 4 dropped cols are only read from OTHER data sources (13F filingDate,
    PEAD dates, position market, market-config currency) -- never from historical_fundamentals rows
pytest -k "cache or fundamental" -> 4 passed
freshness: overall red; prices/fundamentals/signals_log/paper_* GREEN; historical_macro RED
```

## Acceptance-criteria mapping (phase-53.3 — VERBATIM)

| # | Criterion | Result |
|---|-----------|--------|
| 1 | research gate passed (BQ cost/perf + lineage sources) + hot-path audit w/ per-query bytes + partition/cluster-filter gaps | PASS — researcher gate (6 sources); audit: 3 hot tables NOT partitioned/clustered (proven), `SELECT *` gaps at cache.py:153/351 |
| 2 | optimizations land w/ BEFORE/AFTER bytes (dry-run) + cost + freshness/lineage check recorded | PASS — −21.2% (655,079→515,937 dry-run); freshness bands recorded; Sortino lineage discrepancy documented |
| 3 | 30s timeout preserved + RESULTS unchanged (correctness-preserving); NO DROP/unqualified DELETE | PASS — projection-only; timeout/WHERE/ORDER/LIMIT untouched; 0 dropped-col call-sites; 4 tests pass; no DROP/DELETE/schema mutation |
| 4 | live_check_53.3.md records before/after bytes + cost delta + freshness/lineage | PASS — live_check_53.3.md |

## DO-NO-HARM / scope honesty

- **Honest −21.2% (not the researcher's −41%):** the −41% used a 10-col set that would
  drop 2 CONSUMED columns (a result change). The results-preserving projection is all 12
  consumed columns → −21.2%. I kept correctness over a bigger headline number.
- Projection-only: byte-identical results (dropped cols unused; consumers use `.get`).
  30s fallback timeout untouched. No date-filter cargo-cult (tables aren't partitioned).
- NO schema mutation / DROP / DELETE; NO Sortino repoint (result change); NO `.env` edit.
  $0 (dry-run estimation only; no LLM, no bytes billed). The big partition/cluster win +
  the Sortino-lineage fix + the macro refresh are documented operator-gated follow-ups.
