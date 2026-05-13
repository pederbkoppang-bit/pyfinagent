---
step: phase-25.C12
cycle: 78
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_C12.py'
title: Cross-tab Sharpe KPI reconciliation -- backend authoritative (P1)
audit_basis: phase-24.12 F-4 (home page kpiSharpe diverges from /performance perf.sharpe_ratio)
---

# Experiment Results -- phase-25.C12

## Code changes

### `backend/api/paper_trading.py::get_portfolio`
- After the existing `sector_breakdown` try/except, fetches up to 365 snapshots via `await asyncio.to_thread(bq.get_paper_snapshots, limit=365)`.
- Computes `portfolio_sharpe = compute_sharpe_from_snapshots(snapshots_for_sharpe)` -- reuses the canonical helper already used by `/performance` at line 276.
- Assigns `portfolio["sharpe_ratio"] = portfolio_sharpe` (inside the portfolio dict, NOT a top-level response key, so the frontend `PaperPortfolio` interface is the single extension point).
- Per-call try/except: a `get_paper_snapshots` failure logs a warning and leaves `sharpe_ratio = None`; the rest of the response is preserved.
- `phase-25.C12` attribution in the inline comment.

### `frontend/src/lib/types.ts::PaperPortfolio`
- Adds optional field `sharpe_ratio?: number | null` with JSDoc explaining backend-authoritative semantics.

### `frontend/src/lib/api.ts`
- `getPaperPortfolio()` return type unchanged in shape -- the new field is reachable via `PaperPortfolio.sharpe_ratio` (interface extension).

### `frontend/src/app/page.tsx`
- New `apiSharpe` state: `const [apiSharpe, setApiSharpe] = useState<number | null>(null)`.
- In the `Promise.allSettled` handler, when `portfolio.status === "fulfilled"`, calls `setApiSharpe(portfolio.value.portfolio?.sharpe_ratio ?? null)` alongside the existing `setPositions(...)`.
- Line ~163 swap: `const sharpe90 = apiSharpe ?? kpiSharpe(navSeries)` -- backend-authoritative when available, local-fallback otherwise (graceful rolling deploy / fail-open).

### `frontend/src/lib/kpiMetrics.ts::sharpe`
- JSDoc `@deprecated` block added explaining the formula divergence (no risk-free-rate subtraction) and pointing new consumers at the API.

### `tests/verify_phase_25_C12.py` (new file)
- 11 immutable claims with 3 behavioral round-trips:
  - Claims 1-6, 10-11: structural (backend call site, interface extension, api.ts return type, state declaration, swap snippet, deprecation block, response keys preserved, attribution).
  - Claim 7: **Behavioral happy path** -- 60 noisy snapshots; assert `portfolio.sharpe_ratio` is a finite numeric.
  - Claim 8: **Behavioral no-data** -- empty snapshots -> `sharpe_ratio in (None, 0.0)` (graceful).
  - Claim 9: **Behavioral fail-open** -- `get_paper_snapshots` raises -> `sharpe_ratio=None` + rest of response intact (no crash).

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C12.py
get_portfolio: sharpe_ratio fail-open: RuntimeError('BQ down')
PASS: paper_trading_portfolio_endpoint_returns_sharpe_ratio_field
PASS: paper_portfolio_interface_extended_with_sharpe_ratio
PASS: api_ts_getpaperportfolio_returns_paper_portfolio_with_sharpe
PASS: home_page_declares_apisharpe_state
PASS: home_page_uses_api_sharpe_ratio_not_local_kpisharpe
PASS: deprecation_marker_on_kpisharpe_function
PASS: behavioral_get_portfolio_returns_sharpe_ratio_for_valid_snapshots
PASS: behavioral_get_portfolio_empty_snapshots_sharpe_none_or_zero
PASS: behavioral_get_portfolio_snapshot_failure_fails_open
PASS: no_regression_response_keys_preserved
PASS: phase_25_c12_attribution_in_source

11/11 claims PASS, 0 FAIL
```

(The "sharpe_ratio fail-open" log line is emitted by claim 9 behavioral
test -- proves the fail-open path actually executes as designed.)

## Backend + frontend gates

- `python -c "import ast; ast.parse(open('backend/api/paper_trading.py').read())"` -- OK
- `npx tsc --noEmit` (from `frontend/`) -- EXIT=0 (clean)
- 3 behavioral round-trips with mocked BQ + trader run the actual `get_portfolio` coroutine.

## Hypothesis verdict

CONFIRMED. The cross-tab Sharpe divergence root-cause was formula-based (researcher's finding: frontend `kpiMetrics.ts::sharpe` skips the risk-free-rate subtraction that `backend/backtest/analytics.compute_sharpe` performs, creating a ~0.16 Sharpe-unit gap at 4% RFR). After this fix, the home page and `/paper-trading` page consume the same backend-authoritative value computed by the same helper from the same BQ snapshot series -- divergence eliminated by construction.

Three immutable success criteria mapped:
- Criterion 1 (`home_page_uses_api_sharpe_ratio_not_local_kpisharpe`) -- claim 5 (apiSharpe ?? kpiSharpe fallback).
- Criterion 2 (`paper_trading_portfolio_endpoint_returns_sharpe_ratio_field`) -- claim 1 + claims 7-9 behavioral.
- Criterion 3 (`deprecation_marker_on_kpisharpe_function`) -- claim 6 (JSDoc @deprecated block).

## Live-check

Per masterplan: "Home and /paper-trading show identical Sharpe ratio for same window".

After backend restart and frontend rebuild, both pages query the same backend-authoritative value (via different endpoints — `/portfolio` and `/performance`, but both call `compute_sharpe_from_snapshots` on the same BQ snapshot series). Live evidence pending capture in `handoff/current/live_check_25.C12.md`.

## Non-regressions

- `get_portfolio` response shape unchanged at top level: `{portfolio, positions, sector_breakdown}`. `sharpe_ratio` is ADDED INSIDE the `portfolio` dict.
- Existing `compute_sharpe_from_snapshots` helper unchanged.
- `kpiMetrics.ts::sharpe` still exported and callable for backwards compat (just marked deprecated).
- Existing `/performance` endpoint's `sharpe_ratio` unchanged (the `paper-trading` page already reads it; that path is unaffected).
- TS compiles clean.

## Next phase

Q/A pending.
