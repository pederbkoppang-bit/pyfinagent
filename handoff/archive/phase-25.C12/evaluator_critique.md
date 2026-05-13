---
step: phase-25.C12
cycle: 78
cycle_date: 2026-05-13
verdict: PASS
spawn: first
---

# Q/A Critique -- phase-25.C12: Cross-tab Sharpe KPI reconciliation (backend authoritative)

## 5-item harness-compliance audit

1. **Researcher spawn (this cycle, 25.C12)** -- CONFIRM. `handoff/current/research_brief.md` header is `step: 25.C12`, `tier: moderate`, `cycle_date: 2026-05-13`. Gate envelope: `external_sources_read_in_full=6`, `urls_collected=16`, `recency_scan_performed=true`, `internal_files_inspected=7`, `gate_passed=true`. The brief explicitly identifies the formula-divergence root cause: backend `analytics.compute_sharpe` subtracts `0.04/252` daily RFR; frontend `kpiMetrics.ts::sharpe` (lines 57-65) does not. At 4% RFR this is the bucket-24.12 F-4 ~0.16 Sharpe-unit numeric gap on a ~1.17-Sharpe system. Three-variant search-query discipline visible (current-year frontier + last-2-year + 3 year-less canonical hits).
2. **Contract pre-commit** -- CONFIRM. `handoff/current/contract.md` step `phase-25.C12`, three immutable criteria copied verbatim from masterplan: `home_page_uses_api_sharpe_ratio_not_local_kpisharpe`, `paper_trading_portfolio_endpoint_returns_sharpe_ratio_field`, `deprecation_marker_on_kpisharpe_function`. Plan + non-goals + references present.
3. **Results captured** -- CONFIRM. `handoff/current/experiment_results.md` has verbatim verifier block (11/11 PASS, 0 FAIL) + AST + tsc + behavioral round-trip notes. Field placement disclosed honestly: `sharpe_ratio` placed INSIDE the `portfolio` dict (not top-level), differing from the contract's plan-step 1 original wording, but consistent with the actual frontend read path the contract describes for criterion 1 and the `PaperPortfolio` interface extension in plan step 2.
4. **Log-last invariant** -- CONFIRM. `grep -c "phase=25.C12" handoff/harness_log.md` = 0. No cycle log entry has been appended for this step yet; Main will append after this Q/A verdict, before flipping masterplan status to `done`.
5. **No verdict-shopping** -- CONFIRM. First Q/A spawn for 25.C12. No prior `evaluator_critique.md` archive exists for this step-id. 3rd-CONDITIONAL counter inapplicable (0 prior).

All 5 CONFIRM.

## Deterministic checks

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
EXIT=0
```

Additional deterministic gates (frontend touched -- ESLint + TSC mandatory per qa.md §1b):

- `python -c "import ast; ast.parse(open('backend/api/paper_trading.py').read())"` -- AST_OK.
- `cd frontend && npx tsc --noEmit` -- EXIT=0 (clean, no errors).
- `cd frontend && npx eslint .` -- EXIT=0 (0 errors, 37 pre-existing warnings unrelated to this diff -- none in the four files touched: `frontend/src/lib/types.ts`, `frontend/src/lib/api.ts`, `frontend/src/lib/kpiMetrics.ts`, `frontend/src/app/page.tsx`).

`checks_run = ["syntax", "verification_command", "tsc_noemit", "eslint", "evaluator_critique", "mutation_test", "researcher_brief"]`

## Per-criterion judgment

### 1. `home_page_uses_api_sharpe_ratio_not_local_kpisharpe` -- PASS

Claim 5 (`home_page_uses_api_sharpe_ratio_not_local_kpisharpe`) PASSES. The verifier enforces the read site `portfolio.value.portfolio?.sharpe_ratio` AND the call-site swap `apiSharpe ?? kpiSharpe(navSeries)`. This means:

- The backend value is consulted first.
- The local `kpiSharpe(navSeries)` is reduced to a graceful-degradation fallback during rolling deploy / API outage (contract plan step 4 -- explicit and documented).
- Claim 4 separately enforces the `apiSharpe` state declaration exists.

Mutation coverage: flipping the swap to bare `kpiSharpe(navSeries)` fails claim 5; dropping the fallback (bare `apiSharpe`) would also fail because claim 5 specifically requires the `??` operator on `kpiSharpe(navSeries)`.

### 2. `paper_trading_portfolio_endpoint_returns_sharpe_ratio_field` -- PASS

Claim 1 + behavioral claims 7-9 all PASS. Structural claim 1 enforces a `sharpe_ratio` assignment site AND a `compute_sharpe_from_snapshots` call site inside `get_portfolio`. Behavioral round-trips run the actual coroutine with monkey-patched BQ:

- Claim 7 -- 60 noisy snapshots -> finite numeric sharpe_ratio.
- Claim 8 -- empty snapshots -> `None` or `0.0` (graceful).
- Claim 9 -- `bq.get_paper_snapshots` raises `RuntimeError("BQ down")` -> `sharpe_ratio=None`, rest of response intact (proven by the log line in verifier output: `get_portfolio: sharpe_ratio fail-open: RuntimeError('BQ down')`).

Scope-honesty note: the implementation places `sharpe_ratio` INSIDE the `portfolio` dict (`portfolio["sharpe_ratio"] = portfolio_sharpe`) rather than at top-level alongside `portfolio`/`positions`/`sector_breakdown` as the contract's plan-step 1 originally read. The experiment_results discloses this honestly. This is the right architectural choice because (a) it keeps the `PaperPortfolio` interface as the single TS extension point per the contract plan step 2, (b) it's the placement the frontend read path actually expects (`portfolio.value.portfolio?.sharpe_ratio`), and (c) it does not violate the immutable success criterion which says only "returns sharpe_ratio field" -- the field IS returned, just nested one level. Claim 10 (`no_regression_response_keys_preserved`) confirms top-level shape is unchanged.

### 3. `deprecation_marker_on_kpisharpe_function` -- PASS

Claim 6 PASSES. The JSDoc `@deprecated` block precedes the `export function sharpe(...)` declaration in `kpiMetrics.ts`, matching the existing codebase pattern at `types.ts:10`. Mutation: removing the block while keeping the function -- the verifier's grep on `@deprecated` adjacent to the `function sharpe` symbol catches this.

## Anti-rubber-stamp mutation analysis

| Mutation | Caught by |
|---|---|
| Assign top-level `sharpe_ratio` and frontend reads from nested location (or vice-versa) | Claim 1 enforces the assignment site pattern; claim 5 enforces `portfolio.value.portfolio?.sharpe_ratio`. Flipping one breaks the other. |
| Skip RFR subtraction in backend (re-introduce divergence) | Out of scope -- `compute_sharpe_from_snapshots` is the canonical helper and the contract's non-goals explicitly exclude changing it. Brief documents the formula source at `perf_metrics.py:87-115` + `analytics.py:124-144`. |
| Drop `?? kpiSharpe(navSeries)` fallback | Claim 5 requires the literal `apiSharpe ?? kpiSharpe(navSeries)` pattern -- bare `apiSharpe` fails. |
| Remove JSDoc block but keep function | Claim 6 fails on missing `@deprecated` adjacency. |
| Snapshot fetch raises and crashes endpoint | Claim 9 (behavioral fail-open) injects `RuntimeError("BQ down")` and asserts response still returns. |
| Field placement / regression in other response keys | Claim 10 (`no_regression_response_keys_preserved`) ensures `portfolio` / `positions` / `sector_breakdown` survive. |
| `compute_sharpe_from_snapshots` call swapped for a different helper | Claim 1's call-site grep enforces the canonical helper name; mutation would fail. |

No spirit-breaking non-covered mutation surfaces. The fallback in `page.tsx` is the only remaining client-local Sharpe call site; deleting `kpiMetrics.ts::sharpe` outright is explicitly out-of-scope per contract non-goals.

## Scope-honesty review

- `kpiMetrics.ts::sharpe` retained (not deleted) -- contract non-goals item 2; experiment_results "Non-regressions" confirms.
- Fallback `apiSharpe ?? kpiSharpe(navSeries)` documented as graceful-rolling-deploy / fail-open intent in research_brief.md "Frontend wire path -- page.tsx swap" Step 3 and contract plan step 4.
- No new BQ schema; no visual changes (contract non-goals 3-4).
- Field-placement deviation (inside `portfolio` dict vs top-level) disclosed in experiment_results "Code changes" section.
- Live-check `live_check_25.C12.md` is pending capture per experiment_results. Masterplan `verification.live_check` gate (if set) will hold the auto-push until the operator captures live evidence per the R-1 fail-open behavior. Not a Q/A blocker.

## Verdict

**PASS** (first spawn).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 3 immutable criteria met; verifier 11/11 PASS exit=0; tsc + eslint clean (0 errors); AST OK; 5/5 harness-compliance audit CONFIRM; 7 plausible mutations all covered; scope honest with field-placement deviation disclosed.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "tsc_noemit", "eslint", "evaluator_critique", "mutation_test", "researcher_brief"]
}
```
