# Sprint Contract -- phase-25.C12 -- Cross-tab Sharpe KPI reconciliation (backend authoritative)

**Cycle:** phase-25 cycle 22 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.C12
**Priority:** P1
**Audit basis:** bucket 24.12 F-4 -- home page's `kpiSharpe(navSeries)` (`frontend/src/lib/kpiMetrics.ts:57`) diverges from `/paper-trading` page's `perf.sharpe_ratio` (from `/api/paper-trading/performance`)

## Research-gate

Researcher spawned this cycle (agent adec525da9a7396d2). Brief at
`handoff/current/research_brief.md`. Gate envelope: 6 sources read in full,
16 URLs, recency scan performed, 7 internal files inspected, gate_passed=true.

**Critical research finding:** the divergence is FORMULA-based, not input-based:
- Frontend `kpiMetrics.ts:57-65` -- `(mu / sd) * sqrt(252)` with NO risk-free rate subtraction.
- Backend `analytics.compute_sharpe` (via `compute_sharpe_from_snapshots` at `perf_metrics.py:84-112`) -- subtracts `0.04/252` as daily risk-free rate before dividing by std.

At 4% RFR, this creates a systematic ~0.16 Sharpe-unit gap on a ~1.17-Sharpe system. Wiring the home page to the API value eliminates the gap by construction.

Other key conclusions:
- Stripe / Linear / Datadog / Databricks pattern: backend-authoritative SSOT for cross-page KPIs.
- TSDoc `@deprecated` syntax is stable; codebase already uses it at `types.ts:10`.
- `PaperPortfolio` interface at `types.ts:608-617` needs an optional `sharpe_ratio?: number | null` field for backwards-compat during rolling deploy.
- Insertion point in `get_portfolio`: after `sector_breakdown` try/except (after line 205), before `result = {` at line 208.

## Hypothesis

Adding `sharpe_ratio` to the `/portfolio` response (computed via the
already-canonical `compute_sharpe_from_snapshots`) and wiring the home
page to consume that API value (with `kpiSharpe(navSeries)` retained as
a graceful fallback during rolling deploy or API outage), then marking
`kpiMetrics.ts::sharpe` `@deprecated` -- eliminates the cross-tab
divergence by construction (same formula, same input series, same
value rendered on both tabs).

## Success criteria (verbatim from masterplan)

1. `home_page_uses_api_sharpe_ratio_not_local_kpisharpe`
2. `paper_trading_portfolio_endpoint_returns_sharpe_ratio_field`
3. `deprecation_marker_on_kpisharpe_function`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_C12.py`

Live check (per masterplan):
`Home and /paper-trading show identical Sharpe ratio for same window`

## Plan

1. **Backend** -- `backend/api/paper_trading.py::get_portfolio`:
   - After the existing `sector_breakdown` try/except block (after line 205), fetch up to 365 snapshots and compute `portfolio_sharpe = compute_sharpe_from_snapshots(snapshots)` -- reuses the same helper that `/performance` calls at line 276.
   - Add `"sharpe_ratio": portfolio_sharpe` (top-level key) to the response `result` dict, alongside `portfolio` / `positions` / `sector_breakdown`.
   - Per-call try/except so a snapshot-fetch failure surfaces `sharpe_ratio = None` rather than breaking the endpoint.
   - Cache shape is unchanged; the existing `paper:portfolio` TTL refreshes the new field on the same cadence.
2. **Frontend types** -- `frontend/src/lib/types.ts`:
   - Extend `PaperPortfolio` interface (lines 608-617) with `sharpe_ratio?: number | null`.
3. **Frontend API client** -- `frontend/src/lib/api.ts:276`:
   - Extend `getPaperPortfolio()` return type to include the new top-level `sharpe_ratio?: number | null` field on the response wrapper (NOT on the inner `portfolio` object -- the backend writes it at the response top-level per the inserted dict change).
4. **Frontend home page** -- `frontend/src/app/page.tsx`:
   - Add `const [apiSharpe, setApiSharpe] = useState<number | null>(null);`.
   - In the `Promise.allSettled` handler (line 105 area): when `portfolio.status === "fulfilled"`, also `setApiSharpe(portfolio.value.sharpe_ratio ?? null)`.
   - At line 161: replace `const sharpe90 = kpiSharpe(navSeries);` with `const sharpe90 = apiSharpe ?? kpiSharpe(navSeries);` -- backend-authoritative when API has it, local-fallback otherwise.
5. **Deprecation marker** -- `frontend/src/lib/kpiMetrics.ts:57`:
   - Add JSDoc block:
     ```typescript
     /**
      * @deprecated phase-25.C12: prefer the backend-authoritative
      * `sharpe_ratio` field from `/api/paper-trading/portfolio` or
      * `/api/paper-trading/performance`. Local computation uses a
      * different formula (no risk-free rate subtraction) and diverges
      * from the API value by ~0.16 Sharpe units at 4% RFR. New
      * consumers should call the API.
      */
     ```
6. **Verifier** -- `tests/verify_phase_25_C12.py` -- 9+ claims:
   - Claim 1: `get_portfolio` source contains the canonical `"sharpe_ratio":` key in its result dict + a `compute_sharpe_from_snapshots` call site inside the function body.
   - Claim 2: `PaperPortfolio` interface in `types.ts` contains the `sharpe_ratio?: number | null` field.
   - Claim 3: `getPaperPortfolio()` return type in `api.ts` includes `sharpe_ratio?: number | null`.
   - Claim 4: `page.tsx` contains `apiSharpe` state declaration.
   - Claim 5: `page.tsx` line ~161 reads `apiSharpe ?? kpiSharpe(navSeries)` (NOT bare `kpiSharpe(navSeries)`).
   - Claim 6: `kpiMetrics.ts::sharpe` is preceded by a `@deprecated` JSDoc block.
   - Claim 7: **Behavioral happy path** -- import `get_portfolio` route handler; monkey-patch bq + trader to return deterministic snapshots; call the coroutine; assert `result["sharpe_ratio"]` is a float (or None on no-data) and is computed via the canonical helper.
   - Claim 8: **Behavioral no-data fallback** -- empty snapshots -> `result["sharpe_ratio"]` is None or 0.0 (not a crash).
   - Claim 9: **Behavioral fail-open** -- `bq.get_paper_snapshots` raises -> `result["sharpe_ratio"]` is None (or absent), endpoint still returns the rest of the response.
   - Claim 10: grep-level no regression -- `get_portfolio` still returns `portfolio`, `positions`, `sector_breakdown` keys.
   - Claim 11: docstring or inline comment in `get_portfolio` references phase-25.C12 attribution.

## Non-goals

- No change to `compute_sharpe_from_snapshots` semantics (the canonical helper is correct).
- No removal of `kpiMetrics.ts::sharpe` -- it stays for backwards compat, just marked deprecated.
- No frontend visual changes beyond the KPI value source.
- No new BQ schema.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/api/paper_trading.py:160-214` (`get_portfolio`), :249-302 (`get_performance`), :276 (canonical Sharpe call site)
- `backend/services/perf_metrics.py:84-112` (canonical Sharpe helper)
- `frontend/src/lib/kpiMetrics.ts:57-65` (local kpiSharpe -- target of deprecation)
- `frontend/src/app/page.tsx:161` (kpiSharpe call site to swap)
- `frontend/src/app/page.tsx:96-105` (portfolio fetch handler)
- `frontend/src/lib/types.ts:10` (existing `@deprecated` JSDoc pattern), :608-617 (`PaperPortfolio` interface)
- CLAUDE.md `Critical Rules` -- 30s BQ timeout (covered by existing helper)
