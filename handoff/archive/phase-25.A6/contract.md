# Sprint Contract -- phase-25.A6 -- Explicit live-vs-backtest Sharpe reconciliation

**Cycle:** phase-25 cycle 19 (P1 sprint)
**Date:** 2026-05-12
**Step ID:** 25.A6
**Priority:** P1
**Audit basis:** bucket 24.6 F-3 -- `paper_go_live_gate.py:90-94` uses NAV-divergence proxy as a stand-in for the Sharpe-gap measurement

## Research-gate

Researcher spawned this cycle (agent a1dea68c07359c910). Brief at
`handoff/current/research_brief.md`. Gate envelope: 6 sources read in full,
16 URLs, recency scan performed, gate_passed=true.

Key research conclusions:
- **The measurement is wrong, not the threshold.** SR_GAP_THRESHOLD = 0.30 at line 38 is correct per Jacquier et al. arxiv 2501.03938 (Jan 2025): 30-50% IS-to-OOS decay range; 30% is the stricter lower bound. Do NOT change the constant.
- **Canonical live Sharpe:** reuse `backend/services/perf_metrics.py::compute_sharpe_from_snapshots`.
- **Fallback chain:**
  1. Primary: `optimizer_best.json["sharpe"]` (already stored).
  2. Secondary: shadow NAV curve via `compute_reconciliation(bq)["series"][i]["backtest_nav"]` -> apply same Sharpe formula.
  3. Tertiary: existing NAV-divergence proxy with `proxy_fallback=True` flag.
  4. No-data: `gap_within_threshold=None` (gate stays red).
- **All data exists** -- no new BQ schema or migration.

## Hypothesis

Adding a new `compute_sharpe_gap(bq, *, backtest_sharpe_source, ...)` helper
in `perf_metrics.py` that returns an explicit Sharpe-gap dict (with documented
fallback chain), then replacing `paper_go_live_gate.compute_gate`'s
NAV-divergence proxy logic (lines 87-94) with this helper, closes phase-24.6
F-3 without touching the SR_GAP_THRESHOLD constant or the 5-boolean gate
contract.

## Success criteria (verbatim from masterplan)

1. `new_function_compute_live_realized_sharpe_vs_backtest_exists`
2. `paper_go_live_gate_uses_explicit_sharpe_not_nav_proxy`
3. `threshold_at_30pct_per_industry_benchmark`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_A6.py`

Live check (per masterplan):
`Reconciliation report shows explicit live_sharpe - backtest_sharpe gap on next cycle`

## Plan

1. **New helper** -- `backend/services/perf_metrics.py`:
   - Add `import json` and `from pathlib import Path` if absent.
   - Add `compute_sharpe_gap(bq, *, backtest_sharpe_source="optimizer_best", risk_free_rate=0.04, min_snapshots=6) -> dict`.
   - Implementation:
     - Pull paper snapshots via `bq.get_paper_snapshots(limit=365)`.
     - `live_sharpe = compute_sharpe_from_snapshots(snapshots)` -- existing canonical helper. None if `< min_snapshots` or sd=0.
     - Resolve `backtest_sharpe` per the fallback chain.
       - Primary: load `backend/backtest/experiments/optimizer_best.json`, return `data.get("sharpe")` if present.
       - Secondary: call `compute_reconciliation(bq)`; if `series` has >=`min_snapshots` rows, build a NAV list from `backtest_nav`, run the same Sharpe formula manually (use the same windowing semantics as `compute_sharpe_from_snapshots`).
       - Tertiary: if both fail, surface the proxy `latest_divergence_pct / 100.0` with `proxy_fallback=True`. This preserves backwards compatibility with the current gate logic and signals operator that the measurement is the legacy proxy.
       - No-data: all three above unavailable -> return with `gap_within_threshold=None` (gate stays red).
     - Compute `gap_abs = abs(live_sharpe - backtest_sharpe)`, `gap_rel = gap_abs / abs(backtest_sharpe)` when `backtest_sharpe != 0`.
     - `gap_within_threshold = (gap_rel <= 0.30)` when defined; None otherwise.
     - Return dict: `{live_sharpe, backtest_sharpe, gap_abs, gap_rel, threshold, gap_within_threshold, source, note, proxy_fallback, computed_at}`.
2. **Wire `compute_gate`** -- `backend/services/paper_go_live_gate.py`:
   - Replace lines 87-94 (the proxy-only logic) with a call to `compute_sharpe_gap(bq)`.
   - `sr_gap_le = sharpe_gap_result["gap_within_threshold"]` (which can be True/False/None).
   - `booleans["sr_gap_le_30pct"] = bool(sr_gap_le)` (None coerces to False, keeping the gate red on no-data).
   - Add the new dict's diagnostic fields to `details` so the UI can render the explicit comparison: `live_sharpe`, `backtest_sharpe`, `sharpe_gap_rel`, `sharpe_gap_source`.
   - Preserve the `latest_reconciliation_divergence_pct` field (still useful as a sibling signal).
3. **Verifier** -- `tests/verify_phase_25_A6.py` -- 9+ claims:
   - Claim 1: `compute_sharpe_gap` exists with the documented signature in `perf_metrics.py`.
   - Claim 2: SR_GAP_THRESHOLD remains 0.30 (criterion 3 grep).
   - Claim 3: `paper_go_live_gate.py` no longer uses `sr_gap_proxy = latest_divergence_pct / 100.0` AND calls `compute_sharpe_gap(bq)`.
   - Claim 4: `compute_gate`'s returned `details` dict includes `live_sharpe` (or `sharpe_gap`-prefixed) field.
   - Claim 5: **Behavioral primary-source** -- fake bq + fake optimizer_best.json with `sharpe=1.0`; fake paper snapshots that yield a calculable live Sharpe close to 1.2. Assert `compute_sharpe_gap` returns `source="optimizer_best"`, `backtest_sharpe=1.0`, `gap_rel = 0.2/1.0 = 0.2`, `gap_within_threshold=True`.
   - Claim 6: **Behavioral threshold-failure** -- live=2.0, backtest=1.0 -> `gap_rel=1.0 > 0.30` -> `gap_within_threshold=False`.
   - Claim 7: **Behavioral no-data** -- live snapshots empty -> `live_sharpe=None`, `gap_within_threshold=None` (gate stays red).
   - Claim 8: **Behavioral fallback-2 (shadow curve)** -- optimizer_best.json missing, shadow curve has >=6 NAV points -> `source="shadow_curve"`, gap computed from shadow.
   - Claim 9: **Behavioral fallback-3 (proxy)** -- both primary and shadow unavailable, reconciliation has `latest_divergence_pct` -> `source="proxy_fallback"`, `proxy_fallback=True`.
   - Claim 10: **compute_gate integration** -- with a mocked bq + happy-path Sharpe data, the gate's `booleans["sr_gap_le_30pct"]` reflects the new helper's `gap_within_threshold` (not the divergence proxy directly).
   - Claim 11: contract docstring mentions "industry benchmark" + "30%".

## Non-goals

- No change to SR_GAP_THRESHOLD value.
- No new BQ schema.
- No frontend changes.
- No removal of the existing reconciliation divergence (kept as a sibling signal in `details`).

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/services/paper_go_live_gate.py:38, 87-94, 107-127` -- threshold + proxy logic + details dict
- `backend/services/perf_metrics.py:84` -- canonical compute_sharpe_from_snapshots
- `backend/services/reconciliation.py` -- shadow-curve source
- `backend/backtest/experiments/optimizer_best.json` -- primary backtest Sharpe source
- CLAUDE.md `Critical Rules` -- 30s BQ timeout (covered by existing helpers)
