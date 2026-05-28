# Contract — phase-47.4: Sharpe/maxDD metric integrity

**Cycle:** 4 of the production-ready+money push (FREE — no project LLM spend).
**Step:** 47.4 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** moderate-complex (touches perf_metrics.py, the single source of truth).

NOTE: 47.2 (first trade) still PARKED on operator LLM-spend gate. 47.1 + 47.3 done+pushed. 47.4 is the
next unblocked, money-critical free item (corrupts the go-live gate + north-star metric).

## Research-gate summary (PASSED)
Researcher `a4bc11ef3f7d98cbf`, tier=moderate-complex, `gate_passed: true`. 6 sources in full, 17 URLs,
recency scan, 9 internal files. Both live inconsistencies reproduced exactly against BQ. Brief:
`research_brief_phase_47_4_metric_integrity.md`.

**Single shared root cause:** `bq.get_paper_snapshots()` (`bigquery_client.py:1038`) returns rows
`ORDER BY snapshot_date DESC` (newest-first). Two consumers walk the NAV series WITHOUT re-sorting:
- `compute_sharpe_from_snapshots` (`perf_metrics.py:87-115`, cockpit path `paper_trading.py:219`):
  `np.diff(navs)/navs[:-1]` on DESC NAVs flips mean daily return +0.0397 -> -0.0291 -> Sharpe **-5.7156**
  (= displayed -5.72). Sortino +15.59 stayed positive because its path used the chronological
  redLineSeries. Sharpe is sign-order-invariant by definition -> bug.
- `_snapshot_max_dd_pct` (`paper_go_live_gate.py:43-57`): walks DESC, peak=newest(23654), iterates to
  seed(9499.5) -> (23654-9499)/23654 = **60.08%**. Chronological maxDD = **5.31%** (matches cockpit).
  Cockpit RIGHT, gate WRONG.

Caveat (carried, not fixed here): at n_obs=27 a point Sharpe is statistically not meaningful
(Lopez de Prado MinTRL); metrics-v2 already returns None < 30 obs but the cockpit Sharpe does not gate
on sample size. This step corrects the MATH (order), not the small-sample trustworthiness.

## Hypothesis
Sorting snapshots chronologically by `snapshot_date` at the top of both helpers makes Sharpe and
max-drawdown order-invariant, correcting the cockpit Sharpe sign and the gate maxDD (60% -> 5.31%).
Fixing the gate maxDD legitimately flips `max_dd_within_tolerance` False->True (5.31 <= 20) — corrects a
wrongly-red go-live boolean (gate stays 0/5 -> 1/5; still not eligible, needs 100 trades + PSR + DSR + sr_gap).

## Immutable success criteria (verbatim from masterplan.json phase-47.4)
1. compute_sharpe_from_snapshots (perf_metrics.py) and _snapshot_max_dd_pct (paper_go_live_gate.py) sort snapshots chronologically by snapshot_date before computing; results are order-invariant
2. a pytest regression guard asserts Sharpe + maxDD order-invariance (passes post-fix; would fail on the reversed/unsorted pre-fix path); ast.parse clean on edited files
3. live /api/paper-trading/gate details.realized_max_dd_pct corrected from ~60% to the cockpit-consistent ~5% band after backend reload (live_check_47.4.md captures the curl + cockpit Sharpe sign)

## Plan steps
1. `perf_metrics.py::compute_sharpe_from_snapshots` — `snapshots = sorted(snapshots, key=lambda s: s.get("snapshot_date") or "")` before extracting navs (idempotent for compute_paper_sharpe_window which already pre-sorts; fixes the cockpit direct path + all 4 callers).
2. `paper_go_live_gate.py::_snapshot_max_dd_pct` — same chronological sort before extracting navs.
3. NEW `tests/services/test_phase_47_4_metric_order_invariance.py` — behavioral: a growth NAV fixture in chronological vs reversed order; assert Sharpe order-invariant + positive, maxDD order-invariant + small. FAILS on pre-fix code, PASSES after.
4. Verify: pytest green + ast clean (immutable cmd); restart backend, curl /gate showing realized_max_dd_pct ~5% (down from 60%) + cockpit Sharpe sign -> live_check_47.4.md. Fresh Q/A.

## Blast radius
`perf_metrics.py` (single source of truth — sort only, no formula change), `paper_go_live_gate.py`
(sort only). Changes the gate's max_dd boolean (corrects a bug). No trade execution. No LLM spend.
Backend reload required for the live /gate value to refresh.

## References
- `research_brief_phase_47_4_metric_integrity.md` (gate); `roadmap_master.md` workstream 4
- `backend/services/perf_metrics.py:87-115`; `backend/services/paper_go_live_gate.py:43-57`
- `backend/db/bigquery_client.py:1038` (get_paper_snapshots DESC); `.claude/rules/backend-services.md` (single-metric-source)
