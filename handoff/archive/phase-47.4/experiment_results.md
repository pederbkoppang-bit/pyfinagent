# Experiment Results — phase-47.4: Sharpe/maxDD metric integrity

**Cycle:** 4 of the production-ready+money push (FREE — no project LLM spend).
**Step:** 47.4 | **Result:** ready for Q/A.

## What was built / changed (2 one-line sort fixes + 1 new test)
Single shared root cause: `get_paper_snapshots()` returns rows newest-first (`ORDER BY snapshot_date
DESC`); two consumers walked the NAV series without re-sorting -> reversed NAVs negate every return
(Sharpe sign flip) and read growth as a crash (phantom 60% maxDD).

1. `backend/services/perf_metrics.py::compute_sharpe_from_snapshots` — added
   `snapshots = sorted(snapshots, key=lambda s: s.get("snapshot_date") or "")` before computing.
   Fixes the cockpit Sharpe (paper_trading.py:219 /portfolio) + /performance + all 4 callers.
   Idempotent for `compute_paper_sharpe_window` (already pre-sorts).
2. `backend/services/paper_go_live_gate.py::_snapshot_max_dd_pct` — same chronological sort before
   peak-to-trough. Corrects the gate maxDD 60.08% -> 5.31%.
3. NEW `tests/services/test_phase_47_4_metric_order_invariance.py` — 2 BEHAVIORAL guards on a growth
   NAV fixture (chronological vs reversed): Sharpe order-invariant + positive; maxDD order-invariant +
   small. FAILS on pre-fix code, PASSES after.

No formula change (sort only). No trade execution. No LLM spend.

## Verbatim verification output
```
$ python -m pytest tests/services/test_phase_47_4_metric_order_invariance.py -q  -> 2 passed in 1.29s
$ ast.parse(perf_metrics.py) + ast.parse(paper_go_live_gate.py)  -> ast OK ; EXIT_CODE=0

Mutation-resistance proof:
  post-fix Sharpe chron==desc: True (22.91) ; maxDD chron==desc: True (0.495)
  OLD (unsorted) Sharpe chron vs desc: 22.91 vs -23.35 -> DIFFER  (the same +/- sign-flip as -5.72)

LIVE after backend reload:
  /gate realized_max_dd_pct: 60.08% -> 5.3112    | live_sharpe: -5.72 -> 5.42
  /gate max_dd_within_tolerance: False -> True   | promote_eligible: False (still 1/5; needs 100 trades/PSR/DSR)
  /portfolio sharpe_ratio (cockpit, was -5.72): 5.42
```

## Success-criteria mapping (masterplan phase-47.4)
1. Both helpers sort chronologically by snapshot_date; results order-invariant — **MET** (sorts added; mutation proof shows old path differed, new path equal).
2. pytest order-invariance guard passes post-fix / would fail pre-fix; ast clean — **MET** (2 passed; OLD-path proof 22.91 vs -23.35; ast OK).
3. live /gate realized_max_dd_pct ~5% (down from 60%) after reload — **MET** (5.3112; bonus: live_sharpe -5.72->5.42, cockpit sharpe_ratio 5.42, max_dd_within_tolerance False->True). live_check_47.4.md captures it.

## Scope honesty
Sort-only fix to the single-source-of-truth metric module (no formula change). Corrects the cockpit
Sharpe sign + the gate maxDD + the wrongly-red max_dd_within_tolerance boolean. Does NOT touch trade
execution and does NOT make a 27-sample Sharpe statistically trustworthy (Lopez de Prado MinTRL) — a
sample-size gate on the cockpit Sharpe is flagged as a separate follow-up, not claimed here.
`/performance` returned None for a naive top-level parse; authoritative corrected values are on /gate
+ /portfolio (both fed by the fixed function).

## Files
backend/services/perf_metrics.py, backend/services/paper_go_live_gate.py,
tests/services/test_phase_47_4_metric_order_invariance.py, .claude/masterplan.json (phase-47.4 added),
handoff/current/{contract.md, research_brief_phase_47_4_metric_integrity.md, live_check_47.4.md}.
