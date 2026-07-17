# Experiment results — step 70.4 (S3: un-gate throughput — surface + reconcile the silent BUY-gates)

**Phase/step:** phase-70 → 70.4 | **Date:** 2026-07-17 | **Type:** backend observability (always-on, $0) + 2
flag-gated behavior knobs (default-OFF). live_check: none (no UI).

## Files changed (4)

1. **`backend/config/settings.py`** — new `paper_session_budget_reconcile_enabled` (bool, default False).
2. **`backend/services/autonomous_loop.py`** —
   - **G1-A** `_check_session_budget` logs a WARNING at the breach BEFORE raising (never silent).
   - **G1-C** new module var `_effective_session_budget` (defaults to the hidden $1.00); `run_daily_cycle` sets it
     to `paper_max_daily_cost_usd` when the reconcile flag is ON, else $1.00; `summary['session_budget_usd']`
     reflects it.
   - **G1-B** after the two `gather(return_exceptions=True)`, scans the raw results for `BudgetBreachError` (which
     the `isinstance(dict)` filter silently drops) → sets `summary['session_budget_breach'/…]` + WARN.
   - **G3-A** lite parse-fail else-branch marks `_parse_failed=True` + logs WARN + fixes the mislabeled INFO log.
   - **G3-B** `_degraded_scoring_check` now counts `_parse_failed`/`_degraded` (affects only the P1 alert, no trade).
   - **G3-C** the lite return dict adds top-level `_parse_failed`, and (only when `paper_synthesis_integrity_enabled`)
     `_degraded=True` so the cycle loop drops the parse-fail from decide_trades input (fail-safe).
   - **G2-A** folds `trader.buy_rejections` into `summary['buy_rejections'/'buy_rejections_by_reason']` + WARN.
3. **`backend/services/paper_trader.py`** — **G2-A** `self.buy_rejections` accumulator in `__init__`; the
   price-tolerance `return None` appends `{ticker, reason:'price_tolerance', divergence_pct, tolerance_pct, …}`.
4. **`backend/tests/test_phase_70_4_gate_observability.py`** (NEW) — 7 deterministic tests.

## Verification command output (verbatim)

```
$ bash -c 'grep -Eqi "session budget|per-cycle|cost cap" backend/services/autonomous_loop.py && ls backend/tests/ | grep -Eqi "70_4|budget|tolerance|gate"'
VERIFICATION: PASS (exit 0)
$ python -m pytest backend/tests/test_phase_70_4_gate_observability.py -q
7 passed
```
Import-smoke: settings/autonomous_loop/paper_trader import clean; `_effective_session_budget` present; flag default False.

## Criterion evidence

- **C1 (budget never silent):** `test_session_budget_breach_logs_and_raises` — a breach LOGS "SESSION BUDGET
  BREACH" before raising; G1-B surfaces it to `summary['session_budget_breach']`. The effective ceiling is
  reconcilable to `paper_max_daily_cost_usd` (G1-C, flag-gated; OFF → $1.00 byte-identical).
  `test_session_budget_below_ceiling_no_raise` — reconciled $2 ceiling admits a $0.50 cycle.
- **C2 (price-tolerance diagnosable + tunable):** already logged with ticker+drift and tunable via
  `paper_price_tolerance_pct` on HEAD; 70.4 adds the summary surfacing —
  `test_price_tolerance_rejection_is_accumulated` shows a 10%-divergence BUY is rejected AND recorded in
  `trader.buy_rejections` with ticker + divergence; `test_price_tolerance_accumulator_empty_when_within_tolerance`
  shows a within-tolerance BUY records no rejection.
- **C3 (parse-fail counted as degraded, not a score-5 mask):** `test_degraded_check_counts_parse_failed` — three
  `_parse_failed` HOLD-score-5 rows are counted degraded (fires the P1 alert) while a genuine HOLD is not;
  `test_degraded_check_ignores_genuine_hold` confirms a real HOLD-5 is untouched. G3-C (flag) additionally drops a
  parse-fail from decide_trades input under `paper_synthesis_integrity_enabled` — fail-safe (removing a spurious
  neutral can never create a BUY).
- **C4 (flag-gated + always-on split):** `test_flag_present_and_default_off`. Observability additions (G1-A/B,
  G2-A, G3-A/B) are always-on + $0 + change no trade; behavior changes (G1-C budget reconcile, G3-C parse-fail
  drop) are flag-gated default-OFF (byte-identical OFF).

## Regression / do-no-harm
`pytest test_phase_61_2_decision_integrity test_phase_50_2_multicurrency test_phase_70_3 test_phase_70_2` →
61 passed (the degraded machinery, currency paths, and prior phase-70 flags all green). git status: only
backend/config + backend/services (x2) + backend/tests (new) + handoff/. $0; paper-only; NO risk threshold moved
(the budget reconcile is a COST knob; the degraded-guard change touches only the P1 ALERT); historical_macro
FROZEN; every addition fail-safe. Behavior changes DARK-until-token; observability always-on.
