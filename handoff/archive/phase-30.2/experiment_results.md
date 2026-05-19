# Experiment Results -- phase-30.2

**Step:** P1: Wire `backfill_missing_stops` into autonomous_loop Step 5.6.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.

## Summary

Inserted a `trader.backfill_missing_stops()` call into
`backend/services/autonomous_loop.py` Step 5.6 BEFORE the existing
`trader.check_stop_losses()` call. Records the result in
`summary["stop_loss_backfilled"]` for observability. Fail-open via
try/except so a backfill exception does not block the stop-loss
enforcement primitive.

Closes phase-30.0 Stage 7 (FAIL) + P1-2: 7-of-11 open positions had
`stop_loss_price IS NULL` because the phase-25.2 backfill helper had
zero production callers.

## Files touched

| Path | Lines added | Lines removed |
|------|-------------|---------------|
| `backend/services/autonomous_loop.py` | 24 | 0 |
| `backend/tests/test_autonomous_loop_step_5_6.py` (NEW) | 175 | 0 |
| **Total** | **199** | **0** |

Non-comment LOC: ~22 (autonomous_loop wiring + summary key) + ~90
(test). Under the 150-line target for the code change.

**Scope adherence:** the audit's P1-2 named only
`backend/services/autonomous_loop.py` Step 5.6. Plus a new test file
under `backend/tests/` (mandatory per goal directive). No other files
touched.

## Implementation details

### `backend/services/autonomous_loop.py` Step 5.6

Inserted (between line 758 `summary["stop_loss_triggered"] = []` and
the existing `triggered_stops = ...` call):

```python
summary["stop_loss_backfilled"] = []
try:
    backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
    summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
    if backfill_result.get("count_backfilled", 0) > 0:
        logger.info(
            "phase-30.2: backfill_missing_stops synthesized %d stops (skipped %d)",
            backfill_result.get("count_backfilled", 0),
            backfill_result.get("count_skipped", 0),
        )
except Exception as bf_exc:
    logger.exception(
        "phase-30.2: backfill_missing_stops failed (non-fatal; check_stop_losses still runs): %s",
        bf_exc,
    )
```

Reasoning:
- **Async-wrap mandatory** per `.claude/rules/backend-api.md`:
  `save_paper_position` is sync BQ I/O; wrap in `asyncio.to_thread`.
  This mirrors the existing `triggered_stops = await asyncio.to_thread(trader.check_stop_losses)`.
- **Ordering** -- backfill BEFORE check: same-cycle protection. The
  research brief notes (cross-val Section 3 + Kaminski-Lo Source 7)
  that a position currently below its synthesized stop SHOULD trigger
  an immediate sell. That is the desired behavior for the documented
  TER-style hold-at-loss case.
- **Fail-open** -- try/except wrap around the backfill call. The
  stop-loss enforcement primitive `check_stop_losses` is the safety
  control; a transient backfill error must not block it.
- **Observability** -- `summary["stop_loss_backfilled"]` flows to the
  cycle summary so the operator dashboard shows the backfill effect.
  INFO log fires only when `count_backfilled > 0` so the first
  post-fix cycle is visible without spamming on idempotent subsequent
  cycles.

### `backend/tests/test_autonomous_loop_step_5_6.py`

4 test cases:

1. `test_step_5_6_backfill_runs_before_check_stop_losses` -- happy
   path. Mock PaperTrader; assert call-order via `Mock.method_calls`.
   Verifies backfill returns 2 entries, check is invoked with those
   stops visible, triggered=["WDC"], summary populated.
2. `test_step_5_6_idempotent_backfill_no_op` -- cycle 2. Backfill
   returns 0 backfilled / 4 skipped; check still runs. Confirms
   idempotency on subsequent cycles.
3. `test_step_5_6_backfill_exception_does_not_block_check` -- fail
   open. Backfill raises `RuntimeError`; check_stop_losses still
   invoked; check's triggered list flows through.
4. `test_autonomous_loop_step_5_6_contains_backfill_symbol` --
   mirrors the masterplan verification grep predicate. Parses the
   on-disk Step 5.6 block and asserts that the actual CALL line
   (filtering out comment occurrences) for `backfill_missing_stops`
   precedes the call line for `check_stop_losses`. This protects
   against future refactors that reorder the calls or remove the
   wiring.

Mocking strategy: pure `unittest.mock.MagicMock`. Async functions
driven via `asyncio.run()` -- no `pytest-asyncio` dep added (the
codebase doesn't have it; following the goal directive's "no new
deps").

## Verification

### Masterplan verification command (phase-30.2)

```bash
grep -A 5 'Step 5.6' backend/services/autonomous_loop.py | \
  grep -q 'backfill_missing_stops' && \
  python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"
```

Result: **exit 0**.

### Test run

```
$ python -m pytest backend/tests/test_autonomous_loop_step_5_6.py -v
collected 4 items

test_step_5_6_backfill_runs_before_check_stop_losses PASSED [ 25%]
test_step_5_6_idempotent_backfill_no_op PASSED [ 50%]
test_step_5_6_backfill_exception_does_not_block_check PASSED [ 75%]
test_autonomous_loop_step_5_6_contains_backfill_symbol PASSED [100%]

4 passed in 0.01s
```

### Regression check

`python -m pytest backend/tests/test_cycle_heartbeat_alarm.py
backend/tests/test_observability.py` -- 19 passed (7 phase-30.1 +
12 observability), no regressions.

### Syntax check

`python -c "import ast; ast.parse(...)"` on
`backend/services/autonomous_loop.py` and the new test file: OK.

## Hard guardrail attestation

- No mutating BigQuery calls -- the wiring delegates to an existing
  helper (`paper_trader.py::backfill_missing_stops`) which uses
  `save_paper_position` (UPDATE on existing rows; no DROP / ALTER /
  unqualified DELETE).
- No Alpaca calls -- the helper writes only to BQ paper_positions.
- No frontend / `.claude/` / `.mcp.json` touched.
- Diff stays within the audit's proposed-diff scope
  (`backend/services/autonomous_loop.py` only).
- Test ships and passes deterministically.

## Success criteria check

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `autonomous_loop_step_5_6_calls_backfill_missing_stops_before_check_stop_losses` | PASS | Verification grep + test #4 both confirm call-site ordering |
| `syntax_check_passes` | PASS | `python -c "import ast; ast.parse(...)"` returns 0 |
| `after_one_cycle_paper_positions_stop_loss_price_is_null_count_drops_to_zero` | PASS-DEFERRED | Live post-cycle BQ check; loop is PAUSED overnight. Operator verifies in morning via `SELECT COUNT(*) FROM financial_reports.paper_positions WHERE stop_loss_price IS NULL` -- expected drop from 7 to 0. Test #1 verifies the in-cycle wiring; the on-disk effect requires one unpaused cycle. |
| `no_regression_in_existing_stop_loss_enforcement_test` | PASS | All adjacent tests green; no existing stop-loss enforcement test exists in the repo, so the new test #4 establishes the regression baseline. |
