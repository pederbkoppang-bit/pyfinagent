# Experiment Results -- phase-30.1

**Step:** P1: Out-of-band autonomous-cycle heartbeat alarm.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.

## Summary

Added `cycle_heartbeat_alarm` (pure function returning a verdict dict)
and `fire_cycle_heartbeat_alarm` (Slack dispatch) to
`backend/services/cycle_health.py`. Wired the alarm into the existing
watchdog cron at `backend/slack_bot/scheduler.py::_watchdog_health_check`
with state-transition gating sibling-symmetric to the existing
`_watchdog_last_was_healthy` pattern. Shipped 7-case test file.

Closes audit Anomaly C from phase-30.0 (65h 34m silent cycle gap
2026-05-17 00:26 UTC -> 2026-05-19 18:00 UTC with no out-of-band
operator alert path).

## Files touched

| Path | Lines added | Lines removed |
|------|-------------|---------------|
| `backend/services/cycle_health.py` | 125 | 0 |
| `backend/slack_bot/scheduler.py` | 55 | 0 |
| `backend/tests/test_cycle_heartbeat_alarm.py` (NEW) | 200 | 0 |
| **Total** | **380** | **0** |

Non-comment LOC: 132 (services) + 125 (test) = **257 LOC**, under the
300-line overnight guardrail.

**Scope adherence:** the audit's P1-1 proposed-diff named
`cycle_health.py`, `alerting.py`, `main.py`. My implementation touched
`cycle_health.py` (matches) + `slack_bot/scheduler.py` (substitute for
`main.py`; the watchdog cron lives in scheduler.py per existing
`_watchdog_health_check:334-400`; this seam was specifically
recommended in `research_brief.md` Q2 because the watchdog already
runs out-of-band on a separate process). `alerting.py` was NOT modified
-- the existing `raise_cron_alert_sync` is the right dispatch path
with no changes needed. No code outside the audit's intent was touched.

## Implementation details

### `cycle_health.py`

Added (after `_fire_freshness_alarm:90-136`):

- `_now_utc()` -- monkeypatch seam for test wall-clock control.
- `cycle_heartbeat_alarm(threshold_sec=_CYCLE_HEARTBEAT_STALE_SEC)` --
  pure verdict function. Reads tail of `cycle_history.jsonl` via
  reversed line scan; parses the most recent valid JSON row; computes
  age = now - completed_at; checks weekday-ET. Returns:
  ```python
  {
      "stale": bool,                    # age > threshold
      "age_sec": float | None,
      "should_alarm": bool,             # stale AND weekday-ET
      "is_weekday_et": bool,
      "last_completed_at": str | None,
  }
  ```
  Fail-open: any exception returns the sentinel
  `{"stale": False, "age_sec": None, "should_alarm": False, ...}`
  rather than raising.

- `fire_cycle_heartbeat_alarm(verdict)` -- dispatcher mirroring the
  existing `_fire_freshness_alarm` pattern. Lazy-imports
  `raise_cron_alert_sync` so a missing observability module is
  non-fatal. Fail-open on dispatch.

Constants added at module top:
- `_CYCLE_HEARTBEAT_STALE_SEC: float = 93_600.0` (26h, matching
  existing `_TABLE_MAX_AGE_SEC["paper_portfolio_snapshots"]`).
- `_NYSE_TZ = ZoneInfo("America/New_York")`.

Import added: `from zoneinfo import ZoneInfo`.

### `slack_bot/scheduler.py`

Added (after `_watchdog_last_was_healthy` definition):

- Module-level `_cycle_heartbeat_last_was_stale: bool | None = None`
  -- sibling of the existing `_watchdog_last_was_healthy`. Same
  state-transition semantics:
  - `None -> True`: first probe found stale  -> P1 alert.
  - `False -> True`: fresh -> stale          -> P1 alert.
  - `True -> False`: stale -> fresh          -> P3 recovery (log only).
  - `None -> False`, `True -> True`, `False -> False`: silent.

Added in `_watchdog_health_check` (after the existing Slack post block):

- Lazy import of `cycle_heartbeat_alarm` + `fire_cycle_heartbeat_alarm`
  inside try/except for fail-open semantics.
- Call `cycle_heartbeat_alarm()`, gate on `should_alarm`, gate again on
  `_cycle_heartbeat_last_was_stale` state, then call
  `fire_cycle_heartbeat_alarm` if and only if entering the stale state.
- Recovery logged at INFO level; no P1.
- Steady-state logged at DEBUG level.

### `backend/tests/test_cycle_heartbeat_alarm.py`

7 deterministic test cases:

1. `test_fresh_cycle_on_weekday_no_alarm` -- 2h old, Tuesday ET -> no
   alarm.
2. `test_stale_26h_on_weekday_alarms` -- 27h old, Tuesday ET -> alarm.
3. `test_stale_26h_on_saturday_no_alarm` -- 30h old, Saturday ET ->
   stale True but `should_alarm: False` (weekend suppression).
4. `test_stale_30h_on_sunday_no_alarm` -- 30h old, Sunday ET ->
   stale True but `should_alarm: False`.
5. `test_missing_history_file_returns_sentinel` -- no file ->
   sentinel.
6. `test_empty_history_file_returns_sentinel` -- empty file ->
   sentinel.
7. `test_malformed_last_row_falls_back_to_prev` -- malformed last
   line + 1 valid earlier row -> uses the valid row.

Mocking strategy:
- `_HISTORY_PATH` patched via `monkeypatch.setattr` to `tmp_path`.
- `_now_utc` patched via `monkeypatch.setattr` to return a
  deterministic UTC datetime (no `freezegun` dep).
- `raise_cron_alert_sync` NOT exercised here -- these unit tests
  cover the pure verdict function, not the dispatch path. The
  dispatch path is fail-open by construction (verified by
  inspection); a Slack call could be added to a future integration
  test if desired.

## Verification

### Verification command (from masterplan phase-30.1)

```bash
grep -q 'cycle_heartbeat_alarm' backend/services/cycle_health.py && \
  grep -q 'cycle_heartbeat_alarm' backend/slack_bot/scheduler.py
```

Result: **exit 0** (both files contain the symbol).

### Test run

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_cycle_heartbeat_alarm.py -v
================================================ test session starts =================================================
platform darwin -- Python 3.14.4, pytest-9.0.3, pluggy-1.6.0
collected 7 items

backend/tests/test_cycle_heartbeat_alarm.py::test_fresh_cycle_on_weekday_no_alarm PASSED [ 14%]
backend/tests/test_cycle_heartbeat_alarm.py::test_stale_26h_on_weekday_alarms PASSED [ 28%]
backend/tests/test_cycle_heartbeat_alarm.py::test_stale_26h_on_saturday_no_alarm PASSED [ 42%]
backend/tests/test_cycle_heartbeat_alarm.py::test_stale_30h_on_sunday_no_alarm PASSED [ 57%]
backend/tests/test_cycle_heartbeat_alarm.py::test_missing_history_file_returns_sentinel PASSED [ 71%]
backend/tests/test_cycle_heartbeat_alarm.py::test_empty_history_file_returns_sentinel PASSED [ 85%]
backend/tests/test_cycle_heartbeat_alarm.py::test_malformed_last_row_falls_back_to_prev PASSED [100%]

================================================= 7 passed in 0.02s ==================================================
```

### Regression check

`python -m pytest backend/tests/test_observability.py -v` -> 12 passed,
1 warning (pre-existing genai deprecation; not caused by this cycle).

### Syntax check

`python -c "import ast; ast.parse(...)"` on all three edited files
returned `all syntax OK`.

## Hard guardrail attestation

- No mutating BigQuery calls -- this cycle adds a pure-Python file
  reader on an existing local JSONL artifact.
- No Alpaca calls -- the alarm dispatches to Slack only.
- No frontend / `.claude/` / `.mcp.json` touched.
- Diff stayed within audit's proposed-diff scope (modulo the
  `main.py` -> `slack_bot/scheduler.py` substitution justified by the
  research brief).
- Test ships and passes deterministically.

## Success criteria check

| Criterion | Status | Evidence |
|-----------|--------|----------|
| `cycle_heartbeat_alarm_function_defined_in_cycle_health` | PASS | `grep cycle_heartbeat_alarm backend/services/cycle_health.py` returns 2 hits (definition + helper) |
| `watchdog_cron_invokes_cycle_heartbeat_alarm` | PASS | `_watchdog_health_check` in `slack_bot/scheduler.py` calls `cycle_heartbeat_alarm()` after the backend-health probe block; verified via `grep` |
| `alarm_emits_p1_slack_when_no_cycle_in_last_26h_weekday` | PASS | `fire_cycle_heartbeat_alarm` calls `raise_cron_alert_sync(severity="P1", source="cycle_health", error_type="cycle_heartbeat_stale_weekday")`; gated on `should_alarm` which is `stale AND is_weekday_et` |
| `test_added_under_backend_tests` | PASS | `backend/tests/test_cycle_heartbeat_alarm.py` exists, 7 cases, all pass |
