---
step: phase-25.D6
cycle: 77
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_D6.py'
title: Planner plateau-detection lock-file enforcement (P1)
audit_basis: phase-24.6 F-5 (62-experiment plateau bypassed planner Rule 1 strategy-switch)
---

# Experiment Results -- phase-25.D6

## Code changes

### `backend/backtest/quant_optimizer.py`
- New constants `PLATEAU_THRESHOLD: int = 10` and `_PLATEAU_LOCK_PATH = .../handoff/locks/optimizer_plateau.lock`.
- New module-level `write_plateau_lock(run_id, consecutive_discards) -> None` helper -- writes JSON `{created_at, trigger: "plateau_N_discards", consecutive_discards, run_id, cleared_at: null}`, creates parent dir, logs WARNING with the operator-action hint.
- Inside `run_loop`, two plateau-check sites (one after the bottom-of-loop log, one inside the crash branch before `continue`) -- when `consecutive_discards >= PLATEAU_THRESHOLD`, write the lock and `break` out of the loop. The crash branch can't bypass the fence.
- Status step "plateau_locked" emitted on lock fire so the UI can render the halted state.

### `backend/api/backtest.py`
- New helpers `_plateau_lock_path() -> Path` and `_read_plateau_lock() -> dict | None`. Reader is fail-open: corrupt JSON, missing file, or `cleared_at != null` all return None.
- `start_optimizer` guarded -- after existing 400/409 checks, reads the lock; if present, raises `HTTPException(409, detail={"error": "PlateauLockPresent", "message": ..., "lock": payload})`.
- New `DELETE /api/backtest/optimize/lock` route `clear_plateau_lock()` -- 404 if absent; otherwise reads payload, sets `cleared_at`, appends to `handoff/audit/optimizer_plateau_audit.jsonl`, removes the lock file, returns `{status: "cleared", lock: payload}`.

### `tests/verify_phase_25_D6.py` (new file)
- 11 immutable claims with 6 behavioral round-trips:
  - Claims 1-5: structural (threshold, helper signatures, 409 guard, DELETE route).
  - Claim 6: **Behavioral write** -- `write_plateau_lock("run_xyz123", 12)` with temp path -> file exists, JSON shape verified (`consecutive_discards=12`, `run_id`, `trigger="plateau_12_discards"`, `cleared_at: null`, `created_at` set).
  - Claim 7: **Behavioral 409** -- seed a fake lock; `start_optimizer` raises `HTTPException(409)` with `detail.error="PlateauLockPresent"` AND `detail.lock.run_id` matches the payload.
  - Claim 8: **Behavioral clear** -- write fake lock; `clear_plateau_lock()` returns `{status: "cleared", ...}`; file removed; audit JSONL has one row with `cleared_at` set.
  - Claim 9: **Behavioral cleared sentinel** -- lock file with `cleared_at != null` -> `_read_plateau_lock` returns None (treats as absent).
  - Claim 10: **Behavioral corrupt fail-open** -- malformed JSON in lock file -> `_read_plateau_lock` returns None (no exception).
  - Claim 11: **Behavioral 404** -- clear with no lock file -> `HTTPException(404)`.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D6.py
QuantOptimizer: plateau lock written after 12 consecutive discards (run_id=run_xyz123). Operator must DELETE /api/backtest/optimize/lock to acknowledge and resume.
plateau lock file at /var/folders/.../phase25d6_corrupt_ts2tgf8g/optimizer_plateau.lock is corrupt; treating as absent
PASS: plateau_threshold_constant_10
PASS: lock_file_strategy_switch_required_written_on_plateau
PASS: api_backtest_has_plateau_helpers
PASS: optimizer_run_endpoint_returns_409_when_lock_present
PASS: operator_action_clears_lock
PASS: behavioral_write_plateau_lock_emits_correct_json
PASS: behavioral_409_when_lock_present_with_detail_payload
PASS: behavioral_clear_lock_removes_file_and_audits
PASS: behavioral_cleared_lock_treated_as_absent
PASS: behavioral_corrupt_lock_fail_open
PASS: behavioral_clear_raises_404_when_no_lock

11/11 claims PASS, 0 FAIL
```

(The `write_plateau_lock` WARNING and the `plateau lock file ... is corrupt; treating as absent` log are emitted by the behavioral tests -- they prove the fail-open log paths actually run as designed.)

## Backend gates

- `python -c "import ast; ast.parse(open('backend/backtest/quant_optimizer.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/api/backtest.py').read())"` -- OK
- 6 behavioral round-trips exercise actual functions with `unittest.mock.patch` and temp filesystems.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`lock_file_strategy_switch_required_written_on_plateau`) -- claims 2 + 6 (signature + behavioral JSON shape).
- Criterion 2 (`optimizer_run_endpoint_returns_409_when_lock_present`) -- claim 4 (structural grep) + claim 7 (behavioral 409 with PlateauLockPresent error key).
- Criterion 3 (`operator_action_clears_lock`) -- claim 5 (DELETE route registered) + claims 8 + 11 (behavioral: clear works + 404 when absent).

The 62-experiment plateau cannot recur: after N=10 consecutive discards/dsr_rejects/crashes (matching the Keras `ReduceLROnPlateau` default; second tier above the existing `think_harder >= 5` softer signal), the loop halts and persists a lock-file that survives backend restarts. Operators must explicitly acknowledge via `DELETE /api/backtest/optimize/lock` to resume.

## Live-check

Per masterplan: "Trigger 10 consecutive discards; confirm lock file created and next run blocked".

Live evidence pending in `handoff/current/live_check_25.D6.md`. After triggering 10 consecutive bad experiments (a synthetic stress test will work):
- `handoff/locks/optimizer_plateau.lock` is present with the canonical JSON shape.
- `POST /api/backtest/optimize` returns 409 + `detail.error="PlateauLockPresent"`.
- `DELETE /api/backtest/optimize/lock` returns `{status: "cleared", ...}` + appends audit JSONL.

## Non-regressions

- No change to the `think_harder >= 5` softer signal -- preserved as the first tier.
- No frontend changes (endpoint additions only; `get_optimizer_status` shape unchanged this cycle; can be extended later).
- Existing 400/409 guards in `start_optimizer` preserved before the new lock guard.
- Fail-open semantics consistent with existing alerting + state patterns.

## Next phase

Q/A pending.
