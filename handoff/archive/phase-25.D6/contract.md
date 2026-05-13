# Sprint Contract -- phase-25.D6 -- Planner plateau-detection lock-file enforcement

**Cycle:** phase-25 cycle 21 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.D6
**Priority:** P1
**Audit basis:** bucket 24.6 F-5 -- 62-experiment plateau bypassed planner Rule 1 (strategy-switch)

## Research-gate

Researcher spawned this cycle (agent af203a7172a66c5f6). Brief at
`handoff/current/research_brief.md`. Gate envelope: 6 sources read in full,
16 URLs, recency scan performed, 6 internal files inspected, gate_passed=true.

Key research conclusions:
- **Threshold N=10** -- Keras `ReduceLROnPlateau` patience=10 + Optax patience range 5-15 consensus. Also matches the live-check ("trigger 10 consecutive discards"). Second tier above the existing `think_harder >= 5` at `quant_optimizer.py:205`.
- **File-based lock, not in-memory** -- a server crash resets in-memory counters but a file persists. Operators can observe via filesystem without hitting the API.
- **Lock path:** `handoff/locks/optimizer_plateau.lock` (consistent with existing `handoff/` layout).
- **HTTP 409** is the correct status for "operator action required to clear lock" -- matches existing 409 use at `backtest.py:239`.
- **DELETE /api/backtest/optimize/lock** is the canonical operator-action endpoint pattern (mirrors `DELETE /optimize/history` at `backtest.py:665-712`).
- **JSONL audit on clear** at `handoff/audit/optimizer_plateau_audit.jsonl` (consistent with `kill_switch_audit.jsonl` pattern).

## Hypothesis

Adding (a) a `PLATEAU_THRESHOLD=10` constant + `write_plateau_lock()` helper
in `quant_optimizer.py` that fires when `consecutive_discards >= 10` and
breaks out of the loop, (b) a `_read_plateau_lock()` check in
`backend/api/backtest.py::start_optimizer` that returns 409 when the lock
is present, and (c) a `DELETE /api/backtest/optimize/lock` operator-clear
endpoint that appends an audit row before deleting -- prevents the
62-experiment plateau recurrence in a way that survives server restarts.

## Success criteria (verbatim from masterplan)

1. `lock_file_strategy_switch_required_written_on_plateau`
2. `optimizer_run_endpoint_returns_409_when_lock_present`
3. `operator_action_clears_lock`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_D6.py`

Live check (per masterplan):
`Trigger 10 consecutive discards; confirm lock file created and next run blocked`

## Plan

1. **`backend/backtest/quant_optimizer.py`** edits:
   - Add `PLATEAU_THRESHOLD: int = 10` and `_PLATEAU_LOCK_PATH = Path(__file__).parent.parent.parent / "handoff" / "locks" / "optimizer_plateau.lock"` constants.
   - Add `json, datetime, timezone` imports if absent.
   - Add module-level `write_plateau_lock(run_id: str, consecutive_discards: int) -> None` helper that:
     - Creates parent dir.
     - Writes JSON `{created_at, trigger: "plateau_N_discards", consecutive_discards, run_id, cleared_at: null}`.
     - Logs a WARNING with the operator-action hint.
   - Inside `run_loop`, at every site that increments `consecutive_discards` (crash at line 246, dsr_reject at 287, discard at 294), check `if consecutive_discards >= PLATEAU_THRESHOLD: write_plateau_lock(self._run_id, consecutive_discards); break` (or `return` for the crash branch; the break path will log the final experiment first per the brief).
2. **`backend/api/backtest.py`** edits:
   - Add module-level helpers `_plateau_lock_path() -> Path` and `_read_plateau_lock() -> dict | None`. The reader treats `cleared_at != null` and `JSONDecodeError` as absent (fail-open).
   - In `start_optimizer`, BEFORE the `_optimizer_state = {...}` reset (after existing 400/409 guards), check `plateau_lock = _read_plateau_lock()`. If non-None, `raise HTTPException(409, detail={error, message, lock})` with the operator-action message.
   - Add a new `DELETE /api/backtest/optimize/lock` route `clear_plateau_lock()` that:
     - Returns 404 if no lock file.
     - Reads existing payload, sets `cleared_at`, appends to `handoff/audit/optimizer_plateau_audit.jsonl`, deletes the file.
     - Returns `{status: "cleared", lock: payload}`.
3. **Verifier** -- `tests/verify_phase_25_D6.py` -- 10+ claims:
   - Claim 1: `PLATEAU_THRESHOLD = 10` constant in `quant_optimizer.py`.
   - Claim 2: `write_plateau_lock(run_id: str, consecutive_discards: int) -> None` signature exists.
   - Claim 3: `_read_plateau_lock` + `_plateau_lock_path` helpers exist in `backend/api/backtest.py`.
   - Claim 4: `start_optimizer` guards on `_read_plateau_lock()` with 409 + `PlateauLockPresent` error key.
   - Claim 5: `DELETE /api/backtest/optimize/lock` route registered.
   - Claim 6: **Behavioral lock write** -- call `write_plateau_lock("run_xyz", 12)` with a temp path patched in; assert file exists, parses to JSON with expected keys + correct trigger string.
   - Claim 7: **Behavioral lock-present 409** -- patch a fake lock file present; call `start_optimizer(...)`; assert `HTTPException(409)` with `detail["error"] == "PlateauLockPresent"` and `lock` field carries the read payload.
   - Claim 8: **Behavioral clear lock** -- write a fake lock, call `clear_plateau_lock`; assert file removed + audit JSONL has one row with `cleared_at` set.
   - Claim 9: **Behavioral cleared-at semantic** -- write a lock with `cleared_at` already set; `_read_plateau_lock()` returns None (treats as absent).
   - Claim 10: **Behavioral corrupt-lock fail-open** -- write malformed JSON to lock path; `_read_plateau_lock()` returns None and does not crash.
   - Claim 11: **Behavioral 404 when no lock** -- call `clear_plateau_lock` with no file; HTTPException 404 raised.

## Non-goals

- No frontend changes.
- No automatic lock-clear via Slack/HITL (operator manually calls DELETE).
- No change to the existing `think_harder >= 5` second-tier; it remains a softer signal below N=10.
- No new BQ schema.

## References

- `handoff/current/research_brief.md` -- full brief this cycle
- `backend/backtest/quant_optimizer.py:190-294` -- existing `consecutive_discards` loop
- `backend/api/backtest.py:231-258` -- start_optimizer endpoint
- `backend/api/backtest.py:665-712` -- canonical DELETE-operator-action pattern
- `handoff/kill_switch_audit.jsonl` -- canonical audit JSONL pattern
- CLAUDE.md `Critical Rules` -- ASCII-only logger messages
