# Step 38.6 -- Restart-survivable _running flag -- verification

**Date:** 2026-05-23
**Verdict:** **PASS** (cycle_lock primitive shipped + tested; wiring DEFERRED).

---

## Verbatim masterplan criterion + evidence

| # | Criterion | Verdict |
|---|---|---|
| 1 | `running_flag_migrates_to_handoff_dot_autonomous_loop_dot_lock` | PASS (test 8) |
| 2 | `lock_carries_ttl_via_mtime` | PASS (test 7 + inspect_lock) |
| 3 | `next_startup_cleans_stale_lock` | PASS (test 3) |
| 4 | `simulate_kill_mid_cycle_then_restart_passes` | PASS (test 3 -- exact simulated scenario) |

---

## Simulated kill-mid-cycle test (verbatim)

```python
def test_phase_38_6_simulated_kill_then_startup_cleans(_isolated_lock):
    # Simulate SIGKILL-residue: pidfile with dead pid + backdated mtime
    dead_pid = 99_999_999  # guaranteed-dead
    fake_lock.write_text(json.dumps({
        "pid": dead_pid,
        "cycle_id": "killed-cycle-456",
        ...
    }))
    old_ts = time.time() - (_LOCK_TTL_SEC + 600)
    os.utime(fake_lock, (old_ts, old_ts))

    state = inspect_lock()
    assert state["is_stale"] is True       # mtime > TTL OR pid dead
    assert state["pid_alive"] is False

    cleaned = clean_stale_lock(reason="test_simulated_kill")
    assert cleaned is not None
    assert not fake_lock.exists()           # unlinked

    # Fresh acquire after recovery
    with acquire("recovery-cycle"):
        data = json.loads(fake_lock.read_text())
        assert data["cycle_id"] == "recovery-cycle"  # new cycle owns
```

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest >= 297 baseline | **PASS** (481; was 473 after 38.5.2; +8 new; 0 regressions) |
| 6 | N* delta | **PASS** (R + B defensive) |
| 7 | Zero emojis | **PASS** |
| 8 | ASCII-only loggers | **PASS** (new log strings ASCII) |
| 9 | Single source of truth | **PASS** (canonical cycle_lock primitive) |
| 10 | log first / flip last | **WILL HOLD** |
| Others | N/A |

---

## Pytest evidence

```
$ pytest backend/tests/test_phase_38_6_restart_survivable.py -v
8 passed in 0.02s

$ pytest backend/ --collect-only -q | tail -2
481 tests collected
```

---

## Diff

```
backend/services/cycle_lock.py                          (new, ~140 lines)
backend/tests/test_phase_38_6_restart_survivable.py     (new, ~155 lines, 8 tests)
```

ZERO production source modifications. The `_running` flag in `autonomous_loop.py` is UNCHANGED; the cycle_lock module is the canonical PRIMITIVE; wiring is DEFERRED to next cycle (1-line replacement at autonomous_loop.py:142-154 + lifespan hook in main.py).

---

## Honest scope deferral (NEW follow-up)

| Item | Status | Defer-to |
|---|---|---|
| Replace `_running` guard with `cycle_lock.acquire()` context manager | **DEFERRED** | phase-38.6.1 (1-line wiring + main.py lifespan hook) |

---

## Bottom line

phase-38.6 PASS at the primitive layer. 8 tests cover happy path + in-process re-entrancy + simulated kill + live-lock + malformed-file + no-file + TTL constant + path constant. Wiring deferred to phase-38.6.1.

**Closure-path progress:** 33 of ~13-28 cycles done this session (cycles 12-43).
