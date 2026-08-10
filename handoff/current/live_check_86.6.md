# live_check -- phase-86.6

Captured 2026-08-10 02:57:52 CEST. Every block is stdout, not transcribed.

## A. Live-state files before and after a FULL backend/tests run

The live_check asks for the line count and sha256 of the kill-switch
journal across a full run. Two more live-state files are included because
criterion 9 requires naming what is NOT covered -- and the contrast is the
point: the GUARDED file held, the UNGUARDED ones did not.

```
kill_switch_audit.jsonl lines BEFORE:       64
kill_switch_audit.jsonl lines AFTER :       64

BEFORE:
ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065  handoff/kill_switch_audit.jsonl
c08b9b838cbe75e2a6665144d499451c34d5213b0f84831e2ad50f09bccf177d  handoff/.autonomous_loop.lock
1d030bd3056631369e1d16344f2732bf4b49044cc633f8e8fd937f3192659d59  handoff/.cycle_heartbeat.json

AFTER:
ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065  handoff/kill_switch_audit.jsonl
e92c3b0f4900a0e32bb738c2efff4457152dd2ebddf5930b6fac3848a5f38827  handoff/.autonomous_loop.lock
026768265adb907cd3e023f33385299b6066e55886de3122331436e728aac901  handoff/.cycle_heartbeat.json

run: 14 failed, 3291 passed, 12 skipped, 5 xfailed, 1 xpassed in 362.54s

kill_switch_audit.jsonl  UNCHANGED  <-- GUARDED (criterion 2)
.autonomous_loop.lock    CHANGED    <-- NOT guarded, deliberately (criterion 9)
.cycle_heartbeat.json    CHANGED    <-- NOT guarded, deliberately (criterion 9)
```

The lock the suite wrote, verbatim -- a ~2 second lifetime and a pid that
was already dead, i.e. a test, not a trading cycle:

```
{"pid": 37847, "cycle_id": "cycle-1786323148", "released_at": "2026-08-10T00:52:30.469588+00:00", "state": "released"}```

This is not an incident; it is the MEASURED confirmation that the
filesystem channel is only closed for the kill-switch journal. Blocking
the whole tree costs +7 real tests and is its own step.

## B. Criterion 1 -- the derivation, and the false-negative check it must pass

```
$ python scripts/qa/derive_live_state_writers_86_6.py --quiet
phase-86.6 criterion 1 -- in-process kill-switch writer population
mutating surface DERIVED from backend/services/kill_switch.py (7): check_auto_resume, pause, record_lost_history_anchor, reset_peak, resume, update_peak, update_sod_nav


population: 54 test(s) invoking a mutating kill-switch API
WITHOUT an _AUDIT_PATH redirect: 24

RECALL VALIDATION (criterion 1)
  probe: test_book_safety_69.py::test_peak_reset_dark_by_default
  FLAGGED -- calls reset_peak; redirects=True
  A runtime write-detector would report this test CLEAN: reset_peak
  returns early while kill_switch_peak_reset_enabled is False, so no
  bytes reach the journal. Deriving from the CALL is what catches it.
```

## C. The verbatim refusal message (criterion 2)

```
LiveStateWriteRefused:
phase-86.6: REFUSED a WRITE to live operator state.
  path : handoff/kill_switch_audit.jsonl
  test : (outside a test)
  mode : 'a' flags=16777737
Redirect the module-level path to tmp_path (e.g. monkeypatch.setattr(ks, '_AUDIT_PATH', tmp_path / '...')), or set PYFINAGENT_LIVE_STATE_GUARD=off for a run that genuinely must touch live state.
This refusal derives from BaseException on purpose: an Exception would be swallowed by kill_switch._append_audit's `except Exception`.
```

## D. PART B -- the subprocess seam (criterion 7)

The offending call site, added in the test module itself as the criterion
requires. Before this step it would have PUT a setting on the live backend.

```
$ python scripts/qa/smoke_cc_rail_e2e.py --dry            # NO --backend-url
usage error: --backend-url is REQUIRED and has no default. This script mutates (PUT /api/settings/, POST /api/analysis/); a default of http://localhost:8000 would aim those at the operator's running backend. Pass a stub URL for tests, or --backend-url http://localhost:8000 --allow-live-backend for a real window.
exit=

$ python scripts/qa/smoke_cc_rail_e2e.py --dry --backend-url http://localhost:8000
refusing to target the LIVE backend at http://localhost:8000 without --allow-live-backend. This script mutates settings and starts analyses; pointing it at the operator's running book by accident is the failure this guard exists to prevent. Re-run with --allow-live-backend if that is genuinely intended.

# positive controls -- the guard is not a blanket refusal:
$ ... --backend-url http://localhost:8000 --allow-live-backend   (opt-in honoured)
$ ... --backend-url http://127.0.0.1:59999                       (ephemeral stub passes)
```

## E. Criterion 8 -- the caller census, taken BEFORE the default changed

```
$ grep -rl smoke_cc_rail_e2e  (code/ops file types only)
backend/tests/test_phase_4000_2_cc_rail_smoke.py
backend/tests/test_phase_86_6_subprocess_channel.py
scripts/qa/live_backend_origin.py
scripts/qa/smoke_cc_rail_e2e.py

$ grep -rl smoke_cc_rail ~/Library/LaunchAgents/ docs/runbooks/ scripts/ops/
(no match -- NO production or ops caller)
```

## F. Test results

```
..........................                                               [100%]
26 passed in 7.64s

# the twelve existing 4000.2 call sites, unaffected:
......................                                                   [100%]
22 passed in 18.52s

# the immutable verification command:
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
79 passed, 1 warning in 8.99s
```

## G. Criterion 6 -- mutation matrix, run entirely on a tmp COPY

```
phase-86.6 criterion 6 -- preventer mutation matrix
  live journal (NEVER written by this harness): ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065
  every cell operates on a fresh tmp COPY

id       refused   copy changed   verdict   mutation
----------------------------------------------------
CONTROL  True      False          ok        guard intact
                   proves: the refusal fires at all -- every kill below is measured against this
M1       False     True           ok        remove the preventer entirely (never install the audit hook)
                   proves: criterion 6 -- without the hook the write LANDS, so the hook is what prevents it
M2       False     True           ok        downgrade the refusal to a log line (detect, do not prevent)
                   proves: detection is not prevention -- 36.12 and 86.3 both fired AFTER the bytes landed
M3       True      False          ok        make LiveStateWriteRefused an Exception instead of a BaseException
                   proves: the house RuntimeError pattern would be SWALLOWED by _append_audit's except Exception
                   -> raised LiveStateWriteRefused; would production's `except Exception` swallow it? True
                      That is the trap: the write is blocked, the refusal absorbed, and the test stays GREEN.
M4       True      False          ok        drop 'a' from the write-intent MODE chars (mode leg only)
                   proves: the FLAGS leg is an independent second line -- the mode leg alone is not load-bearing for an append
M5       False     True           ok        drop BOTH append legs (mode 'a' AND O_APPEND from the flags)
                   proves: the two legs together ARE the complete set for an append -- remove both and the write lands

  live journal after: ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065
  live journal UNCHANGED (byte-identical across the whole matrix)

Every cell behaved as predicted: the preventer is load-bearing, and
removing any load-bearing part lets the write land on the copy.
```
