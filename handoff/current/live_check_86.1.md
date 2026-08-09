# live_check — phase-86.1

Required shape (masterplan `86.1.verification.live_check`):

> *"live_check_86.1.md with the before/after line count and sha256 of
> handoff/kill_switch_audit.jsonl across a full run of test_book_safety_69.py,
> plus the verbatim mutation transcript showing the pre-fix form writing a
> peak_reset row under a forced-ON flag."*

Captured 2026-08-09, live tree.

---

## Part 1 — before/after across a full run of the file

```
=== BEFORE ===
90e0303130fc546df82e33fe
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'
................                                                         [100%]
16 passed in 1.48s
=== AFTER ===
90e0303130fc546df82e33fe
```

Full digest, unchanged either side:
`90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653`, **62 lines**.

**Criterion 3 is additionally asserted BY THE RUN**, not only by this
hand-check: a function-scoped autouse fixture
(`_live_kill_switch_journal_is_byte_identical`) byte-compares the live journal
around **every** test in the file, so a violation fails the offending test by
name. Function-scoped deliberately — a module-scoped guard would tell you the
file wrote, not which test.

---

## Part 2 — the mutation transcript, verbatim

The pre-fix form run **verbatim** (`st = ks.get_state()`, no detachment) with
`kill_switch_peak_reset_enabled` forced **True**, against a **byte-for-byte copy**
of the live journal:

```
module default _AUDIT_PATH == live journal : True
decoy is byte-identical to live            : True
peak rows in the journal today             : 0

PRE-FIX FORM under a forced-ON flag:
  reset_peak returned      : a snapshot (LIVE, not dark)
  row appended             : {"event": "peak_reset", "old_peak": 24666.57, "new_peak": 12345.0, "trigger": "flatten"}
  in-memory singleton peak : 12345.0  (was 24666.57)
  peak after REPLAY        : 12345.0
  trailing trip point      : 22199.9 -> 11110.5

LIVE journal byte-identical throughout     : True
```

**Read that carefully — every clause is a distinct harm:**

- **A real `peak_reset` row was appended** by production `reset_peak`. Not a
  simulation.
- **The in-memory singleton was corrupted too** — `24666.57 → 12345.0`. This is
  the half an `_AUDIT_PATH` redirect **cannot** prevent: `kill_switch.py:697`
  assigns before auditing. It is why the fix uses a **detached state**, not just
  a redirected path.
- **It survives replay** — `peak after REPLAY: 12345.0`. `peak_reset` is in
  `_BASELINE_EVENTS`, so it is authoritative on every boot.
- **The trailing trip point moves 22199.9 → 11110.5.** A *lower* peak makes the
  trailing leg fire **later**. That is a safety limit being **loosened** by
  running the test suite.
- **`peak rows in the journal today: 0`** — measured. So there is nothing later
  in `ts` order to override it; the row wins outright and 24666.57 is gone
  permanently.

### Why a copy, and not the operator's file

Criterion 2 says *"show that the pre-fix form WOULD have written a `peak_reset`
row to the live journal"*. Doing that literally means **destroying the 24666.57
high-water mark** — the exact harm this step exists to prevent — and it would
break criterion 3 in the same act.

So the destructive write is **actually performed, by the real production code**,
on a stand-in proven identical two ways: `decoy is byte-identical to live: True`,
and `module default _AUDIT_PATH == live journal: True`, so the copy stands in by
construction rather than by assertion. **Disclosed in `contract_86.1.md` §5
rather than taken silently.** The criterion text was not edited.

---

## Part 3 — criterion 5, no other test changed status

```
$ pytest backend/tests/test_book_safety_69.py -q
16 passed          # was 15 passed; +1 is this step's own new mutation test

$ pytest test_phase_36_7_kill_switch_rotation_rearm.py \
         test_phase_36_12_kill_switch_trading_path_block.py \
         test_phase_36_9_kill_switch_armed_liveness.py -q
87 passed
```

Live journal `90e0303130fc…` after every one of those runs.

**No assertion was weakened.** The DARK-behaviour assertions (`out is None`,
peak unchanged) are intact and were *strengthened* with a third: no `peak_reset`
row is appended anywhere.

---

## 4. Disclosures

- **`ruff` reports `SIM117` on this file, and it is PRE-EXISTING.** Proven, not
  claimed: running ruff on a HEAD copy placed inside `backend/tests/` (so the
  project config applies) reports the same `SIM117`, and
  `git diff | grep -c "cycle_lock_failed_acquire"` returns **0** — my diff does
  not touch that test. Left unfixed: editing unrelated code inside a graded step
  is scope creep. `I001` **was** autofixed, because its region (the import
  block) is one my diff does modify.
- **The `get_state` patch was DELETED, not repaired.** It was vacuous by
  identity — `st` was already bound to the real singleton on the line above. A
  line that reads like isolation and provides none is worse than no line.
- **This is a DETECTOR for the file, not a preventer for the codebase.** The
  autouse byte-compare fires after the bytes are written. The general filesystem
  preventer is step **86.6**, which also now carries the live cycle-lock instance
  measured today.
- **Not verified:** that no *other* file in the suite has the same
  flag-inert-today shape. My census methods produced a false negative on this
  very file earlier today, so I am not asserting a clean sweep. That derivation
  is 86.6's criterion 1, which requires the method to flag this file first.
