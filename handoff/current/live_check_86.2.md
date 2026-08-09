# live_check — phase-86.2

Required shape (masterplan `86.2.verification.live_check`): *"live_check_86.2.md
with the verbatim before/after output of measure_sod_date_reachability.py case E,
and the verbatim mutation transcript."*

---

## Part 1 — case E, BEFORE (captured before any change)

```
CASE E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)
  _coerce_nav catches only (TypeError, ValueError); OverflowError is swallowed at :394 and aborts the replay, so every LATER row is lost.
  replayed snapshot : sod_nav=None sod_date=None peak_nav=None
  current_nav       : 80.0   (drop vs sod: None%)
  armed             : False
  daily_baseline_missing=True daily_baseline_stale=False
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : False  (0.0%)
  >>> any_breached      : False
```

**A 20% drawdown and nothing fires.** The only measured path to a TOTAL disarm.

## Part 2 — case E, AFTER

```
CASE E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)
  _coerce_nav catches only (TypeError, ValueError); OverflowError is swallowed at :394 and aborts the replay, so every LATER row is lost.
  replayed snapshot : sod_nav=100.0 sod_date='2026-08-09' peak_nav=100.0
  current_nav       : 80.0   (drop vs sod: 20.0%)
  armed             : True
  daily_baseline_missing=False daily_baseline_stale=False
  daily_loss_breached   : True  (20.0%)
  trailing_dd_breached  : True  (20.0%)
  >>> any_breached      : True
```

**Both legs fire.** Note the case-E *note string* still describes the pre-fix
mechanism — it is the diagnostic's narration, now stale. **Deliberately not
edited**: this script is the step's immutable verification command, and changing
its text mid-step is exactly the kind of quiet edit the criteria exist to
prevent. Flagged here so a reader is not misled by it.

## Part 3 — the mutation transcript, verbatim

Both guards reverted, against an isolated module copy (the real module is never
mutated — that would leak into every other test in the session):

```
kill_switch: audit load FAILED OUTSIDE any single row -- history marked INCOMPLETE, baselines may be partial: OverflowError: int too large to convert to float
MUTANT (both guards reverted):
  sod_nav=None sod_date=None peak_nav=90.0
SHIPPED (same rows):
  sod_nav=100.0 sod_date=2026-08-09 peak_nav=100.0  _history_complete=False

MUTANT KILLED: True
```

**Read `peak_nav=90.0` on the mutant carefully — that is the hazard itself.**
The row *before* the poison applied, then the replay abandoned everything after
it and returned normally. Not "failed to load", but **loaded a partial state and
reported success**: the third state that PostgreSQL redo (which PANICs) and
Kafka Connect (`errors.tolerance=none`) both refuse to enter. The shipped
version applies every well-formed row and marks `_history_complete=False`.

## 4. Criteria deviations — for Q/A to adjudicate, NOT self-cleared

- **Criterion 4's literal wording no longer holds, and the reason is that the
  fix is better than it assumed.** Reverting *only* the widened `except` does
  **not** re-strand: the new per-row `try` contains the fault, skips the row and
  continues. **The single-guard mutant survives by design.** Criterion 4 is met
  by reverting **both** guards (above), and the defence-in-depth property is
  pinned by `test_c4_reverting_only_the_except_is_now_HARMLESS` so a future
  removal of the per-row `try` goes red.
- **Criterion 1's ordering was breached.** It asks for the red test *first*. The
  RED state was captured pre-change — but from the diagnostic, and the gate then
  established the diagnostic **prints** case E without **asserting** it (it exits
  0 with the defect fully present). So it was a witness, not a test. The pytest
  file was written **after** the fix. The both-guards mutation is the evidence it
  genuinely fails without the fix. **No automated check would have caught this
  ordering** — mtime on the diagnostic is unchanged, and the commit carries no
  intra-day ordering.

## 5. Limits

- Live journal `90e0303130fc546df82e33fe1ebb7c782efd75d74e3b7877e16f76fcdbddf653`,
  **62 lines, unchanged** across every run here.
- `update_peak:611/633` and `reset_peak:697` still coerce with a bare
  `float(nav)`, so the same `OverflowError` is reachable from the **in-memory**
  path. **Out of scope by explicit decision** (contract §5) — that is step
  `36.19`, which exists and is pending.
- `ruff` on `kill_switch.py`: 33 → 34 findings, the +1 a `BLE001` for the new
  per-row handler, deliberate and matching the file's existing idiom (it already
  carries 11). The mandated gate set (`F821,F401,F811,E9`) is clean.
