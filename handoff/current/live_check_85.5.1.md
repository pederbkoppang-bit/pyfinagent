# live_check — phase-85.5.1

Required evidence shape (from `.claude/masterplan.json`, `verification.live_check`):

> live_check_85.5.1.md carrying the verbatim measurement that answers criterion 1
> (the production reachability of a None/stale daily anchor), the before/after
> output of the verification command, and the mutation transcript from criterion 4

All three below, verbatim. Captured 2026-08-09 ~02:00-02:45 CEST.

---

## 1. Criterion 1 — the measurement, verbatim

`scripts/diagnostics/measure_sod_date_reachability.py` reaches each candidate
state **through the production `KillSwitchState` replay** and evaluates a **real**
`evaluate_breach` against it. Every case uses a throwaway journal in a temp dir,
and the script **asserts its own isolation** (including that
`_audit_source_paths()` contains no live file) before measuring anything.

```
kill_switch DISARMED: daily anchor STALE (sod_nav=100.0 sod_date='2026-08-07' peak_nav=100.0) -- an unevaluable leg cannot fire. Sources replayed: ['/var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/tmpqlc0ob5q/kill_switch_audit.jsonl']
kill_switch: audit load failed: int too large to convert to float
phase-85.5.1 -- production reachability of a None/STALE daily anchor
measured 2026-08-08T22:42:19.395794+00:00  limits: daily 4.0% / trailing 10.0%
Every case replays an ISOLATED journal through the real KillSwitchState.

============================================================================
CASE C -- UTC ROLLOVER (yesterday's anchor, before today's first roll)
  THE SEVERITY DRIVER. Reachable every UTC day with no fault at all.
  replayed snapshot : sod_nav=100.0 sod_date='2026-08-07' peak_nav=100.0
  current_nav       : 80.0   (drop vs sod: 20.0%)
  armed             : False
  daily_baseline_missing=False daily_baseline_stale=True
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : True  (20.0%)
  >>> any_breached      : True

============================================================================
CASE A/B -- LEGACY ROW (no `date` key, unparseable `ts`)
  Mechanism is live; no row of this shape exists in the live journal.
  replayed snapshot : sod_nav=100.0 sod_date=None peak_nav=100.0
  current_nav       : 80.0   (drop vs sod: 20.0%)
  armed             : False
  daily_baseline_missing=False daily_baseline_stale=True
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : True  (20.0%)
  >>> any_breached      : True

============================================================================
CASE F -- STARTUP, no anchor has ever been written
  Absence is named `missing`, NOT `stale` (kill_switch.py:922-923).
  replayed snapshot : sod_nav=None sod_date=None peak_nav=100.0
  current_nav       : 80.0   (drop vs sod: None%)
  armed             : False
  daily_baseline_missing=True daily_baseline_stale=False
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : True  (20.0%)
  >>> any_breached      : True

============================================================================
CASE E -- OVERSIZED INT aborts the entire audit replay (NEW, found by the gate)
  _coerce_nav catches only (TypeError, ValueError); OverflowError is swallowed at :394 and aborts the replay, so every LATER row is lost.
  replayed snapshot : sod_nav=None sod_date=None peak_nav=None
  current_nav       : 80.0   (drop vs sod: None%)
  armed             : False
  daily_baseline_missing=True daily_baseline_stale=False
  daily_loss_breached   : False  (0.0%)
  trailing_dd_breached  : False  (0.0%)
  >>> any_breached      : False

============================================================================
CASE HEALTHY CONTROL -- same-day anchor, a real 20% drawdown
  If this does NOT breach on both legs, the measurement above is meaningless -- the guard would be broken for every state.
  replayed snapshot : sod_nav=100.0 sod_date='2026-08-08' peak_nav=100.0
  current_nav       : 80.0   (drop vs sod: 20.0%)
  armed             : True
  daily_baseline_missing=False daily_baseline_stale=False
  daily_loss_breached   : True  (20.0%)
  trailing_dd_breached  : True  (20.0%)
  >>> any_breached      : True

============================================================================
VERDICT
  C  UTC rollover     : sod_date STALE, daily leg DISARMED, any_breached=True  <-- reachable EVERY DAY
  A/B legacy row      : sod_date=None, any_breached=True
  F  startup          : missing=True stale=False, any_breached=True
  E  oversized int    : sod_nav=None peak=None, any_breached=False  <-- BOTH legs stranded
  HEALTHY control     : daily=True trailing=True any=True

  ANSWER TO CRITERION 1: YES -- production CAN reach a None/stale
  sod_date, and case C needs no fault at all. Exposure is bounded to
  drawdowns in [daily_limit, trailing_limit) because the trailing leg
  is date-independent -- EXCEPT in case E, where the same fault strands
  the peak too and NOTHING fires for any drawdown.
```

### The answer

**YES — production CAN reach a None/stale `sod_date`.** The severity driver is
**case C, the UTC rollover: reachable every single day with no fault at all.**

But the guard is behaving **correctly** in every case. Exposure is bounded to
drawdowns in **[4%, 10%)** because the trailing leg is date-independent and still
fires — `any_breached` is True in cases C, A/B and F. **Except case E**, where
one malformed row aborts the whole replay and strands *both* legs
(`any_breached=False` on a 20% drawdown). That is a separate defect and is queued,
not absorbed.

## 2. Before / after — the immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_book_safety_69.py -q --timeout=120'

BEFORE (at HEAD, pre-fix):
ERROR backend.services.kill_switch:kill_switch.py:956 kill_switch DISARMED: daily anchor
  STALE (sod_nav=100.0 sod_date=None peak_nav=100.0) -- an unevaluable leg cannot fire.
FAILED backend/tests/test_book_safety_69.py::test_valid_nav_still_breaches
1 failed, 13 passed in 1.47s

AFTER:
15 passed in 1.44s
```

14 → 15 tests because this step added the missing disarm-side test (§3).

## 3. Criterion 4 — the mutation transcript

```
$ source .venv/bin/activate && python scripts/qa/mutation_matrix_85_5_1.py
precondition OK -- baseline green

[KILLED] M1 CRITERION 4: restore the 2-key mock -- the RED test must go RED again   1 failed
[KILLED] M1b the anchor date is hardcoded instead of computed (rots at midnight)    1 failed
[KILLED] M2 the daily-loss leg stops firing at all                                  1 failed
[KILLED] M3 the trailing-DD leg stops firing at all                                 1 failed
[KILLED] M4 the staleness guard stops disarming (phase-36.9 F1 regression)   1 failed, 14 passed

live kill-switch journal untouched (54 lines before and after)

MUTATION MATRIX PASSED -- 5/5 killed, tree restored byte-for-byte, suite green,
live journal untouched.
```

**M4 was LIVE on its first run**, and that was the most useful result: nothing in
`test_book_safety_69.py` could tell a working liveness guard from a disabled one.
A file that proves the switch FIRES but never that it correctly DISARMS is half a
book-safety suite. Fixed by adding
`test_stale_anchor_disarms_the_daily_leg_but_the_trailing_leg_still_fires`.

**M1's target became ambiguous** once that test existed (the same two construction
lines appear twice), and the harness **refused to mutate** rather than silently
picking a site — then M1 was re-anchored on a line unique to the RED test.

## 4. Criterion 5 — no other test changed status, measured as a SET

Measured in a **detached git worktree**, because a full-suite run in the live tree
writes to `handoff/kill_switch_audit.jsonl` and `handoff/.cycle_heartbeat.json`
(the phase-36.28 class). All four polluting constants derive from
`Path(__file__).parents[N]`, so one worktree relocates them all.

**Isolation proven before measuring, not assumed:**

```
OK   kill_switch._AUDIT_PATH        = <WT>/handoff/kill_switch_audit.jsonl
OK   cycle_health._HISTORY_PATH     = <WT>/handoff/cycle_history.jsonl
OK   cycle_health._HEARTBEAT_PATH   = <WT>/handoff/.cycle_heartbeat.json
OK   cycle_lock._LOCK_PATH          = <WT>/handoff/.autonomous_loop.lock
ALL FOUR RELOCATED -- the baseline run cannot touch live state
```

Both arms ran in the **same worktree**, one variable (the test file reverted to
`ebc1e172^` for the BEFORE arm):

```
BEFORE (no fix): 20 FAILED, 4 ERROR   -- 20 failed, 3037 passed, 22 skipped, 5 xfailed
AFTER  (fix)   : 19 FAILED, 4 ERROR   -- 19 failed, 3039 passed, 22 skipped, 5 xfailed

FIXED by this step (in BEFORE, not in AFTER):
  - backend/tests/test_book_safety_69.py::test_valid_nav_still_breaches

NEWLY FAILING (in AFTER, not in BEFORE) -- MUST BE EMPTY:
  (none)

ERROR set identical: True  (4 both sides)
Unchanged failures : 19

CRITERION 5 SET DIFF: PASS
```

**Why the worktree counts (19/20) differ from the live tree's 26:** the worktree
lacks gitignored live files — most importantly `backend.log` (32.5MB) — so
several tests skip or fail differently there. That is precisely why the
comparison is **worktree-vs-worktree**, never worktree-vs-live: the phase-85.5
cycle already lost time to reading that environment difference as a regression.
The claim being made is *"exactly one test changed status, and it is the target"*,
which is measured as a **set** in a single environment.

## 5. The live journal was never touched

```
handoff/kill_switch_audit.jsonl: 54 lines before and after EVERY run in this step
  - the reachability script      : 54 -> 54
  - the scoped pytest runs       : 54 -> 54
  - the mutation matrix          : 54 -> 54 (asserted by the matrix itself)
  - both full-suite arms         : ran in the worktree, live paths never loaded
```

The mutation matrix now **asserts** this as a post-condition, so a future run that
pollutes the journal fails loudly instead of quietly.
