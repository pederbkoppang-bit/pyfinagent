# Experiment results — masterplan step 36.9

**[P0] `armed:true` must mean "this leg can actually fire NOW".** Three ways it didn't.

## What shipped

| File | Change |
|---|---|
| `backend/services/kill_switch.py` | **C2** new `baselines_present` key in BOTH return shapes (presence only -- the pre-36.9 meaning of `armed`); **F1** new `_sod_date_is_stale(sod_date, sod_nav)` + `daily_baseline_stale` folded into `armed` and gating the daily math; **F2** the `nav_invalid` early return now emits `armed: False` (+ `nav_invalid_disarmed`); **F3** `update_sod_nav` refuses a non-positive/non-finite anchor via `_coerce_nav` instead of `float(nav)`; the disarm ERROR log now names staleness instead of asserting absence; `date` imported as `_date` (the method parameter `date` would shadow it) |
| `backend/services/paper_trader.py` | **C2** the 36.12 order gate at :1126 now reads `baselines_present`, not `armed`; **F3** the daily-roll predicate extracted to module-level `sod_anchor_needs_reroll(snap, today)` and corrected to treat `<= 0` as absent; the call site now calls it |
| `backend/tests/test_phase_36_9_kill_switch_armed_liveness.py` | NEW — 28 tests (24 in cycle 1, +4 order-path guards in cycle 2) |
| `backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py` | 7 pre-existing literal date fixtures made relative (HEAD had exactly 7; the amended test adds an 8th `TODAY_UTC` anchor of its own, so the file now greps 8); one 36.7 guard AMENDED (below) |
| `backend/tests/test_phase_23_2_5_kill_switch_no_false_fires.py` | 4 date fixtures made relative |

**Behaviours changed, stated as behaviours** (naming files is not describing behaviour):
1. A daily anchor whose `sod_date` is not today no longer produces a percentage, no longer reports a daily breach, and reports `armed: false` — on GET `/kill-switch`, POST `/resume`, and the MCP risk tool.
   **CORRECTED in cycle 2.** This item originally ended *“The autonomous cycle is unaffected because it re-anchors first.”* That was **measured false** and is withdrawn: the cycle's 36.12 gate reads the flag BEFORE the roll, so staleness reached it and halted the ordinary morning cycle. See the cycle-2 section. The cycle is unaffected *now*, but only because the gate was changed to read `baselines_present`.
2. The **trailing** leg is untouched by staleness and still fires. Protection is not reduced; only the daily leg's false claim is withdrawn.
3. An unmeasurable NAV now reports `armed: false` instead of `armed: true` alongside `any_breached: false`.
4. A non-positive/non-finite start-of-day NAV is refused: no state mutation, no `sod_snapshot` audit row, an ERROR log. `sod_nav` stays `None`, so the next cycle genuinely re-anchors.
5. The daily-roll predicate treats `0.0` as absent, so a book that latched `0.0` before this fix repairs itself on the next cycle rather than waiting for a restart.
6. The disarm log names the actual cause; it previously printed `baseline missing (sod_nav=23838.19 …)` — a number it had just called missing.

## Criterion 1 & 2 — the defects reproduced against pre-fix code

Each fix reverted individually, in memory, then only that finding's tests run. Full capture:
`handoff/current/captures_36.9/prefix_reproductions.txt`.

```
PRE-FIX -- F1 stale sod_date
>       assert r["daily_baseline_stale"] is True
E       assert False is True
1 failed, 23 deselected

PRE-FIX -- F2 nav_invalid reports armed:true
>       assert r["armed"] is False, (
E       AssertionError: a leg that cannot measure cannot fire -- unknown is not healthy
E       assert True is False
3 failed, 21 deselected

PRE-FIX -- F3 sod_nav=0.0 latches
>       assert snap["sod_nav"] is None, f"{bad!r} must not latch as a baseline"
E       AssertionError: 0.0 must not latch as a baseline
E       assert 0.0 is None
```

## Criterion 5 — mutation matrix, one batch at the final baseline

`handoff/current/captures_36.9/mutation_matrix.txt` — **15 mutants, 15 killed, 0 survived**,
baseline `28 passed` (13 in cycle 1 at baseline 24; +2 in cycle 2 pinning the `baselines_present` split from both sides). Each mutation asserts its pattern matched EXACTLY once and that the text
changed, so an inert edit cannot be misread as a survivor.

Two survivors in the first run, both **defects in my own tests**, both fixed rather than explained away:

- `F3_REVERT_consumer_predicate_is_None_only` **survived** because my test re-implemented the
  predicate inline instead of calling it — the exact silent-drift defect the research gate had just
  flagged in `tests/services/test_sod_daily_roll.py`, which I then reproduced. Fixed by extracting
  `sod_anchor_needs_reroll` so production and test share one callable; the test now imports it. A
  second mutant (`always_rerolls`) pins that the predicate must *discriminate*, not merely return True.
- `F3_refusal_still_stamps_the_date` was **semantically inert** — it reordered two assignments inside
  one lock, so its survival carried zero information. Rewritten to make the refusal path stamp the
  date on its way out, which is a real failure; it now dies.

## Criterion 3 & 4 — the wedge, and the healthy path

`test_phase_36_9_refusing_leaves_the_state_the_re_anchor_check_repairs` drives the real predicate and
shows the 409's promise ("the next cycle re-anchors both baselines") is now kept.
`test_phase_36_9_healthy_path_is_byte_for_byte_unchanged` compares the **whole dict** against fixed
numbers (`sod 10000 / peak 12000 / nav 11000` → `daily -10.0`, `trailing 8.3333`, `armed True`), so a
leak into the healthy path fails loudly, and
`test_phase_36_9_a_fresh_anchor_still_fires_the_daily_leg` proves a real 4.0% breach still fires.

## One existing 36.7 guard AMENDED — disclosed, not slipped in

`test_phase_36_7_kill_switch_baseline_restored_from_rotated_v4_file` asserted `armed is True` after
restoring from the real `-v4` row dated `2026-07-24`. **That assertion encoded the live defect**: it is
the same shape :8000 served today. Amended to assert `daily_baseline_stale`, `armed False`, and no
daily percentage — while still asserting 36.7's own subject (restoration works) and that the
**trailing leg still fires**, plus a new sub-case showing the identical restore with a fresh date
behaves exactly as 36.7 originally guaranteed. No 36.7 guarantee was removed.

## Fixture rot fixed (mechanically enumerated, not hand-picked)

**CORRECTED in cycle 2.** This section first claimed *“12 across 3 files”*. The grep does return
12 matches across 3 files — but the 12th (`tests/services/test_sod_daily_roll.py:142`) is
**docstring prose**, not an assignment: `boot replay sees the legacy 04-20 row ->
_sod_date='2026-04-20',` sitting inside a docstring. The true population of executable assignments
is **11 across 2 files** — 4 × `"2026-05-22"` (23_2_5) and 7 in 36_7 (6 × `"2026-07-26"`,
1 × `"2026-07-24"`). All 11 are now relative; the docstring is correctly untouched.

The error is worth naming because it is this project's recurring class: I derived the count
mechanically and then never checked that the *members* were what I called them. A grep finds
candidates; it does not certify that each hit satisfies the predicate in the sentence you write
about it. (A second instance in the same breath: my first re-derivation used `git grep -E` with
`\s`, which POSIX ERE does not define — it silently matched 1 of 12 and would have `confirmed`
any number I wanted.) The six fixtures hardcoding *today* were green only because today is
2026-07-26, and would have failed tomorrow. All in-memory anchors are now computed
(`TODAY_UTC`). Audit-row payloads and replay assertions stay literal deliberately — they pin that
`_load_from_audit` **restores** a stored date, which this step does not touch.

## Verification — measured this cycle

```
$ python -m pytest backend/tests/ -q -k kill_switch          # IMMUTABLE
166 passed, 1 skipped, 2126 deselected

$ python -m pytest backend/tests/test_phase_36_9_kill_switch_armed_liveness.py -q
28 passed

$ python tests/verify_phase_23_2_19.py                        # source-scan pin
phase-23.2.19 verification: ALL PASS (5/5)

$ python -m pytest backend/tests/ -q -k "paper_trader or autonomous or sod"
57 passed
```

The immutable count moved **138 → 166**: all 28 new tests are inside the selector (36.8 shipped a
cycle with zero of its tests selected because the filename lacked `kill_switch`).

**A gate defect I caught in my own tooling.** The first lint run reported `Found 1 error` at
`…36_9…py:1:1`. It was not a lint finding: **zsh does not word-split unquoted variables**, so
`ruff … $SCOPE` passed all five paths as ONE filename and ruff answered `E902 No such file`. I nearly
recorded a broken gate as a result. Re-run with `files=(${(f)SCOPE})`, scope asserted non-empty and
derived from `git diff --name-only`:

```
--- 5 files, derived from git ---
All checks passed!
```

## Out of scope → filed, not fixed here

`tests/services/test_sod_daily_roll.py` re-implements the roll predicate inline at :80/:100/:156 and
`tests/verify_phase_23_2_19.py:47-50` pins it by source-scan. The new shared helper makes converging
them possible, but doing it is not what this step authorizes — it gets its own research-gated step.

## Do-no-harm

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` before and after every run,
including the 13-mutant matrix (`_AUDIT_PATH` redirected to tmp *before* the module singleton is
built). `:8000` GET-only, never restarted or POSTed to. `:3000` never driven. Limits, stops, sector
caps, DSR and PBO byte-untouched — no threshold appears in this diff. No peak reset.

**NOT LIVE:** this code is not on the operator's `:8000`; like 36.12 and 36.8 it needs a restart the
operator has not authorized.


## Cycle 2 -- a Q/A found a money-path regression that all 162 tests were blind to

**The finding, and it was correct.** `armed` has a fourth consumer this step never considered:
`paper_trader.check_and_enforce_kill_switch:1126`, which measures it **before** the daily roll --
36.12's deliberate ordering ("MEASURE THE ARMED STATE BEFORE MUTATING THE BASELINES"). Folding
staleness into `armed` therefore made the **ordinary first cycle of every UTC day** look like lost
history, because a pre-roll anchor is yesterday's by construction.

I reproduced it end-to-end before changing anything, with a discriminating control:

```
STALE anchor (yesterday)  -> blocked=True  reason=kill_switch_disarmed_lost_history  P1s=1
FRESH anchor (today)      -> blocked=False reason=None                               P1s=0
```

Blocked orders, a P1 page, and a fabricated `lost_history_anchor` row into the live audit trail --
every morning. **My cycle-1 claim "the autonomous cycle is unaffected because it re-anchors first"
was measured FALSE**, and the contract compounded it by citing 36.12's order block as the
*compensating measure* for precisely the state that would trip it.

**The fix is a semantic split, not a suppression.** Two different questions had been conflated:

| Question | Key | Read by |
|---|---|---|
| Did we LOSE our baselines? (durable fault; only an operator repairs it) | `baselines_present` | the 36.12 order gate |
| Can this leg fire RIGHT NOW? (presence + freshness + NAV validity) | `armed` | badge, /resume, MCP |

`baselines_present` is the pre-36.9 meaning of `armed` exactly, so the order gate behaves
identically in every case it was built for. Verified by execution across all four states:

```
overnight stale anchor (THE REGRESSION)       blocked=False reason=None                             P1s=0
fresh anchor [CONTROL]                        blocked=False reason=None                             P1s=0
genuine LOST HISTORY (36.12 must still block) blocked=True  reason=kill_switch_disarmed_lost_history P1s=1
lost peak + stale sod (must still block)      blocked=True  reason=kill_switch_disarmed_lost_history P1s=1
```

**Guards added, because the Q/A's deeper point was that no mutation of my suite could have caught
this** -- all 13 of my mutants lived inside a module that never executed the order path, so the
matrix was green while the regression was live. Four new tests drive the REAL
`check_and_enforce_kill_switch` with the pager CAPTURED rather than mocked away, so a test can
assert the P1 is *not* raised (17 false P1s reached the operator's Slack earlier in this phase
because a probe ran with the real dispatcher attached). Two new mutants pin the split from both
sides -- `C2_baselines_present_refolds_staleness` and `C2_order_gate_reads_armed_again`. The matrix
is now **15 killed / 0 survived** at baseline `28 passed`.

The healthy-path whole-dict test failed the moment `baselines_present` appeared -- which is exactly
why it compares the whole dict instead of probing individual fields. Expectation updated
deliberately, with the new key acknowledged rather than the assertion loosened.

**The lesson, stated as the class:** the contract DID enumerate the consumers of `evaluate_breach`,
and got the read surfaces right. What it never asked was which consumer reads the flag **at a
different point in the lifecycle**. A consumer census is not complete until it records *when* each
consumer reads the value, not merely that it does.

## Verification -- re-measured after cycle 2

```
$ python -m pytest backend/tests/ -q -k kill_switch          # IMMUTABLE
166 passed, 1 skipped, 2126 deselected

$ python -m pytest backend/tests/test_phase_36_9_kill_switch_armed_liveness.py -q
28 passed
```


## Cycle 3 -- two operator-facing residuals from the C2 CONDITIONAL

C2 confirmed the C1 regression genuinely closed (it re-mutated the fix from both sides itself and
both mutants died on the new order-path tests) and held at CONDITIONAL on two WARN-severity
operator-facing residuals. Both were real.

**F-B -- the /resume 409 named a cause its own diagnostics refuted. FIXED.** The 36.7 gate refuses on
the new staleness cause but still emitted the ABSENCE text: *"the loss baselines could not be
restored"* while printing `daily_baseline_missing=False, trailing_baseline_missing=False`, and its
remediation described a lost-history block that the cycle-2 split means no longer fires for this
cause. A staleness-specific branch now names the offending date, states the baselines are intact and
the trailing leg still armed, and describes the daily roll that actually clears it. Guarded by a test
that drives the REAL endpoint; mutation-proved (removing the branch -> `1 failed`).

**F-A -- the cockpit badge reads DISARMED on a healthy book every morning. QUEUED as step 36.20.**
`KillSwitchPanel.tsx:137` and `OpsStatusBar.tsx:318` derive `disarmed = breach.armed === false` and
are the only frontend reads of the breach dict; neither reads the new `daily_baseline_stale`. So from
00:00 UTC until that day's first cycle rolls the anchor, a healthy book renders the alarm badge with
Resume disabled. **This is a real operator-visible consequence of this step and it is disclosed here
rather than left for someone to discover.** It is not fixed here because it is a frontend change
requiring :3100 UI evidence, outside this step's authorized scope. 36.20 carries the fix direction
and the trap that a third state must not be `armed: undefined` (both backend gates `.get("armed",
True)` and would fail OPEN).

**An existing P0 guard caught my first wording.** The 409 text initially contained `next cycle
re-anchors`, banned by `test_phase_36_12_no_operator_string_still_promises_an_automatic_re_anchor`
because for LOST HISTORY the automatic anchor WAS the defect. My case is the legitimate daily roll,
but the phrase is ambiguous to a reader, so I reworded the message and the comment rather than narrow
a P0 guard to accommodate my sentence. Immutable went 166 -> 167 (the new 409 test) with that guard
passing unchanged.

**A defect found while verifying, queued as 36.21.** `pytest backend/tests/ -q -k "paper_trading or
resume"` appends four real `pause`/`resume` rows to the git-tracked live audit file. Reproduced
twice; restored via `git show HEAD:` and md5 re-verified both times. Not caused by this step -- the
twelve candidate files each leave the digest unchanged when run alone, so it is an ordering/pollution
effect -- but I found it, so it is queued with the exact repro rather than described in prose.

## Verification -- final

```
$ python -m pytest backend/tests/ -q -k kill_switch          # IMMUTABLE
167 passed, 1 skipped, 2126 deselected

$ python /tmp/.../mutate_36_9.py                              # matrix, one batch
15 killed, 0 survived, of 15 mutants

$ ruff check --select F821,F401,F811,E9 <6 git-derived files>
All checks passed!
```

`handoff/kill_switch_audit.jsonl` md5 `ce8fb93348bb9a3bbe26f2d91b1bc05e` -- verified after every run
above, and after the two deliberate 36.21 reproductions.
