# Contract -- step 86.110

**Step:** 86.110 -- two tests monkeypatch `cycle_health._HISTORY_PATH` but not
`_HEARTBEAT_PATH`, so running them writes a synthetic `cycle_id` into the real,
git-tracked `handoff/.cycle_heartbeat.json`. **P2.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** (`wf_f5a37a43-27c`; 7 sources read in full, 25 URLs,
audit-class dry after 5 rounds; brief `research_brief_86.110.md`, 31,324
chars). **Three premise corrections, and two of them change what the step
does.**

**1. The filing's patch table is wrong in BOTH directions, so criterion 4's
"do not hand-list" is not a style note -- it is the finding.** The filing
frames the population as "tests that patch `_HISTORY_PATH` but not
`_HEARTBEAT_PATH`". That over-reports (`test_cycle_heartbeat_alarm.py:61`
patches one constant but its code path reaches no writer, so it does not leak)
and under-reports (`scripts/smoketest_stages_5_through_13.py:402-409` swaps the
constant **outside pytest entirely**, which no monkeypatch-shaped survey would
find). **The correct denominator is transitive reachability of a writer**,
cross-checked against the patch table -- not the patch table itself.

**2. THE POLLUTION HAS ALREADY SELF-HEALED, so criterion 5's answer is "write
nothing", and that is a finding rather than a dodge.** Verified by me at
contract time: the working tree holds `cycle_id=3e5afddb`, and
`grep -c 3e5afddb handoff/cycle_history.jsonl` returns **2** -- a real cycle
with real ledger rows. The fake `c2` is gone. The file is derived,
last-writer-wins, and has a live writer; hand-restoring it would **manufacture**
a state no cycle produced. The disposition is: stop the leak, write nothing to
the file, and say so.

**3. A provider-only fix would MISS both leaking tests.** They construct
`CycleHealthLog()` directly, so monkeypatching `get_log` -- the idiom
`test_phase_36_17.py:271` already uses -- never intercepts them. The fix must
isolate the module-level constants, not the accessor.

**Two mechanism corrections I re-derived rather than inherited.** The filing's
line-to-name mapping is **inverted**: `:194` is inside
`test_rail_guard_cycle_history_row_carries_flags` and writes `cycle_id="c1"`;
`:211` is inside `test_cycle_history_row_carries_funnel_counts` and writes
`"c2"`. And the writer surface is **larger than the filing says**: the heartbeat
is written by `record_cycle_start` (`cycle_health.py:426`) as well as
`record_cycle_end` (`:492`), both via `_write_heartbeat` (`:555`, writing
`_HEARTBEAT_PATH` at `:558`). An enumeration keyed only on `record_cycle_end`
would itself under-report.

**Prior art:** `tree-is-clean` is the closest external pattern (assert the
working tree is unchanged around a run); ODRepair explicitly excludes
filesystem state; **no pytest plugin does tracked-file snapshot-diff**, so this
is being built, not adopted.

## Hypothesis

The leak is not "two tests forgot a line". It is that a module exposes two
independent path constants and a single public call writes BOTH, so isolating
one is silently partial. The durable fix is an enumeration keyed on *reaching a
writer* plus a guard that fails any test which mutates a git-tracked file --
which catches the class rather than the two known instances.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the leak is REPRODUCED by execution, not asserted: run test_phase_66_1_rail_guard.py's test_cycle_history_row_carries_funnel_counts (or test_rail_guard_cycle_history_row_carries_flags) against a SNAPSHOT of the real handoff/.cycle_heartbeat.json taken immediately before, and show the real file's content changed after the test run despite the test using tmp_path for its own assertions
2. the fix adds `monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / ...)` to BOTH leaking sites (test_phase_66_1_rail_guard.py lines ~194 and ~211), mirroring the already-correct pattern in test_phase_86_38_degradation_visibility.py's `health` fixture -- not a new, third isolation idiom
3. post-fix, the SAME reproduction in criterion 1 is re-run and shows the real file UNCHANGED by the test -- the control (file changes pre-fix) and the fix (file unchanged post-fix) are both demonstrated, not just the fix alone
4. a repo-wide sweep confirms no OTHER call site that reaches record_cycle_end / _write_heartbeat patches _HISTORY_PATH without also patching _HEARTBEAT_PATH -- enumerated from source (every `setattr(.*_HISTORY_PATH` site, cross-checked against whether that test's code path calls record_cycle_end), not hand-listed from this filing's two known sites
5. the currently-polluted handoff/.cycle_heartbeat.json is restored: either regenerated from the real last-completed cycle in cycle_history.jsonl, or explicitly left as the next real cycle's natural write will overwrite it -- state which, and why, rather than leaving the fake c2 value standing uncommitted with no disposition
6. verdict semantics and other steps' status are UNCHANGED: nothing here may flip an unrelated step or alter a prior verdict

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/test_phase_66_1_rail_guard.py\").read())" && echo parses'`

**Immutable live_check:** `live_check_86.110.md` with the before/after
snapshots, the post-fix re-run, the criterion-4 sweep with its command and
output shown, and the disposition of the fake value.

## Plan

**P1 -- criterion 1, reproduced by execution.** Snapshot the real
`handoff/.cycle_heartbeat.json` (SHA-256 + content), run ONE leaking test in a
subprocess, re-read the file, show it changed. **Restore the snapshot
immediately afterwards** so the reproduction does not itself leave the
pollution behind -- a reproduction that dirties the tree is the defect, not a
demonstration of it.

**P2 -- criterion 2, the SAME idiom.** Add
`monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / ...)` at both sites,
mirroring `test_phase_86_38_degradation_visibility.py`'s fixture. No third
idiom, and no provider-level indirection -- both tests construct
`CycleHealthLog()` directly, so a `get_log` patch would not intercept them.

**P3 -- criterion 3, control AND fix.** Both halves, from the same harness:
pre-fix the file changes, post-fix it does not. The pre-fix half is obtained by
reverting the two added lines in a throwaway copy, never by trusting memory of
what the file used to do.

**P4 -- criterion 4, ENUMERATED from reachability.** The criterion forbids a
hand-list, and the gate showed why: the patch table both over- and
under-reports. Compute the population as **every call site that transitively
reaches `_write_heartbeat`** -- which means keying on `record_cycle_start` AND
`record_cycle_end`, not only the latter -- then cross-check against which of
those isolate `_HEARTBEAT_PATH`. Include non-pytest writers
(`scripts/smoketest_stages_5_through_13.py` assigns the constant directly).
State the command and show its output.

**P5 -- the durable guard, which is what makes this more than two lines.** An
autouse session-scoped check that fails any test which leaves a **git-tracked**
file modified. That catches the CLASS -- a future test that writes a different
tracked file, or a third path constant added to this module -- rather than the
two instances. Prior art is `tree-is-clean`; no pytest plugin ships this.

**P6 -- criterion 5, disposition: WRITE NOTHING, and say why.** The fake `c2`
is already gone; the tree holds `3e5afddb`, a real cycle with 2 ledger rows.
The file is derived, last-writer-wins, with a live writer. Regenerating it would
manufacture state. **Recorded as a measurement at evaluation time, not as a
claim carried from the gate** -- if the value has moved again, the artifact says
so.

**P7 -- criterion 6 mutations** with the control observed GREEN first and a
byte-identical restore.

## Scope honesty -- what this step does NOT do

- **It writes nothing to `handoff/.cycle_heartbeat.json`** and does not commit
  it. The operator's standing note calls it fixture-poisoned; it is currently
  NOT, and the correct action either way is to leave it to its writer.
- **It does not refactor `cycle_health`'s two-constant design**, which is the
  root cause. Isolating a module whose one public call writes two paths is a
  wider change; if it is worth doing it gets its own step.
- **It does not fix `scripts/smoketest_stages_5_through_13.py`** if the sweep
  shows it leaks -- that is a script, not a test, and it is reported rather
  than silently edited under this step's name.
- **It flips no step and alters no prior verdict** (criterion 6).

## References

`research_brief_86.110.md` (the reachability-vs-patch-table correction, the
self-heal measurement, the provider-only-fix trap, the `tree-is-clean` prior
art); `backend/services/cycle_health.py:36-37,426,492,555-558`;
`backend/tests/test_phase_86_38_degradation_visibility.py` (the correct idiom).
