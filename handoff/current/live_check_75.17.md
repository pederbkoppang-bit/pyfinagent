# live_check -- Step 75.17 (absent-path verification family: triage + repair)

Date: 2026-07-24. Verbatim captures; rc=$? discipline. Baseline commit for
all byte-identity / git-show comparisons: `7739922d8ab9ed8398dbb97da1a9a8d6ed6894ae`
(the pre-75.17 HEAD; a fixed SHA per CLAUDE.md's immutable-criteria rule,
not "HEAD" -- HEAD moves as later steps land).

## 1. The committed sweep independently reproduces the genuine 10

```
$ .venv/bin/python scripts/qa/sweep_absent_verification_paths.py
phase-75.17 sweep: 739 done/unannotated steps scanned; shape census dict=720 str=126 list=13 none=24
phase-75.17 sweep: CLEAN -- no genuine absent-path defects
```
(run AFTER the 10 annotations landed -- CLEAN is the expected post-repair state)

Against the pre-75.17 (git-show baseline) masterplan snapshot, the SAME
classifier returns exactly the 10:
```
4.14.26    retired      ret=f7e24d0a  backend/agents/skills/neutral_analyst.md
4.14.26    retired      ret=f7e24d0a  backend/agents/skills/devils_advocate_agent.md
4.17.11    never-existed  scripts/go_live_drills/openclaw_runtime_test.py
4.17.12    never-existed  scripts/go_live_drills/f1_recovery_drill.py
4.17.2     never-existed  scripts/go_live_drills/researcher_smoke_test.py
4.17.3     never-existed  scripts/go_live_drills/qa_smoke_test.py
4.17.4     never-existed  scripts/go_live_drills/handoff_e2e_test.py
4.17.5     never-existed  scripts/go_live_drills/coala_memory_layers_test.py
4.17.6     never-existed  scripts/go_live_drills/signal_evidence_test.py
4.17.7     never-existed  scripts/go_live_drills/paper_trade_e2e_test.py
4.17.8     never-existed  scripts/go_live_drills/slack_bot_smoke_test.py
```
Shape census matches the research brief's corrected figures exactly:
`dict=720 str=126 list=13 none=24` (the node's "674 dict" is the stale
75.2.1-era count; brief corrected it to 720).

## 2. Immutable verification command (exit 0)

```
$ .venv/bin/python -m pytest backend/tests/test_phase_75_17_verification_paths.py -q
.............................................                            [100%]
45 passed in 2.36s
```
exit=0

## 3. Byte-identity proof (all 10 touched steps, command + success_criteria)

```
4.17.2   command_match= True criteria_match= True
4.17.3   command_match= True criteria_match= True
4.17.4   command_match= True criteria_match= True
4.17.5   command_match= True criteria_match= True
4.17.6   command_match= True criteria_match= True
4.17.7   command_match= True criteria_match= True
4.17.8   command_match= True criteria_match= True
4.17.11  command_match= True criteria_match= True
4.17.12  command_match= True criteria_match= True
4.14.26  command_match= True criteria_match= True
ALL BYTE-IDENTICAL: True
```
Reproduced independently inside the pytest guard suite
(`test_verification_byte_identical_to_baseline`, parametrized x10) AND via
a standalone script comparing `git show <baseline>` to the live file.

## 4. Exactly-one-superseded_record repo-wide

```
$ grep -c '"superseded_record"' .claude/masterplan.json
14
```
Holders: 4.14.4, 4.14.24, 4.17.9, 68.5 (pre-existing) + 4.17.2, 4.17.3,
4.17.4, 4.17.5, 4.17.6, 4.17.7, 4.17.8, 4.17.11, 4.17.12, 4.14.26 (new) = 14.
The guard test uses `json.loads(..., object_pairs_hook=...)` so a raw
duplicate-key insertion inside one step's object is caught even though a
plain dict-based count would silently collapse it (see M9 below).

## 5. Masterplan diff purity

```
$ git diff HEAD --stat -- .claude/masterplan.json
 .claude/masterplan.json | 148 ++++++++++++++++++++++++++++++++++++++++++++----
 1 file changed, 138 insertions(+), 10 deletions(-)
```
All 10 "-" lines are trailing-comma artifacts of appending a new sibling
key as the object's new last property (JSON syntax requires the
previously-last line to gain a comma) -- each is paired 1:1 with an
identical "+" line plus the comma:
```
-          "max_retries": 3
+          "max_retries": 3,
+          "superseded_record": {
```
Verified programmatically (`test_masterplan_diff_touches_only_the_ten_sibling_insertions`):
every removed line, with a trailing comma appended, appears verbatim in
the added-lines set. Zero real content deletions.

## 6. Regression comparison vs the environment-dependent baseline

CI-equivalent selection (3 env overrides false, `-m "not requires_live"`):
```
$ PAPER_DATA_INTEGRITY_ENABLED=false PAPER_RISK_JUDGE_REJECT_BINDING=false PAPER_SWAP_CHURN_FIX_ENABLED=false \
  .venv/bin/python -m pytest backend/tests/ -q -m "not requires_live"
1555 passed, 2 skipped, 16 deselected, 5 xfailed, 1 xpassed, 1 warning in 100.01s
```
0 failed.

Raw suite (no env overrides, no marker filter):
```
$ .venv/bin/python -m pytest backend/tests/ -q
8 failed, 1553 passed, 12 skipped, 5 xfailed, 1 xpassed, 1 warning in 123.50s
```
The 8 failures are a STRICT SUBSET of the documented 9-red baseline
(`tree_fails_75_15.txt`) -- the one absent item
(`test_phase_23_2_15_known_pass_scripts_still_pass`) is the already-documented
PATH-shell-dependent test. Zero new failures once the pre-existing
collection-count canary (below) was updated.

**One in-scope collateral fix**: adding 45 new tests bumped the collected
count under `-m "not requires_live"` from 1518/1534 to 1563/1579 (deselected
unchanged at 16), which broke the phase-75.16 collection-count canary in
`backend/tests/test_phase_75_ci_gates.py::test_backend_not_requires_live_collection_count_is_stable`
(a hard-coded snapshot string, by design, meant to be bumped whenever tests
are legitimately added -- exactly the situation its own comment
anticipated for the 75.16 addition). Updated the string + comment to the
new baseline; this is the ONLY touched file outside the three
listed above.

## 7. Ruff (derived scope)

```
$ ruff check --select F821,F401,F811 scripts/qa/sweep_absent_verification_paths.py \
  backend/tests/test_phase_75_17_verification_paths.py backend/tests/test_phase_75_ci_gates.py
All checks passed!
```

## 8. Mutation matrix M1-M10 (verbatim, all rows killed or expected-no-op)

Script: scratchpad `mutation_matrix_75_17.py`. Every file mutation applied
to an exact-count-1 pattern with a guaranteed byte-restore (verified via
shasum before/after); the two real touched files
(`scripts/qa/sweep_absent_verification_paths.py`,
`backend/tests/test_phase_75_17_verification_paths.py`) end at the
IDENTICAL checksum they started at:
```
pre-run:  1ba670de972d2648f5dbaa1981cd15c5e035fa06  sweep_absent_verification_paths.py
          2d48f2c007ff9297932a532d2dfaa18f97273596  test_phase_75_17_verification_paths.py
post-run: 1ba670de972d2648f5dbaa1981cd15c5e035fa06  sweep_absent_verification_paths.py
          2d48f2c007ff9297932a532d2dfaa18f97273596  test_phase_75_17_verification_paths.py
```

**Incident during construction (disclosed, not hidden):** an earlier
version of the M4+M5 mutation wrote the SAME file (SWEEP) under two
different variable names (`path` and `p2`) and its second "restore" step
overwrote the first, leaving the committed file corrupted mid-run in a
partially-mutated state (the well-formedness regex widened, everything
else intact). Caught immediately by re-running the guard suite
(`45 passed` became a regex mismatch on manual grep inspection), fixed by
restoring the exact original file content and rewriting the mutation
script so any compound mutation to a single file keeps exactly ONE
original-backup across the whole compound, never two. Final checksums
above confirm the fix holds.

```json
[
 {"mutation": "M3 remove frontend/src fallback from _resolves_on_disk", "killed": true, "tail": "1 failed, 44 deselected in 0.03s"},
 {"mutation": "M6 remove the `test ! -f X` negation pattern", "killed": true, "tail": "1 failed, 44 deselected in 0.03s"},
 {"mutation": "M10 disable glob-prefix re-resolution", "killed": true, "tail": "1 failed, 44 deselected in 0.03s"},
 {"mutation": "M7 FIXTURE mutation: list-shaped step -> dict-shaped (same command)", "killed": true, "tail": "1 failed, 44 deselected in 0.04s"},
 {"mutation": "M5 truncated-plist: 2-layer removal (regex + leading-slash branch)", "killed": true, "tail": "1 failed, 44 deselected in 0.04s"},
 {"mutation": "M4 url-fragment: 2-layer removal -- MEASURED NOT KILLED (3rd incidental defense below)", "killed": false, "tail": "1 passed, 44 deselected in 0.01s"},
 {"mutation": "M4 url-fragment: 3-layer removal (+boundary/mis-split gate) -- MEASURED STILL NOT KILLED (4th defense below)", "killed": false, "tail": "1 passed, 44 deselected in 0.01s"},
 {"mutation": "M4 url-fragment: 4-layer removal (+localhost-URL regex) -- KILLED, proving non-vacuity", "killed": true, "tail": "1 failed, 44 deselected in 0.03s"},
 {"mutation": "M1 truncated masterplan JSON (real file untouched; fed via tmp copy)", "killed": true, "tail": "load_masterplan raised json.JSONDecodeError"},
 {"mutation": "M2 planted absent path in a fixture done-step", "killed": true, "tail": "found=True class=never-existed"},
 {"mutation": "M8 byte-identity break in a tmp copy (real masterplan never touched)", "killed": true, "tail": "mismatch_detected=True"},
 {"mutation": "M9 duplicate superseded_record key (synthetic JSON, identical detection logic)", "killed": true, "tail": "duplicate_key_counts=[2]"}
]
```
`12/12 rows killed-or-expected; survivors: NONE`

**M4 finding (measured, not asserted away):** the URL-fragment guard for
this realistic `curl http://localhost:8000/openapi.json`-shaped command is
defended by FOUR independent layers in the production code: the
well-formedness gate, the leading-slash branch, the boundary/mis-split
gate (incidental: "8000" immediately precedes the token), and the
dedicated localhost-URL regex (the semantically intended guard for this
exact shape). Removing any 1-3 of these leaves the test passing (a
measured non-kill, not a vacuous test); removing all 4 together correctly
flips the test to FAIL, proving the guard genuinely depends on production
behavior and is not tautological.

**Production/test files: net-zero diff.** `scripts/qa/sweep_absent_verification_paths.py`
and `backend/tests/test_phase_75_17_verification_paths.py` are both NEW
files added by this step (not modifications to pre-existing files), so
"restored to original" means restored to the content this step itself
authored -- confirmed byte-identical pre/post mutation-matrix via shasum.

## 9. On-disk equivalents (class-i annotations), independently verified

```
4.17.2  -> scripts/go_live_drills/smoke_test_4_17_2.py   EXISTS
4.17.3  -> scripts/go_live_drills/smoke_test_4_17_3.py   EXISTS
4.17.4  -> scripts/go_live_drills/smoke_test_4_17_4.py   EXISTS
4.17.5  -> scripts/go_live_drills/smoke_test_4_17_5.py   EXISTS
4.17.6  -> scripts/go_live_drills/smoke_test_4_17_6.py   EXISTS
4.17.7  -> scripts/go_live_drills/smoke_test_4_17_7.py   EXISTS
4.17.8  -> scripts/go_live_drills/smoke_test_4_17_8.py   EXISTS
4.17.11 -> scripts/go_live_drills/smoke_test_4_17_11.py  EXISTS
4.17.12 -> scripts/go_live_drills/smoke_test_4_17_12.py  EXISTS
```
`git log --all --diff-filter=A` for every plan-side name (researcher_smoke_test.py,
qa_smoke_test.py, handoff_e2e_test.py, coala_memory_layers_test.py,
signal_evidence_test.py, paper_trade_e2e_test.py, slack_bot_smoke_test.py,
openclaw_runtime_test.py, f1_recovery_drill.py) returns EMPTY -- confirmed
independently (not trusted from the research brief).

4.14.26: `backend/agents/skills/neutral_analyst.md` and
`backend/agents/skills/devils_advocate_agent.md` confirmed ABSENT on disk;
`git log --all --diff-filter=D` for both names to commit `f7e24d0a`
("phase-26.4: Consolidate 6 opinion skills into parameterized stance
prompt"), confirmed independently.
