# experiment_results -- step 86.110

**GENERATE complete for all six criteria.** Verbatim evidence in
`handoff/current/live_check_86.110.md`; this file is the build record.

## What was built

### New files

| File | Purpose |
|---|---|
| `backend/tests/test_phase_86_110_heartbeat_isolation.py` | 13 tests of three deliberately different kinds: behavioural (run the previously-leaking tests in a subprocess against a snapshot), structural (the sweep, including its own positive control), and the runtime guard (it must fire AND repair). |
| `scripts/qa/heartbeat_leak_sweep_86_110.py` | Criterion 4's enumeration, keyed on **reachability of a writer**, with the naive patch-table survey printed as a cross-check rather than used as the rule. |
| `scripts/qa/mutation_86_110.py` | 10-cell matrix, strict scoring. |

### Modified files

| File | Change |
|---|---|
| `backend/tests/test_phase_66_1_rail_guard.py` | Both leaking sites isolate `_HEARTBEAT_PATH`, same idiom as `test_phase_86_38`. |
| `backend/tests/conftest.py` | **Third pollution guard**, beside the existing BQ and Slack-egress ones: an autouse, function-scoped fixture that fails — and REPAIRS — any test which modifies a git-tracked handoff state file. |
| `backend/tests/test_phase_23_2_14_no_reentrant_locks.py` | `EXPECTED_LOCK_COUNT` 20 → 21 with the re-audit the file's own rule demands. See below — this is a regression **86.108 shipped**. |
| `handoff/.cycle_heartbeat.json` | Regenerated from the ledger's last completed cycle (criterion 5). |

## Verbatim verification output

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/test_phase_66_1_rail_guard.py\").read())" && echo parses'
parses
EXIT=0

$ pytest backend/tests/test_phase_86_110_heartbeat_isolation.py -q
13 passed in 4.72s

$ pytest backend/tests/ -q -k "cycle_health or heartbeat or rail_guard or 38_2 or 86_38 or 23_2_14 or 86_110"
71 passed, 3588 deselected

$ python scripts/qa/heartbeat_leak_sweep_86_110.py
POPULATION: 4   ISOLATED TESTS: 3   LEAKING: 0   LEGITIMATE PRODUCTION WRITERS: 1
EXIT=0

$ python scripts/qa/mutation_86_110.py
CONTROL rc=0  collected=13
KILLED=10/10  SURVIVORS=none  UNSCORABLE=none

$ uvx ruff check --select F821,F401,F811 --no-cache <8-file DERIVED scope>
All checks passed!    RUFF_EXIT=0

$ pytest backend/tests/ -q                 (the FULL suite, because the new fixture is global)
20 failed, 3621 passed, 12 skipped, 5 xfailed, 1 xpassed in 8:31 (3,659 collected)
$ grep -c "phase-86.110 test guard"  -> 0  <- my guard fires on NONE of the 20
  CONTROLLED CHECK (same 20-test subset, fixture off vs on): 19 failed / 1 passed BOTH times.
  The fixture causes zero failures. 19 reproduce in isolation; the 1 that passes
  in the subset is a pre-existing full-suite order dependency.
```

## Three things this step found that were not in the filing

1. **The pollution had NOT self-healed.** The filing and this step's own
   research gate both recorded a real `cycle_id` in the working tree. At build
   time the file held **`c2` again** -- the evening's earlier evaluation runs had
   re-polluted it. The contract anticipated this and required criterion 5's
   disposition to be measured at build time rather than carried from the gate.
2. **The sweep's first implementation could not detect the leak it was written
   for.** Its `re.search("_HEARTBEAT_PATH", src)` was satisfied by the
   explanatory comments the fix itself added -- the guard matched its own
   documentation, and the positive control caught it. Rewritten to walk the AST,
   where comments are invisible by construction. Cell P3 pins it.
3. **A REAL REGRESSION FROM 86.108, ALREADY PUSHED.**
   `test_phase_23_2_14_no_reentrant_locks` demands a re-audit whenever the
   `threading.Lock()` roster changes; 86.108's parse-failure ledger added lock
   #21 and **86.108 never ran that guard**, because its regression sweep used a
   `-k` selection that did not include the file. Main carried a red test. The
   guard worked; the SCOPE of the sweep was the defect. *A `-k` selection is not
   a regression suite.* Re-audited and bumped: plain `Lock`, no re-entrancy
   surface (`resolve_rail()` runs before it is taken, `logger.warning` after it
   is released), no `_*_locked` helper introduced.

## Deviations and honest gaps

- **P4 was removed from the matrix as an EQUIVALENT MUTANT**, with the proof
  stated: no file calls `record_cycle_start` without also calling
  `record_cycle_end`, so narrowing `WRITERS` changes no row. The wider set stays
  because it is what keeps the sweep correct the day a start-only caller
  appears -- but this matrix cannot prove that today.
- **Two cells were UNSCORABLE on the first run for a reason in my runner, not
  the guards**: nested pytest output inflated the collected-test count. The
  counter now reads only the final summary line.
- **`cycle_health`'s two-constant design is not refactored.** The sweep and the
  conftest guard contain the class; they do not remove its cause.

## Scope honesty

- No flag promoted, no `.env` written.
- No step flipped, no prior verdict altered (criterion 6).
- `handoff/.cycle_heartbeat.json` IS in this step's diff, deliberately and per
  criterion 5, with its derivation shown.
- No restart is pending from this step: it changes tests, a test fixture and a
  QA script; the only production file touched is none.

## Cycle 2 -- response to the CONDITIONAL (`wf_e7115d07-ae1`)

All six criteria were already MET and independently reproduced. Two WARN
findings, both closed.

**F1 was a real defect in the guard this step shipped.**
`handoff/cycle_history.jsonl` is an append-only ledger the LIVE
`autonomous_loop` writes from this same machine, and a full suite run takes
~8.5 minutes -- so the guard's unconditional restore could have **deleted a
real cycle row and blamed an innocent test.** Each protected file now carries a
rule for what a legitimate concurrent write looks like: `append_only` (the new
content must start with the snapshot and be longer) for the ledger, and
`ledger_backed` (its `cycle_id` must exist in the real ledger) for the
heartbeat. When the rule says legitimate, the guard leaves the file alone.
Because making a guard tolerant risks making it blind, all three arms are
mutated -- cells **P8**, **P9**, **P10**, all KILLED -- and
`test_the_guard_STILL_catches_a_test_leak_after_the_concurrency_fix` is the
standing anti-regression. Matrix is now **9/9**.

**F2**: the full-suite block was measured before this step's own tests existed
(3,646 vs 3,659 collected) while labelled as the shipped tree, and gave a bare
count. Re-run against the shipped tree with the complete failing-ID list, which
is byte-identical to the pre-existing set. The four masterplan-sensitive
failures are pre-existing -- they fail on a dead path reference in step 86.31,
not on this session's filings.

## Cycle 3 -- response to the second CONDITIONAL (`wf_8275f3fa-266`)

All six criteria MET, both cycle-1 blockers confirmed closed. Two evidence-side
findings, both closed.

**F1 -- only the redundant half of my own rule was tested.** Mutating
`append_only` from `after.startswith(before) and len(after) > len(before)` to
`len(after) > len(before)` SURVIVED 13/13: all three fixtures were length-only
discriminable, and the length clause is redundant given `startswith`. Cell P9
mutates the whole compound return, which the length-only cases catch -- so the
matrix read 9/9 while one clause had no falsifying fixture. Closed with a
rewrite LONGER than the snapshot (not a prefix, not catchable by length) plus
cell **P11**, which mutates the prefix half alone. Matrix **10/10**.
*A matrix cell that mutates a compound expression licenses nothing about its
sub-expressions.*

**F2 -- residual (a)'s census was wrong in count and mechanism.** Derived
properly there are **six test-side transitive reachers**, not three: two
isolate `_HEARTBEAT_PATH`, two stub `cycle_health.get_log` (a THIRD idiom the
sweep does not model), and two never execute the function. The substantive
conclusion -- none leaks -- survives and was independently verified, but the
enumeration was wrong and a coverage claim with a wrong census is not one.

Also now disclosed: the conftest guard sits in `backend/tests/`, so it does not
reach the root `tests/` tree the sweep scans.
