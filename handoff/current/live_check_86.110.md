# live_check -- step 86.110 (2026-08-18; exits unpiped)

## 1. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/tests/test_phase_66_1_rail_guard.py\").read())" && echo parses'
parses
EXIT=0
```

## 2. Criterion 1 -- the leak, REPRODUCED BY EXECUTION

Snapshot taken first, so the reproduction cannot itself leave the pollution
behind -- a demonstration that dirties the tree is the defect, not evidence of
it.

```
=== BEFORE ===
content: {"cycle_id": "c2", "event": "end", "updated_at": "2026-08-17T21:03:54.974417+00:00"}
sha256 : ea504fc36e554f79ed66cc516eb2d2b9e5ca1db8a5274809b688fd3b007bffcf

$ pytest backend/tests/test_phase_66_1_rail_guard.py::test_cycle_history_row_carries_funnel_counts -q
1 passed in 0.02s          <- the test PASSES; it uses tmp_path for its own assertions

=== AFTER ===
content: {"cycle_id": "c2", "event": "end", "updated_at": "2026-08-17T22:41:40.533239+00:00"}
sha256 : a8bcd8c9bb183fdc8f87feac25c597b0d4df051d60652c3e8433f53ed94fa490

CHANGED: True
restored: True
```

**A passing unit test wrote into the real, git-tracked heartbeat the dashboard
reads as a liveness signal.**

**A correction to the filing and to this step's own research gate.** Both
recorded that the pollution had self-healed to a real `cycle_id`. When this step
actually ran, the file held **`c2` again** -- the fixture value -- because the
evening's earlier evaluation runs re-polluted it. The self-heal was real and
then undone. This is why criterion 5's disposition is measured at build time
rather than carried from the gate, and the contract said so in advance.

## 3. Criterion 2 -- both sites isolated, with the SAME idiom

`monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")`
added at both sites, mirroring `test_phase_86_38_degradation_visibility.py`'s
`health` fixture. **Not** a provider-level patch: both tests construct
`CycleHealthLog()` directly, so patching `get_log` -- the idiom
`test_phase_36_17.py` uses -- would never intercept them.

**The writer surface is larger than the filing said**, re-derived here: the
heartbeat is written by `record_cycle_start` (`cycle_health.py:426`) as well as
`record_cycle_end` (`:492`), both through `_write_heartbeat` (`:555`, writing
`_HEARTBEAT_PATH` at `:558`).

## 4. Criterion 3 -- control AND fix, both demonstrated

```
=== POST-FIX: the whole rail-guard file re-run ===
9 passed in 0.18s
heartbeat sha BEFORE: ea504fc3...bffcf
heartbeat sha AFTER : ea504fc3...bffcf
UNCHANGED: True
```

The pre-fix half is §2. The post-fix half is above, and it is also asserted as a
standing test (`test_the_previously_leaking_tests_no_longer_touch_the_real_heartbeat`),
which runs the rail-guard suite in a **subprocess** -- in-process, the new
conftest guard would repair the file and mask the measurement.

## 5. Criterion 4 -- the sweep, keyed on REACHABILITY not on the patch table

`scripts/qa/heartbeat_leak_sweep_86_110.py`:

```
POPULATION RULE: a file that calls any of ('record_cycle_end',
'record_cycle_start', '_write_heartbeat') and is not cycle_health.py itself.
SEARCH ROOTS: ('backend', 'scripts', 'tests')

     file                                                    writers                              hist hb
prod backend/services/autonomous_loop.py                     record_cycle_end,record_cycle_start   -    -
     backend/tests/test_phase_38_2_cycle_start_logging.py     record_cycle_end,record_cycle_start   Y    Y
     backend/tests/test_phase_66_1_rail_guard.py              record_cycle_end                      Y    Y
     backend/tests/test_phase_86_38_degradation_visibility.py record_cycle_end,record_cycle_start   Y    Y

POPULATION: 4   ISOLATED TESTS: 3   LEAKING: 0   LEGITIMATE PRODUCTION WRITERS: 1

CROSS-CHECK against the naive patch-table survey:
  naive survey would flag: 2
  OVER-reported by naive: ['backend/tests/test_cycle_heartbeat_alarm.py',
                           'scripts/smoketest_stages_5_through_13.py']
  UNDER-reported by naive: none
EXIT=0
```

**The criterion says "not hand-listed", and the cross-check shows why.** Both
naive hits are over-reports, verified by direct grep: neither
`test_cycle_heartbeat_alarm.py` nor `smoketest_stages_5_through_13.py` calls any
writer, so neither can leak the heartbeat however it patches the history
constant.

*(The research gate framed the smoketest as an UNDER-report. Re-derived here it
is an over-report for THIS question. The gate's framing was about a different
property -- that a monkeypatch survey misses a direct assignment -- which is
true, and is why the cross-check is printed rather than trusted.)*

**`autonomous_loop.py` is classified `prod`, not `LEAK`.** It writes the real
heartbeat on purpose; that is the file's whole point. A sweep that called the
correct behaviour a defect would be the mirror of the naive survey's error.

### The sweep's own positive control, which it FAILED on the first attempt

```
$ <revert the fix, re-run the sweep>
FIRST VERSION  -> exit 0, LEAKING: 0     <- the scanner could not see the leak
AFTER THE FIX  -> exit 1, LEAKING: 1, backend/tests/test_phase_66_1_rail_guard.py
```

The first implementation checked `re.search("_HEARTBEAT_PATH", src)` -- and the
fix's own explanatory comments contain that string, so **the guard was matching
its own documentation.** It now walks the AST and counts only executable shapes
(`setattr(..., "<const>", ...)` and `<x>.<const> = ...`); comments and
docstrings are invisible to `ast.parse`, so they cannot satisfy it by
construction rather than by a rule someone has to remember. Mutation cell **P3**
restores the substring check and is KILLED.

## 6. The durable guard -- the part that addresses the CLASS

`backend/tests/conftest.py` gains a third pollution guard, beside the existing
BQ and Slack-egress ones. Autouse and **function-scoped**, so it names the test
that leaked rather than reporting that something did. It **repairs the file
before failing** -- a guard that only reports leaves most of the damage.

```
$ <revert the fix, run the rail-guard suite>
9 passed, 2 errors
guard message present: True
heartbeat REPAIRED by the guard: True
```

Scoped to a declared list of two protected files, deliberately not to the whole
tree: a broad snapshot would flag legitimate build artefacts and be switched off
within a week.

**Full-suite impact, measured against the SHIPPED tree.** A global autouse
fixture must be shown not to break the suite:

```
$ pytest backend/tests/ -q -p no:cacheprovider
20 failed, 3621 passed, 12 skipped, 5 xfailed, 1 xpassed  in 511.28s (0:08:31)
  (= 3,659 collected, which includes this step's own 13 tests)

$ grep -c "phase-86.110 test guard"   -> 0     <- MY guard fires on NONE of them
```

**The failing IDs, so membership drift is auditable rather than a bare count:**

```
test_phase_23_2_6_sector_cap_emit::test_..._backend_log_has_skipping_buy_evidence
test_phase_40_2_claude_code_v2_1_140_features::test_..._settings_json_still_valid_json_after_edit
test_phase_57_1_reject_binding::test_off_identity_prompts_are_verbatim_constants
test_phase_57_1_reject_binding::test_reject_binding_main_path_off_emits_on_blocks
test_phase_57_1_reject_binding::test_reject_binding_swap_path_off_emits_on_blocks
test_phase_60_3_data_integrity::test_60_3_flag_defaults_off
test_phase_62_4_sentinel::test_infra_path_distinct_exit
test_phase_75_17_verification_paths::test_masterplan_diff_touches_only_the_ten_sibling_insertions
test_phase_75_17_verification_paths::test_sweep_over_live_masterplan_is_clean
test_phase_75_17_verification_paths::test_sweep_shape_census_matches_the_corrected_figures
test_phase_75_19_preflight_calibration::test_live_masterplan_is_currently_clean
test_phase_75_prompt_contracts::test_operator_decision_note_exists_with_token
test_phase_75_sre_ops::test_c1_runbook_and_operator_token_drafted
test_phase_75_sre_ops::test_c6_no_launchctl_bootstrap_executed_in_ops_scripts
test_phase_82_39_outcome_rebuild_query::test_the_sweeps_recall_limit_is_recorded_not_assumed
test_phase_82_48_outcome_write_schema::test_the_fetch_supplies_every_field_the_write_REQUIRES
test_phase_82_48_outcome_write_schema::test_write_really_persists_into_bigquery
test_phase_82_54_cost_budget_columns::test_production_sql_dry_runs_valid
test_phase_86_6_subprocess_channel::test_the_optin_IS_honoured_so_a_real_window_remains_possible
test_portfolio_swap::test_swap_framework_fills_zero_buy_gap
```

**This set is byte-identical to the run taken before this step's own tests
existed**, so the additions changed no membership. A prior revision of this
block quoted "3,608 passed, 20 failed" -- 3,646 collected, i.e. measured
BEFORE this step's tests were added, while labelled as the shipped tree. The
evaluator caught that; the numbers above are the shipped tree.

**The four masterplan-sensitive failures are NOT caused by this session's
filings.** `test_live_masterplan_is_currently_clean` fails on a dead path
reference in step **86.31** (`.claude/hooks/lib/qa_write_guard.py`,
class `never-existed`) -- pre-existing and unrelated to steps 86.112-86.115.

Then a **controlled** check, because my first attempt changed two variables at
once (full suite vs a 20-test subset, AND the fixture):

```
same 20-test subset, fixture DISABLED -> 19 failed, 1 passed
same 20-test subset, fixture ENABLED  -> 19 failed, 1 passed     <- IDENTICAL
```

So the fixture causes **zero** failures. 19 of the 20 reproduce in isolation;
the 1 that passes in the subset (`test_phase_86_6_subprocess_channel`) is a
pre-existing full-suite order dependency, and it passes on its own.

## 7. A REAL REGRESSION THIS FOUND -- from step 86.108, already pushed

The full-suite run failed
`test_phase_23_2_14_no_reentrant_locks::test_phase_23_2_14_threading_lock_count_matches_roster`:

```
E  phase-23.2.14 roster drift: found 21 threading.Lock() instances (expected 20).
E  Re-audit required + bump EXPECTED_LOCK_COUNT in same commit.
E  ... 'backend/agents/parse_failure_ledger.py:110' ...
```

**86.108 shipped that lock and never ran this guard**, because its regression
sweep used a `-k` selection that did not include this file. Main carried a red
test. The guard did exactly what it exists for; the **scope of the sweep** was
the defect. *A `-k` selection is not a regression suite.*

Re-audited per the file's own rule and bumped 20 -> 21: the new lock is a plain
`threading.Lock` (not an `RLock`), and under it the code appends to a bounded
deque and increments three Counters, calling **no** function that could
re-acquire it -- `resolve_rail()` runs before the lock is taken and
`logger.warning` after it is released, both deliberately. No `_*_locked` helper
was introduced, so this file's rules 2 and 3 are vacuously satisfied for it.

## 8. Criterion 5 -- disposition: REGENERATED from the ledger

Not "left alone". The file held `c2` at build time, and `c2` appears in **0** of
the 174 ledger rows -- the dashboard would have been reporting liveness off a
value no cycle ever produced.

```
BEFORE: {"cycle_id": "c2",        "event": "end", "updated_at": "2026-08-17T21:03:54.974417+00:00"}
AFTER : {"cycle_id": "3e5afddb",  "event": "end", "updated_at": "2026-08-17T19:47:15.758944+00:00"}

DERIVED, not manufactured: every field comes from the ledger's last completed
row. cycle_id=3e5afddb has 2 ledger rows; the fake 'c2' it replaces has 0.
```

This is criterion 5's first option ("regenerated from the real last-completed
cycle in `cycle_history.jsonl`"). The second option -- leave it for the next
real cycle -- was rejected **because the value standing there was a lie, not
merely stale**, and the next cycle is up to a day away.
`test_the_heartbeat_value_exists_in_the_real_cycle_ledger` now asserts the
property rather than the value, so it stays true after the next real write.

## 9. Criterion 6 -- mutations, control GREEN first

```
$ python scripts/qa/mutation_86_110.py
CONTROL rc=0  collected=13

P1 KILLED   reverting the isolation at the funnel-counts site is caught BEHAVIOURALLY
P2 KILLED   a sweep that stops recognising monkeypatch.setattr is caught
P3 KILLED   the substring check a comment satisfies -- the sweep's own original defect
P5 KILLED   a runtime guard that never fires is caught
P6 KILLED   a guard that reports without repairing -- leaving most of the damage
P7 KILLED   an empty protected set -- the guard passes vacuously over no files
P8 KILLED   a tolerance rule that makes the guard blind to real leaks is caught
P9 KILLED   treating a rewrite/truncation of the append-only ledger as an append
P10 KILLED  accepting a fixture cycle_id that appears in ZERO ledger rows as legitimate
P11 KILLED  dropping ONLY the prefix half -- survived before the longer-rewrite fixture

KILLED=10/10  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256, and the real
heartbeat was restored after every cell.
```

**P4 was REMOVED as an EQUIVALENT MUTANT, and the removal is itself a finding.**
Narrowing `WRITERS` to `("record_cycle_end",)` -- the filing's own narrower set
-- changes no row in this repo, because **no file calls `record_cycle_start`
without also calling `record_cycle_end`** (measured: zero files under `backend/`
and `scripts/`). Scoring it KILLED would have required pointing it at a test
that fails for an unrelated reason. The wider set is still correct -- it is what
keeps the sweep right the day someone adds a start-only caller -- but this
matrix cannot prove that today, and saying so beats manufacturing a kill.

**Two cells were UNSCORABLE on the first run and the cause was my runner, not
the guards.** Several tests spawn a NESTED pytest whose summary is captured into
the failure text, so summing every `N passed` over the whole output counted the
inner run too (20 != 10). The counter now reads only the final summary line.

## 10. Suite and lint

```
$ pytest backend/tests/test_phase_86_110_heartbeat_isolation.py -q
13 passed in 4.72s

$ pytest backend/tests/ -q -k "cycle_health or heartbeat or rail_guard or 38_2 or 86_38 or 23_2_14 or 86_110"
71 passed, 3588 deselected

$ uvx ruff check --select F821,F401,F811 --no-cache <8-file DERIVED scope>
All checks passed!
RUFF_EXIT=0
```

## 12. Cycle 2 -- closing the CONDITIONAL

Verdict `wf_e7115d07-ae1`: all six criteria MET and independently reproduced,
capped by two WARN findings. **The first was a real defect in the guard I
shipped, and it is the sharpest finding of this step.**

### F1 -- the guard could DESTROY production data

`handoff/cycle_history.jsonl` is an **append-only ledger the live
`autonomous_loop` writes from this same machine** (a real row was appended at
2026-08-17T19:47:15.758944Z), and a full suite run takes ~8.5 minutes. The
original guard restored its snapshot unconditionally, so a real append landing
mid-suite would have been **reverted -- a silently deleted production cycle row,
plus a failure message blaming an innocent test.** Strictly worse than the leak
it was built to stop, and undisclosed in either scope-honesty section.

Fixed by giving each protected file a rule for what a legitimate concurrent
write looks like:

| file | rule | legitimate iff |
|---|---|---|
| `handoff/cycle_history.jsonl` | `append_only` | the new content **starts with** the snapshot and is longer -- a real append. A rewrite or truncation is not. |
| `handoff/.cycle_heartbeat.json` | `ledger_backed` | its `cycle_id` **exists in the real ledger**. That is the same property that made `c2` a lie rather than merely stale: it appears in 0 of 174 rows. |

When the rule says legitimate, the guard leaves the file alone and does not
fail. When in doubt it returns False -- a false alarm costs a re-run, a false
clear costs an undetected pollution.

Making a guard tolerant risks making it blind, so all three arms are mutated:
**P8** (tolerance always true -> guard blind), **P9** (a rewrite counts as an
append), **P10** (any `cycle_id` counts as real). All KILLED, and
`test_the_guard_STILL_catches_a_test_leak_after_the_concurrency_fix` is the
standing anti-regression.

### F2 -- the full-suite block did not reproduce

It was measured before this step's own tests existed (3,646 vs 3,659 collected)
while labelled as the shipped tree, and quoted a bare count with no failing
IDs. Both fixed in §6: current numbers, and the full ID list.

### What the evaluator verified that this step should not re-open

It re-ran the pre-fix HEAD source itself and saw the guard fire twice, naming
both tests. It mutated **each duplicated site individually** -- which the
author's P1 (site 2 only) did not establish -- and found both individually
covered. It recomputed the two SHA-256 values in §2 from the stated content and
confirmed they are not a spliced capture. It confirmed the P4 equivalent-mutant
claim is true, and that the 86.108 lock-roster regression is genuinely repaired.

## 13. Cycle 3 -- closing the second CONDITIONAL

Verdict `wf_8275f3fa-266`: all six criteria MET, both cycle-1 blockers
confirmed closed, capped by two evidence-side findings. Both closed.

### F1 -- only the redundant half of my own rule was tested

The evaluator mutated `_is_legitimate_concurrent_write`'s `append_only` rule
from `after.startswith(before) and len(after) > len(before)` to
`len(after) > len(before)` -- dropping the prefix check entirely -- and **it
SURVIVED 13/13.**

The reason is the fixtures, not the rule: all three cases were **length-only
discriminable** (the rewrite is 20B against a 42B snapshot, the truncation is
0B, the no-growth case is equal), and the length clause is redundant given
`startswith` because equal content is short-circuited upstream by the hash
compare. So the suite exercised only the half that does nothing. Cell P9
mutates the *whole compound return*, which the length-only cases do catch --
which is how the matrix read 9/9 while one clause had no falsifying fixture at
all.

Closed with the case the evaluator named: **a rewrite LONGER than the
snapshot**, which is not a prefix and cannot be caught by length. New cell
**P11** mutates the prefix half alone and is KILLED. Matrix **10/10**.

The general lesson, and it is one this session hit twice: *a matrix cell that
mutates a compound expression licenses nothing about its sub-expressions.*

### F2 -- my residual (a) was wrong in both count and mechanism

I wrote "all three already isolate `_HEARTBEAT_PATH`". Derived properly:

```
$ <every .py calling run_daily_cycle(, with its isolation mechanism>
TRANSITIVE REACHERS: 8   (6 test-side, 2 production)

  backend/api/paper_trading.py                       production caller
  backend/services/autonomous_loop.py                production caller
  backend/tests/test_phase_36_12_kill_switch_...py   stubs get_log      <- THIRD idiom
  backend/tests/test_phase_36_17_halt_stop_loss...py stubs get_log      <- THIRD idiom
  backend/tests/test_phase_85_4_cycle_loudness.py    isolates _HEARTBEAT_PATH
  backend/tests/test_phase_85_6_anchor_deadlock.py   isolates _HEARTBEAT_PATH
  tests/services/test_autonomous_loop_async.py       never executes it (source regex only)
  tests/verify_phase_25_B3.py                        never executes it (source regex only)
```

**Six test-side reachers, not three; only two isolate the constant; two use a
THIRD idiom (stubbing `cycle_health.get_log`) that my sweep does not model at
all; and two never execute the function.** The substantive conclusion --
**none of them leaks** -- survives, and the evaluator verified that
independently. But the enumeration and the mechanism were both wrong, and a
coverage claim whose census is wrong is not a coverage claim.

### A scope bound the evaluator flagged, now stated

The conftest guard lives in `backend/tests/conftest.py`, so it does **not**
cover the root `tests/` tree, which the sweep does scan. Measured: the two
`tests/`-tree reachers never execute `run_daily_cycle`, so nothing there leaks
today -- but the guard's reach is narrower than "any test", and that is a
property of where the file sits rather than a decision.

## 11. Scope honesty

- **`cycle_health`'s two-constant design is the root cause and is NOT
  refactored.** A module whose one public call writes two paths will keep
  producing this bug; the sweep and the conftest guard contain the class, they
  do not remove it. A refactor gets its own step.
- **`scripts/smoketest_stages_5_through_13.py` was not edited.** The sweep shows
  it reaches no writer, so it does not leak; it is reported, not touched.
- **Nothing was promoted and no `.env` was written.**
- **No step was flipped and no prior verdict altered** (criterion 6). The
  `EXPECTED_LOCK_COUNT` bump is a test constant, not a verdict or a status.
- **The heartbeat file IS modified in this step's diff**, deliberately and per
  criterion 5, with the derivation shown in §8.
- **The guard's reach stops at `backend/tests/`.** It does not cover the root
  `tests/` tree that the sweep scans. Nothing there leaks today (both reachers
  are source-regex only), but that is a measured fact, not a design guarantee.
- **The guard writes production state, and that is now bounded rather than
  assumed away.** It restores a protected file only when the change does not
  match a legitimate-concurrent-write rule (§12). It cannot revert a real
  append to the cycle ledger, and it cannot revert a heartbeat naming a real
  cycle. It CAN still restore a change no rule recognises -- that is the
  intended behaviour, and the trade is stated rather than hidden.
