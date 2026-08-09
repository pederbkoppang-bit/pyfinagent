# Experiment results -- phase-36.17

**Step:** 36.17 (P1) -- a halted cycle stops enforcing stop-losses.
**Contract:** `handoff/current/contract_36.17.md` (written BEFORE any code).
**Research:** `handoff/current/research_brief_36.17.md` (gate PASSED, `wf_7b26264d-462`).
**Operator decision:** option **(b)**, recorded in the contract §4 before GENERATE began.

> **Structure note (cycle 6).** Cycles 2-5 grew four meta-sections narrating the
> Q/A process. Each one added claims, anchors and numbers -- i.e. new attack
> surface -- and the compounding was measured: a correct "3 of 44 anchors" became
> wrong ("3 of 48") purely because the sentence reporting it added four more.
> That narrative now lives where it belongs: verbatim verdicts in
> `evaluator_critique_36.17.md`, cycle-by-cycle history in `handoff/harness_log.md`.
> **This file states what was built and carries its proof.** §11 keeps only the
> disclosures, which are not narrative.

---

## 1. What was built

A **SELL-only, exit-only stop-loss pass inside the halt branch** of
`run_daily_cycle`, before its `return summary`.

**Placement:** after `trader.mark_to_market()` (so the stop comparison sees fresh
prices) and before `trader.save_daily_snapshot()` (so the snapshot reflects any
exit). `final_state` at that line is assigned but never read on the halt path --
the halt-path assignment is the **first** of two, and every read of `final_state`
sits below the **second**, hence below the halt's `return summary`. Stated as a
RELATION rather than as line numbers, because the numbers are what broke twice.

**Scope guard:** `if not ks_check.get("triggered"):`. Used rather than a string
comparison on `halt_reason` because `cycle_halt_reason` returns `"breach"` *iff*
`ks_check.get("triggered")` is truthy -- the boolean is the authoritative source
and cannot drift if the reason string is reworded.

**Three deliberate omissions**, each mutation-tested in §7:

1. **Does not run on the `triggered` (breach) path** --
   `check_and_enforce_kill_switch` has already called `flatten_all`, so a second
   pass would duplicate exits, fee events and learn-loop rows over positions that
   no longer exist.
2. **Does not call `backfill_missing_stops`.** Synthesizing a stop level is a NEW
   risk decision (ESMA para 11(5)); the synthesized price
   (`avg_entry_price * (1 - 8%)`) can land ABOVE the current mark, converting
   "this position has no stop" into "sell it at market now" -- a flatten by side
   effect on exactly the branches that deliberately do not flatten.
3. **Does not append to `summary["steps"]`.** It records under the distinct key
   `summary["halt_stop_loss_triggered"]`. Appending was measured to turn two
   `test_phase_36_12...` tests (`:298` and `:374`) RED.

**Failure handling:** the pass is wrapped so an exception records
`summary["halt_stop_loss_error"]` and logs at `exception` level, but never
prevents the halt from completing -- the phase-85.4 loudness guards depend on the
terminal `status`/`halt_reason` set above it. `return summary` remains the last
statement, which is what suppresses BUYs on the `blocked` path (where the switch
is not paused, so `execute_buy`'s own gate returns None).

**Caller count, re-derived.** Before this change `check_stop_losses` had exactly
ONE production caller, which is what set severity: the cycle was the sole
enforcement path, so a halted cycle had none. **This change is what made that
statement stale** -- there are now two:

```
$ grep -rn "trader.check_stop_losses" backend --include="*.py" | grep -v /tests/
backend/services/autonomous_loop.py:1474:                        halt_stops = await asyncio.to_thread(trader.check_stop_losses)
backend/services/autonomous_loop.py:1544:            triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
```

The command is narrowed to `trader.check_stop_losses` deliberately: the bare
symbol also matches the method DEFINITION in `paper_trader.py` and eight comment
lines, so it does not answer "how many call sites".

## 2. Files changed

Derived from the step's own commits, not hand-listed:

```
$ git log --name-only --format= --grep="(36.17)" -- backend/ scripts/ | sort -u | grep -v '^$'
backend/services/autonomous_loop.py
backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py
scripts/qa/mutation_matrix_36_17.py
scripts/qa/verify_36_17_anchors.py
```

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | **the only production change.** Two commits: `e98ca260` (+70, the exit-only pass) and `6ca17793` (+5/-2, comment-only). No other hunk, no other production file. |
| `backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py` | NEW -- 8 tests + isolation fixtures. |
| `scripts/qa/mutation_matrix_36_17.py` | NEW (cycle 6) -- the criterion-7 matrix, re-runnable, and it never writes to the repo. |
| `scripts/qa/verify_36_17_anchors.py` | NEW (cycle 3) -- re-executes every quoted command block and checks prose anchors by content. Carries `--self-test`. |

Contract, research brief and live_check are under `handoff/current/`.

## 3. Criteria 1 + 2 -- REPRODUCE FIRST, recorded verbatim

Two temporary tests asserting the DEFECT (`check_stop_losses.called is False`,
`execute_sell.called is False`) were added and run **against the un-fixed tree**,
with a position priced `41.0` against a `46.0` stop and
`trader.check_stop_losses.return_value = ["WDC"]` set explicitly so the test can
never be vacuous.

**PRE-FIX run (defect present):**

```
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_phase_36_17_paused_cycle_enforces_preexisting_stops
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_phase_36_17_blocked_cycle_enforces_preexisting_stops
2 failed, 4 passed, 3 warnings in 16.24s
```

with the blocked-path failure reading verbatim:

```
>       assert trader.check_stop_losses.called is True, (
            "a blocked cycle did not check stop-losses (phase-36.17)"
        )
E       AssertionError: a blocked cycle did not check stop-losses (phase-36.17)
E       assert False is True
E        +  where False = <MagicMock name='mock.check_stop_losses' id='4710405792'>.called
```

The 4 that passed pre-fix were the **2 REPRODUCE tests** (defect confirmed on
BOTH the `paused` and `blocked` paths) plus `triggered_path_is_unchanged` and
`halt_summary_shape_is_preserved` (correct baselines).

**POST-FIX run of the SAME file -- an exact inversion:**

```
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_REPRODUCE_paused_cycle_never_checks_stops
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_REPRODUCE_blocked_cycle_never_checks_stops
2 failed, 4 passed, 3 warnings in 14.57s
```

The temporary REPRODUCE tests were then removed -- the suite must not carry a
test asserting the broken behaviour is correct. The transcript above is preserved
in the test module's header comment and here.

## 4. Criterion 3 -- the option chosen, and why

**Option (b)**, decided by the operator on 2026-08-09 and recorded in
`contract_36.17.md` §4 **before** GENERATE began: a stop-loss-only pass inside
the halt branch, SELL-only, scoped to `paused` and `blocked`, excluding
`backfill_missing_stops`, with `return summary` still last.

(a) was rejected because it alters the healthy path (double exit plus fee and
learn-loop duplication on a breach cycle) and runs the backfill everywhere. (c)
was rejected because it is defensible only with a human watching, is empirically
refuted by the MCX protective-order incidents, and is not free -- it would
require building an operator stop-check tool and a halted-with-open-positions
alert, neither of which exists. There is no silent fourth option: what shipped is
(b) as written.

## 5. Criterion 4 -- stops enforced AND no BUY

Asserted in both enforcement tests:

- `trader.check_stop_losses.called is True`
- an `execute_sell` call with `reason == "stop_loss_trigger"` for `WDC`
- `summary["halt_stop_loss_triggered"] == ["WDC"]`
- **`trader.execute_buy.called is False`**
- **`decide.called is False`** (the cycle never reaches decide/execute)
- `trader.backfill_missing_stops.called is False`
- `state._paused is True` (the halt was not cleared)

The no-BUY assertion is **falsifiable, not decorative**. As first written it
could not fail -- `decide_trades` was stubbed to `[]`, making `execute_buy`
structurally unreachable -- so the harness now stubs a real `TradeOrder` and
patches `paper_trader._get_live_price`. Mutant **M1** in §7 (delete the halt's
`return summary`) is the standing proof: it fires on `execute_buy.called`.

## 6. Criterion 5 -- `triggered` path unchanged, asserted against a fixture

`test_phase_36_17_triggered_path_is_unchanged` drives the real cycle with
`KS_TRIGGERED` and asserts `check_stop_losses.called is False` and
`"halt_stop_loss_triggered" not in summary`. A fixture-driven assertion, not a
reasoned claim. Mutant **M3** kills it.

## 7. Criterion 6 -- line numbers RE-DERIVED at fix time

```
$ grep -n "return summary" backend/services/autonomous_loop.py | head -1
1510:                return summary
$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1512:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

**Read the RELATION, not the numbers:** the halt's `return summary` is the line
IMMEDIATELY BEFORE the Step 5.6 header. The ordering is unchanged; option (b)
adds enforcement *inside* the halt branch rather than moving Step 5.6. That
relation is what `scripts/qa/verify_36_17_anchors.py` asserts structurally, and
it is the only form of this claim that cannot go stale.

Superseded anchors, kept only to show the drift: the masterplan step text's
`:1334/:1336` and the pre-fix tree's `:1437/:1439` are both **STALE** and do not
reproduce.

## 8. Criterion 7 -- MUTATION TEST, both directions

**Cycle 6 made this re-runnable, which it was not before.** Cycles 1-5 used a
scratch harness that was never committed, so the recorded transcript could not be
regenerated by anyone -- and the artifacts drifted into recording a 9-cell run
while claiming 11 cells. `scripts/qa/mutation_matrix_36_17.py` is now the
authority, and the block below is one real run of it.

The harness **never writes to the repository**: each mutant is applied in memory
inside a throwaway subprocess that registers the mutated module in `sys.modules`
before pytest imports anything. It asserts each anchor matches **exactly once**
(a no-match `str.replace` is indistinguishable from success), requires a GREEN
baseline first (a "killed" mutant proves nothing on a red tree), runs each cell
against the WHOLE module so the transcript names the killing tests, and digests
the target before and after.

```
$ python scripts/qa/mutation_matrix_36_17.py
phase-36.17 criterion 7 -- mutation matrix
target   : backend/services/autonomous_loop.py
md5      : 58bbf24bde4c5161ac05f26f70fb264e  (read-only; mutants are in-memory)
[baseline] un-mutated tree: 8 passed, 1 warning in 14.96s
  KILLED  | M1: delete the halt's `return summary` (cycle falls through to decide/execute)
           proves: criterion 4 no-BUY -- the halt must return before Step 6
           tests : test_phase_36_17_paused_cycle_enforces_preexisting_stops, test_phase_36_17_blocked_cycle_enforces_preexisting_stops, test_phase_36_17_triggered_path_is_unchanged, test_phase_36_17_halt_summary_shape_is_preserved, test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts
           result: 5 failed, 3 passed, 1 warning in 16.50s
  KILLED  | M2: ORDERING REVERTED: disable the exit-only pass entirely (pre-fix behaviour)
           proves: criteria 1+2+7 -- moving the enforcement back must break the reproduce pair
           tests : test_phase_36_17_paused_cycle_enforces_preexisting_stops, test_phase_36_17_blocked_cycle_enforces_preexisting_stops, test_phase_36_17_a_failed_sell_is_not_recorded_as_enforced, test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts, test_phase_36_17_the_halt_exit_is_a_FULL_exit_not_a_partial, test_phase_36_17_stops_are_checked_against_FRESH_marks
           result: 6 failed, 2 passed, 1 warning in 15.48s
  KILLED  | M3: remove the breach-path exclusion (the pass runs on EVERY halt reason)
           proves: criterion 5 -- the `triggered` path must stay observably unchanged
           tests : test_phase_36_17_triggered_path_is_unchanged
           result: 1 failed, 7 passed, 1 warning in 15.37s
  KILLED  | M4: reintroduce backfill_missing_stops into the halt pass
           proves: research Q3 -- synthesizing a stop during a halt is a NEW risk decision
           tests : test_phase_36_17_paused_cycle_enforces_preexisting_stops, test_phase_36_17_blocked_cycle_enforces_preexisting_stops, test_phase_36_17_stops_are_checked_against_FRESH_marks
           result: 3 failed, 5 passed, 1 warning in 14.56s
  KILLED  | M5: append to summary["steps"] inside the halt branch
           proves: measured collision -- two phase-36.12 tests pin summary['steps'][-1]
           tests : test_phase_36_17_halt_summary_shape_is_preserved
           result: 1 failed, 7 passed, 1 warning in 13.83s
  KILLED  | M6: drop the recorded ticker (silent enforcement, no audit surface)
           proves: the exit must be REPORTED in the summary, not just performed
           tests : test_phase_36_17_paused_cycle_enforces_preexisting_stops, test_phase_36_17_blocked_cycle_enforces_preexisting_stops, test_phase_36_17_the_halt_exit_is_a_FULL_exit_not_a_partial
           result: 3 failed, 5 passed, 1 warning in 14.86s
  KILLED  | M-D: `if sl_trade:` -> `if True:` (record an exit that never happened)
           proves: Q/A cycle-3 survivor -- the summary must not claim a stop it did not take
           tests : test_phase_36_17_a_failed_sell_is_not_recorded_as_enforced
           result: 1 failed, 7 passed, 1 warning in 14.38s
  KILLED  | M-E: drop summary['halt_stop_loss_error'] (silent swallow)
           proves: Q/A cycle-3 survivor -- a stop-loss failure must not be swallowed
           tests : test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts
           result: 1 failed, 7 passed, 1 warning in 13.62s
  KILLED  | M-F: logger.exception -> logger.debug (downgrade the loudness)
           proves: paired with M-E; the log level is the half the summary key cannot carry
           tests : test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts
           result: 1 failed, 7 passed, 1 warning in 13.12s
  KILLED  | MUT-B: `quantity=None` -> `quantity=1` (a ONE-SHARE exit reported as enforced)
           proves: Q/A cycle-5 survivor -- a partial fill must not count as a full exit
           tests : test_phase_36_17_the_halt_exit_is_a_FULL_exit_not_a_partial
           result: 1 failed, 7 passed, 1 warning in 13.07s
  KILLED  | MUT-C: delete the halt-path mark_to_market (stops compared against STALE marks)
           proves: Q/A cycle-5 survivor -- §1 claims mark freshness as load-bearing
           tests : test_phase_36_17_stops_are_checked_against_FRESH_marks
           result: 1 failed, 7 passed, 1 warning in 13.92s
[restored] un-mutated tree: 8 passed, 1 warning in 13.63s
[integrity] target md5 unchanged: True (58bbf24bde4c5161ac05f26f70fb264e)
ALL 11 MUTANTS KILLED -- every guard IN THIS MATRIX can fail.
```

**M2 is the "both directions" half of criterion 7**: reverting the enforcement to
pre-fix behaviour turns the reproduce pair RED.

**Four of these eleven cells exist because a Q/A found them SURVIVING**, and two
of those are money-path lies rather than style:

- **M-D** -- `execute_sell` returns None when the position is already gone, so
  the summary recorded a stop as ENFORCED when no sell occurred.
- **MUT-B** -- `paper_trader.py:548` does
  `sell_qty = quantity or position["quantity"]`, so a mutant passing
  `quantity=1` liquidates **one share**, still returns a truthy trade record, and
  the ticker is still appended. The book keeps essentially full exposure while
  the summary reports the stop as ENFORCED. The M-D guard could not see it -- a
  partial fill returns a trade record exactly like a full one. Closed by asserting
  the **argument** (`quantity is None`), not the outcome.
- **MUT-C** -- `check_stop_losses` compares `current_price`, which
  `mark_to_market` refreshes, so deleting the halt-path mark runs the comparison
  on **stale marks**. §1 claims freshness as load-bearing and it had zero covering
  assertion. **My first guard for it also failed and I measured that rather than
  assuming it**: there are two `mark_to_market` calls in the cycle and `.index()`
  returns the first, so the assertion passed even with the halt-path call
  deleted. Corrected to assert the **immediate predecessor** of
  `check_stop_losses`.

## 9. Immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2894 deselected, 1 warning in 15.93s
```

exit code **0**, captured on the tree as committed at `d68f69e5`.

> The `deselected` count MOVES every time this step adds a test, because `-k`
> deselects everything outside the filter. **The load-bearing numbers are
> `224 passed, 1 skipped, exit 0`.**

The phase-36.12 module the brief flagged as a collision risk, run alone:
`25 passed, 1 warning in 8.67s`.

**Lint gate** (`qa.md` §1a), on a DERIVED file set, not a hand-typed one --
`ruff check --select F821,F401,F811` over the 3 changed `.py` files:
`All checks passed!`, exit 0.

### 9b. The anchor verifier's own blind spots, carried into the evidence

`scripts/qa/verify_36_17_anchors.py` re-executes every fenced `$ <cmd>` block it
can run safely and **prints a `note:` for every one it cannot**. The cycle-5 Q/A's
complaint was that those notes existed only on stdout, so a reader of the
evidence never saw the holes. They are reproduced here verbatim:

```
note: experiment_results_36.17.md: `python scripts/qa/mutation_matrix_36_17.py` not re-executable
note: experiment_results_36.17.md: `source .venv/bin/activate && python -m pytest backend/tests/` not re-executable
note: live_check_36.17.md: `source .venv/bin/activate && python -m pytest \` not re-executable
note: live_check_36.17.md: `python -m pytest backend/tests/test_phase_36_17_halt_stop_lo` not re-executable
note: live_check_36.17.md: `source .venv/bin/activate && python -m pytest backend/tests/` not re-executable
```

So **five blocks in this evidence set are not machine-verified**: the mutation
matrix and four pytest captures. pytest is deliberately excluded because
re-running it from inside the verifier is slow and **unsafe concurrently** -- the
module's autouse fixtures byte-compare live files, so a second pytest run turns
the first RED (measured: a false `3 failed`). Both remaining checks (B and C)
pass, and the run exits 0.

**The counts are deliberately NOT quoted here.** The verifier prints its own
block count and anchor recall, and those move every time this artifact is
edited -- that is precisely how a correct "3 of 44" became a wrong "3 of 48" in
cycle 5. **Run the tool for the current figures.**

## 10. Isolation -- measured over the WHOLE live-state set, and one leak found and closed

Cycles 1-5 recorded md5s of **three git-tracked** handoff files and reported
isolation held. That was true but **narrower than it sounded**: it excluded
`handoff/.autonomous_loop.lock`, which is **untracked** -- and which this module
was actually writing.

**Measured, twice, and attributed to this module alone:** running only this test
file rewrote the live lock, leaving the pytest process's own pid in it:

```
before run: 0e9d824adc075e7a250ebf329eafea45
after  run: c4cd2da2fba6a7a005f7d2d6410d48f6
lock content after: {"pid": 22914, "cycle_id": "cycle-1786300366", ...,  "state": "released"}
```

Cause: these tests drive the **real** `run_daily_cycle`, which acquires the real
`cycle_lock`. Consequences, stated at their true size:

- It **cannot** steal a live cycle's lock. phase-85.5 deleted the
  stale-reacquire branch, so contention simply raises.
- It **can** make a scheduled cycle raise `CycleLockError` and be skipped while a
  test holds the lock, and it pollutes the operator's live-state forensics -- a
  dead pid with a ~2s lifetime in that file is exactly what made an earlier
  session misread "is a cycle running?".

**Closed** by redirecting `cycle_lock._LOCK_PATH` to `tmp_path`, plus an autouse
byte-comparison guard so removing the redirect can never be a silent regression.
**The guard is proven able to fail** -- with the redirect removed:

```
E       AssertionError: phase-36.17: a test in this module wrote to the LIVE cycle lock
        /Users/ford/.openclaw/workspace/pyfinagent/handoff/.autonomous_loop.lock.
        Redirect cycle_lock._LOCK_PATH to tmp_path.
E       assert b'{"pid": 233...: "released"}' == b'{"pid": 229...: "released"}'
```

The test file was restored byte-identically afterwards (md5
`d8bfb1ff2ee4554d67e9dfe6cc1fdf5d` before and after that demonstration).

**After the fix, the whole live-state set is byte-identical across a full run**
(these digests are point-in-time and are NOT re-executable evidence -- the live
backend writes two of these files on its own schedule):

```
handoff/kill_switch_audit.jsonl   685bf1a5fd7beaa4f15da2babf133ca2   identical after
handoff/cycle_history.jsonl       6bc251737c8145e0b3891ed1cc5d4b2c   identical after
handoff/.cycle_heartbeat.json     d4a8ba2de8f35348e4df8f775b6a254d   identical after
handoff/.autonomous_loop.lock     ee1ba590743c0cfe00cc72848d5a3260   identical after   <- was leaking
handoff/away_ops/health.jsonl     1e27da828e5f68581c0f94da49ba671e   identical after
```

**Scope, stated so it is not over-read:** this closes the leak for THIS module.
Whether other test modules take the live lock is **not** claimed here -- that
sweep is phase-86.6's, and this is a reproducible instance for it.

The module also carries `captured_alerts` (intercepts `raise_cron_alert_sync`,
which posts to the operator's REAL Slack with no test guard), a
`cycle_health.get_log` stub, `settings.news_screen_enabled = False` and a mocked
`AnalysisOrchestrator` (an unstubbed run makes a REAL ~150s LLM call).

## 11. Disclosures -- self-reported; no automated check would have surfaced these

1. **A tool timeout once left a mutant in the production file** for ~4 minutes
   during cycle 3. Never committed; never executed by any running process
   (the backend had already imported the module). Restored byte-identically
   against `git show`, verified by digest. **Structurally prevented now:** the
   cycle-6 harness never writes to the repo at all.
2. **The cycle-4/5 mutation runs mutated the live production file eleven times
   while an ARMED backend was running.** No harm occurred, and the reason is
   mechanism not luck -- CPython serves an imported module from `sys.modules`
   and never re-reads the file. But a restart inside any of those windows would
   have imported a mutant into an armed trading process. **This is the hazard
   the cycle-6 harness was written to remove.**
3. **A false regression was nearly reported.** A pytest run showed `3 failed`;
   it was a concurrency artifact of a second pytest run tripping the autouse
   live-audit fixture. A clean re-run was green. pytest re-execution was
   therefore removed from the anchor verifier: a guard that manufactures false
   regressions is worse than one with a declared gap.

## 12. What I could NOT verify, stated plainly

- **No live cycle has exercised this path.** All evidence is in-process against
  the real `run_daily_cycle` with a mocked `PaperTrader`. Producing live evidence
  means deliberately halting the book -- an operator action, not taken. The
  step's live_check asks for verbatim test output, which is what is recorded.
- **`check_stop_losses` still silently no-ops on a NULL stop or a 0/None price**
  (`if stop and current`, `paper_trader.py:804`). A halted book whose positions
  have NULL stops still gets nothing, even with this fix. Deliberately NOT closed
  here -- closing it means backfilling, which §1 rejects. The recommendation is
  to ALERT on NULL stops during a halt; queued separately, not smuggled in.
- **The Step 5.4 scale-out ordering hazard is untouched** -- a different defect
  class (commission, not omission), currently DARK
  (`paper_scale_out_enabled=False`, `settings.py:35`). Queued as its own step.
- **The anchor verifier cannot check quotations of its own output.**
  Self-invocation recurses, so blocks quoting it sit in its one structural blind
  spot; they are labelled ABRIDGED where they appear. This is unsolved in
  general -- run the tool for authoritative output.
- **`scripts/qa/mutation_matrix_36_17.py` mutates only
  `autonomous_loop.py`.** The test-isolation guard added in §10 is proven
  falsifiable by the recorded manual demonstration, not by a matrix cell.
- **The Knight Capital 2012 SEC order** could not be fetched (403) and is cited
  nowhere.

## 13. Cycle history

Five prior Q/A cycles: CONDITIONAL, FAIL, FAIL, CONDITIONAL, CONDITIONAL. The
production code has been correct and **byte-identical since cycle 2** (md5
`58bbf24bde4c5161ac05f26f70fb264e`, confirmed by four separate Q/A passes) and is
IN FORCE. Every remaining finding was about evidence quality.

- Verbatim verdicts: `handoff/current/evaluator_critique_36.17.md`.
- Cycle-by-cycle history: `handoff/harness_log.md`.

**IN FORCE proof** (content-last-changed, not mtime -- mtime is not durable, and
a same-content rewrite by a mutation run pushed it past process start, making the
old proof read backwards): production content last changed at commit `6ca17793`,
**2026-08-09 17:54:37 +0200**; the running backend is **pid 6644, started
18:56:00**. Content change precedes process start, and the tree is clean at the
unchanged md5:

```
$ md5 -q backend/services/autonomous_loop.py
58bbf24bde4c5161ac05f26f70fb264e
```
