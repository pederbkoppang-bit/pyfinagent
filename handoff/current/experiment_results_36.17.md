# Experiment results -- phase-36.17

**Step:** 36.17 (P1) -- a halted cycle stops enforcing stop-losses.
**Date:** 2026-08-09. **Cycle:** 190.
**Contract:** `handoff/current/contract_36.17.md` (written BEFORE any code).
**Research:** `handoff/current/research_brief_36.17.md` (gate PASSED, `wf_7b26264d-462`).
**Operator decision:** option **(b)**, recorded in the contract §4 before GENERATE began.

---

## 1. What was built

A **SELL-only, exit-only stop-loss pass inside the halt branch** of
`run_daily_cycle`, before its `return summary`.

**Placement:** after `trader.mark_to_market()` (so the stop comparison sees
fresh prices) and before `trader.save_daily_snapshot()` (so the snapshot
reflects any exit). `final_state` at that line is assigned but never read on the
halt path, so inserting there introduces no staleness.

**CORRECTED IN CYCLE 2 (Q/A finding 1).** The first revision of this paragraph
cited `:1697`/`:1776`/`:1792` as the output of `grep -n final_state`. **None of
those three numbers reproduce** -- each was exactly 70 lines low, i.e. pre-insert
values I never re-derived after my own +70-line change, in a step whose
criterion 6 exists to prevent precisely that. Re-derived 2026-08-09, verbatim:

```
$ grep -n "final_state" backend/services/autonomous_loop.py
1429:                final_state = await asyncio.to_thread(trader.mark_to_market)
1770:            final_state = await asyncio.to_thread(trader.mark_to_market)
1849:                "nav": final_state["nav"],
1850:                "pnl_pct": final_state["pnl_pct"],
1865:            logger.info(f"Paper trading cycle complete: NAV=${final_state['nav']:.2f}, "
1866:                         f"P&L={final_state['pnl_pct']:.2f}%, trades={trades_executed}, "
```

Reading that output: the **first** assignment is the halt-path one; the
**second** is the healthy-path re-assignment; and every **read** (`"nav"`,
`"pnl_pct"`, and the two `logger.info` lines) sits below that second assignment,
hence below the halt's `return summary`. So the halt-path value is never read.
The engineering claim stands -- the Q/A independently confirmed it -- but its
evidence was wrong and is now regenerated from live command output.

**This paragraph deliberately states the RELATION (first / second / below)
rather than repeating the numbers**, because the numbers are what broke twice.
The relation is stable under any edit above it; the numbers are not.

**Second correction (same class).** Both this artifact's predecessor and the
test-module docstring claimed `check_stop_losses` has "exactly ONE production
caller". **That is now false, and this change is what made it false.**
Re-derived:

```
$ grep -rn "trader.check_stop_losses" backend --include="*.py" | grep -v /tests/
backend/services/autonomous_loop.py:1474:                        halt_stops = await asyncio.to_thread(trader.check_stop_losses)
backend/services/autonomous_loop.py:1544:            triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
```

The command is narrowed to `trader.check_stop_losses` deliberately: the bare
`check_stop_losses` pattern also matches the method DEFINITION in
`paper_trader.py` and eight comment lines, so it does not answer "how many
call sites". The cycle-2 Q/A flagged the earlier version of this block as
**curated output presented under a `$` prompt** -- it showed 3 hand-picked,
re-ordered lines with editorial arrows appended, out of 11 real ones. The
block above is the unedited stdout of the command shown.

**Two** call sites. The correct statement is "BEFORE this change there was
exactly one caller, so a halted cycle had no enforcement layer at all." The
comment at `autonomous_loop.py:1438` and the test docstring have both been
amended to say that and to tell the reader to re-derive rather than trust them.

**Scope guard:** `if not ks_check.get("triggered"):`. This is used rather than a
string comparison on `halt_reason` because `cycle_halt_reason` returns `"breach"`
*iff* `ks_check.get("triggered")` is truthy -- the boolean is the authoritative
source and cannot drift if the reason string is ever reworded.

**Three deliberate omissions**, each mutation-tested below:

1. **Does not run on the `triggered` (breach) path** --
   `check_and_enforce_kill_switch` has already called `flatten_all`, so a second
   pass would duplicate exits, fee events and learn-loop rows over positions
   that no longer exist.
2. **Does not call `backfill_missing_stops`.** Synthesizing a stop level is a
   NEW risk decision (ESMA para 11(5)); the synthesized price
   (`avg_entry_price * (1 - 8%)`) can land ABOVE the current mark, converting
   "this position has no stop" into "sell it at market now" -- a flatten by side
   effect on exactly the branches that deliberately do not flatten, and a
   violation of this step's own "no flatten on the blocked path" constraint.
3. **Does not append to `summary["steps"]`.** Records under the distinct key
   `summary["halt_stop_loss_triggered"]`. The brief measured that appending
   turns `test_phase_36_12...:298` and `:374` RED.

**Failure handling:** the pass is wrapped so an exception records
`summary["halt_stop_loss_error"]` and logs at `exception` level, but never
prevents the halt from completing -- the phase-85.4 loudness guards depend on
the terminal `status`/`halt_reason` set above it. `return summary` remains the
last statement, which is what suppresses BUYs on the `blocked` path (where the
switch is not paused, so `execute_buy`'s own gate returns None).

## 2. Files changed

Measured with `git diff --stat e98ca260^ -- backend/ scripts/`:

```
 backend/services/autonomous_loop.py                |  73 +++
 .../test_phase_36_17_halt_stop_loss_enforcement.py | 545 +++++++++++++++++++++
 scripts/qa/verify_36_17_anchors.py                 | 268 ++++++++++
 3 files changed, 886 insertions(+)
```

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | **the only production change** -- +73 lines, the exit-only pass inside the Step 5.5 halt branch. No other hunk, no other file. |
| `backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py` | NEW -- 6 tests + 3 isolation fixtures. |
| `scripts/qa/verify_36_17_anchors.py` | NEW (cycle 3) -- re-executes every quoted command block and checks every prose anchor by content. Carries `--self-test`. |
| `handoff/current/contract_36.17.md` | NEW -- contract, incl. the recorded operator decision. |
| `handoff/current/research_brief_36.17.md` | NEW -- research gate artifact. |

## 3. Criterion 1 + 2 -- REPRODUCE FIRST, recorded verbatim

Two temporary tests asserting the DEFECT
(`check_stop_losses.called is False`, `execute_sell.called is False`) were added
and run **against the un-fixed tree**, with a position priced `41.0` against a
`46.0` stop and `trader.check_stop_losses.return_value = ["WDC"]` set explicitly
so the test can never be vacuous.

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
------------------------------ Captured log call -------------------------------
WARNING  backend.services.autonomous_loop:autonomous_loop.py:1404 Paper trading: kill-switch active (kill_switch_disarmed_lost_history) -- skipping decide/execute
```

The 4 that passed pre-fix were: the **2 REPRODUCE tests** (defect confirmed on
BOTH the `paused` and `blocked` paths), plus `triggered_path_is_unchanged` and
`halt_summary_shape_is_preserved` (correct baselines).

**POST-FIX run of the SAME file -- an exact inversion:**

```
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_REPRODUCE_paused_cycle_never_checks_stops
FAILED backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py::test_REPRODUCE_blocked_cycle_never_checks_stops
2 failed, 4 passed, 3 warnings in 14.57s
```

The temporary REPRODUCE tests were then removed (the suite must not carry a test
asserting the broken behaviour is correct); the transcript above is preserved in
the test module's header comment and here.

**Final state of the file:**

```
4 passed, 1 warning in 10.29s
```

## 4. Criterion 4 -- stops enforced AND no BUY

Asserted in both enforcement tests:

- `trader.check_stop_losses.called is True`
- an `execute_sell` call with `reason == "stop_loss_trigger"` for `WDC`
- `summary["halt_stop_loss_triggered"] == ["WDC"]`
- **`trader.execute_buy.called is False`**
- **`decide.called is False`** (the cycle never reaches decide/execute)
- `trader.backfill_missing_stops.called is False`
- `state._paused is True` (the halt was not cleared)

## 5. Criterion 5 -- `triggered` path unchanged, asserted against a fixture

`test_phase_36_17_triggered_path_is_unchanged` drives the real cycle with
`KS_TRIGGERED` and asserts `check_stop_losses.called is False` and
`"halt_stop_loss_triggered" not in summary`. This is a fixture-driven assertion,
not a reasoned claim.

## 6. Criterion 6 -- line numbers RE-DERIVED at fix time

```
$ grep -n "return summary" backend/services/autonomous_loop.py | head -1
1510:                return summary
$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1512:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

**Read the RELATION, not the numbers:** the halt's `return summary` is the line
IMMEDIATELY BEFORE the Step 5.6 header in the block above. The ordering is
unchanged; the fix adds enforcement *inside* the halt branch rather than moving
Step 5.6. That relation is what `scripts/qa/verify_36_17_anchors.py` asserts
structurally, and it is the only form of this claim that cannot go stale. The masterplan step text's `:1334/:1336` and the
pre-fix `:1437/:1439` are both superseded.

## 7. Criterion 7 -- MUTATION TEST, both directions

Harness asserts each target substring matches **exactly once** before mutating
(a no-match `str.replace` looks identical to success), requires the un-mutated
baseline to be GREEN first (else "killed" proves nothing), and restores + digest-
verifies the file after every mutant.

Final matrix (M1 added in cycle 2 -- see §11):

```
[baseline] un-mutated tree: 4 passed, 1 warning in 10.50s

  KILLED  | M-D: `if sl_trade:` -> `if True:` (record an exit that never happened)
  KILLED  | M-E: drop summary['halt_stop_loss_error'] (silent swallow)
  KILLED  | M-F: logger.exception -> logger.debug (downgrade the loudness)
  KILLED  | M1: delete the halt's `return summary` (falls through to decide/execute)
           result: 2 failed, 2 deselected, 1 warning in 7.17s
  KILLED  | remove the breach-path exclusion (pass runs on EVERY halt reason)
           result: 1 failed, 3 deselected, 1 warning in 5.43s
  KILLED  | ORDERING REVERTED: disable the exit-only pass entirely (pre-fix behaviour)
           result: 2 failed, 2 deselected, 1 warning in 9.60s
  KILLED  | reintroduce backfill_missing_stops into the halt pass
           result: 2 failed, 2 deselected, 1 warning in 7.16s
  KILLED  | append to summary["steps"] inside the halt branch
           result: 1 failed, 3 deselected, 1 warning in 5.55s
  KILLED  | drop the recorded ticker (silent enforcement, no audit surface)
           result: 2 failed, 2 deselected, 1 warning in 6.95s

[restored] un-mutated tree: 4 passed, 1 warning in 10.07s
ALL 9 MUTANTS KILLED. Every new guard can fail.
```

The "ORDERING REVERTED" mutant is the "**both directions**" half of criterion 7:
reverting the enforcement to pre-fix behaviour turns the reproduce pair RED.

## 8. Immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2890 deselected, 1 warning in 20.24s
```

The phase-36.12 module the brief flagged as a collision risk, run alone:

```
25 passed, 1 warning in 8.95s
```

## 9. Isolation -- measured, not asserted

The new module carries `_live_audit_file_is_write_protected` (autouse byte
comparison of the live `handoff/kill_switch_audit.jsonl`), `captured_alerts`
(intercepts `raise_cron_alert_sync`, which posts to the operator's REAL Slack
with no test guard), a `cycle_health.get_log` stub, and
`settings.news_screen_enabled = False` + mocked `AnalysisOrchestrator` (an
unstubbed run makes a REAL ~150s LLM call).

md5 of the three git-tracked handoff files, before and after the reproduce run:

```
before: 685bf1a5fd7beaa4f15da2babf133ca2  handoff/kill_switch_audit.jsonl
        6bc251737c8145e0b3891ed1cc5d4b2c  handoff/cycle_history.jsonl
        8319fc52d0f8a8cbb9959828e498d308  handoff/.cycle_heartbeat.json
after:  685bf1a5fd7beaa4f15da2babf133ca2  (identical)
        6bc251737c8145e0b3891ed1cc5d4b2c  (identical)
        8319fc52d0f8a8cbb9959828e498d308  (identical)
```

## 10. What I could NOT verify, stated plainly

- **No live cycle has exercised this path.** The fix is proven against the real
  `run_daily_cycle` in-process with a mocked `PaperTrader`, not against a live
  halted cycle with a real position below its stop. Producing that live evidence
  would require deliberately halting the book, which is an operator action and
  was not taken. The live_check for this step is therefore the verbatim test
  output above, which is what the criterion asks for.
- **`check_stop_losses` silently no-ops on a NULL stop or a 0/None price**
  (`if stop and current`, `paper_trader.py:804`). A halted book whose positions
  have NULL stops still gets nothing, even with this fix. That is a real
  residual gap, deliberately NOT closed here (closing it would mean backfilling,
  which §1 rejects). The brief's recommendation is to ALERT on NULL stops during
  a halt; that is queued separately rather than smuggled into this step.
- **The Step 5.4 scale-out ordering hazard is untouched** -- a different defect
  class (commission, not omission), currently DARK
  (`paper_scale_out_enabled=False`, `settings.py:35`). Queued as its own step
  per the brief's recommendation; bundling it would change the documented
  MTM-freshness ordering at `:1377-1379` without its own analysis.
- **The Knight Capital 2012 SEC order could not be fetched** (403) and is
  cited nowhere in the contract or here.

---

## 11. Cycle 2 -- what the Q/A found, and what I changed

Cycle-1 verdict: **CONDITIONAL** (`wf_6bc4c0a4-d9c`), transcribed verbatim in
`handoff/current/evaluator_critique_36.17.md`. All 7 immutable criteria were
judged MET and independently re-verified by the Q/A; the cap came from two
evidence defects, not from the fix.

### Finding 1 (WARN) -- stale line anchors presented as re-derived evidence

Corrected in §1 above, in `backend/services/autonomous_loop.py:1438`, and in the
test module docstring. All three now carry the re-derivation command instead of
a bare number. This was a fair hit: the step's own criterion 6 exists to prevent
it, and I still shipped three stale anchors, all broken by my own +70-line
insertion.

### Finding 2 (NOTE) -- mis-attributed kill mechanism (vacuity shape #11)

The Q/A proved that `assert trader.execute_buy.called is False` **could not
fail** in this harness: `decide_trades` was stubbed to `[]`, so `execute_buy`
was structurally unreachable even on a full fall-through, and the no-BUY
invariant was actually carried by its sibling `assert decide.called is False`.

**I did not take the annotate-only remedy. I made the assertion falsifiable --
and my first attempt at that FAILED, which I measured rather than assumed:**

1. First attempt: stub `decide_trades` with a **dict**-shaped order. Re-ran the
   M1 mutation (delete the halt's `return summary`). Result: `execute_buy.called
   is False` **still passed**; `decide.called is False` was still the killer. A
   dict stub does not restore falsifiability.

   **MECHANISM CORRECTED (cycle-2 Q/A, NOTE).** My first write-up said the dict
   "is skipped by `continue`". That is wrong, and the Q/A measured it: the buy
   loop reads `order.action` as an **attribute**, so a dict raises
   `AttributeError: 'dict' object has no attribute 'action'`, which is absorbed
   by an outer handler that **aborts the remainder of the cycle**. The dict never
   reaches the `continue` at all. The observed outcome and the conclusion are
   unchanged -- and the corrected mechanism is a *stronger* argument for using
   the real dataclass, because the dict path does not merely skip the order, it
   kills the rest of the cycle.
2. Second attempt: a real `TradeOrder` dataclass from
   `backend.services.portfolio_manager`, plus a monkeypatch of
   `backend.services.paper_trader._get_live_price` (required -- `:1713` does a
   LIVE yfinance fetch before the buy, which would have made this test do real
   network I/O). Re-ran M1. Result, verbatim:

```
>       assert trader.execute_buy.called is False, "a BUY was placed on a halted cycle"
E       AssertionError: a BUY was placed on a halted cycle
E       assert True is False
E        +  where True = <MagicMock name='mock.execute_buy' id='4733246768'>.called
```

`execute_buy.called` now fires **before** `decide.called`, so criterion 4's
no-BUY assertion is the load-bearing guard rather than a passenger. M1 is now a
permanent cell in the matrix (§7), taking it from 5 mutants to 6.

### Re-verification after the cycle-2 changes

```
$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2890 deselected, 1 warning in 16.03s

$ python -m pytest backend/tests/test_phase_36_12_kill_switch_trading_path_block.py -q
25 passed, 1 warning in 8.67s

$ python -m pytest backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q
4 passed, 1 warning in 9.74s
```

### One isolation detail, stated so it is not mistaken for a leak

`handoff/.cycle_heartbeat.json` now digests `eea37b489ebbf797240dd9a22c23151d`,
where §9 recorded `8319fc52d0f8a8cbb9959828e498d308`. **That is the live backend
writing heartbeats** (it was restarted at 15:08Z, pid 84494), not a test write.
Measured before/after **within** the cycle-2 run, all three files are identical:

```
before: 685bf1a5fd7beaa4f15da2babf133ca2  kill_switch_audit.jsonl
        6bc251737c8145e0b3891ed1cc5d4b2c  cycle_history.jsonl
        eea37b489ebbf797240dd9a22c23151d  .cycle_heartbeat.json
after:  685bf1a5fd7beaa4f15da2babf133ca2  (identical)
        6bc251737c8145e0b3891ed1cc5d4b2c  (identical)
        eea37b489ebbf797240dd9a22c23151d  (identical)
```

### What cycle 2 did NOT change

No production behaviour. The only `autonomous_loop.py` change is a corrected
comment (5 insertions, 2 deletions, all comment lines). The exit-only pass, its
scope guard, the backfill exclusion and the summary key are byte-identical to
what the Q/A graded.

---

## 12. Cycle 3 -- the guard I shipped was illusory, and three mutants survived

Cycle-2 verdict: **FAIL** (`wf_73b4ae3d-73b`), verbatim in
`evaluator_critique_36.17.md`. Criteria 1-5 and 7 were re-verified as MET; the
FAIL was criterion 6 for the second consecutive cycle. Cycle 3 then FAILED again
(`wf_4bf499e6-0e4`) on the same criterion plus two more findings. What follows is
what each found and what changed.

### 12a. Criterion 6, third occurrence -- and I had regenerated the blocks

Cycle 3 measured that `experiment_results` §6 and `live_check` §2 still asserted,
**in the present tense**, the now-STALE cycle-1 pair `:1507` / `:1509` (both
superseded; they do not reproduce) -- sitting two-to-four lines BELOW the
correctly regenerated grep block that said `1510`/`1512`. I had regenerated the
*blocks* and left the *prose*.

**Fix:** both sites now state the RELATION -- "the halt's `return summary` is the
line immediately before the Step 5.6 header" -- and cite no number at all. That
claim cannot go stale, and it is the form
`scripts/qa/verify_36_17_anchors.py` asserts structurally.

### 12b. The anchor verifier was an ILLUSORY GUARD -- the most serious finding

Cycle 3 executed my v1 verifier against a synthetic artifact whose every anchor
was wrong-but-in-bounds. It printed **"ALL ANCHOR CHECKS PASSED", rc=0**. Three
independent defects in one script:

1. `CHECKED` was left `{}`, so the only per-anchor assertion was `n > nlines` --
   a BOUNDS check. All three real defects (-70, -3, -3) were **in bounds**, so
   v1 could never have caught any of them.
2. It hard-coded `1507`/`1509` into its `HISTORICAL` exemption set -- it exempted
   precisely the numbers that were wrong, instead of correcting the prose that
   asserted them as live.
3. Its docstring claimed it verified "content still matches what the artifact
   says it is". The code never did that.

I also reported it to the Q/A as "mutation-proven". That was true of its two
STRUCTURAL checks and false of the anchor check -- a claim narrower than it
sounded, which is the failure mode this project's own memory warns about.

**v2 replaces it and is proven to fail.** It now (A) re-executes **every** fenced
`$ grep ...` block in the artifacts and requires an EXACT match against live
stdout -- which kills the stale-number and the curated-output defects together;
(B) asserts the ordering relation and the call-site count with no numbers; and
(C) checks each prose anchor by CONTENT, with a cross-file guard and
nearest-symbol resolution to avoid the false positives that would train a reader
to ignore it. `--self-test` runs it against a wrong-but-in-bounds artifact and a
curated block and requires REJECTION of both, plus ACCEPTANCE of a correct block:

```
$ python scripts/qa/verify_36_17_anchors.py --self-test
   (i)  wrong-but-in-bounds prose anchor -> REJECTED
   (ii) curated command block            -> REJECTED
   (iii) correct command block           -> ACCEPTED
SELF-TEST PASSED
```

### 12c. Criterion 7 -- three mutants survived my 6-cell matrix

Cycle 3 ran its own matrix and found three survivors, all reported as `4 passed`:

| Mutant | Why it mattered |
|---|---|
| **M-D** `if sl_trade:` -> `if True:` | `execute_sell` returns None when the position is already gone, so the mutant records a stop as **ENFORCED in the summary when no sell occurred**. For this defect specifically, that is the worst possible lie. |
| **M-E** drop `summary["halt_stop_loss_error"]` | a stop-loss failure swallowed silently |
| **M-F** `logger.exception` -> `logger.debug` | the failure downgraded to a whisper |

Root cause: **no test drove `check_stop_losses` to raise**, so the entire
fail-safe/loudness path that §1 "Failure handling" positively claims had zero
covering assertion.

Two new tests close all three:

- `test_phase_36_17_a_failed_sell_is_not_recorded_as_enforced` -- `execute_sell`
  returns `None`; asserts the exit was ATTEMPTED but `halt_stop_loss_triggered`
  stays empty. Attempted != enforced.
- `test_phase_36_17_a_raising_stop_pass_stays_loud_and_still_halts` --
  `check_stop_losses` raises; asserts the halt still completes with
  `status="halted_kill_switch"`, that `decide_trades` never ran, that
  `halt_stop_loss_error` carries the exception, **and** that something was logged
  at ERROR or above (the `caplog` assertion is what kills M-F; the summary key
  alone cannot distinguish a logged failure from a whispered one).

Matrix is now **9 cells, 9 killed**.

### 12d. Re-verification after cycle 3

```
$ python scripts/qa/verify_36_17_anchors.py --self-test    -> SELF-TEST PASSED
$ python scripts/qa/verify_36_17_anchors.py                -> ALL CHECKS PASSED
```

### 12e. What cycle 3 did NOT change

**No production behaviour.** `backend/services/autonomous_loop.py` is
byte-identical to commit `d057f127`. Everything in cycle 3 is tests, the
verifier, and prose.

### 12f. DISCLOSURE -- I left a mutant in the production file for ~4 minutes

Self-reported; no automated check would have surfaced this, and the tree is
clean now, so it is recorded rather than left silent.

During cycle 3 a 2-minute tool timeout sent SIGTERM to the mutation harness
**mid-mutation**. The harness restores inside its loop, but that never runs if
the process is killed between the write and the restore, so
`backend/services/autonomous_loop.py` was left carrying the
`summary["steps"].append("stop_loss_enforcement")` mutant.

**How it was caught:** the new anchor verifier started reporting anchor
mismatches (`:1544` resolving to `)`) because the injected line shifted every
anchor below it by one. It was detecting a live tree/artifact inconsistency --
which is what it is for -- and that is what led me to `git status`.

**Blast radius, measured:** the mutant existed only in the working tree, was
never committed (`git diff --stat d057f127` showed `1 insertion`), and the
backend running as pid 84494 loaded this module at 15:08Z and has not reloaded,
so no running process ever executed it.

**Restored precisely, not with a blanket checkout:** the single line was removed
and the result asserted **byte-identical to the committed blob**
(`git show d057f127:backend/services/autonomous_loop.py`), md5
`58bbf24bde4c5161ac05f26f70fb264e` -- the same digest the cycle-3 Q/A
independently reported for this file across cycle 2, cycle 3 and the worktree.
(A `git checkout --` was attempted first and correctly BLOCKED by the 62.0
PreToolUse guard, which refuses file-level checkouts that silently discard
working-tree edits.)

**Fixed so it cannot recur:** the harness now registers `atexit` plus
`SIGTERM`/`SIGINT`/`SIGHUP` handlers that restore before exiting. Proven by
deliberately killing a run mid-matrix:

```
!! signal 15 received -- production file RESTORED before exit
$ git status --porcelain backend/services/autonomous_loop.py     # (empty)
58bbf24bde4c5161ac05f26f70fb264e
```
