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
halt path -- verified with `grep -n final_state` (reads at `:1776`/`:1792`
belong to the healthy path and use the `:1697` assignment) -- so inserting there
introduces no staleness.

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

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | +70 lines: the exit-only pass inside the Step 5.5 halt branch. No other hunk. |
| `backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py` | NEW -- 4 tests + 3 isolation fixtures. |
| `handoff/current/contract_36.17.md` | NEW -- contract, incl. the recorded operator decision. |
| `handoff/current/research_brief_36.17.md` | NEW -- research gate artifact. |

`git diff --stat backend/services/autonomous_loop.py` -> `1 file changed, 70 insertions(+)`.

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
1507:                return summary
$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1509:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

The halt's `return summary` is at **:1507** and Step 5.6 begins at **:1509** --
the ordering is unchanged; the fix adds enforcement *inside* the halt branch
rather than moving Step 5.6. The masterplan step text's `:1334/:1336` and the
pre-fix `:1437/:1439` are both superseded.

## 7. Criterion 7 -- MUTATION TEST, both directions

Harness asserts each target substring matches **exactly once** before mutating
(a no-match `str.replace` looks identical to success), requires the un-mutated
baseline to be GREEN first (else "killed" proves nothing), and restores + digest-
verifies the file after every mutant.

```
[baseline] un-mutated tree: 4 passed, 1 warning in 10.10s

  KILLED  | remove the breach-path exclusion (pass runs on EVERY halt reason)
           result: 1 failed, 3 deselected, 1 warning in 5.29s
  KILLED  | ORDERING REVERTED: disable the exit-only pass entirely (pre-fix behaviour)
           result: 2 failed, 2 deselected, 1 warning in 6.99s
  KILLED  | reintroduce backfill_missing_stops into the halt pass
           result: 2 failed, 2 deselected, 1 warning in 7.22s
  KILLED  | append to summary["steps"] inside the halt branch
           result: 1 failed, 3 deselected, 1 warning in 5.51s
  KILLED  | drop the recorded ticker (silent enforcement, no audit surface)
           result: 2 failed, 2 deselected, 1 warning in 7.28s

[restored] un-mutated tree: 4 passed, 1 warning in 10.57s
ALL 5 MUTANTS KILLED. Every new guard can fail.
```

The second mutant is the "**both directions**" half of criterion 7: reverting
the enforcement to pre-fix behaviour turns the reproduce pair RED.

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
