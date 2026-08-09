# live_check -- phase-36.17

**Required evidence (immutable, verbatim from `.claude/masterplan.json`):**
"Verbatim test output for the reproduce-then-fix pair on BOTH the paused and the
blocked halt paths, plus the re-derived grep showing the final ordering of Step
5.5's return and Step 5.6."

This file carries exactly that. The full build record, the mutation matrix and
the disclosures are in `handoff/current/experiment_results_36.17.md`; the
verbatim Q/A verdicts are in `handoff/current/evaluator_critique_36.17.md`.

---

## 1. Reproduce-then-fix pair, BOTH halt paths

Both tests drive the **real** `run_daily_cycle` end-to-end with a position priced
`41.0` against a `46.0` stop, and `trader.check_stop_losses.return_value =
["WDC"]` set explicitly so neither test can be vacuous.

### 1a. PRE-FIX -- the defect, reproduced

```
$ source .venv/bin/activate && python -m pytest \
    backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q -p no:randomly

>       assert trader.check_stop_losses.called is True, (
            "a blocked cycle did not check stop-losses (phase-36.17)"
        )
E       AssertionError: a blocked cycle did not check stop-losses (phase-36.17)
E       assert False is True
E        +  where False = <MagicMock name='mock.check_stop_losses' id='4710405792'>.called
E        +    where <MagicMock name='mock.check_stop_losses' id='4710405792'> = <MagicMock id='4709851088'>.check_stop_losses

backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py:302: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  backend.services.autonomous_loop:autonomous_loop.py:1404 Paper trading: kill-switch active (kill_switch_disarmed_lost_history) -- skipping decide/execute

FAILED ...::test_phase_36_17_paused_cycle_enforces_preexisting_stops
FAILED ...::test_phase_36_17_blocked_cycle_enforces_preexisting_stops
2 failed, 4 passed, 3 warnings in 16.24s
```

The **4 passed** include the two temporary `test_REPRODUCE_*` tests, which
asserted the defect directly -- `check_stop_losses.called is False` and
`execute_sell.called is False` -- and PASSED, on **both** the `paused`
(`KS_QUIET` + `state._paused = True`) and `blocked` (`KS_BLOCKED`) paths.

### 1b. POST-FIX -- the same file, exactly inverted

```
FAILED ...::test_REPRODUCE_paused_cycle_never_checks_stops
FAILED ...::test_REPRODUCE_blocked_cycle_never_checks_stops
2 failed, 4 passed, 3 warnings in 14.57s
```

The defect-asserting tests now FAIL; the enforcement tests now PASS. The
temporary REPRODUCE tests were then deleted.

### 1c. Final state of the module, on the tree as committed at `d68f69e5`

```
$ python -m pytest backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q -p no:randomly
8 passed, 1 warning in 15.35s
```

### 1d. Immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q \
    -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2894 deselected, 1 warning in 15.93s
```

exit code **0**. The `deselected` count moves whenever this step adds a test;
the load-bearing numbers are `224 passed, 1 skipped, exit 0`.

The module the research brief flagged as a collision risk, run alone:
`25 passed, 1 warning in 8.67s`.

## 2. Re-derived ordering of Step 5.5's return and Step 5.6

```
$ grep -n "return summary" backend/services/autonomous_loop.py | head -1
1510:                return summary

$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1512:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

**The ordering is deliberately UNCHANGED:** the halt's `return summary` is the
line IMMEDIATELY BEFORE the Step 5.6 header. The claim is stated as a RELATION
because three Q/A cycles caught this artifact shipping stale numbers here.
Option (b) adds enforcement *inside* the halt branch rather than moving Step 5.6
above the halt (option (a), rejected). Keeping `return summary` last is
load-bearing: on the `blocked` path the switch is not paused, so `execute_buy`'s
own gate returns None and that return is the only thing suppressing BUYs.

Superseded anchors, for anyone reading an older artifact -- both **STALE**: the
masterplan step text says `:1334/:1336`; the pre-fix tree was `:1437/:1439`.

## 3. Criterion 7 -- mutation matrix

Re-runnable as of cycle 6 and **it never writes to the repo**:
`python scripts/qa/mutation_matrix_36_17.py`. One real run's stdout, with the
per-cell detail, is in `experiment_results_36.17.md` §8. Result:
**baseline 8 passed -> ALL 11 MUTANTS KILLED -> restored 8 passed**, with the
target md5 asserted unchanged (`58bbf24bde4c5161ac05f26f70fb264e`) across the
whole run.

Deliberately not duplicated here: the previous revision of this file carried its
own copy of the matrix, the two copies drifted apart, and one of them was a
9-cell run recorded under an 11-cell claim. One transcript, one home.

## 4. Live-state isolation, and the leak this cycle found and closed

The earlier isolation claim covered **three git-tracked** handoff files. That was
true but narrower than it sounded: it excluded the **untracked**
`handoff/.autonomous_loop.lock`, which this module was writing, because these
tests drive the real `run_daily_cycle` and it takes the real cycle lock.
Redirected to `tmp_path` with an autouse guard that is proven able to fail. Full
measurement, attribution and the verbatim guard failure are in
`experiment_results_36.17.md` §10.

After the fix, across a full run of the module, the whole live-state set is
byte-identical:

```
handoff/kill_switch_audit.jsonl   685bf1a5fd7beaa4f15da2babf133ca2   identical after
handoff/cycle_history.jsonl       6bc251737c8145e0b3891ed1cc5d4b2c   identical after
handoff/.cycle_heartbeat.json     d4a8ba2de8f35348e4df8f775b6a254d   identical after
handoff/.autonomous_loop.lock     ee1ba590743c0cfe00cc72848d5a3260   identical after
handoff/away_ops/health.jsonl     1e27da828e5f68581c0f94da49ba671e   identical after
```

These digests are point-in-time and are **not** re-executable evidence: the live
backend writes two of these files on its own schedule.

## 5. Scope of this evidence -- stated honestly

In-process evidence against the real `run_daily_cycle` with a mocked
`PaperTrader`. **No live halted cycle with a real breached position was
exercised**, because halting the live book is an operator action and was not
taken. The step's live_check asks for verbatim test output, which is what is
recorded here -- it is not a claim that the path has run in production.
