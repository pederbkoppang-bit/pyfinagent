# live_check -- phase-36.17

**Required evidence (immutable, verbatim from `.claude/masterplan.json`):**
"Verbatim test output for the reproduce-then-fix pair on BOTH the paused and the
blocked halt paths, plus the re-derived grep showing the final ordering of Step
5.5's return and Step 5.6."

Captured 2026-08-09 on the live tree. Backend pid 84494.

---

## 1. Reproduce-then-fix pair, BOTH halt paths

Both tests drive the **real** `run_daily_cycle` end-to-end (mock surface copied
from `test_phase_36_12...py:248-289`) with a position priced `41.0` against a
`46.0` stop, and `trader.check_stop_losses.return_value = ["WDC"]` set
explicitly so neither test can be vacuous.

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

### 1c. Final state

```
$ python -m pytest backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q -p no:randomly
4 passed, 1 warning in 10.29s
```

### 1d. Immutable verification command

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q \
    -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2890 deselected, 1 warning in 20.24s
```

The module the research brief flagged as a collision risk, run alone:

```
$ python -m pytest backend/tests/test_phase_36_12_kill_switch_trading_path_block.py -q -p no:randomly
25 passed, 1 warning in 8.95s
```

## 2. Re-derived ordering of Step 5.5's return and Step 5.6

```
$ grep -n "return summary" backend/services/autonomous_loop.py | head -1
1507:                return summary

$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1509:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

**The ordering is deliberately UNCHANGED**: the halt still returns at `:1507`,
before Step 5.6 at `:1509`. Option (b) adds enforcement *inside* the halt branch
rather than moving Step 5.6 above the halt (option (a), rejected). Keeping the
`return summary` last is load-bearing: on the `blocked` path the switch is not
paused, so `execute_buy`'s own gate returns None and that return is the only
thing suppressing BUYs.

Superseded line numbers, for anyone reading an older artifact: the masterplan
step text says `:1334/:1336`; the pre-fix tree was `:1437/:1439`.

## 3. Mutation matrix (criterion 7)

```
[baseline] un-mutated tree: 4 passed, 1 warning in 10.10s
  KILLED  | remove the breach-path exclusion            -> 1 failed, 3 deselected
  KILLED  | ORDERING REVERTED: disable the pass         -> 2 failed, 2 deselected
  KILLED  | reintroduce backfill_missing_stops          -> 2 failed, 2 deselected
  KILLED  | append to summary["steps"]                  -> 1 failed, 3 deselected
  KILLED  | drop the recorded ticker                    -> 2 failed, 2 deselected
[restored] un-mutated tree: 4 passed, 1 warning in 10.57s
ALL 5 MUTANTS KILLED.
```

## 4. Live-state isolation held

md5 of the three git-tracked handoff files, before and after the reproduce run:

```
685bf1a5fd7beaa4f15da2babf133ca2  handoff/kill_switch_audit.jsonl   (identical after)
6bc251737c8145e0b3891ed1cc5d4b2c  handoff/cycle_history.jsonl       (identical after)
8319fc52d0f8a8cbb9959828e498d308  handoff/.cycle_heartbeat.json     (identical after)
```

## 5. Scope of this evidence -- stated honestly

This is in-process evidence against the real `run_daily_cycle` with a mocked
`PaperTrader`. **No live halted cycle with a real breached position was
exercised**, because halting the live book is an operator action and was not
taken. The step's live_check asks for verbatim test output, which is what is
recorded here -- it is not a claim that the path has run in production.
