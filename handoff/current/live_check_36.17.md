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
1510:                return summary

$ grep -n "Step 5.6: Stop-loss enforcement" backend/services/autonomous_loop.py
1512:            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
```

**The ordering is deliberately UNCHANGED**: the halt's `return summary` is still
the line IMMEDIATELY BEFORE the Step 5.6 header -- see the live output above, and
note the claim is stated as a RELATION because three Q/A cycles caught this
artifact shipping stale numbers here. Option (b) adds enforcement *inside* the halt branch
rather than moving Step 5.6 above the halt (option (a), rejected). Keeping the
`return summary` last is load-bearing: on the `blocked` path the switch is not
paused, so `execute_buy`'s own gate returns None and that return is the only
thing suppressing BUYs.

Superseded line numbers, for anyone reading an older artifact: the masterplan
step text says `:1334/:1336`; the pre-fix tree was `:1437/:1439`.

## 3. Mutation matrix (criterion 7)

```
[baseline] un-mutated tree: 4 passed, 1 warning in 10.50s
  KILLED  | M1: delete the halt's `return summary`      -> 2 failed, 2 deselected
  KILLED  | remove the breach-path exclusion            -> 1 failed, 3 deselected
  KILLED  | ORDERING REVERTED: disable the pass         -> 2 failed, 2 deselected
  KILLED  | reintroduce backfill_missing_stops          -> 2 failed, 2 deselected
  KILLED  | append to summary["steps"]                  -> 1 failed, 3 deselected
  KILLED  | drop the recorded ticker                    -> 2 failed, 2 deselected
[restored] un-mutated tree: 6 passed, 1 warning in 14.03s
ALL 9 MUTANTS KILLED.
```

Cycle 3 added M-D / M-E / M-F after the Q/A found them SURVIVING the 6-cell
matrix. M-D is the one that mattered: `if sl_trade:` -> `if True:` made the
summary record a stop as ENFORCED when `execute_sell` returned None, i.e. an
exit that never happened.

M1 was added in cycle 2 after the Q/A proved the no-BUY assertion was vacuous.
It kills on `assert trader.execute_buy.called is False`, verbatim:

```
>       assert trader.execute_buy.called is False, "a BUY was placed on a halted cycle"
E       AssertionError: a BUY was placed on a halted cycle
E       assert True is False
E        +  where True = <MagicMock name='mock.execute_buy' id='4733246768'>.called
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

---

## 6. Cycle-2 corrections (Q/A `wf_6bc4c0a4-d9c` returned CONDITIONAL)

### 6a. Line anchors, RE-DERIVED (Q/A finding 1)

The cycle-1 artifacts cited three `final_state` anchors that do not reproduce --
each exactly 70 lines low, broken by this step's own +70-line insertion. Actual
output:

```
$ grep -n "final_state" backend/services/autonomous_loop.py
1429:                final_state = await asyncio.to_thread(trader.mark_to_market)
1770:            final_state = await asyncio.to_thread(trader.mark_to_market)
1849:                "nav": final_state["nav"],
1850:                "pnl_pct": final_state["pnl_pct"],
1865:            logger.info(f"Paper trading cycle complete: NAV=${final_state['nav']:.2f}, "
1866:                         f"P&L={final_state['pnl_pct']:.2f}%, trades={trades_executed}, "
```

And the caller count, which this change itself made stale:

```
$ grep -rn "trader.check_stop_losses" backend --include="*.py" | grep -v /tests/
backend/services/autonomous_loop.py:1474:                        halt_stops = await asyncio.to_thread(trader.check_stop_losses)
backend/services/autonomous_loop.py:1544:            triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
```

TWO call sites, not one. Corrected in `autonomous_loop.py:1438`, the test-module
docstring, and `experiment_results_36.17.md` §1.

### 6b. Criterion 4's no-BUY assertion is now FALSIFIABLE (Q/A finding 2)

The Q/A proved `execute_buy.called is False` could not fail (`decide_trades`
stubbed to `[]` made `execute_buy` structurally unreachable). A first fix
attempt with a **dict**-shaped order also failed -- measured, not assumed:
`autonomous_loop.py:1711` reads `order.action` as an attribute, so a dict is
skipped. The working fix uses a real `TradeOrder` plus a patched
`paper_trader._get_live_price` (`:1713` does a live yfinance fetch). See the M1
verbatim failure in §3.

### 6c. Re-verification after the cycle-2 changes

```
$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2890 deselected, 1 warning in 16.03s

$ python -m pytest backend/tests/test_phase_36_12_kill_switch_trading_path_block.py -q
25 passed, 1 warning in 8.67s

$ python -m pytest backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q
4 passed, 1 warning in 9.74s
```

### 6d. Heartbeat digest changed -- and why that is NOT a test leak

`.cycle_heartbeat.json` now digests `eea37b489ebbf797240dd9a22c23151d` vs §4's
`8319fc52d0f8a8cbb9959828e498d308`. **The live backend writes that file** (pid
84494, restarted 15:08Z). Measured before/after WITHIN the cycle-2 run, all
three files are byte-identical, so the suite still wrote nothing.

---

## 7. Cycle-3 evidence (Q/A `wf_4bf499e6-0e4` returned FAIL)

### 7a. Anchor + command-block verification, mechanical

```
$ python scripts/qa/verify_36_17_anchors.py --self-test
   (i)  wrong-but-in-bounds prose anchor -> REJECTED
   (ii) curated command block            -> REJECTED
   (iii) correct command block           -> ACCEPTED
SELF-TEST PASSED

$ python scripts/qa/verify_36_17_anchors.py
  ok   A. command-block fidelity: 8 block(s) re-executed
  ok   B. halt `return summary` :1510 immediately precedes Step 5.6 :1512
  ok   B. check_stop_losses has exactly 2 production call sites
  ok   C. loose prose anchors: 3 checked by CONTENT
ALL CHECKS PASSED.
```

The v1 verifier was an ILLUSORY GUARD -- bounds-only, and it exempted the very
numbers that were wrong. v2 re-executes every quoted command block and checks
prose anchors by content. The self-test is what makes that claim auditable.

### 7b. Final test state

```
$ python -m pytest backend/tests/test_phase_36_17_halt_stop_loss_enforcement.py -q
6 passed, 1 warning in 13.41s

$ python -m pytest backend/tests/ -q -k 'kill_switch or paper_trader or autonomous_loop'
224 passed, 1 skipped, 2892 deselected, 1 warning in 16.11s
```

### 7c. Production file integrity

```
$ git status --porcelain backend/services/autonomous_loop.py     # (empty)
$ md5 -q backend/services/autonomous_loop.py
58bbf24bde4c5161ac05f26f70fb264e
```

Byte-identical to commit `d057f127`, and the same digest the cycle-3 Q/A
independently reported across cycle 2, cycle 3 and the worktree. See
`experiment_results_36.17.md` §12f for a disclosed incident in which a timeout
briefly left a mutant in this file, how it was caught, and the signal-handler
hardening that prevents a recurrence.
