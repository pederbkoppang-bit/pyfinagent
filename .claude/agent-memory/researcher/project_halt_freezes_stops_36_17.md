---
name: halt-freezes-stops-36-17
description: Step 36.17 — the kill-switch halt returns before stop-loss enforcement; execute_sell is deliberately UNGATED; the Step 5.6 test suite never drives run_daily_cycle; summary["steps"][-1] is a measured test collision
metadata:
  type: project
---

Step 36.17 (2026-08-09): should protective stop-loss exits keep running while the
kill switch has HALTED the cycle? Premise CONFIRMED as read — but four things were
not what the step text implied.

**Why:** the step text carried stale anchors (:1334/:1336; real values :1437/:1439)
and framed the fix as a simple reorder. The interesting content is in the
constraints that a reorder would violate.

**How to apply:** when a step proposes "move block X above block Y", first ask what
ELSE runs between them and whether the healthy path changes.

1. **The SELL path is deliberately ungated, and it is documented in-code.**
   `paper_trader.py:278-281` states the asymmetry verbatim: gating sells on `paused`
   "would deadlock the pause — the switch could never close the positions it just
   decided to close. Selling is the safe direction." So "no new BUYs" is enforced at
   the ORDER CHOKE POINT (`execute_buy:282`, fails closed), not merely by the cycle's
   early return. That means running exits during a halt cannot be abused into buying —
   **on the paused path**. On the `blocked` path the switch is NOT paused and baselines
   ARE present, so the BUY gate returns None and blocked-path BUY suppression comes
   ONLY from the cycle's `return summary`. Do not assume one gate covers both reasons.

2. **Synthesizing a stop is a NEW risk decision, not a protective default.**
   `backfill_missing_stops` computes `entry*(1-8%)` from a possibly-stale entry price.
   On an already-drawn-down book the synthesized stop can land ABOVE the mark, so
   backfill-then-check turns "no stop" into "sell at market now" — a flatten by side
   effect on the two branches (`blocked`/`paused`) that deliberately do NOT flatten
   (`paper_trader.py:1468` fires `flatten_all` only on `any_breached and not is_paused`).
   ESMA 2026-02 ¶11(5) classifies stop-loss triggers as algorithmic trading, which is
   the principled version: CHOOSING a level is a decision, ENFORCING a chosen level is
   not. Split them.

3. **The existing Step 5.6 suite is structurally blind to this defect.**
   `backend/tests/test_autonomous_loop_step_5_6.py:44-61` defines a LOCAL reproduction
   `_step_5_6_under_test` because it "avoids importing the 1700-line module under test",
   plus a 50-line line-order scan. Four green tests, none of which can observe that the
   halt returns before Step 5.6 is ever entered. Same class as
   [[feedback_guards_stop_one_seam_short]].

4. **MEASURED test collisions for a halt-branch fix.**
   - `summary["steps"][-1] == "kill_switch_halted"` is asserted at
     `test_phase_36_12_kill_switch_trading_path_block.py:298` (blocked) and `:374`
     (paused). Appending ANY step name inside the halt branch turns both RED.
   - The AST guard at `:460-521` constrains only what sits BETWEEN the
     `halt_reason = cycle_halt_reason(...)` assignment and the `if` — it does NOT
     constrain the `if` body, so inserting inside the branch is invisible to it.
   - `trader.execute_sell.called is False` at `:304/:377` does NOT collide: I measured
     `list(MagicMock()) == []`, so a `for t in trader.check_stop_losses()` loop iterates
     zero times. That is a latent VACUITY — a new test must set
     `check_stop_losses.return_value = ["WDC"]` explicitly.
   - Copying the backfill block verbatim hits
     `TypeError: '>' not supported between instances of 'MagicMock' and 'int'` on
     `count_backfilled > 0`, swallowed by production's try/except at `:1466`.

5. **Severity: `check_stop_losses` has exactly ONE production caller**
   (`autonomous_loop.py:1471`). No API route, no scheduler job, no MCP tool. The daily
   cycle is the SOLE enforcement path. `backfill_missing_stops` has one extra caller,
   `scripts/maintenance/backfill_stops.py:75`, a manual CLI that backfills but never
   enforces. So "accept and document" is not the zero-work option — it requires
   BUILDING an operator stop-check tool that does not exist.

Related: [[project_kill_switch_36_12_traps]], [[project_kill_switch_deadlock_85_6]],
[[project_phase86_killswitch_channels]].
