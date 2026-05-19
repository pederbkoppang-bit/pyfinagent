# Sprint Contract -- phase-30.3

**Step:** phase-30.3 -- P1: Connect stop-loss exits to learn loop.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.
**Cycle owner:** Main + Researcher (complex) + Q/A.

## Research-gate summary

Researcher complex tier delivered to `handoff/current/research_brief.md`.
Envelope: 10 sources read in full, 22 URLs, recency scan present.
gate_passed=true.

Canonical sources cited:
- Kaminski-Lo MIT (stop-loss adds value in momentum).
- Reflexion (Shinn et al., arXiv 2303.11366) -- failure must be
  observable for reflection to function.
- FinMem (Yu et al., arXiv 2311.13743) -- losses trigger risk-profile
  switching.
- Du Memory Survey (arXiv 2603.07670) -- asymmetric memory population
  is a documented failure mode.
- TrustTrade (Li et al., arXiv 2603.22567) -- uniform-memory schema
  for closed actions.
- PER (Schaul et al.) -- stop-outs have high TD-error.

Key design takeaway: stop-outs are the BEST signal for agent learning;
asymmetric capture (Step 7 sells but NOT Step 5.6 stop-outs) is the
documented anti-pattern.

**Line-number correction from researcher:** post phase-30.2 wiring,
`summary["stop_loss_triggered"].append(sl_ticker)` is at `:795`
(NOT `:771`). New `closed_tickers.append(sl_ticker)` goes at `:796`.

**Additional finding (out-of-scope for this step):**
`_learn_from_closed_trades` constructs `OutcomeTracker(settings)` with
no model -> `self._model is None` -> `save_agent_memory` never fires
in production. Model-injection is a separate concern, deferred. The
test exercises the wiring by patching the OutcomeTracker chain.

## Hypothesis

Hoisting `closed_tickers = []` to cycle-top + adding
`closed_tickers.append(sl_ticker)` to Step 5.6 ensures
`_learn_from_closed_trades` receives stop-out tickers. Closes
phase-30.0 Stage 12 (FAIL) + P1-3.

## Immutable success criteria (verbatim from masterplan phase-30.3)

```
verification.command = "grep -B 2 -A 4 'stop_loss_triggered.*append' backend/services/autonomous_loop.py | grep -q 'closed_tickers.append'"
success_criteria = [
  "stop_loss_triggered_tickers_appended_to_closed_tickers",
  "syntax_check_passes",
  "synthetic_test_with_one_stop_out_produces_an_agent_memories_row",
  "no_regression_in_existing_learn_step_test"
]
```

## Plan

1. **`backend/services/autonomous_loop.py`** -- 3 small edits:
   - Hoist: insert `closed_tickers: list[str] = []` at cycle-top
     (line 161 area, after `summary = {...}` and before any step
     code). Researcher Option A is the only timeout-safe choice.
   - Append in Step 5.6: insert `closed_tickers.append(sl_ticker)`
     at line 796 (sibling of the existing `summary["stop_loss_triggered"].append(sl_ticker)`).
   - Remove the redundant `closed_tickers = []` inside Step 7 (currently
     at line 862).

2. **`backend/tests/test_autonomous_loop_step_5_6.py`** (extend) --
   add tests that cover the new wiring without breaking the existing
   4 phase-30.2 tests:
   - Test A: simulated Step 5.6 with one triggered stop -> verify
     `closed_tickers` contains the stop-out ticker.
   - Test B: simulated full sequence of Step 5.6 then Step 7's learn
     invocation -> verify `_learn_from_closed_trades` is called with
     a list that includes the stop-out ticker.
   - Test C: synthetic-stop-out -> `bq.save_agent_memory` is invoked
     via the patched OutcomeTracker chain (satisfies the strict
     literal of the masterplan criterion).
   - Test D: grep-equivalent assertion against the source file (mirrors
     the masterplan verification command -- catch future refactor that
     removes the wiring).

## Hard guardrails

- Diff limited to: `backend/services/autonomous_loop.py` +
  extension to existing `backend/tests/test_autonomous_loop_step_5_6.py`.
- NO mutating BQ. NO Alpaca. NO frontend / .claude / .mcp.json.
- Total diff target: <150 lines.

## References

- `handoff/current/research_brief.md` (this cycle's brief).
- `handoff/archive/phase-30.0/experiment_results.md` Stage 12 + P1-3.
- `backend/services/autonomous_loop.py:160-161` (init site after hoist).
- `backend/services/autonomous_loop.py:794-799` (Step 5.6 append site).
- `backend/services/autonomous_loop.py:862` (Step 7 duplicate init to remove).
- `backend/services/autonomous_loop.py:1611-1637` (`_learn_from_closed_trades`).
- `backend/services/outcome_tracker.py:31-147` (`OutcomeTracker.__init__`
  + `evaluate_recommendation` -- documents the model-injection gap).
