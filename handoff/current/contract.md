# Sprint Contract -- phase-30.2

**Step:** phase-30.2 -- P1: Wire `backfill_missing_stops` into autonomous_loop Step 5.6.
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. Autonomous loop PAUSED.
**Cycle owner:** Main + Researcher (complex) + Q/A.

## Research-gate summary

Researcher complex tier delivered to `handoff/current/research_brief.md`.
Envelope: 6 sources read in full, 16 URLs, recency scan complete,
three-variant search composition visible. `gate_passed: true`.

Canonical sources cited:
- arXiv 2604.27150 (Apr 2026) -- swarm SOTA.
- Kaminski-Lo MIT -- stop-loss adds value in momentum.
- O'Neil CAN SLIM -- 7-8% canonical.
- Quant-Investing 150-yr study -- 10% momentum stop +71.3% return.
- Moments Log -- idempotent-backfill canonical.
- LuxAlgo -- 2%-per-position risk rule.

Key design takeaway: backfill BEFORE check is correct ordering;
immediate-sell of positions already below their synthesized stop IS
the desired behavior.

## Immutable success criteria (verbatim from masterplan phase-30.2)

```
verification.command = "grep -A 5 'Step 5.6' backend/services/autonomous_loop.py | grep -q 'backfill_missing_stops' && python -c \"import ast; ast.parse(open('backend/services/autonomous_loop.py').read())\""
success_criteria = [
  "autonomous_loop_step_5_6_calls_backfill_missing_stops_before_check_stop_losses",
  "syntax_check_passes",
  "after_one_cycle_paper_positions_stop_loss_price_is_null_count_drops_to_zero",
  "no_regression_in_existing_stop_loss_enforcement_test"
]
```

`after_one_cycle_..._null_count_drops_to_zero` is a LIVE post-cycle
BQ check. The autonomous loop is PAUSED overnight, so this criterion
is verified by the operator in the morning via
`SELECT COUNT(*) FROM financial_reports.paper_positions WHERE stop_loss_price IS NULL`
-- expected to drop from 7 to 0 after the first unpause cycle.

## Plan

1. **`backend/services/autonomous_loop.py`** -- modify Step 5.6:
   - Insert `backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)`
     BEFORE the existing `check_stop_losses` call.
   - Record `summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])`.
   - Log INFO when count_backfilled > 0.

2. **`backend/tests/test_autonomous_loop_step_5_6.py`** (new) --
   focused unit test:
   - Mock PaperTrader with `backfill_missing_stops`, `check_stop_losses`,
     `execute_sell`.
   - Assert call-order via Mock parent's `method_calls`.
   - 3 cases: legacy-null-stops, idempotent re-run, fail-open on
     backfill exception.

## Hard guardrails

- Diff limited to: `backend/services/autonomous_loop.py` +
  new `backend/tests/test_autonomous_loop_step_5_6.py`.
- NO mutating BQ. NO Alpaca. NO frontend / .claude / .mcp.json.
- Total diff target: <150 lines.

## References

- `handoff/current/research_brief.md` (this cycle's brief).
- `handoff/archive/phase-30.0/experiment_results.md` Stage 7 + P1-2.
- `backend/services/paper_trader.py:465-532` -- the function being wired.
- `backend/services/autonomous_loop.py:751-777` -- Step 5.6 current code.
