---
step: phase-25.D
cycle: 93
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.D

## What was built/changed

Closed audit bucket 24.4 F-5 by normalizing per-agent contribution
weights to 0-1 and adding a total-contribution summary at the top of
the drawer:

1. **`backend/services/signal_attribution.py`**:
   - Added `_clamp01(w)` helper that clamps any numeric input to [0.0, 1.0]
     (defends against out-of-range scores from upstream).
   - Trader weight: `_clamp01(float(score) / 10.0)` (was `float(score)`).
   - RiskJudge weight: `_clamp01(pos_pct)` (was `float(pos_pct)`; already 0-1, now safe).
   - Quant weight: `_clamp01(float(composite) / 10.0)` (was `float(composite)`).
   - SignalStack weight: `_clamp01(float(conviction_score) / 10.0)` (was raw).
2. **`tests/services/test_signal_attribution.py`** -- updated 3 expected weights:
   - Trader: 7.0 -> 0.7 (`final_score=7`).
   - Quant: 8.45 -> 0.845 (`composite_score=8.45`).
   - SignalStack: 8.0 -> 0.8 (`conviction_score=8.0`).
3. **`frontend/src/components/AgentRationaleDrawer.tsx`** -- new
   `<TotalWeightSummary signals={data.signals} />` component rendered
   at the top of the per-layer cascade. Renders:
   - Label "Total contribution weight".
   - Sum of all weights (mono font).
   - Count + average per signal as a context line.
   - Sky-tinted card (matches the "important" emphasis used for Risk).

## Files changed

| File | Action |
|------|--------|
| `backend/services/signal_attribution.py` | Add _clamp01 + normalize 4 weight sites |
| `tests/services/test_signal_attribution.py` | Update 3 expected weight values |
| `frontend/src/components/AgentRationaleDrawer.tsx` | NEW TotalWeightSummary component |
| `tests/verify_phase_25_D.py` | NEW verifier (5 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D.py

=== phase-25.D verification ===

[PASS] 1. all_weights_normalized_to_0_to_1_in_signal_attribution
        -> clamp01=True trader=True quant=True stack=True
[PASS] 2. total_contribution_weight_summary_displayed_at_top_of_drawer
        -> component=True label=True wire=True
[PASS] 3. pytest_signal_attribution_suite_all_pass
        -> exit=0 passed=22 (expect 22+)
[PASS] 4. behavioral_all_weights_in_zero_one_range
        -> out_of_range=[] all_weights=[('Analyst', 1.0), ('Bull', 0.6), ('Bear', 0.5), ('Trader', 0.9), ('RiskJudge', 0.02), ('Quant', 1.0), ('SignalStack', 0.8)]
[PASS] 5. trader_weight_is_normalized_division_by_ten
        -> trader_weight=0.9 expected=0.9
```

Notable: claim 4 used `composite_score=12.5` intentionally to exercise
the clamp -- result is 1.0 (correctly bounded).

Frontend `npx tsc --noEmit`: clean (excluding pre-existing 25.A12 noise).
Backend AST: clean. All 22 unit tests pass.

## Success criteria -> evidence

1. `all_weights_normalized_to_0_to_1_in_signal_attribution` -- Claim 1 + 4 + 5 PASS:
   structural normalization at 3 sites + clamp01 helper, behavioral test confirms
   ALL 7 returned signals (Analyst, Bull, Bear, Trader, RiskJudge, Quant,
   SignalStack) have weight in [0, 1] -- even when composite_score=12.5 was
   clamped to 1.0.
2. `total_contribution_weight_summary_displayed_at_top_of_drawer` -- Claim 2 PASS:
   `TotalWeightSummary` component defined + "Total contribution weight" label +
   wired ABOVE the Layer cascade at the top of the layered section.

## Out-of-scope / deferred

- Normalizing the Bull/Bear `bull_weight` / `bear_weight` defaults to also use
  /10 if upstream debaters return 0-10 scores: currently no upstream code emits
  >1.0 for these, but a future change could trip them. The _clamp01 helper is
  available for that retrofit when needed.
- Highlighting the highest-weight agent visually in the drawer: out of scope;
  the per-row `weight 0.XX` rendering already gives operators the comparison.

## References

- `handoff/current/research_brief.md`
- `backend/services/signal_attribution.py:55-72` (helper), `:115-119` (Trader), `:140-142` (RiskJudge), `:194-200` (Quant), `:230-236` (SignalStack)
- `frontend/src/components/AgentRationaleDrawer.tsx:181-198` (component)
- `.claude/masterplan.json::25.D`
