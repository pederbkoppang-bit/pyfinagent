---
step: 25.D
slug: normalize-agent-weights-0-1
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.D

## Step ID + masterplan reference

`25.D` -- "Normalize per-agent contribution weights to 0-1 range"
(P2, harness_required, no dep).

## Research-gate summary

Tier=simple. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`.

## Hypothesis

Drawer renders mixed-scale weights (Trader 0-10, RiskJudge 0-1) which
violates Few + Cleveland-McGill scale-consistency. By normalizing all
signals to 0-1 and showing a one-line total at the top, operators can
read contribution-weight at a glance.

## Success criteria (verbatim from masterplan.json)

> `all_weights_normalized_to_0_to_1_in_signal_attribution`
>
> `total_contribution_weight_summary_displayed_at_top_of_drawer`

## Plan steps

1. **`backend/services/signal_attribution.py`** -- in
   `extract_signals_from_analysis`:
   - Trader weight: `(float(score) / 10.0) if isinstance(score,(int,float)) else 0.0`.
   - Add a final `_clamp01(w)` helper to enforce [0,1] on all assigned weights.
2. **In `extract_quant_signals`**:
   - Quant weight: `_clamp01(float(composite) / 10.0)`.
   - SignalStack weight: `_clamp01(float(conviction_score) / 10.0)`.
3. **Update 3 existing tests** in `tests/services/test_signal_attribution.py`:
   - `test_trader_extracts_lite_full_report_reason`: `trader["weight"] == 0.7`.
   - `test_extract_quant_signals_full_candidate`: `quant["weight"] == 0.845`.
   - `test_signalstack_includes_all_overlays`: `stack["weight"] == 0.8`.
4. **`frontend/src/components/AgentRationaleDrawer.tsx`** -- prepend a
   summary block above the Layer cascade:
   ```tsx
   <TotalWeightSummary signals={data.signals} />
   ```
   that computes `sum(signals[i].weight)` and renders as a single line
   with class `text-sm font-mono text-slate-300`.
5. **Verifier** -- `tests/verify_phase_25_D.py` with 5 claims:
   - Claim 1: signal_attribution.py contains the `/10.0` normalization
     on Trader, Quant, SignalStack sites.
   - Claim 2: drawer contains a "Total contribution weight" element.
   - Claim 3: pytest_signal_attribution all green (the updated 22 tests).
   - Claim 4: behavioral round-trip -- call extract_signals_from_analysis +
     extract_quant_signals on sample data and assert ALL signal weights
     are in [0, 1].
   - Claim 5: full_report.analysis.score=7 Trader weight is 0.7 (not 7.0).

## Files

| File | Action |
|------|--------|
| `backend/services/signal_attribution.py` | Normalize 3 sites + clamp |
| `tests/services/test_signal_attribution.py` | Update 3 expected weights |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Add summary block |
| `tests/verify_phase_25_D.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_D.py
```

## Live-check

`Visual: drawer shows weight 0.0-1.0 with total summary`.
Will write `handoff/current/live_check_25.D.md`.

## Risks + mitigations

- **Risk**: Other callers in the codebase assume Trader weight = 0-10 scale.
  **Mitigation**: grep usage: only `paper_trader.py::save_signals_for_trade`
  reads `signal.weight` and just persists it; downstream consumers
  (drawer, attribution queries) treat it as opaque.
- **Risk**: Total weight could exceed 1.0 if all signals are saturated.
  **Mitigation**: Total is informational, not normalized -- a value of 4.5
  means "5 signals each near-saturated"; operators read it as count*avg.

## References

- `handoff/current/research_brief.md`
- `backend/services/signal_attribution.py:73, 113, 189, 225`
- `frontend/src/components/AgentRationaleDrawer.tsx:121-178`
- `.claude/masterplan.json::25.D`
