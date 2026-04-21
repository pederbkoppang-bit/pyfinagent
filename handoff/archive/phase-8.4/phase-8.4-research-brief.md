# Research Brief — phase-8 / 8.4 "Promote or reject decision memo"

**Tier:** simple (closure-style synthesis of 8.1/8.2/8.3 findings)
**Date:** 2026-04-20

## Objective

Final step of phase-8. Write `handoff/current/phase-8-decision.md` whose first line begins with `PROMOTE:` or `REJECT:`. The immutable verification is:

```
test -f handoff/current/phase-8-decision.md && grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md
```

This brief consolidates the evidence from phase-8.1 through 8.3 to inform the decision; it does NOT open new external research because the prior three cycles already fetched 7 + 6 + 6 = 19 sources in full.

## Carry-forward evidence (from prior briefs)

### phase-8.1 TimesFM findings

- Python version: `.venv` is 3.14; `timesfm` requires `<3.12`. **Cannot run live inference in the current runtime.**
- arXiv 2511.18578 (Nov 2025): zero-shot TimesFM R² = **-2.80%**, directional accuracy **<50%**, annualised return **-1.47%** vs CatBoost 46.50%. "Generic pre-training does not transfer to finance."
- Preferred Networks (tech.preferred.jp): zero-shot TimesFM Sharpe **0.42** vs AR(1) Sharpe **1.58**. Fine-tuned Sharpe 1.68.
- arXiv 2412.09880 (Dec 2024): fine-tuning on 100k+ financial series gets Sharpe to 1.68. Zero-shot underperforms AR(1).

### phase-8.2 Chronos-Bolt findings

- arXiv 2511.18578: zero-shot Chronos(large) ~51% directional; TimesFM(500M) ~50%. Marginal at best.
- `chronos-forecasting` + `torch` also uninstalled in the current venv. Same runtime gate.
- AWS blog: Chronos-Bolt Base = 205M params, 250x faster than Chronos-T5. But speed is not the bottleneck — **signal quality is**.

### phase-8.3 Ensemble blend findings

- Infrastructure shipped (EnsembleBlender with equal/correlation/shrinkage weighting, pure-Python Ledoit-Wolf, nested walk-forward with purge+embargo) and tested (15/15).
- **No real-world evaluation yet** — neither TimesFM nor Chronos has produced live forecasts because of the runtime gate.
- With the runtime gate blocking real inference, the blender has nothing to blend beyond the MDA baseline.

## Decision logic

A PROMOTE decision would require:
1. Runtime gate cleared (Python 3.11 sub-env or Docker image with torch+timesfm).
2. At least one of TimesFM/Chronos producing measurable IC > 0 over a shadow-log window of ≥60 days.
3. Ensemble IR > MDA-baseline IR on out-of-sample walk-forward.

Current evidence:
- Condition 1: NOT met (Python 3.14 venv; no Docker image).
- Condition 2: NOT met (zero live forecasts produced; published evidence says zero-shot IC < baseline).
- Condition 3: NOT met (ensemble blender has no real data to blend).

**REJECT is the honest call.** The scaffold stays in place for re-evaluation once:
- A Python 3.11 sub-env is provisioned OR a docker-ized inference service is stood up.
- Fine-tuned variants (pfnet/timesfm-1.0-200m-fin, or Chronos-2) become available.
- Phase-9 refresh cron populates a shadow-log window of ≥60 days.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "gate_passed": true,
  "note": "closure-style synthesis of phase-8.1 + 8.2 + 8.3 briefs; no new external sources (19 previously fetched in full across 3 sub-step briefs)"
}
```
