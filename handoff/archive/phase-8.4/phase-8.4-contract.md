# Sprint Contract — phase-8 / 8.4 (Promote or reject decision memo)

**Step id:** 8.4 -- final phase-8 step. **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple (closure-style synthesis)

## Research-gate summary

Closure brief at `handoff/current/phase-8.4-research-brief.md` consolidating 19 prior in-full sources from 8.1/8.2/8.3. `gate_passed: true` on closure semantics (same pattern as qa_78_v1 and qa_phase5_crypto_removal_v1).

## Hypothesis

Write `handoff/current/phase-8-decision.md` with **`REJECT:`** as the first-line verdict, because:

1. Python 3.14 venv cannot run `timesfm` (requires `<3.12`) or `torch` + `chronos-forecasting` — **zero live forecasts produced**.
2. Published evidence (arXiv 2511.18578 Nov 2025, Preferred Networks) shows zero-shot TimesFM/Chronos underperform AR(1) on equity daily returns.
3. Ensemble blender has no real signals to blend beyond MDA baseline.

Scaffold stays in place for re-evaluation when the runtime gate is cleared (Python 3.11 sub-env or docker-ized service) AND fine-tuned variants AND a ≥60-day shadow-log window exist.

## Immutable criterion

- `test -f handoff/current/phase-8-decision.md && grep -qE '^(PROMOTE|REJECT):' handoff/current/phase-8-decision.md`

## Plan

1. Write `handoff/current/phase-8-decision.md` with the required first-line prefix.
2. Verify with the exact immutable command.
3. Run full regression.
4. Q/A, log, flip. **Phase-8 closes 4/4.**

## Out of scope

- No code. Doc-only.
- No new external research (prior 19 sources suffice).
- No retirement of the phase-8.1/8.2/8.3 modules — they stay as scaffolds for re-evaluation.
- ASCII-only.

## References

- `handoff/current/phase-8.4-research-brief.md`
- Prior in-full sources across 8.1/8.2/8.3 briefs.
- `.claude/masterplan.json` -> phase-8 / 8.4
