# Live-check placeholder -- phase-25.D

**Step:** 25.D -- Normalize per-agent contribution weights to 0-1 range
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Visual: drawer shows weight 0.0-1.0 with total summary"

## Pre-deployment evidence
- 5/5 verifier PASS.
- 22/22 pytest tests pass.
- Frontend tsc clean (excluding pre-existing 25.A12 noise).
- Behavioral test confirms even saturated upstream values (composite_score=12.5)
  produce in-range outputs (1.0 clamped).

## Post-deployment operator workflow
1. Pull main + rebuild frontend:
   ```
   git pull origin main
   source .venv/bin/activate
   cd frontend && npm run dev &
   ```
2. Open a recent paper-trade rationale drawer in the UI; expect:
   - "Total contribution weight" card at the top, sky-tinted.
   - Sum like "4.50 across 7 signals (avg 0.64)".
   - Per-row weights formatted "weight 0.XX" -- all in [0.00, 1.00].
   - No row with weight > 1.00 anywhere.

## Closes audit basis
bucket 24.4 F-5 RESOLVED. The drawer now renders consistent 0-1 weights and a
total-contribution summary so operators can compare agent influence at a glance.

**Audit anchor for next bucket:** 25.C (Layer-1 28-skill output surfacing in drawer),
25.E (P2 backlog).
