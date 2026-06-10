# Live-check placeholder -- phase-25.C

**Step:** 25.C -- Surface Layer-1 28-skill outputs in drawer when full pipeline runs
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Drawer tree on next full-pipeline trade has layer1_skills sub-tree with >=3 entries"

## Pre-deployment evidence
- 7/7 verifier PASS.
- 22/22 pytest tests pass.
- Frontend tsc clean.
- Claim 3 confirms a full-shape analysis dict with insider+options+sector
  produces 3 layer-1 skill rows in the extractor.
- Claim 7 confirms `group_signals_for_drawer` routes them into the
  `layer1_skills` bucket end-to-end.

## Post-deployment operator workflow
1. Pull main + restart backend + rebuild frontend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   cd frontend && npm run dev &
   ```
2. Run a FULL-pipeline analysis (lite_mode=False):
   ```
   curl -X POST http://localhost:8000/api/analysis/ -H 'Content-Type: application/json' \
     -d '{"ticker": "AAPL", "lite_mode": false}'
   ```
3. Open the resulting paper-trade rationale drawer; expect a new
   "Layer-1 Skills" collapsible section between the Total contribution
   weight summary and Analyst layer, containing 3-11 rows depending on
   which skills returned non-N/A signals.

## Closes audit basis
bucket 24.4 F-4 RESOLVED. Operators can now see what each of the 11
enrichment agents reported, gated correctly to full-pipeline trades only.

**Audit anchor for next bucket:** 25.E (drawer summary/full toggle, depends on 25.C done),
25.F3, 25.B6, 25.B10 (P2 backlog).
