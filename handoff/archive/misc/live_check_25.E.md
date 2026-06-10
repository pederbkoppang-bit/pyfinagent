# Live-check placeholder -- phase-25.E

**Step:** 25.E -- Drawer summary vs full toggle (?full=1 query param)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "?full=1 returns >5 signals on a full-pipeline trade"

## Pre-deployment evidence
- 5/5 verifier PASS.
- AST clean on paper_trading.py.
- Frontend tsc clean.

## Post-deployment operator workflow
1. Pull main + restart backend + rebuild frontend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   cd frontend && npm run dev &
   ```
2. Confirm API toggle via curl on a known full-pipeline trade_id:
   ```
   TRADE_ID="<some-real-trade-id>"
   curl -s "http://localhost:8000/api/paper-trading/trades/${TRADE_ID}/rationale?full=1" | jq '.signals | length'
   # Expect >= 5 on a full-pipeline trade (after 25.C surfaced layer-1 skills)

   curl -s "http://localhost:8000/api/paper-trading/trades/${TRADE_ID}/rationale?full=0" | jq '.signals | length'
   # Expect <= 3 (Analyst + Trader + RiskJudge maximum)
   ```
3. Open the drawer in the UI; expect:
   - Default render in compact mode -- 3 rows max.
   - "Show full view" button at top right. Click it; expect the full
     attribution tree to load (Layer-1 Skills, Quant, SignalStack rows
     appear). Button label flips to "Show compact view".

## Closes audit basis
bucket 24.4 F-3 RESOLVED. Operators now control drawer density per-view
without needing to scroll through 20+ rows by default.

**Audit anchor for next bucket:** 25.F3 (still pending), 25.B6, 25.B10,
follow-ups 25.C9.1 / 25.D9.1.
