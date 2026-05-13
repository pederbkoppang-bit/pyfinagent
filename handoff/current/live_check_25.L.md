# Live-check placeholder -- phase-25.L

**Step:** 25.L -- Drawdown alarm with tiered thresholds
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Inject 5%+ drawdown via fixture; Slack alert delivered"

## Pre-deployment evidence
- 6/6 verifier PASS (including behavioral mock round-trip at -6% confirming
  drawdown_warn_3pct (P2) + drawdown_warn_5pct (P1) fire, with critical_10pct
  NOT firing -- correct tier discrimination).
- AST clean on both touched .py files.

## Post-deployment operator workflow
1. Pull main + restart backend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Direct unit-test the alarm with synthetic snapshots:
   ```
   python -c "
   from backend.services.drawdown_alarm import check_drawdown_alarms, emit_drawdown_alarms
   snapshots = [{'total_nav': 10000.0}, {'total_nav': 10100.0}, {'total_nav': 9494.0}]  # -6%
   print('breached:', check_drawdown_alarms(snapshots))
   print('alerts fired:', emit_drawdown_alarms(snapshots, source='live_test'))
   "
   ```
3. Trigger an autonomous cycle and watch Slack for any drawdown alerts:
   ```
   curl -X POST http://localhost:8000/api/paper-trading/run-cycle
   ```
   Expected: only if portfolio NAV vs all-time peak >= 3% drop will alerts fire.

## Closes audit basis
bucket 24.5 F-5(c) + 24.8 RESOLVED.

**Audit anchor for next bucket:** 25.C (Layer-1 28-skill output surfacing),
25.D / 25.E (P2 backlog).
