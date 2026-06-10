# Live-check placeholder -- phase-25.N

**Step:** 25.N -- Cycle-completion summary Slack notification
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Slack post after each completed autonomous cycle with duration, trades, stops, mode"

## Pre-deployment evidence
- 5/5 verifier PASS.
- AST clean on both touched modules.
- Behavioral round-trip in claim 4 confirms the formatter returns 4-block
  Block Kit shape: header + section + divider + context.
- Claim 5 confirms dedup key `cycle_completed_summary` is distinct from the
  failure-path key `cycle_<status>` so the two paths can't collide.

## Post-deployment operator workflow
1. Pull main + restart backend:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "uvicorn backend.main" || true
   python -m uvicorn backend.main:app --reload --port 8000 &
   ```
2. Trigger an autonomous cycle (or wait for the scheduled tick):
   ```
   curl -X POST http://localhost:8000/api/paper-trading/run-cycle
   ```
3. Watch Slack for the cycle-summary post within ~1-5 min of completion.
   Expected payload via webhook (`raise_cron_alert_sync` metadata):
   ```
   [P3] Autonomous trading cycle completed
   cycle_id: <8-char hex>
   started_at: <ISO timestamp>
   duration_sec: <float>
   trades_executed: <int>
   stops_executed: <int>
   mode: full | lite | dry_run
   recommendations_count: <int>
   status: completed
   ```
4. Confirm the failure-path still works by inducing a failure (e.g. pull
   the network briefly during a cycle) -- a P1 alert with
   `error_type=cycle_error` should fire separately, NOT collide with the
   P3 summary.

## Closes audit basis
bucket 24.5 F-5(e) RESOLVED. Operators now get a Slack signal on every
completed cycle, not just on failure.

**Audit anchor for next bucket:** 25.O (error escalation Slack routing),
25.C (Layer-1 28-skill output surfacing).
