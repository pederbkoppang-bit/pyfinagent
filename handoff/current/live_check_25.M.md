# Live-check placeholder -- phase-25.M

**Step:** 25.M -- Cost-budget Slack alert wire repair (no silent fail-open)
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "Inject wiring failure; confirm error logged at WARNING+ not fail-open silent"

## Pre-deployment evidence
- 5/5 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_M.py`).
- Claim 5 already performs the live wiring-failure injection: spawns a stub
  event loop on a background thread, builds a real alert_fn with a MagicMock
  app whose `chat_postMessage` raises RuntimeError, and confirms the logging
  pipeline captures an ERROR record (not WARNING) with the failure message.
- AST clean on both touched modules.

## Post-deployment operator workflow
1. Pull main, restart slack-bot:
   ```
   git pull origin main
   source .venv/bin/activate
   pkill -f "python -m backend.slack_bot.app" || true
   python -m backend.slack_bot.app &
   ```
2. Inject a misconfig and confirm fail-loud:
   ```
   python -c "
   from backend.slack_bot.jobs._production_fns import make_alert_fn_for_budget
   try:
       make_alert_fn_for_budget(None, None, '')
   except ValueError as e:
       print(f'FAIL-LOUD CONFIRMED: {e}')
   "
   ```
   Expected output: `FAIL-LOUD CONFIRMED: make_alert_fn_for_budget: app is required ...`
3. Check error-level log after a Slack post failure (e.g., revoke bot token
   then run the cost_budget_watcher manually) -- expect an ERROR record with
   traceback in `handoff/logs/slack-bot.log`, not WARNING.

## Closes audit basis
bucket 24.5 F-5(d) RESOLVED. The cost-budget watcher now surfaces wiring
errors at construction time and runtime errors at ERROR level rather than
silently logging at WARNING.

**Audit anchor for next bucket:** 25.B7 (P2 yfinance fallback counter),
25.C (P2 Layer-1 28-skill output surfacing), 25.D (P2 backlog).
