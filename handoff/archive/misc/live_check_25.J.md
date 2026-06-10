# Live-check placeholder — phase-25.J

**Step:** 25.J — Trade confirmation Slack
**Date:** 2026-05-12

## Live-check field
> "Slack post visible within 60s of next execute_buy or stop trigger"

## Pre-deployment evidence
- 14/14 verifier PASS (incl. 2 behavioral round-trips)
- Hook architecture: PaperTrader exposes trade_notifier surface; consumer wires the dispatcher
- Backward compat: default `trade_notifier=None` → no-op (existing callers unaffected)
- Stop-loss-trigger sells (from 25.1's Step 5.6) automatically use this hook with `:rotating_light:` Slack icon
- format_trade_confirmation special-cases `reason='stop_loss_trigger'`

## Cross-process bridge (deferred 25.J.1)
The current hook is in-process: when PaperTrader is instantiated with a notifier, the notifier fires per trade. The Slack bot runs in a separate process, so production Slack delivery from autonomous_loop trades requires:
- BQ `paper_trades` polling job in slack_bot scheduler (every 30-60s)
- For each new row, call `notify_trade_confirmation(app, trade)`
- Track last-seen `created_at` to avoid re-posts

## Post-deployment verification (operator workflow)
1. Wire BQ polling job (25.J.1) OR directly inject `trade_notifier` into a backend PaperTrader instance
2. Trigger a paper trade (manual or autonomous cycle)
3. Confirm Slack post in configured channel within 60s
4. Stop-loss-triggered sells get the rotating-light icon

**Audit anchor for next bucket:** P0 sprint COMPLETE. P1 sprint begins; recommended first: 25.A2 (wire bq.save_report into full pipeline) — no deps, closes empty /reports bug.
