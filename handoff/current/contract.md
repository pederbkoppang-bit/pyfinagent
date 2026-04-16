# Sprint Contract -- MAS Harness Cycle 26
Generated: 2026-04-16

## Target
Phase 4.4.3.2: Slack signals tested end-to-end

## Hypothesis
The full signal-to-Slack code path is wired and produces valid Block Kit
messages. Verify via a stdlib-only drill that traces: generate signal dict ->
`SignalsServer.publish_signal` orchestration -> `format_signal_alert` Block Kit
rendering -> `WebClient.chat_postMessage` call site. Live Slack delivery
deferred to launch-week (precedent: 4.4.3.1 deferred runtime curl).

## Success Criteria

### Code-path verification (automated, stdlib-only)
- SC1: `format_signal_alert` exists in `backend/slack_bot/formatters.py`
- SC2: `format_signal_alert` accepts `(signal: dict, trade: dict | None)` signature
- SC3: `format_signal_alert` returns a `list[dict]` of Block Kit blocks
- SC4: Block Kit structure contains required block types: header, section (with fields), divider, context
- SC5: Header block text contains ticker and action (BUY/SELL/HOLD)
- SC6: Section fields include Confidence, Price, Size, Stop
- SC7: Context block contains "PyFinAgent" branding and signal_id
- SC8: `format_signal_alert` handles edge cases: empty dict input, missing fields, None trade
- SC9: `publish_signal` method exists on `SignalsServer` class
- SC10: `publish_signal` source contains `from backend.slack_bot.formatters import format_signal_alert`
- SC11: `publish_signal` source contains `WebClient` and `chat_postMessage` call site
- SC12: `publish_signal` passes `blocks` from `format_signal_alert` to `chat_postMessage`
- SC13: `publish_signal` has ASCII-only `text=` fallback for push notifications
- SC14: `publish_signal` graceful degradation: `slack_not_configured` when no token
- SC15: `_signal_emoji` helper maps BUY->green, SELL->red, HOLD->yellow
- SC16: No Unicode in logger messages within the Slack posting path (security.md rule)

## Excluded
- Live Slack delivery (requires running Slack bot + valid tokens -- launch-week)
- `--dry-run-send-test` CLI flag (not yet implemented; out of scope for this cycle)
- Visual confirmation in desktop/mobile Slack clients (Peder's part of "joint")
