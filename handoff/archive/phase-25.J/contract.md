# Sprint Contract — phase-25.J — Trade confirmation Slack

**Cycle:** phase-25 cycle 9
**Date:** 2026-05-12
**Step ID:** 25.J
**Priority:** P0 (FINAL P0 — completes phase-25.0 sprint)
**Depends on:** 25.1 (DONE — Step 5.6 stop-loss execute_sell uses this hook)

## Research-gate
Reuses phase-24.5 cycle 4 researcher gate (5 sources). Fix per audit F-5(a).

## Hypothesis
Adding a `trade_notifier` callable hook to PaperTrader + a `format_trade_confirmation` formatter + a `notify_trade_confirmation` async helper closes the operator-trade-visibility gap. Stop-loss sells (via 25.1's execute_sell call with reason='stop_loss_trigger') automatically flow through the same notifier — special-cased with rotating-light icon.

## Success criteria (verbatim)
1. execute_buy_emits_slack_message_on_success
2. execute_sell_emits_slack_message_on_success
3. stop_loss_trigger_emits_slack_message

## Plan
1. Add `trade_notifier: Optional[Callable[[dict], None]] = None` kwarg to `PaperTrader.__init__`
2. Add `_maybe_notify_trade(trade)` helper that try/except-wraps the dispatch
3. Call `self._maybe_notify_trade(trade)` at the end of `execute_buy` and `execute_sell` (after BQ persistence + cash update)
4. Add `format_trade_confirmation(trade)` to `formatters.py` with stop_loss special-case
5. Add `async notify_trade_confirmation(app, trade)` to `scheduler.py`
6. Verifier `tests/verify_phase_25_J.py` (14 claims incl. 2 behavioral round-trips)
7. Q/A
8. Cycle 65 log
9. Flip 25.J — phase-25 P0 sprint COMPLETE

## References
- `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md` F-5(a)
- `backend/services/paper_trader.py:32-55` (init + hook), :256, :383 (call sites)
- `backend/slack_bot/formatters.py:627` (new format_trade_confirmation)
- `backend/slack_bot/scheduler.py:notify_trade_confirmation` (new)

## Cross-process gap (deferred to 25.J.1)
Same as 25.K and 25.A8: paper_trader runs in backend process, Slack bot is separate. In-process hook covers the unit-test surface; cross-process delivery via BQ paper_trades polling is 25.J.1 (not blocking this PASS).
