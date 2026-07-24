# Goal Prompt -- goal-phase69-audit-burndown

Theme: Convert the 2026-07-10 ultracode audit (280 agents, double-verified; register: handoff/current/audit_phase69/register.md) into banked P&L protection: stop the working engine from destroying its own book, make the promotion instruments measure what they claim, and un-invert the live signal overlays -- before new alpha is trusted.

## Audit basis (50 confirmed / 30 contested / 4 refuted of 85)
1. execute_sell books non-USD proceeds at FX=1.0 when yfinance+FRED both fail: KRW credited as USD (~1300x phantom cash), EUR exits lose ~14% (paper_trader.py:392); fx_rates ignores its own historical_fx_rates fallback (fx_rates.py:93). Phantom NAV poisons the monotonic kill-switch peak -> spurious full-book flatten.
2. Trailing-DD kill-switch pause is UNRECOVERABLE: peak never resets, both resume paths refuse while the breach persists -- forever, once flattened to cash (kill_switch.py:212; :246 phantom breach on BQ timeout).
3. Any #ford-approvals message containing "clear queue" runs pkill -9 -f python: SIGKILLs the stack (slack_bot/commands.py:295).
4. Locks: a FAILED acquire unlinks the live pidfile (cycle_lock.py:144); unguarded init can strand the flock forever (autonomous_loop.py:167).
5. Promotion instruments broken (offline): DSR z inflated ~sqrt(252) by annualized-Sharpe/daily-T unit mix (backtest/analytics.py:323); walk-forward has NO purging, 90-135d label horizon vs 5d embargo (backtest_engine.py:587); weekend-boundary windows execute zero trades (:488); fracdiff at train only (:794); go-live booleans weaker than documented (paper_go_live_gate.py:111).
6. Live overlays invert on negative composites: macro/news/pead/options/insider multiply SIGNED scores, so positive catalysts DEMOTE names in drawdowns (macro_regime.py:547, news_screen.py:329 et al.); news batches >32 headlines truncate to {} (:282); QMJ Growth never computed (historical_data.py:202); INDPRO missing from regime series (top free-data lift, existing FRED key).

## Steps (masterplan phase-69; queued behind 67.4 revert + phase-68 P0s)
- 69.0 P0 design pack (frontier reasoning at dev time -- Fable if a window is open, else Opus 4.8): FX degradation chain; audited restart-replayable kill-switch peak-reset; sign-safe overlay algebra; DSR/purge corrections vs Bailey-Lopez de Prado reference values.
- 69.1 P0 book-safety: items 1-4, each with a red->green reproduction test. Peak-reset DARK until `KS-PEAK-RESET: APPROVED` (guard-behavior change; thresholds byte-untouched).
- 69.2 P0 gate correctness (offline, zero live surface): DSR units, purge+embargo, boundary snap, fracdiff-at-predict, go-live booleans to documented spec -- fixture-tested; incumbent re-validation waits on the historical_macro freeze (separate token).
- 69.3 P1 signal integrity + free-data lift: sign-safe overlays (flag-gated, ON-vs-OFF live_check), news token cap + parse-retry, QMJ fix; INDPRO + net-liquidity (WALCL-WTREGEN-RRPONTSYD) via a new cached path into _REGIME_SERIES -- historical_macro untouched.
- 69.4 P2 hand-offs, no execution: learn-loop tz TypeError -> 68.4; external-flow/deposit Sharpe corruption + STRING/TIMESTAMP trade query -> 68.5/68.6; FX-1 -> parked 61.3; 30 contested + Slack/UI display defects -> 63.3-style seeds.

## Boundaries (binding)
- $0 metered; free APIs only; paper-only.
- Do-no-harm: kill-switch limits, stops, sector caps, DSR>=0.95/PBO<=0.5 byte-untouched; fixes are fail-safe + ledger-math only.
- Hysteresis fixes stay banned; historical_macro stays frozen.
- Full 5-file protocol per step; harness stays exactly 3 agents; DARK-until-token on every guard-behavior change.

## Definition of done
69.0-69.4 PASS: the engine can no longer book non-USD proceeds at 1.0, brick itself post-flatten, or be SIGKILLed from Slack; corrected DSR/purge pinned to reference values; overlays proven sign-safe live; INDPRO+liquidity in the regime prompt; every non-owned defect filed with its owner -- live book untouched throughout.
