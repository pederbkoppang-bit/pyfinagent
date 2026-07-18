# Money Recon Baseline — 2026-07-18 (feeds GOAL PHASE-72)

Provenance: ultracode recon 2026-07-18 — 3 read-only Explore auditors (P&L via BQ, blockers/dark-flag inventory, trade-funnel trace) + an adversarial verifier that independently re-ran every load-bearing BQ query and re-opened every cited file. Confirmed-to-the-cent facts only; refuted claims listed explicitly.

## Verified P&L facts (BQ `financial_reports.paper_portfolio_snapshots` / `paper_round_trips` / `paper_trades`)

- "No money in ~2 months" is PARTIALLY TRUE: +$972.75 organic since 2026-05-15 (TWR 14.51% -> 19.37%), but flat-to-down since the 05-29 anchor (NAV 24,023.58 -> 23,874.56 on 07-16 = -$149; -$804.82 from the 06-03 peak 24,679.38). The 05-29 row matches the operator's "+20% NAV / +14% alpha / 66% cash" to the decimal.
- $5,000 external deposit 2026-05-13 (the only flow in-window) — use flow-adjusted TWR (`cumulative_pnl_pct`), never raw NAV deltas.
- Trades were NET PROFITABLE: 29 round trips since 05-15 = +$3,194.68 realized (19W/10L). By exit reason: stop_loss_trigger 14 / +$1,941.10 (avg +17.82% — mostly trailing-profit captures), swap_for_higher_conviction 13 / +$1,103.18, sell_signal 2 / +$150.40. Costs negligible: $34.46 transaction + $16.75 analysis (~$51 total, ~0.2% NAV).
- Mechanism of flatness = IDLE CASH: positions_value 15,314.36 (05-15, 13 positions) -> 0.00 (100% cash 07-03..07-08) -> 660.13 (07-16: one AMD position, entered 07-09 avg 545.42, -8.31%). Cash 23,214.43 = 97.24% of NAV. Zero-trade week Jun 15-21; ~1 trade/week late June; NO trades since 07-13.
- Snapshot coverage ENDS 2026-07-16 — nothing for 07-17/07-18; `paper_metrics_v2` 07-17 row has NULL nav.

## Active July cause: degraded LLM scoring rail

- backend.log (rotation 07-06..07-18): "Degraded-scoring guard fired: 6/6 analyses scored 0/degraded" x9; "Meta-scorer ran ENTIRELY on the no-LLM fallback" x20; "consensus=HOLD" x48; "Executing 0 trades" in 8 of 9 cycles (the exception: 07-09, the AMD+MU re-entry).
- Code path: degraded analyses are dropped (`autonomous_loop.py:1103-1109`); `decide_trades` only BUYs on {BUY, STRONG_BUY} and silently `continue`s otherwise with NO log line (`portfolio_manager.py:63,182-189`).
- Suspects: standard tier pinned `claude-sonnet-4-6` Anthropic-routed (`settings.py:30`); cc_rail health probe FAILED; "Meta-scorer LLM-leg repair (credit-exhaustion class)" open follow-up (`cycle_block_summary.md:27`). The away-ops $0-metered/dark posture (06-12..07-06) overlaps the dead-trade window.
- Kill-switch is NOT the blocker: never paused since the manual resume 2026-06-11; peak 24,124.77 (06-22) vs ~23,875 now = ~1% trailing DD vs the 10% cap (`handoff/kill_switch_audit.jsonl`).

## REFUTED by the adversarial verifier — do NOT build on these

1. KS-PEAK-RESET lockout as the active cause — the precondition (>=10% pullback + flatten) never occurred; it is a latent time-bomb only (`kill_switch.py:237-268`, reset_peak DARK).
2. Swap-churn as an active P&L leak — swaps netted +$1,103.18 over the window; the widely-cited -$139.83 was a one-away-week subset. The 0.0-conviction-sentinel code smell is still real (`settings.py:345` flag OFF, `paper_swap_enabled` True at `:329`).
3. The "+14% alpha" headline — ~10pp of it is a `benchmark_pnl_pct` discontinuity 05-23 (14.97) -> 05-26 (4.76), a rebasing artifact, not outperformance. Do not republish alpha until benchmark integrity is fixed.

## Open angles the goal must close

- Benchmark methodology + the 05-23->05-26 discontinuity; benchmark recomputes daily even while NAV is frozen (07-03..08 NAV constant, alpha moves).
- Regime baseline: the benchmark itself was flat over the same 7 weeks (5.84 -> 5.18, dip to 1.87 on 06-10) — a flat book was partly regime-appropriate; the "deploy = profit" counterfactual is unsupported.
- Disentangle deliberate away-ops throttle (06-12..07-06) from live defect before attributing the stall to bugs.
- FX-1.0 mis-booking risk under ALL the P&L numbers (`paper_trader.py:392` / `fx_rates.py:93`; `register.md:18`): KR tickers 000660.KS / 005930.KS / 066570.KS traded in-window; `paper_trades` has no currency column.
- Segment the window — late-May gains (partly benchmark artifact) / June profit-taking-into-cash / July degraded-scoring stall — one verified cause per sub-period; no single-root-cause framing.

## Dark-lever inventory (code defaults from `backend/config/settings.py`; live .env UNCONFIRMED — subagents are permission-blocked from reading it)

- `paper_synthesis_integrity_enabled` (:197) + `paper_risk_judge_shape_fix_enabled` (:311) — OPERATOR-APPROVED 2026-07-09 (`handoff/operator_tokens.jsonl:1`), code default False, live state UNCONFIRMED. These are the BUY-survival-under-rail-degradation fixes.
- `kill_switch_peak_reset_enabled` (:38) — KS-PEAK-RESET token owed; latent permanent-lockout bomb.
- `paper_scale_out_enabled` (:34) OFF — no take-profit ladder; winners exit only via 8% trail or swap.
- `paper_position_recommendation_fix_enabled` (:201) OFF — signal_downgrade SELL structurally dead (`portfolio_manager.py:153-161`).
- `paper_session_budget_reconcile_enabled` (:456) OFF — hidden $1.00 session budget = HALF the visible $2.00 cap; BudgetBreachError swallowed (`autonomous_loop.py:90-120,354-357`).
- Phase-70 diversity bundle all dark (:447-450) — monosector top-5 momentum funnel (`screener.py:249-314`; `autonomous_loop.py:977-985`), per-sector cap 2 (:270).
- `paper_atomic_swap_enabled` (:453), `paper_cross_sector_rotation_enabled` (:454), `paper_avg_entry_fx_fix_enabled` (:455) OFF — non-atomic swap can net -1 position.
- `sign_safe_overlays` (:36), `regime_net_liquidity` (:37), `paper_data_integrity_enabled` (:45), `paper_learn_loop_enabled` (:33 — writer also crash-dead, `register.md:30`) OFF; ~20 alpha-overlay flags default OFF (:362-521).
- Silent BUY-suppressors (active, not flags): no-log non-BUY continue (`portfolio_manager.py:188`); 5% price-tolerance gate (`paper_trader.py:173-202`, `settings.py:560`); intl bad-bar door at logger.debug (`paper_trader.py:1258-1280`); FX-unavailable BUY skip (`paper_trader.py:225-233`).

## Operator greps

If Read on `backend/.env` is denied, ask the operator to run:

```
! grep -nE 'SYNTHESIS_INTEGRITY|RISK_JUDGE|SWAP|SCALE_OUT|SESSION_BUDGET|PAPER_MARKETS|PAPER_TRADING|MODEL|ANTHROPIC|GEMINI' backend/.env
```
