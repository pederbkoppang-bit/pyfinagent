# phase-69.4 Hand-offs — every confirmed audit defect routed to an owner

**No code execution.** This document files each of the 50 CONFIRMED audit findings (register:
`handoff/current/audit_phase69/register.md`) to a disposition so nothing is silently dropped, per
audit-remediation best practice (assign an owner to every finding + track-to-closure + exhaustive
traceability — Origami Risk / Hyperproof / TrustCloud / Webomates / Stell RTM; see `research_brief_69.4.md`).

## Coverage table — all 50 confirmed findings → disposition

Legend: **FIXED-69.2** (offline gate work, DONE) · **OWNED-69.1** (P0 book-safety, pending, phase-68-sequenced)
· **OWNED-69.3** (P1 signal integrity, pending) · **FILE→<owner>** (routed here, no execution)
· **RESIDUAL→63.3** (no named owner in the goal routing → 63.3 verified-defect register with P-level).

### Owned within phase-69 (16)
| # | Location | Disposition |
|---|----------|-------------|
| 28 | analytics.py:323 | FIXED-69.2 (crit1 DSR units) |
| 27 | backtest_engine.py:587 | FIXED-69.2 (crit2 purge+embargo) |
| 26 | backtest_engine.py:488 | FIXED-69.2 (crit3 boundary snap) |
| 29 | backtest_engine.py:794 | FIXED-69.2 (crit4 fracdiff/fill) + FO-69.2-A follow-on (true per-ticker FFD) |
| 9 | paper_go_live_gate.py:111 | FIXED-69.2 (crit5 go-live booleans) |
| 1 | paper_trader.py:392 | OWNED-69.1 (FX=1.0 default) |
| 4 | fx_rates.py:93 | OWNED-69.1 (no last-known FX fallback) |
| 3 | kill_switch.py:212 | OWNED-69.1 (peak never resets — peak_reset DARK until KS-PEAK-RESET token) |
| 10 | kill_switch.py:246 | OWNED-69.1 (current_nav<=0 phantom-breach guard) |
| 41 | commands.py:295 | OWNED-69.1 (clear-queue pkill removal) |
| 18 | cycle_lock.py:144 | OWNED-69.1 (failed-acquire unlinks live pidfile) |
| 17 | autonomous_loop.py:167 | OWNED-69.1 (unguarded init strands flock) |
| 23 | news_screen.py:329 | OWNED-69.3 (sign-safe overlay) |
| 24 | news_screen.py:282 | OWNED-69.3 (news token-cap min() inversion) |
| 25 | macro_regime.py:547 | OWNED-69.3 (sign-safe regime tilt) |
| 30 | historical_data.py:202 | OWNED-69.3 (QMJ Growth dead) |

### Filed to a NAMED owner via 69.4 (13)
| # | Location | Disposition | Claim |
|---|----------|-------------|-------|
| 12 | outcome_tracker.py:50 | **FILE→68.4** | tz-aware minus naive → TypeError; learn loop silently dead |
| 15 | outcome_tracker.py:118 | **FILE→68.4** | BQ datetime into str-only get_report → TypeError; kills eval batch |
| 13 | perf_metrics.py:116 | **FILE→68.6** | compute_sharpe_from_snapshots ignores external_flow_today → phantom returns in go-live Sharpe |
| 38 | bigquery_client.py:957 | **FILE→68.5/68.6** | get_paper_trades_in_window STRING created_at vs TIMESTAMP → fails BQ compile every call |
| 6 | paper_trader.py:1124 | **FILE→61.3** | trailing-stop peak from USD-return MFE; FX component distorts EU/KR trail |
| 14 | paper_round_trips.py:109 | **FILE→61.3** | realized_pnl_usd mixes KRW/EUR/USD as dollars in profit_factor/summarize |
| 42 | formatters.py:247 | **FILE→63.3** | /portfolio renders $0.00/+$0.00 (reads keys absent from envelope+schema) |
| 43 | _production_fns.py:222 | **FILE→63.3** | nightly_outcome_rebuild selects nonexistent cols; heartbeats ok, writes 0 rows |
| 44 | _production_fns.py:348 | **FILE→63.3** | weekly_data_integrity reads 'pct' vs emitted 'delta_pct' → drift always "(0.0%)" |
| 45 | scheduler.py:545 | **FILE→63.3** | digests US-calendar-only; EU/KR-holiday-day trades never reported |
| 46 | scheduler.py:1164 | **FILE→63.3** | nightly_mda_retrain/hourly_signal_warmup run stubs, heartbeat green |
| 47 | cockpit-helpers.tsx:308 | **FILE→63.3** | Risk Monitor reads perf.max_drawdown_pct (never returned) → permanent SAFE |
| 48-50 | layout.tsx:211 / live-portfolio-context.tsx:91,199 | **FILE→63.3** | cockpit no-poll divergence; freshness band never ages; 'P&L (Today)' resets ~$0 daily |

### Residual confirmed — no named owner in goal routing → 63.3 with P-level (19)
| # | Location | P | Claim |
|---|----------|---|-------|
| 2 | paper_trader.py:942 | P1 | external_flow producer dead + deposit bypass + snapshot MERGE clobber (RECOMMEND 68.6-adjacent deposit cluster) |
| 5 | paper_trader.py:555 | P1 | mark_to_market DELETE-then-upsert; transient BQ failure destroys open position (atomicity cluster) |
| 7 | paper_trader.py:1082 | P1 | portfolio cash unguarded read-modify-write; concurrent deposit/trade lost (atomicity cluster) |
| 37 | bigquery_client.py:557 | P1 | upsert_paper_portfolio non-atomic DELETE-then-INSERT; crash between = ledger row gone (atomicity cluster) |
| 39 | api/paper_trading.py:1254 | P1 | POST /deposit unguarded read-modify-write erases concurrent deposit (RECOMMEND deposit cluster) |
| 22 | autonomous_loop.py:1309 | P2 | stop-loss SELLs excluded from trades_executed; 'stops_executed' always 0 (RECOMMEND 68.6 — its crit counts stop-loss sells) |
| 8 | funding_guard.py:36 | P2 | T+1 funding + margin guards dead code (0 callers) |
| 11 | sector_calendars.py:122 | P2 | RTTNews FDA-calendar parser yields 0 events; binary-risk filter dead |
| 16 | reconciliation.py:136 | P2 | yfinance end EXCLUSIVE; final divergence date marks to prior close |
| 19 | cycle_health.py:523 | P1 | freshness flat-24h bands fire hourly false P1 on no-trade/weekend (paging noise) |
| 20 | cycle_health.py:216 | P1 | heartbeat 26h threshold ignores weekend; guaranteed false Monday P1 page (paging noise) |
| 21 | autonomous_loop.py:943 | P2 | session-budget HARD-BLOCK no-op (BudgetBreachError swallowed by gather) |
| 31 | orchestrator.py:2216 | P2 | bias audit reads wrong key; tech/confirmation-bias flags dead |
| 32 | llm_client.py:431 | P1 | 'LLM-spend' block meters BQ bytes-billed not LLM $; can halt trading on BQ scans |
| 33 | orchestrator.py:825 | P2 | per-step timeout doesn't free pipeline; timed-out LLM calls billed+discarded 3x |
| 34 | orchestrator.py:1835 | P2 | self.bq never assigned; yfinance-fallback telemetry write always AttributeError |
| 35 | llm_client.py:989 | P2 | GeminiClient 240s hang guard can't cancel hung HTTP |
| 36 | orchestrator.py:1810 | P2 | blocking network I/O on asyncio event loop (yf.news, BQ ctor) |
| 40 | sovereign_api.py:261 | P2 | cost query ignores cache tokens; understates Anthropic spend ~10x |

## Exhaustive 1..50 checksum (no silent drop)
By subsystem group (register order): Money-path 11 · P&L 5 · Loop/locks/scheduler 6 · Signals 3 ·
Backtest/gates 5 · LLM/orchestrators 6 · DB/API 4 · Slack 6 · Frontend 4 = **50 ✓**.
Every confirmed finding above carries a disposition; **zero silent drops**.

## Contested (30) + refuted (4)
- All **30 CONTESTED** findings → **63.3 seed entries**, each carrying location + claim + the split
  verifier verdict (read `contested.json` before acting — several are "code defect genuine, money impact
  low/degraded"). Full list in the register's "Contested findings (30)" table.
- **4 REFUTED** → no action (dropped by design; not defects): reconciliation.py:187, friday_promotion.py:81,
  llm_client.py:233 (_HOUSE_INSTRUCTIONS), monthly_champion_challenger.py:269.

## Already-filed follow-on
- **FO-69.2-A** (per-ticker time-series FFD for true fracdiff-at-predict; C4 acceptance condition) is filed
  at `handoff/current/audit_phase69/followons_69.2.md` — future live-adjacent step, gated on the
  historical_macro un-freeze. Acknowledged here; no re-file.

## Recommendations for Main / operator (no execution here)
1. **Money-ledger atomicity cluster** (5 / 7 / 37 — non-atomic DELETE+INSERT/upsert on positions, cash, and
   the portfolio singleton) is a coherent P1 group that deserves a dedicated near-term step rather than
   scattered 63.3 seeds; filed to 63.3 P1 by default pending an operator decision.
2. **Deposit/external-flow cluster** (2 / 13 / 39 + 68.6) — recommend consolidating under 68.6's go-live /
   external-flow ownership rather than splitting between 63.3 and 68.6.
3. **Paging-noise pair** (19 / 20 — false weekend/Monday P1 pages) is cheap, high-annoyance; good early 63.3 pick.
