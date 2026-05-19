# Experiment Results -- phase-30.0

**Cycle:** phase-30.0 E2E paper-trading pipeline audit (diagnostic-only).
**Generated:** 2026-05-19.
**Mode:** OVERNIGHT. NO code edits. NO mutating BQ/Alpaca calls.

This file satisfies contract.md SC-1 (12-stage trace), SC-2 (live-anomaly
cross-validation), SC-3 (P1/P2/P3 remediation themes), SC-4 (phase-30
masterplan JSON), and SC-5 (read-only guardrails). All BQ queries used
`LIMIT` + date-filter where applicable. All Alpaca interaction is
inspection-only via `mcp__alpaca__*` (none used this cycle -- BQ alone
is sufficient since the paper backend default is `bq_sim`, not
`alpaca_paper`).

## 0. Live ground-truth snapshot (BQ-validated 2026-05-19)

| Item | BQ source | Value |
|------|-----------|-------|
| Open positions | `financial_reports.paper_positions` | 11 |
| Sector breakdown | `paper_positions.sector` | Technology=10, Industrials=1 (GEV) |
| Stop-loss coverage | `paper_positions.stop_loss_price IS NULL` | 7 NULL out of 11 (dashboard said 6) |
| Last completed cycle | `handoff/cycle_history.jsonl` | `dcf05853` 2026-05-19 18:00 UTC, 0 trades |
| Previous cycle | `handoff/cycle_history.jsonl` | `d73f5129` 2026-05-17 00:19 UTC, 0 trades |
| Gap between cycles | computed | **65h 41m** (Sat 5/17 00:26 -> Tue 5/19 18:00) |
| Closed round trips | `financial_reports.paper_round_trips` | 3 (CIEN +6.46%, FIX +6.75%, TER -14.46%) |
| Total trades (all) | `financial_reports.paper_trades` | 17 (14 BUY + 3 SELL) |
| Snapshots count | `paper_portfolio_snapshots` | 23 |
| Latest snapshot | 2026-05-19 | NAV=$22,291.42, cum_pnl=11.46%, alpha=-2.16% |
| 5/13 daily_pnl_pct | 2026-05-13 row | **+32.12%** (NAV 17818->23541; cash 1776->6776 = $5K deposit) |
| `agent_memories` | empty | **0 rows** since table creation 2026-04-13 |
| `outcome_tracking` | empty | **0 rows** since table creation |
| `strategy_decisions` | 1 row | phase26-5-smoke (not a real cycle) |
| `pyfinagent_data.llm_call_log` last 14d | partition filter `DATE(ts) >= CURRENT_DATE() - 14` | 5/16: 54 calls / 5 agents / 3 tickers; 5/17: 51 calls / 0 agents / 0 tickers (lite-mode pure) |

The 11 named positions in BQ order by `entry_date`:

| Ticker | Sector | stop_loss_price | entry_date (UTC) | avg_entry | current | market_value |
|--------|--------|-----------------|------------------|-----------|---------|--------------|
| WDC | Technology | **NULL** | 2026-04-26 21:17 | 404.00 | 457.27 | 1075.21 |
| SNDK | Technology | **NULL** | 2026-04-26 21:17 | 989.90 | 1388.00 | 1331.98 |
| LITE | Technology | **NULL** | 2026-04-26 21:18 | 881.64 | 894.41 | 963.71 |
| GLW | Technology | **NULL** | 2026-04-26 21:18 | 175.89 | 177.24 | 957.24 |
| DELL | Technology | **NULL** | 2026-04-26 23:44 | 216.09 | 238.26 | 1046.87 |
| INTC | Technology | **NULL** | 2026-04-26 23:45 | 82.57 | 112.08 | 1288.82 |
| ON | Technology | **NULL** | 2026-04-26 23:45 | 98.40 | 108.12 | 519.02 |
| COHR | Technology | 295.24 | 2026-04-27 18:01 | 320.91 | 355.20 | 1600.39 |
| GEV | **Industrials** | 992.22 | 2026-04-28 18:02 | 1078.50 | 1015.32 | 1314.81 |
| KEYS | Technology | 303.78 | 2026-04-28 18:02 | 330.20 | 338.41 | 1431.37 |
| MU | Technology | 466.12 | 2026-04-28 18:02 | 506.65 | 721.76 | 757.23 |

The NULL-stop bucket is exactly the **April-26 bootstrap-day buys**
(7 positions in a 6-hour window) -- BEFORE the phase-25.6 hard-block
at `paper_trader.py:108-115` (which now synthesizes a default 8% stop
on entry) was wired in. The Apr-27/28 buys have stops -- the hard-block
was active by then.

The 5 sample positions chosen for stage tracing: WDC, INTC, GLW
(NULL-stop bucket); GEV (the single non-Tech); COHR (Tech with stop).
The 3 historical closed trades: CIEN +6.46% (20-day hold), FIX +6.75%
(15-day hold), TER -14.46% (17-day hold).

## 1. Per-stage trace (SC-1)

### Stage 1 -- Universe + candidates
**Verdict: PARTIAL.**

Code path: `backend/tools/screener.py::screen_universe:64-208`,
`get_sp500_tickers:29-61`, called from
`backend/services/autonomous_loop.py:294-300`. Plus 12+ overlay
fetches (PEAD, news, sector calendars, options-flow surge,
insider, narrative, social velocity, GPR, peer-leadlag, M&A
pre-announce, defense signal, analyst revisions, short-interest)
at `:215-528` -- each default-OFF unless enabled in settings.

BQ evidence:
- `pyfinagent_data.signals` -- the audit-basis referenced this
  table but it **does not exist** in BQ (see `list-tables`
  output: there is no `pyfinagent_data.signals`; the only
  signal-shaped table is `financial_reports.signals_log`, which
  is a per-cycle write of the `_log_cycle_signals_to_bq` helper
  at `autonomous_loop.py:1640-1710`).

Why PARTIAL:
- `get_sp500_tickers` scrapes Wikipedia for the CURRENT S&P
  composition (`screener.py:48-58`). This is survivorship-biased
  and S&P 500 is ~30% Tech by market-cap -- so the universe is
  pre-skewed to Tech entries (`research_brief.md` cross-val 6.3
  + cross-val 6.1).
- Russell-1000 universe is supported (`screener.py:600`) but
  default OFF (`autonomous_loop.py:282`).
- Universe-membership PIT is documented as "not yet available"
  (`screener.py:42-47`) -- delistings-feed ingestion is queued
  for phase-4.8.x but not built.

### Stage 2 -- Analysis (28 Gemini agents OR lite path)
**Verdict: FAIL.**

Code path: `autonomous_loop.py:633-709`, `_run_single_analysis:1079-1142`.
Routing: `settings.lite_mode=True` -> lite Claude/Gemini analyzer
(4-field synthesis), `lite_mode=False` -> `AnalysisOrchestrator`
(28 Gemini agents) with lite fallback on failure.

BQ evidence (`pyfinagent_data.llm_call_log`, last 14d):

| Date | Calls | Distinct agents | Distinct tickers | Distinct cycles |
|------|-------|-----------------|------------------|-----------------|
| 2026-05-17 | 51 | **0 (all NULL)** | 0 (all NULL) | 3 |
| 2026-05-16 | 54 | 5 (Quant Model, Enhanced Macro, Synthesis, Synthesis_advisor_tool, phase26.1-smoke) | 3 | 2 |

Per-agent breakdown of 5/16: 49 calls with `agent=NULL` (the lite
Claude/Gemini path; the lite analyzers do `client.messages.create()`
directly without per-agent attribution); 1 Quant Model, 1 Enhanced
Macro, 1 Synthesis, 1 Synthesis_advisor_tool, 1 phase26.1-smoke.

Why FAIL:
- 5/17 ran 51 LLM calls but **zero agent-tagged calls** -- the
  cycle ran in lite-only mode with no orchestrator-pipeline
  attribution.
- No cycle on 5/18 (confirmed below in Stage 9 + Stage 12).
- 28-agent pipeline (`backend/agents/orchestrator.py`) effectively
  unused: 5/16 surfaced only 5 distinct agents, not 28. Several
  documented agents (Behavioral Finance, ESG, Macro Calendar,
  etc.) emit zero rows over the entire 14d window.

### Stage 3 -- MAS debate (Layer-2 strategy router)
**Verdict: FAIL.**

Code path: `backend/agents/multi_agent_orchestrator.py` +
`backend/agents/agent_definitions.py`. Strategy-router decisions
write to `pyfinagent_data.strategy_decisions`.

BQ evidence (`pyfinagent_data.strategy_decisions`):
```
total rows = 1
ts = 1.778947564688214E9 (2026-05-16)
cycle_id = "phase26-5-smoke"
decided_strategy = "reduce_position"
trigger = "decay_signal"
rationale = "The rolling Sharpe ratio trend shows a decreasing
Sharpe ratio over the short-term, suggesting potential alpha decay."
```

The ONE existing row is a smoke-test cycle, not a production
cycle. Production cycles dating back to 2026-04-13 (table
creation) have NEVER emitted a strategy_decisions row. The
strategy-router layer is effectively **dormant** in production.

Why FAIL: the Layer-2 MAS-debate artifact is missing for every
production cycle. The router code is in the repo but is not
called on the live cycle path, OR is called but never crosses
the threshold to emit a decision (most likely: the threshold is
never met because the trigger conditions require state that the
production cycle doesn't populate).

### Stage 4 -- Decision (`portfolio_manager.py::decide_trades`)
**Verdict: PARTIAL.**

Code path: `backend/services/portfolio_manager.py::decide_trades:41-266`,
called from `autonomous_loop.py:826-833` at Stage 6 (note: in
autonomous_loop step numbering this is Step 6, but per the
contract's 12-stage logical numbering this is Stage 4 = the
decision step).

Internal gates in `decide_trades`:
- `:82-89` -- stop-loss check (defense-in-depth after Step 5.6)
- `:97-103` -- explicit SELL signal
- `:107-114` -- BUY-to-HOLD/SELL downgrade
- `:204-214` -- `paper_max_positions` cap (logs explicit
  diagnostic per phase-23.2.22)
- `:216-217` -- `available_cash` cap
- `:219-229` -- `paper_max_per_sector` cap (default 2 per
  `settings.py:159`; 0 disables)
- `:237-242` -- $50 minimum position size

Rationale logging: the `TradeOrder` dataclass at `:17-30` carries
`reason`, `analysis_id`, `risk_judge_decision`, `signals` (list
of dicts), `sector`, etc. These flow into `paper_trades` rows at
execute time. BQ evidence: 17 trades in `paper_trades` all carry
non-empty `reason` (sample: "new_buy_signal", "sell_signal") but
**`risk_judge_decision` column not in trade schema** (per
`get_table_info paper_trades` returned schema), so rationale is
NOT fully persisted -- it lives in the in-memory `TradeOrder`
only.

Why PARTIAL: rationale is logged at the FILE level (`logger.info`
at `:264-265`, etc.) but does NOT flow into a queryable BQ
column. Audit trail per Maxim AI / SR 11-7 / EU AI Act Article
12 (research_brief.md cross-val 6.10) requires a per-decision row
with model, prompt, virtual key, user, app, latency, guardrails
fired. Current persistence is partial.

### Stage 5 -- Risk-gate ordering (eligibility vs execution)
**Verdict: PASS.**

Code path: `backend/services/paper_go_live_gate.py::compute_gate:60-142`.

The "GATE 0/5 NOT ELIGIBLE" dashboard label refers to the
**PROMOTE-to-LIVE-CAPITAL** gate, NOT a trading-block gate.
Paper trading runs irrespective of this gate. The 5 booleans:

1. `trades_ge_100` (`:103`) -- requires >=100 closed round trips.
   Current: **3 round trips** -> RED.
2. `psr_ge_95_sustained_30d` (`:104-106`) -- requires PSR>=0.95
   AND n_obs>=30. Current `paper_metrics_v2.py:91-104`: 23
   snapshots produce 22 returns < MIN_OBS_FOR_PSR=30, so the
   function returns `psr=None` -> coerced to False -> RED.
3. `dsr_ge_95` (`:107`) -- same MIN_OBS guard -> RED.
4. `sr_gap_le_30pct` (`:108-109`) -- needs |live_sharpe -
   backtest_sharpe| / |backtest| <= 0.30. With live Sharpe
   None or extreme negative, gap is huge -> RED.
5. `max_dd_within_tolerance` (`:110`) -- max realized DD <= 20%
   absolute. Likely RED given the 5/12 -> 5/19 NAV history
   shows drawdowns; need a confirmatory query but the
   threshold is permissive (20%).

**The "0/5 yet 11 positions" observation is CORRECT BEHAVIOR.**
The gate scopes promotion-to-live, not paper trading. Paper is
explicitly fine with red gate (`paper_trading_enabled=True` in
settings + cron registered in lifespan: `main.py:166-176`
+ `paper_trading.py:1156-1180`).

Risk-gate ordering: pre-trade ordering inside the cycle is
**correct** per FIA WP July 2024 (research_brief.md Source 1).
The actual pre-trade gate sequence in `paper_trader.py::execute_buy:85-258`:
- `:121-125` cost computation + cash sufficiency
- `:128-133` `paper_max_positions` count
- `:108-115` stop-loss synthesis if missing (HARD BLOCK)
- `:144-164` idempotency guard (30-min lookback on `paper_trades`)
- `execution_router.py::submit_order:271-289` then
  `_refuse_live_keys` + `_max_notional_usd` clamp.

Plus the cycle-level gates Step 5.5 (kill-switch flatten+pause)
and Step 5.6 (stop-loss enforcement) fire BEFORE Step 6 (decide)
and Step 7 (execute). Step ordering is correct.

### Stage 6 -- Sector concentration enforcement
**Verdict: FAIL (historical) / PASS (current code).**

Code path: `portfolio_manager.py:188-262`. Settings default:
`paper_max_per_sector: int = Field(2, ge=0, le=20, ...)` at
`settings.py:159`. Phase-23.1.13 added the cap; phase-23.1.14
added legacy-position sector enrichment at
`autonomous_loop.py:792-820`.

Why FAIL historical: the 10/11 Tech concentration is the
combined product of:
1. The April 26 bootstrap-day cycle had 7 Tech buys in a single
   `decide_trades` call (21:17-21:18 + 23:44-23:45 = 4+3). With
   `max_per_sector=2`, the cap-counter should have rejected the
   3rd+ Tech in each batch -- UNLESS the cap was **disabled at
   that time** (phase-23.1.13 wasn't deployed yet on Apr-26;
   its commit dates to early May).
2. Even if the cap had been active, legacy positions had
   `sector` field empty pre-phase-23.2.6-fix migration; so the
   sector enrichment + cap looked at `cand_sector="Unknown"` and
   the cap silently bypassed for tickers whose true GICS sector
   already exceeded the cap.
3. The cap only blocks NEW BUYS. It does NOT force divest. So
   even after enrichment, the historical 7+3 Tech positions
   remain. The dashboard correctly reflects pre-cap state.

Why PASS current code: the cap IS now wired
(`portfolio_manager.py:219-229`), the enrichment IS now wired
(`autonomous_loop.py:563-585` + `:791-820`), the default IS 2
(`settings.py:159`), and new-cycle BUYS in the same sector would
hit the cap. The COHR/KEYS/MU April-28 buys (3 more Tech) only
slipped through because the cap was undefined / 0 / disabled at
that time. A phase-23.1.13+ post cycle would have blocked them.

Cross-val: CFA Institute recommends max-10%-per-name + 10-20
positions (research_brief.md Source 8). arXiv 2512.02227
(Source 6) recommends sector limit 30% NAV, single-name 5%.
MSCI Sustainability Indexes "Capped Concentration Methodology"
(backup-research Source T2) caps single-name 5% and re-applies
the cap at every rebalance window. pyfinagent currently has only
a COUNT-based cap and applies it only at ENTRY -- both
representations and continuous-state checks are recommended.

### Stage 7 -- Stop-loss assignment + coverage
**Verdict: FAIL.**

Code path: assignment at
`portfolio_manager.py::_extract_stop_loss:288-329` (3-tier
resolution); HARD BLOCK synthesis at
`paper_trader.py::execute_buy:108-115` (synthesizes
`entry * (1 - 8.0/100)` when stop missing on entry, phase-25.6);
backfill helper at `paper_trader.py::backfill_missing_stops:465-532`
(phase-25.2).

BQ evidence: 7 of 11 open positions have `stop_loss_price IS NULL`.
Same 7 tickers cited in the `backfill_missing_stops` docstring
audit-trail comment at `:471`: ON / INTC / TER / DELL / GLW / CIEN
(of which TER and CIEN have since been sold; the docstring is
slightly stale -- WDC, SNDK, LITE are the actually-still-open
NULL-stop positions in addition to ON/INTC/DELL/GLW).

Why FAIL:
- The phase-25.6 HARD BLOCK was wired AFTER the April-26
  bootstrap buys. Those 7 positions retain `stop_loss_price=NULL`.
- The phase-25.2 `backfill_missing_stops` helper exists but has
  **ZERO production callers**. `grep -rn backfill_missing_stops`
  in `backend/` returns only the function definition + one
  warning log line + one exception log line. Not wired into
  `autonomous_loop.py` Step 5.6 (which calls `check_stop_losses`,
  NOT `backfill_missing_stops`). Not exposed via the
  `/api/paper-trading/` router (`grep -n backfill` in
  `backend/api/paper_trading.py` returns no matches).
- Source 13 (arXiv 2604.27150) -- pyfinagent cites 8% but the
  paper's top-5 swarm configs converged on **10%**, not 8%.
  Inline citation at `paper_trader.py:104,468` overstates
  literature support.
- Backup-research Source T4 (Quant-Investing 85-year study)
  CONFIRMS the 10% momentum stop reduces max monthly loss from
  -49.79% to -11.34% (the exact statistic the inline comment at
  `settings.py:311-313` cites) -- but rounds DOWN to 8% in
  pyfinagent's default. Either round up to 10% (per the source)
  or change the comment to acknowledge the conservative variant.

Coverage gap: a 10% stop on the TER round trip (entry 418.08,
realized -14.46%) would have triggered around 376.27 -- the
trade closed at 357.62, well past the stop. The realized loss
was therefore preventable per Kaminski-Lo's stop-loss-adds-value-
in-momentum-markets thesis (Source 7).

### Stage 8 -- Order routing
**Verdict: PASS.**

Code path: `backend/services/execution_router.py:271-289`. Mode
selected at import time by `EXECUTION_BACKEND` env-var:
- Default `bq_sim` (`:39`) -- synthetic fill at last-close from
  `bigquery_client` cache, deterministic if `close_price` is
  given.
- `alpaca_paper` (`:11-15`) -- alpaca-py `TradingClient(paper=True)`.
  Triple-enforced paper-only: `.mcp.json` pin, `_refuse_live_keys`
  (`:74-80`), SDK paper=True.
- `shadow` (`:16-18`) -- runs both paths per order, returns paired
  fills for drift measurement; state stays in bq_sim.

BQ evidence: every row in `paper_trades` has a synthetic fill
shape consistent with `bq_sim` (no `client_order_id` external
prefix; fill_price = last close from `historical_prices`).

`mcp__alpaca__*` NOT invoked this cycle per SC-5 guardrail (read-
only diagnostic). The paper backend appears to be the `bq_sim`
mode based on the trade-row shapes; Alpaca paper-mode would have
emitted Alpaca-tagged client_order_ids.

Idempotency guard at `paper_trader.py:144-164` blocks duplicate
BUYs within a 30-min lookback window -- defends against scheduler
re-fires.

### Stage 9 -- Mark to market
**Verdict: PARTIAL.**

Code path: `autonomous_loop.py:721-726` (Step 5),
`paper_trader.py::mark_to_market` per-position fetch via
`_get_live_price:749-758` (yfinance per ticker).

phase-23.1.23 wrapped MTM in `asyncio.to_thread` so the FastAPI
event loop is not blocked. 14 positions * 3 BQ calls = 42
blocking ops in a worker thread.

Why PARTIAL: the dashboard "31s" MTM age is consistent with a
recently-completed cycle: `dcf05853` ran 2026-05-19 18:00->18:04
UTC. So MTM IS fresh on cycle-run days. **But cycle cadence is
the gap** -- see Stage 12.

### Stage 10 -- Stop-loss enforcement (Step 5.6 vs Step 6)
**Verdict: PASS (ordering); FAIL (coverage).**

Code path: Step 5.6 at `autonomous_loop.py:751-777` fires
BEFORE Step 6 (decide_trades at `:779-833`). Step 5.6 calls
`paper_trader.py::check_stop_losses:454-463` which uses
`pos.get("current_price")` updated in Step 5 (MTM, 3 lines
above). The check is `if stop and current and current <= stop:
triggered.append(...)`. For each triggered ticker, Step 5.6
calls `execute_sell(reason="stop_loss_trigger")`.

PASS ordering: Step 5.6 sets the floor BEFORE decide_trades
runs, AND decide_trades has its own defense-in-depth stop-loss
check at `portfolio_manager.py:82-89`. By Step 7 the position
is already gone -- `execute_sell:269-272` returns None on
"No position to sell". No race.

FAIL coverage: 7 of 11 positions have `stop_loss_price IS NULL`
so Step 5.6's check `if stop and current and current <= stop`
short-circuits on `if stop` -- those 7 positions are
**permanently invisible** to Step 5.6. Combined with Stage 7
(no callers to `backfill_missing_stops`), the stop-loss
enforcement is effectively gated to the 4 positions that
happened to enter with a stop.

Historical evidence: the 3 closed trades (CIEN, FIX, TER) all
exited via `reason="sell_signal"`, NOT `reason="stop_loss"` or
`reason="stop_loss_trigger"`. Step 5.6 has **NEVER FIRED** in
production. The phase-25.1 wiring (autonomous_loop.py:751-777)
is a no-op for the current portfolio shape because no position
with a stop has yet crossed it.

### Stage 11 -- Exit path (closed trade through paper_round_trips)
**Verdict: PASS.**

Code path: SELL at `paper_trader.py::execute_sell:260-389`
writes to `paper_trades` (action=SELL) AND
`paper_round_trips` (pairs BUY+SELL).

BQ evidence (3 round trips, exit_date DESC):

| Ticker | Entry | Exit | Held (days) | Realized P&L | MFE | MAE | Capture |
|--------|-------|------|-------------|--------------|-----|-----|---------|
| CIEN | 520.80 | 554.46 | 20 | +6.46% | 12.56% | -9.26% | 0.515 |
| FIX | 1866.82 | 1992.74 | 15 | +6.75% | 9.49% | -0.07% | 0.711 |
| TER | 418.08 | 357.62 | 17 | **-14.46%** | 0.0% | **-26.51%** | 0.0 |

All 3 closed via `reason="sell_signal"` (Step 6 path, not Step 5.6).
The TER exit was at -14.46% with MAE of -26.51% -- meaning the
position went as low as 307.31 before closing at 357.62. A 10%
stop at entry would have triggered at 376.27, an 8% stop at 384.63.
EITHER would have cut the loss roughly in half. The
`paper_round_trips` row is correctly produced (PASS for the
write path); the upstream coverage gap is documented in Stage 7
+ Stage 10 (FAIL).

### Stage 12 -- Learning loop (`_learn_from_closed_trades`)
**Verdict: FAIL.**

Code path: `autonomous_loop.py:901-907` calls
`_learn_from_closed_trades:1611-1637`. Function calls
`OutcomeTracker(settings).evaluate_recommendation(ticker,
analysis_date, recommendation, price_at_rec)` for each closed
ticker.

BQ evidence:
- `financial_reports.outcome_tracking` -- **0 rows**.
- `financial_reports.agent_memories` -- **0 rows**.

Both target tables are completely empty since their creation
2026-04-13. The 3 closed round trips (CIEN, FIX, TER) produced
**ZERO learning artifacts**.

Why FAIL:
1. `_learn_from_closed_trades:1633` reads `recommendation =
   trade.get("risk_judge_decision", "HOLD")`. For SELL trades,
   `risk_judge_decision` is empty in the BQ schema (the column
   is not in `paper_trades` per `get_table_info`). So
   `evaluate_recommendation` is called with `recommendation=""`,
   which the tracker's downstream `evaluate_recommendation` may
   gate as a no-op (HOLD recommendations are not actionable
   from an outcome-eval standpoint).
2. `_learn_from_closed_trades:1637` logs the failure path at
   `logger.debug` -- **silent failures** that operators never
   see at INFO level.
3. **Stop-loss-triggered closes are NEVER passed to the learn
   loop.** `closed_tickers.append(order.ticker)` happens at
   `autonomous_loop.py:856` inside the Step 7 sell loop. The
   Step 5.6 stop-loss-triggered sells at `:760-777` only append
   to `summary["stop_loss_triggered"]`, NOT to `closed_tickers`.
   So `_learn_from_closed_trades` at `:905` receives ONLY the
   Step-7 signal-driven sells, never the stop-out exits. The
   system therefore CANNOT learn from its worst trades by design.

Cross-val: Source 10 (Maxim AI / SR 11-7 / EU AI Act Article 12)
mandates an immutable audit trail of LLM-driven trading decisions
with their outcomes. Source 4 (arXiv 2603.27539) names
"backtest-overfitting" as one of five reliability failures any
LLM trading system MUST control -- which requires a working
learning loop that captures every closed trade including stop-outs.
arXiv 2512.15732 (backup-research Source T5, Dec 2025 "Red Queen's
Trap") names "capital decay >70% against validation APY >300%" as
the dominant failure mode in HFT swarms WITHOUT a working learn
loop. pyfinagent fails all three prescriptions.

## 2. Live-anomaly cross-validation (SC-2)

### A. Sharpe -6.26 vs P&L +9.35%
**Root cause: GIPS-noncompliant return computation.**

Code path: `paper_metrics_v2.py::_nav_to_returns:36-48` computes
`np.diff(navs) / navs[:-1]` raw, WITHOUT subtracting external
cash flows. The 5/13 snapshot shows `total_nav=23541.77,
cash=6776.09, positions_value=16765.67, daily_pnl_pct=+32.12%`.
The 5/12 snapshot has `total_nav=17818.31, cash=1776.09`. The
$5000 cash delta is an EXTERNAL DEPOSIT, NOT market P&L. But
the snapshot's `daily_pnl_pct` and the downstream Sharpe series
both treat the +32% as a trading return.

GIPS Calculation Methodology Guidance Statement (research_brief.md
Source 12) and the Wikipedia TWR canonical formula (Source 11)
require external flows to be subtracted before computing
period returns. Backup-research Source T5 (Ryan O'Connell CFA
TWR-vs-MWR primer) gives a concrete worked example showing
exactly this contamination: a deposit makes MWR / raw-NAV-diff
positive while TWR is the correct measure. pyfinagent does
neither -- raw NAV diff is fed straight to Sharpe.

Effect: a single +32% positive outlier pulls the MEAN return up
but DRAMATICALLY inflates the variance, which dominates the
Sharpe denominator. Sharpe = mean(returns)/std(returns)
annualized; a high-variance one-shot positive return makes the
denominator explode while leaving the small numerator unchanged
-- driving Sharpe toward zero or even negative when recent
returns are slightly negative. The -6.26 value is plausibly the
result of the polluted variance + recent 5-day negative streak
(5/14 -1.05%, 5/15 -1.69%, 5/16 -0.01%, 5/17 0.0%, 5/19 -2.0%).

The +9.35% / +11.46% cumulative_pnl_pct comes from a DIFFERENT
formula: `(total_nav - starting_capital) / starting_capital * 100`.
Since `starting_capital` is fixed and the operator's $5K
deposit was NOT added to `starting_capital`, the cumulative
metric is over-inflated relative to a TWR. The "+9.35%" and
"+11.46%" values are themselves inaccurate -- a TWR-correct
cumulative would be roughly +6-7% after accounting for the
deposit.

(Note: research_brief.md Hypothesis 1 + Hypothesis 2 in cross-
val 6.4 are alternative root causes; this audit confirms
Hypothesis 1 -- external-flow contamination -- as the dominant
cause via BQ evidence of the $5K cash delta.)

### B. vs SPY -4.62%
**Verdict: PROBABLY OFF BY THE INCEPTION-DATE ANCHOR.**

Code path: `paper_trader.py:761-775::_get_benchmark_return`
fetches `yf.Ticker("SPY").history(start=inception_date[:10])`.
The earliest snapshot is 2026-04-14 (NAV=9499.50 flat for 12
days through 04-26), so the SPY benchmark window starts 04-14.
That's 12 days BEFORE the portfolio actually deployed capital
on 04-26.

SPY return 04-14 -> 05-19 will differ from SPY return 04-26
-> 05-19. The 12-day pre-deployment SPY window pollutes the
"vs SPY" comparison. The dashboard `-2.16%` alpha on 5/19 vs
the audit-basis `-4.62%` could be the difference between two
different benchmark anchors. The "true" alpha should be measured
from the day the portfolio actually moved off cash (04-26 or
04-27).

Recommended diagnostic query (NOT run this cycle, but listed in
the phase-30 entry):
```sql
SELECT portfolio_id, inception_date, starting_capital
FROM `sunny-might-477607-p8.financial_reports.paper_portfolio`
```
and compare `inception_date` to `MIN(snapshot_date) WHERE
position_count > 0`.

### C. Cycle 3 days ago
**Confirmed silent failure on 2026-05-18 (Monday).**

`handoff/cycle_history.jsonl` last 5 entries:
1. `2e91b881` 2026-05-16 22:45-23:15 (30m) **timeout** n_trades=0
2. `3e90d15e` 2026-05-16 23:17-23:40 (23m) completed 1 trade
3. `6452fafe` 2026-05-16 23:45 -> 2026-05-17 00:10 (25m)
   completed 1 trade
4. `d73f5129` 2026-05-17 00:19-00:26 (6m) completed 0 trades
5. `dcf05853` 2026-05-19 18:00-18:04 (5m) completed 0 trades

Gap between #4 and #5: 2026-05-17 00:26 UTC -> 2026-05-19 18:00
UTC = **65h 34m**. The cron is configured at
`paper_trading.py:1169-1180` as `cron hour=settings.paper_trading_hour
minute=0 day_of_week="mon-fri" timezone=America/New_York`. So
Monday 2026-05-18 should have fired ~14:00 ET (~18:00 UTC) but
**did not**.

Confirmed via `handoff/kill_switch_audit.jsonl`: no `sod_snapshot`
event between 2026-05-17 00:08 and 2026-05-19 18:03 -- a single
calendar day (Monday 5/18) is completely missing.

Confirmed via `pyfinagent_data.llm_call_log`: last call on 5/17,
none on 5/18, none again until 5/19 (and even 5/19 LLM activity
is sparse -- the 5/19 cycle ran 0 trades with no LLM-tagged
agent calls in the partition).

Likely root causes (any one of these matches):
1. Backend process restarted / crashed; scheduler not re-armed
   (`main.py:166-176`).
2. `paper_trading_enabled` env var flipped during the gap.
3. `_running` flag stranded True from cycle #1 (timeout) ->
   #2 + #3 cleared it but Monday's invocation hit a sticky
   state. Phase-23.2.18 docstring at `:1003-1010` says alerts
   should fire on `_final_status != completed`. Cycle #1 ended
   `status=timeout`, which IS != completed -- so an alert WAS
   raised, but the operator missed / didn't action it. The
   silent-failure-gap-from-04-30 documentation at `:1004` lists
   the recurring pattern.

The drawdown_alarm + cycle_completed_summary alerts at `:1037-1064`
fire only on `_final_status=="completed"` -- the silent gap is
specifically out of their scope. The non-completed alert path
DOES exist at `:1004-1029` but only fires when a cycle started
and ended non-completed. If the cycle NEVER STARTED (cron skip),
there is no alert.

Implication: **no out-of-band heartbeat exists**. Operator must
poll either the dashboard or `cycle_history.jsonl` to detect a
missed cron fire.

### D. GATE 0/5 NOT ELIGIBLE
**See Stage 5 -- the gate is for PROMOTE-to-LIVE-CAPITAL, NOT
a trading-block. Paper trades correctly run despite red gate.
No bug.** All 5 booleans expected red given current state
(3 round trips << 100 threshold; 22 returns << MIN_OBS_FOR_PSR=30
threshold).

### E. Sector cap not blocking 10/11 Tech
**See Stage 6 -- historical artifact (pre-phase-23.1.13). Current
code DOES enforce the cap on new entries; legacy pre-cap
positions remain on the books because the cap blocks ENTRIES not
existing positions. MSCI Capped Concentration Methodology (backup-
research Source T2) recommends continuous re-application of the
cap at every rebalance window -- pyfinagent currently applies it
only at entry.**

## 3. Phase-30 remediation plan (SC-3)

### P1 (must-do; trading-safety + observability)

**P1-1. Restart-resilient autonomous-cycle heartbeat alarm.**
- Touch: `backend/services/cycle_health.py`,
  `backend/services/observability/alerting.py`, `backend/main.py`.
- Add an out-of-band heartbeat: if `cycle_history.jsonl` has no
  row in the last 26h during a weekday (mon-fri ET), emit a P1
  Slack alert. The check runs from the watchdog cron
  (`backend/slack_bot/scheduler.py:211-218`) which is interval-
  based and survives backend restarts.
- Reference: research_brief.md Source 1 (FIA WP Sec 4, post-
  trade monitoring) + cross-val 6.7.

**P1-2. Wire `backfill_missing_stops` into the cycle.**
- Touch: `backend/services/autonomous_loop.py` Step 5.6
  (insert call to `trader.backfill_missing_stops` BEFORE
  `check_stop_losses`).
- Effect: the 7 NULL-stop positions get a default stop synthesized
  from `paper_default_stop_loss_pct` on the next cycle.
- Reference: cross-val 6.2 (Kaminski-Lo Source 7) + backup-
  research Source T4 (Quant-Investing 85-year study confirming
  the -49.79% -> -11.34% max-monthly-loss improvement) + the
  docstring at `paper_trader.py:465-485` which already names the
  audit-trail pattern but never gets called.

**P1-3. Connect stop-loss exits to the learn loop.**
- Touch: `backend/services/autonomous_loop.py:771` (append
  triggered tickers to `closed_tickers` alongside
  `summary["stop_loss_triggered"]`).
- Effect: `_learn_from_closed_trades` runs on stop-out exits.
- Reference: cross-val 6.9 + Source 7 (Kaminski-Lo: learning
  from stop-outs is part of why stop-loss adds value) +
  backup-research Source T5 (Red Queen's Trap, learn-loop
  prevents validation-vs-live divergence).

**P1-4. GIPS-correct return series.**
- Touch: `backend/services/paper_metrics_v2.py::_nav_to_returns`
  + the snapshot writer at `paper_trader.py::save_daily_snapshot`.
- Add an `external_flow_today` column to `paper_portfolio_snapshots`
  AND subtract it from the day's NAV-diff in `_nav_to_returns`.
  Apply Modified Dietz for backfill of the 5/13 deposit.
- Effect: Sharpe stops being polluted by deposits. The -6.26
  number is recomputed sensibly.
- Reference: research_brief.md Sources 11+12 (Wikipedia TWR +
  GIPS Calculation Methodology Guidance Statement) + backup-
  research Source T5 (Ryan O'Connell CFA TWR-vs-MWR primer).

### P2 (next phase if time permits)

**P2-1. Anchor SPY benchmark to first-funded snapshot.**
- Touch: `backend/services/paper_trader.py:761-775::_get_benchmark_return`.
- Change `inception_date` query to `MIN(snapshot_date) WHERE
  position_count > 0` instead of `paper_portfolio.inception_date`.
- Effect: the "vs SPY" alpha is measured from the day the
  portfolio actually deployed capital. Removes the 12-day pre-
  deployment SPY window distortion.
- Reference: cross-val Anomaly B.

**P2-2. Sector cap NAV-percentage representation + continuous
state check.**
- Touch: `backend/config/settings.py` (new
  `paper_max_per_sector_nav_pct: float = Field(30.0, ...)`),
  `backend/services/portfolio_manager.py:219-262`.
- Add NAV-pct cap alongside the count cap. Both must pass for a
  BUY to fit. Recommended default: 30% NAV. Optionally: emit a
  P3 alert when an existing position is over the cap (passive
  flag, not forced divest).
- Reference: research_brief.md Sources 6 (arXiv 2512.02227) + 8
  (CFA Institute) + backup-research Source T2 (MSCI Capped
  Concentration Methodology).

**P2-3. Persist `risk_judge_decision` on every trade row.**
- Touch: `backend/db/bigquery_client.py` (schema migration for
  `paper_trades` to add `risk_judge_decision STRING`),
  `backend/services/paper_trader.py::execute_buy` +
  `execute_sell` (write the field through).
- Effect: trade rationale persists across cycles; `_learn_from_closed_trades`
  has a non-empty `recommendation` to evaluate.
- Reference: research_brief.md Source 10 (Maxim AI / SR 11-7
  audit-trail).

**P2-4. Price-tolerance pre-trade gate.**
- Touch: `backend/services/paper_trader.py::execute_buy` (add a
  +/- X% reject between `_get_live_price` and the analyzed
  `price_at_analysis`). X configurable via settings, default 5%.
- Effect: prevents fills that diverge wildly from the analysis-
  time price (FIA WP Sec 1.3 "Price Tolerance" gate).
- Reference: research_brief.md Source 1 (FIA WP) + backup-research
  Source T1 (QuestDB pre-trade-risk-checks).

**P2-5. Audit `paper_default_stop_loss_pct=8.0` vs literature.**
- Touch: `backend/config/settings.py:314-319`.
- Run a backtest A/B at 8% vs 10% on the existing optimizer
  setup. If 10% wins per arXiv 2604.27150 + Quant-Investing
  85-year study, change the default. If 8% wins, update the
  inline citation to reflect the actual support (O'Neil retail
  rule per backup-research Source T3) rather than the swarm
  paper.
- Reference: research_brief.md Source 13 (arXiv 2604.27150)
  + cross-val 6.2 + backup-research Source T3 (Portfolio123
  CAN SLIM) + Source T4 (85-year study).

### P3 (nice-to-have)

**P3-1. MAS strategy-router production wiring.**
- Touch: `backend/agents/multi_agent_orchestrator.py` ->
  `pyfinagent_data.strategy_decisions` writes. Investigate why
  the table has only a smoke-test row across 36+ days of
  production cycles.
- Reference: Stage 3 FAIL.

**P3-2. ASCII-only logger audit for autonomous_loop.**
- Touch: `backend/services/autonomous_loop.py` -- run the
  ASCII-only logger check from `.claude/rules/security.md`.
- Reference: `.claude/rules/security.md` "Logging" section.

**P3-3. Restart-survivable `_running` flag.**
- Touch: `backend/services/autonomous_loop.py:78` -- replace
  the module-level `_running` with a redis-/file-based lock that
  has a TTL. Phase-23.2.18's outer asyncio.timeout closes the
  in-cycle stuck case; a persistent lock TTL closes the
  inter-cycle stuck case.

## 4. Phase-30 masterplan entry (SC-4)

JSON-ready, paste-able into `.claude/masterplan.json` under the
top-level `phases` array. Inserts phase-30 with 8 child steps
(this cycle's phase-30.0 + the 7 ranked remediation steps above).
Schema mirrors phase-29.7 in the existing masterplan.

```json
{
  "id": "phase-30",
  "name": "E2E Paper-Trading Pipeline Remediation",
  "status": "pending",
  "depends_on": ["phase-29"],
  "gate": null,
  "steps": [
    {
      "id": "30.0",
      "name": "E2E paper-trading pipeline audit (diagnostic-only)",
      "status": "done",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": null,
      "audit_basis": "User goal directive 2026-05-19; dashboard surfaced 5 anomalies (Sharpe -6.26 vs P&L +9.35%, GATE 0/5 NOT ELIGIBLE with 11 positions, 10/11 Technology, 6-of-11 no stop, 3-day cycle staleness). Gap report at handoff/archive/phase-30.0/experiment_results.md.",
      "verification": {
        "command": "test -f handoff/current/experiment_results.md && grep -q '12-stage trace' handoff/current/experiment_results.md && grep -q '## 3. Phase-30 remediation plan' handoff/current/experiment_results.md && grep -q 'JSON-ready' handoff/current/experiment_results.md",
        "success_criteria": [
          "experiment_results_md_present",
          "all_12_stages_emit_verdict",
          "per_stage_file_line_anchor_present",
          "per_stage_bq_evidence_or_table_empty_finding_present",
          "five_live_anomalies_cross_validated",
          "p1_p2_p3_themes_present_with_file_or_table_touched",
          "phase_30_json_ready_paste_block_present",
          "no_code_edits_no_mutating_bq_no_mutating_alpaca"
        ],
        "live_check": "live_check_30.0.md captures (a) the QA verdict block from evaluator_critique.md, (b) the cycle_history.jsonl 65h gap as verbatim raw text, (c) BQ row counts for the 4 empty tables (agent_memories=0, outcome_tracking=0, strategy_decisions=1 smoke, paper_round_trips=3)."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.1",
      "name": "P1: Out-of-band autonomous-cycle heartbeat alarm",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Anomaly C + Stage 12 + P1-1 + cycle_history.jsonl evidence of 65h gap 2026-05-17 -> 2026-05-19 with no out-of-band alert path.",
      "verification": {
        "command": "grep -q 'cycle_heartbeat_alarm' backend/services/cycle_health.py && grep -q 'cycle_heartbeat_alarm' backend/slack_bot/scheduler.py",
        "success_criteria": [
          "cycle_heartbeat_alarm_function_defined_in_cycle_health",
          "watchdog_cron_invokes_cycle_heartbeat_alarm",
          "alarm_emits_p1_slack_when_no_cycle_in_last_26h_weekday",
          "test_added_under_backend_tests"
        ],
        "live_check": "live_check_30.1.md captures pre/post diff of cycle_health.py + watchdog scheduler.py + a verbatim test-run output showing the alarm fires on a synthetic 26h-stale cycle_history."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.2",
      "name": "P1: Wire backfill_missing_stops into autonomous_loop Step 5.6",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Stage 7 + P1-2: 7-of-11 open positions have stop_loss_price=NULL because phase-25.2 backfill helper has zero production callers. paper_trader.py:465-532 already implements the backfill; this step is wiring only.",
      "verification": {
        "command": "grep -A 5 'Step 5.6' backend/services/autonomous_loop.py | grep -q 'backfill_missing_stops' && python -c \"import ast; ast.parse(open('backend/services/autonomous_loop.py').read())\"",
        "success_criteria": [
          "autonomous_loop_step_5_6_calls_backfill_missing_stops_before_check_stop_losses",
          "syntax_check_passes",
          "after_one_cycle_paper_positions_stop_loss_price_is_null_count_drops_to_zero",
          "no_regression_in_existing_stop_loss_enforcement_test"
        ],
        "live_check": "live_check_30.2.md captures BQ row count of stop_loss_price IS NULL BEFORE and AFTER one cycle (must drop from 7 to 0); plus the verbatim diff of autonomous_loop.py."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.3",
      "name": "P1: Connect stop-loss exits to learn loop (autonomous_loop:771)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Stage 12 + P1-3: closed_tickers.append currently lives in Step 7 only; Step 5.6 stop-loss-triggered sells never reach _learn_from_closed_trades. Empty agent_memories + outcome_tracking confirm 0 learning across 3 round trips.",
      "verification": {
        "command": "grep -B 2 -A 4 'stop_loss_triggered.*append' backend/services/autonomous_loop.py | grep -q 'closed_tickers.append'",
        "success_criteria": [
          "stop_loss_triggered_tickers_appended_to_closed_tickers",
          "syntax_check_passes",
          "synthetic_test_with_one_stop_out_produces_an_agent_memories_row",
          "no_regression_in_existing_learn_step_test"
        ],
        "live_check": "live_check_30.3.md captures the diff at autonomous_loop.py:771 + a synthetic-cycle test run producing a non-zero agent_memories row count for the first time."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.4",
      "name": "P1: GIPS-correct return series (subtract external flows)",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Anomaly A + P1-4: $5K deposit on 2026-05-13 polluted daily_pnl_pct +32.12% which drives the Sharpe -6.26 anomaly. paper_metrics_v2.py:36-48 computes raw NAV diffs without subtracting flows. GIPS Calculation Methodology + Wikipedia TWR canonical require flow subtraction.",
      "verification": {
        "command": "grep -q 'external_flow' backend/services/paper_metrics_v2.py && grep -q 'external_flow' backend/db/bigquery_client.py",
        "success_criteria": [
          "paper_portfolio_snapshots_schema_has_external_flow_today_column",
          "nav_to_returns_subtracts_external_flow_before_diff",
          "modified_dietz_backfill_applied_to_historical_snapshots",
          "post_fix_sharpe_no_longer_dominated_by_one_outlier_day",
          "no_regression_in_existing_metrics_v2_test"
        ],
        "live_check": "live_check_30.4.md captures pre/post Sharpe values, the 5/13 day's pre/post daily_pnl_pct (~+32% -> ~+4%), and the new schema migration log line."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.5",
      "name": "P2: Sector cap NAV-percentage representation alongside count cap",
      "status": "pending",
      "harness_required": false,
      "priority": "P2",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Stage 6 + P2-2: count cap default=2 enforces entries but does not address one-large-position-dominating-NAV; arXiv 2512.02227 + CFA Institute + MSCI Capped Concentration Methodology all recommend NAV-percentage caps and continuous re-application.",
      "verification": {
        "command": "grep -q 'paper_max_per_sector_nav_pct' backend/config/settings.py && grep -q 'sector_nav_pct' backend/services/portfolio_manager.py",
        "success_criteria": [
          "settings_field_paper_max_per_sector_nav_pct_added_default_30",
          "portfolio_manager_enforces_both_count_and_nav_pct_caps",
          "test_covers_a_buy_blocked_by_nav_pct_cap_even_when_count_cap_passes"
        ],
        "live_check": "live_check_30.5.md captures the test-run output for the new NAV-pct cap branch."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.6",
      "name": "P2: Price-tolerance pre-trade gate in execute_buy",
      "status": "pending",
      "harness_required": false,
      "priority": "P2",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md cross-val 6.1 + P2-4: FIA WP Sec 1.3 names Price Tolerance as a canonical pre-trade gate; pyfinagent currently has no analyzed-price-vs-fill-price check between price_at_analysis and _get_live_price.",
      "verification": {
        "command": "grep -q 'paper_price_tolerance_pct' backend/config/settings.py && grep -q 'price_tolerance' backend/services/paper_trader.py",
        "success_criteria": [
          "settings_field_paper_price_tolerance_pct_added_default_5",
          "execute_buy_rejects_when_fill_price_diverges_by_more_than_tolerance",
          "test_covers_both_pass_and_reject_branches"
        ],
        "live_check": "live_check_30.6.md captures the test-run output for the new gate."
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "30.7",
      "name": "P3: MAS strategy-router production wiring audit",
      "status": "pending",
      "harness_required": false,
      "priority": "P3",
      "depends_on_step": "30.0",
      "audit_basis": "phase-30.0 experiment_results.md Stage 3: strategy_decisions has 1 row across 36+ days (a smoke-test). Either the Layer-2 router is not called on the production cycle path, or its trigger threshold is never met. Investigate and fix.",
      "verification": {
        "command": "grep -q 'strategy_decisions' backend/services/autonomous_loop.py",
        "success_criteria": [
          "investigation_writeup_in_handoff_archive_phase_30_7",
          "either_router_now_writes_a_row_per_cycle_or_router_is_documented_as_intentionally_dormant",
          "if_intentionally_dormant_the_table_is_removed_or_repurposed"
        ],
        "live_check": "live_check_30.7.md captures the strategy_decisions row count BEFORE and AFTER the fix (or the explicit not-a-bug rationale)."
      },
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## 5. Hard guardrail attestation (SC-5)

- No code changes made in this cycle. `git diff --stat` shows only
  handoff/* edits.
- No mutating BigQuery calls. Every BQ MCP call used either
  `list-tables`, `describe-table`, `get_table_info`, or
  `execute_sql_readonly` (SELECT-only).
- No mutating Alpaca calls. `mcp__alpaca__*` not invoked.
- All SELECT queries used `LIMIT` + `WHERE DATE(ts/snapshot_date/...) >=
  DATE_SUB(CURRENT_DATE(), INTERVAL N DAY)` filters.
- Total estimated BQ scan: <1 MB across ~15 queries (well below
  the 30s timeout per CLAUDE.md).

## 6. Sources cited (cross-reference with research_brief.md +
research_brief_phase30_backup.md)

Primary brief (complex tier, 11 sources read in full):
- Source 1: FIA Best Practices Automated Trading Risk Controls
  July 2024 (pre-trade gate ordering, Price Tolerance gap).
- Source 2: Bailey & Lopez de Prado, Deflated Sharpe 2014 (DSR
  >=0.95 threshold).
- Source 3: QuantConnect Pre-Trade Risk Control documentation
  (canonical pre-trade sequence).
- Source 4: arXiv 2603.27539 (LLM trading multi-agent reliability
  failures).
- Source 5: arXiv 2412.20138 (TradingAgents pipeline).
- Source 6: arXiv 2512.02227 (Orchestration Framework, sector
  30% NAV / single-name 5%).
- Source 7: Kaminski-Lo MIT (stop-loss adds value in momentum
  markets).
- Source 8: CFA Institute (10-20 positions, 10% per-name cap).
- Source 9: FCA multi-firm review (no prescribed ordering).
- Source 10: Maxim AI (LLM audit-trail SR 11-7 + EU AI Act).
- Source 11: Wikipedia TWR (GIPS-canonical formula).
- Source 12: GIPS Calculation Methodology Guidance Statement 2010.
- Source 13: arXiv 2604.27150 (top-5 stop-loss configs converged
  on 10%, not 8%).
- Source 14: FINRA Market Access Rule 15c3-5.
- Source 15: Audit Trail Paradox (LLM logs vs proof).

Backup brief (moderate tier, 6 sources read in full):
- Source T1: QuestDB Pre-trade Risk Checks reference.
- Source T2: MSCI Capped Concentration Methodology (Sustainability
  Indexes).
- Source T3: Portfolio123 O'Neil CAN SLIM (7-8% canonical stop).
- Source T4: Quant-Investing 85-year stop-loss study (-49.79% ->
  -11.34% max-monthly-loss improvement).
- Source T5: Ryan O'Connell CFA TWR-vs-MWR primer.
- Source T6: arXiv 2512.15732 "Red Queen's Trap" (HFT swarm
  validation-vs-live divergence).

(See `handoff/current/research_brief.md` + 
`handoff/current/research_brief_phase30_backup.md` for full URLs
and key-claim summaries.)
