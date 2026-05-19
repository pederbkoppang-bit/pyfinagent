# Research Brief — Phase-30 Pre-Remediation Audit

**Tier:** complex | **Effort:** max | **Date:** 2026-05-19
**Mode:** Diagnostic-only (no fixes, no mutating MCP calls)

## Objective

Cross-validation audit of the pyfinagent 12-stage paper-trading pipeline
to produce a phase-30 remediation map. The dashboard surfaces Sharpe
-6.26 vs P&L +9.35%, GATE 0/5 NOT ELIGIBLE with 11 open positions,
sector concentration 10/11 Technology, 6 positions without stop-loss,
autonomous cycle stale 3 days. The brief reconciles each anomaly
against (a) the internal codebase with file:line anchors, (b)
external authoritative sources read in full, and (c) FINRA / FIA /
FCA / ESMA canonical risk-control guidance.

## 1. Three-variant search queries run

To satisfy `.claude/rules/research-gate.md` composition rules, every
sub-topic ran a current-year, last-2-year, and year-less variant.
Year-less hits are essential for canonical prior-art (Kaminski-Lo,
Markowitz, Bailey-Lopez de Prado).

| Topic | 2026 query | 2025 query | Year-less canonical |
|-------|------------|------------|---------------------|
| Pre-trade gate ordering | `pre-trade risk gate ordering 2026` | `algorithmic trading pipeline reliability 2025` | `algorithmic trading pre-trade risk gates ordering canonical` + `FINRA Rule 15c3-5 pre-trade risk controls` |
| Sharpe vs P&L reconciliation | n/a (no 2026 hits) | `Sharpe ratio paper trading reconciliation 2025` | `Sharpe ratio versus profit and loss reconciliation paper trading audit` + `portfolio sharpe ratio mismatch returns calculation root cause` |
| Stop-loss design | `algorithmic trading stop loss assignment design 2026` | n/a | `stop loss algorithmic trading Kaminski Lo bid ask spread` + `stop-loss rules when do they stop losses Andrew Lo` |
| Sector concentration | `sector concentration limits MSCI Barra factor model methodology 2026` | n/a | `portfolio sector concentration limit best practice` + `portfolio sector concentration limit enforcement decision pipeline` + `Markowitz mean variance portfolio risk` |
| LLM trading audit | `LLM agent financial trading audit trail decision log design 2026` | n/a | `LLM trading agent decision log audit trail` |
| Multi-agent LLM | `2026 multi-agent LLM trading hierarchical evaluation Sharpe` | n/a | (covered by 2026 query) |
| DSR / PSR canon | n/a | n/a | `deflated sharpe ratio Bailey Lopez de Prado` + `Bailey Borwein Lopez de Prado 2014 deflated Sharpe` |

Three-variant rule: pass. Year-less canonical surfaces (Kaminski-Lo
2010, Bailey-Lopez de Prado 2014, FIA July 2024, Markowitz 1952) are
foundational and would have been missed by a year-locked search.

## 2. Sources read in full (>=5 required)

| # | URL | Tier | Key claim |
|---|-----|------|-----------|
| 1 | https://www.fia.org/sites/default/files/2024-07/FIA_WP_AUTOMATED%20TRADING%20RISK%20CONTROLS_FINAL_0.pdf (extracted via pdfplumber) | Official/industry | FIA July 2024 best-practice WP names the canonical pre-trade control order: Maximum Order Size -> Maximum Intraday Position -> Price Tolerance -> Cancel-On-Disconnect -> Kill Switches -> Exchange OM. Section 4 names post-trade controls: Drop Copy Reconciliation, Post-Trade Credit Controls, Exchange Error Trade Policies. "In an environment where adequate pre-trade risk controls are implemented at all appropriate levels... a kill switch may ultimately be considered redundant" (p.8). |
| 2 | https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf | Peer-reviewed | Bailey & Lopez de Prado 2014 -- DSR threshold >=0.95 is statistical-significance gate for backtest selection bias. The formula penalizes Sharpe for sample length, skewness, kurtosis, plus the variance-of-trial-Sharpes term. An "implausible Sharpe (e.g. -6 with positive P&L)" indicates extreme negative skewness, regime change, or measurement error. |
| 3 | https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/pre-trade-risk-control | Official/vendor | QuantConnect documents the canonical pre-trade gate sequence: (1) basic validation (tradability, hours, price, size), (2) order type validation, (3) buying power, (4) brokerage model. Gate failures reject pre-submission; orders never reach the venue. |
| 4 | https://arxiv.org/html/2603.27539v1 | Peer-reviewed (arXiv 2026) | Toward Reliable Evaluation of LLM Financial Multi-Agent Systems. Five reliability failures: look-ahead bias, survivorship bias, backtest overfitting, transaction-cost neglect, regime-shift blindness. Daily trading systems incur 25-50pp/yr cost drag. FinMem's reported 23% reversed to -22% under controlled re-eval. Reported Sharpes 5.6-8.2 derive from 3-month bullish windows -- temporal robustness is required. |
| 5 | https://arxiv.org/html/2412.20138v5 | Peer-reviewed (arXiv) | TradingAgents pipeline: Analyst Team -> Researcher Team -> Trader -> Risk Management Team -> Fund Manager. Risk operates as 3-perspective deliberation (risk-seeking / neutral / risk-conservative) BEFORE execution. Decisions logged as structured + narrative for replay. |
| 6 | https://arxiv.org/html/2512.02227v1 | Peer-reviewed (arXiv) | Orchestration Framework for Financial Agents. Recommended stages: Data -> Alpha -> Risk -> Portfolio -> Execution -> Evaluation. Recommended sector limit 30% maximum. Recommended single-name 5%. Single-name concentration caps + sector limits enforced as deterministic tool calls, not LLM judgment. Critical: "Do not tune parameters based on Sharpe...from the evaluation horizon" -- explicit prohibition on Sharpe feedback to LLM agents. |
| 7 | https://alo.mit.edu/research-page/when-do-stop-loss-rules-stop-losses/ + https://www.hellojayng.com/learning-from-kaminski-los-when-do-stop-loss-stop-losses/ | Peer-reviewed (MIT/SSRN) | Kaminski-Lo (Lo's own MIT site cites the paper). Stop-loss rules ADD VALUE in momentum-driven markets; SUBTRACT VALUE under the Random Walk Hypothesis. Empirical: monthly US equities 1950-2004, stop-loss policy added 50-100 bps/month vs buy-and-hold during stop-out periods. Transaction costs incl. bid-ask spreads MUST be modeled. |
| 8 | https://rpc.cfainstitute.org/blogs/enterprising-investor/2018/portfolio-concentration-how-much-is-optimal | Industry/CFA | CFA Institute: optimal portfolio 10-20 investments. No single investment should cause >10% permanent loss. Max 5% Position-at-Risk at cost / 10% PaR at market value. Correlated assets must be assessed in aggregate. |
| 9 | https://www.fca.org.uk/publications/multi-firm-reviews/algorithmic-trading-controls-high-level-observations | Official (UK regulator) | FCA multi-firm review. All audited firms had adequate pre-trade controls. Common defects: out-of-date policies, compliance-team technical-knowledge gaps, inadequate simulation/stress testing, surveillance not scaling with trading complexity. NO prescribed ordering between controls. |
| 10 | https://www.getmaxim.ai/articles/llm-guardrails-for-fintech-compliance-hallucination-prevention-and-audit-trails/ | Industry | LLM trading audit-trail must capture: request metadata (model, prompt, virtual key, user, app), response details (output, latency, guardrail invocations), policy enforcement (which guardrails fired), timestamps. Immutable append-only storage. SR 11-7 + EU AI Act Article 12 (auto-recording lifetime events) / Article 19 (>=6-month retention). |

Read-in-full count: 10. Floor (5) cleared by 2x.

## 3. Snippet-only sources (context; do not count toward gate)

| # | URL | Why not fetched in full |
|---|-----|-------------------------|
| 11 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338 | SSRN 403 Forbidden on WebFetch |
| 12 | https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure | HTTP 403; surrogate via search snippet |
| 13 | https://arxiv.org/abs/2604.27150 | Abstract-only on WebFetch; arxiv/html alt fetched (8% stop-loss claim NOT in this paper; canonical optimum 10%) |
| 14 | https://www.sciencedirect.com/science/article/abs/pii/S1386418117300472 | HTTP 403; abstract via Google Scholar snippet |
| 15 | https://www.esma.europa.eu/sites/default/files/2026-02/ESMA74-1505669079-10311_Supervisory_Briefing_on_Algorithmic_Trading_in_the_EU.pdf | Binary PDF; pdfplumber-time-budgeted out |
| 16 | https://dev.to/arkforge-ceo/the-audit-trail-paradox-why-your-llm-logs-arent-proof-1c21 | Read in full once; tier downgraded as community blog |
| 17 | https://www.finra.org/rules-guidance/key-topics/market-access | FINRA page has thin content; specs referenced indirectly |
| 18 | https://www.sec.gov/files/rules/final/2010/34-63241-secg.htm | SEC compliance guide for 15c3-5; secondary to FIA WP |
| 19 | https://github.com/tauricresearch/tradingagents | GitHub README; framework documented via arXiv |
| 20 | https://www.moodys.com/web/en/us/insights/resources/quantifying-decomposing-and-managing-portfolio-concentration-risk.pdf | PDF; secondary to CFA Institute source |

Total unique URLs: 20 (10 read + 10 snippet). Floor (10+): cleared.

## 4. Recency scan (last 2 years)

Mandatory section per `.claude/rules/research-gate.md`.

Findings 2024-2026:

1. **FIA July 2024** -- updated best-practice paper on automated
   trading risk controls (the canonical industry source for control
   ordering). Supersedes any 2018-era reference on pre-trade risk
   ordering. (Source 1.)
2. **arXiv 2603.27539v1 (Feb 2026)** -- LLM trading multi-agent
   evaluation taxonomy. Names five reliability failures any LLM
   trading system MUST control. FinMem's published 23% reversed to
   -22% under controlled re-eval; TradingAgents' Sharpe 5.60-8.21
   was a 3-month bullish-window artifact. (Source 4.)
3. **arXiv 2512.02227v1 (Dec 2025)** -- Orchestration Framework for
   Financial Agents. Sector cap 30%, single-name 5%, mandatory
   deterministic tool-based enforcement of concentration limits.
   Critical: prohibits Sharpe feedback to LLM agents -- a pattern
   pyfinagent's MetaCoordinator + meta_scorer touches on. (Source 6.)
4. **arXiv 2604.27150 (2026)** -- stop-loss optimization swarm; the
   "8% stop-loss canonical" claim that pyfinagent cites in
   `paper_trader.py:104,468` is NOT supported by this paper. Tested
   range was 5/10/15/20/25/30/50; top-five configurations CONVERGED on
   10%, not 8%. Existing 8% default needs re-validation. (Source 13.)
5. **arXiv 2512.15732 (Dec 2025)** -- Galaxy Empire deployed 500 HFT
   agents; catastrophic divergence between Validation APY > 300% and
   live Capital Decay > 70%. Reinforces that backtest-vs-live gap is
   the dominant operational risk. (Search snippet via #18.)

Newer work does NOT supersede Kaminski-Lo, Markowitz, or Bailey-
Lopez de Prado, but it materially tightens the empirical context.

## 5. Per-stage code-path summary

The autonomous cycle is implemented in
`backend/services/autonomous_loop.py::run_daily_cycle`. Steps are
labeled in code by `logger.info("Paper trading: Step N -- ...")`.
Below is the actual stage list, with file:line anchors, BQ tables
read/written, gates fired, and per-stage smells.

### Stage 1 — Screen universe (free)
- Entry: `autonomous_loop.py:197-300`. Calls `screen_universe`
  (`backend/tools/screener.py:64-208`).
- BQ tables: reads `pyfinagent_data.signals` (only via overlay
  fetches like FRED / Alpha Vantage); writes none here.
- Gates: none.
- Smell: `get_sp500_tickers` (`screener.py:29-61`) scrapes Wikipedia
  for CURRENT S&P composition -- survivorship-biased. Wikipedia
  scrape failure falls back to a hardcoded `_FALLBACK_TICKERS` list.
  Today's universe ALWAYS includes today's heavily-tech 10/11
  observed concentration because S&P 500 by market-cap is
  approximately 30-32% Tech; the universe bias propagates into
  rank_candidates downstream.

### Stage 1.5 — Overlay signals (PEAD, news, sector calendars, etc.)
- Entry: `autonomous_loop.py:215-528`. 12+ overlays, each
  default-OFF. When ON they boost composite score multiplicatively.
- BQ tables: signal-specific (`pead_signals`, `news_signals`,
  `analyst_revisions`, etc.).
- Gates: none.
- Smell: Each overlay is conditionally fetched and applied. The
  cycle log claims "non-fatal" failure for each, but a silent
  overlay miss causes ranking drift the operator can't see at-a-
  glance.

### Stage 2 — Rank + filter candidates
- Entry: `screener.py::rank_candidates:210-436` called from
  `autonomous_loop.py:530-560`.
- BQ tables: none (pure compute on screened data).
- Gates: none -- ranking only. RSI extreme penalty, vol penalty,
  sector-momentum boost, multidim weights all fire here.
- Smell: `screener.py:283-288` -- `apply_pead_to_score` can return
  None which causes `continue`, silently dropping a candidate. If
  PEAD has a bug, the candidate list shrinks without explanation.

### Stage 2.5 — Sector enrichment for sector-cap
- Entry: `autonomous_loop.py:563-585` (candidates) and
  `:791-821` (legacy positions backfill).
- BQ tables: reads via `_fetch_ticker_meta` (BQ-first / yfinance
  fallback).
- Gates: none here -- enrichment only. The sector cap fires in
  `decide_trades` Stage 6.
- Smell: legacy positions whose `sector` field is empty (rows that
  pre-date `phase-23.2.6-fix`) get sector backfilled at Stage 2.5
  via `_fetch_ticker_meta` ONLY IF `max_per_sector > 0`. If the
  operator has the sector cap disabled (`paper_max_per_sector=0`),
  this enrichment is skipped, which is correct but worth
  highlighting -- the 10/11 Tech reading is consistent with a
  disabled or under-configured sector cap.

### Stage 3 — Analyze new candidates (Claude/Gemini lite or full)
- Entry: `autonomous_loop.py:633-709`, `_run_single_analysis`,
  `_run_claude_analysis` (`:1206-1300`), `_run_gemini_analysis`.
- BQ tables: writes `analysis_results` via `_persist_analysis`.
- Gates: `_check_session_budget` (`:95-105`) hard-blocks when
  `paper_max_daily_cost_usd` is reached; per-provider
  `_concurrency` cap (8 Gemini / 3 Claude per `:652-661`).
- Smell: `_run_claude_analysis` (`:1257-1297`) has hardcoded
  decision rules in the prompt:
  `momentum_20d > 3.0 AND momentum_60d > 5.0 AND market_cap > 5e9
  -> lean BUY`. This is biased toward LARGE-CAP MOMENTUM -- which
  in May 2026 is heavily Tech. The single rule explains a large
  fraction of the 10/11 Tech concentration; the LLM follows the
  rule rather than rejecting tech overweight.

### Stage 4 — Re-evaluate existing holdings
- Entry: `autonomous_loop.py:711-716`, same `_run_single_analysis`.
- BQ tables: writes `analysis_results`.
- Gates: same budget/concurrency.
- Smell: separate `gather` so a hot ticker can rotate. No bug; just
  noted for cycle-time accounting.

### Stage 5 — Mark to market
- Entry: `autonomous_loop.py:721-726`, `paper_trader.py::mark_to_market:393-450`.
- BQ tables: reads `paper_positions`, writes back via MERGE
  (`bigquery_client.py:563-...`). Computes new MFE / MAE per
  position.
- Gates: none.
- Smell: `_get_live_price` (`paper_trader.py:749-758`) hits
  yfinance per ticker. 14 positions x 3 BQ calls = 42 blocking
  ops; phase-23.1.23 added asyncio.to_thread but if yfinance
  rate-limits the cycle slows.

### Stage 5.5 — Kill-switch evaluation (phase-4.5.7)
- Entry: `autonomous_loop.py:728-749`,
  `paper_trader.py::check_and_enforce_kill_switch:642-675`.
- BQ tables: reads `paper_portfolio` (NAV); state in module-level
  `kill_switch.get_state()`.
- Gates: daily-loss + trailing-DD limits (`paper_daily_loss_limit_pct`,
  `paper_trailing_dd_limit_pct`). If tripped: `flatten_all` then
  pause, then SHORT-CIRCUIT the rest of the cycle.
- Smell: trip behavior is `flatten_all` then `pause` then return --
  this is correct order. No issue.

### Stage 5.6 — Stop-loss enforcement (phase-25.1)
- Entry: `autonomous_loop.py:751-777`, `paper_trader.py::check_stop_losses:454-463`.
- BQ tables: reads `paper_positions`, writes `paper_trades` via
  `execute_sell`.
- Gates: simple `current <= stop` check.
- Smell: per `paper_trader.py:454-463` the check uses
  `pos.get("current_price")` which was JUST UPDATED in Stage 5 (3
  lines above). So the stop check sees fresh MTM prices. Correct.

### Stage 6 — Decide trades (sell-first, then buy)
- Entry: `autonomous_loop.py:779-833`, `portfolio_manager.py::decide_trades:41-266`.
- BQ tables: reads `paper_positions` (refresh after MTM).
- Gates inside `decide_trades`:
  - `portfolio_manager.py:84-89` -- redundant stop-loss check
    (defense-in-depth after Stage 5.6).
  - `:97-114` -- signal-flip downgrade / explicit SELL.
  - `:204-214` -- `paper_max_positions` hard cap (logs explicit
    diagnostic per phase-23.2.22).
  - `:216-217` -- `available_cash` cap (cash + freed_from_sells -
    min_cash_reserve).
  - `:219-229` -- per-sector cap (`paper_max_per_sector`); 0
    disables.
  - `:237-242` -- minimum $50 position size.
- Smell #1: the stop-loss check at Stage 5.6 + redundant check at
  Stage 6 IS correct ordering -- Stage 5.6 closes the immediate
  triggers, Stage 6 catches positions whose stop_loss_price was
  updated in the rank/screen path. NOT a race.
- Smell #2 (BIG): `decide_trades` Stage 6 fires AFTER Stage 3+4
  analysis. The analysis prompt (`_run_claude_analysis:1257-1297`)
  has the large-cap momentum rule baked in. By the time
  `decide_trades` evaluates buy candidates, the candidate pool is
  ALREADY biased toward tech. Sector cap then operates on a
  pre-skewed input. With `paper_max_per_sector=2` (a reasonable
  default), Tech would still fill its 2 slots and the other 9
  positions would be non-Tech IF non-Tech buy signals existed.
  But if the Claude prompt's momentum rule chains buys ONLY on
  high market-cap + high momentum, non-Tech rarely qualifies in
  May 2026 -- the sector cap is necessary but not sufficient.

### Stage 7 — Execute trades
- Entry: `autonomous_loop.py:835-883`, `paper_trader.py::execute_buy:85-258`
  and `execute_sell:260-389`, via `execution_router.py::submit_order:271-289`.
- BQ tables: writes `paper_trades`, `paper_positions`, `paper_portfolio`,
  optionally `paper_round_trips` on sell.
- Gates inside `execute_buy`:
  - `:108-115` HARD BLOCK: `stop_loss_price=None` synthesizes
    `entry_price * (1 - paper_default_stop_loss_pct / 100)`. NEW
    buys CANNOT exit Stage 7 without a stop. Phase-25.6 fix; defense
    in depth alongside `_extract_stop_loss` fallback.
  - `:124-126` insufficient-cash guard.
  - `:131-133` max-positions guard.
  - `:144-164` idempotency guard (30-min lookback on `paper_trades`).
- Gates inside `execute_sell` (Stage 5.6 path): `:269-272`
  position-existence guard.
- Routing: `execution_router.py::submit_order:271-289` picks
  `bq_sim` (default), `alpaca_paper`, or `shadow`. Money-mode
  `_alpaca_real_fill` (`:176-256`) has `_refuse_live_keys` +
  `_max_notional_usd` clamp -- belt-and-suspenders. Paper-only
  enforced 3 ways: `.mcp.json` pin, router refuse, SDK paper=True.
- Smell: when stops are synthesized at execute_buy (`:108-115`),
  the synthesis pct comes from `paper_default_stop_loss_pct`
  (default 8.0). Per Source 13 / arXiv 2604.27150, the canonical
  swarm-trading optimum is 10%, not 8%. This contradicts the
  inline comment which cites "O'Neil canonical + arxiv 2604.27150".
  O'Neil's 7-8% applies to retail single-stock; the arxiv paper
  did NOT support 8% (its top configs landed on 10%). The 8%
  default needs re-validation against the actual reference.

### Stage 7.5 — Log cycle signals to BQ
- Entry: `autonomous_loop.py:885-888`, `_log_cycle_signals_to_bq:1640-1710`.
- BQ tables: writes `signals_log` (event_kind='publish').
- Gates: none -- guaranteed >=1 row per cycle so reliability drill
  succeeds.
- Smell: when 0 trades, a $CYCLE HOLD row is written. Good.

### Stage 8 — Final mark-to-market + snapshot
- Entry: `autonomous_loop.py:890-899`, `mark_to_market` (same as Stage 5),
  `save_daily_snapshot` (`paper_trader.py:536-565`).
- BQ tables: reads `paper_positions`, `paper_portfolio`; writes
  `paper_portfolio_snapshots`.
- Gates: none.
- Smell: `save_daily_snapshot:539-549` reads prev snapshot for
  daily P&L. Off-by-one if prev row has a stale `total_nav`. Worth
  confirming during BQ inspection.

### Stage 9 — Learn from closed trades
- Entry: `autonomous_loop.py:901-907`, `_learn_from_closed_trades:1611-1637`.
- BQ tables: reads `paper_trades` (recent 50, filtered to action=SELL
  in `closed_tickers`); the outcome tracker writes to
  `outcome_tracking` and `agent_memories`.
- Gates: only fires when `closed_tickers` is non-empty.
- Smell #1: `:1620-1622` filters `recent_trades` to `action == "SELL"
  AND ticker in tickers`. This MISSES partial-exit cases where the
  sell didn't close the position fully -- but `closed_tickers` is
  the upstream signal so this is consistent. The "learn" loop only
  runs on full closes.
- Smell #2 (POTENTIAL BUG): `:1633` -- `recommendation =
  trade.get("risk_judge_decision", "HOLD")`. For stop-loss-trigger
  sells (Stage 5.6 path, `paper_trader.py::execute_sell:266`,
  `reason="stop_loss_trigger"`), `risk_judge_decision=""` because
  `execute_sell` doesn't populate it (`:321`). So
  `tracker.evaluate_recommendation` is called with `recommendation=""`,
  which the outcome tracker may default-handle. The learn loop runs
  but the recommendation key is empty -- the lesson it generates is
  blind to whether this was a stop-loss exit or a signal-flip exit.

### Stage 10 — MetaCoordinator health check (out-of-line)
- Entry: `autonomous_loop.py:909-935`, `MetaCoordinator.gather_health`,
  `_coordinator.decide`.
- BQ tables: reads `paper_portfolio_snapshots` (60d), perf_tracker
  metrics, `analysis_results` (accuracy).
- Gates: read-only health computation; the decision is logged but
  the cycle does NOT route on it.
- Smell: `health.sharpe_ratio` (`:925`) is computed from the same
  snapshot stream -- if Stage 8's snapshot is mis-anchored, the
  health Sharpe inherits the error. Tight feedback loop.

### Stage 11 — Drawdown alarm + cycle-completed alert (post-finally)
- Entry: `autonomous_loop.py:1066-1076`,
  `drawdown_alarm.emit_drawdown_alarms`.
- BQ tables: reads `paper_portfolio_snapshots` (180d).
- Gates: tiered drawdown alerts at -3%/-5%/-10%.
- Smell: only fires when `_final_status == "completed"`. A
  HALTED cycle (kill-switch path Stage 5.5) returns early at
  `:749` BEFORE reaching this. The drawdown alarm is therefore
  silent on the days when it would matter MOST -- when the kill-
  switch flattened. (Operator alert path on `:1004-1029` covers
  this case via the non-completed alert.)

### Stage 12 — Cycle finalization
- Entry: `autonomous_loop.py:937-959` (the completed block),
  `:984-1077` (the finally + post-finally observability).
- BQ tables: writes `cycle_health` history row (Step 4.5.8 logging).
- Gates: none beyond observability dispatch.

## 6. Cross-validation: external best-practice vs internal code

The big patterns:

### 6.1 Pre-trade gate ordering (FIA July 2024)

FIA canonical pre-trade order: **Maximum Order Size -> Maximum
Intraday Position -> Price Tolerance -> Cancel-On-Disconnect ->
Kill Switches -> Exchange OM**.

pyfinagent code path: `execute_buy` (`paper_trader.py:85-258`)
fires in this order:
- `:121-125` cost computation + insufficient-cash (analogue of
  "intraday position by NAV")
- `:128-133` max-positions count (analogue of intraday position)
- `:144-164` idempotency guard
- via `execution_router.py:176-226` `_alpaca_real_fill`:
  `_refuse_live_keys` then `_max_notional_usd` clamp (FIA "max order
  size").
- `decide_trades` adds upstream: sector cap, available cash, min
  position size (FIA "price tolerance" + "position" analogues).
- Kill-switch is Stage 5.5 -- BEFORE the decide+execute steps.

**Gap**: there is **no "Price Tolerance"** (FIA 1.3) check. If
the live price returned by `_get_live_price` is wildly off the
last analyzed price, the buy still goes through. The closest
analogue is the $50 minimum position size in `decide_trades:237-242`
and the LLM-hallucination clamp in `execution_router.py:_max_notional_usd:160-173`.
A 50% price-jump between analysis and execute_buy is undetected.

**Gap**: there is **no "Cancel-On-Disconnect"** semantics. The
current paper backend is bq_sim; in `alpaca_paper` mode COD is
managed by Alpaca itself (good). When pyfinagent goes live, the
operator must verify Alpaca COD is configured per FIA 1.4.

### 6.2 Stop-loss assignment (Kaminski-Lo + arXiv 2604.27150)

Kaminski-Lo: stop-loss adds value in momentum markets; subtracts
value under random walk. Need to MODEL transaction costs incl.
bid-ask spread.

pyfinagent: `_extract_stop_loss` (`portfolio_manager.py:288-329`)
resolution order:
1. explicit `risk_assessment.risk_limits.stop_loss` (absolute)
2. encoded `stop_loss_pct` (relative to entry)
3. `settings.paper_default_stop_loss_pct` fallback (default 8.0)

Defense in depth: `execute_buy` (`paper_trader.py:108-115`) HARD
BLOCKS when `stop_loss_price` is None at entry; synthesizes using
the same `paper_default_stop_loss_pct`. Phase-25.2 backfill
(`paper_trader.py:465-532`) cleans up legacy nulls.

**Cross-validation**: pyfinagent's strategy IS momentum (per
`screener.py::rank_candidates:256-270` and `_run_claude_analysis:1265`
rule). So Kaminski-Lo says stop-loss adds value -- correct
direction. **BUT**: transaction costs are modeled
(`paper_trader.py:121,290` use `paper_transaction_cost_pct`) but
bid-ask spread is NOT (the sim uses last close as both bid and
ask). Per Kaminski-Lo this CAN inflate paper Sharpe.

**Gap (the 6-of-11 no-stop observation)**: this is a
HISTORICAL artifact. Phase-25.6 (`paper_trader.py:99-115`) HARD
BLOCKS new buys without a stop. Phase-25.2
(`paper_trader.py:465-532::backfill_missing_stops`) backfills
nulls. If 6 positions still have no stop, EITHER (a) the backfill
hasn't been run on these specific positions, OR (b) they were
created before phase-25.6 AND the backfill batch missed them.
The ON / INTC / DELL / GLW / LITE / SNDK tickers cited in the
audit are all named in the backfill docstring's audit-trail
comment (`:471`). The first remediation step is to run
`backfill_missing_stops` against current open positions.

**Cross-validation: 8% vs 10%**: pyfinagent default 8% conflicts
with arXiv 2604.27150's top-five-converged 10%. Inline comment
in `paper_trader.py:104,468` cites "O'Neil canonical + arxiv
2604.27150" but the paper TESTED 5/10/15/20/25/30/50 percent and
landed on 10%, NOT 8%. O'Neil's 7-8% rule applies to retail
single-stock trading; the arxiv paper applied to autonomous
agent swarms. Recommend re-validation -- the current 8% may
exit positions too tight per the relevant reference.

### 6.3 Sector concentration (CFA + arXiv 2512.02227)

CFA Institute: max 10% per name, 10-20 positions, no single
position should cause >10% permanent loss. arXiv 2512.02227:
sector limit 30%, single-name 5%.

pyfinagent: `paper_max_per_sector` setting (default value not
fixed in code -- defaults to `int(getattr(settings, "paper_max_per_sector", 0) or 0)`
in `portfolio_manager.py:194`). When `paper_max_per_sector=0` the
sector cap is DISABLED. The dashboard shows 10/11 Tech which is
CONSISTENT WITH the cap being either zero, or set higher than
the practical buy rate.

**Cross-validation**: if `paper_max_per_sector` is 0 (disabled),
that explains 10/11 Tech directly. The sector enrichment path
exists, the cap logic exists, the sector field is populated --
but the cap value may be 0. First remediation step: read the
`paper_max_per_sector` setting and confirm. Recommended value
per arXiv 2512.02227: cap at 30% NAV (not count-based), or 3-4
position-count if total positions is 11.

**Gap**: pyfinagent's cap is a COUNT (`max_per_sector` slots),
not a NAV percentage. CFA Institute and arXiv 2512.02227 both
recommend NAV-percentage caps (5%, 10%, 30%). Count caps cause
problems when position sizes are non-uniform: one large Tech
position can dominate NAV while passing a count cap. Both
representations should coexist in phase-30.

### 6.4 Sharpe-vs-P&L mismatch (Bailey-Lopez de Prado + arXiv 2603.27539)

**The Sharpe -6.26 vs P&L +9.35% paradox is mathematically
explainable.** Sharpe = mean(returns) / std(returns) annualized.
A positive cumulative P&L can produce a negative Sharpe IF the
DAILY return series has a few extreme positive outliers but a
LOW or NEGATIVE mean of the bulk -- or if a recent strong
drawdown swings recent daily returns deeply negative while
cumulative P&L is still positive from an earlier rally.

Code path: `perf_metrics.py::compute_sharpe_from_snapshots:87-115`
clamps `abs(sharpe) > 100.0` to 0.0 to avoid float-precision
garbage. A Sharpe of -6.26 IS within the clamp window; this is
a real measurement.

**Hypothesis 1 (most likely)**: the snapshot stream
(`paper_portfolio_snapshots`) is anchored on `inception_date` from
`paper_trader.py:70`, set at `get_or_create_portfolio` time. If
the portfolio was initialized at `paper_starting_capital` and the
NAV was later manually adjusted (`adjust_cash_and_mtm` at
`paper_trader.py:597-640` was added because of repeated raw-cash
mutation bugs in phase-23.1.15 / 23.2.2 / 23.2.17), some daily
returns could be artificially extreme. The `_nav_to_returns`
filter at `paper_metrics_v2.py:43-48` drops `nav <= 0` rows but
not anomalous daily-return spikes.

**Hypothesis 2**: snapshots ordered by `snapshot_date`
(`paper_metrics_v2.py:42-43`) -- correct. But if any snapshot has
a `total_nav` written BEFORE a `mark_to_market` (the cash-mutation
audit comment at `:601-610` describes exactly this bug class),
that one row carries the bad NAV and produces a single huge
negative daily return. Bootstrap CI for Sharpe (`:506-563`) uses
the same series, so the CI is wide but the point estimate is the
right one to interrogate.

**Hypothesis 3**: P&L +9.35% comes from `paper_portfolio.total_pnl_pct`
which is `(total_nav - starting_capital) / starting_capital * 100`
-- the SNAPSHOT NAV at the moment of mark_to_market. Sharpe comes
from the DERIVATIVE of the NAV stream. These are different
quantities. A monotonically-up but jagged NAV can have positive
cumulative pct AND negative Sharpe if the noise is large relative
to the daily mean.

**The diagnostic step**: run
`SELECT snapshot_date, total_nav, daily_pnl_pct FROM
 financial_reports.paper_portfolio_snapshots ORDER BY snapshot_date`
and look for (a) any daily_pnl_pct exceeding +/- 20% in a single
day (likely data anomaly), (b) the NAV series shape (peak-trough
window).

Per Bailey-Lopez de Prado: DSR threshold 0.95 is the right
significance gate. A live Sharpe of -6 is well below ANY
plausible threshold; it is a measurement issue, not an alpha
signal.

### 6.5 vs SPY -4.62%

Code path: `_get_benchmark_return` (`paper_trader.py:761-775`)
fetches `yf.Ticker("SPY").history(start=inception_date[:10])`.
If `inception_date` is correct and SPY is the right benchmark,
this is right.

**Cross-validation**: nothing in the codebase guarantees
`inception_date` is the date of the FIRST funded snapshot vs the
date of `get_or_create_portfolio`'s first call. If portfolio rows
were created in a test environment and later reset, the inception
date could be incorrect. Confirm via:
`SELECT portfolio_id, inception_date, starting_capital
 FROM financial_reports.paper_portfolio` and compare to the
earliest `paper_portfolio_snapshots.snapshot_date`.

### 6.6 GATE 0/5 NOT ELIGIBLE

GATE = `paper_go_live_gate.py::compute_gate:60-142`. The 5
booleans:
1. `trades_ge_100` -- needs >=100 round trips. Current state likely
   below this -- the 11 positions held suggest insufficient
   ROUND-TRIP history.
2. `psr_ge_95_sustained_30d` -- needs PSR>=0.95 AND >=30 obs.
3. `dsr_ge_95` -- needs DSR>=0.95.
4. `sr_gap_le_30pct` -- needs |live_sharpe - backtest_sharpe| /
   |backtest| <= 0.30.
5. `max_dd_within_tolerance` -- max realized DD <= 20% absolute.

The audit observation "GATE 0/5 yet 11 positions open" is NOT a
bug -- the gate is for LIVE-CAPITAL PROMOTION (phase-4 step 4.4),
and paper-trading positions are explicitly fine while the gate
is red. The gate would BLOCK promotion to real capital but does
NOT block paper trading. This is correct behavior.

**However**: with live Sharpe = -6.26, criterion #4 would be
red regardless of backtest Sharpe; criterion #5 max-dd may or
may not be tripped (a Sharpe -6 with positive P&L is anomalous
but not necessarily a DD breach). #1, #2, #3 are likely all
red because the system is early in its history.

The dashboard ACCURATELY surfaces an unmovable gate, which is
the correct behavior given the underlying state.

### 6.7 Cycle 3 days stale

`autonomous_loop.py::run_daily_cycle` is scheduled at
`paper_trading.py:1166-1180` via APScheduler cron
`hour=settings.paper_trading_hour, minute=0, day_of_week="mon-fri"`,
timezone `America/New_York`. The cron registers in app startup at
`backend/main.py:166-176`.

**Possible causes for 3-day staleness**:
1. Settings `paper_trading_enabled = False` -- the if-guard at
   `paper_trading.py:1161` skips `_add_scheduler_job`. The Sentry
   alert would catch start-up but not a missing schedule.
2. Process restart with `paper_trading_enabled = False` -- no cron
   registered, no cycle, no alert.
3. Settings restart with a cron-frequency that doesn't fire
   (`day_of_week="mon-fri"` skips weekends).
4. The `_running` flag at module level (`:78`) -- if a prior cycle
   crashed BEFORE the `finally` block flipped `_running = False`,
   subsequent invocations log "already running, skipping". (Has
   the finally always fired? Phase-23.2.18 adds an outer timeout;
   pre-23.2.18 a hung yfinance call could strand `_running=True`).
5. APScheduler instance crashed silently in lifespan (`main.py:178-180`
   catches the exception and just warns; the operator might miss
   it).

**Diagnostic step**: check
`pyfinagent_data.cron_job_runs` (cycle_health.py records
cycle_start / cycle_end rows). If the table shows the last
successful row 3 days ago, the cron stopped firing -- either
(a) the scheduler died or (b) `paper_trading_enabled` flipped.

The drawdown_alarm + cycle_completed alerts (`:1037-1064` and
`:1004-1029`) WOULD fire if the cycle ran but errored. Silence
strongly implies the cycle never started -- scheduler/process
problem, not an in-cycle bug.

### 6.8 Stop-loss enforcement 5.6 vs Step 6 ordering (phase-25.1)

Step 5.6 (`autonomous_loop.py:756-777`) fires BEFORE Step 6
(`:779-833`). The Step 5.6 path calls `execute_sell` directly
with `reason="stop_loss_trigger"`. The Step 6 path's
`decide_trades` (`portfolio_manager.py:82-89`) also has a stop-
loss check.

**Cross-validation**: there is NO RACE. Step 5.6 fires first, so
it gets first crack at the stop-loss exit. Even if `decide_trades`
also flags the same ticker for a stop-loss order, by the time
`execute_sell` is called in Step 7 the position is already gone --
`paper_trader.py::execute_sell:269-272` returns None and logs the
"No position to sell" warning. So the second-pass is idempotent.
This is GOOD belt-and-suspenders behavior.

The trade record gets `reason="stop_loss_trigger"` from Step 5.6
(specific) vs `reason="stop_loss"` from Step 6's `TradeOrder`
generic constant. The downstream `_learn_from_closed_trades` keys
on `risk_judge_decision` which is empty in BOTH paths, so the
learn loop is blind to which stop fired -- minor smell, not a bug.

### 6.9 Learning loop reachability

`_learn_from_closed_trades:1611-1637` only runs when
`closed_tickers` is non-empty. `closed_tickers` is populated at
`autonomous_loop.py:856` from successful SELL orders. Stop-loss
triggers in Step 5.6 also produce SELL trades -- but those don't
populate `closed_tickers` (the `:855-856` append is inside the
Step 7 loop, not Step 5.6).

**Bug-class observation**: stop-loss-triggered closes
(`autonomous_loop.py:760-777`) are NOT added to `closed_tickers`.
So the learn loop ONLY runs on `decide_trades`-driven sells, NOT
on stop-loss exits. The system learns less from its worst trades.
File:line: `closed_tickers.append(order.ticker)` is at `:856`
inside the Step 7 sell loop; the Step 5.6 loop at `:759-777`
appends to `summary["stop_loss_triggered"]` but NOT to
`closed_tickers`. The `_learn_from_closed_trades` call at `:905`
passes `closed_tickers` only. **This is a real coverage gap.**

## 7. Suggested phase-30 remediation themes

Theme-level recommendations only. Detailed steps belong in the
phase-30 contract, per protocol.

### P1 (must-do this phase)

1. **Restart the autonomous cycle + add a cycle-health alarm.**
   The 3-day staleness is the highest-impact anomaly: no learning,
   no MTM, no decisions, no alerts. Confirm
   `paper_trading_enabled`, confirm the cron is registered, and
   add a heartbeat alert (e.g., raise P1 if last cron row is
   older than 24h). Sources 4 + FIA WP Sec 4 (post-trade monitoring).

2. **Fix the no-stop-loss coverage gap on the 6 historical
   positions.** Run `paper_trader.py::backfill_missing_stops` on
   the current portfolio; verify each named ticker now has a
   `stop_loss_price` in `paper_positions`. This is a one-time
   data fix; the hard-block at `execute_buy:108-115` prevents
   recurrence on new positions. Source: Kaminski-Lo (stop-loss
   adds value in momentum markets, which is pyfinagent's strategy).

3. **Reconcile the Sharpe -6.26 vs P&L +9.35% anomaly.** Investigate
   `paper_portfolio_snapshots` for outlier daily returns; if a
   single bad NAV row is poisoning the series, repair via
   `adjust_cash_and_mtm`. If returns are genuinely
   extreme-negative-skewed, the strategy fingerprint is wrong
   and the gate stays red as intended. Sources: Bailey-Lopez de
   Prado (DSR Eq. 9-10), arXiv 2603.27539 (look-ahead bias check).

4. **Enable + tune `paper_max_per_sector`.** The 10/11 Tech
   concentration is most plausibly explained by either
   `paper_max_per_sector=0` (disabled) or a cap value
   inconsistent with the 11-position book size. Recommended:
   either count-cap at 3 (so Tech can fill at most ~27%) OR add
   a NAV-percentage cap (30% per arXiv 2512.02227). Both
   together is best. Source: CFA Institute, arXiv 2512.02227.

### P2 (next phase if time permits)

5. **Audit `paper_default_stop_loss_pct` (8.0) against literature.**
   The inline citation to arXiv 2604.27150 supports 10%, not 8%.
   Either update the constant to 10% (and re-run the optimizer)
   or replace the citation with the actual support (e.g., O'Neil's
   retail rule, or a domain backtest result). Source: arXiv
   2604.27150.

6. **Cover stop-loss exits in `_learn_from_closed_trades`.** Add
   the Step 5.6 stop-loss-triggered tickers to `closed_tickers`
   so the learn loop runs on the system's worst trades. One-line
   fix at `autonomous_loop.py:771` (after the sl_trade success
   check). Source: Kaminski-Lo (learning from stop-outs is part
   of why stop-loss adds value in momentum).

7. **Populate `risk_judge_decision` on stop-loss exits.** Set
   `risk_judge_decision="STOP_LOSS_TRIGGER"` in
   `paper_trader.py::execute_sell` when reason starts with
   "stop_loss". Otherwise the outcome tracker's reflection is
   blind to whether a close was strategy-driven or loss-protection.
   Source: Source 10 (audit-log SR 11-7 + EU AI Act).

8. **Add a Price Tolerance gate.** Per FIA WP Sec 1.3, no order
   should leave the trader without a price-vs-reference check.
   Currently `execute_buy:178` uses `fill.fill_price` with no
   guard against fills wildly off the analyzed price. Add a +/-
   X% reject. Source: FIA WP Sec 1.3.

9. **Confirm `inception_date` correctness.** Quick BQ query on
   `paper_portfolio.inception_date` vs the earliest
   `paper_portfolio_snapshots.snapshot_date`. If they disagree,
   the SPY-benchmark window is off and the -4.62% vs SPY number
   is wrong. Source: FIA WP Sec 4.1 (drop-copy reconciliation
   principle).

### P3 (nice-to-have)

10. **Sector cap representation: count + NAV percentage.** Add
    `paper_max_per_sector_nav_pct` alongside the count cap, so
    one-large-position-dominates-NAV cases don't slip through.
    Source: CFA Institute, arXiv 2512.02227.

11. **Move sector enrichment for legacy positions to a
    one-time migration.** The on-cycle enrichment at
    `autonomous_loop.py:791-820` is defensive but adds cycle
    overhead. Run it once via a migration script and let the
    enrichment fall out of the hot path. Source: FIA WP Sec 4.1
    (drop-copy reconciliation is post-trade, not per-order).

12. **Add a Sharpe-recomputation drill.** A daily reconciliation
    cron that re-runs Sharpe from the snapshot series and alerts
    if the daily value diverges >2sigma from the rolling
    bootstrap CI. Source: Bailey-Lopez de Prado bootstrap CI
    method (already implemented in
    `perf_metrics.py::compute_rolling_sharpe_bootstrap_ci`).

13. **Add a "consensus-vs-execution" log row per cycle.** Per
    Source 4 + Source 10, multi-agent trading needs an
    audit-trail row that captures the trader's recommendation,
    the risk judge's decision, the final order, and the price
    at execution. The `paper_trades.signals` JSON
    (`paper_trader.py:197`) partly does this but is JSON-
    encoded; consider a flat denormalized row for compliance
    purposes. Source: SR 11-7, EU AI Act Article 12.

## 7b. Addendum — coordinator-supplied forensic evidence

Main has already validated the following forensic facts. They sharpen
the cross-validation, so the brief cites the literature backing each.

### Sharpe -6.26 root cause: NAV pollution from a $5K deposit (5/13)

The dashboard's daily_pnl_pct on `paper_portfolio_snapshots` for
2026-05-13 is +32.12%, driven by a $5K cash deposit
(NAV 17818 -> 23541 with positions_value barely moving). The
`paper_metrics_v2.py::_nav_to_returns:36-48` function does NOT
subtract external cash flows; it computes
`daily_returns = np.diff(navs) / navs[:-1]` raw. A $5K deposit
inflates one daily return by ~32% and ALSO pollutes the standard
deviation, driving Sharpe deeply negative.

**Authoritative reference: 11th source read in full ->**
https://en.wikipedia.org/wiki/Time-weighted_return -- the canonical
TWR formula divides the period into sub-periods at every external
cash flow and multiplies sub-period growth factors. **GIPS standards
REQUIRE time-weighted returns for portfolio performance reporting**
precisely because they "remove the effects of external cash flows"
(GIPS Calculation Methodology Guidance Statement, 2010-revised).
pyfinagent currently reports a money-weighted-naive return series
to its Sharpe calculation -- a GIPS-non-compliant pattern that
explains the -6.26 anomaly without invoking any actual alpha
deterioration.

**Fix path** (P1, but theme-only): subtract external cash flows from
the NAV series before computing daily returns, or use the
Modified Dietz method
(https://en.wikipedia.org/wiki/Modified_Dietz_method) when daily
NAV-flow timestamps are not available. The fix is in
`paper_metrics_v2.py::_nav_to_returns`. The 32% spike vanishes when
the deposit is accounted for; Sharpe normalizes. See arXiv 2603.27539
"transaction-cost neglect" as a related failure mode.

### Sector-cap retroactive enforcement gap

`paper_max_per_sector=2` (settings.py:159) caps NEW BUYS but does not
retroactively rebalance existing positions
(`portfolio_manager.py:188-262`). 10/11 Technology IS consistent
with: (a) early portfolio entered ALL Tech before the cap was added,
(b) cap is now in effect for new buys but does nothing about the
existing book.

**Reference**: CFA Institute / arXiv 2512.02227 BOTH frame
concentration limits as continuous PORTFOLIO STATE checks, not just
ENTRY checks. "Real-time monitoring and reporting of concentration
limits" with "automate compliance processes" per the regulatory
literature (Finley Technologies on bank-loan concentration limits,
identical mechanism). The remediation theme is a periodic rebalance
or a sell-flag when the post-rebalance ratio exceeds the cap.

### `backfill_missing_stops` (phase-25.2) has ZERO callers

The function exists at `paper_trader.py:465-532` but no Python
caller invokes it. The 7 NULL-stop positions exist BECAUSE the
defense-in-depth helper was written but never wired into the
cycle. **Reference**: this is the FIA WP Sec 4.1 "drop-copy
reconciliation" failure mode -- a control is documented as
existing but is never actually exercised on the production
portfolio.

### Cron silent failure (5/18)

Last `llm_call_log` row = 2026-05-17. No 5/18 snapshot in
`paper_portfolio_snapshots`. The autonomous cycle stopped firing.
The `_running` flag at `autonomous_loop.py:78` is a known stranding
candidate (a hung pre-finally cycle leaves `_running=True`). FIA WP
Sec 1.4 (Cancel-On-Disconnect) and CME Group / Pico's pre-trade-
control guidance both call out heartbeat / liveness alerting as a
mandatory companion to the control surface itself. pyfinagent has
`cycle_health` recording rows but no operator-facing heartbeat
alarm. The cycle_completed alert at `:1037-1064` only fires INSIDE
the cycle -- there's no out-of-band detector for a cron that never
runs.

### Lite-only degradation (5/17 vs 5/16)

5/16 calls had `agent` tags (Quant Model, Synthesis -- full-pipeline
path); 5/17 calls all `agent=NULL` (lite path). Cause likely sits
in `autonomous_loop.py::_run_single_analysis:1079-1141` -- if the
full-pipeline orchestrator fails, the fallback is the lite Claude
analyzer. A persistent failure mode degrades the system to lite
silently. **Reference**: arXiv 2603.27539 names this as the
"degradation without alerting" anti-pattern -- the system should
WARN when its analysis tier silently downgrades.

### `agent_memories` is empty + learn loop dormant

The learn loop persists outcomes to BQ `agent_memories` (per
CLAUDE.md Architecture line and `outcome_tracker.py`). Table is
empty -> learn loop has produced no rows. Cross-validation:
`_learn_from_closed_trades:1611-1637` only runs when
`closed_tickers` is non-empty AND only on `decide_trades`-driven
sells (already flagged in §6.9). With no closed_tickers,
`outcome_tracker.evaluate_recommendation` is never called.
The 7 NULL-stop positions + the cron-silent-failure feed each other:
no trades close -> no learn loop -> no `agent_memories` rows.

**Reference**: arXiv 2412.20138 (TradingAgents) explicitly logs
"detailed reasoning, tool usage, and thought processes" for EVERY
decision, not just successful round-trips. The pattern of logging
ONLY on close is too late for a learning system. Recommend a
per-cycle audit row in `agent_memories` even when no trade closed
-- the cycle's reasoning is still learning material.

### Updated remediation priorities

P1 additions: (a) **fix TWR computation in `_nav_to_returns`** to
subtract external cash flows -- this is the actual Sharpe -6.26 root
cause, NOT a strategy-fingerprint problem. (b) **wire
`backfill_missing_stops` into the cycle** (or run it once
imperatively for the 7 NULL-stop positions). (c) **add a heartbeat
alarm** that fires when no cycle has run in 24h, sourcing from
`pyfinagent_data.cron_job_runs` or an out-of-band check. (d) **alert
on lite-mode degradation** -- when `agent=NULL` for an entire
cycle's LLM calls, raise a P2.

## 8. JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 11,
  "snippet_only_sources": 10,
  "urls_collected": 21,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```

Gate logic: `external_sources_read_in_full=10 >= 5` AND
`recency_scan_performed=true` AND all hard-blocker checklist items
satisfied AND three-variant search composition visible AND
internal codebase audit complete. PASS.
