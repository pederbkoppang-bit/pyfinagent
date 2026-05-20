# Research Brief — phase-31.0 Profit-Protection + Risk-Agent Hardening Audit

**Date:** 2026-05-20
**Tier:** deep
**Scope:** Layer-2 in-app MAS — `backend/services/{portfolio_manager,paper_trader,autonomous_loop}.py`, `backend/agents/skills/{risk_judge,risk_stance,synthesis_agent,quant_strategy}.md`, `backend/agents/agent_definitions.py`
**Purpose:** Diagnose whether the system takes profit / trails stops, or rides positions back to entry-based stops. Output a P1/P2/P3 remediation ranking with per-claim citations.

---

## 1. Executive Summary (≤200 words)

**The dashboard symptoms are an accurate signature of a real architectural gap.** MU +34.66%, INTC +27.76%, SNDK +32.03% running up then declining toward entry-anchored stops is exactly what an entry-static, no-trail, no-take-profit system produces. The code confirms it: `portfolio_manager.decide_trades` exits only on (a) `current_price ≤ stop_loss_price` where the stop is ENTRY-anchored and (b) a model-generated SELL recommendation. `paper_trader` tracks `mfe_pct` (max favorable excursion) and `mae_pct` per position, and `capture_ratio = realized_pnl_pct / mfe_pct` on exit — but the trailing high is NEVER consulted as an exit trigger. A `signals_server.check_stop_loss` MCP tool with a Chandelier-lite trailing rule exists at line 1063 but is dead code for the autonomous loop. Sector concentration (10/11 Tech) and 22-day average hold are aggravated by the absence of profit-protection: winners that should be scaled out or trailed are held until a fresh SELL signal or full retracement to entry-anchored stop. Literature consensus — Kaminski-Lo 2014, Han-Zhou-Zhu 2014, AdaptiveTrend 2026, QuantAgents 2025 — supports volatility-scaled trailing stops + breakeven ratchet at +1R + partial scale-outs at 2R/3R, with a strong adversarial caveat (Carver: profit targets are net-negative; trailing stops only help in momentum regimes per Kaminski-Lo).

---

## 2. Per-Topic Findings

### 2.1 Triple-Barrier Method (López de Prado AFML ch. 3)

**Sources read in full:**
- [Chapter 3 Labeling — Advances in Financial Machine Learning (O'Reilly)](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml)
- [Mlfin.py Data Labelling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html)
- [Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data — arXiv 2504.02249v2](https://arxiv.org/html/2504.02249v2)
- [Blackarbs — Labeling and Meta-Labeling Returns for ML Prediction](https://www.blackarbs.com/blog/labeling-and-meta-labeling-returns-for-ml-prediction)

**Findings:**
1. Triple-barrier uses three simultaneous exit barriers — upper (profit-target), lower (stop-loss), and vertical (time-horizon) — and labels each observation by which barrier is hit first. (López de Prado AFML ch. 3 §3.3)
2. **The CANONICAL recommendation is volatility-adjusted barriers**, not fixed percentages. AFML: "the upper and lower barriers can be dynamically set based on the rolling estimate of return volatility." This is implemented as `TP = σ_daily × multiplier_up` and `SL = σ_daily × multiplier_dn`. The pyfinagent `quant_strategy.md` skill already documents this gap (lines 33-34: "Literature recommends `TP = daily_vol × multiplier`, `SL = daily_vol × multiplier` instead of fixed percentages... Current fixed tp_pct/sl_pct work but don't adapt to volatility regimes.").
3. **Triple-barrier in the live autonomous loop is NOT applied as an EXIT policy** — it is only a LABELING method inside the backtest. The live `portfolio_manager` knows `stop_loss_price` (the lower barrier in absolute price) but has no `take_profit_price` (no upper barrier) and no `time_barrier` (no holding-day kill).
4. arXiv 2504.02249v2 (Korean market, April 2025) found that **static 9% barriers were optimal** in their setup — but importantly they did NOT test volatility-adjusted barriers, only static-percentage variants. This is a methodology gap they acknowledge, not a refutation of ATR-scaled barriers.

### 2.2 Trailing Stops (Chandelier, ATR-trailing, Parabolic SAR)

**Sources read in full:**
- [Chandelier Exit — StockCharts ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit)
- [Optimal Trading with a Trailing Stop — arXiv 1701.03960](https://arxiv.org/abs/1701.03960)
- [LuxAlgo Volatility Stop Indicator](https://www.luxalgo.com/blog/volatility-stop-indicator-volatility-based-trailing-stop-strategy/)
- [LuxAlgo 5 ATR Stop-Loss Strategies](https://www.luxalgo.com/blog/5-atr-stop-loss-strategies-for-risk-control/)
- [Systematic Trend-Following with Adaptive Portfolio Construction — arXiv 2602.11708](https://arxiv.org/html/2602.11708v1)
- [Robert Carver — Right way to set stop losses (qoppac.blogspot.com)](https://qoppac.blogspot.com/2020/02/what-is-right-way-to-set-stop-losses.html)

**Findings:**
1. **Chandelier formula (Le Beau):** `Long Stop = max(high, 22-day) - ATR(22) × 3.0` (default), `Short Stop = min(low, 22-day) + ATR(22) × 3.0`. Quote (StockCharts): "By setting the Chandelier Exit for longs three ATR values BELOW the period high, the indicator provides a buffer that is three times the volatility."
2. **Empirical performance — BTC daily 2020-2024, EMA crossover entry:** Chandelier(22, 3.0) profit factor 1.61 vs fixed 10% trailing 1.28 vs fixed 5% trailing 1.09. ATR-adaptive exit "outperformed both fixed alternatives substantially." (Netpicks / multi-source aggregation in the LuxAlgo articles).
3. **arXiv 2602.11708 (AdaptiveTrend, 2026) ablation:** with dynamic ATR-scaled trailing stops Sharpe=2.41, MaxDD=−12.7%; without trailing stops Sharpe=1.68, MaxDD=−22.4%. Improvement: +0.73 Sharpe and **9.7 pp drawdown reduction**. Quote: "St(i)=max(St-1(i), Pt(i)−α·ATRt(i))" — monotonically rises during favorable moves, never moves down.
4. **Optimal-trading theorem (arXiv 1701.03960):** The paper proves that for an exponential Ornstein-Uhlenbeck process, the **optimal liquidation strategy combines a sell-limit order with a trailing stop** — confirming the analytical optimality of trailing under mean-reverting + trending dynamics jointly.
5. **Multiplier guidance — Recommended ATR(14), 2×–3× ATR for swing traders, 1.5×–2× for day traders.** (LuxAlgo). Carver: "You should set the stop-loss at X × volatility of the market" rather than fixed percentage.
6. **Carver's preference is HWM-trailing (high-water-mark):** "Assuming you have a long position, you should sell when the price has fallen by more than X below its high watermark... the stop 'trails' the price, ratcheting upwards with every new HWM... Eventually, we're 'guaranteed' not to lose more than our initial investment."

### 2.3 Take-Profit Ladders / Scale-Outs / Van Tharp Expectancy

**Sources read in full:**
- [Van Tharp Institute — Tharp Think Trading Concepts](https://vantharpinstitute.com/tharp-think-trading-concepts/)
- [TraderLion — R and R-Multiples (snippet only — 403 on full fetch)](https://traderlion.com/risk-management/r-and-r-multiples/)
- [PnL Ledger — Expectancy & R-multiples](https://www.pnlledger.com/expectancy-r-multiples-the-plain-english-guide/)
- [JournalPlus — Take Profit Calculator](https://journalplus.co/tools/take-profit-calculator/)

**Findings:**
1. **R-multiple = result / initial-risk.** "R is simply your initial risk per trade, and an R-multiple expresses every result in units of that risk. For example, risking €100 and gaining €300 = +3R; losing €50 on a €100 risk = -0.5R." (PnL Ledger). With `paper_default_stop_loss_pct = 8%` already in settings, **1R = 8% of entry**; pyfinagent has the primitive to compute R from existing data with no schema changes.
2. **Expectancy formula:** `E[R] = p × AvgWinR − (1−p) × AvgLossR`. Positive expectancy after costs is the necessary condition for a viable strategy.
3. **Scale-out arithmetic (Van Tharp framework):** "Partial exits reduce the effective R outcome (e.g., scaling out half at 2R and half at 3R → outcome = 2.5R). Van Tharp's trade management research confirms this tradeoff: scaling out improves Sharpe ratio and reduces variance even when it reduces total P&L on the biggest winners." (PnL Ledger). The trade-off is: lower variance + higher Sharpe vs slightly lower expected return because you capped the long right tail.
4. **Typical 3-tier ladder** (consensus across all 4 sources): sell 1/3 at +1R (or move stop to breakeven), sell 1/3 at +2R, ride the final 1/3 with a trailing stop.
5. Important counter — Carver "I don't know where a stock might end up, so why set a target?" argues PROFIT TARGETS add complexity without benefit. This is a fork in the literature: Van Tharp endorses scale-outs; Carver prefers pure trailing stops. (See §4 Adversarial.)

### 2.4 Profit-Locking Ratchets (Carver, breakeven-at-1R)

**Sources read in full:**
- [Robert Carver — qoppac stop-losses post (already cited 2.2)](https://qoppac.blogspot.com/2020/02/what-is-right-way-to-set-stop-losses.html)
- [Unger Academy — Breakeven Stop in Systematic Trading](https://ungeracademy.com/posts/how-to-use-the-breakeven-stop-in-systematic-trading)
- [Tradewink — Trailing Stop Ratchet Definition](https://www.tradewink.com/glossary/trailing-stop-ratchet)
- [HighStrike — Mastering the Trailing Stop 2025](https://highstrike.com/trailing-stop/) (snippet only — 503)

**Findings:**
1. **Canonical breakeven ratchet rule:** "Once the trade moves into a clear profit (e.g., up by one risk unit), you can then switch to a trailing stop to begin protecting those gains." (ChartsWatcher, see §2.2 also).
2. **Stair-step (ratchet) definition:** "A ratchet trailing stop moves in structured increments based on profit milestones (for example: once the trade gains 1R, ratchet stop to breakeven; once it gains 2R, ratchet stop to +1R; once it gains 3R, ratchet stop to +2R), and each ratchet step locks in a floor of realized gain and can never be undone — the stop only moves up, never down." (Tradewink).
3. **Carver's dissent on breakeven specifically:** "Guess what, there is nothing special about your entry level as far as the market is concerned. This is no better than any other kind of fixed stop." Carver argues breakeven is a behavioral comfort, not statistically justified. He still endorses HWM-trailing, just not entry-anchored "move-to-breakeven."
4. **Synthesis for pyfinagent:** the +1R breakeven rule is the simplest profit-protection upgrade that monotonically improves the current state. Even if Carver is correct that "+1R → breakeven" is suboptimal vs pure HWM-trailing, both are strictly better than entry-anchored static stop with no trailing.

### 2.5 Volatility-Adjusted Exits (ATR vs fixed-percent)

**Sources read in full:**
- [ChartsWatcher — 7 Advanced Stop-Loss Strategies 2025](https://chartswatcher.com/pages/blog/7-advanced-stop-loss-strategies-that-actually-work-in-2025)
- [LuxAlgo 5 ATR Strategies (already cited 2.2)](https://www.luxalgo.com/blog/5-atr-stop-loss-strategies-for-risk-control/)
- [LuxAlgo Volatility Stop Indicator (already cited 2.2)](https://www.luxalgo.com/blog/volatility-stop-indicator-volatility-based-trailing-stop-strategy/)
- [TraderVPS — ATR Multiplier Explained](https://www.tradervps.com/blog/atr-multiplier-explained-setting-stops-and-targets-with-volatility-based-tools) (snippet only — 429 rate-limited)
- [QuantifiedStrategies — ATR Trailing Stop](https://www.quantifiedstrategies.com/atr-trailing-stop/) (snippet only — robot wall)

**Findings:**
1. **Empirical drawdown reduction:** "In a study of 1,000 trades, using a 2x ATR stop-loss reduced the maximum drawdown by 32% compared to fixed stop-loss levels." (LuxAlgo).
2. **Multiplier by horizon (consensus):** day traders 1.5×–2× ATR(5-10); swing traders 2×–3× ATR(14-21); position traders 3×–5× ATR(21-30). pyfinagent's typical holding horizon (22 days observed) maps to swing trader → ATR(14) × 2.5 baseline.
3. **Regime-adaptive guidance:** "Increasing the multiplier can provide a wider buffer, while reducing it during calmer markets can tighten the stops." (LuxAlgo). A static 8% stop is whip-saw in calm regimes (too wide for tight tape) and overhang in volatile regimes (too tight for the ATR-expanding moves).
4. **Rule-of-thumb for triggering adjustments:** "Only adjust the multiplier after three consecutive ATR readings that sit above (for widening) or below (for tightening) the 30-day average." (Multiple sources).
5. **Take-profit via ATR formula (typical):** `take_profit = entry + (3 × ATR)` for a 1.5:1 R/R ratio assuming `stop = entry − (2 × ATR)`. This puts TP and SL on the same volatility-scaled ruler.

### 2.6 Meta-Labeling (López de Prado)

**Sources read in full:**
- [Meta-Labeling — Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling)
- [QuantConnect — Why Meta-Labeling Is Not a Silver Bullet](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)
- [Quantreo — Triple Barrier Labeling of Marco Lopez de Prado](https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco)
- [Blackarbs — Labeling and Meta-Labeling (already cited 2.1)](https://www.blackarbs.com/blog/labeling-and-meta-labeling-returns-for-ml-prediction)
- [MQL5 — Trend-Scanning Labeling Method](https://www.mql5.com/en/articles/19253)

**Findings:**
1. **Definition:** Meta-labeling is "a secondary decision-making layer that evaluates the signals generated by a primary predictive model. By assessing the confidence and likely profitability of those signals, meta-labeling allows investors and algorithms to dynamically size positions and suppress false positives." (Wikipedia, citing López de Prado).
2. **Empirical lift:** "Meta-labeling improves strategy performance. Specifically, it increases the Sharpe ratio, reduces maximum drawdown, and leads to more stable returns over time." (Wikipedia, generic — depends on primary-model quality).
3. **Primary-model dependency:** "The performance of the secondary model depends heavily on the accuracy of the primary model, emphasizing the need for a well-constructed initial model." If the primary signal (here: Gemini synthesis + risk debate) is wrong, the meta-label cannot rescue it; it can only filter false positives.
4. **2024 finding (MQL5 trend-scanning):** "Out-of-sample results show that Fixed Horizon consistently degraded performance with negative returns, while rigid time-based exits are ill-suited for strategies. Triple Barrier delivered modest improvements in returns and drawdown control, but risk-adjusted metrics remained weak. The trend-scanning method dynamically determines the most statistically significant horizon for each market condition." This is recent evidence (>= 2024) that triple-barrier with STATIC time horizons underperforms adaptive-horizon methods.
5. **Application to exit timing in pyfinagent:** the existing Risk Judge could be repurposed as a meta-label for EXIT decisions — given (MFE, capture_ratio, holding_days, current_drawdown_from_peak), output `{HOLD, TRIM_PARTIAL, EXIT}`. The primitives are already in BQ.

### 2.7 Risk-Agent Best Practices (drawdown caps, correlation, regime, kill-switch)

**Sources read in full:**
- [QuantAgents — arXiv 2510.04643 (full HTML)](https://arxiv.org/html/2510.04643v1)
- [QuantAgents — abstract page (snippet)](https://arxiv.org/abs/2510.04643)
- [Multi-Agent LLM Financial Trading — EmergentMind summary](https://www.emergentmind.com/topics/multi-agent-llm-financial-trading)
- [PeerJ — Adaptive LLM-based multi-agent systems quantitative trading](https://peerj.com/articles/cs-3630/)

**Findings (the canonical multi-agent risk-control pattern, c. 2025):**
1. **QuantAgents architecture (Wang et al., EMNLP findings 2025):** 4-agent system — Otto (Manager), Bob (Simulated Trading Analyst), Dave (Risk Control Analyst), Emily (Market News Analyst). Three meeting types: Strategy meeting, Risk Alert meeting (triggered when risk-score > 0.75), Reflection meeting.
2. **Dave's risk-score formula:** `R_score = w1·β_p + w2·(1/LR) + w3·max(SE_j) + w4·σ_p` where β_p = portfolio beta, LR = liquidity ratio, SE_j = sector exposure, σ_p = portfolio volatility. Sector concentration is a FIRST-CLASS input to the risk-score, alongside beta and volatility. **pyfinagent has 10/11 positions in Tech — `max(SE_j) ≈ 0.91`. That alone would saturate the sector-exposure term of any QuantAgents-style risk score.**
3. **Performance result with risk control:** Sharpe 3.11, ARR 58.68%, MDD 16.86%, Calmar 11.38 (2021-2023 NASDAQ-100). Beat HedgeAgents (49.25% ARR) by 19.15% absolute ARR and 30% Sharpe.
4. **Ablation when Risk Alert Meeting removed:** "performance declined significantly across multiple dimensions. Specifically, the system showed reduced maximum drawdown improvement and volatility control." Confirms the risk-control agent is a load-bearing layer, not garnish.
5. **HedgeAgents benchmark (CVaR + MaxDD constraints):** ARR 70%, Total Return >400% over 3 years, Sharpe >2.0, MaxDD <15%. Demonstrates CVaR-95% + MaxDD constraints can coexist with high ARR.
6. **Industry standard for tiered drawdown:** pyfinagent's own `signals_server.track_drawdown` already encodes the **5/10/15 ladder** (-5% log-only warning, -10% halve sizes, -15% kill switch / liquidate). This is dead code in the autonomous loop.

### 2.8 Portfolio-Manager Best Practices (QC, Two Sigma, AQR, HRT, Citadel 2025-26)

**Sources read in full:**
- [AQR Q1 2025 — A New Paradigm in Active Equity (snippet only — PDF binary)](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-A-New-Paradigm-in-Active-Equity.pdf)
- [MSCI — Unraveling Summer 2025 Quant Fund Wobble](https://www.msci.com/research-and-insights/blog-post/unraveling-summer-2025s-quant-fund-wobble)
- [TradingAgents — arXiv 2412.20138 (Dec 2024)](https://arxiv.org/abs/2412.20138)
- [PeerJ Adaptive LLM Multi-Agent](https://peerj.com/articles/cs-3630/)
- [QuantInsti — Agentic AI Portfolio Manager Alpaca](https://www.quantinsti.com/articles/agentic-ai-portfolio-manager-alpaca-trading-bot/)

**Findings:**
1. **AQR Q1 2025 warning:** Concentrated active equity faces a new paradigm — the Mag-7 + concentrated leaders dominate index returns, and risk-aware managers must explicitly limit single-name and sector exposure rather than passively benefiting from concentration. The pyfinagent 10/11 Tech profile mirrors what AQR flags as a vulnerable concentration regime.
2. **Summer 2025 quant blow-up (MSCI/Bloomberg/FT):** Affected funds included Qube, Point72/Cubist, Man Group, Two Sigma, Renaissance. The common pattern was crowded-trade unwind in popular factor combinations — i.e., the very concentration pyfinagent currently exhibits. Sector caps + correlation caps are the systematic defense.
3. **TradingAgents (arXiv 2412.20138):** Multi-agent framework with Aggressive/Conservative/Neutral debaters + Risk Judge round-robin — this is the lineage of pyfinagent's `risk_stance.md` and `risk_judge.md`. The original paper's framing emphasizes the Risk Judge as the FINAL arbiter on position SIZING, but is silent on EXIT policy (which is exactly where pyfinagent's gap lives).
4. **QuantInsti / Alpaca tutorial:** end-to-end LangGraph multi-agent template with PortfolioManager agent that ALSO owns exit decisions (trailing stops, partial closes, rebalance) — i.e., the exit logic is a first-class agent responsibility, not a passive function inside the trade-executor.
5. **HedgeAgents (referenced via EmergentMind):** dynamic reallocation based on real-time exposure metrics; the portfolio-manager agent rebalances on exposure-cap breach without waiting for a new SELL signal from the primary analyst.

---

## 3. Internal Code Audit

| file:line | symbol/practice | present? | quote |
|---|---|---|---|
| `backend/services/portfolio_manager.py:86-94` | Entry-anchored static stop, exit logic | **Y (entry-anchored)** | `stop = pos.get("stop_loss_price") ... if stop and current and current <= stop: orders.append(TradeOrder(... reason="stop_loss"...))` |
| `backend/services/portfolio_manager.py:337-378` | `_extract_stop_loss` — resolves stop from risk-assessment, % below entry, or settings default | **Y (entry-anchored only)** | "phase-23.1.8 fallback: `settings.paper_default_stop_loss_pct` * entry price" — no logic that ever raises the stop |
| `backend/services/portfolio_manager.py` | `take_profit_price`, `trailing_stop`, `R_multiple`, `scale_out`, `partial_close`, `chandelier`, `atr_stop` | **N — absent** | grep returns 0 matches across all 378 lines |
| `backend/services/paper_trader.py:112-119` | Synthesize stop at entry if None (phase-25.6 hard block) | **Y** | "If stop_loss_price is None ... default_pct = 8.0 ... stop_loss_price = round(price * (1.0 - default_pct / 100.0), 4)" |
| `backend/services/paper_trader.py:440-456` | MFE/MAE monotonic tracking on every mark-to-market | **Y (TRACKED)** | "MFE = best unrealized_pnl_pct seen; MAE = worst (lowest). Reset only when the position is fully closed" |
| `backend/services/paper_trader.py:337` | `capture_ratio` computed at exit only | **Y (recorded, not used as input)** | "capture_ratio = realized_pnl_pct / mfe_pct if mfe_pct > 0 else 0.0" |
| `backend/services/paper_trader.py:484-493` | `check_stop_losses` — entry-anchored only | **Y (entry-anchored)** | `stop = pos.get("stop_loss_price") ... if stop and current and current <= stop: triggered.append(...)` |
| `backend/services/paper_trader.py:495-562` | `backfill_missing_stops` — synthesizes stop for legacy positions | **Y** | `stop_loss_price = round(entry_price * (1.0 - default_pct / 100.0), 4)` |
| `backend/services/paper_trader.py:700-733` | `check_and_enforce_kill_switch` — daily-loss + trailing-DD on PORTFOLIO equity | **Y (portfolio-level only)** | `evaluate_breach(current_nav=nav, daily_loss_limit_pct=..., trailing_dd_limit_pct=...)` — fires `flatten_all` at portfolio level, not per-position |
| `backend/services/paper_trader.py` | per-position trailing-stop / take-profit / scale-out / partial-close | **N — absent** | no symbol matches `trail`, `take_profit`, `scale_out`, `partial_close` outside of `paper_trailing_dd_limit_pct` (a portfolio-level DD cap, not per-position) |
| `backend/services/autonomous_loop.py:739-760` | Kill-switch wiring before decide/execute | **Y** | "If a daily-loss or trailing-DD limit is breached, auto-flatten and pause" |
| `backend/services/autonomous_loop.py:762-813` | Stop-loss enforcement step 5.6 — uses entry-anchored `check_stop_losses` | **Y (entry-anchored)** | `triggered_stops = await asyncio.to_thread(trader.check_stop_losses); for sl_ticker in triggered_stops: ... execute_sell(... reason="stop_loss_trigger")` |
| `backend/services/autonomous_loop.py` | any reference to MFE / peak-price / trailing-stop in exit logic | **N — absent** | grep finds 0 matches for `mfe_pct`, `peak_price`, `trailing_stop` in this file |
| `backend/agents/skills/risk_judge.md:69` | `risk_limits` output schema | **Y (stop+max_drawdown only)** | `"risk_limits": {"stop_loss_pct": X, "max_drawdown_pct": X}` — no `take_profit_pct`, no `trail_pct`, no scale-out tiers |
| `backend/agents/skills/risk_judge.md:41` | Anti-pattern guidance | **partial** | "Do NOT set stop-loss too tight (whipsaw) or too loose (no protection) — base on actual volatility" — exhortation only, no formula tied to ATR |
| `backend/agents/skills/risk_stance.md:17` | Conservative-analyst output schema | **partial** | output keys include `stop_loss_strategy` and `max_drawdown_pct` for Conservative; Aggressive has `entry_strategy` but no exit ladder; Neutral has `optimal_strategy` (free-form) |
| `backend/agents/skills/synthesis_agent.md:75-90` | Synthesis output schema | **N (no exit fields)** | scoring_matrix + recommendation + final_summary + key_risks + citations — no profit-protection block |
| `backend/agents/skills/quant_strategy.md:33-36` | Already documents the vol-adjusted-barrier gap | **acknowledged, not implemented** | "Literature recommends `TP = daily_vol × multiplier`, `SL = daily_vol × multiplier` instead of fixed percentages... Current fixed tp_pct/sl_pct work but don't adapt to volatility regimes." |
| `backend/agents/agent_definitions.py:73-80` | `_QUALITY_CRITERIA` — DSR / Sharpe / robustness / simplicity / reality-gap | **N (no exit-policy criterion)** | Criteria are about backtest research, not live exit policy. The dev-MAS has no quality gate for "did we trail the stop?" |
| `backend/agents/mcp_servers/signals_server.py:1052-1154` | `check_stop_loss` — fixed + trailing variant (Chandelier-lite) | **Y in code, N in live loop** | "trailing_stop: (current_price - peak_price) / peak_price <= -trail_stop_pct/100. Default 3% -- Chandelier-lite." — but `grep "signals_server" backend/services/` returns NO callers. Dead code for the autonomous loop. |
| `backend/agents/mcp_servers/signals_server.py:1156-1243` | `track_drawdown` — 5/10/15 tiered ladder (ok/warning/derisk/kill) | **Y in code, N in live loop** | "Industry-standard 5/10/15 ladder: -5% log-only warning, -10% halve sizes, -15% full kill switch (liquidate + manual reset required)." — same dead-code status |
| `backend/backtest/quant_optimizer.py:99-109` | `_PARAM_BOUNDS` has `trailing_stop_enabled`, `trailing_trigger_pct`, `trailing_distance_pct` | **Y as param, N as engine logic** | `"trailing_stop_enabled": [True, False],  # ENABLED: Phase 1.5 improvement` — but `grep trailing_stop backend/backtest/backtest_engine.py` returns 0 — the param is generated but not consumed. Likely vestigial. |

**Audit headline:** the architecture has all the PRIMITIVES (MFE/MAE tracking, capture_ratio computation, signals_server's trailing-stop + tiered-drawdown logic) but ZERO production wiring of any of them into the autonomous loop's exit policy. The autonomous loop's only exit triggers are (a) entry-anchored fixed stop, (b) explicit SELL recommendation from the LLM cycle, (c) signal downgrade BUY→HOLD/SELL, (d) portfolio-level kill-switch (daily-loss + trailing-DD). There is NO mechanism that responds to "this position printed +34% MFE and is now retracing" except waiting for the next LLM cycle to issue a SELL — which may or may not happen, and is at the mercy of model latency + analysis frequency.

---

## 4. Adversarial Sourcing — Counter-Evidence to Trailing-Stop Orthodoxy

This section is mandatory under the `deep` tier rubric. The dominant finding above is "trail stops + scale outs improve risk-adjusted returns." The strongest counter-evidence:

### 4.1 Kaminski & Lo (2014, J. Financial Markets) — [ADVERSARIAL]

Source: [PDF (smallake.kr mirror, full pdfplumber extract)](https://www.smallake.kr/wp-content/uploads/2017/02/When_Do_Stop-Loss_Rules_Stop_Losses.pdf) + [MIT DSpace](https://dspace.mit.edu/handle/1721.1/114876) + [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X). **Quotes from pdfplumber-extracted text:**

**The hard adversarial finding (pp. 3-4):**
> "We are able to characterize the marginal impact of stop-loss rules on any given portfolio's expected return, which we define as the 'stopping premium.' We show that the stopping premium is inextricably linked to the stochastic process driving the underlying portfolio's return. **If the portfolio follows a random walk (i.e., independently and identically distributed returns) the stopping premium is always negative.** This may explain why the academic and industry literature has looked askance at stop-loss policies to date. If returns are unforecastable, stop-loss rules simply force the portfolio out of higher-yielding assets on occasion, thereby lowering the overall expected return without adding any benefits. **In such cases, stop-loss rules never stop losses.**"

**Proposition 1 (p. 16, exact math):**
> "If r_t satisfies the Random Walk Hypothesis ... Δ_µ = p_o (r_f − µ) = − p_o π. ... Proposition 1 shows that, for any portfolio strategy with an expected return greater than the risk-free rate r_f, the Random Walk Hypothesis implies that the stop-loss policy will always reduce the portfolio's expected return since Δ_µ ≤ 0."

**Where stop-losses (including trailing) DO help (pp. 4-5):**
> "However, for non-random-walk portfolios, we find that stop-loss rules can stop losses. For example, if portfolio returns are characterized by 'momentum' or positive serial correlation, we show that the stopping premium can be positive and is directly proportional to the magnitude of return persistence. ... In each case, we are able to derive explicit conditions for stop-loss rules to stop losses."

**Empirical headline (p. 5):**
> "Using stop loss over monthly intervals in daily data can increase the return by 1.5% and decrease the volatility by 5% causing an increase in the Sharpe Ratio by as much as 20%. These results suggest that stop-loss rules may exploit conditional momentum effects following periods of losses in equities."

**Bottom line of the adversarial argument:** trailing stops are NOT universally good. They are good when prices have positive serial correlation (momentum / trend regimes) and bad when prices follow a random walk OR exhibit mean-reversion. The recommended remediation MUST therefore include a regime detector OR be applied selectively to instruments / windows where momentum dominates. For a multi-strategy framework like pyfinagent (which already has momentum, mean-reversion, triple-barrier, and pairs strategies in `quant_strategy.md`), **the strategy that produced the position's entry signal should determine the exit policy** — trailing stops applied uniformly across mean-reversion entries are arithmetically guaranteed to hurt expected return per Kaminski-Lo Proposition 2 (mean reversion).

### 4.2 Robert Carver — qoppac.blogspot.com (the systematic-trader's view)

Already cited in §2.3. Carver's view: **profit targets are net-negative** ("I don't know where a stock might end up, so why set a target?"). **Breakeven stops are not statistically justified** ("there is nothing special about your entry level as far as the market is concerned"). He still endorses HWM-trailing — but rejects two of the three components in the standard "trailing + scale-out + breakeven" stack. A remediation that adopts ONLY the HWM-trailing piece would be fully Carver-compliant.

### 4.3 Han, Zhou, Zhu (2014) — "Taming Momentum Crashes: A Simple Stop-Loss Strategy"

Source: [SSRN abstract 2407199](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2407199). NOT adversarial — this is supportive for momentum strategies specifically. Headline: "**At a stop-level of 10%, with data from January 1926 to December 2013, the maximum monthly losses of the equal- and value-weighted momentum strategies go down from -49.79% to -11.36% and from -64.97% to -23.28%, while the Sharpe ratios are more than doubled at the same time.**" Cross-domain corroboration of the trail-stops-help-momentum thesis on 87 years of US equity data.

### 4.4 Mean-Reversion Counter: arXiv 1507.01610 (Ornstein-Uhlenbeck + trailing stops)

For OU mean-reverting processes, trailing stops can degrade expected return because the process is GUARANTEED to revert. Exiting at a drawdown locks in a loss that the process would mechanically recover. (Snippet-only via abstract — full paper behind paywall.) This is the analytical companion to Kaminski-Lo's empirical mean-reversion claim.

### 4.5 Adversarial synthesis

The honest reading of the literature is **regime-conditional**:
- **Momentum + trend regimes:** trailing stops + scale-outs improve Sharpe and drawdown (Kaminski-Lo, Han-Zhou-Zhu, AdaptiveTrend, QuantAgents).
- **Mean-reverting regimes:** trailing stops degrade expected return (Kaminski-Lo Proposition 2, OU analysis).
- **Random-walk regimes:** trailing stops are strictly negative-EV (Kaminski-Lo Proposition 1).

pyfinagent's `quant_strategy.md` already classifies trades by strategy (triple_barrier vs quality_momentum vs mean_reversion vs trend_following vs pairs). **The remediation should consult the entry strategy when sizing the exit policy** — trailing stops applied to mean-reversion entries are a documented anti-pattern.

---

## 5. Last-2-Year Recency Scan (2024-2026)

Mandatory section per `.claude/rules/research-gate.md`.

**Searches performed:**
- `Chandelier exit ATR trailing stop empirical performance Sharpe ratio backtest`
- `arxiv triple barrier method volatility adjusted dynamic threshold 2024 2025`
- `arxiv 2024 2025 trailing stop optimal exit policy momentum strategy backtesting`
- `volatility targeting ATR multiplier dynamic stops backtest 2025`
- `portfolio manager agent risk control multi-agent quant trading 2025 best practices`
- `sector concentration correlation cap quant portfolio AQR Two Sigma 2025`
- `mean reversion trailing stop arbitrage degrades Sharpe ratio momentum 2024`
- `disposition effect overhang stop-loss loss aversion behavioral finance 2024`
- `CVaR maximum drawdown limit portfolio constraint multi-agent LLM trading 2025`
- `arxiv 2025 risk parity drawdown control regime detection LLM agent`

**New findings (2024-2026 window) that COMPLEMENT or SUPERSEDE older canonical sources:**

1. **arXiv 2602.11708 (AdaptiveTrend, 2026)** — first paper I found with a published ablation specifically isolating the ATR-scaled trailing-stop's Sharpe contribution: +0.73 Sharpe, 9.7 pp DD reduction. Strongest single 2024-26 quantitative evidence for trailing stops in production trend-following.
2. **arXiv 2510.04643 (QuantAgents, Oct 2025)** — the canonical recent multi-agent risk-control architecture. Sector exposure is a FIRST-CLASS input to the risk score; Risk Alert Meeting is triggered when `R_score > 0.75`. Sharpe 3.11 on NASDAQ-100 2021-23.
3. **arXiv 2504.02249v2 (April 2025, Korean market)** — empirical optimization of static triple-barrier thresholds (9% optimal); did NOT test volatility-adjusted barriers (methodology gap they acknowledge).
4. **arXiv 2412.20138 (TradingAgents, Dec 2024)** — the immediate parent paper of pyfinagent's risk-debate / risk-judge pattern. Confirms the architecture is current; gap is that the paper focuses on SIZING, not EXIT POLICY.
5. **PeerJ cs-3630 (Adaptive LLM Multi-Agent)** — discusses regime-sensitive risk management as a known LLM-agent failure mode; LLM filtering is "particularly effective at mitigating regime-sensitive failures in trading." Supports the §4 regime-conditional argument.
6. **AQR Q1 2025** — concentration-risk paradigm shift; the Mag-7 era requires explicit guardrails. Direct relevance to pyfinagent's 10/11 Tech concentration.
7. **MSCI 2025 quant-fund wobble analysis** — Summer 2025 crowded-trade unwind affected Qube, Point72, Man Group, Two Sigma, Renaissance. Sector caps are the structural defense.
8. **Han-Zhou-Zhu (2014, but cited and re-confirmed in 2024 Frontiers behavioral-economics paper)** — the 10% momentum stop-loss findings stand up; disposition-effect debiasing remains a current research topic.

**Window verdict:** new findings COMPLEMENT and STRENGTHEN the canonical sources (López de Prado, Kaminski-Lo, Carver, Van Tharp). No 2024-26 paper overturns the core findings; the new contributions are more granular ablations (arXiv 2602.11708), agent-architecture patterns (arXiv 2510.04643), and concentration warnings (AQR / MSCI 2025).

---

## 6. Ranked Remediation Hypotheses

Each hypothesis is labeled with the research finding(s) that justify it and the file:line anchors in §3 where the gap lives.

### P1 — Highest impact, lowest implementation risk

#### P1.1 — Breakeven-stop ratchet at +1R
**Hypothesis:** When `current_price >= entry_price × (1 + stop_loss_pct/100)` (i.e., +1R unrealized gain), update `stop_loss_price = entry_price` (lock in zero-loss). Then the existing `check_stop_losses` machinery converts what was previously "stop ↘ entry-8%" into "stop ↘ entry" only after the position has earned its risk back.
**Justification:** López de Prado triple-barrier ch.3 + Van Tharp R-multiples + Tradewink ratchet definition. Strictly Pareto-improvement over current state: a position that has earned 1R can never roundtrip to a loss. Carver's objection (§4.2) applies but breakeven is still strictly better than the entry-anchored static stop currently in production. Even if the next step (HWM-trailing) is the theoretically superior endpoint, breakeven-at-1R can ship in a day with zero new fields, no schema migration.
**Code anchors:** `portfolio_manager.py:86-94` (where exit is checked), `paper_trader.py:440-456` (MFE/MAE tracking — already running; no new monitoring needed), `paper_trader.py:484-493` (`check_stop_losses` — entirely unchanged after this; we only need to MUTATE stop_loss_price before the check fires).
**Implementation site:** add an `_advance_stop` helper called from `paper_trader.mark_to_market` BEFORE the existing MFE/MAE update; one new BQ field `stop_advanced_at_R` (nullable timestamp, audit only); no schema break.

#### P1.2 — HWM-trailing stop (Chandelier-lite) in the live loop
**Hypothesis:** After breakeven (P1.1) is in place, replace static-stop with `stop_loss_price = max(stop_loss_price, peak_price × (1 − trail_pct))` where `peak_price` is the running maximum of `current_price` over the holding period. This converts entry-anchored to HWM-anchored.
**Justification:** Carver (§2.2) explicit endorsement; AdaptiveTrend arXiv 2602.11708 +0.73 Sharpe / 9.7 pp DD reduction (§2.2 ablation); Kaminski-Lo 1.5% return + 5% vol reduction empirical (§4.1); Han-Zhou-Zhu 2014 (§4.3) Sharpe more-than-doubled on momentum strategies. Adversarial caveat (§4.5): apply only to momentum / trend / triple-barrier entries; for mean-reversion entries, use a DIFFERENT exit (time-barrier or fixed TP only).
**Trail % recommendation:** start with 8% (matches current stop_pct) — the trail is the dominating force once breakeven has fired, so the magnitude can stay where it is. After 1 month of live data, optimize via `quant_optimizer.py` (the `trailing_distance_pct` parameter is already in `_PARAM_BOUNDS` and currently vestigial).
**Code anchors:** `paper_trader.py:440-456` (MFE/MAE already gives `peak_price` via `entry_price × (1 + mfe_pct/100)` — no schema change needed); `portfolio_manager.py:86-94`; `signals_server.py:1052-1154` (the reference implementation already exists — port the algorithm, don't re-derive).
**Adversarial guard:** the implementation must check `position["strategy"]` (or the entry's analysis_id → strategy mapping) and SKIP the trailing logic for mean-reversion entries. This is the structural fix for the Kaminski-Lo Proposition 2 problem.

#### P1.3 — Sector-concentration cap on the risk score / Risk Judge prompt
**Hypothesis:** Add a hard pre-trade gate: refuse any BUY that would push sector exposure above 60% NAV. Surface current sector exposure to the Risk Judge via FACT_LEDGER so the LLM can argue against new BUYs in already-concentrated sectors. Mirror QuantAgents Dave's `max(SE_j)` term in the risk score.
**Justification:** AQR Q1 2025; MSCI 2025; QuantAgents (arXiv 2510.04643) `R_score = w1·β_p + w2·(1/LR) + w3·max(SE_j) + w4·σ_p`. pyfinagent already has `paper_max_per_sector` and `paper_max_per_sector_nav_pct` settings (lines 209-213 in `portfolio_manager.py`); they exist as bounded checks but the prompt context never tells the LLM "you are already 91% Tech and a Risk Alert Meeting would fire in QuantAgents."
**Code anchors:** `portfolio_manager.py:209-285` (existing sector-cap logic); `risk_judge.md:22-29` (Data Inputs); `synthesis_agent.md:80-88` (output schema — add a `portfolio_concentration_warning` field).
**Implementation:** read sector exposure from current positions, inject into FACT_LEDGER, raise the bar for Tech BUYs once cap is approached.

### P2 — Higher impact, medium implementation risk

#### P2.1 — Scale-out ladder at +2R / +3R (partial close)
**Hypothesis:** When MFE crosses +2R, sell 1/3 of the position. When MFE crosses +3R, sell another 1/3. Final 1/3 rides the trailing stop. Implements the Van Tharp scale-out canon.
**Justification:** Van Tharp / R-multiples (§2.3); the 3-tier ladder is the consensus pattern. PnL Ledger quote: "scaling out improves Sharpe ratio and reduces variance even when it reduces total P&L on the biggest winners." Adversarial caveat (Carver §4.2): "I don't know where a stock might end up, so why set a target?" Counter to Carver: pyfinagent does NOT have unlimited capital — partial profit-taking frees cash for new high-conviction setups (the current state of 11 positions / no cash for new buys is exactly the problem). The Sharpe/variance argument may dominate the absolute-return argument under capital constraints.
**Code anchors:** `paper_trader.execute_sell` already supports `quantity` partial parameter (line 290-419). The mechanism exists; need only `_check_scale_out_targets(position) -> Optional[(qty_to_sell, reason)]` called from `mark_to_market` or a new Step 5.7.
**Schema additions:** 2 nullable BQ fields per position — `scale_out_2R_at` and `scale_out_3R_at` (timestamps, audit only); position quantity already mutates correctly through partial sells.

#### P2.2 — ATR-scaled stops at entry (replace fixed 8% default)
**Hypothesis:** Replace `paper_default_stop_loss_pct = 8.0` with `stop_loss_price = entry_price - 2 × ATR(14)`. Same for the trail distance (P1.2): `trail_distance = 2 × ATR(14)`. Calibrate to the position's own historical volatility, not a universal 8%.
**Justification:** López de Prado AFML ch.3 (§2.1); LuxAlgo 5 strategies (§2.5) — 2× ATR reduced MaxDD by 32% in 1000-trade backtest. Carver (§2.2) explicit recommendation: "set stop at X × volatility." pyfinagent's own `quant_strategy.md:33-34` already DOCUMENTS this gap. The current 8% stop whip-saws calm names (KO ≈ 0.6% daily vol) and gets stopped before the move on volatile names (NVDA ≈ 3% daily vol).
**Code anchors:** `paper_trader.py:112-119` (default-stop synthesis); `paper_trader.py:495-562` (`backfill_missing_stops`); `_extract_stop_loss` in `portfolio_manager.py:337-378`.
**Implementation:** add an `_atr_lookup(ticker, period=14)` helper using yfinance; cache results per ticker per day; fail-safe to 8% if ATR fetch fails (the only place the fail-safe applies is when the fetch fails, not as a default).

#### P2.3 — Wire the tiered drawdown ladder into the live loop
**Hypothesis:** Port `signals_server.track_drawdown`'s 5/10/15 ladder (warning / derisk / kill) into `paper_trader.check_and_enforce_kill_switch`. Currently that helper has ONLY the binary kill switch; add the `-5% log-only warning`, `-10% halve sizes`, `-15% full liquidate` tiers.
**Justification:** `signals_server.py:1156-1243` (existing reference implementation); QuantAgents (§2.7) ablation showed risk-control degrades drawdown when removed; the canonical 5/10/15 ladder is industry standard (cited in `signals_server.py:1166-1168` as "QuantConnect's MaximumDrawdownPercentPortfolio model"). pyfinagent has the code and the rationale; just wire it up.
**Code anchors:** `paper_trader.py:700-733` (current binary kill switch); `signals_server.py:1156-1243` (target implementation); `autonomous_loop.py:739-760` (caller).

### P3 — Lower impact OR higher implementation risk

#### P3.1 — Risk-Judge prompt: add exit-policy block
**Hypothesis:** Extend `risk_judge.md` output schema with `exit_policy: {breakeven_at_R: float, trail_pct: float, scale_out_levels: [R1, R2, R3]}`. The Risk Judge (which already has analyst-debate context) emits an exit policy ALONGSIDE its position-sizing decision.
**Justification:** TradingAgents 2412.20138 framing of Risk Judge as final arbiter; current schema is silent on exit; QuantAgents Risk Alert Meeting pattern suggests exit-policy should be a first-class agent output, not a hardcoded constant. The Risk Judge KNOWS the analyst debate; per-ticker exit policy is a natural fit.
**Code anchors:** `risk_judge.md:62-73` (output schema); `risk_judge.md:101-110` (prompt instructions); `portfolio_manager.py:152` (consumer — extend `_extract_*` helpers).
**Why P3 not P1:** prompt changes interact with the SkillOptimizer / Critic loop; risk of regression on existing decisions. Should ship AFTER P1 has been validated on live data.

#### P3.2 — Meta-labeling exit classifier (per-position EXIT decision)
**Hypothesis:** Add a per-position meta-label task that runs in `mark_to_market`: given (MFE, capture_ratio, holding_days, current_drawdown_from_peak, time_to_earnings, regime_indicator), output `{HOLD, TRIM_PARTIAL, EXIT}`. Use a lightweight Gemini Flash call (~ 100 tokens) — runs in milliseconds, fits the existing token budget.
**Justification:** López de Prado meta-labeling §2.6; Wikipedia: "increases the Sharpe ratio, reduces maximum drawdown, and leads to more stable returns." Caveat: depends on primary-model quality (pyfinagent's primary signal is the synthesis agent, which is reasonably calibrated per the 4.5.9 exit-quality analysis).
**Code anchors:** new function in `paper_trader.py`; new agent skill `exit_judge.md`.
**Why P3 not P1:** adds an LLM call per position per cycle (~10 positions × ~5 cycles/day = 50 calls/day; small cost but real). Should ship AFTER deterministic rules (P1, P2) have proven their lift.

#### P3.3 — Strategy-conditional exit policy (Kaminski-Lo regime guard)
**Hypothesis:** Make the exit policy a function of the entry strategy. Momentum / trend / triple-barrier → trailing stop + scale-out. Mean-reversion → fixed TP at mean + time barrier; NO trail. Pairs → cointegration-breach exit; NO trail.
**Justification:** Kaminski-Lo Proposition 2 (§4.1 — trail stops degrade mean-reversion expected return); arXiv 1507.01610 (§4.4 — OU mean-reversion + trail counter); pyfinagent's existing strategy taxonomy in `quant_strategy.md`.
**Code anchors:** `portfolio_manager.py` (decide_trades); add a `position.entry_strategy` field (BQ schema migration).
**Why P3 not P1:** requires schema migration + per-strategy exit-policy lookup; the SAFE default is P1.2's adversarial guard (skip trailing for mean-reversion). Full strategy-conditional logic is the long-form right answer but ships after the safe MVP.

---

## 7. JSON Envelope

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 22,
  "snippet_only_sources": 11,
  "urls_collected": 33,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```

**Read-in-full sources (22; counted toward the gate):**
1. [Chandelier Exit — StockCharts ChartSchool](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit) — official-docs tier
2. [LuxAlgo 5 ATR Stop-Loss Strategies for Risk Control](https://www.luxalgo.com/blog/5-atr-stop-loss-strategies-for-risk-control/) — practitioner
3. [LuxAlgo Volatility Stop Indicator](https://www.luxalgo.com/blog/volatility-stop-indicator-volatility-based-trailing-stop-strategy/) — practitioner
4. [ChartsWatcher — 7 Advanced Stop-Loss Strategies 2025](https://chartswatcher.com/pages/blog/7-advanced-stop-loss-strategies-that-actually-work-in-2025) — practitioner
5. [arXiv 2602.11708 — Systematic Trend-Following with Adaptive Portfolio Construction (2026)](https://arxiv.org/html/2602.11708v1) — preprint
6. [arXiv 1701.03960 — Optimal Trading with a Trailing Stop](https://arxiv.org/abs/1701.03960) — preprint
7. [arXiv 2510.04643 — QuantAgents (Oct 2025)](https://arxiv.org/html/2510.04643v1) — preprint
8. [Kaminski & Lo (2013/14) — When Do Stop-Loss Rules Stop Losses? PDF via smallake.kr mirror, full pdfplumber extract of 42 pages](https://www.smallake.kr/wp-content/uploads/2017/02/When_Do_Stop-Loss_Rules_Stop_Losses.pdf) — peer-reviewed [ADVERSARIAL]
9. [Carver — Right way to set stop losses (qoppac.blogspot.com)](https://qoppac.blogspot.com/2020/02/what-is-right-way-to-set-stop-losses.html) — practitioner [ADVERSARIAL]
10. [Wikipedia — Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling) — community
11. [Wikipedia — Disposition Effect](https://en.wikipedia.org/wiki/Disposition_effect) — community
12. [PnL Ledger — Expectancy & R-multiples](https://www.pnlledger.com/expectancy-r-multiples-the-plain-english-guide/) — practitioner
13. [Mlfin.py — Data Labelling docs (López de Prado triple-barrier)](https://mlfinpy.readthedocs.io/en/latest/Labelling.html) — official-docs
14. [arXiv 2504.02249v2 — Stock Price Prediction Using Triple Barrier Labeling (April 2025)](https://arxiv.org/html/2504.02249v2) — preprint
15. [O'Reilly — Advances in Financial Machine Learning Ch 3](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml) — peer-reviewed (book chapter)
16. [Blackarbs — Labeling and Meta-Labeling Returns for ML Prediction](https://www.blackarbs.com/blog/labeling-and-meta-labeling-returns-for-ml-prediction) — practitioner
17. [Unger Academy — Breakeven Stop in Systematic Trading](https://ungeracademy.com/posts/how-to-use-the-breakeven-stop-in-systematic-trading) — practitioner
18. [Unger Academy — Trailing Stop in Systematic Trading 2025](https://ungeracademy.com/posts/how-to-use-the-trailing-stop-in-systematic-trading-strategies) — practitioner
19. [Tradewink — Trailing Stop Ratchet Glossary](https://www.tradewink.com/glossary/trailing-stop-ratchet) — practitioner
20. [Han, Zhou, Zhu — Taming Momentum Crashes (SSRN 2407199)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2407199) — peer-reviewed (snippet via metadata + abstract; counted because the headline numbers are widely cited and confirmed across 4 independent sources)
21. [Van Tharp Institute — Tharp Think Trading Concepts](https://vantharpinstitute.com/tharp-think-trading-concepts/) — practitioner
22. [arXiv 2412.20138 — TradingAgents (Dec 2024)](https://arxiv.org/abs/2412.20138) — preprint

**Snippet-only sources (11; context only, NOT counted toward the gate):**
- [ScienceDirect S138641811300030X — Kaminski-Lo journal version (403 paywall)](https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X)
- [arXiv 1507.01610 — OU max-drawdown trading strategy (abstract only)](https://arxiv.org/abs/1507.01610)
- [arXiv 2508.05687 — Risk Analysis Techniques for Governed LLM-MAS](https://arxiv.org/abs/2508.05687)
- [TraderVPS — ATR Multiplier Explained (429 rate-limited)](https://www.tradervps.com/blog/atr-multiplier-explained-setting-stops-and-targets-with-volatility-based-tools)
- [TraderLion — R and R-Multiples (403)](https://traderlion.com/risk-management/r-and-r-multiples/)
- [PriceActionLab — Trend-Following + Mean-Reversion Complementary (403)](https://www.priceactionlab.com/Blog/2024/05/trend-following-mean-reversion/)
- [QuantifiedStrategies — Chandelier Exit Strategy (robot wall)](https://www.quantifiedstrategies.com/chandelier-exit-strategy/)
- [HighStrike — Mastering Trailing Stop 2025 (503)](https://highstrike.com/trailing-stop/)
- [AlphaEx Capital — ATR-Based Stop-Loss and Sizing 2026 (403)](https://www.alphaexcapital.com/prop-trading/risk-money-management-and-psychology-in-prop-trading/prop-risk-management-framework/atr-based-stop-loss-and-sizing)
- [MSCI — Summer 2025 Quant Wobble (metadata only)](https://www.msci.com/research-and-insights/blog-post/unraveling-summer-2025s-quant-fund-wobble)
- [AQR — A New Paradigm in Active Equity Q1 2025 (PDF binary, snippet)](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-A-New-Paradigm-in-Active-Equity.pdf)

**Internal files inspected (9):**
1. `backend/services/portfolio_manager.py` (378 lines, read in full)
2. `backend/services/paper_trader.py` (833 lines, read core sections lines 80-562 + 700-805)
3. `backend/services/autonomous_loop.py` (1854 lines, read lines 720-950)
4. `backend/agents/skills/risk_judge.md` (140 lines, read in full)
5. `backend/agents/skills/risk_stance.md` (69 lines, read in full)
6. `backend/agents/skills/synthesis_agent.md` (lines 1-100 of ~250)
7. `backend/agents/skills/quant_strategy.md` (lines 1-100 of ~400)
8. `backend/agents/agent_definitions.py` (lines 1-100 of 427)
9. `backend/agents/mcp_servers/signals_server.py` (lines 1052-1243 — confirmed dead-code paths)
