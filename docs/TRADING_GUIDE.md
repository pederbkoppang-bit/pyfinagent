# How to Trade pyfinAgent Signals

> A practical guide for Peder -- the sole human trader executing pyfinAgent signals.
> Read this end-to-end before the first live trading day.

---

## 1. What pyfinAgent Is (and Is Not)

pyfinAgent is an autonomous AI-powered trading signal system. It screens the US equity universe daily, analyzes candidates through a multi-agent pipeline (28 analysis agents + 6 orchestration agents), and publishes actionable BUY, SELL, or HOLD signals to Slack.

**What it does:**

- Screens the market daily for candidates
- Runs deep fundamental + technical + sentiment analysis per ticker
- Publishes signals with confidence scores, position sizes, and stop-loss prices
- Tracks paper trading performance against backtest expectations
- Sends morning and evening digests to Slack

**What it does NOT do:**

- Execute real trades. You -- Peder -- are the human-in-the-loop who places every order at your broker.
- Guarantee returns. The system is probabilistic. It will be wrong sometimes.
- Replace your judgment. You always have the final say on whether to act on a signal.

---

## 2. Signal Anatomy

When pyfinAgent publishes a signal, you will see a Slack message in your signals channel with this layout:

**Header:** Ticker symbol + action (e.g., "AAPL BUY")

**Fields section:**

| Field | What it means |
|-------|---------------|
| **Confidence** | A score from 0.00 to 1.00 reflecting the AI's assessed probability that this trade will be profitable. Higher is stronger conviction. See Section 3 for thresholds. |
| **Price** | The price at which the paper trader booked the position (or the current market price at signal time). Use this as your reference -- your actual fill may differ slightly. |
| **Size** | The recommended position size in USD. This is computed by the hybrid sizing formula (see Section 4). It is a recommendation, not a mandate. |
| **Stop** | The stop-loss price. If the stock falls to or below this price, the system will emit a SELL. See Section 5 for how stop-losses work. "N/A" means no explicit stop was set for this signal. |

**Thesis:** A brief text explaining why the AI reached this recommendation. Read this to sanity-check the reasoning -- if the thesis mentions data you know to be stale or wrong, consider skipping the signal.

**Footer:** Contains the signal date, a truncated signal_id (a unique identifier for deduplication), and the timestamp of generation.

**Actions by signal type:**

- **BUY** -- The system recommends opening a new position. Review the thesis, check the size, and decide whether to place the order at your broker.
- **SELL** -- The system recommends closing an existing position. This can be triggered by a sell signal, a signal downgrade (was BUY, now HOLD/SELL), or a stop-loss hit. Act promptly -- the reason field tells you why.
- **HOLD** -- No action required. The system re-evaluated a holding and decided the current position should be maintained. You do not need to do anything.

---

## 3. Confidence Thresholds

Confidence is a number between 0.00 and 1.00. Think of it as the AI's self-assessed conviction:

| Range | Interpretation | Suggested action |
|-------|---------------|-----------------|
| 0.80 - 1.00 | High conviction | Strong candidate. Review thesis and act if you agree. |
| 0.60 - 0.79 | Moderate conviction | Decent candidate. Extra scrutiny recommended -- check if you have independent reasons to agree or disagree. |
| 0.40 - 0.59 | Low conviction | Weak signal. Consider skipping unless you have strong independent conviction. |
| 0.00 - 0.39 | Very low conviction | Very uncertain. Generally skip. The AI is not confident in this call. |

**How confidence affects position sizing:** The half-Kelly arm of the sizing formula (see Section 4) is directly proportional to confidence. A 0.80 confidence signal gets sized at roughly double a 0.40 confidence signal, all else equal. The system self-limits its bets when it is less sure.

---

## 4. Position Sizing

The system uses a hybrid three-arm formula to recommend position sizes. The final size is the **minimum** of all applicable arms -- the most conservative wins.

### Arm (a): Hard percent cap

The strict upper bound. A single position cannot exceed the smaller of:
- 5% of total portfolio equity, OR
- $1,000 USD

This is the defense-in-depth ceiling. It fires regardless of how confident the signal is.

### Arm (b): Half-Kelly (confidence-weighted)

Position size = 0.5 x confidence x equity

Half-Kelly captures roughly 75% of theoretical optimal growth at roughly 50% of the variance. A 0.75 confidence signal on a $10,000 portfolio would size at $3,750 before the hard cap clamps it. In practice the hard cap almost always binds, keeping individual positions conservative.

### Arm (c): Inverse-volatility

When the signal includes annualized volatility data, the system sizes inversely to volatility: more volatile stocks get smaller positions. This decouples sizing from the AI's edge estimate quality.

### Fallback

If the Risk Judge agent does not specify a position percentage, the portfolio manager defaults to 10% of NAV -- still subject to the hard cap.

### What this means for you

The "Size" field on the Slack signal is the system's recommended USD amount. You can:
- Follow it exactly
- Scale it up or down based on your personal risk tolerance
- Skip the signal entirely

The system never sizes a position larger than 5% of equity or $1,000 USD in any case.

---

## 5. Stop-Loss Execution

### How the stop works

The system sets a stop-loss price when opening a position. During each daily cycle, it checks:

> If the current price is **at or below** the stop-loss price, emit a SELL.

This is an inclusive boundary -- a stock priced exactly at the stop triggers the exit. The stop takes precedence over everything: even if a fresh analysis recommends BUY, the stop-loss SELL fires first.

### The backtest stop-loss parameter

The optimizer's best parameters use a stop-loss of approximately 12.9% (sl_pct = 12.92). This means for a stock bought at $100, the stop-loss price would be set around $87.08.

### What you should do

When you see a SELL signal with reason "stop_loss":
1. Act promptly. The stop fired because the position crossed the loss threshold.
2. Place a market sell order (or limit sell near current price) at your broker.
3. Do not re-enter the position on the same day. Wait for a fresh BUY signal.

### Other sell reasons

- **"sell_signal"** -- The AI's latest analysis explicitly recommends selling.
- **"signal_downgrade"** -- The holding was previously a BUY; the latest re-evaluation downgraded it to HOLD or SELL.

---

## 6. Risk Limits

pyfinAgent enforces four hard risk limits. These are coded directly into the system as literals -- they cannot be changed without a code deployment. This is intentional: it prevents accidental relaxation during a live incident.

| Limit | Value | What it does |
|-------|-------|-------------|
| Per-ticker concentration | 10% of portfolio | No single stock can exceed 10% of total portfolio value. Blocks oversized BUYs. |
| Total exposure | 100% of portfolio | The portfolio cannot be leveraged. Total positions cannot exceed 100% of equity. |
| Drawdown kill switch | -15% | If the portfolio draws down 15% from peak, ALL new BUYs are blocked. Only SELLs are allowed. This is the circuit breaker. |
| Daily trade cap | 5 trades per day | No more than 5 trades in a single day. Prevents overtrading. |

If you see a signal that seems to violate these limits (e.g., a BUY when you know the portfolio is near -15% drawdown), the system's risk_check should have already blocked it. If a BUY slips through despite a known risk limit breach, do NOT execute it -- report it in #ford-approvals.

---

## 7. When to Override Ford

You are not obligated to follow every signal. Here are concrete scenarios where you should consider overriding:

**Skip a BUY when:**
- You know an earnings announcement is imminent (within 1-2 days) and want to avoid the volatility
- The market has hit a circuit breaker or there is an extraordinary macro event (e.g., Fed emergency rate decision)
- You already have external exposure to the same sector or stock outside this portfolio
- The position size would push your personal risk tolerance past your comfort zone
- The thesis mentions data points you know to be incorrect or outdated

**Skip a SELL when:**
- You have material non-public information about a positive catalyst (though you should not be trading on MNPI)
- The stop-loss triggered on a temporary intraday dip and the stock has already recovered by the time you see the signal
- In general, be more cautious about overriding SELLs than BUYs -- the system tends to be right about cutting losses

**Escalate when:**
- You see a signal that contradicts the system's own prior signal for the same ticker within the same day
- The Slack bot posts "Missing API key" errors repeatedly (the analysis path may be degraded)
- You notice the morning or evening digest has not arrived for 2+ consecutive days

**How to escalate:** Post in #ford-approvals with a brief description of the issue. Tag the signal_id if relevant. Ford (the autonomous agent) monitors this channel and will investigate.

---

## 8. Daily Workflow

### Morning (before market open, 08:00-09:30 ET)

1. **Check the morning digest** in your Slack signals channel. It shows the overnight portfolio P&L and any analyses that ran.
2. **Review any pending BUY signals.** Read the thesis, check the confidence and size. Decide whether to act.
3. **Place orders at your broker** for any signals you choose to follow. Use limit orders near the signal's reference price for BUYs. Use market orders for urgent SELLs (especially stop-loss triggered).

### During market hours (09:30-16:00 ET)

4. **Monitor for intraday signals.** The system may publish signals during the trading day if the autonomous loop runs mid-day. These follow the same rules.
5. **No action needed on HOLD signals.** They are informational only.

### Evening (after market close)

6. **Check the evening digest.** It summarizes the day's trades, portfolio P&L, and any position changes.
7. **Reconcile.** Compare the paper trading positions shown in the digest with your actual broker positions. Any discrepancy should be flagged in #ford-approvals.

### Weekly

8. **Review the accuracy report.** The system publishes a weekly signal accuracy summary showing hit rate, Wilson confidence interval, and per-group breakdowns. Use this to calibrate your trust in the system over time.

---

## 9. Key Numbers to Remember

| Parameter | Value | Source |
|-----------|-------|--------|
| Backtest Sharpe | 1.17 | optimizer_best.json |
| Deflated Sharpe Ratio (DSR) | 0.95 | optimizer_best.json |
| Paper trading Sharpe floor | 0.82 | 70% of backtest Sharpe |
| Take-profit target | 10.0% | optimizer_best.json (tp_pct) |
| Stop-loss | ~12.9% | optimizer_best.json (sl_pct) |
| Holding period | 90 days | optimizer_best.json (holding_days) |
| Max positions | 10 (paper) / 20 (backtest) | Settings |
| Starting capital (paper) | $10,000 | Settings |
| Min cash reserve | 5% of NAV | Settings |
| Rollback trigger | Live Sharpe < 0.5 for 14 days | Rollback plan |

---

## 10. Important References

- **Rollback Plan:** If live performance degrades, the rollback procedure is documented in `docs/ROLLBACK_PLAN.md`. Familiarize yourself with the stop-signals command before going live.
- **Go-Live Checklist:** All 27 pre-launch items are tracked in `docs/GO_LIVE_CHECKLIST.md`. Every item must be checked before the first live signal.
- **Slack Channels:**
  - Your signals channel -- where BUY/SELL/HOLD alerts and daily digests arrive
  - #ford-approvals -- where Ford posts status updates and where you escalate issues or post your go-live approval
  - #all-pyfinagent -- general project channel

---

## 11. Disclaimer

pyfinAgent is a decision-support tool, not a financial advisor. Past performance (including backtest results and paper trading metrics) does not predict future results. The system can and will produce losing trades. You are solely responsible for every trade you place at your broker.

By using pyfinAgent signals:
- You acknowledge that all trading decisions are yours
- You accept that the system is probabilistic and fallible
- You understand that the backtest Sharpe of 1.17 is a historical metric computed under specific conditions and may not persist in live markets
- You commit to following the rollback plan if live performance degrades

---

## Appendix: System Architecture (Optional Reading)

The signal pipeline works as follows:

1. **Screening:** The system screens the US equity universe daily using free data sources (yfinance). Top candidates are ranked by a composite score.

2. **Analysis:** Each candidate goes through a 15-step analysis pipeline with 28 specialized agents covering fundamental analysis, technical analysis, sentiment analysis, sector-specific research, and risk assessment.

3. **Debate:** Bull and bear agents debate the thesis. A moderator synthesizes the final recommendation.

4. **Risk assessment:** A Risk Judge agent evaluates position sizing and sets the stop-loss price.

5. **Signal publication:** The final signal (BUY/SELL/HOLD) passes through validation, deduplication, and risk_check before reaching Slack.

6. **Paper trading:** The signal is simultaneously booked to the paper trader for performance tracking.

7. **Re-evaluation:** Existing holdings are re-evaluated periodically. Downgrades trigger SELL signals.

The system processes sells before buys each cycle to free up capital before allocating to new positions.

For technical details, see `docs/ARCHITECTURE.md`.
