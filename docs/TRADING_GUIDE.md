# How to Trade pyfinAgent Signals

> A standalone guide for Peder. Covers signal anatomy, confidence interpretation,
> position sizing, stop-loss execution, and when to override.
>
> pyfinAgent generates signals. You trade them. This guide explains what you
> see, what it means, and when to deviate.

---

## 1. Signal Anatomy

Every signal pyfinAgent produces contains six core fields:

| Field | Type | Description |
|-------|------|-------------|
| **ticker** | string | Stock symbol (e.g., `AAPL`, `MSFT`). Alphanumeric + dots only. |
| **signal** | string | One of `BUY`, `SELL`, or `HOLD`. |
| **confidence** | float | 0.0 to 1.0. How strongly the system backs this call. |
| **date** | string | The date the signal was generated (ISO format: `YYYY-MM-DD`). |
| **factors** | list | The supporting evidence, e.g., `"momentum_3m: +12%"`, `"earnings_tone: positive"`. |
| **reason** | string | A plain-English thesis explaining why. |

### What You See in Slack

Signals arrive in `#pyfinagent-signals` as Block Kit messages:

```
[green circle] AAPL BUY               <- Header: emoji + ticker + action
                                          Green = BUY, Red = SELL, Yellow = HOLD

Confidence: 0.82                       <- Fields section
Price: $187.50
Size: $850
Stop: $172.50

Thesis: Strong momentum continuation   <- Reason (truncated to 500 chars)
with positive earnings revision and
institutional accumulation...

PyFinAgent | 2026-04-16 | abc123ef     <- Footer: branding + date + signal_id
```

### Signal Types

- **BUY**: The system recommends opening or adding to a long position.
- **SELL**: The system recommends closing or reducing an existing position.
- **HOLD**: No action recommended. HOLD signals post to Slack for visibility
  but do NOT trigger any trade. Treat them as "no change."

### Signal ID and Deduplication

Each signal gets a unique ID (a short hash like `abc123ef`). If the same signal
is generated twice in one session, the duplicate is silently suppressed. You will
only see each distinct signal once.

---

## 2. Confidence Interpretation

Confidence is a **continuous score from 0.0 to 1.0**, not a letter grade or
bucket. Higher is stronger conviction.

### Reading Confidence Levels

| Range | Interpretation | Your Action |
|-------|---------------|-------------|
| **0.80 - 1.00** | High conviction. Multiple factors agree. | Follow the signal with normal sizing. |
| **0.60 - 0.79** | Moderate conviction. Most factors agree, some mixed. | Follow, but consider reducing size by 25-50%. |
| **0.40 - 0.59** | Low conviction. Factors are split or weak. | Use caution. Consider skipping or taking a half-size position. |
| **0.00 - 0.39** | Very low conviction. Barely above noise. | Strongly consider skipping. The system is not confident. |

These ranges are guidance, not hard rules. Confidence feeds directly into
position sizing (see Section 3), so lower-confidence signals automatically
get smaller positions.

### How Confidence Affects Trading

The system classifies recommendations internally:

- **BUY** or **STRONG_BUY** --> triggers a buy order
- **SELL** or **STRONG_SELL** --> triggers a sell order
- **HOLD** or downgrade from BUY --> triggers a position review (no immediate trade)

The risk judge also produces a `recommended_position_pct` (0-100%) which acts
as a secondary sizing input. When both confidence and the risk judge agree,
the signal is strongest.

---

## 3. Position Sizing

pyfinAgent uses a **hybrid sizing formula** that takes the minimum of up to
three independent calculations. This means the most conservative cap always wins.

### The Three Arms

**(a) Hard Cap (always applied)**

```
size = min(equity * 5.0%, $1,000)
```

- Maximum 5% of your total portfolio value per position
- Maximum $1,000 per position (dollar cap)
- Whichever is smaller

**(b) Half-Kelly (when confidence is available)**

```
size = 0.5 * confidence * equity
```

- Based on the Kelly Criterion, a mathematical formula for optimal bet sizing
- The 0.5 multiplier is "half-Kelly" -- captures ~75% of full-Kelly growth
  at ~50% of the variance
- A confidence of 0.80 on a $20,000 portfolio = 0.5 * 0.80 * $20,000 = $8,000
  (but this will be capped by the hard cap above)

**(c) Inverse-Volatility (when volatility data is available)**

```
size = (target_vol% / annualized_vol) * equity
```

- Allocates more to low-volatility stocks, less to high-volatility ones
- Target volatility defaults to 5% if not specified

### Final Size

The system takes **the minimum** of all available arms:

```
final_size = min(hard_cap, kelly_arm, vol_arm)
```

This conservative approach means you will never see a position that exceeds
any single risk limit.

### Portfolio-Level Risk Constraints

Beyond per-position sizing, the system enforces these portfolio-level limits:

| Constraint | Limit | What It Means |
|------------|-------|---------------|
| **Max per-ticker exposure** | 10% of portfolio | No single stock can exceed 10% of total value |
| **Max total exposure** | 100% of portfolio | No leverage -- total positions cannot exceed portfolio value |
| **Max daily trades** | 5 per day | Prevents overtrading. After 5 trades in a day, new signals are blocked. |
| **Max drawdown (kill switch)** | -15% | If portfolio drops 15% from peak, all new BUYs are blocked. See Section 4. |
| **Max position size** | 5% of equity | Per-position hard cap |
| **Max position USD** | $1,000 | Per-position dollar cap |
| **Stop-loss** | 8% below entry | Fixed stop-loss per position. See Section 4. |
| **Trailing stop** | 3% below peak | Trailing stop per position. See Section 4. |

These limits are **hardcoded in the system** (not configurable via settings files).
They cannot be accidentally relaxed during an incident.

---

## 4. Stop-Loss and Drawdown Protection

### Per-Position Stop-Loss

Each position has two stop-loss mechanisms:

**Fixed Stop-Loss (8% below entry price)**

If a position drops 8% or more from your entry price, it triggers an automatic
SELL. This is based on William O'Neil's CAN SLIM methodology.

Example: You buy AAPL at $100. If the price drops to $92 or below, the system
generates a SELL signal with `reason=stop_loss`.

**Trailing Stop (3% below peak price)**

If a position has risen above your entry price and then drops 3% from its
highest point, it triggers a SELL. This locks in gains on winning positions.

Example: You buy AAPL at $100. It rises to $120 (new peak). If it drops to
$116.40 or below (3% below $120), the system generates a SELL.

### Which Stop Fires First?

The system checks both stops on every evaluation. Whichever triggers first wins.
For a stock that has risen significantly, the trailing stop will usually trigger
before the fixed stop.

### Stop-Loss Precedence

**Important**: A stop-loss SELL takes precedence over any concurrent BUY
re-evaluation signal. If the system simultaneously generates a "BUY based on new
analysis" and a "SELL based on stop-loss breach" for the same ticker, the SELL
wins. This prevents the system from holding a losing position based on
optimistic re-analysis.

### Portfolio-Level Drawdown Protection

The system tracks your overall portfolio drawdown (how far you are below your
peak portfolio value) and applies graduated responses:

| Drawdown | Tier | System Response |
|----------|------|-----------------|
| Better than -5% | OK | Normal operation |
| -5% to -10% | Warning | Logged, no automatic action |
| -10% to -15% | De-risk | Position sizes halved automatically |
| Worse than -15% | Kill | **All new BUYs blocked.** SELLs still allowed (de-risking). Manual reset required. |

If the kill switch fires at -15%, the system will:
1. Block every new BUY signal
2. Continue allowing SELLs (so you can de-risk)
3. Require a manual reset before BUYs resume

This is your circuit breaker. It exists to prevent catastrophic loss during
a market crash or a systematic error in the signal pipeline.

---

## 5. The Daily Signal Flow

Here is what happens every trading day, automatically:

1. **Screen**: The system scans the investment universe (US equities) using
   6 months of price/fundamental data.
2. **Rank**: Candidate stocks are ranked by a composite score.
3. **Analyze**: Top candidates are analyzed by the 28-agent pipeline
   (momentum, sentiment, earnings, macro, options flow, etc.).
4. **Re-evaluate**: Existing holdings are re-analyzed based on new data.
5. **Decide**: The portfolio manager produces trade orders:
   - New BUY signals for top candidates
   - SELL signals for positions hitting stop-loss or downgrade
   - HOLD for positions that remain on-thesis
6. **Risk Check**: Every proposed trade passes through the risk gate
   (Section 3 constraints). Trades that violate any limit are rejected.
7. **Execute (Paper)**: Approved trades execute in the paper portfolio.
8. **Notify**: Signals post to Slack in `#pyfinagent-signals`.
9. **Track**: Signal accuracy is tracked over time (hit/miss/neutral
   classification based on forward returns).

### Your Part

When you see a signal in Slack:

1. **Read the signal** -- ticker, action, confidence, size, thesis.
2. **Check the factors** -- do they make sense given what you know about
   the stock and the current market?
3. **Decide** -- follow the signal, adjust the size, or skip it entirely.
4. **Execute** -- place the order in your broker account manually.
5. **Record** -- note what you did so the weekly review can compare your
   actions against the system's recommendations.

---

## 6. When to Override Ford

pyfinAgent is a tool, not an oracle. You are the final decision-maker.
Here are situations where you should consider overriding or skipping a signal:

### Always Skip or Reduce Size When:

- **Earnings day**: If the stock reports earnings within 1-2 days, the signal
  is based on pre-earnings data. The price will gap on the report. Consider
  waiting until after the announcement.

- **Known corporate event**: Mergers, spinoffs, FDA decisions, major lawsuits.
  The system does not know about events that haven't yet affected price or
  sentiment data.

- **Market-wide stress**: During a broad market selloff (VIX > 30), signals
  based on normal-regime analysis may not apply. Consider reducing all
  position sizes by 50% or pausing entirely.

- **Very low confidence (< 0.40)**: The system is barely above noise. The
  expected value of the trade is marginal at best.

- **You have non-public information**: If you know something material about
  the company that the system cannot know, trust your judgment. (Standard
  insider-trading rules apply -- do not trade on material non-public info.)

### Always Follow Stop-Loss Signals

Do NOT override a stop-loss SELL. The stop-loss exists to limit downside.
Holding through a stop-loss breach because "it will come back" is the single
most common way retail traders turn small losses into large ones.

### The Rollback Trigger

If the system's live performance (measured by Sharpe ratio) drops below 0.5
over a trailing 14-day window:

1. **Stop all signals** -- do not execute any new trades.
2. **Investigate** -- check the Harness tab for recent changes, review signal
   accuracy, look for data quality issues.
3. **Do not restart** until you have identified the cause and explicitly
   re-approved the system.

This is documented in the Go-Live Checklist item 4.4.6.4.

### Your Decision Framework

For every signal, ask yourself:

1. **Do I understand the thesis?** If the reason field is unclear, skip it.
2. **Does the confidence justify the size?** Lower confidence = smaller
   position or skip.
3. **Is there an upcoming event the system cannot know about?** If yes,
   wait or skip.
4. **Am I comfortable with the stop-loss level?** If the stop price is
   too close or too far, adjust in your broker.
5. **Does this fit my overall portfolio?** Even if the system says BUY,
   if you are already overexposed to that sector, skip it.

---

## 7. Key Numbers Reference

A quick-reference table of all hardcoded limits and thresholds:

| Parameter | Value | Source |
|-----------|-------|--------|
| Max position size | 5% of equity | `get_risk_constraints` |
| Max position USD | $1,000 | `get_risk_constraints` |
| Max per-ticker exposure | 10% of portfolio | `get_risk_constraints` |
| Max total exposure | 100% of portfolio | `get_risk_constraints` |
| Max daily trades | 5 | `get_risk_constraints` |
| Fixed stop-loss | 8% below entry | `check_stop_loss` |
| Trailing stop | 3% below peak | `check_stop_loss` |
| Drawdown warning | -5% | `track_drawdown` |
| Drawdown de-risk | -10% (halve sizes) | `track_drawdown` |
| Drawdown kill switch | -15% (block BUYs) | `track_drawdown` / `risk_check` |
| Confidence range | 0.0 to 1.0 | `validate_signal` |
| Half-Kelly multiplier | 0.5 | `size_position` |
| Rollback trigger | Sharpe < 0.5 / 14 days | Go-Live Checklist 4.4.6.4 |
| Transaction cost model | 0.1% per trade | `paper_trader` |

---

## 8. Glossary

- **Confidence**: A 0.0-1.0 score representing the system's conviction in
  a signal. Not a probability of profit -- it is a composite of factor
  agreement and analysis quality.

- **Drawdown**: How far the portfolio has fallen from its all-time peak,
  expressed as a percentage. A -10% drawdown means the portfolio is 10%
  below its highest recorded value.

- **Half-Kelly**: A position sizing method that uses half the mathematically
  optimal bet size (Kelly Criterion). Sacrifices ~25% of growth for ~50%
  less variance.

- **Kill Switch**: The -15% drawdown circuit breaker that blocks all new
  BUY signals. Requires manual reset.

- **Sharpe Ratio**: A measure of risk-adjusted return. Higher is better.
  The backtest Sharpe is 1.17; the paper trading floor is 0.82 (70% of
  backtest).

- **Signal ID**: A short hash (e.g., `abc123ef`) that uniquely identifies
  each signal. Used for deduplication and tracking.

- **Stop-Loss**: An automatic SELL trigger when a position drops below a
  threshold. Fixed stop = 8% below entry. Trailing stop = 3% below peak.

- **Trailing Stop**: A stop-loss that moves up with the stock price but
  never moves down. Locks in gains on winning positions.

---

## Important Reminders

1. **You are the decision-maker.** Ford generates signals. You decide
   whether to trade them. Never feel obligated to follow every signal.

2. **Always honor stop-losses.** The one thing you should NOT override
   is a stop-loss SELL. Cut losses short.

3. **Check the daily signal before acting.** Read the thesis, check the
   confidence, verify there are no upcoming events the system cannot see.

4. **Weekly review.** Compare your actual trades against the system's
   signals. Look for patterns in what you skipped and why. Adjust your
   approach based on evidence, not feelings.

5. **If in doubt, skip.** A missed gain is annoying. A realized loss is
   permanent. The system will generate more signals tomorrow.
