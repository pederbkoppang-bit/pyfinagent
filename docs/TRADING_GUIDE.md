# How to Trade pyfinAgent Signals

> A practical guide for Peder. Covers every signal you will see in Slack,
> what the numbers mean, how to size a trade, when the system protects you,
> and when to override it.

---

## 1. Signal Anatomy -- What You See in Slack

Every trading day at 14:00 CEST, pyfinAgent runs a 28-agent analysis pipeline
on the US equity universe. When it finds a trade worth taking, it posts a
**signal card** to `#pyfinagent-signals`. Here is what each field means:

| Field | Example | What It Means |
|-------|---------|---------------|
| **Header** | "AAPL BUY" | The ticker and the recommended action (BUY, SELL, or HOLD). A green circle means BUY, red means SELL, yellow means HOLD. |
| **Confidence** | `0.72` | How sure the model is, on a scale from 0.00 (no conviction) to 1.00 (maximum conviction). See Section 2 for how to read this. |
| **Price** | `$150.25` | The market price at the time the signal was generated. Use this as your reference entry price. |
| **Size** | `$750.00` | The dollar amount the system recommends you buy or sell. See Section 3 for how this is calculated. |
| **Stop** | `$138.23` | The price at which you should exit the position to limit losses. See Section 4. |
| **Thesis** | "Strong momentum + insider buying" | A short explanation of why the model likes this trade, drawn from the analysis pipeline. |
| **Signal ID** | `abc1234567890def` | A unique 16-character identifier. Use this to look up the signal later or to reference it in conversation with Ford. |
| **Timestamp** | `2026-04-16 14:25` | When the signal was generated (local time). |

**HOLD signals** also appear in Slack. They mean: "I looked at this ticker and
decided not to trade." You do not need to act on HOLD signals. They are
informational -- useful for reviewing what the system considered and rejected.

---

## 2. Confidence Thresholds -- How to Read the Number

Confidence is a continuous score from 0.00 to 1.00. There are no rigid tiers,
but here is a practical interpretation:

| Range | Interpretation | Suggested Action |
|-------|---------------|------------------|
| **0.80 -- 1.00** | High conviction. Multiple factors align strongly. | Trade at the recommended size. |
| **0.60 -- 0.79** | Moderate conviction. Thesis is solid but not overwhelming. | Trade at the recommended size, or reduce by 25% if you are cautious. |
| **0.40 -- 0.59** | Low conviction. The model sees a thesis but uncertainty is high. | Consider skipping, or trade at half the recommended size. This is your judgment call. |
| **Below 0.40** | Very low conviction. Borderline signal. | Skip unless you have independent conviction from your own research. |

**Key rule:** The system never forces a trade. Every signal is a recommendation.
You always have the final say -- see Section 5 on when to override.

**How confidence affects sizing:** The recommended Size field already factors in
confidence (see Section 3). A 0.40 confidence signal will show a smaller dollar
amount than a 0.80 signal, all else equal.

---

## 3. Position Sizing -- How the Dollar Amount Is Calculated

The system uses a **three-arm minimum** formula. It computes three candidate
sizes and takes the smallest one. This ensures you never oversize a position
even if one arm malfunctions.

### The three arms

**(a) Hard equity cap** -- always computed:

    Size = min(portfolio_value x 5%, $1,000)

This is the absolute ceiling. On a $10,000 portfolio, that is $500.
On a $50,000 portfolio, the $1,000 USD cap kicks in.

**(b) Confidence-weighted half-Kelly** -- computed when confidence is available:

    Size = 0.5 x confidence x portfolio_value

Half-Kelly captures roughly 75% of full-Kelly growth at half the variance.
On a $10,000 portfolio with 0.72 confidence, this arm gives $3,600 -- but
arm (a) would cap it at $500.

**(c) Inverse-volatility** -- computed when the signal carries a volatility estimate:

    Size = (5% / annualized_vol) x portfolio_value

This shrinks the position when the stock is volatile and expands it when
the stock is calm. Not always available.

### What you see

The **Size** field in Slack is the final output: `min(arm_a, arm_b, arm_c)`.
On a $10,000 portfolio, most signals will show sizes in the $300--$500 range
because the 5% equity cap and $1,000 USD cap are binding.

### What you should do

Use the Size field as your order amount. If you want to be more conservative,
trade at 50--75% of the recommended size. Never trade more than the recommended
size unless you have a specific reason and are prepared to accept the extra risk.

---

## 4. Stop-Loss Execution -- How the System Protects You

pyfinAgent enforces two layers of stop-loss protection:

### Fixed stop-loss (per position): -8%

If a position drops 8% or more below your entry price, the system flags it
for exit. This is the classic O'Neill 7-8% stop.

    Stop triggers when: (current_price - entry_price) / entry_price <= -8%

**Example:** You buy AAPL at $150.00. The stop fires at $138.00 or below.

The **Stop** field in the Slack card shows this price. When you place your
broker order, set a stop-loss at this price.

### Trailing stop (per position): -3%

If a position has run up from entry and then drops 3% from its peak, the
system flags it for exit. This locks in gains on winning positions.

    Stop triggers when: (current_price - peak_price) / peak_price <= -3%

**Example:** You buy AAPL at $150, it runs to $170 (new peak). The trailing
stop fires at $164.90 or below ($170 x 0.97).

The trailing stop only activates after the position has moved above entry.
It does not conflict with the fixed stop.

### Portfolio-level drawdown kill switch: -15%

If the total portfolio drops 15% from its all-time high, the system blocks
all new BUY signals. You will see a red alert in Slack. SELLs are still
allowed (to de-risk). Trading resumes only after a manual review.

There are also two intermediate tiers:
- **Warning at -5%**: The system logs a warning. No action needed from you.
- **De-risk at -10%**: New position sizes are halved automatically.

### What you should do

1. When you execute a BUY, set a stop-loss order at the **Stop** price shown
   in the Slack card.
2. If the stock runs up significantly, consider manually raising your stop to
   lock in gains (trail it upward, never downward).
3. If you see a kill-switch alert (-15% drawdown), stop all new buys and wait
   for Ford to investigate.

---

## 5. When to Override Ford

pyfinAgent is a signal generator, not an autopilot. You are the decision-maker.
Override the system in these situations:

### Always override (ignore the signal)

- **Earnings within 48 hours.** The model does not reliably predict earnings
  surprises. If a signal fires on AAPL the day before earnings, skip it.
- **Major macro event.** Fed rate decisions, CPI prints, geopolitical shocks.
  If the market is likely to gap overnight, skip new BUY signals.
- **You know something the model does not.** If you have material context
  about a company (product recall, CEO departure, regulatory action) that
  the 28-agent pipeline would not have seen, trust your judgment.
- **The thesis does not make sense to you.** If you read the Thesis field
  and it sounds wrong or contradictory, skip the signal. The model is not
  infallible.

### Consider overriding (use judgment)

- **Low confidence (below 0.40).** The model is not sure. You probably
  should not be either.
- **Sector concentration.** If you already hold 3 tech stocks and the signal
  is a 4th tech BUY, consider skipping for diversification.
- **End of quarter.** Window-dressing flows and rebalancing can distort
  signals near quarter-end.
- **Signal contradicts recent price action.** If the model says BUY but the
  stock just broke a major support level, the signal may be stale.

### Never override (follow the signal)

- **Stop-loss triggers.** If the system says exit, exit. Do not move your
  stop further away. The 8% fixed stop and 3% trailing stop exist to prevent
  small losses from becoming large ones.
- **Kill-switch alerts.** If the portfolio is down 15%, the system is telling
  you to stop. Listen.

### How to communicate an override

If you skip or modify a signal, post a note in `#ford-approvals` with the
signal ID and your reason. This helps Ford learn from your overrides and
improve future signals.

---

## 6. Daily Workflow Summary

1. **14:00 CEST** -- Signals arrive in `#pyfinagent-signals`.
2. **Review each signal.** Read the header, confidence, thesis.
3. **Check for overrides.** Earnings? Macro event? See Section 5.
4. **If trading:** Open your broker, place a market or limit order for the
   recommended Size at approximately the Price shown. Set a stop-loss at the
   Stop price.
5. **If skipping:** No action needed. Optionally note in `#ford-approvals`.
6. **End of day:** Check the evening digest in `#pyfinagent-signals` for a
   portfolio summary.

---

## 7. Quick Reference Card

| Parameter | Value | Source |
|-----------|-------|--------|
| Max position size (% of equity) | 5% | `get_risk_constraints` |
| Max position size (absolute) | $1,000 | `get_risk_constraints` |
| Fixed stop-loss | -8% from entry | `get_risk_constraints` |
| Trailing stop | -3% from peak | `get_risk_constraints` |
| Max per-ticker exposure | 10% of portfolio | `get_risk_constraints` |
| Max total exposure | 100% (no leverage) | `get_risk_constraints` |
| Max daily trades | 5 | `get_risk_constraints` |
| Drawdown warning | -5% | `get_risk_constraints` |
| Drawdown de-risk (halve sizes) | -10% | `get_risk_constraints` |
| Drawdown kill switch (block BUYs) | -15% | `get_risk_constraints` |
| Signal delivery time | 14:00 CEST (weekdays) | `PAPER_TRADING_HOUR=14` |
| Confidence range | 0.00 -- 1.00 | `validate_signal` |

---

*This guide was written by Ford (Cycle 28, 2026-04-16). Peder: please read
end-to-end and post acknowledgement in `#ford-approvals` to satisfy checklist
item 4.4.5.5.*
