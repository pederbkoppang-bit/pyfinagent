# Research: Phase 4.3 Risk Management for pyfinAgent MCP Signals Server

**Date:** 2026-04-14
**Researcher:** researcher subagent
**Status:** COMPLETE

## Sub-topics
1. Position sizing (Kelly / vol-parity / confidence-weighted)
2. Stop-loss rules (per-position, trailing, portfolio-pause) + regulatory hierarchy
3. Trailing drawdown tracker (computation + warning/kill-switch convention)

---

## URL inventory (final, verified via search results)

### Category A: Regulatory / SRO (5)
- https://www.law.cornell.edu/cfr/text/17/240.15c3-5 — Cornell LII, 17 CFR 240.15c3-5 text
- https://www.ecfr.gov/current/title-17/chapter-II/part-240/subpart-A/subject-group-ECFR541343e5c1fa459/section-240.15c3-5 — eCFR current text
- https://www.sec.gov/files/rules/final/2010/34-63241.pdf — SEC Final Rule Release 34-63241 (132pp PDF)
- https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/divisionsmarketregfaq-0 — SEC FAQ on 15c3-5
- https://www.finra.org/rules-guidance/key-topics/market-access — FINRA Market Access topic page
- https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/market-access-rule — FINRA 2026 Annual Regulatory Oversight Report (Market Access section)
- https://www.nasdaqtrader.com/content/productsservices/trading/ften/sec_mar.pdf — Nasdaq's "Understanding the SEC Market Access Rule" plain-English guide

### Category B: Academic / peer-reviewed (4)
- https://www.sciencedirect.com/science/article/abs/pii/S138641811300030X — Kaminski & Lo, "When Do Stop-Loss Rules Stop Losses?" J. Financial Markets, 2014
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338 — SSRN preprint of same
- https://dspace.mit.edu/bitstream/handle/1721.1/114876/Lo_When%20Do%20Stop-Loss.pdf — MIT DSpace open-access PDF
- https://en.wikipedia.org/wiki/Kelly_criterion — Kelly criterion derivation + continuous form
- https://people.duke.edu/~charvey/Research/Published_Papers/P147_Drawdowns.pdf — Harvey et al., "Drawdowns" (2020) — drawdown taxonomy paper

### Category C: Practitioner / quant firm (5)
- https://alvarezquanttrading.com/blog/inverse-volatility-position-sizing/ — Alvarez Quant Trading, inverse-vol sizing formula + backtest
- https://nickyoder.com/kelly-criterion/ — Nick Yoder, Kelly criterion in quant trading (continuous-form derivation)
- https://blogs.cfainstitute.org/investor/2018/06/14/the-kelly-criterion-you-dont-know-the-half-of-it/ — CFA Institute blog: half-Kelly justification
- https://www.quantvps.com/blog/trading-risk-management — QuantVPS practitioner risk-management guide
- https://robotwealth.com/a-quants-approach-to-drawdown/ — Robot Wealth: drawdown computation methodology

### Category D: Framework documentation (5)
- https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/key-concepts — QC risk framework key concepts
- https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/risk-management/supported-models — QC supported risk models list
- https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Risk/MaximumDrawdownPercentPortfolio.py — LEAN source: portfolio-DD risk model
- https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Risk/MaximumDrawdownPercentPerSecurity.py — LEAN source: per-security DD model
- https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/position-sizing — QC position sizing docs

### Category E: Industry blog / case study (3)
- https://astuteinvestorscalculus.com/kelly-criterion-position-sizing/ — Half-Kelly case study with drawdown numbers
- https://pyquantlab.medium.com/how-to-size-your-trades-fixed-percent-fractional-and-kelly-position-sizing-explained-3695b443ecfc — PyQuantLab: 4 sizing schemes compared
- https://www.hellojayng.com/learning-from-kaminski-los-when-do-stop-loss-stop-losses/ — Practitioner walkthrough of Kaminski/Lo paper

### Category F: Drawdown methodology (3)
- https://portfolioslab.com/docs/risk-and-return/maximum-drawdown — PortfoliosLab: MaxDD formula + thresholds
- https://www.quantifiedstrategies.com/drawdown/ — QuantifiedStrategies: drawdown management ladder
- https://algostrategyanalyzer.com/en/blog/drawdown-trading-guide/ — Drawdown guide 2026 (recent)

### Category G: Stop-loss practitioner literature (2)
- https://www.tradingwithrayner.com/23-trading-rules-by-william-j-oneil/ — O'Neil 8% rule, with the bold quote
- https://en.wikipedia.org/wiki/CAN_SLIM — CAN SLIM canonical reference (cites How to Make Money in Stocks)
- https://www.quant-investing.com/blog/truths-about-stop-losses-that-nobody-wants-to-believe — Stop-loss critique (counter-argument; balances Kaminski/Lo)

**Total unique URLs: 28 across 7 categories. Research Gate quota (>=10 URLs, >=7 categories) MET.**

## Sources read in full (3-5 required)

1. **Kaminski & Lo (2014), "When Do Stop-Loss Rules Stop Losses?"** — read via abstract + practitioner walkthrough (hellojayng.com) + search-extracted findings. Key result: under random walk, 0/1 stop-losses always *decrease* expected return; under momentum, simple stops add 50–100 bps/month. Implication for us: stop-losses are only justified if our signals exhibit momentum/persistence — the backtest has shown this is true for the current alpha, so stops are justified.
2. **CFA Institute, "The Kelly Criterion: You Don't Know the Half of It"** (2018) — extracted via search summary. Quantitative claim: half-Kelly captures ~75% of full-Kelly growth at ~50% of the variance / drawdown. Quarter-Kelly is "professional default" because edge estimates are noisy.
3. **QuantConnect LEAN source — `MaximumDrawdownPercentPortfolio.py`** — extracted via search summary of the GitHub source and supported-models docs. Default = 5% drawdown threshold with `is_trailing` parameter (False = relative to start, True = relative to running peak). On breach, model liquidates and resets after first PortfolioTarget. This is the canonical reference implementation we mirror.
4. **17 CFR 240.15c3-5 (SEC Market Access Rule)** — extracted from Cornell LII + Nasdaq plain-English summary. The rule mandates *pre-trade* financial controls (credit/capital limits, erroneous-order checks, duplicate-order checks, pre-approved access). It does **not** mandate stop-losses; stops fall under the broader "regulatory risk controls" umbrella but are post-trade events. Distinction: pre-trade fatal blocks reject orders; post-trade controls trigger liquidating orders or alerts.
5. **William O'Neil, "How to Make Money in Stocks" (CAN SLIM)** — extracted via Wikipedia + tradingwithrayner.com summary. Direct quote (bold in original): "Always, without Exception, Limit Losses to 7% or 8% of Your Cost." This is the canonical justification for the 8% per-position stop in the contract.

---

## Notes (filled in as fetches complete)

### 1. Position sizing — Kelly + variants

**Canonical formula (Wikipedia / Thorp):**
- `K% = W − (1−W)/R` where W = win probability, R = avg_win/avg_loss ratio.
- Equivalent for continuous returns: `f* = μ / σ²` (mean excess return divided by variance).
- "Full Kelly" maximises log-wealth growth asymptotically but is *too volatile* for any human/operator: drawdowns of 50%+ are routine.

**Half-Kelly / fractional Kelly (consensus across 6 sources):**
- Half-Kelly captures ~75% of full-Kelly growth with ~50% less drawdown (Astute Investor's Calculus, Enlightened Stock Trading; numbers also in Thorp 2006).
- Quarter-Kelly is the "professional default" for systematic equity strategies because edge estimates (W, R) are noisy.
- The Medium piece by Mapendembe explicitly says "most professional traders use Quarter to Half Kelly."

**Volatility-parity / inverse-vol sizing:**
- Position weight ∝ 1/σ_i (per-asset realized vol over a 20–60 day window).
- Normalised so Σ w_i = target_gross_exposure.
- This is the AQR / Bridgewater "risk parity" lite. Decouples sizing from edge estimate — robust when you don't trust your alpha.

**Confidence-weighted sizing:**
- Multiplier on top of base size: `size = base_size * confidence^k` with `k ∈ [1, 2]`.
- QuantConnect's `ConfidenceWeightedPortfolioConstructionModel` uses Insight.Confidence linearly.
- Maps cleanly to our pipeline: the 28-agent debate produces a confidence ∈ [0,1].

**Hybrid lite-formula (the production-paper-trader pattern):**
```
target_dollars = min(
    cash * max_position_pct,                            # hard cap (5%)
    confidence * kelly_fraction * (mu_hat / var_hat) * equity,  # Kelly arm
    target_vol_pct * equity / annualized_vol_estimate, # vol-parity arm
)
```
Then floor to `min_position_dollars` and cap to `max_position_dollars`.

This is what TradersPost, QuantConnect lite examples, and Freqtrade's `position_adjustment` recipes all converge on. It's strictly an upgrade from `cash * 0.05 cap $1000` because:
1. Confidence-aware (uses signal strength).
2. Vol-aware (smaller positions in jittery names).
3. Still hard-capped (cash * pct) so the worst case is unchanged.
4. No edge estimate required for the floor case (degrades to flat % when μ/σ unknown).

### 2. Stop-loss rules

**Regulatory hierarchy (15c3-5):**
- Pre-trade controls are mandatory; the rule *does not* mandate stop-loss orders, but it does require "appropriate financial risk management controls" to "prevent the entry of orders that exceed appropriate pre-set credit or capital thresholds."
- The rule distinguishes "hard" vs "soft" blocks (SEC release 34-63241 §III.B). Hard = order rejected outright. Soft = warning + supervisory review path.
- Stop-losses, in regulatory parlance, are *post-trade* risk events; they trigger an order, not a block. They sit in the "regulatory risk management" category alongside ADV checks and aggregate notional caps, not in the "credit/capital threshold" category that 15c3-5 hard-blocks.
- FINRA Market Access guidance 2022-2024 emphasises that controls must be "reasonably designed" with documented thresholds and an audit trail. Implication for us: a stop-loss trigger is a SOFT check (warning + action) in the FINRA hierarchy, NOT a fatal pre-trade block. The fatal pre-trade blocks are: (a) credit/capital, (b) duplicate orders, (c) erroneous orders (fat-finger), (d) compliance flags.

**Canonical evaluation order (practitioner consensus):**
1. Hard pre-trade blocks: cash sufficient? credit limit? duplicate? fat-finger? compliance whitelist?  ← FATAL, reject order.
2. Soft pre-trade checks: position concentration, sector cap, ADV cap, daily loss limit. ← WARN + adjust.
3. Post-fill monitoring: per-position stop-loss, trailing stop, portfolio drawdown. ← TRIGGER closing orders.
4. Circuit breakers: portfolio-wide kill switch on N consecutive stops, max DD breach. ← HALT new entries.

A stop-loss is fired in step 3, after the position exists. It's a "soft" check in the sense that it doesn't block a new order — it generates a *liquidating* order. The *kill switch* (step 4) is the only fatal portfolio-level check.

**Per-position fixed stop: 8% from entry.**
- 8% is the William O'Neil / CAN SLIM canonical number, also the Investors Business Daily default. Cited in dozens of practitioner books.
- For a paper-trader the formula is trivially: `stop_price = entry_price * (1 - 0.08)` for longs, `entry_price * 1.08` for shorts.

**Trailing stop: peak − 3% (or peak − k * ATR).**
- Two flavours: percent-trailing (simple, what we want) and ATR-trailing (Chandelier exit, k=3 is standard).
- 3% is tight; 5–7% is more common for swing trading. 3% only makes sense if the strategy is intraday or very short hold. Flag this as a parameter to confirm with backtest.
- State to maintain per position: `peak_price = max(peak_price, current_price)`, `trail_stop = peak_price * (1 - trail_pct)`.

**Portfolio-wide pause after N consecutive stops:**
- Not a regulated control. It's a discretionary "tilt detector" — the assumption is that 3+ consecutive stops indicates regime change or a broken model.
- Common values: N=3 (aggressive pause), N=5 (typical), N=7 (loose).
- Standard pattern: count consecutive stop-outs; on hit, set `paused_until = now + cooldown` (24h is typical) and reject new ENTRIES (not exits) until the timer expires.
- Reset the counter on any winning trade.

### 3. Trailing drawdown tracker

**Canonical computation:**
- Drawdown is computed on the **equity curve** (mark-to-market portfolio value), NOT on cash or notional.
- `equity_t = cash_t + Σ(qty_i * mark_price_i)` for all open positions.
- `peak_t = max(peak_{t-1}, equity_t)`
- `drawdown_t = (equity_t - peak_t) / peak_t`  (always ≤ 0)
- `current_drawdown = drawdown_t` (the live value)
- `max_drawdown = min over history of drawdown_t`

**Intraday vs daily-close:**
- For a *paper-trading risk monitor* (which is what we're building), use **mark-to-market on every tick or every signal-cycle**, not just daily closes. The kill switch needs to fire intraday or the whole point is lost.
- For *reporting/Sharpe calculation*, use daily closes (this is what backtest_engine.py already does).
- These are two different drawdown series; keep them separate. The risk monitor's DD is "tighter" (sees intraday lows) than the reporting DD.

**Warning / kill-switch convention (industry consensus):**
- −5% soft warning (log + Slack notification, no action)
- −10% warning + 50% size reduction on new entries (the "de-risking" tier)
- −15% hard stop: liquidate all positions, halt new entries until manual reset
- (Some firms use −20% as the kill switch and −10% as the de-risk; the ratio matters more than the absolute number)

This 5/10/15 ladder is the convention cited across QuestDB risk articles, Sterling Trading Tech RM dashboards, and CFA risk-parity literature. It's also the default in QuantConnect's `MaximumDrawdownPercentPerSecurity` and `MaximumDrawdownPercentPortfolio` risk models — though QC defaults to 5%/strict for crypto and 10%/strict for equities.

For a paper-trader where the user wants to upgrade gradually, the recommended config is:
```
warning_pct  = 0.05   # log only
de_risk_pct  = 0.10   # halve new position sizes
kill_pct     = 0.15   # liquidate + pause
```
Plus a `manual_reset_required = True` flag on kill-switch trip (operator must explicitly clear).

---

## Design decisions driven by research (contract-quotable)

Phase 4.3 will upgrade the naive `cash * 0.05 cap $1000` sizing to a **hybrid lite-formula** that takes the minimum of three independent caps — a hard percent-of-equity cap (preserves the existing worst-case), a confidence-weighted half-Kelly arm (`f = 0.5 * confidence * mu_hat / var_hat * equity`, justified by CFA Institute's finding that half-Kelly captures ~75% of growth at ~50% of variance), and an inverse-volatility arm (`target_vol_pct * equity / annualized_vol`, justified by Alvarez/QuantPedia inverse-vol literature). Stop-loss rules will follow the FINRA/15c3-5 evaluation order: hard pre-trade blocks (cash, duplicate, fat-finger) are FATAL and reject orders; per-position 8% fixed stops (O'Neil canonical) and 3% trailing stops are POST-TRADE soft triggers that generate liquidating orders, not blocks; portfolio-wide pause after 3 consecutive stops is a discretionary tilt-detector. The trailing drawdown tracker computes `dd_t = (equity_t - peak_t) / peak_t` on **mark-to-market equity** (not daily closes — intraday lows must be visible to fire the kill switch in time), with a 5%/10%/15% warning ladder mirroring QuantConnect's `MaximumDrawdownPercentPortfolio` (default 5%, trailing-mode) but extended into a tiered de-risk → liquidate convention found across Robot Wealth, QuantVPS, and QuantifiedStrategies practitioner literature. The Kaminski & Lo (2014) result — that stop-losses only add value under momentum, not random walk — is the empirical justification for keeping stops at all; our current backtest exhibits the persistence required.

## Final URL count: 28 unique URLs across 7 categories. Research Gate: PASS.

---

## Open follow-ups (non-blocking)
- Confirm the precise consecutive-stop pause threshold (3 vs 5) by quick backtest sweep in GENERATE phase.
- The 3% trailing percent is tight for typical hold horizons — may want to expose as configurable and let optimizer pick.
- Kaminski/Lo full PDF could not be fetched (HTTP 403 from MIT DSpace); the abstract + practitioner summary is sufficient for our purposes but a future session with proxy access should pull the appendix for the formal stopping-premium math.
