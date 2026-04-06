# Phase 0.3 — Backtest vs Reality Gap Analysis

## Audit Date: 2026-03-25
## Auditor: Ford

---

This document catalogs assumptions in the backtest that differ from live trading.
Each gap is assessed for impact on expected real-world performance.

---

## 1. Execution Price

**Backtest**: Trades execute at the close price on the signal date.
**Reality**: Market-On-Close (MOC) orders approximate this, but:
- MOC orders must be placed before the exchange deadline (typically 15:50 ET for NYSE)
- Actual fill may differ slightly from official close
- After-hours news can cause next-day gaps

**Impact**: LOW. MOC orders are widely available and match our assumption closely.

**Mitigation**: Use VWAP or limit orders in live trading; add 1-2 bps slippage buffer.

---

## 2. Slippage & Market Impact

**Backtest**: Zero slippage. We can buy any quantity at the exact close price.
**Reality**: Large orders move the market (Almgren & Chriss, 2000):
- Impact ∝ σ × √(order_size / daily_volume)
- For S&P 500 stocks with our position sizes ($5K-$10K per position), impact is minimal
- For small-caps or illiquid names, impact could be significant

**Impact**: LOW for S&P 500 (liquid, ~$1B+ daily volume). HIGH if we expand to small-caps.

**Mitigation**: 
- Track `amihud_illiquidity` — already in feature vector, not yet used as a filter
- Add minimum liquidity threshold (e.g., avg daily volume > $10M) before trading
- Consider TWAP execution for larger positions

---

## 3. Partial Fills

**Backtest**: Orders always fill completely.
**Reality**: In liquid S&P 500 names, this is realistic. Could be an issue for:
- Illiquid options (if we ever trade options based on flow signals)
- Extended hours trading
- During market stress (flash crashes, circuit breakers)

**Impact**: LOW for our current universe and position sizes.

---

## 4. Transaction Costs

**Backtest**: Flat percentage (default 0.1% per trade = 10 bps).
**Reality**: 
- Most US brokers: $0 commission for equities (Schwab, Fidelity, Interactive Brokers Lite)
- IBKR Pro: $0.005/share, min $1
- Real cost is primarily spread (bid-ask): ~1-5 bps for S&P 500 stocks
- SEC fee: ~$0.02 per $1,000 sold
- FINRA TAF: $0.000145 per share sold

**Assessment**: Our 0.1% (10 bps) per-side estimate is CONSERVATIVE for S&P 500.
Actual round-trip cost ≈ 2-10 bps, not 20 bps.

**Impact**: Our backtest slightly UNDERESTIMATES performance by ~10-18 bps per round trip.

**Mitigation**: Consider reducing `transaction_cost_pct` to 0.05% for S&P 500 testing.

---

## 5. Dividends

**Backtest**: yfinance adjusted close prices account for dividends implicitly.
**Reality**: Dividends are paid as cash; reinvestment decision varies.

**Assessment**: Adjusted close correctly accounts for total return including dividends. ✅

---

## 6. Corporate Actions (Splits, Mergers, Delistings)

**Backtest**: yfinance adjusted close handles splits automatically.
**Reality**: 
- Mergers/acquisitions: stock may be delisted or converted; backtest uses last available price
- Bankruptcy: stock goes to ~$0; if we hold, we lose everything
- Spinoffs: may not be captured in adjusted close

**Impact**: MEDIUM for individual events, LOW overall for diversified portfolio.

**Mitigation**: Stop-loss in the backtest (SL barrier) limits downside. In live trading, use actual stop-loss orders.

---

## 7. Survivorship Bias

**Backtest**: We screen from CURRENT S&P 500 constituents.
**Reality**: Companies that were removed from S&P 500 (poor performers, bankruptcies) are not in our universe, making historical returns look better than they were.

**Impact**: MEDIUM. This is a well-known bias that can add 1-2% annual return to backtests.

**Mitigation**: Phase 1 item — obtain historical S&P 500 constituent lists and use them for point-in-time screening.

---

## 8. Look-Ahead Bias in Fundamentals

**Backtest**: We use `filing_date` (SEC filing date) as the availability date.
**Reality**: 
- Some fundamental data may be available before filing (preliminary earnings)
- FRED macro data has publication lag AND gets revised retroactively
- We use current (revised) FRED values, not first-release

**Impact**: LOW for fundamentals (filing_date is conservative). MINOR for FRED revisions.

---

## 9. Regime Changes

**Backtest**: Model trained on 2018-2025 data.
**Reality**: Market regimes change:
- Interest rate environment (ZIRP 2020 → 5%+ 2023 → cuts 2024)
- COVID crash (2020): unprecedented speed and recovery
- Quantitative tightening (2022): different correlation structure

**Impact**: MEDIUM. Model performance will vary across regimes.

**Mitigation**: 
- Walk-forward expanding window naturally adapts to new regimes ✅
- Turbulence index can flag regime changes (already coded, not yet integrated)
- Consider adding regime-conditional features in Phase 1

---

## 10. Rebalancing Frequency

**Backtest**: Trades only at window boundaries (every 3 months).
**Reality**: Could trade more frequently if signals change.

**Impact**: LOW — our holding period (weeks to months) matches the window frequency.

---

## Overall Assessment

| Gap | Impact | Phase |
|-----|--------|-------|
| Execution price | LOW | Manageable with MOC orders |
| Slippage | LOW (S&P 500) | Monitor if expanding universe |
| Partial fills | LOW | N/A for S&P 500 |
| Transaction costs | FAVORABLE | Backtest overestimates costs |
| Dividends | NONE | Adjusted close handles correctly |
| Corporate actions | LOW-MEDIUM | Stop-loss mitigates |
| Survivorship bias | MEDIUM | Phase 1 fix |
| Fundamentals look-ahead | LOW | Filing_date is conservative |
| Regime changes | MEDIUM | Walk-forward adapts; add turbulence |
| Rebalancing frequency | LOW | Matches holding period |

**Net assessment**: The backtest is moderately conservative (overestimates costs, doesn't benefit from survivorship bias in labels). The main risk is regime change, mitigated by walk-forward training. Survivorship bias in candidate selection is the biggest unaddressed issue (Phase 1).

**Expected real-world performance**: Within ±20% of backtest results for S&P 500 stocks, assuming normal market conditions. Wider gap during market stress events.
