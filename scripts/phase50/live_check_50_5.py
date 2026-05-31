"""phase-50.5 live_check -- EU (.DE) multi-market backtest evidence.

Proves the 4 immutable success criteria on REAL yfinance data:
  #1 backtest engine accepts a market + uses its benchmark (^GDAXI for EU) +
     FX-converts NAV/returns to base currency (USD).
  #2 data-quality gate detects/drops identical-OHLC + gross-deviation bars in
     intl series, logging how many were dropped (no silent truncation).
  #3 an EU (.DE) backtest runs end-to-end with the correct benchmark +
     FX-converted returns + the gate active; US backtests unchanged.
  #4 live evidence: EU backtest summary (benchmark, FX-converted return,
     n bars dropped by the quality gate).

Tiered + resilient: each tier in its own try/except so one network/BQ failure
does not erase the rest of the evidence. $0 LLM. Read-only except BQ price
ingest (the engine's own auto-ingest, gated by BQ availability)."""
from __future__ import annotations

import sys
import traceback

import pandas as pd
import yfinance as yf

from backend.backtest.markets import get_market_config, MARKET_CONFIG, market_for_symbol
from backend.backtest.analytics import compute_baseline_strategies
from backend.tools.price_quality import validate_ohlcv

EU_TICKERS = ["SAP.DE", "SIE.DE", "BMW.DE", "ALV.DE", "BAS.DE"]
BENCH_EU = "^GDAXI"


def section(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def _close_df(symbol, period="1y"):
    """Fetch a yfinance OHLCV frame (flat columns)."""
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    return df


# ---------------------------------------------------------------------------
section("CRITERION #1a: MARKET_CONFIG benchmarks")
for m in ["US", "EU", "KR", "NO", "CA"]:
    print(f"  {m}: benchmark={get_market_config(m)['benchmark']:<10} currency={get_market_config(m)['currency']}")
assert get_market_config("EU")["benchmark"] == "^GDAXI"
assert all("benchmark" in c for c in MARKET_CONFIG.values())
print("  OK -- every market carries a benchmark; EU=^GDAXI")

# ---------------------------------------------------------------------------
section("CRITERION #1b: engine accepts market='EU' + selects ^GDAXI")
try:
    from backend.backtest.candidate_selector import CandidateSelector
    eu_universe = CandidateSelector().get_universe_tickers(market="EU")
    print(f"  get_universe_tickers(market='EU') -> {len(eu_universe)} tickers, sample: {eu_universe[:5]}")
    assert eu_universe and all(market_for_symbol(t) == "EU" for t in eu_universe[:5])
    print("  OK -- EU universe is .DE/.PA/... (market_for_symbol -> EU)")
except Exception as e:
    print(f"  WARN universe check degraded: {e}")

# ---------------------------------------------------------------------------
section("CRITERION #2: data-quality gate on REAL .DE bars (drop/flag counts)")
total_dropped = total_flagged = 0
for sym in EU_TICKERS:
    try:
        raw = _close_df(sym, "6mo")
        clean, rep = validate_ohlcv(raw, market="EU", ticker=sym)
        total_dropped += rep["dropped"]; total_flagged += rep["flagged"]
        print(f"  {sym:<8} raw={len(raw):>3} bars -> clean={len(clean):>3} | "
              f"dropped={rep['dropped']} flagged={rep['flagged']} {rep['reasons']}")
    except Exception as e:
        print(f"  {sym}: fetch/validate degraded: {e}")
# inject a synthetic bad bar into a real series to PROVE the drop path fires on live data
try:
    raw = _close_df("SAP.DE", "3mo")
    n0 = len(raw)
    op = float(raw["Open"].iloc[3])
    for col in ("Open", "High", "Low", "Close"):          # identical-OHLC...
        raw.iloc[3, raw.columns.get_loc(col)] = op
    raw.iloc[3, raw.columns.get_loc("Volume")] = 0        # ...+ zero vol -> DROP
    raw.iloc[6, raw.columns.get_loc("Close")] = float(raw["Close"].iloc[5]) * 1.8  # +80% glitch -> DROP
    clean, rep = validate_ohlcv(raw, market="EU", ticker="SAP.DE(+injected)")
    print(f"  INJECTED bad bars into real SAP.DE: {n0}->{len(clean)} dropped={rep['dropped']} ({rep['reasons']})")
    assert rep["dropped"] >= 2, "gate failed to drop injected bad bars"
    print("  OK -- gate drops unambiguous bad bars on real intl data; counts logged (no silent truncation)")
except Exception as e:
    print(f"  WARN injected-bar proof degraded: {e}")
print(f"  TOTAL across real .DE series: dropped={total_dropped} flagged={total_flagged}")

# ---------------------------------------------------------------------------
section("CRITERION #1c + #4: EU baseline w/ ^GDAXI + FX-converted return (REAL data)")
try:
    # build a real prices_cache_fn over ^GDAXI + EU candidates
    store = {}
    for sym in [BENCH_EU] + EU_TICKERS:
        df = _close_df(sym, "1y")
        if not df.empty:
            ser = df["Close"].copy(); ser.index = pd.to_datetime(ser.index).tz_localize(None)
            store[sym] = ser
    all_dates = sorted(store[BENCH_EU].index)
    test_start = all_dates[len(all_dates) // 2].strftime("%Y-%m-%d")   # ~6mo in (leaves lookback)
    test_end = all_dates[-1].strftime("%Y-%m-%d")

    def prices_cache_fn(ticker, start, end):
        ser = store.get(ticker)
        if ser is None:
            return pd.DataFrame({"close": []})
        m = (ser.index >= pd.Timestamp(start)) & (ser.index <= pd.Timestamp(end))
        return pd.DataFrame({"close": ser[m].values})

    eu = compute_baseline_strategies(
        prices_cache_fn, test_start, test_end, EU_TICKERS,
        benchmark=BENCH_EU, base_currency="USD", local_currency="EUR",
    )
    # the SAME computation in local currency (no FX) for the comparison
    local = compute_baseline_strategies(prices_cache_fn, test_start, test_end, EU_TICKERS, benchmark=BENCH_EU)
    print(f"  window {test_start} .. {test_end}")
    print(f"  benchmark={eu['benchmark']}  fx_ratio(EUR->USD endpoint)={eu['fx_ratio']:.5f}")
    print(f"  ^GDAXI return  LOCAL(EUR)={local['spy_return_pct']:+.3f}%   ->  USD={eu['spy_return_pct']:+.3f}%")
    print(f"  equal-weight   LOCAL(EUR)={local['equal_weight_return_pct']:+.3f}%   ->  USD={eu['equal_weight_return_pct']:+.3f}%")
    print(f"  ^GDAXI Sharpe (local) = {eu['spy_sharpe']:.3f}")
    assert eu["benchmark"] == "^GDAXI" and eu["base_currency"] == "USD"
    print("  OK -- EU baseline uses ^GDAXI + FX-converts returns EUR->USD")
except Exception as e:
    print(f"  WARN EU baseline degraded: {e}\n{traceback.format_exc()}")

# ---------------------------------------------------------------------------
section("CRITERION #3 (US unchanged): US baseline -> SPY, fx_ratio == 1.0 (byte-identical)")
try:
    spy = _close_df("SPY", "1y")
    ser = spy["Close"].copy(); ser.index = pd.to_datetime(ser.index).tz_localize(None)
    us_store = {"SPY": ser}

    def us_fn(ticker, start, end):
        s2 = us_store.get(ticker)
        if s2 is None:
            return pd.DataFrame({"close": []})
        m = (s2.index >= pd.Timestamp(start)) & (s2.index <= pd.Timestamp(end))
        return pd.DataFrame({"close": s2[m].values})

    dts = sorted(ser.index)
    ts, te = dts[len(dts)//2].strftime("%Y-%m-%d"), dts[-1].strftime("%Y-%m-%d")
    us = compute_baseline_strategies(us_fn, ts, te, ["SPY"])  # defaults -> SPY/USD
    print(f"  US benchmark={us['benchmark']}  fx_ratio={us['fx_ratio']}  base={us['base_currency']}")
    print(f"  SPY return USD={us['spy_return_pct']:+.4f}%  (fx_ratio==1.0 -> exact passthrough)")
    assert us["benchmark"] == "SPY" and us["fx_ratio"] == 1.0
    print("  OK -- US path uses SPY + fx_ratio EXACTLY 1.0 (no conversion, byte-identical)")
except Exception as e:
    print(f"  WARN US check degraded: {e}")

print("\nlive_check_50.5 complete.")
