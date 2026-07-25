"""
Multi-dimensional anomaly detector.
Uses Z-score and IQR methods to identify statistical outliers across
financial metrics, price behavior, and enrichment signals.

Research basis: Goldman Sachs 127-dimensional anomaly detection (ref 16).
"""

import logging

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Z-score thresholds
_Z_STRONG = 2.0   # strong anomaly
_Z_MODERATE = 1.5  # moderate anomaly


def _aligned_ohlcv_arrays(hist):
    """Extract close/volume/high/low from ONE frame, guaranteed index-aligned.

    phase-80.31. This exists as a named function rather than four inline
    `.to_numpy()` calls so the alignment invariant has a single, directly
    testable seam. The previous code built each array with its own
    `hist["X"].dropna()`, and because `Volume` is `int64` -- which cannot
    hold NaN -- that dropna was a structural no-op while the three price
    columns lost yfinance's malformed trailing bar. `close[i]` and
    `volume[i]` then described different sessions.

    The caller drops malformed rows ONCE on the frame before calling this,
    so every array here is a straight read of the same index.

    Returns `(close, volume, high, low)`.
    """
    return (
        hist["Close"].to_numpy(),
        hist["Volume"].to_numpy(),
        hist["High"].to_numpy(),
        hist["Low"].to_numpy(),
    )


def _z(value: float, mean: float, std: float) -> float | None:
    """Compute Z-score. Returns None if std is zero."""
    return (value - mean) / std if std > 0 else None


def _append_if_anomalous(
    anomalies: list, metric: str, value: float, z_score: float | None,
    mean: float = 0, std: float = 0, note: str = "",
) -> None:
    """Append anomaly if z_score exceeds the moderate threshold."""
    if z_score is not None and abs(z_score) >= _Z_MODERATE:
        entry = {
            "metric": metric,
            "value": round(value, 2),
            "z_score": round(z_score, 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "severity": "high" if abs(z_score) >= _Z_STRONG else "moderate",
        }
        if note:
            entry["note"] = note
        anomalies.append(entry)


def get_anomaly_scan(ticker: str) -> dict:
    """
    Perform multi-dimensional anomaly detection on a ticker.
    Analyzes price, volume, valuation, technical, and fundamental metrics.

    Returns anomalies with Z-scores and classifications.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        # phase-80.31: drop malformed rows ONCE, on the FRAME, before anything
        # reads a column.
        #
        # This used to be four independent `hist["X"].dropna()` calls, one per
        # column. Column-wise dropna destroys positional correspondence the
        # moment two columns have different NaN patterns -- and that is exactly
        # the shape yfinance returns: the most recent bar carries NaN
        # Open/High/Low/Close but a REAL int64 Volume (measured AAPL 1y:
        # 251 raw rows -> close/high/low/open 250, volume 251).
        #
        # So `close[-1]` was day T-1 while `volume[-5:]` spanned T-4..T: the
        # volume anomaly compared a five-day window that included a malformed
        # session against a baseline drawn from a different set of days, then
        # reported a precise-looking mean/std/z triple with no indication the
        # inputs were misaligned.
        #
        # Ordering matters twice over:
        #  - BEFORE the `len(hist) < 20` guard, so the sufficiency check counts
        #    USABLE rows rather than raw ones;
        #  - subset is the OHLC columns only. Volume is deliberately NOT in the
        #    subset: a legitimately zero/absent volume must not discard an
        #    otherwise-good price bar.
        #
        # NOTE this is a row-wise drop, so it DISCARDS the malformed session's
        # real volume rather than restoring any price. It cannot make the
        # system less conservative (see the step's do-no-harm note), which is
        # why -- unlike phase-80.27 -- it needs no dark-launch flag.
        hist = hist.dropna(subset=["Open", "High", "Low", "Close"])

        if hist.empty or len(hist) < 20:
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": "Insufficient data for anomaly detection.",
            }

        anomalies: list[dict] = []

        close, volume, high, low = _aligned_ohlcv_arrays(hist)

        # phase-80.31: ENFORCE the alignment invariant at runtime, do not
        # merely arrange for it. Unequal lengths mean the four arrays no
        # longer describe the same sessions, so every z-score below would be
        # computed across mismatched days -- precisely the defect this step
        # exists to remove, and one that produced precise-looking numbers with
        # no outward sign anything was wrong. Fail loudly instead: a missing
        # anomaly scan is recoverable, a silently wrong one is not.
        # Length equality is necessary but NOT sufficient -- an off-by-one
        # ROLL keeps every length identical while desyncing the sessions. So
        # also spot-check both ends against the frame the arrays came from,
        # which is the cheapest check that actually pins POSITION.
        _len_ok = len(close) == len(volume) == len(high) == len(low) == len(hist)
        # NaN-tolerant comparison: `nan == nan` is False, so a legitimately
        # absent volume must not be mistaken for a desync.
        def _same(a, b) -> bool:
            try:
                if a != a and b != b:  # both NaN
                    return True
            except TypeError:
                pass
            return bool(a == b)

        _pos_ok = _len_ok and all(
            _same(arr[i], hist[col].iloc[i])
            for col, arr in (("Close", close), ("Volume", volume),
                             ("High", high), ("Low", low))
            for i in (0, -1)
        )
        if not (_len_ok and _pos_ok):
            logger.error(
                "anomaly_detector %s: OHLCV arrays are misaligned "
                "(close=%d volume=%d high=%d low=%d frame=%d, endpoints_ok=%s) "
                "-- refusing to score anomalies across mismatched sessions "
                "(phase-80.31)",
                ticker, len(close), len(volume), len(high), len(low),
                len(hist), _pos_ok,
            )
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": (
                    "Error: price/volume series are misaligned, so no anomaly "
                    "scan could be computed."
                ),
                "data": {},
            }

        # ── Price Return Anomalies ──────────────────────────────
        if len(close) >= 60:
            # 20-day return vs historical 20d returns
            recent_return_20d = (close[-1] / close[-21] - 1) * 100
            returns_20d = [(close[i] / close[i - 20] - 1) * 100 for i in range(20, len(close))]
            if len(returns_20d) > 1:
                mu, sigma = float(np.mean(returns_20d)), float(np.std(returns_20d))
                z = _z(recent_return_20d, mu, sigma)
                _append_if_anomalous(anomalies, "20d_price_return", recent_return_20d, z, mu, sigma)

            # 5-day return vs historical 5d returns
            recent_return_5d = (close[-1] / close[-6] - 1) * 100
            returns_5d = [(close[i] / close[i - 5] - 1) * 100 for i in range(5, len(close))]
            if len(returns_5d) > 1:
                mu, sigma = float(np.mean(returns_5d)), float(np.std(returns_5d))
                z = _z(recent_return_5d, mu, sigma)
                _append_if_anomalous(anomalies, "5d_price_return", recent_return_5d, z, mu, sigma)

        # ── Volume Anomaly ──────────────────────────────────────
        if len(volume) >= 60:
            # phase-80.31: `Volume` is NOT in the row-drop subset above, so a
            # legitimately zero/absent volume never discards an otherwise-good
            # price bar. The cost is that a NaN volume (only possible on a
            # float64 Volume column -- yfinance returns int64 for every market
            # this project trades, measured) now reaches this window, where it
            # would make std NaN, make `_z` return None, and SILENTLY drop the
            # anomaly. Silent suppression of a data-quality problem is the
            # phase-80.27 defect family, so make it explicit and logged.
            if not np.isfinite(volume[-60:]).all():
                logger.warning(
                    "anomaly_detector %s: non-finite value(s) in the 60-day "
                    "volume window -- skipping volume_5d_vs_60d rather than "
                    "scoring against a NaN baseline (phase-80.31)",
                    ticker,
                )
            else:
                recent_vol = float(np.mean(volume[-5:]))
                hist_vol = float(np.mean(volume[-60:]))
                std_vol = float(np.std(volume[-60:]))
                z = _z(recent_vol, hist_vol, std_vol)
                _append_if_anomalous(anomalies, "volume_5d_vs_60d", recent_vol, z, hist_vol, std_vol)

        # ── Volatility Anomaly ──────────────────────────────────
        if len(close) >= 60:
            daily_returns = np.diff(np.log(close))
            recent_vol_20d = float(np.std(daily_returns[-20:]) * np.sqrt(252) * 100)
            vols_20d = [
                float(np.std(daily_returns[i:i + 20]) * np.sqrt(252) * 100)
                for i in range(0, len(daily_returns) - 20, 5)
            ]
            if len(vols_20d) > 1:
                mu, sigma = float(np.mean(vols_20d)), float(np.std(vols_20d))
                z = _z(recent_vol_20d, mu, sigma)
                _append_if_anomalous(anomalies, "realized_volatility_20d", recent_vol_20d, z, mu, sigma)

        # ── Technical Anomalies ─────────────────────────────────
        if len(close) >= 50:
            # RSI (14-period)
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = float(np.mean(gains[-14:]))
            avg_loss = float(np.mean(losses[-14:]))
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0
            # RSI extremes: below 30 = oversold, above 70 = overbought
            if rsi < 30:
                _append_if_anomalous(anomalies, "rsi_oversold", rsi, -2.0, 50, 10,
                                     note=f"RSI {rsi:.0f} indicates oversold conditions")
            elif rsi > 70:
                _append_if_anomalous(anomalies, "rsi_overbought", rsi, 2.0, 50, 10,
                                     note=f"RSI {rsi:.0f} indicates overbought conditions")

            # Price vs 50-day SMA deviation
            sma_50 = float(np.mean(close[-50:]))
            pct_from_sma50 = ((close[-1] - sma_50) / sma_50) * 100
            # Compute historical deviations from rolling 50d SMA
            sma_devs = []
            for i in range(50, len(close)):
                sma = float(np.mean(close[i - 50:i]))
                sma_devs.append(((close[i] - sma) / sma) * 100)
            if len(sma_devs) > 1:
                mu, sigma = float(np.mean(sma_devs)), float(np.std(sma_devs))
                z = _z(pct_from_sma50, mu, sigma)
                _append_if_anomalous(anomalies, "price_vs_sma50", pct_from_sma50, z, mu, sigma,
                                     note=f"Price is {pct_from_sma50:+.1f}% from 50d SMA")

        if len(close) >= 200:
            # Price vs 200-day SMA deviation
            sma_200 = float(np.mean(close[-200:]))
            pct_from_sma200 = ((close[-1] - sma_200) / sma_200) * 100
            sma_devs_200 = []
            for i in range(200, len(close)):
                sma = float(np.mean(close[i - 200:i]))
                sma_devs_200.append(((close[i] - sma) / sma) * 100)
            if len(sma_devs_200) > 1:
                mu, sigma = float(np.mean(sma_devs_200)), float(np.std(sma_devs_200))
                z = _z(pct_from_sma200, mu, sigma)
                _append_if_anomalous(anomalies, "price_vs_sma200", pct_from_sma200, z, mu, sigma,
                                     note=f"Price is {pct_from_sma200:+.1f}% from 200d SMA")

        # Largest single-day move in last 5 days vs distribution
        if len(close) >= 60:
            daily_pct = np.abs(np.diff(close) / close[:-1]) * 100
            max_recent_move = float(np.max(daily_pct[-5:]))
            mu, sigma = float(np.mean(daily_pct)), float(np.std(daily_pct))
            z = _z(max_recent_move, mu, sigma)
            _append_if_anomalous(anomalies, "max_daily_move_5d", max_recent_move, z, mu, sigma,
                                 note=f"Largest daily move in last 5 days: {max_recent_move:.2f}%")

        # Price vs 52-week range
        if len(close) >= 200:
            high_52w = float(np.max(high[-252:])) if len(high) >= 252 else float(np.max(high))
            low_52w = float(np.min(low[-252:])) if len(low) >= 252 else float(np.min(low))
            range_52w = high_52w - low_52w
            if range_52w > 0:
                position = ((close[-1] - low_52w) / range_52w) * 100
                # Near 52-week low (<10%) or high (>90%) is notable
                if position < 10:
                    _append_if_anomalous(anomalies, "near_52w_low", position, -2.0, 50, 20,
                                         note=f"Price at {position:.0f}% of 52-week range (near low)")
                elif position > 90:
                    _append_if_anomalous(anomalies, "near_52w_high", position, 2.0, 50, 20,
                                         note=f"Price at {position:.0f}% of 52-week range (near high)")

        # ── Fundamental Anomalies ───────────────────────────────
        pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        if pe and forward_pe and pe > 0 and forward_pe > 0:
            pe_gap = ((pe - forward_pe) / pe) * 100
            if abs(pe_gap) > 20:
                _append_if_anomalous(anomalies, "pe_trailing_vs_forward_gap", pe_gap,
                                     pe_gap / 10, 0, 10,
                                     note=f"Trailing PE {pe:.1f} vs Forward PE {forward_pe:.1f}")

        # Revenue growth vs profit margin divergence
        rev_growth = info.get("revenueGrowth", 0)
        profit_margin = info.get("profitMargins", 0)
        if rev_growth and profit_margin:
            if rev_growth > 0.15 and profit_margin < 0.05:
                _append_if_anomalous(anomalies, "growth_margin_divergence",
                                     round(rev_growth * 100, 1), 2.5, 0, 0,
                                     note=f"Revenue growing {rev_growth*100:.0f}% but margin only {profit_margin*100:.1f}%")

        # Debt/equity spike
        de_ratio = info.get("debtToEquity")
        if de_ratio and de_ratio > 150:
            _append_if_anomalous(anomalies, "debt_to_equity", de_ratio,
                                 (de_ratio - 100) / 50, 100, 50)

        # Short ratio anomaly
        short_ratio = info.get("shortRatio")
        if short_ratio and short_ratio > 4:
            _append_if_anomalous(anomalies, "short_ratio", short_ratio,
                                 (short_ratio - 2) / 1.5, 2, 1.5)

        # Beta anomaly (far from 1.0)
        beta = info.get("beta")
        if beta is not None:
            if beta > 1.8 or beta < 0.3:
                z_beta = (beta - 1.0) / 0.4
                _append_if_anomalous(anomalies, "beta_extreme", beta, z_beta, 1.0, 0.4,
                                     note=f"Beta of {beta:.2f} is far from market average")

        # ── Classify overall signal ─────────────────────────────
        total = len(anomalies)
        high_severity = sum(1 for a in anomalies if a.get("severity") == "high")

        risk_metrics = {"debt_to_equity", "short_ratio", "growth_margin_divergence",
                        "rsi_overbought", "near_52w_high"}
        risk_anomalies = sum(
            1 for a in anomalies
            if a.get("z_score", 0) < -_Z_MODERATE or a["metric"] in risk_metrics
        )
        opportunity_anomalies = total - risk_anomalies

        if total == 0:
            signal = "NORMAL"
            summary = "No significant statistical anomalies detected."
        elif risk_anomalies > opportunity_anomalies:
            signal = "ANOMALY_RISK"
            summary = f"Detected {total} anomalies ({high_severity} high severity), primarily risk-related."
        else:
            signal = "ANOMALY_OPPORTUNITY"
            summary = f"Detected {total} anomalies ({high_severity} high severity), potentially indicating opportunity."

        return {
            "ticker": ticker,
            "signal": signal,
            "summary": summary,
            "anomaly_count": total,
            "anomalies": anomalies,
            "current_price": round(float(close[-1]), 2) if len(close) > 0 else None,
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"Anomaly detection failed: {e}",
        }
