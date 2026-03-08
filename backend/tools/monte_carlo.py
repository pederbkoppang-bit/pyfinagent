"""
Monte Carlo VaR simulation engine.
Runs 1,000 Geometric Brownian Motion paths to estimate Value-at-Risk,
expected shortfall, and probability distributions.

Research basis: Goldman Sachs stress-testing framework (ref 16).
"""

import logging
import math

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

NUM_SIMULATIONS = 1_000
TRADING_DAYS_YEAR = 252


def get_monte_carlo_simulation(ticker: str) -> dict:
    """
    Run Monte Carlo VaR simulation using historical daily returns.

    Returns structured risk metrics including VaR, expected shortfall,
    and probability of various return thresholds.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        if hist.empty or len(hist) < 60:
            return {
                "ticker": ticker,
                "signal": "ERROR",
                "summary": "Insufficient price history for Monte Carlo simulation.",
            }

        # Calculate daily log returns
        close = hist["Close"].dropna().values
        log_returns = np.diff(np.log(close))
        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns))
        current_price = float(close[-1])

        # Horizons: 3M, 6M, 1Y trading days
        horizons = {
            "3M": 63,
            "6M": 126,
            "1Y": TRADING_DAYS_YEAR,
        }

        results = {}
        rng = np.random.default_rng(seed=42)

        for label, days in horizons.items():
            # GBM simulation: S(t) = S(0) * exp((mu - sigma^2/2)*t + sigma*W(t))
            dt = 1.0  # daily steps
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * math.sqrt(dt)

            # Generate random walks
            random_shocks = rng.standard_normal((NUM_SIMULATIONS, days))
            daily_returns = drift + diffusion * random_shocks
            price_paths = current_price * np.exp(np.cumsum(daily_returns, axis=1))

            final_prices = price_paths[:, -1]
            returns_pct = (final_prices - current_price) / current_price * 100

            # VaR (losses are negative returns)
            var_95 = float(np.percentile(returns_pct, 5))
            var_99 = float(np.percentile(returns_pct, 1))

            # Expected Shortfall (average of worst 5%)
            worst_5pct = returns_pct[returns_pct <= np.percentile(returns_pct, 5)]
            es_95 = float(np.mean(worst_5pct)) if len(worst_5pct) > 0 else var_95

            # Probabilities
            prob_gain_20 = float(np.mean(returns_pct >= 20) * 100)
            prob_loss_20 = float(np.mean(returns_pct <= -20) * 100)
            prob_positive = float(np.mean(returns_pct > 0) * 100)

            # Percentile bands for fan chart
            percentiles = {}
            for p in [5, 10, 25, 50, 75, 90, 95]:
                path_percentile = np.percentile(price_paths, p, axis=0)
                # Sample at intervals for chart data
                indices = np.linspace(0, days - 1, min(days, 50), dtype=int)
                percentiles[f"p{p}"] = [float(path_percentile[i]) for i in indices]

            results[label] = {
                "var_95": round(var_95, 2),
                "var_99": round(var_99, 2),
                "expected_shortfall_95": round(es_95, 2),
                "prob_gain_20_pct": round(prob_gain_20, 1),
                "prob_loss_20_pct": round(prob_loss_20, 1),
                "prob_positive": round(prob_positive, 1),
                "median_return": round(float(np.median(returns_pct)), 2),
                "mean_return": round(float(np.mean(returns_pct)), 2),
                "std_return": round(float(np.std(returns_pct)), 2),
                "percentile_paths": percentiles,
            }

        # Overall signal based on 6M risk profile
        six_month = results.get("6M", {})
        var_6m = six_month.get("var_95", 0)

        if var_6m > -10:
            signal = "LOW_RISK"
        elif var_6m > -20:
            signal = "MODERATE_RISK"
        elif var_6m > -35:
            signal = "HIGH_RISK"
        else:
            signal = "EXTREME_RISK"

        summary = (
            f"Monte Carlo ({NUM_SIMULATIONS} simulations): "
            f"6M VaR(95%)={six_month.get('var_95', 'N/A')}%, "
            f"Prob(+20%)={six_month.get('prob_gain_20_pct', 'N/A')}%, "
            f"Prob(-20%)={six_month.get('prob_loss_20_pct', 'N/A')}%"
        )

        return {
            "ticker": ticker,
            "signal": signal,
            "summary": summary,
            "current_price": round(current_price, 2),
            "daily_mu": round(mu, 6),
            "daily_sigma": round(sigma, 6),
            "annualized_volatility": round(sigma * math.sqrt(TRADING_DAYS_YEAR) * 100, 2),
            "num_simulations": NUM_SIMULATIONS,
            "horizons": results,
        }

    except Exception as e:
        logger.error(f"Monte Carlo simulation failed for {ticker}: {e}")
        return {
            "ticker": ticker,
            "signal": "ERROR",
            "summary": f"Monte Carlo simulation failed: {e}",
        }
