"""
Backtest analytics — Sharpe, Deflated Sharpe Ratio, baselines, reporting.
All computations are purely numerical, zero LLM cost.
"""

import math
import numpy as np
from scipy import stats

from backend.backtest.backtest_engine import BacktestResult


def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
    """Annualized Sharpe ratio from daily returns."""
    if len(returns) < 5:
        return 0.0
    excess = returns - risk_free_rate / 252
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def compute_max_drawdown(nav_series: np.ndarray) -> float:
    """Maximum peak-to-trough decline (%). Input: NAV series, not returns."""
    if len(nav_series) < 2:
        return 0.0
    peak = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - peak) / peak
    return float(drawdown.min() * 100)


def compute_alpha(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Annualized alpha = excess return over benchmark."""
    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    min_len = min(len(portfolio_returns), len(benchmark_returns))
    port_total = float(np.prod(1 + portfolio_returns[:min_len]) - 1)
    bench_total = float(np.prod(1 + benchmark_returns[:min_len]) - 1)
    return (port_total - bench_total) * 100


def compute_hit_rate(predictions: list[dict]) -> float:
    """% of correct directional calls."""
    valid = [p for p in predictions if p.get("correct") is not None]
    if not valid:
        return 0.0
    return sum(1 for p in valid if p["correct"]) / len(valid)


def compute_information_ratio(active_returns: np.ndarray) -> float:
    """Information Ratio = mean(active_return) / std(active_return), annualized."""
    if len(active_returns) < 5:
        return 0.0
    std = active_returns.std()
    if std == 0:
        return 0.0
    return float((active_returns.mean() / std) * np.sqrt(252))


def compute_deflated_sharpe(
    observed_sr: float,
    num_trials: int,
    variance_of_srs: float = 0.5,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    T: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts observed Sharpe for:
    - Number of trials (backtest configs tested)
    - Return distribution non-normality
    - Sample length

    Returns probability [0, 1] that observed SR >= expected max SR under null.
    DSR >= 0.95 means the result is statistically significant.
    """
    if num_trials < 1 or T < 10 or observed_sr == 0:
        return 0.0

    # Expected maximum Sharpe ratio under null for num_trials independent trials
    # E[max(SR)] ≈ sqrt(V) * [(1-γ)*Φ^{-1}(1-1/N) + γ*Φ^{-1}(1-1/(N*e))]
    # Simplified: E[max(SR)] ≈ sqrt(2*log(N)) * sqrt(V) (Euler approx)
    e_max_sr = math.sqrt(variance_of_srs) * (
        (1 - 0.5772) * stats.norm.ppf(1 - 1 / max(num_trials, 2))
        + 0.5772 * stats.norm.ppf(1 - 1 / (max(num_trials, 2) * math.e))
    )

    # Standard error of the Sharpe ratio (accounting for non-normality)
    se_sr = math.sqrt(
        (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) / T
    )

    if se_sr == 0:
        return 0.0

    # Test statistic
    z = (observed_sr - e_max_sr) / se_sr

    # Probability (one-sided)
    dsr = float(stats.norm.cdf(z))
    return max(0.0, min(1.0, dsr))


def compute_baseline_strategies(
    prices_cache_fn,
    test_start: str,
    test_end: str,
    candidate_tickers: list[str],
) -> dict:
    """
    Compute 3 baseline strategies over the same test period:
    1. Buy-and-hold SPY
    2. Equal-weight top candidates
    3. Momentum-only (top quartile by trailing 6M return)
    """
    from backend.backtest import cache

    # 1. SPY baseline
    spy_prices = prices_cache_fn("SPY", test_start, test_end)
    spy_return = 0.0
    if not spy_prices.empty and len(spy_prices) > 1:
        spy_return = float(
            (spy_prices["close"].iloc[-1] / spy_prices["close"].iloc[0] - 1) * 100
        )

    # 2. Equal-weight all candidates
    eq_returns = []
    for ticker in candidate_tickers[:50]:
        p = prices_cache_fn(ticker, test_start, test_end)
        if not p.empty and len(p) > 1:
            ret = float(p["close"].iloc[-1] / p["close"].iloc[0] - 1)
            eq_returns.append(ret)
    eq_weight_return = float(np.mean(eq_returns) * 100) if eq_returns else 0.0

    # 3. Momentum-only (top quartile by 6M trailing return)
    momentum_scores = {}
    for ticker in candidate_tickers[:50]:
        p = prices_cache_fn(ticker, test_start, test_start)
        # Need lookback — get 6 months before test_start
        import pandas as pd
        lookback_start = (pd.Timestamp(test_start) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        p_lb = prices_cache_fn(ticker, lookback_start, test_start)
        if not p_lb.empty and len(p_lb) > 20:
            momentum_scores[ticker] = float(p_lb["close"].iloc[-1] / p_lb["close"].iloc[0] - 1)

    # Top quartile by momentum
    if momentum_scores:
        sorted_tickers = sorted(momentum_scores, key=lambda t: momentum_scores[t], reverse=True)
        top_quartile = sorted_tickers[: max(len(sorted_tickers) // 4, 1)]
        mom_returns = []
        for ticker in top_quartile:
            p = prices_cache_fn(ticker, test_start, test_end)
            if not p.empty and len(p) > 1:
                ret = float(p["close"].iloc[-1] / p["close"].iloc[0] - 1)
                mom_returns.append(ret)
        momentum_return = float(np.mean(mom_returns) * 100) if mom_returns else 0.0
    else:
        momentum_return = 0.0

    return {
        "spy_return_pct": spy_return,
        "equal_weight_return_pct": eq_weight_return,
        "momentum_return_pct": momentum_return,
    }


def generate_report(
    result: BacktestResult,
    num_trials: int = 1,
) -> dict:
    """
    Full backtest analytics report with DSR and baseline comparison.
    """
    # Compute return distribution stats for DSR
    all_predictions = [p for w in result.windows for p in w.predictions]

    # Per-window Sharpe variance for DSR
    window_sharpes = [w.sharpe_ratio for w in result.windows if w.sharpe_ratio != 0]
    sr_variance = float(np.var(window_sharpes)) if len(window_sharpes) > 1 else 0.5

    # Return stats from NAV history
    if len(result.nav_history) > 2:
        navs = np.array([n["nav"] for n in result.nav_history])
        daily_returns = np.diff(navs) / navs[:-1]
        skew = float(stats.skew(daily_returns)) if len(daily_returns) > 5 else 0.0
        kurt = float(stats.kurtosis(daily_returns, fisher=False)) if len(daily_returns) > 5 else 3.0
        T = len(daily_returns)
    else:
        skew, kurt, T = 0.0, 3.0, 252

    dsr = compute_deflated_sharpe(
        observed_sr=result.aggregate_sharpe,
        num_trials=max(num_trials, 1),
        variance_of_srs=sr_variance,
        skewness=skew,
        kurtosis=kurt,
        T=T,
    )

    # Top features by MDA (primary) and MDI (secondary)
    mda_sorted = sorted(result.feature_importance_mda.items(), key=lambda x: x[1], reverse=True)[:15]
    mdi_sorted = sorted(result.feature_importance_mdi.items(), key=lambda x: x[1], reverse=True)[:15]

    report = {
        "aggregate": {
            "sharpe_ratio": result.aggregate_sharpe,
            "deflated_sharpe_ratio": dsr,
            "dsr_significant": dsr >= 0.95,
            "total_return_pct": result.aggregate_return_pct,
            "alpha_pct": result.aggregate_alpha_pct,
            "max_drawdown_pct": result.aggregate_max_drawdown_pct,
            "hit_rate": result.aggregate_hit_rate,
            "total_trades": result.total_trades,
            "num_windows": len(result.windows),
            "num_trials": num_trials,
        },
        "per_window": [
            {
                "window_id": w.window_id,
                "train_period": f"{w.train_start} → {w.train_end}",
                "test_period": f"{w.test_start} → {w.test_end}",
                "sharpe_ratio": w.sharpe_ratio,
                "total_return_pct": w.total_return_pct,
                "max_drawdown_pct": w.max_drawdown_pct,
                "hit_rate": w.hit_rate,
                "num_trades": w.num_trades,
            }
            for w in result.windows
        ],
        "feature_importance": {
            "mda_top_15": [{"feature": f, "importance": round(v, 4)} for f, v in mda_sorted],
            "mdi_top_15": [{"feature": f, "importance": round(v, 4)} for f, v in mdi_sorted],
        },
        "nav_history": result.nav_history,
        "strategy_params": result.strategy_params,
    }

    return report
