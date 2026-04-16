"""
PerformanceSkill — Single source of truth for all P&L and portfolio metrics.

Every optimizer loop, API endpoint, and paper-trading module should delegate
to these canonical formulas instead of computing independently.

Scalar metric for optimization:
    get_scalar_metric() = risk_adjusted_return × (1 − min(0.3, turnover_ratio × tx_cost_pct))

Research basis:
    - analytics.py Sharpe (risk-free adjusted, √252 annualized) is THE Sharpe formula
    - López de Prado Ch. 3-8 for backtest-specific metrics (DSR, sample weights)
    - Transaction cost penalty prevents over-trading (not in Karpathy — our hybrid extension)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from backend.backtest.analytics import compute_sharpe, compute_max_drawdown

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant, used in DSR threshold (Bailey & Lopez de Prado 2014, Eq. 8).
_EULER_GAMMA = 0.5772156649015329


# ── Position-Level Metrics ───────────────────────────────────────


def compute_position_pnl(
    quantity: float, current_price: float, cost_basis: float
) -> tuple[float, float]:
    """
    Canonical position P&L.

    Returns:
        (unrealized_pnl, unrealized_pnl_pct)
    """
    market_value = quantity * current_price
    pnl = market_value - cost_basis
    pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
    return round(pnl, 2), round(pnl_pct, 2)


def compute_return_pct(
    current_price: float, entry_price: float
) -> float:
    """Simple return percentage: ((current - entry) / entry) × 100."""
    if entry_price <= 0:
        return 0.0
    return round(((current_price - entry_price) / entry_price) * 100, 2)


# ── Portfolio-Level Metrics ──────────────────────────────────────


def compute_portfolio_pnl(
    nav: float, starting_capital: float
) -> tuple[float, float]:
    """
    Portfolio-level P&L.

    Returns:
        (total_pnl, total_pnl_pct)
    """
    pnl = nav - starting_capital
    pnl_pct = (pnl / starting_capital * 100) if starting_capital > 0 else 0.0
    return round(pnl, 2), round(pnl_pct, 2)


def compute_alpha(
    portfolio_pnl_pct: float, benchmark_pnl_pct: float
) -> float:
    """Alpha = portfolio cumulative return − benchmark cumulative return."""
    return round(portfolio_pnl_pct - benchmark_pnl_pct, 2)


def compute_sharpe_from_snapshots(
    snapshots: list[dict],
    nav_key: str = "total_nav",
    risk_free_rate: float = 0.04,
) -> float:
    """
    Compute Sharpe ratio from daily NAV snapshots.

    Converts NAV series → daily returns, then delegates to the canonical
    analytics.compute_sharpe (risk-free adjusted, √252 annualized).
    """
    if len(snapshots) < 6:
        return 0.0

    navs = [s.get(nav_key, 0) for s in snapshots if s.get(nav_key)]
    if len(navs) < 6:
        return 0.0

    nav_arr = np.array(navs, dtype=float)
    # Daily returns from NAV series
    daily_returns = np.diff(nav_arr) / nav_arr[:-1]

    return round(compute_sharpe(daily_returns, risk_free_rate), 2)


# ── Benchmark Comparison ─────────────────────────────────────────


def compute_benchmark_return(holding_days: int, annual_rate: float = 0.10) -> float:
    """
    Expected benchmark return for a holding period.

    Uses geometric compounding (not simple arithmetic) for accuracy
    over long holding periods.

    Args:
        holding_days: Number of calendar days held.
        annual_rate: Assumed annual benchmark return (default 10% for SPY).

    Returns:
        Expected return percentage.
    """
    if holding_days <= 0:
        return 0.0
    daily_rate = (1 + annual_rate) ** (1 / 365) - 1
    return round(((1 + daily_rate) ** holding_days - 1) * 100, 2)


def beat_benchmark(
    return_pct: float, holding_days: int, annual_rate: float = 0.10
) -> bool:
    """Did the position beat the geometric benchmark return?"""
    return return_pct > compute_benchmark_return(holding_days, annual_rate)


# ── Turnover & Transaction Costs ─────────────────────────────────


def compute_turnover_ratio(
    trades: list[dict],
    avg_nav: float,
    period_days: int = 365,
) -> float:
    """
    Annual portfolio turnover = total sell value / average NAV.

    Higher turnover → higher transaction cost drag.
    """
    if avg_nav <= 0 or period_days <= 0:
        return 0.0
    sell_value = sum(
        abs(t.get("total_value", 0))
        for t in trades
        if t.get("action") == "SELL"
    )
    # Annualize
    annualized = sell_value * (365 / period_days)
    return round(annualized / avg_nav, 4)


def compute_tx_cost_drag(
    turnover_ratio: float, tx_cost_pct: float = 0.001
) -> float:
    """
    Transaction cost drag as a fraction of returns.

    Capped at 0.3 (30%) to prevent degenerate values.
    """
    return min(0.3, turnover_ratio * tx_cost_pct)


# ── Scalar Optimization Metric ───────────────────────────────────


@dataclass
class ScalarMetricInputs:
    """Inputs for the unified scalar metric."""
    avg_return_pct: float
    benchmark_beat_rate: float
    turnover_ratio: float = 0.0
    tx_cost_pct: float = 0.001  # 0.1% default


def get_scalar_metric(inputs: ScalarMetricInputs) -> float:
    """
    THE single metric all optimization loops converge on.

    scalar = risk_adjusted_return × (1 − tx_cost_drag)

    where:
        risk_adjusted_return = avg(return_pct) × beat_benchmark_rate
        tx_cost_drag = min(0.3, turnover_ratio × tx_cost_pct)

    This extends Karpathy's single-metric approach with a transaction cost
    penalty that prevents the optimizer from discovering "churn alpha" —
    strategies that look good on paper but lose to friction in practice.
    """
    risk_adjusted = inputs.avg_return_pct * inputs.benchmark_beat_rate
    drag = compute_tx_cost_drag(inputs.turnover_ratio, inputs.tx_cost_pct)
    return round(risk_adjusted * (1 - drag), 4)


def get_scalar_metric_from_bq(bq_client, trades: Optional[list] = None) -> float:
    """
    Convenience: compute scalar metric from BigQuery performance stats
    and optional trade history.
    """
    stats = bq_client.get_performance_stats()
    avg_return = stats.get("avg_return") or 0.0
    benchmark_rate = stats.get("benchmark_beat_rate") or 0.0

    turnover = 0.0
    if trades:
        # Rough NAV estimate from trade values
        total_value = sum(abs(t.get("total_value", 0)) for t in trades)
        avg_nav = total_value / max(len(trades), 1)
        if avg_nav > 0:
            turnover = compute_turnover_ratio(trades, avg_nav)

    return get_scalar_metric(ScalarMetricInputs(
        avg_return_pct=avg_return,
        benchmark_beat_rate=benchmark_rate,
        turnover_ratio=turnover,
    ))


# ── Advanced Risk-Adjusted Metrics (PSR, DSR, Sortino, Calmar) ───


def _sr_per_period(returns: np.ndarray, ddof: int = 1) -> float:
    """Non-annualized per-period Sharpe. PSR/DSR require this form, not annualized."""
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=ddof))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def compute_psr(returns: Sequence[float], sr_star: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012, Eq. 9).

    PSR(SR*) = CDF((SR_hat - SR*) * sqrt(n-1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2))

    Where g4 is RAW kurtosis (normal = 3). scipy's default is excess kurtosis --
    we add 3. Denominator guarded to avoid division-by-zero at extreme SR.
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < 5:
        return 0.0
    from scipy.stats import norm, skew, kurtosis
    sr_hat = _sr_per_period(arr)
    g3 = float(skew(arr, bias=False))
    g4 = float(kurtosis(arr, fisher=True, bias=False)) + 3.0  # excess -> raw
    var_term = 1.0 - g3 * sr_hat + (g4 - 1.0) / 4.0 * sr_hat * sr_hat
    var_term = max(var_term, 1e-8)
    z = (sr_hat - sr_star) * math.sqrt(n - 1) / math.sqrt(var_term)
    return float(norm.cdf(z))


def compute_dsr(
    returns: Sequence[float],
    all_trial_sharpes: Sequence[float],
    n_trials: Optional[int] = None,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, Eq. 8 + Eq. 10).

    Corrects PSR for selection bias when N strategies were tested. The threshold
    SR* is derived from the cross-sectional variance of trial Sharpes and the
    expected maximum of N draws from that distribution.

    SR* = sqrt(Var[SR]) * ((1-gamma)*PPF(1-1/N) + gamma*PPF(1-1/(N*e)))
    DSR = PSR(SR*)
    """
    arr = np.asarray(list(returns), dtype=float)
    trials = np.asarray(list(all_trial_sharpes), dtype=float)
    n = n_trials if n_trials is not None else max(len(trials), 2)
    if len(arr) < 5 or n < 2 or len(trials) < 2:
        return 0.0
    from scipy.stats import norm
    var_sr = float(trials.var(ddof=1))
    if var_sr <= 0.0:
        return compute_psr(arr, sr_star=0.0)
    sr_star = math.sqrt(var_sr) * (
        (1.0 - _EULER_GAMMA) * float(norm.ppf(1.0 - 1.0 / n))
        + _EULER_GAMMA * float(norm.ppf(1.0 - 1.0 / (n * math.e)))
    )
    return compute_psr(arr, sr_star=sr_star)


def compute_sortino(
    returns: Sequence[float], mar: float = 0.0, periods_per_year: int = 252
) -> float:
    """Sortino ratio, annualized. MAR default 0 (downside = negative returns)."""
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 5:
        return 0.0
    excess = arr - mar
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return 0.0
    dstd = float(downside.std(ddof=1))
    if dstd == 0.0:
        return 0.0
    return float(excess.mean() / dstd * math.sqrt(periods_per_year))


def compute_calmar(
    returns: Sequence[float], periods_per_year: int = 252
) -> float:
    """Calmar = annualized return / |max drawdown|. Returns 0 if no drawdown."""
    arr = np.asarray(list(returns), dtype=float)
    if len(arr) < 5:
        return 0.0
    cum = np.cumprod(1.0 + arr)
    dd_pct = compute_max_drawdown(cum)
    if dd_pct >= 0.0:
        return 0.0
    annualized = float(arr.mean() * periods_per_year)
    return float(annualized / (abs(dd_pct) / 100.0))


def compute_rolling_sharpe_bootstrap_ci(
    returns: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.04,
    seed: Optional[int] = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI on annualized Sharpe. Percentile method by default; falls
    back to stationary block bootstrap when |lag-1 autocorr| > 0.2 to preserve
    serial dependence (Politis & Romano 1994).

    Returns (sharpe_point, ci_low, ci_high).
    """
    arr = np.asarray(list(returns), dtype=float)
    n = len(arr)
    if n < 10:
        return 0.0, 0.0, 0.0

    point = compute_sharpe(arr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)

    # Lag-1 autocorr check
    mean = arr.mean()
    centered = arr - mean
    denom = float((centered * centered).sum())
    if denom == 0.0:
        return point, point, point
    acf1 = float((centered[:-1] * centered[1:]).sum() / denom)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=float)

    if abs(acf1) <= 0.2:
        # Standard IID bootstrap
        idx = rng.integers(0, n, size=(n_resamples, n))
        for i in range(n_resamples):
            samples[i] = compute_sharpe(arr[idx[i]], risk_free_rate, periods_per_year)
    else:
        # Stationary block bootstrap: geometric block lengths with mean ~ sqrt(n)
        p = 1.0 / max(1.0, math.sqrt(n))
        for i in range(n_resamples):
            resample = np.empty(n, dtype=float)
            t = 0
            start = int(rng.integers(0, n))
            while t < n:
                resample[t] = arr[start % n]
                if rng.random() < p:
                    start = int(rng.integers(0, n))
                else:
                    start += 1
                t += 1
            samples[i] = compute_sharpe(resample, risk_free_rate, periods_per_year)

    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(samples, alpha))
    high = float(np.quantile(samples, 1.0 - alpha))
    return point, low, high
