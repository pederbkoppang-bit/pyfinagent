"""
Backtest analytics — Sharpe, Deflated Sharpe Ratio, baselines, reporting.
All computations are purely numerical, zero LLM cost.
"""

import math
from datetime import datetime

import numpy as np
from scipy import stats

from backend.backtest.backtest_engine import BacktestResult


def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.

    Per Lo (2002) and Sharpe (1994), the √T annualization factor must match
    the actual return observation frequency. Default 252 (daily). Pass
    periods_per_year=12 for monthly, =4 for quarterly, etc.
    """
    if len(returns) < 5:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std()
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


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


def compute_round_trips(all_trades: list[dict]) -> list[dict]:
    """
    Pair BUY→SELL trades into round-trips using FIFO matching by ticker.
    Returns list of dicts with entry/exit prices, P&L, holding days, commission.
    """
    open_buys: dict[str, list[dict]] = {}  # ticker → list of BUY trades (FIFO)
    round_trips: list[dict] = []

    for trade in all_trades:
        ticker = trade.get("ticker", "")
        action = trade.get("action", "")
        if action == "BUY":
            open_buys.setdefault(ticker, []).append(trade)
        elif action == "SELL":
            buys = open_buys.get(ticker, [])
            if not buys:
                continue
            buy = buys.pop(0)  # FIFO
            entry_price = buy.get("price", 0)
            exit_price = trade.get("price", 0)
            quantity = trade.get("quantity", 0)
            commission = buy.get("commission", 0) + trade.get("commission", 0)
            gross_pnl = (exit_price - entry_price) * quantity
            net_pnl = gross_pnl - commission

            # Holding days
            try:
                d_entry = datetime.fromisoformat(buy.get("date", ""))
                d_exit = datetime.fromisoformat(trade.get("date", ""))
                holding_days = (d_exit - d_entry).days
            except (ValueError, TypeError):
                holding_days = 0

            cost_basis = entry_price * quantity
            pnl_pct = (net_pnl / cost_basis) if cost_basis > 0 else 0.0  # decimal ratio: 0.11 = 11%

            round_trips.append({
                "ticker": ticker,
                "entry_date": buy.get("date", ""),
                "exit_date": trade.get("date", ""),
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "quantity": round(quantity, 4),
                "gross_pnl": round(gross_pnl, 2),
                "commission": round(commission, 2),
                "net_pnl": round(net_pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "holding_days": holding_days,
                "probability": round(buy.get("probability", 0), 4),
            })

    return round_trips


def compute_trade_statistics(round_trips: list[dict], avg_nav: float = 100_000.0) -> dict:
    """
    TradingView-style trade performance statistics from round-trip trades.
    Returns dict with profit_factor, win_rate, expectancy, SQN, cost metrics, etc.
    """
    if not round_trips:
        return {}

    pnls = [rt["net_pnl"] for rt in round_trips]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    n_trades = len(round_trips)
    n_wins = len(wins)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    total_commission = sum(rt["commission"] for rt in round_trips)
    total_volume = sum(rt["entry_price"] * rt["quantity"] + rt["exit_price"] * rt["quantity"] for rt in round_trips)

    # Profit factor
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    if profit_factor == float("inf"):
        profit_factor = 99.9  # Cap for display

    # Win/loss averages
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 99.9

    # Expectancy
    expectancy = float(np.mean(pnls))

    # SQN — Van Tharp: sqrt(n) * mean(pnl) / std(pnl)
    pnl_arr = np.array(pnls)
    pnl_std = float(pnl_arr.std()) if n_trades > 1 else 0.0
    sqn = (math.sqrt(n_trades) * expectancy / pnl_std) if pnl_std > 0 else 0.0

    # Best / worst (as decimal ratio for frontend: 0.05 = 5%)
    pnl_pcts = [rt["pnl_pct"] for rt in round_trips if "pnl_pct" in rt]
    if not pnl_pcts:
        # Fallback: compute from cost basis
        pnl_pcts = [
            rt["net_pnl"] / (rt["entry_price"] * rt["quantity"])
            for rt in round_trips
            if rt.get("entry_price", 0) * rt.get("quantity", 0) > 0
        ]
    best_trade = float(max(pnl_pcts)) if pnl_pcts else 0.0
    worst_trade = float(min(pnl_pcts)) if pnl_pcts else 0.0

    # Holding periods
    win_holdings = [rt["holding_days"] for rt in round_trips if rt["net_pnl"] > 0]
    loss_holdings = [rt["holding_days"] for rt in round_trips if rt["net_pnl"] <= 0]
    avg_holding_days_win = float(np.mean(win_holdings)) if win_holdings else 0.0
    avg_holding_days_loss = float(np.mean(loss_holdings)) if loss_holdings else 0.0

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    current_streak = 0
    last_was_win = None
    for p in pnls:
        is_win = p > 0
        if is_win == last_was_win:
            current_streak += 1
        else:
            current_streak = 1
            last_was_win = is_win
        if is_win:
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)

    # Cost metrics
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    commission_pct_of_profit = (total_commission / gross_profit * 100) if gross_profit > 0 else 0.0
    avg_cost_per_trade = total_commission / n_trades if n_trades > 0 else 0.0
    turnover_rate = total_volume / avg_nav if avg_nav > 0 else 0.0
    # Break-even win rate: 1 / (1 + payoff_ratio)
    break_even_win_rate = (1 / (1 + payoff_ratio)) if payoff_ratio > 0 else 1.0

    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": len(losses),
        "profit_factor": round(profit_factor, 2),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "expectancy": round(expectancy, 2),
        "sqn": round(sqn, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "avg_holding_days_win": round(avg_holding_days_win, 1),
        "avg_holding_days_loss": round(avg_holding_days_loss, 1),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "total_commission": round(total_commission, 2),
        "commission_pct_of_profit": round(commission_pct_of_profit, 2),
        "avg_cost_per_trade": round(avg_cost_per_trade, 2),
        "turnover_rate": round(turnover_rate, 2),
        "break_even_win_rate": round(break_even_win_rate, 4),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


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
    Returns total return % and annualized Sharpe for each.
    """
    import pandas as pd

    # 1. SPY baseline
    spy_prices = prices_cache_fn("SPY", test_start, test_end)
    spy_return = 0.0
    spy_sharpe = 0.0
    if not spy_prices.empty and len(spy_prices) > 1:
        spy_close = spy_prices["close"]
        spy_return = float((spy_close.iloc[-1] / spy_close.iloc[0] - 1) * 100)
        spy_daily = np.diff(spy_close.values) / spy_close.values[:-1]
        spy_sharpe = compute_sharpe(spy_daily)

    # 2. Equal-weight all candidates
    # Collect daily close series for each candidate, compute portfolio daily returns
    eq_close_series = []
    for ticker in candidate_tickers[:50]:
        p = prices_cache_fn(ticker, test_start, test_end)
        if not p.empty and len(p) > 1:
            eq_close_series.append(p["close"].values)

    eq_weight_return = 0.0
    eq_weight_sharpe = 0.0
    if eq_close_series:
        # Align to the shortest series length, compute daily returns per stock
        min_len = min(len(s) for s in eq_close_series)
        if min_len > 1:
            stock_returns = []
            for series in eq_close_series:
                trimmed = series[:min_len]
                daily_ret = np.diff(trimmed) / trimmed[:-1]
                stock_returns.append(daily_ret)
            # Equal-weight portfolio daily return = mean across stocks
            portfolio_daily = np.mean(stock_returns, axis=0)
            eq_weight_return = float((np.prod(1 + portfolio_daily) - 1) * 100)
            eq_weight_sharpe = compute_sharpe(portfolio_daily)

    # 3. Momentum-only (top quartile by 6M trailing return)
    momentum_scores = {}
    for ticker in candidate_tickers[:50]:
        lookback_start = (pd.Timestamp(test_start) - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        p_lb = prices_cache_fn(ticker, lookback_start, test_start)
        if not p_lb.empty and len(p_lb) > 20:
            momentum_scores[ticker] = float(p_lb["close"].iloc[-1] / p_lb["close"].iloc[0] - 1)

    momentum_return = 0.0
    momentum_sharpe = 0.0
    if momentum_scores:
        sorted_tickers = sorted(momentum_scores, key=lambda t: momentum_scores[t], reverse=True)
        top_quartile = sorted_tickers[: max(len(sorted_tickers) // 4, 1)]
        mom_close_series = []
        for ticker in top_quartile:
            p = prices_cache_fn(ticker, test_start, test_end)
            if not p.empty and len(p) > 1:
                mom_close_series.append(p["close"].values)

        if mom_close_series:
            min_len = min(len(s) for s in mom_close_series)
            if min_len > 1:
                stock_returns = []
                for series in mom_close_series:
                    trimmed = series[:min_len]
                    daily_ret = np.diff(trimmed) / trimmed[:-1]
                    stock_returns.append(daily_ret)
                portfolio_daily = np.mean(stock_returns, axis=0)
                momentum_return = float((np.prod(1 + portfolio_daily) - 1) * 100)
                momentum_sharpe = compute_sharpe(portfolio_daily)

    return {
        "spy_return_pct": spy_return,
        "spy_sharpe": spy_sharpe,
        "equal_weight_return_pct": eq_weight_return,
        "eq_weight_sharpe": eq_weight_sharpe,
        "momentum_return_pct": momentum_return,
        "momentum_sharpe": momentum_sharpe,
    }


def generate_report(
    result: BacktestResult,
    num_trials: int = 1,
    baselines: dict | None = None,
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
        "analytics": {
            "sharpe": result.aggregate_sharpe,
            "deflated_sharpe": dsr,
            "dsr_significant": dsr >= 0.95,
            "total_return_pct": result.aggregate_return_pct,
            "alpha": result.aggregate_alpha_pct,
            "max_drawdown": result.aggregate_max_drawdown_pct,
            "hit_rate": result.aggregate_hit_rate,
            "n_trades": result.total_trades,
            "n_windows": len(result.windows),
            "information_ratio": 0.0,
            "num_trials": num_trials,
        },
        "per_window": [
            {
                "window_id": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "n_candidates": w.n_candidates,
                "n_train_samples": w.n_train_samples,
                "n_features": w.n_features,
                "sharpe_ratio": w.sharpe_ratio,
                "total_return_pct": w.total_return_pct,
                "max_drawdown_pct": w.max_drawdown_pct,
                "hit_rate": w.hit_rate,
                "num_trades": w.num_trades,
                "feature_importance_mda": dict(
                    sorted(w.feature_importance_mda.items(), key=lambda x: x[1], reverse=True)[:15]
                ) if w.feature_importance_mda else {},
                "feature_importance_mdi": dict(
                    sorted(w.feature_importance_mdi.items(), key=lambda x: x[1], reverse=True)[:15]
                ) if w.feature_importance_mdi else {},
            }
            for w in result.windows
        ],
        "feature_importance": {
            "mda_top_15": [{"feature": f, "importance": round(v, 4)} for f, v in mda_sorted],
            "mdi_top_15": [{"feature": f, "importance": round(v, 4)} for f, v in mdi_sorted],
        },
        "equity_curve": [
            {"date": n.get("date", ""), "equity": n.get("nav", 0)}
            for n in result.nav_history
        ],
        "nav_history": result.nav_history,
        "strategy_params": result.strategy_params,
    }

    # Round-trip trades + trade statistics (always include keys for frontend)
    all_trades = getattr(result, "all_trades", None) or []
    round_trips = compute_round_trips(all_trades) if all_trades else []
    avg_nav = float(np.mean([n["nav"] for n in result.nav_history])) if result.nav_history else result.strategy_params.get("starting_capital", 100_000)
    trade_stats = compute_trade_statistics(round_trips, avg_nav) if round_trips else {}
    report["trades"] = round_trips
    report["trade_statistics"] = trade_stats

    # Add baselines if provided
    if baselines:
        report["baselines"] = {
            "spy": {
                "total_return_pct": baselines.get("spy_return_pct", 0),
                "sharpe": baselines.get("spy_sharpe", 0),
            },
            "equal_weight": {
                "total_return_pct": baselines.get("equal_weight_return_pct", 0),
                "sharpe": baselines.get("eq_weight_sharpe", 0),
            },
            "momentum": {
                "total_return_pct": baselines.get("momentum_return_pct", 0),
                "sharpe": baselines.get("momentum_sharpe", 0),
            },
        }

    return report
