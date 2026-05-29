"""phase-48.2: $0 mock tests for the rotation real-engine adapter.

Mocks ONLY engine.run_backtest (returns a hand-built REAL BacktestResult) and lets
the REAL pure-numpy generate_report + compute_pbo run on it -- zero backtest, zero
BQ, zero LLM, zero macro preload. Proves the metric-extraction wiring, the
load-bearing undersize-matrix guard (no false-good 0.0), the strategy-name reject,
warm-cache clear-once discipline, and the end-to-end registry->adapter->producer->
selector composition.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.backtest.analytics import compute_pbo, generate_report
from backend.backtest.backtest_engine import BacktestResult, WindowResult, STRATEGY_REGISTRY
from backend.autoresearch.strategy_backtest_adapter import (
    make_engine_backtest_fn,
    _assemble_pbo_matrix,
    _daily_returns_from_nav,
    _default_param_grid,
    _extract_dsr_sharpe,
)

_BASE = {
    "strategy": "triple_barrier",
    "tp_pct": 10.0,
    "sl_pct": 12.0,
    "holding_days": 90,
    "mr_holding_days": 15,
    "max_positions": 20,
}


def _navs(n: int = 45, start: float = 100000.0, drift: float = 0.0008, seed_offset: int = 0) -> list[dict]:
    """Deterministic rising-then-noisy nav series of length n (no RNG)."""
    out, nav = [], start
    for i in range(n):
        noise = ((i * 37 + seed_offset * 13) % 11 - 5) * 0.0006
        nav = nav * (1 + drift + noise)
        out.append({"date": f"2024-{((i // 28) % 12) + 1:02d}-{(i % 28) + 1:02d}", "nav": round(nav, 2), "cash": 0.0})
    return out


def _make_fake_result(strategy="triple_barrier", aggregate_sharpe=1.3, n=45, seed_offset=0,
                      window_sharpes=(1.2, 1.4, 1.1)) -> BacktestResult:
    windows = [
        WindowResult(
            window_id=i, train_start="2023-01-01", train_end="2023-12-31",
            test_start="2024-01-01", test_end="2024-03-31", sharpe_ratio=s,
            total_return_pct=0.0, alpha_pct=0.0, max_drawdown_pct=5.0, hit_rate=0.55, num_trades=10,
        )
        for i, s in enumerate(window_sharpes)
    ]
    return BacktestResult(
        windows=windows, aggregate_sharpe=aggregate_sharpe, aggregate_return_pct=12.0,
        aggregate_max_drawdown_pct=6.0, aggregate_hit_rate=0.55, total_trades=30,
        nav_history=_navs(n=n, seed_offset=seed_offset),
        strategy_params={"strategy": strategy, "starting_capital": 100000},
    )


class _FakeEngine:
    """Stub engine: run_backtest returns a fake result whose nav series varies by params."""

    def __init__(self, params: dict, n: int = 45):
        self._params, self._n = params, n

    def run_backtest(self, skip_cache_clear: bool = False, **_):
        off = int(self._params.get("holding_days", 90) or 90) + int(self._params.get("mr_holding_days", 15) or 15)
        return _make_fake_result(strategy=self._params.get("strategy", "triple_barrier"),
                                 n=self._n, seed_offset=off)


# ---- pure-helper tests --------------------------------------------------------
def test_daily_returns_from_nav():
    assert _daily_returns_from_nav(_navs(n=10)).shape == (9,)
    assert _daily_returns_from_nav([]).size == 0
    assert _daily_returns_from_nav([{"nav": 1.0}, {"nav": 2.0}]).size == 0  # <3 rows
    assert _daily_returns_from_nav([{"nav": 0.0}, {"nav": 1.0}, {"nav": 2.0}]).size == 0  # zero divisor


def test_default_param_grid_validates_and_holds_strategy_fixed():
    grid = _default_param_grid(_BASE, 8)
    assert len(grid) == 8
    assert grid[0] == _BASE                                   # variant 0 == seed
    assert all(g["strategy"] == "triple_barrier" for g in grid)  # categorical fixed
    assert all(30 <= g["holding_days"] <= 252 for g in grid)
    mr = _default_param_grid({"strategy": "mean_reversion", "mr_holding_days": 8}, 6)
    assert all(5 <= g["mr_holding_days"] <= 30 for g in mr)
    with pytest.raises(ValueError):                           # unknown -> raise (no silent fallback)
        _default_param_grid({"strategy": "bogus"}, 4)


def test_extract_dsr_sharpe_matches_generate_report():
    res = _make_fake_result(aggregate_sharpe=1.5)
    dsr, sharpe = _extract_dsr_sharpe(res, num_trials=8)
    rep = generate_report(res, num_trials=8)
    assert dsr == rep["analytics"]["deflated_sharpe"]
    assert 0.0 <= dsr <= 1.0
    assert sharpe == 1.5 == res.aggregate_sharpe


def test_compute_pbo_happy_path_hand_matrix():
    mat = np.array([[((i * 7 + j * 13) % 11 - 5) * 0.001 for j in range(4)] for i in range(40)])
    pbo = compute_pbo(mat, S=16)
    assert isinstance(pbo, float) and 0.0 <= pbo <= 1.0


def test_assemble_pbo_matrix_guard():
    # too short (T<32) -> None
    assert _assemble_pbo_matrix([_make_fake_result(n=10, seed_offset=i) for i in range(4)], min_rows=32) is None
    # <2 usable columns -> None
    assert _assemble_pbo_matrix([_make_fake_result(n=45), BacktestResult(nav_history=[])], min_rows=32) is None
    # healthy -> (T>=32, N=4)
    m = _assemble_pbo_matrix([_make_fake_result(n=45, seed_offset=i) for i in range(4)], min_rows=32)
    assert m is not None and m.ndim == 2 and m.shape[1] == 4 and m.shape[0] >= 32


# ---- adapter / boundary tests -------------------------------------------------
def test_adapter_emits_full_metrics_and_clears_cache_once():
    calls = {"clear": 0}
    fn = make_engine_backtest_fn(
        lambda p: _FakeEngine(p), num_param_variants=4,
        clear_cache_fn=lambda: calls.__setitem__("clear", calls["clear"] + 1),
    )
    out = fn(dict(_BASE))
    assert {"dsr", "pbo", "sharpe"} <= set(out)
    assert isinstance(out["dsr"], float) and isinstance(out["pbo"], float) and isinstance(out["sharpe"], float)
    assert 0.0 <= out["dsr"] <= 1.0 and 0.0 <= out["pbo"] <= 1.0
    assert out["n_variants"] == 4
    assert calls["clear"] == 1  # warm-cache discipline: cleared exactly once per bake-off


def test_adapter_undersize_matrix_emits_no_pbo_so_producer_skips():
    from backend.autoresearch.strategy_candidate_producer import build_per_strategy_candidates
    # short nav -> matrix undersized -> adapter omits pbo -> producer SKIPS (no false-good 0.0)
    short_fn = make_engine_backtest_fn(
        lambda p: _FakeEngine(p, n=10), num_param_variants=4, clear_cache_fn=lambda: None,
    )
    raw = short_fn(dict(_BASE))
    assert "pbo" not in raw and "dsr" in raw  # the guard fired
    cands = build_per_strategy_candidates([{"id": "tb", "params": dict(_BASE)}], short_fn)
    assert cands == []  # producer skipped the pbo-less candidate


def test_adapter_unknown_strategy_raises_and_producer_skips():
    from backend.autoresearch.strategy_candidate_producer import build_per_strategy_candidates
    fn = make_engine_backtest_fn(lambda p: _FakeEngine(p), num_param_variants=4, clear_cache_fn=lambda: None)
    with pytest.raises(ValueError):
        fn({"strategy": "bogus"})
    cands = build_per_strategy_candidates([{"id": "bad", "params": {"strategy": "bogus"}}], fn)
    assert cands == []  # producer caught the raise + skipped


def test_end_to_end_registry_adapter_producer_selector():
    from backend.autoresearch.strategy_registry import load_seed_strategies
    from backend.autoresearch.strategy_candidate_producer import build_per_strategy_candidates
    from backend.autoresearch.strategy_selector import select_best_strategy

    fn = make_engine_backtest_fn(lambda p: _FakeEngine(p), num_param_variants=4, clear_cache_fn=lambda: None)
    configs = load_seed_strategies(base_params=_BASE)  # 4 seeds: tb / mr / qm / tb_risk_managed
    cands = build_per_strategy_candidates(configs, fn)
    assert len(cands) == 4  # all 4 seed strategies are valid + produce healthy matrices
    for c in cands:
        assert set(c.keys()) == {"strategy", "dsr", "pbo", "params", "sharpe"}
        assert isinstance(c["dsr"], float) and isinstance(c["pbo"], float)
    verdict = select_best_strategy(cands, incumbent=None, num_trials=len(configs))
    assert "selected_id" in verdict and "reason" in verdict and "ranked" in verdict


@pytest.mark.skip(reason="live: multi-minute, real BQ + macro preload (run opt-in only, not $0/CI)")
def test_live_adapter_one_seed():
    # Documents the REAL wiring for a future live-run cycle. NOT run by default.
    # from backend.config.settings import get_settings
    # from backend.db.bigquery_client import BigQueryClient
    # from backend.backtest.cache import clear_cache
    # from scripts.harness.run_harness import make_engine  # constructor-kwargs factory
    # settings, bq = get_settings(), BigQueryClient()
    # factory = lambda params: make_engine(params, settings, bq)
    # fn = make_engine_backtest_fn(factory, num_param_variants=2, clear_cache_fn=clear_cache)
    # out = fn({"strategy": "triple_barrier", "holding_days": 90, "tp_pct": 10.0, "sl_pct": 12.0})
    # assert isinstance(out["dsr"], float) and isinstance(out["pbo"], float)
    pass
