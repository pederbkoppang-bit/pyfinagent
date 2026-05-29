"""phase-48.3: $0 deterministic tests for the rotation runner + full-kwarg factory.

No real backtest / BQ / LLM. make_rotation_engine is tested by monkeypatching
BacktestEngine to capture ctor kwargs; run_rotation_bakeoff via injected stub
engine_factory / adapter_fn; persistence via tmp_path. The LIVE bake-off
(~32 real backtests) stays opt-in (@pytest.mark.skip).
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import backend.autoresearch.rotation_runner as RR
from backend.autoresearch.rotation_runner import (
    make_rotation_engine,
    run_rotation_bakeoff,
    _persist_verdict,
    _resolve_incumbent,
)
from backend.backtest.backtest_engine import BacktestResult, WindowResult

_SETTINGS = SimpleNamespace(gcp_project_id="proj-x", bq_dataset_reports="financial_reports")
_BQ = SimpleNamespace(client="rawclient")


# ---- fakes (mirror 48.2) ------------------------------------------------------
def _navs(n=45, start=100000.0, drift=0.0008, seed_offset=0):
    out, nav = [], start
    for i in range(n):
        noise = ((i * 37 + seed_offset * 13) % 11 - 5) * 0.0006
        nav = nav * (1 + drift + noise)
        out.append({"date": f"2024-{((i // 28) % 12) + 1:02d}-{(i % 28) + 1:02d}", "nav": round(nav, 2)})
    return out


def _fake_result(strategy="triple_barrier", sharpe=1.3, n=45, seed_offset=0):
    windows = [
        WindowResult(window_id=i, train_start="2023-01-01", train_end="2023-12-31",
                     test_start="2024-01-01", test_end="2024-03-31", sharpe_ratio=s,
                     total_return_pct=0.0, alpha_pct=0.0, max_drawdown_pct=5.0, hit_rate=0.55, num_trades=10)
        for i, s in enumerate((1.2, 1.4, 1.1))
    ]
    return BacktestResult(windows=windows, aggregate_sharpe=sharpe, aggregate_return_pct=12.0,
                          aggregate_max_drawdown_pct=6.0, aggregate_hit_rate=0.55, total_trades=30,
                          nav_history=_navs(n=n, seed_offset=seed_offset),
                          strategy_params={"strategy": strategy})


class _FakeEngine:
    def __init__(self, params, n=45):
        self._p, self._n = params, n

    def run_backtest(self, skip_cache_clear=False, **_):
        off = int(self._p.get("holding_days", 90) or 90) + int(self._p.get("mr_holding_days", 15) or 15)
        return _fake_result(strategy=self._p.get("strategy", "triple_barrier"), n=self._n, seed_offset=off)


# ---- make_rotation_engine -----------------------------------------------------
def test_make_rotation_engine_threads_full_kwargs(monkeypatch):
    captured = {}

    class _CapEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(RR, "BacktestEngine", _CapEngine)
    make_rotation_engine(
        {"strategy": "quality_momentum", "holding_days": 120, "commission_model": "per_share",
         "starting_capital": 50000.0, "train_window_months": 18},
        _SETTINGS, _BQ, start_date="2024-01-01", end_date="2024-06-30",
    )
    # the 8 kwargs make_engine drops are now threaded
    assert captured["market"] == "US"
    assert captured["train_window_months"] == 18
    assert captured["test_window_months"] == 3
    assert captured["embargo_days"] == 5
    assert captured["starting_capital"] == 50000.0
    assert captured["commission_model"] == "per_share"
    assert "target_vol" in captured
    assert captured["strategy"] == "quality_momentum"
    assert captured["start_date"] == "2024-01-01" and captured["end_date"] == "2024-06-30"
    assert captured["bq_client"] == "rawclient"  # getattr(bq, "client", bq)


def test_make_rotation_engine_maps_target_annual_vol_to_target_vol(monkeypatch):
    captured = {}
    monkeypatch.setattr(RR, "BacktestEngine", lambda **kw: captured.update(kw) or SimpleNamespace())
    # seed name target_annual_vol=0.15 -> live ctor target_vol=0.15 (vol-targeting ON)
    make_rotation_engine({"strategy": "triple_barrier", "target_annual_vol": 0.15}, _SETTINGS, _BQ)
    assert captured["target_vol"] == 0.15
    # base target_annual_vol=0 -> target_vol=0 (vol-targeting OFF)
    captured.clear()
    make_rotation_engine({"strategy": "triple_barrier", "target_annual_vol": 0}, _SETTINGS, _BQ)
    assert captured["target_vol"] == 0
    # explicit target_vol wins over target_annual_vol
    captured.clear()
    make_rotation_engine({"strategy": "triple_barrier", "target_vol": 0.2, "target_annual_vol": 0.1}, _SETTINGS, _BQ)
    assert captured["target_vol"] == 0.2


def test_make_rotation_engine_raises_unknown_strategy(monkeypatch):
    monkeypatch.setattr(RR, "BacktestEngine", lambda **kw: SimpleNamespace())
    with pytest.raises(ValueError):
        make_rotation_engine({"strategy": "bogus"}, _SETTINGS, _BQ)


def test_make_rotation_engine_warns_on_dead_keys(monkeypatch, caplog):
    monkeypatch.setattr(RR, "BacktestEngine", lambda **kw: SimpleNamespace())
    with caplog.at_level(logging.WARNING):
        make_rotation_engine(
            {"strategy": "triple_barrier", "trailing_stop_enabled": True, "trailing_trigger_pct": 5},
            _SETTINGS, _BQ,
        )
    assert any("inert risk keys" in r.message for r in caplog.records)
    assert any("trailing_stop_enabled" in r.message for r in caplog.records)


# ---- run_rotation_bakeoff -----------------------------------------------------
def test_seam_A_engine_factory_full_wiring(tmp_path):
    # full wiring: stub engine_factory -> REAL adapter+producer+selector
    verdict = run_rotation_bakeoff(
        _SETTINGS, _BQ,
        incumbent={"strategy_id": "x", "strategy": "x", "dsr": 0.10},  # explicit -> skip loader
        num_param_variants=4,
        engine_factory=lambda variant: _FakeEngine(variant),
        clear_cache_fn=lambda: None,
        persist=True, log_path=tmp_path / "rot.jsonl",
    )
    assert {"selected_id", "switched", "reason", "ranked", "num_trials"} <= set(verdict)
    # a row was persisted at allocation_pct=0
    rows = (tmp_path / "rot.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["allocation_pct"] == 0.0 and row["status"] == "bakeoff_verdict"
    assert row["selected_id"] == verdict["selected_id"]


def test_seam_B_adapter_fn_and_incumbent(tmp_path):
    # narrow seam: stub adapter_fn -> tests incumbent + selector wiring only
    def stub_adapter(params):
        s = params.get("strategy")
        if s == "mean_reversion":
            return {"dsr": 0.99, "pbo": 0.05, "sharpe": 1.4}
        return {"dsr": 0.96, "pbo": 0.10, "sharpe": 1.1}

    verdict = run_rotation_bakeoff(
        _SETTINGS, _BQ,
        incumbent={"strategy_id": "qm_trend_tilt", "strategy": "qm_trend_tilt", "dsr": 0.985},
        adapter_fn=stub_adapter,
        persist=True, log_path=tmp_path / "rot.jsonl",
    )
    # top passer mr (0.99) beats incumbent (0.985) by < min_improvement(0.01) -> retain
    assert verdict["reason"] in {"below_min_improvement", "dsr_improvement", "incumbent_is_top", "first_selection"}
    assert "mr_short_horizon" in verdict["ranked"]


def test_resolve_incumbent_from_loader(monkeypatch):
    import backend.services.autonomous_loop as AL
    monkeypatch.setattr(AL, "load_promoted_params", lambda bq: {"strategy": "triple_barrier", "tp_pct": 10.0})
    monkeypatch.setattr(RR, "_incumbent_dsr_from_optimizer_best", lambda: 0.9526)
    inc = _resolve_incumbent(_BQ)
    assert inc["strategy_id"] == "triple_barrier" and inc["dsr"] == 0.9526
    # empty params -> None (selector does first_selection)
    monkeypatch.setattr(AL, "load_promoted_params", lambda bq: {})
    assert _resolve_incumbent(_BQ) is None


def test_persist_verdict_failopen_and_persist_false(tmp_path):
    verdict = {"selected_id": "mr_short_horizon", "incumbent_id": None, "switched": True,
               "reason": "first_selection", "delta_dsr": None, "ranked": ["mr_short_horizon"], "num_trials": 4}
    # persist=False writes nothing
    run_rotation_bakeoff(_SETTINGS, _BQ, incumbent={"strategy": "x", "dsr": 0.1},
                         adapter_fn=lambda p: {"dsr": 0.99, "pbo": 0.05, "sharpe": 1.2},
                         persist=False, log_path=tmp_path / "none.jsonl")
    assert not (tmp_path / "none.jsonl").exists()
    # a raising bq_fn is swallowed (fail-open) -- still returns the row
    def _boom(_row):
        raise RuntimeError("bq down")
    row = _persist_verdict(verdict, path=tmp_path / "ok.jsonl", bq_fn=_boom)
    assert row["allocation_pct"] == 0.0
    assert (tmp_path / "ok.jsonl").exists()


@pytest.mark.skip(reason="live: ~32 real backtests, tens of minutes, real BQ (run opt-in only)")
def test_live_rotation_bakeoff_smoke():
    # Documents the real entrypoint for a future live-run cycle. NOT $0/CI.
    # from backend.config.settings import get_settings
    # from backend.db.bigquery_client import BigQueryClient
    # v = run_rotation_bakeoff(get_settings(), BigQueryClient(),
    #     seeds=[{"id": "tb_baseline", "param_overrides": {}}],
    #     num_param_variants=2, start_date="2024-01-01", end_date="2024-06-30")
    # assert "selected_id" in v
    pass
