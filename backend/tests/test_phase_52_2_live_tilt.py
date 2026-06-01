"""phase-52.2: live 52-week-high tilt -- byte-identity (OFF) + faithful-to-52.1 (ON).

Pins: with the flag OFF (default) rank_candidates is byte-identical (no composite_score_raw
written, identical order); with the flag ON the ranking tilts toward 52w-high proximity AND
matches the phase-52.1 replay's hi52_tilt_basket EXACTLY (so the live ranking == what 52.1 measured).
$0, no network.
"""
import importlib.util
import pathlib

import pytest

from backend.tools.screener import rank_candidates

# 52.1 reference logic lives in the replay (scripts/, not a package) -> load by path.
_RP = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "sector_neutral_replay.py"
_spec = importlib.util.spec_from_file_location("snr_uut_522", _RP)
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def _row(t, mom, pct):
    return {"ticker": t, "sector": "Tech", "momentum_1m": mom, "momentum_3m": mom,
            "momentum_6m": mom, "rsi_14": 50, "volatility_ann": 0.3,
            "sma_50_distance_pct": 0.0, "pct_to_52w_high": pct}


def _data():
    return [_row("T1", 0.30, 0.99), _row("T2", 0.28, 0.70), _row("T3", 0.26, 0.95),
            _row("T4", 0.24, 0.60), _row("T5", 0.22, 0.98)]


def test_flag_off_is_byte_identical_default():
    a = [r["ticker"] for r in rank_candidates(_data(), top_n=5, strategy="momentum")]
    b = [r["ticker"] for r in rank_candidates(_data(), top_n=5, strategy="momentum", momentum_52wh_tilt=False)]
    assert a == b == ["T1", "T2", "T3", "T4", "T5"]  # pure-composite order


def test_flag_off_writes_no_raw_field():
    # composite_score_raw is written ONLY by a re-scoring pass -> its absence witnesses
    # the OFF path never touched scored.
    out = rank_candidates(_data(), top_n=5, strategy="momentum")
    assert all("composite_score_raw" not in r for r in out)


def test_flag_on_tilts_toward_52w_high():
    base = [r["ticker"] for r in rank_candidates(_data(), top_n=3, strategy="momentum")]
    tilt = [r["ticker"] for r in rank_candidates(_data(), top_n=3, strategy="momentum",
                                                 momentum_52wh_tilt=True, momentum_52wh_tilt_k=1.0)]
    assert base == ["T1", "T2", "T3"]
    assert tilt != base                       # the tilt changed the basket
    assert "T2" not in tilt and "T3" in tilt   # near-high T3 displaced far-from-high T2


def test_live_tilt_matches_52_1_replay_logic():
    # the LIVE tilt basket must equal the 52.1 replay's hi52_tilt_basket on the same data
    ranked_all = rank_candidates(_data(), top_n=len(_data()), strategy="momentum")  # production composites
    replay_basket = replay.hi52_tilt_basket(ranked_all, k=1.0, top_n=3)
    live_basket = [r["ticker"] for r in rank_candidates(_data(), top_n=3, strategy="momentum",
                                                        momentum_52wh_tilt=True, momentum_52wh_tilt_k=1.0)]
    assert live_basket == replay_basket


def test_missing_pct_is_noop_for_that_name():
    rows = _data() + [_row("NOPCT", 0.29, None)]  # 2nd-highest composite, no 52wh data
    out = [r["ticker"] for r in rank_candidates(rows, top_n=6, strategy="momentum",
                                                momentum_52wh_tilt=True, momentum_52wh_tilt_k=1.0)]
    assert "NOPCT" in out  # tilt 1.0 -> ranked on its raw composite, not dropped/crashed
