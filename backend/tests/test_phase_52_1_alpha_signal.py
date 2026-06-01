"""phase-52.1: 52-week-high momentum tilt -- mechanism + feature tests.

Tests the centered multiplicative tilt logic (hi52_tilt_basket) and the price-only
52w-high feature (build_screen_row.pct_to_52w_high), deterministically. The empirical
ON-vs-OFF measurement is the replay's job (live_check_52.1.md); these pin the math.

$0, no network. The replay lives under scripts/ (not an importable package) -> load by path.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd

_RP_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "sector_neutral_replay.py"
_spec = importlib.util.spec_from_file_location("sector_neutral_replay_uut", _RP_PATH)
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def _row(ticker, composite, pct):
    return {"ticker": ticker, "composite_score": composite, "pct_to_52w_high": pct}


def test_tilt_breaks_tie_toward_higher_52wh():
    # equal composite -> the name nearer its 52w high must rank first after the tilt
    rows = [_row("FAR", 1.0, 0.70), _row("NEAR", 1.0, 0.95)]
    out = replay.hi52_tilt_basket(rows, k=1.0, top_n=2)
    assert out == ["NEAR", "FAR"]


def test_tilt_is_centered_no_change_when_all_equal_pct():
    # all at the same pct (== mean) -> tilt == 1.0 for all -> composite order preserved
    rows = [_row("A", 3.0, 0.8), _row("B", 2.0, 0.8), _row("C", 1.0, 0.8)]
    out = replay.hi52_tilt_basket(rows, k=1.0, top_n=3)
    assert out == ["A", "B", "C"]


def test_tilt_cannot_overturn_a_large_composite_gap_at_small_k():
    # a gentle tilt (k=0.5) should not flip a big composite lead
    rows = [_row("STRONG", 5.0, 0.70), _row("WEAK", 1.0, 0.99)]
    out = replay.hi52_tilt_basket(rows, k=0.5, top_n=1)
    assert out == ["STRONG"]


def test_tilt_handles_missing_pct():
    rows = [_row("X", 2.0, None), _row("Y", 1.0, 0.9)]
    out = replay.hi52_tilt_basket(rows, k=1.0, top_n=2)
    assert set(out) == {"X", "Y"}  # no crash; X (pct=None) gets tilt 1.0


def test_pct_to_52w_high_feature():
    # a 260-day series whose LAST price is the max -> pct_to_52w_high == 1.0
    at_high = pd.Series(np.linspace(50.0, 100.0, 260))
    row = replay.build_screen_row("AT", "Tech", at_high)
    assert row is not None and abs(row["pct_to_52w_high"] - 1.0) < 1e-9
    # a series that peaked then fell to 80% of the high -> pct ~ 0.80
    peaked = pd.Series(list(np.linspace(50.0, 100.0, 200)) + list(np.linspace(100.0, 80.0, 60)))
    row2 = replay.build_screen_row("PK", "Tech", peaked)
    assert row2 is not None and abs(row2["pct_to_52w_high"] - 0.80) < 0.02
