"""phase-8.3 tests for backend.backtest.ensemble_blend.

Pure-Python math tests; no numpy/sklearn dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from backend.backtest.ensemble_blend import EnsembleBlender


def test_init_defaults():
    b = EnsembleBlender()
    assert b.component_names == ("mda", "timesfm", "chronos")
    assert b.weighting_method == "equal"
    assert b.n_splits == 5
    assert b.last_weights == {"mda": 1.0 / 3, "timesfm": 1.0 / 3, "chronos": 1.0 / 3}


def test_init_rejects_unknown_weighting_method():
    with pytest.raises(ValueError):
        EnsembleBlender(weighting_method="magic")


def test_init_rejects_empty_components():
    with pytest.raises(ValueError):
        EnsembleBlender(component_names=())


def test_pearson_perfect_positive_and_negative():
    b = EnsembleBlender()
    assert abs(b._compute_ic([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]) - 1.0) < 1e-9
    assert abs(b._compute_ic([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0]) + 1.0) < 1e-9


def test_pearson_zero_variance_is_zero():
    b = EnsembleBlender()
    assert b._compute_ic([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_equal_weights_returned_when_component_missing():
    b = EnsembleBlender(weighting_method="correlation")
    # Missing "chronos" signal -> fallback to equal.
    w = b.fit_weights(
        {"mda": [0.1, 0.2, 0.3, 0.4, 0.5], "timesfm": [0.2, 0.3, 0.4, 0.5, 0.6]},
        [0.01, 0.02, 0.03, 0.04, 0.05],
    )
    assert pytest.approx(w["mda"]) == 1.0 / 3
    assert pytest.approx(w["timesfm"]) == 1.0 / 3
    assert pytest.approx(w["chronos"]) == 1.0 / 3


def test_correlation_weighted_rewards_high_ic():
    b = EnsembleBlender(weighting_method="correlation", n_splits=2)
    # mda perfectly correlates with returns; other two anti-correlate.
    returns = [0.01 * i for i in range(20)]
    w = b.fit_weights(
        {
            "mda": list(returns),
            "timesfm": list(reversed(returns)),
            "chronos": [0.0] * 20,
        },
        returns,
    )
    # mda should get substantially more weight (IC=1), chronos should get ~0.
    assert w["mda"] > w["chronos"]
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0


def test_blend_equal_weights():
    b = EnsembleBlender()  # equal weights
    sig = {
        "mda": {("AAPL", "2026-04-19"): 1.0, ("MSFT", "2026-04-19"): 2.0},
        "timesfm": {("AAPL", "2026-04-19"): 3.0, ("MSFT", "2026-04-19"): 4.0},
        "chronos": {("AAPL", "2026-04-19"): 5.0, ("MSFT", "2026-04-19"): 6.0},
    }
    out = b.blend(sig)
    assert pytest.approx(out[("AAPL", "2026-04-19")]) == 3.0  # (1+3+5)/3
    assert pytest.approx(out[("MSFT", "2026-04-19")]) == 4.0  # (2+4+6)/3


def test_blend_drops_unknown_component_and_logs():
    b = EnsembleBlender()
    sig = {
        "mda": {("AAPL", "2026-04-19"): 1.0},
        "unknown_component": {("AAPL", "2026-04-19"): 99.0},
    }
    out = b.blend(sig)
    # Unknown is dropped; mda alone, so weight renormalizes to 1.0 -> output = 1.0.
    assert out == {("AAPL", "2026-04-19"): 1.0}


def test_blend_handles_missing_key_per_component():
    b = EnsembleBlender()
    sig = {
        "mda": {("AAPL", "2026-04-19"): 2.0},
        "timesfm": {},  # no AAPL
        "chronos": {("AAPL", "2026-04-19"): 4.0},
    }
    out = b.blend(sig)
    # Only mda + chronos contribute; weights renormalize to 0.5 each.
    assert pytest.approx(out[("AAPL", "2026-04-19")]) == 3.0


def test_walk_forward_splits_respects_chronology():
    b = EnsembleBlender(n_splits=3)
    splits = b._walk_forward_splits(24)
    assert len(splits) == 3
    for train, test in splits:
        assert max(train) < min(test)  # strict chronology


def test_ledoit_wolf_shrinkage_shape_and_bounds():
    b = EnsembleBlender()
    # 20 samples, 3 dims
    x = [[float(i), float(2 * i), float(3 * i)] for i in range(20)]
    cov, shrinkage = b._ledoit_wolf_shrinkage(x)
    assert len(cov) == 3 and len(cov[0]) == 3
    assert 0.0 <= shrinkage <= 1.0


def test_shrinkage_method_produces_simplex_weights():
    b = EnsembleBlender(weighting_method="shrinkage", n_splits=2)
    returns = [0.01 * ((-1) ** i) for i in range(30)]
    signals = {
        "mda": [r + 0.001 for r in returns],
        "timesfm": [-r + 0.001 for r in returns],
        "chronos": [0.0] * 30,
    }
    w = b.fit_weights(signals, returns)
    assert pytest.approx(sum(w.values()), abs=1e-9) == 1.0
    for v in w.values():
        assert 0.0 <= v <= 1.0


def test_cv_ic_shape():
    b = EnsembleBlender(n_splits=2)
    returns = [0.01 * i for i in range(18)]
    signals = {
        "mda": list(returns),
        "timesfm": [r + 0.001 for r in returns],
        "chronos": [r - 0.001 for r in returns],
    }
    out = b.cv_ic(signals, returns)
    assert set(out.keys()) == {"ic_mean", "ic_std", "ic_ir", "n_splits"}
    assert out["n_splits"] >= 1
    assert out["ic_mean"] > 0.9  # perfectly-correlated components


def test_module_is_ascii_only():
    mod_path = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "backtest"
        / "ensemble_blend.py"
    )
    mod_path.read_bytes().decode("ascii")
