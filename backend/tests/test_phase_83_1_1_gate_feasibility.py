"""phase-83.1.1: go/no-go gate-feasibility criteria tests.

 C1  required-SR grid computed across N in at least {1,10,45,100} and >=2
     measured-window T values; every figure RECORDED, none asserted against
     a threshold (so these tests check presence/shape/type, never values
     against targets)
 C2  the computation CALLS analytics.compute_deflated_sharpe (routing spy on
     the module attribute the function-scoped import resolves at call time)
 C3  V measured over an actually-run trial set, recorded WITH its integer
     trial count >= 2
 C4  span counts use the horizon THIS PHASE pre-registers (126, read from
     the preregistration at runtime), never the engine's 1.5x-holding-days
     135 -- silent equation fails
 C5  compute_pbo_checked payloads at the intended shape with refused,
     gate_grade, columns_diverse recorded verbatim
 C6  the kill-rule file's mtime precedes every result artifact this step
     produced (strict st_mtime_ns; equal passes -- fresh clones stamp
     identically; population asserted NON-EMPTY so the guard is never
     vacuous)
 C7  touching the kill rule FORWARD makes the C6 assertion fail -- executed
     against the REAL tracked file with its original mtime restored in
     finally (it cannot be unlinked like 83.1's throwaway mutant)
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from backend.backtest import gate_feasibility as gf

_RESULTS = gf.RESULTS_PATH
_KILL = gf.KILL_RULE_PATH


def _results() -> dict:
    assert _RESULTS.exists(), "results artifact missing -- run gate_feasibility.run_all()"
    return json.loads(_RESULTS.read_text())


# ── C1: grid shape + recorded-not-asserted ───────────────────────────


def test_c1_grid_covers_required_n_and_t():
    r = _results()
    grid = r["required_annualized_sr_grid"]
    for n in (1, 10, 45, 100):
        assert n in r["n_grid"], f"N={n} missing from the grid"
    t_labels = {k.split("|")[1] for k in grid}
    assert len(t_labels) >= 2, f"fewer than two T windows: {t_labels}"
    # |cells| == |V| x |T| x |N| -- a dropped cell is a missing record
    v_labels = {k.split("|")[0] for k in grid}
    n_vals = {k.split("|")[2] for k in grid}
    assert len(grid) == len(v_labels) * len(t_labels) * len(n_vals), (
        f"grid incomplete: {len(grid)} cells != "
        f"{len(v_labels)}x{len(t_labels)}x{len(n_vals)}"
    )
    for key, val in grid.items():
        assert isinstance(val, (int, float)) and val > 0, f"non-numeric cell {key}"
    # the N=1 collapse is RECORDED, not hidden
    assert "n1_collapse_note" in r and "byte-identical" in r["n1_collapse_note"]


# ── C2: the deflation is the repo function, not a reimplementation ───


def test_c2_required_sr_routes_through_compute_deflated_sharpe(monkeypatch):
    import backend.backtest.analytics as analytics

    calls = []
    real = analytics.compute_deflated_sharpe

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(analytics, "compute_deflated_sharpe", spy)
    out = gf.required_annualized_sr(num_trials=10, T=2883, variance_of_srs=0.008169)
    assert calls, "required_annualized_sr did not invoke analytics.compute_deflated_sharpe"
    assert isinstance(out, float) and 0 < out < 50
    # periods_per_year matched to the daily return frequency (criterion 1)
    assert all(c.get("periods_per_year") == 252 for c in calls)


# ── C3: V measured with its trial count ──────────────────────────────


def test_c3_v_recorded_with_integer_trial_count():
    v = _results()["v_measurement"]
    assert isinstance(v["n_trials_measured_over"], int)
    assert v["n_trials_measured_over"] >= 2, "V without a sample size >= 2 does not count"
    assert isinstance(v["v_measured"], float) and v["v_measured"] > 0
    # the recorded count is the actual number of trial Sharpes recorded
    assert len(v["trial_sharpes"]) == v["n_trials_measured_over"]
    # the secondary (short-window) V also carries its count
    vs = _results()["v_short_window"]
    assert isinstance(vs["n_trials_measured_over"], int) and vs["n_trials_measured_over"] >= 2


# ── C4: the pre-registered horizon, never the engine's 135 ───────────


def test_c4_horizon_is_preregistered_not_engine_derived():
    spans = _results()["span_counts"]
    prereg = json.loads(gf.PREREG_PATH.read_text())["label_horizon"]["trading_days"]
    assert spans["horizon_trading_days"] == prereg, (
        "recorded horizon does not match the 83.1 preregistration"
    )
    # the engine's holding-days-derived horizon, computed from the ENGINE
    # default rather than hardcoded lore: int(90 * 1.5) == 135
    import inspect

    from backend.backtest.backtest_engine import BacktestEngine

    sig = inspect.signature(BacktestEngine.__init__)
    holding_default = sig.parameters["holding_days"].default
    engine_horizon = int(holding_default * 1.5)
    assert spans["horizon_trading_days"] != engine_horizon, (
        f"horizon silently equated with the engine purge horizon {engine_horizon}"
    )
    # spans recorded per candidate source, arithmetic consistent
    for name, src in spans["per_source"].items():
        assert src["independent_spans"] == round(src["sessions"] / prereg, 2), name


# ── C5: verbatim compute_pbo_checked payloads at the intended shape ──


def test_c5_pbo_payloads_recorded_verbatim():
    pbo = _results()["pbo_feasibility"]
    assert pbo["intended_shape"] == {"T": 2883, "N": 45, "S": 16}
    assert len(pbo["pure_noise_multi_seed"]) >= 3, "a single-seed PBO is not a statistic"
    for entry in pbo["pure_noise_multi_seed"] + pbo["one_real_edge_multi_seed"]:
        payload = entry["payload"]
        for field in ("refused", "gate_grade", "columns_diverse"):
            assert field in payload, f"payload missing {field} (seed {entry['seed']})"
        assert payload["refused"] is None  # shape is adequate: never refused
        assert payload["gate_grade"] is True  # N=45 >= 10


def test_c5b_narrative_ranges_match_the_recorded_payloads():
    """Cycle-1 Q/A FAIL guard: the artifact's stated ranges must equal the
    min/max of the payloads they summarize -- a hand-written range that
    contradicts the record must turn this red (kills the q7/q8 mutant class:
    falsifying either a payload value or the reading_note)."""
    pbo = _results()["pbo_feasibility"]
    noise = [e["payload"]["pbo"] for e in pbo["pure_noise_multi_seed"]]
    edge = [e["payload"]["pbo"] for e in pbo["one_real_edge_multi_seed"]]
    ranges = pbo["measured_ranges"]
    assert ranges["pure_noise_min"] == round(min(noise), 4)
    assert ranges["pure_noise_max"] == round(max(noise), 4)
    assert ranges["one_real_edge_min"] == round(min(edge), 4)
    assert ranges["one_real_edge_max"] == round(max(edge), 4)
    # the prose note carries the same measured numbers, not imported ones
    note = pbo["reading_note"]
    for val in (min(noise), max(noise), min(edge), max(edge)):
        assert f"{val:.4f}" in note, (
            f"reading_note does not carry the measured value {val:.4f}"
        )


# ── cycle-4 guards (Q/A wf_48465ea7-38e): the record cannot drift ────


def test_c1b_spot_cells_equal_live_recomputation():
    """qa15 killer: recorded grid VALUES must equal a live recomputation
    through the module (equality-to-recomputation, not a threshold --
    criterion 1's no-target rule is untouched)."""
    grid = _results()["required_annualized_sr_grid"]
    spots = [
        ("V=measured_full_sample|T=gdelt(2883)|N=45", 45, 2883, 0.008169),
        ("V=measured_full_sample|T=gpr(10477)|N=10", 10, 10477, 0.008169),
        ("V=module_default_NOT_MEASURED|T=gdelt(2883)|N=45", 45, 2883, 0.5),
    ]
    for key, n, t, v in spots:
        assert grid[key] == gf.required_annualized_sr(n, t, v), (
            f"recorded cell {key} does not equal live recomputation"
        )


def test_c5c_seed_population_pinned():
    """qa14 killer: the seed tuples are pinned -- pruning the harshest seed
    (or any seed) breaks the census even if ranges stay self-consistent."""
    pbo = _results()["pbo_feasibility"]
    assert [e["seed"] for e in pbo["pure_noise_multi_seed"]] == [1, 2, 3, 4, 5]
    assert [e["seed"] for e in pbo["one_real_edge_multi_seed"]] == [7, 8, 9]


def test_c3b_v_equals_variance_of_its_own_trial_sharpes():
    """qa6 killer, two independent pins: (a) v_measured equals the ddof=1
    variance of its recorded sharpes at 2e-6 (measured 4dp-rounding gap is
    1.4e-7; dropping even ONE trial moves it 1e-5, 5.00x the tolerance
    -- measured by the cycle-4 Q/A over all 24 single-drop variants);
    (b) the recorded count equals the SOURCE artifact's run count, so a
    consistently-truncated set still dies."""
    import statistics

    v = _results()["v_measurement"]
    recomputed = statistics.variance(v["trial_sharpes"])
    assert abs(v["v_measured"] - round(recomputed, 6)) <= 2e-6, (
        f"v_measured {v['v_measured']} is not the variance of its own "
        f"recorded trial_sharpes (recomputed {round(recomputed, 8)})"
    )
    src = json.loads(
        (gf._82_3_FULL).read_text()
    )
    n_source = sum(len(s["runs"]) for s in src["per_strategy"].values())
    assert v["n_trials_measured_over"] == n_source, (
        f"recorded trial count {v['n_trials_measured_over']} != source "
        f"artifact run count {n_source} -- the trial set was narrowed"
    )


# ── C6: kill rule predates every result artifact ─────────────────────


def _step_result_artifacts() -> list[pathlib.Path]:
    return [_RESULTS]


def _assert_kill_rule_predates_results() -> None:
    arts = _step_result_artifacts()
    assert arts, "result-artifact population EMPTY -- the ordering guard is vacuous"
    k_ns = _KILL.stat().st_mtime_ns
    for art in arts:
        assert art.exists(), f"{art.name} missing"
        assert not art.stat().st_mtime_ns < k_ns, (
            f"result artifact {art.name} predates the kill rule -- the phase "
            "could be talked past its own arithmetic"
        )


def test_c6_kill_rule_written_before_results():
    assert _KILL.exists(), "kill-rule file missing"
    clauses = json.loads(_KILL.read_text())["clauses"]
    assert len(clauses) >= 1
    for c in clauses:
        for field in ("recorded_quantity", "comparison", "threshold", "threshold_source"):
            assert c.get(field) not in (None, ""), f"clause {c.get('id')} missing {field}"
    _assert_kill_rule_predates_results()


# ── C7: the ordering guard is mutation-tested on the REAL file ───────


def test_c7_touching_kill_rule_forward_fails_criterion_6():
    orig_ns = _KILL.stat().st_mtime_ns
    newest = max(a.stat().st_mtime_ns for a in _step_result_artifacts())
    try:
        t = newest + 60_000_000_000  # 60s after the newest result
        os.utime(_KILL, ns=(t, t))
        with pytest.raises(AssertionError):
            _assert_kill_rule_predates_results()
    finally:
        os.utime(_KILL, ns=(orig_ns, orig_ns))
    # restoration proven, not assumed
    assert _KILL.stat().st_mtime_ns == orig_ns
