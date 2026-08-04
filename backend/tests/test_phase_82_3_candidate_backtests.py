"""phase-82.3 -- guards the backtest-evidence artifacts.

The 56 runs are expensive (8h34m + 1h23m measured) and are NOT re-run here.
These tests assert the artifacts they produced are present, parseable and carry
all four required metrics as numbers -- so a truncated, half-written or
null-filled result cannot be mistaken for evidence.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "backend" / "backtest" / "experiments" / "results"
TSV = REPO / "backend" / "backtest" / "experiments" / "quant_results.tsv"

PASS_A_GLOB = "*_phase_82_3_full_sample_3strat.json"
PASS_B_GLOB = "*_phase_82_3_short_window_4strat.json"

# The evidential pass. qarp is deliberately absent: historical_fundamentals has
# zero rows before 2024-06-30, so it cannot train on this window (step 82.21).
PASS_A_STRATEGIES = {"triple_barrier", "stretch_regime", "reversion_sigma"}
PASS_B_STRATEGIES = PASS_A_STRATEGIES | {"qarp"}

# compute_pbo (analytics.py) returns 0.0 SILENTLY when T < S*2, and 0.0 PASSES
# a `<= 0.5` gate. Any matrix at or below this is a fabricated pass.
MIN_PBO_T = 32


def _latest(pattern: str) -> Path:
    hits = sorted(RESULTS.glob(pattern))
    assert hits, f"no 82.3 artifact matching {pattern} under {RESULTS}"
    return hits[-1]


def _load(pattern: str) -> dict:
    return json.loads(_latest(pattern).read_text(encoding="utf-8"))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ── criterion 1: a result row per candidate exists ───────────────────

@pytest.mark.parametrize("pattern,expected", [
    (PASS_A_GLOB, PASS_A_STRATEGIES),
    (PASS_B_GLOB, PASS_B_STRATEGIES),
])
def test_result_artifact_exists_and_covers_every_strategy(pattern, expected):
    d = _load(pattern)
    got = set(d.get("per_strategy") or {})
    assert got == expected, f"{pattern}: expected {sorted(expected)}, got {sorted(got)}"


@pytest.mark.parametrize("pattern", [PASS_A_GLOB, PASS_B_GLOB])
def test_every_strategy_has_the_full_config_grid(pattern):
    """K=8 was chosen to match the repo's own _DEFAULT_K. A short grid would
    make PBO near-quantised against the 0.5 threshold."""
    d = _load(pattern)
    for strat, v in d["per_strategy"].items():
        runs = v.get("runs") or []
        assert len(runs) == 8, f"{strat}: {len(runs)} runs, expected the K=8 grid"


# ── criterion 3: all four metrics are NUMBERS, not nulls ─────────────

@pytest.mark.parametrize("pattern", [PASS_A_GLOB, PASS_B_GLOB])
def test_all_four_metrics_are_numbers_not_nulls(pattern):
    """DSR, PBO, turnover and net-of-cost return. A null here is the shape a
    half-failed run leaves behind, and it must not read as evidence."""
    d = _load(pattern)
    problems = []
    for strat, v in d["per_strategy"].items():
        if not _is_number(v.get("pbo")):
            problems.append(f"{strat}: pbo={v.get('pbo')!r}")
        for r in v.get("runs") or []:
            for key in ("dsr", "turnover_rate", "net_of_cost_return_pct"):
                if not _is_number(r.get(key)):
                    problems.append(f"{strat}/{r.get('config')}: {key}={r.get(key)!r}")
    assert not problems, "non-numeric metrics:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("pattern", [PASS_A_GLOB, PASS_B_GLOB])
def test_pbo_is_not_a_silently_fabricated_zero(pattern):
    """THE trap this step exists to avoid. compute_pbo returns 0.0 when
    T < S*2 = 32, and 0.0 passes the `<= 0.5` veto. A per-window PnL matrix
    (T ~ 27 on this sample) would have produced a clean-looking pass for every
    strategy. The T-axis must be daily NAV returns."""
    d = _load(pattern)
    for strat, v in d["per_strategy"].items():
        shape = v.get("pbo_matrix_shape")
        assert shape, f"{strat}: no pbo_matrix_shape recorded"
        T, N = shape
        assert T >= MIN_PBO_T, (
            f"{strat}: PBO matrix T={T} < {MIN_PBO_T}; compute_pbo returns 0.0 "
            "SILENTLY here and 0.0 PASSES the gate"
        )
        assert N >= 2, f"{strat}: PBO needs N>=2 columns, got {N}"
        if v.get("pbo") == 0.0:
            pytest.fail(f"{strat}: pbo is exactly 0.0 -- suspect a silent under-length return")


def test_pass_a_pbo_matrix_uses_daily_returns_not_windows():
    """~27 walk-forward windows vs ~1600 trading days. The magnitude alone
    distinguishes the two constructions."""
    d = _load(PASS_A_GLOB)
    for strat, v in d["per_strategy"].items():
        T = v["pbo_matrix_shape"][0]
        assert T > 500, (
            f"{strat}: T={T} looks like per-window returns, not daily NAV returns"
        )


# ── criterion 2: same sample, one flag state ─────────────────────────

@pytest.mark.parametrize("pattern,expected_start", [
    (PASS_A_GLOB, "2018-01-01"),
    (PASS_B_GLOB, "2024-07-01"),
])
def test_all_strategies_share_one_sample_and_one_flag_state(pattern, expected_start):
    """A comparison across different windows or different point-in-time states
    is not a comparison.

    Q/A cycle-2: the first version asserted only that sample.start/end were
    TRUTHY and that the flag was a bool -- so mutating the window to 1900-01-01,
    or flipping point-in-time to False, both SURVIVED. It was an existence and
    type check wearing the name of a comparison guard. It now pins the actual
    window each pass claims, and pins the flag to True (look-ahead removed);
    False is a different experiment and must not pass silently.
    """
    d = _load(pattern)
    assert d["sample"]["start"] == expected_start, (
        f"sample window is {d['sample']['start']}, expected {expected_start} -- "
        "these runs are not the pass they claim to be"
    )
    assert d["sample"]["end"], "no sample end recorded"
    assert d.get("macro_point_in_time_enabled") is True, (
        "the runs were made with point-in-time macro DISABLED, so they carry "
        "publication-lag look-ahead (82.15) and are not comparable with runs "
        "that do not"
    )


@pytest.mark.parametrize("pattern", [PASS_A_GLOB, PASS_B_GLOB])
def test_tsv_rows_agree_with_the_artifact_window(pattern):
    """Every TSV row sharing a pass_label must record the same window as the
    artifact -- otherwise the appended evidence and the source disagree.

    Q/A cycle-3: the first version used a DEFAULT ARGUMENT (pattern=PASS_A_GLOB)
    rather than @parametrize, so pytest collected one instance bound to pass A
    and pass B's 4 TSV rows were never checked -- while the handoff prose
    claimed "every TSV row". Now parametrized over both passes.
    """
    import csv as _csv
    d = _load(pattern)
    want = (d["sample"]["start"], d["sample"]["end"])
    with TSV.open(encoding="utf-8") as fh:
        rows = [r for r in _csv.DictReader(fh, delimiter="\t")
                if (r.get("run_id") or "").startswith("82.3-pass")]
    seen = set()
    for r in rows:
        pj = json.loads(r["params_json"])
        if pj.get("pass_label") == d.get("pass_label"):
            seen.add((pj["sample_start"], pj["sample_end"]))
    assert seen == {want}, f"TSV windows {seen} disagree with the artifact {want}"


def test_the_incumbent_is_present_in_both_passes():
    """Every candidate is measured AGAINST triple_barrier; without it in the
    same artifact there is no baseline."""
    for pattern in (PASS_A_GLOB, PASS_B_GLOB):
        assert "triple_barrier" in _load(pattern)["per_strategy"]


# ── criterion 1 (second half): appended to quant_results.tsv ─────────

def test_results_are_appended_to_quant_results_tsv():
    assert TSV.exists(), f"{TSV} missing"
    with TSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    ours = [r for r in rows if (r.get("run_id") or "").startswith("82.3-pass")]
    assert ours, "no 82.3 rows appended to quant_results.tsv"

    seen = set()
    for r in ours:
        params = json.loads(r["params_json"])
        seen.add((params["pass_label"], params["strategy"]))
        assert _is_number(params.get("pbo")), f"{params['strategy']}: pbo not a number in the TSV row"
        assert _is_number(params.get("turnover_rate"))
        assert _is_number(params.get("net_of_cost_return_pct"))
        assert _is_number(float(r["dsr"]))

    for strat in PASS_A_STRATEGIES:
        assert ("full_sample_3strat", strat) in seen, f"pass A row missing for {strat}"
    for strat in PASS_B_STRATEGIES:
        assert ("short_window_4strat", strat) in seen, f"pass B row missing for {strat}"


# ── the gate outcome is recorded, whatever it was ────────────────────

def test_no_strategy_silently_claims_a_gate_pass():
    """The measured outcome was 0/3 on the evidential pass. This asserts the
    artifact still supports that reading -- if a future re-run flips it, the
    change is visible here rather than in prose."""
    d = _load(PASS_A_GLOB)
    import statistics as st
    for strat, v in d["per_strategy"].items():
        dsr = st.median([r["dsr"] for r in v["runs"]])
        assert dsr < 0.95, (
            f"{strat}: median DSR {dsr:.4f} now clears 0.95 -- the recorded "
            "0-of-3 gate outcome is stale and the design pack must be updated"
        )
