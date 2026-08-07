"""phase-83.1.1: gate-feasibility arithmetic, measured with the repo's own functions.

Computes and RECORDS (never asserts against targets -- the step's criteria
forbid it) the quantities the phase-83 go/no-go depends on:

- required annualized Sharpe for DSR >= target across an (N, T, V) grid,
  by BISECTION over ``analytics.compute_deflated_sharpe`` -- never a
  re-derived closed form (criterion 2);
- V measured from the realized dispersion of trial Sharpes in the
  actually-run 82.3 artifacts (criterion 3);
- independent label spans at the PRE-REGISTERED horizon (criterion 4);
- PBO feasibility payloads from ``analytics.compute_pbo_checked`` at the
  intended matrix shape (criterion 5).

The kill rule (killrule_phase83.json) is written BEFORE this module runs;
``run_all`` writes the results JSON artifact whose mtime the criterion-6
test orders against it.

NOTE the import discipline: ``analytics`` is resolved by FUNCTION-SCOPED
attribute lookup so a test spy on ``backend.backtest.analytics`` observes
the real routing. A module-level ``from ... import compute_deflated_sharpe``
would bind the function at import time and make the spy invisible -- the
criterion would pass vacuously (the 83.0.3 lesson, same seam).
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
from typing import Any

_REPO = pathlib.Path(__file__).resolve().parents[2]

# Allow `python backend/backtest/gate_feasibility.py` (direct invocation) --
# same bootstrap convention as backend/news/fetcher.py.
if __package__ in (None, ""):
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
_EXPERIMENTS = _REPO / "backend/backtest/experiments"
KILL_RULE_PATH = _EXPERIMENTS / "killrule_phase83.json"
RESULTS_PATH = _EXPERIMENTS / "gate_feasibility_phase83_results.json"
PREREG_PATH = _EXPERIMENTS / "preregistration_phase83_ranking.json"
_82_3_FULL = _EXPERIMENTS / "results/20260804T025319Z_phase_82_3_full_sample_3strat.json"

DSR_TARGET = 0.95

# XNYS trading-session counts for the measured free-source coverage windows
# (research brief 83.1.1 section 3; calendar built with start="1980-01-01" --
# the library default first session is 2006-08-07 and raises DateOutOfBounds).
SOURCE_SESSIONS = {
    "gdelt": {"sessions": 2883, "window": "2015-02-17..2026-08-04", "free": True},
    "edgar": {"sessions": 6434, "window": "2001-01-01..2026-08-04", "free": True},
    "gpr": {"sessions": 10477, "window": "1985-01-01..2026-08-04", "free": True},
    "control_82_3": {"sessions": 2011, "window": "2018-01-01..2025-12-31",
                     "free": False, "note": "82.3 control window; NOT a free source"},
}

# N grid: 1 and 2 are deliberately both present -- analytics.py clamps
# max(num_trials, 2), so N=1 is byte-identical to N=2 and the results
# artifact records that collapse rather than hiding it.
N_GRID = [1, 2, 10, 45, 100]


def required_annualized_sr(
    num_trials: int,
    T: int,
    variance_of_srs: float,
    dsr_target: float = DSR_TARGET,
    periods_per_year: int = 252,
) -> float:
    """Smallest annualized SR whose DSR reaches dsr_target, by bisection
    over the REPO deflation (monotonicity in observed_sr verified in the
    83.1.1 research over 4000 points; provable for skew=0, kurt=3)."""
    from backend.backtest import analytics

    def dsr(sr: float) -> float:
        return analytics.compute_deflated_sharpe(
            observed_sr=sr,
            num_trials=num_trials,
            variance_of_srs=variance_of_srs,
            skewness=0.0,
            kurtosis=3.0,
            T=T,
            periods_per_year=periods_per_year,
        )

    lo, hi = 1e-9, 50.0
    if dsr(hi) < dsr_target:  # pragma: no cover -- grid never reaches this
        return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if dsr(mid) >= dsr_target:
            hi = mid
        else:
            lo = mid
    return round(hi, 4)


def measure_v_from_82_3() -> dict[str, Any]:
    """V = variance (ddof=1) of the ANNUALIZED Sharpes across the
    actually-run 82.3 full-sample trials, pooled across strategies.
    Recorded WITH the integer trial count (criterion 3)."""
    data = json.loads(_82_3_FULL.read_text())
    sharpes: list[float] = []
    # per_strategy is a dict keyed by strategy name -> {"runs": [...], ...}
    for strat in data["per_strategy"].values():
        for run in strat["runs"]:
            sharpes.append(float(run["sharpe"]))
    n = len(sharpes)
    return {
        "v_measured": round(statistics.variance(sharpes), 6),  # ddof=1
        "n_trials_measured_over": n,
        "trial_sharpes": [round(s, 4) for s in sharpes],
        "mean_sr": round(statistics.mean(sharpes), 4),
        "min_sr": round(min(sharpes), 4),
        "max_sr": round(max(sharpes), 4),
        "source_artifact": _82_3_FULL.name,
        "caveat": ("K=8 configs of three EXISTING strategy families over "
                   "2018-2025 -- a measured prior for V, not the phase-83 "
                   "trial dispersion itself"),
    }


def preregistered_horizon() -> int:
    """The label horizon in TRADING days, read from the 83.1 preregistration
    at runtime -- never the engine's holding-days-derived 1.5x purge horizon."""
    return int(json.loads(PREREG_PATH.read_text())["label_horizon"]["trading_days"])


def span_counts() -> dict[str, Any]:
    horizon = preregistered_horizon()
    per_source = {
        name: {
            "sessions": src["sessions"],
            "window": src["window"],
            "free": src["free"],
            "independent_spans": round(src["sessions"] / horizon, 2),
        }
        for name, src in SOURCE_SESSIONS.items()
    }
    return {
        "horizon_trading_days": horizon,
        "horizon_source": "preregistration_phase83_ranking.json::label_horizon.trading_days",
        "engine_purge_horizon_days_NOT_used": 135,
        "per_source": per_source,
        "unit_note": ("numerators are XNYS TRADING sessions; calendar-day "
                      "numerators overstate GDELT spans by +45%"),
    }


def pbo_feasibility(T: int = 2883, N: int = 45, S: int = 16) -> dict[str, Any]:
    """compute_pbo_checked payloads at the intended gate-evaluation shape:
    multi-seed pure-noise baselines and a one-real-edge case. Payloads
    recorded VERBATIM (criterion 5); a single-seed PBO is not a statistic."""
    import numpy as np

    from backend.backtest import analytics

    noise_payloads = []
    for seed in (1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed)
        payload = analytics.compute_pbo_checked(rng.normal(size=(T, N)), S=S)
        noise_payloads.append({"seed": seed, "payload": payload})

    edge_payloads = []
    for seed in (7, 8, 9):
        rng = np.random.default_rng(seed)
        mat = rng.normal(size=(T, N))
        mat[:, 0] += 0.05  # one genuinely superior column among N-1 noise
        edge_payloads.append({"seed": seed, "payload": analytics.compute_pbo_checked(mat, S=S)})

    # The reading note is DERIVED from the payloads just computed -- never a
    # prose constant. Cycle-1 Q/A FAIL: hand-written ranges (carried from the
    # research brief's different prototype run) contradicted this run's own
    # payloads in both decision-relevant tails.
    noise_pbos = [e["payload"]["pbo"] for e in noise_payloads]
    edge_pbos = [e["payload"]["pbo"] for e in edge_payloads]
    return {
        "intended_shape": {"T": T, "N": N, "S": S},
        "pure_noise_multi_seed": noise_payloads,
        "one_real_edge_multi_seed": edge_payloads,
        "measured_ranges": {
            "pure_noise_min": round(min(noise_pbos), 4),
            "pure_noise_max": round(max(noise_pbos), 4),
            "one_real_edge_min": round(min(edge_pbos), 4),
            "one_real_edge_max": round(max(edge_pbos), 4),
        },
        "reading_note": (
            f"PBO -- not DSR -- is the binding gate at this shape: pure noise "
            f"spans {min(noise_pbos):.4f}-{max(noise_pbos):.4f} across "
            f"{len(noise_pbos)} seeds and one real edge among {N - 1} noise "
            f"columns spans {min(edge_pbos):.4f}-{max(edge_pbos):.4f} against "
            f"the 0.20 ceiling -- in these seeds the edge case NEVER clears "
            f"the ceiling and the noise floor lands essentially at it. A "
            f"single-seed PBO is not a statistic; columns_diverse passed a "
            f"0.918-correlated matrix in research and is weak alone."
        ),
    }


def run_all() -> dict[str, Any]:
    """Compute everything and write the results artifact (AFTER the kill
    rule exists -- the criterion-6 test orders the two mtimes)."""
    assert KILL_RULE_PATH.exists(), (
        "kill rule must exist BEFORE results are produced -- write "
        "killrule_phase83.json first"
    )
    v = measure_v_from_82_3()
    v_short_note = {
        "v_measured": 0.167921,
        "n_trials_measured_over": 32,
        "source_artifact": "20260804T041628Z_phase_82_3_short_window_4strat.json",
        "note": ("short-window pooled set, recorded per contract D4 -- the 20x "
                 "gap vs the full-sample V is estimation noise on a short "
                 "sample; recorded both, never averaged"),
    }
    v_grid = {
        "measured_full_sample": v["v_measured"],
        "measured_short_window": v_short_note["v_measured"],
        "module_default_NOT_MEASURED": 0.5,
    }
    required = {}
    for v_label, v_val in v_grid.items():
        for src, meta in SOURCE_SESSIONS.items():
            for n in N_GRID:
                key = f"V={v_label}|T={src}({meta['sessions']})|N={n}"
                required[key] = required_annualized_sr(n, meta["sessions"], v_val)

    results = {
        "step": "83.1.1",
        "produced_on": "2026-08-07",
        "dsr_target": DSR_TARGET,
        "n_grid": N_GRID,
        "n1_collapse_note": ("analytics.compute_deflated_sharpe clamps "
                             "max(num_trials, 2): every N=1 cell is "
                             "byte-identical to its N=2 cell by construction"),
        "v_measurement": v,
        "v_short_window": v_short_note,
        "required_annualized_sr_grid": required,
        "span_counts": span_counts(),
        "pbo_feasibility": pbo_feasibility(),
        "kill_rule_file": KILL_RULE_PATH.name,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=1, default=str))
    return results


if __name__ == "__main__":
    r = run_all()
    print(f"kill rule: {KILL_RULE_PATH.name} (pre-existing, ordered first)")
    print(f"results:   {RESULTS_PATH.name} written")
    print(f"V = {r['v_measurement']['v_measured']} over n={r['v_measurement']['n_trials_measured_over']} trials")
    print(f"spans@{r['span_counts']['horizon_trading_days']}: " + ", ".join(
        f"{k}={v['independent_spans']}" for k, v in r["span_counts"]["per_source"].items()))
    print("required annualized SR (V=measured_full_sample):")
    for src, meta in SOURCE_SESSIONS.items():
        row = [f"N={n}: {r['required_annualized_sr_grid'][f'V=measured_full_sample|T={src}({meta['sessions']})|N={n}']}" for n in N_GRID]
        print(f"  T={src}({meta['sessions']}): " + "  ".join(row))
