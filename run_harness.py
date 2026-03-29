"""
Three-Agent Harness Loop — Anthropic harness design pattern applied to quant research.

Planner (heuristic) -> Generator (QuantStrategyOptimizer) -> Evaluator (independent backtests)

Usage:
    python run_harness.py [--cycles N] [--iterations-per-cycle N] [--dry-run]
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -- Setup path and env before any backend imports --
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv("backend/.env")

# Suppress noisy GCP logs
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("harness")

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.backtest.backtest_engine import BacktestEngine
from backend.backtest.analytics import generate_report
from backend.backtest.quant_optimizer import QuantStrategyOptimizer

# -- Paths --
HANDOFF_DIR = Path(__file__).parent / "handoff"
BEST_PARAMS_PATH = Path(__file__).parent / "backend" / "backtest" / "experiments" / "optimizer_best.json"
TSV_PATH = Path(__file__).parent / "backend" / "backtest" / "experiments" / "quant_results.tsv"
HARNESS_LOG = HANDOFF_DIR / "harness_log.md"

# -- Sub-periods for evaluator --
SUB_PERIODS = [
    ("2018-2020", "2018-01-01", "2020-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2025", "2023-01-01", "2025-12-31"),
]


def load_best_params() -> dict:
    """Load current best params from optimizer_best.json."""
    with open(BEST_PARAMS_PATH) as f:
        return json.load(f)


def save_best_params(data: dict):
    """Save params back to optimizer_best.json."""
    BEST_PARAMS_PATH.write_text(json.dumps(data, default=str, indent=2), encoding="utf-8")


def progress_cb(data: dict):
    """Compact progress callback for backtest engine."""
    step = data.get("step", "")
    detail = data.get("step_detail", "")
    window = data.get("window", 0)
    total = data.get("total_windows", 0)
    if window:
        print(f"\r  [{window}/{total}] {step}: {detail}        ", end="", flush=True)


def make_engine(params: dict, settings, bq, start_date=None, end_date=None, tx_cost_pct=None):
    """Create a fresh BacktestEngine from params dict."""
    return BacktestEngine(
        bq_client=bq.client,
        project=settings.gcp_project_id,
        dataset=settings.bq_dataset_reports,
        start_date=start_date or params.get("start_date", "2018-01-01"),
        end_date=end_date or params.get("end_date", "2025-12-31"),
        strategy=params.get("strategy", "triple_barrier"),
        holding_days=params.get("holding_days", 90),
        tp_pct=params.get("tp_pct", 10.0),
        sl_pct=params.get("sl_pct", 10.0),
        frac_diff_d=params.get("frac_diff_d", 0.4),
        mr_holding_days=params.get("mr_holding_days", 15),
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 4),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        learning_rate=params.get("learning_rate", 0.1),
        max_positions=params.get("max_positions", 20),
        top_n_candidates=params.get("top_n_candidates", 50),
        transaction_cost_pct=tx_cost_pct if tx_cost_pct is not None else 0.1,
        progress_callback=progress_cb,
    )


def run_backtest(params: dict, settings, bq, start_date=None, end_date=None, tx_cost_pct=None) -> dict:
    """Run a single backtest with a fresh engine. Returns analytics dict."""
    engine = make_engine(params, settings, bq, start_date, end_date, tx_cost_pct)
    result = engine.run_backtest()
    report = generate_report(result, num_trials=1)
    return report["analytics"]


def run_backtest_full(params: dict, settings, bq, start_date=None, end_date=None, tx_cost_pct=None) -> dict:
    """Run backtest and return full report (analytics + per_window). Used by evaluator for hardening tests."""
    engine = make_engine(params, settings, bq, start_date, end_date, tx_cost_pct)
    result = engine.run_backtest()
    report = generate_report(result, num_trials=1)
    return report


# ── AGENT 1: Heuristic Planner ──────────────────────────────────

def run_planner(cycle: int, previous_critique: dict | None) -> dict:
    """
    Rule-based planner. Reads experiment log, identifies patterns, suggests next direction.
    Returns a plan dict with hypothesis + suggestions.
    """
    plan = {
        "cycle": cycle,
        "hypothesis": "Continue parameter optimization with random perturbation",
        "suggestions": [],
        "excluded_params": [],
        "strategy_change": False,
    }

    # Read TSV experiment history
    experiments = _read_tsv_experiments()
    if not experiments:
        plan["hypothesis"] = "No experiment history -- establish baseline first"
        return plan

    # Count kept/discarded per param
    param_stats = {}
    for exp in experiments:
        param = exp.get("param_changed", "--")
        status = exp.get("status", "")
        if param == "--" or param == "warm-start" or param == "standalone":
            continue
        # Extract param name (format: "param_name: old -> new")
        pname = param.split(":")[0].strip() if ":" in param else param
        if pname not in param_stats:
            param_stats[pname] = {"kept": 0, "discarded": 0, "recent_discards": 0}
        if status == "keep":
            param_stats[pname]["kept"] += 1
            param_stats[pname]["recent_discards"] = 0
        elif status in ("discard", "dsr_reject", "crash"):
            param_stats[pname]["discarded"] += 1
            param_stats[pname]["recent_discards"] += 1

    # Rule 1: Check for plateau (last 10 experiments all discarded)
    recent = experiments[-10:]
    recent_kept = sum(1 for e in recent if e.get("status") == "keep")
    if len(recent) >= 10 and recent_kept == 0:
        plan["suggestions"].append("PLATEAU: Last 10 experiments all discarded. Consider strategy change.")
        plan["strategy_change"] = True

    # Rule 2: Check for small improvement plateau
    recent_keeps = [e for e in experiments[-15:] if e.get("status") == "keep"]
    if recent_keeps:
        max_delta = max(abs(float(e.get("delta", 0))) for e in recent_keeps)
        if max_delta < 0.02:
            plan["suggestions"].append(
                f"DIMINISHING RETURNS: Best improvement in last 15 experiments is {max_delta:.4f}. "
                "Consider structural changes."
            )

    # Rule 3: Saturated params (5+ consecutive discards)
    for pname, stats in param_stats.items():
        if stats["recent_discards"] >= 5:
            plan["excluded_params"].append(pname)
            plan["suggestions"].append(f"SATURATED: {pname} has {stats['recent_discards']} consecutive discards. Excluding.")

    # Rule 4: If previous evaluator found sub-period weakness, target that regime
    if previous_critique:
        weak_periods = previous_critique.get("weak_periods", [])
        for wp in weak_periods:
            plan["suggestions"].append(f"WEAKNESS: Sub-period '{wp}' had low Sharpe. Target that regime.")

        if previous_critique.get("verdict") == "FAIL":
            plan["hypothesis"] = "Previous cycle FAILED evaluation. Need defensive parameter changes."
            plan["suggestions"].append("REVERT: Consider reverting to last known-good baseline.")

    # Rule 5: Default -- continue with random perturbation
    if not plan["suggestions"]:
        plan["suggestions"].append("No strong signals. Continue random perturbation.")
        plan["hypothesis"] = "Random parameter search still has value at current iteration count."

    return plan


def write_contract(plan: dict, cycle: int, current_sharpe: float):
    """Write handoff/contract.md -- the sprint contract between generator and evaluator."""
    content = f"""# Sprint Contract -- Cycle {cycle}
Generated: {datetime.now(timezone.utc).isoformat()}

## Hypothesis
{plan['hypothesis']}

## Current Baseline
- Sharpe: {current_sharpe:.4f}

## Success Criteria (from evaluator_criteria.md)
- Statistical Validity: DSR >= 0.95, Sharpe > 0
- Robustness: ALL sub-periods Sharpe > 0
- Reality Gap: 2x costs Sharpe > 0.5

## Planner Suggestions
"""
    for s in plan["suggestions"]:
        content += f"- {s}\n"

    if plan["excluded_params"]:
        content += f"\n## Excluded Parameters\n"
        for p in plan["excluded_params"]:
            content += f"- {p}\n"

    (HANDOFF_DIR / "contract.md").write_text(content, encoding="utf-8")
    logger.info("Wrote handoff/contract.md")


# ── AGENT 2: Generator (QuantStrategyOptimizer wrapper) ──────────

def run_generator(iterations: int, settings, bq) -> dict:
    """
    Run the QuantStrategyOptimizer for N iterations.
    Returns generator results dict.
    """
    best_data = load_best_params()
    params = best_data["params"]
    initial_sharpe = best_data.get("sharpe", 0)

    logger.info("Generator: starting %d iterations from Sharpe=%.4f", iterations, initial_sharpe)

    engine = make_engine(params, settings, bq)
    optimizer = QuantStrategyOptimizer(
        backtest_engine=engine,
        status_callback=lambda state: None,
        dsr_threshold=0.95,
    )

    t0 = time.time()
    optimizer.run_loop(max_iterations=iterations, use_llm=False)
    elapsed = time.time() - t0

    final = optimizer.export_best()
    final_sharpe = final["sharpe"] or 0
    delta = final_sharpe - initial_sharpe

    result = {
        "initial_sharpe": initial_sharpe,
        "final_sharpe": final_sharpe,
        "delta": delta,
        "kept": final["kept"],
        "discarded": final["discarded"],
        "num_trials": final["num_trials"],
        "elapsed_sec": round(elapsed, 1),
        "params": final["params"],
        "dsr": final["dsr"],
    }

    # Write experiment_results.md
    content = f"""# Experiment Results -- Generator Output
Generated: {datetime.now(timezone.utc).isoformat()}

## Summary
- Initial Sharpe: {initial_sharpe:.4f}
- Final Sharpe: {final_sharpe:.4f}
- Delta: {delta:+.4f}
- Kept: {final['kept']} / Discarded: {final['discarded']}
- Iterations: {iterations}
- Elapsed: {elapsed:.0f}s
- DSR: {final['dsr']:.4f}

## Best Parameters
```json
{json.dumps(final['params'], indent=2, default=str)}
```
"""
    (HANDOFF_DIR / "experiment_results.md").write_text(content, encoding="utf-8")
    logger.info("Generator: done. Sharpe %.4f -> %.4f (%+.4f), kept=%d, discarded=%d",
                initial_sharpe, final_sharpe, delta, final["kept"], final["discarded"])
    return result


# ── AGENT 3: Evaluator (independent validation) ─────────────────

def run_evaluator(params: dict, settings, bq) -> dict:
    """
    Independent evaluator: sub-period backtests + 2x cost stress test.
    Uses SEPARATE engine instances (core Anthropic insight).
    Returns grades + verdict.
    """
    logger.info("Evaluator: starting independent validation...")
    results = {}

    # Sub-period backtests (use full reports for feature importance stability)
    sub_period_features = {}
    for label, start, end in SUB_PERIODS:
        logger.info("  Evaluator: running %s (%s to %s)", label, start, end)
        try:
            sub_report = run_backtest_full(params, settings, bq, start_date=start, end_date=end)
            a = sub_report["analytics"]
            results[label] = a
            # Extract top features for stability check
            fi = sub_report.get("feature_importance", {})
            mda = fi.get("mda_top_15", [])
            sub_period_features[label] = set(f["feature"] for f in mda[:10])
            logger.info("  %s: Sharpe=%.4f DSR=%.4f Return=%.1f%%",
                        label, a["sharpe"], a["deflated_sharpe"], a["total_return_pct"])
        except Exception as e:
            logger.error("  %s: FAILED - %s", label, e)
            results[label] = {"error": str(e), "sharpe": -999}

    # 2x cost stress test (full period)
    logger.info("  Evaluator: running 2x cost stress test")
    try:
        a_2x = run_backtest(params, settings, bq, tx_cost_pct=0.2)
        results["2x_costs"] = a_2x
        logger.info("  2x_costs: Sharpe=%.4f Return=%.1f%%", a_2x["sharpe"], a_2x["total_return_pct"])
    except Exception as e:
        logger.error("  2x_costs: FAILED - %s", e)
        results["2x_costs"] = {"error": str(e), "sharpe": -999}

    # Full-period baseline (for statistical validity + hardening tests)
    logger.info("  Evaluator: running full-period baseline")
    try:
        full_report = run_backtest_full(params, settings, bq)
        a_full = full_report["analytics"]
        results["full_period"] = a_full
        # Store per-window returns for concentration and autocorrelation tests
        results["full_period"]["window_returns"] = [
            w["total_return_pct"] for w in full_report.get("per_window", [])
        ]
        logger.info("  full_period: Sharpe=%.4f DSR=%.4f", a_full["sharpe"], a_full["deflated_sharpe"])
    except Exception as e:
        logger.error("  full_period: FAILED - %s", e)
        results["full_period"] = {"error": str(e), "sharpe": -999}

    # -- Phase 2.8 Hardening Tests --

    # 1. Window concentration check: no single window should drive >30% of total return
    full = results.get("full_period", {})
    if "error" not in full:
        window_returns = full.get("window_returns", [])
        if window_returns:
            total_abs = sum(abs(r) for r in window_returns)
            if total_abs > 0:
                max_concentration = max(abs(r) / total_abs for r in window_returns)
                results["window_concentration"] = {
                    "max_concentration_pct": round(max_concentration * 100, 1),
                    "pass": max_concentration < 0.3,
                    "n_windows": len(window_returns),
                }
                logger.info("  Window concentration: %.1f%% (%s)",
                           max_concentration * 100, "PASS" if max_concentration < 0.3 else "FAIL")

    # 2. Ljung-Box autocorrelation test on returns
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        import numpy as np
        window_returns = full.get("window_returns", [])
        if len(window_returns) >= 10:
            lb_result = acorr_ljungbox(np.array(window_returns), lags=[5], return_df=True)
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[0])
            results["ljung_box"] = {
                "p_value": round(lb_pvalue, 4),
                "pass": lb_pvalue > 0.05,  # p > 0.05 means no significant autocorrelation
                "interpretation": "No autocorrelation (good)" if lb_pvalue > 0.05 else "Serial correlation detected (bad)",
            }
            logger.info("  Ljung-Box: p=%.4f (%s)", lb_pvalue,
                        "PASS" if lb_pvalue > 0.05 else "FAIL")
    except ImportError:
        logger.warning("  Ljung-Box: statsmodels not installed, skipping")
    except Exception as e:
        logger.warning("  Ljung-Box: failed - %s", e)

    # 3. Slippage stress test (5bps execution slippage on top of transaction costs)
    logger.info("  Evaluator: running 5bps slippage stress test")
    try:
        base_tx = params.get("transaction_cost_pct", 0.1)
        slippage_tx = base_tx + 0.05  # add 5bps execution slippage
        a_slip = run_backtest(params, settings, bq, tx_cost_pct=slippage_tx)
        results["slippage_5bps"] = a_slip
        results["slippage_5bps"]["survives"] = a_slip.get("sharpe", 0) > 0.5
        logger.info("  5bps slippage: Sharpe=%.4f (%s)",
                    a_slip.get("sharpe", 0), "PASS" if a_slip.get("sharpe", 0) > 0.5 else "FAIL")
    except Exception as e:
        logger.error("  Slippage test: FAILED - %s", e)
        results["slippage_5bps"] = {"error": str(e), "sharpe": -999}

    # 4. Feature importance stability across sub-periods
    if len(sub_period_features) >= 2:
        all_feature_sets = list(sub_period_features.values())
        # Jaccard similarity between each pair
        jaccard_scores = []
        pairs = []
        labels = list(sub_period_features.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a_set, b_set = all_feature_sets[i], all_feature_sets[j]
                if a_set or b_set:
                    jaccard = len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0
                    jaccard_scores.append(jaccard)
                    pairs.append(f"{labels[i]} vs {labels[j]}")

        if jaccard_scores:
            avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
            # Common features across ALL sub-periods
            common_features = set.intersection(*all_feature_sets) if all_feature_sets else set()
            results["feature_stability"] = {
                "avg_jaccard": round(avg_jaccard, 3),
                "pairs": {p: round(j, 3) for p, j in zip(pairs, jaccard_scores)},
                "common_features": sorted(common_features),
                "pass": avg_jaccard > 0.3,  # At least 30% overlap
                "n_common": len(common_features),
            }
            logger.info("  Feature stability: avg Jaccard=%.3f, %d common features (%s)",
                        avg_jaccard, len(common_features), "PASS" if avg_jaccard > 0.3 else "FAIL")

    # -- GRADE (anti-leniency: grade each criterion BEFORE verdict) --
    grades = _grade_results(results, params)

    # Write evaluator critique
    _write_critique(results, grades)

    return grades


def _grade_results(results: dict, params: dict) -> dict:
    """Grade against evaluator_criteria.md. Anti-leniency: lower score when uncertain."""

    # 1. Statistical Validity (weight 40%)
    full = results.get("full_period", {})
    full_sharpe = full.get("sharpe", -999)
    full_dsr = full.get("deflated_sharpe", 0)

    if "error" in full:
        stat_score = 1
        stat_notes = f"Full period backtest failed: {full.get('error')}"
    elif full_dsr >= 0.99 and full_sharpe > 0.5:
        stat_score = 9
        stat_notes = f"DSR={full_dsr:.4f} (>0.99), Sharpe={full_sharpe:.4f}"
    elif full_dsr >= 0.95 and full_sharpe > 0:
        stat_score = 7
        stat_notes = f"DSR={full_dsr:.4f} (>=0.95), Sharpe={full_sharpe:.4f}"
    elif full_sharpe > 0:
        stat_score = 5
        stat_notes = f"Sharpe positive but DSR={full_dsr:.4f} (<0.95)"
    else:
        stat_score = 2
        stat_notes = f"Sharpe={full_sharpe:.4f} (<=0), DSR={full_dsr:.4f}"

    # 2. Robustness (weight 30%)
    sub_sharpes = {}
    weak_periods = []
    for label, _, _ in SUB_PERIODS:
        sp = results.get(label, {})
        s = sp.get("sharpe", -999)
        sub_sharpes[label] = s
        if s <= 0:
            weak_periods.append(label)

    cost_2x = results.get("2x_costs", {})
    cost_2x_sharpe = cost_2x.get("sharpe", -999)

    all_positive = all(s > 0 for s in sub_sharpes.values())
    all_above_03 = all(s > 0.3 for s in sub_sharpes.values())

    if all_above_03 and cost_2x_sharpe > 0.7:
        robust_score = 9
        robust_notes = f"All sub-periods >0.3, 2x costs Sharpe={cost_2x_sharpe:.4f}"
    elif all_positive and cost_2x_sharpe > 0.5:
        robust_score = 7
        robust_notes = f"All sub-periods positive, 2x costs Sharpe={cost_2x_sharpe:.4f}"
    elif all_positive:
        robust_score = 6
        robust_notes = f"All sub-periods positive but 2x costs Sharpe={cost_2x_sharpe:.4f} (<0.5)"
    elif len(weak_periods) == 1:
        robust_score = 4
        robust_notes = f"Weak period: {weak_periods[0]}, 2x costs Sharpe={cost_2x_sharpe:.4f}"
    else:
        robust_score = 2
        robust_notes = f"Multiple weak periods: {weak_periods}"

    # 3. Simplicity (weight 15%)
    # Count active params (non-default or explicitly tuned)
    active_count = len([k for k in params if k not in ("start_date", "end_date", "train_window_months",
                                                        "test_window_months", "embargo_days", "starting_capital")])
    if active_count <= 12:
        simple_score = 9
        simple_notes = f"{active_count} active params (<=12)"
    elif active_count <= 15:
        simple_score = 8
        simple_notes = f"{active_count} active params (<=15)"
    elif active_count <= 20:
        simple_score = 6
        simple_notes = f"{active_count} active params (<=20)"
    else:
        simple_score = 4
        simple_notes = f"{active_count} active params (>20, over-parameterized)"

    # 4. Reality Gap (weight 15%) — includes slippage test from Phase 2.8
    slip = results.get("slippage_5bps", {})
    slip_sharpe = slip.get("sharpe", -999) if "error" not in slip else -999
    slip_survives = slip.get("survives", False)

    if cost_2x_sharpe > 0.7 and slip_survives:
        reality_score = 9
        reality_notes = f"2x costs Sharpe={cost_2x_sharpe:.4f}, +5bps slippage Sharpe={slip_sharpe:.4f}"
    elif cost_2x_sharpe > 0.7:
        reality_score = 8
        reality_notes = f"2x costs Sharpe={cost_2x_sharpe:.4f} (>0.7), slippage={'untested' if slip_sharpe == -999 else f'{slip_sharpe:.4f}'}"
    elif cost_2x_sharpe > 0.5:
        reality_score = 7 if slip_survives else 6
        reality_notes = f"2x costs Sharpe={cost_2x_sharpe:.4f} (>0.5), slippage={'untested' if slip_sharpe == -999 else f'{slip_sharpe:.4f}'}"
    elif cost_2x_sharpe > 0:
        reality_score = 5
        reality_notes = f"2x costs Sharpe={cost_2x_sharpe:.4f} (positive but <0.5)"
    else:
        reality_score = 3
        reality_notes = f"2x costs Sharpe={cost_2x_sharpe:.4f} (negative -- would lose money)"

    # Weighted composite
    composite = (stat_score * 0.4 + robust_score * 0.3 + simple_score * 0.15 + reality_score * 0.15)

    # Verdict (anti-leniency: grade BEFORE verdict, never upgrade after)
    min_score = min(stat_score, robust_score, simple_score, reality_score)
    below_6 = sum(1 for s in [stat_score, robust_score, simple_score, reality_score] if s < 6)

    if min_score >= 7:
        verdict = "PASS"
    elif below_6 <= 1 and min_score >= 5:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"

    return {
        "statistical_validity": {"score": stat_score, "notes": stat_notes},
        "robustness": {"score": robust_score, "notes": robust_notes},
        "simplicity": {"score": simple_score, "notes": simple_notes},
        "reality_gap": {"score": reality_score, "notes": reality_notes},
        "composite": round(composite, 2),
        "verdict": verdict,
        "weak_periods": weak_periods,
        "sub_sharpes": sub_sharpes,
        "cost_2x_sharpe": cost_2x_sharpe,
        "full_sharpe": full_sharpe,
        "full_dsr": full_dsr,
    }


def _write_critique(results: dict, grades: dict):
    """Write handoff/evaluator_critique.md."""
    content = f"""# Evaluator Critique
Generated: {datetime.now(timezone.utc).isoformat()}

## Verdict: {grades['verdict']}

Composite Score: {grades['composite']}/10

## Criterion Scores (graded independently, anti-leniency protocol)

### 1. Statistical Validity (40%): {grades['statistical_validity']['score']}/10
{grades['statistical_validity']['notes']}

### 2. Robustness (30%): {grades['robustness']['score']}/10
{grades['robustness']['notes']}

### 3. Simplicity (15%): {grades['simplicity']['score']}/10
{grades['simplicity']['notes']}

### 4. Reality Gap (15%): {grades['reality_gap']['score']}/10
{grades['reality_gap']['notes']}

## Sub-Period Results
"""
    for label, _, _ in SUB_PERIODS:
        r = results.get(label, {})
        if "error" in r:
            content += f"- {label}: ERROR - {r['error']}\n"
        else:
            content += f"- {label}: Sharpe={r.get('sharpe', 0):.4f}, Return={r.get('total_return_pct', 0):.1f}%, MaxDD={r.get('max_drawdown', 0):.1f}%\n"

    r2x = results.get("2x_costs", {})
    if "error" in r2x:
        content += f"- 2x Costs: ERROR - {r2x['error']}\n"
    else:
        content += f"- 2x Costs: Sharpe={r2x.get('sharpe', 0):.4f}, Return={r2x.get('total_return_pct', 0):.1f}%\n"

    # Phase 2.8 hardening results
    conc = results.get("window_concentration")
    if conc:
        content += f"\n### Window Concentration\n"
        content += f"- Max concentration: {conc['max_concentration_pct']:.1f}% ({'PASS' if conc['pass'] else 'FAIL'} — threshold: <30%)\n"

    lb = results.get("ljung_box")
    if lb:
        content += f"\n### Ljung-Box Autocorrelation\n"
        content += f"- p-value: {lb['p_value']:.4f} ({'PASS' if lb['pass'] else 'FAIL'} — need p>0.05)\n"
        content += f"- {lb['interpretation']}\n"

    slip = results.get("slippage_5bps")
    if slip and "error" not in slip:
        content += f"\n### Slippage Stress Test (+5bps)\n"
        content += f"- Sharpe: {slip.get('sharpe', 0):.4f} ({'PASS' if slip.get('survives') else 'FAIL'} — need >0.5)\n"

    fs = results.get("feature_stability")
    if fs:
        content += f"\n### Feature Importance Stability\n"
        content += f"- Average Jaccard similarity: {fs['avg_jaccard']:.3f} ({'PASS' if fs['pass'] else 'FAIL'} — need >0.3)\n"
        content += f"- Common features across all sub-periods ({fs['n_common']}): {', '.join(fs['common_features'][:10])}\n"
        for pair, score in fs.get("pairs", {}).items():
            content += f"  - {pair}: {score:.3f}\n"

    content += f"""
## Decision
{"SAVE as new baseline." if grades['verdict'] == 'PASS' else "REVERT to previous best." if grades['verdict'] == 'FAIL' else "CONDITIONAL -- needs fixes before next cycle."}
"""
    (HANDOFF_DIR / "evaluator_critique.md").write_text(content, encoding="utf-8")
    logger.info("Wrote handoff/evaluator_critique.md (verdict=%s)", grades["verdict"])


# ── Planner: update research_plan.md ────────────────────────────

def update_research_plan(plan: dict, grades: dict, generator_result: dict, cycle: int):
    """Update handoff/research_plan.md with heuristic planner suggestions."""
    content = f"""# Research Plan -- Cycle {cycle + 1}

## Context
Cycle {cycle} completed. Generator ran {generator_result['num_trials']} trials.
Sharpe: {generator_result['initial_sharpe']:.4f} -> {generator_result['final_sharpe']:.4f} ({generator_result['delta']:+.4f})
Evaluator verdict: {grades['verdict']} (composite {grades['composite']}/10)

## Analysis of Experiment Log Patterns
"""
    experiments = _read_tsv_experiments()
    if experiments:
        recent = experiments[-20:]
        kept_count = sum(1 for e in recent if e.get("status") == "keep")
        content += f"\n- Recent 20 experiments: {kept_count} kept, {len(recent) - kept_count} discarded\n"

        # Param analysis
        param_stats = {}
        for exp in experiments:
            param = exp.get("param_changed", "--")
            if ":" in param:
                pname = param.split(":")[0].strip()
                status = exp.get("status", "")
                if pname not in param_stats:
                    param_stats[pname] = {"kept": 0, "discarded": 0}
                if status == "keep":
                    param_stats[pname]["kept"] += 1
                else:
                    param_stats[pname]["discarded"] += 1

        if param_stats:
            content += "\n### Per-Parameter Keep/Discard Rates\n"
            for pname, stats in sorted(param_stats.items(), key=lambda x: x[1]["kept"], reverse=True):
                total = stats["kept"] + stats["discarded"]
                rate = stats["kept"] / total * 100 if total > 0 else 0
                content += f"- {pname}: {stats['kept']}/{total} kept ({rate:.0f}%)\n"

    content += "\n## Proposed Research Directions (prioritized)\n\n"

    if grades["verdict"] == "FAIL":
        content += "### Direction: Recovery\n"
        content += "Previous cycle failed evaluation. Focus on:\n"
        for wp in grades.get("weak_periods", []):
            content += f"- Fix weak sub-period: {wp}\n"
        if grades["cost_2x_sharpe"] < 0.5:
            content += "- Reduce transaction cost sensitivity (lower turnover)\n"
    elif grades["verdict"] == "CONDITIONAL":
        content += "### Direction: Fix Conditional Issues\n"
        if grades["robustness"]["score"] < 7:
            content += "- Improve robustness across sub-periods\n"
        if grades["reality_gap"]["score"] < 7:
            content += "- Reduce reality gap (2x costs must survive)\n"
    else:
        content += "### Direction: Continue Optimization\n"
        content += "Strategy passed evaluation. Continue parameter search.\n"

    for suggestion in plan.get("suggestions", []):
        content += f"- {suggestion}\n"

    content += f"""
## Success Criteria for Next Cycle
- Sharpe >= {max(grades['full_sharpe'], 1.0):.2f}
- DSR >= 0.95
- All sub-periods Sharpe > 0.3
- 2x costs Sharpe > 0.7
"""

    (HANDOFF_DIR / "research_plan.md").write_text(content, encoding="utf-8")
    logger.info("Updated handoff/research_plan.md for cycle %d", cycle + 1)


# ── Harness Log ─────────────────────────────────────────────────

def append_harness_log(cycle: int, plan: dict, generator_result: dict, grades: dict, elapsed: float):
    """Append cycle summary to handoff/harness_log.md."""
    entry = f"""
---

## Cycle {cycle} -- {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

**Planner hypothesis:** {plan['hypothesis']}
**Generator:** {generator_result['num_trials']} trials, Sharpe {generator_result['initial_sharpe']:.4f} -> {generator_result['final_sharpe']:.4f} ({generator_result['delta']:+.4f}), kept={generator_result['kept']}, elapsed={generator_result['elapsed_sec']:.0f}s
**Evaluator verdict:** {grades['verdict']} (composite {grades['composite']}/10)
- Statistical: {grades['statistical_validity']['score']}/10
- Robustness: {grades['robustness']['score']}/10
- Simplicity: {grades['simplicity']['score']}/10
- Reality Gap: {grades['reality_gap']['score']}/10
- Sub-periods: {', '.join(f'{k}={v:.4f}' for k, v in grades.get('sub_sharpes', {}).items())}
- 2x costs: Sharpe={grades.get('cost_2x_sharpe', 0):.4f}
**Decision:** {'SAVE baseline' if grades['verdict'] == 'PASS' else 'REVERT to previous' if grades['verdict'] == 'FAIL' else 'CONDITIONAL -- kept with warning'}
**Total cycle time:** {elapsed:.0f}s
"""

    # Create or append
    if HARNESS_LOG.exists():
        existing = HARNESS_LOG.read_text(encoding="utf-8")
    else:
        existing = "# Harness Log\n\nAutomated three-agent harness loop. Each cycle: Planner -> Generator -> Evaluator.\n"

    HARNESS_LOG.write_text(existing + entry, encoding="utf-8")
    logger.info("Appended cycle %d to harness_log.md", cycle)


# ── Utilities ────────────────────────────────────────────────────

def _read_tsv_experiments() -> list[dict]:
    """Read quant_results.tsv into list of dicts."""
    if not TSV_PATH.exists():
        return []
    experiments = []
    try:
        with open(TSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                experiments.append(row)
    except Exception as e:
        logger.warning("Failed to read TSV: %s", e)
    return experiments


# ── Main Harness Loop ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Three-Agent Harness Loop (Anthropic pattern)")
    parser.add_argument("--cycles", type=int, default=3, help="Number of harness cycles (default: 3)")
    parser.add_argument("--iterations-per-cycle", type=int, default=10, help="Optimizer iterations per cycle (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only, don't run backtests")
    args = parser.parse_args()

    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    bq = BigQueryClient(settings)

    logger.info("=" * 60)
    logger.info("THREE-AGENT HARNESS LOOP")
    logger.info("Cycles: %d, Iterations/cycle: %d, Dry-run: %s", args.cycles, args.iterations_per_cycle, args.dry_run)
    logger.info("=" * 60)

    # Load initial baseline
    best_data = load_best_params()
    previous_best = copy.deepcopy(best_data)
    logger.info("Loaded baseline: Sharpe=%.4f, DSR=%.4f", best_data.get("sharpe", 0), best_data.get("dsr", 0))

    previous_critique = None

    for cycle in range(1, args.cycles + 1):
        cycle_start = time.time()
        logger.info("\n" + "=" * 60)
        logger.info("CYCLE %d/%d", cycle, args.cycles)
        logger.info("=" * 60)

        # ── Step 1: PLANNER ──
        logger.info("-- PLANNER --")
        plan = run_planner(cycle, previous_critique)
        for s in plan["suggestions"]:
            logger.info("  Suggestion: %s", s)

        # Save pre-generator baseline for revert
        pre_cycle_best = copy.deepcopy(load_best_params())

        # Write contract
        write_contract(plan, cycle, pre_cycle_best.get("sharpe", 0))

        if args.dry_run:
            logger.info("DRY RUN -- skipping generator and evaluator")
            append_harness_log(cycle, plan,
                               {"num_trials": 0, "initial_sharpe": 0, "final_sharpe": 0,
                                "delta": 0, "kept": 0, "elapsed_sec": 0},
                               {"verdict": "DRY_RUN", "composite": 0,
                                "statistical_validity": {"score": 0, "notes": "dry run"},
                                "robustness": {"score": 0, "notes": "dry run"},
                                "simplicity": {"score": 0, "notes": "dry run"},
                                "reality_gap": {"score": 0, "notes": "dry run"},
                                "sub_sharpes": {}, "cost_2x_sharpe": 0, "full_sharpe": 0,
                                "full_dsr": 0, "weak_periods": []},
                               time.time() - cycle_start)
            continue

        # ── Step 2: GENERATOR ──
        logger.info("-- GENERATOR --")
        generator_result = run_generator(args.iterations_per_cycle, settings, bq)

        # ── Step 3: EVALUATOR (independent, separate engines) ──
        logger.info("-- EVALUATOR --")
        post_gen_best = load_best_params()
        grades = run_evaluator(post_gen_best["params"], settings, bq)

        # ── Step 4: DECIDE ──
        if grades["verdict"] == "PASS":
            logger.info("PASS -- saving new baseline (Sharpe=%.4f)", generator_result["final_sharpe"])
            # optimizer_best.json already updated by generator
        elif grades["verdict"] == "FAIL":
            logger.info("FAIL -- reverting to pre-cycle baseline (Sharpe=%.4f)", pre_cycle_best.get("sharpe", 0))
            save_best_params(pre_cycle_best)
        else:
            # CONDITIONAL -- keep but warn
            logger.info("CONDITIONAL -- keeping result with warnings")

        # ── Step 5: UPDATE research plan ──
        update_research_plan(plan, grades, generator_result, cycle)

        # ── Step 6: LOG ──
        cycle_elapsed = time.time() - cycle_start
        append_harness_log(cycle, plan, generator_result, grades, cycle_elapsed)

        previous_critique = grades
        logger.info("Cycle %d complete in %.0fs. Verdict: %s", cycle, cycle_elapsed, grades["verdict"])

    logger.info("\n" + "=" * 60)
    logger.info("HARNESS COMPLETE -- %d cycles finished", args.cycles)
    final_best = load_best_params()
    logger.info("Final best: Sharpe=%.4f, DSR=%.4f", final_best.get("sharpe", 0), final_best.get("dsr", 0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
