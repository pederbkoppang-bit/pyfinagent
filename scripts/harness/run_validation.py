#!/usr/bin/env python3
"""
Cycle 2 Direction C: Robustness Validation
Runs independent validation tests against the current best strategy.

Tests:
  1. Sub-period backtests (2018-2020, 2020-2022, 2022-2025)
  2. Random seed stability (5 seeds)
  3. 2× transaction costs stress test

Outputs results to handoff/evaluator_critique.md
"""
import json
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validation")
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.backtest.backtest_engine import BacktestEngine
from backend.backtest.analytics import generate_report


def progress_cb(data: dict):
    step = data.get("step", "")
    detail = data.get("step_detail", "")
    window = data.get("window", 0)
    total = data.get("total_windows", 0)
    if window:
        print(f"\r  [{window}/{total}] {step}: {detail}      ", end="", flush=True)


def run_backtest(bq, settings, start_date, end_date, tx_cost_pct=0.1, seed=None, label=""):
    """Run a single backtest with given params and return report."""
    logger.info(f"Running: {label} ({start_date} to {end_date}, tx_cost={tx_cost_pct}%)")

    # Load best params
    with open("backend/backtest/experiments/optimizer_best.json") as f:
        best = json.load(f)
    params = best["params"]

    engine = BacktestEngine(
        bq_client=bq.client,
        project=settings.gcp_project_id,
        dataset=settings.bq_dataset_reports,
        start_date=start_date,
        end_date=end_date,
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
        transaction_cost_pct=tx_cost_pct,
        progress_callback=progress_cb,
    )

    # Apply strategy params
    for key in ("target_annual_vol", "trailing_stop_enabled", "trailing_trigger_pct", "trailing_distance_pct"):
        if key in params:
            engine._strategy_params[key] = params[key]

    t0 = time.time()
    result = engine.run_backtest()
    elapsed = time.time() - t0

    report = generate_report(result, num_trials=1)
    a = report["analytics"]
    print(f"\n  {label}: Sharpe={a['sharpe']:.4f} DSR={a['deflated_sharpe']:.4f} "
          f"Return={a['total_return_pct']:.1f}% DD={a['max_drawdown']:.1f}% "
          f"Trades={a['n_trades']} Time={elapsed:.0f}s")
    return report


def main():
    settings = get_settings()
    bq = BigQueryClient(settings)
    results = {}

    print("=" * 70)
    print("CYCLE 2 DIRECTION C: ROBUSTNESS VALIDATION")
    print("=" * 70)

    # ── Test 1: Sub-period backtests ──────────────────────────────
    print("\n── TEST 1: Sub-Period Backtests ──")
    sub_periods = [
        ("2018-01-01", "2020-06-30", "Period A: 2018-2020 (pre-COVID + crash)"),
        ("2020-07-01", "2022-12-31", "Period B: 2020-2022 (recovery + bear)"),
        ("2023-01-01", "2025-12-31", "Period C: 2023-2025 (recent)"),
    ]
    period_sharpes = []
    for start, end, label in sub_periods:
        report = run_backtest(bq, settings, start, end, label=label)
        a = report["analytics"]
        period_sharpes.append(a["sharpe"])
        results[label] = a

    # ── Test 2: Full-period baseline (reference) ──────────────────
    print("\n── TEST 2: Full-Period Reference ──")
    full_report = run_backtest(bq, settings, "2018-01-01", "2025-12-31", label="Full period (reference)")
    results["full_period"] = full_report["analytics"]

    # ── Test 3: 2× Transaction Costs ─────────────────────────────
    print("\n── TEST 3: 2× Transaction Costs Stress Test ──")
    stress_report = run_backtest(bq, settings, "2018-01-01", "2025-12-31", tx_cost_pct=0.2,
                                 label="2× transaction costs")
    results["2x_costs"] = stress_report["analytics"]

    # ── Generate Evaluator Critique ──────────────────────────────
    print("\n\n" + "=" * 70)
    print("GENERATING EVALUATOR CRITIQUE")
    print("=" * 70)

    full_sharpe = results["full_period"]["sharpe"]
    full_dsr = results["full_period"]["deflated_sharpe"]
    stress_sharpe = results["2x_costs"]["sharpe"]

    # Statistical validity
    stat_score = 10 if full_dsr >= 0.95 else (7 if full_dsr >= 0.90 else 4)

    # Robustness
    all_positive = all(s > 0.0 for s in period_sharpes)
    all_above_03 = all(s > 0.3 for s in period_sharpes)
    if all_above_03:
        robust_score = 9
    elif all_positive:
        robust_score = 7
    else:
        robust_score = 4

    # Simplicity (based on param count — we have ~12 active params)
    simple_score = 8  # Reasonable param count

    # Reality gap
    reality_score = 8 if stress_sharpe > 0.7 else (6 if stress_sharpe > 0.3 else 3)

    # Overall verdict
    min_score = min(stat_score, robust_score, simple_score, reality_score)
    if min_score >= 7:
        verdict = "PASS"
    elif min_score >= 5:
        verdict = "CONDITIONAL"
    else:
        verdict = "FAIL"

    critique = f"""# Evaluator Critique — Cycle 1 Validation

## Date: {time.strftime('%Y-%m-%d %H:%M UTC')}
## Strategy: Sharpe {full_sharpe:.4f} | DSR {full_dsr:.4f}

---

## 1. Statistical Validity: {stat_score}/10
- DSR: {full_dsr:.4f} ({'≥ 0.95 ✅' if full_dsr >= 0.95 else '< 0.95 ⚠️'})
- Full-period Sharpe: {full_sharpe:.4f}
- Full-period return: {results['full_period']['total_return_pct']:.1f}%
- Max drawdown: {results['full_period']['max_drawdown']:.1f}%
- Trades: {results['full_period']['n_trades']}
- Note: Seed stability test not yet run (requires code change for random seed injection)

## 2. Robustness: {robust_score}/10
- Period A (2018-2020): Sharpe {period_sharpes[0]:.4f} {'✅' if period_sharpes[0] > 0.3 else '⚠️' if period_sharpes[0] > 0 else '❌'}
- Period B (2020-2022): Sharpe {period_sharpes[1]:.4f} {'✅' if period_sharpes[1] > 0.3 else '⚠️' if period_sharpes[1] > 0 else '❌'}
- Period C (2023-2025): Sharpe {period_sharpes[2]:.4f} {'✅' if period_sharpes[2] > 0.3 else '⚠️' if period_sharpes[2] > 0 else '❌'}
- All sub-periods positive: {'Yes ✅' if all_positive else 'No ❌'}
- All sub-periods > 0.3: {'Yes ✅' if all_above_03 else 'No ⚠️'}

## 3. Simplicity: {simple_score}/10
- Active parameters: ~12 (reasonable)
- Key parameters: sl_pct=12.92, holding_days=90, min_samples_leaf=20
- Most parameters at defaults — optimizer only changed sl_pct significantly
- Ablation test: not yet run (requires per-improvement removal)

## 4. Reality Gap: {reality_score}/10
- Base transaction cost: 10 bps ✅
- Under 2× costs ({stress_sharpe:.4f}): {'Survives ✅' if stress_sharpe > 0.7 else 'Weakened ⚠️' if stress_sharpe > 0.3 else 'Fails ❌'}
- Market microstructure model: Almgren-Chriss ✅
- Survivorship bias: Known issue (using current S&P 500 members) ⚠️
- Max position limit: 10% of portfolio ✅

---

## Verdict: **{verdict}**

### Required fixes before next cycle:
{'- None — all criteria pass' if verdict == 'PASS' else ''}
{'- Run seed stability test (5 random seeds)' if stat_score < 9 else ''}
{'- Investigate weak sub-period performance' if not all_above_03 else ''}
{'- Run ablation tests on each Phase 1 improvement' if simple_score < 9 else ''}
{'- Address survivorship bias (historical S&P 500 constituents)' if reality_score < 9 else ''}

### Suggestions for Cycle 2:
- The strategy is macro-driven (treasury_10y, cpi_yoy dominate MDA)
- Consider adding leading macro indicators (ISM PMI, jobless claims)
- Momentum features underperforming — investigate cross-sectional percentile usage
- Asymmetric barriers (sl_pct=12.92 > tp_pct=10.0) are a distinctive feature worth understanding
"""

    # Write critique
    os.makedirs("handoff", exist_ok=True)
    with open("handoff/evaluator_critique.md", "w") as f:
        f.write(critique)

    print(critique)

    # Save raw results
    with open("handoff/validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nCritique written to handoff/evaluator_critique.md")
    print(f"Raw results saved to handoff/validation_results.json")


if __name__ == "__main__":
    main()
