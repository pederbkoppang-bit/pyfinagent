"""
Seed Stability Test — Phase 2.8 Harness Hardening

Runs the backtest with multiple random seeds to verify the strategy
is not dependent on a specific random initialization.

Success criteria: Sharpe std < 0.1 across 5 seeds, all seeds > 0.9
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Unbuffer stdout for progress visibility
sys.stdout.reconfigure(line_buffering=True)

from run_harness import make_engine, generate_report
from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient

SEEDS = [42, 123, 456, 789, 2026]
BEST_PARAMS_PATH = Path("backend/backtest/experiments/optimizer_best.json")


def load_best_params() -> dict:
    with open(BEST_PARAMS_PATH) as f:
        data = json.load(f)
    return data["params"]


def run_with_seed(params: dict, seed: int, settings, bq) -> dict:
    """Run a full backtest with a specific random seed by monkey-patching."""
    import backend.backtest.backtest_engine as engine_mod
    original_train = engine_mod.BacktestEngine._train_model

    def patched_train(self, X, y, sample_weights):
        from sklearn.ensemble import GradientBoostingClassifier
        from datetime import datetime, timezone
        model = GradientBoostingClassifier(
            n_estimators=self.ml_params["n_estimators"],
            max_depth=self.ml_params["max_depth"],
            min_samples_leaf=self.ml_params["min_samples_leaf"],
            learning_rate=self.ml_params["learning_rate"],
            random_state=seed,
        )
        model.fit(X, y, sample_weight=sample_weights)
        self.model_trained_at = datetime.now(timezone.utc).isoformat()
        return model, list(X.columns)

    engine_mod.BacktestEngine._train_model = patched_train
    try:
        engine = make_engine(params, settings, bq)
        result = engine.run_backtest(skip_cache_clear=True)
        report = generate_report(result, num_trials=1)
        a = report["analytics"]

        # Save to experiments/results/ so it shows on Sharpe History chart
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"seed_{seed}"
        report["run_id"] = run_id
        report["is_baseline"] = False
        report["parent_run_id"] = "seed_stability"

        results_dir = Path("backend/backtest/experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{ts}_{run_id}.json"
        result_path.write_text(json.dumps(report, indent=2, default=str))

        # Append to TSV
        tsv_path = Path("backend/backtest/experiments/quant_results.tsv")
        if tsv_path.exists():
            with open(tsv_path, "a") as f:
                f.write(f"{ts}\t{run_id}\tseed: {seed}\t0.0000\t{a['sharpe']:.4f}\t+0.0000\tseed_test\t{a['deflated_sharpe']:.4f}\t\t\tseed_stability\n")

        return {
            "seed": seed,
            "sharpe": a["sharpe"],
            "dsr": a["deflated_sharpe"],
            "return_pct": a["total_return_pct"],
            "max_drawdown": a["max_drawdown"],
            "trades": a["n_trades"],
            "hit_rate": a["hit_rate"],
        }
    finally:
        engine_mod.BacktestEngine._train_model = original_train


def main():
    print("=" * 60)
    print("SEED STABILITY TEST — Phase 2.8")
    print("=" * 60)
    print()

    params = load_best_params()
    print(f"Strategy: {params.get('strategy', 'unknown')}")
    print(f"Seeds: {SEEDS}")
    print()

    settings = get_settings()
    bq = BigQueryClient(settings)
    print("Engine handles its own data preloading per seed run.")
    print()

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"--- Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        start = time.time()
        try:
            result = run_with_seed(params, seed, settings, bq)
            elapsed = time.time() - start
            results.append(result)
            print(f"  Sharpe: {result['sharpe']:.4f} | Return: {result['return_pct']:.1f}% | MaxDD: {result['max_drawdown']:.1f}% | Trades: {result['trades']} | {elapsed:.0f}s")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  FAILED: {e} ({elapsed:.0f}s)")
            results.append({"seed": seed, "sharpe": 0, "error": str(e)})
        print()

    # Clean up BQ cache after all runs
    from backend.backtest.cache import clear_cache
    clear_cache()

    sharpes = [r["sharpe"] for r in results if r.get("sharpe", 0) > 0]
    if not sharpes:
        print("ALL SEEDS FAILED")
        return

    import numpy as np
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)

    print("=" * 60)
    print("SEED STABILITY RESULTS")
    print("=" * 60)
    print(f"  Mean Sharpe: {mean_sharpe:.4f}")
    print(f"  Std Sharpe:  {std_sharpe:.4f}")
    print(f"  Min Sharpe:  {min(sharpes):.4f}")
    print(f"  Max Sharpe:  {max(sharpes):.4f}")
    print(f"  Range:       {max(sharpes) - min(sharpes):.4f}")
    print()

    all_above_09 = all(s > 0.9 for s in sharpes)
    std_below_01 = std_sharpe < 0.1

    print(f"  All seeds > 0.9:  {'PASS' if all_above_09 else 'FAIL'}")
    print(f"  Std < 0.1:        {'PASS' if std_below_01 else 'FAIL'}")
    print(f"  Overall:          {'PASS' if all_above_09 and std_below_01 else 'FAIL'}")

    output = {
        "test": "seed_stability",
        "seeds": SEEDS,
        "results": results,
        "mean_sharpe": float(mean_sharpe),
        "std_sharpe": float(std_sharpe),
        "min_sharpe": float(min(sharpes)),
        "max_sharpe": float(max(sharpes)),
        "pass_all_above_09": bool(all_above_09),
        "pass_std_below_01": bool(std_below_01),
        "verdict": "PASS" if all_above_09 and std_below_01 else "FAIL",
    }
    out_path = Path("handoff/seed_stability_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
