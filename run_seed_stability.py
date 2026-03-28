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

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.backtest.backtest_engine import BacktestEngine
from backend.backtest.cache import preload_prices, preload_fundamentals, preload_macro, init_cache
from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient

SEEDS = [42, 123, 456, 789, 2026]
BEST_PARAMS_PATH = Path("backend/backtest/experiments/optimizer_best.json")


def load_best_params() -> dict:
    with open(BEST_PARAMS_PATH) as f:
        data = json.load(f)
    return data["params"]


def run_with_seed(params: dict, seed: int) -> dict:
    """Run a full backtest with a specific random seed."""
    # Monkey-patch the random_state in the engine
    import backend.backtest.backtest_engine as engine_mod
    original_train = engine_mod.BacktestEngine._train_model

    def patched_train(self, X, y, sample_weights):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=self.ml_params["n_estimators"],
            max_depth=self.ml_params["max_depth"],
            min_samples_leaf=self.ml_params["min_samples_leaf"],
            learning_rate=self.ml_params["learning_rate"],
            random_state=seed,
        )
        model.fit(X, y, sample_weight=sample_weights)
        from datetime import datetime, timezone
        self.model_trained_at = datetime.now(timezone.utc).isoformat()
        return model, list(X.columns)

    engine_mod.BacktestEngine._train_model = patched_train

    try:
        settings = get_settings()
        bq = BigQueryClient(settings)
        # Map optimizer param names to engine param names
        engine_params = {}
        skip_keys = {"starting_capital", "target_annual_vol", "trailing_stop_enabled",
                      "trailing_trigger_pct", "trailing_distance_pct"}
        for k, v in params.items():
            if k in skip_keys:
                continue
            engine_params[k] = v
        engine = BacktestEngine(
            bq_client=bq,
            project=settings.gcp_project_id,
            dataset="financial_reports",
            **engine_params,
        )
        result = engine.run_backtest(skip_cache_clear=True)
        return {
            "seed": seed,
            "sharpe": result.aggregate_sharpe,
            "return_pct": result.aggregate_return_pct,
            "max_drawdown": result.aggregate_max_drawdown_pct,
            "trades": result.total_trades,
            "hit_rate": result.aggregate_hit_rate,
        }
    finally:
        engine_mod.BacktestEngine._train_model = original_train


def main():
    print("=" * 60)
    print("SEED STABILITY TEST — Phase 2.8")
    print("=" * 60)
    print()

    # Load best params
    params = load_best_params()
    print(f"Strategy: {params.get('strategy', 'unknown')}")
    print(f"Seeds: {SEEDS}")
    print()

    # Preload data once — get tickers from BQ
    print("Preloading data...")
    settings = get_settings()
    bq = BigQueryClient(settings)
    init_cache(bq.client, settings.gcp_project_id, "financial_reports")
    q = f"SELECT DISTINCT ticker FROM `{settings.gcp_project_id}.financial_reports.historical_prices` LIMIT 500"
    tickers = [row.ticker for row in bq.client.query(q, timeout=30).result()]
    print(f"  {len(tickers)} tickers from BQ")
    preload_prices(tickers, params["start_date"], params["end_date"])
    preload_fundamentals(tickers)
    preload_macro()
    print("Data preloaded.")
    print()

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"--- Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        start = time.time()
        result = run_with_seed(params, seed)
        elapsed = time.time() - start
        results.append(result)
        print(f"  Sharpe: {result['sharpe']:.4f} | Return: {result['return_pct']:.1f}% | MaxDD: {result['max_drawdown']:.1f}% | Trades: {result['trades']} | {elapsed:.0f}s")
        print()

    # Analyze stability
    sharpes = [r["sharpe"] for r in results]
    import numpy as np
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    min_sharpe = min(sharpes)
    max_sharpe = max(sharpes)

    print("=" * 60)
    print("SEED STABILITY RESULTS")
    print("=" * 60)
    print(f"  Mean Sharpe: {mean_sharpe:.4f}")
    print(f"  Std Sharpe:  {std_sharpe:.4f}")
    print(f"  Min Sharpe:  {min_sharpe:.4f}")
    print(f"  Max Sharpe:  {max_sharpe:.4f}")
    print(f"  Range:       {max_sharpe - min_sharpe:.4f}")
    print()

    # Pass/fail
    all_above_09 = all(s > 0.9 for s in sharpes)
    std_below_01 = std_sharpe < 0.1
    print(f"  All seeds > 0.9:  {'PASS' if all_above_09 else 'FAIL'}")
    print(f"  Std < 0.1:        {'PASS' if std_below_01 else 'FAIL'}")
    print(f"  Overall:          {'PASS' if all_above_09 and std_below_01 else 'FAIL'}")

    # Save results
    output = {
        "test": "seed_stability",
        "seeds": SEEDS,
        "results": results,
        "mean_sharpe": float(mean_sharpe),
        "std_sharpe": float(std_sharpe),
        "min_sharpe": float(min_sharpe),
        "max_sharpe": float(max_sharpe),
        "pass_all_above_09": all_above_09,
        "pass_std_below_01": std_below_01,
        "verdict": "PASS" if all_above_09 and std_below_01 else "FAIL",
    }
    out_path = Path("handoff/seed_stability_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
