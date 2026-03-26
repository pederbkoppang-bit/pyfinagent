#!/usr/bin/env python3
"""
Run the quant optimizer (autoresearch loop) from CLI.
Usage: python run_optimizer.py [--strategy blend] [--iterations 50]
"""
import argparse
import json
import logging
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_optimizer")

# Suppress noisy loggers
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="PyFinAgent Quant Optimizer")
    parser.add_argument("--strategy", default="blend", help="Strategy to use (default: blend)")
    parser.add_argument("--iterations", type=int, default=50, help="Max optimizer iterations")
    parser.add_argument("--baseline-only", action="store_true", help="Just run baseline, no optimization")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv("backend/.env")

    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.backtest.backtest_engine import BacktestEngine
    from backend.backtest.quant_optimizer import QuantStrategyOptimizer
    from backend.backtest.analytics import generate_report

    settings = get_settings()
    bq = BigQueryClient(settings)

    logger.info("Creating BacktestEngine: strategy=%s, %s to %s",
                args.strategy, settings.backtest_start_date, settings.backtest_end_date)

    def progress_cb(data: dict):
        step = data.get("step", "")
        detail = data.get("step_detail", "")
        window = data.get("window", 0)
        total = data.get("total_windows", 0)
        if window:
            print(f"\r  [{window}/{total}] {step}: {detail}      ", end="", flush=True)
        else:
            print(f"\r  {step}: {detail}      ", end="", flush=True)

    engine = BacktestEngine(
        bq_client=bq.client,
        project=settings.gcp_project_id,
        dataset=settings.bq_dataset_reports,
        start_date=settings.backtest_start_date,
        end_date=settings.backtest_end_date,
        strategy=args.strategy,
        progress_callback=progress_cb,
        commission_model=settings.backtest_commission_model,
        commission_per_share=settings.backtest_commission_per_share,
    )

    if args.baseline_only:
        logger.info("Running baseline backtest (strategy=%s)...", args.strategy)
        t0 = time.time()
        result = engine.run_backtest()
        elapsed = time.time() - t0
        report = generate_report(result, num_trials=1)
        a = report["analytics"]
        print(f"\n\n=== BASELINE RESULTS (strategy={args.strategy}) ===")
        print(f"  Sharpe:     {a['sharpe']:.4f}")
        print(f"  DSR:        {a['deflated_sharpe']:.4f} ({'✅' if a['dsr_significant'] else '❌'})")
        print(f"  Return:     {a['total_return_pct']:.1f}%")
        print(f"  Max DD:     {a['max_drawdown']:.1f}%")
        print(f"  Hit Rate:   {a['hit_rate']:.1%}")
        print(f"  Trades:     {a['n_trades']}")
        print(f"  Windows:    {a['n_windows']}")
        print(f"  Time:       {elapsed:.0f}s")
        # Save result
        from pathlib import Path
        outpath = Path("backend/backtest/experiments/results")
        outpath.mkdir(parents=True, exist_ok=True)
        fname = outpath / f"baseline_{args.strategy}_{time.strftime('%Y%m%dT%H%M%S')}.json"
        fname.write_text(json.dumps(report, indent=2, default=str))
        print(f"  Saved:      {fname}")
        return

    # Run optimizer
    logger.info("Starting optimizer: strategy=%s, max_iterations=%d", args.strategy, args.iterations)

    def status_cb(iterations, best_sharpe, best_dsr, kept, discarded,
                  current_step="", current_detail="", run_id=""):
        if current_step == "experiment_result":
            print(f"\n  [{iterations}] Sharpe={best_sharpe:.4f} DSR={best_dsr:.4f} kept={kept} discarded={discarded}")
        elif current_step == "baseline_complete":
            print(f"\n  Baseline: Sharpe={best_sharpe:.4f} DSR={best_dsr:.4f}")

    optimizer = QuantStrategyOptimizer(engine, status_callback=status_cb)

    def on_result(report):
        a = report["analytics"]
        print(f"  → Result: Sharpe={a['sharpe']:.4f} Return={a['total_return_pct']:.1f}% DD={a['max_drawdown']:.1f}%")

    t0 = time.time()
    optimizer.run_loop(
        max_iterations=args.iterations,
        use_llm=False,
        on_result=on_result,
    )
    elapsed = time.time() - t0

    print(f"\n\n=== OPTIMIZER COMPLETE ===")
    print(f"  Best Sharpe: {optimizer.best_sharpe:.4f}")
    print(f"  Best DSR:    {optimizer.best_dsr:.4f}")
    print(f"  Kept:        {optimizer.kept}")
    print(f"  Discarded:   {optimizer.discarded}")
    print(f"  Total time:  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Best params: {json.dumps(optimizer.best_params, indent=2)}")


if __name__ == "__main__":
    main()
