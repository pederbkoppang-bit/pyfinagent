#!/usr/bin/env python3
"""
Run a single targeted backtest experiment.
Modify params manually and see the result.
"""
import json
import logging
import os
import sys
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("experiment")
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from dotenv import load_dotenv
load_dotenv("backend/.env")

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.backtest.backtest_engine import BacktestEngine
from backend.backtest.analytics import generate_report

settings = get_settings()
bq = BigQueryClient(settings)

# ── EXPERIMENT PARAMS ──
# Tweak these to test different configurations
EXPERIMENTS = [
    {
        "name": "baseline_tb",
        "strategy": "triple_barrier",
        "params": {}  # default params
    },
    {
        "name": "wider_barriers",
        "strategy": "triple_barrier",
        "params": {
            "tp_pct": 15.0,   # was 10
            "sl_pct": 8.0,    # was 10 — asymmetric: let winners run
        }
    },
    {
        "name": "more_positions",
        "strategy": "triple_barrier",
        "params": {
            "max_positions": 30,   # was 20
            "top_n_candidates": 80, # was 50
        }
    },
    {
        "name": "aggressive_ml",
        "strategy": "triple_barrier",
        "params": {
            "n_estimators": 300,    # was 200
            "max_depth": 5,         # was 4
            "min_samples_leaf": 15, # was 20
        }
    },
]

# Pick which experiment to run from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="wider_barriers", help="Experiment name")
parser.add_argument("--list", action="store_true", help="List experiments")
args = parser.parse_args()

if args.list:
    for exp in EXPERIMENTS:
        print(f"  {exp['name']}: {exp['params']}")
    sys.exit(0)

exp = next((e for e in EXPERIMENTS if e["name"] == args.exp), None)
if not exp:
    print(f"Unknown experiment: {args.exp}")
    print("Available:", [e["name"] for e in EXPERIMENTS])
    sys.exit(1)

logger.info("Running experiment: %s (params: %s)", exp["name"], exp["params"])

engine = BacktestEngine(
    bq_client=bq.client,
    project=settings.gcp_project_id,
    dataset=settings.bq_dataset_reports,
    start_date=settings.backtest_start_date,
    end_date=settings.backtest_end_date,
    strategy=exp["strategy"],
    commission_model=settings.backtest_commission_model,
    commission_per_share=settings.backtest_commission_per_share,
)

# Apply experiment params
for key, value in exp["params"].items():
    if key in ("n_estimators", "max_depth", "min_samples_leaf", "learning_rate"):
        engine.ml_params[key] = value
    elif key == "max_positions":
        engine.trader.max_positions = value
    elif key == "tp_pct":
        engine.tp_pct = value
    elif key == "sl_pct":
        engine.sl_pct = value
    elif key == "top_n_candidates":
        engine.top_n_candidates = value
    elif key == "holding_days":
        engine.holding_days = value
    else:
        engine._strategy_params[key] = value

t0 = time.time()
result = engine.run_backtest()
elapsed = time.time() - t0

report = generate_report(result, num_trials=1)
a = report["analytics"]

print(f"\n\n=== EXPERIMENT: {exp['name']} ===")
print(f"  Strategy:   {exp['strategy']}")
print(f"  Params:     {json.dumps(exp['params'], indent=2)}")
print(f"  Sharpe:     {a['sharpe']:.4f}")
print(f"  DSR:        {a['deflated_sharpe']:.4f} ({'✅' if a['dsr_significant'] else '❌'})")
print(f"  Return:     {a['total_return_pct']:.1f}%")
print(f"  Max DD:     {a['max_drawdown']:.1f}%")
print(f"  Hit Rate:   {a['hit_rate']:.1%}")
print(f"  Trades:     {a['n_trades']}")
print(f"  Time:       {elapsed:.0f}s ({elapsed/60:.1f}m)")

# Save result
from pathlib import Path
outpath = Path("backend/backtest/experiments/results")
outpath.mkdir(parents=True, exist_ok=True)
fname = outpath / f"exp_{exp['name']}_{time.strftime('%Y%m%dT%H%M%S')}.json"
report["experiment"] = exp
fname.write_text(json.dumps(report, indent=2, default=str))
print(f"  Saved:      {fname}")

# Force GC
del engine, result, report
gc.collect()
