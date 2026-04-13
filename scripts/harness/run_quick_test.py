#!/usr/bin/env python3
"""Quick single-run validation to verify Phase 1.2 fix."""
import json, os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("quick_test")
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

with open("backend/backtest/experiments/optimizer_best.json") as f:
    best = json.load(f)
params = best["params"]

def progress_cb(data):
    step = data.get("step", "")
    detail = data.get("step_detail", "")
    window = data.get("window", 0)
    total = data.get("total_windows", 0)
    if window:
        print(f"\r  [{window}/{total}] {step}: {detail}      ", end="", flush=True)

engine = BacktestEngine(
    bq_client=bq.client,
    project=settings.gcp_project_id,
    dataset=settings.bq_dataset_reports,
    start_date="2018-01-01",
    end_date="2025-12-31",
    strategy=params.get("strategy", "triple_barrier"),
    holding_days=params.get("holding_days", 90),
    tp_pct=params.get("tp_pct", 10.0),
    sl_pct=params.get("sl_pct", 12.923403579416114),
    min_samples_leaf=params.get("min_samples_leaf", 20),
    max_positions=params.get("max_positions", 20),
    top_n_candidates=params.get("top_n_candidates", 50),
    transaction_cost_pct=0.1,
    progress_callback=progress_cb,
)

for key in ("target_annual_vol", "trailing_stop_enabled", "trailing_trigger_pct", "trailing_distance_pct"):
    if key in params:
        engine._strategy_params[key] = params[key]

t0 = time.time()
result = engine.run_backtest()
elapsed = time.time() - t0

report = generate_report(result, num_trials=1)
a = report["analytics"]
print(f'\n\n=== RESULT ===')
print(f'Sharpe={a["sharpe"]:.4f} DSR={a["deflated_sharpe"]:.4f} Return={a["total_return_pct"]:.1f}% DD={a["max_drawdown"]:.1f}% Trades={a["n_trades"]} Time={elapsed:.0f}s')
print(f'Previous optimizer result: Sharpe=1.1705')
print(f'Previous validation result: Sharpe=-4.5716')
print(f'Delta from optimizer: {a["sharpe"] - 1.1705:+.4f}')
