"""Quick sub-period + 2x cost validation on pre-Phase-1.2 code."""
import json, sys, os, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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

def progress_cb(data: dict):
    step = data.get("step", "")
    detail = data.get("step_detail", "")
    window = data.get("window", 0)
    total = data.get("total_windows", 0)
    if window:
        print(f"\r  [{window}/{total}] {step}: {detail}      ", end="", flush=True)

def run_test(label, start_date, end_date, tx_cost_pct=0.1):
    t0 = time.time()
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
    result = engine.run_backtest()
    elapsed = time.time() - t0
    report = generate_report(result, num_trials=1)
    a = report["analytics"]
    print(f"\n  {label}: Sharpe={a['sharpe']:.4f} DSR={a['deflated_sharpe']:.4f} "
          f"Return={a['total_return_pct']:.1f}% DD={a['max_drawdown']:.1f}% "
          f"Trades={a['n_trades']} Time={elapsed:.0f}s")
    return a

results = {}
tests = [
    ("Period A: 2018-2020", "2018-01-01", "2020-12-31", 0.1),
    ("Period B: 2020-2022", "2020-01-01", "2022-12-31", 0.1),
    ("Period C: 2023-2025", "2023-01-01", "2025-12-31", 0.1),
    ("2x_costs", "2018-01-01", "2025-12-31", 0.2),
]

for label, start, end, cost in tests:
    print(f"\n{'='*60}")
    print(f"Running: {label} ({start} to {end}, tx_cost={cost}%)")
    print(f"{'='*60}")
    try:
        results[label] = run_test(label, start, end, cost)
    except Exception as e:
        print(f"\n❌ {label}: FAILED - {e}")
        import traceback; traceback.print_exc()
        results[label] = {"error": str(e)}

with open("handoff/subperiod_validation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, r in results.items():
    if "error" in r:
        print(f"  ❌ {name}: ERROR - {r['error']}")
    else:
        status = "✅" if r["sharpe"] > 0 else "❌"
        print(f"  {status} {name}: Sharpe={r['sharpe']:.4f}, Return={r['total_return_pct']:.1f}%")
print("\nSaved to handoff/subperiod_validation_results.json")
