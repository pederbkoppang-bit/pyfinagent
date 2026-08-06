#!/usr/bin/env python3
"""phase-82.51 -- A/B the publication-lag embargo on a REAL backtest.

Criterion 4 asks for before/after Sharpe and trade-count deltas from a real
backtest. Run this TWICE with the embargo as the ONLY variable:

    FUNDAMENTALS_EMBARGO_DAYS=0  python scripts/backtest/run_82_51_embargo_ab.py
    FUNDAMENTALS_EMBARGO_DAYS=60 python scripts/backtest/run_82_51_embargo_ab.py

WHY `qarp`: it is the only strategy whose LABEL function reads a
fundamentals-only key (derived live by
`label_fundamentals_dependent_strategies()`), so its labels move directly with
the embargo. Note the step's claim that non-fundamentals strategies would show
a ZERO delta is FALSE -- every strategy is feature-dependent via
`_NUMERIC_FEATURES`, so `triple_barrier` would move too, just less
interpretably.

WHY the window starts 2025-06-30: at a 2025-03-31 cutoff a 60-day embargo takes
5-quarter coverage from 41.7% of the universe to 0.0%, so an earlier start
would measure the coverage hole rather than the embargo.

PRE-REGISTERED EXPECTATION: Sharpe DOWN or flat, n_trades DOWN or flat. A
delta of exactly 0.0000 is an ALARM -- it most likely means the embargo was not
actually applied.
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for noisy in ("google", "urllib3", "backend.backtest"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

from dotenv import load_dotenv  # noqa: E402

load_dotenv("backend/.env")

from backend.backtest.analytics import generate_report  # noqa: E402
from backend.backtest.backtest_engine import BacktestEngine  # noqa: E402
from backend.backtest import fundamentals_coverage as fc  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.db.bigquery_client import BigQueryClient  # noqa: E402

settings = get_settings()
bq = BigQueryClient(settings)

print("=== phase-82.51 embargo A/B ===")
print(f"FUNDAMENTALS_EMBARGO_DAYS = {fc.FUNDAMENTALS_EMBARGO_DAYS}")
print(f"raw coverage start        = {fc.FUNDAMENTALS_COVERAGE_START}")
print(f"effective coverage start  = {fc.effective_coverage_start()}")
print(f"label-dependent strategies= {sorted(fc.label_fundamentals_dependent_strategies())}")
sys.stdout.flush()


def progress_cb(data):
    w, t = data.get("window", 0), data.get("total_windows", 0)
    if w:
        print(f"\r  [{w}/{t}] {data.get('step','')}: {data.get('step_detail','')}      ",
              end="", flush=True)


engine = BacktestEngine(
    bq_client=bq.client,
    project=settings.gcp_project_id,
    dataset=settings.bq_dataset_reports,   # financial_reports, NOT pyfinagent_data
    start_date="2025-06-30",
    end_date="2026-02-28",                 # the table's MAX(report_date)
    train_window_months=6,
    test_window_months=2,
    strategy="qarp",
    holding_days=90,
    tp_pct=10.0,
    sl_pct=12.923403579416114,
    min_samples_leaf=20,
    max_positions=20,
    top_n_candidates=50,
    transaction_cost_pct=0.1,
    progress_callback=progress_cb,
)

t0 = time.time()
result = engine.run_backtest()
elapsed = time.time() - t0

report = generate_report(result, num_trials=1)
a = report["analytics"]

print("\n\n=== RESULT ===")
print(f'embargo_days={fc.FUNDAMENTALS_EMBARGO_DAYS}')
print(f'sharpe={a["sharpe"]:.4f}')
print(f'deflated_sharpe={a["deflated_sharpe"]:.4f}')
print(f'n_trades={a["n_trades"]}')
print(f'total_return_pct={a["total_return_pct"]:.4f}')
print(f'max_drawdown={a["max_drawdown"]:.4f}')
print(f'elapsed_s={elapsed:.0f}')
print(f'data_availability={result.data_availability}')
