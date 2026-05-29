"""phase-48.4: live rotation bake-off SMOKE -- first REAL validation of the
48.1-48.3 rotation machinery on actual backtests. AUDIT-ONLY (allocation_pct=0,
no deploy). $0 LLM (quant-only backtests). ~6-12+ min real compute.

Window 2022-01-01..2024-06-30 (default 12/3 -> 6 walk-forward windows, T~367 >>
the 32 PBO floor); 2 seeds (tb_baseline + qm_trend_tilt) x num_param_variants=2
= 4 real backtests. Keeps the BQ cache warm across both strategies (clear once
at the end). Prints SMOKE_RESULT_JSON: <json> with the selector verdict + the
captured per-seed {dsr,pbo,sharpe,...}.

Run: python scripts/run_rotation_smoke.py   (background it; tens of minutes possible)
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# Make `backend` importable when run as `python scripts/run_rotation_smoke.py`
# (sys.path[0] is scripts/, not the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.backtest.cache import clear_cache
    from backend.autoresearch.rotation_runner import make_rotation_engine, run_rotation_bakeoff
    from backend.autoresearch.strategy_backtest_adapter import make_engine_backtest_fn

    settings = get_settings()
    bq = BigQueryClient(settings)
    START, END = "2022-01-01", "2024-06-30"
    seeds = [
        {"id": "tb_baseline", "param_overrides": {}},
        {"id": "qm_trend_tilt", "param_overrides": {"strategy": "quality_momentum", "holding_days": 120}},
    ]

    print(f"[smoke] starting live rotation bake-off: {len(seeds)} seeds x 2 variants, {START}..{END}", flush=True)

    # full-kwarg real engine factory (closes over settings+bq)
    def factory(variant: dict):
        return make_rotation_engine(variant, settings, bq, start_date=START, end_date=END)

    # do NOT clear between strategies (warm cache across both -- same universe+window); clear ONCE at end
    real_adapter = make_engine_backtest_fn(factory, num_param_variants=2, clear_cache_fn=lambda: None)

    captured: dict[str, dict] = {}

    def capturing_adapter(params: dict) -> dict:
        m = real_adapter(params)
        captured[str(params.get("strategy"))] = m
        print(f"[smoke] scored strategy={params.get('strategy')}: {m}", flush=True)
        return m

    try:
        verdict = run_rotation_bakeoff(
            settings, bq,
            seeds=seeds,
            num_param_variants=2,
            start_date=START, end_date=END,
            persist=True,            # writes ONE rotation_log.jsonl row at allocation_pct=0 (audit-only)
            adapter_fn=capturing_adapter,
            log_path=None,           # default backend/backtest/experiments/rotation_log.jsonl
        )
    finally:
        try:
            clear_cache()
        except Exception as exc:
            print(f"[smoke] clear_cache warn: {exc}", flush=True)

    print("SMOKE_RESULT_JSON:" + json.dumps({"verdict": verdict, "captured": captured}, default=str), flush=True)
    print("[smoke] SMOKE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("[smoke] SMOKE_FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
