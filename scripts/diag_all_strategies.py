"""phase-48.5 grounding: which STRATEGY_REGISTRY strategies actually TRADE (and at
what Sharpe) on 2022-01-01..2024-06-30? Determines whether a MEANINGFUL rotation
seed set exists (>=2-3 distinct trading strategies) or whether rotation is stuck
on triple_barrier. $0 LLM; warm cache across strategies (clear once at end).
triple_barrier already known: 160 trades, Sharpe ~1.82. quality_momentum already
known: 0 trades (degenerate). Tests the remaining: mean_reversion, factor_model,
meta_label (+ quality_momentum again for confirmation)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.backtest.cache import clear_cache
    from backend.autoresearch.rotation_runner import make_rotation_engine

    s = get_settings()
    bq = BigQueryClient(s)
    START, END = "2022-01-01", "2024-06-30"
    strategies = ["mean_reversion", "factor_model", "meta_label", "quality_momentum"]
    print(f"[diag-all] testing {strategies} over {START}..{END} (warm cache)", flush=True)
    results = {}
    try:
        for strat in strategies:
            try:
                eng = make_rotation_engine({"strategy": strat}, s, bq, start_date=START, end_date=END)
                r = eng.run_backtest(skip_cache_clear=True)
                results[strat] = {
                    "total_trades": r.total_trades,
                    "aggregate_sharpe": round(r.aggregate_sharpe, 4),
                    "aggregate_return_pct": round(r.aggregate_return_pct, 3),
                    "n_windows": len(r.windows),
                    "nav_len": len(r.nav_history),
                }
                print(f"[diag-all] {strat}: {results[strat]}", flush=True)
            except Exception as exc:
                results[strat] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[diag-all] {strat} FAILED: {exc}", flush=True)
    finally:
        try:
            clear_cache()
        except Exception:
            pass
    import json
    print("DIAG_ALL_JSON:" + json.dumps(results, default=str), flush=True)
    print("[diag-all] DIAG_ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception:
        print("[diag-all] DIAG_ALL_FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
