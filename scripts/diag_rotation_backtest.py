"""phase-48.4 diagnostic: characterize WHY the smoke backtests came back degenerate
(sharpe=0, short nav, undersized PBO matrix). Runs ONE tb_baseline backtest over
the smoke window and dumps total_trades / aggregate_sharpe / nav_history length /
per-window trades+candidates. $0 LLM. ~5-10 min cold."""
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
    strat = sys.argv[1] if len(sys.argv) > 1 else "triple_barrier"
    print(f"[diag] running ONE {strat} backtest (FRESH, isolated) 2022-01-01..2024-06-30", flush=True)
    eng = make_rotation_engine({"strategy": strat}, s, bq,
                               start_date="2022-01-01", end_date="2024-06-30")
    try:
        r = eng.run_backtest(skip_cache_clear=True)
    finally:
        try:
            clear_cache()
        except Exception:
            pass
    print(f"DIAG total_trades={r.total_trades} aggregate_sharpe={r.aggregate_sharpe} "
          f"aggregate_return_pct={r.aggregate_return_pct} n_windows={len(r.windows)} "
          f"nav_history_len={len(r.nav_history)}", flush=True)
    for w in r.windows:
        print(f"  window {w.window_id}: test {w.test_start}..{w.test_end} "
              f"sharpe={w.sharpe_ratio} num_trades={w.num_trades} n_candidates={w.n_candidates} "
              f"n_train={w.n_train_samples}", flush=True)
    if r.nav_history:
        print(f"  nav_history head={r.nav_history[:2]} tail={r.nav_history[-2:]}", flush=True)
    print(f"  all_trades_count={len(r.all_trades)} sample={r.all_trades[:2]}", flush=True)
    print("[diag] DIAG_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception:
        print("[diag] DIAG_FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
