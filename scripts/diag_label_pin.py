"""phase-48.5 root-cause pin: WHY do mean_reversion/factor_model/quality_momentum
produce 0 training samples? Preload the cache (as run_backtest does), then for
real 2022/2023 candidates inspect the exact feature values + run each label
method, printing which None-path triggers. $0 LLM, ~1 min. Read-only diagnosis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.backtest import cache
    from backend.autoresearch.rotation_runner import make_rotation_engine

    s = get_settings()
    bq = BigQueryClient(s)
    tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "WMT"]
    cache.init_cache(bq.client, s.gcp_project_id, s.bq_dataset_reports)
    # mimic run_backtest preload: backward buffer included
    print(f"[pin] preloading prices {tickers} 2020-06-01..2023-12-31", flush=True)
    cache.preload_prices(tickers + ["SPY"], "2020-06-01", "2023-12-31")
    cache.preload_fundamentals(tickers)
    cache.preload_macro()

    eng = make_rotation_engine({"strategy": "mean_reversion"}, s, bq,
                               start_date="2022-01-01", end_date="2023-06-30")
    dp = eng.data_provider
    dates = ["2022-06-01", "2023-03-01"]
    feat_keys = ["price_at_analysis", "sma_50_distance", "rsi_14", "momentum_6m",
                 "momentum_12m", "annualized_volatility", "quality_score", "pb_ratio"]
    for t in tickers[:3]:
        for d in dates:
            px = dp.get_point_in_time_prices(t, d)
            fv = dp.build_feature_vector(t, d)
            print(f"\n[pin] {t}@{d}: lookback_rows={len(px)} fv_keys={len(fv)}", flush=True)
            for k in feat_keys:
                print(f"   {k} = {fv.get(k)}", flush=True)
            # run each label method
            try:
                mr = eng._compute_mean_reversion_label(t, d)
            except Exception as e:
                mr = f"ERR:{e}"
            try:
                qm = eng._compute_quality_momentum_label(t, d)
            except Exception as e:
                qm = f"ERR:{e}"
            try:
                fac = eng._compute_factor_label(t, d)
            except Exception as e:
                fac = f"ERR:{e}"
            try:
                tb = eng._compute_triple_barrier_label(t, d)
            except Exception as e:
                tb = f"ERR:{e}"
            print(f"   LABELS: mean_reversion={mr} quality_momentum={qm} factor={fac} triple_barrier={tb}", flush=True)
    print("\n[pin] PIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception:
        print("[pin] PIN_FAILED:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
