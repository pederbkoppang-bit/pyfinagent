"""phase-82.3 -- backtest evidence for the overpriced-market candidates.

Runs a K=8 factorial per strategy over {triple_barrier (incumbent), stretch_regime,
qarp, reversion_sigma} = 32 walk-forward runs on ONE sample, then reports DSR,
PBO, turnover and net-of-cost return per strategy.

DESIGN NOTES -- each one is load-bearing, from the 82.3 research gate:

* PBO's T-AXIS IS DAILY NAV RETURNS, NOT PER-WINDOW RETURNS.
  `analytics.compute_pbo` returns **0.0 silently when T < S*2** (32 at S=16), and
  0.0 PASSES the `<= 0.5` gate. Per-window returns give T ~ 27 -> a fabricated
  pass on every strategy. Daily NAV gives T ~ 1900.

* ONE MATRIX PER STRATEGY. Bailey/Borwein/Lopez de Prado/Zhu Algorithm 2.3 wants
  columns that are configurations of the SAME model. A pooled cross-strategy
  matrix answers a different question (which strategy the search would pick), so
  it is reported separately as `pbo_selection` and is NOT the gate number.

* K=8 MATCHES THE REPO'S OWN `_DEFAULT_K` (`strategy_backtest_adapter.py:70`).
  At K=3 the omega statistic has a 3-atom support and is near-quantised against
  a 0.5 threshold. No source states a hard minimum N, so this is calibration,
  not a citable rule.

* `total_return_pct` IS ALREADY NET OF COMMISSION -- deducted from cash on every
  fill path in `backtest_trader.py`. Do NOT subtract costs again. It is net of
  commission only: no slippage, spread or market impact.

* TURNOVER comes from `trade_statistics.turnover_rate`
  (`analytics.py:488`), which is computed from the trader's own volume rather
  than from `result.all_trades` -- the latter is CAPPED AT 500
  (`backtest_engine.py:380`), so deriving turnover from it would rank strategies
  partly by how badly each was truncated.

* CONFIGS VARY ONLY WITHIN `make_engine`'s THREADED KWARGS. It silently ignores
  anything it does not thread, so a grid over an unthreaded knob would produce 8
  identical runs and a meaningless PBO.

* `skip_cache_clear=True` per run with ONE `clear_cache()` at the end. Without
  it every run refetches the whole cache (~20 min each becomes far worse).

MEASURED RUNTIME: 20.3 +/- 0.3 min per run -> 32 runs ~ 10.8h. The figure in
`.claude/rules/backend-backtest.md` ("<30s per experiment") is stale by ~40x;
see queued step 82.20.

Usage:
    python scripts/harness/run_82_3_candidate_backtests.py [--smoke] [--strategies a,b]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

OUT_DIR = Path("backend/backtest/experiments/results")
PROGRESS = Path("handoff/logs/82_3_progress.jsonl")

STRATEGIES = ["triple_barrier", "stretch_regime", "qarp", "reversion_sigma"]

# K=8 factorial over MODEL hyperparameters only. The horizon stays at each
# strategy's natural value so the columns are configurations of the same model
# rather than of different models.
GRID = list(product([3, 4], [10, 20], [0.05, 0.1]))  # max_depth, min_samples_leaf, learning_rate

BASE = {
    "market": "US",
    "start_date": "2018-01-01",
    "end_date": "2025-12-31",
    "holding_days": 90,
    "mr_holding_days": 15,
    "tp_pct": 10.0,
    "sl_pct": 12.923403579416114,
    "frac_diff_d": 0.4,
    "max_positions": 20,
    "top_n_candidates": 50,
    "n_estimators": 200,
}


def _log(rec: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[82.3] {rec}", flush=True)


def _daily_returns(result) -> np.ndarray | None:
    """Daily NAV return series -- the T-axis for CSCV."""
    nav = getattr(result, "nav_history", None) or []
    vals = []
    for p in nav:
        v = p.get("nav") if isinstance(p, dict) else None
        if v is not None:
            vals.append(float(v))
    if len(vals) < 40:
        return None
    a = np.asarray(vals, dtype=float)
    r = np.diff(a) / np.where(a[:-1] == 0, np.nan, a[:-1])
    return r[np.isfinite(r)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one short run per strategy to prove the harness, not evidence")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--start", default=BASE["start_date"])
    ap.add_argument("--end", default=BASE["end_date"])
    ap.add_argument("--label", default="full_sample",
                    help="pass label; keeps the two passes separable in the artifact")
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    grid = GRID[:1] if args.smoke else GRID
    BASE["start_date"], BASE["end_date"] = args.start, args.end

    from backend.config.settings import get_settings
    from backend.db.bigquery_client import BigQueryClient
    from backend.backtest import cache
    from backend.backtest.analytics import compute_pbo, generate_report
    from scripts.harness.run_harness import make_engine

    settings = get_settings()
    bq = BigQueryClient(settings)

    # Point-in-time state is part of the evidence: ALL runs must share ONE flag
    # state or the results are not comparable (82.15).
    pit = bool(getattr(settings, "macro_point_in_time_enabled", True))
    _log({"event": "start", "pass": args.label, "window": [args.start, args.end],
          "strategies": strategies, "K": len(grid),
          "total_runs": len(strategies) * len(grid),
          "macro_point_in_time_enabled": pit})

    per_strategy: dict[str, list[np.ndarray]] = {s: [] for s in strategies}
    summary: dict[str, dict] = {}
    t0 = time.time()

    try:
        for strat in strategies:
            rows = []
            for i, (max_depth, msl, lr) in enumerate(grid):
                params = dict(BASE, strategy=strat, max_depth=max_depth,
                              min_samples_leaf=msl, learning_rate=lr)
                if args.smoke:
                    params["start_date"], params["end_date"] = "2022-01-01", "2023-06-30"
                tag = f"{strat}[d{max_depth}_l{msl}_r{lr}]"
                started = time.time()
                try:
                    engine = make_engine(params, settings, bq,
                                         start_date=params["start_date"],
                                         end_date=params["end_date"])
                    result = engine.run_backtest(skip_cache_clear=True)
                    rep = generate_report(result)
                    an = rep.get("analytics", {})
                    ts = rep.get("trade_statistics", {}) or {}
                    rets = _daily_returns(result)
                    if rets is not None:
                        per_strategy[strat].append(rets)
                    row = {
                        "strategy": strat, "config": tag,
                        "max_depth": max_depth, "min_samples_leaf": msl, "learning_rate": lr,
                        "sharpe": an.get("sharpe"),
                        "dsr": an.get("deflated_sharpe"),
                        "net_of_cost_return_pct": an.get("total_return_pct"),
                        "turnover_rate": ts.get("turnover_rate"),
                        "total_commission": ts.get("total_commission"),
                        "n_trades": an.get("n_trades"),
                        "max_drawdown": an.get("max_drawdown"),
                        "n_daily_returns": int(len(rets)) if rets is not None else 0,
                        "elapsed_s": round(time.time() - started, 1),
                    }
                    rows.append(row)
                    _log({"event": "run_ok", **row})
                except Exception as exc:
                    _log({"event": "run_fail", "config": tag,
                          "error": f"{type(exc).__name__}: {exc}",
                          "elapsed_s": round(time.time() - started, 1)})
                    traceback.print_exc()
            summary[strat] = {"runs": rows}

        # ── PBO: one matrix per strategy, daily-return columns ──────
        for strat, series in per_strategy.items():
            if len(series) < 2:
                summary.setdefault(strat, {})["pbo"] = None
                summary[strat]["pbo_note"] = (
                    f"only {len(series)} usable column(s); compute_pbo needs N>=2 "
                    "and returns 0.0 SILENTLY below that, which would false-pass the gate")
                continue
            T = min(len(s) for s in series)
            matrix = np.column_stack([s[-T:] for s in series])
            if T < 32:
                summary[strat]["pbo"] = None
                summary[strat]["pbo_note"] = (
                    f"T={T} < S*2=32; compute_pbo returns 0.0 SILENTLY here and 0.0 "
                    "PASSES the <=0.5 gate, so no PBO is reported rather than a fabricated pass")
                continue
            summary[strat]["pbo"] = float(compute_pbo(matrix, S=16))
            summary[strat]["pbo_matrix_shape"] = [int(T), int(matrix.shape[1])]

            # TRIAL-DIVERSITY DIAGNOSTIC. CSCV ranks the columns against each
            # other across time subsets, so near-identical columns make the
            # ranking noise-driven and PBO correspondingly weak -- regardless of
            # how large N is. Measured live during pass A: two triple_barrier
            # configs differing only in learning_rate produced IDENTICAL trade
            # counts (1004) and a Sharpe delta of 5e-05. Reporting PBO without
            # this number would make it look more informative than it is.
            try:
                C = np.corrcoef(matrix, rowvar=False)
                iu = np.triu_indices_from(C, k=1)
                pair = C[iu]
                pair = pair[np.isfinite(pair)]
                summary[strat]["column_corr_mean"] = float(np.mean(pair)) if pair.size else None
                summary[strat]["column_corr_min"] = float(np.min(pair)) if pair.size else None
                summary[strat]["trial_diversity_note"] = (
                    "mean pairwise correlation of the K config columns; >0.99 means the "
                    "hyperparameter grid barely moved the strategy, so PBO is measuring a "
                    "near-degenerate search and must be read as weak evidence")
            except Exception as exc:
                summary[strat]["column_corr_mean"] = None
                summary[strat]["trial_diversity_error"] = f"{type(exc).__name__}: {exc}"

        out = {
            "step": "82.3",
            "pass_label": args.label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "macro_point_in_time_enabled": pit,
            "sample": {"start": BASE["start_date"], "end": BASE["end_date"], "market": "US"},
            "K_per_strategy": len(grid),
            "grid": "max_depth x min_samples_leaf x learning_rate (2x2x2)",
            "smoke": args.smoke,
            "elapsed_total_s": round(time.time() - t0, 1),
            "per_strategy": summary,
            "notes": [
                "total_return_pct is ALREADY net of commission (backtest_trader deducts on every fill); not re-subtracted",
                "turnover_rate comes from trade_statistics (trader volume), NOT from result.all_trades which is capped at 500",
                "PBO uses DAILY NAV returns; per-window returns give T~27 < S*2 and compute_pbo would return 0.0 silently, which PASSES the gate",
                "one PBO matrix per strategy (columns = configs of the same model), per Bailey et al. Algorithm 2.3",
                "reversion_sigma is purged at holding_days*1.5=135d against a 15d label horizon (backtest_engine.py:665 is strategy-blind, queued as 82.19) -- a WIN is clean, a LOSS is CONFOUNDED",
            ],
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = OUT_DIR / f"{stamp}_phase_82_3_{args.label}.json"
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        _log({"event": "done", "path": str(path), "elapsed_s": out["elapsed_total_s"]})
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "runs"}
                          for k, v in summary.items()}, indent=2))
        return 0
    finally:
        try:
            cache.clear_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
