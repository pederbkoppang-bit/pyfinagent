#!/usr/bin/env python3
"""
scripts/ablation/run_ablation.py

Feature ablation runner. For a target feature (or cluster), zeroes it
out across both the training and prediction feature matrices, runs
one walk-forward backtest, compares the resulting Sharpe + DSR to the
locked baseline from optimizer_best.json, and logs a verdict row to
backend/backtest/experiments/feature_ablation_results.tsv.

Verdict gate (research-backed, see
/Users/ford/.claude/plans/parsed-tinkering-stallman.md Research Gate):
  * keep        -- delta_sharpe <  -0.05 AND baseline_dsr >= 0.95
                   (feature is load-bearing: removing it hurts enough
                   to pay its own keep; floor inherited from
                   evaluator-criteria.md Simplicity rule)
  * discard     -- delta_sharpe >= 0
                   (removing the feature didn't hurt or actually
                   helped; feature is noise)
  * inconclusive -- -0.05 <= delta_sharpe < 0
                   (weak negative signal; not strong enough to commit
                   the feature to the permanent vector)

Usage:

    source .venv/bin/activate

    # Ablate one specific feature
    python scripts/ablation/run_ablation.py --feature momentum_1m

    # Pick the next feature with no TSV row (for launchd)
    python scripts/ablation/run_ablation.py --next-untested

    # Batch-run every untested feature (one-shot seeding, ~30-60min)
    python scripts/ablation/run_ablation.py --batch

    # Cluster mode (Lopez de Prado's Clustered Feature Importance):
    # hierarchical clustering on Spearman-correlation distance, flat cut
    # at 0.15 (= 0.85 correlation), then ablate whole clusters at once.
    python scripts/ablation/run_ablation.py --cluster-mode

Each invocation is one backtest. The baseline is read from
backend/backtest/experiments/optimizer_best.json; if that file does
not exist or does not contain a sharpe/dsr pair, the runner computes
a fresh baseline on the first run of the session (using an empty
ablation mask) and caches it in-memory for the rest of the session.

Emits a single line to stdout:
  <feature> delta=<float> dsr=<float> verdict=<keep|discard|inconclusive>
and appends one row to the TSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Imports deferred to after sys.path injection
from backend.backtest.backtest_engine import BacktestEngine, _NUMERIC_FEATURES  # noqa: E402
from backend.backtest.analytics import generate_report  # noqa: E402
from backend.config.settings import get_settings  # noqa: E402
from backend.db.bigquery_client import BigQueryClient  # noqa: E402

EXPERIMENTS_DIR = REPO / "backend" / "backtest" / "experiments"
TSV_PATH = EXPERIMENTS_DIR / "feature_ablation_results.tsv"
BEST_PARAMS_PATH = EXPERIMENTS_DIR / "optimizer_best.json"

TSV_HEADER = [
    "run_id",
    "timestamp",
    "feature_name",
    "cluster_id",
    "baseline_sharpe",
    "ablated_sharpe",
    "delta_sharpe",
    "baseline_dsr",
    "ablated_dsr",
    "delta_dsr",
    "verdict",
    "cost_tier",
]

# Research-backed thresholds (see plan file Research Gate section).
SHARPE_DELTA_FLOOR = -0.05  # From .claude/rules/evaluator-criteria.md Simplicity rule
DSR_THRESHOLD = 0.95        # Bailey & Lopez de Prado 2014
CLUSTER_CORR_THRESHOLD = 0.85  # Lopez de Prado Clustered Feature Importance


@dataclass
class BaselineCache:
    sharpe: float
    dsr: float


def _ensure_tsv_header() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not TSV_PATH.exists():
        with TSV_PATH.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(TSV_HEADER)


def _already_tested_features() -> set[str]:
    if not TSV_PATH.exists():
        return set()
    tested: set[str] = set()
    with TSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("feature_name", "").strip()
            if name:
                tested.add(name)
    return tested


def _load_best_params() -> dict:
    if BEST_PARAMS_PATH.exists():
        try:
            return json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _make_engine(settings, ablation_mask: Optional[set[str]] = None) -> BacktestEngine:
    """Construct a BacktestEngine with the locked best params and an
    optional ablation mask. Mirrors the wiring quant_optimizer does
    internally, minus the strategy-param mutation loop.
    """
    best = _load_best_params()
    params = best.get("params", best) if best else {}

    bq_client = BigQueryClient(settings)
    engine = BacktestEngine(
        bq_client=bq_client,
        project=settings.gcp_project_id,
        dataset=settings.bq_dataset_reports,
        strategy=params.get("strategy", "triple_barrier"),
        start_date=params.get("start_date", "2023-01-01"),
        end_date=params.get("end_date", "2025-12-31"),
        holding_days=int(params.get("holding_days", 90)),
        tp_pct=float(params.get("tp_pct", 10.0)),
        sl_pct=float(params.get("sl_pct", 10.0)),
        target_vol=float(params.get("target_vol", 0.15)),
        max_positions=int(params.get("max_positions", 20)),
        top_n_candidates=int(params.get("top_n_candidates", 50)),
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 4)),
        min_samples_leaf=int(params.get("min_samples_leaf", 20)),
        learning_rate=float(params.get("learning_rate", 0.1)),
    )
    engine._ablation_mask = set(ablation_mask or [])
    return engine


def _run_backtest(settings, ablation_mask: Optional[set[str]]) -> tuple[float, float]:
    """Run one walk-forward backtest and return (sharpe, dsr)."""
    engine = _make_engine(settings, ablation_mask=ablation_mask)
    result = engine.run_backtest(skip_cache_clear=True)
    report = generate_report(result, num_trials=1)
    sharpe = float(report["analytics"]["sharpe"])
    dsr = float(report["analytics"].get("deflated_sharpe", 0.0))
    return sharpe, dsr


def _get_baseline(settings, cache: Optional[BaselineCache]) -> BaselineCache:
    if cache is not None:
        return cache
    best = _load_best_params()
    if best and "sharpe" in best and "dsr" in best:
        return BaselineCache(sharpe=float(best["sharpe"]), dsr=float(best["dsr"]))
    # Fresh baseline
    sharpe, dsr = _run_backtest(settings, ablation_mask=None)
    return BaselineCache(sharpe=sharpe, dsr=dsr)


def _verdict(delta_sharpe: float, baseline_dsr: float) -> str:
    if delta_sharpe < SHARPE_DELTA_FLOOR and baseline_dsr >= DSR_THRESHOLD:
        return "keep"
    if delta_sharpe >= 0:
        return "discard"
    return "inconclusive"


def _log_row(
    feature_name: str,
    cluster_id: str,
    baseline: BaselineCache,
    ablated_sharpe: float,
    ablated_dsr: float,
    run_id: str,
    cost_tier: str,
) -> str:
    _ensure_tsv_header()
    delta_sharpe = ablated_sharpe - baseline.sharpe
    delta_dsr = ablated_dsr - baseline.dsr
    verdict = _verdict(delta_sharpe, baseline.dsr)
    with TSV_PATH.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerow([
            run_id,
            time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            feature_name,
            cluster_id,
            f"{baseline.sharpe:.4f}",
            f"{ablated_sharpe:.4f}",
            f"{delta_sharpe:+.4f}",
            f"{baseline.dsr:.4f}",
            f"{ablated_dsr:.4f}",
            f"{delta_dsr:+.4f}",
            verdict,
            cost_tier,
        ])
    print(
        f"{feature_name} delta={delta_sharpe:+.4f} dsr={ablated_dsr:.4f} "
        f"verdict={verdict}",
        flush=True,
    )
    return verdict


def _pick_next_untested() -> Optional[str]:
    tested = _already_tested_features()
    for feat in _NUMERIC_FEATURES:
        if feat not in tested:
            return feat
    return None


def _cluster_features(settings) -> list[set[str]]:
    """Hierarchical clustering on pairwise Spearman correlation distance
    of the feature matrix. Returns a list of clusters (each a set of
    feature names). Requires scipy.
    """
    import numpy as np
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # Build a feature matrix from one walk-forward training batch.
    # Reuses the engine's _build_training_data for determinism.
    engine = _make_engine(settings, ablation_mask=None)
    tickers = engine._screen_universe(engine.start_date)[: engine.top_n_candidates]
    X, _, _ = engine._build_training_data(tickers, engine.start_date, engine.end_date)
    if X.empty or len(X.columns) < 2:
        return [{c} for c in _NUMERIC_FEATURES if c in (X.columns if not X.empty else [])]

    corr = X.corr(method="spearman").abs().fillna(0.0)
    # Convert correlation to distance (1 - |rho|); diagonal to 0.
    dist = 1.0 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")
    # Flat cut at distance = 1 - CLUSTER_CORR_THRESHOLD (= 0.15)
    labels = fcluster(link, t=(1.0 - CLUSTER_CORR_THRESHOLD), criterion="distance")
    groups: dict[int, set[str]] = {}
    for col, label in zip(corr.columns, labels):
        groups.setdefault(int(label), set()).add(col)
    return list(groups.values())


def run_single(settings, feature: str, baseline: BaselineCache, run_id: str) -> str:
    s, d = _run_backtest(settings, ablation_mask={feature})
    return _log_row(feature, "", baseline, s, d, run_id, settings.cost_tier)


def run_cluster(
    settings, cluster: set[str], baseline: BaselineCache, run_id: str, cluster_id: str
) -> str:
    s, d = _run_backtest(settings, ablation_mask=cluster)
    feature_name = "|".join(sorted(cluster))
    return _log_row(feature_name, cluster_id, baseline, s, d, run_id, settings.cost_tier)


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature ablation runner")
    parser.add_argument("--feature", help="Ablate a single feature by name")
    parser.add_argument(
        "--next-untested",
        action="store_true",
        help="Ablate the next feature with no TSV row (for launchd cron)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Ablate every untested feature sequentially (~30-60min wall clock)",
    )
    parser.add_argument(
        "--cluster-mode",
        action="store_true",
        help="Cluster correlated features first, then ablate whole clusters",
    )
    args = parser.parse_args()

    settings = get_settings()
    run_id = str(uuid.uuid4())[:8]
    baseline: Optional[BaselineCache] = None

    if args.cluster_mode:
        clusters = _cluster_features(settings)
        print(f"[ablation] discovered {len(clusters)} clusters", flush=True)
        baseline = _get_baseline(settings, baseline)
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                # Singleton -- treat as per-feature ablation
                feat = next(iter(cluster))
                run_single(settings, feat, baseline, run_id)
            else:
                run_cluster(settings, cluster, baseline, run_id, f"c{i:02d}")
        return 0

    if args.feature:
        baseline = _get_baseline(settings, baseline)
        run_single(settings, args.feature, baseline, run_id)
        return 0

    if args.next_untested:
        feat = _pick_next_untested()
        if feat is None:
            print("all-features-tested", flush=True)
            return 0
        baseline = _get_baseline(settings, baseline)
        run_single(settings, feat, baseline, run_id)
        return 0

    if args.batch:
        tested = _already_tested_features()
        untested = [f for f in _NUMERIC_FEATURES if f not in tested]
        if not untested:
            print("all-features-tested", flush=True)
            return 0
        print(f"[ablation] batch running {len(untested)} features", flush=True)
        baseline = _get_baseline(settings, baseline)
        for feat in untested:
            run_single(settings, feat, baseline, run_id)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
