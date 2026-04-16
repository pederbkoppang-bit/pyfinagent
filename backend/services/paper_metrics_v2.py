"""
paper_metrics_v2 -- Evaluation-grade metrics for the paper-trading dashboard.

Orchestrator only: all math delegates to backend.services.perf_metrics per the
"single source of truth" convention. This module pulls NAV snapshots + trades
from BigQuery, calls perf_metrics, and returns a response dict for the
GET /api/paper-trading/metrics-v2 endpoint.

References (cited in perf_metrics.py):
  - Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier", Eq. 9
  - Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", Eq. 8
  - Efron & Tibshirani (1993), "An Introduction to the Bootstrap"
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from backend.services.perf_metrics import (
    compute_calmar,
    compute_dsr,
    compute_psr,
    compute_rolling_sharpe_bootstrap_ci,
    compute_sortino,
)

logger = logging.getLogger(__name__)

MIN_OBS_FOR_PSR = 30  # PSR/DSR unstable below this; Bailey & Lopez de Prado 2012 footnote.


def _nav_to_returns(snapshots: list[dict], nav_key: str = "total_nav") -> np.ndarray:
    """Convert NAV snapshots (oldest -> newest) to daily simple returns."""
    if not snapshots:
        return np.array([], dtype=float)
    # Accept either order; normalize by snapshot_date when present.
    ordered = list(snapshots)
    if ordered and "snapshot_date" in ordered[0]:
        ordered = sorted(ordered, key=lambda s: str(s.get("snapshot_date")))
    navs = np.array([float(s.get(nav_key) or 0.0) for s in ordered], dtype=float)
    navs = navs[navs > 0.0]
    if len(navs) < 2:
        return np.array([], dtype=float)
    return np.diff(navs) / navs[:-1]


def _trial_sharpes_from_trades(trades: list[dict]) -> list[float]:
    """
    Estimate the set of trial Sharpes for DSR's variance-of-trials term.
    Fallback heuristic: treat each unique ticker traded as a 'trial'; use the
    distribution of per-ticker returns as a proxy. This keeps DSR computable
    when we lack a formal N_strategies registry.
    """
    by_ticker: dict[str, list[float]] = {}
    for t in trades:
        tkr = t.get("ticker")
        if not tkr:
            continue
        r = t.get("realized_pnl_pct")
        if r is None:
            continue
        by_ticker.setdefault(tkr, []).append(float(r) / 100.0)
    out: list[float] = []
    for _, rets in by_ticker.items():
        if len(rets) < 2:
            continue
        arr = np.asarray(rets, dtype=float)
        std = float(arr.std(ddof=1))
        if std == 0.0:
            continue
        out.append(float(arr.mean() / std))
    return out


def compute_metrics_v2(bq: Any, snapshot_limit: int = 365, trade_limit: int = 1000) -> dict:
    """
    Build the metrics-v2 response. No BigQuery writes here -- pure read + compute.
    Callers handle persistence via persist_metrics_v2().
    """
    snapshots = bq.get_paper_snapshots(limit=snapshot_limit) or []
    trades = bq.get_paper_trades(limit=trade_limit) or []
    returns = _nav_to_returns(snapshots)
    n_obs = int(len(returns))
    now_iso = datetime.now(timezone.utc).isoformat()

    if n_obs < MIN_OBS_FOR_PSR:
        return {
            "psr": None,
            "dsr": None,
            "sortino": None,
            "calmar": None,
            "rolling_sharpe": None,
            "rolling_sharpe_ci_low": None,
            "rolling_sharpe_ci_high": None,
            "n_obs": n_obs,
            "n_strategies_tested": 0,
            "computed_at": now_iso,
            "note": "insufficient_data",
            "min_obs_required": MIN_OBS_FOR_PSR,
        }

    trial_sharpes = _trial_sharpes_from_trades(trades)
    n_trials = max(len(trial_sharpes), 2)

    psr = compute_psr(returns, sr_star=0.0)
    dsr = compute_dsr(returns, trial_sharpes, n_trials=n_trials) if trial_sharpes else 0.0
    sortino = compute_sortino(returns)
    calmar = compute_calmar(returns)
    sharpe, ci_low, ci_high = compute_rolling_sharpe_bootstrap_ci(returns)

    return {
        "psr": round(psr, 4),
        "dsr": round(dsr, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "rolling_sharpe": round(sharpe, 4),
        "rolling_sharpe_ci_low": round(ci_low, 4),
        "rolling_sharpe_ci_high": round(ci_high, 4),
        "n_obs": n_obs,
        "n_strategies_tested": len(trial_sharpes),
        "computed_at": now_iso,
        "note": None,
    }


def persist_metrics_v2(bq: Any, metrics: dict) -> None:
    """
    Best-effort persistence to pyfinagent_pms.paper_metrics_v2. Swallows errors;
    the endpoint must not 500 if the write leg fails. Table is created lazily
    if missing (idempotent, matches other paper_* migration style).
    """
    if metrics.get("note") == "insufficient_data":
        return
    try:
        table = bq._pt_table("paper_metrics_v2")
        from google.cloud import bigquery
        schema = [
            bigquery.SchemaField("snapshot_date", "DATE"),
            bigquery.SchemaField("nav", "FLOAT64"),
            bigquery.SchemaField("rolling_sharpe", "FLOAT64"),
            bigquery.SchemaField("psr", "FLOAT64"),
            bigquery.SchemaField("dsr", "FLOAT64"),
            bigquery.SchemaField("sortino", "FLOAT64"),
            bigquery.SchemaField("calmar", "FLOAT64"),
            bigquery.SchemaField("rolling_sharpe_ci_low", "FLOAT64"),
            bigquery.SchemaField("rolling_sharpe_ci_high", "FLOAT64"),
            bigquery.SchemaField("n_strategies_tested", "INT64"),
            bigquery.SchemaField("trades_to_date", "INT64"),
            bigquery.SchemaField("computed_at", "TIMESTAMP"),
        ]
        tbl_ref = bigquery.Table(table, schema=schema)
        try:
            bq.client.create_table(tbl_ref, exists_ok=True)
        except Exception as e:
            logger.debug(f"metrics-v2 create_table skipped: {e}")
        row = {
            "snapshot_date": datetime.now(timezone.utc).date().isoformat(),
            "nav": None,
            "rolling_sharpe": metrics.get("rolling_sharpe"),
            "psr": metrics.get("psr"),
            "dsr": metrics.get("dsr"),
            "sortino": metrics.get("sortino"),
            "calmar": metrics.get("calmar"),
            "rolling_sharpe_ci_low": metrics.get("rolling_sharpe_ci_low"),
            "rolling_sharpe_ci_high": metrics.get("rolling_sharpe_ci_high"),
            "n_strategies_tested": metrics.get("n_strategies_tested"),
            "trades_to_date": metrics.get("n_obs"),
            "computed_at": metrics.get("computed_at"),
        }
        errors = bq.client.insert_rows_json(table, [row])
        if errors:
            logger.warning(f"metrics-v2 insert errors: {errors}")
    except Exception as e:
        logger.warning(f"metrics-v2 persist failed: {e}")
