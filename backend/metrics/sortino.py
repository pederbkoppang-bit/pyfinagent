"""phase-10.5 Sortino ratio with configurable Minimum Acceptable Return.

Canonical Sortino & Price (1994) formulation:
    DD  = sqrt((1/T) * sum(min(0, R_t - MAR) ** 2))
    SOR = (mean(R) - MAR) / DD * sqrt(periods_per_year)

Notes:
- DD sums squared DOWNSIDE deviations over ALL T periods (clip at 0 above MAR).
- The existing `backend/services/perf_metrics.compute_sortino` uses
  `std(ddof=1)` on negative-only values and is left untouched for back-compat
  with `paper_metrics_v2.py`. This module is the canonical LPM_2 form.
- MAR accepts a scalar, a 1-D array (per-period), or `None` (fetched via
  `mar_fetch_fn`, default is BQ `pyfinagent_data.historical_macro` with
  fallback to `backend/backtest/analytics.get_risk_free_rate` and hardcoded
  0.045).
- Zero-downside (all returns above MAR) returns `float('nan')` per Empyrical
  convention. Not `+inf` (JSON-unsafe) and not `0.0` (indistinguishable from
  "insufficient samples").

ASCII-only. Fail-open on MAR fetch.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ANNUAL_MAR = 0.045  # 3M T-Bill proxy; sprint_calendar.yaml line 37


def sortino(
    returns: Sequence[float] | np.ndarray,
    *,
    mar: float | Sequence[float] | np.ndarray | None = None,
    periods_per_year: int = 252,
    mar_fetch_fn: Callable[[], float] | None = None,
) -> float:
    """Canonical LPM_2 Sortino ratio.

    Parameters
    ----------
    returns : sequence of per-period returns (NOT annualized)
    mar : per-period MAR (scalar or same-length array), or None to fetch
    periods_per_year : annualization factor (252 daily, 12 monthly, 52 weekly)
    mar_fetch_fn : injectable; default fetches from historical_macro -> DTB3
        -> 0.045. Fetcher returns ANNUALIZED rate; this function divides by
        `periods_per_year` to get per-period MAR.

    Returns
    -------
    Annualized Sortino ratio as float. Returns `float('nan')` when:
      - `returns` has fewer than 2 samples
      - all returns are above MAR (zero downside deviation)
    """
    arr = np.asarray(list(returns), dtype=float)
    if arr.size < 2:
        return float("nan")

    if mar is None:
        fetcher = mar_fetch_fn or _default_mar_fetcher
        try:
            annual_mar = float(fetcher())
        except Exception as exc:
            logger.warning("sortino: mar_fetch_fn fail-open to %.4f: %r", _DEFAULT_ANNUAL_MAR, exc)
            annual_mar = _DEFAULT_ANNUAL_MAR
        mar_arr = np.full_like(arr, annual_mar / float(periods_per_year), dtype=float)
    elif np.isscalar(mar):
        mar_arr = np.full_like(arr, float(mar), dtype=float)
    else:
        mar_arr = np.asarray(list(mar), dtype=float)  # type: ignore[arg-type]
        if mar_arr.shape != arr.shape:
            raise ValueError(
                f"mar shape {mar_arr.shape} does not match returns shape {arr.shape}"
            )

    excess = arr - mar_arr
    downside_excess = np.clip(mar_arr - arr, a_min=0.0, a_max=None)
    dd2 = float(np.mean(downside_excess ** 2))
    if dd2 <= 0.0:
        # All returns above MAR -> downside deviation is 0; Sortino is
        # undefined (division by zero). Return NaN per Empyrical convention.
        return float("nan")

    dd = math.sqrt(dd2)
    return float(excess.mean() / dd) * math.sqrt(periods_per_year)


def _default_mar_fetcher() -> float:
    """Fetch an ANNUALIZED 3-month T-Bill rate.

    Priority:
      1. BQ `pyfinagent_data.historical_macro` (DGS3MO series) -- latest row
      2. `backend.backtest.analytics.get_risk_free_rate()` (local DTB3 cache)
      3. Hardcoded 0.045 fallback
    """
    # Tier 1: BQ historical_macro.
    try:
        from google.cloud import bigquery
        project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
        client = bigquery.Client(project=project)
        sql = f"""
            SELECT value
            FROM `{project}.pyfinagent_data.historical_macro`
            WHERE series_id IN ('DGS3MO', 'DTB3')
              AND value IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
        """
        rows = list(client.query(sql).result())
        if rows and rows[0].get("value") is not None:
            # FRED publishes DGS3MO / DTB3 as annualized percent (e.g., 4.5).
            value = float(rows[0]["value"])
            annualized = value / 100.0 if value > 1.0 else value
            return annualized
    except Exception as exc:
        logger.info("sortino: BQ historical_macro lookup fail-open: %r", exc)

    # Tier 2: local DTB3 CSV cache via analytics.
    try:
        from backend.backtest.analytics import get_risk_free_rate
        rate = float(get_risk_free_rate())
        if rate > 0.0:
            return rate
    except Exception as exc:
        logger.info("sortino: analytics.get_risk_free_rate fail-open: %r", exc)

    # Tier 3: hardcoded fallback.
    return _DEFAULT_ANNUAL_MAR


__all__ = ["sortino"]
