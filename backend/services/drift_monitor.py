"""phase-4.8 step 4.8.4 Model-drift monitor: PSI + rolling IC.

Two post-deployment drift signals:

1. **PSI** (Population Stability Index): feature-distribution shift
   between a baseline sample and a current sample. Formula:
       PSI = Sum_i (a_i - e_i) * ln(a_i / e_i)
   where a_i / e_i are the ACTUAL / EXPECTED fraction in bin i.
   Thresholds (Siddiqi 2006, industry standard):
     <0.10 = no shift, 0.10-0.25 = minor, >0.25 = significant.

2. **20-day rolling IC** (Information Coefficient): Spearman rank
   correlation between model predictions and realized forward
   returns, over a rolling 20-day window. IC <= 0 sustained = loss
   of predictive edge (Qian/Hua/Sorensen 2007).

Auto-freeze policy:
  - PSI > 0.25 -> freeze
  - IC_20d sustained <= 0 for IC_FREEZE_SUSTAINED_DAYS -> freeze
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


PSI_FREEZE_THRESHOLD = 0.25
IC_FREEZE_MAX = 0.0
IC_FREEZE_SUSTAINED_DAYS = 5
IC_WINDOW = 20


def compute_psi(
    baseline: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index.

    Bins are defined from the baseline quantiles (so the bin edges
    reflect the expected distribution). Zero-bin counts are floored
    at `eps` fraction so the log term is finite.
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(current, dtype=float)
    if b.size == 0 or c.size == 0:
        return 0.0
    edges = np.quantile(b, np.linspace(0, 1, bins + 1))
    # widen the outer edges so current values outside baseline range
    # still bin
    edges[0] = min(edges[0], c.min()) - 1e-9
    edges[-1] = max(edges[-1], c.max()) + 1e-9
    e_hist, _ = np.histogram(b, bins=edges)
    a_hist, _ = np.histogram(c, bins=edges)
    e_frac = np.maximum(e_hist / max(b.size, 1), eps)
    a_frac = np.maximum(a_hist / max(c.size, 1), eps)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # numpy has no scipy; implement rank + Pearson-on-ranks.
    if x.size != y.size or x.size < 3:
        return 0.0
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


def compute_ic(
    predictions: list[float] | np.ndarray,
    forward_returns: list[float] | np.ndarray,
) -> float:
    p = np.asarray(predictions, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    return _spearman(p, r)


def rolling_ic(
    predictions: list[float] | np.ndarray,
    forward_returns: list[float] | np.ndarray,
    window: int = IC_WINDOW,
) -> list[float]:
    p = np.asarray(predictions, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if p.size != r.size or p.size < window:
        return []
    out: list[float] = []
    for i in range(window, p.size + 1):
        out.append(_spearman(p[i - window:i], r[i - window:i]))
    return out


def _seed_model(
    seed: str,
    psi_anomaly: bool = False,
    ic_anomaly: bool = False,
    n: int = 250,
) -> dict[str, Any]:
    h = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    baseline = rng.normal(0.0, 1.0, 500).tolist()
    current = rng.normal(0.0, 1.0, 500).tolist()
    if psi_anomaly:
        # Shift + stretch -> PSI > 0.25 easily
        current = (rng.normal(2.5, 2.0, 500)).tolist()
    # Predictions/returns for IC
    predictions = rng.normal(0.0, 1.0, n)
    # Baseline: predictions correlated with returns (real edge)
    returns = 0.35 * predictions + rng.normal(0.0, 1.0, n)
    if ic_anomaly:
        # Break the edge: returns strongly anti-correlated with
        # predictions so Spearman IC stays negative across every
        # rolling window. The -1.0 coefficient + small residual
        # guarantees rank correlation near -1; a weaker weight gets
        # drowned out by noise and fails the audit's sustained-
        # negative check.
        returns = -1.0 * predictions + 0.2 * rng.normal(0.0, 1.0, n)
    return {
        "baseline": baseline,
        "current": current,
        "predictions": predictions.tolist(),
        "returns": returns.tolist(),
    }


def _evaluate_model(
    name: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    psi = compute_psi(data["baseline"], data["current"])
    ic_series = rolling_ic(data["predictions"], data["returns"])
    ic_20d = ic_series[-1] if ic_series else 0.0
    ic_trend_sustained_neg = (
        len(ic_series) >= IC_FREEZE_SUSTAINED_DAYS
        and all(v <= IC_FREEZE_MAX
                for v in ic_series[-IC_FREEZE_SUSTAINED_DAYS:])
    )

    reasons: list[str] = []
    if psi > PSI_FREEZE_THRESHOLD:
        reasons.append(f"psi_exceeded ({psi:.4f} > {PSI_FREEZE_THRESHOLD})")
    if ic_trend_sustained_neg:
        tail = ic_series[-IC_FREEZE_SUSTAINED_DAYS:]
        reasons.append(
            "ic_sustained_negative "
            f"(last {IC_FREEZE_SUSTAINED_DAYS}d: "
            f"{[round(v, 3) for v in tail]})"
        )

    return {
        "name": name,
        "psi": round(psi, 6),
        "ic_20d": round(ic_20d, 6),
        "ic_20d_trend_last": [round(v, 4) for v in ic_series[-IC_FREEZE_SUSTAINED_DAYS:]],
        "frozen": bool(reasons),
        "freeze_reasons": reasons,
        "psi_threshold": PSI_FREEZE_THRESHOLD,
        "ic_freeze_sustained_days": IC_FREEZE_SUSTAINED_DAYS,
    }


def run(
    models: list[dict[str, Any]] | None = None,
    *,
    seed_prefix: str = "drift",
) -> dict[str, Any]:
    """Return the drift-monitor snapshot for one or more models.

    When `models` is None, seeds three synthetic models ("momentum",
    "mean_reversion", "triple_barrier") with benign baseline/current
    + predictions/returns so the signature contract is testable
    without live data. Records `data_source: "seeded"` on the output.
    """
    if models is None:
        seeded = [
            {"name": n, "data": _seed_model(f"{seed_prefix}-{n}")}
            for n in ("momentum", "mean_reversion", "triple_barrier")
        ]
        data_source = "seeded"
    else:
        seeded = models
        data_source = "live"

    return {
        "data_source": data_source,
        "config": {
            "psi_threshold": PSI_FREEZE_THRESHOLD,
            "ic_freeze_max": IC_FREEZE_MAX,
            "ic_freeze_sustained_days": IC_FREEZE_SUSTAINED_DAYS,
            "ic_window": IC_WINDOW,
        },
        "models": [_evaluate_model(m["name"], m["data"]) for m in seeded],
    }


__all__ = [
    "IC_FREEZE_MAX",
    "IC_FREEZE_SUSTAINED_DAYS",
    "IC_WINDOW",
    "PSI_FREEZE_THRESHOLD",
    "compute_psi",
    "compute_ic",
    "rolling_ic",
    "run",
]
