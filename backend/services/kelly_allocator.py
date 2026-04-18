"""phase-4.8 step 4.8.3 Fractional-Kelly multi-strategy allocator.

Kelly 1956 / Thorp 1971 solution for N strategies:

    f* = Sigma^{-1} * mu          # full Kelly (growth-optimal)
    f_frac = k * f*                # fractional Kelly, k in [0.25, 0.5]

Fractional Kelly trades some asymptotic growth for dramatically
reduced drawdowns (MacLean/Thorp/Ziemba 2010). We clip each
allocation to `cap` (default 30%) and renormalize so the
portfolio stays fully invested.

Sign convention: positive alloc = long. Negative alloc (Kelly can
return them when mu has negative entries) is clipped to 0 -- this
allocator is long-only by design.

Reject non-PSD or singular Sigma with ValueError (fail-loud over
silent bad allocations).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_FRACTION = 0.25   # MacLean/Thorp/Ziemba 2010 recommendation
DEFAULT_CAP = 0.30        # single-strategy concentration ceiling


def fractional_kelly(
    mu: Sequence[float],
    Sigma: Sequence[Sequence[float]],
    k: float = DEFAULT_FRACTION,
    cap: float = DEFAULT_CAP,
) -> list[float]:
    """Return per-strategy allocation fractions (long-only, capped).

    Parameters
    ----------
    mu : length-N excess-return vector (annualized or any consistent
        frequency; units must match Sigma).
    Sigma : N x N covariance matrix (must be symmetric, PSD,
        non-singular).
    k : fractional Kelly scale (0 < k <= 1); default 0.25.
    cap : per-strategy max allocation in [0, 1]; default 0.30.

    Returns
    -------
    allocations : length-N list of floats in [0, cap]; sum <= 1.
    """
    mu_v = np.asarray(mu, dtype=float)
    S = np.asarray(Sigma, dtype=float)
    n = mu_v.size
    if S.shape != (n, n):
        raise ValueError(f"Sigma shape {S.shape} does not match mu length {n}")
    if not np.allclose(S, S.T, atol=1e-9):
        raise ValueError("Sigma must be symmetric")
    eigs = np.linalg.eigvalsh(S)
    if eigs.min() <= 1e-12:
        raise ValueError(
            f"Sigma must be positive definite (min eig {eigs.min()}); "
            "provide regularized / shrunk estimate"
        )
    if not (0 < k <= 1):
        raise ValueError(f"k must be in (0, 1], got {k}")
    if not (0 < cap <= 1):
        raise ValueError(f"cap must be in (0, 1], got {cap}")

    # Full Kelly: Sigma^{-1} * mu
    try:
        full_kelly = np.linalg.solve(S, mu_v)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Sigma inversion failed: {e}") from e

    # Fractional Kelly
    f = k * full_kelly
    # Long-only: clip negative
    f = np.clip(f, 0.0, None)
    # Per-strategy cap
    f = np.minimum(f, cap)
    # Renormalize if sum > 1 (stay fully invested at 1x notional)
    s = float(f.sum())
    if s > 1.0:
        f = f * (1.0 / s)
        # After renormalization every entry <= 1/N * (1/s) * cap,
        # which is guaranteed <= cap when cap*N >= 1. For cap=0.30
        # and N<=5 this is tight (0.30 * 5 = 1.5 >= 1) -- safe.
        # For N where cap*N < 1 (e.g., cap=0.1, N=5, total <0.5) no
        # renorm is needed. Defensive re-cap:
        f = np.minimum(f, cap)
    return [float(x) for x in f]


__all__ = ["fractional_kelly", "DEFAULT_FRACTION", "DEFAULT_CAP"]
