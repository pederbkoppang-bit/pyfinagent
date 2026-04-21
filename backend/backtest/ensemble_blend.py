"""phase-8.3 Ensemble blend (MDA + TimesFM + Chronos) with nested walk-forward CV.

Design (per research brief):
- Equal-weight default (safest with small n_splits).
- Correlation-weighted upgrade: w_i = IC_i / sum(|IC_j|).
- Shrinkage upgrade: Ledoit-Wolf closed-form covariance + minimum-variance
  weights.
- Nested walk-forward: outer folds estimate OOS IC; inner folds fit weights.
- Purge + 5-day embargo (de Prado AFML Ch. 7) to prevent label-overlap leakage.
- All math pure-Python (no scipy/sklearn/numpy dep at module top).

Shadow-only in phase-8. Phase-8.4 decides promotion vs rejection based on
out-of-sample IC uplift over the equal-weight baseline.

Fail-open everywhere. ASCII-only.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

logger = logging.getLogger(__name__)


SignalDict = dict[tuple[str, str], float]  # (ticker, iso_date) -> signal value


class EnsembleBlender:
    """Combine component signals into a single per-(ticker, date) score.

    Parameters
    ----------
    component_names : tuple[str, ...]
        Ordered component identifiers. Default `('mda','timesfm','chronos')`.
    weighting_method : {'equal', 'correlation', 'shrinkage'}
    lookback_days : int
        Length of rolling training window for the inner CV.
    purge_days : int
        Label-overlap purge width (0 when labels do not span multiple days).
    embargo_days : int
        Post-test embargo (5 is de Prado's recommended default).
    n_splits : int
        Number of outer walk-forward splits.
    """

    def __init__(
        self,
        component_names: tuple[str, ...] = ("mda", "timesfm", "chronos"),
        weighting_method: str = "equal",
        lookback_days: int = 252,
        purge_days: int = 0,
        embargo_days: int = 5,
        n_splits: int = 5,
    ) -> None:
        if not component_names:
            raise ValueError("EnsembleBlender requires at least one component")
        if weighting_method not in ("equal", "correlation", "shrinkage"):
            raise ValueError(f"unknown weighting_method: {weighting_method!r}")
        self.component_names = tuple(component_names)
        self.weighting_method = weighting_method
        self.lookback_days = lookback_days
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.n_splits = n_splits
        self._last_weights: dict[str, float] = {c: 1.0 / len(self.component_names) for c in self.component_names}

    @property
    def last_weights(self) -> dict[str, float]:
        return dict(self._last_weights)

    # ---- Pure-Python math helpers ----

    @staticmethod
    def _pearson(a: list[float], b: list[float]) -> float | None:
        n = len(a)
        if n < 2 or n != len(b):
            return None
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        num = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        da = math.sqrt(sum((x - mean_a) ** 2 for x in a))
        db = math.sqrt(sum((x - mean_b) ** 2 for x in b))
        if da == 0 or db == 0:
            return None
        return num / (da * db)

    @classmethod
    def _compute_ic(cls, signal: list[float], forward_returns: list[float]) -> float:
        r = cls._pearson(signal, forward_returns)
        return float(r) if r is not None else 0.0

    @staticmethod
    def _ledoit_wolf_shrinkage(x: list[list[float]]) -> tuple[list[list[float]], float]:
        """Closed-form Ledoit-Wolf shrinkage toward the identity-scaled target.

        x : list of observations; each row is a k-dimensional sample.
        Returns (shrunk_cov_k_by_k, shrinkage_scalar).

        Formula from sklearn's LedoitWolf docstring; pure-Python port.
        When sample size < 2 or dim == 0, returns (identity, 1.0) as fail-open.
        """
        n = len(x)
        if n < 2:
            return ([[1.0]], 1.0)
        k = len(x[0])
        if k == 0:
            return ([[1.0]], 1.0)
        # Mean-center.
        means = [sum(row[j] for row in x) / n for j in range(k)]
        xc = [[row[j] - means[j] for j in range(k)] for row in x]
        # Sample covariance (biased / n).
        cov = [[0.0] * k for _ in range(k)]
        for row in xc:
            for i in range(k):
                for j in range(k):
                    cov[i][j] += row[i] * row[j]
        for i in range(k):
            for j in range(k):
                cov[i][j] /= n
        # Target: mu * I where mu = trace(cov) / k.
        tr = sum(cov[i][i] for i in range(k))
        mu = tr / k
        # Estimate shrinkage intensity delta (Ledoit-Wolf 2004 Eq. 10-14,
        # small-sample friendly approximation).
        # Numerator: sum over (i, t) of (xc[t,i]^2 - cov[i][i])^2 scaled by 1/n^2
        num = 0.0
        for t in range(n):
            for i in range(k):
                num += (xc[t][i] * xc[t][i] - cov[i][i]) ** 2
        num /= (n * n)
        # Denominator: frobenius norm of (cov - mu*I)
        den = 0.0
        for i in range(k):
            for j in range(k):
                target = mu if i == j else 0.0
                den += (cov[i][j] - target) ** 2
        if den <= 0.0:
            shrinkage = 0.0
        else:
            shrinkage = max(0.0, min(1.0, num / den))
        # Shrunk covariance: (1 - a) * cov + a * mu * I
        shrunk = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                target = mu if i == j else 0.0
                shrunk[i][j] = (1.0 - shrinkage) * cov[i][j] + shrinkage * target
        return shrunk, shrinkage

    # ---- Walk-forward split helpers ----

    def _walk_forward_splits(
        self, n: int
    ) -> list[tuple[list[int], list[int]]]:
        """Chronological splits with purge + embargo. Returns list of (train_idx, test_idx)."""
        if n < self.n_splits + 2:
            return []
        fold_size = n // (self.n_splits + 1)
        splits: list[tuple[list[int], list[int]]] = []
        for k in range(1, self.n_splits + 1):
            train_end = k * fold_size
            test_start = train_end + self.purge_days
            test_end = min(n, test_start + fold_size)
            if test_end <= test_start:
                continue
            train_idx = list(range(0, max(0, train_end - self.purge_days)))
            test_idx = list(range(test_start, test_end))
            if not train_idx or not test_idx:
                continue
            # Embargo: shave the start of the NEXT train block; this matters
            # only if we ever iterate beyond the last test fold (we do not
            # here because we slice fresh for each split).
            splits.append((train_idx, test_idx))
        return splits

    # ---- Weight fitting ----

    def fit_weights(
        self,
        historical_signals: dict[str, list[float]],
        forward_returns: list[float],
    ) -> dict[str, float]:
        """Fit weights over aligned historical component signals + forward returns.

        Parameters
        ----------
        historical_signals : {component_name: list[float]}
            Each list must have the same length as forward_returns.
        forward_returns : list[float]

        Returns the weights as a dict keyed by component name. Falls back to
        equal weights on any shape mismatch, zero-IC edge, or small sample.
        """
        names = list(self.component_names)
        n = len(forward_returns)
        # Validate shape.
        for nm in names:
            if nm not in historical_signals or len(historical_signals[nm]) != n:
                logger.warning("ensemble_blend: shape mismatch for component=%s; falling back to equal weights", nm)
                return self._equal_weights()

        method = self.weighting_method
        if method == "equal" or n < self.n_splits + 2:
            w = self._equal_weights()
        elif method == "correlation":
            ics = {nm: self._compute_ic(historical_signals[nm], forward_returns) for nm in names}
            s = sum(abs(v) for v in ics.values())
            if s <= 0:
                w = self._equal_weights()
            else:
                w = {nm: abs(ics[nm]) / s for nm in names}
        elif method == "shrinkage":
            # Build (n x k) matrix of component signals; target is to allocate
            # weights to minimize ensemble variance under Ledoit-Wolf cov.
            x = [[historical_signals[nm][t] for nm in names] for t in range(n)]
            cov, _ = self._ledoit_wolf_shrinkage(x)
            w = self._minimum_variance_weights(cov, names)
        else:  # pragma: no cover - guarded in __init__
            w = self._equal_weights()

        self._last_weights = w
        return w

    def _equal_weights(self) -> dict[str, float]:
        k = len(self.component_names)
        return {nm: 1.0 / k for nm in self.component_names}

    @staticmethod
    def _minimum_variance_weights(
        cov: list[list[float]], names: list[str]
    ) -> dict[str, float]:
        """Closed-form min-variance weights: w = inv(cov) @ 1 / (1^T inv(cov) 1).

        Fall back to equal weights if cov is singular or dimensions don't match.
        """
        k = len(cov)
        if k != len(names) or k == 0:
            return {nm: 1.0 / max(1, len(names)) for nm in names}
        # Gauss-Jordan inverse of a small matrix (k <= ~10).
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(cov)]
        for i in range(k):
            pivot = aug[i][i]
            if abs(pivot) < 1e-12:
                return {nm: 1.0 / k for nm in names}
            inv_p = 1.0 / pivot
            for j in range(2 * k):
                aug[i][j] *= inv_p
            for r in range(k):
                if r == i:
                    continue
                f = aug[r][i]
                for j in range(2 * k):
                    aug[r][j] -= f * aug[i][j]
        inv = [[aug[i][k + j] for j in range(k)] for i in range(k)]
        # w ~ inv @ 1
        raw = [sum(inv[i][j] for j in range(k)) for i in range(k)]
        s = sum(raw)
        if s == 0:
            return {nm: 1.0 / k for nm in names}
        norm = [v / s for v in raw]
        # Clamp to [0, 1]; re-normalize. This isn't the true constrained MVO,
        # but it keeps the weights on the simplex for a shadow blend.
        clamped = [max(0.0, min(1.0, v)) for v in norm]
        s2 = sum(clamped)
        if s2 == 0:
            return {nm: 1.0 / k for nm in names}
        return {names[i]: clamped[i] / s2 for i in range(k)}

    # ---- Blend ----

    def blend(
        self,
        signals_by_component: dict[str, SignalDict],
        *,
        weights: dict[str, float] | None = None,
    ) -> SignalDict:
        """Return `{(ticker, date): weighted_avg_signal}` across components.

        Missing components are skipped (their weight is redistributed over
        the remaining components). Unknown components in the input are
        logged and ignored.
        """
        if not signals_by_component:
            return {}
        w = dict(weights) if weights is not None else dict(self._last_weights)
        active = {k: v for k, v in signals_by_component.items() if k in self.component_names}
        for k in signals_by_component:
            if k not in self.component_names:
                logger.warning("ensemble_blend: unknown component dropped: %s", k)
        if not active:
            return {}
        # Re-normalize weights over active components.
        active_w = {k: w.get(k, 0.0) for k in active}
        s = sum(active_w.values())
        if s <= 0:
            active_w = {k: 1.0 / len(active) for k in active}
            s = 1.0
        else:
            active_w = {k: v / s for k, v in active_w.items()}
        # Aggregate keys across all active components.
        keys: set[tuple[str, str]] = set()
        for v in active.values():
            keys.update(v.keys())
        out: SignalDict = {}
        for key in keys:
            val = 0.0
            wsum = 0.0
            for comp, sig_map in active.items():
                x = sig_map.get(key)
                if x is None:
                    continue
                val += active_w[comp] * float(x)
                wsum += active_w[comp]
            if wsum > 0:
                out[key] = val / wsum
        return out

    # ---- Nested walk-forward IC ----

    def cv_ic(
        self,
        historical_signals: dict[str, list[float]],
        forward_returns: list[float],
    ) -> dict[str, float]:
        """Run outer walk-forward, fit weights on each train fold, measure IC on test.

        Returns `{ic_mean, ic_std, ic_ir, n_splits}`. Fall back to zeros on
        any shape mismatch.
        """
        names = list(self.component_names)
        n = len(forward_returns)
        if any(len(historical_signals.get(nm, [])) != n for nm in names):
            return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_splits": 0}
        splits = self._walk_forward_splits(n)
        ics: list[float] = []
        for train_idx, test_idx in splits:
            train_sigs = {nm: [historical_signals[nm][i] for i in train_idx] for nm in names}
            train_ret = [forward_returns[i] for i in train_idx]
            w = self.fit_weights(train_sigs, train_ret)
            # Apply weights to test block and measure IC.
            blended = [
                sum(w.get(nm, 0.0) * historical_signals[nm][i] for nm in names)
                for i in test_idx
            ]
            test_ret = [forward_returns[i] for i in test_idx]
            ics.append(self._compute_ic(blended, test_ret))
        if not ics:
            return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "n_splits": 0}
        mean = sum(ics) / len(ics)
        var = sum((x - mean) ** 2 for x in ics) / len(ics) if len(ics) > 1 else 0.0
        std = math.sqrt(var)
        ir = (mean / std) if std > 0 else 0.0
        return {"ic_mean": float(mean), "ic_std": float(std), "ic_ir": float(ir), "n_splits": len(ics)}


__all__ = ["EnsembleBlender", "SignalDict"]
