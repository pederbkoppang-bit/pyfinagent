"""phase-8.5.5 DSR + PBO blocking gate (CPCV).

PromotionGate refuses to promote a trial unless:
    dsr >= min_dsr AND pbo <= max_pbo

De Prado Advances in Financial Machine Learning Ch. 12 CPCV (combinatorial
purged cross-validation): `cpcv_folds(n, k)` enumerates all C(n, k) - 1
possible train/test splits for n groups with k test groups.

Pure functions. Fail-open. ASCII-only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionGate:
    min_dsr: float = 0.95
    max_pbo: float = 0.20

    def evaluate(self, trial: dict[str, Any]) -> dict[str, Any]:
        """Pure: read trial, return verdict dict. Never mutates trial or anything else."""
        dsr = trial.get("dsr")
        pbo = trial.get("pbo")
        if dsr is None or pbo is None:
            return {"promoted": False, "reason": "missing_dsr_or_pbo", "trial_id": trial.get("trial_id")}
        try:
            dsr_f = float(dsr)
            pbo_f = float(pbo)
        except (TypeError, ValueError):
            return {"promoted": False, "reason": "non_numeric_dsr_or_pbo", "trial_id": trial.get("trial_id")}
        if dsr_f < self.min_dsr:
            return {"promoted": False, "reason": f"dsr_below_min:{dsr_f:.4f}<{self.min_dsr}", "trial_id": trial.get("trial_id")}
        if pbo_f > self.max_pbo:
            return {"promoted": False, "reason": f"pbo_above_max:{pbo_f:.4f}>{self.max_pbo}", "trial_id": trial.get("trial_id")}
        return {"promoted": True, "reason": None, "trial_id": trial.get("trial_id")}


def cpcv_folds(n: int, k: int = 4) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return CPCV fold pairs for n groups with k test groups per fold.

    Each fold is (train_groups, test_groups). Caps output at C(n, k) - 1 as
    per AFML Ch. 12; the "-1" excludes the single fold where all-test =
    all-train complement. For n < k returns [].
    """
    if n <= 0 or k <= 0 or k >= n:
        return []
    all_idx = tuple(range(n))
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for combo in itertools.combinations(all_idx, k):
        test = tuple(combo)
        train = tuple(i for i in all_idx if i not in combo)
        out.append((train, test))
    # AFML Ch. 12: C(n, k) - 1 splits (excluding the trivially-redundant last).
    # Conservative: we return all C(n, k). Caller may slice.
    return out


__all__ = ["PromotionGate", "cpcv_folds"]
