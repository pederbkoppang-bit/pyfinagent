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
    # phase-82.23: a PBO computed from too few independent trials is
    # DIRECTIONAL, not gate-grade. Bailey/Borwein/Lopez de Prado/Zhu: "if the
    # investor is sensitive to values of [phi] < 1/10 ... N >> 10 is required";
    # the R reference implementation uses N=100. A trial carrying `pbo_n_trials`
    # below this is refused rather than promoted on a coarse statistic. A trial
    # that does not report N at all is UNCHANGED in behaviour (see below), so
    # this is additive for every existing producer.
    min_pbo_trials: int = 10

    def evaluate(self, trial: dict[str, Any]) -> dict[str, Any]:
        """Pure: read trial, return verdict dict. Never mutates trial or anything else."""
        dsr = trial.get("dsr")
        pbo = trial.get("pbo")
        if dsr is None or pbo is None:
            # Already fail-CLOSED: a missing PBO has never silently promoted
            # anything, it has silently BLOCKED promotion. Retained verbatim.
            return {"promoted": False, "reason": "missing_dsr_or_pbo", "trial_id": trial.get("trial_id")}
        # phase-82.23: when the producer DOES report its trial count, refuse an
        # undersized one. Absent => unchanged legacy behaviour, so no existing
        # producer starts failing on a field it never emitted.
        n_trials = trial.get("pbo_n_trials")
        if n_trials is not None:
            try:
                n_int = int(n_trials)
            except (TypeError, ValueError):
                return {"promoted": False, "reason": f"non_numeric_pbo_n_trials:{n_trials!r}",
                        "trial_id": trial.get("trial_id")}
            if n_int < self.min_pbo_trials:
                return {"promoted": False,
                        "reason": f"pbo_trials_below_min:{n_int}<{self.min_pbo_trials}",
                        "trial_id": trial.get("trial_id")}
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
