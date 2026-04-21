"""phase-8.5.5 gate verification script."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.gate import PromotionGate, cpcv_folds


def case_dsr_below_threshold() -> tuple[bool, str]:
    g = PromotionGate()
    trial = {"trial_id": "t1", "dsr": 0.90, "pbo": 0.10}
    v = g.evaluate(trial)
    if v["promoted"]:
        return False, "promoted despite dsr=0.90 < 0.95"
    if "dsr_below_min" not in (v.get("reason") or ""):
        return False, f"reason did not cite dsr_below_min: {v['reason']!r}"
    return True, "dsr below 0.95 -> rejected"


def case_pbo_above_threshold() -> tuple[bool, str]:
    g = PromotionGate()
    trial = {"trial_id": "t2", "dsr": 0.99, "pbo": 0.30}
    v = g.evaluate(trial)
    if v["promoted"]:
        return False, "promoted despite pbo=0.30 > 0.20"
    if "pbo_above_max" not in (v.get("reason") or ""):
        return False, f"reason did not cite pbo_above_max: {v['reason']!r}"
    return True, "pbo above 0.20 -> rejected"


def case_cpcv_applied() -> tuple[bool, str]:
    folds = cpcv_folds(n=6, k=2)
    # C(6,2) = 15 expected fold pairs
    if len(folds) != 15:
        return False, f"expected 15 CPCV folds for n=6 k=2, got {len(folds)}"
    train, test = folds[0]
    if len(test) != 2:
        return False, "test-fold size mismatch"
    if set(train).intersection(set(test)):
        return False, "train/test overlap in CPCV fold (purge violated)"
    if len(train) + len(test) != 6:
        return False, "train+test != n"
    return True, f"cpcv_folds(6,2) -> 15 clean folds"


def case_rejection_and_revert_regression() -> tuple[bool, str]:
    """A rejected trial must NOT mutate the input dict or the gate."""
    g = PromotionGate()
    trial = {"trial_id": "t3", "dsr": 0.80, "pbo": 0.50, "state": {"counter": 0}}
    before = copy.deepcopy(trial)
    v = g.evaluate(trial)
    if v["promoted"]:
        return False, "should have been rejected"
    if trial != before:
        return False, f"trial mutated by evaluate(): {trial} != {before}"
    # Gate itself is frozen dataclass; attempting to mutate would raise.
    try:
        g.min_dsr = 0.10  # type: ignore[misc]
        return False, "gate was mutable (frozen=True should block)"
    except Exception:
        pass
    return True, "rejection did not mutate trial; gate is frozen"


def main() -> int:
    cases = [
        ("dsr_gt_0_95_required", case_dsr_below_threshold),
        ("pbo_lt_0_2_required", case_pbo_above_threshold),
        ("cpcv_applied", case_cpcv_applied),
        ("rejection_and_revert_regression_passes", case_rejection_and_revert_regression),
    ]
    all_pass = True
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}: {name} -- {msg}")
        if not ok:
            all_pass = False
    print("---")
    print("PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
