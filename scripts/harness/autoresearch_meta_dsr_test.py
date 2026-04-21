"""phase-8.5.10 meta-search DSR verification."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.autoresearch.meta_dsr import (
    TrialLedger, meta_dsr, required_dsr, cpcv_applied_on,
    STRICT_DSR_MIN, LOOSE_DSR_MIN,
)


def case_every_trial_logged_including_abandoned() -> tuple[bool, str]:
    ledger = TrialLedger()
    ledger.log({"trial_id": "a", "dsr": 0.90, "abandoned": False})
    ledger.log({"trial_id": "b", "dsr": 0.00, "abandoned": True})
    ledger.log({"trial_id": "c", "dsr": 0.98, "abandoned": False})
    if ledger.n != 3:
        return False, f"expected n=3, got {ledger.n}"
    if ledger.n_abandoned != 1:
        return False, f"expected n_abandoned=1, got {ledger.n_abandoned}"
    return True, f"ledger logged 3 trials including 1 abandoned"


def case_dsr_recomputed_at_cumulative_N() -> tuple[bool, str]:
    trials = [{"dsr": 0.99} for _ in range(10)]
    out_small = meta_dsr(trials, cumulative_n=10)
    out_large = meta_dsr(trials, cumulative_n=10000)
    # Larger cumulative N -> larger penalty -> lower adjusted DSR
    if out_large["mean"] >= out_small["mean"]:
        return False, f"penalty did not increase with N: small={out_small['mean']}, large={out_large['mean']}"
    return True, f"adjusted dsr: n=10 mean={out_small['mean']:.4f} -> n=10000 mean={out_large['mean']:.4f}"


def case_dsr_gt_0_99_required_when_N_gt_50() -> tuple[bool, str]:
    assert required_dsr(10) == LOOSE_DSR_MIN == 0.95
    assert required_dsr(50) == LOOSE_DSR_MIN == 0.95
    assert required_dsr(51) == STRICT_DSR_MIN == 0.99
    assert required_dsr(1000) == STRICT_DSR_MIN == 0.99
    return True, f"required_dsr: N<=50 -> 0.95, N>50 -> 0.99"


def case_cpcv_applied_on_promoted_only() -> tuple[bool, str]:
    # Promoted WITHOUT cpcv_applied -> violation
    t1 = {"trial_id": "t1", "promoted": True, "abandoned": False, "cpcv_applied": False}
    if cpcv_applied_on(t1):
        return False, "promoted trial without cpcv_applied=True should violate"
    # Promoted WITH cpcv_applied -> ok
    t2 = {"trial_id": "t2", "promoted": True, "abandoned": False, "cpcv_applied": True}
    if not cpcv_applied_on(t2):
        return False, "promoted trial with cpcv_applied=True should pass"
    # Non-promoted -> vacuously true
    t3 = {"trial_id": "t3", "promoted": False}
    if not cpcv_applied_on(t3):
        return False, "non-promoted trial should vacuously pass"
    # Abandoned -> vacuously true
    t4 = {"trial_id": "t4", "promoted": True, "abandoned": True}
    if not cpcv_applied_on(t4):
        return False, "abandoned trial should vacuously pass even if promoted flag set"
    return True, "cpcv_applied_on gated to promoted-non-abandoned trials"


def main() -> int:
    cases = [
        ("every_trial_logged_including_abandoned", case_every_trial_logged_including_abandoned),
        ("dsr_recomputed_at_cumulative_N", case_dsr_recomputed_at_cumulative_N),
        ("dsr_gt_0_99_required_when_N_gt_50", case_dsr_gt_0_99_required_when_N_gt_50),
        ("cpcv_applied_on_promoted_only", case_cpcv_applied_on_promoted_only),
    ]
    ok_all = True
    for name, fn in cases:
        try:
            ok, msg = fn()
        except Exception as exc:
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"{'PASS' if ok else 'FAIL'}: {name} -- {msg}")
        if not ok:
            ok_all = False
    print("---")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
