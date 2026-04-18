"""phase-4.8 step 4.8.3 Kelly-allocator audit.

Discriminating tests:
1. Baseline dry-run: max allocation <= cap (30%).
2. Renormalization: sum of allocations <= 1.0 within a tolerance.
3. **Covariance-based mixing** (the teeth): with two otherwise-
   identical strategies, switching the correlation between them
   from 0.0 to 0.9 must SHIFT the allocation vector. If it doesn't,
   the allocator is a mu-only rule and the contract is violated.
4. Singular Sigma raises ValueError.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.kelly_allocator import fractional_kelly  # noqa: E402

OUT = REPO / "handoff" / "kelly_allocator_audit.json"


def _cov_from_corr(sigmas: np.ndarray, corr: np.ndarray) -> np.ndarray:
    C = np.outer(sigmas, sigmas) * corr
    return (C + C.T) / 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # 1. Baseline: dry-run artifact max <= 0.30
    dry = json.loads((REPO / "handoff" / "allocator_output.json").read_text())
    max_alloc = max(s["alloc_pct"] for s in dry["strategies"])
    t1 = max_alloc <= 0.30 + 1e-9
    if not t1:
        reasons.append(f"max alloc {max_alloc} > 0.30 cap")

    # 2. Total alloc <= 1 + eps
    total = sum(s["alloc_pct"] for s in dry["strategies"])
    t2 = total <= 1.0 + 1e-9
    if not t2:
        reasons.append(f"sum alloc {total} > 1")

    # 3. Covariance-based mixing: two strategies with identical
    #    mu and sigma but correlation {0.0 vs 0.9} must give
    #    DIFFERENT allocations. Use small mu so fractional-Kelly
    #    stays below the cap and the covariance effect is visible.
    mu = np.array([0.01, 0.01])
    sigmas = np.array([0.15, 0.15])
    corr_lo = np.array([[1.0, 0.0], [0.0, 1.0]])
    corr_hi = np.array([[1.0, 0.9], [0.9, 1.0]])
    alloc_lo = np.array(fractional_kelly(mu, _cov_from_corr(sigmas, corr_lo)))
    alloc_hi = np.array(fractional_kelly(mu, _cov_from_corr(sigmas, corr_hi)))
    drift = float(np.linalg.norm(alloc_lo - alloc_hi))
    # Identical inputs up to correlation -> allocations must differ.
    t3 = drift > 1e-6
    if not t3:
        reasons.append(
            f"covariance test: alloc_lo={alloc_lo.tolist()}, "
            f"alloc_hi={alloc_hi.tolist()}, drift={drift}"
        )

    # 4. Singular Sigma raises ValueError.
    t4 = False
    try:
        fractional_kelly(
            [0.1, 0.1],
            # singular: two identical rows
            [[0.04, 0.04], [0.04, 0.04]],
        )
        reasons.append("singular Sigma did not raise")
    except ValueError:
        t4 = True

    verdict = "PASS" if (t1 and t2 and t3 and t4) else "FAIL"
    result = {
        "step": "4.8.3",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "baseline_max_alloc_le_cap": t1,
        "baseline_total_alloc_le_1": t2,
        "covariance_based_mixing": t3,
        "singular_sigma_raises": t4,
        "covariance_test": {
            "alloc_lo_corr": alloc_lo.tolist(),
            "alloc_hi_corr": alloc_hi.tolist(),
            "drift": round(drift, 6),
        },
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "t1_cap": t1, "t2_sum": t2, "t3_cov": t3, "t4_singular": t4,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
