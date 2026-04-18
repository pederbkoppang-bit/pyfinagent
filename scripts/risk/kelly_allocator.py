"""phase-4.8 step 4.8.3 CLI: Fractional-Kelly allocator dry-run.

Seeds 5 strategies (momentum / mean-reversion / triple-barrier /
factor / quality) with realistic annualized mu and a correlation
matrix, runs `fractional_kelly`, and writes
`handoff/allocator_output.json`.
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

from backend.services.kelly_allocator import (  # noqa: E402
    DEFAULT_CAP, DEFAULT_FRACTION, fractional_kelly,
)

OUT = REPO / "handoff" / "allocator_output.json"


STRATEGIES = [
    {"name": "momentum",       "mu_ann": 0.09, "sigma_ann": 0.16},
    {"name": "mean_reversion", "mu_ann": 0.06, "sigma_ann": 0.13},
    {"name": "triple_barrier", "mu_ann": 0.08, "sigma_ann": 0.15},
    {"name": "factor_model",   "mu_ann": 0.07, "sigma_ann": 0.14},
    {"name": "quality",        "mu_ann": 0.05, "sigma_ann": 0.11},
]

# Seeded correlation matrix (symmetric, realistic for moderately-
# correlated US-equity strategies).
_CORR = np.array([
    [1.00, -0.10, 0.55, 0.30, 0.25],
    [-0.10, 1.00, -0.05, 0.10, 0.15],
    [0.55, -0.05, 1.00, 0.40, 0.30],
    [0.30, 0.10, 0.40, 1.00, 0.50],
    [0.25, 0.15, 0.30, 0.50, 1.00],
])


def build_mu_sigma() -> tuple[np.ndarray, np.ndarray]:
    sigmas = np.array([s["sigma_ann"] for s in STRATEGIES])
    mu = np.array([s["mu_ann"] for s in STRATEGIES])
    Sigma = np.outer(sigmas, sigmas) * _CORR
    # Symmetrize defensively.
    Sigma = (Sigma + Sigma.T) / 2
    return mu, Sigma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP)
    args = ap.parse_args()

    mu, Sigma = build_mu_sigma()
    allocs = fractional_kelly(mu, Sigma, k=args.fraction, cap=args.cap)
    full = np.linalg.solve(Sigma, mu).tolist()

    strategies_out = []
    for i, s in enumerate(STRATEGIES):
        strategies_out.append({
            "name": s["name"],
            "mu_annual": s["mu_ann"],
            "sigma_annual": s["sigma_ann"],
            "kelly_full": round(full[i], 4),
            "kelly_fractional": round(full[i] * args.fraction, 4),
            "alloc_pct": round(allocs[i], 4),
            "capped": round(allocs[i], 4) == round(args.cap, 4),
        })

    result = {
        "step": "4.8.3",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "config": {
            "fraction": args.fraction,
            "cap": args.cap,
        },
        "strategies": strategies_out,
        "total_alloc_pct": round(sum(allocs), 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "total_alloc_pct": result["total_alloc_pct"],
        "max_alloc_pct": max(s["alloc_pct"] for s in strategies_out),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
