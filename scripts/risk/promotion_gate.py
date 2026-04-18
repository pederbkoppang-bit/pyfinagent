"""phase-4.8 step 4.8.5 CLI for gradual-rollout promotion gate.

`--dry-run` reads backend/backtest/experiments/optimizer_best.json,
evaluates three seeded candidates through `evaluate_stage`, and
ensures optimizer_best.json has `allocation_pct` set (defaults to
the canary 5% when no prior stage exists).

Emits `handoff/promotion_gate_output.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.services.promotion_gate import (  # noqa: E402
    STAGES, evaluate_stage, update_optimizer_best,
)

OPTIMIZER_BEST = REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
OUT = REPO / "handoff" / "promotion_gate_output.json"


# Seeded evaluation candidates exercising different branches.
CANDIDATES = [
    {
        "name": "benign_advance",
        "challenger": {"psr": 0.98, "pbo": 0.15, "kill_events": 0},
        "champion":   {"psr": 0.92, "pbo": 0.20, "kill_events": 0},
        "current_stage": 0, "days_at_stage": 20,
        "consecutive_failures": 0,
    },
    {
        "name": "psr_failure",
        "challenger": {"psr": 0.85, "pbo": 0.18, "kill_events": 0},
        "champion":   {"psr": 0.95, "pbo": 0.20, "kill_events": 0},
        "current_stage": 1, "days_at_stage": 35,
        "consecutive_failures": 1,
    },
    {
        "name": "insufficient_days",
        "challenger": {"psr": 0.97, "pbo": 0.10, "kill_events": 0},
        "champion":   {"psr": 0.92, "pbo": 0.20, "kill_events": 0},
        "current_stage": 0, "days_at_stage": 5,
        "consecutive_failures": 0,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    evaluations = [
        {
            "name": c["name"],
            "input": {
                "challenger": c["challenger"],
                "champion": c["champion"],
                "current_stage": c["current_stage"],
                "days_at_stage": c["days_at_stage"],
                "consecutive_failures": c["consecutive_failures"],
            },
            "result": evaluate_stage(
                challenger=c["challenger"],
                champion=c["champion"],
                current_stage=c["current_stage"],
                days_at_stage=c["days_at_stage"],
                consecutive_failures=c["consecutive_failures"],
            ),
        }
        for c in CANDIDATES
    ]

    # Ensure optimizer_best.json has allocation_pct + stage (default
    # canary 5% if this is the first deploy).
    existing = {}
    if OPTIMIZER_BEST.exists():
        existing = json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8"))
    if "allocation_pct" not in existing:
        existing_stage = 0
        existing_alloc = STAGES[0]  # 0.05 canary default
        update_optimizer_best(
            OPTIMIZER_BEST,
            allocation_pct=existing_alloc,
            stage=existing_stage,
        )
        file_updated = True
    else:
        # Preserve in-place; don't reset an existing stage.
        file_updated = False

    # Re-read post-update
    final = json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8"))

    report = {
        "step": "4.8.5",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "stages": STAGES,
        "optimizer_best_path": str(OPTIMIZER_BEST.relative_to(REPO)),
        "file_updated": file_updated,
        "optimizer_best_snapshot": {
            "allocation_pct": final.get("allocation_pct"),
            "stage": final.get("stage"),
            "preserved_keys": [k for k in final.keys()
                                if k not in {"allocation_pct", "stage",
                                              "challenger_run_id"}],
        },
        "evaluations": evaluations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "file_updated": file_updated,
        "allocation_pct": final.get("allocation_pct"),
        "stage": final.get("stage"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
