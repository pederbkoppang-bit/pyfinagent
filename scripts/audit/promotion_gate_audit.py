"""phase-4.8 step 4.8.5 audit: promotion-gate teeth.

Four tests:
(a) allocation_pct field present in optimizer_best.json
(b) `evaluate_stage` returns advance for benign inputs + min-days met
(c) gate BLOCKS advance when PSR < champion (returns "regress" or
    "demote", not "advance")
(d) gate BLOCKS advance when days_at_stage < min (returns "hold"
    with insufficient-days reason)

Emits handoff/promotion_gate_audit.json. `--check` exits 1 on FAIL.
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

from backend.services.promotion_gate import evaluate_stage  # noqa: E402

OPTIMIZER_BEST = REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
OUT = REPO / "handoff" / "promotion_gate_audit.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # (a) file has allocation_pct
    t_a = OPTIMIZER_BEST.exists()
    blob = {}
    if t_a:
        blob = json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8"))
        t_a = "allocation_pct" in blob
    if not t_a:
        reasons.append("optimizer_best.json missing allocation_pct")

    # (b) benign advance
    r_b = evaluate_stage(
        challenger={"psr": 0.97, "pbo": 0.18, "kill_events": 0},
        champion={"psr": 0.90, "pbo": 0.20, "kill_events": 0},
        current_stage=0, days_at_stage=20,
    )
    t_b = (r_b["decision"] == "advance"
           and r_b["next_allocation_pct"] > 0.05 + 1e-9)
    if not t_b:
        reasons.append(f"benign advance failed: {r_b}")

    # (c) PSR-failure -> NOT advance
    r_c = evaluate_stage(
        challenger={"psr": 0.70, "pbo": 0.18, "kill_events": 0},
        champion={"psr": 0.95, "pbo": 0.20, "kill_events": 0},
        current_stage=1, days_at_stage=40,
    )
    t_c = (r_c["decision"] != "advance"
           and any("psr_below_champion" in s for s in r_c["reasons"]))
    if not t_c:
        reasons.append(f"psr failure should have blocked advance: {r_c}")

    # (d) insufficient-days -> hold (not advance) with reason
    r_d = evaluate_stage(
        challenger={"psr": 0.97, "pbo": 0.10, "kill_events": 0},
        champion={"psr": 0.92, "pbo": 0.20, "kill_events": 0},
        current_stage=0, days_at_stage=3,
    )
    t_d = (r_d["decision"] == "hold"
           and any("days_at_stage_insufficient" in s for s in r_d["reasons"]))
    if not t_d:
        reasons.append(f"insufficient-days should have blocked advance: {r_d}")

    all_ok = t_a and t_b and t_c and t_d
    verdict = "PASS" if all_ok else "FAIL"

    result = {
        "step": "4.8.5",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "allocation_pct_field_present": t_a,
        "allocation_pct_value": blob.get("allocation_pct") if isinstance(blob, dict) else None,
        "stage_value": blob.get("stage") if isinstance(blob, dict) else None,
        "benign_advance_ok": t_b,
        "psr_failure_blocks": t_c,
        "days_insufficient_blocks": t_d,
        "eval_samples": {
            "benign": r_b, "psr_failure": r_c, "days_insufficient": r_d,
        },
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "t_a": t_a, "t_b": t_b, "t_c": t_c, "t_d": t_d,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
