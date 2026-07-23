"""phase-4.8 step 4.8.5 CLI for gradual-rollout promotion gate.

Evaluates three seeded candidates through `evaluate_stage` and ensures
optimizer_best.json has `allocation_pct` set (defaults to the canary 5%
when no prior stage exists).

`--dry-run` (phase-75.8, gap6-10): strictly NO-WRITE against
optimizer_best.json. Both writers (the allocation-stage init and the
gauntlet-hash stamp) are guarded; the dry-run branch prints the
would-be mutation instead (terraform-plan idiom). Only the
`handoff/promotion_gate_output.json` preview report is emitted.

phase-4.9.7 extension: `--require-gauntlet` loads the latest Gauntlet
report (from `handoff/gauntlet/<strategy>/report.json`), runs the
pass-criteria evaluator, and REFUSES to promote unless
`overall_pass=True`. On pass, the SHA-256 of the report is stored
under `gauntlet_report_hash` in `optimizer_best.json` so downstream
auditors can verify which Gauntlet run authorised the promotion.

phase-75.8 (gap6-01, consumer side): before evaluating, rejects any
report carrying the dry-run stub fingerprint -- `bt_drawdown` exactly
equal to `drawdown` for every non-skipped regime -- because such a
report is fabricated (stub) evidence that passes all four evaluator
hard gates by construction.

Emits `handoff/promotion_gate_output.json`.
"""
from __future__ import annotations

import argparse
import hashlib
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
from backend.backtest.gauntlet.evaluator import evaluate as evaluate_gauntlet  # noqa: E402

OPTIMIZER_BEST = REPO / "backend" / "backtest" / "experiments" / "optimizer_best.json"
OUT = REPO / "handoff" / "promotion_gate_output.json"
GAUNTLET_ROOT = REPO / "handoff" / "gauntlet"


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


def _load_gauntlet_report(strategy: str) -> tuple[dict, str, Path]:
    """Load the most recent gauntlet report for `strategy`.

    Returns (report_dict, sha256_hex, path). Raises FileNotFoundError
    if the report is missing -- the caller (promotion gate) uses this
    to block with a clear diagnostic.
    """
    path = GAUNTLET_ROOT / strategy / "report.json"
    if not path.exists():
        raise FileNotFoundError(
            f"gauntlet report missing at {path}; run "
            f"scripts/risk/gauntlet.py --strategy {strategy} first"
        )
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8")), sha, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--require-gauntlet",
        action="store_true",
        help="phase-4.9.7: block promotion unless gauntlet evaluator returns overall_pass",
    )
    ap.add_argument(
        "--strategy",
        default="baseline",
        help="strategy name under handoff/gauntlet/<strategy>/ (default: baseline)",
    )
    args = ap.parse_args()

    gauntlet_status: dict | None = None
    gauntlet_hash: str | None = None
    if args.require_gauntlet:
        try:
            report, gauntlet_hash, rp = _load_gauntlet_report(args.strategy)
        except FileNotFoundError as e:
            print(json.dumps({"blocked": True, "reason": str(e)}))
            return 1
        # phase-75.8 (gap6-01): stub-fingerprint rejection. The dry-run
        # stub copies drawdown into bt_drawdown for every regime, so exact
        # equality across ALL non-skipped regimes marks fabricated
        # evidence (real live-vs-backtest drawdowns never match to the
        # last bit on every regime). Skipped regimes are filtered like the
        # evaluator does; an empty non-skipped list is NOT fingerprinted
        # (all([]) is True -- guard the vacuous case explicitly).
        non_skipped = [
            r for r in (report.get("per_regime", []) or [])
            if not r.get("skipped")
        ]
        if non_skipped and all(
            r.get("bt_drawdown") == r.get("drawdown") for r in non_skipped
        ):
            print(json.dumps({
                "blocked": True,
                "reason": (
                    "stub fingerprint: bt_drawdown == drawdown exactly for "
                    "all non-skipped regimes -- report is dry-run/stub "
                    "evidence, not a live gauntlet run"
                ),
                "report_path": str(rp.relative_to(REPO)),
            }))
            return 1
        verdict = evaluate_gauntlet(report)
        gauntlet_status = {
            "strategy": args.strategy,
            "report_path": str(rp.relative_to(REPO)),
            "report_hash": gauntlet_hash,
            "overall_pass": bool(verdict["overall_pass"]),
            "reasons": list(verdict.get("reasons", [])),
            "drawdown_ratio_cap": verdict.get("drawdown_ratio_cap"),
        }
        if not verdict["overall_pass"]:
            print(json.dumps({
                "blocked": True,
                "reason": "gauntlet evaluator overall_pass=False",
                "reasons": verdict["reasons"],
            }))
            return 1

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
    # canary 5% if this is the first deploy). phase-75.8 (gap6-10): under
    # --dry-run this writer is guarded -- print the would-be mutation.
    existing = {}
    if OPTIMIZER_BEST.exists():
        existing = json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8"))
    file_updated = False
    if "allocation_pct" not in existing:
        existing_stage = 0
        existing_alloc = STAGES[0]  # 0.05 canary default
        if args.dry_run:
            print(json.dumps({
                "dry_run": True,
                "would_update": str(OPTIMIZER_BEST.relative_to(REPO)),
                "would_set": {"allocation_pct": existing_alloc,
                              "stage": existing_stage},
            }))
        else:
            update_optimizer_best(
                OPTIMIZER_BEST,
                allocation_pct=existing_alloc,
                stage=existing_stage,
            )
            file_updated = True

    # phase-4.9.7: stamp the gauntlet hash in-place so auditors can
    # cross-reference which gauntlet report authorised the promotion.
    # phase-75.8 (gap6-10): guarded under --dry-run like the init writer.
    if gauntlet_status is not None and gauntlet_hash is not None:
        post = json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8")) if OPTIMIZER_BEST.exists() else {}
        if post.get("gauntlet_report_hash") != gauntlet_hash:
            if args.dry_run:
                print(json.dumps({
                    "dry_run": True,
                    "would_update": str(OPTIMIZER_BEST.relative_to(REPO)),
                    "would_set": {"gauntlet_report_hash": gauntlet_hash,
                                  "gauntlet_strategy": args.strategy},
                }))
            else:
                post["gauntlet_report_hash"] = gauntlet_hash
                post["gauntlet_strategy"] = args.strategy
                post["gauntlet_stamped_at"] = datetime.now(timezone.utc).isoformat()
                OPTIMIZER_BEST.write_text(
                    json.dumps(post, indent=2) + "\n", encoding="utf-8"
                )
                file_updated = True

    # Re-read post-update (file may legitimately not exist under --dry-run
    # on a fresh deploy, since the init writer is guarded).
    final = (
        json.loads(OPTIMIZER_BEST.read_text(encoding="utf-8"))
        if OPTIMIZER_BEST.exists() else {}
    )

    report = {
        "step": "4.9.7",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "require_gauntlet": bool(args.require_gauntlet),
        "gauntlet_status": gauntlet_status,
        "stages": STAGES,
        "optimizer_best_path": str(OPTIMIZER_BEST.relative_to(REPO)),
        "file_updated": file_updated,
        "optimizer_best_snapshot": {
            "allocation_pct": final.get("allocation_pct"),
            "stage": final.get("stage"),
            "gauntlet_report_hash": final.get("gauntlet_report_hash"),
            "preserved_keys": [k for k in final.keys()
                                if k not in {"allocation_pct", "stage",
                                              "challenger_run_id",
                                              "gauntlet_report_hash",
                                              "gauntlet_strategy",
                                              "gauntlet_stamped_at"}],
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
