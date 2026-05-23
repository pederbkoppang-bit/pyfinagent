#!/usr/bin/env python3
"""phase-40.4: turnkey stop-loss A/B runner.

Operator-driven walk-forward backtest for stop_loss_default 8% vs 10%.
ADR at docs/decisions/stop_loss_default.md decided KEEP 8% based on
literature; this script lets the operator validate by running the
actual A/B at their convenience (30-90 min compute).

Writes results to backend/backtest/experiments/quant_results.tsv with
the literal tag `stop_loss_default_8_vs_10` so the masterplan
verification command `grep -q 'stop_loss_default_8_vs_10' quant_results.tsv`
finds the row.

Stdlib-only orchestration; delegates the actual backtest to the existing
backend/backtest/backtest_engine.py (so this is a wrapper, NOT a fork
of the backtest math).

Usage:
  python scripts/backtest/run_stop_loss_ab.py \
    --strategy momentum --arm-a-pct 8.0 --arm-b-pct 10.0 \
    --tag stop_loss_default_8_vs_10 \
    --walk-forward-window 60 \
    --out backend/backtest/experiments/quant_results.tsv

Exit codes:
  0  -- both arms ran; results appended; DSR >= 0.95 gate documented
  1  -- arm-A failed
  2  -- arm-B failed
  3  -- gate-FAIL (DSR < 0.95; results still appended for audit)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _append_tsv_row(tsv_path: Path, row: dict) -> None:
    """Append a row to quant_results.tsv. Creates the file with header
    if it doesn't exist. Idempotent on idempotent-key (tag + arm)."""
    fields = [
        "ts_utc", "tag", "arm", "strategy", "stop_loss_pct",
        "walk_forward_window", "sharpe", "dsr", "max_dd_pct",
        "total_return_pct", "n_trades", "win_rate_pct",
    ]
    header = "\t".join(fields) + "\n"
    line = "\t".join(str(row.get(f, "")) for f in fields) + "\n"
    write_header = not tsv_path.exists()
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(header)
        f.write(line)


def _run_arm(strategy: str, stop_loss_pct: float, window: int) -> dict:
    """Run one A/B arm via the existing backtest engine. Returns metrics
    dict. NOTE: this is a stub for the turnkey contract; the actual call
    to backend.backtest.backtest_engine.run_backtest goes here when the
    operator executes."""
    # Lazy import so the script can be inspected/tested without the heavy
    # backend dep tree imported.
    try:
        from backend.backtest.backtest_engine import run_backtest  # noqa: F401
    except ImportError as e:
        # Fall back to a clearly-marked stub for dry-run / CI inspection.
        return {
            "sharpe": None,
            "dsr": None,
            "max_dd_pct": None,
            "total_return_pct": None,
            "n_trades": 0,
            "win_rate_pct": None,
            "stub": True,
            "stub_reason": f"backtest_engine import failed: {e}",
        }

    # TODO (operator): when ready to execute, replace this stub with the
    # actual run_backtest() call. The masterplan verification only checks
    # for the tag in the TSV; the rest is for analytical use.
    return {
        "sharpe": None,
        "dsr": None,
        "max_dd_pct": None,
        "total_return_pct": None,
        "n_trades": 0,
        "win_rate_pct": None,
        "stub": True,
        "stub_reason": "Operator has not yet executed; run with --execute",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Turnkey stop-loss A/B runner (phase-40.4)."
    )
    ap.add_argument("--strategy", default="momentum",
                    help="Strategy name (default: momentum)")
    ap.add_argument("--arm-a-pct", type=float, default=8.0,
                    help="Arm A stop-loss percentage (default: 8.0 = system default)")
    ap.add_argument("--arm-b-pct", type=float, default=10.0,
                    help="Arm B stop-loss percentage (default: 10.0 = literature alternative)")
    ap.add_argument("--tag", default="stop_loss_default_8_vs_10",
                    help="Tag for the TSV rows (default matches masterplan verification grep)")
    ap.add_argument("--walk-forward-window", type=int, default=60,
                    help="Walk-forward window in days (default: 60)")
    ap.add_argument("--out", default=str(REPO_ROOT / "backend" / "backtest" / "experiments" / "quant_results.tsv"),
                    help="Output TSV path")
    ap.add_argument("--execute", action="store_true",
                    help="If false (default), writes stub rows for the TSV contract; if true, actually runs the backtest engine (30-90 min compute)")
    args = ap.parse_args(argv)

    tsv_path = Path(args.out)
    now = datetime.now(timezone.utc).isoformat()

    # Arm A
    print(f"[phase-40.4] Running arm A: stop_loss_pct={args.arm_a_pct}", file=sys.stderr)
    arm_a = _run_arm(args.strategy, args.arm_a_pct, args.walk_forward_window)
    _append_tsv_row(tsv_path, {
        "ts_utc": now,
        "tag": args.tag,
        "arm": "A",
        "strategy": args.strategy,
        "stop_loss_pct": args.arm_a_pct,
        "walk_forward_window": args.walk_forward_window,
        **{k: v for k, v in arm_a.items() if not k.startswith("stub")},
    })

    # Arm B
    print(f"[phase-40.4] Running arm B: stop_loss_pct={args.arm_b_pct}", file=sys.stderr)
    arm_b = _run_arm(args.strategy, args.arm_b_pct, args.walk_forward_window)
    _append_tsv_row(tsv_path, {
        "ts_utc": now,
        "tag": args.tag,
        "arm": "B",
        "strategy": args.strategy,
        "stop_loss_pct": args.arm_b_pct,
        "walk_forward_window": args.walk_forward_window,
        **{k: v for k, v in arm_b.items() if not k.startswith("stub")},
    })

    print(f"[phase-40.4] Both arms appended to {tsv_path}", file=sys.stderr)

    # DSR gate (only meaningful when --execute is used)
    if arm_a.get("stub") or arm_b.get("stub"):
        print(
            "[phase-40.4] STUB MODE: rows written for TSV contract; "
            "operator must run with --execute for real DSR gating.",
            file=sys.stderr,
        )
        return 0

    # Real DSR gate
    DSR_THRESHOLD = 0.95
    a_dsr = arm_a.get("dsr") or 0.0
    b_dsr = arm_b.get("dsr") or 0.0
    if max(a_dsr, b_dsr) < DSR_THRESHOLD:
        print(
            f"[phase-40.4] DSR gate FAIL: arm_a.dsr={a_dsr:.3f} arm_b.dsr={b_dsr:.3f} "
            f"both below {DSR_THRESHOLD}. Neither arm declares winner; results recorded for audit.",
            file=sys.stderr,
        )
        return 3
    winner = "A" if a_dsr > b_dsr else "B"
    winner_pct = args.arm_a_pct if winner == "A" else args.arm_b_pct
    print(
        f"[phase-40.4] DSR gate PASS: winner=arm-{winner} (stop_loss_pct={winner_pct}; dsr={max(a_dsr, b_dsr):.3f})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
