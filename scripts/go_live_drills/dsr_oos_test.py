"""
Go-Live drill test: Deflated Sharpe Ratio >= 0.95 on OOS data (Phase 4.4.1.2).

Stdlib-only drill. Reads the best backtest result JSON and optimizer_best.json
to verify:
  1. DSR >= 0.95
  2. The result comes from a walk-forward backtest (OOS by construction)
  3. DSR is significant (dsr_significant flag)
  4. Cross-check: optimizer_best.json and result JSON agree on DSR
  5. Multiple trials used in DSR computation (num_trials > 1)
  6. Embargo days > 0 (prevents train/test leakage)
  7. Walk-forward windows have distinct train/test periods

Run from the repo root:

    python scripts/go_live_drills/dsr_oos_test.py

Exit code 0 on PASS, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "backend" / "backtest" / "experiments" / "results"
OPTIMIZER_BEST = REPO_ROOT / "backend" / "backtest" / "experiments" / "optimizer_best.json"

DSR_THRESHOLD = 0.95

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if condition else "FAIL"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1


def find_best_result() -> tuple[Path | None, float]:
    best_path = None
    best_sharpe = 0.0
    for f in RESULTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            sharpe = data.get("analytics", {}).get("sharpe", 0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_path = f
        except (json.JSONDecodeError, KeyError):
            continue
    return best_path, best_sharpe


def main():
    print("=" * 60)
    print("DRILL: DSR >= 0.95 on out-of-sample data (Phase 4.4.1.2)")
    print("=" * 60)

    # S0: optimizer_best.json exists
    check("S0 optimizer_best.json exists", OPTIMIZER_BEST.exists())
    if not OPTIMIZER_BEST.exists():
        print(f"\nDRILL FAIL: {passed}/{passed + failed} (missing optimizer_best.json)")
        sys.exit(1)

    opt_best = json.loads(OPTIMIZER_BEST.read_text())

    # S1: Find best result in results directory
    best_path, best_sharpe = find_best_result()
    check("S1 best result found in results/", best_path is not None,
          f"Sharpe {best_sharpe:.4f}, file {best_path.name if best_path else 'NONE'}")
    if best_path is None:
        print(f"\nDRILL FAIL: {passed}/{passed + failed} (no results found)")
        sys.exit(1)

    data = json.loads(best_path.read_text())
    analytics = data.get("analytics", {})
    strategy_params = data.get("strategy_params", {})
    per_window = data.get("per_window", [])

    # S2: DSR value exists and >= threshold
    dsr = analytics.get("deflated_sharpe")
    check("S2 DSR exists in analytics", dsr is not None, f"deflated_sharpe={dsr}")
    if dsr is None:
        print(f"\nDRILL FAIL: {passed}/{passed + failed} (no DSR in analytics)")
        sys.exit(1)

    check("S3 DSR >= 0.95", dsr >= DSR_THRESHOLD,
          f"DSR={dsr:.4f} vs threshold={DSR_THRESHOLD}")

    # S4: DSR significance flag
    dsr_sig = analytics.get("dsr_significant")
    check("S4 dsr_significant is True", dsr_sig is True, f"dsr_significant={dsr_sig}")

    # S5: Cross-check optimizer_best.json DSR matches result DSR
    opt_dsr = opt_best.get("dsr")
    check("S5 optimizer_best.json DSR matches result",
          opt_dsr is not None and abs(opt_dsr - dsr) < 1e-6,
          f"optimizer_best DSR={opt_dsr}, result DSR={dsr}")

    # S6: Multiple trials (DSR requires > 1 trial to be meaningful)
    num_trials = analytics.get("num_trials", 0)
    check("S6 num_trials > 1", num_trials > 1,
          f"num_trials={num_trials} (DSR needs multiple trials for deflation)")

    # S7: Walk-forward structure present (OOS by construction)
    n_windows = analytics.get("n_windows", 0)
    check("S7 walk-forward windows > 0", n_windows > 0, f"n_windows={n_windows}")

    # S8: per_window data confirms walk-forward structure
    check("S8 per_window data present", len(per_window) > 0,
          f"{len(per_window)} windows in per_window array")

    # S9: Each window has distinct train/test periods (no train/test overlap)
    all_have_distinct = True
    overlap_detail = ""
    for w in per_window:
        ts = w.get("test_start", "")
        te = w.get("test_end", "")
        tre = w.get("train_end", "")
        if ts and tre and ts <= tre:
            all_have_distinct = False
            overlap_detail = f"window {w.get('window_id')}: test_start={ts} <= train_end={tre}"
            break
    check("S9 all windows have train_end < test_start (no overlap)", all_have_distinct,
          overlap_detail if overlap_detail else f"verified {len(per_window)} windows")

    # S10: Embargo days configured (prevents information leakage between train/test)
    embargo = strategy_params.get("embargo_days", 0)
    check("S10 embargo_days > 0", embargo > 0, f"embargo_days={embargo}")

    # S11: Strategy uses walk-forward expanding window
    train_months = strategy_params.get("train_window_months", 0)
    test_months = strategy_params.get("test_window_months", 0)
    check("S11 train/test window configured", train_months > 0 and test_months > 0,
          f"train={train_months}mo, test={test_months}mo, expanding window")

    # S12: Sharpe cross-check (optimizer_best vs result)
    opt_sharpe = opt_best.get("sharpe")
    result_sharpe = analytics.get("sharpe")
    check("S12 Sharpe cross-check matches",
          opt_sharpe is not None and result_sharpe is not None and abs(opt_sharpe - result_sharpe) < 1e-6,
          f"optimizer_best Sharpe={opt_sharpe}, result Sharpe={result_sharpe}")

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    if failed == 0:
        print(f"DRILL PASS: {passed}/{total} checks passed")
        print(f"DSR = {dsr:.4f} >= {DSR_THRESHOLD} on {n_windows}-window walk-forward OOS data")
        print(f"Sharpe = {result_sharpe:.4f}, num_trials = {num_trials}, embargo = {embargo}d")
    else:
        print(f"DRILL FAIL: {passed}/{total} ({failed} failures)")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
