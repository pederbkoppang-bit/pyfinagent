"""
Go-Live drill test: walk-forward return concentration (Phase 4.4.1.4).

Stdlib-only drill. Analyzes the best backtest result's equity curve
to verify that no single walk-forward window contributes > 30% of
total return. Uses only json, sys, pathlib -- no numpy, pandas, or
backend imports.

Run from the repo root:

    python scripts/go_live_drills/walk_forward_concentration_test.py

Exit code 0 on PASS, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "backend" / "backtest" / "experiments" / "results"

CONCENTRATION_THRESHOLD = 0.30  # 30%

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


def nav_at_date(nav_history: list[dict], target_date: str) -> float | None:
    exact = None
    closest_before = None
    closest_after = None
    for entry in nav_history:
        d = entry["date"]
        n = entry["nav"]
        if d == target_date:
            exact = n
            break
        if d < target_date:
            if closest_before is None or d > closest_before[0]:
                closest_before = (d, n)
        elif d > target_date:
            if closest_after is None or d < closest_after[0]:
                closest_after = (d, n)
    if exact is not None:
        return exact
    if closest_after is not None:
        return closest_after[1]
    if closest_before is not None:
        return closest_before[1]
    return None


def main():
    print("=" * 60)
    print("WALK-FORWARD RETURN CONCENTRATION TEST -- Phase 4.4.1.4")
    print("=" * 60)
    print()

    # S0: Find best result file
    print("S0: Locate best result file")
    best_path, best_sharpe = find_best_result()
    check("S0.1 best result found", best_path is not None,
          f"sharpe={best_sharpe:.4f}" if best_path else "no result files")
    if best_path is None:
        print(f"\nFINAL: {passed} passed, {failed} failed")
        sys.exit(1)

    data = json.loads(best_path.read_text())
    check("S0.2 best sharpe > 1.0", best_sharpe > 1.0,
          f"sharpe={best_sharpe:.4f}, file={best_path.name}")
    print()

    # S1: Validate per_window structure
    print("S1: Validate walk-forward windows")
    per_window = data.get("per_window", [])
    check("S1.1 per_window present", len(per_window) > 0,
          f"{len(per_window)} windows")
    check("S1.2 >= 10 windows", len(per_window) >= 10,
          f"{len(per_window)} windows (need diversification)")

    if per_window:
        w0 = per_window[0]
        check("S1.3 test_start key exists", "test_start" in w0)
        check("S1.4 test_end key exists", "test_end" in w0)
    print()

    # S2: Validate equity curve
    print("S2: Validate equity curve / NAV history")
    nav_history = data.get("nav_history", [])
    equity_curve = data.get("equity_curve", [])
    check("S2.1 nav_history present", len(nav_history) > 0,
          f"{len(nav_history)} points")
    check("S2.2 equity_curve present", len(equity_curve) > 0,
          f"{len(equity_curve)} points")

    if nav_history:
        first_date = nav_history[0]["date"]
        last_date = nav_history[-1]["date"]
        first_nav = nav_history[0]["nav"]
        last_nav = nav_history[-1]["nav"]
        total_return = (last_nav - first_nav) / first_nav * 100
        check("S2.3 total return > 0", total_return > 0,
              f"{total_return:.2f}% ({first_date} to {last_date})")
    print()

    # S3: Compute per-window returns from equity curve
    print("S3: Per-window return concentration analysis")
    window_returns = []
    skipped = 0

    for i, w in enumerate(per_window):
        ts = w.get("test_start")
        te = w.get("test_end")
        if not ts or not te:
            skipped += 1
            continue

        nav_start = nav_at_date(nav_history, ts)
        nav_end = nav_at_date(nav_history, te)

        if nav_start is None or nav_end is None:
            skipped += 1
            continue

        dollar_return = nav_end - nav_start
        pct_return = (nav_end - nav_start) / nav_start * 100
        window_returns.append({
            "window_id": i,
            "test_start": ts,
            "test_end": te,
            "nav_start": nav_start,
            "nav_end": nav_end,
            "dollar_return": dollar_return,
            "pct_return": pct_return,
        })

    check("S3.1 windows computed", len(window_returns) > 0,
          f"{len(window_returns)} windows, {skipped} skipped")

    if not window_returns:
        print(f"\nFINAL: {passed} passed, {failed} failed")
        sys.exit(1)

    total_dollar_return = sum(w["dollar_return"] for w in window_returns)
    check("S3.2 total dollar return > 0", total_dollar_return > 0,
          f"${total_dollar_return:,.2f}")

    # Compute each window's contribution to total return
    print()
    print("  Window return breakdown:")
    max_contribution = 0.0
    max_window = None

    for w in window_returns:
        if total_dollar_return != 0:
            contribution = w["dollar_return"] / total_dollar_return
        else:
            contribution = 0
        abs_contribution = abs(contribution)

        marker = ">>>" if abs_contribution > CONCENTRATION_THRESHOLD else "   "
        print(f"  {marker} W{w['window_id']:2d}: {w['test_start']} to {w['test_end']}  "
              f"ret={w['pct_return']:+7.2f}%  contribution={contribution:+6.1%}")

        if abs_contribution > max_contribution:
            max_contribution = abs_contribution
            max_window = w
    print()

    # S4: The critical check
    print("S4: Concentration threshold check")
    check("S4.1 max contribution < 30%",
          max_contribution < CONCENTRATION_THRESHOLD,
          f"max={max_contribution:.1%} in W{max_window['window_id']} "
          f"({max_window['test_start']} to {max_window['test_end']})"
          if max_window else "no windows")

    # S5: Soft robustness notes (informational, not gating)
    print()
    print("S5: Robustness notes (informational)")
    positive_windows = sum(1 for w in window_returns if w["dollar_return"] > 0)
    negative_windows = sum(1 for w in window_returns if w["dollar_return"] < 0)
    zero_windows = sum(1 for w in window_returns if w["dollar_return"] == 0)
    print(f"  [NOTE] Window distribution: {positive_windows} positive, "
          f"{negative_windows} negative, {zero_windows} flat")
    print(f"  [NOTE] Flat windows are expected: engine takes no positions "
          f"when ML filter rejects all candidates")

    top3_contribution = sum(
        sorted([abs(w["dollar_return"]) / abs(total_dollar_return)
                for w in window_returns if total_dollar_return != 0],
               reverse=True)[:3]
    ) if total_dollar_return != 0 else 1.0
    print(f"  [NOTE] Top-3 windows contribute {top3_contribution:.1%} of total return")

    # Summary
    print()
    print("=" * 60)
    verdict = "PASS" if failed == 0 else "FAIL"
    print(f"VERDICT: {verdict} ({passed} passed, {failed} failed)")
    print(f"Max single-window contribution: {max_contribution:.1%} "
          f"(threshold: {CONCENTRATION_THRESHOLD:.0%})")
    print(f"Source: {best_path.name} (Sharpe {best_sharpe:.4f})")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
