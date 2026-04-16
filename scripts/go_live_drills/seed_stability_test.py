"""
Go-Live drill test: seed stability (Phase 4.4.1.3).

Stdlib-only drill. Verifies that the seed stability test has been run
with 5 different random seeds and that the resulting Sharpe values
have std < 0.1 with all seeds > 0.9.

Run from the repo root:

    python scripts/go_live_drills/seed_stability_test.py

Exit code 0 on PASS, 1 on any failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "backend" / "backtest" / "experiments" / "results"
SEED_RESULTS_PATH = REPO_ROOT / "handoff" / "seed_stability_results.json"

EXPECTED_SEEDS = [42, 123, 456, 789, 2026]
STD_THRESHOLD = 0.1
MIN_SHARPE = 0.9

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


def main():
    global passed, failed

    print("=" * 60)
    print("SEED STABILITY DRILL -- Phase 4.4.1.3")
    print("=" * 60)
    print()

    # S0: seed_stability_results.json exists
    check("S0 results file exists", SEED_RESULTS_PATH.exists(),
          str(SEED_RESULTS_PATH.name))

    if not SEED_RESULTS_PATH.exists():
        print(f"\nFATAL: {SEED_RESULTS_PATH} not found. Run scripts/harness/run_seed_stability.py first.")
        sys.exit(1)

    data = json.loads(SEED_RESULTS_PATH.read_text())

    # S1: correct seeds
    actual_seeds = data.get("seeds", [])
    check("S1 correct seeds tested", actual_seeds == EXPECTED_SEEDS,
          f"expected {EXPECTED_SEEDS}, got {actual_seeds}")

    # S2: all 5 results present
    results = data.get("results", [])
    n_results = len(results)
    check("S2 all 5 results present", n_results == 5,
          f"{n_results}/5 results")

    # S3: no errors in any result
    errors = [r for r in results if "error" in r]
    check("S3 no seed errors", len(errors) == 0,
          f"{len(errors)} errors" if errors else "clean")

    # S4: extract Sharpe values
    sharpes = [r.get("sharpe", 0) for r in results if r.get("sharpe", 0) > 0]
    check("S4 all seeds produced Sharpe", len(sharpes) == 5,
          f"{len(sharpes)}/5 valid Sharpe values")

    if len(sharpes) < 2:
        print("\nFATAL: Not enough Sharpe values to compute statistics.")
        sys.exit(1)

    # S5: mean Sharpe
    mean_sharpe = sum(sharpes) / len(sharpes)
    check("S5 mean Sharpe reasonable", mean_sharpe > MIN_SHARPE,
          f"mean={mean_sharpe:.4f}")

    # S6: std Sharpe < 0.1
    variance = sum((s - mean_sharpe) ** 2 for s in sharpes) / len(sharpes)
    std_sharpe = variance ** 0.5
    check("S6 std Sharpe < 0.1", std_sharpe < STD_THRESHOLD,
          f"std={std_sharpe:.4f}, threshold={STD_THRESHOLD}")

    # S7: all seeds > 0.9
    min_sharpe = min(sharpes)
    all_above = all(s > MIN_SHARPE for s in sharpes)
    check("S7 all seeds Sharpe > 0.9", all_above,
          f"min={min_sharpe:.4f}, values={[round(s, 4) for s in sharpes]}")

    # S8: range check (sanity)
    max_sharpe = max(sharpes)
    range_val = max_sharpe - min_sharpe
    check("S8 range < 0.3 (sanity)", range_val < 0.3,
          f"range={range_val:.4f} ({min_sharpe:.4f} to {max_sharpe:.4f})")

    # S9: per-seed result files exist in experiments/results/
    seed_files_found = 0
    for seed in EXPECTED_SEEDS:
        pattern = f"*seed_{seed}.json"
        matches = list(RESULTS_DIR.glob(pattern))
        if matches:
            seed_files_found += 1
    check("S9 per-seed result files saved", seed_files_found == 5,
          f"{seed_files_found}/5 files in experiments/results/")

    # S10: cross-check mean_sharpe from file
    file_mean = data.get("mean_sharpe", 0)
    check("S10 mean Sharpe cross-check", abs(file_mean - mean_sharpe) < 0.001,
          f"file={file_mean:.4f}, computed={mean_sharpe:.4f}")

    # S11: cross-check std from file
    file_std = data.get("std_sharpe", 0)
    check("S11 std Sharpe cross-check", abs(file_std - std_sharpe) < 0.001,
          f"file={file_std:.4f}, computed={std_sharpe:.4f}")

    # S12: verdict field
    verdict = data.get("verdict", "")
    check("S12 verdict is PASS", verdict == "PASS",
          f"verdict={verdict}")

    # S13: per-seed trade counts reasonable (no degenerate runs)
    trade_counts = [r.get("trades", 0) for r in results if "trades" in r]
    min_trades = min(trade_counts) if trade_counts else 0
    check("S13 all seeds have trades", min_trades > 100,
          f"min trades={min_trades}, counts={trade_counts}")

    print()
    print(f"Results: {passed}/{passed + failed} PASS")

    if failed > 0:
        print("DRILL FAILED")
        sys.exit(1)
    else:
        print("DRILL PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
