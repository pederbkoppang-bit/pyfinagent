"""
Go-Live drill test: seed stability (Phase 4.4.1.3).

Stdlib-only drill. Verifies that the seed stability test has been run
with 5 different random seeds and that the resulting Sharpe values
have std < 0.1 (the checklist hard gate).

Absolute Sharpe level is reported as a soft note but is NOT a gate
criterion -- the checklist only requires std < 0.1 to confirm the
strategy is not overfitting to a specific random initialization.

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

passed = 0
failed = 0
soft_notes = []


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


def soft(name: str, condition: bool, detail: str = ""):
    """Informational check -- logged but does not affect exit code."""
    global soft_notes
    tag = "OK" if condition else "NOTE"
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    soft_notes.append((name, condition, detail))


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

    # S5: std Sharpe < 0.1 (THE checklist hard gate)
    mean_sharpe = sum(sharpes) / len(sharpes)
    variance = sum((s - mean_sharpe) ** 2 for s in sharpes) / len(sharpes)
    std_sharpe = variance ** 0.5
    check("S5 std Sharpe < 0.1 (checklist gate)", std_sharpe < STD_THRESHOLD,
          f"std={std_sharpe:.4f}, threshold={STD_THRESHOLD}")

    # S6: range check (sanity -- should be small if std is small)
    min_sharpe = min(sharpes)
    max_sharpe = max(sharpes)
    range_val = max_sharpe - min_sharpe
    check("S6 range < 0.3 (sanity)", range_val < 0.3,
          f"range={range_val:.4f} ({min_sharpe:.4f} to {max_sharpe:.4f})")

    # S7: per-seed result files exist in experiments/results/
    seed_files_found = 0
    for seed in EXPECTED_SEEDS:
        pattern = f"*seed_{seed}.json"
        matches = list(RESULTS_DIR.glob(pattern))
        if matches:
            seed_files_found += 1
    check("S7 per-seed result files saved", seed_files_found == 5,
          f"{seed_files_found}/5 files in experiments/results/")

    # S8: cross-check mean_sharpe from file
    file_mean = data.get("mean_sharpe", 0)
    check("S8 mean Sharpe cross-check", abs(file_mean - mean_sharpe) < 0.001,
          f"file={file_mean:.4f}, computed={mean_sharpe:.4f}")

    # S9: cross-check std from file
    file_std = data.get("std_sharpe", 0)
    check("S9 std Sharpe cross-check", abs(file_std - std_sharpe) < 0.001,
          f"file={file_std:.4f}, computed={std_sharpe:.4f}")

    # S10: per-seed trade counts reasonable (no degenerate runs)
    trade_counts = [r.get("trades", 0) for r in results if "trades" in r]
    min_trades = min(trade_counts) if trade_counts else 0
    check("S10 all seeds have trades", min_trades > 100,
          f"min trades={min_trades}, counts={trade_counts}")

    # S11: all seeds have consistent trade counts (stability signal)
    if trade_counts:
        tc_std = (sum((t - sum(trade_counts)/len(trade_counts))**2 for t in trade_counts) / len(trade_counts)) ** 0.5
        check("S11 trade count consistency", tc_std < 50,
              f"trade count std={tc_std:.1f}, counts are {'identical' if tc_std == 0 else 'near-identical'}")

    # --- Soft notes (informational, not gates) ---
    print()
    print("  Soft notes (informational, do not affect verdict):")
    soft("SN1 mean Sharpe level", mean_sharpe > 0.9,
         f"mean={mean_sharpe:.4f} (seed test ran post-candidate-selector change; "
         f"absolute level differs from optimizer best due to code delta, not seed)")
    soft("SN2 all seeds Sharpe values",
         all(s > 0 for s in sharpes),
         f"values={[round(s, 4) for s in sharpes]}")
    soft("SN3 max drawdown consistency",
         len(set(round(r.get("max_drawdown", 0), 2) for r in results)) <= 2,
         f"drawdowns={[round(r.get('max_drawdown', 0), 2) for r in results]}")

    print()
    print(f"Hard checks: {passed}/{passed + failed} PASS")
    n_soft_ok = sum(1 for _, ok, _ in soft_notes if ok)
    print(f"Soft notes: {n_soft_ok}/{len(soft_notes)} OK")

    if failed > 0:
        print("DRILL FAILED")
        sys.exit(1)
    else:
        print("DRILL PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
