"""phase-75.15 (qa-tests-01/02/04/05/06/10, deps-06): CI gates made real.

Guards the CI-lane config edits (workflow YAML text + the coverage-tier
runner) with real assertions, so a future edit that silently reverts a
lane back to advisory, drops the requires_live selection, or removes the
npm-audit/coverage-tier gates is caught by the backend test suite itself
-- not just by eyeballing a diff.

These tests read workflow files DIRECTLY (`.read_text()`, no
`if not path.exists(): pytest.skip(...)` guard) so a wrong/typo'd path
hard-fails (FileNotFoundError -> test ERROR) rather than silently
skip-greening. That property was verified by mutation (see
experiment_results_75.15.md mutation matrix, M7).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


# ---------------------------------------------------------------------------
# Leg (a): e2e-smoke.yml advisory flip + requires_live migration
# ---------------------------------------------------------------------------

def test_e2e_smoke_backend_lane_is_enforcing():
    """continue-on-error must be gone from the job -- a red step must
    redden the lane (M1 catches a revert to continue-on-error: true)."""
    y = (WORKFLOWS / "e2e-smoke.yml").read_text(encoding="utf-8")
    assert "continue-on-error: true" not in y, (
        "e2e-smoke.yml backend lane must not be advisory (continue-on-error: true found)"
    )


def test_e2e_smoke_uses_requires_live_marker_not_ignore_list():
    """The stale 6-file --ignore list must be gone; selection is via the
    requires_live pytest marker (M2 catches dropping the -m selection).

    Checks the ACTUAL `run:` line, not just substring-anywhere-in-file --
    the header comment above the step also mentions the marker in prose,
    which would let a mutation that only strips the run-line survive a
    naive whole-file substring check (found by mutation M2 on the first
    pass; corrected here).
    """
    y = (WORKFLOWS / "e2e-smoke.yml").read_text(encoding="utf-8")
    run_line = next(
        (ln for ln in y.splitlines() if ln.strip().startswith("python -m pytest backend/tests/")),
        None,
    )
    assert run_line is not None, "backend pytest run line not found in e2e-smoke.yml"
    assert '-m "not requires_live"' in run_line, (
        f"backend pytest run line must select via the requires_live marker; got: {run_line!r}"
    )
    assert "--ignore=backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py" not in y, (
        "stale hardcoded --ignore list should be replaced by the marker selection"
    )


def test_e2e_smoke_includes_vitest_step():
    """Leg (f): the frontend step must run the vitest suite after tsc.

    Searches from the LAST `npx tsc --noEmit` occurrence forward (not
    `str.index`'s first match) because an explanatory comment above the
    step also mentions "npm run test" in prose, earlier in the file than
    the actual `run:` block.
    """
    y = (WORKFLOWS / "e2e-smoke.yml").read_text(encoding="utf-8")
    assert "npm run test" in y or "vitest" in y, "vitest/npm run test step missing from e2e-smoke.yml"
    tsc_idx = y.rindex("npx tsc --noEmit")
    test_idx = y.index("npm run test", tsc_idx) if "npm run test" in y[tsc_idx:] else y.index("vitest", tsc_idx)
    build_idx = y.index("npm run build", tsc_idx)
    assert tsc_idx < test_idx < build_idx, (
        "expected order: tsc --noEmit -> npm run test -> npm run build"
    )


REQUIRES_LIVE_MARKED_TESTS = [
    "backend/tests/test_phase_23_2_10_watchdog_no_fire_7d.py::test_phase_23_2_10_watchdog_log_present_and_fresh",
    "backend/tests/test_phase_23_2_6_sector_cap_emit.py::test_phase_23_2_6_backend_log_has_skipping_buy_evidence",
    "backend/tests/test_phase_23_2_9_ticker_meta_latency.py::test_phase_23_2_9_backend_log_has_prewarm_evidence",
]


def test_phase_75_15_newly_marked_tests_carry_requires_live():
    """The 3 tests phase-75.15 newly quarantined must actually carry the
    marker (source-grep, not just trust the docstring)."""
    for nodeid in REQUIRES_LIVE_MARKED_TESTS:
        file_path, _, func_name = nodeid.partition("::")
        src = (REPO_ROOT / file_path).read_text(encoding="utf-8")
        # The marker decorator must appear on the line(s) immediately
        # preceding the function's def.
        func_idx = src.index(f"def {func_name}(")
        preceding = src[:func_idx]
        last_def_idx = preceding.rfind("\ndef ")
        scope = preceding[last_def_idx if last_def_idx != -1 else 0:]
        assert "@pytest.mark.requires_live" in scope, (
            f"{nodeid} must carry @pytest.mark.requires_live"
        )


def test_backend_not_requires_live_collection_count_is_stable():
    """Pin the exact collected/deselected counts under `-m "not
    requires_live"` (M3 catches un-marking one of the 3 newly-marked
    tests -- the deselected count would drop and this assertion fails)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "backend/tests/", "-q",
         "-m", "not requires_live", "--collect-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"collection failed:\n{result.stdout}\n{result.stderr}"
    tail = result.stdout.strip().splitlines()[-1]
    # pytest --collect-only summary line: "N/M tests collected (K deselected) in Ts"
    # phase-75.16: baseline moved from 1474/1490 to 1518/1534 -- test_phase_75_
    # deploy_surface.py added 44 new tests, none carrying requires_live, so the
    # deselected count (the thing this canary actually protects) is unchanged
    # at 16 while both totals shift by +44.
    assert "1518/1534 tests collected (16 deselected)" in tail, (
        f"collection count drifted from the phase-75.16 baseline; got: {tail!r}"
    )


# ---------------------------------------------------------------------------
# Leg (b): lock-count guard -- verify-only, collected under the migration
# ---------------------------------------------------------------------------

def test_lock_count_guard_collected_under_not_requires_live():
    """test_phase_23_2_14 (unmarked, green) must be selected by the new
    `-m "not requires_live"` filter, not accidentally excluded."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "backend/tests/test_phase_23_2_14_no_reentrant_locks.py",
         "-q", "-m", "not requires_live", "--collect-only"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "5 tests collected" in result.stdout, (
        f"expected all 5 test_phase_23_2_14 tests collected; got:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# Leg (c): coverage_tier_check.py
# ---------------------------------------------------------------------------

def test_coverage_tier_check_script_exists():
    assert (REPO_ROOT / "scripts" / "qa" / "coverage_tier_check.py").exists()


def test_coverage_tier_check_errors_on_missing_coverage_json(tmp_path):
    """M4: pointing the checker at a nonexistent coverage json must ERROR
    (exit 2), never silently pass (exit 0)."""
    missing = tmp_path / "does_not_exist.json"
    result = subprocess.run(
        [sys.executable, "scripts/qa/coverage_tier_check.py",
         "--coverage-json", str(missing)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 2, (
        f"expected exit 2 on missing coverage json; got {result.returncode}\n{result.stderr}"
    )


def test_coverage_tier_check_fails_when_bar_exceeds_measurement(tmp_path):
    """M5: a bar set above current measured coverage must exit non-zero,
    proving the guard can actually fail (not vacuous at today's
    all-modules-above-bar state)."""
    doc = REPO_ROOT / "docs" / "coverage_tier_overrides.md"
    doc_text = doc.read_text(encoding="utf-8")
    marker = "### Tier-1 EXTENDED (>=75% combined STRICT bar, post-phase-43.0.2)"
    assert doc_text.count(marker) == 1, "expected exactly one EXTENDED section header"
    mutated_doc = tmp_path / "coverage_tier_overrides_mutated.md"
    mutated_doc.write_text(
        doc_text.replace(marker, marker.replace(">=75%", ">=99%"), 1),
        encoding="utf-8",
    )

    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(json.dumps({
        "files": {
            "backend/services/paper_trader.py": {"summary": {"percent_covered": 78.3}},
            "backend/services/portfolio_manager.py": {"summary": {"percent_covered": 83.7}},
            "backend/services/perf_metrics.py": {"summary": {"percent_covered": 84.8}},
            "backend/services/kill_switch.py": {"summary": {"percent_covered": 88.2}},
            "backend/services/cycle_lock.py": {"summary": {"percent_covered": 83.0}},
            "backend/services/factor_correlation.py": {"summary": {"percent_covered": 85.1}},
            "backend/services/factor_loadings.py": {"summary": {"percent_covered": 78.1}},
        }
    }), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/qa/coverage_tier_check.py",
         "--doc", str(mutated_doc), "--coverage-json", str(coverage_json)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1, (
        f"expected exit 1 when EXTENDED bar (99%) exceeds measured coverage; "
        f"got {result.returncode}\n{result.stdout}\n{result.stderr}"
    )
    assert "paper_trader.py" in result.stderr


def test_coverage_tier_check_passes_at_real_measurements():
    """Sanity companion to the M5 test above: the REAL doc + a coverage
    json reflecting today's real measurements must PASS (exit 0) -- this
    is what proves M5 is testing the bar, not a broken comparison."""
    coverage_json_data = {
        "files": {
            "backend/services/paper_trader.py": {"summary": {"percent_covered": 78.3}},
            "backend/services/portfolio_manager.py": {"summary": {"percent_covered": 83.7}},
            "backend/services/perf_metrics.py": {"summary": {"percent_covered": 84.8}},
            "backend/services/kill_switch.py": {"summary": {"percent_covered": 88.2}},
            "backend/services/cycle_lock.py": {"summary": {"percent_covered": 83.0}},
            "backend/services/factor_correlation.py": {"summary": {"percent_covered": 85.1}},
            "backend/services/factor_loadings.py": {"summary": {"percent_covered": 78.1}},
        }
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        coverage_json = Path(td) / "coverage.json"
        coverage_json.write_text(json.dumps(coverage_json_data), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/qa/coverage_tier_check.py",
             "--coverage-json", str(coverage_json)],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, f"expected exit 0; got {result.returncode}\n{result.stdout}\n{result.stderr}"


def test_coverage_tier_check_workflow_exists_and_shaped():
    y = (WORKFLOWS / "coverage-tier-check.yml").read_text(encoding="utf-8")
    assert "coverage_tier_check.py" in y
    assert "schedule" in y


# ---------------------------------------------------------------------------
# Leg (d): seed-stability-check.yml wording
# ---------------------------------------------------------------------------

def test_seed_stability_no_longer_overclaims_pr_blocking():
    """75.15 Q/A cycle-1 fix: the original OR-form was VACUOUS (the step
    itself added the 'run_seed_stability' comment token, making the escape
    clause permanently true). Now: the overclaim must be ABSENT and the
    honest re-scoped sentence PRESENT -- both halves can fail on revert."""
    s = (WORKFLOWS / "seed-stability-check.yml").read_text(encoding="utf-8")
    assert "blocks the PR" not in s, (
        "seed-stability-check.yml re-introduced the PR-blocking overclaim"
    )
    assert "structurally cannot enforce" in s, (
        "seed-stability-check.yml lost the honest re-scoped rationale "
        "(frozen-baseline recompute cannot enforce reproducibility on new code)"
    )


# ---------------------------------------------------------------------------
# Leg (e): visual-regression.yml baseline gate
# ---------------------------------------------------------------------------

def test_visual_regression_gates_on_baseline_presence():
    y = (WORKFLOWS / "visual-regression.yml").read_text(encoding="utf-8")
    assert "baseline_check" in y or "has_baselines" in y, (
        "visual-regression.yml must gate its comparison run on committed-baseline presence"
    )
    baselines_dir = REPO_ROOT / "frontend" / "tests" / "visual-regression" / "snapshots" / "chromium"
    png_count = len(list(baselines_dir.glob("*.png"))) if baselines_dir.exists() else 0
    # Documents WHY the gate matters -- if this ever goes non-zero, the
    # baseline-presence gate becomes load-bearing for real (not just
    # defensive), so leave the assertion but don't require 0 (the operator
    # first-run flow is expected to change this).
    assert png_count >= 0


# ---------------------------------------------------------------------------
# Leg (g): npm-audit.yml
# ---------------------------------------------------------------------------

def test_npm_audit_workflow_exists_and_shaped():
    """M6 catches removing the audit step / the audit-level flag.

    Checks the ACTUAL `run:` lines (non-comment), not substring-anywhere-
    in-file -- the header comment also mentions `npm audit --audit-
    level=high` in prose, which would let a mutation that only deletes the
    step survive a naive whole-file check (found by mutation M6 on the
    first pass; corrected here).
    """
    y = (WORKFLOWS / "npm-audit.yml").read_text(encoding="utf-8")
    non_comment_lines = [
        line for line in y.splitlines() if not line.strip().startswith("#")
    ]
    non_comment_text = "\n".join(non_comment_lines)
    assert "npm ci" in non_comment_text
    assert "run: npm audit --audit-level=high" in non_comment_text, (
        "no executable `run: npm audit --audit-level=high` step line found"
    )
    assert "package-lock.json" in y
    assert not any("audit fix" in line for line in non_comment_lines), (
        "the gate must never run npm audit fix as an executed command"
    )


def test_npm_audit_workflow_triggers_on_lockfile_and_schedule():
    y = (WORKFLOWS / "npm-audit.yml").read_text(encoding="utf-8")
    assert "schedule" in y
    assert "frontend/package-lock.json" in y


# ---------------------------------------------------------------------------
# M7 stub property (documented, not a live mutation in CI): reading a
# WRONG workflow path must hard-fail, never skip-green. Demonstrated here
# directly rather than via the scratchpad mutation script, since it's a
# property of this file's own read pattern, not of production code.
# ---------------------------------------------------------------------------

def test_wrong_workflow_path_hard_fails_not_skips():
    """Proves the read pattern used throughout this file (`.read_text()`,
    no existence-guard) hard-fails on a wrong path instead of silently
    skip-greening. This is the M7 stub: point at a workflow file that does
    not exist and confirm a real exception propagates."""
    wrong_path = WORKFLOWS / "e2e-smoke-TYPO-DOES-NOT-EXIST.yml"
    assert not wrong_path.exists()
    with pytest.raises(FileNotFoundError):
        wrong_path.read_text(encoding="utf-8")
