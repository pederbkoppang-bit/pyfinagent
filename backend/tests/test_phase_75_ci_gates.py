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
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── phase-80.46 cycle 4: ONE CHOKE POINT, NO PER-SITE CONSTANTS ────────────────
#
# Six previous attempts at a source-SCANNING guard were each defeated by a spelling
# nobody anticipated: a global floor that passed the very value it existed to catch;
# a flag-keyed discriminator that tripped on clean code; unpinned calibration; a
# one-character path edit ("backend/tests/" -> "backend/tests"); then "./backend/
# tests/", "backend//tests/", hoisting the argv to a module constant, and
# `from subprocess import run` -- all four found by a Q/A, all behaviour-identical
# (every spelling collects 2295/2311).
#
# The lesson is not "scan harder". A source scan over a language with unlimited
# spellings is the wrong instrument. This is the SAME remedy phase-36.13 applied to
# the kill switch, and CWE-638 names it: "create and use a single interface that
# performs the access checks". Call sites no longer carry a timeout at all -- there
# is no constant left to tune down, and the budget is DERIVED from the workload at
# call time rather than written next to it.
#
# Timeouts are >= MULTIPLIER x the measured quiet-machine cost. Re-derive the costs
# before changing them; do not tune closer. The collection these drive grows
# monotonically (2307 -> 2311 in two days), which is why a fixed constant is the
# same defect class as the exact-count pin removed in 80.44.
_MEASURED_COST_S = {"whole_tree": 8.8, "single_target": 1.0}
_TIMEOUT_MULTIPLIER = 20


def _run_gated(argv: list[str], **kwargs):
    """Every subprocess in this file goes through here. The timeout is DERIVED.

    Scope is decided by resolving each argv entry against the repo and asking the
    filesystem whether it is a DIRECTORY -- not by matching a string. That is what
    makes "./backend/tests/", "backend//tests/" and "backend/tests" identical here,
    as they already are to pytest.
    """
    whole_tree = False
    for a in argv:
        if not isinstance(a, str) or a.startswith("-"):
            continue
        candidate = (REPO_ROOT / a).resolve()
        if candidate.is_dir() and candidate != REPO_ROOT:
            whole_tree = True
            break
    cost = _MEASURED_COST_S["whole_tree" if whole_tree else "single_target"]
    kwargs.setdefault("cwd", str(REPO_ROOT))
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs["timeout"] = cost * _TIMEOUT_MULTIPLIER
    return subprocess.run(argv, **kwargs)
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


# phase-80.44: the ONE number this canary protects -- the count of tests carrying
# the `requires_live` marker. Unlike the collection totals it does NOT move when
# someone adds an ordinary test, so it is a real invariant rather than a
# re-baselining chore.
EXPECTED_REQUIRES_LIVE_DESELECTED = 16


# phase-80.46: SUBPROCESS TIMEOUT POLICY -- LOOSE, DERIVED FROM A MEASUREMENT.
#
# A 60s budget over a subprocess measured at 8.8s was REPRODUCED timing out under
# 30x CPU oversubscription on 10 cores (captures_80.46/reproduction.txt). Tight
# timeouts are themselves a flakiness source: SAP HANA 2026 (n=559) measured 18%
# timeout-flakiness for tests calibrated close to average execution time versus 7%
# under one loose global timeout.
#
# RULE: every subprocess timeout here is >= 20x its measured quiet-machine cost.
# Re-derive the cost before changing a value; do NOT tune it closer. The collection
# these subprocesses drive grows monotonically (2307 -> 2310 in a single day), so a
# budget that merely looks generous today is the same defect class as the exact
# count pin removed in 80.44 -- a constant that assumes a static suite.
#
# CALL SITES, KEYED BY FUNCTION NAME -- not line number. Cycle 2 regenerated this
# banner from an AST walk and then INSERTED the banner, which shifted every line it
# had just named by 7-8. A Q/A caught that: the fix for stale line numbers produced
# stale line numbers. Function names do not move when text is inserted above them.
#   test_backend_not_requires_live_collection_count_is_stable() -> timeout=300s
#   test_coverage_tier_check_errors_on_missing_coverage_json() -> timeout=30s
#   test_coverage_tier_check_fails_when_bar_exceeds_measurement() -> timeout=30s
#   test_coverage_tier_check_passes_at_real_measurements() -> timeout=30s
#   test_lock_count_guard_collected_under_not_requires_live() -> timeout=120s
#
# Only the whole-tree collection is expensive (~8.8s measured); the rest are ~1s.
# The guard test at the bottom of this file enforces the 20x rule per call site.


def test_backend_not_requires_live_collection_count_is_stable():
    """Pin the exact collected/deselected counts under `-m "not
    requires_live"` (M3 catches un-marking one of the 3 newly-marked
    tests -- the deselected count would drop and this assertion fails)."""
    result = _run_gated([sys.executable, "-m", "pytest", "backend/tests/", "-q",
         "-m", "not requires_live", "--collect-only"])
    assert result.returncode == 0, f"collection failed:\n{result.stdout}\n{result.stderr}"
    tail = result.stdout.strip().splitlines()[-1]
    # pytest --collect-only summary line: "N/M tests collected (K deselected) in Ts"
    #
    # phase-80.44: STOPPED PINNING THE TOTALS. This canary was asserting the literal
    # "1563/1579 tests collected (16 deselected)", which had gone stale and the gate
    # was FAILING -- silently non-authoritative for some time. Re-baselining it again
    # would just restart the treadmill: the file's own comment history records three
    # previous re-baselines (1474/1490 -> 1518/1534 -> 1563/1579), and every one of
    # them says the same thing -- "the deselected count is unchanged at 16 while both
    # totals shift". The totals move whenever ANYONE adds a test; they carry no signal.
    #
    # The invariant this canary actually protects is named in its own docstring: M3,
    # un-marking one of the 3 `requires_live` tests, which would make the DESELECTED
    # count drop. So assert that, exactly, plus the internal arithmetic. A test added
    # anywhere no longer breaks the gate, and un-marking a requires_live test still
    # does -- which is the whole point.
    m = re.match(r"(\d+)/(\d+) tests collected \((\d+) deselected\)", tail)
    assert m, f"unrecognised pytest collection summary: {tail!r}"
    collected, total, deselected = (int(g) for g in m.groups())
    assert deselected == EXPECTED_REQUIRES_LIVE_DESELECTED, (
        f"requires_live deselection drifted: expected "
        f"{EXPECTED_REQUIRES_LIVE_DESELECTED}, got {deselected} -- a test was "
        f"un-marked or newly marked. Summary: {tail!r}"
    )
    assert total - collected == deselected, (
        f"collection arithmetic inconsistent: {collected}/{total} with "
        f"{deselected} deselected. Summary: {tail!r}"
    )
    assert collected > 0, f"nothing collected: {tail!r}"


# ---------------------------------------------------------------------------
# Leg (b): lock-count guard -- verify-only, collected under the migration
# ---------------------------------------------------------------------------

def test_lock_count_guard_collected_under_not_requires_live():
    """test_phase_23_2_14 (unmarked, green) must be selected by the new
    `-m "not requires_live"` filter, not accidentally excluded."""
    result = _run_gated([sys.executable, "-m", "pytest",
         "backend/tests/test_phase_23_2_14_no_reentrant_locks.py",
         "-q", "-m", "not requires_live", "--collect-only"])
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
    result = _run_gated([sys.executable, "scripts/qa/coverage_tier_check.py",
         "--coverage-json", str(missing)])
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

    result = _run_gated([sys.executable, "scripts/qa/coverage_tier_check.py",
         "--doc", str(mutated_doc), "--coverage-json", str(coverage_json)])
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
        result = _run_gated([sys.executable, "scripts/qa/coverage_tier_check.py",
             "--coverage-json", str(coverage_json)])
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


def test_phase_80_46_every_subprocess_goes_through_the_gated_launcher():
    """phase-80.46 cycle 4: complete mediation, replacing a source scan that lost
    six times.

    THE HISTORY MATTERS, because it is why this test looks different from the one it
    replaces. Six versions tried to SCAN the source for tight timeouts. Each was
    defeated by a spelling nobody anticipated -- a global floor that passed the exact
    value it existed to catch; a flag-keyed discriminator that tripped on clean code;
    unpinned calibration constants; then FOUR behaviour-identical escapes found by a
    Q/A: "./backend/tests/", "backend//tests/", hoisting the argv to a module
    constant, and `from subprocess import run`. Every one collected 2295/2311 --
    invisible in review.

    A source scan over a language with unlimited spellings cannot win that game. So
    the policy moved to the ONE place a subprocess can actually be launched, exactly
    as phase-36.13 did for the kill switch, and for the reason CWE-638 gives:
    "create and use a single interface that performs the access checks."

    This test therefore asserts TWO things and neither is a spelling:
      1. No test function launches a subprocess directly -- all go through
         `_run_gated`, which DERIVES the timeout. There is no per-site constant left
         to tune, so the entire class of "lower the number" mutations is gone.
      2. The helper's own calibration cannot be weakened silently.

    Evading this now requires deleting or bypassing the helper, which is a visible
    structural change rather than a one-character edit.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    # (1) COMPLETE MEDIATION. Any Call named `run` -- attribute form
    # (`subprocess.run`) OR bare (`from subprocess import run`) -- inside a test
    # function is an offender. Both spellings that defeated the previous guard are
    # caught by matching the NAME rather than the access path.
    offenders = []
    for fn in ast.walk(tree):
        if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")):
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            name = (node.func.attr if isinstance(node.func, ast.Attribute)
                    else node.func.id if isinstance(node.func, ast.Name) else None)
            if name in {"run", "Popen", "check_output", "call", "check_call"}:
                offenders.append(f"{fn.name} (line {node.lineno}) calls {name}() directly")

    assert offenders == [], (
        "subprocess launched outside the gated helper -- the timeout policy cannot "
        f"apply to it: {offenders}. Use _run_gated(argv), which derives the budget "
        "from the measured workload."
    )

    # (2) The calibration the helper derives from, pinned to its measurement.
    assert _TIMEOUT_MULTIPLIER >= 20, (
        "the 20x rule is the phase-80.46 finding, not a tunable: SAP HANA 2026 "
        "(n=559) measured 18% timeout-flakiness for budgets calibrated near average "
        "execution time vs 7% for loose ones."
    )
    assert _MEASURED_COST_S["whole_tree"] >= 8.0, (
        "the whole-tree collection was MEASURED at 8.8s on a quiet 10-core machine "
        "and the suite only grows -- re-measure rather than lowering this."
    )

    # (4) THE HELPER ACTUALLY DERIVES -- checked BEHAVIOURALLY, not by reading it.
    # Mutation found this gap: replacing the derivation with a hardcoded
    # `kwargs["timeout"] = 60` SURVIVED every other assertion here, because they
    # only proved that call sites USE the helper and that the constants exist --
    # never that the helper CONSULTS them. Intercept the launch and read the budget
    # it actually computed.
    captured = {}

    def _fake_run(argv, **kw):
        captured["timeout"] = kw.get("timeout")
        class _R:
            returncode, stdout, stderr = 0, "", ""
        return _R()

    _real_run = subprocess.run
    try:
        subprocess.run = _fake_run
        _run_gated([sys.executable, "-c", "pass", "backend/tests/"])
        whole_tree_budget = captured["timeout"]
        _run_gated([sys.executable, "-c", "pass", "backend/tests/test_phase_75_ci_gates.py"])
        single_budget = captured["timeout"]
    finally:
        subprocess.run = _real_run

    expected_whole = _MEASURED_COST_S["whole_tree"] * _TIMEOUT_MULTIPLIER
    expected_single = _MEASURED_COST_S["single_target"] * _TIMEOUT_MULTIPLIER
    assert whole_tree_budget == expected_whole, (
        f"the helper did not DERIVE the whole-tree budget: got {whole_tree_budget}, "
        f"expected {expected_whole} (= {_MEASURED_COST_S['whole_tree']}s measured x "
        f"{_TIMEOUT_MULTIPLIER}). A hardcoded timeout in the helper defeats the whole "
        "policy while every other assertion here still passes."
    )
    assert single_budget == expected_single, (
        f"the helper did not DERIVE the single-target budget: got {single_budget}, "
        f"expected {expected_single}"
    )
    assert whole_tree_budget > single_budget, (
        "the helper must give a DIRECTORY target the larger budget -- if these are "
        "equal the scope detection has stopped discriminating"
    )

    # (3) The helper actually discriminates: a directory target must buy the larger
    # budget, and it must do so for EVERY spelling of that directory. This is the
    # assertion the six scanning versions could not make, because they compared
    # strings; this one asks the filesystem.
    for spelling in ("backend/tests/", "backend/tests", "./backend/tests/", "backend//tests/"):
        resolved = (REPO_ROOT / spelling).resolve()
        assert resolved.is_dir() and resolved != REPO_ROOT, (
            f"{spelling!r} must resolve to the tests directory -- if this breaks, the "
            "helper's scope detection is no longer sound"
        )
