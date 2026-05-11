"""phase-23.8.2 verifier — 10-claim source-level assertion.

Run: source .venv/bin/activate && python3 tests/verify_phase_23_8_2.py

Exits 0 on PASS, 1 on FAIL.

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md R-2 Option A
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def _check(name: str, ok: bool, detail: str = ""):
    flag = "PASS" if ok else "FAIL"
    RESULTS.append((flag, name, detail))


def _grep_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _absent_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle not in path.read_text(encoding="utf-8")


def main() -> int:
    settings_path = REPO / ".claude" / "settings.json"
    project_md = REPO / ".claude" / "context" / "project.md"
    runbook = REPO / "docs" / "runbooks" / "per-step-protocol.md"
    harness_log = REPO / "handoff" / "harness_log.md"

    # 1. settings.json valid JSON
    try:
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        settings_ok = True
    except Exception as e:
        settings = {}
        settings_ok = False
    _check(
        "1. settings_json_valid",
        settings_ok,
        ".claude/settings.json must be valid JSON",
    )

    # 2. TaskCompleted key removed
    hooks = settings.get("hooks", {})
    _check(
        "2. task_completed_hook_block_removed",
        "TaskCompleted" not in hooks,
        f"'TaskCompleted' must not be in hooks, found: {list(hooks.keys())}",
    )

    # 3. Other expected hook keys intact (regression check)
    expected_hooks = {
        "PreToolUse", "ConfigChange", "InstructionsLoaded",
        "PostToolUse", "TeammateIdle", "Stop", "SubagentStop",
    }
    missing = expected_hooks - set(hooks.keys())
    _check(
        "3. other_hook_keys_intact",
        len(missing) == 0,
        f"Missing expected hooks: {sorted(missing)}" if missing else "",
    )

    # 4. project.md no longer lists "TaskCompleted gate,"
    _check(
        "4. project_md_no_longer_lists_task_completed_gate",
        _absent_in_file(project_md, "TaskCompleted gate,"),
        ".claude/context/project.md must not contain 'TaskCompleted gate,'",
    )

    # 5. per-step-protocol.md no longer contains the old line 226 prose
    _check(
        "5. per_step_protocol_old_line_226_prose_removed",
        _absent_in_file(runbook, "The TaskCompleted hook should fire"),
        "docs/runbooks/per-step-protocol.md must not contain 'The TaskCompleted hook should fire'",
    )

    # 6. per-step-protocol.md no longer contains the old line 248 prose
    _check(
        "6. per_step_protocol_old_line_248_prose_removed",
        _absent_in_file(runbook, "TaskCompleted hook is load-bearing"),
        "docs/runbooks/per-step-protocol.md must not contain 'TaskCompleted hook is load-bearing'",
    )

    # 7. per-step-protocol.md DOES contain the retirement note
    _check(
        "7. per_step_protocol_has_retirement_note",
        _grep_in_file(runbook, "retired in phase-23.8.2"),
        "docs/runbooks/per-step-protocol.md must contain 'retired in phase-23.8.2'",
    )

    # 8. EXPECTED-FAIL test: the historical step 2.13 assertion now fails
    # (we run the exact assertion from masterplan.json:214 and confirm it raises)
    try:
        # This is the verbatim historical assertion. If TaskCompleted is gone,
        # this assert raises AssertionError. That is the CORRECT post-cycle
        # behavior — the audit explicitly supersedes the historical check.
        with open(settings_path, encoding="utf-8") as f:
            s = json.load(f)
        assert 'TaskCompleted' in s['hooks']
        # If we get here, the hook is STILL present — which means our delete failed.
        historical_assertion_now_fails = False
    except AssertionError:
        historical_assertion_now_fails = True
    _check(
        "8. step_2_13_historical_assertion_now_expectedly_fails",
        historical_assertion_now_fails,
        "The audit explicitly supersedes the step 2.13 historical TaskCompleted assertion (R-2 Option A); this test asserts the assertion now raises",
    )

    # 9. harness_log Cycle 39 documents the 2.13 breakage
    _check(
        "9. harness_log_has_2_13_breakage_disclosure",
        (
            _grep_in_file(harness_log, "phase=23.8.2")
            and _grep_in_file(harness_log, "step 2.13")
            and _grep_in_file(harness_log, "immutability")
            and _grep_in_file(harness_log, "H-2")
        ),
        "handoff/harness_log.md must document the controlled step 2.13 breakage in the phase=23.8.2 cycle entry",
    )

    # 10. bash -n on all remaining 9 hooks
    hooks_dir = REPO / ".claude" / "hooks"
    all_hooks_ok = True
    failed_hooks: list[str] = []
    for h in sorted(hooks_dir.glob("*.sh")):
        try:
            r = subprocess.run(
                ["bash", "-n", str(h)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                all_hooks_ok = False
                failed_hooks.append(h.name)
        except Exception:
            all_hooks_ok = False
            failed_hooks.append(h.name)
    _check(
        "10. no_regressions_other_hooks_bash_syntax_valid",
        all_hooks_ok,
        f"failed: {failed_hooks}" if failed_hooks else "",
    )

    # Print results
    print("=== phase-23.8.2 verifier ===")
    n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
    for flag, name, detail in RESULTS:
        if flag == "PASS":
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {detail}")
    total = len(RESULTS)
    print(f"{'PASS' if n_pass == total else 'FAIL'} ({n_pass}/{total}) EXIT={'0' if n_pass == total else '1'}")
    return 0 if n_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
