"""phase-23.8.1 verifier — 10-claim source-level + behavioral assertion.

Run: source .venv/bin/activate && python3 tests/verify_phase_23_8_1.py

Exits 0 on PASS, 1 on FAIL. Each criterion is a single source grep
or a synthetic run of the gate helper.

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md R-1
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def _check(name: str, ok: bool, detail: str = ""):
    flag = "PASS" if ok else "FAIL"
    RESULTS.append((flag, name, detail))
    return ok


def _grep_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def main() -> int:
    hook = REPO / ".claude" / "hooks" / "auto-commit-and-push.sh"
    gate_lib = REPO / ".claude" / "hooks" / "lib" / "live_check_gate.py"
    cmd = REPO / "CLAUDE.md"
    mp = REPO / ".claude" / "masterplan.json"

    # 1. Hook contains the gate invocation + WARN message
    _check(
        "1. hook_contains_live_check_gate_logic",
        (
            _grep_in_file(hook, "live_check_gate.py")
            and _grep_in_file(hook, "auto-push skipped")
            and _grep_in_file(hook, "GATE_DECISION")
        ),
        "auto-commit-and-push.sh must invoke live_check_gate.py and contain the WARN message",
    )

    # 2. Hook bash syntax valid
    try:
        r = subprocess.run(
            ["bash", "-n", str(hook)],
            capture_output=True, text=True, timeout=10,
        )
        bash_ok = (r.returncode == 0)
    except Exception:
        bash_ok = False
    _check(
        "2. hook_bash_syntax_valid",
        bash_ok,
        "bash -n must accept the hook",
    )

    # 3. Gate helper python syntax valid (ast.parse)
    helper_ok = gate_lib.exists()
    if helper_ok:
        try:
            import ast
            ast.parse(gate_lib.read_text(encoding="utf-8"))
            helper_syntax_ok = True
        except SyntaxError:
            helper_syntax_ok = False
    else:
        helper_syntax_ok = False
    _check(
        "3. hook_python_heredoc_ast_parses",
        helper_ok and helper_syntax_ok,
        ".claude/hooks/lib/live_check_gate.py must exist and parse",
    )

    # Import the gate function for behavioral tests
    sys.path.insert(0, str(REPO / ".claude" / "hooks"))
    try:
        from lib.live_check_gate import gate_decision
        import_ok = True
    except Exception as e:
        import_ok = False
        gate_decision = None
        print(f"  [debug] import failed: {e}")

    if not import_ok:
        for n in (4, 5, 6):
            _check(
                f"{n}. behavioral_test_skipped_gate_import_failed",
                False,
                "gate_decision could not be imported",
            )
    else:
        # 4. Backward compat: no live_check -> proceed
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mp_file = tmp / "masterplan.json"
            handoff = tmp / "handoff" / "current"
            handoff.mkdir(parents=True, exist_ok=True)
            mp_file.write_text(json.dumps({
                "phases": [{"id": "phase-test", "steps": [
                    {"id": "test.1", "status": "done", "verification": {
                        "command": "true",
                        "success_criteria": ["x"],
                    }},
                ]}],
            }))
            decision = gate_decision(str(mp_file), "test.1", str(handoff))
            _check(
                "4. backward_compat_no_live_check_proceeds",
                decision == "proceed",
                f"expected 'proceed', got '{decision}'",
            )

        # 5. Gate fires: live_check set + artifact missing -> skip
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mp_file = tmp / "masterplan.json"
            handoff = tmp / "handoff" / "current"
            handoff.mkdir(parents=True, exist_ok=True)
            mp_file.write_text(json.dumps({
                "phases": [{"id": "phase-test", "steps": [
                    {"id": "test.2", "status": "done", "verification": {
                        "command": "true",
                        "success_criteria": ["x"],
                        "live_check": "synthetic evidence required",
                    }},
                ]}],
            }))
            decision = gate_decision(str(mp_file), "test.2", str(handoff))
            _check(
                "5. gate_fires_when_required_skips_push",
                decision == "skip",
                f"expected 'skip', got '{decision}'",
            )

        # 6. Gate passes: live_check set + artifact present -> passed
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mp_file = tmp / "masterplan.json"
            handoff = tmp / "handoff" / "current"
            handoff.mkdir(parents=True, exist_ok=True)
            (handoff / "live_check_test.3.md").write_text(
                "synthetic curl output — confirmed live behavior"
            )
            mp_file.write_text(json.dumps({
                "phases": [{"id": "phase-test", "steps": [
                    {"id": "test.3", "status": "done", "verification": {
                        "command": "true",
                        "success_criteria": ["x"],
                        "live_check": "synthetic evidence required",
                    }},
                ]}],
            }))
            decision = gate_decision(str(mp_file), "test.3", str(handoff))
            _check(
                "6. gate_passes_when_artifact_present",
                decision == "passed",
                f"expected 'passed', got '{decision}'",
            )

    # 7. CLAUDE.md documents the field
    _check(
        "7. claude_md_documents_live_check_field",
        (
            _grep_in_file(cmd, "verification.live_check")
            and _grep_in_file(cmd, "handoff/current/live_check_")
            and _grep_in_file(cmd, "phase-23.8.1")
        ),
        "CLAUDE.md must reference verification.live_check, the live_check_ artifact path, and phase-23.8.1",
    )

    # 8. Step 23.8.1 in masterplan has NO live_check field for itself
    # (otherwise the hook would block this very step's auto-push)
    try:
        d = json.loads(mp.read_text(encoding="utf-8"))
        from lib.live_check_gate import find_step
        step = find_step(d, "23.8.1")
        has_no_self_gate = (
            step is not None
            and isinstance(step.get("verification"), dict)
            and not step["verification"].get("live_check")
        )
    except Exception:
        has_no_self_gate = False
    _check(
        "8. step_23_8_1_does_not_set_live_check_for_itself",
        has_no_self_gate,
        "step 23.8.1 must not set verification.live_check on itself (chicken-and-egg)",
    )

    # 9. harness_log has the qa.md deferral note for Cycle 38
    hl = REPO / "handoff" / "harness_log.md"
    _check(
        "9. harness_log_has_qa_md_deferral_note_for_cycle_38",
        (
            _grep_in_file(hl, "phase=23.8.1")
            and _grep_in_file(hl, "qa.md")
            and _grep_in_file(hl, "deferred")
            and _grep_in_file(hl, "Separation of duties on agent edits")
        ),
        "harness_log.md must contain phase=23.8.1 cycle with qa.md deferral note citing the CLAUDE.md separation-of-duties rule",
    )

    # 10. No regressions: bash -n on all 9 hooks
    hooks_dir = REPO / ".claude" / "hooks"
    all_hooks_ok = True
    failed_hooks = []
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
        f"all .claude/hooks/*.sh must pass bash -n; failed: {failed_hooks}" if failed_hooks else "",
    )

    # Print results
    print("=== phase-23.8.1 verifier ===")
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
