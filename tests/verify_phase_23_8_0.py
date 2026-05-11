"""phase-23.8.0 verifier — 12-claim source-level assertion.

Run: source .venv/bin/activate && python3 tests/verify_phase_23_8_0.py

Exits 0 on PASS, 1 on FAIL. Each criterion is a single grep / exists /
import check. Criteria are immutable per the contract at
handoff/current/contract.md (phase-23.8.0).

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = []


def _check(name: str, ok: bool, detail: str = ""):
    flag = "PASS" if ok else "FAIL"
    RESULTS.append((flag, name, detail))
    return ok


def _grep_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8")


def _absent_in_file(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle not in path.read_text(encoding="utf-8")


def main() -> int:
    ad = REPO / "backend" / "agents" / "agent_definitions.py"
    arch = REPO / "ARCHITECTURE.md"
    cmd = REPO / "CLAUDE.md"
    mpj = REPO / "backend" / "backtest" / "experiments" / "meta_plan.json"
    pa = REPO / "backend" / "agents" / "planner_agent.py"
    tpa = REPO / "tests" / "agents" / "test_planner_meta_plan_config.py"
    hl = REPO / "handoff" / "harness_log.md"

    # 1. Ford renamed
    _check(
        "1. ford_label_renamed_to_slack_orchestrator",
        _grep_in_file(ad, 'name="Ford (Slack Orchestrator)"'),
        "agent_definitions.py must contain 'Ford (Slack Orchestrator)'",
    )

    # 2. Researcher renamed
    _check(
        "2. researcher_label_renamed_to_slack_researcher",
        _grep_in_file(ad, 'name="Slack Researcher"'),
        "agent_definitions.py must contain 'Slack Researcher'",
    )

    # 3. Communication prose updated (no old "Ford (Main Agent)" string)
    _check(
        "3. communication_prose_updated",
        (
            _absent_in_file(ad, "Ford (Main Agent)")
            and _grep_in_file(ad, "Ford / Slack Orchestrator")
        ),
        "agent_definitions.py Communication prose must mention new labels and drop the old 'Ford (Main Agent)' string",
    )

    # 4. ARCHITECTURE.md Layer-2 labels updated
    _check(
        "4. architecture_md_layer2_labels_updated",
        (
            _absent_in_file(arch, "Ford (Main Agent)")
            and _grep_in_file(arch, "Ford (Slack Orch.)")
            and _grep_in_file(arch, "Slack Researcher")
        ),
        "ARCHITECTURE.md must use new Layer-2 labels in the diagram",
    )

    # 5. CLAUDE.md three-agent rule scoped
    _check(
        "5. claude_md_three_agent_rule_scoped_to_layer3",
        (
            _grep_in_file(cmd, "Harness MAS layer (Layer 3) is exactly 3 agents")
            and _grep_in_file(cmd, "_inventory.json")
        ),
        "CLAUDE.md must scope the 'exactly 3 agents' rule to Layer 3 and cite _inventory.json",
    )

    # 6. meta_plan.json exists with 7 keys
    ok_mpj = mpj.exists()
    if ok_mpj:
        try:
            data = json.loads(mpj.read_text(encoding="utf-8"))
            required = (
                "sharpe_target", "annual_return_min_pct", "annual_return_max_pct",
                "max_drawdown_pct", "max_trades_per_month",
                "sector_concentration_max_pct", "cost_stress_multiple",
            )
            ok_mpj = all(k in data for k in required)
        except Exception:
            ok_mpj = False
    _check(
        "6. meta_plan_json_exists_with_7_keys",
        ok_mpj,
        "backend/backtest/experiments/meta_plan.json must exist with all 7 numeric keys",
    )

    # 7. planner_agent reads from JSON (no hardcoded triple-quoted META_PLAN)
    _check(
        "7. planner_agent_reads_from_meta_plan_json",
        (
            _absent_in_file(pa, 'META_PLAN = """')
            and _grep_in_file(pa, "_load_meta_plan_text")
            and _grep_in_file(pa, "self.meta_plan_text")
        ),
        "planner_agent.py must use _load_meta_plan_text(); old hardcoded constant must be gone",
    )

    # 8. test_planner_meta_plan_config exists + passes
    test_exists = tpa.exists()
    test_passes = False
    if test_exists:
        try:
            r = subprocess.run(
                ["python3", "-m", "pytest", str(tpa), "-q", "--no-header"],
                capture_output=True, text=True, timeout=60, cwd=REPO,
            )
            test_passes = (r.returncode == 0)
        except Exception as e:
            test_passes = False
    _check(
        "8. test_planner_meta_plan_config_passes",
        test_exists and test_passes,
        "tests/agents/test_planner_meta_plan_config.py must exist and pass",
    )

    # 9. ARCHITECTURE.md has the 28->5->3 mapping paragraph
    _check(
        "9. architecture_md_has_28_to_5_to_3_mapping_paragraph",
        (
            _grep_in_file(arch, "28 Layer-1 skills")
            and _grep_in_file(arch, "5 progressive-disclosure layers")
            and _grep_in_file(arch, "3 rows in lite-mode")
            and _grep_in_file(arch, "phase-23.2.A-agent-rationale-audit.md")
        ),
        "ARCHITECTURE.md must contain the 28->5->3 mapping paragraph citing the prior audit",
    )

    # 10. harness_log has both deferral notes
    _check(
        "10. harness_log_has_r5_and_r6_deferral_notes",
        (
            _grep_in_file(hl, "phase=23.8.0")
            and _grep_in_file(hl, "R-5 deferred")
            and _grep_in_file(hl, "R-6 deferred")
        ),
        "handoff/harness_log.md must contain phase=23.8.0 cycle with R-5 and R-6 deferral notes",
    )

    # 11. No regressions: active modules importable
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import backend.agents.planner_agent; "
             "import backend.agents.agent_definitions; "
             "import backend.services.autonomous_loop; print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        )
        active_ok = (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        active_ok = False
    _check(
        "11. no_import_regressions_active_modules",
        active_ok,
        "planner_agent + agent_definitions + autonomous_loop must all still import",
    )

    # 12. Deferred stubs still importable (R-6 NOT applied)
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import backend.autonomous_harness; "
             "import backend.agents.meta_coordinator; print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        )
        stubs_ok = (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        stubs_ok = False
    _check(
        "12. no_import_regressions_deferred_stubs_still_importable",
        stubs_ok,
        "autonomous_harness.py + meta_coordinator.py must remain importable (R-6 deferred)",
    )

    # Print results
    print("=== phase-23.8.0 verifier ===")
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
