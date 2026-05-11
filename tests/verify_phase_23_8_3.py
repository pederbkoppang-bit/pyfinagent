"""phase-23.8.3 verifier — 10-claim source-level + import-regression assertion.

Run: source .venv/bin/activate && python3 tests/verify_phase_23_8_3.py

Exits 0 on PASS, 1 on FAIL.

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md R-6 (closure)
"""
from __future__ import annotations

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
    meta = REPO / "backend" / "agents" / "meta_coordinator.py"
    auto = REPO / "backend" / "autonomous_harness.py"
    me_init = REPO / "backend" / "meta_evolution" / "__init__.py"
    me_alpha = REPO / "backend" / "meta_evolution" / "alpha_velocity.py"
    audit_doc = REPO / "docs" / "audits" / "dev-mas-2026-05-11" / "04-remediation.md"
    harness_log = REPO / "handoff" / "harness_log.md"

    _check(
        "1. meta_coordinator_header_no_longer_says_deprecated_phase_4_stub",
        _absent_in_file(meta, "DEPRECATED — Phase 4 stub"),
        "meta_coordinator.py must not contain 'DEPRECATED — Phase 4 stub'",
    )

    _check(
        "2. meta_coordinator_header_says_active_with_importers",
        (
            _grep_in_file(meta, "ACTIVE legacy MAS coordinator")
            and _grep_in_file(meta, "autonomous_loop.py")
            and _grep_in_file(meta, "skill_optimizer.py")
        ),
        "meta_coordinator.py header must call itself ACTIVE and name both live importers",
    )

    _check(
        "3. autonomous_harness_header_no_longer_says_deprecated_phase_4_stub",
        _absent_in_file(auto, "DEPRECATED — Phase 4 stub"),
        "autonomous_harness.py must not contain 'DEPRECATED — Phase 4 stub'",
    )

    _check(
        "4. autonomous_harness_header_says_active_with_callers",
        (
            _grep_in_file(auto, "ACTIVE")
            and _grep_in_file(auto, "phase4_9_redteam.py")
            and _grep_in_file(auto, "FINRA")
        ),
        "autonomous_harness.py header must call itself ACTIVE and cite the phase4_9_redteam.py FINRA usage",
    )

    _check(
        "5. meta_evolution_init_contrast_label_updated",
        (
            _absent_in_file(me_init, "DEPRECATED `backend/agents/meta_coordinator.py`")
            and _grep_in_file(me_init, "phase-23.8.3")
        ),
        "meta_evolution/__init__.py must replace the old DEPRECATED contrast label and cite phase-23.8.3",
    )

    _check(
        "6. meta_evolution_alpha_velocity_contrast_label_updated",
        (
            _absent_in_file(me_alpha, "DEPRECATED `backend/agents/meta_coordinator.py`")
            and _grep_in_file(me_alpha, "phase-23.8.3")
        ),
        "alpha_velocity.py must replace the old DEPRECATED contrast label and cite phase-23.8.3",
    )

    _check(
        "7. audit_remediation_md_has_r6_closure_note",
        (
            _grep_in_file(audit_doc, "CLOSURE (phase-23.8.3")
            and _grep_in_file(audit_doc, "header-correction")
            and _grep_in_file(audit_doc, "preserved as a historical record")
        ),
        "04-remediation.md must have the R-6 closure note citing header-correction and historical-record preservation",
    )

    try:
        r = subprocess.run(
            ["python3", "-c",
             "import backend.agents.meta_coordinator; "
             "import backend.autonomous_harness; print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        )
        target_ok = (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        target_ok = False
    _check(
        "8. no_regressions_targeted_modules_import",
        target_ok,
        "meta_coordinator + autonomous_harness must still import (header edits should not break syntax)",
    )

    try:
        r = subprocess.run(
            ["python3", "-c",
             "import backend.services.autonomous_loop; "
             "import backend.agents.skill_optimizer; print('OK')"],
            capture_output=True, text=True, timeout=30, cwd=REPO,
        )
        live_ok = (r.returncode == 0 and "OK" in r.stdout)
    except Exception:
        live_ok = False
    _check(
        "9. no_regressions_live_importers_still_import",
        live_ok,
        "autonomous_loop + skill_optimizer must still import (they depend on the targeted modules)",
    )

    _check(
        "10. harness_log_has_r6_closure_note",
        (
            _grep_in_file(harness_log, "phase=23.8.3")
            and _grep_in_file(harness_log, "R-6")
            and _grep_in_file(harness_log, "header correction")
        ),
        "harness_log.md must contain phase=23.8.3 cycle with R-6 closure framing",
    )

    print("=== phase-23.8.3 verifier ===")
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
