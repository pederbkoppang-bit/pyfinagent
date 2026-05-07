"""phase-23.3.3: immutable verification.

Asserts the dormant phase-9 jobs are now wired + the regression tests pass.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_register_call_in_start_scheduler():
    rel = "backend/slack_bot/scheduler.py"
    text = _read(rel)
    ast.parse(text)
    # Find the start_scheduler function and confirm the call inside it
    fn_start = text.find("def start_scheduler")
    # Next def AFTER start_scheduler (any kind)
    nxt = re.search(r"\n(?:async )?def ", text[fn_start + 1:])
    fn_end = (fn_start + 1 + nxt.start()) if nxt else len(text)
    body = text[fn_start:fn_end]
    assert "register_phase9_jobs(_scheduler)" in body, \
        "start_scheduler must call register_phase9_jobs(_scheduler)"
    return f"OK {rel}"


def check_safety_kwargs_in_mapping():
    rel = "backend/slack_bot/scheduler.py"
    text = _read(rel)
    # Confirm misfire_grace_time and coalesce both present in the mapping
    mapping_match = re.search(r"def register_phase9_jobs\(.*?\n    return registered", text, re.DOTALL)
    assert mapping_match, "register_phase9_jobs body not found"
    body = mapping_match.group(0)
    assert "misfire_grace_time" in body, \
        "mapping must include misfire_grace_time"
    assert '"coalesce": True' in body, \
        "mapping must include coalesce=True"
    # Tier-correct grace times
    assert "3600" in body, "daily grace 3600 missing"
    assert "7200" in body, "weekly grace 7200 missing"
    assert "600" in body, "hourly grace 600 missing"
    return "OK safety kwargs (misfire_grace_time + coalesce) per tier"


def check_pytest_passes():
    rel = "tests/services/test_phase9_registration.py"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", rel, "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.strip().splitlines()[-10:])
        raise AssertionError(f"pytest failed: {tail}")
    return f"OK {rel}"


def check_audit_findings():
    rel = "handoff/current/phase-23.3.3-audit-findings.md"
    p = ROOT / rel
    assert p.exists(), f"missing: {rel}"
    text = p.read_text()
    assert "register_phase9_jobs" in text and "dormant" in text.lower(), \
        "audit findings must explain the dormancy"
    assert "operator" in text.lower() and "restart" in text.lower(), \
        "audit findings must name the operator-restart caveat"
    return f"OK {rel}"


def main() -> int:
    checks = [
        check_register_call_in_start_scheduler,
        check_safety_kwargs_in_mapping,
        check_pytest_passes,
        check_audit_findings,
    ]
    failed = 0
    for fn in checks:
        try:
            print(fn())
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print(f"\nphase-23.3.3 verification: ALL PASS ({len(checks)}/{len(checks)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
