"""phase-4.8 step 4.8.8 Supply-chain audit.

Checks:
1. root requirements.txt exists and includes backend/requirements.txt
   via `-r` directive (not a stub).
2. backend/requirements.txt still has the 5 exact-pinned LLM
   clients from Cycle 65: anthropic, openai, google-cloud-
   aiplatform, fastmcp, alpaca-py.
3. .github/workflows/pip-audit.yml exists with:
   - `pip-audit` command
   - `--strict` flag
   - `schedule:` with a weekly cron expression
4. `pip-audit --requirement backend/requirements.txt --strict`
   exits 0.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT_REQS = REPO / "requirements.txt"
BACKEND_REQS = REPO / "backend" / "requirements.txt"
WORKFLOW = REPO / ".github" / "workflows" / "pip-audit.yml"
OUT = REPO / "handoff" / "supply_chain_audit.json"

EXPECT_PINNED = [
    "anthropic",
    "openai",
    "google-cloud-aiplatform",
    "fastmcp",
    "alpaca-py",
]


def _pinned(pkg: str, reqs_text: str) -> bool:
    pat = rf"^{re.escape(pkg)}==\d"
    return bool(re.search(pat, reqs_text, re.MULTILINE))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    reasons: list[str] = []

    # 1. root requirements.txt
    t1 = ROOT_REQS.exists()
    if t1:
        root_text = ROOT_REQS.read_text(encoding="utf-8")
        includes = bool(
            re.search(r"^-r\s+backend/requirements\.txt\s*$",
                       root_text, re.MULTILINE)
        )
        if not includes:
            t1 = False
            reasons.append("root requirements.txt missing -r backend/requirements.txt")
    else:
        reasons.append("root requirements.txt missing")

    # 2. 5 exact pins
    pinned_status: dict[str, bool] = {}
    if BACKEND_REQS.exists():
        reqs_text = BACKEND_REQS.read_text(encoding="utf-8")
        for pkg in EXPECT_PINNED:
            pinned_status[pkg] = _pinned(pkg, reqs_text)
        t2 = all(pinned_status.values())
        if not t2:
            missing = [p for p, v in pinned_status.items() if not v]
            reasons.append(f"LLM pins missing/unpinned: {missing}")
    else:
        t2 = False
        reasons.append("backend/requirements.txt missing")

    # 3. workflow structure
    t3 = WORKFLOW.exists()
    workflow_checks: dict[str, bool] = {}
    if t3:
        wf = WORKFLOW.read_text(encoding="utf-8")
        workflow_checks["has_pip_audit"] = "pip-audit" in wf
        workflow_checks["has_strict"] = "--strict" in wf
        workflow_checks["has_schedule"] = "schedule:" in wf
        # Weekly cron: "cron: \"0 X * * Y\"" where Y is day-of-week 0-6
        workflow_checks["weekly_cron"] = bool(
            re.search(r"cron:\s*\"[^\"]*\*\s*\*\s*[0-6]\"", wf)
        )
        t3 = all(workflow_checks.values())
        if not t3:
            missing = [k for k, v in workflow_checks.items() if not v]
            reasons.append(f"workflow checks failed: {missing}")
    else:
        reasons.append("workflow file missing")

    # 4. pip-audit --strict run locally
    pip_audit_ok = False
    pip_audit_summary = ""
    try:
        r = subprocess.run(
            ["pip-audit", "--requirement", str(BACKEND_REQS),
             "--strict", "--progress-spinner", "off"],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        pip_audit_summary = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
        pip_audit_ok = r.returncode == 0
        if not pip_audit_ok:
            reasons.append(f"pip-audit rc={r.returncode}; tail: {pip_audit_summary[:200]}")
    except FileNotFoundError:
        reasons.append("pip-audit not on PATH (activate venv first)")
    except subprocess.TimeoutExpired:
        reasons.append("pip-audit timed out after 120s")

    all_ok = t1 and t2 and t3 and pip_audit_ok
    verdict = "PASS" if all_ok else "FAIL"
    summary = {
        "step": "4.8.8",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "root_requirements_includes_backend": t1,
        "llm_clients_pinned": t2,
        "pinned_status": pinned_status,
        "pip_audit_in_ci": t3,
        "workflow_checks": workflow_checks,
        "pip_audit_no_vulns": pip_audit_ok,
        "pip_audit_tail": pip_audit_summary,
        "reasons": reasons,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "pinned": t2,
        "ci": t3,
        "pip_audit_clean": pip_audit_ok,
    }))
    if args.check and not all_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
