#!/usr/bin/env python3
"""
One-time extraction: Parse PLAN.md into .claude/masterplan.json

Usage:
    python scripts/generate_masterplan.py
    python scripts/generate_masterplan.py --diff  # Show differences between PLAN.md and existing masterplan.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PLAN_PATH = PROJECT_ROOT / "PLAN.md"
MASTERPLAN_PATH = PROJECT_ROOT / ".claude" / "masterplan.json"
HANDOFF_DIR = PROJECT_ROOT / "handoff"


def find_contract(phase_id: str) -> str | None:
    """Find matching handoff contract file for a phase/step."""
    patterns = [
        f"contract_phase_{phase_id}.md",
        f"contract_phase_{phase_id.replace('.', '_')}.md",
        f"contract_{phase_id}.md",
    ]
    for pattern in patterns:
        path = HANDOFF_DIR / pattern
        if path.exists():
            return f"handoff/{pattern}"
    return None


def parse_status(line: str, header: str = "") -> str:
    """Map emoji/text markers to status enum."""
    combined = f"{header} {line}"
    if "✅" in combined:
        return "done"
    if "⚠️ GATE" in combined or "BLOCKED" in combined.upper():
        return "blocked"
    if "~60% BUILT" in combined:
        return "in-progress"
    if "🔄 QUEUED" in combined or "QUEUED" in combined.upper():
        return "pending"
    if "🔄" in combined or "IN PROGRESS" in combined.upper():
        return "in-progress"
    if "FUTURE" in combined.upper():
        return "pending"
    # Check for checkbox markers
    if "- [x]" in line:
        return "done"
    if "- [🔄]" in line:
        return "in-progress"
    if "- [ ]" in line:
        return "pending"
    return "pending"


def has_harness_tag(lines: list[str], start: int, end: int) -> bool:
    """Check if a section has a > Harness: tag."""
    for i in range(start, min(end, len(lines))):
        if lines[i].strip().startswith("> **Harness:**") or lines[i].strip().startswith("> Harness:"):
            return True
    return False


def extract_gate(header: str, lines: list[str], start: int, end: int) -> dict | None:
    """Extract gate information from phase header and content."""
    # Check header and first ~20 lines of content for GATE markers
    combined = header
    for i in range(start, min(end, len(lines))):
        combined += " " + lines[i]
    if "⚠️ GATE" in combined or "GATE:" in combined.upper():
        reason = "Requires approval before starting"
        for i in range(start, min(end, len(lines))):
            if "REQUIRES" in lines[i].upper() and "APPROVAL" in lines[i].upper():
                reason = lines[i].strip().lstrip("#").strip().lstrip("⚠️ ").strip()
                break
        return {
            "type": "approval",
            "approver": "Peder",
            "reason": reason,
            "approved": False,
        }
    return None


def parse_plan():
    """Parse PLAN.md into masterplan schema."""
    text = PLAN_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")

    phases = []
    current_phase = None
    current_step = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Phase headers: ## Phase N: Name STATUS
        if line.startswith("## Phase "):
            # Save previous phase
            if current_phase:
                if current_step:
                    current_phase["steps"].append(current_step)
                    current_step = None
                phases.append(current_phase)

            # Parse phase header
            header = line.lstrip("# ").strip()
            # Extract phase number
            parts = header.split(":")
            phase_num = parts[0].replace("Phase ", "").strip().split(" ")[0]
            phase_name = ":".join(parts[1:]).strip() if len(parts) > 1 else header

            # Clean up phase name (remove status markers, parentheticals)
            for marker in ["✅", "⚠️", "🔄", "📋", "COMPLETE", "CORE IMPLEMENTED",
                           "OPERATIONAL", "GATE", "FUTURE"]:
                phase_name = phase_name.replace(marker, "")
            # Remove trailing status in parens/dashes
            phase_name = phase_name.split("—")[0].split("(")[0].strip().rstrip(" -–")

            # Find end of phase section
            phase_end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## Phase ") or (lines[j].startswith("## ") and "Phase" not in lines[j] and j > i + 5):
                    phase_end = j
                    break

            status = parse_status(line, header)

            # Determine dependencies
            depends_on = []
            phase_int = phase_num.split(".")[0]
            if phase_int.isdigit() and int(phase_int) > 0:
                depends_on = [f"phase-{int(phase_int) - 1}"]

            gate = extract_gate(header, lines, i, min(i + 20, phase_end))

            current_phase = {
                "id": f"phase-{phase_num}",
                "name": phase_name,
                "status": status,
                "depends_on": depends_on,
                "gate": gate,
                "steps": [],
            }

        # Step headers: ### N.M Name STATUS
        elif line.startswith("### ") and current_phase:
            # Save previous step
            if current_step:
                current_phase["steps"].append(current_step)
                current_step = None

            header = line.lstrip("# ").strip()

            # Try to extract step ID (e.g., "2.0", "2.6.0", "4.1")
            import re
            step_match = re.match(r"(\d+\.\d+(?:\.\d+)?)\s+(.+)", header)
            if step_match:
                step_id = step_match.group(1)
                step_name = step_match.group(2)
            else:
                # Try other patterns like "Implemented improvements"
                i += 1
                continue

            # Clean up step name
            for marker in ["✅", "⚠️", "🔄", "PASS", "FAIL", "CONDITIONAL PASS",
                           "COMPLETE", "OPERATIONAL", "IMPLEMENTED", "ACTIVATED", "LIVE"]:
                step_name = step_name.replace(marker, "")
            step_name = step_name.split("—")[0].split("(")[0].strip().rstrip(" -–")

            # Find end of step section
            step_end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("### ") or lines[j].startswith("## "):
                    step_end = j
                    break

            step_status = parse_status(line, header)
            harness_req = has_harness_tag(lines, i, step_end)

            # Build verification if harness required
            verification = None
            if harness_req and step_status != "done":
                verification = {
                    "command": "source .venv/bin/activate && python scripts/harness/run_harness.py --dry-run --cycles 1",
                    "success_criteria": ["evaluator_critique_pass", "no_regressions"],
                }

            current_step = {
                "id": step_id,
                "name": step_name,
                "status": step_status,
                "harness_required": harness_req,
                "verification": verification,
                "contract": find_contract(step_id),
                "retry_count": 0,
                "max_retries": 3,
            }

        i += 1

    # Save last phase/step
    if current_step and current_phase:
        current_phase["steps"].append(current_step)
    if current_phase:
        phases.append(current_phase)

    # Determine overall phase status from steps
    for phase in phases:
        if phase["steps"]:
            statuses = [s["status"] for s in phase["steps"]]
            if all(s == "done" for s in statuses):
                if phase["status"] != "blocked":
                    phase["status"] = "done"
            elif any(s == "in-progress" for s in statuses):
                if phase["status"] not in ("blocked",):
                    phase["status"] = "in-progress"
            elif any(s == "pending" for s in statuses) and any(s == "done" for s in statuses):
                if phase["status"] not in ("blocked",):
                    phase["status"] = "in-progress"

    masterplan = {
        "$schema": "masterplan-v1",
        "project": "pyfinagent",
        "goal": "Ship a validated, evidence-based trading signal system by May 2026",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
    }

    return masterplan


def diff_masterplans(old_path: Path, new_data: dict):
    """Show differences between existing and newly parsed masterplan."""
    if not old_path.exists():
        print("No existing masterplan.json found. Run without --diff to create it.")
        return

    old_data = json.loads(old_path.read_text())
    old_phases = {p["id"]: p for p in old_data.get("phases", [])}
    new_phases = {p["id"]: p for p in new_data.get("phases", [])}

    for pid, new_phase in new_phases.items():
        old_phase = old_phases.get(pid)
        if not old_phase:
            print(f"  NEW PHASE: {pid}: {new_phase['name']}")
            continue
        if old_phase["status"] != new_phase["status"]:
            print(f"  CHANGED: {pid} status: {old_phase['status']} -> {new_phase['status']}")

        old_steps = {s["id"]: s for s in old_phase.get("steps", [])}
        for step in new_phase.get("steps", []):
            old_step = old_steps.get(step["id"])
            if not old_step:
                print(f"    NEW STEP: {step['id']}: {step['name']}")
            elif old_step["status"] != step["status"]:
                print(f"    CHANGED: {step['id']} status: {old_step['status']} -> {step['status']}")


if __name__ == "__main__":
    masterplan = parse_plan()

    if "--diff" in sys.argv:
        diff_masterplans(MASTERPLAN_PATH, masterplan)
    else:
        MASTERPLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        MASTERPLAN_PATH.write_text(
            json.dumps(masterplan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Summary
        total_steps = sum(len(p["steps"]) for p in masterplan["phases"])
        done = sum(1 for p in masterplan["phases"] for s in p["steps"] if s["status"] == "done")
        in_prog = sum(1 for p in masterplan["phases"] for s in p["steps"] if s["status"] == "in-progress")
        pending = sum(1 for p in masterplan["phases"] for s in p["steps"] if s["status"] == "pending")
        blocked = sum(1 for p in masterplan["phases"] for s in p.get("steps", []) if s["status"] == "blocked")

        print(f"Generated {MASTERPLAN_PATH}")
        print(f"  Phases: {len(masterplan['phases'])}")
        print(f"  Steps:  {total_steps}")
        print(f"  Done:   {done}")
        print(f"  Active: {in_prog}")
        print(f"  Pending: {pending}")
        print(f"  Blocked: {blocked}")
