"""phase-4.8 step 4.8.6 DR runbooks + drill-log audit.

Checks:
1. Three runbook files exist under docs/runbooks/.
2. Each has the 6 required sections: Scope, Trigger, Response
   Steps, Rollback, RTO Target, Last Drill.
3. Each runbook's Response Steps has >=4 numbered steps (not a
   placeholder stub).
4. handoff/dr_drill_log.md has 3 drill entries with both
   rto_target_minutes and rto_actual_minutes populated.
5. Each drill's rto_actual_minutes <= rto_target_minutes (drills
   passed). A regression where targets were auto-matched would
   fail a separate plausibility check (target > 0 and
   actual > 0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK_DIR = REPO / "docs" / "runbooks"
DRILL_LOG = REPO / "handoff" / "dr_drill_log.md"
OUT = REPO / "handoff" / "dr_runbooks_audit.json"

REQUIRED = ("broker_outage", "data_feed_outage", "llm_outage")
REQUIRED_SECTIONS = (
    "## Scope", "## Trigger", "## Response Steps", "## Rollback",
    "## RTO Target", "## Last Drill",
)


def _check_runbook(path: Path) -> dict:
    reasons: list[str] = []
    if not path.exists():
        return {"ok": False, "reasons": [f"missing: {path}"]}
    text = path.read_text(encoding="utf-8")
    for sec in REQUIRED_SECTIONS:
        if sec not in text:
            reasons.append(f"missing section '{sec}'")
    # Response Steps numbered list: at least 4 items starting "N."
    rs_idx = text.find("## Response Steps")
    rs_block = text[rs_idx:rs_idx + 4000] if rs_idx >= 0 else ""
    num_items = len(re.findall(r"^\d+\.\s", rs_block, re.MULTILINE))
    if num_items < 4:
        reasons.append(f"Response Steps has {num_items} numbered items (<4)")
    return {"ok": not reasons, "reasons": reasons, "numbered_steps": num_items}


def _parse_drills() -> dict:
    if not DRILL_LOG.exists():
        return {"ok": False, "reasons": ["missing dr_drill_log.md"],
                "drills": []}
    text = DRILL_LOG.read_text(encoding="utf-8")
    # Each drill block starts with "## Drill N -- <scenario> (<date>)"
    blocks = re.split(r"^##\s+Drill\s+\d+\s+--\s+", text, flags=re.MULTILINE)
    drills: list[dict] = []
    reasons: list[str] = []
    for blk in blocks[1:]:
        # scenario is the first word after "-- "
        m = re.match(r"([a-z_]+)\s*", blk)
        scenario = m.group(1) if m else "?"
        target = re.search(
            r"rto_target_minutes\*?\*?:\s*(\d+)", blk
        )
        actual = re.search(
            r"rto_actual_minutes\*?\*?:\s*(\d+)", blk
        )
        verdict = re.search(
            r"verdict\*?\*?:\s*([A-Z]+)", blk
        )
        t = int(target.group(1)) if target else None
        a = int(actual.group(1)) if actual else None
        v = verdict.group(1) if verdict else None
        drills.append({
            "scenario": scenario, "rto_target": t,
            "rto_actual": a, "verdict": v,
        })
    # Validate structure
    scenarios_found = {d["scenario"] for d in drills}
    for req in REQUIRED:
        if req not in scenarios_found:
            reasons.append(f"missing drill for scenario '{req}'")
    for d in drills:
        if d["rto_target"] is None or d["rto_actual"] is None:
            reasons.append(f"{d['scenario']} missing rto values")
        elif d["rto_target"] <= 0 or d["rto_actual"] <= 0:
            reasons.append(f"{d['scenario']} has non-positive rto value(s)")
        elif d["rto_actual"] > d["rto_target"]:
            reasons.append(
                f"{d['scenario']} actual {d['rto_actual']} > target {d['rto_target']}"
            )
    return {"ok": not reasons, "reasons": reasons, "drills": drills}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    runbook_results = {}
    for name in REQUIRED:
        path = RUNBOOK_DIR / f"{name}.md"
        runbook_results[name] = _check_runbook(path)
    drills_result = _parse_drills()

    three_runbooks_ok = all(r["ok"] for r in runbook_results.values())
    three_drills_ok = (drills_result["ok"]
                       and len([d for d in drills_result["drills"]
                                if d["scenario"] in REQUIRED]) == 3)
    rto_measured = all(
        d["rto_actual"] is not None and d["rto_target"] is not None
        for d in drills_result["drills"]
    )

    verdict = "PASS" if (three_runbooks_ok and three_drills_ok
                          and rto_measured) else "FAIL"
    summary = {
        "step": "4.8.6",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "three_runbooks_landed": three_runbooks_ok,
        "three_tabletop_drills_logged": three_drills_ok,
        "rto_per_scenario_measured": rto_measured,
        "runbooks": runbook_results,
        "drills": drills_result["drills"],
        "drill_reasons": drills_result["reasons"],
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(OUT),
        "verdict": verdict,
        "three_runbooks": three_runbooks_ok,
        "three_drills": three_drills_ok,
        "rto_measured": rto_measured,
    }))
    if args.check and verdict != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
