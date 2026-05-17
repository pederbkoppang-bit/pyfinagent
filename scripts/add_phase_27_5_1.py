#!/usr/bin/env python3
"""Insert phase-27.5.1 after phase-27.5 in masterplan.json.

Born from phase-27.5 Q/A CONDITIONAL verdict 2026-05-17: full Gemini
pipeline runs end-to-end (3 tickers persisted), but cycle TIMED OUT at
1800s budget after only 3/15. Q/A required a follow-up before flipping
27.5 done. Parallelism chosen over budget raise — same cost, lower
latency, aligns with north-star "make most money" (faster cycles =
more market opportunities per day).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

existing_ids = {s["id"] for p in data["phases"] for s in p.get("steps", [])}
if "27.5.1" in existing_ids:
    print("27.5.1 already present; nothing to do")
    raise SystemExit(0)

NEW = {
    "id": "27.5.1",
    "name": "Parallelize per-ticker analysis (asyncio.gather, concurrency=4) so the full Gemini path fits the 1800s cycle budget",
    "status": "pending",
    "harness_required": True,
    "priority": "P0",
    "depends_on_step": "27.4",
    "audit_basis": (
        "phase-27.5 Q/A 2026-05-17 returned CONDITIONAL (agent "
        "a7bcaa71fac947e64): full Gemini pipeline runs end-to-end "
        "(3 tickers persisted: STX, AMD, FIX) but cycle TIMED OUT "
        "at 1800s budget after only 3/15 tickers. Q/A required "
        "follow-up before flipping 27.5 done. Parallelism (Q/A "
        "option #2) chosen over budget raise (#1) because it aligns "
        "with north-star (faster cycles = more market opportunity "
        "per day) AND keeps API cost identical. Concurrency capped "
        "at 4 (Gemini AI Studio default RPM allows ~60-300 depending "
        "on tier; 4 concurrent is conservative and per-ticker yfinance "
        "fetches don't trip rate limits)."
    ),
    "verification": {
        "command": (
            "source .venv/bin/activate && "
            "grep -qE 'asyncio\\.gather|asyncio\\.TaskGroup|asyncio\\.Semaphore' "
            "backend/services/autonomous_loop.py && "
            "python -c \"import ast; ast.parse(open('backend/services/autonomous_loop.py').read()); print('syntax OK')\""
        ),
        "success_criteria": [
            "asyncio_gather_or_TaskGroup_or_Semaphore_introduced_in_step_3_analysis_loop",
            "concurrency_capped_at_documented_value_via_Semaphore",
            "both_lite_AND_full_paths_compatible_with_concurrent_execution",
            "syntax_passes_for_autonomous_loop_py",
            "fresh_cycle_runs_all_15_tickers_under_1800s_budget"
        ],
        "live_check": (
            "fresh run-now cycle with standard=gemini-2.5-flash completes "
            "status=completed within the 1800s budget AND persists >=14 of "
            "15 analyses to BQ analysis_results; captured in handoff/current/"
            "live_check_27.5.md (rewritten or appended)"
        )
    },
    "retry_count": 0,
    "max_retries": 3
}

for p in data["phases"]:
    if p["id"] == "phase-27":
        for i, s in enumerate(p["steps"]):
            if s["id"] == "27.5":
                p["steps"].insert(i + 1, NEW)
                print(f"inserted 27.5.1 after 27.5 at index {i+1}")
                break
        break

data["updated_at"] = datetime.now(timezone.utc).isoformat()
MP.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("OK")
