#!/usr/bin/env python3
"""Insert phase-27.6.1, .2, .3 after phase-27.6 in masterplan.json.

Born from phase-27.6 Q/A CONDITIONAL verdict 2026-05-17 (agent
aa86f7cd0a7722772): Claude full path failed on 14/14 tickers with two
NEW orthogonal upstream bugs (SEC 429 + QuantAgent NoneType). Lite
Claude rescued all 14, but the strict `zero_Full_orchestrator_failed_lines`
criterion was violated.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

existing = {s["id"] for p in data["phases"] for s in p.get("steps", [])}

NEW_STEPS = [
    {
        "id": "27.6.1",
        "name": "SEC EDGAR rate-limit guard: cache company_tickers.json + per-host throttle + User-Agent header",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6",
        "audit_basis": (
            "phase-27.6 Q/A 2026-05-17 (agent aa86f7cd0a7722772) returned "
            "CONDITIONAL: 8 of 14 Claude full-path failures were 'Ingestion "
            "Agent Error: ERROR:429 Client Error: Too Many Requests for "
            "url: https://www.sec.gov/files/company_tickers.json'. "
            "Concurrency=8 (per 27.5.1) fires 8 simultaneous bulk downloads "
            "without throttling. SEC EDGAR fair-access requires "
            "User-Agent header + ≤10 req/sec; bulk endpoint may be stricter. "
            "Fix: cache company_tickers.json for 24h (it changes daily) + "
            "per-host concurrency limit of 2 + courtesy User-Agent."
        ),
        "verification": {
            "command": (
                "source .venv/bin/activate && python -c \"import ast; "
                "ast.parse(open('backend/agents/orchestrator.py').read()); "
                "print('syntax OK')\" && "
                "grep -qE 'company_tickers.*cache|_TICKERS_CACHE|cached_ticker' "
                "backend/agents/orchestrator.py"
            ),
            "success_criteria": [
                "company_tickers_json_cached_in_memory_or_on_disk_for_24h",
                "per_host_throttle_added_for_sec_gov",
                "User-Agent_header_added_per_SEC_fair_access_policy",
                "next_Claude_cycle_zero_SEC_429_errors"
            ],
            "live_check": "fresh Claude cycle has zero `Ingestion Agent Error: ERROR:429` log lines"
        },
        "retry_count": 0,
        "max_retries": 3
    },
    {
        "id": "27.6.2",
        "name": "QuantAgent NoneType safety: defensive `.get()` / `or {}` on upstream dep that returns None on Claude path",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6",
        "audit_basis": (
            "phase-27.6 Q/A 2026-05-17 (agent aa86f7cd0a7722772) returned "
            "CONDITIONAL: 6 of 14 Claude full-path failures were "
            "'ERROR: QuantAgent failed for X: NoneType object has no "
            "attribute get'. Affected: STX, COHR, INTC, DELL, SNDK, WDC. "
            "Cycle #8 (Gemini) had ZERO of these — so this is Claude-"
            "pathway-specific (either different upstream branch or "
            "different defensive coercion). Fix: defensive `.get()` / "
            "`or {}` on the upstream dependency before the `.get()` call."
        ),
        "verification": {
            "command": (
                "source .venv/bin/activate && python -c \"import ast; "
                "ast.parse(open('backend/agents/orchestrator.py').read()); "
                "print('syntax OK')\" && "
                "! grep -nE '\\bquant\\b.*\\bNone\\b' backend/agents/orchestrator.py | "
                "grep -v -E 'or \\{\\}|safe|guard' | head -1"
            ),
            "success_criteria": [
                "QuantAgent_get_path_has_or_empty_dict_guard",
                "regression_test_added_or_existing_test_passes",
                "next_Claude_cycle_zero_QuantAgent_NoneType_errors"
            ],
            "live_check": "fresh Claude cycle has zero `QuantAgent failed.*NoneType` log lines"
        },
        "retry_count": 0,
        "max_retries": 3
    },
    {
        "id": "27.6.3",
        "name": "Re-run Claude full-path smoke after 27.6.1 + 27.6.2 — zero full-orchestrator failures required",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6.2",
        "audit_basis": (
            "phase-27.6 Q/A required a clean re-smoke after the two upstream "
            "fixes. Same gate as 27.6 but with success_criterion "
            "zero_Full_orchestrator_failed_lines holding cleanly this time."
        ),
        "verification": {
            "command": (
                "test -f handoff/current/live_check_27.6.3.md && "
                "grep -q 'cycle_id' handoff/current/live_check_27.6.3.md && "
                "grep -q 'lite_mode.*[Ff]alse' handoff/current/live_check_27.6.3.md && "
                "grep -qE 'analyses_persisted.*1[4-9]|analyses_persisted.*2[0-9]' "
                "handoff/current/live_check_27.6.3.md && "
                "grep -qE 'full_orchestrator_failures.*: 0|zero full|0 full' "
                "handoff/current/live_check_27.6.3.md"
            ),
            "success_criteria": [
                "model_set_to_claude-sonnet-4-6",
                "full_cycle_completed_status_completed",
                "min_14_analyses_persisted",
                "zero_Full_orchestrator_failed_lines",
                "live_check_27.6.3.md_captures_full_path_persistence_via_Claude_pipeline"
            ],
            "live_check": "live_check_27.6.3.md captures cycle_id, status=completed, persist count, AND full-path-success per ticker"
        },
        "retry_count": 0,
        "max_retries": 3
    }
]

inserted = 0
for p in data["phases"]:
    if p["id"] == "phase-27":
        for i, s in enumerate(p["steps"]):
            if s["id"] == "27.6":
                # Insert all 3 right after 27.6
                for n, new_step in enumerate(NEW_STEPS):
                    if new_step["id"] in existing:
                        continue
                    p["steps"].insert(i + 1 + n, new_step)
                    inserted += 1
                break
        break

data["updated_at"] = datetime.now(timezone.utc).isoformat()
MP.write_text(json.dumps(data, indent=2), encoding="utf-8")
print(f"inserted {inserted} new sub-steps")
