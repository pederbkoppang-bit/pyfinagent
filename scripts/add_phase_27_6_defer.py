#!/usr/bin/env python3
"""Insert phase-27.6.4 + 27.6.5 (deferred infrastructure-layer follow-ups).

Born from cycle #10 evidence 2026-05-17: even with our process-side
fixes (27.6.1 Semaphore=2, 27.6.2 NoneType guard), the full Claude
path is blocked by TWO infrastructure rate limits we can't fix from
within the orchestrator process:

  - SEC 429 on the Cloud Function side (it doesn't cache the CIK map;
    every ticker triggers a fresh SEC fetch; SEC rate-limits the IP).
  - Anthropic 429 on `/v1/messages` — Claude's per-minute RPM cap is
    tighter than Gemini's, and concurrency=8 with ~10 Claude calls per
    ticker exceeds the cap.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

existing = {s["id"] for p in data["phases"] for s in p.get("steps", [])}

NEW = [
    {
        "id": "27.6.4",
        "name": "(DEFERRED, Cloud Function redeploy) Cache CIK map inside the Ingestion Agent CF so it doesn't re-fetch SEC per ticker",
        "status": "pending",
        "harness_required": True,
        "priority": "P1",
        "depends_on_step": "27.6",
        "audit_basis": (
            "Cycle #10 (2026-05-17 02:47-02:58) showed every uncached "
            "ticker still gets `Ingestion: Failed to fetch CIK map from "
            "SEC: 429`. Our 27.6.1 Semaphore(2) limits concurrent calls "
            "from our process to the Cloud Function, but the CF still "
            "fetches SEC fresh per ticker — SEC rate-limits the CF's IP "
            "regardless of our throttle. Fix requires CF code change "
            "(cache `company_tickers.json` in-memory in the CF for the "
            "lifetime of the CF instance; refresh daily) + CF redeploy. "
            "Out of scope for this session because: (1) CF deploys are "
            "Peder-approved infra changes, (2) the CF source repo may "
            "be separate. Tickers already cached in GCS (file path "
            "`<TICKER>/<form>_<date>_<doc>.json`) succeed regardless."
        ),
        "verification": {
            "command": (
                "echo 'manual gate — CF redeploy required'; "
                "echo 'verification: after CF redeploy, cycle log shows zero \"Failed to fetch CIK map from SEC: 429\" lines'"
            ),
            "success_criteria": [
                "CIK_map_cached_in_CF_instance_memory",
                "cycle_log_has_zero_failed_to_fetch_CIK_map_lines",
                "CF_redeploy_approved_by_Peder"
            ],
            "live_check": "fresh Claude cycle post-CF-redeploy has zero SEC 429s for any ticker"
        },
        "retry_count": 0,
        "max_retries": 3
    },
    {
        "id": "27.6.5",
        "name": "Per-provider concurrency cap: Claude tighter than Gemini due to lower RPM ceiling",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6",
        "audit_basis": (
            "Cycle #10 2026-05-17 02:58:39 hit `HTTP/1.1 429 Too Many "
            "Requests` from api.anthropic.com/v1/messages. Concurrency=8 "
            "(per 27.5.1, sized for Gemini's higher RPM) is too aggressive "
            "for Anthropic tier-1 (~50 RPM input, ~10 RPM output). Fix: "
            "split the cycle-wide _analysis_semaphore into a per-provider "
            "lookup — Gemini=8 (current), Claude=2-4 (conservative for "
            "default tier; raisable per Anthropic plan)."
        ),
        "verification": {
            "command": (
                "source .venv/bin/activate && python -c \"import ast; "
                "ast.parse(open('backend/services/autonomous_loop.py').read()); print('syntax OK')\" && "
                "grep -qE \"claude.*[Ss]emaphore|provider.*concurrency|_semaphore_for_model\" "
                "backend/services/autonomous_loop.py"
            ),
            "success_criteria": [
                "per_provider_concurrency_helper_in_autonomous_loop",
                "claude_cap_at_or_below_4",
                "gemini_cap_preserved_at_8",
                "fresh_Claude_cycle_zero_Anthropic_429s"
            ],
            "live_check": "fresh Claude cycle has zero `HTTP/1.1 429 Too Many Requests` from api.anthropic.com"
        },
        "retry_count": 0,
        "max_retries": 3
    }
]

for p in data["phases"]:
    if p["id"] == "phase-27":
        for i, s in enumerate(p["steps"]):
            if s["id"] == "27.6.3":
                for n, new_step in enumerate(NEW):
                    if new_step["id"] in existing:
                        continue
                    p["steps"].insert(i + 1 + n, new_step)
                    print(f"inserted {new_step['id']}")
                break
        break

data["updated_at"] = datetime.now(timezone.utc).isoformat()
MP.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("OK")
