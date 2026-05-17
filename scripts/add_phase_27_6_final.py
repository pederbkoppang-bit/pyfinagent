#!/usr/bin/env python3
"""Append phase-27.6.7 and 27.6.8 — newly-discovered Claude-path bugs.

Cycle #11 (2026-05-17 03:36-03:39 CEST) with 27.6.1+.2+.5+.6 all applied
exposed TWO additional pre-existing Claude-path bugs that the prior
schema/routing failures had masked:

  - `coroutine 'AnalysisOrchestrator.fetch_insider_data' was never awaited`
    — an async function is being called without `await` somewhere in the
    Claude path. RuntimeWarning surfaces in cycle log.
  - `Messages.create() got an unexpected keyword argument 'betas'` —
    Anthropic SDK call passes a kwarg the installed SDK version rejects.
    Multiple enrichment agents (Anomaly Detection, Social Sentiment,
    Patent Innovation) fail with this.

Both are pre-existing latent bugs in the Claude path that the upstream
schema bug (27.1) was masking. Fix requires careful per-call-site
investigation; not a one-line change.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

existing = {s["id"] for p in data["phases"] for s in p.get("steps", [])}

NEW = [
    {
        "id": "27.6.7",
        "name": "Fix unawaited coroutine: AnalysisOrchestrator.fetch_insider_data must be awaited at every call site",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6.6",
        "audit_basis": (
            "Cycle #11 2026-05-17 03:39:21 surfaced: `RuntimeWarning: "
            "coroutine AnalysisOrchestrator.fetch_insider_data was "
            "never awaited`. The async function is invoked somewhere "
            "(likely an enrichment-agent dispatch path) without await, "
            "so its return value is a coroutine object — downstream "
            "`.get(...)` raises `'coroutine' object has no attribute "
            "'get'`. Caused full orchestrator failure on STX in cycle "
            "#11. Fix requires grepping ALL call sites of "
            "fetch_insider_data + verifying each is awaited or wrapped "
            "in asyncio.run / asyncio.create_task."
        ),
        "verification": {
            "command": (
                "source .venv/bin/activate && python -c \"import ast; "
                "ast.parse(open('backend/agents/orchestrator.py').read()); print('syntax OK')\" && "
                "! grep -E 'fetch_insider_data\\(' backend/agents/orchestrator.py | grep -v '^.*await\\|^.*async def' | head -1"
            ),
            "success_criteria": [
                "every_fetch_insider_data_call_is_awaited_or_wrapped",
                "no_RuntimeWarning_coroutine_never_awaited_for_fetch_insider_data",
                "fresh_Claude_cycle_zero_coroutine_attribute_errors"
            ],
            "live_check": "fresh Claude cycle log has zero 'coroutine object has no attribute' errors"
        },
        "retry_count": 0,
        "max_retries": 3
    },
    {
        "id": "27.6.8",
        "name": "Fix Anthropic SDK 'betas' kwarg incompatibility in enrichment agents (Anomaly/Social/Patent)",
        "status": "pending",
        "harness_required": True,
        "priority": "P0",
        "depends_on_step": "27.6.6",
        "audit_basis": (
            "Cycle #11 2026-05-17 03:39:41 surfaced repeated: "
            "`Enrichment agent X failed: Messages.create() got an "
            "unexpected keyword argument 'betas'`. Affected: Anomaly "
            "Detection, Social Sentiment, Patent Innovation (and "
            "likely others). The installed anthropic SDK does not "
            "accept `betas=` in `messages.create(**kwargs)`. Either "
            "wrong kwarg name (correct may be `extra_headers={'anthropic-beta': ...}` "
            "or the beta API moved to `client.beta.messages.create`), "
            "or `betas` was added in an SDK version we haven't installed. "
            "Fix requires checking the anthropic SDK reference + the "
            "kwarg's call-site in our orchestrator."
        ),
        "verification": {
            "command": (
                "source .venv/bin/activate && python -c \"import ast; "
                "ast.parse(open('backend/agents/orchestrator.py').read()); print('syntax OK')\" && "
                "! grep -E 'betas\\s*=' backend/agents/orchestrator.py | head -1"
            ),
            "success_criteria": [
                "no_betas_kwarg_passed_to_anthropic_Messages_create",
                "if_beta_feature_needed_use_client.beta.messages_or_extra_headers",
                "fresh_Claude_cycle_zero_betas_kwarg_errors"
            ],
            "live_check": "fresh Claude cycle log has zero `unexpected keyword argument 'betas'` errors"
        },
        "retry_count": 0,
        "max_retries": 3
    }
]

for p in data["phases"]:
    if p["id"] == "phase-27":
        for i, s in enumerate(p["steps"]):
            if s["id"] == "27.6.6":
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
