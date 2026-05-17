#!/usr/bin/env python3
"""Insert phase-27.5.2 after phase-27.5.1 in masterplan.json.

Born from phase-27.5.1 / cycle #7 outcome 2026-05-17: parallelism
(concurrency=8) brought 15-ticker cycle wall time from ~120 min (serial)
to ~23 min — well within the 1800s budget AND completed ALL 8 steps
including Step 9 Learning + 1 executed trade. BUT only 10/15 tickers
persisted because the daily cost-budget hard-block (cost_budget_daily_usd
defaulting to $5) tripped at $5.15 mid-cycle.

Raising the daily cap to $25 (5x) and promoting it from a getattr-default
to a proper Settings field. Still conservative for an autonomous LLM
trading system; expected full-cycle Gemini Flash cost is ~$1-3.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

MP = Path(".claude/masterplan.json")
data = json.loads(MP.read_text(encoding="utf-8"))

existing_ids = {s["id"] for p in data["phases"] for s in p.get("steps", [])}
if "27.5.2" in existing_ids:
    print("27.5.2 already present")
    raise SystemExit(0)

NEW = {
    "id": "27.5.2",
    "name": "Raise daily cost-budget cap (cost_budget_daily_usd) so 15-ticker full Gemini cycle doesn't trip mid-batch",
    "status": "pending",
    "harness_required": True,
    "priority": "P0",
    "depends_on_step": "27.5.1",
    "audit_basis": (
        "phase-27.5.1 cycle #7 (3e90d15e) 2026-05-17 completed all 8 "
        "steps end-to-end (first cycle in session to do so, including "
        "Step 9 Learning + 1 executed trade) but persisted only 10/15 "
        "tickers because cost_budget_daily_usd defaulted to $5 and was "
        "tripped at $5.15 by the concurrent-8 batch. The 5 missed "
        "tickers (GLW, SNDK, WDC, LITE, CIEN) all failed with "
        "'cost_budget tripped (cached): reason=daily $5.15 >= cap $5.00'. "
        "Raising cap to $25 (5x) — still conservative for an autonomous "
        "Gemini Flash trading system."
    ),
    "verification": {
        "command": (
            "source .venv/bin/activate && python -c \""
            "from backend.config.settings import Settings; "
            "s=Settings(); "
            "assert s.cost_budget_daily_usd >= 20.0, f'cap too low: {s.cost_budget_daily_usd}'; "
            "assert hasattr(s, 'cost_budget_monthly_usd'), 'monthly cap missing'; "
            "print(f'PASS daily=${s.cost_budget_daily_usd} monthly=${s.cost_budget_monthly_usd}')\""
        ),
        "success_criteria": [
            "cost_budget_daily_usd_added_to_Settings_as_proper_field",
            "cost_budget_monthly_usd_also_added_for_completeness",
            "default_daily_cap_at_least_20_usd",
            "field_pickup_via_env_var_override_works",
            "fresh_cycle_does_not_trip_cost_budget_mid_batch"
        ],
        "live_check": (
            "fresh run-now cycle with standard=gemini-2.5-flash + concurrency=8 "
            "completes status=completed AND persists >=14 of 15 analyses to BQ "
            "without any 'cost_budget tripped' log lines"
        )
    },
    "retry_count": 0,
    "max_retries": 3
}

for p in data["phases"]:
    if p["id"] == "phase-27":
        for i, s in enumerate(p["steps"]):
            if s["id"] == "27.5.1":
                p["steps"].insert(i + 1, NEW)
                print(f"inserted 27.5.2 after 27.5.1 at index {i+1}")
                break
        break

data["updated_at"] = datetime.now(timezone.utc).isoformat()
MP.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("OK")
