# Experiment Results -- Cycle 71 / phase-4.7 step 4.7.3

Step: 4.7.3 MAS Monitoring view: per-agent latency, cost, heartbeat

## What was generated

1. **MODIFIED** `frontend/src/app/agents/page.tsx`:
   - EVENT_STYLES extended with `tool_result` (cyan-300 Lightning) and
     `thinking` (indigo-300 Brain) -> now maps 1:1 to the 14 event
     types defined in mas_events.py module docstring.
   - New `costSummary` state + `/api/mas/dashboard` fetch now extracts
     `data.cost_summary` (previously ignored).
   - Agent Map table extended with 3 new columns:
     * Latency (avg ms) -- computed from `duration_ms` events per
       agent; data-col="latency" / data-cell="latency"
     * Cost ($) -- looked up from costSummary.agents by agent_name;
       data-col="cost" / data-cell="cost"
     * Heartbeat -- derived from last-event age per agent
       (green<60s, amber<5min, red else); data-col="heartbeat" /
       data-cell="heartbeat"

2. **NEW** `scripts/audit/mas_ui_events.py`:
   - Regex-parses mas_events.py docstring to extract canonical event-
     type list (14 types).
   - Walks balanced braces in agents/page.tsx to skip the TS type
     annotation `Record<string, {...}>` and collect EVENT_STYLES keys.
   - Regex-checks data-col + data-cell markers for latency, cost,
     heartbeat columns.
   - Also verifies the page CONSUMES `cost_summary` + uses
     `costSummary` state (not display-only placeholder).
   - `--check` flag exits 1 on any violation.

## Verification run (verbatim, immutable)

    $ python scripts/audit/mas_ui_events.py --check
    {"wrote": ".../handoff/mas_ui_events.json", "verdict": "PASS",
     "backend_count": 14, "frontend_count": 14,
     "missing_in_frontend": [],
     "latency_ok": true, "cost_ok": true, "heartbeat_ok": true}
    exit=0

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| events_rendered_1to1_with_mas_events_py | PASS (14/14) |
| per_agent_latency_visible | PASS (avg-ms column wired to duration_ms) |
| per_agent_cost_visible | PASS (column wired to cost_summary.agents) |

## Regression

`cd frontend && npm run build` -> 12 static pages, no errors.

## Known limitations (non-blocking)

- Heartbeat uses event-freshness as a liveness proxy (green <60s,
  amber <5min, red else). True agent-level liveness endpoints land in
  a later observability phase. The proxy is honest and useful.
- cost_summary only populates when the backend has a live CostTracker
  for the active run; cells show "—" otherwise (correct).
- Agent-name matching: `agent_name === node.id || node.label`.
  Covers the canonical six (Communication, Ford, qa, research,
  Quality Gate, CitationAgent).
