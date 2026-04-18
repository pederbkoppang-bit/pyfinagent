# Contract -- Cycle 71 / phase-4.7 step 4.7.3

Step: 4.7.3 MAS Monitoring view: per-agent latency, cost, heartbeat

## Hypothesis

Gap analysis (Explore): frontend `/agents` page already rendered a
live event stream + agent map, but:
- EVENT_STYLES covered 12 of 14 event types defined in
  `backend/agents/mas_events.py` (missing `tool_result` + `thinking`).
- Agent Map table had no per-agent latency, cost, or heartbeat
  columns.
- `cost_summary` from `/api/mas/dashboard` was fetched by the backend
  but never consumed by the frontend.

Closing these three gaps + shipping an audit script satisfies all
three immutable criteria without touching the backend event schema.
Heartbeat is derived from the last event age per agent (green <60s,
amber <5min, red otherwise) -- event-freshness is a proper liveness
proxy for agents whose whole observable surface is the MAS event bus.

## Scope

Files modified / created:

1. **MODIFY** `frontend/src/app/agents/page.tsx`
   - Extend EVENT_STYLES with `tool_result` and `thinking`.
   - Add `costSummary` state + fetch from `/api/mas/dashboard`
     cost_summary block.
   - Extend the Agent Map table with 3 new columns: Latency (avg ms
     from `duration_ms` in events), Cost ($ from cost_summary), and
     Heartbeat (age-band dot + label).
   - Mark cells with `data-col` and `data-cell` attributes so the
     audit script can verify presence without a runtime browser.

2. **NEW** `scripts/audit/mas_ui_events.py`
   - Parses `backend/agents/mas_events.py` module docstring to
     extract canonical event-type list.
   - Parses the EVENT_STYLES object literal in `agents/page.tsx`
     by walking balanced braces (handles TS type annotation
     `Record<string, { ... }>` correctly).
   - Regex-checks the three data-attribute markers for latency, cost,
     heartbeat columns.
   - `--check` flag exits 1 on any violation.
   - Emits `handoff/mas_ui_events.json` with the full audit report.

## Immutable success criteria

1. `events_rendered_1to1_with_mas_events_py`: every event-type named
   in `mas_events.py` docstring maps 1:1 to EVENT_STYLES keys
   (no missing, extras are allowed only if they're explicit aliases).
2. `per_agent_latency_visible`: table has `data-col="latency"` and
   `data-cell="latency"` markers rendering a computed avg latency
   per agent.
3. `per_agent_cost_visible`: table has `data-col="cost"` +
   `data-cell="cost"` AND the page consumes `cost_summary` from the
   dashboard endpoint (not just a display-only placeholder).

## Verification (immutable, from masterplan.json)

    python scripts/audit/mas_ui_events.py --check

Audit passes when verdict == "PASS".

## References

- backend/agents/mas_events.py (14 event types in module docstring)
- backend/api/mas_events.py (/api/mas/dashboard -> cost_summary)
- backend/agents/cost_tracker.py (per-agent cost breakdown source)
- .claude/rules/frontend.md Glass Box convention (every I/O visible)
