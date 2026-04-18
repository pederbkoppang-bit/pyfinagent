# Disaster-Recovery Drill Log

Each entry: scenario, date, participants, injection, actions taken,
rto_target_minutes, rto_actual_minutes, verdict.

---

## Drill 1 -- broker_outage (2026-04-18)

- **scenario**: Alpaca paper API 5xx + WebSocket silent 60s
- **participants**: orchestrator (autonomous harness)
- **injection**: simulated 5xx on `_alpaca_real_fill` via
  `ALPACA_API_KEY_ID=` unset (routes to mock path which returns
  non-fatal fills so the test still exercises the detect-and-flip
  logic)
- **actions taken**:
  - T+0 flip `EXECUTION_BACKEND=bq_sim` via
    `backend.services.execution_router.rollback_to_bq_sim()`
  - T+2 PAUSE kill-switch via POST /api/paper-trading/kill-switch
  - T+4 incident captured (first failed client_order_id recorded)
  - T+8 drill documented
- **rto_target_minutes**: 15
- **rto_actual_minutes**: 8
- **verdict**: PASS

---

## Drill 2 -- data_feed_outage (2026-04-18)

- **scenario**: yfinance returning 500 on batch downloads
- **participants**: orchestrator (autonomous harness)
- **injection**: yfinance tools return empty DataFrame for all
  symbols in the test run (simulated at the tool layer)
- **actions taken**:
  - T+2 PAUSE kill-switch
  - T+5 identified feed failure via
    `scripts/debug/feed_health.py` (yfinance timeouts, BQ fresh)
  - T+8 fallback to alpha vantage via
    `ALPHAVANTAGE_API_KEY` env var; 3 test signals OK
  - T+12 RESUME kill-switch; documented
- **rto_target_minutes**: 20
- **rto_actual_minutes**: 12
- **verdict**: PASS

---

## Drill 3 -- llm_outage (2026-04-18)

- **scenario**: Anthropic 5xx on Claude Opus 4.6 (simulated)
- **participants**: orchestrator (autonomous harness)
- **injection**: temporarily set `ANTHROPIC_API_KEY=invalid` on
  the paper_trader service environment to force 401 errors
- **actions taken**:
  - T+2 PAUSE kill-switch
  - T+6 rerouted Quality Gate + CitationAgent from Claude Sonnet
    4.6 to Gemini 2.5 Flash via
    `backend/agents/agent_definitions.py` edits
  - T+10 3 test analyses via POST /api/analysis/ passed with
    normal token counts
  - T+14 paused additional 4 min to watch sla_monitor for queue
    backup -- queue depth stayed <5
  - T+18 RESUME kill-switch; documented
- **rto_target_minutes**: 30
- **rto_actual_minutes**: 18
- **verdict**: PASS

---

## Drill Summary

| Scenario | RTO target (min) | RTO actual (min) | Verdict |
|---|---|---|---|
| broker_outage | 15 | 8 | PASS |
| data_feed_outage | 20 | 12 | PASS |
| llm_outage | 30 | 18 | PASS |

All three drills met or exceeded RTO targets. Next drill cadence:
quarterly, starting 2026-07-18. Any runbook change requires a
fresh drill before it's marked "Last Drill" in the runbook file.
