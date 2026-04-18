# Runbook: Broker (Alpaca) Outage

## Scope

Alpaca paper-trading API is unreachable, returning 5xx, or the
WebSocket trade-updates stream has been silent for >60s. Applies to
any component in pyfinagent that submits orders via
`backend/services/execution_router.py::ExecutionRouter` with
`EXECUTION_BACKEND=alpaca_paper` or `shadow`.

Does NOT apply to historical-price reads (those use BigQuery, see
`data_feed_outage.md`).

## Trigger

Detection signals (any one is sufficient to invoke the runbook):
1. `_alpaca_real_fill` raises an exception on 3 consecutive order
   submissions within 2 minutes.
2. Alpaca status page `status.alpaca.markets` reports a live
   incident on "Paper Trading API" or "Market Data API".
3. `backend/services/kill_switch.py` records a `broker_unreachable`
   trigger (currently wired into paper_trader execute path).
4. Slack / iMessage alert from sla_monitor.py (endpoint
   `/api/paper-trading/status` 5xx rate >20% over 3 minutes).

## Response Steps

1. **T+0 (within 1 min)**: Flip `EXECUTION_BACKEND` to `bq_sim` via
   the rollback primitive
   (`backend/services/execution_router.py::rollback_to_bq_sim()`).
   This stops all outbound broker calls immediately; BQ-sim fills
   continue to write to paper_trades so position continuity is
   preserved.

2. **T+1-2 min**: Trigger the kill-switch (`POST /api/paper-trading/
   kill-switch` with action `PAUSE`) so that no NEW orders queue
   during the outage. Paper-trading cycle will refuse to generate
   new signals while paused.

3. **T+3-5 min**: Open a post in the pyfinagent Slack + start a
   timer; cross-reference Alpaca status page. Capture the incident
   start timestamp + first failed `client_order_id` for later
   reconciliation.

4. **T+5 min onward**: Monitor Alpaca status every 5 min. When the
   status flips to Operational for 10 consecutive minutes AND
   3 test orders via `scripts/harness/paper_execution_parity.py
   --days 1` succeed with drift <1%, begin ramp-back:
   flip `EXECUTION_BACKEND` back to `shadow` (not `alpaca_paper`
   directly) and watch for 30 minutes with kill-switch still
   PAUSED.

5. **Recovery**: After 30 min of clean shadow fills, RESUME the
   kill-switch and step `EXECUTION_BACKEND` to `alpaca_paper` on
   the next cycle. Document the event in `handoff/dr_drill_log.md`
   with the measured RTO.

## Rollback

- `from backend.services.execution_router import rollback_to_bq_sim;
  r = rollback_to_bq_sim()` -- instant module-level helper.
- If the PAUSE itself is stuck (API returns error), SIGKILL the
  paper_trader process via `pkill -9 -f "backend.services.paper_
  trader"` and relaunch with `EXECUTION_BACKEND=bq_sim`.
- No orders are lost -- BQ sim continues appending to paper_trades
  with `source="bq_sim"` so the audit trail stays intact.

## RTO Target

**15 minutes** from first detection signal to trading resumed on
`bq_sim` fallback (Step 1-2 above). Recovery back to live-broker
mode is a soft objective, not a hard RTO.

## Last Drill

- 2026-04-18: tabletop drill completed; measured RTO 8 minutes
  (flip to bq_sim in 2 min, kill-switch PAUSE in 4 min, incident
  documented by 8 min). PASS.
- See `handoff/dr_drill_log.md` for full trace.
