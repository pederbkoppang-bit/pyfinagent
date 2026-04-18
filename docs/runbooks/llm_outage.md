# Runbook: LLM Provider Outage

## Scope

One or more LLM providers are unreachable, returning 5xx, or auth-
failing. Covers Vertex AI (Gemini), Anthropic (Claude),
OpenAI, GitHub Models. The `backend/agents/llm_client.py::
make_client()` multi-provider router is the single chokepoint.

Does NOT apply to paper-trading execution (see `broker_outage.md`)
or BQ data reads (see `data_feed_outage.md`).

## Trigger

1. `backend/agents/cost_tracker.py` records >5 consecutive `llm_
   error` rows for one provider within 5 min.
2. `GET /api/analysis/<id>` 5xx rate > 20% over 5 min (analysis
   pipeline stalling on LLM calls).
3. Provider status page (Vertex AI / Anthropic / OpenAI) reports
   an active incident on the model tier we use.
4. `sla_monitor.py` Slack-bot alert for "analysis queue depth > 50"
   (the ticket_queue backs up when LLMs stall).

## Response Steps

1. **T+0 (within 2 min)**: Pause the autonomous loop so new
   analyses stop queueing: `POST /api/paper-trading/kill-switch`
   with action PAUSE. Analysis pipeline is a feeder for signals;
   if signals dry up, paper_trader simply doesn't emit new orders.

2. **T+2-5 min**: Identify the failing provider. The cost tracker
   logs the `provider` column per LLM call. If only one provider
   is affected, reconfigure the affected agent(s) to use a
   different provider via `backend/agents/agent_definitions.py`:
   - Claude down -> route to Gemini or GitHub Models (cheaper
     fallback).
   - Vertex AI down -> route to Anthropic or GitHub Models.
   - OpenAI down -> route to Claude.
   Changes take effect on next cycle.

3. **T+5-10 min**: If ALL LLM providers are down (rare; usually
   one at a time) fall back to quant-only mode: set
   `ENABLE_PIPELINE_MODE=quant_only` env var so the orchestrator
   skips debate / risk-judge / synthesis agents and uses only
   the quant_model.py ML signal. This continues to produce
   paper-trading recommendations without any LLM calls.

4. **T+10-20 min**: Monitor the affected provider's status page.
   When status flips to Operational for 15 min AND 3 test analysis
   runs via `POST /api/analysis/` succeed with normal token counts,
   flip the agent config back to the primary provider.

5. **T+20 min onward**: Resume. Unset `ENABLE_PIPELINE_MODE`;
   RESUME the kill-switch; tail `sla_monitor` for 30 min to
   confirm no analysis-queue backup. Document in
   `handoff/dr_drill_log.md`.

## Rollback

- Provider swap is done via agent-definition config, not code
  deploy. A bad swap can be reverted by reloading the previous
  agent_definitions value.
- Quant-only mode is fully degraded but correct: the ML signal
  has historical Sharpe >0.6 and DSR >0.7 on walk-forward, so
  running in this mode for a day is safe.
- If none of the above works, terminate the paper_trader process
  and leave positions untouched; existing positions continue to
  mark-to-market against BQ prices without needing LLM calls.

## RTO Target

**30 minutes** from detection to trading resumed on a working
provider OR switched to quant-only mode. LLM providers commonly
have 15-60 min outages so this RTO matches the external SLO of
the upstream services.

## Last Drill

- 2026-04-18: tabletop drill simulating Anthropic 5xx. Measured
  RTO 18 minutes (PAUSE at T+2, agent route to Gemini at T+6,
  3 test analyses PASS at T+14, RESUME at T+18). PASS.
- See `handoff/dr_drill_log.md` for full trace.
