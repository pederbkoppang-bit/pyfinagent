# Sprint Contract -- phase-31.0.3 (Smoketest Stage 3)

(Written to smoketest dir to avoid the optimizer-cron clobbering of
`handoff/current/contract.md`.)

**Step:** Stage 3 -- Gemini full-path on NVDA via AnalysisOrchestrator.
**Date:** 2026-05-20.
**Mode:** Loop PAUSED. NO production `analysis_results` write.

## Hypothesis

Invoking `AnalysisOrchestrator(settings).run_full_analysis("NVDA")`
with `_persist_analysis` mocked produces:
1. Non-error return (full report dict).
2. New rows in `pyfinagent_data.llm_call_log` with >=3 distinct agent
   tags.
3. NO new row in `financial_reports.analysis_results`.

## Baseline (pre-run)

`llm_call_log` last 24h: 16 rows, 0 distinct_agents (lite-only).

## Immutable success criteria

1. Orchestrator completes without raising.
2. >=10 new `llm_call_log` rows + >=3 distinct agent tags (full path
   would be >=20; lite_mode would be ~10-15).
3. No new `analysis_results` row for NVDA today (mocked persist).
4. Output persisted to STAGE_3_gemini_full_path_output.json.

## Plan

1. Capture baseline (done: 16 rows, 0 agents).
2. Background-spawn Python script that:
   - Patches `_persist_analysis` (try multiple patch paths).
   - Calls `run_full_analysis("NVDA")` under asyncio.
   - Writes output to STAGE_3_gemini_full_path_output.json.
3. Wait ~3-10 min.
4. Compute post-run delta + assertions.

## Hard guardrails

- Loop PAUSED.
- _persist_analysis mocked -> no analysis_results write.
- Vertex AI cost ~$0.20-$1 for full NVDA run.
- If orchestrator raises (Vertex auth/network), capture traceback and
  mark PARTIAL.

## References

- Morning goal Stage 3.
- `.claude/rules/backend-agents.md` -- 28-agent pipeline doc.
- `backend/agents/orchestrator.py:1398` -- run_full_analysis entry.
