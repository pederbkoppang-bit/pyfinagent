# Sprint Contract — phase-24.9 — LLM Provider Conformance

**Cycle:** phase-24 cycle 13
**Date:** 2026-05-12
**Step ID:** 24.9
**Priority:** P2

## Research-gate
`gate_passed: true` (tier=complex). 7 sources: Anthropic prompt-caching, extended-thinking, batch-processing, tool-use, Files+Batch guide, Claude prompt-caching practical guide, Gemini thinking mode.

```json
{"tier":"complex","external_sources_read_in_full":7,"snippet_only_sources":10,"urls_collected":17,"recency_scan_performed":true,"internal_files_inspected":5,"gate_passed":true}
```

## Hypothesis
Infrastructure largely correct. Features (batch, files, citations) unused. Thinking budgets may need tuning.

**Researcher verdict: CONFIRMED with 3 bugs + 3 unused features:**
- **Bug 1**: `cost_tracker.py:147` charges 1.25x for cache writes; should be 2.0x since `llm_client.py:773-779` already uses `ttl: "1h"`. ~60% cost under-report on writes.
- **Bug 2**: System prompt at `llm_client.py:751-759` is ~35 tokens — below Anthropic's 4096 minimum cache threshold; `cache_control` sent but silently skipped.
- **Bug 3**: In-process `_cache_hits/_cache_misses` counters at `llm_client.py:709-713` are misleadingly low (per-instance, while server-side cache reads occur across instances). Use `UsageMeta.cache_read_input_tokens` for truth.
- **Unused 1**: Batch API (50% discount) — non-interactive Steps 1-7 in `multi_agent_orchestrator.py` could batch
- **Unused 2**: Files API — skill markdowns re-sent every call; `file_id` reference would eliminate
- **Unused 3**: Native Citations — bespoke `CitationAgent` at `multi_agent_orchestrator.py:1284-1333` runs an EXTRA LLM call

**Correct/working:**
- Gemini grounding correctly Gemini-only (`orchestrator.py:407-419` gates on `client.supports_grounding`)
- Thinking budgets 8192/4096 mostly defensible; Synthesis 4096 candidate for reduction

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. findings_audits_prompt_caching_depth_ephemeral_1h
12. findings_audits_extended_thinking_budgets
13. findings_audits_tool_use_loop_max_turns
14. findings_audits_structured_output_schema_enforcement
15. findings_audits_google_search_grounding_usage
16. findings_identifies_unused_anthropic_features_batch_files_citations

**Verifier:** `python3 tests/verify_phase_24_9.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 54 log
5. live_check_24.9.md
6. Flip
