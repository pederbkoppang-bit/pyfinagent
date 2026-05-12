---
step: phase-24.9
cycle: 13
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_9.py'
title: LLM provider conformance audit (Claude + Gemini) (P2)
---

# Experiment Results — phase-24.9

**Action:** READ-ONLY. Findings + brief + contract. No code changes.

## Verbatim verifier output (after canonical URL fix)

```
=== phase-24.9 (llm-conformance) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_9_llm_conformance_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_anthropic_com_engineering
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_9_cycle_entry
  [PASS] executive_summary_section_present
  [PASS] findings_audits_prompt_caching_depth_ephemeral_1h
  [PASS] findings_audits_extended_thinking_budgets
  [PASS] findings_audits_tool_use_loop_max_turns
  [PASS] findings_audits_structured_output_schema_enforcement
  [PASS] findings_audits_google_search_grounding_usage
  [PASS] findings_identifies_unused_anthropic_features_batch_files_citations
FAIL (15/16) EXIT=1
```

15/16 PASS. Log-last only FAIL.

## Hypothesis verdict
CONFIRMED with 3 bugs + 3 unused features:
- **Bug 1**: `cost_tracker.py:147` 1.25x instead of 2.0x for cache writes (60% under-report)
- **Bug 2**: System prompt ~35 tokens at `llm_client.py:751-759` — below 4096 cache threshold; silently skipped
- **Bug 3**: In-process cache-hit counters at `llm_client.py:709-713` are per-instance; use `UsageMeta.cache_read_input_tokens`
- **Unused**: Batch API (50% discount), Files API (file_id), native Citations (no extra LLM call)
- **Correct**: Gemini grounding gated correctly; thinking budgets defensible (Synthesis 4096 candidate for reduction)

## Phase-25 candidates (6)
- 25.A9 (P1) — Fix cache-write premium 1.25x → 2.0x
- 25.B9 (P1) — Bump system prompt above 4096-token cache threshold
- 25.C9 (P1) — Adopt Batch API (50% savings on non-interactive steps)
- 25.D9 (P1) — Adopt Files API for skill markdowns
- 25.E9 (P1) — Adopt native Citations (deprecate CitationAgent)
- 25.F9 (P2) — Profile + tune Synthesis thinking (4096 → 2048)

## Next: Q/A
