---
step: 25.D9.1
slug: caller-side-files-api-adoption
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.D9.1

## Step ID + masterplan reference

`25.D9.1` -- "Caller-side Files API adoption (skill_file_id wiring in
run_*_agent calls)" (P2, harness_required, depends on `25.D9` done
at cycle 81).

## Research-gate summary

Tier=moderate. Brief at `handoff/current/research_brief.md`,
`gate_passed=true` (5 sources fetched in full, 12 URLs collected,
recency scan present). Key findings:
- Files API beta header `files-api-2025-04-14` unchanged.
- Document blocks support `cache_control` (independent of system prompt cache).
- Sonnet 4.6 cache minimum is 1024 tokens; larger skill markdowns qualify.
- Token reduction is 98-99.5% per call (5K-15K skill tokens -> ~8 token file_id ref).
- 11 enrichment `run_*_agent` callers at `orchestrator.py:835-919` are the gap.
- `_skill_file_ids` is empty on Gemini path -> helper must return None.

## North-star alignment

Directly cuts the LLM-cost denominator in Net System Alpha. Compounds with:
- 25.B9 (cycle 80, system-prompt cache): same prompts now also pay 0.1x.
- 25.C9 (cycle 84) + 25.C9.1 (cycle 103, today): batch path now uses tiny doc-block refs instead of inlined markdown.
- Backtests with 28-agent x 10-ticker pipelines see the largest dollar savings.

## Hypothesis

Adding a small `_skill_gen_config(skill_stem) -> dict | None` helper to
AnalysisOrchestrator + wiring each of the 11 enrichment call sites to
pass `generation_config=self._skill_gen_config(...)` makes Claude
provider runs route through the cached `file_id` document block
(already supported in `llm_client.py:1220-1247` from 25.D9). Gemini
provider runs see `_skill_file_ids` empty -> helper returns `None` ->
existing inline path preserved.

## Success criteria (verbatim from masterplan.json)

> `orchestrator_has_skill_gen_config_helper`
>
> `enrichment_agents_pass_generation_config_with_skill_file_id`
>
> `helper_returns_none_when_skill_file_ids_empty_gemini_fallback`

## Plan steps

1. **Add `_skill_gen_config(skill_stem: str) -> dict | None`** method on
   `AnalysisOrchestrator` (place near `_run_enrichment_batch`).
2. **Wire 11 enrichment call sites** in `run_*_agent` methods to pass
   `generation_config=self._skill_gen_config("<skill_stem>")`:
   - `run_insider_agent` -> `"insider_agent"`
   - `run_options_agent` -> `"options_agent"`
   - `run_social_sentiment_agent` -> `"social_sentiment_agent"`
   - `run_patent_agent` -> `"patent_agent"`
   - `run_earnings_tone_agent` -> `"earnings_tone_agent"`
   - `run_alt_data_agent` -> `"alt_data_agent"`
   - `run_sector_agent` -> `"sector_agent"`
   - `run_nlp_sentiment_agent` -> `"nlp_sentiment_agent"`
   - `run_anomaly_agent` -> `"anomaly_agent"`
   - `run_scenario_agent` -> `"scenario_agent"`
   - `run_quant_model_agent` -> `"quant_model_agent"`
3. **Verifier** -- `tests/verify_phase_25_D9_1.py` with 5 claims:
   - C1: `_skill_gen_config` method exists with the right signature.
   - C2: 11+ call sites pass `generation_config=self._skill_gen_config(...)`.
   - C3: behavioral -- when `_skill_file_ids` is empty (Gemini path),
     `_skill_gen_config(any_stem)` returns `None`.
   - C4: behavioral -- when `_skill_file_ids = {"insider_agent": "file_xyz"}`,
     `_skill_gen_config("insider_agent")` returns `{"skill_file_id": "file_xyz"}`.
   - C5: behavioral -- missing stem returns `None` (not KeyError).

## Files

| File | Action |
|------|--------|
| `backend/agents/orchestrator.py` | Add `_skill_gen_config` helper + 11 call-site wires |
| `tests/verify_phase_25_D9_1.py` | NEW (5 claims) |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_D9_1.py
```

## Live-check

`Run a Claude-mode analysis; observe BQ cost_tracker rows show skill-token input drop to ~50 tokens per agent (was 700-1500)`.

## Risks + mitigations

- **Risk**: skill-stem keys differ between `bulk_upload_all` and the
  caller wiring (e.g., underscore-vs-dash mismatch).
  **Mitigation**: research brief identified `"insider_agent"` etc. as the
  canonical stems. C4 behavioral test confirms one stem end-to-end.
- **Risk**: existing tests that mock `generate_content` may need to
  accept the new `generation_config` kwarg.
  **Mitigation**: `_generate_with_retry` already accepts
  `generation_config` (line 511); the new helper just *populates* it
  when applicable. Pre-existing tests aren't affected.
- **Risk**: 1h prompt cache compounding (`cache_control` on document
  block) is documented as a follow-up in this cycle, NOT shipped here.
  **Mitigation**: criterion does not require cache_control on the doc
  block. Future 25.D9.2 can add cache_control once 25.D9.1 is in.

## References

- `handoff/current/research_brief.md` (5 sources, gate_passed=true)
- `backend/agents/orchestrator.py:467-480, 835-919` (skill_file_ids + call sites)
- `backend/agents/llm_client.py:1092-1247` (upload + skill_file_id routing)
- `backend/config/prompts.py:36-172` (SkillFileIdCache + bulk_upload_all)
- `.claude/masterplan.json::25.D9.1`
