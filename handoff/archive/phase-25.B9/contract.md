# Sprint Contract -- phase-25.B9 -- Bump system prompt above 4096-token cache threshold

**Cycle:** phase-25 cycle 24 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.B9
**Priority:** P1
**Audit basis:** bucket 24.9 F-2 -- `llm_client.py:851-860` system prompt ~10-400 tokens; below 4096-token threshold; `cache_control` silently no-ops

## Research-gate

Researcher spawned this cycle (agent a36e6b0584dde92ba). Brief at
`handoff/current/research_brief.md`. Gate envelope: 7 sources read in full,
17 URLs, recency scan performed, 7 internal files inspected, gate_passed=true.

Key research conclusions:
- **Cache thresholds confirmed:** Opus 4.7=4096, Sonnet 4.6=2048, Haiku 4.5=4096. Below these, `cache_control` silently no-ops. Current pyfinagent system prompt is ~10-400 tokens -- 100% cache misses on every call.
- **Target prompt size:** **4500-5000 tokens** (~15,750-17,500 chars) -- clears the 4096 floor for all models with ~10% safety margin.
- **DO NOT inline `skills/*.md` content** -- SkillOptimizer modifies these; including them invalidates cache on every optimization cycle.
- **DO NOT inline JSON schemas** -- they change per Pydantic model.
- **Inline constant approach** preferred over file-load (file-load deferred to 25.D9 Files API for compound savings).
- **Token estimation:** Anthropic heuristic `chars / 3.5`. Exact count via `client.messages.count_tokens(...)` (free, rate-limited).
- **Cache hit-rate proxy:** `cache_read / (cache_read + cache_creation)` -- the cost_tracker already records both fields at `cost_tracker.py:90-96`.
- **Insertion point:** `llm_client.py:852` -- replace literal `"You are a financial analysis AI."` with `_HOUSE_INSTRUCTIONS` constant defined near the top of the file.

## Hypothesis

Adding a substantive 4500-5000-token `_HOUSE_INSTRUCTIONS` constant
covering persona + behavioral mandates + JSON output rules + financial-
analysis reasoning framework + agent interaction rules + anti-patterns +
safety anchor, prefixed to every system prompt in the ClaudeClient path
-- moves the system prompt above the 4096-token cache write threshold
on Opus 4.7 / Haiku 4.5 and above the 2048-token threshold on Sonnet
4.6, causing `cache_control={"type":"ephemeral","ttl":"1h"}` to actually
register. Subsequent calls within 1h read the cached prefix at 0.1x cost
(90% discount on input tokens).

## Success criteria (verbatim from masterplan)

1. `system_prompt_consolidates_skill_and_schema_above_4096_tokens`
2. `usage_meta_cache_read_input_tokens_grows_post_25_B9`
3. `cache_hit_rate_proxy_increases_to_30_percent_or_higher`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_B9.py`

Live check (per masterplan):
`BQ cost_tracker_events show cache_read_input_tokens > 0 in cycles post-25.B9`

## Plan

1. **`backend/agents/llm_client.py`** -- single-file edit:
   - Add module-level `_HOUSE_INSTRUCTIONS: str` constant near the top of the file (before `class OpenAIClient` or near other module constants). Content: substantive 4500-5000-token financial-analysis house-instructions including persona / mandates / JSON output rules / reasoning framework / interaction rules / anti-patterns / safety anchor.
   - Update `ClaudeClient.generate_content` (line 852 area): replace `system_prompt = "You are a financial analysis AI."` with `system_prompt = _HOUSE_INSTRUCTIONS`.
   - The existing schema-append block at lines 853-860 stays unchanged -- dynamic content appends AFTER the cached prefix.
   - The existing `cache_control={"type":"ephemeral","ttl":"1h"}` at line 879 stays unchanged -- no behavioral change in cache wiring, just the prompt size that lets it register.
2. **Verifier** -- `tests/verify_phase_25_B9.py` -- 10+ claims:
   - Claim 1: `_HOUSE_INSTRUCTIONS` constant exists in `llm_client.py`.
   - Claim 2: `_HOUSE_INSTRUCTIONS` length >= 14_336 chars (= 4096 tokens * 3.5 chars/token Anthropic heuristic). Some headroom: ~15_000-17_500 chars target.
   - Claim 3: `_HOUSE_INSTRUCTIONS` contains key sections: persona header, JSON output rules, safety anchor language, FACT_LEDGER / reasoning framework references.
   - Claim 4: `system_prompt = _HOUSE_INSTRUCTIONS` assignment present in `generate_content` (replacing the old short literal).
   - Claim 5: Old literal `"You are a financial analysis AI."` NOT present as a bare string in the `system_prompt = ...` line (still allowed in comments / docstrings).
   - Claim 6: `cache_control={"type": "ephemeral", "ttl": "1h"}` still present (no regression on caching wiring).
   - Claim 7: `_HOUSE_INSTRUCTIONS` does NOT include `skills/*.md` content or Pydantic schemas (grep-check for common skill / schema markers).
   - Claim 8: **Behavioral token-count check** -- estimate tokens via the chars/3.5 heuristic; assert >= 4096.
   - Claim 9: **Behavioral cache-hit-rate proxy** -- mock anthropic client, simulate 2 sequential calls returning `cache_creation_input_tokens=5000` then `cache_read_input_tokens=5000`; cost_tracker proxy = 5000/(5000+5000) = 0.5 >= 0.30. Use `cost_tracker.CostTracker.record(...)` directly.
   - Claim 10: **Behavioral no-regression** -- `ClaudeClient(model="claude-sonnet-4-6", api_key="sk-test", enable_prompt_caching=True)` constructs without error; the `_get_client` lazy-init still works (no module-level Anthropic import side effects).

## Non-goals

- No Files API integration (25.D9 will move `_HOUSE_INSTRUCTIONS` to a Files-API-cached attachment for compound savings).
- No removal of the per-cycle `skills/*.md` SkillOptimizer hot path -- those continue to flow through user-message content, not system prompt.
- No change to non-Claude paths (OpenAIClient / GitHubModelsClient / GeminiClient unaffected).
- No new BQ schema or migration.

## References

- `handoff/current/research_brief.md` -- full brief
- `backend/agents/llm_client.py:751-900` (ClaudeClient.generate_content), :852 (target literal), :874-881 (cache_control wiring), :866-873 (existing comment explaining the gap)
- `backend/agents/cost_tracker.py:90-167` (cache_creation_input_tokens + cache_read_input_tokens tracking; `MODEL_PRICING` for cost math)
- Anthropic docs: prompt-caching + token-counting (cited in brief)
- arXiv 2601.06007v2 -- "Don't Break the Cache" (linear savings with prompt size)
- OWASP LLM01:2025 -- prompt-injection safety mitigation for long system prompts
