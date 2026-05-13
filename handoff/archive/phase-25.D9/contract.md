# Sprint Contract -- phase-25.D9 -- Adopt Files API for skill markdowns

**Cycle:** phase-25 cycle 25 (P1 sprint)
**Date:** 2026-05-13
**Step ID:** 25.D9
**Priority:** P1
**Audit basis:** bucket 24.9 F-5 -- skill markdowns 500-3000 tokens each re-injected every call; Files API `file_id` reference eliminates

## Research-gate

Researcher spawned this cycle (agent a046483e7d2762e3c). Brief at
`handoff/current/research_brief.md`. Gate envelope: 5 sources read in full,
15 URLs, recency scan performed, 7 internal files inspected, gate_passed=true.

Key research conclusions:
- **Upload pattern mirrors `sec_insider.py:311-334`**: `client.beta.files.upload(file=(name, bytes, mime_type))` returns object with `.id` attribute (NOT `.file_id`).
- **MIME type for .md files = `text/plain`** -- there's no `text/markdown` in Anthropic's supported MIME table; all markdown uses `text/plain` in a `document` content block.
- **`betas=["files-api-2025-04-14"]` REQUIRED on every `messages.create`** that references a file_id. SDK injects the beta header on upload but NOT on messages.create.
- **Cost math:** file_id reference ~8 tokens vs ~1,490 tokens inline avg. 28-skill pipeline saves ~41,720 input tokens/analysis (~98.5% reduction).
- **NOT available on Vertex / Bedrock** -- guard behind `isinstance(client, ClaudeClient)`; Gemini path falls back to inline.
- **Cache invalidation via SHA256 hash** -- recompute on access; re-upload when hash changes. SkillOptimizer's three `reload_skills()` sites must invalidate the file_id cache.
- **Disk-sidecar cache** at `backend/agents/skills/.skill_file_ids.json` for cross-restart reuse without re-uploading unchanged skills.
- **Files persist indefinitely; no auto-expiration** -- storage free; per-call token billing at standard input rates.

## Hypothesis

Adding (a) `ClaudeClient.upload_file_to_anthropic_files_api(path, mime_type)`
helper, (b) `SkillFileIdCache` class in `backend/config/prompts.py` with
SHA256-keyed disk-persistent cache, (c) orchestrator-startup bulk-upload
of all skill files when the client is Claude, (d) `messages.create`
kwargs injection of `betas` + document block when `config["skill_file_id"]`
is provided, and (e) cache-invalidation at all 3 `reload_skills()` sites
-- delivers ~98.5% token reduction on the skill body per call without
breaking Gemini callers or SkillOptimizer's hot-reload behavior.

## Success criteria (verbatim from masterplan)

1. `upload_file_function_in_llm_client`
2. `skill_file_ids_loaded_at_orchestrator_startup`
3. `per_cycle_skill_content_input_tokens_reduced_by_at_least_90_percent`

Verification command (immutable):
`source .venv/bin/activate && python3 tests/verify_phase_25_D9.py`

Live check (per masterplan):
`BQ cost_tracker_events show ~97% reduction in skill-content input tokens post-25.D9`

## Plan

1. **`backend/agents/llm_client.py`** -- `ClaudeClient`:
   - Add method `upload_file_to_anthropic_files_api(self, file_path: str | Path, mime_type: str = "text/plain") -> str`. Mirror `sec_insider.py:311-334` exactly: `client.beta.files.upload(file=(path.name, path.read_bytes(), mime_type))` -> return `.id`.
2. **`backend/config/prompts.py`** -- new class:
   - `SkillFileIdCache` with classmethods: `_hash(path) -> sha256_hex`, `bulk_upload_all(client) -> dict[name, file_id]`, `get_or_upload(name, client) -> file_id`, `invalidate(name, client) -> None`, `invalidate_stale(client) -> None`, `_load_disk_cache() -> None`, `_save_disk_cache() -> None`.
   - Disk cache path: `backend/agents/skills/.skill_file_ids.json` -- shape `{agent_name: {hash: str, file_id: str}}`.
   - `invalidate_stale` re-hashes each tracked agent and re-uploads if the hash differs from cache.
3. **`backend/config/prompts.py::reload_skills`** -- update signature to accept optional `anthropic_client=None`. When provided, calls `SkillFileIdCache.invalidate_stale(client)` after clearing `_skill_cache`.
4. **`backend/agents/orchestrator.py::AnalysisOrchestrator.__init__`** -- after the existing `_load_memories_from_bq()` call (around line 437), check `isinstance(self.general_client, ClaudeClient)` and if so, attempt `SkillFileIdCache.bulk_upload_all(client._get_client())`. Store on `self._skill_file_ids: dict[name, file_id]`. Per-call try/except (fail-open: log warning + empty dict if upload fails -> the existing inline-skill path stays intact as fallback).
5. **`backend/agents/llm_client.py::ClaudeClient.generate_content`** -- after the kwargs dict build (before the `messages.create` call), check `config.get("skill_file_id")`. If present, REPLACE the messages kwarg with a structured content array containing the document block + the user text block, AND add `"betas": ["files-api-2025-04-14"]` to kwargs (preserving any existing betas).
6. **`backend/agents/skill_optimizer.py`** -- at the 3 `reload_skills()` sites (lines 428, 436, 454), pass the appropriate client so the file_id cache is invalidated on optimizer rewrites. If the client isn't readily available in those scopes, call `SkillFileIdCache.invalidate(agent_name, client)` per-agent directly (depending on which is cleaner).
7. **Verifier** -- `tests/verify_phase_25_D9.py` -- 10+ claims:
   - Claim 1: `ClaudeClient.upload_file_to_anthropic_files_api` signature.
   - Claim 2: `SkillFileIdCache` class declared in `prompts.py` with the documented classmethods.
   - Claim 3: Orchestrator __init__ contains the bulk-upload bridge.
   - Claim 4: ClaudeClient.generate_content injects `"betas"` + document block when `config["skill_file_id"]` is set.
   - Claim 5: `reload_skills` signature accepts `anthropic_client=None` (backwards-compat default).
   - Claim 6: `skill_optimizer.py` invalidates cache at the 3 reload sites.
   - Claim 7: Disk cache path is `backend/agents/skills/.skill_file_ids.json`.
   - Claim 8: **Behavioral upload** -- mock `client.beta.files.upload` returning an object with `.id="file_xyz"`; assert helper returns `"file_xyz"`.
   - Claim 9: **Behavioral cache hit** -- pre-populate the disk cache with a matching hash; `get_or_upload` returns the cached file_id WITHOUT calling upload.
   - Claim 10: **Behavioral cache miss** -- modify the skill file (change content -> hash mismatch); `get_or_upload` triggers re-upload.
   - Claim 11: **Behavioral document-block injection** -- mock the Anthropic client; assert `betas` includes `"files-api-2025-04-14"` and the messages kwarg has the document block when `config["skill_file_id"]` is set.
   - Claim 12: **Behavioral fallback** -- bulk_upload raising does NOT crash the orchestrator (`_skill_file_ids = {}` and existing inline path still works).

## Non-goals

- No live upload in this cycle's CI -- behavioral tests use mocked `beta.files.upload`.
- No Gemini-side change (Files API not available on Vertex).
- No BQ schema for cache_invalidation telemetry; the disk JSON cache is sufficient.
- No removal of the existing inline-skill path -- it remains the fallback when `skill_file_id` is not in config.

## References

- `handoff/current/research_brief.md` -- full brief
- `backend/tools/sec_insider.py:311-334` -- canonical upload pattern
- `backend/config/prompts.py:18,21,24,55,66` -- skill loader + cache anchor points
- `backend/agents/llm_client.py:1104-1300` -- ClaudeClient.generate_content (insertion points)
- `backend/agents/orchestrator.py:317-438` -- `AnalysisOrchestrator.__init__` (bulk-upload insertion)
- `backend/agents/skill_optimizer.py:428,436,454` -- 3 `reload_skills()` sites (cache-invalidation insertion)
- Anthropic Files API docs (cited in brief)
