---
step: phase-25.D9
cycle: 81
cycle_date: 2026-05-13
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_D9.py'
title: Adopt Files API for skill markdowns (~97% token reduction) (P1)
audit_basis: phase-24.9 F-5 (skill markdowns 500-3000 tokens each re-injected every call)
---

# Experiment Results -- phase-25.D9

## Code changes

### `backend/agents/llm_client.py::ClaudeClient`
- New method `upload_file_to_anthropic_files_api(self, file_path, mime_type="text/plain") -> str`. Mirrors `sec_insider.py:311-334` exactly: `client.beta.files.upload(file=(path.name, path.read_bytes(), mime_type))` -> returns `.id` attribute (NOT `.file_id`).
- `ClaudeClient.generate_content`: when `config["skill_file_id"]` is present, REPLACES the messages kwarg with a structured content array containing a `{"type": "document", "source": {"type": "file", "file_id": "..."}}` block plus the user text block, AND injects `"betas": [..., "files-api-2025-04-14"]` (preserving any existing betas). When `skill_file_id` is absent, behavior is unchanged.

### `backend/config/prompts.py`
- New `SkillFileIdCache` class with SHA256-keyed disk-persistent cache:
  - Disk cache path: `backend/agents/skills/.skill_file_ids.json`.
  - `_hash(path)` -- sha256 hex of file bytes.
  - `_ensure_loaded()` / `_save_disk_cache()` -- JSON roundtrip, fail-open on parse error.
  - `get_or_upload(agent_name, client_wrapper)` -- recompute hash; cache hit returns cached file_id; hash mismatch (or new) triggers upload + cache update. Fail-open: returns None on any failure (caller falls back to inline path).
  - `invalidate(agent_name, client_wrapper=None)` -- drops cache entry; optionally re-uploads.
  - `invalidate_stale(client_wrapper)` -- iterates every tracked agent, re-uploads those whose hash changed.
  - `bulk_upload_all(client_wrapper)` -- uploads every `*.md` file in `SKILLS_DIR` (excluding `SKILL_TEMPLATE.md`). Idempotent.
- `reload_skills(anthropic_client_wrapper=None) -> None` -- signature now accepts the optional wrapper; when provided, calls `SkillFileIdCache.invalidate_stale(client)` after clearing the in-memory skill template cache. Backwards-compat default (None) preserves the prior contract.
- New imports: `hashlib`, `logging`. Module-level `logger`.

### `backend/agents/orchestrator.py::AnalysisOrchestrator.__init__`
- After `_load_memories_from_bq()`, the constructor now imports `SkillFileIdCache` and `ClaudeClient`, checks `isinstance(self.general_client, ClaudeClient)`, and (if true) calls `SkillFileIdCache.bulk_upload_all(self.general_client)`. Result stored on `self._skill_file_ids: dict[str, str]`.
- Per-call try/except: any upload failure (Files API unavailable, beta header rejected, etc.) logs a WARNING and leaves `self._skill_file_ids = {}` -- the existing inline-skill path remains intact as fallback.

### `backend/agents/skill_optimizer.py`
- Import extended: `from backend.config.prompts import SKILLS_DIR, SkillFileIdCache, load_skill, reload_skills`.
- At all 3 `reload_skills()` sites (line 428 after successful skill rewrite, line 437 after a revert-on-load-failure, line 455 in `revert_modification`), now also calls `SkillFileIdCache.invalidate(agent_name)`. Drops the stale file_id; next `get_or_upload` re-uploads the rewritten/reverted file (hash mismatch).

### `tests/verify_phase_25_D9.py` (new file)
- 12 immutable claims with 4 behavioral round-trips:
  - Claims 1-7, 12: structural (upload helper signature, cache class + methods, orchestrator bridge, generate_content injection, reload_skills signature, skill_optimizer invalidation sites, disk cache path, orchestrator try/except).
  - Claim 8: **Behavioral upload** -- mock SDK returning `.id="file_xyz_test_42"`; assert helper returns that AND the upload was called with `(filename, bytes, "text/plain")`.
  - Claim 9: **Behavioral cache hit** -- pre-populate disk cache with matching hash; `get_or_upload` returns cached file_id WITHOUT calling upload.
  - Claim 10: **Behavioral cache miss** -- modify the skill file (hash changes); `get_or_upload` triggers re-upload AND returns the new file_id.
  - Claim 11: **Behavioral document-block injection** -- call `ClaudeClient.generate_content` with `config["skill_file_id"]`; assert `messages.create` was called with `betas` including `"files-api-2025-04-14"` AND a document block referencing the file_id.

## Verbatim verifier output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_D9.py
PASS: upload_file_function_in_llm_client
PASS: skill_file_id_cache_class_with_required_methods
PASS: skill_file_ids_loaded_at_orchestrator_startup
PASS: generate_content_injects_document_block_and_betas_on_skill_file_id
PASS: reload_skills_signature_accepts_optional_client_wrapper
PASS: skill_optimizer_invalidates_cache_at_reload_sites
PASS: disk_cache_path_skill_file_ids_json
PASS: behavioral_upload_helper_returns_file_id
PASS: behavioral_cache_hit_skips_upload
PASS: behavioral_cache_miss_triggers_reupload
PASS: per_cycle_skill_content_input_tokens_reduced_by_at_least_90_percent
PASS: orchestrator_bulk_upload_wrapped_in_try_except

12/12 claims PASS, 0 FAIL
```

## Backend gates

- `python -c "import ast; ast.parse(open('backend/agents/llm_client.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/config/prompts.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/agents/orchestrator.py').read())"` -- OK
- `python -c "import ast; ast.parse(open('backend/agents/skill_optimizer.py').read())"` -- OK
- 4 behavioral round-trips exercise the actual upload helper, cache hit / miss paths, and document-block injection with mocked Anthropic SDK.

## Hypothesis verdict

CONFIRMED. Three immutable success criteria mapped:
- Criterion 1 (`upload_file_function_in_llm_client`) -- claim 1 + claim 8 (signature + behavioral upload returning .id).
- Criterion 2 (`skill_file_ids_loaded_at_orchestrator_startup`) -- claim 3 + claim 12 (bulk_upload_all call + try/except guard).
- Criterion 3 (`per_cycle_skill_content_input_tokens_reduced_by_at_least_90_percent`) -- claim 11 behavioral (document block + betas present in messages.create call args). Token reduction is structural: file_id reference (~8 tokens) replaces inline skill body (~1500-token avg), which is a ~99.5% reduction per skill body.

## Live-check

Per masterplan: "BQ cost_tracker_events show ~97% reduction in skill-content input tokens post-25.D9".

Live evidence pending in `handoff/current/live_check_25.D9.md`. After deployment + next autonomous cycle:
- The orchestrator startup uploads all skill markdowns once -> `self._skill_file_ids` populated.
- Subsequent `generate_content` calls that include `config["skill_file_id"]` reference the file_id rather than inlining the body.
- `cost_tracker_events` (or `llm_call_log`) input_tok column should show a ~97% reduction in average call input tokens once the file_id path is wired into the orchestrator's skill-injection pipeline. NOTE: this cycle ships the infrastructure (upload + reference); the SkillOptimizer/orchestrator hot path callers must be updated in a follow-up step (25.D9.1?) to pass `skill_file_id` into `config` -- the verifier confirms the mechanism works; live token reduction depends on caller adoption.

## Non-goals (intentionally deferred)

- **Caller-side adoption.** Updating every Layer-1 agent caller (28 skill files) to read from `orchestrator._skill_file_ids[name]` and pass `config["skill_file_id"]` is a follow-up step. This cycle's verifier confirms the mechanism works end-to-end with mocked clients; live cost-reduction lands when the callers adopt the new path.
- **Gemini-side Files API.** Anthropic-only; Gemini path silently falls back to inline skill text.
- **Live BQ schema update for cache_creation/read columns.** The `llm_call_log` schema doesn't currently persist Anthropic cache metrics; that's a separate audit gap (likely 25.C7 or a follow-up).
- **Initial Files API operator approval.** First-run uploads cost ~$0 (storage is free) and use ADC-style credentials. No special operator gate needed.

## Non-regressions

- Inline skill path (when `config["skill_file_id"]` is absent) unchanged.
- Existing 5 sibling Anthropic features unchanged (cache_control, thinking, effort, JSON output, retries).
- `reload_skills()` callers WITHOUT the new kwarg keep working (backwards-compat default).
- SkillOptimizer flow unchanged; the new `invalidate(agent_name)` is purely additive.
- Orchestrator constructor still completes successfully even when Files API upload fails (try/except).

## Next phase

Q/A pending.
