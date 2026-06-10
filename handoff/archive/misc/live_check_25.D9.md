# Live-check placeholder -- phase-25.D9

**Step:** 25.D9 -- Adopt Files API for skill markdowns
**Date:** 2026-05-13

## Live-check field (per masterplan)
> "BQ cost_tracker_events show ~97% reduction in skill-content input tokens post-25.D9"

## Pre-deployment evidence
- 12/12 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_D9.py`).
- 4 behavioral round-trips: upload helper returns `.id`, cache hit skips upload, cache miss triggers re-upload, generate_content injects document block + `files-api-2025-04-14` beta header.
- Backend AST clean for all 4 touched files.

## Two-phase deployment

This step ships the INFRASTRUCTURE: upload helper + cache + orchestrator-startup bulk upload + `generate_content` document-block injection path. The CALLER-SIDE adoption (each agent passes `config["skill_file_id"]` from `orchestrator._skill_file_ids[name]`) is a follow-up step (likely 25.D9.1). Live token reduction won't materialize until callers adopt the new path.

### Phase 1: bulk-upload runs at orchestrator startup
After restart:
```
source .venv/bin/activate
python -m uvicorn backend.main:app --reload --port 8000
```
Expected log line:
```
phase-25.D9: uploaded N skill files to Anthropic Files API
```
Verify the disk cache:
```
cat backend/agents/skills/.skill_file_ids.json | jq .
```
Expected shape:
```json
{
  "bull_agent": {"hash": "<sha256>", "file_id": "file_..."},
  "bear_agent": {"hash": "<sha256>", "file_id": "file_..."},
  ...
}
```

### Phase 2: caller adoption (follow-up step)
Each downstream agent (Bull, Bear, Synthesis, Risk Judge, etc.) needs to pass `config["skill_file_id"] = orchestrator._skill_file_ids.get(agent_name)` when invoking `general_client.generate_content(...)`. Once adopted, `llm_call_log` rows for those agents will show ~98.5% lower `input_tok` values.

## SkillOptimizer hot path verification
When SkillOptimizer modifies a skill:
1. The new content is written.
2. `reload_skills()` clears the in-memory template cache.
3. `SkillFileIdCache.invalidate(agent_name)` drops the file_id (hash mismatch on next access).
4. Next `generate_content` for that agent triggers a re-upload via `get_or_upload`.
Verify by:
- Run SkillOptimizer in a test cycle.
- Check `.skill_file_ids.json` -- the modified agent's `hash` should change AND `file_id` should be the new uploaded one.

## Closes audit basis
phase-24.9 F-5 RESOLVED structurally. The mechanism for ~98.5% skill-body token reduction is in place; live cost reduction lands when downstream callers adopt the `config["skill_file_id"]` path (25.D9.1 follow-up).

**Audit anchor for next bucket:** 25.D9.1 (caller-side adoption) OR 25.C9 (Batch API for 50% non-interactive savings) OR 25.E9 (native Citations).
