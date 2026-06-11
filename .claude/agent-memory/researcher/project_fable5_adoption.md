---
name: fable5-adoption
description: Fable 5 model adoption facts (phase-59.1) — frontmatter alias, effort baseline, June-22 credit cliff, the verification-command coverage gap, and the one breaking test
metadata:
  type: project
---

Phase-59.1 (2026-06-11): adopt `claude-fable-5` on the quality-first rare-event roles in both MAS layers.

**Validated facts (official docs, accessed 2026-06-11):**
- Frontmatter `model:` accepts the alias **`fable`** (sub-agents + model-config docs). Use `model: fable`, mirroring `model: opus`. Full id `claude-fable-5` also valid.
- Requires **Claude Code v2.1.170+** (local is 2.1.172 — OK). `fable` won't show in older pickers.
- Specs: **1M context default, 128K max output**, $10/$50 per Mtok (2x Opus 4.8).
- Effort: Fable accepts low/medium/high/xhigh/max. **Fable baseline is `high`, NOT `xhigh`** (differs from Opus 4.8's "start with xhigh"). Fable doc: "lower effort on Fable 5 often exceeds `xhigh` on prior models." `effort: max` stays valid for the gate roles (deliberate over-spec).
- "At the highest effort, Fable 5 reflects on and validates its own work" — the evaluator-gate match.
- Classifier fallback to Opus 4.8 on cyber/bio/chem/distillation (finance unaffected); Claude Code handles it, no code change.

**ECONOMICS SHIFT (supersedes phase-29.2 "Max flat-fee" rationale):** Fable free on Max/Pro/Team only **June 9-22 2026**; from **June 23** Fable draws Max usage credits (~2x Opus burn official, steeper in unbounded agentic sessions). The "Max flat-fee removes per-token ceiling" line in CLAUDE.md:56 + researcher.md:7-14 + qa.md:7-13 is now STALE — reframe as "cost contained by rare-event FREQUENCY (fires once per masterplan step) + operator quality-first pre-approval."

**Rare-event grounding (code, not vibes):** Layer-2 `mas_main`="Ford (Slack Orchestrator)" + `mas_qa`="Analyst" are operator-paced Slack/iMessage routing (agent_definitions.py:181/229; multi_agent_orchestrator.py), NOT the per-ticker Gemini pipeline (that's the 28 Gemini agents in orchestrator.py, Gemini-LOCKED, out of scope). `autoresearch_strategic` = nightly 2am cron (autoresearch/cron.py:25 "0 2 * * *"). All three are rare -> Fable-eligible. mas_communication/mas_research/autoresearch_fast/smart stay cheap.

**model_tiers.py gotcha:** `EFFORT_SUPPORTED_MODELS` (:183-191) and `MODEL_EFFORT_FALLBACK` (:233-242) have NO `claude-fable-5` entry. If a role is repinned to Fable without adding it, `model_supports_effort()` returns False and llm_client.py:1481 SILENTLY DROPS the effort param. Add `"claude-fable-5"` to the tuple + `("claude-fable-5","xhigh")` to the fallback.

**THE BREAKING TEST:** `test_agent_map_live_model.py::test_endpoint_injects_live_model_field` asserts `main_node.live_model == "claude-opus-4-8"` (resolves mas_main through build tier). Repinning mas_main->fable BREAKS it. Update literal to "claude-fable-5" (same as the 56.2 4-7->4-8 precedent already noted in that file's comment). The other model-id tests (test_apply_model_to_all_agents 52/79, phase_39_1, etc.) are DYNAMIC (read _BUILD_TIER or use override stand-ins) and do NOT break.

**VERIFICATION-COMMAND COVERAGE GAP (false-green risk):** the 59.1 cmd `pytest -k 'fable or model_tiers or phase_59'` collects only 1 test today (phase_37_2 via name-substring) and does NOT catch test_agent_map_live_model. So fixing model_tiers but forgetting the test update -> cmd exits 0 while real suite is RED. The new test MUST be named `test_phase_59_1_*` (matches `phase_59`) to enter the -k net, and Q/A must run the FULL backend/tests suite, not just -k.

See [[research_gate_discipline]]. Session model is user-default (no `model` key in .claude/settings.json) — operator's `/model fable` already in effect; repo doesn't contradict it.
