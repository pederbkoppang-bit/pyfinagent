# Contract — Step 59.1

**Step id:** 59.1 — Fable 5 model adoption (both layers, quality-first)
**Date:** 2026-06-11
**Phase:** phase-59 (operator-directed; 8 in-session pre-approvals recorded 2026-06-11)
**Researcher gate:** PASSED — `handoff/current/research_brief.md` (tier=moderate-complex, 7 external sources read in full, 17 URLs, recency scan; 9 internal files; envelope `gate_passed: true`)

## Research-gate summary

Validated against official docs: subagent frontmatter accepts the alias **`model: fable`** (sub-agents + model-config docs; requires Claude Code v2.1.170+, local is **2.1.172** ✓); **`effort: max` remains valid** (Fable effort table low→max; recommended baseline is `high` — keeping `max` is a documented over-spec for the gate roles); **`maxTurns`** is the documented stall lever (qa 12→30, researcher 30→40 per the 5 observed mid-evaluation stalls). Economics: Fable is Max-free only June 9-22, 2026; from June 23 it draws usage credits (~2x Opus burn) — this **invalidates the phase-29.2 "Max flat-fee" rationale** in CLAUDE.md and both agent comment blocks; reframe to rare-event frequency + operator quality-first pre-approval. Layer-2 traps found: `EFFORT_SUPPORTED_MODELS` (model_tiers.py:183-191) and `MODEL_EFFORT_FALLBACK` (:233-242) lack claude-fable-5 — without entries, `model_supports_effort()` returns False and `llm_client.py:1481` silently DROPS the effort param; add both (fallback tier xhigh). Rare-event grounding: mas_main/mas_qa are the operator-paced Slack/iMessage orchestrator+analyst (agent_definitions.py:181/229); autoresearch_strategic is a nightly cron; the 28 per-ticker pipeline agents are Gemini-locked (out of scope). Test inventory: `pytest -k 'fable or model_tiers or phase_59'` collects only 1 test today and does NOT catch `test_agent_map_live_model.py::test_endpoint_injects_live_model_field` which WILL break on the mas_main repin — the new test must be named `test_phase_59_1_*` and the breaking test must be updated in the same change; Q/A must run the FULL suite. Ticket map: repin cost ≈ $0.18/day (negligible; decision to record). Adversarial sources ("Fable overkill for high-volume roles") reconciled: metered roles stay off Fable.

## Hypothesis

Pinning Fable 5 on the four rare-event roles (researcher, qa, mas_main, autoresearch_strategic — plus the negligible-cost ticket agents) captures the "longer the task, the larger the lead" quality gain where it matters, with zero change to metered per-ticker spend, provable by unit tests over the resolution paths.

## Immutable success criteria (verbatim from .claude/masterplan.json, step 59.1)

1. "Layer-3: .claude/agents/researcher.md and qa.md pin Fable 5 using the researcher-validated frontmatter syntax for this Claude Code version, retain their effort settings, and raise maxTurns to 40 (researcher) and 30 (qa); the file comments record the 2026-06-11 operator pre-approval, the June-22 Max-credit economics, and the session-restart requirement; live_check_59.1.md instructs the next session to run scripts/qa/verify_qa_roster_live.sh"

2. "Layer-2: backend/config/model_tiers.py pins mas_main and autoresearch_strategic to claude-fable-5 while mas_qa and every per-ticker/per-analysis role keeps its existing model (cost discipline: no metered per-ticker path changes model); MODEL_EFFORT_FALLBACK gains a claude-fable-5 entry with a researcher-grounded effort tier; the ticket_queue_processor agent_model_map is updated (or explicitly kept) per the researcher's rare-event cost analysis with the decision recorded; a unit test covers the new resolution paths and the unchanged per-ticker pins"

3. "CLAUDE.md's effort-policy section is updated for Fable 5 (model id, $10/$50 pricing, the June-22 Max-credit change superseding the flat-fee rationale, the classifier-fallback note) without deleting the Opus 4.8 history; the change is additive and cites the operator's 2026-06-11 approval"

4. "the verification command exits 0 and live_check_59.1.md records the pin diff map (old->new per role), the unchanged-roles list, and the restart/roster-verify instruction"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests -k 'fable or model_tiers or phase_59' -q && test -f handoff/current/live_check_59.1.md`

## Plan

1. Layer-3: researcher.md (`model: fable`, maxTurns 40) + qa.md (`model: fable`, maxTurns 30); effort keys untouched; comment blocks rewritten (operator pre-approval 2026-06-11, June-23 credit economics superseding flat-fee, restart caveat).
2. Layer-2: model_tiers.py — mas_main :49 + autoresearch_strategic :60 → "claude-fable-5"; mas_qa :51 UNCHANGED (per-ticker); EFFORT_SUPPORTED_MODELS += "claude-fable-5"; MODEL_EFFORT_FALLBACK += ("claude-fable-5","xhigh"); comments cite the Fable effort doc (recommended high; project runs xhigh per quality-first).
3. ticket_queue_processor.py agent_model_map: main + q-and-a → claude-fable-5 (~$0.18/day, quality-first), research stays claude-sonnet-4-6 (cost-efficient) — decision recorded in the map comment + live_check.
4. Tests: NEW `backend/tests/test_phase_59_1_fable_adoption.py` (resolution paths: mas_main/autoresearch_strategic → fable; mas_qa + per-ticker pins unchanged; effort NOT dropped for claude-fable-5 via model_supports_effort + fallback tier; ticket map assertions); UPDATE `test_agent_map_live_model.py` (mas_main live_model → claude-fable-5).
5. CLAUDE.md effort-policy: additive Fable 5 paragraph (id, $10/$50, June-22→23 credit change, classifier fallback, alias notes, /model guidance) — Opus 4.8 history preserved as history.
6. FULL suite run (the -k false-green risk is known; Q/A re-runs full). live_check_59.1.md (pin diff map, unchanged-roles list, restart/roster instruction, test output). experiment_results.md → fresh Q/A → log → flip.

## Constraints

- No metered per-ticker path changes models (mas_qa, gemini_*, autoresearch_fast/smart, lite analyzers untouched).
- Agent .md edits take effect NEXT session (snapshot semantics); this step's own qa spawn runs the OLD snapshot — fine (pins don't change protocol); separation-of-duties satisfied by the operator's in-session pre-approval.
- Additive CLAUDE.md edit; no emojis; ASCII logger strings n/a (no logger changes).

## References

- handoff/current/research_brief.md (researcher 59.1, gate_passed: true; code.claude.com sub-agents/model-config docs, platform effort doc, anthropic.com Fable-5 announcement, pricing docs, Claude Code changelog)
- Operator pre-approvals: 8 AskUserQuestion answers 2026-06-11 (layers=both; cost=quality-first; caps=raise both; governance=pre-approve+restart-later)
- Code anchors: .claude/agents/{researcher,qa}.md frontmatter; model_tiers.py:49,51,60,183-191,233-242; llm_client.py:1481; ticket_queue_processor.py:165-169; test_agent_map_live_model.py; CLAUDE.md:56-62
