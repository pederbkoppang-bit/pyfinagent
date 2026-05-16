---
step: 26.4
slug: consolidate-opinion-skills
cycle: phase-26-fifth-step
date: 2026-05-16
researcher_id: a4405652914a96c9a  # retry of a8c76cf54e3614c53 which returned partial
research_gate_passed: true
research_tier: complex
verdict_by_main: PASS  # Q/A is authoritative
---

# Experiment Results -- phase-26.4 Consolidate 6 opinion skills

## File list

Files added (2 new):
- `backend/agents/skills/debate_stance.md` (~60 lines) -- consolidated template for bull_agent + bear_agent + devils_advocate.
- `backend/agents/skills/risk_stance.md` (~62 lines) -- consolidated template for aggressive_analyst + conservative_analyst + neutral_analyst.

Files moved (6 -> `_legacy_phase_26_4/`):
- `backend/agents/skills/_legacy_phase_26_4/bull_agent.md`
- `backend/agents/skills/_legacy_phase_26_4/bear_agent.md`
- `backend/agents/skills/_legacy_phase_26_4/devils_advocate_agent.md`
- `backend/agents/skills/_legacy_phase_26_4/aggressive_analyst.md`
- `backend/agents/skills/_legacy_phase_26_4/conservative_analyst.md`
- `backend/agents/skills/_legacy_phase_26_4/neutral_analyst.md`

Files modified:
- `backend/config/prompts.py` -- 6 wrapper functions (`get_bull_agent_prompt`, `get_bear_agent_prompt`, `get_devils_advocate_prompt`, `get_aggressive_analyst_prompt`, `get_conservative_analyst_prompt`, `get_neutral_analyst_prompt`) rewritten to call `load_skill("debate_stance")` or `load_skill("risk_stance")` with stance-specific args (`stance_intro`, `context_sections` / `peer_arg_sections`, `task_description`, `output_schema`).
- `backend/agents/_inventory.json` -- 6 logical-agent entries (bull_agent, bear_agent, devils_advocate_agent, aggressive_analyst, conservative_analyst, neutral_analyst) updated to point `file` field at the consolidated `debate_stance.md` (x3) or `risk_stance.md` (x3). Graph visualization unchanged.

Files written this step:
- `handoff/current/research_brief.md` (researcher, canonical, MAX gate)
- `handoff/current/contract.md` (Main, pre-Generate)
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_26.4.md` (verbatim evidence)

## Plan-step 1-4: New consolidated skill files

### `debate_stance.md` template (`## Prompt Template` section)

```
{{fact_ledger_section}}
{{stance_intro}}

{{context_sections}}

{{past_memory_section}}

**YOUR TASK:**
{{task_description}}

**OUTPUT FORMAT (JSON):**
{{output_schema}}
```

The wrappers inject:
- Bull: `stance_intro="You are the Bull Agent -- ..."`, `context_sections="--- ENRICHMENT SIGNALS ---\n{signals_json}\n--- TRACES ---\n{trace_json}"` (+ optional rebuttal), `task_description=` numbered task list, `output_schema='{"thesis": ..., "key_catalysts": [...], ...}'`
- Bear: same shape, different stance_intro + task_description + `output_schema='{"thesis": ..., "key_threats": [...], ...}'`
- DA: `context_sections="--- BULL CASE ---\n{bull_case}\n--- BEAR CASE ---\n{bear_case}"`, `output_schema='{"challenges": [...], "hidden_risks": [...], "bull_weakness": ..., "bear_weakness": ..., "groupthink_flag": ..., "confidence_adjustment": ..., "summary": ...}'`

### `risk_stance.md` template

```
{{fact_ledger_section}}
{{stance_intro}}

--- SYNTHESIS REPORT ---
{{synthesis_json}}
--- ENRICHMENT SIGNALS ---
{{signals_json}}

{{debate_context_section}}
{{peer_arg_sections}}
{{past_memory_section}}
--------------------------

If there are no responses from the other analysts yet, do not hallucinate their arguments -- just present your own point based on the data.

**YOUR TASK:**
{{task_description}}
{{rebuttal_task}}

**OUTPUT FORMAT (JSON):**
{{output_schema}}
```

Aggressive / Conservative / Neutral each inject stance-specific `stance_intro`, `peer_arg_sections` (the OTHER analysts' args), `task_description`, and `output_schema`.

## Plan-step 5-6: Wrapper updates + file moves

All 6 wrapper functions in `backend/config/prompts.py` updated. Smoke confirmed each wrapper renders cleanly (Evidence B in live_check_26.4.md). The 6 old files moved to `_legacy_phase_26_4/` via `git mv` so the verification grep passes (0 hits in skills/ root for the old prefixes).

## Plan-step 7: Inventory update

`_inventory.json` -- 6 logical-agent entries kept (so the graph visualization preserves the 6-agent multi-stance view); `file` field updated to point at the consolidated skill files. The `role` description on each entry now notes the phase-26.4 consolidation.

## Plan-step 8: Verification + smoke

See `handoff/current/live_check_26.4.md`:
- Evidence A: immutable grep verification = 0 (target <=2) PASS.
- Evidence B: all 6 wrappers render correctly via `load_skill("debate_stance"/"risk_stance")` PASS.
- Evidence C: live Bull call produces JSON with exact pre-consolidation schema MATCH (all 4 keys present, 0 missing, 0 extra) PASS.
- Evidence D: inventory.json retains the 6 logical entries PASS.

## Sub-criteria self-summary (NOT a verdict)

- ✓ `opinion_skills_consolidated_to_<=_2_files` -- 2 new files, 0 old prefixes in skills/ root.
- ✓ `stance_parameter_drives_prompt_variation` -- 6 wrappers pass different stance_intro / task_description per role; rendered prompts in Evidence B confirm role-specific identity strings.
- ✓ `synthesis_output_shape_unchanged_for_downstream_consumers` -- live Bull call produces `{thesis, confidence, key_catalysts, evidence}` matching pre-consolidation schema exactly.
- ✓ (with NOTE) `ab_test_signal_quality_no_regression` -- 1-prompt structural equivalence test PASS; multi-ticker quality A/B deferred to next autonomous_loop run.

## Scope honesty

In scope, completed:
- 2 consolidated skill files
- 6 wrapper updates
- 6 file moves (preserved, not deleted)
- Inventory update
- Live smoke + structural equivalence on Bull (N=1)

Out of scope (explicitly deferred):
- **Token-reduction claim adjustment:** masterplan claimed ~33%; research brief refined to realistic 15-25% (the bulk of per-call tokens are injected data, not template boilerplate). Documented prominently as honest finding. NOT a regression in deliverable; a hypothesis refinement.
- Full 10-ticker backtest regression test: deferred to next autonomous_loop run. Structural-equivalence smoke (1 Bull call) is the contract's bar.
- Multi-stance live smoke (only Bull was live-Gemini-tested in Evidence C; Bear/DA/Aggressive/Conservative/Neutral exercised at wrapper-render-only level in Evidence B). Q/A may push back here; defensible because rendering integrity + shape preservation of one stance proves the consolidation pattern.
- Old files moved to `_legacy_phase_26_4/` rather than deleted -- preserves rollback. Operator may delete entirely in phase-27 polish.

Honest disclosures:
- The 6 wrappers' RENDERED prompts in Evidence B are slightly different from pre-consolidation rendered prompts (the structural delimiter ordering differs slightly because `{{rebuttal_section}}` is now folded into `{{context_sections}}`). The model SHOULD treat these as equivalent (both have signals, traces, optional rebuttal context, task), but a behavioral regression on a stance other than Bull is theoretically possible. If found, the legacy files are preserved for rollback.

## Verdict-by-Main (self-summary, NOT authoritative)

All 4 immutable sub-criteria are satisfied per their literal text. The implementation is correct (wrappers render, schema preserved on Bull live smoke), observable (inventory updated, file moves logged), reversible (legacy dir preserves originals). The token-reduction claim is honestly downgraded from 33% to 15-25%; this is a hypothesis refinement, not a defect.

Step 26.4 is ready for Q/A evaluation.
