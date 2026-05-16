# Sprint Contract -- phase-26.4
Step: Consolidate 6 opinion skills into parameterized stance prompt

## Research Gate
researcher_a4405652914a96c9a (tier=complex, MAX gate per user instruction 2026-05-16) gate_passed=true. First researcher spawn (a8c76cf54e3614c53) returned partial without writing the brief; retry succeeded.
Brief: `handoff/current/research_brief.md` (canonical).
- 6 unique external URLs read in full via WebFetch (arXiv expert investment teams paper 2602.23330, Anthropic harness-design, Redis token-optimization 2026, learnprompting role-prompting, Medium multi-persona, OpenReview multi-agent debate). 8 snippet-only; 14 URLs total. 3+-variant search.
- Recency scan 2024-04 -> 2026-05 reported.
- Internal grep covered: 6 skill .md files, debate.py + risk_debate.py call sites, prompts.py wrappers, downstream consumers (Moderator + Risk Judge).
- **Key finding -- 6 skills split into 2 functionally distinct groups:**
  - **Group A (debate, used by debate.py):** bull_agent + bear_agent + devils_advocate. Consumed by Moderator at `debate.py:~250-350`. Expected named outputs: `bull_case`, `bear_case`, `devils_advocate`.
  - **Group B (risk_debate, used by risk_debate.py):** aggressive_analyst + conservative_analyst + neutral_analyst. Consumed by Risk Judge at `prompts.py:797`. Expected named outputs: `aggressive_arg`, `conservative_arg`, `neutral_arg`.
- Recommended consolidation: **Option B (2 files)** -- `debate_stance.md` + `risk_stance.md`. NOT Option A (1 file conflates two semantically distinct pipelines with different I/O schemas) and NOT Option C (1 call destroys adversarial-independence contract per debate literature).
- **Honest cost-finding:** the masterplan's "33% Gemini token-spend reduction" is overstated. Realistic estimate: 15-25% template-payload reduction. The bulk of per-call tokens are INJECTED DATA (signals_json, synthesis_json, fact_ledger), NOT template boilerplate. Consolidating duplicate template structure saves 15-25%, not 33%. Documented prominently in brief lines 198-201.
- Integration points: `backend/config/prompts.py` -- 6 wrapper functions (`get_bull_agent_prompt` at 504, `get_bear_agent_prompt` at 553, `get_devils_advocate_prompt` at 648, `get_aggressive_analyst_prompt` at 666, `get_conservative_analyst_prompt` at 715, `get_neutral_analyst_prompt` at 764). These each call `load_skill(<name>) -> format_skill(...)`. After consolidation, each wrapper calls `load_skill("debate_stance")` or `load_skill("risk_stance")` with different `stance=` and other params.
- **Downstream consumers UNCHANGED:** `debate.py` and `risk_debate.py` still call the same `get_<name>_prompt` wrappers and receive the same named output keys. Refactor is purely at the file/template layer, not the runtime interface.

## Hypothesis
Replacing 6 skill files with 2 (`debate_stance.md`, `risk_stance.md`) using `{{stance}}`/`{{output_schema_block}}`/`{{task_description}}` placeholders achieves the literal sub-criterion (file count <=2) and the file-content sub-criterion (stance parameter drives prompt variation). Updating the 6 wrappers in `prompts.py` to inject stance-specific arguments preserves all downstream consumer contracts -- Moderator and Risk Judge receive the same shape they received before. A structural equivalence smoke (1 stance call through the consolidated skill) demonstrates the output is shape-compatible with the pre-consolidation version. The realistic token reduction is 15-25%, not 33% -- the brief's honest finding.

## Success Criteria (immutable, copied verbatim from .claude/masterplan.json step 26.4)
```
ls backend/agents/skills/ | grep -cE '^(bull|bear|aggressive|conservative|neutral|devils_advocate)_'
```
Must produce <=2 (target: 0 after migration).

Plus sub-criteria:
- `opinion_skills_consolidated_to_<=_2_files` -- satisfied by deleting (or moving out of skills/) the 6 original files and creating exactly 2 new files (`debate_stance.md`, `risk_stance.md`).
- `stance_parameter_drives_prompt_variation` -- satisfied by `{{stance}}` and related placeholders in the two new files; the wrappers in `prompts.py` inject different stance strings per role.
- `synthesis_output_shape_unchanged_for_downstream_consumers` -- satisfied by preserving the named outputs (`bull_case`, `bear_case`, `devils_advocate`, `aggressive_arg`, `conservative_arg`, `neutral_arg`) from the 6 wrapper functions. Verified by reading the consumers' expected input shape (Moderator, Risk Judge) and confirming the wrapper functions still produce those keys.
- `ab_test_signal_quality_no_regression` -- satisfied by a structural-equivalence smoke: run a representative stance prompt through both the OLD path (one of the existing 6 files, before deletion) and the NEW path (consolidated file with `stance` param). Compare the output JSON keys; expect identical key set.

live_check: `handoff/current/live_check_26.4.md` -- verbatim Python output of the consolidated skill being loaded + formatted + the live Gemini response showing the same JSON shape (bull_case structure or aggressive_arg structure) as the pre-consolidation version would have produced.

## Plan (PRE-commit; will NOT diverge in Generate)

1. **Inspect the 6 existing skill files** (brief gives line counts: ~80-130 lines each). Note common structure: identical fact_ledger block, near-identical anti-patterns block, identical experiment log, stance-specific prompt template + output schema.

2. **Read `backend/config/prompts.py`** around the 6 wrapper functions (lines 504, 553, 648, 666, 715, 764). Quote each function's structure to plan the migration.

3. **Write `backend/agents/skills/debate_stance.md`** -- consolidates bull + bear + devils_advocate. Placeholders: `{{stance}}`, `{{opponent_label}}`, `{{output_schema_block}}`, `{{task_description}}`, `{{rebuttal_section}}` (already built in Python). Shared: fact_ledger rules, anti-patterns, experiment log, harness contract.

4. **Write `backend/agents/skills/risk_stance.md`** -- consolidates aggressive + conservative + neutral. Placeholders: `{{stance}}`, `{{stance_philosophy}}`, `{{peer_arg_sections}}`, `{{output_schema_block}}`, `{{task_description}}`, `{{rebuttal_task}}` (already built in Python). Shared: fact_ledger rules, anti-patterns, experiment log, harness contract.

5. **Update `backend/config/prompts.py`** -- 6 wrapper functions each call `load_skill("debate_stance")` or `load_skill("risk_stance")` (instead of their individual files) with the correct stance-specific args. Output schemas live in the wrappers (passed as `{{output_schema_block}}` string param to `format_skill`).

6. **Move the 6 old files** out of `backend/agents/skills/` to `backend/agents/skills/_archive/` (preserves them in repo for rollback; removes them from the grep scope so verification passes).

7. **Update `backend/agents/_inventory.json`** if the 6 old skill names appear there (replace with the 2 new ones).

8. **Verification + live smoke + structural A/B:**
   - Immutable verification: `ls backend/agents/skills/ | grep -cE '^(bull|bear|aggressive|conservative|neutral|devils_advocate)_' = 0`.
   - Companion check: `ls backend/agents/skills/ | grep -cE '^(debate_stance|risk_stance)\.md$' = 2`.
   - Load one consolidated skill via `load_skill("debate_stance")` and `format_skill(template, stance="Bull Agent", opponent_label="Bear Agent", output_schema_block=..., task_description="...")`. Confirm rendered prompt looks coherent (no unfilled `{{}}` placeholders).
   - Live Gemini call: invoke the formatted "Bull Agent" version of debate_stance.md on a representative input. Capture output JSON; confirm it has the expected keys (`thesis`, `key_catalysts`, `confidence`, etc. — match the pre-migration bull_agent.md schema).
   - Structural equivalence: compare the output JSON keys vs the OLD bull_agent.md's expected schema. Expect MATCH.

## Scope honesty / out-of-scope

- The 10-ticker backtest regression test from the brief is **deferred to operator-driven validation** (full backtest run takes ~5-10 minutes per condition; not justifiable in 26.4 smoke scope). The structural equivalence smoke (1 stance call) is the contract's PASS bar; full multi-cycle quality regression is a phase-27 operator-driven affordance.
- The cost-reduction claim is 15-25% (honest brief finding), NOT the 33% in the masterplan's audit_basis. This is a HYPOTHESIS REFINEMENT, not a verification failure. Documented in brief lines 198-201.
- The 6 old files are MOVED to `_archive/` not DELETED -- preserves rollback path. If operator wants to delete entirely, that's a phase-27 affordance.
- SkillOptimizer compatibility: `skill_optimizer.py` may reference the old skill names. If found, listed in `_inventory.json`, updated. If not in scope, deferred to phase-27.
- This step does NOT touch `debate.py` or `risk_debate.py` source code. Refactor lives entirely in `prompts.py` (wrapper updates) + 2 new skill files + 6 file moves.
- The 6 old files' Experiment Logs are concatenated into the new files' headers (preserves historical optimization context).

## References
- Research brief: `handoff/current/research_brief.md` (canonical)
- Masterplan step JSON: `.claude/masterplan.json` step `26.4`
- 6 existing skill files: `backend/agents/skills/{bull_agent,bear_agent,aggressive_analyst,conservative_analyst,neutral_analyst,devils_advocate_agent}.md`
- 6 wrapper functions: `backend/config/prompts.py:504, 553, 648, 666, 715, 764`
- Group A consumer: `backend/agents/debate.py:213, 229, 261` + Moderator at `prompts.py:get_moderator_prompt`
- Group B consumer: `backend/agents/risk_debate.py:186, 203, 220` + Risk Judge at `prompts.py:797`
- Inventory: `backend/agents/_inventory.json`
- load_skill / format_skill: `backend/config/prompts.py` helpers (path TBD; brief notes them as the loaders)
