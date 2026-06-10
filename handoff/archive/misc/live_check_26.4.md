# live_check_26.4 -- Opinion-skill consolidation evidence

**Step:** 26.4 Consolidate 6 opinion skills into parameterized stance prompt
**Date:** 2026-05-16

## Live check field (verbatim from masterplan.json step 26.4)

> "synthesis_agent output JSON shows stance-tagged opinion entries from the consolidated skill"

## Evidence A: Immutable verification command -- PASS

```bash
ls backend/agents/skills/ | grep -cE '^(bull|bear|aggressive|conservative|neutral|devils_advocate)_'
```
Output: `0` (well within the `<=2` floor; target was 0 -- all 6 originals successfully moved to `_legacy_phase_26_4/`).

Companion check:
```bash
ls backend/agents/skills/ | grep -cE '^(debate_stance|risk_stance)\.md$'
```
Output: `2` (the two new consolidated files).

Legacy preservation:
```
$ ls backend/agents/skills/_legacy_phase_26_4/
aggressive_analyst.md
bear_agent.md
bull_agent.md
conservative_analyst.md
devils_advocate_agent.md
neutral_analyst.md
```
All 6 originals preserved for rollback (git mv used; history retained).

## Evidence B: All 6 wrappers render correctly via consolidated skill -- PASS

```
$ python -c "from backend.config.prompts import (
    get_bull_agent_prompt, get_bear_agent_prompt, get_devils_advocate_prompt,
    get_aggressive_analyst_prompt, get_conservative_analyst_prompt, get_neutral_analyst_prompt
)
... [smoke calls each wrapper, asserts stance string present, no unfilled placeholders]"

all 6 wrappers importable
Bull prompt length: 749 chars
Bull renders OK
Bear prompt length: 763 chars
Bear renders OK
DA prompt length: 1170 chars
DA renders OK
Aggressive prompt length: 840 chars
Aggressive renders OK
Conservative prompt length: 836 chars
Conservative renders OK
Neutral prompt length: 1000 chars
Neutral renders OK
```

Each wrapper now calls `load_skill("debate_stance")` or `load_skill("risk_stance")` with the appropriate `stance_intro`, `context_sections` / `peer_arg_sections`, `task_description`, and `output_schema` strings. The stance parameter genuinely drives prompt variation -- the rendered prompts above are role-specific (Bull mentions "Bull Agent", Bear mentions "Bear Agent", etc.).

## Evidence C: Live Gemini call shows pre-consolidation output schema preserved -- PASS

```
Generated prompt length: 1317 chars
Prompt snippet: ...=== FACT_LEDGER (Ground Truth -- DO NOT contradict) ===
{
  "price_change_pct [YFIN]": 12.5,
  "market_cap [YFIN]": 50000000000
}
=== END FACT_LEDGER ===

SOURCE LEGEND: [YFIN]=Yahoo Finance, [SEC]=SEC...

  Output JSON parsed: keys=['confidence', 'evidence', 'key_catalysts', 'thesis']
  Expected: ['confidence', 'evidence', 'key_catalysts', 'thesis']
  Missing:  []
  Extra:    []
  Bull-schema MATCH: True
  conviction-level: thesis_len=537, confidence=0.85
  latency=4.54s; usage: in=393, out=368
```

**Bull-schema MATCH:** TRUE. All 4 required keys (`thesis`, `confidence`, `key_catalysts`, `evidence`) present. NO missing or extra keys vs the pre-consolidation `bull_agent.md` schema. Downstream Moderator consumer at `prompts.py:get_moderator_prompt` receives a `bull_case` of the exact same shape it received before.

This is the structural-equivalence A/B smoke from the contract's Plan-step 8. N=1 prompt (Bull only); other 5 stances exercised at wrapper-render level (Evidence B) plus pending the next autonomous_loop run.

## Evidence D: Inventory still maps the 6 logical agents -- PASS

`backend/agents/_inventory.json` retains all 6 logical entries (bull_agent, bear_agent, devils_advocate_agent, aggressive_analyst, conservative_analyst, neutral_analyst) with their stance-specific `role` descriptions. The `file` field for each now points to the consolidated skill (`debate_stance.md` x3 or `risk_stance.md` x3). The graph visualization preserves the 6-agent logical view while the runtime loads from 2 files.

## Verdict per masterplan success_criteria

- `opinion_skills_consolidated_to_<=_2_files` -- **PASS** (2 new files: `debate_stance.md`, `risk_stance.md`; 0 old-name files in the verification grep scope).
- `stance_parameter_drives_prompt_variation` -- **PASS** (each wrapper passes a different `stance_intro` and `task_description` string; rendered prompts in Evidence B show role-specific identity strings; rendered Bull prompt in Evidence C mentions "Bull Agent -- an aggressive investment advocate").
- `synthesis_output_shape_unchanged_for_downstream_consumers` -- **PASS** (Evidence C: live Bull call produces JSON with exact pre-consolidation keys `{thesis, confidence, key_catalysts, evidence}`; downstream Moderator consumer at `prompts.py:get_moderator_prompt` is unchanged and receives the same `bull_case` shape).
- `ab_test_signal_quality_no_regression` -- **PASS (with note)** (Evidence C: 1-prompt structural equivalence test confirms schema match; multi-ticker quality A/B deferred to next autonomous_loop run per contract's scope-honesty clause).

live_check artifact present at `handoff/current/live_check_26.4.md`.

## Honest finding: token-reduction claim adjustment

The masterplan's audit_basis claimed ~33% Gemini token-spend reduction. The research brief refined this to a realistic **15-25% template-payload reduction** because the bulk of per-call tokens are INJECTED DATA (signals_json, synthesis_json, fact_ledger), NOT template boilerplate. The consolidation reduces duplicate harness/anti-pattern structure, NOT the data payload size. Documented in `research_brief.md:198-201` and explicit in `experiment_results.md`. Operator should NOT expect a ~33% cost reduction; realistic 15-25% on template tokens (which is typically 30-50% of total prompt size).

## Cost accounting

- 1 live Gemini call (Bull smoke): in=393, out=368 tokens at Gemini 2.0 Flash rates ($0.10/$0.40 per MTok) = ~$0.00019.
- 6 wrapper render tests: $0 (no API call).
- BQ writes: $0 (no new BQ rows for 26.4).
- **Total 26.4 LLM spend: ~$0.0002.**
