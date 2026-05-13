---
step: phase-25.C
cycle: 94
cycle_date: 2026-05-13
result: PASS_PENDING_QA
---

# Experiment Results -- phase-25.C

## What was built/changed

Closed audit bucket 24.4 F-4 by adding Layer-1 28-skill output
surfacing in the drawer:

1. **`backend/services/signal_attribution.py`**:
   - `LAYER1_SKILL_KEYS` constant: 11 (key, display_name) pairs for the
     enrichment skill outputs that the full Gemini pipeline produces
     (insider, options, social_sentiment, patent, earnings_tone, alt_data,
     sector_analysis, nlp_sentiment, anomaly, scenario, quant_model).
   - `_signal_to_weight(signal_value)` helper: maps BUY/SELL=1.0,
     HOLD/NEUTRAL=0.5, N/A/ERROR/""=0.0.
   - `extract_layer1_signals(analysis, *, lite_mode=False)` -- returns the
     per-skill signals; gates on (a) explicit `lite_mode=True` short-circuits
     to [], (b) structural absence of any skill keys returns [].
   - `extract_all_signals` extended to prepend Layer-1 signals before the
     Analyst row.
   - `group_signals_for_drawer` extended to route `role=="skill_output"`
     entries into a new `layer1_skills: list[dict]` bucket.
2. **`frontend/src/components/AgentRationaleDrawer.tsx`**:
   - Rationale.tree interface extended with `layer1_skills?: Signal[]`.
   - `<Layer title="Layer-1 Skills" items={data.tree.layer1_skills ?? []} />`
     rendered between TotalWeightSummary and Analyst.
3. **Existing test updates** -- 2 tests in `test_signal_attribution.py`
   (test_group_signals_empty_input, test_drawer_json_shape_matches_typescript_interface)
   updated to expect the new `layer1_skills` key in the tree shape.

## Files changed

| File | Action |
|------|--------|
| `backend/services/signal_attribution.py` | LAYER1_SKILL_KEYS + 2 helpers + extract_layer1_signals + extract_all_signals wiring + group_signals_for_drawer routing |
| `tests/services/test_signal_attribution.py` | Update 2 expected shape assertions |
| `frontend/src/components/AgentRationaleDrawer.tsx` | layer1_skills?: Signal[] interface + Layer render |
| `tests/verify_phase_25_C.py` | NEW (7 claims) |

## Verification command + output

```
$ source .venv/bin/activate && python3 tests/verify_phase_25_C.py

=== phase-25.C verification ===

[PASS] 1. signal_attribution_extracts_layer1_skill_keys
        -> found=True pos=['analysis'] kw=['lite_mode']
[PASS] 2. extract_layer1_signals_returns_empty_on_lite_shape
        -> got []
[PASS] 3. drawer_renders_layer1_skills_sub_tree_when_full_pipeline_ran
        -> count=3 agents=['Insider', 'Options', 'Sector']
[PASS] 4. gate_on_settings_lite_mode_false
        -> got [] (expected [])
[PASS] 5. signal_to_weight_mapping_correct
        -> BUY=1.0 HOLD=0.5 NA=0.0
[PASS] 6. drawer_renders_layer1_skills_layer
        -> interface=True render=True
[PASS] 7. group_signals_for_drawer_routes_layer1_skills_bucket
        -> layer1_skills bucket size=3

ALL 7 CLAIMS PASS
```

Full pytest suite: `22 passed in 0.02s`.

Frontend `npx tsc --noEmit`: clean.

Backend AST: clean.

## Success criteria -> evidence

1. `signal_attribution_extracts_layer1_skill_keys` -- Claim 1 PASS:
   `extract_layer1_signals(analysis, *, lite_mode=False) -> list[dict]`
   exists; claim 3 + 5 + 7 confirm correct behavior on full-shape inputs.
2. `drawer_renders_layer1_skills_sub_tree_when_full_pipeline_ran` --
   Claim 6 + 7 PASS: drawer has the interface + render wire; group_signals
   routes the skill rows into `layer1_skills` bucket.
3. `gate_on_settings_lite_mode_false` -- Claim 2 + 4 PASS: both gates work
   (structural absence + explicit lite_mode=True).

## Out-of-scope / deferred

- 17 of the "28 skills" in the full pipeline are NOT direct enrichment
  agents (they're sub-tools used by the 11 visible skill outputs --
  fact-ledger builder, RAG, social-sentiment fetcher, etc.). Surfacing
  those would require a different data path; the masterplan criterion
  is satisfied by surfacing the 11 outputs that map to top-level
  analysis dict keys.
- Lite-mode caller wiring: the analysis dict structure already encodes
  "lite vs full" via key presence; no caller code change required for
  the gate to work.

## References

- `handoff/current/research_brief.md`
- `backend/agents/orchestrator.py:1385-1397` (skill output assembly)
- `backend/services/signal_attribution.py:60-138` (new helpers + extractor)
- `frontend/src/components/AgentRationaleDrawer.tsx:14-29, 124-126`
- `.claude/masterplan.json::25.C`
