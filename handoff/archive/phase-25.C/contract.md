---
step: 25.C
slug: layer1-skill-outputs-in-drawer
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.C

## Step ID + masterplan reference

`25.C` -- "Surface Layer-1 28-skill outputs in drawer when full pipeline runs"
(P2, harness_required, no dep).

## Research-gate summary

Tier=moderate. Brief at `handoff/current/research_brief.md`,
`gate_passed=true`. Layer-1 skill output structure established by
inspection of orchestrator.py:1385-1397.

## Hypothesis

When the full Gemini pipeline runs, the analysis dict carries 11
enrichment-agent skill outputs (`insider`, `options`, etc.) each with
`{signal, summary, analysis}`. Currently the signal_attribution
extractor ignores them, so the drawer never shows the agents that
gathered the underlying evidence.

By adding `extract_layer1_signals` + wiring into `extract_all_signals` +
`group_signals_for_drawer`, the drawer renders a "Layer-1 Skills"
section between the total-weight summary and the Analyst layer.

In lite mode the analysis dict doesn't carry these keys, so the
function naturally returns [] -- satisfying the gate criterion.

## Success criteria (verbatim from masterplan.json)

> `signal_attribution_extracts_layer1_skill_keys`
>
> `drawer_renders_layer1_skills_sub_tree_when_full_pipeline_ran`
>
> `gate_on_settings_lite_mode_false`

## Plan steps

1. **`backend/services/signal_attribution.py`**:
   - Add `LAYER1_SKILL_KEYS: list[tuple[str, str]]` = list of
     `(analysis_dict_key, display_agent_name)` pairs for the 11 skills.
   - Add `_signal_to_weight(signal_value: str) -> float` helper:
     BUY|SELL -> 1.0, HOLD|NEUTRAL -> 0.5, N/A|ERROR|"" -> 0.0.
   - Add `extract_layer1_signals(analysis, *, lite_mode=False) -> list[dict]`
     that returns the per-skill signals (empty list when lite_mode=True or
     no skill keys are present).
   - Extend `extract_all_signals` to prepend layer-1 signals before the
     Analyst row.
   - Extend `group_signals_for_drawer` to route `role=="skill_output"`
     entries into a new `layer1_skills: list[dict]` bucket.
2. **`frontend/src/components/AgentRationaleDrawer.tsx`**:
   - Extend the Rationale.tree interface with `layer1_skills?: Signal[]`.
   - Render a `<Layer title="Layer-1 Skills" items={data.tree.layer1_skills ?? []} />` between TotalWeightSummary and Analyst.
3. **Verifier** -- `tests/verify_phase_25_C.py` with 6 claims:
   - Claim 1: `extract_layer1_signals` exists with correct signature.
   - Claim 2: returns [] on lite-shape analysis (no skill keys).
   - Claim 3: returns >=3 entries on full-shape analysis with insider/options/sector populated.
   - Claim 4: gate -- explicit `lite_mode=True` short-circuits to [].
   - Claim 5: signal-to-weight mapping correct (BUY=1.0, HOLD=0.5, N/A=0.0).
   - Claim 6: drawer renders `layer1_skills` (TS source has the new Layer wire).

## Files

| File | Action |
|------|--------|
| `backend/services/signal_attribution.py` | Add `extract_layer1_signals` + wire |
| `frontend/src/components/AgentRationaleDrawer.tsx` | Render Layer-1 Skills layer |
| `tests/verify_phase_25_C.py` | NEW |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_C.py
```

## Live-check

`Drawer tree on next full-pipeline trade has layer1_skills sub-tree with >=3 entries`.
Will write `handoff/current/live_check_25.C.md`.

## Risks + mitigations

- **Risk**: drawer becomes dense (now 7 layers instead of 6).
  **Mitigation**: each Layer is a collapsible `<details>` -- operators can
  close the Layer-1 Skills section when they want the high-level summary.
- **Risk**: empty-skill rows pad noise (e.g., `insider` returning `signal=N/A`).
  **Mitigation**: extractor skips entries with empty summary OR N/A signal.

## References

- `handoff/current/research_brief.md`
- `backend/agents/orchestrator.py:1385-1397` (skill output shape)
- `backend/services/signal_attribution.py` (extractors)
- `frontend/src/components/AgentRationaleDrawer.tsx`
- `.claude/masterplan.json::25.C`
