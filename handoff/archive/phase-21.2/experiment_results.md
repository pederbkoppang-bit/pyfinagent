---
step: phase-21.2
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - frontend/src/lib/types.ts (+apply_model_to_all_agents on ModelConfig + FullSettings)
  - frontend/src/lib/api.ts (updateModelConfig accepts new flag)
  - backend/api/settings_api.py (ModelConfig + FullSettings + SettingsUpdate + ModelConfigUpdate accept bool; _FIELD_TO_ENV mapping)
  - frontend/src/app/settings/page.tsx (checkbox + help text below Models row)
---

# Experiment Results -- phase-21.2

## What was done

Wired phase-21.1's backend flag through to the Settings UI. Operator can
now toggle "Apply Standard Model to all agents" in /settings Models tab;
the flag persists to backend/.env and is read by `resolve_model()` on
every agent dispatch.

## Verification

```
$ cd frontend && npx tsc --noEmit
(exit 0; no output)

$ npm run build
... 14 routes built; settings page = 16.4 kB ...
```

## Files touched

| Path | Action |
|------|--------|
| frontend/src/lib/types.ts | edit (+apply_model_to_all_agents on 2 types) |
| frontend/src/lib/api.ts | edit (updateModelConfig signature) |
| backend/api/settings_api.py | edit (4 Pydantic models + _FIELD_TO_ENV mapping) |
| frontend/src/app/settings/page.tsx | edit (+checkbox + help block, ~22 lines) |
| handoff/current/contract.md | rewrite (rolling) |
| handoff/current/experiment_results.md | rewrite (this) |

## Cycle-2 fix applied

Q/A cycle-1 returned CONDITIONAL flagging a real round-trip-display bug:
`_settings_to_full()`, `get_model_config()`, and `update_model_config()` were
constructing their response models WITHOUT reading `settings.apply_model_to_all_agents`.
The Pydantic default of `False` always won, so the toggle would have appeared
unchecked on Settings page reload even after a successful save (PUT correctly
wrote the env var, but GET masked it).

Fixed by adding `apply_model_to_all_agents=bool(getattr(s, "apply_model_to_all_agents", False))`
to all three response constructors. Also wired `update_model_config()` to write
`APPLY_MODEL_TO_ALL_AGENTS` to .env when the field is in the body.

Re-verified: 10/10 pytest still PASS + npm run build still exit 0.

## Honest disclosures

1. **Cycle-2 fix needed** -- Q/A caught a real round-trip-display bug missed in cycle-1. Real defect; honest fix.
2. **No new tests this cycle** -- the backend override logic was already tested in 21.1; the frontend toggle is a thin wiring change covered by `tsc --noEmit + npm run build`.
2. **`gemini_model` field misnaming preserved** (it's the Standard model selector for any provider, not just Gemini). Renaming would break the existing settings.py / env var compatibility chain. Documented for a separate cleanup cycle.
3. **Environment plumbing** -- the new flag uses `APPLY_MODEL_TO_ALL_AGENTS` env var. Backend restart picks it up via pydantic Settings autoload.
4. **Cycle-2 not needed** (clean first pass).

## Closes

UAT-21.2.

## Next

Spawn Q/A. After PASS: log + flip + archive. Then phase-21.3 (per-model skill optimization design doc).
