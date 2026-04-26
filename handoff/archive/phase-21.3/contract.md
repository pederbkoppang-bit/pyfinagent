---
step: phase-21.2
title: Settings frontend toggle for apply_model_to_all_agents
cycle_date: 2026-04-26
harness_required: true
verification: cd frontend && npx tsc --noEmit && npm run build
research_brief: handoff/current/phase-21.1-research-brief.md
---

# Contract -- phase-21.2

Frontend wiring for phase-21.1's backend flag. Adds:
- `apply_model_to_all_agents?: boolean` to `ModelConfig` + `FullSettings` types
- `updateModelConfig({ apply_model_to_all_agents })` API support
- Backend Pydantic ModelConfig + FullSettings + SettingsUpdate + ModelConfigUpdate accept the bool
- `_FIELD_TO_ENV` mapping `apply_model_to_all_agents` -> `APPLY_MODEL_TO_ALL_AGENTS`
- Settings UI Models tab: checkbox below Standard/Deep-Think pickers with help text

Verification: `cd frontend && npx tsc --noEmit && npm run build` -> exit 0 (already passing).
