---
step: phase-23.1.6
title: Settings page Signal Stack toggles + backend settings_api.py extension for 13 new fields
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "from backend.api.settings_api import FullSettings, SettingsUpdate; expected = {\"macro_regime_filter_enabled\",\"macro_regime_model\",\"pead_signal_enabled\",\"pead_signal_model\",\"pead_signal_lookback_quarters\",\"news_screen_enabled\",\"news_screen_model\",\"news_screen_max_headlines\",\"sector_calendars_enabled\",\"sector_calendars_lookahead_days\",\"meta_scorer_enabled\",\"meta_scorer_model\",\"meta_scorer_max_batch\"}; missing = expected - set(FullSettings.model_fields.keys()); assert not missing, f\"Missing in FullSettings: {missing}\"; missing = expected - set(SettingsUpdate.model_fields.keys()); assert not missing, f\"Missing in SettingsUpdate: {missing}\"; upd = SettingsUpdate(macro_regime_filter_enabled=True, news_screen_max_headlines=50); assert upd.macro_regime_filter_enabled is True and upd.news_screen_max_headlines == 50; print(\"ok 13 fields wired in FullSettings + SettingsUpdate\")"'
research_brief: handoff/current/phase-23.1.6-research-brief.md
---

# Contract — phase-23.1.6

## Hypothesis

The 13 new backend settings (5 enable flags + 4 model selectors + 4 numeric controls) added in cycles 1-5 must be exposed via the Settings API + frontend Settings page so the operator can toggle them without editing `.env` files. Default-OFF discipline is preserved at the backend layer; the UI surfaces them with clear labels + sub-descriptions.

## Plan

1. **Backend `backend/api/settings_api.py`** — add the 13 fields per the research brief:
   - `FullSettings` Pydantic class: 13 new fields with default values matching `backend/config/settings.py`
   - `SettingsUpdate` Pydantic class: 13 new optional fields with `Field(None, ge=..., le=...)` validators on the numeric ones
   - `_FIELD_TO_ENV` dict: 13 new field-to-env-var mappings
   - `_settings_to_full()`: 13 new `bool(getattr(s, ...))` lines
   - Model validation: extend the existing model-name validator loop to cover `macro_regime_model`, `pead_signal_model`, `news_screen_model`, `meta_scorer_model`
2. **Frontend `frontend/src/lib/types.ts`** — extend `FullSettings` interface with 13 new optional fields (all `?` for backward compat).
3. **Frontend `frontend/src/app/settings/page.tsx`** — NEW BentoCard "Signal Stack (Phase 23.1)" placed after the existing "Model Configuration" BentoCard:
   - 5 toggle rows (one per cycle 1-5 enable flag)
   - Model pickers for the 4 LLM-using signals (sector_calendars is data-pull only, no model)
   - Numeric inputs for `pead_signal_lookback_quarters`, `news_screen_max_headlines`, `sector_calendars_lookahead_days`, `meta_scorer_max_batch`
   - Brief sub-description per row explaining what the signal does
   - Disabled-state styling (`opacity-40 pointer-events-none`) on model+numeric controls when the toggle is off
   - Phosphor icons via `@/lib/icons` (Brain, TrendUp, Newspaper, Calendar, Scales — all already exported)
4. **Tests** at `tests/api/test_settings_api_signal_stack.py`:
   - `FullSettings()` instantiates with all 13 fields at correct defaults
   - `SettingsUpdate(...)` accepts each field individually
   - `SettingsUpdate(news_screen_max_headlines=600)` raises ValueError (above ge/le)
   - `SettingsUpdate(meta_scorer_max_batch=4)` raises ValueError (below floor)
   - `_settings_to_full(settings)` propagates the 13 fields from `Settings` to `FullSettings`

## Out of scope

- "Why this candidate" panel on the Signals or Paper Trading page (Brief Option A/B) — DEFERRED to Phase 2 as `phase-23.2.X`. Surfacing requires extending `screener.rank_candidates` to attach `_signal_tags` to each candidate dict (an additional contract surface) AND the API endpoint that exposes them. Tracking note added to `handoff/harness_log.md`.
- Model-pricing display in the Signal Stack card (Phase 2 — uses existing model_costs hooks)
- Signals page redesign (this cycle ONLY adds the backend + Settings UI)
- E2E browser test (Phase 2 — current dev cycle has no claude-in-chrome MCP available)

## Files modified

- `backend/api/settings_api.py` — 13 fields added in 5 places (FullSettings, SettingsUpdate, _FIELD_TO_ENV, _settings_to_full, model-validation loop)
- `frontend/src/lib/types.ts` — 13 new optional fields on `FullSettings`
- `frontend/src/app/settings/page.tsx` — NEW BentoCard with toggles + controls
- `tests/api/test_settings_api_signal_stack.py` — NEW (8 tests)

## Verification

Front-matter command does NOT require the frontend to be running. It verifies the 13 fields are wired through the backend settings layer (the necessary precondition for the UI to function). Frontend compile is verified separately via `cd frontend && npx tsc --noEmit && npm run lint` (recipe in experiment_results).

## References

- `handoff/current/phase-23.1.6-research-brief.md` — full brief (538 lines, 4 external + 13 internal files inspected, gate_passed: true)
- `frontend/src/app/settings/page.tsx:768-808` — existing BentoCard / toggle pattern to mirror
- `backend/api/settings_api.py:84-203` — current FullSettings + SettingsUpdate + _FIELD_TO_ENV
- `backend/config/settings.py:158-170` — the 13 source-of-truth fields from cycles 1-5
- `frontend/src/lib/types.ts` (≈line 549) — FullSettings TypeScript interface
- `frontend/src/lib/icons.ts` — Brain, TrendUp, Newspaper, Calendar, Scales (already exported)
