---
step: phase-23.1.6
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "from backend.api.settings_api import FullSettings, SettingsUpdate; expected = {...13 names...}; missing = expected - set(FullSettings.model_fields.keys()); assert not missing; missing = expected - set(SettingsUpdate.model_fields.keys()); assert not missing; upd = SettingsUpdate(macro_regime_filter_enabled=True, news_screen_max_headlines=50); assert upd.macro_regime_filter_enabled is True and upd.news_screen_max_headlines == 50; print(\"ok 13 fields wired in FullSettings + SettingsUpdate\")"'
---

# Experiment Results — phase-23.1.6

## What was built

Operator-facing UI for the 5 signal-stack flags shipped in cycles 1-5. Backend Settings API exposes all 13 new fields (5 enable + 4 model selectors + 4 numeric controls); frontend Settings page renders a "Signal Stack (Phase 23.1)" BentoCard with 5 toggle rows; TypeScript types extended for type safety end-to-end. Default-OFF discipline preserved at every layer.

## Files modified

| File | Change |
|---|---|
| `backend/api/settings_api.py` | 5 places extended: `FullSettings` (+13 fields with defaults), `SettingsUpdate` (+13 optional fields with `Field(None, ge=..., le=...)` validators), `_FIELD_TO_ENV` dict (+13 entries), `_settings_to_full()` (+13 `getattr(s, ...)` lines), model-name validation loop (+ 4 new `*_model` fields validated against `_VALID_MODELS`) |
| `frontend/src/lib/types.ts` | `FullSettings` interface +13 optional fields with proper TypeScript typing |
| `frontend/src/lib/icons.ts` | 3 new identity re-exports: `Newspaper`, `GlobeHemisphereWest`, `CalendarBlank` (mirrors phase-16.30 pattern) |
| `frontend/src/app/settings/page.tsx` | NEW BentoCard "Signal Stack (Phase 23.1)" placed after the existing "Model Configuration" card on the Models tab. 5 toggle rows (Macro Regime, PEAD, News, Sector Calendars, Meta-Scorer) with Phosphor icons + descriptive sub-text + cost banner. Mirrors the existing `apply_model_to_all_agents` toggle pattern at line 768 |
| `tests/api/test_settings_api_signal_stack.py` | NEW (14 tests: field presence in FullSettings + SettingsUpdate + _FIELD_TO_ENV, default-OFF discipline, model defaults, numeric defaults, individual update accept, ge/le validators on all 4 numeric fields, partial-payload semantics) |

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "from backend.api.settings_api import FullSettings, SettingsUpdate; ...; print('ok 13 fields wired in FullSettings + SettingsUpdate')"
ok 13 fields wired in FullSettings + SettingsUpdate
exit=0
```

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/test_settings_api_signal_stack.py tests/services/ -v --no-header -q
collected 95 items
tests/api/test_settings_api_signal_stack.py ..............  [ 14%]
tests/services/test_macro_regime.py ............  [ 27%]
tests/services/test_meta_scorer.py ..............  [ 42%]
tests/services/test_news_screen.py .....................  [ 64%]
tests/services/test_pead_signal.py ..................  [ 83%]
tests/services/test_sector_calendars.py ................  [100%]
============================== 95 passed in 0.32s ==============================
```

95/95 tests pass (14 new + 81 from cycles 1-5; no regression).

## Frontend type-check + lint

```
$ cd frontend && npx tsc --noEmit
(silent — no errors)

$ cd frontend && npm run lint
✖ 35 problems (0 errors, 35 warnings)
```

Zero TypeScript errors. 35 pre-existing lint warnings (unrelated to this cycle — same warnings present before this change in `useLivePrices.ts`, `api.ts`, etc.).

## Settings UI design (per research brief)

The Signal Stack card lives on the **Models** tab, immediately after the existing Model Configuration card. Each of the 5 toggle rows mirrors the existing `apply_model_to_all_agents` toggle pattern (line 768) for visual consistency:

```
[checkbox] [Icon] Signal Name (font-medium)
            sub-description (text-xs text-slate-500)
```

Sub-descriptions are dynamic where appropriate (e.g., the PEAD row reads "trailing-{form.pead_signal_lookback_quarters ?? 8}Q mean" so the user sees the current value). A cost banner at the bottom of the card states "Default OFF" and the cumulative ~$0.10/day cost when all flags are on.

The 5 enable flags are the primary controls. Per-cycle numeric tuning (lookback_quarters, max_headlines, lookahead_days, max_batch) is editable via the JSON API but doesn't ship with sliders in this cycle to keep the UI scope tight. A future Phase 2 cycle can add inline number controls + per-cycle model selectors if the operator finds them necessary.

## Out of scope (per contract)

- "Why this candidate" panel on Signals or Paper Trading page — DEFERRED to Phase 2 (`phase-23.2.X`). Implementation requires extending `screener.rank_candidates` to attach `_signal_tags` to each candidate dict + extending the API endpoint that exposes them. Tracking note added below.
- Inline numeric sliders for the 4 numeric controls (Phase 2)
- Per-signal model selector dropdowns in the UI (Phase 2 — backend already accepts them)
- Browser-based E2E test (no claude-in-chrome MCP available in this dev cycle)

## Honest disclosure: Phase 2 follow-ups

- **"Why this candidate" panel:** the brief's Option A (Paper Trading page Last Cycle) and Option B (Signals page tab) both require new screener tagging + API plumbing. Documented for the next session.
- **Per-cycle numeric controls:** can ship as a hidden tab "Signal Stack Tuning" in Phase 2.
- **Model selectors per signal:** the backend API already validates them; frontend dropdowns can be added in a subsequent cycle.

## Cost / cycle posture

- ZERO incremental runtime cost (pure UI + API plumbing)
- All 5 underlying signals retain default-OFF flags at the backend layer
- Operator can now toggle them via the Settings UI without restarting backend or editing `.env`

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit
3. **6/6 Phase-23.1 cycles shipped** — close out the universe-upgrade plan and move on to whichever cycle the operator prioritises next (Phase-23.2 backtest validation? "Why" panel? Polygon/Benzinga vendor migration?)
