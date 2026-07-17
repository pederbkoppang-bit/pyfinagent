# Contract — step 70.5 (General/observability polish: deposit-aware Starting-capital + cron reschedule)

**Phase:** phase-70 (LAST step) | **Step:** 70.5 | **Priority:** P3 | harness_required: true
**Cycle:** 1 | Date: 2026-07-17 | **Type:** frontend (UI) + backend. $0, paper-only. live_check: Playwright
capture of the Starting-capital display post/with-deposit.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0). Envelope: **gate_passed=true**, tier=moderate,
**6 external sources read in full**, 11 snippet-only, 17 URLs, recency scan performed, 10 internal files
re-anchored on HEAD 03414593. Brief: `research_brief_70.5.md`. Grounding: APScheduler 3.x docs (reschedule via a
fresh trigger; `modify_job` leaves a stale `next_run_time` on a trigger swap — GH#234, so reuse
`add_job(replace_existing=True)`); Smashing/NN-g dashboard heuristics (show the live/effective value, never a
stale configured value; label non-live values honestly).

## Confirmed on HEAD

- `FullSettings.paper_starting_capital` (settings_api.py:362) is the .env CONFIG (immutable at runtime), but
  `deposit_funds` upserts `paper_portfolio.starting_capital` in BQ (paper_trading.py:1249) — they diverge after a
  Top-up, and the Manage page renders the stale config (manage/page.tsx:214) with a hint that falsely implies it
  tracks deposits. BQ truth is already exposed via the paper-trading context (`portfolio.starting_capital`).
- `paper_trading_hour` (settings.py:368, default 10) is captured ONCE at startup (`_add_scheduler_job` :1305) and
  is NOT writable (absent from `SettingsUpdate` :156-170, `_FIELD_TO_ENV` :295, `FullSettings`).

## Design (minimal correct)

**#17 Starting capital — UI-only:** destructure `portfolio` from `usePaperTradingData()` and render
`portfolio?.starting_capital ?? manageSettings.paper_starting_capital ?? 10000` in the ReadOnlyField; `refresh()`
(page.tsx:86) updates it live after a deposit. Correct the hint to the truth. No backend/no new fetch.

**#15 Cron reschedule — backend + frontend:** add `paper_trading_hour` to `SettingsUpdate` (Field(None, ge=0,
le=23)), `_FIELD_TO_ENV` (`PAPER_TRADING_HOUR`), `FullSettings` + `_settings_to_full`; in `update_settings`, after
`get_settings.cache_clear()` + fresh `get_settings()`, if `paper_trading_hour` is in the body call a NEW
`reschedule_paper_job(settings)` in paper_trading.py — it reuses `_add_scheduler_job(settings)`
(`add_job(replace_existing=True)` → fresh trigger + recomputed `next_run`), GUARDED by
`_scheduler and _scheduler.get_job('paper_trading_daily')` (never creates the job when paper trading is off),
fail-open (a reschedule error must never 500 the save). Frontend: add a 0–23 hour `PaperSettingNum` to the Manage
grid + extend `FullSettings`/`PaperNumKey` TS types.

## Immutable success criteria (verbatim from masterplan.json 70.5)

1. The Manage-page Starting capital reflects deposits (matches paper_portfolio.starting_capital after a Top-up),
   OR the label is corrected to state it is the configured base -- no UI value that silently diverges from the BQ
   truth
2. Changing paper_trading_hour via the settings surface reschedules the APScheduler job without a backend restart
   (verified by the job's next_run reflecting the new hour)
3. No regression to the fresh-per-cycle reading of the position/sector caps

Verification command (immutable):
`bash -c 'grep -Eqi "paper_trading_hour" backend/api/settings_api.py backend/api/paper_trading.py && python -c "import ast; ast.parse(open(\'backend/api/paper_trading.py\').read())"'`
Live check: `live_check_70.5.md` with a Playwright capture of the Starting-capital display post-deposit.

## Plan
2 (this contract). 3. GENERATE: manage/page.tsx (Starting-capital re-fetch + hint + hour input); settings_api.py
(SettingsUpdate/_FIELD_TO_ENV/FullSettings/_settings_to_full + reschedule call); paper_trading.py
(reschedule_paper_job); types.ts + cockpit-helpers.tsx PaperNumKey (paper_trading_hour); test
test_phase_70_5_reschedule.py. Verify: command + import-smoke + `npx tsc --noEmit` (NEVER npm run build — clobbers
the live :3000 dev server) + pytest. Live Playwright capture (skip-auth :3100). 4. Q/A + live gate. 5. LOG. 6. FLIP.

## Boundaries (binding)
$0; paper-only; NO risk threshold moved; NO regression to fresh-per-cycle cap reads (do NOT cache settings for the
cycle body or move cap reads into _add_scheduler_job); historical_macro FROZEN; reschedule fail-open; frontend
rules (no emoji, Phosphor, navy/slate); harness stays 3 agents.

## References
research_brief_70.5.md; confirmed_findings.json (#15/#17). Code: settings_api.py:156-170/295/362/445,
paper_trading.py:38/1249/1289-1322, manage/page.tsx:30/212-216, cockpit-helpers.tsx:433, types.ts:570/617,
config/settings.py:368.
