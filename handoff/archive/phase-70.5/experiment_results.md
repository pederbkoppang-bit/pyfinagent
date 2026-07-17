# Experiment results — step 70.5 (deposit-aware Starting-capital + cron reschedule)

**Phase/step:** phase-70 → 70.5 (LAST phase-70 step) | **Date:** 2026-07-17 | **Type:** frontend + backend. $0,
paper-only. live_check: Playwright capture of the Starting-capital display.

## Files changed (6)

1. **`frontend/src/app/paper-trading/manage/page.tsx`** — destructure `portfolio` from `usePaperTradingData()`;
   the Starting-capital ReadOnlyField renders `portfolio?.starting_capital ?? paper_starting_capital ?? 10000`
   (BQ deposit truth, live via `refresh()`) with a corrected hint; added a "Daily run hour (ET, 0-23)"
   `PaperSettingNum`.
2. **`frontend/src/components/paper-trading/cockpit-helpers.tsx`** — `paper_trading_hour` added to `PaperNumKey`.
3. **`frontend/src/lib/types.ts`** — `paper_trading_hour?` added to `FullSettings`; corrected the misleading
   `paper_starting_capital` comment.
4. **`backend/api/settings_api.py`** — `paper_trading_hour` added to `FullSettings`, `SettingsUpdate`
   (Field(None, ge=0, le=23)), `_FIELD_TO_ENV` (`PAPER_TRADING_HOUR`), `_settings_to_full`; `update_settings`
   calls `reschedule_paper_job(settings)` (fail-open) when `paper_trading_hour` is in the PUT body.
5. **`backend/api/paper_trading.py`** — `reschedule_paper_job(settings)`: reuses `_add_scheduler_job`
   (`add_job(replace_existing=True)` → fresh trigger + recomputed `next_run`, not `modify_job`), guarded by
   `_scheduler and _scheduler.get_job('paper_trading_daily')` (never creates the job when off), fail-open.
6. **`backend/tests/test_phase_70_5_reschedule.py`** (NEW) — 4 tests.

## Verification command output (verbatim)

```
$ bash -c 'grep -Eqi "paper_trading_hour" backend/api/settings_api.py backend/api/paper_trading.py && python -c "import ast; ast.parse(open(\"backend/api/paper_trading.py\").read())"'
VERIFICATION: PASS (exit 0)
$ python -m pytest backend/tests/test_phase_70_5_reschedule.py -q
4 passed
$ (cd frontend && npx tsc --noEmit)   # exit 0 (NEVER npm run build)
```
Import-smoke: settings_api + paper_trading import clean; `reschedule_paper_job` present.

## Criterion evidence

- **C1 (Starting capital reflects deposits) — LIVE Playwright (skip-auth :3100):** the display shows **$20 000**
  (BQ `paper_portfolio.starting_capital`, from a $10k deposit over the $10k config), NOT the stale $10,000;
  corrected hint "Reflects deposits (Top up fund)." `shows_20000: true, shows_10000_stale: false`. See
  `live_check_70.5.md` + `captures_70.5/70.5-starting-capital-deposit.png` (visually verified).
- **C2 (hour reschedules without restart):** `test_phase_70_5_reschedule.py` — `reschedule_paper_job` re-adds the
  cron with the new hour (`add_job(hour=18, replace_existing=True)` → fresh `next_run`), is a guarded no-op when
  no live job exists, and is fail-open; `paper_trading_hour` is now writable. Bootstrapping caveat: the running
  `:8000` is on pre-70.5 code, so live activation needs the standard one-time restart to load the new code;
  thereafter hour changes reschedule without a restart (documented in the live_check).
- **C3 (no cap-read regression):** the fresh-per-cycle reads of `paper_max_per_sector`/`paper_max_positions` are
  untouched (70.5 adds only a reschedule side-effect + a display fix; no settings caching, no cap-read move).

## Do-no-harm / scope
frontend + backend API only; $0; paper-only; NO risk threshold moved; reschedule fail-open (never 500s the save);
historical_macro FROZEN; no operator config mutated during the capture (read-only). Closes phase-70 (6/6).
