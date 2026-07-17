# live_check — step 70.5 (deposit-aware Starting-capital + cron reschedule)

## Method (phase-59.2 canonical)
Captured against a skip-auth `:3100` dev server (operator `:3000` untouched, verified `302→/login` after teardown).
Same code + same backend `:8000` + same BQ data as `:3000`. `@playwright/mcp@0.0.76`. Frontend `npx tsc --noEmit`
exit 0 (NOT `npm run build` — that clobbers the live `:3000` dev-server chunk manifest; phase-70.4 lesson).
Capture: `handoff/current/captures_70.5/70.5-starting-capital-deposit.png` (element screenshot, 1109×895).

## Criterion 1 — Starting capital reflects deposits (BQ truth, not stale config)

The operator's BQ `paper_portfolio.starting_capital` is **$20,000** (a $10k deposit over the $10k `.env` base);
the config `paper_starting_capital` is still **$10,000**. Pre-fix the Manage page rendered the stale $10,000;
post-fix it renders the live BQ value.

Live DOM + visual (screenshot):
```
STARTING CAPITAL: $20 000
hint: "Reflects deposits (Top up fund). Total P&L % is measured against this."
shows_20000: true | shows_10000_stale: false | hour_field_present: true
```
The fix reads `portfolio.starting_capital` from `usePaperTradingData()` (already exposed via
`/api/paper-trading/portfolio`), so it reflects deposits live via `refresh()` — no backend change, no new fetch.

## Criterion 2 — paper_trading_hour reschedules the cron without a restart

Proven by `backend/tests/test_phase_70_5_reschedule.py` (4 passed): `reschedule_paper_job` re-adds the cron via
`add_job(hour=<new>, replace_existing=True)` (APScheduler recomputes `next_run_time` from the fresh trigger —
deliberately not `modify_job`, GH#234), is a guarded no-op when no live job exists (never CREATES the job on a
settings PUT when paper trading is off), and is fail-open. `update_settings` calls it when `paper_trading_hour`
is in the PUT body; `paper_trading_hour` is now writable (in `SettingsUpdate` + `_FIELD_TO_ENV` + `FullSettings`)
and editable via the new Manage "Daily run hour (ET, 0-23)" field.

**Note (bootstrapping):** the running operator `:8000` is on pre-70.5 code (my `settings_api`/`paper_trading`
changes are on disk, not yet loaded — that is why the live GET returns `paper_trading_hour: null` and the hour
field renders empty). The reschedule takes effect after the standard ONE-time backend restart to load the new
code; thereafter every hour change reschedules WITHOUT a restart (which is the criterion). The reschedule logic
itself is proven by the unit tests against a mocked scheduler.

## Criterion 3 — no regression to fresh-per-cycle cap reads
Untouched: `paper_max_per_sector`/`paper_max_positions` are still read fresh each cycle via `_scheduled_run` →
`get_settings()` → `run_daily_cycle` → `autonomous_loop`/`portfolio_manager` (kept fresh by the PUT's
`cache_clear()`). 70.5 adds only a reschedule side-effect + a display fix; it does NOT cache settings for the
cycle body or move cap reads into `_add_scheduler_job`.

## Do-no-harm
$0; paper-only; NO risk threshold moved; reschedule fail-open; historical_macro FROZEN; no operator config
mutated during the capture (read-only navigation; no PUT issued).
