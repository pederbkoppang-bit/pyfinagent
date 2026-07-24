# Research Brief — phase-70.5 (Manage-page observability polish)

**Step:** 70.5 (LAST phase-70 step) — general/observability polish; $0, paper-only
**Researcher:** Layer-3 (effort max)
**Date:** 2026-07-17
**HEAD:** `03414593bec497ff11ea931a0123e48a1594dc96`
**Tier:** moderate (two independent, well-scoped fixes; one UI-label, one backend-scheduler; no algorithmic novelty, but a real APScheduler runtime-reschedule design decision + a config-vs-BQ-truth divergence)

---

## Objective (verbatim from spawn)

1. **Starting-capital display** on the Manage page must reflect deposits (match
   `paper_portfolio.starting_capital` after a Top-up) **OR** the label must be
   corrected to state it is the CONFIGURED base — no UI value that silently
   diverges from BQ truth. (Audit finding #17.)
2. **`paper_trading_hour` reschedule**: changing it via the settings surface must
   reschedule the APScheduler job WITHOUT a backend restart (audit finding #15).
   NOTE: `paper_trading_hour` is currently NOT in the writable `SettingsUpdate`
   model, so it is not even editable via the UI/PUT today — confirm and design
   accordingly.
3. **NO regression** to fresh-per-cycle reading of `paper_max_per_sector` /
   `paper_max_positions` (re-read each cycle via `get_settings()`).

**Binding constraints:** $0; paper-only; NO risk threshold moved; NO regression
to fresh-per-cycle cap reads; `historical_macro` FROZEN; frontend rules (no emoji,
Phosphor, navy/slate); any UI claim needs a live Playwright capture (Q/A gate).

---

## Internal RE-ANCHOR (exact HEAD lines — verified 2026-07-17)

`internal_files_inspected = 10` — `settings_api.py`, `paper_trading.py`,
`config/settings.py`, `main.py`, `services/autonomous_loop.py`,
`services/paper_trader.py` (backend); `manage/page.tsx`,
`components/paper-trading/cockpit-helpers.tsx`, `lib/paper-trading-context.tsx`,
`lib/types.ts` (frontend). All line references below verified at HEAD `03414593`.

### Finding #17 — Starting-capital divergence (config value vs BQ truth)

**The two divergent sources:**

| Source | Where | Value |
|---|---|---|
| CONFIG (what the UI shows) | `backend/config/settings.py:264` — `paper_starting_capital: float = Field(10000.0, ...)`; surfaced by `settings_api.py:362` `paper_starting_capital=float(getattr(s, "paper_starting_capital", 10000.0))` inside `_settings_to_full` | static `.env`/default (10000.0) — **never mutated by a deposit** |
| BQ TRUTH (what a deposit changes) | `backend/api/paper_trading.py:1249` — `deposit_funds` upserts `paper_portfolio.starting_capital` = `new_starting` (`starting_before + req.amount`) via `bq.upsert_paper_portfolio(updated)` at line 1254 | grows with every Top-up |

**The render path (stale value reaches the screen):**
- `frontend/src/app/paper-trading/manage/page.tsx:52` — `getFullSettings()` → `GET /api/settings/` → `settings_api.py:383` `get_all_settings` → `_settings_to_full(settings)` (line 391) → `paper_starting_capital` from CONFIG.
- `frontend/src/app/paper-trading/manage/page.tsx:212-216` renders:
  ```tsx
  <ReadOnlyField
    label="Starting capital"
    value={`$${(manageSettings.paper_starting_capital ?? 10000).toLocaleString()}`}
    hint="Adjust via Top up fund (deposit only)."
  />
  ```
- `ReadOnlyField` component: `frontend/src/components/paper-trading/cockpit-helpers.tsx:413-431` (label + read-only value box + optional hint; navy/slate tokens already correct).

**The divergence:** `paper_starting_capital` in `FullSettings` is `settings.paper_starting_capital` (config, immutable at runtime). `deposit_funds` mutates `paper_portfolio.starting_capital` in BQ but never touches the config field. After a Top-up, the deposit success banner reports the true `new_starting_capital` (page.tsx:79), yet the "Starting capital" ReadOnlyField still shows the config base (10000) → the two silently diverge. The hint even says "Adjust via Top up fund (deposit only)" — which is a lie about *this field* (deposits never move it).

**Where the BQ truth is already exposed (reuse target):**
- `paper_trading.py:152` — `GET /api/paper-trading/status` already returns
  `"starting_capital": portfolio.get("starting_capital")` (BQ truth).
- `paper_trading.py:335` — `GET /api/paper-trading/portfolio` also returns
  `"starting_capital": portfolio.get("starting_capital")` (BQ truth).
- Frontend already calls both: `getPaperTradingStatus()` and `getPaperPortfolio()`
  are imported in `manage/page.tsx:22-24` and invoked at line 85. So the true value
  is **already fetchable client-side with zero new backend work.**

### Finding #15 — `paper_trading_hour` captured once at startup

- `backend/config/settings.py:368` — `paper_trading_hour: int = Field(10, description="Hour (ET) to run daily trading cycle (0-23)")`.
- `backend/api/paper_trading.py:1299-1322` — `_add_scheduler_job(settings)` calls
  `_scheduler.add_job(_scheduled_run, "cron", hour=settings.paper_trading_hour, minute=0, day_of_week="mon-fri", timezone=ZoneInfo("America/New_York"), id=_scheduler_job_id, name="Paper trading daily run", replace_existing=True, misfire_grace_time=3600, coalesce=True)`. **`settings.paper_trading_hour` is read exactly once, at the moment the job is added** (startup via `init_scheduler`).
- `backend/api/paper_trading.py:1289-1296` — `init_scheduler(scheduler)` stores the
  scheduler in module global `_scheduler` (line 1292) and calls `_add_scheduler_job`
  once if `settings.paper_trading_enabled`.
- Scheduler object: `backend/main.py:266-269` — `from apscheduler.schedulers.asyncio import AsyncIOScheduler; scheduler = AsyncIOScheduler(); init_scheduler(scheduler); scheduler.start()`. So `_scheduler` is an **`AsyncIOScheduler`** (in-memory default MemoryJobStore; no serialization).
- Job id: `backend/api/paper_trading.py:38` — `_scheduler_job_id = "paper_trading_daily"`.
- **`paper_trading_hour` is NOT writable today:** absent from `SettingsUpdate`
  (`settings_api.py:125-170`) AND absent from `_FIELD_TO_ENV` (`settings_api.py:257-308`).
  So `PUT /api/settings/` cannot change it, and even if it could, `update_settings`
  (`settings_api.py:397-450`) only writes `.env` + `get_settings.cache_clear()` +
  `get_api_cache().invalidate("settings:*")` (lines 445-446) — it has **no scheduler
  reference and no reschedule call**. `FullSettings` also does not expose it
  (`settings_api.py:100-122` lists `paper_starting_capital` … `paper_cycle_max_seconds`
  but no `paper_trading_hour`).
- Installed APScheduler: **3.11.2** (`backend/requirements.txt:55` `APScheduler>=3.10.0`;
  venv reports 3.11.2). → 3.x API (`reschedule_job` / `modify_job` / `add_job(replace_existing=True)`).

### Regression guard #3 — caps ARE re-read fresh each cycle (must NOT break)

- `backend/api/paper_trading.py:1325-1332` — `_scheduled_run` calls
  `settings = get_settings()` fresh (line 1327), then `run_daily_cycle(settings)`.
- `backend/services/autonomous_loop.py:252` — `run_daily_cycle(settings=None, ...)`;
  line 309 `settings = settings or get_settings()`; line 1367 reads
  `max_per_sector = int(getattr(settings, "paper_max_per_sector", 0) or 0)` from that
  fresh settings; `backend/services/paper_trader.py:221` reads
  `self.settings.paper_max_positions` (PaperTrader is constructed per-cycle with the
  fresh settings). Because `update_settings` calls `get_settings.cache_clear()`
  (`settings_api.py:445`), the NEXT cycle's `get_settings()` returns the new caps.
  **This path is orthogonal to the scheduler trigger** — rescheduling the job does
  not touch it, so the correct 70.5 fix leaves it intact. (Guard: do NOT move cap
  reads to startup/module scope; do NOT capture settings once in `init_scheduler`.)

---

## Source table

| # | Source | Tier | Read in full | Relevance |
|---|--------|------|--------------|-----------|
| S1 | [APScheduler 3.x User Guide](https://apscheduler.readthedocs.io/en/3.x/userguide.html) | 2 Official docs | YES | reschedule/modify jobs at runtime; scheduler must be running |
| S2 | [APScheduler 3.x `schedulers.base` module API](https://apscheduler.readthedocs.io/en/3.x/modules/schedulers/base.html) | 2 Official docs | YES | exact signatures `reschedule_job`/`modify_job`/`add_job(replace_existing)` |
| S3 | [APScheduler GitHub issue #234](https://github.com/agronholm/apscheduler/issues/234) | 2 Official repo | YES | later→earlier cron `modify_job` stale-next_run_time gotcha |
| S4 | [Smashing Magazine — UX Strategies for Real-Time Dashboards (Sep 2025)](https://www.smashingmagazine.com/2025/09/ux-strategies-real-time-dashboards/) | 3 Authoritative blog | YES | live vs stale/cached/configured value labeling; "reveal the true state" |
| S5 | [APScheduler 3.x `job` module (Job.reschedule/modify)](https://apscheduler.readthedocs.io/en/3.x/modules/job.html) | 2 Official docs | YES | Job-level `reschedule`/`modify`/`remove`; attributes incl. `next_run_time` |
| S6 | [Pencil & Paper — Visibility of System Status (NN/g heuristic)](https://www.pencilandpaper.io/articles/visibility-system-status) | 3 Authoritative blog | YES | show the actual current state, not a default; financial-context redundancy |

**Snippet-only set (evaluated, not read in full):**

| URL | Why snippet-only |
|---|---|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | CronTrigger arg reference — covered by S1/S2 |
| https://apscheduler.readthedocs.io/en/master/api.html | APScheduler 4.x API — NOT our version (3.11.2); avoided to prevent 4.x/3.x confusion |
| https://github.com/agronholm/apscheduler/issues/471 | "dynamically add jobs" — duplicate of S1 guidance |
| https://www.peterspython.com/en/blog/basic-job-scheduling-with-apscheduler | community tutorial — lower tier than official docs |
| https://docs.openwebui.com/reference/env-configuration/ | strong analog: persisted DB value silently overrides env var (mirror of finding #17) — snippet captured the key point |
| https://www.nngroup.com / https://www.pencilandpaper.io (NN/g heuristic origin) | heuristic origin — S6 read in full |
| https://www.uxpin.com/studio/blog/dashboard-design-principles/ | dashboard principles roundup — S4 stronger |
| https://excited.agency/blog/dashboard-ux-design | dashboard UX roundup — S4 stronger |
| https://www.saasalliance.io/how-to-create-a-value-based-dashboard-for-your-users/ | value-based dashboards — off-target |
| https://learn.microsoft.com/.../uielement.visibility | WPF Visibility property — false-positive keyword hit |
| https://developer.android.com/training/system-ui | Android system UI — false-positive keyword hit |

**URLs collected: 17** (6 read in full + 11 snippet-only).

---

## Per-claim citations (external) — populated as sources are read

- **C1 (APScheduler can reschedule a cron job at runtime without restart):** S1, S2.
- **C2 (exact method to change the hour):** S2 —
  `reschedule_job(job_id, trigger='cron', hour=H, minute=0, day_of_week='mon-fri', timezone=...)`.
- **C3 (add_job replace_existing=True already updates in place):** S2.
- **C4 (later→earlier cron modify gotcha):** S3.
- **C5 (dashboards must not show stale config as if live; label or show live):** S4, S6.

---

## Recency scan (last 2 years)

**APScheduler (topic A):** No API drift relevant to us in the last 2 years. The
project is pinned to **APScheduler 3.11.2** (3.x line; `backend/requirements.txt:55`).
The 3.x `reschedule_job`/`modify_job`/`add_job(replace_existing=True)` surface is
stable across the whole 3.x series (docs served as `3.11.2.post*`). APScheduler
**4.x** exists (readthedocs `/en/master/`) with a redesigned API
(`scheduler.add_schedule` / `configure_task`), but adopting it is out of scope and
would be a breaking dependency change — the 3.x methods below are the correct target.
No new finding supersedes the reschedule guidance. The 2017 issue #234 (S3) is old
but still the canonical characterization of the `modify_job` later→earlier gotcha,
and it directly informs the "use a fresh trigger" recommendation below.

**Dashboard config-vs-live (topic B):** 1 new finding in-window — Smashing Magazine
(Sep 2025, S4) crystallizes the modern pattern: when a shown value is not the live
one, label it explicitly ("Data as of…", freshness widget, status color) and
"reveal the true state of the system"; "hiding failures damages trust more than
acknowledging them." This complements (does not supersede) the older canonical NN/g
"Visibility of System Status" heuristic (S6). Both point the same way: the Manage
page must either show the true (BQ) starting capital or relabel the field so it does
not masquerade as the post-deposit value.

---

## 3-variant query disclosure

**Topic A — APScheduler runtime reschedule:**
1. Current-year frontier (2026): `"APScheduler reschedule_job modify_job change cron trigger at runtime 2026"`.
2. Year-less canonical: `"APScheduler change job schedule dynamically without restart"`.
3. Last-2-year / version pin: covered by the version-pinned `3.x` docs (3.11.2.post*)
   surfaced by both queries — the docs are inherently version-stamped, so a bare
   `2025` query adds nothing over the pinned-version doc hits.

**Topic B — config-vs-live dashboard value:**
1. Current-year frontier (2026): `"visibility of system status show actual value not configured default UI 2026"`.
2. Year-less canonical: `"dashboard show effective live value not stale configured value settings UI clarity"` (surfaced the NN/g heuristic canon + Smashing).
3. Last-2-year: the Smashing Magazine 2025 source (S4) is the in-window hit.

---

## Design recommendations

Both fixes are small and independent. #17 is **frontend-only**; #15 is
**backend + frontend**. Neither touches a risk threshold, the fresh-per-cycle cap
reads, or `historical_macro`.

### Fix #17 — Starting-capital display (RECOMMEND: re-fetch BQ truth; UI-only)

**Root cause (re-anchored above):** the ReadOnlyField shows the CONFIG field
`FullSettings.paper_starting_capital` (`.env` base, immutable at runtime), while a
Top-up mutates `paper_portfolio.starting_capital` in BQ. They diverge after any
deposit, and the hint "Adjust via Top up fund (deposit only)" falsely implies the
displayed field tracks deposits. (types.ts:570 even mis-comments the config field as
"mutated only via deposit" — the confusion is codified in the type.)

**Recommended (minimal + correct): show the BQ truth, keep the hint.** The true value
is **already available client-side with zero new fetch and zero backend change**:
- `usePaperTradingData()` already exposes `portfolio: PaperPortfolio | null`
  (`frontend/src/lib/paper-trading-context.tsx:32`; `PaperPortfolio.starting_capital:
  number` at `frontend/src/lib/types.ts:617`), fetched by the paper-trading layout.
- The Manage page already destructures the context: `const { refresh } =
  usePaperTradingData();` (`manage/page.tsx:30`).

Change (frontend only, `frontend/src/app/paper-trading/manage/page.tsx`):
1. Line 30 → `const { refresh, portfolio } = usePaperTradingData();`
2. Lines 212-216 → render `portfolio?.starting_capital ?? manageSettings.paper_starting_capital ?? 10000`:
   ```tsx
   <ReadOnlyField
     label="Starting capital"
     value={`$${(portfolio?.starting_capital ?? manageSettings.paper_starting_capital ?? 10000).toLocaleString()}`}
     hint="Reflects deposits (Top up fund). Base is set at portfolio init."
   />
   ```
   The `?? manageSettings.paper_starting_capital` keeps a correct value on first
   paint before the portfolio resolves. After a deposit, `handleDeposit` already calls
   `refresh()` (line 86), which re-fetches the context `portfolio` → the field updates
   live. This makes the existing hint TRUE and matches the deposit success banner's
   `new_starting_capital` (line 79). **Change type: UI (frontend) only. Files:
   `frontend/src/app/paper-trading/manage/page.tsx` (2 lines).** No backend change —
   BQ truth is already exposed at `paper_trading.py:152` (`/status`) and `:335`
   (`/portfolio`). Grounded in S4 + S6: show the true state, don't render a config
   default as if it were live.

**Fallback (also acceptable per the spawn's "OR"): relabel.** If, for any reason, the
team prefers zero coupling to the portfolio object, keep the config value but make the
label honest — label `"Configured base capital"`, hint `"The .env-configured seed for
a fresh portfolio. Your live starting capital (after deposits) is shown on Overview /
in the Top-up banner."` This is a pure-string frontend change (kills the silent
divergence by no longer claiming to reflect deposits) but is strictly less useful than
showing the BQ truth, so it is the fallback, not the primary. **Change type: UI only.**

> Recommendation: **re-fetch** (primary). It costs ~2 lines, uses data already in the
> context, preserves the useful information, and makes the existing hint accurate. The
> fresh-per-cycle cap reads (#3) are untouched — this is a pure display change.

### Fix #15 — `paper_trading_hour` reschedule without restart (backend + frontend)

**Root cause (re-anchored):** `paper_trading_hour` is captured ONCE at startup in
`_add_scheduler_job` (`paper_trading.py:1305`), and is not writable at all today
(absent from `SettingsUpdate`, `_FIELD_TO_ENV`, and `FullSettings`). To satisfy the
audit ("changing it reschedules the job without a restart") we must make it writable
AND reschedule on the change.

**APScheduler mechanism (S1, S2, S3, S5 — version 3.11.2):**
- `reschedule_job(job_id, jobstore=None, trigger=None, **trigger_args)` — "Constructs
  a new trigger for a job and updates its next run time." This is the documented way
  to change a cron job's hour at runtime; it **recomputes `next_run_time`** (S1/S2).
- `modify_job(job_id, jobstore=None, **changes)` — general attribute changes;
  **does NOT recompute `next_run_time`** for a trigger swap unless you also pass
  `next_run_time`. Issue #234 (S3) is exactly this trap: a later→earlier cron change
  via `modify_job` left the stale `next_run_time` = tomorrow. **Do NOT use `modify_job`
  for the trigger change.**
- `add_job(..., replace_existing=True)` — "replace an existing job with the same id
  (but retain the number of runs)." Builds a **fresh Job with a fresh trigger and a
  freshly-computed `next_run_time`**, so it sidesteps the #234 stale-time trap. The
  scheduler must be running for the change to take effect immediately (it is —
  `main.py:269` `scheduler.start()`).

**Recommended (minimal + DRY + gotcha-proof): reuse `_add_scheduler_job` via
`add_job(replace_existing=True)`.** `_add_scheduler_job` is ALREADY the single
definition of the cron args (hour, minute=0, day_of_week="mon-fri", America/New_York
tz, misfire_grace_time=3600, coalesce=True) and already passes `replace_existing=True`.
Re-invoking it with fresh settings replaces the job in place with the new hour, in ONE
place, with a fresh next_run_time — no duplicated cron config, no #234 risk.

Backend changes:
1. `backend/api/settings_api.py`
   - `SettingsUpdate` (after line 170): add
     `paper_trading_hour: Optional[int] = Field(None, ge=0, le=23)`.
   - `_FIELD_TO_ENV` (in the paper block ~line 295-307): add
     `"paper_trading_hour": "PAPER_TRADING_HOUR"`.
   - `FullSettings` (paper block ~line 100-122): add `paper_trading_hour: int = 10`;
     and in `_settings_to_full` (~line 361-377) add
     `paper_trading_hour=int(getattr(s, "paper_trading_hour", 10))` so the UI can read
     the current value.
   - In `update_settings` (after the `get_settings.cache_clear()` at line 445 and the
     fresh `settings = get_settings()` at line 447): if `"paper_trading_hour" in
     updates`, reschedule:
     ```python
     if "paper_trading_hour" in updates:
         try:
             from backend.api.paper_trading import reschedule_paper_job
             reschedule_paper_job(settings)  # fresh settings has the new hour
         except Exception:
             logger.warning("paper_trading_hour reschedule failed", exc_info=True)
     ```
     Function-local import avoids any import-order coupling (verified: neither module
     imports the other, so a top-level import would also be safe — function-local is
     the conservative choice). Fail-open: a reschedule error must not 500 the settings
     save (the .env value is already persisted; next restart picks it up).
2. `backend/api/paper_trading.py` — add a thin wrapper near `_add_scheduler_job`
   (~line 1298):
   ```python
   def reschedule_paper_job(settings) -> bool:
       """Re-point the daily cron at the new paper_trading_hour without a restart.
       Reuses _add_scheduler_job (add_job replace_existing=True) so the cron args
       live in ONE place and next_run_time is recomputed fresh (avoids APScheduler
       issue #234's modify_job later->earlier stale-time trap)."""
       if not _scheduler or not _scheduler.get_job(_scheduler_job_id):
           return False  # scheduler disabled / job not running -> nothing to do
       _add_scheduler_job(settings)
       logger.info("Paper trading rescheduled: daily at %s:00 ET", settings.paper_trading_hour)
       return True
   ```
   Guarded so a settings PUT never *creates* the job when paper trading is disabled
   (only reschedules an existing one). `_add_scheduler_job` already returns early if
   `_scheduler is None`.

   **Alternative (more surgical, but duplicates cron args):**
   `_scheduler.reschedule_job(_scheduler_job_id, trigger="cron", hour=settings.paper_trading_hour,
   minute=0, day_of_week="mon-fri", timezone=ZoneInfo("America/New_York"))`. This
   preserves misfire_grace_time/coalesce (only swaps the trigger) and also recomputes
   next_run_time, but re-states dow/minute/tz in a second place — DRY loss + drift
   risk vs. reusing `_add_scheduler_job`. Prefer the wrapper above.

Frontend change (to make it operable + satisfy "via the settings surface" end-to-end):
3. `frontend/src/app/paper-trading/manage/page.tsx` — add an hour field (0-23) in the
   Trading-settings grid. `PaperSettingNum` currently keys off the `PaperNumKey` union
   (`cockpit-helpers.tsx:433-443`) which does NOT include `paper_trading_hour`; extend
   that union (and `SettingsUpdate`'s TS type in `types.ts`) to include
   `paper_trading_hour`, or add a small dedicated hour input. Add
   `paper_trading_hour?: number` to the `FullSettings` + `SettingsUpdate` TS types in
   `frontend/src/lib/types.ts`. **Change type: UI + type.** A live Playwright capture
   is required (edit hour → Save → confirm no 422 and the value round-trips; Q/A gate).

**Backend-only vs full-stack framing for #15:** the load-bearing audit fix is the
backend (writable field + reschedule-on-PUT). The frontend field is what makes it
usable "via the settings surface" from the Manage page; without it the only way to
exercise the new PUT is a raw API call. Recommend shipping both, but if scope must be
cut, the backend is mandatory and the UI field is the operability layer.

### Regression guard #3 (fresh-per-cycle caps) — untouched by both fixes

Neither fix goes near the cap-read path. `paper_max_positions` /
`paper_max_per_sector` are read fresh each cycle: `_scheduled_run` →
`get_settings()` (paper_trading.py:1327) → `run_daily_cycle(settings)` →
`autonomous_loop.py:1367` (`paper_max_per_sector`) and `paper_trader.py:221`
(`paper_max_positions`), and `update_settings` already calls
`get_settings.cache_clear()` (settings_api.py:445) so the next cycle sees new caps.
**Guard for GENERATE:** do NOT capture settings once at module/`init_scheduler`
scope for caps; do NOT move cap reads into `_add_scheduler_job`. The scheduler
reschedule only swaps the cron trigger — it must not cache settings for the cycle body.

### UI-change checklist (both fixes)
- No emoji; Phosphor icons only; navy/slate palette (ReadOnlyField already compliant:
  `cockpit-helpers.tsx:422-430`).
- Any UI claim (Starting-capital reflects a deposit; hour field saves + reschedules)
  needs a **live Playwright capture** on the skip-auth :3100 instance per
  `.claude/rules/frontend.md` "Live-UI verification" (Q/A gate), operator :3000
  untouched.

---

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 10,
  "gate_passed": true
}
```
