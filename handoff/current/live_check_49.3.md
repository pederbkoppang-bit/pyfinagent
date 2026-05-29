# live_check_49.3 -- cron-control UI (autonomous evidence + operator visual section)

## Autonomous evidence (verified 2026-05-29)

### Build + types (criterion #4)
- `cd frontend && npx tsc --noEmit` -> **0 type-errors**.
- `cd frontend && npm run build` -> **EXIT=0, "✓ Compiled successfully"**, full route table printed (incl. /cron), no prerender error.
- Build-fix discovered during verification: `/observability/page.tsx` had a PRE-EXISTING SSG-prerender break ("a[d] is not a function" in webpack-runtime) -- a live-data client page that must not be statically prerendered; added `export const dynamic = "force-dynamic"`. This is latent (the app runs on `next dev`, which is why it went unnoticed) and unrelated to the cron change, but it blocked `npm run build`, so fixing it was necessary to satisfy criterion #4. NOTE: one INTERMITTENT `/agents` webpack flake ("Cannot find module for page") was observed on one of three build runs and did NOT recur; it is a pre-existing Next 15 SSG-worker/cache flakiness (a `rm -rf frontend/.next` clean build is the standard mitigation -- I am permission-denied for `rm -rf`; operator can run it). Tracked as a follow-on.

### API wiring (criterion #1) -- matches backend cron_dashboard_api.py
- frontend/src/lib/api.ts:384-411 -- `pauseJob`/`resumeJob`/`triggerJob` POST to `/api/jobs/{encodeURIComponent(id)}/pause|resume|trigger` with confirmation tokens `PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB` + a reason.
- frontend/src/lib/types.ts:1141 -- `JobInfo.controllable?: boolean` (optional, matches backend emitting it only on main_apscheduler rows); :1146 -- `JobControlResponse`.

### UI gating + conventions (criteria #2, #3)
- cron/page.tsx: Actions column renders pause/resume ONLY when `j.controllable===true` (line 364) and a trigger button ONLY when `j.id==='paper_trading_daily'` (line 390 -- the backend ticket_queue->400 asymmetry). pause/resume toggle by `j.status==='paused'`.
- Per-row in-flight `SpinnerGap animate-spin` (line 366); `window.confirm` before each action; `await load()` after (pessimistic, server-confirmed status); an actionError rose banner with dismiss.
- Icons (`Pause`/`Play`/`Lightning`/`SpinnerGap`) imported from `@/lib/icons` (line 16); NO direct `@phosphor-icons/react` import; NO emoji (grep clean); `aria-label` on every icon button.
- Subtitle updated from "Read-only." to reflect the new controls.
- Backend endpoints already live-verified in phase-49.2 (pause/resume round-trip + 404/400 cases), so the UI calls a proven API.

### Frontend dev server
- `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` -> `GET http://localhost:3000/` returns HTTP 302 (auth redirect to /login -- server up + serving).

## OPERATOR TO CONFIRM (visual render -- cannot be verified autonomously behind the NextAuth wall, per frontend.md rule 5)
Load `/cron` (authenticated) and confirm:
1. The **Jobs** tab table shows a new **Actions** column.
2. Action buttons appear ONLY on the 2 backend-owned rows (`paper_trading_daily`, `ticket_queue_process_batch`); other rows show `--`. The **Trigger** (lightning) button appears ONLY on `paper_trading_daily`.
3. Dark-theme contrast is correct (amber Pause / emerald Play / sky Trigger on navy).
4. Live round-trip: click Pause on `paper_trading_daily` -> confirm -> the Status pill flips to `paused` (amber) + next-run clears; click Resume -> back to `scheduled`. (Leave it RESUMED.)
5. Trigger does not double-fire (returns 409 if a cycle is already running).
