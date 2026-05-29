# Experiment results -- phase-49.3: Cron-control UI

**Date:** 2026-05-29 | **Result: built + autonomously-verified (build/types/API/conventions); visual render = operator live_check** | $0 LLM | Frontend (Next 15).

## What was built
An Actions column on the existing `/cron` dashboard page exposing the phase-49.2 cron-control endpoints: pause/resume on the 2 backend-owned rows, trigger on `paper_trading_daily` only. Plus the api.ts client methods + the type field.

## Files changed
1. **frontend/src/lib/types.ts** -- `JobInfo += controllable?: boolean` (optional; backend emits it only on main_apscheduler rows) + new `JobControlResponse` interface.
2. **frontend/src/lib/api.ts** -- `pauseJob`/`resumeJob`/`triggerJob` (POST `/api/jobs/{encodeURIComponent(id)}/pause|resume|trigger`, tokens PAUSE_JOB/RESUME_JOB/TRIGGER_JOB + reason), mirroring `postPaperKillSwitchAction`.
3. **frontend/src/app/cron/page.tsx** -- Actions column: pause/resume when `controllable===true` (toggled by status), trigger when `j.id==='paper_trading_daily'`; `window.confirm` + per-row `SpinnerGap` + pessimistic `await load()`; actionError rose banner w/ dismiss; icons from `@/lib/icons`; aria-labels; subtitle updated from "Read-only.".
4. **frontend/src/app/observability/page.tsx** -- `export const dynamic = "force-dynamic"` to fix a PRE-EXISTING SSG-prerender break that blocked `npm run build` (necessary to satisfy criterion #4; unrelated to cron but discovered during build verification).

## Verification (autonomous)
- `npx tsc --noEmit` -> **0 type-errors**.
- `npm run build` -> **EXIT=0, "✓ Compiled successfully"**, full route table (incl. /cron), no prerender error.
- API-wiring grep: api.ts:384-411 has the 3 methods + exact tokens; types.ts:1141 has controllable + :1146 JobControlResponse.
- Conventions: icons via `@/lib/icons` (no direct @phosphor-icons import); NO emoji (grep clean); trigger gated to paper_trading_daily (cron/page.tsx:390); pause/resume gated to controllable (:364); aria-label on every button.
- Frontend dev server restarted (kickstart) -> HTTP 302 (auth redirect; up + serving).
- Backend endpoints already live-verified in 49.2, so the UI calls a proven API.

## Success criteria mapping
1. types + api.ts methods/tokens -- YES (grep proofs).
2. Actions column gated to controllable + trigger gated to paper_trading_daily + subtitle updated -- YES.
3. confirmation + in-flight SpinnerGap + pessimistic re-fetch + icons from @/lib/icons + no emoji -- YES.
4. `npm run build` SUCCEEDS -- YES (EXIT=0), AFTER fixing a pre-existing /observability SSG-prerender blocker via force-dynamic.
5. live_check_49.3.md -- written with autonomous evidence + an explicit OPERATOR-TO-CONFIRM visual section (the render cannot be verified behind the NextAuth wall).

## Scope honesty / flags
- **Visual render is NOT autonomously verified** (NextAuth wall, frontend.md rule 5) -> delegated to the operator live_check (the designed gate). The build/types/API/conventions ARE verified.
- **Pre-existing production-build issues surfaced** (NOT caused by 49.3): (a) `/observability` SSG-prerender break -- FIXED here via force-dynamic; (b) an intermittent `/agents` "Cannot find module for page" webpack flake observed once (did not recur) -- pre-existing Next 15 SSG-worker/cache flakiness; the clean fix is `rm -rf frontend/.next` (I am permission-denied for rm -rf; operator can run it). Recommend adding a masterplan step to harden the production build (clean-cache CI + audit any other SSG-incompatible client pages).
- Risk-limits UI (49.1's endpoints) is a SEPARATE follow-on (phase-49.4), not in this step.
