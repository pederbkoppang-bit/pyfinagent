# Contract -- phase-49.3: Cron-control UI

**Step id:** 49.3 | **Priority:** P2 (P7 "cron enable+trigger" UI) | **depends_on:** 49.2
**Date:** 2026-05-29 | **harness_required:** true | **$0 LLM** | Frontend (Next 15 + React 19 + TS + Tailwind)

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher gate: **5 sources read in full, recency scan, 13 URLs, 7 internal files, gate_passed=true**). Decisive:
- **cron/page.tsx** (`JobsTab`): fetches `getAllJobs()`, renders rows at `:294-320` (4 cols Job/Schedule/Next-run/Status); does NOT read `controllable` yet; subtitle `:114` says "Read-only." -> add a 5th **Actions** column + update subtitle.
- **Backend contract:** `controllable` emitted ONLY on `main_apscheduler` rows (`_job_to_dict`) -> TS field **optional**. Endpoints `POST /api/jobs/{id}/pause|resume|trigger`, body `{confirmation, reason}`, tokens **exactly** `PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB`. **ASYMMETRY:** trigger is implemented ONLY for `paper_trading_daily`; `ticket_queue_process_batch` trigger -> HTTP 400 -> the Trigger button gates on `j.id==='paper_trading_daily'`, not just `controllable`.
- **api.ts:** `apiFetch<T>(path, init?)`; POST mirror = `postPaperKillSwitchAction` (`:366-379`). Add `pauseJob`/`resumeJob`/`triggerJob` with `encodeURIComponent(jobId)`.
- **types.ts:** `JobInfo` (`:1131`) += `controllable?: boolean`; add `JobControlResponse {status; job_id?; job?; detail?}`.
- **icons.ts:** `Pause`, `Play`, `Lightning` (trigger glyph -- NOT `ArrowsClockwise` which means refresh here), `SpinnerGap` (in-flight). Import from `@/lib/icons`; NO emoji; NO direct `@phosphor-icons/react`.
- **Confirm pattern:** `window.confirm()` (backend already enforces a token; actions reversible/low-blast) + per-row busy state (`SpinnerGap animate-spin`) + `await load()` after (pessimistic, server-confirmed).
- **External UX:** NN/G proportionate action-named confirmation; pessimistic server-confirmed state for a financial control; accessible icon buttons (aria-label, focus ring, color-not-alone).

## Verification split (NextAuth wall)
Authenticated-page VISUAL verification is impossible autonomously (frontend.md rule 5). So:
- **Autonomous (qa-verifiable):** `cd frontend && npm run build` (Next 15 strict TS + ESLint + compile); API-wiring grep (paths/methods/tokens/trigger-gating match backend); convention adherence (icons source, no emoji, `controllable===true` gating, aria-label).
- **Operator `live_check_49.3.md`:** the rendered Actions column (only on the 2 controllable rows), a live pause->amber->resume round-trip, a `paper_trading_daily` trigger that doesn't double-fire.

## Hypothesis
Adding a `controllable`-gated Actions column to the existing cron page (mirroring the existing kill-switch control pattern) + 3 api.ts methods + the type field, with the trigger button gated to `paper_trading_daily`, delivers the P7 cron-control UI; it is fully autonomously-verifiable at the build/API/convention level, with the visual render delegated to an operator live_check.

## Success criteria (IMMUTABLE -- verbatim from masterplan step 49.3)
1. frontend/src/lib/types.ts: JobInfo gains an optional controllable?: boolean field + a JobControlResponse type; frontend/src/lib/api.ts gains pauseJob/resumeJob/triggerJob methods POSTing to /api/jobs/{encodeURIComponent(id)}/pause|resume|trigger with the EXACT confirmation tokens PAUSE_JOB/RESUME_JOB/TRIGGER_JOB + a reason
2. frontend/src/app/cron/page.tsx renders a per-row Actions column: pause/resume controls ONLY when controllable===true (toggled by the row's running/paused status), and a trigger control ONLY when j.id==='paper_trading_daily' (the backend ticket_queue->400 asymmetry); the prior 'Read-only.' subtitle is updated
3. each action uses a confirmation step + shows per-row in-flight state (SpinnerGap) + re-fetches the job list afterward to show server-confirmed status (pessimistic); icons are imported from @/lib/icons (Pause/Play/Lightning/SpinnerGap) with NO emoji and NO direct @phosphor-icons/react import
4. cd frontend && npm run build SUCCEEDS (Next 15 strict type-check + ESLint + compile) with the changes
5. live_check_49.3.md records the autonomous evidence (build pass + API-wiring grep proofs + convention checks) AND flags the rendered Actions column + a live pause->paused->resume round-trip + a non-double-firing trigger for OPERATOR visual confirmation

**live_check:** REQUIRED -- autonomous evidence (build PASS + grep proofs) + an explicit OPERATOR-TO-CONFIRM visual section.

## Plan steps
1. **types.ts**: `JobInfo += controllable?: boolean`; add `JobControlResponse`.
2. **api.ts**: add `pauseJob(jobId, reason?)`, `resumeJob(jobId, reason?)`, `triggerJob(jobId, reason?)` after the existing job methods, mirroring `postPaperKillSwitchAction` (POST + JSON body `{confirmation, reason}`), `encodeURIComponent(jobId)`.
3. **cron/page.tsx**: add an Actions `<th>`/`<td>` (5th col); render pause/resume (toggle by status) only when `j.controllable===true`; render trigger only when `j.id==='paper_trading_daily'`; per-row busy state + `window.confirm` + `await load()`; import Pause/Play/Lightning/SpinnerGap from `@/lib/icons`; aria-label on each icon button; update the "Read-only." subtitle.
4. **Verify**: `cd frontend && npm run build`; grep proofs; write live_check_49.3.md (autonomous evidence + operator-visual section); restart the dev server only if needed (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`).
5. **EVALUATE**: fresh qa (no self-eval). Then harness_log.md (LAST), then flip masterplan 49.3 -> done.

## Safety / scope notes
- Purely additive UI on an existing page; no backend change. The controls call the already-live, already-validated 49.2 endpoints.
- pause/resume are reversible; trigger reuses the backend's triple-guard (no double-fire). Confirmation + pessimistic re-fetch prevent fat-finger + stale UI.
- Risk-limits UI is a SEPARATE follow-on (phase-49.4) -- not in this step.
- Visual render is operator-verified (auth wall); the build + API-wiring + conventions are the autonomous gate.

## References
- handoff/current/research_brief.md (gate-passing brief)
- frontend/src/app/cron/page.tsx:114 (subtitle), :161 (getAllJobs), :294-320 (rows)
- frontend/src/lib/api.ts:65 (apiFetch), :366-379 (postPaperKillSwitchAction mirror)
- frontend/src/lib/types.ts:1131 (JobInfo)
- frontend/src/lib/icons.ts (Pause/Play/Lightning/SpinnerGap)
- backend/api/cron_dashboard_api.py (controllable flag + the 3 endpoints + tokens + ticket_queue 400)
- .claude/rules/frontend.md + frontend-layout.md
- NN/G confirmation-dialog + button-states; React useTransition/useOptimistic
