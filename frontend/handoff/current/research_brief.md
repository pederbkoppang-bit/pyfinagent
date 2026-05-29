# Research Brief — phase-49.3: cron-control UI

**Tier:** moderate
**Step:** Add pause/resume/trigger buttons to the cron dashboard page, wired to
phase-49.2 endpoints (`POST /api/jobs/{id}/pause|resume|trigger`).
**Stack:** Next.js 15 + React 19 + TS 5.6 + Tailwind.
**Constraint that shapes the whole design:** Authenticated-page VISUAL verification
is impossible autonomously (NextAuth wall, frontend.md rule 5). Autonomous
verification = `npm run build` (type-check + compile + lint) + API-wiring
correctness (paths/methods/body match backend) + convention adherence. The visual
render is delegated to operator `live_check_49.3.md`.

---

## Internal code inventory (Q1-Q6, file:line anchors)

### Q1 — `frontend/src/app/cron/page.tsx` structure

| Aspect | Finding (file:line) |
|--------|---------------------|
| Page shell | Correct two-zone shell at `cron/page.tsx:105-145` (`flex h-screen overflow-hidden` + Sidebar + fixed header + scrollable `flex-1 overflow-y-auto scrollbar-thin`). Compliant with frontend-layout.md §1. |
| Tabs | `TabId = "jobs" \| "logs"` at `:31`; pill tabs at `:119-135` (canonical pattern). |
| Header subtitle | `:114` reads "...Read-only." — **must be updated** when controls are added (it is no longer read-only for the 2 controllable jobs). |
| Job fetch | `JobsTab` at `:150`. `getAllJobs()` at `:161`, 5s poll at `:182-188`, `MAX_CONSECUTIVE_FAILURES=5` at `:55` (matches polling-failure-limit rule). Loading/error/empty states all present (`:202-246`). |
| Job grouping | `useMemo` groups by `j.source` at `:194-200` (Rules-of-Hooks-safe — called every render). |
| Job ROW shape | `:294-320`. 4 columns: **Job** (id + description), **Schedule**, **Next run**, **Status pill**. `key={`${j.source}-${j.id}`}` at `:295`. Status pill via `statusClasses()` at `:77-88` (scheduled=emerald, paused=amber, manifest=slate). |
| `controllable` read? | **NO.** The page does NOT yet read `controllable`. The backend adds it (see Q1-backend) but the frontend `JobInfo` type and the row JSX ignore it. **This is the gap phase-49.3 fills.** |
| Where buttons go | New 5th column `<th>Actions</th>` after Status (`:291`), with a matching `<td>` after `:318`. Render buttons ONLY when `j.controllable === true`; otherwise render an em-dash / empty cell so the table stays aligned. |

**Backend confirmation that `controllable` exists on the row** (`backend/api/cron_dashboard_api.py:199-201`): `_job_to_dict` sets `"controllable": cron_control.is_controllable(job.id)`. Only `main_apscheduler` rows get this key — `_static_to_dict` (slack_bot/launchd manifest rows, `:204-213`) and the slack_bot/launchd merge blocks (`:431-467`) do **NOT** emit `controllable`, so it is `undefined` there → the TS type must make it **optional** (`controllable?: boolean`) and the row must treat `undefined`/`false` identically (no buttons).

### Q2 — `frontend/src/lib/api.ts` POST-control pattern

| Aspect | Finding (file:line) |
|--------|---------------------|
| `apiFetch` signature | `apiFetch<T>(path, init?: RequestInit): Promise<T>` at `api.ts:65`. Sets `Content-Type: application/json`, Bearer token, `credentials: "include"`, 30s AbortController timeout (`:78-80`), 401→`/login` (`:112-116`), structured error messages for 422/500/404 (`:125-134`). |
| POST-with-body idiom | `postPaperKillSwitchAction` at `:366-379` — the canonical mirror. Shape: `apiFetch(path, { method: "POST", body: JSON.stringify({ confirmation: action }) })`. |
| POST-no-body idiom | `triggerPaperTradingCycle` at `:299-301` — `apiFetch("/api/paper-trading/run-now", { method: "POST" })`. |
| Existing jobs methods | `getAllJobs()` at `:735-737` → `apiFetch("/api/jobs/all")`. `getLogTail` at `:739-745`. |
| Where to add | After `getLogTail` (`:745`), in the same phase-23.2.23 cron block. |

**Exact backend contract for the 3 new endpoints** (`cron_dashboard_api.py:481-527`):
- Request model `CronControlRequest` (`:481-484`): `{ confirmation: str, reason: str = "manual" (max_length 200) }`.
- `POST /api/jobs/{job_id}/pause` (`:487`): requires `confirmation == "PAUSE_JOB"` else HTTP 400; 404 if job not controllable. Returns `{status:"paused", job: <state>}`.
- `POST /api/jobs/{job_id}/resume` (`:499`): requires `confirmation == "RESUME_JOB"`; returns `{status:"resumed", job: <state>}`.
- `POST /api/jobs/{job_id}/trigger` (`:511`): requires `confirmation == "TRIGGER_JOB"`; 404 if not controllable; returns `{status:"triggered", job_id, detail}`. **ASYMMETRY (critical):** trigger is implemented **only for `paper_trading_daily`**; for `ticket_queue_process_batch` it returns **HTTP 400** "trigger not supported ... (pause/resume only)" (`:525-527`).
- Controllable job IDs (`backend/services/cron_control.py:36-38`): exactly **`paper_trading_daily`** and **`ticket_queue_process_batch`** (both `source=main_apscheduler`).

So the 3 methods should be:
```ts
export function pauseJob(jobId: string, reason = "manual"): Promise<JobControlResponse> {
  return apiFetch(`/api/jobs/${encodeURIComponent(jobId)}/pause`, {
    method: "POST",
    body: JSON.stringify({ confirmation: "PAUSE_JOB", reason }),
  });
}
// resumeJob -> /resume, confirmation "RESUME_JOB"
// triggerJob -> /trigger, confirmation "TRIGGER_JOB"
```
`encodeURIComponent(jobId)` matters — IDs contain no slashes today but the path-param interpolation should be defensive (mirrors `getPaperTradeRationale` at `:393-395`).

### Q3 — `frontend/src/lib/types.ts`

| Aspect | Finding (file:line) |
|--------|---------------------|
| `JobInfo` | `:1131-1139`. Fields: id, source, schedule, next_run, last_run, status, description. **Add `controllable?: boolean;`** (optional — manifest/slack/launchd rows omit it). |
| `JobSource` | `:1129` — `"main_apscheduler" \| "slack_bot" \| "launchd"`. No change. |
| `AllJobsResponse` | `:1141-1145`. No change (jobs/generated_at/n_total). |
| Control response type | **None exists — add one.** e.g. `export interface JobControlResponse { status: string; job_id?: string; job?: unknown; detail?: unknown; }`. The backend returns slightly different shapes per action (`job` for pause/resume, `job_id`+`detail` for trigger); a permissive interface with optional fields is the honest type. Mirrors how `getPaperKillSwitchState` returns `Promise<unknown>` (`:362`) — the app already tolerates loosely-typed control responses. A typed-but-permissive interface is an improvement over `unknown`. |

### Q4 — `frontend/src/lib/icons.ts` (MUST import from here, never `@phosphor-icons/react`)

| Need | Export (file:line) | Use |
|------|--------------------|-----|
| Pause | `Pause` at `icons.ts:225` | Pause button |
| Resume / Play | `Play` at `:226` | Resume button |
| Trigger / run-now | `Lightning` at `:220` (also `ArrowClockwise`/`ArrowsClockwise` at `:179-180`) | Trigger button — **`Lightning`** is the closest semantic match (fire-now) and is already the project's "run/trigger" glyph; `ArrowsClockwise` already means "refresh" on this page (`:7`, `:261`), so do NOT reuse it for trigger or the two actions collide visually. |
| In-flight spinner | `SpinnerGap` at `:234` (alias `IconSpinner` at `:145`) | Per-button loading indicator (`animate-spin`). |
| Warning (errors) | `Warning` already imported on the page (`:8`). | Inline action error. |

All four (`Pause`, `Play`, `Lightning`, `SpinnerGap`) are confirmed exports. NO emojis anywhere (strict project rule + auto-memory `feedback_no_emojis`).

### Q5 — Conventions (`.claude/rules/frontend.md` + `frontend-layout.md`)

- **Shell / dark theme**: already correct on the page. New buttons must use the navy+slate palette (NOT zinc): borders `border-navy-700`, hover `hover:bg-navy-700/40` or `hover:bg-navy-800/60`, text `text-slate-200/300/400`. Action accents: resume/positive `sky-600`/`emerald`, pause/caution `amber`, destructive `rose-600` (frontend.md dark-mode rules 1+6; color-coding green/amber/red).
- **Confirmation pattern (TWO precedents exist):**
  1. **`window.confirm()`** — used by `OpsStatusBar.tsx:96-103` (pause/resume/flatten), `KillSwitchShortcut.tsx:12`, `HomeQuickActionsPanel.tsx:81`, and `backtest/page.tsx:196,593`. Lightweight, one-liner, restates the action.
  2. **In-app `ConfirmModal`** — `KillSwitchPanel.tsx:212-268`: a `role="dialog" aria-modal="true"` overlay with Cancel (safe) + a colored Confirm button, "recorded in audit log" note, busy-disabled buttons.
  **Recommendation: `window.confirm()`** for phase-49.3. Rationale: (a) it is the pattern used by the most-similar action sites (OpsStatusBar pause/resume), (b) the backend ALREADY enforces a confirmation TOKEN (`PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB`) server-side, so `window.confirm` is the second layer of a defense-in-depth design, not the only guard, and (c) these actions are reversible (pause↔resume) and low-blast-radius (2 jobs), so the heavier modal (reserved for FLATTEN_ALL, which liquidates positions) is overkill per NN/G "don't cry wolf" (see external findings). Use action-specific wording, not Yes/No.
- **Loading/error states**: per-row in-flight state (disable the row's buttons + show `SpinnerGap animate-spin` on the clicked one). On error, surface the message — the page has no per-row error slot today, so either `window.alert(msg)` (matches `OpsStatusBar.tsx:109`) or a small inline rose text under the row. After a successful action, call `load()` to refresh the job list so the status pill reflects the server-confirmed state (pessimistic refresh — see external findings).
- **scrollbar-thin**: already on the scroll zone (`:139`). No new scroll containers.
- **Dev-server restart after changes (auto-memory `feedback_npm_install_requires_launchctl_kickstart`):** for a code-only change (no `npm install`), the launchd-managed dev server hot-reloads. If a restart is needed, use `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` — **NOT `pkill`** (pkill races the launchd watchdog and serves stale 404 CSS bundles). `npm install` is NOT needed here (no new deps; all icons already exported).

### Q6 — Build / verify path

- Command: `cd frontend && npm run build` (per CLAUDE.md "Frontend build check" + frontend Quick Start). `package.json` script `build` → `next build`.
- Next.js 15 `next build` runs **TypeScript type-checking AND ESLint** as part of the production build (unless `typescript.ignoreBuildErrors`/`eslint.ignoreDuringBuilds` is set in `next.config` — confirm it is not). A type error in the new api.ts methods or the `JobInfo` change WILL fail the build. This is the primary autonomous gate.
- TS is `strict` (frontend.md "TypeScript 5.6 strict"). The permissive control-response interface must still satisfy strict null/优先 checks.

---

## External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://www.nngroup.com/articles/confirmation-dialog/ | 2026-05-29 | doc (UX authority) | WebFetch full | Confirmation warranted for "serious consequences"; use action-named buttons not Yes/No; **no default Yes** / best no default; "don't cry wolf" — overuse → ignored. |
| https://react.dev/reference/react/useTransition | 2026-05-29 | doc (official) | WebFetch full | `isPending` true from first `startTransition` until action completes; `disabled={isPending}`; React 19 accepts async fns in `startTransition`; wrap post-`await` state in nested `startTransition`; errors caught via error boundary. |
| https://www.nngroup.com/articles/button-states-communicate-interaction/ | 2026-05-29 | doc (UX authority) | WebFetch full | Loading state = spinner left of label, may animate to check on success; disabled = action unavailable; **color alone insufficient — use stroke/outline + secondary cue**; focus ring within 100-150ms for keyboard. |
| https://react.dev/reference/react/useOptimistic | 2026-05-29 | doc (official) | WebFetch full | Optimistic shows a TEMPORARY guessed value; on failure auto-reverts to prior `value`. Canonical uses = likes/follows/"Submitting…". **NOT for state the user must see server-confirmed** → use `useTransition` + render server response. |
| https://www.sparkcodehub.com/airflow/scheduling/pause-resume | 2026-05-29 | doc (peer practitioner) | WebFetch full | Airflow pauses a DAG via a **toggle**; "No new runs are scheduled; existing runs finish"; **no confirmation step** in their flow; paused shown as toggle "Off". Peer precedent for "pause = stop NEW work, let in-flight finish" (matches our backend semantics). |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://simonhearne.com/2021/optimistic-ui-patterns/ | blog | Fetched, but article only covers WHERE optimistic works (carts/likes), not the financial-exclusion nuance — superseded by the React useOptimistic doc read in full. |
| https://adamsilver.io/blog/the-problem-with-disabled-buttons-and-what-to-do-instead/ | blog | Fetched; argues against PERMANENTLY disabled (form-validation) buttons. Not directly about in-flight disabling; the NN/G button-states full read covers the in-flight case better. |
| https://www.uxtigers.com/post/inactive-buttons | blog (Nielsen) | Snippet: show/disable/hide trade-off; 60% prefer always-clickable + feedback. Reinforces NN/G. |
| https://www.patternfly.org/components/button/accessibility/ | doc (design system) | Snippet: ARIA for icon buttons. Covered by WCAG findings. |
| https://github.com/apache/airflow/discussions/41247 | community | Snippet: "pause all DAGs" UI discussion; peer evidence operators want scheduler pause controls. |
| https://www.uxpin.com/studio/blog/button-states/ | blog | Snippet: 2026 button-states guide; redundant with NN/G full read. |
| https://medium.com/@fikrim69/optimistic-ui-pessimistic-ui-... | blog | Snippet: "pessimistic prioritizes consistency over instant feedback" — supports the pessimistic choice for financial state. |
| https://www.nngroup.com/articles/proximity-consequential-options/ | doc (UX authority) | Snippet: keep destructive far from benign — informs button spacing/ordering. |

**URLs collected (unique):** 13+ (5 read-in-full + 8 snippet-only above; additional search hits not listed).

### Search-query variants run (3-variant discipline)

1. **Current-year frontier (2026/2025):** "React 19 useTransition button loading state POST request 2026"; "admin panel control button design patterns 2025"; "scheduler job pause resume run-now button UI 2026 Airflow Dagster Prefect".
2. **Last-2-year window:** covered by the 2025/2026 hits above (Airflow-2026 comparisons, button-states-2026 guides, React-19-Jan-2026 patch note).
3. **Year-less canonical:** "confirmation dialog for destructive admin actions UX best practices" (→ NN/G confirmation-dialog, the founding reference); "optimistic vs pessimistic UI updates"; "button states design loading disabled hover focus active accessibility" (→ NN/G button-states). The year-less queries surfaced the two canonical NN/G articles that anchor the recommendation.

---

## Recency scan (2024-2026)

Searched the last-2-year window explicitly. **Findings that complement (not supersede) the canonical sources:**
1. **React 19 (Dec 2024) `useTransition` async support** is the modern idiom for in-flight button state — but a **Jan 17 2026 React patch** fixed a Fiber-reconciler race where a fast-resolving action left the UI stuck in pending (vercel/next.js Discussion #88767). Implication: ensure the project's React/Next is current; for a plain client-fetch (no Server Action) the race is not triggered, which is another reason to prefer the existing client-fetch pattern over Server Actions here.
2. **`useOptimistic`** (React 19) exists but the official doc (read in full, 2026) confirms it is for non-critical UI; **no new finding overturns** the "pessimistic for financial state" guidance.
3. **WCAG 2.2 (Oct 2023, now the 2025-2026 compliance baseline):** icon-only action buttons need an `aria-label`; focus rings mandatory; color-not-alone. No change to the canonical button-state guidance, just a firmer compliance floor.
**No finding in the window contradicts the design below.**

---

## Recommended design (autonomously-verifiable vs operator-eyes)

### What to build
1. **`types.ts`**: add `controllable?: boolean;` to `JobInfo` (`:1131`); add `export interface JobControlResponse { status: string; job_id?: string; job?: unknown; detail?: unknown; }`.
2. **`api.ts`** (after `:745`): add `pauseJob`, `resumeJob`, `triggerJob` exactly as in Q2 — POST with `{confirmation: "<VERB>", reason}`, `encodeURIComponent(jobId)`, return `Promise<JobControlResponse>`.
3. **`cron/page.tsx`**:
   - Add 5th column `Actions` (header `:291`, cell after `:318`). Render buttons **only when `j.controllable === true`**; else render a dim `--`.
   - Buttons per controllable row: **Pause** (when `status==="scheduled"`), **Resume** (when `status==="paused"`), and **Trigger** (`Lightning`) **only when `j.id === "paper_trading_daily"`** — because the backend returns HTTP 400 for trigger on `ticket_queue_process_batch` (`cron_dashboard_api.py:525-527`). Showing a trigger button that always errors would be a wiring bug; gate it on the id.
   - Per-row in-flight state: a `busyJobId`/`busyAction` state (or a `Record<string, action>`); disable the row's buttons + swap the clicked button's icon to `SpinnerGap animate-spin` while the POST is in flight (pessimistic: NN/G + React useTransition).
   - On click: `window.confirm("<action-specific message>")` first (e.g. "Pause job 'paper_trading_daily'? Scheduled runs stop until resumed; an in-flight run finishes."), then call the api method, then `await load()` to refresh from the server (pessimistic — show server-confirmed status, never an optimistic guess; React `useOptimistic` doc).
   - On error: `window.alert(msg)` (matches `OpsStatusBar.tsx:109`) or inline rose text. Either is convention-compliant.
   - Icon buttons MUST carry `aria-label` (e.g. `aria-label="Pause paper_trading_daily"`) and a visible focus ring (WCAG 2.2; NN/G button-states). Color is not the only cue — each button also has its glyph.
   - Update the page subtitle (`:114`) so it no longer claims "Read-only" (or scope the claim to logs).
   - Implementation choice: `useTransition` is available, but the page already uses an explicit `refreshing`/`loading` boolean idiom (`:154`, `:336`) and plain `async` handlers. Mirror that (a `busy` state + plain `async` handler) for consistency rather than introducing `useTransition` mid-file. Either is correct; consistency wins. Note: if `useTransition` is used, wrap any post-`await` `setState` in a nested `startTransition` (React 19 caveat from the full read).

### Autonomously verifiable (no operator)
- `cd frontend && npm run build` passes → TS strict type-check (the `controllable?` field, the 3 method signatures, the response interface all compile) + ESLint + production compile.
- **API-wiring correctness** (auditable by reading code against `cron_dashboard_api.py`): paths `/api/jobs/{id}/pause|resume|trigger`, method POST, body `{confirmation, reason}`, confirmation tokens exactly `PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB`, trigger gated to `paper_trading_daily`.
- **Convention adherence** (greppable): icons imported from `@/lib/icons` (not `@phosphor-icons/react`); no emoji; navy/slate palette; `aria-label` on icon buttons; buttons gated on `controllable===true`.
- Optional: a unit-level assertion that `pauseJob`/`resumeJob`/`triggerJob` build the right path+body (if the project adds a tiny test — not required by the step).

### Needs operator eyes (`handoff/current/live_check_49.3.md`)
- The actual rendered Actions column on the authenticated `/cron` page (dark-theme contrast, button spacing/alignment, that buttons appear ONLY on the 2 controllable `main_apscheduler` rows and NOT on slack/launchd rows).
- A real pause→status-pill-flips-to-amber→resume round-trip against the running backend, and a trigger on `paper_trading_daily` (confirm the 409-guard path doesn't double-fire). Live-system evidence (curl or screenshot) is what the `live_check` gate (CLAUDE.md `verification.live_check`) converts from "agent claimed PASS" into an auditable artifact.

### 2-3 external-source-backed UX practices applied
1. **Confirm only proportionate friction; name the action in the button/prompt, no Yes/No, no default-Yes** (NN/G confirmation-dialog). → `window.confirm` with action-specific wording; reserve the heavy modal for FLATTEN_ALL.
2. **Pessimistic, server-confirmed state for a financial control — not optimistic** (React `useOptimistic` doc + NN/G button-states loading-state). → disable + spinner during the POST, then `load()` to show the backend's actual status.
3. **Accessible icon buttons: `aria-label`, focus ring, color-not-alone** (NN/G button-states + WCAG 2.2). → every icon button labeled, focusable, with a glyph in addition to color.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (NN/G confirmation, React useTransition, NN/G button-states, React useOptimistic, Airflow pause/resume)
- [x] 10+ unique URLs total (13+ incl. snippet-only)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (page, api, types, icons, backend endpoint, cron_control, OpsStatusBar/KillSwitchPanel confirm patterns)
- [x] Contradictions / consensus noted (two confirm patterns; optimistic-vs-pessimistic resolved to pessimistic for financial state)
- [x] All claims cited per-claim with file:line or URL

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 8,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
