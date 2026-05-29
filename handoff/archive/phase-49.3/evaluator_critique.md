# Evaluator Critique -- phase-49.3: Cron-control UI

**Q/A verdict: PASS** | Fresh single Q/A (merged qa-evaluator + harness-verifier), no self-eval.
**Date:** 2026-05-29 | **Cycle:** first Q/A for 49.3 (no verdict-shopping).
(Overwrites the prior stale phase-49.2 critique content -- archive hook had not rotated yet.)

---

## 5-item harness-compliance audit (ran FIRST)

1. **Research gate -- PASS.** `handoff/current/research_brief.md` is a 49.3 cron-UI brief; envelope `gate_passed: true`, `external_sources_read_in_full: 5` (NN/G confirmation-dialog, React useTransition, NN/G button-states, React useOptimistic, Airflow pause/resume -- all read in full via WebFetch), `recency_scan_performed: true` (2024-2026 section present), `urls_collected: 13`, `internal_files_inspected: 7`. Cited by contract.md §"Research-gate summary (PASSED)" + §References.
2. **Contract-before-generate -- PASS.** `git log`: `d025240f phase-49.3: PLAN` PRECEDES `16f3ee03 phase-49.3: GENERATE`. Contract success criteria are verbatim-identical to masterplan 49.3 `verification.success_criteria` (diffed all 5).
3. **experiment_results.md -- PASS.** Present; lists 4 changed files, verbatim verification (tsc 0 errors, build EXIT=0), live evidence (kickstart -> HTTP 302), success-criteria mapping, and a scope-honesty section.
4. **Log-last -- PASS.** NO `phase=49.3` entry in `handoff/harness_log.md`; masterplan 49.3 still `status: in_progress`. Correct ordering (log appends AFTER this PASS, before status flip).
5. **No verdict-shopping -- PASS.** First Q/A spawn for 49.3; no prior 49.3 verdict in harness_log to flip.

---

## Deterministic checks (run by Q/A, verbatim)

- **`npx tsc --noEmit`** -> `error TS` count = **0**.
- **ESLint (REQUIRED -- diff touches `frontend/**`):** `npx eslint .` -> **EXIT=0**, `52 problems (0 errors, 52 warnings)`. All 52 are `warning`-severity and pre-existing (the `eslint .` errors-only gate is satisfied). **`react-hooks/rules-of-hooks`: NOT triggered anywhere** (grepped `rules-of-hooks|called conditionally|change the order|fewer hooks` -> none). This is the canonical guard for the phase-23.2.23 hook-order bug class that previously shipped in THIS file; it is clean. The 3 cron/page.tsx findings (`:213`, `:437`, `:538`) are all `react-hooks/set-state-in-effect` **warnings** (the "Error:" text is the rule's own message label, not the severity) on PRE-EXISTING effects (the poll-`load()` effect at :213, plus two unrelated tabs at :437/:538) -- NOT introduced by 49.3's `runAction`.
- **`npm run build`** -> **BUILD_EXIT=0**, "Compiled successfully", full route table incl. `○ /cron 11.4 kB` and `○ /observability 1.79 kB`. No prerender error (grepped `a[d] is not a function | Cannot find module for page | Error occurred prerendering | Failed to compile | Type error:` -> none). The definitive run did NOT hit the intermittent `/agents` flake.
- **API-wiring grep:** `api.ts:384-411` -> `pauseJob`/`resumeJob`/`triggerJob` POST `/api/jobs/${encodeURIComponent(jobId)}/pause|resume|trigger` with tokens `PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB` + `reason`. `types.ts:1141` -> `controllable?: boolean`; `:1146` -> `JobControlResponse`.
- **Convention grep:** icons imported from `@/lib/icons` (`cron/page.tsx:16`); NO direct `@phosphor-icons/react` import; NO emoji (grep clean on cron/page.tsx + api.ts); `aria-label` on every action button (`:373` Resume, `:383` Pause, `:394` Trigger, `:366` Spinner "Working").
- **Secret/debug scan on the GENERATE diff:** no secret literals; no `console.log`/`debugger` added.

## Backend-contract verification (verified against source, not trusting the brief)

- `cron_control.py:36-38` -- `CONTROLLABLE` = exactly `{paper_trading_daily, ticket_queue_process_batch}`. Confirms the optional TS field + the `controllable===true` row gate.
- `cron_dashboard_api.py:489/501/513` -- tokens are EXACTLY `PAUSE_JOB`/`RESUME_JOB`/`TRIGGER_JOB` (HTTP 400 on mismatch). Frontend token strings match.
- `cron_dashboard_api.py:517` + `:527` -- **trigger is implemented ONLY for `paper_trading_daily`; every other id -> HTTP 400** "trigger not supported ... (pause/resume only)". This is precisely why the Trigger button is gated on `j.id === "paper_trading_daily"` (`cron/page.tsx:390`), not merely `controllable`. Gating it only on `controllable` would surface a Trigger button on `ticket_queue_process_batch` that always 400s -- the wiring AVOIDS that bug. Correct.
- `cron_dashboard_api.py:200` -- `controllable` emitted via `is_controllable()` on `main_apscheduler` rows only -> optional `controllable?` field is the honest type.

## Code-review heuristics (5 dimensions evaluated)

Diff is `frontend/**` only (4 files) + handoff docs. No backend execution path, no risk-guard, no `perf_metrics`, no `kill_switch`, no `paper_trader`, no Sharpe/drawdown math touched -> Dimension-2 (trading-domain) and Dimension-4 (financial-logic-without-behavioral-test) BLOCK heuristics are N/A for this diff. Security (D1): no secret-in-diff, no command/prompt injection, no dep-pin removal (purely additive). Code-quality (D3): no `console.log`, no broad-except, icon buttons typed, aria-labels present. LLM-evaluator (D5): first cycle, no prior verdict to flip -> sycophancy/second-opinion-shopping N/A. **No findings.**

## Success-criteria evaluation (1-5 vs masterplan 49.3)

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | types.ts `controllable?` + `JobControlResponse`; api.ts pause/resume/triggerJob POST w/ EXACT tokens + reason + `encodeURIComponent` | **PASS** | types.ts:1141,1146; api.ts:384-411 (method bodies verified -- tokens + `encodeURIComponent(jobId)` + reason all present) |
| 2 | Actions column: pause/resume ONLY when `controllable===true` (toggled by status), trigger ONLY when `j.id==='paper_trading_daily'`; subtitle updated | **PASS** | cron/page.tsx:364 (`j.controllable ?`), :369 (status toggle `paused`->Resume / else Pause), :390 (`j.id === "paper_trading_daily"` gate), :404 (`--` for non-controllable); subtitle now "...can be paused, resumed, or triggered." -- NO "Read-only" string remains |
| 3 | confirmation + per-row SpinnerGap in-flight + pessimistic re-fetch; icons from @/lib/icons (Pause/Play/Lightning/SpinnerGap); NO emoji; NO direct @phosphor-icons | **PASS** | :194 `window.confirm`; :365-366 per-row `SpinnerGap animate-spin` gated on `busyJob===j.id`; :201 `await load()` after action (pessimistic); icons.ts exports all 4 confirmed; emoji grep clean; no direct phosphor import |
| 4 | `npm run build` SUCCEEDS (Next 15 strict TS + ESLint + compile) | **PASS** | BUILD_EXIT=0 + "Compiled successfully" + tsc 0 errors + eslint EXIT=0. Required the force-dynamic `/observability` fix (in-scope -- see below) |
| 5 | live_check_49.3.md records autonomous evidence + flags rendered column + live round-trip + non-double-firing trigger for OPERATOR | **PASS** | live_check_49.3.md present: autonomous evidence section (build/tsc/API/conventions with file:line) + explicit "OPERATOR TO CONFIRM" section (5 items incl. Actions column on the 2 rows, pause->amber->resume round-trip, trigger no-double-fire) |

## Scope-honesty assessment (orchestrator's explicit questions)

- **Visual-render-pending-operator is BY DESIGN, not a gap.** The verification split is encoded in masterplan 49.3's `live_check` field, the contract §"Verification split (NextAuth wall)", and frontend.md rule 5 (authenticated-page visual verification is impossible autonomously). Criterion #5 *requires* exactly this shape (autonomous evidence + an OPERATOR-TO-CONFIRM section). live_check_49.3.md delivers a cleanly-delimited operator section. Shipping build/types/API/conventions as autonomously-verified while delegating pixels to an auditable operator artifact is the documented `verification.live_check` discipline (CLAUDE.md) -- it converts "agent claimed PASS" into an operator-auditable artifact. This is correct, not overclaiming. Q/A does NOT fail the step for "visual unverified".
- **force-dynamic observability fix is IN-SCOPE.** Criterion #4 requires `npm run build` to SUCCEED. The pre-existing `/observability` SSG-prerender break ("a[d] is not a function") blocked the build, so the fix was a precondition for #4. It is a minimal 6-line additive `export const dynamic = "force-dynamic"` (verified present at observability/page.tsx:13; the comment explains the latent next-dev-only nature). Correctly disclosed in experiment_results §"Files changed" item 4 + §"Scope honesty", and in live_check. Reasonable and correctly attributed.
- **Intermittent `/agents` flake disclosed honestly** as pre-existing Next 15 SSG-worker flakiness that did not recur, with a `rm -rf .next` mitigation recommendation and a follow-on step suggestion. Not an overclaim; my definitive build run did not hit it.

## Notes

- Visual render PENDING OPERATOR is EXPECTED (the live_check gate), NOT a violation.
- A separate masterplan step to harden the production build (clean-cache CI + audit other SSG-incompatible client pages) is a reasonable follow-on per experiment_results -- out of scope for 49.3.

---

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 5 immutable criteria met. Deterministic: tsc=0 errors, eslint EXIT=0 (0 errors, react-hooks/rules-of-hooks clean -- the prior hook-order bug class in this file), npm run build EXIT=0 'Compiled successfully'. API wiring verified against backend source (cron_control.py:36-38 CONTROLLABLE set; cron_dashboard_api.py:489/501/513 exact tokens; :517/:527 trigger-only-paper_trading_daily asymmetry -> Trigger button correctly gated on j.id, not just controllable). Conventions clean (icons from @/lib/icons, no emoji, no direct phosphor, aria-labels). Visual-render-pending-operator is the designed live_check split (frontend.md rule 5), not a gap. force-dynamic observability fix is in-scope (precondition for criterion #4) and correctly disclosed.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax_tsc", "eslint_frontend", "npm_build", "verification_command", "api_wiring_grep", "backend_contract_source_verify", "convention_grep", "code_review_heuristics", "research_brief", "experiment_results", "live_check"],
  "harness_compliance": {
    "research_gate": "PASS -- research_brief.md gate_passed:true, 5 sources read in full, recency scan, 13 URLs, 7 internal files, cited by contract",
    "contract_before_generate": "PASS -- PLAN (d025240f) precedes GENERATE (16f3ee03); criteria verbatim from masterplan 49.3",
    "experiment_results_present": "PASS -- files + verification output + live evidence + scope honesty",
    "log_last": "PASS -- no phase=49.3 entry in harness_log; masterplan 49.3 still in_progress",
    "no_verdict_shopping": "PASS -- first Q/A for 49.3"
  }
}
```
