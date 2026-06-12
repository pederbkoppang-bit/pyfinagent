# live_check -- phase-61.1: Activate the dark fixes + deploy phase-60 code

Status: PARTIAL (sections A-D COMPLETE; E pending the 2026-06-12 18:00 UTC cycle).
Remaining blocker: criterion-4 evidence requires the first post-flag daily cycle.

## A. Operator flag tokens (criterion 1 -- recorded verbatim)

Source: AskUserQuestion, local session 2026-06-11 (~23:30 CEST), goal-phase61-churn-integrity.

- "60.2 FLAG: ON (Recommended)"  -> PAPER_SWAP_CHURN_FIX_ENABLED=true
- "60.3 FLAG: ON (Recommended)"  -> PAPER_DATA_INTEGRITY_ENABLED=true
- "57.1 FLAG: ON (Recommended)"  -> PAPER_RISK_JUDGE_REJECT_BINDING=true

Install decision (phase-61 itself), same session, verbatim: "Install + begin 61.1 now
(Recommended)" -- install commit 255d6cc9.

.env append: DONE by operator keystroke 2026-06-12 ~08:04 local (`!`-prefixed
in-session). Precondition grep returned zero hits (no pre-existing flag lines);
printf appended the provenance comment + three KEY=true lines. Note: the terminal
wrapped the comment string mid-paste, leaving a harmless non-KEY line in .env
(dotenv ignores lines without '='); the three flag lines load correctly (section C).

## B. Frontend kickstart + Playwright capture (criterion 3 -- COMPLETE)

Before (2026-06-11 ~21:00 CEST probe): /login threw ChunkLoadError, stale bundle --
console log .playwright-mcp/console-2026-06-11T18-58-57-517Z.log (404 on
/_next/static/chunks/app/login/page.js, favicon 404, ChunkLoadError stack).

Action: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend` -> "kickstarted";
`curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/login` -> 200.

After (2026-06-12 04:12 UTC, Playwright MCP):
- Page URL http://localhost:3000/login, title "PyFinAgent -- AI Financial Analyst",
  fully styled login card (Google + Passkey buttons).
- Console: ZERO errors (only the React DevTools info line) --
  .playwright-mcp/console-2026-06-12T04-12-17-492Z.log
- Screenshot: .playwright-mcp/page-2026-06-12T04-12-23-333Z.png
- Snapshot: .playwright-mcp/page-2026-06-12T04-12-17-914Z.yml

## C. Backend flag state

Baseline (pre-append, pre-restart, fresh interpreter, 2026-06-12 ~04:15 UTC):

    PRE-RESTART FLAGS: churn_fix False | data_integrity False | rj_binding False

Post-append (fresh interpreter, 2026-06-12 ~06:04 UTC, verbatim):

    FRESH-INTERPRETER FLAGS: churn_fix True | data_integrity True | rj_binding True

The running process (restarted 08:05:49 local, AFTER the append) therefore booted with
all three flags ON.

## D. Restart evidence (criterion 2 -- COMPLETE)

Pre-restart process state (ps -axo pid,lstart,command, verbatim):

    77557 tor. 11 jun. 11.43.34 2026   .../Python .../.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
    77559 tor. 11 jun. 11.43.34 2026   /usr/bin/caffeinate -i -s .../.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

Phase-60.4 commit (git log, verbatim):

    b0fe1983 2026-06-11 16:30:22 +0200 phase-60.4: observability + ops residuals (AW-7/AW-1/AW-2/AW-10, hygiene) -- PASS, CLOSES PHASE-60

=> pre-restart process (11:43:34) PREDATED the phase-60 commits; phase-60.2/60.3/60.4
code was NOT loaded. Restart safety: GO per research_brief.md (MemoryJobStore,
forward-only CronTrigger next_run_time, no run-on-startup caller; misfire window moot).

RESTART EXECUTED 2026-06-12: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`
-> "kickstarted"; backend health http 200. Post-restart process state (verbatim):

    84680 fre. 12 jun. 08.05.49 2026   .../Python .../.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
    84682 fre. 12 jun. 08.05.49 2026   /usr/bin/caffeinate -i -s .../.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000

new lstart 2026-06-12 08:05:49 (+0200) > phase-60.4 commit 2026-06-11 16:30:22 +0200
=> CRITERION 2 SATISFIED (phase-60.2/60.3/60.4 code loaded). Old PID pair 77557/77559
gone (no zombies; single uvicorn+caffeinate pair as expected).

Scheduler check post-restart (curl /api/paper-trading/status, verbatim excerpt):

    "scheduler_active": true,
    "next_run": "2026-06-12T14:00:00-04:00",
    "loop": {"running": false, "last_run": null, "last_result": null}

=> job registered for 18:00 UTC today, did NOT re-fire on startup -- the research-brief
no-double-cycle prediction held live. NAV at restart: 23896.04 (+19.48%), 2 positions.

## E. First post-flag cycle evidence (criterion 4 -- PENDING)

Due after the 2026-06-12 18:00 UTC cycle. Required verbatim BQ rows from
financial_reports.paper_trades (+ analysis_results cross-check):
- zero SELL rows with reason='swap_for_higher_conviction' whose ticker lacks a
  same-cycle analysis_results row (60.2 ON), and
- zero executed trades with risk_judge_decision='REJECT' (57.1 ON).
