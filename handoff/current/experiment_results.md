# Experiment Results -- Step 61.1 (GENERATE, in progress)

**Step:** 61.1 -- Activate the dark fixes + deploy phase-60 code. **Date:** 2026-06-12.
**State:** Criteria 1-3 COMPLETE; criterion 4 gated on the 2026-06-12 18:00 UTC cycle;
criterion 5 (log-last) queued for the close.

## What was done (chronological)

1. Phase-61 installed: payload appended to .claude/masterplan.json (5 steps, all
   pending), goal file renamed to handoff/current/goal_phase61_churn_integrity.md,
   active_goal.md refreshed, commit 255d6cc9 pushed. Operator install decision verbatim:
   "Install + begin 61.1 now (Recommended)".
2. Research gate: first researcher spawn died at the session usage limit having written
   only the brief skeleton (handoff/current/research_brief.md, write-first held).
   Fresh respawn completed the brief after the 01:20 reset -- gate_passed: true,
   5 sources read in full, recency scan, GO verdict on the restart (double-cycle risk
   zero: MemoryJobStore + forward-only CronTrigger + no run-on-startup caller;
   watchdog/kickstart -k safe; single-process uvicorn).
3. Contract written: handoff/current/contract.md (criteria copied verbatim from
   masterplan; cycle-2-aware plan).
4. Operator tokens collected (AskUserQuestion, verbatim): "60.2 FLAG: ON (Recommended)",
   "60.3 FLAG: ON (Recommended)", "57.1 FLAG: ON (Recommended)".
5. Frontend (criterion 3, COMPLETE): launchctl kickstart -k gui/$(id -u)/
   com.pyfinagent.frontend; /login HTTP 200; Playwright capture clean (zero console
   errors, ChunkLoadError gone). Evidence in live_check_61.1.md section B.
6. Baselines captured (live_check sections C/D): flags False/False/False on fresh
   interpreter; running uvicorn PID 77557 lstart 2026-06-11 11:43:34 vs phase-60.4
   commit b0fe1983 at 16:30:22 +0200 -- phase-60 code confirmed NOT loaded.

## File list (this step so far)

- .claude/masterplan.json (phase-61 appended, 61.1 still pending -- no flips)
- handoff/current/goal_phase61_churn_integrity.md (renamed from _DRAFT)
- handoff/current/active_goal.md (refresh payload)
- handoff/current/research_brief.md (61.1 gate)
- handoff/current/contract.md (61.1)
- handoff/current/live_check_61.1.md (sections A-D partially filled, E pending)
- handoff/current/experiment_results.md (this file)
- NO source-code changes (per contract scope: 61.1 is config/ops only)

7. OPERATOR .env append executed 2026-06-12 ~08:04 local (grep precondition: zero hits;
   printf appended comment + three KEY=true lines; harmless mid-paste line-wrap in the
   comment, dotenv ignores non-KEY lines). Fresh-interpreter flags: True True True.
8. Backend restart executed: kickstart -k -> new PID pair 84680/84682, lstart
   2026-06-12 08:05:49 > phase-60.4 commit 16:30:22 +0200 (criterion 2 satisfied);
   health 200; /api/paper-trading/status: scheduler_active true, next_run
   2026-06-12T14:00:00-04:00, loop idle -- no startup re-fire, matching the research
   prediction. Old PIDs 77557/77559 gone (no zombies). Evidence verbatim in
   live_check_61.1.md sections C/D.

## Verification command output (verbatim, post-restart)

    PRE-RESTART FLAGS:  churn_fix False | data_integrity False | rj_binding False
    POST-APPEND FLAGS:  churn_fix True  | data_integrity True  | rj_binding True

## Remaining to close the step

1. After the 2026-06-12 18:00 UTC cycle: pull BQ rows for criterion 4 into live_check
   section E (zero sentinel swap-outs; zero REJECT-executed trades).
2. Spawn fresh Q/A on the updated evidence; append harness_log.md (log-last); only then
   flip 61.1 to done.
