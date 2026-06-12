# Contract -- phase-61.1: Activate the dark fixes + deploy phase-60 code

Date: 2026-06-12 (session start 2026-06-11 evening). Goal: goal-phase61-churn-integrity.
Install commit: 255d6cc9 (operator decision verbatim: "Install + begin 61.1 now (Recommended)").

## Research-gate summary

Brief: handoff/current/research_brief.md (198 lines; tier simple; gate_passed: true;
5 external sources read in full incl. APScheduler 3.x user guide, launchctl man page,
pydantic-settings docs, uvicorn server-behavior docs, Fowler feature-toggles; recency scan
performed; 12 internal files inspected). Headline findings:

- GO on tonight's restart: double-cycle risk is ZERO with code-level certainty --
  (1) AsyncIOScheduler at backend/main.py:267 uses the default in-memory MemoryJobStore
  (no cross-restart state); APScheduler 3.11.2 base.py:1066-1068 computes a fresh job's
  next_run_time strictly forward from now; (2) misfire_grace_time=3600/coalesce=True
  (paper_trading.py:1299-1322) is moot >5.5h past the 18:00 UTC fire; (3) exactly three
  run_daily_cycle callers, none on startup. Next fire: 2026-06-12T14:00:00-04:00.
- Watchdog safe: kickstart -k on the same label; launchd enforces single instance;
  uvicorn here is single-process (no --workers/--reload in the plist) so the
  caffeinate->uvicorn pair tears down together.
- get_settings() lru_cache (settings.py:539-541): restart is the only deterministic flag
  pickup; no stale module-level Settings snapshots found; Slack bot unaffected.
- All three flag definitions confirmed (settings.py:311 / :42 / :277) with 7 reader
  sites (portfolio_manager.py:196/:471/:561, autonomous_loop.py:805/:1948/:2228,
  data_integrity.py:17/:114).
- Precondition: verify backend/.env has no existing lines for the three env names before
  appending (researcher was permission-denied on .env; Main also denied -- operator runs
  the grep + append via `!` commands; this is consistent with the secrets deny rule).
- Frontend label com.pyfinagent.frontend confirmed; kickstart is the documented
  stale-chunk remedy.

## Hypothesis

The phase-60.2/60.3 fixes and the 57.1 binding gate are correct (each passed its own
step's Q/A) but inert: flags default OFF and the running backend process (PID 77557,
started 2026-06-11 11:43:34) predates all phase-60 commits. Appending the three env lines
per the operator's tokens and restarting will (a) load phase-60.2/60.3/60.4 code,
(b) activate the churn fix, data-integrity normalization, and binding REJECT gate, and
(c) the next daily cycle (2026-06-12 18:00 UTC) will show zero sentinel-driven swap-outs
and zero REJECT-executed buys.

## Operator tokens (recorded verbatim, AskUserQuestion 2026-06-11 local session)

- "60.2 FLAG: ON (Recommended)"  -> PAPER_SWAP_CHURN_FIX_ENABLED=true
- "60.3 FLAG: ON (Recommended)"  -> PAPER_DATA_INTEGRITY_ENABLED=true
- "57.1 FLAG: ON (Recommended)"  -> PAPER_RISK_JUDGE_REJECT_BINDING=true

## Immutable success criteria (verbatim from .claude/masterplan.json, phase-61 step 61.1)

1. "the operator's verbatim flag tokens (60.2 FLAG / 60.3 FLAG / 57.1 FLAG, each ON or
   KEEP OFF) are recorded in handoff/current/live_check_61.1.md and backend/.env matches
   them exactly; no flag changed without its token"
2. "post-restart, the running uvicorn process start time is later than the phase-60.4
   commit timestamp (ps -o lstart vs git log evidence pasted verbatim), proving
   phase-60.2/60.3/60.4 code is loaded"
3. "frontend kickstarted via launchctl; Playwright capture shows
   http://localhost:3000/login loads without ChunkLoadError"
4. "first post-restart daily-cycle evidence in live_check_61.1.md as verbatim BQ rows: if
   60.2 FLAG: ON, zero swap_for_higher_conviction SELLs of holdings lacking a same-cycle
   analysis_results row; if 57.1 FLAG: ON, zero executed trades with
   risk_judge_decision='REJECT'"
5. "handoff/harness_log.md cycle entry appended before the status flip"

verification.command (verbatim): cd /Users/ford/.openclaw/workspace/pyfinagent && source
.venv/bin/activate && python -c "from backend.config.settings import get_settings; s =
get_settings(); print('churn_fix', s.paper_swap_churn_fix_enabled, 'data_integrity',
s.paper_data_integrity_enabled, 'rj_binding', s.paper_risk_judge_reject_binding)" && test
-f handoff/current/live_check_61.1.md

live_check (verbatim): live_check_61.1.md containing: verbatim operator flag tokens,
ps -o lstart output post-restart vs commit timestamps, Playwright screenshot path for
/login, and first post-flag cycle BQ rows from financial_reports.paper_trades

## Plan

1. Operator runs (Main is .env-denied by design):
   `! grep -nE "^(PAPER_SWAP_CHURN_FIX_ENABLED|PAPER_DATA_INTEGRITY_ENABLED|PAPER_RISK_JUDGE_REJECT_BINDING)=" backend/.env`
   (expect zero hits), then the append of the three lines + provenance comment.
2. Main: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`; wait for health;
   run the verification command (fresh interpreter settings print = True True True);
   `ps -o pid,lstart,command` vs `git log -1 --format=%ci` for the phase-60.4 commit;
   curl /api/paper-trading/status asserting next_run 2026-06-12T14:00:00-04:00 (proves
   flags loaded AND no double cycle).
3. Main: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`; Playwright
   navigate /login + snapshot; assert no ChunkLoadError in console.
4. Write live_check_61.1.md (criteria 1-3 evidence; criterion 4 marked PENDING until the
   2026-06-12 18:00 UTC cycle) + experiment_results.md.
5. Spawn fresh Q/A. Expected honest verdict: CONDITIONAL pending criterion-4 cycle
   evidence -- that is the correct verdict tonight, not a failure of the step.
6. Append harness_log.md cycle entry (log-last). NO masterplan status flip until
   criterion 4 evidence lands after the 2026-06-12 cycle and a fresh Q/A passes the step.

## Out of scope for this step

Any code edit (61.2-61.5 own those); any change to paper_swap_min_delta_pct; any
hysteresis-family work; disturbing phase-58.1.

## References

- handoff/current/research_brief.md (this step's gate)
- handoff/current/goal_phase61_churn_integrity.md (goal prompt, CRITICAL constraints)
- handoff/archive/phase-60/ (60.2 replay + live_check promotion sections)
- APScheduler 3.x user guide; launchctl man page; pydantic-settings docs; uvicorn
  server-behavior docs; Fowler feature-toggles (full list in research_brief.md)
