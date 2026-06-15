# live_check -- phase-61.1: Activate the dark fixes + deploy phase-60 code

Status: COMPLETE (sections A-E). Criterion 4 closed 2026-06-15 (AM away session) on the
first post-flag cycle `5f15fdbe` (2026-06-12 18:00 UTC). PASS-with-caveat: the negative
assertions are satisfied AND vacuous (n_trades=0); positive guardrail-fires evidence is the
neutralized-env activation-test witness (28 passed). See section E.

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

## E. First post-flag cycle evidence (criterion 4 -- COMPLETE, 2026-06-15 AM)

The three fixes went live-ON 2026-06-12 (restart 08:05:49 local = ~06:05 UTC, section D).
The first post-restart daily cycle ran the same day at 18:00 UTC. Weekend cycles do NOT run
(scheduler `day_of_week="mon-fri"`), so `5f15fdbe` is the ONLY post-flag cycle to date; the
next is Mon 2026-06-15 18:00 UTC (after this AM session).

### E.1 First post-flag cycle ran (handoff/cycle_history.jsonl, verbatim)

    {"cycle_id": "5f15fdbe", "started_at": "2026-06-12T18:00:00.291916+00:00", "completed_at": "2026-06-12T18:39:55.305834+00:00", "duration_ms": 2395013, "status": "completed", "n_trades": 0, "error_count": 0, "data_source_ages": {}, "bq_ingest_lag_sec": null, "meta_scorer_degraded": true}

=> completed, n_trades=0. (Disclosed: `meta_scorer_degraded: true` -- a meta-scorer health
flag on this cycle, tracked separately; it does not alter the trade-count evidence.)

### E.2 Criterion 4a -- 60.2 ON: zero swap_for_higher_conviction SELLs (BQ, verbatim)

Query against `sunny-might-477607-p8.financial_reports.paper_trades` (us-central1), via ADC
Python client (the pinned MCP is location-locked to US; CLAUDE.md BQ rule 6 fallback):

    SELECT created_at, ticker, action, reason, analysis_id, risk_judge_decision
    FROM `...financial_reports.paper_trades`
    WHERE created_at >= '2026-06-12' AND action='SELL' AND reason='swap_for_higher_conviction'
    -- ROWS: 0

=> ZERO post-flag swap_for_higher_conviction SELLs. Criterion 4a satisfied. (The "lacking a
same-cycle analysis_results row" qualifier is moot: there are no such SELL rows at all.)

### E.3 Criterion 4b -- 57.1 ON: zero executed REJECT trades (BQ, verbatim)

    SELECT created_at, ticker, action, reason, risk_judge_decision
    FROM `...financial_reports.paper_trades`
    WHERE created_at >= '2026-06-12' AND risk_judge_decision='REJECT'
    -- ROWS: 0

And all post-flag trades for completeness:

    SELECT ... WHERE created_at >= '2026-06-12' ORDER BY created_at
    -- ROWS: 0

=> ZERO post-flag executed trades, hence zero executed REJECT trades. Criterion 4b satisfied.

### E.4 VACUOUSNESS DISCLOSURE (honest caveat -- do not over-read E.2/E.3)

n_trades=0 means the guardrails were NOT actively exercised in production this cycle. The
criterion-4 conditions are negative assertions ("zero bad rows") and are *vacuously* satisfied
when no trades occur. Per the research brief: there is NO queryable "generated-but-blocked"
table -- a correctly-blocked REJECT is in-memory `summary["risk_judge_blocked"]` + a
`logger.warning` only, never persisted to BQ. So live production evidence for these guardrails
is *necessarily* absence-in-paper_trades; it cannot positively show a block fired. The
literature remedy for a vacuous pass is an "interesting witness" -- provided in E.6.

### E.5 PRE-FLAG CONTRAST -- the antecedent CAN occur (BQ, verbatim, 06-08..06-11)

    SELECT created_at, ticker, action, reason, risk_judge_decision
    FROM `...paper_trades`
    WHERE created_at >= '2026-06-08' AND created_at < '2026-06-12'
      AND (risk_judge_decision='REJECT' OR reason='swap_for_higher_conviction')
    ORDER BY created_at
    -- ROWS: 6
    2026-06-08T18:11:20Z | STX       | SELL | swap_for_higher_conviction | rj=
    2026-06-08T18:11:34Z | DELL      | SELL | swap_for_higher_conviction | rj=
    2026-06-09T18:12:08Z | MU        | SELL | swap_for_higher_conviction | rj=
    2026-06-09T18:12:22Z | SNDK      | SELL | swap_for_higher_conviction | rj=
    2026-06-09T18:12:39Z | 066570.KS | BUY  | swap_buy                   | rj=REJECT   <-- REJECT that EXECUTED (the exact 57.1 audit-basis bug)
    2026-06-10T18:39:40Z | DELL      | SELL | swap_for_higher_conviction | rj=

=> Pre-flag, a REJECT buy executed (066570.KS, 06-09) and swap-churn SELLs fired repeatedly.
These are precisely the events 57.1 / 60.2 now block. The antecedent is real, not impossible.

### E.6 POSITIVE WITNESS -- guardrails fire when triggered (activation tests)

The "ON blocks" behavior is proven by the dedicated regression tests. Plain `pytest` shows 4
failures that are PURE `.env`-bleed artifacts (the tests assert process-default-OFF / off-path
behavior, but the live backend/.env now sets the flags ON, which pydantic-settings reads when
the test omits an explicit override -- e.g. `_make_settings()` then
`assert s_off.paper_risk_judge_reject_binding is False`). Neutralizing the bleed via OS env
vars (which take precedence over .env in pydantic-settings) makes ALL pass:

    PAPER_RISK_JUDGE_REJECT_BINDING=false PAPER_DATA_INTEGRITY_ENABLED=false \
    PAPER_SWAP_CHURN_FIX_ENABLED=false python -m pytest \
      backend/tests/test_phase_60_2_churn_fix.py \
      backend/tests/test_phase_57_1_reject_binding.py \
      backend/tests/test_phase_60_3_data_integrity.py -q
    -- 28 passed, 1 warning in 3.34s

The ON-legs assert exactly the block: `flag-ON must block the REJECT BUY`
(test_reject_binding_main_path_off_emits_on_blocks) and the churn-fix block. This is the
"interesting witness" the vacuousness remedy requires: the guardrails DO block when triggered.

NOTE (out of scope for 61.1; phase-63 defect-register candidate): the 4 plain-run failures
are a TEST-ISOLATION defect (tests read live backend/.env instead of pinning the flag) that
appeared when the flags were flipped ON pre-departure. It is NOT a guardrail regression (E.6
neutralized run proves the logic is intact) and NOT a trading-behavior change. Recorded for
phase-63; not fixed here (would be scope creep on a criterion-4 evidence step).

### E.7 Verdict basis

Criterion 4 closes as **PASS-with-caveat**: the literal negative-assertion conditions are
satisfied on the first post-flag cycle (E.2/E.3), the wiring is verified (research brief),
the antecedent can occur (E.5), and the guardrails provably block when triggered (E.6). The
caveat (E.4 vacuousness) is disclosed, not hidden. Q/A ruling is the authority.
