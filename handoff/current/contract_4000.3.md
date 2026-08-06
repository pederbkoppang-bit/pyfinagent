# Contract -- 4000.3: the operator-gated LIVE smoke window

Step id: 4000.3 (phase-4000, P0, depends_on 4000.1=done + 4000.2=done).
Written 2026-08-06 AFTER the research gate (wf_b7081c09-538, gate_passed=true,
tier=simple, 6 sources read in full, 20 URLs) and BEFORE the window opens.

## Preconditions (all three recorded BEFORE this contract)

1. RAIL TIERS: AS CONFIGURED -- handoff/harness_log.md:31182 (operator,
   2026-08-06; 79.55 closed 5857bc3c).
2. Window authorization + END-STATE RULE stated in advance -- the operator
   action block appended to handoff/harness_log.md 2026-08-06 ("4000.3
   LIVE-WINDOW AUTHORIZATION"): all-pass = rail stays ON; any FAIL = rail
   STAYS ON, evidence queued as defect steps, any OFF-flip is the operator's
   own later call; cap trip = loud stop + report, no retry. Mechanically: the
   pre-window flag is True, the smoke runs WITHOUT --keep-on, so the ExitStack
   restore returns the flag to True on every path (pass/fail/cap/crash).
3. Single-writer window -- git add -An captures taken immediately before and
   after the window, recorded in live_check_4000.3.md.

## Research-gate summary -- window parameters it fixes

- TICKER: AAPL. A manual POST /api/analysis/ persists exactly ONE
  analysis_results row (analysis.py:210-311) and touches no paper_positions /
  paper_trades / strategy_decisions; decide_trades is fed by in-cycle
  rank_candidates (autonomous_loop.py:890 -> screener.py:249, zero
  analysis_results references). The one indirect seam is _fetch_ticker_meta
  metadata (paper_trading.py:1159-1240) -- AAPL avoids even that, since NTAP
  (the only open position) is the one ticker whose metadata the cycle re-reads.
- SCHEDULER COLLISION (hard constraint): paper_trading_daily next_run
  2026-08-06T18:00:00Z (cron mon-fri 14:00 ET) in the SAME process/event loop.
  An overlap would inject cycle rail rows into the window's llm_call_log
  bracket and could blow the <=30 cap with traffic that is not the window's.
  The window therefore opens immediately and must reach its post-flush
  evidence capture BEFORE 18:00 UTC, with the wall-clock margin recorded.
- DEADLINE: --analysis-deadline 3600 (default 1800 is too tight: one step can
  burn 3 attempts x 180s step budget; the deadline is a client-side poll bound
  only and cannot raise rail calls or server budgets).
- EXPECTED CALLS: --expected-calls-per-analysis 30 -- the maximum the
  pre-start gate permits -- DISCLOSED against the ~25-33 static bound; a
  legitimate loud cap trip (exit 3) is an anticipated outcome that closes the
  window and reports.
- E6 honesty: the rail pipeline runs UNGROUNDED (ClaudeCodeClient
  supports_grounding=False, claude_code_client.py:583 -- Market/Competitor/
  DeepDive/EnhancedMacro fall back ungrounded), and CLI dollar figures are
  LIST-RATE local computations "not relevant for billing" on Max
  (code.claude.com/docs/en/costs) -- E6 USD is a proxy, E2's $0-metered claim
  rests on the llm_call_log metered rule + spend bracket. Attribution: both
  live pins are claude-*, so ~100% of window tokens draw the Max pool.
- FAILURE SHAPE: a FAILed analysis is terminal-with-error in the poll body
  (E4 goes red), leaves no analysis_results row and no partial signals; the
  in-memory _tasks entry is the only residue. Do NOT restart the backend
  before evidence capture (the poll 404s after restart).
- E4 leg 4 (persisted row) is owed as LIVE evidence here per the 4000.2 R10
  disclosure: after a completed analysis, capture the analysis_results row for
  AAPL in the window (bounded, date-filtered query) in the live_check.

## Immutable success criteria (copied VERBATIM from .claude/masterplan.json 4000.3)

1. "scripts/qa/verify_phase_4000_3_live_smoke.sh exists, is bounded to this step's own artifacts, and accepts them at handoff/current/ OR handoff/archive/phase-4000.3/ paths so it stays green after archival."
2. "handoff/harness_log.md contains, dated before the window: a 'RAIL TIERS' line; an operator flag-flip token that states the end-state decision rule in advance; and the single-writer confirmation. The check script greps all three."
3. "live_check_4000.3.md contains the machine-readable per-check E1-E6 verdicts, at least one complete CLI envelope with a modelUsage map, the llm_call_log window rows with the 4000.1(c) rule restated inline, and the E6 percent-of-weekly-pool-per-day figure as an explicit number. The check script asserts each by content marker."
4. "The rail-call counter from the smoke's own output shows <=30 calls and <=2 tickers for the window."
5. "The end state of paper_use_claude_code_route is recorded with a verbatim GET /api/settings capture taken after the window, and it matches the pre-stated decision rule applied to the observed check results; if any check failed, a restoration capture is present. Silence on the end state fails the step."
6. "The two git add -An captures bracketing the window are present in the live_check and show no foreign in-flight changes were staged into this step's commits."
7. "If a backend restart occurred inside this step, the live_check shows it happened after the 'RAIL TIERS' line's timestamp and that parent and child workers were both cycled."
8. "Every FAILed check (if any) has a corresponding queued masterplan step id named in the live_check, per auto-memory feedback_queue_discovered_defects_in_masterplan."

## Window plan

1. Pre-window: date -u; git add -An capture; confirm next paper_trading_daily
   run is still >=2h away.
2. Run (background, output teed to the scratchpad):
   .venv/bin/python scripts/qa/smoke_cc_rail_e2e.py --live --ticker AAPL \
     --expected-calls-per-analysis 30 --analysis-deadline 3600
   (no --keep-on; llm-log + spend sources default to bigquery; probe model
   default haiku; flush wait default 65s.)
3. Post-window: git add -An capture; post-window GET /api/settings capture;
   ONE bounded analysis_results query for the AAPL row (E4 leg 4 live
   evidence); assemble live_check_4000.3.md; build the criterion-1 check
   script; experiment_results_4000.3.md; Q/A via the Workflow rail; log; flip
   (tree-quiet check first).
4. No backend restart is needed (4000.1(b): the running process post-dates
   78.2) -- criterion 7 is the vacuous-true branch and the live_check says so.

## Non-scope

One window, one ticker, one invocation. No retry loops against the live rail.
No settings changes beyond the smoke's own PUT/restore pair. No masterplan
edits except queued-defect steps at the flip.
