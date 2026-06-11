# Active Goal -- Post-away-week review -> evidence-gated fixes -> go-live runway

Set by operator 2026-06-10. Supersedes the 2026-06-01 goal, whose autonomous scope
completed on main the same day (53.5 e2e-smoke capstone shipped; 53.4 deferred by the
operator). Full specification: handoff/current/goal_post_away_review.md (the goal prompt).

## North star
(verbatim masterplan.json::goal) "Ship an Intelligence Engine trading system that
maximizes Net System Alpha = Profit - (Risk Exposure + Compute Burn) by dynamically
shifting capital to the highest-earning strategy, recursively self-improving under hard
risk caps, within a 15-slot daily Claude-routine budget." THIS GOAL'S LENS: you cannot
maximize what you cannot measure -- the away week showed the P&L readout itself is
corrupted for non-USD markets; measurement integrity comes first.

## Scope -- in order
1. phase-55 -- away-week forensic review (55.1 data/trading, 55.2 ops/skills,
   55.3 synthesis + operator checkpoint). $0; NO fixes.
2. OPERATOR CHECKPOINT -- replies: 'LLM SPEND: APPROVED <budget> | DECLINED' +
   'PHASE-57: LEVER | FEATURE'
3. phase-56 -- data-correctness + ops fixes (56.1 FX/value/fee + NAV root cause,
   56.2 approve flow / degraded-scoring guard / watchdog / kill-switch follow-up /
   test quarantine). No fix without a 55.x finding ID.
4. phase-57 -- evidence-selected improvement (operator picks a 55.3 candidate spec;
   the full payload is authored at install; not pre-installed).
5. phase-58 -- go-live runway resumption (spend-gated; CLOSES this goal).

## Founding principles (non-negotiable)
- Full harness loop per step: researcher FIRST (>=5 sources read in full, recency
  scan) -> contract.md (criteria verbatim) -> GENERATE -> ONE fresh qa -> harness_log.md
  append -> masterplan flip. No self-evaluation; no verdict-shopping.
- REVIEW BEFORE FIX: phase-56+ steps cite 55.x finding IDs or they FAIL.
- DO-NO-HARM: the US pure-quant momentum core (+20% NAV) stays byte-identical unless a
  config flag is explicitly enabled; no live flag flips inside this goal.
- NO NAIVE NO-TRADE BAND: 53.1's Ledoit-Wolf REJECT is binding.
- UI claims are verified in the live UI via Playwright MCP (NextAuth wall); captures go
  in the live_checks.

## Effort policy
Main xhigh; Researcher + Q/A max (CLAUDE.md effort policy unchanged).

## Done-definition (HARD STOP)
- 58.1 closed (either spend branch) + phase-57 variant installed and executed or
  explicitly deferred by the operator + cycle_block_summary.md refreshed with a crisp
  operator ask list. Write the summary and stop.

## Constraints / gates
- OPERATOR-GATED: LLM API spend (live cycles), pip installs, BQ DROP / unqualified
  DELETE / historical-row backfill, launchctl changes.
- Local runs are main-based: merge claude/sweet-feynman-zhs8p3 before starting.

## Stop conditions
- SOFT STOP: 12 cycles elapsed OR a blocker needing the operator -> summary + crisp ask.

## Cycle ledger (this run)
- (appended per cycle)
- 2026-06-10: install `5d2abb8a`; 55.1 PASS `3222133d`; 55.2 PASS `a747d86b`; 55.3 PASS `2983694f` (checkpoint posted, ts 1781111785.584429); 56.1 PASS `17e53d00`; 56.2 PASS `236b1f86`. Phase-55+56 CLOSED. SOFT STOP: phase-57 install + 58.1 hard-gated on the operator's two verbatim Slack replies (none yet). Ask list: handoff/current/cycle_block_summary.md.
- 2026-06-11: operator decisions recorded in-session (verbatim: LLM SPEND: APPROVED $25 / PHASE-57: FEATURE / backfill execute / F-9: APPROVED); services restarted (v6.37.6); 9-row backfill EXECUTED; phase-57 installed `af4aa8d6` + 57.1 PASS `78b264bf` (Cycle 48). Phase-58.1 window RUNNING; goal closes on window evidence.
