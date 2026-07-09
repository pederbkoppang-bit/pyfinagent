# Goal Prompt -- goal-phase68-real-fill-runway

Set: 2026-07-10 (operator in-session /goal, Stop hook active). Installed as masterplan
phase-68 (depends_on phase-66), QUEUED BEHIND phase-67's P0s. **67.4 Sunday revert
untouchable.**

## Theme (operator verbatim)

"Real-Fill Runway: every fill ever made is synthetic; convert the mock engine into an
Alpaca-paper-executed, price-verified, learning system producing GO_LIVE_CHECKLIST
4.4.2.2/.4/.5 evidence. Alpha re-opening DEFERRED to phase-69 (hold strategy constant
while measuring divergence)."

## Audit basis (operator-supplied; each claim tool-verified at the owning step's
## research gate before it is acted on)

1. Execution stuck on bq_sim (execution_router.py default); EXECUTION_BACKEND env
   never populated (pydantic env_file doesn't export; launchd plist lacks env keys;
   researcher memory project_funnel_zero_trade_66_2.md); creds absent ->
   _alpaca_mock_fill.
2. Corrupted money evidence: AMD/MU avg_entry $545/$1005 vs real ~$150/~$110
   (live_check_66.2.md section 9 + Cycle-77 register). avg_entry feeds trailing-stop
   math = LIVE RISK. Also: DESC-order phantom drawdown pages + hash-fallback price
   path.
3. Learn-loop 0 rows ever; the only remaining blocker is paper_learn_loop_enabled=False
   (the return_pct bug was fixed in 47.7).
4. DON'T RE-FIX: alerting imports fixed in 66.1; cc-rail guard done in 66.1/66.4.
5. The Alpaca paper account carries stray shorts (-13,842.89 short_market_value, the
   2026-06-10 MCP drill artifact per the 66.2 close): flatten BEFORE drift
   measurement. alpaca-py is pinned (0.43.2).

## Steps (masterplan phase-68; verification objects live in .claude/masterplan.json)

- **68.0 P0, CALENDAR-GATED before 2026-07-12 (Fable window):** deep-research + design
  pack -- env propagation to the launchd process; shadow-mode TRUE order semantics
  (code-traced); alpaca-py paper order/reconciliation; AMD/MU price-defect hypothesis
  tree. Deliver research_brief_68.0.md + design_execution_cutover_68.md (config
  precedence, shadow isolation, order-id idempotency, rollback=env flip, PKLIVE guards
  kept). NOTE: research spawned 2026-07-10 on the session's Opus roster snapshot
  (Fable agent-file pins bind next session); Main (Fable) authors the design doc.
- **68.1 P0 (dep 68.0):** EXECUTION_BACKEND demonstrably reaches execution_router in
  the launchd process; startup log = mode+source; default byte-identical bq_sim;
  missing creds logs LOUD; paper-only triple-enforcement tested. DARK.
- **68.2 P1 (dep 68.1; token `ALPACA-RESET: APPROVED`):** flatten stray shorts
  (before/after paste); >=5 days shadow paired fills to BQ + drift report; shadow
  NEVER mutates bq_sim position state.
- **68.3 P0 (dep 68.2; DARK until `EXEC-BACKEND: ALPACA_PAPER`):** cutover -- >=3
  SCHEDULED cycles with source='alpaca_paper' fills; reconciliation <2% drift, zero
  orphans; mismatch pages P1; one rollback drill; risk caps byte-untouched.
- **68.4 P1 (dep 68.0; DARK until `PAPER-LEARN-LOOP: ENABLE`):** dark write-drill vs
  LIVE schemas (financial_reports / us-central1); the token ask states the measured
  reflection cost; post-token the first real sell-close writes outcome_tracking +
  agent_memories rows; NO manufactured SELLs.
- **68.5 P0 (dep 68.0):** root-cause AMD/MU prices to file:line with repro; correct
  the rows auditably; pre-persist fill-price sanity gate (block+page on deviation vs
  an independent quote; covers close-cache + hash-fallback); fix the DESC phantom
  with a regression test; FX-1 root cause -> hand the fix to parked 61.3; file
  defects into 63.3 seeds (do NOT execute 63.3).
- **68.6 P1 (dep 68.3; also needs 68.5):** weekly go-live tracker from REAL fills
  only, immutable clean-window start: Sharpe vs 0.82, missed days (stop-loss sells
  count), divergence per 4.4.2.5. Feeds 65.3/65.4 + 58.1; flips nothing.
- **68.7 P2 (DARK until `STALE 35-44: APPROVED`):** 66.5-style triage table for stale
  phases 35/39/43/44 (recommend 35.3 superseded-by 68.6); no status surgery before
  the token. (Existing open ask STALE-PHASE-TRIAGE-35-44 amended with this token.)

## Window plan (operator verbatim)

"67.6 first; install 68 + spawn 68.0 research now; 68.5/68.4 briefs Sat; 67.5 if
capacity; Sun: commit briefs, then 67.4. 68.1+ continues on Opus."
(State at install time: 67.6 AND 67.5 already closed PASS -- ahead of plan; 68.0
research spawned at install.)

## Boundaries (binding -- violation = FAIL)

- $0 metered; paper-only TRIPLE-ENFORCED, no live key ever.
- Do-no-harm on stops/caps/kill-switch/thresholds.
- DARK-until-token on every behavior change (tokens above; away-ops token mechanics).
- Fable doctrine + STALL WATCH (qa.md frontmatter); 67.4 Sunday revert untouchable.
- No bypass of parked 61.x / 63-65 work; 68.5 HANDS OFF findings, never executes them.
- Full 5-file protocol per step; harness stays exactly 3 agents.

## Definition of done (whole goal)

68.0-68.7 all PASS per their immutable criteria: the engine executes on Alpaca paper
through its normal gates with price-verified fills, reconciliation + rollback proven,
the learn-loop writes real outcome rows, and the weekly tracker produces
GO_LIVE_CHECKLIST 4.4.2.2/.4/.5 evidence from real fills -- with strategy/alpha
untouched (phase-69's job).
