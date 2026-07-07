# Contract -- 66.2 Redeploy capital via the NORMAL path (goal-phase66-reactivation)

Step: 66.2 | Cycle 73 (prep) | 2026-07-07 | Operator present
Sequencing: operator authorized PREP 2026-07-07 ~15:57 UTC (AskUserQuestion;
masterplan sequencing_note) while 66.1 awaits tonight's scheduled cycle. PREP =
research + this contract + funnel tooling + the two read-only integrity checks.
The >=5-healthy-rail-trading-day BUY/no-BUY evidence clock starts ONLY after 66.1
closes. NO trading-behavior change of any kind.

## Research-gate summary

research_brief_66.2.md (tier complex, gate_passed: true; 8 read-in-full / 35 URLs /
recency scan / 16 internal files). Load-bearing:
- A CASH-DEPLOYMENT BUY PATH EXISTS: decide_trades section-2
  (backend/services/portfolio_manager.py:147-399) emits BUYs directly from _BUY_RECS
  analyses; the swap path (:439) needs sector-blocked candidates (impossible on an
  empty book) -> paper_swap_churn_fix_enabled is a NON-FACTOR for the first BUY.
- RAIL DEATH = STRUCTURAL ALL-HOLD: empty CC response -> {"action":"HOLD",
  "confidence":0} (autonomous_loop.py:2092-2096) -> zero buy candidates; 06-11 was
  already 3/5 degraded (log-proven). Zero-trades 06-11..07-06 is fully explained by
  the rail; no second hidden gate suspected for BUYs.
- FUNNEL COUNTING SOURCES: n_trades NEVER counts Step-5.6 stop-loss SELLs (the book
  emptied invisibly to cycle_history) -> the funnel tool must count paper_trades
  (BQ) + llm_call_log analysis rows + cycle summary keys + 66.1's
  rail_skipped/breaker_tripped; stages with no counter are enumerated in the brief.
- ALPACA: the autonomous loop has NEVER touched Alpaca (launchd env lacks
  EXECUTION_BACKEND/keys; execution is bq_sim always). short_market_value is SIGNED
  (equity = cash + long_mv + short_mv, official docs) -> -13,842.89 is REAL short
  exposure on the paper account created by NON-ENGINE sessions (MCP/drills).
  Check design: read-only get_account_info; NO orders.
- paper_portfolio SINGLE ROW IS BY DESIGN (aggregate; write path cited in brief);
  the real integrity question is USD-magnitude correctness of KR paper_trades rows.
- PREDICTION (falsifiable): tonight may STILL show rail failure because the 66.1
  probe/rail run in the LAUNCHD context (keychain access differs from the
  interactive session where the probe passes); if so, 66.1's new
  rail_skipped/breaker labeling + probe-failure P1 (imports fixed) make it VISIBLE
  -- that is the system working, and the launchd-keychain fix becomes the next
  action. If the rail is healthy: expect <=2 BUYs (sector caps) or a correctly
  gated 0.
- BONUS DEFECT (register): phantom "drawdown -61.51%" P1 from a DESC snapshot trap
  in drawdown_alarm.

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.2)

1. "EITHER (a) first BUY row in financial_reports.paper_trades dated after the 66.1
   deploy, executed by the ordinary scheduled pipeline with risk_judge_decision
   recorded (BQ paste), OR (b) after >=5 consecutive healthy-rail trading days with
   zero BUYs, a Q/A-verified per-stage funnel diagnosis with candidate counts at
   every gate (signals -> scorer -> risk judge -> execution) from BQ/logs
   distinguishing gates-correctly-reject from pipeline-defect"
2. "No gate threshold, risk cap, position limit, or entry criterion modified to
   manufacture trades (git diff evidence over the step's commit range)"
3. "Alpaca short_market_value anomaly (-13842.89 long-only) explained with evidence
   or filed as a verified defect with root cause"
4. "paper_portfolio single-US-row representation for EU/KR markets verified as
   intended design (cite code) or filed as a verified defect"

Verification command (immutable): the BQ BUY query in the masterplan step.
live_check: live_check_66.2.md with the BUY BQ row or the 5-day funnel diagnosis,
plus both integrity-check verdicts.

## Plan (prep now; evidence clock post-66.1)

1. Build scripts/diagnostics/funnel_report.py (READ-ONLY): per date-range +
   per-market stage counts from existing sources only (universe -> analyses
   dispatched (llm_call_log) -> rail health (cycle_history rail_skipped/
   breaker_tripped + claude_rail_healthy) -> non-HOLD recommendations
   (strategy_decisions/persisted analyses) -> risk-judge outcomes -> executed
   trades (paper_trades incl. stop-loss SELLs)); missing-counter stages reported
   as "no instrumentation" lines, never silently omitted.
2. Prove the tool on the outage window (2026-06-10..07-06): it must SHOW the
   all-HOLD collapse -- that is its acceptance test.
3. Integrity check A (criterion 3): read-only Alpaca account inspection; verdict =
   explained (non-engine orders; engine never connected) + defect-register entry
   for the account hygiene question; NO orders placed.
4. Integrity check B (criterion 4): cite the by-design write path + run the KR-row
   USD-magnitude BQ check; verdict accordingly.
5. After 66.1 closes (tonight): criterion 1 resolves via (a) on the first ordinary
   BUY or via (b) after the 5-day clock; interim funnel reports appended per cycle.
6. Fresh Q/A at each evidence milestone; expected first verdict CONDITIONAL
   (criterion 1 clock running) with criteria 2/3/4 closed.

## Scope boundaries

READ-ONLY throughout prep: no gate/threshold/sizing/entry changes (criterion 2), no
.env, no orders (Alpaca or paper), no trading code edits. The funnel tool is a new
standalone diagnostics script only.

## References

research_brief_66.2.md; portfolio_manager.py:147-399/:439; autonomous_loop.py:
2092-2096; docs.alpaca.markets account fields; 50.x multi-market design;
funnel-instrumentation practice per brief section 7.
