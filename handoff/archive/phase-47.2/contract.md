# Contract — phase-47.2: First autonomous trade end-to-end

**Cycle:** 6 (resumes the parked 47.2). **Step:** 47.2 | **Phase:** phase-47 | **Status:** in-progress |
**Harness:** required | **Tier:** moderate-complex (executes a real paper trade).

**Authorization for the LLM cost:** operator said "you have my full approval" + "run non-stop to HARD
STOP" (HARD STOP explicitly requires a live n_trades>=1). The `paper_trading_daily` cron runs this exact
Gemini cycle automatically every weekday (standing pre-authorized operation); triggering it now is that
same operation ~18h early, to produce the operator's #1-priority trade. Treated as approved.

## Research-gate summary (PASSED, from 47.2)
Researcher `a58bbea89ecbd5d69`, `gate_passed: true`. 5 sources in full, 13 URLs, recency scan, 6 internal
files. Brief: `research_brief_phase_47_2_first_trade.md`. Validated root cause: `decide_trades` blocks
100% of buys on the per-sector COUNT cap (book is 11-12 Tech vs cap 2; candidates all Tech semis), so
ROTATION (sell weak Tech, buy strong Tech) is the only buy path. sod_date already wired (no fix).

## Code-verification (this session)
`portfolio_manager.py:354-518` + `settings.py:222-235`: the swap-rotation path is ENABLED
(paper_swap_enabled=True), CONFIGURED (paper_swap_max_per_cycle=2, paper_swap_min_delta_pct=25%), and
CORRECT (delta math valid for 0-10 final_scores). It never fired only because the running backend
PREDATED the swap commit 69c710ec -- fixed by the cycle-2/4 restarts (backend now has the code). No code
fix needed; the path just needs a real cycle to fire it.

## Hypothesis
A real `/run-now` cycle, with the rotation code now loaded + fresh prices (47.1) + correct metrics (47.4),
fires the swap path: it sells the lowest-conviction Tech holding and buys a higher-conviction Tech
candidate (e.g. STX final_score 8.0), producing >=1 trade with all safety caps intact (kill-switch,
stop-loss, sector NAV-pct backstop, max-per-cycle=2). If 0 trades, observe the post-restart swap-decision
log lines and escalate Fix B1 (swap-score resolution / lower min_delta).

## Immutable success criteria (verbatim from masterplan.json phase-47.2)
1. POST /api/paper-trading/run-now returns n_trades >= 1 with non-empty candidate analyses
2. a fresh financial_reports.paper_trades row dated today exists
3. root cause of empty new_candidates identified + fixed without disabling safety
4. live_check_47.2.md captures the run-now response + BQ paper_trades row dated today

(Note: criterion 3's wording says "empty new_candidates" -- the research REFUTED that; the validated
cause is sector-cap-without-rotation, resolved by the loaded rotation path. No safety disabled.)

## Plan steps
1. POST /api/paper-trading/run-now (real cycle, dry_run=false). Capture n_trades + candidate count.
2. Read backend.log for the post-restart swap-decision lines (Swap skip / swap pair / sector NAV-pct).
3. Query BQ financial_reports.paper_trades for a row dated today.
4. If n_trades>=1 -> write experiment_results + live_check_47.2.md -> fresh Q/A -> log -> flip 47.2 done.
   If n_trades=0 -> diagnose from the swap log, apply the precise Fix B1 (code), re-run once.

## Blast radius
Executes REAL (virtual) paper trades -- reversible via flatten-all. Fires the Gemini Layer-1 pipeline
(LLM cost = one standing daily cycle). No DROP/DELETE. Kill-switch + stop-loss + sector caps stay intact.

## References
- `research_brief_phase_47_2_first_trade.md`; `roadmap_master.md` workstream 2
- `backend/services/portfolio_manager.py:354-518` (swap path); `backend/services/autonomous_loop.py` (run_daily_cycle)
- `backend/config/settings.py:222-235` (paper_swap_* defaults)
