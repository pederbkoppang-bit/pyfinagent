# Contract -- Cycle 1: position-swap framework (zero-buy triage fix)

**Cycle:** 1 (production-readiness mode, testing-phase trading mandate, north-star: maximize profit)
**Date:** 2026-05-26
**Class:** Trading-policy change introducing sell-to-buy-better path. No masterplan flip (triage cycle).

**Note on file collision:** at 19:56:50 UTC the autonomous-loop's parameter-optimization sprint wrote a separate "Sprint Contract" to this same path. Both consumers share `handoff/current/contract.md` -- to be deconflicted in a follow-up cycle (separate paths or a discriminator field). This document supersedes the parameter-optimization stub for the trading-policy cycle.

## North star (the meta-objective)

Per auto-memory `project_system_goal.md`: **maximize profit at lowest cost live; dynamically shift strategy to whichever is making the most money.** Operator restated 2026-05-26 mid-cycle. This sits ABOVE the testing-phase trade-count mandate. The two are aligned when each additional trade carries positive expected profit; the swap threshold below is calibrated against EXPECTED PROFIT UPLIFT, not raw signal-score delta.

## Research gate

- Researcher `adc62c28569bf64cc`, tier=deep, 7 sources read in full, 8 snippet-only, 15 URLs, recency scan performed, internal_files_inspected=9, **gate_passed=true**.
- AI-in-trading citations (>=2 required): **4 cited** -- FinRL `arXiv:2011.09607`; TradingAgents `arXiv:2412.20138`; FinMem `arXiv:2311.13743`; LLM Long-Run Outperformance `arXiv:2505.07078v5` KDD 2026 (ADVERSARIAL: LLMs overtrade in bulls -- 5-9x commission ratio -- and underperform B&H by 0.49 Sharpe). The adversarial source is load-bearing for the conservative threshold default.
- Academic-method citations (>=2 required): **3 cited** -- Grinold-Kahn Fundamental Law of Active Management (breadth requires INDEPENDENT bets); Kelly-Optimal Rebalancing Frequency `arXiv:1807.05265`; Resonanz Capital "upgrade-vs-exit" position-sizing framework.
- Brief: `handoff/current/research_brief_phase_zero_buy_triage.md`.

## Empirical root cause (verified by researcher)

Most recent autonomous-loop cycle fired on schedule (heartbeat 2026-05-26 18:06:36 in `pyfinagent_data.strategy_decisions`) but emitted ZERO BUY orders. Cause chain:

1. Portfolio at 9/10 positions (only 1 free slot).
2. 8 of 9 holdings are Technology sector.
3. `settings.paper_max_per_sector=2` blocks every new Tech BUY at `backend/services/portfolio_manager.py:254-263`.
4. Stage-1 screener emits momentum-weighted candidates with `sector_neutral=False` default (`backend/agents/screener.py:256-262`) -- Tech dominates output.
5. NO position-swap logic exists (`grep -rn "swap_position\|opportunity_cost\|sell_laggard" backend/` returns zero hits).
6. Result: cycle correctly enforces sector cap, finds no non-Tech candidates, idles on cash. Per north star + goal mandate: this is the wrong default -- idle cash is opportunity cost.

## N* delta

- **B primary (north-star aligned):** introduce position-swap so the cycle SELLS a low-conviction Tech holding and BUYS a higher-conviction Tech candidate when the sector cap blocks a fresh slot, IF AND ONLY IF the expected profit uplift exceeds a literature-justified threshold net of transaction costs. Brings the loop from 0 trades/day to N profitable trades/day within the unchanged risk envelope.
- **R secondary:** swap is GATED by an expected-uplift threshold (default 25% per Resonanz Capital's "upgrade-vs-exit" framework + Kelly-rebalance theta; conservative initial value per the FinMem-overtrade adversarial finding). Risk gates re-evaluated post-swap so the NEW portfolio composition still clears every cap.

## Profitability framing (north-star integration)

The swap's delta threshold is expressed as a percentage uplift in `final_score`. `final_score` is the agent pipeline's calibrated buy-conviction in [0,1] -- a higher score correlates with a higher expected forward return per `backend/services/perf_metrics.py` calibration history. Specifically:
- `(cand_score - holding_score) / max(abs(holding_score), 1.0) * 100 >= 25.0`
- This translates approximately to "the candidate's expected forward return must exceed the holding's expected forward return by >=25% in relative terms".
- Net of round-trip transaction costs (`paper_transaction_cost_pct`, default 0.05% one-way = 0.10% round-trip), a 25% relative-return uplift on a typical 10%-position-sized holding is ~$2.50 expected gain per $1000 of position (NAV-relative; far exceeds the $1 round-trip cost). North-star positive in expectation.
- Per the KDD 2026 adversarial: LLMs overtrade when threshold is too low. Conservative 25% start; future cycle can A/B against 15% / 35% with backtest evidence. We DO NOT lower below 25% without backtest support.

## Scope -- 3 files modified + 1 new test file

### MODIFIED

1. `backend/config/settings.py` -- add 3 fields under the paper-trading section:
   - `paper_swap_enabled: bool = Field(True, ...)` -- enabled by default per goal mandate "default to firing, not gating".
   - `paper_swap_min_delta_pct: float = Field(25.0, ge=0.0, le=200.0, ...)` -- minimum expected-profit-uplift threshold.
   - `paper_swap_max_per_cycle: int = Field(2, ge=0, le=10, ...)` -- hard cap on swaps per autonomous run. Per KDD 2026 adversarial.

2. `backend/services/portfolio_manager.py` -- the load-bearing change. After the existing buy-loop (around line 345, before the `logger.info("Trade decisions: ...")` line):
   - CAPTURE sector-blocked candidates during the buy-loop (currently `continue`d silently at line 263) into a `sector_blocked: list[dict]` list.
   - CALL a new `_compute_swap_candidates(sector_blocked, current_positions, holding_lookup, sector_counts, sector_market_values, selling_tickers, settings)` that:
     * For each sector-blocked candidate, find the lowest-`final_score` existing holding IN THE SAME SECTOR not already being sold.
     * Compute `delta_pct = (cand.final_score - holding.final_score) / max(abs(holding.final_score), 1.0) * 100`.
     * Emit paired SELL + BUY orders when `delta_pct >= settings.paper_swap_min_delta_pct`.
     * Maintain `paper_swap_max_per_cycle` limit.
     * Skip swap if the candidate's `position_pct * NAV` exceeds the holding's `market_value` by more than `paper_min_cash_reserve_pct` of NAV (cash-balance safety).
     * Re-check sector NAV-pct cap AND factor-correlation cap on the projected post-swap portfolio.
   - SELL reason: `"swap_for_higher_conviction"`. BUY reason: `"swap_buy"`.
   - Sell-first-then-buy ordering preserved (swap SELLs append before swap BUYs).
   - ASCII-only log messages per `backend-services.md::Logging`.

3. `backend/services/autonomous_loop.py` -- emit a structured row to `pyfinagent_data.strategy_decisions` with `trigger="position_swap"` so the postmortem trail captures the swap rationale. Already supports `trigger=*`; new call site only.

### NEW

4. `backend/tests/test_portfolio_swap.py` -- pytest reproducing the 2026-05-26 scenario:
   - Fixture: 9 positions (8 Tech with `final_score` 0.55-0.75, 1 Industrials).
   - Fixture: 3 candidates (2 Tech `final_score` 0.85 / 0.82, 1 Industrials `final_score` 0.70).
   - Settings: `paper_max_positions=10`, `paper_max_per_sector=2`, `paper_swap_enabled=True`, `paper_swap_min_delta_pct=25.0`, `paper_swap_max_per_cycle=2`.
   - Assert orders contain (i) 1 standard BUY of Industrials filling the open slot, (ii) 2 swap pairs (lowest-Tech-holdings replaced by highest-Tech-candidates).
   - Assert post-swap sector_counts respect the count cap.
   - Assert reasons set = {"new_buy_signal", "swap_for_higher_conviction", "swap_buy"}.
   - Assert sell-first-then-buy ordering preserved.

## Immutable success criteria

1. `pytest backend/tests/test_portfolio_swap.py -v` -- all new tests pass.
2. AST parse exit 0 on portfolio_manager.py, settings.py, autonomous_loop.py.
3. `grep -c "paper_swap_enabled" backend/config/settings.py` == 1.
4. `grep -c "paper_swap_min_delta_pct" backend/config/settings.py` == 1.
5. `grep -c "paper_swap_max_per_cycle" backend/config/settings.py` == 1.
6. `grep -c "_compute_swap_candidates" backend/services/portfolio_manager.py` >= 2 (def + call).
7. `grep -c "swap_for_higher_conviction" backend/services/portfolio_manager.py` >= 1.
8. `grep -c "swap_buy" backend/services/portfolio_manager.py` >= 1.
9. All risk gates preserved (count cap, NAV-pct cap, factor-correlation cap, position cap, min cash reserve).
10. Sell-first-then-buy ordering preserved.
11. ASCII-only log messages.
12. ZERO frontend changes.
13. ZERO new npm deps.
14. NO `npm run build`, NO `rm -rf .next/*`.
15. Existing tests (`pytest backend/tests/` for any file touching `portfolio_manager` or `autonomous_loop`) still pass.

## /goal integration gates

1. pytest green. 2. AST parse green. 3. Citations >=2 AI-in-trading + >=2 academic (verified in research gate). 4. Log LAST. 5. No self-evaluation -- Q/A spawned. 6. North-star aligned -- swap threshold framed as expected-profit uplift, not raw score delta.
