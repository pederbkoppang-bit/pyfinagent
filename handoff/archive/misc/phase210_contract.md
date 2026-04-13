# Phase 2.10 Contract: Karpathy AutoResearch Integration (Autonomous Feature Discovery)

**Date:** 2026-03-29 (prepared in advance, execution pending Phase 2.9 completion)

**Hypothesis:**
PyfinAgent can autonomously discover new trading features (technical indicators, factor combinations, sentiment signals) using LLM-guided feature generation, without human-directed trial-and-error. Using Karpathy's AutoResearch pattern + FAMOSE methodology, we'll create a feature proposer that iteratively:
1. Suggests new features based on data analysis + domain knowledge
2. Evaluates on held-out window (2-week test period)
3. Keeps features with Sharpe improvement ≥ +0.02 (Bailey & López de Prado threshold)
4. Prunes redundant features via mRMR selection

**Success Criteria (Research-Backed):**

1. **Feature Proposer Implementation** (Karpathy AutoResearch template)
   - `backend/agents/feature_generator.py`: LLM agent reads current features + market data schema
   - ReAct loop: reason → code generation → test → evaluate → next iteration
   - Fixed budget: 5-min backtest per feature proposal (scale to full 27-window on acceptance)
   - Metadata: timestamp, source agent, candidate features, test results

2. **Evaluation Logic** (FAMOSE + Harvey/Lo methodology)
   - Candidate feature tested on 2-week holdout (2024-12-16 to 2024-12-31)
   - Acceptance threshold: Sharpe improvement ≥ +0.02 AND DSR ≥ 0.95 (Bailey & López de Prado, 2014)
   - Correlation check: reject if correlated > 0.85 with existing features (FAMOSE mRMR principle)
   - Out-of-sample validation: feature must improve sub-period performance (at least 1 of 3 periods)

3. **Autonomous Loop Integration**
   - `run_harness.py` enhancement: after every 5-experiment plateau, trigger feature generator
   - Feature generator proposes 3-5 candidates per batch
   - Evaluator tests candidates in parallel (time-permitting)
   - Winner integrated into next optimizer cycle

4. **Budget & Constraints**
   - Feature space bounded: only technical indicators + cross-sector combinations (no illiquid data sources)
   - Avoid hallucinations: provide whitelist of available data sources + 10 example features
   - Cost tracking: each feature proposal = $0.05-0.15 (Claude API call), log to budget dashboard
   - Max 10 features/week to avoid overfitting explosion

5. **Audit Trail** (Anthropic harness principle)
   - Every feature has: source agent, proposal timestamp, test result, keep/discard reason
   - `handoff/feature_discovery_log.tsv`: append-only log of all proposals
   - Evaluator can trace why a feature was rejected (for retroactive learning)

**Fail Conditions:**
- LLM proposes invalid code (syntax errors, references non-existent data sources)
- Features improve in-sample but fail out-of-sample (DSR < 0.95)
- Feature induces crashes or timeouts in backtest
- Cost per feature exceeds $0.25 (suggests inefficient proposing)
- All 5 proposals rejected (indicates feature space exhaustion or bad constraints)

**Acceptance Criteria:**
- ≥1 feature proposed and accepted (Sharpe improvement ≥ +0.02, DSR ≥ 0.95)
- Feature integrated into next optimizer run
- Audit trail captured and logged
- No false positives (accepted feature doesn't degrade sub-period performance)
- Cost per accepted feature ≤ $0.20

**Handoff Files:**
- `backend/agents/feature_generator.py` — ReAct agent implementation
- `handoff/phase210_experiment_results.md` — proposed features + results
- `handoff/feature_discovery_log.tsv` — append-only audit trail
- `handoff/phase210_evaluator_critique.md` — PASS/FAIL + evidence

**Timeline:**
- Prep: 2026-03-29 (research + contract, this file)
- Execution: Upon Phase 2.9 completion + Peder's approval of Phase 3 budget ($2-5/cycle)
- Duration: 2-3 hours (5 feature proposals + evaluation + integration)
- Expected completion: 2026-03-29 14:00-16:00 UTC (same day if budget approved)

**Budget Dependency:**
- **BLOCKED until Peder approves Phase 3 budget** ($2-5 per feature cycle = $20-50/month for Phase 3)
- Without approval: skip to Phase 2.10 optimization (CPU-only, zero cost)

**Phase Dependency:**
- Requires Phase 2.9 PASS (multi-market layer must be stable before feature proposer can query diverse markets)
- Feeds into Phase 3 (LLM-guided Planner will coordinate feature discovery + parameter optimization)

---

**RESEARCH CITATIONS:**

1. Karpathy, A. (2026). "AutoResearch: AI agents running research autonomously." GitHub.
   - Provides: iterative code modification pattern, fixed-time budget design, metrics tracking
   - Application: `feature_generator.py` will follow "program the program" principle

2. FAMOSE (2026). "Feature AugMentation and Optimal Selection agEnt" — arXiv:2602.17641
   - Provides: ReAct loop design, mRMR feature selection, handling of feature space explosion
   - Application: our evaluator will use mRMR for collinearity pruning, ReAct for iterative refinement

3. Bailey & López de Prado (2014). "The Probability of Backtest Overfitting" — Journal of Portfolio Management
   - Provides: DSR threshold (0.95) for statistical significance in financial backtests
   - Application: Phase 2.10 features must pass DSR ≥ 0.95 to be accepted

4. Harvey, Liu, Zhu (2016). "... and the Cross-Section of Expected Returns" — Journal of Finance
   - Provides: multiple testing correction, t-stat ≥ 3.0 for claiming findings
   - Application: feature improvement threshold (Sharpe +0.02) is conservative relative to multiple testing burden

---

**EVALUATOR SIGN-OFF (pending execution)**

Once Phase 2.10 executes, evaluator will:
1. Verify ≥1 feature proposed
2. Check all proposed features have valid syntax + data sources
3. Confirm DSR ≥ 0.95 for accepted features
4. Verify no correlation > 0.85 with existing features
5. Confirm audit trail captures all proposals and rationale
6. Sign PASS if all criteria met

**Decision:** PASS → Phase 2.10 COMPLETE → Integrate feature into Phase 3 Planner → Feed to next optimizer cycle
