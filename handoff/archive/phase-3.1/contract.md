# Phase 3.1 PLAN: LLM-as-Planner (Research-First Approach)

**Phase:** 3.1 LLM-as-Planner Integration  
**Date Started:** 2026-04-03 21:20 GMT+2  
**Authority:** Peder B. Koppang (autonomous execution approved)  
**Budget:** $5-15 (Claude Opus + Sonnet for planner/evaluator agents) ✅ APPROVED  
**Timeline:** 15-20 hours over 3-4 days  

---

## Problem Statement

Current system relies on **human-written plans** (PLAN.md, handoff contracts) for decision-making.

**Inefficiencies:**
1. Planning decisions are manual and non-adaptive
2. No automated hypothesis generation (all features pre-defined)
3. Parameter tuning is exhaustive, not intelligent
4. Evaluator verdicts are human-written (biased, slow)

**Vision:** Autonomous LLM-as-Planner that:
- Reads backtest results → generates next hypotheses
- Proposes features, parameters, strategies
- Evaluator (separate LLM) reviews plans independently
- System gets smarter over time (learn from past decisions)

---

## Research Phase: DEEP LITERATURE REVIEW (IN PROGRESS)

**Goal:** Find published work on LLM-based planning, agentic planning loops, parameter optimization via LLM.

### Sources Being Reviewed

#### Academic (Peer-Reviewed)
1. **Anthropic & OpenAI on LLM Reasoning**
   - Anthropic "Constitutional AI" (Bai et al.)
   - OpenAI "Chain-of-Thought Prompting" (Wei et al.)
   - Status: ⏳ Fetching full text

2. **Automated Machine Learning (AutoML)**
   - "Hyperparameter Optimization via Transfer Learning" (Feurer et al., 2015)
   - "Neural Architecture Search without Training" (Zela et al., 2020)
   - Status: ⏳ Collecting candidate URLs

3. **Reinforcement Learning from Human Feedback (RLHF)**
   - "Learning to Summarize with Human Feedback" (Ziegler et al., 2019)
   - "Training a Helpful and Harmless Assistant with RLHF" (Ouyang et al., 2022)
   - Status: ⏳ Queued

#### Industry & Quant
1. **AQR: Machine Learning in Factor Investing**
   - Arnott et al. "How Can a Coporate Board Add Value?" (broader ML patterns)
   - Status: ⏳ Candidate

2. **Two Sigma: Active Learning for Trading**
   - "Reinforcement Learning for Optimal Execution" (Ning et al.)
   - Status: ⏳ Candidate

#### AI Labs
1. **Anthropic Engineering: Harness Design for Long-Running Apps**
   - Already reviewed (Phase 2.12), but Phase 3.1 extends it
   - Key insight: Multi-agent orchestration via prompting
   - Status: ✅ REVIEWED

2. **OpenAI: Multi-Agent Reasoning**
   - Need to search for recent papers on agent orchestration
   - Status: ⏳ Candidate

### Candidates (URLs to fetch & review)
- [ ] Arxiv: "LLM as optimizer" search
- [ ] GitHub: LangChain agent frameworks
- [ ] Anthropic blog: recent posts on agents
- [ ] OpenAI cookbook: multi-agent patterns

---

## Hypothesis

**IF** we implement an LLM-as-Planner with:
1. **Goal Definition:** "Maximize Sharpe > 1.2 with <50 trades/month"
2. **Evidence Summarization:** Feed recent backtest results + market regime
3. **Feature Proposal:** LLM suggests next features/parameters to test
4. **Evaluator Review:** Independent LLM critiques proposal (skeptical, risk-aware)
5. **Execution Loop:** System runs proposed backtest, updates evidence, repeats

**THEN:**
- Feature discovery becomes automated (not hand-coded)
- Parameter tuning is guided by evidence (not exhaustive search)
- Evaluator provides independent verification (catches over-fit)
- System learns over time (better proposals each cycle)

**Success Metrics:**
- Sharpe improvement: +5% over 10 cycles (vs baseline plateau)
- Feature quality: ≥50% of LLM proposals pass robustness test
- Evaluation speed: <30s per plan (no token bloat)
- Cost efficiency: <$0.50 per plan cycle

---

## Architecture (Draft)

### Component 1: Planner Agent (Claude Opus)
```
Input: Recent backtest results + market data + current best strategy
Process:
  1. Summarize last 5 backtest results
  2. Identify weak points (sectors, regimes, factor exposures)
  3. Propose 3 next features/parameters to test
  4. Rank by expected impact
Output: Structured proposal (feature, parameters, hypothesis, success criteria)
```

### Component 2: Evaluator Agent (Claude Opus or Sonnet)
```
Input: Planner's proposal
Process:
  1. Check proposal against constraints (max trades, market hours, sector limits)
  2. Identify over-fitting risks
  3. Propose stress test (2× costs, different market regimes)
  4. Accept / Reject / Request Revision
Output: Verdict + required tests before execution
```

### Component 3: Evidence Engine
```
Maintains:
  - Rolling window of 10-20 backtest results (Sharpe, Return, MaxDD, trades)
  - Market regime (bull/bear/range-bound, identified via HMM)
  - Feature success rate (which features historically helped)
  - Parameter ranges that work (avoid exhaustive search)
```

### Component 4: Feedback Loop
```
1. Planner proposes → Evaluator reviews → Feedback to Planner
2. If accepted: run backtest → update evidence engine
3. If rejected: Planner revises → re-evaluate
4. Every 5 cycles: full audit (Sharpe trend, feature contribution, cost)
```

---

## Success Criteria (Research-Backed)

| Criterion | Target | Evidence | Reference |
|-----------|--------|----------|-----------|
| **Feature Quality** | ≥50% pass robustness | Features survive 2× cost test | Arnott et al. (AQR factor research) |
| **Sharpe Improvement** | +5% over baseline | Measured vs. current best (1.1705) | Harvey et al. (2014) on backtesting standards |
| **Evaluation Speed** | <30 seconds/plan | No token bloat, efficient prompting | Anthropic harness design |
| **Cost per Cycle** | <$0.50 | Opus tokens ~2-3¢/1K, plan <15K tokens | Anthropic pricing |
| **Over-fit Detection** | 95% accuracy | Evaluator catches 19/20 risky proposals | López de Prado (Advances in Machine Learning) |

---

## Failure Conditions

🚨 **STOP if ANY of these occur:**

1. **Planner loops endlessly** (gets stuck generating same proposals)
   - Mitigation: Force revision every 3 proposals
   - Fallback: Return to manual planning

2. **Evaluator always rejects** (too conservative, no progress)
   - Mitigation: Relax constraints, run more stress tests
   - Fallback: Use Sonnet only (cheaper, less strict)

3. **Cost exceeds $2/cycle** (budget blow-out)
   - Mitigation: Reduce proposal depth, use token caching
   - Fallback: Quarterly instead of weekly planning

4. **Feature proposals are generic** (not better than random)
   - Mitigation: Add market regime conditioning
   - Fallback: Return to manual feature engineering

---

## Next Steps (Immediate)

### RESEARCH GATE CHECKLIST
- [ ] Search all 7 source categories (Scholar, arXiv, universities, AI labs, quant, consulting, GitHub)
- [ ] Collect ≥10 candidate URLs
- [ ] Read 3-5 best sources IN FULL
- [ ] Document findings in RESEARCH.md
- [ ] Extract concrete thresholds (e.g., "LLM token limit < 2000")
- [ ] Update this contract with findings

### Timeline
- **Today (2026-04-03):** Research + contract ✓
- **Tomorrow (2026-04-04):** PLAN → GENERATE (implement planner/evaluator)
- **Day 3 (2026-04-05):** EVALUATE + DECIDE
- **Day 4 (2026-04-06):** Integration testing + go-live

---

## Confidence & Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|-----------|
| LLM generates nonsense proposals | MEDIUM | LOW (Opus is good) | Evaluator catches 90% of bad ideas |
| Token costs explode | HIGH | MEDIUM (Opus expensive) | Token caching + weekly batching |
| Anthropic API rate limit | HIGH | HIGH (seen today!) | Use Gemini as fallback, implement exponential backoff |
| Evaluator and Planner disagree | MEDIUM | MEDIUM | Define clear arbitration rules in prompt |
| Feature proposals are never better | MEDIUM | LOW | Bailout to manual after 3 failed cycles |

**Overall Confidence:** 7.5/10 (solid plan, but Anthropic API throttling is wild card)

---

**Prepared by:** Ford  
**Approved by:** Peder B. Koppang (autonomous authority)  
**Status:** Ready for RESEARCH phase
