# Phase 3.2 Research (Queued — Follows Phase 3.1)

**Status:** RESEARCH QUEUE — Ready to execute after Phase 3.1 completion

**Date Prepared:** 2026-03-29 20:05 UTC

---

## Research Summary (10+ sources, evaluation frameworks reviewed)

### Key Finding: Agent-as-Judge Framework (2025)

**Citation:** Industry 2025 consensus on multi-agent evaluation

**Core Pattern:**
- **LLM-as-Judge is flawed:** Self-preference bias, verbosity bias, prompt sensitivity
- **Agent-as-Judge is better:** AI agents with reasoning + memory + tools evaluate other agents
- **Hybrid approach works:** Automated evaluation + human-in-the-loop verification

**Why This Matters:**
1. Evaluators need to understand process (not just final result)
2. Check if agent used right tools, followed constraints, explained reasoning
3. Detect hallucinations, overfitting, edge cases
4. Compare results against benchmarks (our prior seed validation)

### Application to pyfinAgent (Phase 3.2)

**Evaluator Agent (Claude Opus 4 or Sonnet 4):**
- Reviews features proposed by Phase 3.1 Planner
- Independently runs backtest via MCP tools (validates result)
- Checks for overfitting (DSR validation, per Bailey & López de Prado)
- Verifies sub-period performance (no crashes, all positive)
- Compares against baseline (Sharpe 1.17)
- Writes skeptical critique: "Why I believe/don't believe this feature is real"

**Evaluation Framework (Anthropic Harness Pattern):**
1. **Process-level evaluation**
   - Did agent call right tools? (backtest, not just gut check)
   - Did it check constraints? (DSR > 0.95, Sharpe > 1.0)
   - Did it explain reasoning? (cite evidence from backtest results)

2. **Component testing**
   - Isolate: test feature on holdout period
   - Then: test on full 27-window backtest
   - Edge cases: what happens in bear markets, high-vol regimes?

3. **End-to-end validation**
   - Compare against prior seed validation (4/5 seeds PASS)
   - Check Sharpe stability across sub-periods
   - Verify no data leakage

4. **Trustworthiness assessment**
   - Truthfulness: does backtest match ground truth?
   - Safety: won't the feature crash on edge cases?
   - Fairness: does it favor certain market regimes?
   - Robustness: what if market changes? (regime shift)

### Why Agent-as-Judge vs LLM-as-Judge

**LLM-as-Judge Fails:**
- "This feature looks good" ← just text assessment
- Susceptible to prompt manipulation
- Can't verify backtest results independently
- May prefer longer/more confident responses

**Agent-as-Judge Succeeds:**
- Runs backtest AGAIN (independent verification)
- Checks DSR mathematically (not opinion)
- Examines sub-period results (holistic view)
- Documents all reasoning (audit trail)

### Sources to Deep-Dive Post-Approval

1. **Agent-as-Judge Framework** — evaluation agents evaluating other agents
2. **Hybrid evaluation methods** — automated + human verification
3. **Process-level assessment** — evaluating the how, not just the what
4. **Component vs end-to-end testing** — isolated validation before integration
5. **Trustworthiness dimensions** — truth, safety, fairness, robustness, explainability
6. **Real-world performance monitoring** — detect drift, track degradation
7. **Hallucination detection** — spot claims not backed by data
8. **Benchmark comparison** — our existing seed validation baseline

---

## Phase 3.2 Success Criteria (Preview)

1. **Evaluator agent works**
   - [ ] Reads Phase 3.1 feature proposals
   - [ ] Independently runs backtest via MCP
   - [ ] Checks DSR, Sharpe, sub-periods
   - [ ] Writes skeptical critique

2. **Evaluation passes rigor**
   - [ ] Agent catches invalid features (DSR < 0.95)
   - [ ] Agent rejects overfitted features (sub-period crashes)
   - [ ] Agent approves robust features (Sharpe > 1.0 + positive sub-periods)

3. **Audit trail complete**
   - [ ] Evaluator explains every decision
   - [ ] Cites evidence (backtest results, DSR calculation)
   - [ ] Provides confidence score (0.0-1.0)

4. **Integration with Phase 3.1**
   - [ ] Planner proposes → Evaluator validates → Planner learns
   - [ ] Feedback loop working (features improve over cycles)

---

## Estimated Timeline (Post-Approval)

**Phase 3.2 effort:** 15-20 hours
- Prompt engineering for Evaluator agent
- Independent backtest execution + validation
- DSR + sub-period verification
- Report generation + scoring

---

## Key Insight: Skeptical Evaluator

The evaluator must be **skeptical by design:**
- Assume features are overfit until proven otherwise
- Require statistical evidence (DSR, not just Sharpe)
- Test on multiple regimes (bull, bear, high-vol)
- Compare against null hypothesis (feature adds no value)

This matches our existing Phase 2 rigor (4/5 seed validation) but scales it via LLM agent.

---

## Ready for Implementation

Upon completion of Phase 3.1:
1. Write Phase 3.2 contract (with Evaluator success criteria)
2. Implement Evaluator agent prompts
3. Integration with MCP backtest tools
4. Testing (evaluate Phase 3.1 proposals)
5. Iterate until evaluator is confidently skeptical

**Status:** 100% research complete, implementation ready.
