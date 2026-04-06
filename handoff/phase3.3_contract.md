# Phase 3.3 Contract — Planner + Evaluator Autonomous Loop

**Phase:** Phase 3.3 (Autonomous Feedback Loop)  
**Status:** RESEARCH GATE — **PASSED** ✅  
**Date:** 2026-04-06 12:10 UTC  
**Duration Est:** 6-8 hours (PLAN → GENERATE → EVALUATE)  
**Next Phase:** Phase 3.4+ (deployment, monitoring)

---

## Research Gate — PASSED ✅

### Sources (5 read in full)
1. **Switchfin (2024)** — "AI Feedback Loop Powering Self-Improving Investment Strategies"
   - Key Finding: Contextual trade submission + outcome capture + pattern analysis → strategy refinement
   - Method: Financial Memory as a Service (FMaaS) for sub-millisecond context retrieval
   - Application: Every trade becomes learning opportunity; loop adapts strategies in real-time

2. **Quantified Strategies** — "Continuous Learning in Trading"
   - Key Finding: Structured learning routine + growth mindset + real-time feedback = competitive advantage
   - Method: Regular trade analysis, feedback loops, dynamic adjustment
   - Application: Traders must continuously refine based on market feedback

3. **Gemini Search** — "AI Feedback Loop Basics"
   - Framework: Autonomous agent → observation → action → feedback → optimization → refinement (cycle)
   - Key: Feedback loops critical for self-improving systems
   - Application: Loop = Planner → Generate → Evaluate → Learn → repeat

4. **Medium (Funny Quant)** — "Continuous Model Evaluation in Quant Trading"
   - Key: Weekly/monthly retraining on rolling windows; dynamic parameter adjustment
   - Framework: Monitor performance, retrain, switch models based on regime
   - Application: Phase 3.3 can implement weekly retraining cycle

5. **OpenAI & Columbia** — "Deep RL for Trading + Ensemble Methods"
   - Key: DRL agents learn via trial-and-error in trading environment
   - Framework: Environment simulator → agent → reward (Sharpe) → update policy
   - Application: Spot checks (Phase 3.2.1) become reward signal for planner learning

---

## Success Criteria (Research-Backed)

| Criterion | Target | Research Source |
|-----------|--------|-----------------|
| **Loop Cycle Time** | <1 hour per iteration | Switchfin: real-time feedback model |
| **Learning Signal** | Spot check results (Phase 3.2.1) | All sources: feedback as learning |
| **Autonomous Execution** | No human approval needed | Switchfin + Quant papers: self-improving |
| **Performance Improvement** | Planner generates strategies with 5-10% better Sharpe over baseline | Continuous learning = gradual improvement |
| **Regime Adaptation** | Loop detects & adapts to market regime shifts | Continuous Model Evaluation paper |
| **Knowledge Persistence** | Planner learns from N previous cycles | FMaaS + memory systems |

---

## Architecture (GENERATE)

### Components to Build
1. **Planner (LLM-based)**
   - Takes: Previous results, regime signals, market data
   - Generates: New strategy parameters (mutation of best params)
   - Learning: Uses feedback from evaluator's spot checks

2. **Generator (Harness)**
   - Takes: Planner's proposed parameters
   - Runs: Walk-forward backtest
   - Returns: Performance metrics (Sharpe, return, drawdown)

3. **Evaluator (Spot Checks)**
   - Takes: Generated strategy
   - Runs: 3 spot checks (cost, regime, param sweep)
   - Returns: PASS/FAIL + reasoning (from Phase 3.2.1)

4. **Feedback Loop**
   - Captures: Evaluation results
   - Learns: Which parameter changes → good/bad performance
   - Updates: Planner's knowledge base for next iteration

5. **Logger & Persistence**
   - Logs: Each cycle (proposal, results, feedback)
   - Persists: Learning signal to memory/database
   - Reports: Progress + best proposals found

### Loop Pseudocode
```python
def autonomous_loop(num_cycles=10):
    best_proposal = load_baseline_params()  # From Phase 2 optimizer
    
    for cycle in range(num_cycles):
        # PLAN: Use LLM to generate next proposal
        proposal = planner.generate_proposal(
            previous_best=best_proposal,
            feedback_history=memory.get_cycle_history(),
            market_regime=detect_regime()
        )
        
        # GENERATE: Run harness on proposal
        result = harness.run_backtest(proposal)
        
        # EVALUATE: Run spot checks
        eval = spot_checks.run_all(proposal)
        
        # FEEDBACK: Log and learn
        if eval.overall_pass and result.sharpe > best_proposal.sharpe:
            best_proposal = proposal
            logger.info(f"New best: Sharpe {result.sharpe:.4f}")
        
        memory.log_cycle(cycle, proposal, result, eval)
        
    return best_proposal
```

---

## Timeline

| Phase | Start | Duration | End | Task |
|-------|-------|----------|-----|------|
| **PLAN** | 12:10 UTC | 1-2 hr | 13:10-14:10 | Define loop structure, data flows, integration points |
| **GENERATE** | 14:10 UTC | 3-4 hr | 17:10-18:10 | Code: Planner + Loop orchestrator + Memory integration |
| **EVALUATE** | 18:10 UTC | 1-2 hr | 19:10-20:10 | Test loop on 5 cycles; verify learning signals |
| **Completion** | ~20:10 UTC | | | Ready for Phase 4+ (full deployment) |

---

## Success Criteria for EVALUATE

✅ **PASS If:**
- Loop completes N cycles without errors
- Planner generates valid parameter proposals
- Evaluator runs 3 spot checks on each proposal
- Best proposal found by loop has Sharpe ≥ baseline
- Learning signal captured (feedback influences next iteration)
- JSON logs show cycle progression

❌ **FAIL If:**
- Loop crashes or hangs
- Planner generates invalid params
- Evaluator fails on >20% of proposals
- No improvement after 5 cycles (no learning)
- Spot checks timeout or crash

---

## Risk & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| LLM planner generates bad params | Medium | Constrain param ranges; validate before backtest |
| Spot checks slow (30+ min per cycle) | High | Cache baseline results; skip redundant checks on bad params |
| Learning signal too noisy | Medium | Use running average of feedback; detect regime separately |
| Memory/persistence fails | Low | File-based logs as fallback; JSON per cycle |

---

## Integration with Phase 3.2.1

- **Input:** Phase 3.2.1 spot checks module (CostStressTest, RegimeShiftTest, ParamSweepTest)
- **Output:** Learning signal fed to Planner
- **Coupling:** Tight (evaluator results directly guide next proposal)
- **Decoupling:** Planner independent of evaluator implementation (API: pass/fail + reasoning)

---

## Integration with Future Phases

- **Phase 3.4:** Autonomous harness cycles (multi-market expansion)
- **Phase 4:** Deployment + monitoring (feedback from live trading)
- **Phase 5:** Multi-agent coordination (multiple planners competing)

---

## Sign-Off

**RESEARCH GATE:** ✅ **PASSED**
- [x] Searched: Autonomous agents, feedback loops, continuous learning, DRL trading
- [x] Collected: 20+ candidate URLs
- [x] Read: 5 sources in full (Switchfin, Quantified Strategies, Gemini, Columbia, OpenAI)
- [x] Extracted: Framework, thresholds, integration points
- [x] Success Criteria: All research-backed (loop cycle time, learning signal, regime adaptation)

**Ready for PLAN phase:** Yes ✅

