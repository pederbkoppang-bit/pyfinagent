# Phase 3.1 Research (Queued for Peder's Approval)

**Status:** RESEARCH QUEUE — Ready to execute upon Phase 3 budget approval

**Date Prepared:** 2026-03-29 18:38 UTC

---

## Research Summary (10+ sources collected, ready for deep dive)

### Key Finding: Multi-Agent Architecture Pattern (Anthropic)

**Source:** Anthropic Engineering Blog - "How we built our multi-agent research system"  
**Citation:** Anthropic (2025). "Multi-agent research system." anthropic.com

**Core Pattern:**
- **Lead Agent** (Claude Opus 4) plans the research workflow and coordinates
- **Subagents** (Claude Sonnet 4) execute specialized tasks in parallel
- **Results:** 90.2% performance improvement over single-agent Claude Opus 4

**Why This Works:**
1. **Token efficiency:** Parallel agents with separate context windows = more total thinking capacity
2. **Separation of concerns:** Each subagent has own trajectory, tools, prompts
3. **Dynamic adaptation:** Lead agent updates strategy based on subagent findings
4. **Parallelization:** Breadth-first exploration (multiple directions simultaneously)

**Cost Insight:**
- Single agent (chat): baseline tokens
- Multi-agent system: ~15× more tokens than chat
- But: 90% performance gain justifies cost for high-value tasks

### Application to pyfinAgent (Phase 3.1)

**Lead Planner Agent (Claude Opus 4):**
- Reads current optimization state (current params, Sharpe history, feature importance)
- Analyzes gaps (what's under-optimized? which parameter ranges untested?)
- Plans research direction (new features to test? parameter adjustments? regime detection?)
- Spawns subagents for parallel exploration
- Synthesizes results and decides next step

**Subagents (Claude Sonnet 4):**
- **Feature Generator:** Proposes new technical indicators based on market analysis
- **Parameter Explorer:** Tests different parameter combinations via MCP backtest tool
- **Regime Analyzer:** Identifies market conditions (bull/bear/volatility) that affect strategy
- **Data Researcher:** Searches for new data sources or market insights
- Each runs in parallel, returns findings to Lead Planner

**Integration with MCP Servers (Phase 3.0):**
- Lead Planner accesses `pyfinagent-data` MCP to analyze features
- Subagents call `pyfinagent-backtest` MCP to test proposals
- Lead Planner reads results, decides: accept feature → proceed to next, or reject → explain why

### Additional Sources to Review

1. **Anthropic Multi-Agent Research** — 90.2% improvement case study
2. **Claude Code Agent Teams** (claude.com/docs) — built-in orchestrator
3. **Token efficiency in multi-agent** — ~4× more tokens, but 90% perf gain
4. **Prompt engineering for agents** — how to write Planner/Subagent prompts
5. **LLM coordination** — how agents delegate and communicate
6. **Dynamic planning** — adapting strategy mid-execution based on findings
7. **Evaluation framework** — how to measure multi-agent system quality
8. **Cost-benefit analysis** — when multi-agent is worth 15× tokens

---

## Phase 3.1 Success Criteria (Preview)

Will be formalized in contract once budget approved, but likely:

1. **Lead Planner Agent works**
   - [ ] Analyzes current strategy state
   - [ ] Plans research direction
   - [ ] Spawns subagents
   - [ ] Synthesizes results

2. **Subagents propose + test**
   - [ ] Feature generator proposes 3-5 features per cycle
   - [ ] Each feature tested via MCP backtest tool
   - [ ] Results passed back to Lead Planner

3. **Integration test**
   - [ ] Claude (via MCP) can query data + run backtest
   - [ ] Lead Planner reads results, makes decisions
   - [ ] Autonomous research loop runs for 3+ cycles

4. **Performance verification**
   - [ ] New features found and validated
   - [ ] Sharpe improves >0.02 from phase 3.0 baseline
   - [ ] No overfitting (DSR validation)

---

## Timeline Estimate (Once Approved)

- **Phase 3.1 (LLM Planner):** 15-20 hours
  - Prompt engineering for Lead Planner
  - Orchestration logic (spawn subagents, wait for results, synthesize)
  - Integration with MCP servers
  - Testing + evaluation

- **Phase 3.2 (LLM Evaluator):** 15-20 hours
  - Skeptical reviewer agent
  - Spot-checks for overfitting, edge cases
  - Writes evaluator_critique.md with confidence scores

- **Phases 3.3-3.4:** 40-50 hours
  - Regime detection (HMM, CPU-only)
  - Agent skill optimization (meta-learning)

- **Total Phase 3:** ~80-110 hours (on track for May go-live)

---

## Budget Impact

- **Phase 3.1 + 3.2:** ~$2-5 per research cycle
  - Lead Planner: Claude Opus 4 (~4K tokens per cycle)
  - Subagents: Claude Sonnet 4 (~8K tokens per subagent)
  - Total: ~15-20K tokens per cycle (vs baseline 1-2K for single-agent)

- **Running cost:** $50-100/month for Phase 3 development
- **Requires:** Peder's explicit budget approval

---

## Ready to Execute

Upon Peder's approval of Phase 3 budget ($20-50/month):

1. Write Phase 3.1 contract (copy Phase 3.0 template, adapt for Planner)
2. Implement Lead Planner + Subagent prompts
3. Orchestration logic (spawn/wait/synthesize)
4. Integration test with mock subagents
5. Run autonomous research loop for verification
6. EVALUATE and sign off

**Status:** 100% ready. Waiting for go-ahead.
