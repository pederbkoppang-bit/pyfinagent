# Phase 2.12: Harness Optimization — PLAN Contract

**Date:** 2026-04-02 06:35 UTC  
**Status:** DRAFT (awaiting Peder approval to execute)  
**Research:** ✅ COMPLETE (24KB findings in RESEARCH.md)

---

## Problem Statement

Backtest harness currently:
- **Cost:** Full Claude API calls for each optimization cycle (Anthropic SDK)
- **Speed:** 40-50 min per 5-seed evaluation cycle
- **Efficiency:** Identical prompt calculations repeated across seeds
- **ROI:** No cost optimization pathway before go-live

**Goal:** Reduce LLM costs by 90% while maintaining Sharpe robustness (σ≤0.99%)

---

## Success Criteria (Research-Backed)

### Primary Metric
- **Prompt Cache Hit Rate:** ≥85% (reduces redundant prompt processing by ~90%)
- **Cost Reduction:** ≥85% vs baseline (verified via cost tracking)
- **Execution Time:** ≤20 min per 5-seed cycle (vs current 40-50 min)

### Secondary Metrics
- **Sharpe Robustness:** Std dev ≤0.99% (Phase 2.8 baseline: σ=0.99%)
- **Return Consistency:** Std dev ≤1.0% across seed runs (Phase 2.8 baseline: σ=1.0%)
- **Max Drawdown:** Std dev ≤0.05% (Phase 2.8 baseline: σ=0.05%)

### Failure Criteria
- Prompt cache hit rate <50% → redesign required
- Sharpe robustness σ >1.2% → regress to Phase 2.8 parameters
- Cost reduction <50% → implementation incomplete

---

## Design Overview

### Tier 1: Prompt Caching (Anthropic SDK native)
**Implementation:** Cache backtest parameters, evaluation criteria, and success thresholds at the Claude API level
- **Cache Key:** SHA256(backtest_ticker + eval_period + optimizer_params)
- **TTL:** 5-10 minutes (suitable for sequential seed tests)
- **Expected Savings:** 85% of prompt processing cost (per Anthropic documentation)

### Tier 2: Memory System (4-tier hierarchy)
1. **Seed Memory:** Per-run LLM state (synthesis, critique, revisions) — cache in local dict
2. **Batch Memory:** Cross-seed parameters learned (momentum thresholds, ML model tune) — persist to `.json`
3. **Strategy Memory:** Long-term learnings from all phases (effective lookback, sector filters) — load on startup
4. **Macro Memory:** Market regime indicators (VIX, yield curve) — static, pre-cached

**Implementation:** Replace raw BQ agent queries with pre-filtered, in-memory lookups (90%+ speed improvement)

### Tier 3: Batch Optimization
**Implementation:** Group 5 seeds into single Claude request (prompt once per batch vs. once per seed)
- **Single Prompt:** "Here are 5 seed configurations. Synthesize & critique all together."
- **Response:** Batch critique with differentiated insights per seed
- **Cost:** 5 requests → 1 request (80% reduction alone)

### Tier 4: Incremental Evaluation
**Implementation:** Cache intermediate results and only re-evaluate changed parameters
- **Delta Tracking:** Monitor which hyperparams changed since last cycle
- **Partial Re-run:** Only evaluate seeds affected by param changes
- **Cost:** Estimated 60% reduction for minor tuning iterations

---

## Implementation Roadmap

### Phase 2.12A: Prompt Caching Setup (1-2h)
1. Initialize Claude client with `cache_control={"type": "ephemeral"}` (per Anthropic spec)
2. Wrap backtest execution with cache-aware request packaging
3. Add cost tracking instrumentation (cost before vs. after prompt caching)
4. Verify cache hit rate in logs

### Phase 2.12B: Memory System (2-3h)
1. Implement Tier 1 (seed dict) + Tier 2 (`.json` persistence)
2. Load Tier 3 (strategy memory) and Tier 4 (macro cache) from Phase 2.9
3. Replace BQ agent queries with memory lookups
4. Benchmark: Compare performance (speed + accuracy) vs. full BQ

### Phase 2.12C: Batch Optimization (1-2h)
1. Refactor optimizer to group seeds into single Claude request
2. Test batch synthesis + critique logic
3. Verify all 5 seeds evaluated with full context (not individually)

### Phase 2.12D: Evaluation (2-3h)
1. Run 5-cycle evaluation (25 seeds total) with cost + speed tracking
2. Verify Sharpe robustness within ±0.2% of Phase 2.8 baseline (1.1705)
3. Document cost savings (target: 85% reduction)
4. A/B test: Full prompt caching vs. no caching to isolate savings

---

## Metrics & Monitoring

### Cost Tracking
```
Baseline (Phase 2.8): ~$X per 5-seed cycle
Phase 2.12 Target: <$X * 0.15 per 5-seed cycle
Verification: Cost logs + Claude API usage dashboard
```

### Performance Tracking
- **Sharpe:** Mean ± σ for each seed run
- **Return:** Mean ± σ for each seed run
- **Execution Time:** Wall time per cycle (target: <20 min)
- **Cache Hit Rate:** Log percentage after each request

### Success Definition
**PASS:** Sharpe σ ≤0.99% AND cost reduction ≥85% AND time ≤20 min  
**CONDITIONAL:** If any metric marginal, run Phase 2.12D+ (refinement)  
**FAIL:** Revert to Phase 2.8, document why caching failed

---

## Rollback Plan

If Phase 2.12 underperforms:
1. Disable prompt caching (1 line config change)
2. Revert to Phase 2.8 backtest harness (git revert or branch restore)
3. Document failure mode → update RESEARCH.md with findings
4. Log incident to memory/incidents.md

---

## Research Justification

**Prompt Caching:** Anthropic Claude API native feature, documented 85%+ token savings for repeated prompts (per Anthropic API docs, Section 5.2)

**Memory System:** Follows phase 2.9 design (Strategy Memory layer) + established caching patterns in pyfinAgent (orchestrator agent memory BM25 lookups)

**Batch Optimization:** Reduces sequential API calls; follows multi-agent synthesis pattern from Phase 3.2.1 (real agent spawning in queue processor)

---

## Owner & Approval

**Created By:** Ford  
**Approval Required:** Peder B. Koppang  
**Approval Status:** ⏳ Pending  
**Timeline:** Ready to execute immediately upon approval
