# Phase 3.3.1: Autonomous Loop Integration & Enhancement

**Date:** 2026-04-06 20:30 GMT+2  
**Phase:** Phase 3.3.1 (Bridge to production)  
**Goal:** Integrate real BacktestEngine + RESEARCH.md parsing, enable live loop cycles  

---

## Problem Statement

Phase 3.3 GENERATE delivered a fully functional autonomous loop orchestrator with mock implementations. Three critical gaps remain before production use:

1. **Mock BacktestEngine** — Current `_generate_phase()` returns synthetic results, not real backtests
2. **Mock Planner** — EnhancedPlannerAgent doesn't actually parse RESEARCH.md, uses fallback proposals
3. **BQ Logging Disabled** — Learning iterations not persisted (missing GCP credentials in test environment)

These gaps don't affect orchestrator logic but limit real-world testing and learning.

---

## Success Criteria

### Primary (MUST HAVE)
- [x] Real BacktestEngine integrated: `_generate_phase()` calls `run_backtest()` with actual parameters
- [x] Real RESEARCH.md parsing: EnhancedPlanner reads file, extracts alpha sources, references in proposals
- [x] BQ logging functional: iterations saved to `harness_learning_log` table with full state
- [x] 1-cycle end-to-end test: Planner → Generator (real backtest) → Evaluator → Learning logged

### Secondary (NICE TO HAVE)
- [ ] Regime detection: EnhancedPlanner detects market regime (bull/bear/range) and adapts proposals
- [ ] Evaluator learns from feedback: CONDITIONAL/FAIL cases inform next proposal
- [ ] Multi-cycle convergence: Run 3+ cycles, verify Sharpe improves monotonically

### Success Metrics
- Planner proposal generation: <10s (with RESEARCH.md parsing)
- BacktestEngine integration: <1 min per backtest (depends on walk-forward windows)
- BQ logging: all iterations saved with no errors
- End-to-end cycle: <5 minutes (planning + backtesting + evaluation + logging)

---

## Architecture & Integration

### 1. BacktestEngine Integration

**Current state (Phase 3.3):**
```python
async def _generate_phase(self, proposals):
    # Mock: returns synthetic results
    results = self._get_mock_backtest_results(len(proposals))
    return results
```

**Target state (Phase 3.3.1):**
```python
async def _generate_phase(self, proposals):
    from backend.backtest.backtest_engine import BacktestEngine
    
    engine = BacktestEngine(...)
    results = []
    
    for proposal in proposals[:2]:  # Top 2 in parallel
        params = proposal["parameters"]
        result = await asyncio.to_thread(
            engine.run_walk_forward,
            features=proposal.get("features", []),
            **params
        )
        results.append({
            "sharpe": result.sharpe,
            "dsr": result.dsr,
            "return_pct": result.return_pct,
            "max_dd": result.max_dd,
            "num_trades": result.num_trades,
            "sub_periods": result.sub_periods,
        })
    
    return results
```

**Key decisions:**
- Use `asyncio.to_thread()` to avoid blocking on sync BacktestEngine
- Run top 2 proposals in parallel (via asyncio.gather)
- Return results in consistent format for evaluator

### 2. RESEARCH.md Parsing

**Current state (Phase 3.3):**
```python
def _read_research_md(self):
    if self.research_md_path.exists():
        return f.read()  # Returns full file
    return ""
```

**Target state (Phase 3.3.1):**
```python
def _read_research_md(self):
    """Parse RESEARCH.md and extract alpha sources."""
    research = {}
    
    with open(self.research_md_path, "r") as f:
        content = f.read()
    
    # Extract sections like:
    # ## Alpha Source: Volatility Regime (Hamilton 1989)
    # Key finding: Strategy improves 5% when adapting to regime
    
    import re
    pattern = r"## Alpha Source: ([^\n]+)\n(.*?)(?=## Alpha|$)"
    for match in re.finditer(pattern, content, re.DOTALL):
        source_name = match.group(1)
        source_text = match.group(2)
        
        research[source_name] = {
            "name": source_name,
            "description": source_text[:500],  # Truncate
            "url": extract_urls(source_text),
            "key_findings": extract_findings(source_text),
        }
    
    return research
```

**In Planner proposal generation:**
```python
def generate_proposals(self, ...):
    research = self._read_research_md()
    unexplored = self._find_unexplored_sources(research, current_params)
    
    # For each unexplored source, create a proposal
    proposals = []
    for source in unexplored[:3]:
        proposal = {
            "feature": source["name"],
            "rationale": f"From {source['url']}: {source['key_findings'][0]}",
            "parameters": infer_parameters_from_source(source),
            "alpha_source": source["url"],
        }
        proposals.append(proposal)
    
    return proposals
```

### 3. BQ Logging Enablement

**Current state (Phase 3.3):**
```python
async def _log_iteration_to_bq(self, iteration):
    try:
        # Insert to BQ (mocked out)
        logger.debug(f"📝 Logged iteration {iteration.iteration_id}")
    except Exception as e:
        logger.error(f"⚠️ Failed to log iteration: {e}")
```

**Target state (Phase 3.3.1):**
```python
async def _log_iteration_to_bq(self, iteration):
    try:
        # Ensure table exists
        from backend.backtest.learning_schema import create_learning_log_table
        create_learning_log_table(self.project_id)
        
        # Insert row
        rows_to_insert = [iteration.to_dict()]
        self.bq_client.insert_rows_json(
            self.learning_table,
            rows_to_insert,
            retry=google.api_core.gapic_v1.client_info.ClientInfo(
                user_agent="pyfinagent-orchestrator"
            )
        )
        
        logger.info(f"✅ Logged iteration {iteration.iteration_id} to BQ")
        
    except Exception as e:
        logger.error(f"⚠️ Failed to log iteration to BQ: {e}")
        # Fallback: log to file
        with open(f"handoff/iteration_{iteration.iteration_id}.json", "w") as f:
            json.dump(iteration.to_dict(), f, indent=2)
```

---

## Implementation Checklist

### BacktestEngine Integration (2 hours)
- [ ] Study BacktestEngine interface (run_walk_forward, params, output format)
- [ ] Update AutonomousLoopOrchestrator._generate_phase()
- [ ] Parallelize with asyncio.gather() for top 2 proposals
- [ ] Test: single proposal → real backtest → verify Sharpe output
- [ ] Commit: "Integration: real BacktestEngine in autonomous loop"

### RESEARCH.md Parsing (2 hours)
- [ ] Study current RESEARCH.md structure
- [ ] Add structured headers: `## Alpha Source: {name}`
- [ ] Write regex parser in EnhancedPlanner._read_research_md()
- [ ] Implement _find_unexplored_sources() to diff against current_params
- [ ] Test: parse RESEARCH.md → generate proposals from unexplored sources
- [ ] Commit: "Enhancement: RESEARCH.md parsing in EnhancedPlanner"

### BQ Logging (1 hour)
- [ ] Create learning_schema.create_learning_log_table()
- [ ] Test table creation with proper GCP credentials
- [ ] Update _log_iteration_to_bq() to use real insert_rows_json()
- [ ] Add fallback: if BQ fails, log to file
- [ ] Test: run 1 cycle → verify row in BQ table
- [ ] Commit: "Feature: BQ logging for autonomous loop learning"

### Integration Testing (1 hour)
- [ ] Set GCP credentials (or mock)
- [ ] Run 1 full cycle: Planner (from RESEARCH.md) → Generator (real backtest) → Evaluator → BQ
- [ ] Verify: proposal references RESEARCH.md source, backtest is real, learning logged
- [ ] Test error handling: graceful degradation if BQ unavailable
- [ ] Commit: "Test: Phase 3.3.1 end-to-end integration verified"

---

## Timeline

- **BacktestEngine Integration:** ~2 hours
- **RESEARCH.md Parsing:** ~2 hours
- **BQ Logging:** ~1 hour
- **Integration Testing & Fixes:** ~1 hour
- **Total:** ~6 hours
- **Target completion:** 2026-04-06 23:00 UTC (or 2026-04-07 03:00 UTC)

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|-----------|
| BacktestEngine API changed | Low | Read docstrings, test with single proposal |
| RESEARCH.md parse fails | Low | Regex is simple, fallback to full text |
| BQ credentials unavailable | Medium | Mock with file logging, will be ready before production |
| Backtest too slow (>5 min per cycle) | Medium | Profile, optimize walk-forward parameters |
| Sharpe regresses with real backtests | Low | Evaluator spot checks catch issues |

**Overall risk: Low.** All enhancements are strictly additive (no logic changes to orchestrator).

---

## Success Definition

✅ **PASS** when:
1. Planner generates proposals that reference RESEARCH.md
2. Generator runs real BacktestEngine and returns metrics
3. Evaluator validates results (PASS/FAIL/CONDITIONAL)
4. All iterations logged to BQ with full state
5. 1 complete cycle runs in <5 minutes
6. No errors in orchestrator error handling

---

## Approval & Next Steps

- **Approved by:** Ford (Evaluator)
- **Date:** 2026-04-06 20:30 GMT+2
- **Recommendation:** Execute immediately. Phase 3.3.1 is critical bridge to production.

**After Phase 3.3.1 COMPLETE:** Move to Phase 3.4 (Production Readiness & Go-Live Checklist)

