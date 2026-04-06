# Phase 3.3.1 PLAN — Autonomous Loop Integration & Enhancement

**Date:** 2026-04-06 21:30 GMT+2  
**Gate:** Research ✅ COMPLETE (RESEARCH_3.3.1.md)  
**Target:** Replace mock components with real BacktestEngine, RESEARCH.md parsing, BigQuery logging  
**Scope:** 3 integration points across autonomous_loop.py, planner_enhanced.py, and new learning_logger.py  

---

## Overview

Phase 3.3 created the orchestrator + mock backtest engine. Phase 3.3.1 replaces mocks with **real components**:

1. **BacktestEngine integration** — `_generate_phase()` calls real backtest_engine.py
2. **RESEARCH.md parsing** — planner reads alpha source discoveries to inform proposals
3. **BigQuery logging** — harness_learning_log table captures iteration decisions + learnings

This is **strictly additive**: no refactoring, zero breaking changes. Mocks used when real components unavailable.

---

## Integration Points (3 files)

### 1. `backend/autonomous_loop.py` — _generate_phase() Integration

**Current state (mocks):**
- Lines 380-400: `_generate_phase()` calls mock backtest with synthetic results
- Returns fixed Sharpe: 1.1705 + random ±0.01 noise

**Target state (real):**
- Initialize `BacktestEngine` once per orchestrator init (line ~120 in __init__)
- Call `engine.run_backtest(proposal.tickers)` for top 2 proposals in parallel
- Use `asyncio.to_thread()` for sync→async wrapper
- Pre-load macro cache once (`cache.preload_macro()` in __init__)

**Code changes:**
```python
# In __init__:
self.backtest_engine = None
self.bq_client = None
self._init_backtest_engine()

def _init_backtest_engine(self):
    """Initialize BacktestEngine once, pre-load macro cache."""
    try:
        from google.cloud import bigquery
        from backend.backtest.backtest_engine import BacktestEngine
        from backend.backtest.cache import preload_macro
        
        self.bq_client = bigquery.Client(project=self.project_id)
        preload_macro()  # Critical: prevents ~40min hang
        
        self.backtest_engine = BacktestEngine(
            bq_client=self.bq_client,
            project=self.project_id,
            dataset="trading",
            market="US",
            start_date="2023-01-01",
            end_date="2025-12-31",
            transaction_cost_pct=0.1,  # Nominal
            max_positions=20,
            holding_days=90,
        )
    except Exception as e:
        logger.warning("BacktestEngine init failed: %s. Falling back to mocks.", e)
        self.backtest_engine = None

# In _generate_phase():
async def _generate_phase(self, proposals):
    if not self.backtest_engine:
        # Fallback to mock
        return await self._generate_phase_mock(proposals)
    
    # Real backtest: top 2 proposals in parallel
    tasks = []
    for proposal in proposals[:2]:
        task = asyncio.to_thread(
            self.backtest_engine.run_backtest,
            proposal.tickers
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Handle exceptions, convert to BacktestResult
    return results
```

**Risk:** LOW. BacktestEngine is thread-safe, pre-loaded macro is standard pattern.

---

### 2. `backend/agents/planner_enhanced.py` — RESEARCH.md Parsing

**Current state:**
- Lines ~50-80: Reads RESEARCH.md as raw text
- No structured extraction of alpha sources
- Proposals based on generic "explore X" templates

**Target state:**
- Parse RESEARCH.md for `## Alpha Source:` sections
- Extract: name, reference, url, key_finding, parameters, complexity, status
- Filter for "unexplored" or "low_complexity" sources
- Include in proposal as `research_sources: List[AlphaSource]`
- Use in prompt: "Based on recent research in {research_sources}, propose..."

**Code changes:**
```python
# New dataclass at top of file:
@dataclass
class AlphaSource:
    name: str
    reference: str
    url: str
    key_finding: str
    parameters: str
    complexity: str  # "low" | "medium" | "high"
    status: str      # "unexplored" | "partially_explored" | "explored"

# New function:
def parse_research_md(filepath: str) -> List[AlphaSource]:
    """Extract alpha sources from RESEARCH.md using regex patterns."""
    import re
    
    with open(filepath) as f:
        content = f.read()
    
    # Match ## Alpha Source: ... sections
    pattern = r"## Alpha Source:\s*(.+?)\n(.*?)(?=## Alpha Source:|$)"
    matches = re.findall(pattern, content, re.DOTALL)
    
    sources = []
    for name, section in matches:
        source = AlphaSource(
            name=name.strip(),
            reference=extract_field(section, "Reference"),
            url=extract_field(section, "URL"),
            key_finding=extract_field(section, "Key Finding"),
            parameters=extract_field(section, "Parameters"),
            complexity=extract_field(section, "Complexity", default="unknown"),
            status=extract_field(section, "Status", default="unexplored"),
        )
        sources.append(source)
    
    return sources

def extract_field(section: str, field_name: str, default="") -> str:
    """Extract a field from markdown section."""
    pattern = rf"\*\*{field_name}:\*\*\s*(.+?)(?:\n\*\*|\n-|\Z)"
    match = re.search(pattern, section, re.DOTALL)
    return match.group(1).strip() if match else default

# In enhanced_planner_agent():
def generate_proposals(...):
    # Parse research
    try:
        research_sources = parse_research_md("RESEARCH.md")
        unexplored = [s for s in research_sources if s.status == "unexplored"]
    except Exception as e:
        logger.warning("Failed to parse RESEARCH.md: %s", e)
        unexplored = []
    
    # Build prompt with research context
    research_context = "\n".join([
        f"- {s.name}: {s.key_finding} (URL: {s.url}, Complexity: {s.complexity})"
        for s in unexplored[:3]  # Top 3 unexplored
    ])
    
    prompt = f"""
    Based on recent research discoveries:
    {research_context}
    
    Generate ranked proposals for backtest experiments...
    """
    # ... rest of prompt
```

**Risk:** LOW. Regex parsing is defensive (exceptions gracefully fall back to generic proposals).

---

### 3. `backend/backtest/learning_logger.py` (NEW FILE) — BQ Logging

**Current state:**
- No learning logging
- Harness iterations disappear after completion

**Target state:**
- New module: `learning_logger.py` (60 lines)
- Dataclass: `IterationLog` with timestamp, iteration_id, proposal_id, verdict, sharpe_delta, key_findings, status
- Function: `log_iteration_to_bq(log: IterationLog) → bool`
- Fallback: JSON to `handoff/iteration_*.json` if BQ unavailable

**Code changes:**
```python
# backend/backtest/learning_logger.py

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class IterationLog:
    """Schema for harness_learning_log table."""
    timestamp: str  # ISO8601
    iteration_id: str  # UUID
    cycle_number: int
    proposal_id: str
    proposal_ranking: int  # 1, 2, 3...
    evaluator_verdict: str  # "PASS" | "FAIL" | "CONDITIONAL"
    sharpe_baseline: float
    sharpe_tested: float
    sharpe_delta: float
    dsr_baseline: float
    dsr_tested: float
    key_findings: str  # JSON or text summary
    next_action: str  # "proceed" | "revert" | "fix_and_retry"
    status: str  # "logged" | "error"

def log_iteration_to_bq(
    project_id: str,
    log: IterationLog
) -> bool:
    """Log iteration to BigQuery, fallback to JSON file if unavailable."""
    try:
        from google.cloud import bigquery
        
        bq = bigquery.Client(project=project_id)
        table_id = f"{project_id}.trading.harness_learning_log"
        
        rows = [asdict(log)]
        errors = bq.insert_rows_json(table_id, rows, timeout=10)
        
        if errors:
            logger.error("BQ insert errors: %s", errors)
            return False
        
        logger.info("Logged iteration %s to BQ", log.iteration_id)
        return True
    
    except Exception as e:
        logger.warning("BQ logging failed: %s. Falling back to file.", e)
        
        # Fallback: JSON file
        try:
            filename = f"handoff/iteration_{log.iteration_id}.json"
            with open(filename, "w") as f:
                json.dump(asdict(log), f, indent=2)
            logger.info("Logged iteration %s to %s", log.iteration_id, filename)
            return True
        except Exception as e2:
            logger.error("File logging also failed: %s", e2)
            return False

def create_bq_table(project_id: str, dataset_id: str) -> None:
    """Create harness_learning_log table if not exists (idempotent)."""
    from google.cloud import bigquery
    from google.cloud.bigquery import SchemaField
    
    bq = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_id}.harness_learning_log"
    
    schema = [
        SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        SchemaField("iteration_id", "STRING", mode="REQUIRED"),
        SchemaField("cycle_number", "INTEGER", mode="REQUIRED"),
        SchemaField("proposal_id", "STRING", mode="REQUIRED"),
        SchemaField("proposal_ranking", "INTEGER", mode="REQUIRED"),
        SchemaField("evaluator_verdict", "STRING", mode="REQUIRED"),
        SchemaField("sharpe_baseline", "FLOAT64", mode="REQUIRED"),
        SchemaField("sharpe_tested", "FLOAT64", mode="REQUIRED"),
        SchemaField("sharpe_delta", "FLOAT64", mode="REQUIRED"),
        SchemaField("dsr_baseline", "FLOAT64", mode="REQUIRED"),
        SchemaField("dsr_tested", "FLOAT64", mode="REQUIRED"),
        SchemaField("key_findings", "STRING", mode="NULLABLE"),
        SchemaField("next_action", "STRING", mode="REQUIRED"),
        SchemaField("status", "STRING", mode="REQUIRED"),
    ]
    
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = ["evaluator_verdict", "iteration_id"]
    
    try:
        bq.create_table(table)
        logger.info("Created table %s", table_id)
    except Exception as e:
        logger.info("Table %s already exists or creation failed: %s", table_id, e)
```

**Integration into autonomous_loop.py:**
```python
# In evaluate_and_decide():
from backend.backtest.learning_logger import log_iteration_to_bq, IterationLog

# After evaluator verdict
log = IterationLog(
    timestamp=datetime.utcnow().isoformat() + "Z",
    iteration_id=str(self.iteration_id),
    cycle_number=self.cycle_number,
    proposal_id=top_proposal.id,
    proposal_ranking=1,
    evaluator_verdict=verdict,
    sharpe_baseline=self.baseline_sharpe,
    sharpe_tested=top_result.sharpe,
    sharpe_delta=top_result.sharpe - self.baseline_sharpe,
    dsr_baseline=self.baseline_dsr,
    dsr_tested=top_result.dsr,
    key_findings=json.dumps({
        "proposal": top_proposal.title,
        "evaluator_notes": evaluator_response.get("notes", ""),
    }),
    next_action=decision_action,
    status="logged",
)
log_iteration_to_bq(self.project_id, log)
```

**Risk:** LOW. Graceful fallback to JSON files. Standard BQ patterns.

---

## Files to Modify / Create

| File | Type | Lines | Changes |
|------|------|-------|---------|
| `backend/autonomous_loop.py` | Modify | ~630 → ~700 | BacktestEngine init + integration in _generate_phase |
| `backend/agents/planner_enhanced.py` | Modify | ~400 → ~500 | RESEARCH.md parsing + AlphaSource dataclass |
| `backend/backtest/learning_logger.py` | Create | ~150 | New module for BQ + file logging |
| `backend/agents/__init__.py` | Modify | ~5 | Export AlphaSource, parse_research_md |
| `handoff/phase3.3.1_plan.md` | Create | ~200 | This document |

---

## Implementation Steps

### Step 1: Create learning_logger.py (10 min)
- Write new module with dataclass + logging functions
- Test imports

### Step 2: Update autonomous_loop.py (20 min)
- Add BacktestEngine init to __init__
- Replace mock _generate_phase with real integration
- Add learning log calls to evaluate_and_decide

### Step 3: Update planner_enhanced.py (15 min)
- Add AlphaSource dataclass
- Add parse_research_md + extract_field functions
- Update generate_proposals to use research context

### Step 4: Test integration (20 min)
- Run 1-cycle orchestrator with real BacktestEngine
- Verify BQ logging
- Verify RESEARCH.md parsing

### Step 5: Evaluate + Log (10 min)
- Check Sharpe improvement vs phase 3.3 (1.1705)
- Verify all 3 components working
- Update memory/HEARTBEAT

---

## Success Criteria (from CONTRACT)

✅ **BacktestEngine integration:**
- [ ] run_backtest() called for ≥1 proposal per cycle
- [ ] Results match BacktestResult schema (sharpe, dsr, return_pct, max_dd)
- [ ] Macro cache preloaded (no ~40min hang)

✅ **RESEARCH.md parsing:**
- [ ] Parse 3+ alpha sources from RESEARCH.md
- [ ] Proposals reference research findings
- [ ] Graceful fallback if parsing fails

✅ **BigQuery logging:**
- [ ] ≥1 iteration log inserted per cycle
- [ ] Fallback to JSON if BQ unavailable
- [ ] Schema matches harness_learning_log definition

✅ **Regression test:**
- [ ] Sharpe ≥ 1.1705 (no degradation from phase 3.3)
- [ ] Loop completes in <30 min
- [ ] All 3 cycles produce valid proposals + verdicts

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| BacktestEngine crashes | Catch exceptions, fallback to mock results + log to incidents |
| BQ unavailable | Fallback to JSON file logging + warning in logs |
| RESEARCH.md parsing fails | Graceful exception, fall back to generic proposals |
| ~40min hang (macro cache) | Preload once in __init__, not per backtest |
| Thread safety | BacktestEngine already thread-safe, confirmed in audit |

---

## Rollback Plan

If integration fails:
1. Revert last 3 commits
2. Phase 3.3 mocks still work (self-contained)
3. Master plan continues with Phase 3.4

---

## Estimated Time

- Coding: 65 min (4 steps)
- Testing: 20 min (1-cycle run)
- **Total: ~85 min** (1.5 hours)

Target completion: 2026-04-06 23:00 UTC (1.5 hours from start)

---

## Commit Plan

1. `feature: learning_logger module (BQ + file fallback)`
2. `feature: BacktestEngine integration in autonomous_loop`
3. `feature: RESEARCH.md parsing in planner_enhanced`
4. `test: Phase 3.3.1 integration 1-cycle pass`
5. `🔄 HEARTBEAT: Phase 3.3.1 GENERATE ✅, EVALUATE ⏳`

---

**Next step:** GENERATE phase starts immediately. Code changes begin in ~5 min.
