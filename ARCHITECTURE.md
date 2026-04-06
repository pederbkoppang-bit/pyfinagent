# pyfinAgent Architecture

Three layers, each with a distinct purpose and model provider.

## Layer 1: Analysis Pipeline (Gemini)

**File:** `backend/agents/orchestrator.py` (1477 lines)
**Models:** Gemini (Vertex AI)
**Purpose:** Per-ticker 15-step enrichment + synthesis

```
Screener → [15 enrichment agents in parallel] → Debate → Synthesis → Critic
```

Each ticker runs through: yfinance, alphavantage, FRED, SEC insider, options flow,
social sentiment, alt data, earnings tone, Monte Carlo, anomaly detection, NLP sentiment,
patent tracker, sector analysis, quant model, and macro analysis.

**Memory:** `FinancialSituationMemory` (BM25 situation→lesson retrieval in `memory.py`)
**Traces:** `DecisionTrace` + `AnalysisContext` (XAI audit trail in `trace.py`)

---

## Layer 2: MAS Orchestrator (Anthropic)

**File:** `backend/agents/multi_agent_orchestrator.py`
**Models:** Claude Opus 4.6 (Ford, Q&A), Claude Sonnet 4.6 (Communication, Researcher, Quality Gate, Citation)
**Purpose:** Slack/iMessage routing, research, Q&A — following Anthropic's multi-agent research pattern

**References:**
- https://www.anthropic.com/engineering/multi-agent-research-system
- https://www.anthropic.com/engineering/harness-design-long-running-apps

### Flow (from Anthropic's diagram)

```
User (Slack/iMessage)
    │
    ▼
Communication Agent (Sonnet 4.6) — classifies into 3 tiers
    │
    ├─ TRIVIAL → local response (no API)
    ├─ SIMPLE/MODERATE → Ford (Opus) solo with [DELEGATE:xxx] option
    └─ COMPLEX → Ford spawns subagents in parallel
                    │
                    ├─▶ Q&A Analyst (Opus) — with tool loop + thinking
                    └─▶ Researcher (Sonnet) — with tool loop + thinking
                    │
                    ▼
                Ford synthesizes → "More research needed?" loop (max 3)
                    │
                    ▼
                Quality Gate (Sonnet, 0.0-1.0 scoring rubric)
                    │
                    ▼
                CitationAgent (Sonnet, adds [1] [2] markers)
                    │
                    ▼
                Persist → Memory
```

### Key Design Decisions

| Principle | Implementation |
|-----------|---------------|
| Separation of generation & evaluation | Ford never self-evaluates; Quality Gate is a separate Sonnet instance |
| 4-component delegation | objective, output format, tool guidance, boundaries |
| Interleaved thinking | Subagents get `thinking={"type": "enabled", "budget_tokens": 2048}` |
| Observation masking | ACON-inspired compression at 60% context window in tool loops |
| Quality Gate rubric | 4 criteria (Accuracy, Completeness, Groundedness, Conciseness), 0.0-1.0, threshold < 0.6 |
| Few-shot calibration | 3 graded examples in Quality Gate prompt |
| Iterative research | "More research needed?" loop with MAX_RESEARCH_ITERATIONS=3 |
| File-based communication | handoff/ directory artifacts (contract, results, critique, plan, log) |

### Agent Inventory

| Agent | Model | Role | Spawned When |
|-------|-------|------|-------------|
| Communication | Sonnet 4.6 | Router/classifier + Quality Gate | Always (first step) |
| Ford (Main) | Opus 4.6 | Orchestrator, planner, synthesizer | Always (except TRIVIAL) |
| Q&A Analyst | Opus 4.6 | Quantitative analysis with harness tools | COMPLEX or [DELEGATE:qa] |
| Researcher | Sonnet 4.6 | Literature/evidence with harness tools | COMPLEX or [DELEGATE:research] |
| Quality Gate | Sonnet 4.6 | Skeptical reviewer (reuses Communication model) | All non-trivial responses |
| CitationAgent | Sonnet 4.6 | Adds source markers to Q&A/Research responses | Q&A/Research > 200 chars |

### Memory System

| Type | File | Purpose |
|------|------|---------|
| Working | (in context) | Current turn observations |
| Episodic | `memory/YYYY-MM-DD.md` via `harness_memory.py` | Daily logs, plan persistence |
| Semantic | `MEMORY.md` via `harness_memory.py` | Curated long-term facts |
| Procedural | `AGENTS.md`, `SOUL.md` | System prompts (loaded externally) |
| Observation Masking | `harness_memory.py` ObservationMasker | ACON-inspired compression at 60% |

### Harness Tools (7, read-only)

All subagents get these tools to ground analysis in real experiment data:

1. `read_evaluator_critique` — verdict, scores, weak periods
2. `read_experiment_results` — last cycle changes
3. `read_research_plan` — planner hypothesis
4. `read_experiment_log` — experiment summary
5. `read_best_params` — optimizer_best.json
6. `read_contract` — sprint contract
7. `read_harness_log` — cycle history

---

## Layer 3: Harness Loop (File-based)

**File:** `run_harness.py`
**Purpose:** Autonomous Planner → Generator → Evaluator backtest cycles

```
Planner (proposes strategy) → Generator (runs backtest) → Evaluator (grades result)
    ▲                                                              │
    └──────────── feedback loop (PASS/FAIL/CONDITIONAL) ───────────┘
```

**Communication:** File-based via `handoff/` directory:
- `contract.md` — sprint contract (negotiated before work)
- `experiment_results.md` — generator output
- `evaluator_critique.md` — evaluator verdict + scores
- `research_plan.md` — planner's next direction
- `harness_log.md` — cycle history

**Spot Checks** (Phase 3.2.1):
- `backend/backtest/spot_checks.py` — Cost stress (2×), Regime shift, Parameter sweep
- `backend/backtest/spot_checks_harness.py` — Harness integration entry point

---

## Deprecated Files (Phase 4 stubs)

| File | Status | Notes |
|------|--------|-------|
| `backend/agents/meta_coordinator.py` | DEPRECATED | Cross-loop sequencing, Phase 4 |
| `backend/autonomous_harness.py` | DEPRECATED | Self-driving loop skeleton, Phase 4 |
| `backend/services/autonomous_loop.py` | Active but uses deprecated MetaCoordinator | Paper trading daily cycle |

---

## Entry Points

| Surface | Entry | Routes To |
|---------|-------|-----------|
| Slack | `slack_bot/assistant_handler.py` → `get_orchestrator()` | Layer 2 (MAS) |
| API `/analyze` | `api/analysis.py` → `AnalysisOrchestrator` | Layer 1 (Pipeline) |
| API `/investigate` | `api/investigate.py` → `AnalysisOrchestrator` | Layer 1 (Pipeline) |
| Harness CLI | `run_harness.py` | Layer 3 (Harness) |
| Paper Trading | `services/autonomous_loop.py` | Layer 1 + MetaCoordinator |
