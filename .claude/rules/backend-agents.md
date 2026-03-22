---
paths:
  - "backend/agents/**"
---

# Backend Agents — AI Pipeline Conventions

## Agent Architecture
- 20+ specialized agents orchestrated by `orchestrator.py` through a 15-step pipeline
- All agents use Gemini 2.0 Flash via `LLMClient` abstraction (`llm_client.py`)
- Multi-provider support: Gemini, GitHub Models, Anthropic, OpenAI — routed by `make_client()`
- RAG (Step 3) and Google Search Grounding (Steps 4/5/9/10) always use Gemini regardless of selected model

## Key Modules
- `orchestrator.py` — Main pipeline (15 steps), manages `AnalysisContext` session memory
- `debate.py` — Multi-round Bull↔Bear + Devil's Advocate + Moderator (TradingAgents pattern)
- `risk_debate.py` — Aggressive/Conservative/Neutral analysts + Risk Judge
- `schemas.py` — Pydantic output schemas for Gemini structured output
- `trace.py` — DecisionTrace dataclass (agent name, timestamp, signal, confidence, evidence, reasoning)
- `bias_detector.py` — Tech bias, confirmation bias, recency bias detection
- `conflict_detector.py` — Parametric vs real-time knowledge conflict flagging
- `cost_tracker.py` — Per-agent token/cost tracking (28 models across 4 providers)
- `memory.py` — BM25-based FinancialSituationMemory (learns from outcomes)
- `compaction.py` — Deterministic compact-state helpers for constrained-context models
- `skill_optimizer.py` — Autoresearch-style prompt optimization loop
- `meta_coordinator.py` — Cross-loop sequencing (QuantOpt → SkillOpt → PerfOpt)

## Skills System
- Agent prompts in `skills/*.md`, loaded via `load_skill()` + `format_skill()` with `{{variable}}` placeholders
- Skill cache keyed by file modification time — edits auto-picked up
- SkillOptimizer modifiable sections: `## Prompt Template`, `## Skills & Techniques`, `## Anti-Patterns`
- Fixed harness (UNTOUCHABLE): data tools, orchestrator pipeline, output schemas, BQ schema, evaluation formula

## Design Patterns
- **Reflection Loop**: Synthesis↔Critic, max 2 iterations (`MAX_SYNTHESIS_ITERATIONS`)
- **Quality Gates**: Data quality threshold skips debate/risk when insufficient
- **Sector Routing**: Sector-aware tool skipping (e.g., patents skipped for Financial Services)
- **Glass Box**: Every agent I/O visible in UI. DecisionTrace for full audit trail.

## Cost Controls
- Output token limits: Enrichment 1024, Debate 1536, Moderator 2048, Synthesis 4096
- Lite Mode: ~39 → ~20 LLM calls (skips Deep Dive, DA, Risk Assessment, limits debate rounds)
- Prompt truncation: enrichment sections capped at 1,500 chars, market context at 12,000 chars
- Extended thinking: opt-in via `ENABLE_THINKING=true`, tiered budgets (Critic 8192, Synthesis 4096)
