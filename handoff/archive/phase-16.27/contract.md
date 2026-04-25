---
step: phase-16.27
title: Trading-MAS benefit analysis design doc
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverable: docs/architecture/trading-mas-evaluation.md
---

# Sprint Contract -- phase-16.27

Research-only deliverable. No code. The output is a design doc grounded in 2024-2026 multi-agent trading literature + internal code audit.

## Research-gate summary

`handoff/current/phase-16.27-research-brief.md`. tier=simple, 6 in-full, 16 URLs, recency scan, gate_passed=true.

Key research findings shaping the doc:
- TradingAgents (arXiv 2412.20138) — 7-agent pattern; SR 8.21 on AAPL 6-mo bull (overfit risk, no DSR)
- HedgeAgents (ACM 2025) — +24.49% Sharpe over FinGPT baseline; +39.3% from memory wiring alone (biggest single lever)
- FinRL Contest 2025 — RL ensemble + LLM signal; ensemble outperforms individuals
- Public.com Agentic Brokerage — 2026 production, no published numbers
- LLM-as-Judge (Galileo) — multi-judge ensemble for risk/compliance, 23% lift, 80%+ human agreement
- Bailey & Lopez de Prado (2014) — DSR/PBO required for any reported SR claim

Internal-audit refinement: pyfinagent already has 2 of the 3 typical Beta agents (Analyst = Layer-1 synthesis; Risk Officer = Risk Judge). Only **Fund Manager** is new. Plug-in: `autonomous_loop.py:207-217`.

## Success criteria (verbatim, immutable)

```
test -f docs/architecture/trading-mas-evaluation.md && grep -qE 'Recommendation|Estimated benefit|Plug-in point' docs/architecture/trading-mas-evaluation.md
```

- design_doc_exists
- current_state_mapped
- options_compared
- plug_in_point_identified
- recommendation_present
- no_code_written

## Plan steps

1. Write `docs/architecture/trading-mas-evaluation.md` (12 sections, ~270 lines)
2. Verify file exists + grep for required keywords
3. Spawn Q/A
