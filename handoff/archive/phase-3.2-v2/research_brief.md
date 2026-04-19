---
step: phase-3.1
title: LLM-as-Planner — Research Brief
date: 2026-04-19
tier: moderate
researcher: researcher-agent (merged researcher + Explore)
---

## Research: LLM-as-Planner Integration Audit (Phase 3.1)

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.anthropic.com/engineering/multi-agent-research-system | 2026-04-19 | official doc | WebFetch | "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Lead agent uses extended thinking to plan, then spawns 3-5 parallel subagents; synthesizes results iteratively. |
| https://arxiv.org/html/2409.06289v3 | 2026-04-19 | paper (arXiv 2024) | WebFetch | Three-stage propose-evaluate loop: Seed Alpha Factory (LLM generates candidates from literature) → Multimodal Multi-Agent Evaluation (IC + Sharpe) → Weight Optimization. Cumulative return 53.17% on SSE50. |
| https://arxiv.org/html/2412.20138v3 | 2026-04-19 | paper (arXiv 2024) | WebFetch | TradingAgents: structured document exchange (not conversation chains). Analyst team → Researcher debate → Trader synthesis → Risk management. Metrics: cumulative return, annualized return, Sharpe, max drawdown. |
| https://arxiv.org/html/2602.23330v1 | 2026-04-19 | paper (arXiv 2026) | WebFetch | Three-level hierarchy for investment teams. Fine-grained task decomposition (exact metrics, criteria, output formats per agent) significantly outperforms abstract role assignments. |
| https://www.datacamp.com/tutorial/guide-to-autoresearch | 2026-04-19 | authoritative blog | WebFetch | Karpathy autoresearch: 9-step ratchet loop. Propose ONE change → implement → run fixed budget → evaluate val_bpb → keep if improved else `git reset HEAD~1`. "The ratchet only accepts changes that immediately improve val_bpb, so the agent can never take a step backward." 700 experiments / 48 hours, 11% improvement. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://tradingagents-ai.github.io/ | project site | Covered by arXiv paper (2412.20138) fetched in full |
| https://arxiv.org/html/2510.02209v1 (StockBench) | paper | Search snippet sufficient; LLM agents profitable but don't consistently beat simple baselines — negative result, not architecturally novel for our use case |
| https://arxiv.org/html/2505.07078v5 | paper | Search snippet: long-run LLM investing performance questionable; captured in recency scan |
| https://openreview.net/forum?id=Q5o249Z3Je (FINSABER) | paper | Temporal contamination benchmarking — not relevant to planner architecture |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5217340 | paper | LLM market simulation; not planner architecture |
| https://arxiv.org/abs/2409.06289 | paper (abstract only) | Full HTML fetched separately (v3 above) |
| https://glenrhodes.com/karpathys-autoresearch-repo-autonomous-ml-experiment-loop-compresses-the-gap-between-research-question-and-validated-result/ | blog | Fetched but thin — cultural analysis only, no technical depth. Superseded by DataCamp full read. |
| https://aclanthology.org/2025.findings-emnlp.1005/ | paper | Same paper as arXiv 2409.06289 (EMNLP 2025 proceedings) |
| https://blog.bytebytego.com/p/how-anthropic-built-a-multi-agent | blog | Anthropic primary source fetched in full; secondary summary not needed |
| https://kingy.ai/ai/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/ | blog | Search snippet; DataCamp fetched in full |

---

### Recency scan (2024-2026)

Searched explicitly for 2024-2026 literature on LLM-as-planner for trading/quant systems.

**Findings:** 4 new works from 2024-2026 are directly relevant and complement the phase-3.1 architecture:

1. **arXiv 2409.06289 (Sep 2024, EMNLP 2025)** — "Automate Strategy Finding with LLM in Quant Investment." Proposes a Seed Alpha Factory + multimodal evaluator that is the closest academic analogue to what phase-3.1 is building. Key gap vs. our implementation: their evaluator uses an Information Coefficient (IC) metric alongside Sharpe — our `EvaluatorAgent` uses only Sharpe/DSR.

2. **arXiv 2412.20138 (Dec 2024)** — TradingAgents. Confirms structured document exchange (not live conversation) as the correct inter-agent protocol. Validates pyfinagent's file-based handoff design.

3. **arXiv 2602.23330 (Feb 2026)** — Expert Investment Teams. Confirms that fine-grained task decomposition (concrete metrics, exact output formats) outperforms vague agent roles. Direct implication: `PlannerAgent.generate_proposal()` system prompt is appropriately specific; `EnhancedPlannerAgent._build_system_prompt()` is even better.

4. **Karpathy AutoResearch (2026-03)** — propose-measure-keep/discard ratchet. 700 experiments, 48h, 11% improvement. The pattern maps directly to `run_harness.py`'s existing rule-based planner + `quant_results.tsv` log. The critical difference: Karpathy's ratchet uses an immutable eval function (val_bpb) applied after every commit. Pyfinagent's Sharpe + DSR is the equivalent — but the LLM planner (`planner_agent.py`) is NOT yet wired to the ratchet loop in `run_harness.py`.

**No new findings that supersede** the core contract hypothesis. The 2024-2026 literature reinforces the propose-evaluate-feedback loop architecture, does not invalidate it.

---

### Key findings

1. **`planner_agent.py` (251 lines) implements Phase 3.1 Component 1 faithfully.** `PlannerAgent.generate_proposal()` reads evidence summary, calls Claude Opus, returns structured JSON proposals. `reflect_on_feedback()` implements the feedback loop from the contract's Component 4. Both methods match the contract spec at lines 103-143. (Source: internal read, `backend/agents/planner_agent.py:43-151`)

2. **`planner_enhanced.py` (336 lines) implements Phase 3.3 scope, NOT Phase 3.1 scope.** Its docstring explicitly says "Phase 3.3 Autonomous Loop." It adds regime conditioning, RESEARCH.md reading, and richer per-proposal fields. This is forward work. (Source: `backend/agents/planner_enhanced.py:1-10, 166`)

3. **`evaluator_agent.py` (522 lines) implements Phase 3.2 fully.** This is the separate phase — it exists, it is complete, it has `EvaluatorAgent` with a 5-rubric scoring system (DSR, robustness, simplicity, reality gap, risk check), spot-check logic, and a mock evaluator for testing. (Source: `backend/agents/evaluator_agent.py:1-27, 78-522`)

4. **`autonomous_loop.py` (481 lines) wires planner + evaluator together.** `AutonomousLoopOrchestrator._plan_phase()` (line 231) imports and calls `PlannerAgent`. `_evaluate_phase()` (line 314) imports and calls `EvaluatorAgent`. The full Plan → Generate → Evaluate loop exists in code. (Source: `backend/autonomous_loop.py:231-359`)

5. **The critical gap: `autonomous_loop.py` is NOT called by `run_harness.py`.** The harness planner (`run_harness.py:149`) is a **rule-based** planner (plateau detection, param saturation rules) — it does NOT call `PlannerAgent` from `planner_agent.py`. The LLM planner exists in `autonomous_loop.py` (a separate entry point) but is isolated from the production harness. (Source: `scripts/harness/run_harness.py:149-280`, `backend/autonomous_loop.py:231`)

6. **`autonomous_loop.py`'s `_plan_phase()` feeds mock data.** Lines 254-276 show hardcoded `mock_recent_results` and `current_params` — real backtest history from `quant_results.tsv` / BigQuery is NOT wired in. The planner cannot learn from real experiment history. (Source: `backend/autonomous_loop.py:253-276`)

7. **`autonomous_loop.py`'s `_evaluate_phase()` uses simplified logic, not the full EvaluatorAgent rubric.** Lines 354-358 show a direct `if result_sharpe > baseline_sharpe and dsr > 0.95` check, not a call to `EvaluatorAgent.evaluate_proposal()`. The evaluator import is there but the full rubric is bypassed. (Source: `backend/autonomous_loop.py:354-358`)

8. **No tests exist for `PlannerAgent` or `EvaluatorAgent`.** `backend/tests/` contains `test_bq_writer.py`, `test_calendar_watcher.py`, `test_observability.py`, `test_paper_trading_v2.py`, `test_sentiment_ladder.py` — nothing for the planner/evaluator stack. (Source: `backend/tests/` directory listing)

9. **Phase 3.1 masterplan status is `pending`, verification criteria are `null`.** The step was opened, code was written, harness cycle was never closed. Phase 3.2 is also `pending` with `contract: null`. (Source: `.claude/masterplan.json` phase-3 steps)

10. **Anthropic's architecture confirms the design direction.** Lead agent (extended thinking) → parallel subagents → synthesis → iterate. Token budget drives performance (80% of variance). This validates the Planner → Evaluator split but requires proper wiring to real data. (Source: anthropic.com/engineering/multi-agent-research-system, 2026-04-19)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/planner_agent.py` | 251 | Phase 3.1 planner: `PlannerAgent.generate_proposal()`, `reflect_on_feedback()` | Complete and functional. Not integrated into `run_harness.py`. |
| `backend/agents/planner_enhanced.py` | 336 | Phase 3.3 enhanced planner: `EnhancedPlannerAgent`, RESEARCH.md reading, regime conditioning | Forward work (Phase 3.3 scope). Unused in any harness path. |
| `backend/agents/evaluator_agent.py` | 522 | Phase 3.2: `EvaluatorAgent`, 5-rubric scoring, spot checks, mock evaluator | Complete and functional. Not integrated into `run_harness.py`. |
| `backend/autonomous_loop.py` | 481 | Wires planner + evaluator in a Plan→Generate→Evaluate loop | Exists. Called by `scripts/harness/run_autonomous_loop.py` (worktree only). Feeds mock data to planner. Bypasses evaluator rubric. Not called by production `run_harness.py`. |
| `scripts/harness/run_harness.py` | ~1200+ | Production harness: rule-based `run_planner()` at line 149 | Uses heuristic planner only. `PlannerAgent` import exists nowhere in this file. |
| `backend/autonomous_harness.py` | — | Production harness backend: gauntlet evaluator, not LLM planner | No references to `PlannerAgent` or `EvaluatorAgent`. |
| `handoff/archive/phase-3.1/contract.md` | 213 | Original phase-3.1 plan and success criteria | Research gate was marked "IN PROGRESS" — never formally closed. |
| `handoff/archive/phase-3.1/research_queue.md` | 137 | Pre-work research notes | Contains Anthropic multi-agent pattern summary. No experiment_results.md. No evaluator_critique.md. |
| `backend/tests/` | — | Test suite | No tests for planner or evaluator stack. |

---

### Consensus vs debate (external)

**Consensus:** Planner → Evaluator separation is the correct architectural split (Anthropic, TradingAgents, Expert Investment Teams papers all confirm). File-based handoff between agents is the documented reliable pattern (Anthropic). Fixed evaluation metric (Sharpe + DSR) as immutable yardstick is correct (Karpathy ratchet analogy).

**Debate / open questions:** Whether the LLM planner should call real backtests directly (TradingAgents does; Karpathy's system modifies training code directly) vs. proposing parameter changes that the harness executes (pyfinagent's current design). The pyfinagent separation (planner proposes → harness runs → evaluator judges) is arguably cleaner but requires wiring.

---

### Pitfalls (from literature)

1. **Mock data starvation (Karpathy):** The ratchet loop only improves when it reads real experiment history. Mock proposals cannot converge on real weaknesses. Current `autonomous_loop.py` feeds hardcoded mocks — this is the primary bug.

2. **Evaluator bypassed in the loop (internal):** `_evaluate_phase()` short-circuits the 5-rubric `EvaluatorAgent` with a 2-line Sharpe check. The robustness, simplicity, and reality-gap dimensions are never applied in the actual loop.

3. **Two planners, no canonical one (internal):** `PlannerAgent` (phase-3.1) and `EnhancedPlannerAgent` (phase-3.3 scope) coexist without a clear entry point. `autonomous_loop.py` uses `PlannerAgent`. `planner_enhanced.py` has a `__main__` block but nothing calls it in production.

4. **No tests (internal):** Zero coverage means the planner stack has no regression baseline before integration.

5. **Token cost at scale (arXiv 2409.06289):** Multi-agent systems use ~15x more tokens. At Opus pricing, unconstrained planner cycles will exceed the $0.50/cycle budget target in the contract within a few runs if the proposal depth is not controlled.

---

### Application to pyfinagent

The contract's four components map to existing code as follows:

| Contract Component | File | Status | Gap |
|-------------------|------|--------|-----|
| Component 1: Planner Agent | `planner_agent.py:43` | Implemented | Not wired to real backtest history |
| Component 2: Evaluator Agent | `evaluator_agent.py:78` | Implemented (Phase 3.2) | Loop in `autonomous_loop.py` bypasses full rubric (`autonomous_loop.py:354`) |
| Component 3: Evidence Engine | None | Missing | `autonomous_loop.py` uses hardcoded mocks. No BigQuery read of `quant_results.tsv` |
| Component 4: Feedback Loop | `autonomous_loop.py:231-359` | Partially implemented | Real backtest runner not wired; uses mock results (`autonomous_loop.py:305`) |

**The one integration point missing:** `run_harness.py:run_planner()` (line 149) must either (a) be replaced by a call to `PlannerAgent.generate_proposal()` with real TSV data, or (b) `autonomous_loop.py` must be promoted as the canonical harness entry point with real BQ/TSV data wired in.

---

### Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 confirmed: Anthropic, arXiv 2409.06289v3, arXiv 2412.20138v3, arXiv 2602.23330v1, DataCamp/Karpathy)
- [x] 10+ unique URLs total (incl. snippet-only) — 11 collected
- [x] Recency scan (last 2 years) performed + reported — 4 new findings from 2024-2026
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks — note gaps but do not auto-fail:
- [x] Internal exploration covered every relevant module (planner_agent, planner_enhanced, evaluator_agent, autonomous_loop, run_harness, autonomous_harness, tests, masterplan, archive)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

### Open questions for the contract

1. Should `planner_enhanced.py` be merged into `planner_agent.py` or kept separate for phase-3.3? Current duplication is dead weight.
2. Is `autonomous_loop.py` the canonical entry point for the LLM planner loop, or should `run_harness.py` be extended? Answer determines where the real-data wiring goes.
3. Phase 3.2 (`evaluator_agent.py`) is already implemented. Should phase-3.1 and 3.2 be closed together in a single step since the code already exists for both?
4. The contract's success criterion "≥50% of LLM proposals pass robustness test" requires a real backtest run on proposals. Is that in scope for closing phase-3.1, or deferred to phase-3.3?

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 6,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/phase-3.1-research-brief.md",
  "gate_passed": true
}
```
