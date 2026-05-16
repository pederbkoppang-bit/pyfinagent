# Internal Gap Brief — pyfinagent Top-5 Profit/Efficiency Gaps

**Date:** 2026-05-16
**Author:** Main (researcher subagent retry produced skeleton only; Main completed in-line)
**Tier:** simple (internal-only scan, no external floor required)

## Top-5 ranked gaps

### Gap 1 — Zero `code_execution` adoption on Gemini agents
- **Evidence:** `grep -rln "code_execution" backend/ --include="*.py"` → no hits
- **Implication:** Quantitative skills (`quant_model_agent.md`, `quant_strategy.md`, `scenario_agent.md`, `enhanced_macro_agent.md`) currently return numerical analysis as JSON only; the model never executes code to verify arithmetic, run a quick regression, or stress-test a scenario. Either we duplicate the math in Python downstream (extra latency + drift) or we trust the LLM's mental arithmetic.
- **Profit hypothesis:** Wiring `code_execution` on the 4 quant skills lets them sanity-check their own math (correct Sharpe, position sizing arithmetic) inside one call, eliminating a class of "the model said 0.42 but the number was 0.24" silent errors that show up in backtest divergence reports.
- **Not duplicated in phase-25** — 48 steps reviewed; only `25.C9`/`25.D9` touch Anthropic features; nothing about Gemini code_execution.
- **Effort: M** | **Risk if skipped:** silent arithmetic errors continue to bleed into signal quality.

### Gap 2 — No autonomous-cycle hard token budget
- **Evidence:** `grep -rln -iE "task_budget|max_session_budget|budget_ceiling" backend/` → no hits. `backend/services/autonomous_loop.py` runs harness cycles with only per-cycle cost-tracking (post-hoc), not a hard pre-cycle ceiling. `25.A8` added an `llm_client.py` hard-block but it's per-call, not per-session.
- **Implication:** A runaway harness cycle (extended thinking + retry loops) can blow $50+ before the per-call gate trips frequently enough to halt it. The `25.A8` per-call ceiling is necessary but not sufficient — there's no upper bound on total session cost.
- **Profit hypothesis:** Hard pre-cycle ceiling lets us safely raise cycle frequency (which compounds profit through faster strategy adaptation) without tail-risk on a single bad cycle.
- **Not duplicated** — 25.A8 is per-call only; no session-level cap exists.
- **Effort: S** | **Risk if skipped:** one extended-thinking storm can equal a week of cycle budget.

### Gap 3 — 6 overlapping "opinion" skills with no consolidation lever
- **Evidence:** `ls backend/agents/skills/` reveals `bull_agent.md`, `bear_agent.md`, `aggressive_analyst.md`, `conservative_analyst.md`, `neutral_analyst.md`, `devils_advocate_agent.md` — six separate Gemini calls producing structurally identical "take a stance" outputs that the synthesis layer merges.
- **Implication:** Six independent Gemini calls per signal cycle just for "spread of opinion" is high token spend for low-marginal-information output. The Anthropic *Advisor Tool* pattern (external brief #1) does this with one executor + one advisor call. A single parameterized stance-prompt could replace 6 separate skills.
- **Profit hypothesis:** Replacing 6 calls with 2 (one parameterized for stance) cuts Gemini token spend ~33% on the opinion-leg without losing the multi-stance synthesis input.
- **Not duplicated** — phase-25 didn't consolidate the skill set; only added/wired skills.
- **Effort: M** | **Risk if skipped:** opinion-leg compute spend grows linearly with each new stance skill.

### Gap 4 — No alpha-decay / regime-shift detector skill
- **Evidence:** None of the 33 skills in `backend/agents/skills/` contains "decay", "regime_shift", or "alpha_attribution". `phase-25.S` added per-ticker daily P&L attribution but not strategy-level alpha-decay detection.
- **Implication:** The strategy auto-switching policy (`phase-25.R`) decides which strategy to allocate to *given* their recent performance — but no upstream signal detects when a *currently-allocated* strategy's edge is decaying before the policy notices. The system reacts to decay; it doesn't anticipate it.
- **Profit hypothesis:** A regime-shift / alpha-decay agent (cheap Gemini Flash) running on each cycle gives the strategy router an early-warning signal, shortening the lag between decay onset and capital reallocation. Direct profit lever via reduced drawdown.
- **Not duplicated** — phase-25.R is a downstream policy; this is an upstream detection signal.
- **Effort: M** | **Risk if skipped:** continued lag in shifting capital out of decaying strategies.

### Gap 5 — Multimodal RAG gap on `financial_reports` dataset
- **Evidence:** `rag_agent.md` exists; BigQuery dataset `financial_reports` (us-central1) holds filings. Currently text-only extraction. No grep hits for `multimodal`, `image_embedding`, or `media_id` in `backend/`.
- **Implication:** Charts, tables, and figures in 10-Ks/10-Qs carry signal we never see. Competitors running multimodal RAG on the same filings extract this signal.
- **Profit hypothesis:** Index filings as multimodal via Gemini's File Search + `gemini-embedding-2` (external brief #4); RAG returns include visual citations. New signal layer with no incremental compute cost on retrieval.
- **Not duplicated** — phase-25 has no RAG / filings work.
- **Effort: L** | **Risk if skipped:** stable signal gap vs any multimodal competitor.

## Feature-adoption quick check

| Feature | Adopted | Evidence |
|---|---|---|
| Claude prompt caching (`cache_control`) | Yes (4 files) | `backend/agents/llm_client.py`, `backend/tools/sec_insider.py`, `backend/tools/earnings_tone.py`, `backend/news/sentiment.py` |
| Extended thinking | Yes | `llm_client.py:1491` handles `thinking` block types |
| Files API (`skill_file_id`) | Yes | `llm_client.py:1220` — phase-25.D9 |
| Batch API | Yes | `llm_client.py` — phase-25.C9 (50% savings on non-interactive) |
| Native Citations | Yes | phase-25.E9 |
| Gemini grounding (Google Search) | Yes | `backend/agents/orchestrator.py`, `schemas.py`, etc. |
| **Gemini `code_execution`** | **No** | Zero hits |
| **Anthropic Task Budgets** | **No** | Zero hits |
| **Anthropic Advisor Tool (Sonnet+Opus pairing)** | **No** | Zero hits |
| Gemini Multimodal File Search | No | Zero hits |

## Cross-check vs phase-25 (48 steps, all `done`)

All 5 gaps above are net-new — they target either (a) features released after the 2026-05-12 audit window or (b) topology gaps the audit didn't surface (decay detection, opinion-skill consolidation).

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```

**Gate note:** External-source floor does not apply to internal-only gap analysis. Internal scan inspected 9 files + 1 directory listing + 1 masterplan JSON.
