---
step: 24.4
title: Agent topology + per-agent rationale flow audit (P0)
date: 2026-05-12
tier: complex
---

## Research: Agent Topology and Per-Agent Rationale Flow Audit (phase-24.4)

### Queries run (three-variant discipline)

| Variant | Queries |
|---------|---------|
| Current-year frontier (2026) | "multi-agent debate LLM independent evaluator consensus 2026", "independent evaluator agent design 2026", "AI evaluator independence multi-agent pipeline 2026" |
| Last-2-year window (2025) | "NIST AI evaluation methodology independent evaluator design 2025", "AI evaluator independence multi-agent pipeline 2025" |
| Year-less canonical | "Society of Minds Du et al multi-agent debate language models", "multi-agent debate LLM consensus voting", "independent evaluator agentic AI" |

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-12 | Official doc | WebFetch | "Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator" — prevents game-of-telephone information loss |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-12 | Official doc | WebFetch | "agents tend to respond by confidently praising the work — even when, to a human observer, the quality is obviously mediocre"; "separating the agent doing the work from the agent judging it proves to be a strong lever" |
| https://www.anthropic.com/engineering/building-effective-agents | 2026-05-12 | Official doc | WebFetch | "one LLM call generates a response while another provides evaluation and feedback in a loop"; evaluator must be independent to identify weaknesses the generator overlooks |
| https://composable-models.github.io/llm_debate/ | 2026-05-12 | Academic project page | WebFetch | Du et al. Society of Minds: agents first propose independently, then critique; "the final answer generated after such a procedure is both more factually accurate" — independent generation before observation is key |
| https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai | 2026-05-12 | Official doc (NIST) | WebFetch | "automated tools... act as adversarial verifiers" using "a strict rubric" compared to "human-curated corpus of reference documents" — independent third-party checkers, not self-referential |
| https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents | 2026-05-12 | Official doc | WebFetch | Separation of grader types (code-based, model-based, human) — "distinct grader types evaluate different aspects without overlap, reducing bias" |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/2305.14325 | Paper (Du et al. 2023) | arXiv page returned only abstract; project page read in full instead |
| https://openreview.net/forum?id=FQepisCUWu | Paper (ChatEval 2024) | Snippet only — project page and direct WebFetch not performed (budget) |
| https://openreview.net/forum?id=Vusd1Hw2D9 | Paper (2026 debate judge) | Snippet only — full fetch not within budget |
| https://www.nist.gov/ai-test-evaluation-validation-and-verification-tevv | Official NIST | Snippet only — sufficient from NIST probe page |
| https://www.nist.gov/news-events/news/2026/02/new-report-expanding-ai-evaluation-toolbox-statistical-models | Official NIST | Snippet only — complementary to probe page |
| https://arxiv.org/html/2604.26561v1 | Paper (2026 AI Council) | Snippet only — three-phase deliberation pattern noted |
| https://dl.acm.org/doi/10.5555/3692070.3692537 | Paper (ICML 2024) | Snippet only — Du et al. ICML version |
| https://www.sitepoint.com/the-definitive-guide-to-agentic-design-patterns-in-2026/ | Blog (2026) | Snippet only — "independent evaluator" as named pattern |
| https://www.langchain.com/state-of-agent-engineering | Industry report | Snippet only — 57% orgs have agents in prod; quality is top barrier |
| https://assets.amazon.science/48/5d/20927f094559a4465916e28f41b5/enhancing-llm-as-a-judge-via-multi-agent-collaboration.pdf | Paper (Amazon 2024/25) | Snippet only — multi-agent judge collaboration |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on multi-agent debate independence and evaluator aliasing. Results:

- **2026 finding**: Anthropic published "Demystifying Evals for AI Agents" (2026), establishing code-based + model-based + human grader separation as production best practice. Directly applicable.
- **2026 finding**: NIST published "Building Evaluation Probes into Agentic AI" (ongoing 2026), formalizing "adversarial verifier" probes that compare against reference corpora rather than self-referential generation output.
- **2025/2026 finding**: LangChain "State of AI Agents" (2026): quality is the top deployment barrier (32% of respondents). Human-in-the-loop review for nuanced evaluation remains at 59.8% adoption. Online evals at 37.3%.
- **2025 finding**: "Adaptive heterogeneous multi-agent debate" (Springer 2025) extends Du et al. with heterogeneous model types — diversity of model family further reduces output aliasing vs. homogeneous debate panels.
- **No superseding canonical shift**: Du et al. (2023) "society of minds" pattern remains foundational; 2024-2026 work builds on it rather than overturning it. The key principle — independent generation before peer critique — is confirmed across all recent work.

---

### Key findings

1. **Independent generation is structurally required to prevent aliasing.** Du et al. (2023): agents must produce outputs independently before observing peers. When an evaluator sees the generator's output before producing its own, outputs converge (alias). The RiskJudge-reads-Trader pattern is the textbook form of this failure. (Source: Du et al., https://composable-models.github.io/llm_debate/)

2. **Self-evaluation is the strongest single anti-pattern.** Anthropic (harness-design): "agents tend to respond by confidently praising the work — even when, to a human observer, the quality is obviously mediocre." An evaluator that receives the generator's reasoning as its own input will inherit that reasoning. (Source: https://www.anthropic.com/engineering/harness-design-long-running-apps)

3. **Subagents must write to independent storage; passing outputs through conversation creates game-of-telephone degradation.** Anthropic (multi-agent research system): structured storage with lightweight references prevents cascading fidelity loss. In pyfinagent: the lite-path writes a single `reason` field; both Trader AND RiskJudge pull from it. (Source: https://www.anthropic.com/engineering/built-multi-agent-research-system)

4. **Evaluator independence requires distinct LLM call + distinct prompt context.** Anthropic (building-effective-agents): "one LLM call generates a response while another provides evaluation and feedback." RiskJudge in lite mode makes zero independent LLM calls — it post-processes Trader output. (Source: https://www.anthropic.com/engineering/building-effective-agents)

5. **Adversarial verifiers must compare against trusted reference material, not self-produced output.** NIST: evaluation probes use "human-curated corpus of reference documents" to evaluate, not a re-reading of the same generation. A risk evaluation grounded only in the trader's reasoning is not adversarial. (Source: https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai)

6. **Weight=0.0 is a sentinel for "no independent size recommendation was produced."** The lite-path `_run_claude_analysis` returns `risk_assessment: {"reason": analysis["reason"]}` with no `recommended_position_pct`. The signal_attribution code explicitly checks `risk_weight == 0.0 AND risk_rationale == trader_rationale_trimmed` to flag this. The flag exists; the fix does not.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/autonomous_loop.py` | ~800+ | Daily cycle; contains `_run_claude_analysis` lite path (L619-740) | Active — lite path is the primary paper-trading analysis path |
| `backend/services/signal_attribution.py` | 298 | Converts raw analysis dicts to per-trade signal rows; `extract_signals_from_analysis` is the aliasing detection site | Active — aliasing detection at L131-154 flags but does not fix |
| `backend/api/paper_trading.py` | ~540+ | Paper trading endpoints; `/trades/{trade_id}/rationale` at L460-511 | Active — calls `group_signals_for_drawer(signals)` on BQ-stored signals column |
| `frontend/src/components/AgentRationaleDrawer.tsx` | 232 | Renders progressive-disclosure rationale tree; knows about `lite_path` flag | Active — renders amber "lite-path" badge when `s.lite_path == true` but drawer cannot distinguish if text is byte-identical without the flag |
| `backend/agents/multi_agent_orchestrator.py` | 1513 | Layer-2 MAS (Slack bot); NOT the paper-trading decision path | Active but irrelevant to paper-trade rationale — this is the Slack/iMessage orchestrator, separate from the autonomous_loop paper-trade decision path |
| `backend/agents/agent_definitions.py` | 428 | Layer-2 agent configs (4 types: COMMUNICATION, MAIN, QA, RESEARCH) | Active — these are Slack-bot agents, not paper-trade agents |
| `backend/agents/_inventory.json` | 94 | Canonical agent topology; 43 nodes across 4 layers | Active — 3 Layer-3 harness agents, 7 Layer-2 MAS agents, 28+ Layer-1 skill agents, ~10 Layer-4 services |
| `backend/agents/planner_agent.py` | 60+ read | Layer-3 harness planner; reads backtest results, proposes next parameters | Active — not in paper-trade rationale path |
| `backend/agents/evaluator_agent.py` | not read (not in rationale path) | Layer-2 EvaluatorAgent (Gemini); backtests proposed trade plans | Active — not in rationale drawer path |

---

### The Smoking Gun (verbatim grep evidence)

The aliasing is at `backend/services/signal_attribution.py:L117-154`:

```python
# ── Risk layer ────
risk = analysis.get("risk_assessment") or {}
if isinstance(risk, dict):
    decision = risk.get("decision") or ""
    # phase-23.1.7: lite shape uses {"reason": "..."}; add as fallback alongside reasoning/rationale.
    reasoning = (
        risk.get("reasoning")
        or risk.get("rationale")
        or risk.get("reason")  # <-- this is the lite-path field
        or ""
    )
    ...
    risk_rationale = _trim(reasoning) or f"Decision: {decision}"
    risk_weight = float(pos_pct) if isinstance(pos_pct, (int, float)) else 0.0
    # phase-23.2.A-fix: Option B -- detect lite-path duplicate.
    trader_rationale_trimmed = _trim(trader_note) or f"Recommendation: {rec}"
    is_lite_dup = (
        risk_weight == 0.0
        and risk_rationale == trader_rationale_trimmed  # byte-identical check
    )
```

The upstream source is `backend/services/autonomous_loop.py:L711-720`:

```python
return {
    ...
    "risk_assessment": {"reason": analysis["reason"]},  # <-- same "reason" field as trader
    ...
    "full_report": {
        "analysis": analysis,  # analysis["reason"] is the trader rationale
    },
}
```

And the Trader layer reads from:

```python
trader_note = (
    analysis.get("trader_note")
    or analysis.get("recommendation_reason")
    or (analysis.get("full_report") or {}).get("analysis", {}).get("reason")  # <-- same reason
    or ""
)
```

**Both Trader and RiskJudge resolve to `analysis["reason"]` — the single LLM response sentence from `_run_claude_analysis`.** No separate risk LLM call exists in the lite path.

---

### Consensus vs debate (external)

**Consensus**: All five external sources agree — independent generation before observation is structurally required to prevent output aliasing. Evaluator agents that receive the generator's reasoning as primary input will converge on it. The evaluator-optimizer pattern (Anthropic), society of minds (Du et al.), and NIST adversarial verifiers all independently assert this.

**No dissent found**: No source argues that sharing generator context with evaluators is beneficial for independence. The Anthropic demystifying-evals piece shows distinct grader types with non-overlapping concerns, not cascaded graders that read each other.

---

### Pitfalls (from literature)

1. **Grading the same thing twice from the same input** — the exact failure mode in pyfinagent. NIST: reference corpus must be independent of generation output.
2. **Treating a sentiment flag as a fix** — the `lite_path=True` badge is a documentation flag, not a structural fix. The drawer still shows the RiskJudge row; the operator may not notice the amber badge.
3. **Weight=0.0 as a UI signal without a backend remedy** — a zero weight shown in the drawer creates the impression that RiskJudge ran but had zero conviction, not that RiskJudge never ran.
4. **Layer-1 28-skill outputs never surfaced in lite mode** — the full 28-agent Gemini pipeline is entirely bypassed by `_run_claude_analysis`. Even in full mode, there is no mechanism to surface individual skill outputs in the drawer tree (no `quant_model`, `bull_agent`, etc. rows appear).
5. **Sparse drawer by default** — `group_signals_for_drawer` only groups what is already in the stored `signals` JSON column. If the full pipeline was never run, the column has 3-4 entries maximum.

---

### Application to pyfinagent (file:line anchors)

#### Finding 1: The aliasing site

`backend/services/autonomous_loop.py:L719`:
```python
"risk_assessment": {"reason": analysis["reason"]},
```
The lite-path LLM call (`_run_claude_analysis`, L670-690) produces ONE reason sentence. It is stored as `risk_assessment.reason`. The Trader layer reads from `full_report.analysis.reason` (L104-107 in signal_attribution.py). Both resolve to the same string. **No independent risk LLM call is made.**

#### Finding 2: Aliasing detection at L131-154 is cosmetic

`backend/services/signal_attribution.py:L139-154`: `is_lite_dup` is detected and the rationale text is replaced with the human-readable message `"Lite-path: Risk Judge inherited Trader's reasoning..."`. The `lite_path=True` flag is set. The frontend renders an amber badge. **This is a display patch, not a fix** — the RiskJudge still has weight=0.0 and no independent content.

#### Finding 3: RiskJudge weight=0.0 tracing

`backend/services/signal_attribution.py:L126-130`:
```python
pos_pct = risk.get("recommended_position_pct")
...
risk_weight = float(pos_pct) if isinstance(pos_pct, (int, float)) else 0.0
```
The lite path never populates `recommended_position_pct`. This is confirmed by `autonomous_loop.py:L711-740` — the lite return dict has no `recommended_position_pct` key in `risk_assessment`.

#### Finding 4: Full Layer-1 28-skill pipeline not surfaced

`_inventory.json:L42` confirms 28 Layer-1 skill agents exist. `extract_signals_from_analysis` in `signal_attribution.py` only extracts: Analyst, Bull/Bear, Trader, RiskJudge (5 signal types). No Layer-1 skill individual output (e.g., `NLPSentimentAgent`, `ScenarioAgent`, `BiasDetector`) is extracted. Even when full Gemini pipeline runs, the individual skill rationales are not persisted into the `signals` column.

#### Finding 5: Drawer shows 3 of ~20 agents

With the lite path, `extract_signals_from_analysis` produces at most 3 rows: Analyst (absent in lite — no `analyst_summary`), Trader (1 row), RiskJudge (1 row, flagged lite-path), plus potentially a Quant row if a `candidate` dict is passed. So: **2-3 rows visible, not 20**. In full Gemini path without quant signals: 4-5 rows (Analyst + Bull + Bear + Trader + RiskJudge). The inventory has 43 nodes across 4 layers; drawer coverage is 10-25% at best.

#### Finding 6: Layer-2 MAS (multi_agent_orchestrator.py) is NOT the paper-trade path

The investigation of `multi_agent_orchestrator.py` shows it is the **Slack/iMessage Layer-2 orchestrator** (responding to Slack commands, classifying queries, spawning COMMUNICATION/MAIN/QA/RESEARCH agents). It does NOT drive paper trading decisions. Paper trading flows through `autonomous_loop.py` → `_run_single_analysis` (full Gemini path) or `_run_claude_analysis` (lite path) → `extract_signals_from_analysis`. The "Trader" and "RiskJudge" in the drawer are NOT from `multi_agent_orchestrator.py`. **The hypothesis as stated was wrong in identifying the file — the aliasing occurs in `autonomous_loop.py` + `signal_attribution.py`, not in `multi_agent_orchestrator.py`.**

---

### Phase-25 candidates

| # | Candidate | File(s) | Verifier |
|---|-----------|---------|----------|
| **25.A** | **Decouple RiskJudge in lite path**: add a second LLM call in `_run_claude_analysis` with a risk-specific prompt (position sizing, max notional, drawdown risk) that runs AFTER the Trader decision, with the Trader decision as context but NOT as the only output | `backend/services/autonomous_loop.py:L619-740` | Assert `risk_assessment.reasoning != analysis["reason"]` in a unit test; assert `risk_weight > 0` on BQ row |
| **25.B** | **Remove the cosmetic aliasing patch and fix the root cause**: delete `is_lite_dup` branch in `signal_attribution.py:L131-154` once 25.A decouples the calls; replace with honest weight and independent rationale | `backend/services/signal_attribution.py:L117-157` | Assert `lite_path` key absent from RiskJudge signal; assert `weight > 0` |
| **25.C** | **Surface Layer-1 skill outputs in drawer when full pipeline ran**: `extract_signals_from_analysis` currently ignores Layer-1 keys. Extend it to extract synthesis sub-components (NLP sentiment, scenario weights, bias flags) into a new `layer1_skills` tree node. Gate behind `settings.lite_mode == False` check. | `backend/services/signal_attribution.py:L57-157`, `frontend/src/components/AgentRationaleDrawer.tsx` | Assert drawer tree has `layer1_skills` key with >=3 entries on a full-pipeline trade |
| **25.D** | **Add per-agent contribution weight to all signals**: Trader weight is currently `final_score` (1-10 scale); RiskJudge weight is `recommended_position_pct`. Normalize both to 0-1 range and display alongside each row. Add a "total weight" summary line at the top of the drawer. | `backend/services/signal_attribution.py`, `frontend/src/components/AgentRationaleDrawer.tsx` | Visual regression test: all weight values 0.0-1.0; sum displayed correctly |
| **25.E** | **Expand drawer from "3 main" to "all contributing" toggle**: add a `?full=1` query param to `/trades/{trade_id}/rationale` that returns all persisted signals without tree-grouping; frontend toggle between "summary" (current 3-5 rows) and "full" (all rows) views | `backend/api/paper_trading.py:L460-511`, `frontend/src/components/AgentRationaleDrawer.tsx` | Assert full mode returns >5 signals on a full-pipeline trade |
| **25.F** | **Add regression test for byte-identical detection**: unit test that feeds identical Trader + RiskJudge strings through `extract_signals_from_analysis` and asserts the output is flagged as `lite_path=True`. Add a second assertion that the full-path (with `recommended_position_pct`) produces distinct rationale and `lite_path` absent. | `tests/` new file | `pytest tests/test_signal_attribution.py::test_lite_path_detection` passes |

---

### Research Gate Checklist

Hard blockers -- `gate_passed` is true if all checked:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (incl. snippet-only) (16 URLs collected)
- [x] Recency scan (last 2 years) performed + reported (2024-2026 window, 5 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (autonomous_loop.py:L619, L711, L719; signal_attribution.py:L117, L126, L131, L139; paper_trading.py:L460; AgentRationaleDrawer.tsx:L1-232)

Soft checks -- note gaps but do not auto-fail:

- [x] Internal exploration covered every relevant module (6 internal files read in full, 2 partially)
- [x] Contradictions / consensus noted (hypothesis file location corrected: not multi_agent_orchestrator.py)
- [x] All claims cited per-claim (not just listed in a footer)
- [ ] evaluator_agent.py not read in full (determined to be outside the rationale path based on _inventory.json)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
