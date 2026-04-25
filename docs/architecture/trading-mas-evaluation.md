# Trading-MAS Benefit Evaluation

**Date:** 2026-04-25
**Author:** pyfinagent harness MAS (Main + Researcher + Q/A)
**Status:** Research deliverable — design recommendation, NOT a build commit
**Cycle:** phase-16.27
**Audience:** Peder + future MAS sessions

User's instruction: *"check it up against our app and see how we can benefit from it [a multi-agent system for trading decisions, modeled on the existing harness MAS]"*.

This document evaluates whether and how to add a **trading-decision MAS** layer between the existing Layer-1 28-agent analysis pipeline and the rule-based `portfolio_manager.decide_trades()`. It is grounded in:

- **Internal code audit** (3 Explore + 2 researcher cycles in this session)
- **2024-2026 academic + industry literature** (TradingAgents, HedgeAgents, FinRL, Public.com, LLM-as-Judge)
- **Risk doctrine** (Bailey & Lopez de Prado on backtest overfitting; Anthropic harness design)

---

## 1. Current state — where decisions are made today

The trading decision flow today, end-to-end:

```
backend/services/autonomous_loop.py::run_daily_cycle()
  └─> Step 1: screen candidates (universe filter)
  └─> Step 2: AnalysisOrchestrator.run_full_analysis(ticker)            [Layer-1, 28 agents]
        ├─ enrichment (12 data agents): insider, options, social, patent, earnings, macro, alt, sector, NLP, anomaly, scenario, quant
        ├─ debate (Bull ↔ Bear ↔ Devil's Advocate ↔ Moderator)
        ├─ risk_debate (Aggressive ↔ Conservative ↔ Neutral ↔ Risk Judge)
        ├─ synthesis → SynthesisReport JSON
        └─ critic loop (max 2 reflections)
  └─> Step 3: outcome_tracker.evaluate_recent (RECORDS, doesn't FEED BACK)
  └─> Step 4: agent_memories.retrieve (LOADS at orchestrator start, NOT mid-session)
  └─> Step 5: mark-to-market existing positions
  └─> Step 6: portfolio_manager.decide_trades(candidate_analyses, holdings)
        ├─ rule-based: sell-first-then-buy
        ├─ Risk Judge sets position_pct (1-5%)
        └─ TradeOrder list
  └─> Step 7: execution_router.submit_order(...) → bq_sim or alpaca_paper
  └─> Step 8: snapshot, persist, log
```

**Where the LLM reasons today:** in Layer-1 (28 Gemini-or-Claude agents producing the synthesis) and in `Risk Judge` (one LLM call sizing the position). The transition from "synthesis" to "trade order" is **one rule-based function** — `decide_trades()` — with no second-look reasoning.

**Where MAS already exists:**
- **Harness MAS** (`Main + Researcher + Q/A`): governs `phase-X.Y` cycles, lives outside the trading path
- **In-app MAS** (`backend/agents/multi_agent_orchestrator.py`): 4 agents (mas_main / mas_research / mas_qa / mas_communication), serves Slack and `/api/agents/*` queries; **does not participate in daily trading**

---

## 2. Gap analysis — where alpha leaks

| Layer | Decision | LLM-reasoned? | Gap |
|-------|----------|---------------|-----|
| Layer-1 (analysis) | recommendation, confidence, risk_assessment | Yes (28 agents) | Strong |
| Risk Judge | position_pct (1-5%) | Yes (1 LLM call) | OK |
| **decide_trades** | **what + how much to BUY/SELL given holdings + macro** | **No (rule-based)** | **HERE** |
| execution_router | route + clamp + idempotent | No (rule-based) | Correct (no LLM in execution) |
| outcome_tracker | win/loss labels | Records only | Open feedback loop |
| agent_memories | BM25 retrieval | Loads at startup | No mid-session refresh |

The **single biggest gap**: `decide_trades()` is a single-pass function. Layer-1 says "BUY AAPL with 0.78 confidence"; `decide_trades` adds nothing — no second-look, no portfolio-level reasoning ("I'm already 35% tech, do I want more AAPL exposure?"), no macro overlay ("rates are spiking, downsize tech bets"), no risk-officer veto.

A trading MAS at this seam would close the gap.

---

## 3. Three architectural options

### Option Alpha — 5-agent file-based, harness-like

**Agents (5):** Lead Trader, Macro Analyst, Technical Analyst, Risk Judge, Execution Approver.
**Communication:** file-based handoff per ticker per cycle. Output: `trade_directive_<ticker>_<ts>.json`.
**Plug-in:** between Layer-1 synthesis and `decide_trades()`.
**Risk:** 5 Opus/Sonnet calls per ticker × 5 tickers per cycle = ~25 calls × ~$0.05 = **$1.25/cycle**. Daily cost ~$1.25 × 21 trading days = **$26/month** in LLM costs alone.
**Complexity:** high; file I/O latency; deterministic audit trail.

### Option Beta — 3-agent async in-process

**Agents (3 NEW):** Trader, Risk Officer, Synthesizer. Reads Layer-1 synthesis JSON, writes `trade_directive`.
**Communication:** async in-process; no file I/O.
**Plug-in:** `autonomous_loop.py:207-217` (between Step 5 mark-to-market and Step 6 `decide_trades`).
**Risk:** 3 calls per ticker × 5 = ~15 calls × ~$0.03 = **$0.45/cycle**. ~**$10/month** LLM costs.
**Complexity:** low; mirrors `multi_agent_orchestrator.py` patterns; minimal disruption to Layer-1.

### Option Gamma — Beta + Learner agent (closes feedback loop)

Adds **6th agent: Learner**. Reads `outcome_tracker` results (7/30/90-day windows), writes lessons to `agent_memories` BQ table. On next cycle, agents retrieve via BM25 — closing the open feedback loop.
**Risk:** **HIGHEST.** Self-improving loops can encode survivor bias, regime-overfitting, drift. Requires Q/A review of every memory write before persist. HedgeAgents (ACM 2025) reports +39.3% SR from memory wiring alone — the biggest reported lever — but the same paper warns about overfitting to bull-market windows.

---

## 4. Recommendation: a **REFINEMENT** of Option Beta

After deeper internal-code audit, the user's app already has 2 of the 3 Beta agents:

- **Analyst** — Layer-1's `synthesis` IS the analyst's output; consume it directly, don't re-analyze.
- **Risk Officer** — `Risk Judge` already exists in `risk_debate.py` and runs as part of Layer-1; we can extract its output into the trading MAS without re-prompting.
- **Fund Manager** (NEW) — the only net-new agent. Reads synthesis + risk_assessment + portfolio state, decides "given my current 35% tech exposure and macro=EASING, do I take this BUY for AAPL or pass?"

**Refined Beta = "Layer-1 + Existing Risk Judge + NEW Fund Manager."** One new agent. ~80 lines of code. ~$0.15/cycle ($3/month LLM). Plugs in at `autonomous_loop.py:207-217`.

**A/B harness:** flag `TRADING_MAS_ENABLED=false` (default OFF). When ON, route through Fund Manager. Compare DSR (NOT raw SR) over 6+ months. Lopez de Prado's overfit floor: discount any reported SR by `f(N_trials)` where `N_trials` = number of MAS configurations tested.

---

## 5. Estimated benefit (concrete)

**Single defensible MAS-vs-single-LLM benchmark in 2025-2026 literature:**

| Source | Year | Architecture | Baseline | MAS | Sharpe lift |
|--------|------|--------------|----------|-----|-------------|
| HedgeAgents (ACM 2025) | 2025 | 4 agents + 3 conferences (Budget, Experience, Emergency) | FinGPT single-LLM SR=1.93 | SR=2.41 | **+24.49%** |
| HedgeAgents memory ablation | 2025 | Same 4 agents, w/ vs w/o BM25 memory | SR=1.73 (no memory) | SR=2.41 (with memory) | **+39.3%** |
| TradingAgents (arXiv 2412.20138) | 2024 | 7 agents (analysts + Bull/Bear + Trader + Risk + Fund Manager) | (no published baseline) | SR=8.21 (AAPL 6mo bull) | not directly comparable; no DSR adjustment |
| FinRL Contest 2025 (arXiv 2504.02281) | 2025 | RL ensemble + LLM signal augmentation | individual RL agents | best ensemble SR=1.08 | ensemble outperforms individuals |

**The honest number:** **+10-25% Sharpe lift is plausible** given:
- HedgeAgents +24.49% is a single point on one benchmark, NOT a generalizable rate
- HedgeAgents memory ablation suggests **wire memory FIRST** if you can only do one thing — the biggest reported lever
- TradingAgents 8.21 SR is overfit (6-month bull window, no DSR)

**Cost estimate (refined Beta):** ~$0.15/cycle × 21 trading days = **~$3/month** in LLM costs. Vs. estimated alpha: 1bps daily lift on a $10k paper portfolio = ~$1/day = ~$20/month. **Marginal ROI is positive but slim.** This is research-grade, not production-grade alpha.

**Latency:** Beta adds ~100-200ms per ticker × 5 tickers = **<1s** to the daily cycle. The cycle already takes 49-93 seconds today; 1s overhead is noise.

---

## 6. Risk profile

### Risks specific to a trading MAS

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Overfit to bull windows** | HIGH | DSR gate, not raw SR. Min 6-month holdout. (Bailey & Lopez de Prado 2014; ScienceDirect 2024) |
| **Survivor bias from outcome-tracker** | HIGH | Track ALL recommendations (not just successful ones). Ground-truth W/L labels via `paper_round_trips`. |
| **Regime shift (Beta works in EASING, fails in HIKING)** | MEDIUM | Per-regime A/B (compare MAS vs single-LLM separately for EASING / HIKING / NEUTRAL macro states from FRED) |
| **Fund Manager hallucinates portfolio facts** | MEDIUM | Pass current portfolio state (positions, cash, NAV) as STRUCTURED input, not free-text. Include "you currently hold X% of $TICKER" as a literal field. |
| **Self-improving Gamma loop runs away** | HIGH (Gamma only) | Q/A review every memory write before persist. 30-day washout before learnings affect decisions. Drift detector (compare new memories' BM25 score distribution vs prior week). |
| **Cost runaway** | LOW | Beta is $0.15/cycle. Cap via existing `cost_budget_api.py` ($5 daily / $50 monthly). |
| **Claude/Gemini rate limit blocks daily cycle** | MEDIUM | Existing `llm_client::make_client` multi-provider routing already handles this. Beta's NEW Fund Manager calls go through the same routing. |

### Risks specific to *NOT* building it

- **Open feedback loop**: outcome_tracker records data but no agent reads it for future decisions. The system currently records its own ignorance.
- **No portfolio-level reasoning**: every BUY decision is local-to-ticker, no macro/concentration overlay.
- **Existing 28-agent pipeline produces signals, but `decide_trades` discards their nuance**: confidence=0.78 vs confidence=0.55 are treated identically by the rule-based path.

---

## 7. Plug-in point — concrete file:line

**`backend/services/autonomous_loop.py:207-217`** (between Step 5 mark-to-market and Step 6 `decide_trades`):

```python
# CURRENT
candidate_analyses = await asyncio.gather(*analyze_tasks)  # Layer-1 outputs
mark_to_market(holdings)
trades = decide_trades(candidate_analyses, holdings, ...)  # rule-based

# WITH BETA-REFINED MAS (TRADING_MAS_ENABLED=true)
candidate_analyses = await asyncio.gather(*analyze_tasks)
mark_to_market(holdings)
if settings.trading_mas_enabled:
    candidate_analyses = await fund_manager_review(
        candidate_analyses, holdings, macro_signal=fred_macro_signal,
    )  # NEW: Fund Manager re-ranks / vetoes / adjusts confidence
trades = decide_trades(candidate_analyses, holdings, ...)
```

`fund_manager_review()` is a **new ~80-line async function** in `backend/services/fund_manager.py` (new file). It:
1. Reads each candidate's synthesis JSON (existing Layer-1 output)
2. Reads current `holdings` + cash + NAV
3. Reads `fred_macro_signal` (already retrieved by autonomous_loop)
4. Calls one Claude/Gemini message with structured prompt: "Given portfolio state, candidates, and macro, return per-ticker {action, position_pct_adjustment, veto_reason} JSON"
5. Returns updated `candidate_analyses` with adjusted confidence/position
6. Logs the Fund Manager's reasoning to `handoff/fund_manager_decisions.jsonl` (audit trail)

---

## 8. Industry precedent

| Source | Pattern | Output | Validates which option? |
|--------|---------|--------|-------------------------|
| Anthropic, "How we built our multi-agent research system" (2025) | Lead + parallel subagents + file-based handoff + Citation Agent | research synthesis | Alpha (file-based) ✓ |
| Anthropic, "Harness Design for Long-Running Apps" (2025) | Single-provider serial harness | long-running app outputs | Beta (in-process) ✓ |
| TradingAgents (arXiv 2412.20138) | 7-agent: analysts + Bull/Bear + Trader + Risk + Fund Manager | daily BUY/SELL/HOLD | **Refined Beta** ✓ (Fund Manager is the new agent) |
| HedgeAgents (arXiv 2502.13165, ACM 2025) | 4-agent + 3 conferences + memory | multi-asset rebalance | Memory wiring (Gamma) — biggest lever |
| FinRL Contest 2025 | RL ensemble + LLM signal augmentation | stock + crypto allocations | Ensemble validates A/B framework |
| Public.com Agentic Brokerage (2026) | Production agentic brokerage for retail | portfolio automations | Validates plug-in-between-signal-and-execution |
| Galileo LLM-as-Judge (2025) | Multi-judge ensemble for compliance | risk approval | Validates Fund Manager as judge-style agent |

**Strongest validators of Refined Beta:** TradingAgents (Fund Manager pattern) + HedgeAgents (memory wiring as ROI lever). Strongest counter-evidence: Public.com is in production but publishes no SR numbers — black box.

---

## 9. Q/A discipline cost

If we ship Refined Beta, the Fund Manager's outputs become a **new decision surface** that the harness MAS must audit. Specifically:

- Each Fund Manager decision should append to `handoff/fund_manager_decisions.jsonl` (audit trail)
- Daily Q/A review (or weekly) spot-checks: did the Fund Manager's veto reasons hold up post-trade? Did it concentrate / de-concentrate the portfolio appropriately?
- Memory writes (if Gamma is added later) require harness Q/A review before persist — this is non-negotiable per Anthropic harness design

**Estimated Q/A overhead:** ~30 min/week of Q/A subagent time for spot-check. Same protocol as the masterplan harness.

---

## 10. Recommendation

**Ship Refined Beta** (Fund Manager only, A/B flag default OFF) **as a follow-up cycle**, NOT today. Specifically:

1. **NOT for Monday paper-trading day-1.** Untested-in-production reasoning layer between signal and execution. Risk > reward at launch.
2. **Build the skeleton (~80 lines, 1-2 days)** when paper-trading is steady-state (≥2 weeks of clean cycles).
3. **A/B with `TRADING_MAS_ENABLED=true` for 6 weeks** vs the rule-based baseline. Measure: directional accuracy, DSR, max drawdown — NOT raw SR.
4. **If A/B PASSes (DSR ≥ 0.95)** flip default to ON.
5. **Wire memory (Gamma)** only AFTER Refined Beta has 6 months of clean data. Gamma is the highest-ROI lever (HedgeAgents +39.3% SR) but also highest risk.

### NOT recommended now
- **Alpha** (5 agents file-based) — over-engineered for current scale; defer until portfolio is multi-strategy
- **Gamma** (learner loop) — defer 6 months; needs production data first
- **TradingAgents-style 7-agent** — more agents ≠ more alpha; HedgeAgents 4-agent beats it on the same paper-trading universe

### Smallest first step

`backend/services/fund_manager.py` (new file, ~80 lines):

```python
async def fund_manager_review(
    candidate_analyses: list[dict],
    holdings: dict,
    macro_signal: str,
    cost_tier: str = "build",
) -> list[dict]:
    """Refined-Beta Fund Manager: re-rank/veto candidates given portfolio + macro.
    Returns updated candidate_analyses with adjusted confidence + optional veto.
    Audit trail: handoff/fund_manager_decisions.jsonl.
    """
    if not get_settings().trading_mas_enabled:
        return candidate_analyses  # A/B flag OFF: passthrough
    # ... LLM prompt construction + structured-output JSON ...
    # ... append decision to JSONL ...
    return updated_candidate_analyses
```

Plug into `autonomous_loop.py:207-217` per section 7.

---

## 11. What this evaluation does NOT do

- **Does NOT build the trading MAS.** Code-only file is `docs/architecture/trading-mas-evaluation.md` (this doc).
- **Does NOT enable any flag.** `TRADING_MAS_ENABLED` does not exist as a setting yet. Adding it is a follow-up.
- **Does NOT promise alpha.** The +24.49% Sharpe figure is from HedgeAgents on their benchmark, not pyfinagent.
- **Does NOT close phase-10.7.x** (Alpha Velocity, Recursive Prompt Optimization, Cron Budget Allocator) — those are different work tracks. This doc is precursor evidence for whether 10.7-style learning-loop investments are worth the complexity.

---

## 12. Sources

Industry / academic:
- TradingAgents (Yang et al., arXiv 2412.20138, 2024)
- HedgeAgents (ACM Web Conference 2025, arXiv 2502.13165)
- FinRL Contests Benchmark (arXiv 2504.02281, 2025)
- LLM-as-a-Judge in Financial Services (Galileo 2025)
- Public.com Agentic Brokerage (PR Newswire 2026)
- The Probability of Backtest Overfitting (Bailey & Lopez de Prado, SSRN 2326253, 2014)
- Backtest Overfitting in the ML Era (ScienceDirect 2024)
- Anthropic, "How we built our multi-agent research system" (2025)
- Anthropic, "Harness Design for Long-Running Apps" (2025)

Internal anchors:
- `backend/services/autonomous_loop.py:207-217` — plug-in point
- `backend/services/portfolio_manager.py::decide_trades` — current rule-based logic
- `backend/agents/multi_agent_orchestrator.py` — existing in-app MAS pattern (4 agents)
- `backend/agents/orchestrator.py::AnalysisOrchestrator.run_full_analysis` — Layer-1 28-agent pipeline output
- `.claude/agents/researcher.md`, `.claude/agents/qa.md` — harness MAS pattern (3-agent file-based serial)
- `handoff/current/phase-16.27-research-brief.md` — research brief that fed this evaluation
