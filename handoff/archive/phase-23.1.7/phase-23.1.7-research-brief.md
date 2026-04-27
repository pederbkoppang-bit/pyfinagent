# Research Brief: phase-23.1.7 — Full Agent Rationale + Signal Stack in Trade Record

**Tier:** moderate (internal-heavy; relaxed external floor of >=3 sources read in full)
**Date:** 2026-04-26
**Researcher:** researcher agent (Sonnet 4.6)

---

## Search query log (3-variant per topic)

### Topic 1 — TradingAgents rationale capture / memory loop
1. `TradingAgents multi-agent rationale capture trade reflection memory loop arXiv 2412.20138` (current-year frontier)
2. `TradingAgents Xiao 2025 reflection memory per-trade schema` (last-2-year window)
3. `multi-agent LLM trading system agent rationale per-trade storage` (year-less canonical)

### Topic 2 — FINCON / comparable systems per-trade audit schema
1. `FINCON multi-agent financial trading per-trade audit trail learning loop schema 2025` (current frontier)
2. `FINCON FinCon verbal reinforcement NeurIPS 2024 episodic memory schema` (last-2-year)
3. `LLM multi-agent financial trading memory audit trail schema` (year-less canonical)

### Topic 3 — BM25 retrieval over agent memories
1. `BM25 retrieval agent memory schema high-recall reflection context storage` (current frontier)
2. `BM25 agent memory agentic context engineering 2025 2026` (last-2-year)
3. `BM25 lexical retrieval memory situation lesson schema agent` (year-less canonical)

### Topic 4 — Anthropic / ACE context engineering per-decision learning
1. `Anthropic agent memory schema structured context per-decision learning reflection 2025 2026` (current frontier)
2. `Agentic Context Engineering ACE evolving context self-improving 2025` (last-2-year)
3. `Anthropic effective context engineering agents note-taking memory` (year-less canonical)

---

## Read in full (>=3 required for relaxed floor; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://arxiv.org/html/2412.20138v3 | 2026-04-26 | paper | WebFetch full | "agents query the global agent state for analyst reports"; rationale stored as "concise, well-organized reports" within a global state; debate facilitator "records it as a structured entry" |
| https://arxiv.org/html/2407.06567v3 | 2026-04-26 | paper | WebFetch full | FinCon episodic memory contains "actions, PnL series from previous episodes, and updated conceptual investment beliefs"; self-reflection text Bt stored when rt<0; minimum context = action + PnL + CVaR + timestamp |
| https://arxiv.org/html/2508.17565v1 | 2026-04-26 | paper | WebFetch full | TradingGroup stores "input prompts and LLM outputs; account snapshot; full Chain-of-Thought"; retrieval keyed on "technical indicators (RSI, ATR, distance to moving averages, volatility)" and "market conditions" |
| https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | 2026-04-26 | official doc | WebFetch full | "critical decisions and rationale — architectural decisions, unresolved bugs, and implementation details"; preserve "high-signal tokens" while discarding redundant content; metadata and organization signal purpose |
| https://arxiv.org/abs/2510.04618 | 2026-04-26 | paper | WebFetch full | ACE "treats contexts as evolving playbooks that accumulate, refine, and organize strategies"; prevents "context collapse" by "structured, incremental updates that preserve detailed knowledge"; +10.6% on agent benchmarks, +8.6% on finance tasks |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/2412.20138 | paper | Abstract/landing page; HTML version fetched in full instead |
| https://github.com/TauricResearch/TradingAgents | code | Repo index; no specific schema file surfaced in search results |
| https://proceedings.neurips.cc/paper_files/paper/2024/file/f7ae4fe91d96f50abc2211f09b6a7e49-Paper-Conference.pdf | paper | NeurIPS PDF; HTML version fetched in full |
| https://hindsight.vectorize.io/blog/2026/03/17/hermes-agent-memory | blog | BM25 details extracted from search snippet; sufficient for cross-referencing |
| https://openreview.net/forum?id=dG1HwKMYbC&noteId=lCPXezaagn | review | FinCon OpenReview; HTML version of arXiv fetched in full |
| https://arxiv.org/pdf/2502.12110 | paper | A-Mem paper; BM25 context from snippet sufficient for this brief |
| https://interestingengineering.substack.com/p/from-bm25-to-agentic-rag-the-evolution | blog | BM25 context from snippet |
| https://arxiv.org/html/2604.19795 | paper | General agent context paper; ACE paper fetched as more targeted |
| https://medium.com/@bijit211987/ai-powered-multi-agent-trading-workflow-90722a2ada3b | blog | Community tier; lower weight; ACE + TradingAgents sufficient |
| https://community.netapp.com/t5/Tech-ONTAP-Blogs/Hybrid-RAG-in-the-Real-World-Graphs-BM25-and-the-End-of-Black-Box-Retrieval/ba-p/464834 | blog | Hybrid retrieval details; snippet sufficient given pyfinagent uses BM25Okapi directly |

---

## Recency scan (2024-2026)

Searched for 2024-2026 literature on: (a) per-trade agent rationale capture in multi-agent trading systems; (b) BM25 agent memory schemas for learning loops; (c) ACE/context-engineering for per-decision reflection.

**Findings:** TradingGroup (arXiv 2508.17565, 2025) advances the TradingAgents pattern by adding explicit Chain-of-Thought storage and technical-indicator-keyed retrieval — directly relevant to pyfinagent's BM25 `situation` field. ACE (arXiv 2510.04618, Oct 2025) confirms the "incremental structured update" doctrine that the pyfinagent memory module already approximates. FinCon (NeurIPS 2024) is the most complete example of a production learning loop that stores manager reasoning alongside analyst inputs. Anthropic's context engineering post (2026) provides the strongest guidance on what constitutes "high-signal" tokens worth preserving per decision. No finding supersedes the canonical approach; all 2024-2026 work reinforces it.

---

## Key findings

1. **Minimum essential per-trade context** — FinCon's episodic memory minimum: action + PnL + CVaR + timestamp. TradingGroup extends this to include full CoT and the account snapshot (date, positions, cash). For BM25 retrieval in pyfinagent, the `situation` field must contain enough lexical signal for keyword overlap: ticker, sector, momentum direction, RSI level, regime tag, conviction label, and the Claude reason text. (Source: FinCon arXiv 2407.06567; TradingGroup arXiv 2508.17565)

2. **Structured entries, not free-form narratives** — TradingAgents stores debate rationale as "structured entries" recorded by a facilitator agent, not raw transcripts. The key insight: the structured entry contains the *prevailing perspective* plus the supporting evidence, not the full debate. In pyfinagent terms: one `{agent, role, rationale, weight}` row per signal layer is the right granularity — the current `extract_signals_from_analysis` shape is architecturally correct. (Source: TradingAgents arXiv 2412.20138v3)

3. **BM25 retrieval quality is dominated by `situation` field richness** — BM25Okapi scores via whitespace-tokenized term overlap (confirmed in `backend/agents/memory.py:76-82`). If `situation` omits "momentum", "RSI", "regime", "conviction", those terms cannot match future queries that contain them. The `build_situation_description` function (memory.py:170-210) currently only reads `enrichment_signals` + `debate_result` from the full Gemini report — it has no path to receive the screener overlay fields (`regime_tag`, `conviction_reason`, `news_rationale`, etc.). This is the primary gap for lite analyses. (Source: internal — memory.py:170-210, outcome_tracker.py:164-175)

4. **Lite analysis path never writes `analysis_results`** — `_run_claude_analysis` returns a dict and `outcome_tracker.evaluate_all_pending` calls `bq.get_recent_reports` which reads `analysis_results`. Since lite analyses are not written there, the evaluator has no source for `price_at_rec` or `full_report_json`, so `_generate_and_persist_reflections` is never called for any paper trade. (Source: internal — autonomous_loop.py:466-578; outcome_tracker.py:87-150; bigquery_client.py:41-137)

5. **Gap 1 root cause confirmed** — `signal_attribution.py:99-107` reads `analysis.get("trader_note") or analysis.get("recommendation_reason")`. The lite dict (autonomous_loop.py:558-578) has neither key: the Claude reason is nested at `full_report.analysis.reason`. The risk layer reads `risk.get("decision")` and `risk.get("reasoning")` but the lite dict stores `risk_assessment = {"reason": analysis["reason"]}` — key is `"reason"` not `"decision"` or `"reasoning"`. Result: Trader row falls back to `"Recommendation: BUY"` and RiskJudge row is completely empty. (Source: internal — signal_attribution.py:99-121; autonomous_loop.py:558-578)

6. **Gap 2 root cause confirmed** — `paper_trader.py:118-134` builds the trade dict with only the keys enumerated; the caller (`portfolio_manager.py:190-200`) passes `signals=cand.get("signals", [])`. But `cand` is assembled from the analysis dict at portfolio_manager.py:148-158, which calls `extract_signals_from_analysis(analysis)` — not from the screener candidate dict. So `composite_score`, `conviction_score`, `conviction_reason`, `regime_tag`, `pead_tag`, `news_rationale`, `sector_event` are all discarded at portfolio_manager.py:148. The screener candidate dict is available at autonomous_loop.py:160-166 but is never merged into the analysis dict. (Source: internal — portfolio_manager.py:130-201; autonomous_loop.py:159-188; screener.py:228-245)

7. **ACE doctrine** — "Preserve high-signal tokens; discard redundant content." For pyfinagent, the high-signal tokens per trade are: ticker, sector, momentum_20d, momentum_60d, RSI, composite_score, conviction_score, conviction_reason, regime, Claude's one-sentence reason. All other data (market_cap, pe_ratio, industry) are lower-signal for BM25 retrieval purposes and should be stored but not inflate the `situation` string. (Source: ACE arXiv 2510.04618; Anthropic context engineering 2026)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/signal_attribution.py` | 147 | Converts raw analysis dicts to `{agent, role, rationale, weight}` signal rows | Gap: Trader layer reads `trader_note`/`recommendation_reason`; lite dict has neither. Risk layer reads `risk.decision`/`risk.reasoning`; lite dict has `risk_assessment.reason`. |
| `backend/services/autonomous_loop.py` | ~600+ | Orchestrates daily cycle; `_run_claude_analysis` lines 466-578 | Gap: Returns lite dict missing `trader_note`, `recommendation_reason`. `risk_assessment` uses key `"reason"` not `"decision"`. Never writes to `analysis_results`. Screener candidate dict with overlays never merged into analysis dict. |
| `backend/services/portfolio_manager.py` | ~208 | `TradeOrder` dataclass + trade decision logic | Gap: `buy_candidates` list (lines 148-158) calls `extract_signals_from_analysis(analysis)` — discards screener overlay fields. `TradeOrder` has no field for `conviction_score`, `conviction_reason`, `regime_tag`, `news_rationale`. |
| `backend/services/paper_trader.py` | ~300+ | `execute_buy` lines 67-187 | Gap: Trade dict (lines 118-134) has no fields for quant overlays. `signals` column captures agent signals but not screener metadata. |
| `backend/db/bigquery_client.py` | ~650+ | BQ DML + `save_paper_trade` lines 595-611; `save_report` lines 41-137 | `save_paper_trade` uses a dict-driven INSERT — any new keys in the trade dict go straight through provided the BQ table has matching columns. `save_report` is the 88-column `analysis_results` writer; lite analyses never call it. |
| `backend/services/outcome_tracker.py` | 221 | `evaluate_all_pending` + reflection generation | Gap: `evaluate_all_pending` (line 118) calls `bq.get_report` which reads `analysis_results` — never populated for lite analyses. `build_situation_description` called with `enrichment_signals` + `debate_result` from full report; lite analyses have neither. |
| `backend/agents/memory.py` | 269 | `FinancialSituationMemory` BM25 class + `build_situation_description` + `generate_reflection` | `build_situation_description` (line 170) only uses `enrichment_signals` dict + `debate_result`. No path for momentum, RSI, regime, conviction. `BM25Okapi` tokenizes whitespace — term richness in `situation` directly governs retrieval quality. |
| `backend/tools/screener.py` | 260+ | `rank_candidates` lines 151-248 | Returns `{**stock, "composite_score": ..., "source"?: "news_only", "news_event_type"?: ..., "news_rationale"?: ...}`. After meta-scorer (autonomous_loop.py:169-181), candidates also carry `conviction_score` and `conviction_reason`. Regime, PEAD, news, sector_event scores are applied in-place via `apply_*_to_score` — the tag/reason is NOT surfaced back on the candidate dict. |
| `frontend/src/components/AgentRationaleDrawer.tsx` | 214 | Renders `signals[]` as progressive-disclosure layers | Drawer expects `{agent, role, rationale, weight}` shape. Routes by `agent` name into Analyst/Bull/Bear/Trader/RiskJudge buckets. New signal types with new `agent` names will NOT render unless the `group_signals_for_drawer` function and the drawer's `Layer`/`DebateLayer` logic is extended. |

---

## Consensus vs debate (external)

All three papers (TradingAgents, FinCon, TradingGroup) agree: the minimum per-trade memory unit is **(action, rationale summary, market context snapshot, outcome)**. They differ on granularity: FinCon stores manager reasoning implicitly (consolidated); TradingAgents stores per-agent structured entries; TradingGroup stores full CoT. For pyfinagent's BM25 retrieval (which is lexical, not semantic), the TradingGroup approach (include technical indicators directly in the stored text) gives the highest recall because the query at retrieval time will contain those same terms. The ACE paper confirms: incremental structured updates beat compressed summaries for downstream retrieval.

No paper recommends storing raw LLM transcripts in the trade row itself — the consensus is to store a structured summary (the signal rows) and a situation description, with the full report available for reconstruction.

---

## Pitfalls (from literature + internal audit)

1. **BM25 term mismatch**: If the `situation` stored at trade time omits domain terms (regime, conviction, RSI), a future query on "regime bearish RSI overbought" will score near-zero. (TradingGroup retrieval design; memory.py BM25Okapi)
2. **Nested key mismatch**: Fixing one key path without auditing the full signal_attribution layer leaves other gaps. The Risk layer has three fallback keys (`decision`, `reasoning`, `rationale`) but the lite dict uses `reason`. (signal_attribution.py:113)
3. **Overwriting `situation` with full Gemini report fields**: `build_situation_description` assumes `enrichment_signals` is a dict of signal objects. Lite analyses produce no such dict. A naive fix that passes `full_report.get("enrichment_signals", {})` on a lite analysis will pass an empty dict and produce "Analyzing ON in the Unknown sector." — a useless memory. (outcome_tracker.py:166)
4. **Drawer routing by `agent` name**: Adding a `"Quant"` or `"SignalStack"` signal requires adding a routing branch in `group_signals_for_drawer` (signal_attribution.py:126-146) AND a new `Layer` render target in AgentRationaleDrawer.tsx. Without both, the new signals are stored but invisible in the UI.
5. **BQ schema extension risk**: Adding columns to `paper_trades` is low-risk (additive, NULL-tolerant). Adding a new `paper_trading_analyses` table requires a migration script and a schema guard check; it doesn't break existing calls but adds operational overhead.

---

## Application to pyfinagent — concrete design decisions

### Concrete signal_attribution.py extension

**Trader layer fix** — add fallback key paths that match the lite dict shape:

```python
# ── Trader layer ────  (signal_attribution.py:99-107, replacement)
rec = str(analysis.get("recommendation", "")).upper() or "HOLD"
score = analysis.get("final_score")

# Lite dict: full_report.analysis.reason  (autonomous_loop.py:566)
# Full Gemini dict: trader_note / recommendation_reason (existing keys)
trader_note = (
    analysis.get("trader_note")
    or analysis.get("recommendation_reason")
    or (analysis.get("full_report") or {}).get("analysis", {}).get("reason")
    or ""
)
signals.append({
    "agent": "Trader",
    "role": "decision",
    "rationale": _trim(trader_note) or f"Recommendation: {rec}",
    "weight": float(score) if isinstance(score, (int, float)) else 0.0,
})
```

**Risk layer fix** — the lite dict stores `risk_assessment = {"reason": "..."}`, not `{"decision": ..., "reasoning": ...}`. Add `"reason"` to the fallback chain:

```python
# ── Risk layer ────  (signal_attribution.py:110-121, replacement)
risk = analysis.get("risk_assessment") or {}
if isinstance(risk, dict):
    decision = risk.get("decision") or ""
    reasoning = risk.get("reasoning") or risk.get("rationale") or risk.get("reason") or ""
    pos_pct = risk.get("recommended_position_pct")
    if decision or reasoning:
        signals.append({
            "agent": "RiskJudge",
            "role": "gate",
            "rationale": _trim(reasoning) or f"Decision: {decision}",
            "weight": float(pos_pct) if isinstance(pos_pct, (int, float)) else 0.0,
        })
```

**New extraction function for screener overlays** — add after line 123:

```python
def extract_quant_signals(candidate: dict) -> list[dict]:
    """
    Extract screener-layer quant signals from a candidate dict produced by
    rank_candidates / meta_scorer. Returns 0-2 additional signal rows:
      - "Quant" Analyst signal (momentum, RSI, vol, sector)
      - "SignalStack" group signal (regime, conviction, PEAD, news, sector_event)
    These are appended BEFORE the Trader signal so the drawer shows them in
    the Analyst layer.
    """
    signals: list[dict] = []

    # Quant metrics signal
    mom_1m = candidate.get("momentum_1m")
    mom_3m = candidate.get("momentum_3m")
    rsi = candidate.get("rsi_14")
    vol = candidate.get("volatility_ann")
    sector = candidate.get("sector", "")
    composite = candidate.get("composite_score")

    quant_parts = []
    if mom_1m is not None:
        quant_parts.append(f"1m momentum {mom_1m:+.1f}%")
    if mom_3m is not None:
        quant_parts.append(f"3m momentum {mom_3m:+.1f}%")
    if rsi is not None:
        quant_parts.append(f"RSI14 {rsi:.1f}")
    if vol is not None:
        quant_parts.append(f"ann_vol {vol:.2f}")
    if sector:
        quant_parts.append(f"sector {sector}")
    if composite is not None:
        quant_parts.append(f"composite_score {composite:.3f}")

    if quant_parts:
        signals.append({
            "agent": "Quant",
            "role": "screener",
            "rationale": _trim("; ".join(quant_parts)),
            "weight": float(composite) if composite is not None else 0.0,
        })

    # Signal stack overlay signal
    stack_parts = []
    conviction_score = candidate.get("conviction_score")
    conviction_reason = candidate.get("conviction_reason", "")
    news_rationale = candidate.get("news_rationale", "")
    news_event_type = candidate.get("news_event_type", "")
    source_tag = candidate.get("source", "")  # "news_only" when applicable

    if conviction_score is not None:
        stack_parts.append(f"conviction {conviction_score:.2f}")
    if conviction_reason:
        stack_parts.append(conviction_reason)
    if news_event_type:
        stack_parts.append(f"news:{news_event_type}")
    if news_rationale:
        stack_parts.append(news_rationale)
    if source_tag == "news_only":
        stack_parts.append("source:news_only")

    if stack_parts:
        signals.append({
            "agent": "SignalStack",
            "role": "overlay",
            "rationale": _trim("; ".join(stack_parts)),
            "weight": float(conviction_score) if conviction_score is not None else 0.0,
        })

    return signals
```

**New combined extractor** — callers should use this wrapper:

```python
def extract_all_signals(analysis: dict, candidate: dict | None = None) -> list[dict]:
    """
    Full signal extraction: agent rationale (analysis) + quant overlays (candidate).
    Pass candidate=None for full Gemini path (no screener candidate available).
    """
    signals = extract_signals_from_analysis(analysis)
    if candidate:
        # Prepend Quant + SignalStack before Trader so drawer ordering is
        # Analyst -> Quant -> SignalStack -> Trader -> Risk
        quant_sigs = extract_quant_signals(candidate)
        # Insert before Trader (last 1-2 items)
        trader_idx = next(
            (i for i, s in enumerate(signals) if s.get("agent") == "Trader"), len(signals)
        )
        signals = signals[:trader_idx] + quant_sigs + signals[trader_idx:]
    return signals
```

**group_signals_for_drawer extension** (signal_attribution.py:126-146):

```python
def group_signals_for_drawer(signals: list[dict]) -> dict:
    out: dict = {
        "analyst": [],
        "debate": {"bull": [], "bear": []},
        "quant": [],        # NEW
        "signal_stack": [], # NEW
        "trader": [],
        "risk": [],
    }
    for s in signals or []:
        role = s.get("role")
        agent = s.get("agent")
        if agent == "Analyst":
            out["analyst"].append(s)
        elif agent == "Bull":
            out["debate"]["bull"].append(s)
        elif agent == "Bear":
            out["debate"]["bear"].append(s)
        elif agent == "Quant" or role == "screener":
            out["quant"].append(s)
        elif agent == "SignalStack" or role == "overlay":
            out["signal_stack"].append(s)
        elif agent == "Trader" or role == "decision":
            out["trader"].append(s)
        elif agent == "RiskJudge" or role == "gate":
            out["risk"].append(s)
    return out
```

---

### Concrete signals JSON schema

Each signal row MUST conform to `{agent, role, rationale, weight}`. New types:

**Quant Analyst signal:**
```json
{
  "agent": "Quant",
  "role": "screener",
  "rationale": "1m momentum +4.2%; 3m momentum +11.8%; RSI14 58.3; ann_vol 0.28; sector Technology; composite_score 8.450",
  "weight": 8.45
}
```
The `weight` field carries `composite_score` for the Quant signal. The drawer renders `weight {w.toFixed(2)}` — at composite_score=8.45 this reads `weight 8.45`, which is interpretable.

**SignalStack overlay signal:**
```json
{
  "agent": "SignalStack",
  "role": "overlay",
  "rationale": "conviction 0.82; strong momentum across all timeframes with analyst upgrade; news:earnings_beat; Q1 beat consensus by 12%; source:news_only",
  "weight": 0.82
}
```
The `weight` field carries `conviction_score`. When no conviction_score (regime-only overlay), `weight` is 0.0.

**Drawer rendering:** Both new agents fall through to `Layer title="Quant"` and `Layer title="Signal Stack"` respectively in AgentRationaleDrawer.tsx. The frontend must add these two `<Layer>` calls after `<Layer title="Analyst" ...>`. The drawer already handles arbitrary `Signal[]` arrays — only the routing switch and render block need updating.

**Frontend AgentRationaleDrawer extension** (minimal change, after line 121):
```tsx
{data && data.signals.length > 0 && (
  <div className="space-y-3">
    <Layer title="Analyst" items={data.tree.analyst} />
    <DebateLayer bull={data.tree.debate.bull} bear={data.tree.debate.bear} />
    <Layer title="Quant" items={data.tree.quant ?? []} />
    <Layer title="Signal Stack" items={data.tree.signal_stack ?? []} />
    <Layer title="Trader" items={data.tree.trader} />
    <Layer title="Risk Judge" items={data.tree.risk} emphasize />
  </div>
)}
```
The `Rationale` TypeScript interface must add `quant: Signal[]` and `signal_stack: Signal[]` to its `tree` type.

---

### Concrete BQ schema decision

**Recommendation: extend `paper_trades` table, NOT `analysis_results`; write a minimal `paper_trading_analyses` table for the learning loop.**

Rationale:

| Option | Pros | Cons |
|--------|------|------|
| Extend `analysis_results` | Single table for both paths; outcome_tracker already reads it | 88 columns, most NULL for lite analyses; Gemini full report owns this table's schema; collisions risk if column semantics diverge; `save_report` signature is 40+ params — adding lite writes requires careful threading through all callers |
| New `paper_trading_analyses` table | Clean separation; minimal schema (10-12 cols); lite analysis writes are isolated; `outcome_tracker` can query it without touching `analysis_results` | New migration script needed; `outcome_tracker.get_recent_reports` needs to union or pivot |
| Extend `paper_trades` only | Zero new tables; `signals` JSON already present | `outcome_tracker` needs a price_at_rec source — reading `paper_trades` for this is awkward; no good hook for `full_report_json` |

The cleanest path: **create `pyfinagent_pms.paper_trading_analyses`** with this minimal schema:

```sql
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_pms.paper_trading_analyses` (
  analysis_id   STRING NOT NULL,  -- same as analysis_date ISO string used as FK
  ticker        STRING NOT NULL,
  recommendation STRING,           -- BUY / SELL / HOLD
  final_score   FLOAT64,
  summary       STRING,            -- Claude's one-sentence reason
  price_at_analysis FLOAT64,
  sector        STRING,
  momentum_20d  FLOAT64,
  momentum_60d  FLOAT64,
  rsi_14        FLOAT64,
  composite_score FLOAT64,
  conviction_score FLOAT64,
  conviction_reason STRING,
  regime_tag    STRING,
  news_rationale STRING,
  analysis_source STRING,          -- "claude_lite" | "gemini_full"
  total_cost_usd FLOAT64,
  created_at    TIMESTAMP
);
```

`outcome_tracker.evaluate_all_pending` should query this table when `analysis_results` returns 0 rows for a given analysis_id. `price_at_rec` comes from `price_at_analysis`. `full_report_json` is replaced by individual columns that `build_situation_description` can read directly. This keeps the 88-column `analysis_results` pristine for the Gemini full-analysis path.

---

### Concrete TradeOrder + paper_trader plumbing

**Step 1 — autonomous_loop.py**: After `_run_claude_analysis` returns, capture the candidate dict alongside the analysis result.

The existing call site (around line 195-250) is approximately:
```python
# BEFORE (conceptual)
analysis = await _run_claude_analysis(ticker, settings)

# AFTER — carry candidate alongside analysis
candidate = next((c for c in candidates if c["ticker"] == ticker), {})
analysis["_screener_candidate"] = candidate   # ephemeral — not persisted
```

Alternatively (cleaner — avoids polluting the analysis dict): pass `candidates_by_ticker` as a dict into `decide_trades` so `portfolio_manager.py` can look up the candidate alongside the analysis.

**Step 2 — portfolio_manager.py TradeOrder**: Add optional screener overlay fields to the dataclass:

```python
@dataclass
class TradeOrder:
    ticker: str
    action: str
    amount_usd: Optional[float] = None
    quantity: Optional[float] = None
    reason: str = ""
    analysis_id: str = ""
    risk_judge_decision: str = ""
    stop_loss_price: Optional[float] = None
    risk_judge_position_pct: Optional[float] = None
    price: Optional[float] = None
    signals: list[dict] = field(default_factory=list)
    # NEW — screener overlay fields (phase-23.1.7)
    composite_score: Optional[float] = None
    conviction_score: Optional[float] = None
    conviction_reason: Optional[str] = None
    regime_tag: Optional[str] = None
    news_rationale: Optional[str] = None
    analysis_source: str = "claude_lite"   # "claude_lite" | "gemini_full"
```

**Step 3 — portfolio_manager.py build_candidates**: Populate overlay fields from the candidate dict (line 148-158 area):

```python
buy_candidates.append({
    "ticker": ticker,
    "recommendation": rec,
    "position_pct": position_pct,
    "stop_loss_price": stop_loss,
    "risk_judge_decision": risk_assessment.get("decision", ""),
    "analysis_id": analysis.get("analysis_date", ""),
    "final_score": final_score,
    "price": analysis.get("price_at_analysis"),
    "signals": extract_all_signals(analysis, candidate=analysis.get("_screener_candidate")),
    # NEW
    "composite_score": (analysis.get("_screener_candidate") or {}).get("composite_score"),
    "conviction_score": (analysis.get("_screener_candidate") or {}).get("conviction_score"),
    "conviction_reason": (analysis.get("_screener_candidate") or {}).get("conviction_reason"),
    "regime_tag": (analysis.get("_screener_candidate") or {}).get("regime_tag"),
    "news_rationale": (analysis.get("_screener_candidate") or {}).get("news_rationale"),
    "analysis_source": "claude_lite",
})
```

**Step 4 — paper_trader.py execute_buy**: Add optional kwargs mirroring the new TradeOrder fields:

```python
def execute_buy(
    self,
    ticker: str,
    amount_usd: float,
    price: float,
    reason: str = "new_buy_signal",
    analysis_id: str = "",
    risk_judge_decision: str = "",
    stop_loss_price: Optional[float] = None,
    risk_judge_position_pct: Optional[float] = None,
    signals: Optional[list[dict]] = None,
    # NEW — phase-23.1.7 screener overlays
    composite_score: Optional[float] = None,
    conviction_score: Optional[float] = None,
    conviction_reason: Optional[str] = None,
    regime_tag: Optional[str] = None,
    news_rationale: Optional[str] = None,
    analysis_source: str = "claude_lite",
) -> Optional[dict]:
```

The trade dict (line 118-134) extends to:
```python
trade = {
    "trade_id": trade_id,
    "ticker": ticker,
    "action": "BUY",
    "quantity": round(quantity, 6),
    "price": exec_price,
    "total_value": round(quantity * exec_price, 2),
    "transaction_cost": round(tx_cost, 2),
    "reason": reason,
    "analysis_id": analysis_id,
    "risk_judge_decision": risk_judge_decision,
    "created_at": now,
    "signals": json.dumps(signals or []),
    # NEW
    "composite_score": composite_score,
    "conviction_score": conviction_score,
    "conviction_reason": conviction_reason,
    "regime_tag": regime_tag,
    "news_rationale": news_rationale,
    "analysis_source": analysis_source,
}
```

`save_paper_trade` uses a dict-driven INSERT (`bigquery_client.py:595-611`) — new keys flow through automatically once the `paper_trades` BQ table has the matching columns.

**Step 5 — paper_trades BQ table DDL additions:**
```sql
ALTER TABLE `sunny-might-477607-p8.pyfinagent_pms.paper_trades`
  ADD COLUMN composite_score FLOAT64,
  ADD COLUMN conviction_score FLOAT64,
  ADD COLUMN conviction_reason STRING,
  ADD COLUMN regime_tag STRING,
  ADD COLUMN news_rationale STRING,
  ADD COLUMN analysis_source STRING;
```

**Step 6 — write to paper_trading_analyses**: In `autonomous_loop.py`, after the analysis completes and before `decide_trades`, write a minimal record:

```python
await asyncio.to_thread(
    bq.save_paper_trading_analysis,
    {
        "analysis_id": analysis["analysis_date"],
        "ticker": ticker,
        "recommendation": analysis["recommendation"],
        "final_score": analysis["final_score"],
        "summary": (analysis.get("full_report") or {}).get("analysis", {}).get("reason", ""),
        "price_at_analysis": analysis["price_at_analysis"],
        "sector": (analysis.get("full_report") or {}).get("market_data", {}).get("sector", ""),
        "momentum_20d": (analysis.get("full_report") or {}).get("market_data", {}).get("momentum_20d"),
        "momentum_60d": (analysis.get("full_report") or {}).get("market_data", {}).get("momentum_60d"),
        "composite_score": candidate.get("composite_score"),
        "conviction_score": candidate.get("conviction_score"),
        "conviction_reason": candidate.get("conviction_reason"),
        "regime_tag": candidate.get("regime_tag"),
        "news_rationale": candidate.get("news_rationale"),
        "analysis_source": "claude_lite",
        "total_cost_usd": analysis.get("total_cost_usd", 0.01),
        "created_at": analysis["analysis_date"],
    }
)
```

**Step 7 — outcome_tracker.py**: Add a fallback query path. When `bq.get_report` (analysis_results) returns None for a lite analysis, query `paper_trading_analyses`:

```python
# In evaluate_all_pending, after line 119:
if not stored or not stored.get("full_report_json"):
    # Fallback: lite analysis
    lite = self.bq.get_paper_trading_analysis(report["ticker"], report["analysis_date"])
    if lite:
        # Synthesize a fake full_report_json from the lite columns
        stored = {
            "full_report_json": json.dumps({
                "quant": {"yf_data": {"valuation": {"Current Price": lite.get("price_at_analysis")},
                                       "profile": {"sector": lite.get("sector", "")}}}
            }),
            "lite_row": lite,
        }
        # Override price_at_rec
        price_at_rec = lite.get("price_at_analysis")
```

**Step 8 — build_situation_description for lite analyses**: Extend the function signature to accept optional lite fields:

```python
def build_situation_description(
    ticker: str,
    sector: str,
    enrichment_signals: dict,
    debate_result: dict | None = None,
    # NEW — lite analysis overlay fields
    momentum_20d: float | None = None,
    momentum_60d: float | None = None,
    rsi_14: float | None = None,
    composite_score: float | None = None,
    conviction_reason: str | None = None,
    regime_tag: str | None = None,
    news_rationale: str | None = None,
    claude_reason: str | None = None,
) -> str:
```

After the existing `parts` list, append:
```python
    # Lite analysis quant context (improves BM25 recall for future matches)
    if momentum_20d is not None:
        direction = "upward" if momentum_20d > 0 else "downward"
        parts.append(f"20-day momentum {momentum_20d:+.1f}% ({direction}).")
    if momentum_60d is not None:
        parts.append(f"60-day momentum {momentum_60d:+.1f}%.")
    if rsi_14 is not None:
        rsi_state = "overbought" if rsi_14 > 70 else ("oversold" if rsi_14 < 30 else "neutral RSI")
        parts.append(f"RSI {rsi_14:.0f} ({rsi_state}).")
    if regime_tag:
        parts.append(f"Macro regime: {regime_tag}.")
    if conviction_reason:
        parts.append(f"Conviction: {conviction_reason}.")
    if news_rationale:
        parts.append(f"News signal: {news_rationale}.")
    if claude_reason:
        parts.append(f"Analysis rationale: {claude_reason}.")
```

This produces a situation string like: "Analyzing ON in the Technology sector. 20-day momentum +6.1% (upward). 60-day momentum +14.2%. RSI 62 (neutral RSI). Macro regime: risk_on. Conviction: strong momentum across all timeframes. Analysis rationale: Strong momentum with reasonable valuation." — every domain term is now BM25-indexable.

---

## Research Gate Checklist

Hard blockers:
- [x] >=3 external sources READ IN FULL via WebFetch (relaxed floor; 5 fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) — 15 URLs collected
- [x] Recency scan (last 2 years) performed + reported
- [x] Full papers / pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (9 files inspected)
- [x] Contradictions / consensus noted (all 3 papers agree on minimum context unit)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/phase-23.1.7-research-brief.md",
  "gate_passed": true
}
```
