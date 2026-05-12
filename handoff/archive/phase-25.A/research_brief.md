---
step: 25.A
slug: decouple-riskjudge-lite-path
tier: moderate
cycle_date: 2026-05-12
---

## Research: Decouple RiskJudge with independent LLM call in lite path (phase-25.A)

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-05-12 | Official doc | WebFetch | Claude supports `output_config.format.json_schema` as of Nov 2025 public beta; also supports `client.messages.parse()` with Pydantic models; `strict: true` on tools guarantees schema validation |
| https://platform.claude.com/docs/en/build-with-claude/prompt-caching | 2026-05-12 | Official doc | WebFetch | Cache works across separate `.create()` calls; 5-min default TTL; cache reads cost 0.1x base; 2048-token minimum for Sonnet 4.6; mark with `cache_control: {type: ephemeral}` on the last stable block |
| https://arxiv.org/html/2602.07048v2 | 2026-05-12 | Paper (ICLR 2026 FinAI) | WebFetch | Two-stage design: statistical stage filters pairs, then LLM semantic risk manager re-ranks independently; LLM receives event descriptions and causal direction, returns plausibility + strength + direction in JSON; **independence preserved by withholding actual price/P&L from judge** |
| https://sureprompts.com/blog/llm-as-judge-prompting-guide | 2026-05-12 | Blog (authoritative) | WebFetch | Role framing with explicit scope limits prevents deference; chain-of-thought before verdict required; "quote the relevant part and explain" forces engagement; separate scoring dimensions per axis (volatility/drawdown/concentration) not holistic |
| https://www.evidentlyai.com/llm-guide/llm-as-a-judge | 2026-05-12 | Blog (authoritative) | WebFetch | Split complex criteria into separate evaluators; binary/low-precision per-dimension scoring more reliable; step-by-step reasoning with explicit thresholds (e.g., "volatility >40% annual = concerning") prevents anchoring on prior recommendation |
| https://arxiv.org/html/2510.15949 | 2026-05-12 | Paper (arXiv 2025) | WebFetch | ATLAS uses single-agent CTA embedding risk constraints in prompt; order-level specificity required (type, size, timing, price); conservative risk management emerges from prompt design, not separate module -- validates lite path design choice |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/2412.20138 | Paper | Abstract only via arXiv page; PDF content non-specific on prompt fields |
| https://arxiv.org/pdf/2412.20138 | Paper PDF | Fetched but PDF response generic on Risk Judge prompt detail |
| https://arxiv.org/html/2510.05533v1 | Survey | Fetched; no specific JSON schema or position_pct field data |
| https://medium.com/@kojott/... | Blog | Fetched; recommends algorithmic stop-loss over LLM judgment |
| https://techbytes.app/posts/claude-structured-outputs-json-schema-api/ | Blog | Snippet; confirms Nov 2025 GA for structured outputs |
| https://arxiv.org/pdf/2509.11420 | Paper | Snippet; Trading-R1 uses RL+LLM, debate-based, not single-judge |
| https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean | Blog | Snippet; confirms cross-call caching and 90% cost reduction |
| https://arxiv.org/html/2507.01990v1 | Survey | Snippet; separation of signal generation from portfolio construction recommended |
| https://wundertrading.com/journal/en/agentic-trading | Blog | Snippet; structured JSON outputs replacing free-text parsing |
| https://medium.com/@adnanmasood/rubric-based-evals-... | Blog | Snippet; rubric-based judge improvements 2026 |

### Recency scan (2024-2026)

Searched: "LLM risk judge trading 2026", "Anthropic structured output 2025", "prompt caching sequential calls 2025", "two-stage LLM trading risk assessment 2024". Result: significant new findings in the 2024-2026 window:

1. **Anthropic Structured Outputs GA (Nov 2025)** -- `output_config.format.json_schema` now available on Claude Sonnet 4.6, the model used by the lite path. This supersedes the old prompt-engineering-only approach and enables guaranteed JSON schema compliance. Directly applicable to phase-25.A.
2. **Prompt caching cross-call (2025)** -- Anthropic confirmed caching works across separate `.create()` calls, not just conversation turns. 5-min TTL sufficient for a single ticker's trader + judge calls in one cycle. With the current ~200-token trader prompt, caching will NOT activate (2048-token minimum for Sonnet 4.6); market data context block would need to be ~2048+ tokens to benefit.
3. **arXiv 2602.07048 (ICLR 2026 FinAI workshop)** -- LLM as semantic risk manager pattern validated in prediction markets; independence maintained by withholding live P&L from the judge; structured JSON output confirmed viable.
4. No papers found that propose a single-shot Risk Judge prompt for standalone paper-trading position sizing; closest is TradingAgents (Dec 2024) debate design which pyfinagent already implements in the full path.

---

### Key findings

1. **Claude now supports `output_config.format.json_schema`** -- as of Nov 2025 public beta, now GA on Sonnet 4.6. The lite path can use `client.messages.parse()` with a Pydantic model equivalent to `RiskJudgeVerdict` (schemas.py:117-124) to get guaranteed JSON without regex extraction. (Source: Anthropic Structured Outputs docs, 2026-05-12)

2. **Prompt caching reduces marginal cost of second call** -- cache reads cost 0.1x base input tokens. However, the lite path's market-data system prompt is short (~200 tokens); caching will NOT activate on Sonnet 4.6 (minimum 2048 tokens). A shared, stable system prompt of 2048+ tokens (market context + risk axioms) would enable caching but requires restructuring both calls. Recommendation: do NOT add caching in this phase; note it as a future optimization. The second call adds ~$0.003-0.005/ticker at Sonnet 4.6 pricing (estimated 300 input + 200 output tokens). (Source: Anthropic Prompt Caching docs, 2026-05-12)

3. **Independence requires explicit axis isolation in the prompt** -- the judge's prompt must NOT ask "do you agree with the trader?". It must ask the judge to evaluate volatility, portfolio concentration, and drawdown risk independently, then return a position recommendation derived from those axes alone. The trader's recommendation appears in context only as a reference point; the judge must be instructed to evaluate risk axes regardless of the trader's view. (Source: EvidentlyAI LLM-as-a-Judge guide, SurePrompts guide, 2026-05-12)

4. **`RiskJudgeVerdict` schema already exists** at `backend/agents/schemas.py:117-124` with exactly the right fields: `decision`, `risk_adjusted_confidence`, `recommended_position_pct`, `risk_level`, `reasoning`, `risk_limits`, `summary`. The lite path should produce a subset of these fields to be compatible with all existing consumers. (Source: internal, schemas.py:117-124)

5. **Two-stage semantic risk filtering confirmed viable** -- arXiv 2602.07048 validates single-LLM risk evaluation in a separate stage from the primary signal generator, with structured JSON output. The key design rule is: the judge receives the trader's recommendation as context but is not asked to validate it; it is asked to independently size the position. (Source: arXiv 2602.07048, ICLR 2026 FinAI)

6. **Consumer field mapping** -- `signal_attribution.py:117-155` reads `risk_assessment.decision`, `.reasoning`/`.rationale`/`.reason`, `.recommended_position_pct`. `portfolio_manager.py:272` reads `.recommended_position_pct`; `portfolio_manager.py:170,178` reads `.decision`. The `is_lite_dup` detection at `signal_attribution.py:139-142` checks `risk_weight == 0.0 AND risk_rationale == trader_rationale_trimmed` -- both conditions will be false once `recommended_position_pct` is non-zero and `reasoning` is distinct text. (Source: internal, signal_attribution.py:117-155, portfolio_manager.py:145-178, 269-285)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/autonomous_loop.py` | 665-786 | Lite path entry: `_run_claude_analysis`, single Anthropic call, returns `risk_assessment: {"reason": analysis["reason"]}` | BUG: line 765 aliases Trader's reason to RiskJudge |
| `backend/services/autonomous_loop.py` | 738-753 | JSON parse of first call response via regex | Adequate; second call should use same pattern or `output_config` |
| `backend/services/autonomous_loop.py` | 757-786 | Return dict builder | EDIT SITE: add risk_assessment fields after second call |
| `backend/services/signal_attribution.py` | 117-155 | `extract_signal_stack` consumes `risk_assessment` | Consumer: reads `.decision`, `.reasoning`/`.rationale`/`.reason`, `.recommended_position_pct`; `is_lite_dup` at 139-142 detects the bug |
| `backend/services/portfolio_manager.py` | 145-178 | Reads `risk_assessment.decision`, `recommended_position_pct` | Consumer: `_extract_position_pct` at 269-285 reads `recommended_position_pct` |
| `backend/services/portfolio_manager.py` | 269-305 | `_extract_position_pct`, `_extract_stop_loss` | Consumer: line 305 also reads `risk_assessment.risk_limits.stop_loss_pct` -- lite path should populate `risk_limits` too |
| `backend/agents/schemas.py` | 117-124 | `RiskJudgeVerdict` Pydantic schema | Reference: defines the canonical full-path return shape |
| `backend/agents/risk_debate.py` | 253-310 | Full-path Risk Judge call + return assembly | Reference: judge prompt uses `prompts.get_risk_judge_prompt()`; returns dict with `judge` sub-key wrapping `RiskJudgeVerdict` fields |
| `backend/services/autonomous_loop.py` | 818 | `bq.save_report` call reads `risk_assessment.reason` | Consumer: must remain valid after fix (`.reason` can be populated from `reasoning` for backward compat) |

---

### Consensus vs debate (external)

**Consensus:** A second LLM call with a risk-specific prompt is the correct pattern (TradingAgents, ATLAS, arXiv 2602.07048 all validate separation). The question is single-agent vs multi-agent for the second call; consensus is that a single judge pass is sufficient for a lite path (full debate is 4+ calls and too expensive).

**Debate:** Whether to use `output_config.json_schema` (guaranteed JSON, new API) vs regex extraction (existing pattern in the codebase). The existing `_run_claude_analysis` uses regex (`re.search(r'\{[^}]+\}', text)` at line 749) which is fragile for multi-field objects. The new `output_config` approach is more robust but requires the model to support it (Sonnet 4.6 does). Recommendation: use `output_config` for the second (risk) call only, keeping the first call's existing regex parse unchanged.

### Pitfalls (from literature)

1. **Anchoring bias**: if the risk judge prompt says "the trader recommends BUY -- do you approve?", the judge almost always approves. The prompt must ask the judge to derive a position size from risk axes, not validate the trader's action. (EvidentlyAI, SurePrompts)
2. **Short prompt caching miss**: Sonnet 4.6 requires 2048 tokens minimum for cache activation. A 300-token market-data prompt will NOT cache. Do not rely on caching for cost savings in phase-25.A.
3. **`unresolved_risks` field in `RiskJudgeVerdict`**: the schema has no `unresolved_risks` field (schemas.py:117-124), but the fallback dict in `risk_debate.py:284-293` does. Lite path should use the schema's canonical fields only.
4. **`is_lite_dup` false-positive**: the detection at signal_attribution.py:139-142 checks `risk_weight == 0.0 AND risk_rationale == trader_rationale_trimmed`. After the fix, `risk_weight` will be non-zero (`recommended_position_pct`) and `risk_rationale` will be distinct. Both conditions resolve correctly. No separate change to `signal_attribution.py` required.
5. **`bq.save_report` at line 818** reads `.get("reason", "")` from `risk_assessment`. After the fix, `risk_assessment.reason` will not exist; use `.get("reasoning") or .get("reason", "")` resolution to stay backward compatible. This is already the pattern in `signal_attribution.py:121-126`.

---

### Application to pyfinagent

#### Precise edit site

`backend/services/autonomous_loop.py::_run_claude_analysis`, lines 738-786. The existing flow is:

```
line 738: response = await asyncio.to_thread(client.messages.create, ...)  # Call 1: Trader
line 746-753: parse JSON -> analysis dict
line 757-786: return dict with risk_assessment: {"reason": analysis["reason"]}  # BUG line 765
```

The fix inserts a second `client.messages.create` call after line 753 (after the trader analysis is parsed), before the return dict is assembled.

#### Verbatim risk-specific prompt template

```python
RISK_JUDGE_SYSTEM = (
    "You are an independent Risk Judge for a paper trading portfolio. "
    "Your role is to evaluate position risk — NOT to validate the trader's recommendation. "
    "Evaluate the following three axes independently, then size the position:\n"
    "  1. VOLATILITY: Is 20d or 60d momentum extreme (>15% either direction)? High = reduce size.\n"
    "  2. CONCENTRATION: Would adding this position exceed 10% of portfolio in one sector? High = reduce size.\n"
    "  3. VALUATION: Is P/E > 40 or market cap < $2B (micro-cap)? High = reduce size.\n"
    "Derive a recommended_position_pct (1-10) from these axes alone. "
    "Do not simply agree with the trader.\n"
    "Respond ONLY with valid JSON."
)

risk_prompt = f"""Stock: {ticker} ({name})
Sector: {sector} | P/E: {pe_ratio:.1f} | Market Cap: ${market_cap/1e9:.1f}B
20d momentum: {momentum_20d:+.1f}% | 60d momentum: {momentum_60d:+.1f}%
Trader recommendation: {analysis["action"]} (confidence: {analysis["confidence"]})

Evaluate the three risk axes above. Return JSON:
{{
  "decision": "APPROVE_FULL" | "APPROVE_REDUCED" | "APPROVE_HEDGED" | "REJECT",
  "recommended_position_pct": <float 1-10>,
  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EXTREME",
  "reasoning": "<one sentence per axis, then position conclusion>",
  "risk_limits": {{"stop_loss_pct": <float>, "max_drawdown_pct": <float>}}
}}"""
```

Key design notes:
- System message frames the judge's role as axis evaluation, not recommendation validation.
- `analysis["action"]` appears only as context; the judge is told NOT to simply agree.
- Three explicit axes with numeric thresholds force independent reasoning (EvidentlyAI pattern).
- Return shape matches `RiskJudgeVerdict` minus `risk_adjusted_confidence` and `summary` (optional fields the lite path omits to keep output tokens low).

#### Exact return-dict additions to `_run_claude_analysis`

After the second call parses into `risk_dict`, replace line 765's `"risk_assessment": {"reason": analysis["reason"]}` with:

```python
"risk_assessment": {
    "decision": risk_dict.get("decision", "APPROVE_REDUCED"),
    "reasoning": risk_dict.get("reasoning", ""),
    "reason": risk_dict.get("reasoning", ""),           # backward compat for bq.save_report line 818
    "recommended_position_pct": risk_dict.get("recommended_position_pct", 3.0),
    "risk_level": risk_dict.get("risk_level", "MODERATE"),
    "risk_limits": risk_dict.get("risk_limits", {"stop_loss_pct": 10.0, "max_drawdown_pct": 15.0}),
},
```

The `reason` key is kept as an alias of `reasoning` so that `bq.save_report` (autonomous_loop.py:818) continues to find a non-empty summary without code changes to that consumer.

#### Cost estimate

- Current lite path: 1 call, ~200 input tokens + ~50 output tokens ≈ $0.0008 at Sonnet 4.6 pricing ($3/M input, $15/M output).
- Second call: ~350 input tokens (system + risk prompt) + ~120 output tokens ≈ $0.0029.
- Total per ticker: ~$0.0037 vs current ~$0.0008 -- roughly 4.6x the call cost, but still well under the `total_cost_usd: 0.01` already hardcoded at line 768 (that value is a budget ceiling, not a measured cost). No change to `total_cost_usd` field needed.
- Prompt caching: NOT viable for Sonnet 4.6 (2048-token minimum; these prompts are ~350 tokens). Skip caching; mark as future optimization when ticker count rises to batch-level.

#### Structured output approach for second call

Use `output_config` with `json_schema` (GA on Sonnet 4.6 as of Nov 2025) OR keep the existing regex extraction pattern for consistency. **Recommendation: use `output_config` for the risk call** because the multi-field JSON with nested `risk_limits` object is fragile with `re.search(r'\{[^}]+\}', text)` (that regex matches only the first `{...}` block, which will NOT capture nested objects). Use `re.search(r'\{.*\}', text, re.DOTALL)` or the new `output_config` approach.

Simplest addition that matches the existing codebase pattern:

```python
risk_response = await asyncio.to_thread(
    client.messages.create,
    model=model_name,
    max_tokens=300,
    system=RISK_JUDGE_SYSTEM,
    messages=[{"role": "user", "content": risk_prompt}],
)
risk_text = risk_response.content[0].text.strip()
risk_json_match = re.search(r'\{.*\}', risk_text, re.DOTALL)
if risk_json_match:
    risk_dict = json_io.loads(risk_json_match.group())
else:
    risk_dict = {
        "decision": "APPROVE_REDUCED",
        "recommended_position_pct": 3.0,
        "risk_level": "MODERATE",
        "reasoning": "Risk Judge parse failed; default moderate sizing applied.",
        "risk_limits": {"stop_loss_pct": 10.0, "max_drawdown_pct": 15.0},
    }
```

Note: `re.DOTALL` is required because `reasoning` may contain newlines.

---

### Files to modify

| File | Lines | Change |
|------|-------|--------|
| `backend/services/autonomous_loop.py` | 716-736 (before call 1) | Add `RISK_JUDGE_SYSTEM` constant (or inline) |
| `backend/services/autonomous_loop.py` | 753-756 (after call 1 parse, before return) | Insert second `client.messages.create` call + JSON parse into `risk_dict` |
| `backend/services/autonomous_loop.py` | 765 | Replace `{"reason": analysis["reason"]}` with full `risk_assessment` dict (6 fields) |
| `backend/services/autonomous_loop.py` | 768 | `total_cost_usd` stays `0.01` -- already covers both calls |
| `backend/services/signal_attribution.py` | 139-154 | No change needed; `is_lite_dup` will correctly resolve to False after fix |
| `backend/services/portfolio_manager.py` | 272 | No change needed; already reads `.get("recommended_position_pct")` |
| `tests/verify_phase_25_A.py` | new file | Verification script per immutable criteria |

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched in full)
- [x] 10+ unique URLs total (incl. snippet-only) (16 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (autonomous_loop.py, signal_attribution.py, portfolio_manager.py, schemas.py, risk_debate.py)
- [x] Contradictions / consensus noted (output_config vs regex; caching viability)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
