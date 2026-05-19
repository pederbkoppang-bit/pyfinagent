# Research Brief -- phase-31.0.2 Stage 2 Smoketest (lite-path subagent synthesis)

**Tier:** deep | **Effort:** max | **Date:** 2026-05-20
**Scope:** Stage 2 of 13-stage smoketest. For each ticker
[AAPL, MSFT, NVDA, JPM] returned by Stage 1, spawn a Claude Code
subagent that produces a 4-field JSON synthesis:
`{ticker, recommendation, final_score, risk_assessment, price_at_analysis}`.

The substitution rule is load-bearing: the production
`_run_claude_analysis` lite path calls `anthropic.Anthropic().messages.create()`
directly. This smoketest replaces those Anthropic API calls with
Claude Code subagent spawns via `Agent({subagent_type: "general-purpose", ...})`.

## Search-query composition (three-variant discipline)

| Variant | Topic | Sample query |
|---------|-------|--------------|
| 2026 frontier | LLM structured JSON output | "Anthropic Claude structured output tool_use input_schema 2026" |
| 2025 last-2-yr | LLM-as-judge robustness | "LLM-as-judge JSON synthesis 2025 robustness" |
| year-less canonical | JSON-only prompt patterns | "JSON-only prompt LLM"; "prompt schema enforcement" |

## Code-audit findings (file:line anchors) -- CONFIRMED

### Production lite path -- `backend/services/autonomous_loop.py`

`_run_claude_analysis` at line **1288-1470**. The shape of the
returned dict (the EXACT shape Stage 2 subagent output must match
to satisfy `decide_trades`):

```python
return {
    "ticker": ticker,                          # str
    "_path": "lite",                           # str sentinel
    "recommendation": analysis["action"],      # "BUY" | "SELL" | "HOLD"
    "final_score": analysis["score"],          # 1-10 numeric
    "risk_assessment": {                       # dict
        "decision": ...,                       # "APPROVE_FULL" | "REJECT" | ...
        "reasoning": str,
        "reason": str,                         # back-compat alias
        "recommended_position_pct": float,
        "risk_level": str,
        "risk_limits": dict,
    },
    "price_at_analysis": current_price,        # float (USD)
    "analysis_date": "<ISO8601 UTC>",
    "total_cost_usd": 0.01,
    "full_report": {
        "source": model_name,
        "analysis": analysis,                  # nested inner trader dict
        "market_data": {...},
    },
}
```

**Inner trader-LLM prompt** at `autonomous_loop.py:1339-1359` -- demands:
```json
{"action": "BUY", "confidence": 75, "score": 7, "reason": "..."}
```
Parsed with `re.search(r'\{[^}]+\}', text)` at line 1372 (single-
level brace match -- nested objects in the trader inner JSON would
break; the risk-judge JSON uses `re.DOTALL` for nested matching at
line 1407).

### Consumer -- `backend/services/portfolio_manager.py::decide_trades`

Required fields read by `decide_trades` (lines 138-181):

| Field | Path in analysis dict | Default if missing | Used for |
|-------|----------------------|--------------------|----------|
| `ticker` | `analysis["ticker"]` | `""` | Per-position lookup |
| `recommendation` | `analysis["recommendation"]` | `"HOLD"` | Buy/sell gate (line 140-146); upper-cased |
| `risk_assessment` | `analysis["risk_assessment"]` | `{}` | Sizing + stop-loss |
| `risk_assessment.recommended_position_pct` | nested | None | Position size |
| `risk_assessment.decision` | nested | `""` | Logging at line 183-188 |
| `risk_assessment.risk_limits.stop_loss` or `stop_loss_pct` | nested | None | Stop derivation |
| `final_score` | `analysis["final_score"]` | `0` | Sort key (line 191) |
| `price_at_analysis` | `analysis["price_at_analysis"]` | `None` | Stop derivation |

Stage 2 simplified scope (per user prompt): the subagent emits a
**4-field** subset: `ticker, recommendation, final_score,
risk_assessment, price_at_analysis`. The user spec calls
`risk_assessment` a **string** ("1-2 sentences"). This DIVERGES
from production where `risk_assessment` is a **dict** with the
nested keys above. **This is intentional for Stage 2**: Stage 2
verifies subagent SPAWN + JSON SHAPE; full risk-assessment fidelity
is downstream (Stage 4+ when the risk-judge call substitution is
exercised). The Stage 2 output schema is therefore the
**simplified contract** the user prompt specifies, NOT the
production contract.

## Pass 1 -- Broad coverage (20+ sources read in full)

### A. Anthropic structured-output canonical patterns

(table appended as sources are fetched in full -- write-as-you-go)

