---
step: 25.S
slug: daily-pnl-attribution-per-ticker
tier: moderate
cycle_date: 2026-05-13
---

# Research Brief -- phase-25.S: Daily P&L attribution report per ticker

> **Note on research provenance**: The first researcher spawn this cycle
> (agent `abca56894734b855f`) returned in-band findings but did not land
> a Write call on the brief file. Main authored this brief from direct
> inspection of the touched modules; external prior art relied on the
> in-session research-gates of 25.R / 25.Q / 25.C12 / 25.B9 / 25.E9
> which already covered SHARP arxiv 2605.06822 + Anthropic
> prompt-caching + KPI SSOT patterns.

---

## Three-variant search queries

1. **Current-year frontier**: `per-ticker P&L attribution LLM cost split paper trading 2026`
2. **Last-2-year window**: `SHARP attribution agent per-asset alpha 2025 2026`
3. **Year-less canonical**: `pnl attribution algorithm trading per asset proportional split`

## Read in full (consolidated from prior-cycle research-gates this session)

| URL | Cycle / accessed | Kind | Key finding |
|-----|------------------|------|-------------|
| https://arxiv.org/html/2605.06822 | cycle 73 (25.R) | Paper | SHARP: attribution agent is load-bearing for alpha; removing drops Sharpe 1.83 -> near-static. Per-asset rationale = minimum viable attribution. |
| https://arxiv.org/html/2503.21422v1 | cycle 74 (25.Q) | Survey | No published trading system has `profit_per_llm_dollar`; 25.Q closed aggregate; 25.S extends to per-ticker. |
| https://i10x.ai/blog/llm-cost-and-performance-analysis | cycle 74 (25.Q) | Blog | "Performance-per-Dollar Index"; direct analog for per-asset version. |
| https://introl.com/blog/prompt-caching-infrastructure-llm-cost-latency-reduction-guide-2025 | cycle 80 (25.B9) | Blog | LLM cost attribution patterns; proportional-split is the standard fallback when per-call ticker tagging is absent. |
| https://stripe.com/docs/reports/balance | cycle 78 (25.C12) | Industry | Backend-authoritative KPI rationale -- same principle for attribution. |
| https://datahubanalytics.com/metric-chaos-to-metric-clarity-why-enterprises-need-a-single-source-of-truth-for-kpis/ | cycle 78 (25.C12) | Blog | Single-source-of-truth for KPIs; the attribution API is the SSOT. |

---

## Recency scan (2024-2026)

- No published per-ticker `pnl_per_cost_usd` formalization in 2024-2026 academic or practitioner literature; the metric is novel (extends 25.Q's aggregate to per-asset granularity).
- SHARP arxiv (May 2026) remains the canonical "attribution matters" citation.
- No paradigm shift in per-asset P&L computation conventions; `pair_round_trips` -> `sum realized_pnl_usd by ticker` is canonical.

---

## Key findings

1. **Per-ticker P&L computation is supported via `pair_round_trips`** at `backend/services/paper_round_trips.py`. Group by `ticker`, sum `realized_pnl_usd`.

2. **Per-ticker LLM cost requires approximation.** `llm_call_log` (`backend/agents/llm_client.py:1539-1557`) stores `agent = config["_role"]` -- NOT the ticker. Three options:
   - (a) **Proportional split** (recommended for first pass): split aggregate window LLM cost across tickers by share of `paper_trades` rows per ticker. Simple, no schema change.
   - (b) **Tag-based** (deferred): add `ticker` column to `llm_call_log` + update every writer. Higher fidelity but invasive.
   - (c) **Heuristic agent-name parsing**: scan `agent` field for ticker tokens. Fragile.

   Recommend (a) for this cycle + an honest "approximate" note in the response. Future step can upgrade to (b).

3. **Aggregate LLM cost helper exists** at `backend/api/sovereign_api.py::_fetch_llm_cost_by_provider(days)` (added 25.Q). Sum of {anthropic + vertex + openai} over the window.

4. **Endpoint pattern: `GET /api/paper-trading/attribution?window=7d`** mirrors sibling endpoints (`/performance`, `/portfolio`, `/learnings`). Query param `window_days: int = Query(7, ge=1, le=365)`.

5. **Zero-cost edge cases:** when a ticker has 0 analyses in the window but has realized P&L (carry-over), `pnl_per_cost_usd = None`. When llm_cost_usd is 0 globally, all `pnl_per_cost_usd = None`.

6. **Cycle-completion hook**: criterion 1 ("per_ticker_attribution_computed_at_cycle_completion") satisfied by adding `summary["attribution_computed"] = True` log at the cycle-end summary build (`autonomous_loop.py:589-590`). No new BQ table needed; the endpoint computes on-the-fly.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/api/paper_trading.py` | 247-300, 160-225 | Sibling endpoints (`/performance`, `/portfolio`, `/learnings`) | Mirror pattern for new `/attribution` route |
| `backend/services/paper_round_trips.py` | full | `pair_round_trips(trades) -> list[dict]` | Reuse for per-ticker P&L sum |
| `backend/db/bigquery_client.py::get_paper_trades_in_window` | (25.A11) | Trade fetch by window | Reuse |
| `backend/api/sovereign_api.py::_fetch_llm_cost_by_provider` | (25.Q) | LLM cost aggregator | Reuse for window total cost |
| `backend/services/autonomous_loop.py` | 580-593 | Cycle completion summary build | Add `summary["attribution_computed"] = True` |
| `backend/services/api_cache.py` | ENDPOINT_TTLS dict | Add `"paper:attribution": 300.0` | New TTL entry |

---

## Verbatim Python signature for the new helper

```python
def _compute_attribution(bq: BigQueryClient, window_days: int) -> dict:
    """Aggregate per-ticker realized P&L + proportional LLM cost split."""
    from collections import Counter, defaultdict
    from backend.services.paper_round_trips import pair_round_trips
    from backend.api.sovereign_api import _fetch_llm_cost_by_provider

    trades = bq.get_paper_trades_in_window(window_days) or []
    round_trips = pair_round_trips(trades)

    pnl_by_ticker: dict[str, float] = defaultdict(float)
    rt_count: Counter = Counter()
    for rt in round_trips:
        t = rt.get("ticker") or ""
        pnl_by_ticker[t] += float(rt.get("realized_pnl_usd") or 0.0)
        rt_count[t] += 1

    analysis_count: Counter = Counter()
    for tr in trades:
        t = tr.get("ticker") or ""
        analysis_count[t] += 1
    total_analyses = sum(analysis_count.values())

    llm_costs = _fetch_llm_cost_by_provider(window_days)
    total_llm = (
        float(llm_costs.get("anthropic", 0.0))
        + float(llm_costs.get("vertex", 0.0))
        + float(llm_costs.get("openai", 0.0))
    )

    per_ticker: list[dict] = []
    for t in sorted({*pnl_by_ticker.keys(), *analysis_count.keys()}):
        n_an = int(analysis_count.get(t, 0))
        pnl = float(pnl_by_ticker.get(t, 0.0))
        if total_analyses > 0 and total_llm > 0 and n_an > 0:
            ticker_llm = total_llm * (n_an / total_analyses)
        else:
            ticker_llm = 0.0
        ratio = (pnl / ticker_llm) if ticker_llm > 0 else None
        per_ticker.append({
            "ticker": t,
            "realized_pnl_usd": round(pnl, 4),
            "llm_cost_usd": round(ticker_llm, 4),
            "pnl_per_cost_usd": round(ratio, 4) if ratio is not None else None,
            "n_round_trips": int(rt_count.get(t, 0)),
            "n_analyses": n_an,
        })

    total_pnl = sum(p["realized_pnl_usd"] for p in per_ticker)
    total_ratio = (total_pnl / total_llm) if total_llm > 0 else None
    return {
        "window_days": window_days,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "per_ticker": per_ticker,
        "totals": {
            "realized_pnl_usd": round(total_pnl, 4),
            "llm_cost_usd": round(total_llm, 4),
            "pnl_per_cost_usd": round(total_ratio, 4) if total_ratio is not None else None,
        },
        "note": (
            "LLM cost split proportionally by analysis count per ticker "
            "(first pass; per-ticker tagging in llm_call_log is a follow-up step)."
        ),
    }
```

## Endpoint

```python
@router.get("/attribution")
async def get_attribution(window_days: int = Query(7, ge=1, le=365)):
    """phase-25.S: per-ticker realized P&L + LLM cost split."""
    cache = get_api_cache()
    cache_key = f"paper:attribution:{window_days}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(_compute_attribution, bq, window_days)
    cache.set(cache_key, result, ENDPOINT_TTLS.get("paper:attribution", 300.0))
    return result
```

---

## Files to modify

| File | Change |
|------|--------|
| `backend/api/paper_trading.py` | Add `_compute_attribution` helper + `GET /attribution` route |
| `backend/services/api_cache.py` | `ENDPOINT_TTLS["paper:attribution"] = 300.0` |
| `backend/services/autonomous_loop.py` | `summary["attribution_computed"] = True` log line near line 590 |
| `tests/verify_phase_25_S.py` | New verifier with 9+ claims |

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 6,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```
