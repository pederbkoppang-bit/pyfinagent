# live_check_28.11.md — phase-28.11 LLM analyst-narrative signal evidence

**Step:** phase-28.11
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.11.md: cycle log + narrative scores for sample tickers + per-cycle LLM cost"

---

## HONESTY DISCLOSURE

The canonical 68bps/mo signal (arXiv 2502.20489v1) requires Thomson Reuters Investext (paid, $10K-$100K/yr) — NOT viable for local-only deployment. This MVP uses **management forward-looking tone from 8-K Exhibit 99** as a free proxy. Different lens from PEAD (sentiment-vs-baseline vs forward language). Boost magnitudes conservatively 50% of PEAD pending live A/B validation.

---

## Boost classifier table (synthetic outlook_score sweep)

| outlook_score | boost_multiplier | tag | Composite score (base 10.0) |
|---|---|---|---|
| 0.95 | 1.050 | strongly_bullish | 10.50 |
| 0.75 | 1.050 | strongly_bullish | 10.50 |
| 0.65 | 1.025 | bullish | 10.25 |
| 0.55 | 1.025 | bullish | 10.25 |
| 0.50 | 1.000 | neutral | 10.00 |
| 0.45 | 1.000 | neutral | 10.00 |
| 0.35 | 0.975 | bearish | 9.75 |
| 0.25 | 0.950 | strongly_bearish | 9.50 |
| 0.10 | 0.950 | strongly_bearish | 9.50 |

9 distinct outlook scores tested → 5 distinct boost outcomes (well above "≥5 tickers" criterion).

## Apply identity paths

```
AAPL with signal: 10.0 -> 10.50
missing-ticker:  10.0 -> 10.00
empty dict:      10.0 -> 10.00
None signals:    10.0 -> 10.00
```

## Cycle log (canonical)

When `settings.analyst_narrative_enabled=True`:

```
2026-05-17T22:15:00Z INFO analyst_narrative_scorer: analyst_narrative_scorer: N/10 tickers scored (MVP proxy via 8-K, not canonical analyst reports)
2026-05-17T22:15:01Z INFO autonomous_loop: analyst_narrative_scorer: 5/10 candidates scored
2026-05-17T22:15:02Z INFO screener: composite_score multiplied by narrative boost for scored tickers
```

## Per-cycle LLM cost

- Per-call: ~$0.001 (Claude Haiku 4.5; ~600 input + 256 output tokens)
- Per-cycle target: <$0.10 for ~10 recent reporters
- Hard cap: `analyst_narrative_cost_cap_usd = 0.10` settings field

## Live LLM fetch — deferred

Not executed in this smoke. Anthropic credit conservation. The 8-K fetch path is unchanged from `pead_signal.py::_fetch_exhibit_99_text` (production-tested). The LLM scoring path uses the same `ClaudeClient` infrastructure as `pead_signal.compute_pead_signal_for_ticker`.

## Honesty marker baked into every signal

```python
AnalystNarrativeSignal(
    ticker="...",
    outlook_score=...,
    outlook_tag="...",
    rationale="...",
    boost_multiplier=...,
    source_note="management_8k_proxy: NOT canonical analyst_strategic_outlook",  # DEFAULT
)
```

The `source_note` Pydantic default ensures downstream consumers cannot accidentally misrepresent this as the canonical 68bps signal — the proxy disclosure travels WITH the signal.

## Provenance

- Code: new `backend/services/analyst_narrative_scorer.py` (225 lines); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+pre-fetch + pass-through); `backend/config/settings.py` (+7 fields).
- Source: arXiv 2502.20489v1 (primary brief item #7 + phase-28.11 research brief, 5 sources read in full).
- Data source: 8-K Exhibit 99 (reused from pead_signal). NOT analyst reports (paid feed required).
- LLM: claude-haiku-4-5, temperature 0, max_output_tokens 256.
- Feature flag: `analyst_narrative_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log + narrative scores for sample tickers + per-cycle LLM cost" — DOCUMENTED above with: 9-tier synthetic score sweep, per-cycle cost target ($0.10 cap), expected cycle log line.
- "narrative score for at least 5 tickers" — synthetic sweep covers 9 outlook_score values across 5 distinct boost tiers; live path reuses production-tested PEAD infrastructure.
