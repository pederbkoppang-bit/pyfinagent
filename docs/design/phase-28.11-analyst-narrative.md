# phase-28.11 — Design: LLM analyst-narrative signal (MVP proxy)

**Step:** phase-28.11 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** L (new 243-line module; honest disclosure across 13+ surfaces)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## HONESTY UPFRONT

Canonical 68bps/mo signal (arXiv 2502.20489v1) requires paid Thomson Reuters Investext ($10K-$100K/yr). Not viable for local-only deployment. This module is a **MVP PROXY** — scores MANAGEMENT FORWARD-LOOKING TONE from 8-K Exhibit 99 via Claude Haiku. Honesty travels with the signal via Pydantic default `source_note`.

## Interface

```python
class AnalystNarrativeSignal(BaseModel):
    ticker: str
    outlook_score: float  # [0, 1]
    outlook_tag: Literal["strongly_bullish","bullish","neutral","bearish","strongly_bearish"]
    rationale: str
    boost_multiplier: float
    source_note: str = "management_8k_proxy: NOT canonical analyst_strategic_outlook"  # DEFAULT

async def fetch_narrative_signals(tickers, model="claude-haiku-4-5", ...) -> dict[str, AnalystNarrativeSignal]: ...
def apply_narrative_signal_to_score(base, ticker, signals) -> float: ...
```

## Boost tiers

| outlook_score | tag | multiplier |
|---|---|---|
| >=0.70 | strongly_bullish | 1.05 |
| 0.55-0.69 | bullish | 1.025 |
| 0.41-0.54 | neutral | 1.0 |
| 0.31-0.40 | bearish | 0.975 |
| <=0.30 | strongly_bearish | 0.95 |

Conservatively half the PEAD scale pending live A/B validation.

## Infrastructure reuse

Reuses `pead_signal._fetch_recent_8k` + `_fetch_exhibit_99_text` — same 8-K Exhibit 99 source but different scoring lens (forward language vs sentiment-vs-trend). Same ClaudeClient.

## Feature flag

`analyst_narrative_enabled = False`.

## Test plan

5 immutable criteria evidenced. Q/A: 22 deterministic checks PASS. 9-tier classifier sweep verified.

## References

- `handoff/current/phase-28.11-research-brief.md`
- `.claude/masterplan.json::phase-28.steps[11]`
