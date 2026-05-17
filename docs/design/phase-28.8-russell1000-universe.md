# phase-28.8 — Design: Russell-1000 universe expansion

**Step:** phase-28.8 (Candidate Picker Expansion — post-launch)
**Date:** 2026-05-17
**Effort:** M (3 files; new function + 60-ticker extras list + flag-conditional universe selection)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend/tools/screener.py`:

```python
IWB_HOLDINGS_URL = "https://www.ishares.com/us/products/239707/...IWB_holdings..."

def get_russell1000_tickers() -> list[str]:
    """3-tier fallback chain:
    1. Local 180-day cache
    2. iShares IWB CSV download (browser User-Agent)
    3. SP500 list + _RUSSELL_1000_EXTRA_FALLBACK (deduped)"""
```

`backend/services/autonomous_loop.py` adds flag-conditional universe selection before `screen_universe()`.

## Data source (post-fallback)

The IWB direct URL returned 10MB of HTML in this environment (browser-protected). The fallback path activated and produced 515 tickers (SP500 503 + 12 effective extras after dedup).

Critical: **SNDK, WDC, MU all present** in the fallback list — the documented reference-case names from the primary brief.

## Cost guard

Existing two-pass design (cheap `screen_universe` yfinance-only → top-N cap → expensive analysis on `paper_analyze_top_n=5`) caps downstream cost. Russell-1000 mode adds ~30-60s wall-clock per cycle (515 tickers vs 503) but **zero LLM cost change**.

## Feature flag

`russell1000_universe_enabled = False` default. Production behavior unchanged.

## Cache TTL

180 days = FTSE Russell semi-annual reconstitution cadence.

## Source rationale

- **Primary brief item #10** — Sandisk/SNDK reference case where SP500-only universe excluded the spinoff during the early rally
- **iShares IWB** — canonical Russell-1000 holdings source (issues with direct CSV path; fallback covers reference cases)
- **FTSE Russell 2026 reconstitution schedule** — semi-annual reconstitution justifies 180-day cache

## Test plan

All 7 checks passed. Q/A PASS.

## Known follow-up

**phase-28.8-followup-iwb-csv-parser** — IWB direct download returns HTML, not CSV. Options:
- A: install `etf-scraper` PyPI package
- B: use stockanalysis.com IWB page (different scrape pattern)
- C: FTSE Russell paid feed

Default-OFF means production is unaffected; operator picks the option later.

## References

- `handoff/current/phase-28.8-research-brief.md`
- `handoff/current/contract.md`, `experiment_results.md`, `evaluator_critique.md`, `live_check_28.8.md`
- `docs/audits/phase-28.8-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[8]`
