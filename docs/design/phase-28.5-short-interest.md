# phase-28.5 — Design: Short-interest exclusion filter

**Step:** phase-28.5 (Candidate Picker Expansion)
**Date:** 2026-05-17
**Effort:** S (4-file change, one new 213-line module)
**Author:** Main (Claude Code session, Opus 4.7 xhigh)

## Interface

`backend.tools.screener.screen_universe()` signature, before vs after:

```python
# BEFORE (post-phase-28.0)
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
) -> list[dict]: ...

# AFTER (post-phase-28.5)
def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
    short_interest_lookup: Optional[dict[str, float]] = None,  # NEW
    short_interest_threshold: float = 0.10,                     # NEW
) -> list[dict]: ...
```

## Inputs / Outputs / Integration points

- **Inputs:** unchanged for current callers (both new kwargs default to None / 0.10). When the feature flag is on, autonomous_loop builds the lookup and passes it.
- **Outputs:** same `list[dict]` shape. When exclusion fires, the excluded ticker simply doesn't appear in results (no new field added).
- **Integration points:**
  - `backend/services/autonomous_loop.py` — pre-fetches lookup when `settings.short_interest_filter_enabled=True`, passes both kwargs to `screen_universe()`. Mirrors the existing `sector_lookup` pattern.
  - `backend/api/backtest.py:195` — calls `screen_universe()` with no kwargs, unchanged (back-compat). When ready, backtest can optionally pass the lookup too.
  - Tests in `tests/services/test_screener_sector_propagation.py` and `tests/verify_phase_23_1_13.py` — unchanged (they exercise `sector_lookup`, not the new kwargs).

## Data sources (priority order)

1. **FINRA bimonthly equity short-interest CSV** (preferred) — published on the 15th and end-of-month per cycle. Free, bulk. Covers all exchange-listed equities since June 2021. Local 14-day cache. Format: pipe-delimited.
2. **yfinance `Ticker.info["shortPercentOfFloat"]`** (fallback) — per-ticker HTTP call to Yahoo Finance. Throttled at 0.5s/call to respect 2024-tightened rate limits. Cost: ~250s for full S&P 500.
3. **Empty dict** (degraded) — when both fail; screener no-ops on exclusion, preserves cycle.

## Feature flag

`settings.short_interest_filter_enabled: bool = Field(False, ...)` — default OFF. Production behavior unchanged until explicitly enabled.

Two supporting fields:
- `short_interest_threshold: float = Field(0.10, ...)` — cutoff (10% of float, approximate top-decile for S&P 500 large-caps)
- `short_interest_cache_days: int = Field(14, ...)` — FINRA cache TTL matching bimonthly publication cadence

## Test plan

1. **Immutable verification command** (from `.claude/masterplan.json::phase-28.steps[5].verification.command`):
   ```bash
   source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE 'short.{0,30}(ratio|interest|exclusion)' backend/tools/screener.py
   ```
2. **Signature smoke** — `inspect.signature(screen_universe)` must include both new kwargs.
3. **Settings defaults** — `Settings().short_interest_filter_enabled == False`, threshold `0.10`, cache_days `14`.
4. **Back-compat smoke** — `screen_universe(tickers=['AAPL','MSFT','TSLA'], period='1mo')` returns 3 results.
5. **Exclusion smoke** — `screen_universe(tickers=['AAPL','MSFT','TSLA'], period='1mo', short_interest_lookup={'TSLA': 0.15}, short_interest_threshold=0.10)` returns 2 results (TSLA excluded).
6. **Live data path** — `fetch_short_interest_lookup(fallback_tickers=['TSLA','GME','AMC','AAPL','MSFT'])` returns valid shortPercentOfFloat values for each (yfinance fallback path, FINRA URL probe currently returns 403 — see follow-up).
7. **Q/A pass** — fresh `qa` subagent reads all artifacts, returns verdict.

All seven passed; see `docs/audits/phase-28.5-smoke-test-2026-05-17.md` for verbatim outputs.

## Source rationale (threshold 0.10)

- Boehmer-Jones-Zhang 2008: top-decile underperforms 1.16%/mo (NY Fed PDF — binary; cited via search snippets)
- Oxford RAPS 2022 (32-country, read in full): 1 SD increase in short interest predicts 0.62% lower monthly returns; confirms cross-sectional predictability internationally
- Quantpedia (read in full): top-decile short underperforms 215 bps/month EW (1988-2002); OOS decay for long-short, but exclusion-only is robust
- Practitioner tutorial (Medium, 2022-2025 data, read in full): >10% shortPercentOfFloat = elevated; >25% = extreme; demonstrates signal continues to work post-meme era

The 10% cutoff corresponds to the approximate top-decile for S&P 500 large-caps and aligns with the "elevated" threshold in practitioner literature.

## Known follow-up

The FINRA bulk-CSV URL pattern (`cdn.finra.org/equity/regsho/monthly/shrt<DATE>.csv`) returns HTTP 403 in this environment. Correct URL likely requires the documented `https://api.finra.org/data/group/otcMarket/name/equityShortInterest` REST endpoint or FINRA portal authentication. **Track as phase-28.5-followup-finra-url** (tiny patch; default-OFF feature is unblocked).

## References

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md`
- `handoff/current/phase-28.5-research-brief.md`
- `handoff/current/live_check_28.5.md`
- `docs/audits/phase-28.5-smoke-test-2026-05-17.md`
- `.claude/masterplan.json::phase-28.steps[5]`
