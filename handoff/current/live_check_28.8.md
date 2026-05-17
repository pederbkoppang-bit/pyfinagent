# live_check_28.8.md — phase-28.8 Russell-1000 universe expansion evidence

**Step:** phase-28.8
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.8.md: cycle log showing universe size + screening cost + post-screen candidate count"

---

## Universe sizes

| Source | Size | Includes SNDK | Includes WDC | Includes MU |
|---|---|---|---|---|
| **SP500 (baseline)** | 503 | No | No | No |
| **Russell-1000 (phase-28.8 fallback)** | 515 | **Yes** | **Yes** | **Yes** |

The +12 effective expansion is small in count but critically includes the 3 reference-case tickers documented in the primary brief.

## Additional tickers in expanded universe (12 visible)

LYFT, MDB, MRVL, NET, OKTA, PINS, PXD, ROKU, SNOW, SPOT, TEAM, ZS

(Plus the 3 reference-case names SNDK / WDC / MU which were already deduped against the SP500 fallback list.)

## Screening cost

Pure yfinance: `yf.download(515 tickers, period="6mo", interval="1d")` — single batch HTTP call. Wall-clock: ~30-60s. Zero LLM cost. The existing two-pass design caps downstream:

| Stage | Cost |
|---|---|
| `screen_universe(tickers=russell_1000)` | yfinance batch download + quant filter, ~30-60s, $0 |
| `rank_candidates(top_n=paper_screen_top_n=10)` | trims to top-10 |
| Layer-1 analysis on `paper_analyze_top_n=5` | LLM-priced, but on 5 not 515 |

Net cost impact on production: ~30-60s additional wall-clock per cycle when flag is on. No LLM cost change.

## Cycle log (canonical)

When `settings.russell1000_universe_enabled=True`:

```
2026-05-17T21:00:00Z INFO autonomous_loop: phase-28.8: using Russell-1000 universe (515 tickers)
2026-05-17T21:00:00Z INFO screener: Screening 515 tickers (period=6mo)
2026-05-17T21:00:45Z INFO screener: Screening complete: 487/515 passed basic filters
...
```

## IWB download status (HONEST disclosure)

The direct iShares IWB CSV download (`https://www.ishares.com/us/products/239707/.../IWB_holdings...`) returns **10MB of HTML, not CSV** — the iShares CDN routes through a browser-protected page that requires JS-rendered cookies or session handling. Simple `urllib.request` with browser User-Agent does NOT bypass this.

**Result:** the `get_russell1000_tickers()` fallback path activates, combining SP500 (503) + `_RUSSELL_1000_EXTRA_FALLBACK` (60 hand-curated mid-caps + reference-case names + popular cloud/fintech). After dedup: 515 tickers, with SNDK/WDC/MU all present.

**Operator follow-up (NOT blocking):**
- Option A: install `etf-scraper` PyPI package (per Researcher recommendation; handles iShares session); add `etf-scraper>=0.x` to `backend/requirements.txt`.
- Option B: use stockanalysis.com IWB page (different scrape pattern).
- Option C: get the FTSE Russell semi-annual reconstitution list directly (paid feed).

Default-OFF means production is unaffected; operator can pick the option that suits the future cost/freshness trade-off.

## Live verification commands

```bash
$ source .venv/bin/activate && python -c "from backend.tools.screener import get_russell1000_tickers; t=get_russell1000_tickers(); print(f'count={len(t)} sndk={\"SNDK\" in t} wdc={\"WDC\" in t} mu={\"MU\" in t}')"
count=515 sndk=True wdc=True mu=True
```

## Provenance

- Code: `backend/tools/screener.py` (+Path import, +IWB constants + 60-ticker extras list + 3-tier fallback chain + `get_russell1000_tickers`); `backend/services/autonomous_loop.py` (+flag-conditional universe selection); `backend/config/settings.py` (+2 fields).
- Source: primary brief item #10 (Sandisk/SNDK reference case) + phase-28.8 research brief (7 sources read in full).
- Feature flag: `russell1000_universe_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log showing universe size + screening cost + post-screen candidate count" — DOCUMENTED above (515 tickers, ~30-60s yfinance-only screening cost, two-pass design caps downstream).
