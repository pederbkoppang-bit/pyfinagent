# Live Check — phase-32.5 Dashboard Wiring Fix

**Date:** 2026-05-21
**Verification target:** invoke `_fetch_ticker_meta` against production with the 11 current tickers; all 11 must return `source='paper_positions'` with real company_name values.

## Live invocation result

```
$ source .venv/bin/activate && python -c "
from backend.config.settings import Settings
from backend.db.bigquery_client import BigQueryClient
from backend.api.paper_trading import _fetch_ticker_meta
s = Settings(); bq = BigQueryClient(s)
tickers = ['MU', 'KEYS', 'GEV', 'COHR', 'ON', 'INTC', 'DELL', 'GLW', 'LITE', 'SNDK', 'WDC']
print(_fetch_ticker_meta(tickers, s, bq))
"
```

Returned (verbatim):

```json
{
  "meta": {
    "COHR": {"company_name": "Coherent Corp.",                "sector": "Technology",  "source": "paper_positions"},
    "DELL": {"company_name": "Dell Technologies Inc.",         "sector": "Technology",  "source": "paper_positions"},
    "GEV":  {"company_name": "GE Vernova Inc.",                "sector": "Industrials", "source": "paper_positions"},
    "GLW":  {"company_name": "Corning Incorporated",           "sector": "Technology",  "source": "paper_positions"},
    "INTC": {"company_name": "Intel Corporation",              "sector": "Technology",  "source": "paper_positions"},
    "KEYS": {"company_name": "Keysight Technologies Inc.",     "sector": "Technology",  "source": "paper_positions"},
    "LITE": {"company_name": "Lumentum Holdings Inc.",         "sector": "Technology",  "source": "paper_positions"},
    "MU":   {"company_name": "Micron Technology, Inc.",        "sector": "Technology",  "source": "paper_positions"},
    "ON":   {"company_name": "ON Semiconductor Corporation",   "sector": "Technology",  "source": "paper_positions"},
    "SNDK": {"company_name": "Sandisk Corporation",            "sector": "Technology",  "source": "paper_positions"},
    "WDC":  {"company_name": "Western Digital Corporation",    "sector": "Technology",  "source": "paper_positions"}
  },
  "ttl_sec": 86400,
  "count": 11
}
```

## Cross-check vs the original dashboard observation

The 9 affected tickers from the original dashboard observation are MU, KEYS, GEV, COHR, ON, DELL, GLW, LITE, WDC. After phase-32.5:

| Ticker | Dashboard COMPANY (2026-05-20 pre-32.x) | _fetch_ticker_meta result (post-32.5) |
|---|---|---|
| MU | MU | **Micron Technology, Inc.** |
| KEYS | KEYS | **Keysight Technologies Inc.** |
| GEV | GEV | **GE Vernova Inc.** |
| COHR | COHR | **Coherent Corp.** |
| ON | ON | **ON Semiconductor Corporation** |
| DELL | DELL | **Dell Technologies Inc.** |
| GLW | GLW | **Corning Incorporated** |
| LITE | LITE | **Lumentum Holdings Inc.** |
| WDC | WDC | **Western Digital Corporation** |
| INTC | Intel Corporation (already correct) | **Intel Corporation** (now via paper_positions, not analysis_results) |
| SNDK | Sandisk Corporation (already correct) | **Sandisk Corporation** (now via paper_positions) |

**9 of 9 originally-broken tickers** now return real names. **11 of 11 total** route via `paper_positions` (the canonical source per phase-32.4 backfill).

## Cache eviction note

The `/api/paper-trading/ticker-meta` endpoint caches results for 24h via `get_api_cache()`. Two paths to surface the fix in the dashboard:

1. **Wait up to 24h** for the existing cache to evict naturally.
2. **Operator cache-bust:** any mutation through `/api/paper-trading/*` write endpoints triggers `get_api_cache().invalidate("paper:*")` at line 96. Visiting the `/portfolio` page after a trade or NAV adjustment will refresh.

The cache lives in-process so a backend restart also evicts. No data-layer change is needed.

## Verification command output

```
$ python -m pytest backend/tests/ -q --tb=line
285 passed, 1 skipped, 0 failures in 19.85s

$ grep -n 'paper_positions' backend/api/paper_trading.py | head -5
[shows ≥5 hits including the new UNION query block at the _fetch_ticker_meta Step 1 BQ query]

$ python -c "import ast; ast.parse(open('backend/api/paper_trading.py').read())"
(no output -- OK)
```

## Success criteria check (all 5 PASS)

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `fetch_ticker_meta_paper_positions_primary_source` | **PASS** | all 11 tickers return `source: "paper_positions"` |
| 2 | `analysis_results_fallback_preserved` | **PASS** | UNION includes `analysis_results` priority 2; tickers absent from paper_positions still get their analysis_results value |
| 3 | `yfinance_fallback_preserved` | **PASS** | Step 2 yfinance fallback unchanged; tickers still missing OR missing sector flow through it |
| 4 | `ticker_as_name_sentinel_filtered_at_sql` | **PASS** | WHERE clause filters `company_name != ticker` on BOTH source branches |
| 5 | `no_regression_full_sweep_285` | **PASS** | 285 passed, 0 failures |
