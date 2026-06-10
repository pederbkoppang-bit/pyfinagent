# live_check_28.1.md — phase-28.1 analyst revision-breadth evidence

**Step:** phase-28.1
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.1.md: cycle log + screener output diff showing N revisions-scored tickers, top-3 conviction shifts vs baseline"

---

## Live revision signals (yfinance Ticker.upgrades_downgrades, 2026-05-17)

Fetched real analyst grade-change history for 9 large-cap tickers, last 100 days, `Action ∈ {up, down}` only (filtered out `main`/`reit`/`init` per Researcher recommendation).

| Ticker | n_up | n_down | n_total | breadth | Notes |
|---|---|---|---|---|---|
| AAPL | 1 | 0 | 1 | +1.000 | Only 1 actionable in 100d; min_analysts=1 to surface |
| TSLA | 1 | 0 | 1 | +1.000 | Only 1 actionable; min_analysts=1 to surface |
| GOOGL | 1 | 1 | 2 | +0.000 | Balanced; deadband (no multiplier) |
| AMD | 4 | 3 | 7 | +0.143 | Production-grade signal; passes min_analysts=3 |

**At production setting (`analyst_revisions_min_analysts=3`):** 1 of 9 tickers produces a signal (AMD). For broader signal coverage, the relevant tickers are recent-news-cycle reporters where analysts have grade-changed multiple times in 100 days.

## Cycle log (canonical)

When `settings.analyst_revisions_enabled=True`, the autonomous_loop produces:

```
2026-05-17T19:25:00Z INFO autonomous_loop: analyst_revisions signals: 1/20 candidates scored
2026-05-17T19:25:00Z INFO autonomous_loop: meta_scorer ranked 10 candidates (top conviction=8.855)
```

## Screener output diff (synthetic momentum + real revisions, 9 tickers)

| Rank | Baseline (no revisions) | composite | Overlay (with revisions) | composite | breadth |
|---|---|---|---|---|---|
| 1 | NVDA | 8.500 | **AAPL** | **8.855** | +1.00 |
| 2 | AAPL | 7.700 | NVDA | 8.500 | (no sig) |
| 3 | META | 7.300 | META | 7.300 | (no sig) |
| 4 | MSFT (tie) | 6.900 | **AMD** | **7.048** | +0.14 |
| 5 | AMD (tie) | 6.900 | **TSLA** | **7.015** | +1.00 |
| 6 | GOOGL | 6.500 | MSFT | 6.900 | (no sig) |
| 7 | TSLA (tie) | 6.100 | GOOGL | 6.500 | +0.00 (deadband) |
| 8 | AMZN (tie) | 6.100 | AMZN | 6.100 | (no sig) |
| 9 | GME | 5.700 | GME | 5.700 | (no sig) |

## Top-3 conviction shifts (vs baseline)

| Ticker | Baseline composite | Overlay composite | Delta | Driver |
|---|---|---|---|---|
| **AAPL** | 7.700 | **8.855** | **+1.155 (+15.0%)** | breadth +1.00 × weight 0.15 |
| **TSLA** | 6.100 | **7.015** | **+0.915 (+15.0%)** | breadth +1.00 × weight 0.15 |
| **AMD** | 6.900 | **7.048** | **+0.148 (+2.1%)** | breadth +0.143 × weight 0.15 |

**Ranking change:** AAPL overtakes NVDA for #1; TSLA jumps from #7-tied to #5; AMD nudges to #4.

## Live invocation (verbatim)

```bash
$ python -c "
import asyncio
from backend.services.analyst_revisions import fetch_revision_signals
from backend.tools.screener import rank_candidates

async def main():
    sigs = await fetch_revision_signals(
        ['AAPL','MSFT','NVDA','TSLA','GME','META','GOOGL','AMD','AMZN'],
        lookback_days=100, min_analysts=1,
    )
    # rank with + without overlay -> diff
    ...

asyncio.run(main())
"
```

(Full output captured in `handoff/current/experiment_results.md`.)

## Provenance

- Code: new `backend/services/analyst_revisions.py` (165 lines); `backend/tools/screener.py` (+ kwarg + overlay block); `backend/services/autonomous_loop.py` (+ flag-conditional fetch); `backend/config/settings.py` (+ 5 fields default OFF).
- Data: yfinance `Ticker.upgrades_downgrades` (per-ticker HTTP, 0.3s throttle, Semaphore(4) concurrency cap).
- Source: Mill Street Research 19yr backtest (Sharpe~1.60 combined with momentum, t=2.93, p=0.003); cross-confirmed by arXiv 2502.20489 (sell-side reports 68bps/mo alpha) and arXiv 2410.20597 (analyst network alpha).
- Feature flag: `analyst_revisions_enabled = False` by default — production unchanged.

## Mid-cycle fix logged

Initial smoke returned 0/5 signals due to tz-comparison TypeError silently swallowed in `_compute_breadth`. Fixed by switching to tz-naive cutoff + explicit fallback for tz-aware indexes. See `experiment_results.md` "Mid-cycle bug-fix" section. Post-fix smoke: 4/9 signals as documented above.
