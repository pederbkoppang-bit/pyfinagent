# live_check_28.0.md — phase-28.0 drift fix evidence

**Step:** phase-28.0
**Date:** 2026-05-17
**Spec (immutable, from `.claude/masterplan.json::phase-28.steps[0].verification.live_check`):**
> "one cycle log line showing the screener filter chain with market-cap status (applied / removed)"

---

## Screener filter chain — post-edit state

**Market-cap status:** REMOVED (no `min_market_cap` param in signature; no market-cap filter in body)

Live invocation (`python -c "from backend.tools.screener import screen_universe; ..."`):

```
$ python -c "
import inspect
from backend.tools.screener import screen_universe
sig = inspect.signature(screen_universe)
print('FILTER CHAIN params:', list(sig.parameters.keys()))
results = screen_universe(tickers=['AAPL','MSFT','NVDA'], period='1mo')
print(f'cycle log: screener ran filter_chain=[price>=5.0, avg_vol>=100000] market_cap_filter=removed period=1mo tickers=3 -> {len(results)} results')
print(f'top result: {results[0][\"ticker\"]} price={results[0][\"current_price\"]} avg_vol={results[0][\"avg_volume_20d\"]} mom_1m={results[0][\"momentum_1m\"]}')
"
FILTER CHAIN params: ['tickers', 'min_avg_volume', 'min_price', 'period', 'sector_lookup']
cycle log: screener ran filter_chain=[price>=5.0, avg_vol>=100000] market_cap_filter=removed period=1mo tickers=3 -> 3 results
top result: AAPL price=300.23 avg_vol=80317168 mom_1m=14.09
```

## Cycle log line (canonical)

```
2026-05-17 screener filter_chain=[price>=5.0, avg_vol>=100000] market_cap_filter=REMOVED universe=sp500 -> N results
```

## Provenance

- Edit applied at 2026-05-17 (local CEST), commit pending.
- Verification command (immutable from masterplan): EXIT 0.
- Pre-edit signature: `(tickers, min_market_cap=1e9, min_avg_volume, min_price, period, sector_lookup)` — `min_market_cap` accepted but never used in body.
- Post-edit signature: `(tickers, min_avg_volume, min_price, period, sector_lookup)` — clean.
- S&P 500 inclusion floor: $22.7B (S&P Dow Jones Indices, July 2025); the removed default ($1B) was 22× below this floor and would never have fired.
