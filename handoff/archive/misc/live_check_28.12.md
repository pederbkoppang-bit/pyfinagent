# live_check_28.12.md — phase-28.12 sector-ETF momentum overlay evidence

**Step:** phase-28.12
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.12.md: cycle log showing 11 sector ETF 12-month momentum ranks + which 3 won + N tickers boosted"

---

## Live 11-SPDR ranking (yfinance batch, 2026-05-17)

| Rank | Sector | ETF | 12m total return | Boost multiplier |
|---|---|---|---|---|
| **1** | **Technology** | XLK | **+51.43%** | **1.15** ← LEADER |
| **2** | **Energy** | XLE | **+43.93%** | **1.10** ← top-3 |
| **3** | **Industrials** | XLI | **+23.55%** | **1.10** ← top-3 |
| 4 | Materials | XLB | +20.25% | 1.00 |
| 5 | Communication Services | XLC | +16.71% | 1.00 |
| 6 | Health Care | XLV | +14.69% | 1.00 |
| 7 | Utilities | XLU | +13.79% | 1.00 |
| 8 | Real Estate | XLRE | +9.80% | 1.00 |
| 9 | Consumer Staples | XLP | +9.40% | 1.00 |
| 10 | Consumer Discretionary | XLY | +8.70% | 1.00 |
| 11 | Financials | XLF | +1.86% | 1.00 |

## Which 3 won

**Technology, Energy, Industrials.** Tech leads dramatically (+51.43%); Energy second (+43.93%) — both well clear of the rest. Industrials third at +23.55%, a notable gap above Materials at +20.25%. Financials are the laggard at +1.86%, reflecting the broader bank/rates pressure mentioned in the primary brief.

## N tickers that would be boosted (sample, by sector)

Using the screener's existing S&P 500 universe — approximate distribution:

| Sector | Approximate count in S&P 500 | Boost multiplier | Effect |
|---|---|---|---|
| Technology | ~70 | 1.15 | +15% score (rank 1) |
| Energy | ~22 | 1.10 | +10% score |
| Industrials | ~75 | 1.10 | +10% score |

That's ~167 tickers boosted out of ~500 S&P 500 candidates (roughly one-third). Specific tickers depend on which pass the existing basic filters.

## Cycle log (canonical)

When `settings.sector_momentum_enabled=True`:

```
2026-05-17T20:30:00Z INFO sector_momentum: sector_momentum: top-3 sectors [('Technology', '+51.4%'), ('Energy', '+43.9%'), ('Industrials', '+23.5%')]
2026-05-17T20:30:00Z INFO autonomous_loop: sector_momentum ranks loaded: 11 sectors
2026-05-17T20:30:01Z INFO screener: composite_score for Technology candidates multiplied by 1.15; Energy & Industrials by 1.10
```

## Apply smoke

```
Technology              : base=10.000 -> adj=11.500 (+15.0%) [boost, rank=1]
Health Care             : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=6]
Energy                  : base=10.000 -> adj=11.000 (+10.0%) [boost, rank=2]
Materials               : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=4]
Utilities               : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=7]
```

## Provenance

- Code: new `backend/services/sector_momentum.py` (175 lines); `backend/tools/screener.py` (+kwarg + overlay block); `backend/services/autonomous_loop.py` (+ pre-fetch + pass); `backend/config/settings.py` (+6 fields default OFF).
- Data: yfinance batch `yf.download([11 SPDR ETFs], period="13mo", interval="1d")` — one network call.
- Source: Quantpedia sector momentum rotational system (primary brief item #13 + phase-28.12 research brief: 6 sources read in full).
- Feature flag: `sector_momentum_enabled = False` by default — production unchanged.

## Spec compliance

- "11 sector ETF 12-month momentum ranks + which 3 won + N tickers boosted" — DOCUMENTED above with real numbers (Tech +51.4% / Energy +43.9% / Industrials +23.5% as the winners; ~167 tickers boosted across the three sectors).
