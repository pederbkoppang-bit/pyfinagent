# Stage 1 Results -- screen + rank + sector enrichment

**Step:** phase-31.0.1 -- Smoketest Stage 1.
**Date:** 2026-05-20.
**Verdict:** **PASS.**

## Summary

Executed Test Design #3 (full production chain: screen -> rank ->
sector enrichment) per researcher recommendation. All 4 tickers
returned with non-empty sector + numeric composite_score.

## Execution

```python
from backend.tools.screener import screen_universe, rank_candidates
import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA", "JPM"]

# 1a: screen
screen_data = screen_universe(tickers=tickers, period="6mo")
# -> 4 rows with ticker + current_price + momentum + rsi + etc.

# 1b: rank (composite_score added)
ranked = rank_candidates(screen_data, top_n=4, strategy="momentum")
# -> 4 rows ordered by composite_score desc, with composite_score field

# 1c: sector enrichment (mirrors _fetch_ticker_meta)
sector_lookup = {t: yf.Ticker(t).info.get("sector", "Unknown") for t in tickers}
for row in ranked:
    row["sector"] = sector_lookup.get(row["ticker"], "Unknown")
```

## Output (4 rows)

| Ticker | Sector | composite_score | current_price | momentum_3m | rsi_14 |
|--------|--------|-----------------|---------------|-------------|--------|
| NVDA | Technology | 15.283 | 220.61 | 17.36 | 58.5 |
| AAPL | Technology | 8.107 | 298.97 | 13.20 | 84.1 |
| MSFT | Technology | -1.557 | 417.42 | 4.70 | 45.4 |
| JPM | Financial Services | -3.986 | 295.70 | -3.75 | 36.9 |

Notes:
- 3 Tech + 1 Financials basket -- exercises phase-30.5 sector NAV-pct
  cap in downstream stages.
- Sector taxonomy is yfinance native (`Technology`, `Financial
  Services`); pyfinagent's downstream code accepts both yfinance + GICS
  per the brief.
- composite_score is signed -- NVDA + AAPL positive (momentum-favored),
  MSFT + JPM negative (lower momentum / higher RSI extreme on AAPL
  84.1 risks overbought penalty).

## Assertions (all PASS)

1. `len(ranked) == 4` -> PASS
2. Each row has `ticker in tickers` -> PASS
3. Each row has numeric `current_price` -> PASS
4. Each row has numeric `composite_score` -> PASS
5. Each row has non-empty `sector` string -> PASS
6. Schema invariants `{ticker, current_price, composite_score, sector}` -> PASS

## Output file

`handoff/smoketest_20260520/STAGE_1_screen_universe_output.json` (52
lines, machine-readable for downstream Stages 4+ to consume).

## Hard guardrails attestation

- NO LLM calls. Pure code execution + yfinance read.
- NO production BQ writes.
- NO Alpaca calls.
- Loop STAYS PAUSED.
- Live yfinance reads only (read-only data fetch per goal).

## Verdict

**PASS.** All 6 assertions green. Stage 1 deliverable persisted.
Researcher gate was honest-`false` on the 20-source floor (18 of 20
external sources fetched in full); content-completeness of the brief
fully covers the Stage 1 test design. Q/A judges whether the floor
miss matters for a P3-equivalent smoke verification.
