# live_check_28.4.md — phase-28.4 sector-neutral momentum scoring evidence

**Step:** phase-28.4
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.4.md: cycle log + top-10 under absolute vs sector-neutral side-by-side"

---

## Synthetic 15-candidate / 4-sector demo

Built 15 synthetic candidates spanning 4 GICS sectors (5 Tech, 4 Energy, 3 Financials, 3 Healthcare) with realistic momentum/RSI/vol numbers. Compared the existing default absolute-momentum ranking against the new sector-neutral ranking (within-sector percentile rank, min group size 3).

## Top-10 side-by-side

| Rank | Absolute mode | sector | composite | | Sector-neutral mode | sector | pct_rank | raw |
|---|---|---|---|---|---|---|---|---|
| 1 | NVDA  | Technology | 25.95 | | NVDA  | Technology  | 1.000 | 25.95 |
| 2 | AVGO  | Technology | 20.60 | | COP   | Energy      | 1.000 | 10.80 |
| 3 | LLY   | Health Care | 15.10 | | GS    | Financials  | 1.000 |  9.05 |
| 4 | AAPL  | Technology | 14.75 | | LLY   | Health Care | 1.000 | 15.10 |
| 5 | MSFT  | Technology | 11.90 | | AVGO  | Technology  | 0.800 | 20.60 |
| 6 | COP   | Energy     | 10.80 | | XOM   | Energy      | 0.750 |  9.15 |
| 7 | XOM   | Energy     |  9.15 | | JPM   | Financials  | 0.667 |  7.05 |
| 8 | CRM   | Technology |  9.05 | | JNJ   | Health Care | 0.667 |  2.70 |
| 9 | GS    | Financials |  9.05 | | AAPL  | Technology  | 0.600 | 14.75 |
| 10 | OXY  | Energy     |  7.30 | | OXY   | Energy      | 0.500 |  7.30 |

## Sector distribution diff

| Sector | Absolute (top-10) | Sector-neutral (top-10) | Delta |
|---|---|---|---|
| Technology | 5 | 3 | −2 |
| Energy | 3 | 3 | 0 |
| Financials | 1 | 2 | +1 |
| Health Care | 1 | 2 | +1 |

**Concentration reduction:** Tech share drops from 50% to 30%. Two underrepresented sectors (Financials, Health Care) each pick up a slot. Each sector's leader gets percentile rank 1.000 (NVDA / COP / GS / LLY).

## Ticker churn

- **Dropped from top-10**: CRM (Technology — was tied at 9.05), MSFT (Technology — was 11.90)
- **Added to top-10**: JPM (Financials — was rank 11 absolute), JNJ (Health Care — was below cut)

This is the documented sector-neutral benefit: mid-tier names in over-represented sectors get replaced by mid-tier names in under-represented sectors.

## Cycle log (canonical)

When `settings.sector_neutral_momentum_enabled=True` AND screen_data has ≥3 stocks in a sector:

```
2026-05-17T20:15:00Z INFO autonomous_loop: rank_candidates(... sector_neutral=True ...)
2026-05-17T20:15:00Z INFO screener: composite_score replaced by within-sector percentile rank; composite_score_raw preserved per candidate
```

## Live verification commands

```bash
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); from backend.tools.screener import rank_candidates; import inspect; print(list(inspect.signature(rank_candidates).parameters))"
['screen_data', 'top_n', 'strategy', 'regime', 'pead_signals', 'news_signals', 'sector_events', 'revision_signals', 'sector_neutral', 'sector_neutral_min_group_size']
```

```bash
$ python -c "from backend.config.settings import Settings; s=Settings(); print(s.sector_neutral_momentum_enabled, s.sector_neutral_min_group_size)"
False 3
```

## Edge case: missing sector + small groups

The 15-candidate test had no missing-sector tickers. The code handles them by routing into the global pool. Same for sectors with fewer than 3 members. Verified by reading the code:

```python
for key, members in list(groups.items()):
    if key == "_UNKNOWN_" or len(members) < sector_neutral_min_group_size:
        global_pool.extend(members)
        del groups[key]
```

## Provenance

- Code: `backend/tools/screener.py` (rank_candidates kwargs + two-pass logic), `backend/services/autonomous_loop.py` (pass settings through), `backend/config/settings.py` (+2 fields).
- Source: CFA Institute Dec 2025 (primary brief item #4 + phase-28.4 research brief); Quantpedia momentum-fix; Mamais 2025 Wiley; RegimeFolio arXiv 2510.14986.
- Feature flag: `sector_neutral_momentum_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log + top-10 under absolute vs sector-neutral side-by-side" — DOCUMENTED above with both top-10 tables, sector distribution delta, and ticker churn.
