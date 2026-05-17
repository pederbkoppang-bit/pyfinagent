# Experiment Results — phase-28.12 — Sector-ETF momentum overlay

**Step ID:** phase-28.12
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | Added 6 fields after sector_neutral block: `sector_momentum_enabled` (False), `sector_momentum_lookback_months` (12), `sector_momentum_top_n` (3), `sector_momentum_boost_top` (1.10), `sector_momentum_boost_leader` (1.15), `sector_momentum_cache_hours` (24). |
| `backend/tools/screener.py` | Added `sector_momentum_ranks=None` kwarg to `rank_candidates`. Inserted `apply_sector_momentum_to_score` call in per-stock loop AFTER analyst_revisions block. |
| `backend/services/autonomous_loop.py` | Pre-fetch sector ETF momentum ranks when flag is on; pass to `rank_candidates`. Surface top-3 sectors in cycle summary. |

### Files created

| File | Purpose |
|---|---|
| `backend/services/sector_momentum.py` | New 175-line module. `RankedSector` Pydantic model. `fetch_sector_momentum_ranks()` async helper: yfinance batch download (11 SPDR ETFs in one round-trip) → 12m total return per ETF → rank → boost multiplier (1.15 #1, 1.10 top-N, 1.0 otherwise). `apply_sector_momentum_to_score()` helper. JSON cache 24h. |

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/services/sector_momentum.py').read()); print('syntax OK')" && grep -q 'sector_momentum_enabled' backend/config/settings.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Live `fetch_sector_momentum_ranks()` — real 11-SPDR data

```
INFO backend.services.sector_momentum: sector_momentum: top-3 sectors [('Technology', '+51.4%'), ('Energy', '+43.9%'), ('Industrials', '+23.5%')]

Returned 11 sectors

Rank Sector                   ETF   12m return  boost
   1 Technology               XLK       51.43%   1.15 <- TOP
   2 Energy                   XLE       43.93%   1.10 <- TOP
   3 Industrials              XLI       23.55%   1.10 <- TOP
   4 Materials                XLB       20.25%   1.00
   5 Communication Services   XLC       16.71%   1.00
   6 Health Care              XLV       14.69%   1.00
   7 Utilities                XLU       13.79%   1.00
   8 Real Estate              XLRE       9.80%   1.00
   9 Consumer Staples         XLP        9.40%   1.00
  10 Consumer Discretionary   XLY        8.70%   1.00
  11 Financials               XLF        1.86%   1.00
```

**REAL DATA:** As of 2026-05-17, the picker would boost Technology stocks +15%, Energy +10%, Industrials +10% — and leave the other 8 sectors at identity. Tech leads with +51.43% trailing 12m return. Financials are the laggard at +1.86%.

### 3. `apply_sector_momentum_to_score()` smoke

```
Technology              : base=10.000 -> adj=11.500 (+15.0%) [boost, rank=1]
Health Care             : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=6]
Energy                  : base=10.000 -> adj=11.000 (+10.0%) [boost, rank=2]
Materials               : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=4]
Utilities               : base=10.000 -> adj=10.000 ( +0.0%) [identity, rank=7]
```

Boost applied correctly to top-3; identity for non-top sectors. **PASS.**

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `sector_momentum_module_created` | `backend/services/sector_momentum.py` exists, imports cleanly, 11 GICS sectors loaded | PASS |
| `top_3_sector_logic_documented` | Module docstring + settings field descriptions cite Quantpedia + 1.15/1.10 boost rationale; `top_n=3` is the parameterized default | PASS |
| `feature_flag_sector_momentum_enabled_default_false` | `Settings().sector_momentum_enabled == False` confirmed | PASS |
| `live_check_lists_winning_sectors_and_boost_recipients` | live_check_28.12.md captures all 11 sectors ranked + which 3 won + per-sector boost multipliers | PASS |

---

## Artifact shape

`fetch_sector_momentum_ranks` returns `dict[GICS_sector, RankedSector]`:

```python
{
    "Technology": RankedSector(sector="Technology", etf="XLK", momentum=0.5143, rank=1, boost_multiplier=1.15),
    "Energy":     RankedSector(sector="Energy",     etf="XLE", momentum=0.4393, rank=2, boost_multiplier=1.10),
    ...
}
```

`apply_sector_momentum_to_score(base, sector, ranks)` returns `base * ranks[sector].boost_multiplier` when sector is in ranks; identity otherwise.

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 21, flip phase-28.12. Next per proposal sequencing: **28.7 — Multidimensional momentum composite** (M effort).
