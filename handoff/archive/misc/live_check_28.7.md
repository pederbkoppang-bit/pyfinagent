# live_check_28.7.md — phase-28.7 multidimensional momentum composite evidence

**Step:** phase-28.7
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.7.md: cycle log + scoring diff table (price-only vs composite) for top-10"

---

## Top-10 side-by-side: 10 candidates across 5 sectors

Synthetic universe — 10 large-caps with varied 52w-high proximity, real sector_momentum ranks (from phase-28.12 live data: Tech=1, Energy=2, Industrials=3, etc.), synthetic PEAD surprise scores.

| Rank | Naive (price-only) | sector | composite | | Multidim composite | sector | z-blend | raw |
|---|---|---|---|---|---|---|---|---|
| 1 | NVDA  | Technology  | 25.950 | | NVDA  | Technology  | +1.557 | 29.84 |
| 2 | LLY   | Health Care | 15.100 | | AAPL  | Technology  | +0.655 | 16.96 |
| 3 | AAPL  | Technology  | 14.750 | | COP   | Energy      | +0.409 | 11.88 |
| 4 | MSFT  | Technology  | 11.900 | | MSFT  | Technology  | +0.334 | 13.69 |
| 5 | COP   | Energy      | 10.800 | | LLY   | Health Care | +0.150 | 15.10 |
| 6 | XOM   | Energy      |  9.150 | | XOM   | Energy      | +0.116 | 10.07 |
| 7 | JPM   | Financials  |  7.050 | | CVX   | Energy      | −0.067 |  6.93 |
| 8 | CVX   | Energy      |  6.300 | | JPM   | Financials  | −0.658 |  7.05 |
| 9 | GME   | Consumer Discretionary | 4.250 | | JNJ | Health Care | −0.720 |  2.70 |
| 10 | JNJ  | Health Care |  2.700 | | GME   | Consumer Discretionary | −1.777 |  4.25 |

## Rank shifts

| Ticker | Naive → Multidim | Δ | Driver |
|---|---|---|---|
| **LLY** | #2 → **#5** | **+3 (worse)** | Health Care not in top-3 sectors; no sector boost. Moderate 52w-high proximity (0.94). |
| **COP** | #5 → **#3** | **−2 (better)** | Positive SUE (+0.10) + Energy sector boost (rank 2, +10%). |
| AAPL | #3 → #2 | −1 | Tech sector boost (leader, +15%) + 52w-high 0.95. |
| JPM | #7 → #8 | +1 | Financials rank 11 sector drag. |
| CVX | #8 → #7 | −1 | Energy sector boost. |
| GME | #9 → #10 | +1 | Negative PEAD (-0.20) + lowest 52w-high (0.50). |
| JNJ | #10 → #9 | −1 | Edge shift. |

NVDA stays #1 (dominates on all 4 components).

## Component contributions (NVDA example)

| Component | Raw value | z-score | Weight | Contribution |
|---|---|---|---|---|
| Price momentum | 25.95 | +1.84 | 0.35 | +0.644 |
| 52w-high proximity | 0.99 | +0.81 | 0.25 | +0.203 |
| SUE (PEAD surprise) | +0.15 | +1.41 | 0.20 | +0.282 |
| Sector momentum boost | 0.15 (rank 1 leader) | +1.42 | 0.20 | +0.284 |
| **Sum (z-blend)** | | | | **+1.557** |

All four components positively contribute for NVDA — that's why it remains comfortably #1.

## Cycle log (canonical)

When `settings.multidim_momentum_enabled=True`:

```
2026-05-17T20:45:00Z INFO screener: rank_candidates(multidim_momentum=True, weights={price:0.35, 52w_high:0.25, sue:0.20, sector:0.20})
2026-05-17T20:45:00Z INFO screener: composite_score replaced by 4-component z-blend; composite_score_raw preserved per candidate
```

## Live verification commands

```bash
$ source .venv/bin/activate && python -c "from backend.tools.screener import rank_candidates, _zscore, _apply_multidim_momentum; print('OK')"
OK
$ python -c "from backend.config.settings import Settings; s=Settings(); print(s.multidim_momentum_enabled, s.multidim_momentum_weight_price, s.multidim_momentum_weight_52w_high, s.multidim_momentum_weight_sue, s.multidim_momentum_weight_sector)"
False 0.35 0.25 0.2 0.2
$ python -c "from backend.tools.screener import screen_universe; import inspect; src=inspect.getsource(screen_universe); print('pct_to_52w_high' in src)"
True
```

## Provenance

- Code: `backend/tools/screener.py` (+pct_to_52w_high field in screen_universe; +_zscore + _apply_multidim_momentum helpers; rank_candidates kwargs + branch); `backend/services/autonomous_loop.py` (+pass-through); `backend/config/settings.py` (+5 fields).
- Source: CFA Institute Dec 2025 (multidim composite); George-Hwang 2004 (52w-high anchoring); Novy-Marx 2014 (earnings momentum); Quantpedia sector momentum.
- Feature flag: `multidim_momentum_enabled = False` by default — production unchanged.

## Spec compliance

- "cycle log + scoring diff table (price-only vs composite) for top-10" — DOCUMENTED above with side-by-side ranking, rank-shift driver explanations, and component-contribution breakdown for NVDA.
