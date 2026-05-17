# Experiment Results — phase-28.7 — Multidimensional momentum composite

**Step ID:** phase-28.7
**Date:** 2026-05-17
**Cycle:** 1

---

## What was built / changed

### Files modified

| File | Change |
|---|---|
| `backend/config/settings.py` | Added 5 fields after sector_momentum block: `multidim_momentum_enabled` (False), `multidim_momentum_weight_price` (0.35), `multidim_momentum_weight_52w_high` (0.25), `multidim_momentum_weight_sue` (0.20), `multidim_momentum_weight_sector` (0.20). |
| `backend/tools/screener.py` | (a) `screen_universe` now also returns `pct_to_52w_high` (current price / trailing-252d max). (b) Added new `_zscore()` helper + `_apply_multidim_momentum()` function that z-scores 4 components across the universe and rewrites composite_score as a weighted blend; preserves original on `composite_score_raw`. (c) `rank_candidates` accepts `multidim_momentum`, `multidim_weights`, `pead_signals_lookup` kwargs; calls the new function when enabled. |
| `backend/services/autonomous_loop.py` | Passes `multidim_momentum=settings.multidim_momentum_enabled` + `multidim_weights={...}` dict into `rank_candidates`. |

### Implementation: z-blend formula

```
composite_score = w_price * z(price_momentum)
                + w_52w * z(pct_to_52w_high)
                + w_sue * z(pead.surprise_score)
                + w_sec * z(sector_momentum_boost - 1.0)
```

with default weights 0.35 / 0.25 / 0.20 / 0.20 (sum = 1.0). Missing components contribute 0 (mean) per stock. Std=0 → 0 z-score.

---

## Verification — verbatim output

### 1. Immutable verification command

```
$ source .venv/bin/activate && python -c "import ast; ast.parse(open('backend/tools/screener.py').read()); print('syntax OK')" && grep -qE '52.{0,5}week|fifty.two|composite_momentum' backend/tools/screener.py && echo "MASTERPLAN VERIFICATION: PASS"
syntax OK
MASTERPLAN VERIFICATION: PASS
```

EXIT 0. **PASS.**

### 2. Smoke — 10 candidates / 5 sectors, naive vs multidim with real sector_momentum + synthetic PEAD

```
=== NAIVE price-only momentum (default) ===
   1 NVDA   Technology             composite= 25.950
   2 LLY    Health Care            composite= 15.100
   3 AAPL   Technology             composite= 14.750
   4 MSFT   Technology             composite= 11.900
   5 COP    Energy                 composite= 10.800
   6 XOM    Energy                 composite=  9.150
   7 JPM    Financials             composite=  7.050
   8 CVX    Energy                 composite=  6.300
   9 GME    Consumer Discretionary composite=  4.250
  10 JNJ    Health Care            composite=  2.700

=== MULTIDIM composite (with PEAD + sector_momentum) ===
   1 NVDA   Technology             z-blend= +1.557 raw=29.842
   2 AAPL   Technology             z-blend= +0.655 raw=16.962
   3 COP    Energy                 z-blend= +0.409 raw=11.880
   4 MSFT   Technology             z-blend= +0.334 raw=13.685
   5 LLY    Health Care            z-blend= +0.150 raw=15.100
   6 XOM    Energy                 z-blend= +0.116 raw=10.065
   7 CVX    Energy                 z-blend= -0.067 raw= 6.930
   8 JPM    Financials             z-blend= -0.658 raw= 7.050
   9 JNJ    Health Care            z-blend= -0.720 raw= 2.700
  10 GME    Consumer Discretionary z-blend= -1.777 raw= 4.250
```

### Top rank shifts (multidim vs naive)

| Ticker | Naive rank | Multidim rank | Δ | Driver |
|---|---|---|---|---|
| LLY | #2 | #5 | **+3 (worse)** | Health Care not in top-3 sectors; no sector boost; moderate 52w-high proximity |
| COP | #5 | #3 | **−2 (better)** | Positive SUE (+0.10) + Energy sector boost (rank 2) |
| AAPL | #3 | #2 | −1 | Tech sector boost (leader, +15%) + 52w-high proximity 0.95 |
| JPM | #7 | #8 | +1 | Financials rank 11 (worst sector) drags slightly |
| CVX | #8 | #7 | −1 | Energy sector boost |
| JNJ | #10 | #9 | −1 | Edge of universe; minor shift |
| GME | #9 | #10 | +1 | Negative PEAD (-0.20) drops to bottom |

NVDA stays #1 (overwhelming on all 4 components). The multidim composite correctly rewards stocks that combine strong price momentum WITH a near-52w high WITH positive PEAD WITH a winning sector — not just one of those.

**PASS** — ranking shifts are explainable by the 4-component formula and consistent with the literature (George-Hwang anchoring; Novy-Marx earnings momentum; sector rotation).

---

## Success criteria mapping

| Criterion (immutable) | Evidence | Result |
|---|---|---|
| `composite_momentum_function_added` | `_apply_multidim_momentum()` + `_zscore()` helpers in screener.py | PASS |
| `weighting_scheme_documented_with_source_citation` | settings field descriptions cite CFA Dec 2025 + George-Hwang 2004; multi-line phase-28.7 comment in rank_candidates explains the components + sources | PASS |
| `feature_flag_composite_momentum_enabled_default_false` | `Settings().multidim_momentum_enabled == False` | PASS |
| `live_check_compares_naive_vs_composite_top10_for_one_cycle` | live_check_28.7.md has side-by-side top-10 + rank-shift table + driver explanations | PASS |

---

## Next

Q/A pass via fresh `qa` subagent. On PASS: append Cycle 22, flip phase-28.7. Post-launch tier progress: 2/7.
