# live_check — step 65.3 (US+KR per-market health baseline)

Per the immutable `live_check` field ("= the baseline doc itself"), the deliverable IS
**`handoff/away_ops/market_health_baseline.md`**. This file records the immutable-command output + a summary so an
operator can audit at a glance.

## Immutable command output

```
$ test -f handoff/away_ops/market_health_baseline.md && grep -c 'HEALTHY-THRESHOLD' handoff/away_ops/market_health_baseline.md
10
# exit 0  (10 HEALTHY-THRESHOLD lines; 4 verbatim SQL blocks; pre/post-61.1-fix split present)
```

## Baseline summary (all from live BQ, $0 read-only)

| market | buys | sells | win rate | fees (USD) | fees % NAV | median hold |
|--------|------|-------|----------|------------|------------|-------------|
| US | 11 | 17 | 70.6% (n=17, descriptive) | $20.30 | 0.085% | 3d |
| KR | 5 | 5 | 20.0% (n=5, descriptive) | $4.82 | 0.020% | 1d |
| EU | 0 | 0 | — (no trades yet) | $0 | 0% | — |

**Churn split (criterion 3):** PRE-FIX (06-01→06-11, `paper_swap_churn_fix_enabled` OFF) = 12 swap-exits (US 10, KR 2),
many ≤1d holds. POST-FIX (06-12+, flag ON) = **0 swap-exits** (fix holds) but thin sample (US 2 sells / KR 1),
confounded by the away-ops quiet period → trend PENDING more cycles. Segments NEVER merged.

**Sample-size honesty:** US 17 / KR 5 closed < the ~30/metric inferential minimum → win-rate/PF are DESCRIPTIVE; 65.4
should judge primarily against the STRUCTURAL HEALTHY-THRESHOLD lines (holding-days, churn-swap-hold, fee-drag,
liveness).

## Method / boundaries
Read-only BQ SELECT via the Python client (ADC, us-central1) + Python aggregation. ZERO metered LLM. NO production
code changed (git = only the baseline doc + handoff). $0; live book untouched; historical_macro FROZEN. Full per-market
tables + exit-reason mix + holding-day distribution + the 4 verbatim SQL blocks are in
`handoff/away_ops/market_health_baseline.md`.
