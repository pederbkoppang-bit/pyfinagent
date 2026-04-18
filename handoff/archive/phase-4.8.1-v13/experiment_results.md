# Experiment Results -- Cycle 80 / phase-4.8 step 4.8.3

Step: 4.8.3 Fractional-Kelly multi-strategy allocator (30% cap)

## What was generated

1. **NEW** `backend/services/kelly_allocator.py`
   - `fractional_kelly(mu, Sigma, k=0.25, cap=0.30)` -- real
     `Sigma^{-1} * mu` via `np.linalg.solve`; scale by k; clip
     long-only; per-strategy cap; renormalize if sum > 1.
   - Rejects non-symmetric / non-PSD / singular Sigma with
     explicit `ValueError` (fail-loud).

2. **NEW** `scripts/risk/kelly_allocator.py`
   - `--dry-run` seeds 5 strategies (momentum, mean-reversion,
     triple-barrier, factor_model, quality) with realistic mu/sigma
     + correlation matrix.
   - Emits `handoff/allocator_output.json` with per-strategy
     alloc_pct, kelly_full, kelly_fractional, capped flag.

3. **NEW** `scripts/audit/kelly_allocator_audit.py`
   Four teeth checks:
   - cap binding: max(alloc_pct) <= 0.30
   - renorm to 1.0 sum
   - **covariance mixing**: uncorrelated 2-strategy fixture
     (mu=0.01, sigma=0.15) gives alloc 0.111 each; same fixture
     at corr=0.9 gives 0.058 each -- diversification penalty
     visible because small mu keeps cap from binding
   - singular Sigma raises ValueError

## Verification (verbatim, immutable)

    $ python scripts/risk/kelly_allocator.py --dry-run && \
      python -c "import json; r=json.load(open('handoff/allocator_output.json')); \
                  assert max(s['alloc_pct'] for s in r['strategies']) <= 0.30"
    {"total_alloc_pct": 1.0, "max_alloc_pct": 0.202}
    exit=0

    $ python scripts/audit/kelly_allocator_audit.py --check
    {"verdict": "PASS", "t1_cap": true, "t2_sum": true,
     "t3_cov": true, "t4_singular": true}

## Success criteria

| Criterion | Result |
|-----------|--------|
| per_strategy_alloc_computed | PASS (5 strategies, 7 fields each) |
| single_strategy_cap_30pct | PASS (max=0.202 <=0.30) |
| covariance_based_mixing | PASS (drift=0.074 when corr 0->0.9) |

## Dry-run allocation (dry-run; informational)

| Strategy | mu_ann | sigma_ann | alloc_pct |
|---|---|---|---|
| momentum | 9% | 16% | 0.202 |
| mean_reversion | 6% | 13% | 0.202 |
| triple_barrier | 8% | 15% | 0.202 |
| factor_model | 7% | 14% | 0.202 |
| quality | 5% | 11% | 0.192 |
| **total** | -- | -- | **1.00** |

All strategies contribute; total fully-invested; no single
strategy > 30% cap. Renorm path exercised (fractional Kelly
summed to 2.526 before capping; capped to 1.50 then normalized).

## Known limitations (tracked follow-up)

- Dry-run mu/sigma are hardcoded seeds for a reproducible demo.
  Live covariance estimation (Ledoit-Wolf shrinkage on the last
  252 strategy-return days) is a separate step.
- Long-only clipping discards negative Kelly solutions. Short-
  enabled mode is out-of-scope.
- k=0.25 is per MacLean/Thorp/Ziemba 2010; adaptive k tuning
  (based on realized DSR) is a later phase-10 concern.
