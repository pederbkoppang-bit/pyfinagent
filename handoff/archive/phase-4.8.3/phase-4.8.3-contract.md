# Contract -- Cycle 80 / phase-4.8 step 4.8.3

Step: 4.8.3 Fractional-Kelly multi-strategy allocator (30% cap)

## Hypothesis

Kelly 1956 / Thorp 1971 multi-asset optimal allocation for N
strategies with expected excess returns mu and covariance Sigma is
`f* = Sigma^{-1} * mu`. Full Kelly has ruinous drawdowns (growth-
optimal but ergodic-bet-blowing); the standard practitioner fix is
**fractional Kelly** (MacLean/Thorp/Ziemba 2010 review): scale
f* by k in [0.25, 0.50]. A per-strategy hard cap (30% here)
guards against concentration even when one strategy dominates Kelly.

Ship the allocator as a library + CLI + audit:
1. `backend/services/kelly_allocator.py::fractional_kelly(mu, Sigma,
   k=0.25, cap=0.30)` returning per-strategy allocation.
2. `scripts/risk/kelly_allocator.py --dry-run` with 5 seeded
   strategies (momentum / mean-reversion / triple-barrier / factor
   / quality) whose mu + Sigma come from a deterministic seed.
3. Audit proves covariance REALLY drives the mix: two strategies
   with identical mu but different correlations must get different
   allocations.

## Scope

Files created:

1. **NEW** `backend/services/kelly_allocator.py`
   - `fractional_kelly(mu, Sigma, k=0.25, cap=0.30)` ->
     list[float] allocation fractions summing to <=1.0.
   - Enforces per-strategy cap: after `k * Sigma^{-1} * mu`,
     clip each to [0, cap]. If the clipped sum > 1, renormalize
     to sum=1 (stay-fully-invested rule).
   - Rejects non-PSD or singular Sigma with explicit ValueError
     (fail-loud over silent bad allocations).

2. **NEW** `scripts/risk/kelly_allocator.py`
   - `--dry-run` seeds 5 strategies + runs the library.
   - Emits `handoff/allocator_output.json` with `strategies: [{
     name, mu_annual, sigma_annual, alloc_pct, kelly_full,
     kelly_fractional, capped}]` and top-level `config` (k, cap).
   - Exits 0 always in dry-run.

3. **NEW** `scripts/audit/kelly_allocator_audit.py`
   - Discriminating tests:
     (a) max allocation <= 0.30 (the cap)
     (b) allocations sum to ~1.0 (fully invested)
     (c) **covariance test**: with two otherwise-identical
         strategies, swapping correlation 0.0 -> 0.9 SHIFTS
         allocations (not a constant output).
     (d) rejects singular covariance with ValueError.

## Immutable success criteria

1. per_strategy_alloc_computed -- JSON has 5 strategies with
   alloc_pct each.
2. single_strategy_cap_30pct -- max alloc_pct <= 0.30.
3. covariance_based_mixing -- audit (c) passes, proving the
   allocator actually uses Sigma (not mu-only).

## Verification (immutable)

    python scripts/risk/kelly_allocator.py --dry-run && \
    python -c "import json; r=json.load(open('handoff/allocator_output.json')); assert max(s['alloc_pct'] for s in r['strategies']) <= 0.30"

Plus:

    python scripts/audit/kelly_allocator_audit.py --check

## Anti-rubber-stamp

qa must check:
- Sigma is actually inverted (numpy.linalg.solve) -- not a stub.
- Fractional k default is 0.25 (conservative industry standard).
- Cap is ENFORCED AFTER the Kelly computation (clip + renorm),
  not by hardcoding alloc = min(x, 0.30) without a renormalization.
- Covariance-mixing test actually flips allocations when correlation
  changes (audit fixture (c)).

## References

- Kelly 1956 "A New Interpretation of Information Rate"
- Thorp 1971 "Portfolio Choice and the Kelly Criterion"
- MacLean/Thorp/Ziemba 2010 "The Kelly Capital Growth Investment
  Criterion" (World Scientific) -- recommends k=0.25 for low vol-drag
- Rising 2013 "The Kelly Criterion" -- 30-50% cap conventions
