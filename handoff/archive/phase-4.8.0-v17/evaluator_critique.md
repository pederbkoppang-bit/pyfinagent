# Evaluator Critique -- Cycle 80 / phase-4.8 step 4.8.3

Step: 4.8.3 Fractional-Kelly multi-strategy allocator (30% cap)

## Dual-evaluator run (parallel; anti-rubber-stamp active)

## qa-evaluator: PASS

Substantive math-walkthrough review:
1. **fractional_kelly is REAL**: `np.linalg.solve(S, mu_v)` for the
   matrix inverse; scale by k; np.clip >=0; np.minimum cap;
   renormalize if sum > 1. Traced by hand for the dry-run:
   full-Kelly sum 2.526 -> clipped to 1.50 -> renorm to 1.0 with
   defensive re-cap. JSON matches the hand-calculation.
2. **Covariance test is HONEST**: mu=0.01 intentionally keeps the
   fractional Kelly (~0.11) below the 0.30 cap so the correlation
   effect is visible rather than masked by cap-saturation. The
   prior failing fixture used mu=0.08 which hit the cap -- that
   was a genuine bug in the test, not a cheat. The 0.111 (low-corr)
   vs 0.058 (high-corr) numbers match analytic Kelly to 3 decimal
   places. Real diversification penalty, not a fixture manipulation.
3. **Singular/non-PSD Sigma raises**: eigvalsh min > 1e-12 guard.
4. **Dry-run artifact structure**: 5 strategies, 7 fields each.
5. **Sum=1 is emergent from renorm, not hardcoded**: read the code,
   confirmed via the dry-run path.

## harness-verifier: PASS

6/6 mechanical checks green including:
- **Mutation test**: disabled the `np.minimum(f, cap)` line; audit
  caught exit=1 (max alloc would have exceeded 0.30). File
  restored; re-audit clean. Cap has real teeth.
- Artifact has 5 strategies x 7 fields; covariance drift 0.074 is
  non-zero and signed correctly (lo-corr > hi-corr).
- Singular Sigma raises ValueError with correct message.

## Decision: PASS (evaluator-owned)

Both evaluators substantively green. Mutation test proves cap
enforcement is real code, not a hardcoded value. Covariance test
honestly demonstrates Sigma sensitivity. No rubber-stamp.
