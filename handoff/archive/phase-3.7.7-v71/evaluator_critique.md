# Evaluator Critique -- Cycle 79 / phase-4.8 step 4.8.2

Step: 4.8.2 Portfolio CVaR + factor-exposure gate

## Dual-evaluator run (parallel; anti-rubber-stamp active)

## qa-evaluator: PASS

6-point substantive review:
1. **CVaR math**: Rockafellar-Uryasev historical method. `losses =
   -returns`; threshold = `np.quantile(losses, 0.975)`; CVaR =
   `tail.mean()` over `losses >= threshold`. Returns positive loss
   magnitude. `max(..., 0.0)` is a defensive floor only when no
   losses exist, not a cheat.
2. **FF3 math**: real OLS via `np.linalg.lstsq` with R^2 correctly
   computed from residual sum of squares. Subtracts rf properly.
   Not a stub.
3. **Gate discrimination**: all three fixtures use REAL
   construction -- no hardcoded betas. Beta-trip fixture regresses
   to 2.179 because port was constructed as `2.2 * mkt + noise`.
4. **blocking_reasons populated by real checks**: append only when
   `cvar > CVAR_LIMIT_PCT` and `abs(beta) > BETA_CAP`. Benign
   passes an empty list.
5. **Seeded transparency**: `data_source` field reflects the path;
   seed/n/sigma recorded in meta. Scope honestly disclosed.
6. **No constant-return fakes**: empty-array path returns 0
   defensively; not the normal route.

## harness-verifier: PASS

6/6 mechanical checks green:
- Immutable verification exits 0.
- Audit runs clean (verdict=PASS, benign_ok, cvar_trip_ok,
  beta_trip_ok all true).
- Artifact structure correct.
- CVaR constant-loss test: `compute_cvar([-0.05]*252) == 0.05`.
- FF3 recovers beta ~1.5 from constructed 1.5*mkt series; R^2>0.5.
- **Gate-teeth mutation**: injected `if False and cvar > ...` to
  disable both blocking paths -> audit returned rc=1 (caught the
  silenced gate). File restored verbatim.

## Decision: PASS (evaluator-owned)

Both evaluators substantively green with a deliberate mutation test
proving the gate has real teeth, not a constant-true. No rubber-stamp.
