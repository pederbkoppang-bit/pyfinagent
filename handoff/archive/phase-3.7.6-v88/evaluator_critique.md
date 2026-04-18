# Evaluator Critique -- Cycle 81 / phase-4.8 step 4.8.4

Step: 4.8.4 Drift monitor (PSI + 20-day rolling IC)

## Dual-evaluator run (parallel, anti-rubber-stamp)

## qa-evaluator: PASS

Substantive 6-point review:
1. **PSI canonical**: `sum((a_frac - e_frac) * log(a_frac /
   e_frac))` with quantile bins + eps floor. Siddiqi 2006 form.
2. **Spearman canonical**: `argsort().argsort()` produces ranks,
   then Pearson on ranks. Standard textbook implementation.
3. **Rolling window = 20**: `IC_WINDOW = 20`, loop `range(window,
   p.size + 1)` yields one value per day. Correct.
4. **Thresholds compared dynamically**: reasons appended only
   when `psi > 0.25` or `ic_trend_sustained_neg`. Benign fixture
   (psi=0.013, ic=+0.43) produces empty reasons; trip fixtures fire
   correct strings.
5. **Fixture honesty**: psi-trip 1.87 is dramatic but representative
   of regime break; ic-trip -0.93 is worst-case demonstrator of the
   sustained-negative logic. Threshold is what's under test.
6. **Threshold 0.25 = Siddiqi standard**.

## harness-verifier: PASS

6/6 mechanical checks green:
- Immutable verification exits 0; 3 models with psi+ic_20d.
- Audit clean with all 3 fixtures ok.
- Artifact structure correct.
- **PSI formula test**: identical distribution -> PSI < 0.02;
  shifted distribution -> PSI > 0.2 (correct shape).
- **Spearman test**: `x` vs `x^3` (monotone nonlinear) -> IC=1.0;
  `x` vs `-x^3` -> IC=-1.0. Rank correlation, not value correlation.
- **Mutation test**: disable the `if psi > PSI_FREEZE_THRESHOLD:`
  line -> audit rc=1. Cap teeth proven. File restored.

## Decision: PASS (evaluator-owned)

Both evaluators substantively green with canonical-form formula
verification + mutation resistance proof. No rubber-stamp.
