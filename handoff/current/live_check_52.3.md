# live_check -- phase-52.3: DSR/SR-difference robustness gate for the 52wh edge -> REJECT

**Step:** 52.3 | **Date:** 2026-06-01 | **Result shape:** the rigorous test of whether the 52.1
+0.05 Sharpe improvement is real -> **REJECT** (not statistically distinguishable from noise). The
52wh tilt stays OFF (the 52.2 wiring remains dormant). NO live change. The PBO/DSR overfitting
control (the goal's explicit discipline) PREVENTED enabling a noise-edge on the +20% engine.

**Command:**
```
source .venv/bin/activate
PYTHONPATH=. python scripts/ablation/dsr_52wh_verdict.py   # reads the pinned _52wh_paired_returns.json
```

## Verbatim verdict
```
=== phase-52.3: 52wh-tilt robustness verdict (a-priori rule) ===
n_rebalances=47  n_boot=5000
SR_tilt=1.445  SR_base=1.388  delta=+0.057
PRIMARY  Ledoit-Wolf stationary-bootstrap p (one-sided, H0 SR_tilt<=SR_base) = 0.2420
         -> R1 (p < 0.05): False
         bootstrap 90% CI for delta = [-0.073, +0.188]  (se=0.080)
         -> R2 (delta >= +0.05 AND CI_low > 0): False
SECONDARY DSR(abs SR=1.45, 5 trials, T=47) = 1.000  (weak here; report only)

VERDICT: REJECT -- the +0.05 edge is NOT statistically distinguishable from noise
```

## Interpretation (criterion #1/#2 -- a-priori rule, honestly reported)
- **PRIMARY (Ledoit-Wolf SR-difference, stationary bootstrap, one-sided): p = 0.242** -> R1 FAILS (need < 0.05). The bootstrap SE of the delta (0.080) is LARGER than the delta itself (0.057), and the 90% CI [-0.073, +0.188] straddles zero -> R2 FAILS (CI lower bound < 0). The +0.057 point estimate is well within the noise band of trying 5 configs over 47 monthly rebalances. (Matches the run-to-run drift +0.047..+0.057 -- the edge IS the noise.)
- **SECONDARY DSR = 1.000** -- as the researcher proved, DSR is a WEAK discriminator here (it returns ~1.0 for both 1.45 and 1.39 absolute Sharpes; it answers "is 1.45 a fluke of the search" -> no, but NOT "is the +0.05 delta real" -> that's the SR-difference test, which says no).
- **VERDICT: REJECT.** A-priori rule = ENABLE iff (R1 AND R2); both fail -> REJECT. The 52wh tilt flag stays OFF (dormant). McLean-Pontiff would haircut it further live, so this is the right call.

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | the +0.05 improvement tested rigorously (SR-difference/paired + DSR haircut), reusing compute_deflated_sharpe, on the existing replay data | Ledoit-Wolf stationary-bootstrap on the 47 pinned paired returns (PRIMARY) + compute_deflated_sharpe (SECONDARY) | PASS |
| 2 | a-priori rule -> ENABLE/REJECT; honest 'not robust -> do not enable' is valid | rule fixed in the contract BEFORE running; verdict REJECT, reported with all stats | PASS |
| 3 | NO live change; 52wh flag stays OFF | diff = analytics.py (+sharpe_diff_test) + replay dump + verdict script + test + pinned JSON; screener.py/autonomous_loop/the flag UNTOUCHED (still OFF) | PASS |
| 4 | live_check records p/DSR + rule + verdict | this file | PASS |

## pytest (the SR-difference test is correct)
```
$ python -m pytest backend/tests/test_phase_52_3_dsr.py -q
.....                                                                    [100%]
5 passed in 1.89s
```
(identical series -> not significant; clearly-better -> p<0.05 + CI_low>0; worse -> not significant; deterministic seed; None/short -> safe.)

## What this CLOSES (the element-2 alpha-signal question)
Across ALL tested element-2 levers on our universe: rotation (architecturally disconnected + losing
strategies, REJECT), sector-neutral breadth (-0.166 Sharpe, REJECT), vol-scaling (+0.015, marginal),
52wh tilt (+0.057 point but p=0.24, NOT robust -> REJECT). **No statistically-robust alpha
enhancement was found among the cheap price-based levers.** The +20% momentum engine STANDS. This is
the honest, overfitting-controlled outcome -- not a failure: it prevented shipping noise as alpha.

## Recommendation / next
- **Keep the 52wh tilt OFF** (the 52.2 wiring stays dormant; available for re-test if more data accrues).
- **The bigger-edge path** (if pursued): 52.4 residual momentum (Blitz-Huij-Martens; higher-evidenced; a larger build) -- the only remaining cited lever with a plausibly-larger effect. Otherwise ACCEPT the engine as-is and let the multi-market expansion (now live) be the money lever.
- **MEASURE Monday's first multi-market cycle** -- the real money test; the EU/KR universe (structurally non-tech) is the live expansion that doesn't depend on a marginal US-momentum tweak.
- Reproducibility: the pinned `_52wh_paired_returns.json` + the seeded bootstrap make this verdict deterministic for the Q/A.
