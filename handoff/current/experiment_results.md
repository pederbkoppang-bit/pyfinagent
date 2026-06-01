# experiment_results -- phase-52.3: DSR/SR-difference robustness gate -> REJECT the 52wh edge

**Step:** 52.3 | **Date:** 2026-06-01 | **$0 LLM** | no pip | **NO live change** | GENERATE complete

## Outcome in one line
Rigorously tested the 52.1 +0.05 Sharpe improvement (paired Ledoit-Wolf SR-difference, stationary
bootstrap) -> **REJECT (one-sided p=0.242, 90% CI [-0.073,+0.188] straddles 0)**: the edge is NOT
statistically distinguishable from noise. The 52wh tilt stays OFF (52.2 wiring dormant). The PBO/DSR
overfitting control worked -- it prevented enabling a noise-edge on the live +20% engine.

## What was built / changed (analysis only; NO live engine change)

| File | Change |
|------|--------|
| `backend/backtest/analytics.py` | NEW `sharpe_diff_test(ret_a, ret_b, ppy=12, n_boot, block, seed, ci)` -- Ledoit-Wolf (2008) SR-difference via a stationary (Politis-Romano 1994) bootstrap of the JOINT paired rows; one-sided p + central CI + se; deterministic. (The repo had no SR-difference test; grep=0.) |
| `scripts/ablation/sector_neutral_replay.py` | dumps the paired monthly arrays (baseline, hi52_k0.5) + the 5 config Sharpes -> `handoff/current/_52wh_paired_returns.json` (the reproducibility PIN). |
| `scripts/ablation/dsr_52wh_verdict.py` | NEW -- loads the pinned JSON, runs the PRIMARY LW test + SECONDARY DSR, applies the a-priori rule -> ENABLE/REJECT. |
| `backend/tests/test_phase_52_3_dsr.py` | NEW 5 tests (sharpe_diff_test: identical->not-sig; clearly-better->p<0.05+CI_low>0; worse->not-sig; deterministic; None/short-safe). |

## The verdict (criterion #1/#2)
```
SR_tilt=1.445  SR_base=1.388  delta=+0.057  (n=47, n_boot=5000)
PRIMARY Ledoit-Wolf one-sided p = 0.2420  -> R1 (p<0.05): False
        bootstrap 90% CI for delta = [-0.073, +0.188]  (se=0.080)  -> R2 (delta>=+0.05 AND CI_low>0): False
SECONDARY DSR(abs SR=1.45, 5 trials) = 1.000  (weak discriminator -- report only)
VERDICT: REJECT (a-priori rule: ENABLE iff R1 AND R2; both fail)
```
The bootstrap SE (0.080) > the delta (0.057); the CI straddles zero -> the +0.057 is within selection-bias/small-sample noise (consistent with the +0.047..+0.057 run-to-run drift -- the edge IS the noise). DSR is weak here exactly as the researcher proved (it can't tell 1.45 from 1.39).

## Research basis (gate PASSED)
`research_brief.md` (researcher `af86058ca2cd0d154`, 5 sources read in full via pdfplumber). Decisive: DSR is the WRONG primary test for a DIFFERENCE (proven: compute_deflated_sharpe ~1.0 for both); the canonical test is paired Ledoit-Wolf 2008 SR-difference + stationary bootstrap (Politis-Romano). A-priori rule fixed before computing.

## Verification command output (verbatim)
```
$ python -m pytest backend/tests/test_phase_52_3_dsr.py -q
.....                                                                    [100%]
5 passed in 1.89s
```
Verdict reproduced -> live_check_52.3.md (deterministic via the pinned JSON + seeded bootstrap).

## Scope / safety (criterion #3)
- NO live engine change. Diff = analytics.py (+1 function) + the replay dump + the verdict script + the test + the pinned JSON. screener.py / autonomous_loop / the momentum_52wh_tilt_enabled flag are UNTOUCHED (flag still OFF; the tilt stays dormant).
- A-priori rule fixed in the contract before running -> no p-hacking; all stats reported regardless of verdict.

## Artifact shape
- `sharpe_diff_test(...) -> {delta, p_one_sided, ci_low, ci_high, sr_a, sr_b, se, n, n_boot}`
- verdict script -> ENABLE/REJECT against the a-priori rule.

## What this CLOSES + next
Across ALL tested element-2 levers (rotation REJECT, sector-neutral -0.166 REJECT, vol-scaling +0.015 marginal, 52wh +0.057 but p=0.24 NOT robust): **no statistically-robust price-based alpha enhancement found on our universe.** The +20% momentum engine STANDS -- the overfitting-controlled, honest outcome. NEXT: 52.4 residual momentum (bigger-edge, bigger-build, the only remaining cited lever) IF a larger edge is wanted; otherwise accept the engine + let the LIVE multi-market expansion be the money lever. MEASURE Monday's first multi-market cycle (the real test).
