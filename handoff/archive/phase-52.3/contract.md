# Contract -- phase-52.3: DSR/SR-difference robustness gate for the 52wh edge (is +0.05 real?)

**Step id:** 52.3 | **Priority:** P1 | **depends_on:** 52.2 | **Date:** 2026-06-01
**harness_required:** true | **$0 LLM** | no pip | **NO live change** | a-priori rule -> ENABLE or REJECT

## Research-gate summary (PASSED)
`handoff/current/research_brief.md` (researcher `af86058ca2cd0d154`: gate_passed=true, tier complex, 5 sources read IN FULL -- 4 binary PDFs recovered via pdfplumber per research-gate.md). Decisive:
- **DSR is the WRONG primary test** -- it deflates an ABSOLUTE max-of-N Sharpe; the +0.05 is a DIFFERENCE (baseline 1.39 vs tilt 1.44). Proven empirically in-session: `compute_deflated_sharpe` returns ~1.0 for BOTH 1.44 and 1.39 at T=47 -> can't adjudicate the delta.
- **PRIMARY = paired Ledoit-Wolf (2008) SR-difference test with a STATIONARY-BOOTSTRAP p-value** (Politis-Romano 1994). delta = SR_tilt - SR_base (annualized sqrt(12)); studentized statistic; resample the JOINT (base_ret_i, tilt_ret_i) rows (geometric block ~T^(1/3)~4mo); one-sided p = (#{d*_m >= d}+1)/(M+1), M>=1000. Robust to fat tails + autocorrelation (a naive paired t-test understates SE -> sanity-only, NOT the gate). Ledoit-Wolf does NOT exist in the repo (grep=0) -> ADD ~40-60 LOC (numpy+scipy, both deps). $0.
- **SECONDARY: DSR** on the tilt's absolute Sharpe, deflated for N_eff trials (reuse `compute_deflated_sharpe` analytics.py:239) -- WEAK discriminator here, report only.
- **TERTIARY: PBO** (reuse `compute_pbo` analytics.py:184, S=6 at T=47) -- report only.
- **Reproducibility:** PIN the yfinance prices/paired arrays (live yfinance drifts +0.047..+0.057) -> dump the paired monthly arrays to a JSON once + run the deterministic (seeded) LW test on it so the Q/A reproduces the verdict exactly.
- **A-PRIORI RULE (state BEFORE computing -> no p-hacking): ENABLE iff ALL:** R1 (primary) one-sided LW stationary-bootstrap p < 0.05 (M>=1000); R2 (magnitude) delta >= +0.05 ann AND bootstrap 90% CI lower bound for delta > 0. REJECT if R1 fails OR R2 CI-lower <= 0. (R3 DSR>=0.95 / R4 PBO corroborating.) McLean-Pontiff: a small in-sample edge decays ~26-58% live -> require a comfortable, not knife-edge, pass.

## Hypothesis
The 52.1 +0.05 Sharpe improvement, tested rigorously (paired Ledoit-Wolf SR-difference, stationary-bootstrap p-value) on the ~47 paired monthly returns + deflated for the configs tried, is MOST LIKELY within selection-bias/small-sample noise (p >= 0.05 and/or the bootstrap CI lower bound <= 0) -> the a-priori rule REJECTS -> the 52wh tilt stays OFF (dormant wiring from 52.2), and element-2 "promote the highest earner" is NOT satisfied by this edge. (If it surprisingly clears the rule, ENABLE is the verdict -- still a separate operator-gated flag flip.)

## Success criteria (IMMUTABLE -- verbatim from masterplan step 52.3)
1. the +0.05 52wh-tilt Sharpe IMPROVEMENT over baseline is tested rigorously (a canonical SR-difference / paired test on the per-rebalance return differences + a multiple-testing/DSR haircut for the configs tried), reusing the codebase's compute_deflated_sharpe where applicable, on the existing replay data
2. an A-PRIORI decision rule (set BEFORE computing) determines ENABLE vs REJECT; the result is honestly reported (a 'not statistically robust -> do NOT enable' outcome is VALID and expected if the edge is within noise)
3. NO live engine change; the 52wh flag stays OFF unless/until the rule says ENABLE (and even then enabling is a separate operator-gated action)
4. live_check_52.3.md records the test statistics (p-value / DSR / PBO) + the a-priori rule + the ENABLE/REJECT verdict

**Verification command:** `pytest backend/tests/test_phase_52_3_dsr.py` + `test -f live_check_52.3.md`.
**live_check:** REQUIRED -- the LW SR-difference stats (p, delta, CI) + DSR/PBO + the a-priori rule + ENABLE/REJECT; NO flag flip.

## Plan steps (GENERATE)
1. **analytics.py:** add `sharpe_diff_test(ret_a, ret_b, periods_per_year=12, n_boot=2000, block=4, seed=42) -> {delta, p_one_sided, ci90_low, ci90_high, sr_a, sr_b}` -- the Ledoit-Wolf SR-difference with a stationary (Politis-Romano) bootstrap of the JOINT paired rows; deterministic (seeded).
2. **sector_neutral_replay.py:** dump the per-config monthly arrays (baseline, hi52_k0.5) + the 5 config Sharpes to `handoff/current/_52wh_paired_returns.json` at the end of a run (the reproducibility pin).
3. **Run the replay once** -> the JSON (the pinned paired arrays).
4. **scripts/ablation/dsr_52wh_verdict.py:** load the JSON; run `sharpe_diff_test(hi52_k0.5, baseline)` (PRIMARY R1/R2); `compute_deflated_sharpe(1.44, N_eff, var-of-5-config-Sharpes, skew/kurt, T)` (SECONDARY); optionally `compute_pbo` (TERTIARY); apply the A-PRIORI RULE -> print ENABLE/REJECT + all stats.
5. **test** `backend/tests/test_phase_52_3_dsr.py`: `sharpe_diff_test` on synthetic -- identical series -> p ~ high (>0.05, delta~0); a clearly-better series -> p < 0.05, delta > 0, CI-low > 0 (deterministic via seed). (Pins the test's correctness, not the empirical verdict.)
6. **Verify:** pytest; run dsr_52wh_verdict.py on the pinned JSON -> capture {delta, p, CI, DSR, PBO, verdict} into `live_check_52.3.md` with the a-priori rule + the McLean-Pontiff caveat. Honest verdict (likely REJECT).
7. **EVALUATE:** fresh qa. Then harness_log.md (LAST), then flip masterplan 52.3 -> done.

## Safety / scope notes
- **NO live change.** Diff = analytics.py (+sharpe_diff_test) + the replay dump + the verdict script + the test + the pinned JSON. screener.py / autonomous_loop / the 52wh flag are UNTOUCHED (flag stays OFF).
- **A-priori rule fixed BEFORE computing** (above) -> GENERATE is pure compute-and-compare, no p-hacking. Report ALL stats regardless of verdict.
- **Honest REJECT is the expected + valid outcome** (criterion #2): a +0.05/~+0.05%/mo edge over 47 obs is likely within noise; REJECT keeps the (already-OFF) tilt dormant + points to 52.4 residual momentum (bigger edge) or accepting the engine as-is.
- Reproducibility: the pinned JSON + seeded bootstrap make the verdict deterministic for the Q/A.
- $0 LLM; no pip; no spend; no flag flip.

## References
- handoff/current/research_brief.md (52.3 gate) + .claude/agent-memory/researcher/ (52.3 memory)
- Ledoit-Wolf 2008 (econ.uzh.ch/.../iewwp320.pdf -- SR-difference, eq 6/9); Politis-Romano 1994 (stationary bootstrap); Bailey-LdP 2014 (DSR); McLean-Pontiff 2016 (decay); Benhamou et al. 2019 (corroborating)
- backend/backtest/analytics.py:239 (compute_deflated_sharpe), :184 (compute_pbo), :125 (compute_sharpe); scripts/ablation/sector_neutral_replay.py:177 (the monthly[config] arrays to dump)
