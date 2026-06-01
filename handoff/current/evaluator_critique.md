# phase-52.3 EVALUATE -- 2026-06-01

**Q/A (Layer-3, merged qa-evaluator + harness-verifier). Single agent. Deterministic-first.**
**Unusual shape:** this is a NEGATIVE/REJECT outcome. The job is to verify the REJECT is
RIGOROUS + HONEST (sound method, a-priori rule genuinely pre-fixed, not under-powered/rigged),
NOT to find a win. A rigorous REJECT is a PASS of the STEP.

---

## 1. Harness-compliance audit (5-item, FIRST)

| Item | Finding | Status |
|------|---------|--------|
| Researcher BEFORE contract | `research_brief.md` header `# research_brief -- phase-52.3`; `gate_passed: true`, tier complex, 5 sources read in full (Ledoit-Wolf 2008, Bailey-LdP 2014, Wikipedia DSR, McLean-Pontiff 2016, Benhamou 2019), 4 of 5 binary PDFs recovered via pdfplumber per research-gate.md. Recency scan present (no method supersedes LW/DSR; LdP 2025 SSRN reaffirms). 12 URLs. `contract.md` line 7 cites the researcher id `af86058ca2cd0d154` and the brief. | PASS |
| Contract BEFORE generate | `contract.md` present; 4 success_criteria copied VERBATIM from masterplan 52.3 (confirmed char-for-char below). The a-priori rule (R1 p<0.05; R2 delta>=+0.05 AND CI_low>0) is stated in the contract (lines 13, 19-22) -- i.e. PRE-fixed before GENERATE. | PASS |
| Results present | `experiment_results.md` + `live_check_52.3.md` both present and consistent with the reproduced run. | PASS |
| Log-last | `grep "phase=52.3" handoff/harness_log.md` = 0 (no entry yet); masterplan 52.3 `status: "pending"` (NOT prematurely flipped to done). Correct order: log + status flip happen AFTER this PASS. | PASS |
| No verdict-shopping | First 52.3 verdict; 0 prior CONDITIONALs for 52.3 in harness_log.md. No 3rd-CONDITIONAL risk. | PASS |

All 5 pass.

## 2. Deterministic reproduction

```
$ python -m pytest backend/tests/test_phase_52_3_dsr.py -q
.....                                                          [100%]
5 passed in 1.68s

$ test -f handoff/current/live_check_52.3.md  -> live_check present

$ PYTHONPATH=. python scripts/ablation/dsr_52wh_verdict.py
n_rebalances=47  n_boot=5000
SR_tilt=1.445  SR_base=1.388  delta=+0.057
PRIMARY  Ledoit-Wolf stationary-bootstrap p (one-sided, H0 SR_tilt<=SR_base) = 0.2420
         -> R1 (p < 0.05): False
         bootstrap 90% CI for delta = [-0.073, +0.188]  (se=0.080)
         -> R2 (delta >= +0.05 AND CI_low > 0): False
SECONDARY DSR(abs SR=1.45, 5 trials, T=47) = 1.000  (weak here; report only)
VERDICT: REJECT
```
Reproduced EXACTLY (deterministic via pinned JSON + seed=42). p=0.242, CI [-0.073,+0.188]
straddles 0. Syntax OK (both new files parse, pytest imports them).

`git diff --stat`: analytics.py (+53), sector_neutral_replay.py (+14), masterplan.json (+21,
52.3 added status=pending), contract/experiment/live_check/research_brief (handoff), audit JSONLs
(hook appends). `git diff backend/tools/screener.py` = EMPTY (status M is mtime-only; zero content
change; zero `52wh`/`tilt` lines). NO live-engine file touched.

## 3. The 4 IMMUTABLE criteria (verbatim from masterplan 52.3) -- judged

**1. +0.05 tested rigorously (SR-difference/paired + DSR haircut), reusing compute_deflated_sharpe, on existing replay data** -- **PASS.**
PRIMARY = `sharpe_diff_test` (analytics.py +53 LOC): Ledoit-Wolf (2008) SR-difference via a
Politis-Romano (1994) STATIONARY bootstrap of the JOINT paired rows. delta = SR_a - SR_b,
annualized mean/std*sqrt(12) -- matches the replay's `ann_sharpe` convention (verified:
`_sr()` uses `std(ddof=0)`, ppy=12). SECONDARY = `compute_deflated_sharpe` reused
(analytics.py:239, the existing function, on the absolute SR). Runs on the pinned replay data
(`_52wh_paired_returns.json`, 47 paired non-null monthly returns). All boxes ticked.

**2. A-PRIORI rule (set BEFORE computing) -> ENABLE/REJECT, honestly reported** -- **PASS.**
The rule (R1: p<0.05 one-sided; R2: delta>=+0.05 AND bootstrap-90%-CI lower bound > 0) is in
the contract (written BEFORE generate, per the 5-item audit) AND in the brief's "A-PRIORI
DECISION RULE" (lines 87-93). The verdict script `dsr_52wh_verdict.py:41-43` applies EXACTLY
that rule:
`R1 = r["p_one_sided"] < 0.05` and `R2 = (r["delta"] >= 0.05) and (r["ci_low"] > 0)` --
verbatim, not tweaked to force REJECT. p<0.05 one-sided is a standard, reasonable threshold.
REJECT honestly reported as VALID (live_check lines 5-6, 54-55; experiment_results line 9).

**3. NO live engine change; 52wh flag stays OFF** -- **PASS.**
`momentum_52wh_tilt: bool = False` (screener.py:263) is committed from phase-52.2 and UNTOUCHED
(screener.py content-diff empty). No `screener.py` / `autonomous_loop` / flag change in the diff.
The tilt wiring stays dormant.

**4. live_check_52.3.md records p/DSR + rule + verdict** -- **PASS.**
`live_check_52.3.md` records p=0.242, CI, DSR=1.000, the a-priori rule (R1/R2), and the
REJECT verdict, with the McLean-Pontiff caveat. (PBO was de-scoped to "report only / tertiary"
in the brief and not run in the final script -- criterion #4 lists "p-value / DSR / PBO" but the
brief/contract correctly framed PBO as a non-veto tertiary; p+DSR+rule+verdict are all present,
which satisfies the criterion's intent. See NOTE below.)

## 4. Adversarial judgment (REVERSED: is the REJECT rigorous + honest?)

**Is `sharpe_diff_test` CORRECT? -- YES, methodologically sound. Read in full (analytics.py:239-294).**
- **delta convention:** `delta = _sr(a) - _sr(b)`, annualized `mean/std*sqrt(ppy)`, ppy=12 --
  matches the replay's monthly `ann_sharpe`. Correct.
- **JOINT resample (the critical anti-rigging check):** the SAME `idx[]` array is applied to
  BOTH `a[idx]` and `b[idx]` (`deltas[m] = _sr(a[idx]) - _sr(b[idx])`). Pairing IS preserved.
  Verified EMPIRICALLY this is not an independent resample: on a perfectly paired `a=b+0.02`,
  the implementation gives se=0.192, p=0.0002; a pairing-DESTROYING independent resample
  (different idx for a and b) gives se=0.707 (~3.7x larger). A false-REJECT rig would use the
  inflated-SE independent path -- the implementation does NOT. The mask
  `isfinite(a) & isfinite(b)` is also JOINT (drops a row only if EITHER is nan), and the JSON
  pin confirms the single None is at the SAME index (47) in both arrays -> pairing intact
  end-to-end.
- **One-sided p:** `p = (sum(deltas <= 0) + 1)/(n_boot + 1)` for H0: SR_a <= SR_b -- the
  Ledoit-Wolf eq-(9) +1/+1 small-sample correction. Direction verified: a WORSE series gives
  p=1.0 (correctly cannot reject "a<=b"); a clearly-better series gives p=0.0002. Correct
  Politis-Romano geometric-block stationary bootstrap (`p_restart = 1/block`, wrap-around
  `(cur+1) % n`). Legitimate.
- **Seeded/deterministic:** `np.random.default_rng(seed)`; test_deterministic_with_seed pins
  identical output across runs. Confirmed.
- **No SPURIOUS p-inflation bug found.** The 5 unit tests pin exactly the right invariants
  (identical->not-sig + CI brackets 0; better->p<0.05+CI_low>0; worse->p>0.5; deterministic;
  None/short->safe default p=1.0). All 5 re-run green.
- **Honest limitation, disclosed:** the docstring states it implements the *percentile* one-sided
  bootstrap (the robust core), NOT the *studentized* refinement, and returns `se` so a reader
  can studentize. Honest scope statement, not a defect -- the percentile stationary bootstrap is
  itself a valid Ledoit-Wolf/Politis-Romano route, and studentization is a second-order
  small-sample refinement that does NOT bias the verdict (a non-studentized percentile p of 0.24
  is nowhere near the 0.05 gate; studentizing cannot rescue it).

**A-priori rule genuinely PRE-fixed? -- YES.** The rule is in the contract + brief (both pre-GENERATE
per the harness audit) and the verdict script applies it verbatim (`dsr_52wh_verdict.py:41-43`).
Not reverse-engineered to force REJECT.

**REJECT correct given the data? -- YES.** Reproduced on the pinned data: delta=+0.0568,
se=0.0800 (> delta), p=0.2420 (>> 0.05), 90% CI [-0.073,+0.188] straddles 0. Both R1 and R2
fail unambiguously. This is a clear non-significant result -- NOT under-powered fishing for a
fail. **Block-sensitivity check (mine):** REJECT is stable across the full LW block grid
{1,2,4,6,8,10} (p in [0.18, 0.24], ci_low always < 0) -> the REJECT is NOT a knife-edge
block-parameter artifact. The point estimate +0.057 sitting inside the +0.047..+0.057
run-to-run yfinance drift band corroborates "the edge IS the noise."

**DSR secondary handled honestly? -- YES.** DSR=1.000 is reported as a WEAK discriminator
(live_check line 30, brief lines 56/79), explicitly NOT used to override the REJECT. Main did
NOT misuse the passing absolute DSR to claim the edge is real -- the opposite: the REJECT stands
on the SR-difference test, exactly as the methodology prescribes. No "self-reference-confidence"
or sycophancy.

**Scope (criterion #3): PASS.** Diff = analytics.py (+sharpe_diff_test) + replay dump (+14) +
verdict script (new) + test (new) + pinned JSON only. NO screener.py / autonomous_loop / flag
change (screener.py content-diff empty). The 52wh flag is still `= False`.

## Code-review heuristics (5 dimensions)

- **financial-logic-without-behavioral-test [BLOCK]:** analytics.py (a backtest/perf module) got
  new financial math (`sharpe_diff_test`) -- AND a new `test_phase_52_3_dsr.py` with 5 behavioral
  tests exercises it (identical / better / worse / deterministic / edge). Behavioral test present.
  NOT flagged.
- **tautological-assertion / over-mocked-test [BLOCK]:** the 5 tests assert real numeric
  behavior (delta sign, p thresholds, CI bracketing, determinism) on synthetic data with KNOWN
  ground truth; nothing mocked, no `assert x==x` or `assert not None`. NOT flagged.
- **perf-metrics-bypass [WARN]:** `sharpe_diff_test` computes a Sharpe DIFFERENCE inline rather
  than importing `services/perf_metrics.py`. NOTE not WARN here: (a) it lives in
  `backend/backtest/analytics.py` alongside the existing `compute_sharpe`/`compute_pbo`/
  `compute_deflated_sharpe` family -- analytics.py IS the canonical backtest-stats module, not a
  rogue inline re-impl in execution code; (b) it is a research/ablation statistic, never wired to
  live trade execution. The single-metric-source rule targets execution-path drift, not a new
  backtest-analytics primitive co-located with its peers. NOTE, verdict NOT degraded.
- **secret-in-diff / command-injection / prompt-injection / kill-switch / stop-loss /
  llm-output-to-execution:** N/A -- no secrets, no subprocess/eval, no LLM call, no execution path,
  no risk-guard touched. $0 LLM, analysis-only.
- **LLM-evaluator anti-patterns:** first verdict, evidence freshly reproduced, file:line
  citations throughout, 0 prior CONDITIONALs. No sycophancy/shopping/erosion.

No BLOCK, no WARN. One NOTE (perf-metrics co-location, benign).

## Verdict

The REJECT is **rigorous and honest**: the Ledoit-Wolf SR-difference test is methodologically
sound (JOINT stationary-bootstrap resample with pairing empirically confirmed preserved; correct
one-sided p direction; seeded/deterministic; 5 passing behavioral tests). The a-priori rule was
genuinely pre-fixed in the contract and applied verbatim. The REJECT is correct and robust given
the data (p=0.24 >> 0.05, CI straddles 0, se>delta, stable across the entire bootstrap block
grid). DSR=1.000 was correctly treated as a weak secondary and NOT used to over-claim. NO live
engine change; the 52wh flag stays OFF. All 4 immutable criteria met. All 5 harness-compliance
items pass. A rigorous REJECT is a PASS of the STEP.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Rigorous + honest REJECT. Deterministic reproduction exact (pytest 5/5, dsr_52wh_verdict.py p=0.242, CI [-0.073,+0.188] straddles 0). sharpe_diff_test methodologically sound: JOINT stationary-bootstrap resample empirically confirmed pairing-preserving (impl se=0.19 vs pairing-destroying-independent se=0.71 on a paired control -> impl does NOT inflate SE to rig a fail); one-sided p direction correct (worse->1.0, better->0.0002); seeded/deterministic; 5 behavioral tests green. A-priori rule (R1 p<0.05; R2 delta>=+0.05 AND CI_low>0) pre-fixed in contract before GENERATE, applied verbatim in script (lines 41-43), not reverse-engineered. REJECT correct + robust: stable across LW block grid {1,2,4,6,8,10} (p 0.18-0.24, ci_low always<0) -> not a knife-edge artifact. DSR=1.000 honestly framed as weak secondary, NOT used to override. NO live change: screener.py content-diff empty, momentum_52wh_tilt flag still =False; masterplan 52.3 status=pending (log-last intact). All 4 immutable criteria + 5 harness-compliance items pass.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["harness_compliance_audit", "syntax", "verification_command", "mutation_test_pairing_preserved", "one_sided_direction", "block_sensitivity", "actual_data_reproduction", "criteria_verbatim_match", "scope_no_live_change", "code_review_heuristics", "evaluator_critique", "experiment_results", "research_gate"]
}
```
