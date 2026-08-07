# Contract — Step 83.1.1: THE GO/NO-GO — measure the gate arithmetic before building anything

- **Step id:** 83.1.1 (P0, phase-83; depends_on: 83.1 — done, commit 2e1fb9ca)
- **Tier (named field):** T3 — executor Main (Opus 5, effort max); Q/A via qa-verdict Workflow (opus/max).
- **Date:** 2026-08-07, autonomous drain, cycle 172

## Research-gate summary

`handoff/current/research_brief_83.1.1.md` — gate_passed: **true** (7 external sources read in full / 27 URLs / recency scan / 14 internal files; the inversion and V measurement PROTOTYPED with 16,080 real `compute_deflated_sharpe` invocations, printed into the brief). Decisive findings:

1. **V measured from actually-run trial sets** (82.3 artifacts): pooled n=24 full-sample (2018-2025) → **V = 0.008169** (ddof=1; mean SR 0.5733, max 0.8246); pooled n=32 short-window → **V = 0.167921**. The 20× spread is estimation noise on the short sample — and it is exactly why the two 2026-08-04 feasibility verdicts contradicted each other: both arithmetically right, different V. Record BOTH, never average; caveat recorded that these are K=8 configs of existing families (a prior, not the phase-83 V).
2. **PBO, not DSR, is the binding gate**: at the intended (T=2883, N=45) shape, pure independent noise gives PBO 0.41-0.77 across 5 seeds; ONE genuine edge among 44 noise columns gives 0.18/0.38/0.51 vs the 0.20 ceiling. A single-seed PBO is not a statistic. `columns_diverse` is weak (a 0.918-correlated matrix passed it).
3. **The deflation floor √V·E[max_N] contains no T** (analytics.py:429-432); measured: +263% history buys -11% on required SR, cutting N 45→10 buys -23%. [MAIN CORRECTION 2026-08-07, cycle 4: those two percentages are the V=0.5 slice — the researcher's brief computed them at V=0.5 with the qualifier, which this summary dropped. At the MEASURED V=0.008169 the pair is −33.6% / −8.7% and the lever ordering REVERSES; see the corrected experiment_results bullet.] `max(num_trials, 2)` at :430 makes **N=1 byte-identical to N=2** — recorded as the collapse it is.
4. **Bailey/LdP Eq.(1) primary text confirms V is a VARIANCE** (pypdf extraction; a WebFetch summary of the same PDF FABRICATED the opposite claim — all PDF claims in the brief come from source text).
5. **Spans at the pre-registered 126 TRADING days**: GDELT 22.88, EDGAR 51.06, GPR 83.15, 82.3-control 15.96 (calendar-day numerators overstate GDELT +45%; XNYS calendar needs `start="1980-01-01"` or it raises DateOutOfBounds).
6. **Criterion-2 trap**: the new module must use a FUNCTION-SCOPED `from backend.backtest import analytics` attribute lookup, or the spy is invisible and the criterion passes vacuously (the 83.0.3 lesson, same seam).
7. **Discovered defect to queue**: the repo's LIVE V (analytics.py:752-754) is a cross-WINDOW dispersion inside one run with a silent 0.5 fallback below 2 windows — NOT the Bailey cross-TRIAL dispersion. Out of scope here.

## Immutable success criteria (verbatim from `.claude/masterplan.json` 83.1.1)

1. "the annualized Sharpe required to reach DSR >= 0.95 is computed with backend/backtest/analytics.py::compute_deflated_sharpe -- not a re-derived formula -- with periods_per_year matched to the return frequency, across trial counts N in at least {1, 10, 45, 100} and across at least two sample lengths T corresponding to measured free-source coverage windows; every resulting figure is RECORDED in the step artifact and none is asserted against a threshold"
2. "a test asserts the computation calls compute_deflated_sharpe from that module rather than reimplementing the deflation, and fails if the module's function is not invoked"
3. "the trial-Sharpe dispersion V is MEASURED from the realized dispersion of Sharpes across an actually-run trial set, and a test asserts the recorded V is accompanied by the integer trial count it was measured over and that this count is at least 2 -- a V recorded without its sample size does not satisfy this criterion"
4. "the count of INDEPENDENT label spans is computed as the sample span in days divided by the theme label horizon and recorded per candidate source, and a test asserts the horizon used equals the value this step pre-registers rather than the engine's holding-days-derived 1.5x horizon, failing if the two are silently equated"
5. "PBO feasibility is evaluated with backend/backtest/analytics.py::compute_pbo_checked on the intended matrix shape and its refused, gate_grade and columns_diverse fields are recorded verbatim in the step artifact"
6. "a kill rule naming the recorded quantity and the comparison that stops the phase is written to its own file, and a test asserts that file's mtime precedes the mtime of every result artifact this step produces"
7. "the ordering guard is mutation-tested: touching the kill-rule file so its mtime follows the results makes the criterion-6 test FAIL"

**Verification command (immutable):** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_1_gate_feasibility.py -q`

**live_check (immutable):** verbatim console table of required SR by (T, N) with the producing call; measured V with its trial count; span counts per source with the horizon used; verbatim compute_pbo_checked payload; `ls -la` full timestamps showing the kill rule predates every result artifact → `handoff/current/live_check_83.1.1.md`.

## Explicit decisions

- **D1 — KILL RULE WRITTEN FIRST**, before any result artifact: `backend/backtest/experiments/killrule_phase83.json` (deliberately NOT under `results/` so it never joins the 83.1 artifact population). Four clauses, each (recorded_quantity, comparison, threshold, threshold_source): **K1** required_annualized_sr at (N=45 cap, T=GDELT, V=measured-full-sample) > 0.8246 (the best full-sample Sharpe the repo ever produced — 82.3 stretch_regime) → STOP; **K2** independent_spans@126 < 16 (= the CSCV S=16 subset count) → STOP for that source; **K3** PBO refused OR gate_grade false OR pbo > PromotionGate.max_pbo (read at runtime) on the gate evaluation → STOP; **K4** actual trial count > 45 (the pre-registered cap) → STOP. Write-once: any later edit pushes its mtime past the results and turns criterion 6 red — that is the design, not a hazard.
- **D2 — new importable module `backend/backtest/gate_feasibility.py`** holding `required_annualized_sr()` (bisection over the REPO function, bracket [1e-9, 50]; monotonicity was verified in research) and the measurement drivers — with a FUNCTION-SCOPED `analytics` attribute lookup so the criterion-2 spy is real (finding 6).
- **D3 — results recorded as a real JSON artifact** `backend/backtest/experiments/gate_feasibility_phase83_results.json` (not markdown fences — the multi-fence regex trap found by this research), written AFTER the kill rule; figures RECORDED, none asserted against targets (criterion 1's own wording).
- **D4 — the (N, T) grid**: N = {1, 2, 10, 45, 100} (N=2 added to expose the N=1 collapse explicitly), T = {2883 GDELT, 6434 EDGAR, 10477 GPR, 2011 82.3-control(labelled NOT-free)}, V = {0.008169 measured-full, 0.167921 measured-short, 0.5 module-default(labelled DEFAULT-NOT-MEASURED)}. periods_per_year=252, T in daily observations.
- **D5 — horizon read from the pre-registration at runtime** (`label_horizon.trading_days` = 126) and asserted ≠ int(engine holding_days default 90 × 1.5) = 135; spans use TRADING-session numerators (calendar-day numerators overstate +45%).
- **D6 — PBO feasibility at the intended shape (T=2883, N=45, S=16)** with multi-seed noise baselines and the one-real-edge case recorded verbatim; the single-seed caveat and the columns_diverse weakness recorded beside the payloads.
- **D7 — discovered defect queued** as its own step (83.1.4): analytics.py:752-754 live V is cross-window-not-cross-trial with a silent 0.5 fallback.
- **D8 — criterion-7 mutation restores the ORIGINAL kill-rule mtime in finally** (os.utime forward, expect AssertionError, restore saved st_mtime_ns — this file cannot be unlinked like 83.1's throwaway mutant).

## Plan

1. Write `killrule_phase83.json` (D1) — FIRST.
2. Write `gate_feasibility.py` (D2): `required_annualized_sr()`, `measure_v_from_82_3()`, `span_counts()`, `pbo_feasibility()`, `run_all()` writing the results JSON (D3).
3. Run it; capture the console table + payloads for the live_check.
4. Write the 7-criteria test file (reusing the strict-mtime idiom + population-non-empty assert; the C2 routing spy + an inline-reimplementation mutant; C4's prereg-vs-135 assertion).
5. Mutation matrix: (m1) reimplement the deflation inline in required_annualized_sr → C2 spy fails; (m2) V recorded without its count → C3 fails; (m3) horizon silently set to 135 → C4 fails; (m4) touch kill rule forward → C6 fails (this IS criterion 7, in-suite); (m5) drop a required-SR grid cell → C1 cell-count fails.
6. `experiment_results_83.1.1.md` + live_check → qa-verdict → transcribe → harness_log → flip. Re-derive every fenced measurement after the final edit.

## References

`research_brief_83.1.1.md` (Bailey & Lopez de Prado DSR primary text via pypdf; Harvey-Liu; arXiv:2507.07107 recency scan; in-repo analytics.py audit with line anchors; the 16,080-invocation inversion prototype).
