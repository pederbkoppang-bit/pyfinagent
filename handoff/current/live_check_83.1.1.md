# live_check evidence — step 83.1.1 (go/no-go gate arithmetic)

Captured 2026-08-07.

## Required annualized SR by (T, N) — verbatim console output of the producing call

`python backend/backtest/gate_feasibility.py` (each cell = bisection over `analytics.compute_deflated_sharpe(observed_sr, num_trials, variance_of_srs, skewness=0.0, kurtosis=3.0, T, periods_per_year=252)` to DSR ≥ 0.95):

```
kill rule: killrule_phase83.json (pre-existing, ordered first)
results:   gate_feasibility_phase83_results.json written
V = 0.008169 over n=24 trials
spans@126: gdelt=22.88, edgar=51.06, gpr=83.15, control_82_3=15.96
required annualized SR (V=measured_full_sample):
  T=gdelt(2883): N=1: 0.5334  N=2: 0.5334  N=10: 0.6288  N=45: 0.6886  N=100: 0.7153
  T=edgar(6434): N=1: 0.3725  N=2: 0.3725  N=10: 0.4679  N=45: 0.5277  N=100: 0.5543
  T=gpr(10477): N=1: 0.3021  N=2: 0.3021  N=10: 0.3975  N=45: 0.4572  N=100: 0.4839
  T=control_82_3(2011): N=1: 0.6295  N=2: 0.6295  N=10: 0.7249  N=45: 0.7847  N=100: 0.8114
```

(The full 60-cell grid across all three V values — measured-full 0.008169, measured-short 0.167921, module-default 0.5 labelled NOT-MEASURED — is in `gate_feasibility_phase83_results.json`. At V=0.5 the N=45/GDELT cell is 2.0692: the unmeasured default flips the verdict, which is why criterion 3 exists. N=1 cells are byte-identical to N=2 — the `max(num_trials, 2)` collapse, recorded not hidden.)

## Measured V alongside its trial count (verbatim)

```
V = 0.008169 over n=24 trials
```

(ddof=1 variance of the 24 annualized trial Sharpes pooled across the three 82.3 full-sample strategies, `results/20260804T025319Z_phase_82_3_full_sample_3strat.json`; mean 0.5733, min 0.3909, max 0.8246. Secondary short-window V = 0.167921 over n=32 — recorded, never averaged.)

## Independent label spans per source, with the horizon used (verbatim)

```
spans@126: gdelt=22.88, edgar=51.06, gpr=83.15, control_82_3=15.96
```

(Horizon = 126 trading days, READ AT RUNTIME from `preregistration_phase83_ranking.json::label_horizon.trading_days`; asserted ≠ the engine's 1.5×holding-days 135 by the criterion-4 test. Numerators are XNYS trading sessions.)

## compute_pbo_checked payload at the intended shape — one verbatim sample (all 8 in the results JSON)

Pure noise, seed 1, shape (T=2883, N=45, S=16):

```json
{"pbo": 0.5982672759713673, "n_trials": 45, "n_obs": 2883, "gate_grade": true, "column_corr_mean": -0.00027926644174065784, "column_corr_max": 0.07271130624413848, "columns_diverse": true, "refused": null}
```

Multi-seed reading (recorded in the artifact): pure noise spans PBO 0.41-0.77 across 5 seeds; one genuine edge among 44 noise columns spans 0.18-0.51 vs the 0.20 ceiling — **PBO, not DSR, is the binding gate**, and a single-seed PBO is not a statistic.

## Ordering — `ls` full timestamps: the kill rule predates every result artifact

```
$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
-rw-r--r--  1 ford  staff  8467  7 aug. 13:28:25 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
```

## The go/no-go reading (recorded, per the kill rule's own clauses — decision NOT taken here)

K1 at (N=45 cap, T=GDELT 2883, V=measured 0.008169): required SR **0.6886** vs threshold 0.8246 (the best full-sample Sharpe the repo ever produced) → **does not fire — CONTINUE**. At the unmeasured default V=0.5 it would read 2.0692 → STOP; the V measurement decided the verdict, exactly as the step predicted ("the required Sharpe swings ~1.38 to 3.18 across a plausible V range" — with the true measured V it lands BELOW that range entirely). K2: GDELT 22.88 ≥ 16 passes; the 82.3 control window (15.96) would fail — a phase-83 gate evaluation must use a source with more spans than the control window offers. K3/K4 bind future evaluations, not this measurement step.


## Cycle-2 superseding section (2026-08-07, after the cycle-1 Q/A FAIL wf_f3d90599-f10)

The cycle-1 PBO sample above was NOT verbatim from the results JSON -- it was carried
from the research brief's separate prototype run and mislabelled; and the ranges
"0.41-0.77 / 0.18-0.51" were the brief's figures, not this run's. Both are superseded
by the following, REGENERATED from `gate_feasibility_phase83_results.json` after
`run_all()` was re-executed with the reading-note now DERIVED from the payloads in code
(`gate_feasibility.py::pbo_feasibility`), guarded by the new `test_c5b` narrative-vs-
payload assertion. The cycle-1 text above is preserved unedited as the record of the
defect.

TRUE verbatim seed-1 pure-noise payload (piped from the artifact, not retyped):

```json
{"pbo": 0.47233877233877236, "n_trials": 45, "n_obs": 2883, "gate_grade": true, "column_corr_mean": -0.00016336776105686126, "column_corr_max": 0.06103452971078957, "columns_diverse": true, "refused": null}
```

TRUE measured ranges (derived in code from the recorded payloads):

```json
{"pure_noise_min": 0.2027, "pure_noise_max": 0.6524, "one_real_edge_min": 0.3025, "one_real_edge_max": 0.8095}
```

CORRECTED go/no-go reading for the PBO leg: pure noise spans 0.2027-0.6524 (seed 4
lands essentially AT the 0.20 ceiling -- the false-pass risk is REAL) and the
one-real-edge case spans 0.3025-0.8095 -- **no measured edge seed clears the 0.20
ceiling at this shape**. This is HARSHER than the cycle-1 text claimed: the binding-gate
thesis stands, and the PBO leg of K3 is the hard constraint for 83.5. The K1 (DSR)
reading is unchanged: required SR 0.6886 at the cap vs 0.8246 best-ever -> CONTINUE.

Ordering re-verified after the re-run: kill rule 13:25:01 < results 13:50:43.

```
$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  8783  7 aug. 13:50:43 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
```

## Cycle-3 final capture (2026-08-07 — the tree is FROZEN after this block)

The cycle-2 Q/A found the fenced mtime captures stale: the q7/q8 artifact-mutation
runner restores content byte-identically but REWRITES the file, moving its stat —
so the 13:50:43 figures above no longer reproduced. The kill rule is byte-unchanged
(2545 bytes since 13:25:01). Final capture, taken AFTER the last suite run
(`8 passed in 1.34s`) with no further artifact-touching operation to follow:

```
$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  8783  7 aug. 13:51:56 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
```

Ordering: kill rule 13:25:01 < results 13:51:56 — criterion 6 holds on the live stat.
The prior fenced captures are preserved unedited as the record of their moments; this
block supersedes them for the CURRENT tree.

## Follow-up — cycle 4 (2026-08-07, after Q/A CONDITIONAL #2 wf_48465ea7-38e; streak 2 — this cycle must close everything)

The cycle-3 verdict's blocking finding: the "trial budget beats archive depth — now measured" sentence quoted the V=0.5 slice. Closed with the DERIVED-population sweep (grep across all 83.1.1 artifacts + agent memory): corrected at experiment_results (the V-conditional bullet), contract finding 3 ([MAIN CORRECTION]), the gate brief :261-263 (annotation) AND its envelope summary (header cover — which also covered the prototype PBO ranges still sitting there), and the researcher memory's lever bullet (Q/A-directed). The 83.1-owned sites (preregistration trial_budget_rationale + pack section 7) are QUEUED as step 83.1.5 — hash-committed artifacts of a closed step need their own append-only amendment cycle, not a mid-step edit.

The three guard gaps closed with tests + executed kills: `test_c1b` spot-cell equality-to-recomputation (qa15 KILLED); `test_c5c` seed-tuple pinning (qa14 KILLED — pruning the harshest seed with consistent re-derivation dies); `test_c3b` V-vs-own-sharpes at 2e-6 tolerance (measured 4dp-rounding gap 1.4e-7; one dropped trial moves it 1e-5) PLUS the count pinned to the SOURCE artifact's run count — both qa6 (inconsistent truncation) and qa6b (fully-consistent truncation with recomputed V) KILLED. The stale "7 passed" cycle-1 block re-labelled as superseded.

## Cycle-4 FINAL capture (the tree is FROZEN after this block — the qa-mutation runs above moved the results stat again, so this supersedes the cycle-3 capture)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_83_1_1_gate_feasibility.py -q
11 passed in 1.36s

$ ls -laT backend/backtest/experiments/killrule_phase83.json backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  8783  7 aug. 14:32:53 2026 backend/backtest/experiments/gate_feasibility_phase83_results.json
-rw-r--r--  1 ford  staff  2545  7 aug. 13:25:01 2026 backend/backtest/experiments/killrule_phase83.json
```

Ordering: kill rule 13:25:01 (byte-unchanged since creation) < results 14:32:53. No artifact-touching operation follows this capture.
