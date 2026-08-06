---
name: project-dsr-trial-count-reset-82-25
description: phase-82.25 DSR num_trials warm-start reset -- what is measured vs asserted about N, the live optimizer_best.json schema-v1 gap, and the go-live gate headroom trap
metadata:
  type: project
---

`quant_optimizer.py` resets `self.num_trials = 1` at BOTH warm-start paths (`:821`
optimizer_best.json, `:863` result_store), so a carried-forward DSR is deflated as if the
strategy were found on the first attempt.

**Why:** DSR is strictly decreasing in N (`analytics.py:429-432`, Bailey Eq.1). Bailey & LdP
call the undisclosed trial count "the most important piece of information missing from
virtually all backtests"; LdP & Lewis 2018 name the count a **meta-research variable ...
related to the research process itself, rather than the outcome** and list "not track" as a
cause of unassessable discoveries. So N is scoped to the DISCOVERY, not the session.

**How to apply / MEASURED facts that will save a future session:**

- **The live `optimizer_best.json` is schema v1 and has NO `num_trials`.** 82.22 changed the
  WRITER (`:792`) only; the artifact was last written `2026-07-24T11:04:51Z`, before 82.22
  landed (`be04da12`, 2026-08-04), and the optimizer has not run since (historical_macro
  frozen). Keys present: `params, sharpe, dsr, run_id, kept, discarded, saved_at`. Any step
  saying "82.22 added the field to optimizer_best.json" is talking about the writer, not the
  file. The unknown-prior-count branch is the PRODUCTION path, not the edge case.
- **`_load_previous_best` never reads `num_trials` at all** -- write side (`:792`) and read
  side (`:809-834`) are disconnected. Not "reads it but ignores it".
- `:226` (`num_trials = 1` on the COLD baseline) is **legitimate**; only `:821`/`:863` are the
  defect. A fix touching `:226` is over-reach.
- `:256 self.num_trials += 1` sits **before** the `try:` at `:274`, so crashed experiments
  count. That matters: `quant_results.tsv` = 537 data rows of which **300 are crashes**
  (193 discard, 27 BASELINE, 7 evaluated, 6 seed_test, 2 keep, 2 dsr_reject; 36 distinct base
  run_ids).
- **`max(num_trials, 2)` at `analytics.py:430-431`** means N=1 and N=2 give the SAME
  `e_max_sr`. A test comparing N=1 vs N=2 finds no difference. Also TWO different clamps:
  `compute_deflated_sharpe` returns `0.0` for N<1 (`:417`) while `generate_report` passes
  `max(num_trials,1)` (`:768`).
- **Over-counting is the SAFE direction** -- DSR paper App.3: "using M instead of N will
  overstate E[max{SR}]", i.e. lowers DSR. So a plain cumulative counter needs no ONC/effective-N
  clustering. Repo already says the same at `strategy_backtest_adapter.py:45`.
- **Go-live headroom trap:** `paper_go_live_gate.py:42 DSR_THRESHOLD = 0.95`, live artifact
  `dsr = 0.9525811126193078` -- clears by 0.0026. A fix must change only FUTURE computations at
  `:285`; retroactively re-deflating the persisted number closes the gate AND fabricates a
  figure (its own N is unrecorded).
- **Run 60617e0b headline is REPRODUCIBLE, not folklore** (10 result JSONs on disk) and is
  stronger than the step claims: exp01 (N=2, DSR 0.6387492887307706) and exp10 (N=11, DSR
  0.008813951184271042) share the **identical Sharpe 0.6455483635957818** -- 72x on trial count
  alone. Across all ten the fall is NOT strictly monotone (exp04->05, exp07->08 rise) because
  Sharpe also moves; monotonicity holds only at fixed Sharpe.
- **`backend/autoresearch/meta_dsr.py` already states this exact doctrine** ("recompute DSR at
  the cumulative sample size across ALL trials, including abandoned ones") and exposes
  `cumulative_n`, but is DEAD for it: the only production import is `LOOSE_DSR_MIN` at
  `evaluator_agent.py:48`; `meta_dsr()`/`TrialLedger` have zero production callers, and the
  penalty `0.1*sqrt(log(n))` is a self-declared placeholder. Reuse the vocabulary, not the formula.
- **`num_trials` in `strategy_backtest_adapter/strategy_selector/strategy_candidate_producer/
  rotation_runner` is a DIFFERENT variable** (count of seed configs in a bake-off). Never
  conflate.
- Inferring N from the TSV is undermined by `DELETE /api/backtest/optimize/history`
  (`api/backtest.py:838-857`) which deletes TSV + optimizer_best.json + result JSONs together.
- Fixture precedent: `backend/tests/test_phase_82_22_optimizer_best_provenance.py:32-49`
  `_optimizer(**attrs)` bypasses `__init__` via `__new__` -- so it CANNOT exercise
  `_load_previous_best` (called from `__init__:157`). A warm-start test needs a real
  constructor with `qo._BEST_PARAMS_PATH` monkeypatched. `:194-200` already asserts
  `"num_trials" in d` with the message "num_trials is required as input to step 82.25".

**Source-fetch notes:** `en.wikipedia.org/wiki/Deflated_Sharpe_ratio` **does not exist**
(curl returns "Wikipedia does not have an article with this exact name") despite appearing in
search results. `ams.org/notices/201405/rnoti-p458.pdf` = 403;
`davidhbailey.com/dhbpapers/pseudo-math.pdf` = 404; but
`davidhbailey.com/dhbpapers/deflated-sharpe.pdf` WORKS and WebFetch saves the binary locally --
extract verbatim with `pypdf` rather than trusting the fetch summary. The Nomura QMS deck and
the codemacher-hosted LdP&Lewis PDF are both good full-text fallbacks.

Related: [[project_psr_dsr_formulas]], [[project_pbo_single_strategy_cpcv]],
[[project_pbo_level_and_dead_gate_82_27]], [[project_strategy_rotation_seed_set]].
