# experiment_results -- steps 82.22 + 82.23

**GENERATE complete for both.** Contracts: `handoff/current/contract_82.22.md`,
`handoff/current/contract_82.23.md`. Research:
`handoff/current/research_brief_82.22_82.23.md` (gate_passed=true, 8 sources
read in full, 33 URLs, 14 internal files) -- ONE gate feeding both contracts,
since they are one coupled surface.

## Verification command output (verbatim) -- REGENERATED 2026-08-04

The previous block recorded `10 passed` / `CONTROL -> 28 passed`. That was
captured BEFORE the criterion-4 deflation tests and the criterion-1 DSR tests
existed, and was never re-run -- a stale "verbatim" capture, which the 82.22
Q/A caught. Worse, the criterion-4 tests shipped briefly RED (a `TypeError` on
a kwarg I guessed rather than read) and the evaluator observed that failure
mid-run. Both regenerated below from the current tree.

```
$ python -m pytest backend/tests/test_phase_82_22_optimizer_best_provenance.py -q
15 passed

$ python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py -q
21 passed

$ ... both together
36 passed

$ python scripts/qa/check_optimizer_best_provenance.py   # MUST be red on the live file
  optimizer_best.json claims sharpe=1.1704633657934074 for run '60617e0b', but none of that run's 10 saved artifacts p
   exit=1
```

## 82.22 -- what was wrong, and the fix

`_save_best_params` (`backend/backtest/quant_optimizer.py:720`) wrote
`"run_id": self._run_id` -- the CURRENT run -- beside `self.best_sharpe` /
`self.best_dsr`, which `_load_previous_best` may have INHERITED from an earlier
run without recording the source. When a run beat nothing (`kept == 0`), the
prior run's numbers were re-stamped with the new run's identity.

**Additive schema only** (all 15 consumers are `dict.get`-based; renaming would
break named readers): `metrics_run_id`, `metrics_source_artifact`,
`warm_started_from`, `num_trials`, `schema_version`. `run_id` keeps its
original meaning -- the run that WROTE the file. `metrics_run_id` says which run
PRODUCED the metrics.

**Absence must not read as freshness.** A file with no `metrics_run_id` is
reported as UNDECLARED provenance, never as self-attributed to `run_id`.

**The checker is red on the live file, as the criterion demands**, and it locates
the true origin by search rather than by being told: `52eb3ffe-exp10`.

### A correction the research forced

1.1705 is **not "correct but mis-labelled" -- it is STALE.** Six of run
60617e0b's ten trials returned the identical Sharpe **0.6455483636** (measured
by `Counter` over the artifacts): the incumbent params re-measured under current
code and data. The honest present-day figure is ~0.646, not 1.17.

## 82.23 -- the gate was never fail-open

**CORRECTION carried from the contract.** An earlier framing of mine said
"PBO <= 0.5 was never computed, so the gate ran on one term", implying silent
promotion. Measured, that is wrong in the failure DIRECTION:

| gate | ceiling | on missing PBO | live? |
|---|---|---|---|
| `backend/autoresearch/gate.py:22` | **0.20** | **fail-CLOSED** | YES |
| `backend/services/promotion_gate.py:37` | 0.5 | fail-OPEN (defaults 0.0) | NO -- zero callers |
| `backend/backtest/analytics.py:198` | 0.5 | doc only | n/a |

A missing PBO never promoted anything; it BLOCKED promotion.

**The real defect is one line:** `compute_pbo` returns **0.0 -- the best possible
value -- when N < 2 or T < S*2**, and 0.0 passes every ceiling above. An
undersized matrix does not fail to inform; it MANUFACTURES A PASS.

**Shipped:**
- `compute_pbo_checked` -- returns a dict, never a bare float, so a consumer
  cannot receive a PBO without the N it came from. Refuses (`pbo: None`) instead
  of returning the false-good 0.0.
- `PBO_CEILING_LIVE = 0.20`, `PBO_CEILING_CANONICAL = 0.50`,
  `PBO_MIN_TRIALS_GATE_GRADE = 10` -- the three ceilings reconciled in one place
  with a note on which is enforced.
- `PromotionGate` refuses a PBO whose reported trial count is below the floor.
  **Additive**: a producer that never emitted `pbo_n_trials` behaves exactly as
  before.
- The adapter now emits `pbo_n_trials` / `pbo_n_obs` / `pbo_gate_grade`.

### Criterion 4 -- I declared it unmet, then met it

I told the Q/A that criterion 4 ("a trial-diversity number accompanies every
reported pbo") was NOT met, because I emitted `pbo_n_trials` but no column
correlation. **It is now met**: `compute_pbo_checked` returns
`column_corr_mean` / `column_corr_max` / `columns_diverse` alongside every
value, and the adapter forwards `pbo_column_corr_mean` /
`pbo_columns_diverse` so the number reaches the gate rather than merely
existing in the helper.

The diagnostic DISCRIMINATES rather than decorating: independent columns
measure 0.003 and near-duplicates 0.9999, and a test asserts both. This matters
because CSCV ranks the N columns against each other, so correlated columns make
PBO noise-driven **however large N is** -- exactly what phase-82.3 measured on
its short window (0.967-0.979).

**Disclosure:** the 82.23 Q/A was spawned BEFORE this was added, so its evidence
block says criterion 4 is unmet. That statement was true when written and is now
stale. I am not re-spawning to hide it -- the next cycle will grade the current
tree, and the sequence is recorded here.

**PBO deliberately NOT added to `generate_report`**, and a test pins that: it
receives ONE `BacktestResult`, so N=1 -> a hard-coded 0.0 -> an unconditional
PASS on every run. A test exists so a future "why has the report no PBO?" is not
answered by adding it.

**And the optimizer's own trials cannot be the matrix.** Bailey Algorithm 2.3:
for a guided search the columns must be "the final outcome of each guided search
... and not the intermediate steps". `QuantOptimizer.run_loop` IS a guided
search, so stacking its ten nav_histories is wrong in an unquantified direction.

## CRITERION 4 WAS UNMET WHEN I SPAWNED THE Q/A -- I flagged it, then fixed it

82.22's criterion 4 requires asserting **the deflation math is UNCHANGED** -- a
fixture showing DSR falls monotonically as `num_trials` rises. My first suite
did not contain that test: I had persisted `num_trials` into the schema and
mistaken that for pinning the statistic. I said so explicitly in the Q/A spawn
prompt rather than hoping it passed unnoticed, then added the tests.

**Two vacuity traps hit while adding them, both self-caught:**

1. **I guessed the function signature** (`n_obs`, `skew`, `sr_variance`) instead
   of reading it -- the real one is `variance_of_srs`, `skewness`, `T`,
   `periods_per_year`. Immediate `TypeError`, so this one failed loudly.
2. **The first version passed on a floating-point crumb.** At
   `observed_sr=0.65` the DSR SATURATES: `[1.0, 1.0, 1.0, 0.9999999999997232]`.
   `dsrs[0] > dsrs[-1]` was True by **3e-13** -- a monotonicity assert that
   would have survived deflation being switched off entirely. Moved to
   `observed_sr=0.15`, which sits on the responsive part of the curve, and the
   assert now demands a **material** gradient (`> 0.5`, straddling 0.5) rather
   than any positive difference.

A second test measures the same gradient on the REAL run-60617e0b artifacts, so
a fixture that drifts from production cannot hide a change.

**And my first mutation of it was incomplete**, which nearly produced a false
"the guard is weak" conclusion: the expected-max-Sharpe term uses `num_trials`
TWICE, and I neutralised only the first. With both neutralised the guard dies
correctly. A surviving mutant is only evidence when the mutation is complete.

## Mutation matrix (in-tree, restored, 0 MUTANT markers)

```
82.22  M1 revert to unconditional self-attribution   -> 2 failed
82.22  M2 always disclaim (cheap way to pass)        -> 2 failed
82.23  M1 wrapper returns the false-good 0.0         -> 2 failed
82.23  M2 gate ignores the trial count               -> 2 failed
82.23  M3 gate refuses ALWAYS (cheap way to pass)    -> 2 failed
82.22  M4 num_trials neutralised in BOTH DSR terms   -> 1 failed
CONTROL                                              -> 30 passed
```

Both "revert the fix" AND "pass by always refusing" are caught, so neither
direction is a free pass.

## Scope honesty

- **No live-funnel change**: `backend/services`, `backend/tools`,
  `backend/agents` untouched. Changed: `quant_optimizer.py`, `analytics.py`,
  `gate.py`, `strategy_backtest_adapter.py` + two test files + one checker.
- **The live file is NOT rewritten.** The fix corrects the WRITER; regenerating
  `optimizer_best.json` requires an optimizer run, which is out of scope and
  gated on the historical_macro state. The checker stays red until then, which
  is the honest state.
- Two spin-off defects queued rather than folded in: **82.25** (`num_trials`
  reset to 1 on every warm start, under-deflating a carried-forward DSR) and
  **82.26** (`_DEFAULT_K = 8` vs the paper's `N >> 10`; phase-82.3's PBO figures
  are N=8 and now disclosed as not gate-grade in the design pack).
- Two pre-existing test failures in `test_price_tolerance_gate.py` and
  `test_phase_70_4_gate_observability.py` are NOT mine: neither file imports any
  module I changed, and both were in the 32 the 82.3 Q/A classified as
  pre-existing.
- Three `F401`s in `quant_optimizer.py` are pre-existing (verified against
  `git show HEAD:`); my own unused imports were removed.


## CYCLE 2 (82.22) -- disposition of the cycle-1 CONDITIONAL

Criteria 2, 3 and 4 confirmed MET with mutation-verified behavioural guards.
Three findings, all closed.

**B1 [BLOCK] -- MY CHECKER VERIFIED HALF THE CRITERION.** Criterion 1 says the
recorded "**sharpe/dsr**" must be reproducible. My checker compared
**sharpe only** -- it read `deflated_sharpe` into `observed` and then discarded
it. The Q/A proved it with two probes: a file with a matching sharpe and a
FABRICATED dsr (0.99 against the artifact's 0.05) returned `verified`, and so
did a file with **no `dsr` key at all**.

That is the worse half to have skipped. DSR is the money-path statistic: it is
what the promotion gate spends (`DSR >= 0.95`) and the bar every rotation
challenger must clear. A provenance check that validates the headline number
while leaving the gated one unverified defeats its own purpose.

Fixed: both statistics must now reproduce from the SAME artifact, with a
distinct `dsr_mis_attributed` status for the insidious shape where sharpe
matches and dsr does not. Both probes now fail; reverting to sharpe-only kills
2 tests.

**And the mirror guard could not have caught it.** My
`test_checker_passes_when_metrics_reproduce_from_the_named_run` used
`dsr == deflated_sharpe == 0.5` -- identical values, so a sharpe-only
implementation passed it just as happily as a both-check. **A fixture whose
two fields are aliased cannot distinguish a check that reads one from a check
that reads both.** De-aliased to 0.6455/0.3771.

**B2 [BLOCK] -- the "verbatim" capture was stale, again.** It recorded
`10 passed` / `CONTROL -> 28 passed`, captured before the criterion-4 and
criterion-1 tests existed. Regenerated from the current tree: **15 / 21 / 36**.

Worse than stale: **the criterion-4 tests shipped briefly RED.** I added them
mid-evaluation, and the first version raised `TypeError` on a kwarg I had
GUESSED (`n_obs`, `skew`, `sr_variance`) instead of reading the real signature
(`variance_of_srs`, `skewness`, `T`). The evaluator ran the immutable command
during that window and saw `1 failed, 11 passed`. Editing a suite while it is
being graded is my error; the honest record is that the evaluator's first
observation was a red suite.

**B3 [WARN] -- sequencing hazard, and I am acting on it.** Flipping 82.22 fires
`git add -A`, which would sweep un-verdicted 82.23 production code
(`analytics.py`, `gate.py`, the adapter) into a commit under 82.22's name --
the audit-the-commit-not-the-diff class. **I will not flip 82.22 until 82.23
has its own verdict**, so both close under their own names.
