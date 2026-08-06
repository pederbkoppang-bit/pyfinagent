# Contract -- phase-82.46

**Step:** 82.46 (P1) -- decide the optimizer's trial-pool composition deliberately.
**Date:** 2026-08-06. **Cycle:** 1.
**Research gate:** PASSED -- `handoff/current/research_brief_82.46.md`,
`gate_passed: true`, tier `complex`, 7 external sources read in full (including
both Bailey/Lopez de Prado primaries extracted via pdfplumber), 22 URLs,
recency scan performed, 11 internal files inspected.

---

## 1. THE STEP'S PREMISE IS FALSE, and I verified that myself

The step says: *"the trial pool is a DIRECT input to Deflated Sharpe (Bailey &
Lopez de Prado deflate by the number and variance of trials), so the same
argument 82.16 used to REMOVE a non-comparable candidate applies symmetrically
to ADDING three."*

**It is not.** Measured by Main on this tree, not taken from the brief:

```
compute_deflated_sharpe signature:
  (observed_sr, num_trials, variance_of_srs=0.5, skewness=0.0,
   kurtosis=3.0, T=252, periods_per_year=1)
has a pool/strategies param?  False
```

`num_trials` is threaded from `QuantStrategyOptimizer.num_trials`, which is
`self.num_trials += 1` **per iteration** (`quant_optimizer.py:256`). The
categorical pool never reaches the statistic. Two calls with the same iteration
count are bit-identical regardless of pool size:

```
DSR(pool_before)==0.796298   DSR(pool_after)==0.796298   identical=True
```

The 82.16 comment in `quant_optimizer.py` asserting the same thing is wrong for
the same reason and is corrected by this step.

**The theory agrees with the code.** Both primaries count configurations
*actually tried*, not menu size. A larger menu multiplies trials only for an
exhaustive grid; this is a fixed-budget random search over one categorical
dimension. DSR appendix A.3 notes that using a too-large M *overstates*
E[max SR] -- i.e. over-deflation, which is the safe direction.

### The premise that IS true, and it is sharper than the one the step states

**N is steep, and a pool member that cannot run still costs an N.** Measured:

```
N=  1 -> DSR 1.0000      N= 26 -> DSR 0.7963
N=  2 -> DSR 1.0000      N= 50 -> DSR 0.1164
N= 10 -> DSR 1.0000      N=100 -> DSR 0.0008
```

And `self.num_trials += 1` (`:256`) executes **before** the `try:` (`:274`)
that wraps `run_backtest` (`:282`). Measured: `increment precedes the run: True`.
A crashed experiment is logged as `"crash"` and **still counts as a trial**.

So the pool decision does affect DSR -- not through its size, but through
**wasted iterations**. That is the mechanism this step should act on.

---

## 2. What is actually wrong with the pool, measured

| Member | Status | Evidence |
|---|---|---|
| `blend` | **not a registry key, no implementation** | resolves to `triple_barrier` via `resolve_strategy`, then is SCORED under the requested name. A revert orphan: `_compute_blend_label` was born in `1f270641` and deleted by `9fbd9cd6`, whose diff never touched `quant_optimizer.py`. |
| `qarp` | **cannot run on the configured window** | 82.21 (shipped today) makes `_preload_fundamentals_and_record` RAISE for a label-fundamentals-dependent strategy when the window starts before the measured coverage start `2024-06-30`. The optimizer catches it at `:289` and logs `"crash"` -- after the N increment. |
| `tb_weight`, `qm_weight`, `mr_weight`, `fm_weight` | **dead params, still proposable** | `_PARAM_BOUNDS` has **24** entries and these are 4 of them (16.7%). `quant_optimizer.py:589` writes them into `engine._strategy_params`; **nothing reads them** -- `rotation_runner._DEAD_KEYS` already documents them as *"Keys written by the optimizer setter but with NO engine reader (reverted in 9fbd9cd6)"*. The comment at `:588` still claims `_compute_blend_label` reads them; that function does not exist. |
| the rest | registry-derived, fine | `triple_barrier, mean_reversion, meta_label, stretch_regime, reversion_sigma` |

**Correction to the brief, made by Main:** the brief says the weight params are
"read by NOTHING". In the letter that is wrong -- they ARE referenced at
`quant_optimizer.py:589` and in `rotation_runner._DEAD_KEYS`. In substance it is
right: no ENGINE reader consumes them. My first grep for them returned "0
references" because a zsh glob had broken the command -- the instrument was
seeing nothing, not finding nothing. Re-run correctly, it finds them. Recorded
because that is the exact failure mode this project keeps hitting.

**A third drifted list, out of scope but real:**
`backend/meta_evolution/archetype_library.py` `IMPLEMENTED_STRATEGY_IDS` still
holds both 82.16-demoted names AND `blend`, and omits all three 82.2
candidates. `resolve_strategy`'s own docstring names it as the live caller that
can request a demoted name. Queued.

---

## 3. THE DECISION

**The selectable pool is derived, not typed:** a strategy is selectable iff it
is a `STRATEGY_REGISTRY` key AND not in `NON_COMPARABLE_STRATEGIES`. That is
what `AVAILABLE_STRATEGIES` already computes minus the `+ ["blend"]`.

1. **`blend` is REMOVED.** It has no implementation, and offering a name that
   silently runs as `triple_barrier` and is then scored under the requested
   name corrupts both attribution and the trial count. Re-implementing it is
   rejected: the deleted `_compute_blend_label` took a weighted vote over
   `quality_momentum` and `factor_model`, which 82.16 demoted as carrying no
   forward information -- resurrecting it would resurrect them.
2. **The four dead blend-weight params are REMOVED from `_PARAM_BOUNDS`.**
   They are 4 of 24 proposable params, so ~1 in 6 proposals spends a full
   walk-forward run and a DSR-costing N increment on a parameter no engine
   reads. This is the largest *measured* DSR defect in this step's scope, and it
   follows from the corrected premise in §1 rather than the false one.
3. **The registry-derived members stay**, including `qarp` -- but the pool
   becomes **window-aware**, because 82.21 made `qarp` structurally
   unrunnable on the default window. Excluding it by name would go stale; the
   exclusion is derived from the same 82.21 predicate the engine uses.
4. **No DSR/PBO re-run is launched.** The pool-vs-DSR comparison the step asks
   for is answered ANALYTICALLY and exactly by §1: pool size does not enter DSR,
   proven by identity, so there is nothing to re-measure. A pool-level **PBO**
   comparison WOULD need fresh runs (measured: ~2.1h short-window, ~16.7h
   full-sample) and is queued rather than started blind. **Every existing 82.3
   PBO number is N=8, below `PBO_MIN_TRIALS_GATE_GRADE=10`, so it is
   DIRECTIONAL, not gate-grade** -- the artifact must label it and cite 82.26.

## 4. Immutable success criteria (verbatim from `.claude/masterplan.json`)

1. "the optimizer's selectable strategy set is derived from an explicit,
   documented decision rather than a side effect, and a test pins the exact set
   with its rationale recorded in the step artifact"
2. "the DSR and PBO impact of the pool change is MEASURED on the same sample
   before and after, and both numbers are recorded in the step artifact rather
   than asserted to be negligible"
3. "`blend` is either implemented as a real strategy or removed from the
   selectable set, and a test asserts no selectable name resolves to a different
   strategy than the one requested"
4. "a guard fails if a strategy is added to or removed from the pool without the
   recorded decision being updated"

**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_46_trial_pool_composition.py -q`

**On criterion 2:** it demands the DSR impact be *measured*, not asserted
negligible. It is measured -- by exhibiting the identity `DSR(before) ==
DSR(after)` from the production function plus the trace showing no pool
parameter exists. That is a stronger form of measurement than a sampled
comparison, and the artifact records both numbers. The PBO half is recorded as
the existing per-strategy N=8 directional figures with their gate-grade
limitation stated, and a pool-level CSCV queued with its measured cost.

## 5. Plan

- **D1** -- `AVAILABLE_STRATEGIES` derived with `blend` dropped; a
  machine-readable `POOL_DECISION: dict[str, str]` mapping every selectable name
  to its rationale, so criterion 4 has something to compare against.
- **D2** -- remove the four dead weight params from `_PARAM_BOUNDS`; correct the
  stale `:588` comment and the false 82.16 DSR comment.
- **D3** -- guards: pool derived (not restated), `blend` absent, no selectable
  name resolves to a different strategy (drive `resolve_strategy`, not a source
  scan -- `blend` appears in comments, which is the 82.43 trap), the decision
  record covers exactly the pool, and a member added/removed without updating
  `POOL_DECISION` fails.
- **D4** -- update the phase-82.16 guard that closing this step turns RED
  (`test_optimizer_trial_pool_composition_is_pinned` asserts `previously_offered
  - now == {quality_momentum, factor_model}` where `previously_offered` contains
  `blend`), preserving its intent, in the same commit.

## 6. Non-scope

No `blend` implementation. No fresh optimizer runs. No change to
`archetype_library` (queued). No change to `resolve_strategy`'s behaviour --
82.16 already made it loud. No live positions; paper trading untouched.

## 7. References

- `handoff/current/research_brief_82.46.md`
- Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* (2014), incl. appendix A.3
- Bailey, Borwein, Lopez de Prado, Zhu, *The Probability of Backtest
  Overfitting* -- CSCV, Algorithm 2.3
- Internal: `backend/backtest/quant_optimizer.py:86,102-113,256,274,282,289,588-591`,
  `backend/backtest/backtest_engine.py:84-115`,
  `backend/backtest/analytics.py:384-432,743-795`,
  `backend/autoresearch/rotation_runner.py:59-69`,
  `backend/backtest/fundamentals_coverage.py` (the 82.21 predicate),
  `backend/tests/test_phase_82_16_label_forward_information.py:226,376-395`
