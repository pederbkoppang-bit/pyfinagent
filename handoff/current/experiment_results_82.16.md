# Experiment Results -- masterplan step 82.16

**Step:** 82.16 (P1) -- two STRATEGY_REGISTRY labels carry zero forward information
**Date:** 2026-08-05 | **Cycle:** 1
**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_16_label_forward_information.py -q`

---

## 0. Research findings, re-measured by Main before adoption

**(a) Confirmed, and no third method is broken.** Main read
`_compute_quality_momentum_label` directly: it calls
`build_feature_vector(ticker, entry_date)` and classifies on `momentum_6m` and
`quality_score` -- both entry-date features, both in `_NUMERIC_FEATURES`. No forward
price. `_compute_factor_label` is the same shape.

**(b) Measured on the committed 82.2 fixture (880 rows), post-entry closes collapsed:**

| Strategy | non-None labels | changed under mutation |
|---|---|---|
| triple_barrier | 880 | 375 |
| meta_label | 880 | 375 |
| **quality_momentum** | 880 | **0** |
| mean_reversion | 880 | 119 |
| **factor_model** | **0** | 0 |
| stretch_regime | 880 | 565 |
| qarp | 528 | 462 |
| reversion_sigma | 409 | 321 |

**(c) THE TRAP.** `factor_model`'s `0 changed` is **evidentially empty** -- it returns
`None` on all 880 rows (it needs `momentum_12m`, which the fixture omits). "Nothing
changed when I destroyed the future" is indistinguishable from "there was nothing to
change". **The verdict survives on source reading; the NUMBER is not evidence.** Hence
the coverage precondition (§2).

**(d) `mean_reversion` moves on only 119/880** -- which is why criterion 1's "at least
one row" is correct and must not be tightened.

**(e) `meta_label` needs no special case:** it aliases `_compute_triple_barrier_label`
and passes on its own (880/375); meta-labelling is a post-label *sizing* stage.

**(f) The tautology is set-theoretically checkable, not rhetorical.** Every input to
both broken labels is itself a training feature: `momentum_6m` + `quality_score` are the
complete input set of `quality_momentum`; `momentum_1m`/`momentum_12m`/
`annualized_volatility`/`pe_ratio`/`pb_ratio`/`quality_score`/`dividend_yield` the
complete set of `factor_model` -- all inside `_NUMERIC_FEATURES`.

**(g) `quality_score` has a SECOND, independent degeneracy.** It is assigned only inside
`if fundamentals:`, and the label coerces `None -> 0.0`. On a fundamentals-free span the
label collapses to "-1 if momentum_6m < -5 else 0", with **+1 unreachable**. Combined
with 82.21 (zero `historical_fundamentals` rows before 2024-06-30, ~81% of the window),
a forward-looking rewrite would still have nothing to train on.

**(h) This is CIRCULAR ANALYSIS (Kriegeskorte et al. 2009), not look-ahead bias** -- and
**a shuffled-label permutation test would PASS it**, because a tautological label
produces a genuine, hugely significant feature-label dependency. Worth stating: that is
the reflex check, and it does not work here.

**(i) Bailey & Lopez de Prado:** DSR's deflation term is built from the number and
variance of trials, so the trial POOL is a direct input. A tautological candidate
corrupts it and *can win the selection*. That is the literature basis for remove-rather-
than-caveat.

---

## 1. The decision: DEMOTE, not rewrite

The criteria permit either. This step demotes, on measured grounds:

1. A forward-looking rewrite is **a new strategy under an old name** -- it would silently
   change what every historical `quality_momentum` row in `quant_results.tsv` means
   (7 `factor_model` + 4 `quality_momentum` rows exist).
2. `quality_momentum` would **remain untrainable** regardless (§0g).
3. The literature requires removal from the pool (§0i).

Demotion is to a named `NON_COMPARABLE_STRATEGIES` mapping carrying the reason in code.
**The methods are KEPT**, so nothing referencing them by name breaks and either is
restorable if given a real forward label.

**Live book unaffected:** `optimizer_best.json` is `triple_barrier` (measured).

---

## 2. What was built

| File | Change |
|---|---|
| `backend/backtest/backtest_engine.py` | `NON_COMPARABLE_STRATEGIES` + the two names out of `STRATEGY_REGISTRY`; `resolve_strategy()` seam; `__init__` records `requested_strategy` / `strategy_was_demoted` |
| `backend/backtest/quant_optimizer.py` | `AVAILABLE_STRATEGIES` **derived** from the registry instead of restated |
| `backend/tests/test_phase_82_16_label_forward_information.py` | NEW -- 32 tests |

**The coverage precondition is the load-bearing addition.**
`test_label_has_coverage_on_the_fixture` asserts each registered strategy produces >= 1
non-None label *before* its mutation result counts. Without it, any strategy that
silently returns `None` passes criterion 1 for free -- exactly how `factor_model`
measured 0/880.

**The optimizer list was ALREADY DRIFTED** -- the hand-written literal was missing all
three 82.2 candidates, so the optimizer could never select them. Deriving it fixes that
too.

---

## 3. A consequence of MY OWN change, found and handled

Demoting a name made the engine's pre-existing coercion **reachable**.
`BacktestEngine.__init__` silently coerced an unregistered strategy to `triple_barrier`.
Harmless while every name was registered; after the demotion a caller asking for
`quality_momentum` would get a run **reported as** `quality_momentum` that actually ran
`triple_barrier` -- worse than the tautological label it replaced, because now the
provenance is wrong too.

And a caller really can ask: `meta_evolution/archetype_library.py` keeps its **own**
`IMPLEMENTED_STRATEGY_IDS` frozenset that still lists both names (verified: it imports
fine, it is independent of `STRATEGY_REGISTRY`; reconciling it is 82.17's file).

Fixed by extracting `resolve_strategy()` -- a module-level seam returning
`(effective_name, was_demoted)` and warning loudly. Verified live:

```
qm   -> ('triple_barrier', True)    + WARNING naming the strategy and the reason
tb   -> ('triple_barrier', False)   no warning
junk -> ('triple_barrier', False)   unchanged legacy behaviour
```

**Autoresearch blast radius, measured not assumed:**
`autoresearch/strategy_registry.py:104` seeds `"strategy": "quality_momentum"`. The
adapter's `_default_param_grid` raises on a name outside `STRATEGY_REGISTRY`, and
`strategy_candidate_producer` catches and **skips** with a warning. Net effect: the
non-comparable strategy is **excluded from the candidate pool** rather than silently
backtested under the wrong name -- precisely §0i. Pinned by a test that drives the real
guard (and also asserts a registered strategy is still accepted, so the guard cannot
pass by rejecting everything). `archetype_library`, `strategy_registry` and the adapter
all still import.

---

## 4. Verification command output (verbatim, unpiped)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_16_label_forward_information.py -q; echo "BARE_EXIT=$?"
................................                                         [100%]
32 passed in 7.54s
BARE_EXIT=0
```

**No skips.** An earlier draft of the autoresearch guard called
`make_engine_backtest_fn` with the wrong signature and SKIPPED -- a guard that does not
run proves nothing. Re-pointed at `_default_param_grid`, the function that actually
carries the check.

---

## 5. Mutation matrix -- CONTROL re-derived in the same run

CONTROL `rc=0 passed=31`; POST-RESTORE `rc=0 passed=31`. **Zero survivors.**

| Mutant | Mutation | Result | Tests ACTUALLY killed |
|---|---|---|---|
| M1 | `quality_momentum` restored to the registry (the defect back) | **KILLED** | `test_label_depends_on_post_entry_prices`, `test_non_forward_strategy_is_out_of_the_registry`, `test_optimizer_cannot_select_a_demoted_strategy` |
| M2 | Optimizer list restated instead of derived | **KILLED** | `test_optimizer_cannot_select_a_demoted_strategy`, `test_optimizer_list_is_derived_from_the_registry_not_restated` |
| M3b | Demotion reason emptied | **KILLED** | `test_non_forward_strategy_is_out_of_the_registry` |
| M5 | Demoted method deleted rather than kept | **KILLED** | `test_demoted_methods_are_kept_so_the_demotion_is_reversible` |
| M6b | `resolve_strategy` stops flagging a demotion | **KILLED** | `test_requesting_a_demoted_strategy_is_loud_not_silent` |
| M7 | `__init__` bypasses `resolve_strategy` | **KILLED** | `test_engine_source_flags_a_demoted_request` |
| M8 | A registered strategy returns `None` everywhere | **KILLED** | `test_label_has_coverage_on_the_fixture`, `test_label_depends_on_post_entry_prices` |

### 5.1 Three mutants survived the first matrix -- what each actually meant

- **M6 (real gap).** I had added the loud-coercion behaviour with **no guard at all**.
  Fixed by the `resolve_strategy` seam + two behavioural tests.
- **M3 (bad mutant, not a survivor).** My mutation replaced the reason string with
  `"" "x: ..."`, which Python **concatenates** into a non-empty string -- so it never
  emptied anything. Re-run correctly as M3b: killed.
- **M4 (redundant assertion).** It removed a duplicate coverage check inside the
  mutation test while the dedicated `test_label_has_coverage_on_the_fixture` still
  stood. Rather than argue equivalence, I added **M8** -- make a registered strategy
  return `None` everywhere -- which kills the dedicated coverage guard directly and
  proves the protection is real.

### 5.2 A tautological test I wrote and replaced

My first demotion-loudness test **set the attributes itself and then asserted them** --
a guard re-implementing what it checks. Replaced with one that calls the production
`resolve_strategy`, plus an AST guard that `__init__` actually calls it (M7 proves that
one fails when the production path bypasses the seam).

---

## 6. Regression

```
$ python -m pytest test_phase_82_2_candidate_strategies.py test_phase_82_3_candidate_backtests.py \
      test_phase_82_13_preload_refusal_handling.py test_phase_82_22_optimizer_best_provenance.py \
      test_phase_82_27_pbo_sweep_producer.py -q
88 passed in 16.03s
```

---

## 7. Scope honesty

**Changed:** registry membership, one derived constant, one resolver seam, one new test
file. **NOT changed:** no label semantics; none of the five forward-looking strategies;
`optimizer_best.json`; historical `quant_results.tsv` rows (their meaning is preserved
precisely BY not redefining the names); `archetype_library`'s `IMPLEMENTED_STRATEGY_IDS`
(82.17's file); any live position, credential or operator-gated flag. Paper trading left
running.

**Not claimed:** that a full backtest was run for a demoted strategy end-to-end; that
`archetype_library`'s list was reconciled (it was not -- it still lists both names, and
that is now a known inconsistency owned by 82.17); that the 82.21 fundamentals interaction
was re-measured this session (it is quoted from 82.21, not re-derived).

---

# CYCLE 2 -- response to the cycle-1 Q/A CONDITIONAL

Verdict verbatim in `evaluator_critique_82.16.md`; raw return at
`qa_returns/82.16_cycle1.output.json`. Three findings, all accepted, all closed and each
mutation-verified.

## 8. Finding 1 -- my criterion-4 test claimed a mechanism it never implemented

`test_guard_detects_a_deliberately_non_forward_stub`'s docstring said it "registers a
non-forward stub in a TEMPORARY COPY of the registry". **It did no such thing.** The body
built a closure, called `_run` twice and asserted `changed == 0` -- it never touched
`STRATEGY_REGISTRY` and never invoked the real guard. The Q/A proved the hole precisely:
**weakening the real guard's threshold to `>= 0` would have left criterion 4 GREEN.** A
negative control that re-computes the check proves nothing about the check.

Rewritten to do what it always claimed: monkeypatch a **copy** of the registry containing
the stub (never the module global -- it is imported by `strategy_backtest_adapter`), then
assert the REAL guard raises via `pytest.raises(AssertionError, match="IDENTICAL labels")`
-- and that the same real guard still ACCEPTS `triple_barrier`, so it cannot pass by
rejecting everything.

**Mutation-verified (C1):** weakening the real guard to `changed >= 0` now kills
`test_guard_detects_a_deliberately_non_forward_stub`.

## 9. Finding 2 -- the demoted names were not the only route to the silent coercion

`AVAILABLE_STRATEGIES` still offers `blend`, which is **not** a `STRATEGY_REGISTRY` key
and has **no implementation** in `backtest_engine.py`. `resolve_strategy` warned only for
DEMOTED names, so `blend` -- and any typo -- still became `triple_barrier` silently and
was then scored under the requested name. My `quant_optimizer.py` comment claimed the
derivation removed "exactly the silent wrongness 82.16 exists to remove", which that
surviving path **contradicted**.

Fixed both: `resolve_strategy` now warns on **any** unregistered name, and the
overclaiming comment is replaced with the two honest caveats. **Mutation-verified (C2).**

`blend`'s fate is a decision, not a cleanup, so it is queued (§11) rather than changed
here.

## 10. Finding 3 -- I analysed only the removal side of a two-sided change

The old hand-written `AVAILABLE_STRATEGIES` had **drifted** and omitted all three 82.2
candidates. Deriving from the registry therefore does two things, and §2 described only
one:

```
old = {triple_barrier, quality_momentum, mean_reversion, factor_model, meta_label, blend}
new = {triple_barrier, mean_reversion, meta_label, qarp, reversion_sigma, stretch_regime, blend}
  removed: quality_momentum, factor_model          (intended, analysed)
  ADDED  : qarp, reversion_sigma, stretch_regime   (NOT analysed)
```

The Q/A's point is exactly right and uses this step's own literature against it: **the
trial pool is a direct input to DSR deflation**, so the Bailey/Lopez de Prado argument for
removing a non-comparable candidate applies **symmetrically to adding three**. I framed
the enlargement purely as a drift fix.

I did not revert it -- the derivation is structurally correct and the drift was itself a
defect -- but the enlargement is now **pinned by a test** so it is visible and
intentional, and the pool-composition decision is queued with a requirement to MEASURE
the DSR/PBO effect rather than assert it is small. **Mutation-verified (C3).**

## 11. Queued

**82.46 (P1) -- optimizer trial-pool composition.** Decide the selectable set
deliberately; measure the DSR/PBO impact of the enlargement before/after on the same
sample; resolve `blend` (implement it or remove it, but do not leave a name that silently
aliases the incumbent). Sequenced **before** 82.24 and 82.26, since it decides the pool
those two measure over.

## 12. Verification + mutation

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_16_label_forward_information.py -q; echo "BARE_EXIT=$?"
..................................                                       [100%]
34 passed in 7.63s
BARE_EXIT=0
```

CONTROL 34 / POST-RESTORE 34.

| Mutant | Targets | Result | Test killed |
|---|---|---|---|
| C1 | real guard weakened to `changed >= 0` | **KILLED** | `test_guard_detects_a_deliberately_non_forward_stub` |
| C2 | `resolve_strategy` stops warning on unregistered non-demoted names | **KILLED** | `test_unregistered_non_demoted_names_are_also_loud` |
| C3 | optimizer list restated instead of derived | **KILLED** | `test_optimizer_list_is_derived_from_the_registry_not_restated`, `test_optimizer_trial_pool_composition_is_pinned` |

Cycle-1's M1-M8 are unchanged and still killed.

## 13. What changed in cycle 2

`backend/backtest/backtest_engine.py` (warn on any unregistered name),
`backend/backtest/quant_optimizer.py` (comment corrected -- no behaviour change),
`backend/tests/test_phase_82_16_label_forward_information.py` (criterion-4 rewritten, two
guards added), `.claude/masterplan.json` (82.46). **No change to registry membership, to
any label, or to which strategies are selectable** -- cycle 2 fixed guards and claims, not
behaviour.
