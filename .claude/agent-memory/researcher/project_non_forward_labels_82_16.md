---
name: non-forward-labels-82-16
description: Step 82.16 measured facts -- factor_model's "0/880 changed" is VACUOUS (all-None), the 82.2 test already IS the 82.16 test over a hand-written list, and removal without patching AVAILABLE_STRATEGIES fabricates TSV rows
metadata:
  type: project
---

Measured 2026-08-05 for masterplan step 82.16 (non-forward-looking labels in
`STRATEGY_REGISTRY`). Anchors move -- re-derive before citing.

**The registry has 8 keys / 7 methods, not 5** (phase-82.2 added `stretch_regime`,
`qarp`, `reversion_sigma`). Every line number in the masterplan 82.16 text is
stale by ~66 (`:1124`->`:1190`, `:1208`->`:1274`, `:742`->`:808`, `:1145`->`:1211`).

**MEASURED across all 8 names on the committed 82.2 fixture (880 rows), mutation
= collapse every post-entry bar to 0.5x entry:**

| strategy | non-None | changed |
|---|---|---|
| triple_barrier / meta_label | 880 | 375 |
| quality_momentum | 880 | **0** |
| mean_reversion | 880 | 119 |
| factor_model | **0** | 0 |
| stretch_regime | 880 | 565 |
| qarp | 528 | 462 |
| reversion_sigma | 409 | 321 |

**Why:** the fixture's `build_feature_vector` omits `momentum_12m`, so
`_compute_factor_label` early-returns None on every row -- its archived
"0/880 changed" is **arithmetically true but evidentially empty** (`None != None`
is False). The source verdict still holds (no `cached_prices` call anywhere in
its body), but any new guard MUST assert per-strategy `n_labelled > 0` or it
inherits the same blind spot. Same class as
[[feedback_a_green_suite_can_be_blind]] -- assert the guard's own preconditions.

**Other durable traps:**
- `mean_reversion` moves on only 119/880 because `if not is_oversold and not
  is_overbought: return 0` precedes the forward fetch -- any "N% of rows must
  change" threshold FAILS a correct method.
- The 82.2 test `test_candidate_label_depends_on_post_entry_prices` is ALREADY
  this test, over a hand-written 3-name list. Generalise, don't duplicate.
- `_engine()` in the 82.2 test sets no `tp_pct`/`sl_pct` -> AttributeError on
  triple_barrier AND meta_label. Un-patched you see
  `AssertionError: Cache not initialized` FIRST, which masks it.
- `@pytest.mark.parametrize` over an empty registry collects ZERO tests and exits
  0 -- that is exactly why 82.16 criterion 3 demands a non-empty assertion.
- `quality_score` None is coerced to 0.0 in `_compute_quality_momentum_label`, so
  with no fundamentals `+1` is unreachable and `-1` fires on momentum alone -- a
  SECOND degeneracy stacked on the non-forward one.
- REMOVAL BLAST RADIUS: `quant_optimizer.py` `AVAILABLE_STRATEGIES` is a
  hand-written literal; removing a name from the registry without patching it
  makes the optimizer log `"strategy": "quality_momentum"` for a run that
  silently executed triple_barrier (ctor clamp). `archetype_library` archetypes
  declare `is_implemented=True` for both offenders -- inert today, but step 82.17
  will derive that frozenset and they will raise at IMPORT.
  `optimizer_best.json` is `triple_barrier`, so the live book is UNAFFECTED;
  `quant_results.tsv` carries 7 factor_model + 4 quality_momentum rows.

**Literature naming (the useful half):** this is **circular analysis / double
dipping** (Kriegeskorte 2009, PMC2841687), NOT look-ahead bias -- it is the
mirror image. Kapoor & Narayanan's [L2] "illegitimate features" and Apicella
et al.'s "direct target leakage" describe the same identity traversed the other
way. **A shuffled-label permutation test would PASS a tautological label** --
if a reviewer asks for one, say so. Bailey & Lopez de Prado's DSR takes the trial
POOL (`N`, `Var[{SR_n}]`) as input, which is the formal reason a non-comparable
candidate must be REMOVED rather than caveated.

Full brief: `handoff/current/research_brief_82.16.md` (archived to
`handoff/archive/phase-82.16/` on step close).
