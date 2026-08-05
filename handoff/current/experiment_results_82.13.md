# Experiment Results -- masterplan step 82.13

**Step:** 82.13 (P1) -- the backtest engine discards `preload_macro`'s refusal
**Date:** 2026-08-05 | **Cycle:** 1
**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_13_preload_refusal_handling.py -q`

---

## 0. Research findings re-measured by Main before adoption

**(a) `preload_macro` has FIVE return paths and `0` is AMBIGUOUS.** Measured directly:

```
loaded       rc=1    outcome=loaded
empty        rc=0    outcome=empty
warm         rc=1    outcome=warm        (positive total, loaded NOTHING this call)
refused_stale        rc=0 outcome=refused_stale
refused_unparseable  rc=0 outcome=refused_unparseable
```

So `if not cache.preload_macro(): abort` is a trap on two counts: it cannot tell a
REFUSAL from an EMPTY TABLE, and the warm path (which fires on every optimizer
iteration under `skip_cache_clear=True`) returns a positive total having loaded
nothing. **This is why the fix is a status accessor, not a check on the int.**

**(b) THE "~40-MINUTE HANG" IS UNMEASURED FOLKLORE.** The brief traced it to
`CLAUDE.md:30` and found no benchmark, test or timing artefact behind it anywhere in
the repo. **This document does not repeat that number as measured.** The defensible
claim is the one that is actually observable in the source: an empty `_macro_full`
sends `cached_macro` into **one 30s-timeout BQ round-trip per distinct cutoff date**,
where zero were intended.

---

## 1. What was built

| File | Change |
|---|---|
| `backend/backtest/cache.py` | `_macro_status` stamped on **all five** return paths; `macro_load_status()`, `macro_was_refused()`, `macro_is_loaded()`, `set_macro_unavailable()`, `macro_is_unavailable()`, `reset_macro_status()`; **fallback suppression** in `cached_macro`; the lying docstring corrected |
| `backend/backtest/backtest_engine.py` | `BacktestResult.data_availability` (defaulted); `_preload_macro_and_record()` (`:278`) acts on a refusal; result labelled from that record (`:375`) |
| `backend/backtest/analytics.py` | report surfaces availability at top level **and** inside `["analytics"]` |
| `backend/tests/test_phase_82_13_preload_refusal_handling.py` | NEW -- 16 tests |

**The `int` return contract is UNCHANGED**, so none of the other 14 `preload_*` call
sites move.

**Design decision -- macro-free mode, not fail-fast.** The brief recommended raising by
default behind a settings flag, arguing the degraded path is "an unbounded sequence of
30s BQ round-trips producing a macro-blind model". That argument is sound against a
degraded mode that *leaves the fallback armed* -- but this implementation **suppresses
the fallback**, so the degraded run is fast rather than unbounded. What remains is a
macro-blind run, and the model-cards / datasheets literature is explicit that the
correct treatment for a missing-feature run is to **label it**, not to refuse it. So:
labelled macro-free mode, loudly warned, and no new settings flag to test in two states.
The fail-fast alternative is recorded here rather than silently dropped.

**Why the fallback suppression is load-bearing, not bookkeeping:** a "degraded mode"
that only sets a flag would still issue one BQ query per cutoff. It would be labelled
about the harm instead of removing it.

---

## 2. Verification command output (verbatim, unpiped)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_13_preload_refusal_handling.py -q; echo "BARE_EXIT=$?"
................                                                         [100%]
16 passed in 4.46s
BARE_EXIT=0
```

---

## 3. Mutation matrix -- CONTROL re-derived in the same run

CONTROL `rc=0 passed=16`; POST-RESTORE `rc=0 passed=16`.

| Mutant | Mutation | Result | Tests ACTUALLY killed |
|---|---|---|---|
| M1 | Engine stops branching on `macro_was_refused` (restores the discard) | **KILLED** | `test_engine_branches_on_the_status_accessor_not_the_return_value`, `test_engine_records_a_refusal_and_enters_macro_free_mode` |
| M2 | Refusal no longer suppresses the fallback | **KILLED** | `test_engine_records_a_refusal_and_enters_macro_free_mode` |
| M3 | `cached_macro` ignores macro-free mode | **KILLED** | `test_refusal_puts_the_cache_into_macro_free_mode_so_the_fallback_cannot_fire` |
| M4 | Stale refusal mis-stamped `empty` (ambiguity restored) | **KILLED** | `test_engine_records_a_refusal_and_enters_macro_free_mode`, `test_zero_return_is_ambiguous_between_empty_and_refused` |
| M5 | Result no longer derives availability | **KILLED** | `test_run_backtest_labels_the_result_from_that_record` |
| M6 | `report["analytics"]` loses `macro_available` | **KILLED** | `test_report_surfaces_availability_at_top_level_and_inside_analytics` |
| M7 | Success path also flips macro-free on (always-degrade) | **KILLED** | `test_engine_leaves_a_successful_run_untouched`, `test_engine_does_not_treat_an_empty_table_as_a_refusal` |
| M8 | Docstring reverts to a row-count claim | **KILLED** | `test_docstring_no_longer_claims_the_return_is_a_row_count` |
| M9 | `available = bool(macro_rows)` instead of `macro_is_loaded()` | **EQUIVALENT** (§3.2) | -- |

### 3.1 FOUR mutants survived the first version of this suite -- one root cause

M2, M3, M5 and M7 all survived initially. Every one of my guards stopped at the **cache
boundary**: they exercised the primitives (`set_macro_unavailable`, `preload_macro`) in
isolation and never drove the **engine**, which is what the criteria are actually about
("causes the backtest ENGINE to surface...", "the RUN RESULT records..."). A green suite
said nothing about the wiring.

Fixed by extracting `BacktestEngine._preload_macro_and_record()` -- the engine's **real**
code path, not a test-only copy -- and adding four engine-level guards that drive it.

**M3 survived for a second, sharper reason:** my guard raised `AssertionError` from a
fake `query()`, but `cached_macro` wraps the call in `except Exception`, which **swallows
it**. The guard could not observe its own defect. Replaced with a **counting** client and
`assert client.n == 0`, which is visible through the except.

### 3.2 M9 is an EQUIVALENT mutant -- measured, not argued

Required a behavioural differential before calling it a gap. Across all three reachable
non-refusal outcomes:

```
loaded  rc=1  bool(rc)=True   macro_is_loaded()=True   AGREE=True
empty   rc=0  bool(rc)=False  macro_is_loaded()=False  AGREE=True
warm    rc=1  bool(rc)=True   macro_is_loaded()=True   AGREE=True
```

No differential exists today. `macro_is_loaded()` is still the better expression -- it
states the invariant directly and will not drift if a future path ever returns a positive
count without populating `_macro_full` -- but reporting M9 as a surviving gap would be a
false finding.

---

## 4. Criterion 3 -- the enumeration is an AST walk, not a grep

`_preload_call_sites()` parses each source file and classifies a call **discarded iff its
direct parent is an `ast.Expr` statement**. Grep cannot substitute: this repo contains
`preload_*` mentions in prose comments, in `settings.py`, and a *local function of the
same name* inside a test file. The enumeration is asserted **non-empty** (an empty scan
and a clean codebase produce identical output), and the walk is scoped to
`backend/scripts/tests/dev` -- `rglob` over the repo root also crawls `.venv` and made the
test take minutes.

---

## 5. Regression

```
$ python -m pytest test_phase_82_0_macro_ingestion.py test_phase_82_12_string_column_guards.py \
      test_phase_82_15_macro_point_in_time.py test_phase_82_3_candidate_backtests.py -q
77 passed in 5.21s
```

**One in-flight breakage, caused by this step and fixed:** growing `cache.py` moved the
lines cited in 82.12's classification table, and
`test_classified_line_numbers_still_point_at_a_row_read` went red. That guard doing
exactly its job. Anchors re-derived (`530 -> 612`, `590 -> 672`) rather than the guard
loosened.

---

## 6. Discovered defects -- QUEUED, not absorbed

| Step | What |
|---|---|
| **82.43** (P1) | `historical_data.py`'s bare `if macro:` silently drops **all six** macro features when `cached_macro` returns `{}`. 82.13 fixed the layer above; this one has *other* causes (early cutoff, vintage miss, BQ timeout) that a refusal-shaped flag would mislabel. Requires measuring whether a shorter vector reaches the model. |
| **82.44** (P3) | A `cache.py` comment cites `backtest_engine.py:308` for a call site that was at `:317` and has since moved again. The docstring half of this defect was fixed here; the cross-file citation is queued as a class sweep. |

---

## 7. Scope honesty

**Changed:** macro status plumbing, one engine method, one result field, one report key,
one new test file, two new pending steps. **NOT changed:** macro SLAs, the `int` return
contract, `preload_prices`/`preload_fundamentals` (82.40 owns their missing gates), any
live position, any credential, any operator-gated flag. Paper trading left running. The
other 7 discarding `preload_*` call sites are **classified**, not fixed -- criterion 3
asks for classification.

**Not claimed:** that a full `run_backtest` was executed end-to-end (the guards drive
`_preload_macro_and_record`, the real seam, plus an AST assertion that the result is
constructed from its record); that the per-cutoff fallback's wall-clock cost was measured
(see §0b -- the folklore number is not repeated).
