# Experiment Results -- phase-82.21 (cycle 1)

**Step:** 82.21 (P1). **Date:** 2026-08-06.
**Contract:** `handoff/current/contract_82.21.md`.
**Research brief:** `handoff/current/research_brief_82.21.md` (`gate_passed: true`,
tier `complex`, 8 sources read in full, 41 URLs).

---

## 1. The decision (criterion 4)

**DECISION: build SEC EDGAR XBRL.** The operator's verbatim instruction and the
full derivation are in `contract_82.21.md` §2. The short form:

Branch A as worded -- *"accept that fundamentals-dependent strategies are
evaluable from 2024-07 on"* -- has a **falsified premise**, measured, not
argued:

- `data_ingestion.py:278` writes `"filing_date": report_date` with the comment
  *"true filing date not available from yfinance"*, and `cache.py:612/:631`
  filter `report_date <= cutoff`. Measured mean publication lag 66 days. So the
  *covered* window leaks look-ahead too; it is not point-in-time.
- MinBTL (Bailey/Borwein/Lopez de Prado/Zhu 2014, Thm 2) inverts at 1.67 years
  to ~2.3 independent configurations, against gates of DSR >= 0.95 / PBO <= 0.5.

So Branch A is not a shorter evaluation window -- it is **no evaluation**.
Branch B is free, is the only path to real history *and* a true `filed`
vintage, and was measured feasible today. The EDGAR ingester is **not built
here** (no criterion requires it; the verification command is one pytest
module) -- it is queued as its own research-gated step.

**Owed back to the operator:** Branch A as worded is not available. If you would
rather not fund the EDGAR work, the honest form of that choice is
*"fundamentals-dependent strategies are retired"*, not *"evaluable from
2024-07"*. Either way this step's code is identical.

---

## 2. What was built

### D1 (criterion 1) -- coverage pinned where production can see it

`backend/backtest/fundamentals_coverage.py` (NEW) + a checked-in snapshot
`backend/backtest/_fundamentals_coverage.json`, **measured live against
BigQuery during GENERATE**:

```json
{
  "min_report_date": "2024-06-30",
  "max_report_date": "2026-02-28",
  "n_rows": 4798,
  "n_tickers": 503,
  "date_format": "%Y-%m-%d",
  "report_date_bq_type": "STRING",
  "rows_with_non_iso_report_date": 0,
  "measured_at": "2026-08-06"
}
```

The step's own figures reproduce exactly (4798 / 503 / `2024-06-30` /
`COUNTIF(report_date < '2024-06-30') = 0`).

`FUNDAMENTALS_COVERAGE_START` is a constant in production code so the guard
tests **the code's belief**, not just its own literal; `snapshot_drift()` fails
when the two disagree. `report_date` is a BQ **STRING**, so `MIN()` is
lexicographic -- valid only because every value is zero-padded ISO. The guard
asserts that with a regex **and** with the measured
`rows_with_non_iso_report_date == 0`. No `isinstance(v, date)` guard exists
anywhere in this step; on a STRING column it could never fire.

### D2 (criterion 2) -- explicit unavailability at the feature builder

`historical_data.py`: `features["fundamentals_available"]` is now set on
**every** return path -- seeded `False` at construction (so the short-price
early return carries it too) and set to `bool(fundamentals)` at the
fundamentals block. Without the seed, `fv.get("fundamentals_available")` would
have three outcomes (True/False/**absent**) and "absent" reintroduces exactly
the ambiguity the criterion forbids.

The discriminating predicate now exists:
`pe_ratio is None AND available is True` = genuine null (a loss-making company;
`pe_ratio` is only assigned when `net_income > 0`) versus
`pe_ratio is None AND available is False` = structural. Those two were
byte-identical before.

The flag is deliberately **not** in `_NUMERIC_FEATURES`: it is a perfect proxy
for `date >= 2024-07`, so as a model input it is a regime dummy and the
classifier would learn the coverage boundary instead of the economics. A guard
asserts it stays out.

### D3 (criterion 3) -- refuse for the dependent set, record for everyone else

`_preload_fundamentals_and_record()` mirrors 82.13's
`_preload_macro_and_record` (same file, same shape, same reason: the refusal
path must be drivable against the engine's real code, not a copy). Called once
from `run_backtest`, asserted structurally by AST.

- **REFUSE** (raise) when the resolved strategy is label-fundamentals-dependent
  AND the window starts before coverage.
- **RECORD** otherwise -- `data_availability` gains `fundamentals`, defaulted
  so no existing construction site changes, and `analytics.py` double-writes it
  into `report["analytics"]` because two consumers read only that.

**The refusal set is DERIVED, never hardcoded.** The rule, written down before
it was applied:

> Let `F` = feature keys assigned ONLY inside the `if fundamentals:` block of
> `build_feature_vector`. A strategy is label-fundamentals-dependent iff its
> registry label function reads a key in `F`, transitively.

Derived output, reproducing the brief independently:

```
F size: 17
label-dependent (derived): ['qarp']
independent: ['mean_reversion', 'meta_label', 'reversion_sigma', 'stretch_regime', 'triple_barrier']
```

A hardcoded `{"qarp"}` would go stale the next time a strategy is added, and a
stale gate is worse than no gate because it reads as coverage.

---

## 3. Verbatim verification command output

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_21_fundamentals_coverage.py -q
.....................                                                    [100%]
21 passed in 1.49s
```

Regression sweeps:

```
$ python -m pytest backend/tests/ -q -k "82_0 or 82_12 or 82_13 or 82_16 or 82_2 or 82_21 or 82_22 or 82_25 or 82_27 or 82_3 or 82_5"
254 passed, 2451 deselected, 1 warning in 36.58s

$ python -m pytest backend/tests/ -q -k "backtest or historical or analytic or optimizer or strategy or cache"
147 passed, 1 skipped, 2557 deselected, 1 warning in 6.04s
```

Derived sizes (`wc -l`, `git diff --numstat`, AST -- not typed):

```
$ wc -l backend/backtest/fundamentals_coverage.py backend/tests/test_phase_82_21_fundamentals_coverage.py
     307 backend/backtest/fundamentals_coverage.py
     409 backend/tests/test_phase_82_21_fundamentals_coverage.py

$ git diff --numstat -- backend/backtest/historical_data.py backend/backtest/backtest_engine.py backend/backtest/analytics.py backend/tests/test_phase_82_13_preload_refusal_handling.py
9       1       backend/backtest/analytics.py
73      1       backend/backtest/backtest_engine.py
31      1       backend/backtest/historical_data.py
19      1       backend/tests/test_phase_82_13_preload_refusal_handling.py

$ python3 -c "...ast walk for test_ functions..."
21
```

---

## 4. Mutation matrix

Script: `scratchpad/mutate_82_21.py`. Each mutant asserts its target exists
before replacing, hashes the file after writing, always restores, and the run
re-verifies the restored tree.

| # | Mutant (production site) | Result |
|---|---|---|
| M1 | hardcode the availability flag to `False` | KILLED |
| M2 | hardcode it to `True` | KILLED |
| M3 | drop the seed flag from the early-return path | KILLED |
| M4 | `if fundamentals:` -> `if True:` at the feature builder | KILLED |
| M5 | delete the engine refusal (`raise` -> `pass`) | KILLED |
| M6 | make the derived dependent set empty (gate silently off) | KILLED |
| M7 | claim a later coverage start without re-measuring | KILLED |
| M8 | make `window_is_covered` always `True` | KILLED |
| M9 | drop `fundamentals` from the `BacktestResult` default | KILLED |
| M10 | remove the `analytics` double-write | KILLED |
| M11 | remove the recorder call from `run_backtest` | KILLED |
| M12 | add the flag to `_NUMERIC_FEATURES` (regime dummy) | KILLED |
| M13 | paraphrase the operator instruction in the artifact | KILLED |
| M14 | drop the recorded DECISION line | KILLED |
| M15 | let the snapshot loader tolerate a missing key | KILLED |
| M16 | let the derivation return an empty `F` instead of raising | KILLED |

**16 of 16 mutants died.** That licenses exactly "these 16 were killed" -- it is
NOT a claim that no survivor exists.

### The survivor, and the production bug it exposed

**M16 survived the first run**, and chasing it found a real defect in my own
derivation rather than only a missing test.

My vacuity guard covered "the `if fundamentals:` block is gone" but not "the
block exists and `F` comes out empty". Writing the missing fixture -- a key
assigned both inside and outside the block -- did **not** produce an empty `F`,
because my `outside` set was computed as `assigned(fn) - inside`. Subtracting
`inside` means a key assigned in BOTH places was classified as
fundamentals-only, which is the opposite of the rule the module documents
("assigned ONLY inside"). Fixed by excluding the block subtrees by node
identity instead of by set subtraction.

**Honest scope of that fix:** it changed no live answer. `F` is 17 keys and the
dependent set is `{qarp}` both before and after, because no key in the real
`build_feature_vector` is assigned in both places. It removed a latent
misclassification, not an active one -- and it was found by a mutant, not by
reading the code I had just written.

---

## 4b. Cycle-2 corrections (Q/A CONDITIONAL -> fixed)

Cycle-1 verdict: CONDITIONAL, all four criteria MET, two blockers. Verbatim at
`handoff/current/evaluator_critique_82.21.md` (raw return in
`qa_returns/82.21_cycle1.output.json`).

**B1 -- the Python lint gate was RED on a file this step created.**
`uvx ruff check --select F821,F401,F811` over the git-derived scope exited 1:
`F401 SNAPSHOT_PATH imported but unused` in the new test module. I never ran
that gate; the Q/A did. Import removed; the gate now exits 0 over a
git-derived, asserted-non-empty 6-file scope:

```
$ FILES=$( { git diff --name-only HEAD -- '*.py'; git ls-files -o --exclude-standard -- '*.py'; } | sort -u )
$ test -n "$FILES" || exit 1
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811
All checks passed!
ruff exit=0
```

**B2 -- a latent false-negative in my own gate, and the more serious one.**
`label_fundamentals_dependent_strategies` skipped every callee whose name
started with `_compute`, while its docstring promised reads are inherited
"transitively through any helper it calls". **Every** label function in
`STRATEGY_REGISTRY` is named `_compute_*`, so the exclusion matched exactly this
module's dominant convention and could only ever produce FALSE NEGATIVES -- and
a false negative here lets a fundamentals-dependent strategy run on an uncovered
window, which is precisely what the gate exists to prevent.

I have no justification for that exclusion; it was wrong, not a trade-off.
Removed -- the derivation now follows every callee defined in the module, with
cycle protection. The docstring now also states the deliberate
over-approximation in `_string_keys_read` (it collects every string-constant
subscript in scope, so it over-REFUSES rather than under-refusing; loud beats
silent).

**Why my 16-mutant matrix missed it: no guard drove a transitive hop at all.**
The derivation tests exercised only DIRECT reads, so the entire transitive
branch was uncovered and no mutation of it could fail. Added
`test_derivation_follows_a_transitive_hop_between_compute_helpers`, which drives
`_compute_outer_label -> _compute_inner_helper` -- exactly the shape the old
exclusion dropped -- plus two new mutants:

| # | Mutant | Result |
|---|---|---|
| M17 | restore the `_compute` callee exclusion (the Q/A's finding) | KILLED |
| M18 | stop following transitive callees entirely | KILLED |

Full re-run: **18 of 18 mutants killed, 0 survived**, restored tree GREEN.

**Scope of the B2 fix, stated honestly:** no live answer changed. `F` is 17 keys
and the dependent set is `{qarp}` before and after, which the Q/A independently
confirmed by re-deriving with full recursion. It was a latent false-negative in
a gate whose entire value proposition is "DERIVED, never stale" -- so it
mattered, but it was not producing a wrong result today.

```
$ python3 -c "...fundamentals_only_feature_keys / label_fundamentals_dependent_strategies..."
F: 17 | dependent: ['qarp']

$ python -m pytest backend/tests/test_phase_82_21_fundamentals_coverage.py -q
......................                                                   [100%]
22 passed in 1.45s

$ python -m pytest backend/tests/ -q -k "backtest or fundamental or analytics or 82_1 or 82_2 or historical or cache or strategy"
358 passed, 1 skipped, 2347 deselected, 1 warning in 36.07s
```

No production behaviour changed in cycle 2 beyond the derivation fix above; no
criterion was touched.

## 5. Files changed

| File | Change |
|---|---|
| `backend/backtest/fundamentals_coverage.py` | NEW, 326 lines |
| `backend/backtest/_fundamentals_coverage.json` | NEW, measured snapshot |
| `backend/tests/test_phase_82_21_fundamentals_coverage.py` | NEW, 443 lines, 22 tests |
| `backend/backtest/historical_data.py` | `+31 / -1` -- availability flag on every return path |
| `backend/backtest/backtest_engine.py` | `+73 / -1` -- `_preload_fundamentals_and_record`, default extension, call site |
| `backend/backtest/analytics.py` | `+9 / -1` -- double-write into `report["analytics"]` |
| `backend/tests/test_phase_82_13_preload_refusal_handling.py` | `+19 / -1` -- see §6 |
| `.claude/masterplan.json` | queued follow-on steps (§7) |
| `handoff/current/*_82.21.md` | contract, brief, results |

Everything else in the dirty tree belongs to other sessions and is not staged;
commits use deliberate `git add <paths>`.

## 6. A prior step's test I deliberately changed

`test_phase_82_13_preload_refusal_handling.py::test_result_carries_data_availability_by_default`
asserted `r.data_availability == {"macro": True}` by **exact equality**, which
also pinned the dict's size. Extending the record with `fundamentals` broke it.

Its stated intent -- *"a default result must claim availability, so only an
explicit refusal marks a run degraded"* -- is untouched and now enforced more
strongly: the rewrite asserts `macro is True`, that the record is non-empty,
**and that no key defaults to a degraded value, for any key present now or
later**. 82.13's own design note says the field is defaulted precisely so it can
be extended without disturbing existing sites; the exact-equality assertion was
incidental over-specification. Disclosed here, in the test docstring, and in the
contract -- not silently edited.

## 7. Queued out of scope (each its own research-gated step)

Named in the contract's non-scope and **filed as real steps before this
artifact was graded** -- verify with:

```
$ python3 -c "import json; d=json.load(open('.claude/masterplan.json')); \
s={str(x['id']):x for x in d['phases'][105]['steps']}; \
print([(i, s[i]['status'], s[i]['priority'], len(s[i]['verification']['criteria'])) \
for i in ('82.50','82.51','82.52','82.53')])"
[('82.50', 'pending', 'P1', 5), ('82.51', 'pending', 'P1', 5), ('82.52', 'pending', 'P2', 4), ('82.53', 'pending', 'P2', 5)]
```

1. **82.50 -- the SEC EDGAR XBRL ingester** -- the decision above; 3-5 days plus an
   annually recurring taxonomy obligation.
2. **82.51 -- the publication-lag embargo** (`cache.py:612/:631`) -- the covered
   window leaks a measured 66-day mean lag on EVERY fundamentals read. This is
   the one that makes the current 2024-07+ data non-point-in-time, and it is
   independent of the branch choice.
3. **82.52 -- `quality_momentum`'s `or 0` fallback** -- `fv.get("quality_score", 0) or 0`
   makes `> 0.3` unreachable and `< 0.1` always true without fundamentals: a
   structurally bearish label manufactured from missing data. The method is
   retained so the demotion stays reversible, which makes it a live landmine.
4. **82.53 -- the silent feature-set shrink** -- an uncovered run trains on 22 of 37
   features with no record of it.

## 8. What this step does NOT do

It does not deepen coverage, does not touch `cache.py`'s cutoff filter (that is
item 2 above -- changing it here would alter every backtest's inputs inside a
step about *visibility*), does not re-register the demoted strategies, and does
not adopt any paid source. No live positions touched; paper trading untouched.
