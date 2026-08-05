# Experiment Results -- masterplan step 82.12

**Step:** 82.12 (P1) -- defect-class sweep: vacuous type guards on STRING BQ columns
**Date:** 2026-08-05 | **Cycle:** 1
**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_12_string_column_guards.py -q`

---

## 0. The headline, stated before the evidence: THE SWEEP FINDS ZERO OF WHAT IT WAS SENT TO FIND

The research gate measured it and I reproduced it: **there are no remaining vacuous
date/`isinstance` guards.** 82.0 closed the only one. `_STRING_DATE_TIMESTAMP_COLS` is
6/6 correct. Both numeric-named STRING columns are false positives.

That makes this a dangerous step shape -- "go find more of the same" when there is no
more of the same invites either an empty report that reads as a wasted cycle, or padded
false positives. The brief's re-framing was adopted: **build the oracle, prove the
surface clean BY CONSTRUCTION, and leave a standing check.** "Zero" then becomes a
measured result rather than an assertion.

**And the sweep did find a live P1** -- same root cause, different symptom. See §3.

---

## 1. Research findings RE-MEASURED by Main

**(a) The live P1.** `backend/slack_bot/jobs/_production_fns.py:220` selects `timestamp`
and `realized_pnl` from `paper_trades`. Measured against the live schema:

```
paper_trades columns: 18
  'timestamp'          present=False
  'realized_pnl'       present=False
  'created_at'         present=True  type=STRING
  'realized_pnl_pct'   present=True  type=FLOAT
```

**(b) `dry_run` is a free, authoritative instrument.** All four predicted cases
reproduced at $0:

```
FAIL | broken _production_fns query     | 400 Unrecognized name
OK   | fixed equivalent                 | bytes_billed=4936   (dry run: $0)
FAIL | STRING date >= CURRENT_DATE()    | 400 No matching signature for >=
OK   | STRING date >= '2026-01-01'      | bytes_billed=56808  (dry run: $0)
```

**(c) Oracle scale.** The brief's counts reproduce EXACTLY:
`tables: 33  columns: 477` and
`{'STRING': 189, 'FLOAT': 182, 'INTEGER': 48, 'TIMESTAMP': 31, 'DATE': 13, 'BOOLEAN': 6, 'JSON': 5, 'RECORD': 3}`.

---

## 2. What was built

| File | Change |
|---|---|
| `backend/db/schema_oracle.py` | NEW -- live+snapshot oracle, two-sided derivation, `dry_run` helper, drift diff |
| `backend/db/_schema_snapshot.json` | NEW -- checked-in oracle (33 tables / 477 columns) so guards run without ADC |
| `backend/tests/test_phase_82_12_string_column_guards.py` | NEW -- 20 tests, the immutable verification target |
| `.claude/masterplan.json` | three new queued steps (§3); 82.12 flip LAST |

**Two-sided derivation, because a name regex is itself a hand-written list.** Criterion 1
demands the scope be derived from live schemas "rather than from a hand-written list". A
regex over column NAMES satisfies that in letter and violates it in spirit -- it is a
hand-written list of name tokens, and the brief measured that it under-covers
(`calendar_events.window`, `analysis_results.overall_reliability`,
`strategy_decisions.decay_attribution`). So the candidate set is *every column the oracle
declares STRING* (128 distinct names), and `test_string_column_set_is_derived_from_declared_types_not_names`
proves no name filter is in the gate path by asserting that STRING columns carrying **no**
date/time token are still in scope.

**Every stage asserts non-empty.** Not ceremony: the researcher's first scanner reported
"0 unknown identifiers", which reads exactly like a clean bill of health, and was a
relative-path bug plus a non-greedy regex. *A checker that scans nothing and a clean
codebase produce identical output.*

---

## 3. Criterion 1 -- the derived scope (measured)

```
SQL side: files=294  sql_literals=13  tables_resolved=1  columns_in_oracle=477
PY  side: files=294  string_columns_in_oracle=128  hits=3  distinct sites=2
```

### An honest limitation of the SQL side, stated rather than buried

`tables_resolved = 1` is **low coverage**, and it is the number a reader should be
suspicious of. The SQL extractor only resolves *hardcoded* fully-qualified table names in
backticks; most SQL in this repo is built through `bq._pt_table(...)` helpers and
f-strings, which it does not resolve. So the SQL half of the join is a **spot check, not
a sweep**, and I am not claiming otherwise. The load-bearing half is the Python side
(294 files, 128 STRING column names) -- which is also where the actual 82.0 defect lived,
since that defect was never in SQL at all. Extending SQL resolution through the
`_pt_table` indirection is real work and is **not** claimed as done here.

### Narrowing that mattered

The first Python-side run produced **175 hits across 61 sites** -- almost all
`status == "done"` / `action == "BUY"`. String *equality* on a STRING column is correct
and is not date/number semantics; counting it buried the two real hits under noise.
Scope narrowed to **ordering** comparisons (`<`, `<=`, `>`, `>=`) and **non-concat**
arithmetic (`+` on a string is legitimate concatenation). Result: 3 hits / 2 sites, both
pinned by tests so the narrowing cannot silently reverse.

---

## 4. Criterion 2 -- classification of every scope member (file:line)

| Site | Column | Verdict | Why |
|---|---|---|---|
| `backend/services/outcome_tracker.py:100` | `analysis_date` | **CORRECT** | Reads `analysis_results.analysis_date`, which the oracle declares **TIMESTAMP**, so the `isinstance(datetime)` branch really does execute. Both shapes handled explicitly (`:101` passthrough, `:104` `fromisoformat(str(...))`). It entered the scope only because the same column NAME is STRING on `outcome_tracking` -- the Python side matches names across tables, a **disclosed over-approximation** resolved by reading, never by a name filter. |
| `backend/tools/sec_insider.py:226` | `date` | **CORRECT** | `t["date"] >= cutoff_date` where `cutoff_date = ...strftime("%Y-%m-%d")` (`:160`). Both sides zero-padded ISO-8601, so lexical ordering **is** chronological ordering -- correct by construction, the known-good pattern the sweep must not cry wolf on. These dicts also come from SEC Form-4 parsing, not a BQ row at all. |

**VACUOUS: 0. NEEDS-COERCION: 0.** `test_every_derived_scope_member_is_classified` fails
if a new site appears unclassified **or** if a classification entry describes a site that
no longer exists, and `test_classified_line_numbers_still_point_at_a_row_read` re-derives
each `file:line` instead of trusting the table.

---

## 5. Criterion 3 -- nothing vacuous left unfixed; the NAME variant queued

Zero vacuous hits, so the "fix or queue" clause has an empty subject *for the type
variant*. The **name variant** is where the real defect is, and it is queued:

| New step | What |
|---|---|
| **82.39** (P1) | `_production_fns.py` selects `paper_trades.timestamp` / `.realized_pnl`, neither of which exists. BQ 400s, `except Exception` swallows it, `nightly_outcome_rebuild` has been running on **zero trades**. |
| **82.40** (P2) | `preload_prices` / `preload_fundamentals` have **no staleness gate at all** -- a MISSING guard beside a now strictly-gated `preload_macro`. Cross-referenced to 82.21 so the SLA is not set to a value that makes every historical run refuse. |
| **82.41** (P3) | `pyrightconfig.json` pins `.venv312` / Python 3.12 while the project runs 3.14, and sets no `typeCheckingMode` -- measured 0 diagnostics, i.e. a configured-but-inert checker that looks like a passing gate. |

---

## 6. Criterion 4 -- the fixed guard, fed the PRODUCTION type FROM THE ORACLE

The guard fixed for this class is `cache.preload_macro`'s staleness gate (repaired in
82.0). The fixture reads the declared type **out of the oracle** rather than hardcoding
`str`, so a future `STRING -> DATE` migration breaks the test loudly instead of quietly
testing the wrong thing.

- `test_fixture_emits_the_type_the_oracle_declares` -- precondition on the fixture itself.
- `test_guard_fires_on_stale_data_of_the_production_type` -- **positive**: 400-day-old
  macro must yield 0. Pre-82.0 this returned non-zero, because `isinstance(rd, date)` was
  False for every `str` row and the branch never ran.
- `test_guard_does_not_fire_on_fresh_data_of_the_production_type` -- **negative**: a guard
  that always refuses is not a guard.
- `test_guard_fails_closed_on_unparseable_dates`.

---

## 7. Recall test -- "found nothing" vs "cannot see anything"

The single most important test in the file. The instrument is run against the phase-25.D7
guard **exactly as it stood before 82.0** and must flag it
(`test_instrument_detects_the_known_pre_fix_defect`); and run against the coercing form
82.0 replaced it with, where it must **not** fire
(`test_instrument_does_not_flag_the_fixed_coercing_form`). Without the first, every
"clean" result this instrument produces is worthless.

---

## 8. Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_12_string_column_guards.py -q
....................                                                     [100%]
20 passed in 3.07s
```

---

## 9. Mutation matrix -- CONTROL re-derived in the same run

CONTROL `rc=0 passed=20`; POST-RESTORE `rc=0 passed=20`.

| Mutant | Form | Mutation | Result | Tests ACTUALLY killed |
|---|---|---|---|---|
| M1 | code | Oracle returns an empty schema | **KILLED** | 10 tests incl. `test_derived_scope_is_non_empty` |
| M2 | code | FQ-table regex made non-greedy | **EQUIVALENT** (see §9.1) | -- |
| M3 | code | isinstance-date context unrecognised (instrument blind) | **KILLED** | `test_instrument_detects_the_known_pre_fix_defect`, `test_every_derived_scope_member_is_classified` |
| M4 | code | String equality re-admitted as date/number semantics | **KILLED** | `test_string_equality_is_not_treated_as_date_or_number_semantics` |
| M5 | code | String concatenation re-admitted as numeric | **KILLED** | `test_ordering_comparison_and_numeric_arithmetic_are_in_scope` |
| M6 | code | Keyword filter re-applied blindly -> `timestamp` hidden | **KILLED** | `test_query_selecting_nonexistent_columns_is_detected` |
| M7 | code | Alias `re.sub` strip removed | **EQUIVALENT** (see §9.1) | -- |
| M8 | code | `refresh_snapshot` no longer refuses an empty oracle | **KILLED** | `test_refresh_refuses_to_persist_an_empty_oracle` |
| M9 | code | `preload_macro` staleness gate disarmed (the 82.0 defect restored) | **KILLED** | `test_guard_fires_on_stale_data_of_the_production_type` |
| M10 | code | `preload_macro` fail-closed branch removed | **KILLED** | `test_guard_fails_closed_on_unparseable_dates` |
| M11 | code | Queued step 82.39 deleted from the masterplan | **KILLED** | `test_the_nonexistent_column_defect_is_queued_as_its_own_step` |
| F1 | fixture | Fixture emits a python `date` where production emits `str` | **KILLED** | `test_fixture_emits_the_type_the_oracle_declares` |
| F2 | fixture | Snapshot claims `historical_macro.date` is DATE (drift lie) | **KILLED** | `test_fixture_emits_the_type_the_oracle_declares`, `test_instrument_detects_the_known_pre_fix_defect` |

### 9.1 The two survivors are EQUIVALENT MUTANTS -- proven, not assumed

Neither is reported as a finding, because neither has a behavioural differential. I
required one before calling a survivor a gap.

- **M2 (non-greedy regex).** Measured: `derive_scope` output is byte-identical.
  Non-greedy still captures `('sunny-might-477607-p8', 'financial_reports',
  'paper_trades')` because the trailing backtick anchors the final group. The
  researcher's real bug was a *different* regex; this mutation does not reproduce it.
- **M7 (alias `re.sub` removed).** Measured: `unknown_columns` identical, because the
  separately-collected `aliases` set already filters `pnl`. **This means the `re.sub`
  alias-strip is redundant dead code** -- a simplification, not a defect. Disclosed
  rather than silently removed, since removing it is a change this step's contract does
  not scope.

### 9.2 A guard that was passing for the WRONG reason, caught before the Q/A

`test_the_nonexistent_column_defect_is_queued_as_its_own_step` was first written as
`assert "_production_fns" in masterplan_text`. It **passed before the step was queued**:
measured, `_production_fns` already occurs 1x and `nightly_outcome_rebuild` 7x in
unrelated steps. Same substring-scan class as 82.10's M4. Rewritten to walk the actual
step objects and require one OPEN step naming the full signature (`_production_fns`,
`paper_trades`, `timestamp`, `realized_pnl`) **and** carrying verification criteria.
Mutant M11 confirms it now dies when the step is removed.

---

## 10. Regression + scope honesty

```
$ python -c "import ast; ast.parse(...)  # schema_oracle.py"          syntax OK
$ python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py -q  (see below)
```

**Changed:** one new module, one generated snapshot, one new test file, three new pending
masterplan steps. **NOT changed:** no query semantics, no schema migration, no
`_STRING_DATE_TIMESTAMP_COLS` behaviour, no live position, no credential, no
operator-gated flag. Paper trading left running. `handoff/current/phase83_research_raw/`
(another session's work) untouched and excluded from the commit.

**Explicitly NOT claimed:** that this sweep covers all SQL in the repo (see §3 -- SQL
resolution is a spot check at `tables_resolved=1`); that `dry_run` validation is wired
into CI (it is available as `schema_oracle.dry_run`, exercised manually, not enforced);
that static analysis could have caught any of this (measured: it cannot -- a BQ row is an
untyped dict and the decisive fact lives in the schema).

---

# CYCLE 2 -- response to the cycle-1 Q/A CONDITIONAL

Verdict verbatim in `handoff/current/evaluator_critique_82.12.md`; raw return at
`handoff/current/qa_returns/82.12_cycle1.output.json`. Two WARN findings, both accepted.

## 11. RETRACTION -- my headline claim exceeded the instrument's recall

Section 0 asserted "**there are no remaining vacuous date/isinstance guards**". The Q/A
ran **8 shapes of the same defect class** through `_row_key_reads` and measured that the
instrument saw only **1**. So the claim was an Overgeneralization: what I had measured was
"zero hits under this instrument", and I stated it as "zero defects exist". I disclosed
the SQL-side limit prominently (§3) and disclosed the Python-side recall envelope --
*the half I called load-bearing* -- nowhere.

**Fixed by raising the recall, not by softening the sentence.** `_row_key_reads` now
resolves a row value through attribute access (`row.date` -- a BQ `Row` supports it),
`AnnAssign`, tuple unpacking, walrus, and **call arguments**. Measured after the change:

```
SEEN   | subscript_then_isinstance     SEEN   | annassign
SEEN   | get_then_isinstance           SEEN   | tuple_unpack
SEEN   | attribute_access              SEEN   | helper_wrapped_call
SEEN   | walrus
MISSED | attribute_on_bound_value      <- known, accepted, and now ASSERTED
```

**7 of 8, up from 1 of 8.** The remaining miss (`rd = row["date"]` then `rd.year` --
implicit date semantics with no isinstance and no comparison) needs type inference, not
an AST walk. It is now a **parametrized test** (`test_recall_envelope_is_measured_not_assumed`)
that asserts each shape SEEN or MISSED, so any recall change in either direction fails
the suite and forces the completeness wording to be restated.

**The corrected claim:** zero vacuous guards **under an instrument with the recall
envelope asserted above**. Not "zero exist".

## 12. The anti-cry-wolf test was ILLUSORY -- and is now real

The Q/A proved `test_instrument_does_not_flag_the_fixed_coercing_form` passed for the
wrong reason: the instrument could not resolve a row value through *any* call, so it
returned `[]` for the correct coercing form **and** for a genuinely vacuous
helper-wrapped `isinstance(passthrough(row.get("date")), date)`. It demonstrated
blindness to both, not discrimination between them; no mutation could turn it red.

Now that call arguments resolve, the two forms come apart -- measured:

```
FIXED (coercing)                       -> isinstance_date hits: []
VACUOUS (helper-wrapped isinstance)    -> isinstance_date hits: [('date','isinstance_date')]
```

Replaced by `test_instrument_discriminates_coercing_from_vacuous_helper`, which asserts
**both** halves. The claim now has a failure mode.

## 13. A false-positive class I found while raising recall

Raising recall surfaced 2 new hits at `backend/api/paper_trading.py:737,1294`
(`len(raw) > 50`) attributed to the column `signals`. **Wrong:** `raw = trade.get("signals")`
is bound in a *different function*, and my binder was module-wide, so a name leaked
across unrelated functions. Fixed -- bindings now resolve **per function**
(`_reads_in_scope`, which does not descend into nested function bodies). Both false
positives disappeared. A sweep that cries wolf is worse than one with a stated recall
limit, because every future reader has to re-litigate it.

## 14. Re-derived scope and classification (grew 2 -> 5 sites, all CORRECT)

| Site | Column | Verdict |
|---|---|---|
| `backend/backtest/cache.py:530` | `report_date` | **CORRECT** -- explicit `str()` + lexical ISO vs cutoff |
| `backend/backtest/cache.py:590` | `date` | **CORRECT** -- same pattern; same module as the 82.0 defect but NOT the same shape (operates on the production type directly rather than isinstance-testing it) |
| `backend/services/outcome_tracker.py:101` | `analysis_date` | **CORRECT** (unchanged) |
| `backend/tools/sec_insider.py:226` | `date` | **CORRECT** (unchanged) |
| `backend/tools/sec_insider.py:240` | `date` | **CORRECT** (unchanged) |

**VACUOUS: 0. NEEDS-COERCION: 0** -- now on an instrument with 7/8 measured recall
rather than 1/8.

## 15. Verification + mutation re-run

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_12_string_column_guards.py -q
............................                                             [100%]
28 passed in 4.42s
```

Mutation matrix re-run: **CONTROL 28 / POST-RESTORE 28**, all mutants KILLED except the
two previously proven EQUIVALENT (M2, M7) -- unchanged status, still reported as
equivalents rather than as survivors. Regression: `test_phase_82_0_macro_ingestion.py` +
`test_phase_82_10_freshness_paging.py` = **32 passed**.

## 16. What changed in cycle 2

`backend/db/schema_oracle.py` (recall + per-function binder) and
`backend/tests/test_phase_82_12_string_column_guards.py` (recall-envelope test,
discrimination test, 2 new classifications). No masterplan change. No behaviour change to
any production query, guard, or job -- `schema_oracle.py` has no production callers yet;
it is an instrument plus its tests.

---

# CYCLE 3 -- response to the cycle-2 Q/A CONDITIONAL

Verdict verbatim in `evaluator_critique_82.12.md`; raw return at
`qa_returns/82.12_cycle2.output.json`. THREE WARN findings, all accepted, all closed and
each one mutation-verified.

## 17. Finding 1 -- a DEAD OR-CLAUSE in the guard I had just added to fix an overclaim

The recall envelope read
`seen = any(k=="date" and c=="isinstance_date") or any(k=="date")`. Clause A is a strict
**subset** of clause B, so `A or B == B` and clause A was dead. The envelope asserted only
"the key was seen in *some* context" -- not that the `isinstance_date` context is
recognised, which is the property §11 claims it measures. The Q/A proved it by killing the
context regex: `seen` stayed `True`.

Fixed by deleting the redundant clause. **Mutation-verified:** killing the context regex
now fails `test_recall_envelope_is_measured_not_assumed` (plus 3 more).

## 18. Finding 2 -- the other half of the discrimination test had no failing mutation

`assert vacuous_hits` was a genuine kill. `assert not fixed_hits` was not: `_FIXED_SOURCE`
contained no `isinstance` applied to a resolvable row value, so it returned `[]` under
every mutation.

Fixed by giving `_FIXED_SOURCE` a **resolvable and correct** guard --
`isinstance(row.get("date"), str)`, which is the right way to test a STRING column and
must never be reported as a date-guard defect. **Mutation-verified:** forcing
`ctx = "isinstance_date"` unconditionally makes that a false positive and
`test_instrument_discriminates_coercing_from_vacuous_helper` goes RED.

## 19. Finding 3 -- the envelope omitted the shape ALL FOUR real coercers use

The Q/A ran 12 shapes to my 8 and found the **two-statement bind-through-call** form
(`rd = parse(row["date"])` then `isinstance(rd, date)`) invisible -- the exact shape used
by `cache.py:137/382`, `reconciliation.py:41`, `paper_round_trips.py:39`,
`wash_sale_filter.py:30-32`. An envelope that misses what the codebase actually writes is
not an envelope.

### The fix taught me the principle I had missed

Binding through a call is **directional**, and my first attempt got it wrong in the
expensive direction. Resolving calls for *every* context took the scope **5 -> 18 sites**,
and I read all 13 new ones: every one was a **correctly coerced value** --
`paper_trader` `_l2u` (an FX *rate*, not the `market` string), `analytics.py:479` `d_entry`
(a `datetime` from `fromisoformat`), `cycle_health.py:516` `hb_updated`. The reason is
simple once stated:

> For **`isinstance`**, a call between the row read and the guard is the **DEFECT** shape
> -- a non-coercing helper leaves a `str` and the guard is vacuous.
> For **arithmetic / ordering**, the same call is the **FIX** shape -- a coercer's whole
> job is to change the type.

So the binder now records *whether* a binding passed through a call, keeps such bindings
visible to `isinstance`, and suppresses them for `arith` / `order_compare`. Measured:

```
bind-through-call + isinstance -> [('date', 'isinstance_date')]     (defect shape: SEEN)
bind-through-call + arith      -> []                                (fix shape: not flagged)
scope sites: 5                                                      (back from 18)
```

Both shapes are now rows in the asserted envelope. **Mutation-verified:** removing the
bind-through-call resolution fails `test_recall_envelope_is_measured_not_assumed`.

## 20. What the cycle-2 Q/A established that I could not have

It ran a **second, differently-operationalized census** -- every
`isinstance(<x>, date|datetime|_dt)` site across all 294 backend files (7 sites) -- read
all of them, and found every one has an explicit `str` branch; a repo-wide scan for the
missed bind-through-call shape returned **0 sites**. So the recall residual hid **zero
vacuous guards**, and the "VACUOUS: 0" headline survives an instrument I did not build.
That is stronger evidence for the headline than anything in this document, and it is the
Q/A's, not mine.

## 21. Verification + mutation

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_12_string_column_guards.py -q
..............................                                           [100%]
30 passed in 5.07s
```

| Mutant | Targets finding | Result | Tests ACTUALLY killed |
|---|---|---|---|
| Q1 kill the `isinstance_date` context regex | 1 | **KILLED** | `test_recall_envelope_is_measured_not_assumed`, `test_instrument_detects_the_known_pre_fix_defect`, `test_instrument_discriminates_coercing_from_vacuous_helper`, `test_every_derived_scope_member_is_classified` |
| Q2 drop the `isinstance_other` filter | 2 | **KILLED** | `test_every_derived_scope_member_is_classified` |
| Q3 remove bind-through-call in the binder | 3 | **KILLED** | `test_recall_envelope_is_measured_not_assumed` |
| Q4 force `ctx = "isinstance_date"` always | 2 (the named one) | **KILLED** | `test_instrument_discriminates_coercing_from_vacuous_helper`, `test_every_derived_scope_member_is_classified` |

CONTROL clean before and POST-RESTORE clean after every mutant.

## 22. Scope after cycle 3

The derived scope is the SAME 5 sites as cycle 2, all classified **CORRECT**; **VACUOUS:
0, NEEDS-COERCION: 0** -- now under an instrument whose envelope includes the shape every
real coercer in this repo uses. Changed this cycle: `backend/db/schema_oracle.py` and
`backend/tests/test_phase_82_12_string_column_guards.py` only. No masterplan change, no
production behaviour change -- `schema_oracle.py` still has no production callers; the
only importer is its test file.
