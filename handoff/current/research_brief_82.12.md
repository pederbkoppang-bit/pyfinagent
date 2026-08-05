# Research Brief -- Step 82.12: Vacuous type-assumption guards over BigQuery STRING columns

**Tier:** complex | **Audit-class:** true (loop-until-dry, K=2)
**Researcher:** Layer-3 researcher (merged external + internal)
**Started:** 2026-08-05
**Status:** IN PROGRESS -- written incrementally (write-first discipline)

---

## 0. The question

Step 82.12 (P1) is a DEFECT-CLASS SWEEP. The confirmed instance (fixed in 82.0):
`backend/backtest/cache.py::preload_macro` tested `isinstance(rd, datetime.date)`
against `historical_macro.date`, declared `('date','STRING','REQUIRED')`. The guard
was VACUOUS -- it never fired, so it never refused stale macro; it silently served
212-day-old data.

The job: SWEEP, not spot-fix. Derive the STRING-typed date/number column scope FROM
LIVE SCHEMAS, find every consumer, classify each guard vacuous / correct /
needs-coercion, with file:line.

Sections below are appended as work proceeds.

---

## 1. Search queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| Q1 | `BigQuery Python client RowIterator type mapping STRING DATE TIMESTAMP to Python types` | **YEAR-LESS canonical** |
| Q2 | `unsatisfiable predicate always-false condition static analysis dead code detection` | **YEAR-LESS canonical** |
| Q3 | `mutation testing guard clause equivalence mutant prove a condition can fail` | **YEAR-LESS canonical** |
| Q4 | `schema drift type mismatch data pipeline silent failure detection 2026` | **current-year frontier** |
| Q5 | `vacuous guard clause test that always passes branch coverage 2025` | **last-2-year window** |

Q2 is the one that paid: the year-less variant returned the AdaCore/MISRA-C treatment and
Microsoft's CA1508 analyzer -- the decades-old canonical framing of this exact defect
class. Q4 (2026-locked) returned only LLM-era data-pipeline blog posts and no primary
analysis literature, which is precisely the failure mode the year-less rule exists to
prevent.

---

## 2. Sources read in full

### S1. BigQuery GoogleSQL Conversion rules -- OFFICIAL DOCS
URL: https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_rules
Accessed 2026-08-05. Fetched via `curl -sL` + tag-strip (WebFetch returns nav-only on
cloud.google.com -- known, see auto-memory `feedback_gcloud_docs_fetch.md`). Page's own
"Last updated 2026-07-31 UTC".

VERBATIM, the cast/coerce table row for STRING:

> **From type** STRING -- **Cast to**: `BOOL INT64 NUMERIC BIGNUMERIC FLOAT64 STRING
> BYTES DATE DATETIME TIME TIMESTAMP RANGE` -- **Coerce to**: *(empty)*

and for DATE:

> **From type** DATE -- **Cast to**: `STRING DATE DATETIME TIMESTAMP` -- **Coerce to**: *(empty)*

Supertype table:

> STRING -> Supertypes: `STRING`
> DATE   -> Supertypes: `DATE`

Literal-coercion caveat (this is the trap that makes people *think* it works):

> "GoogleSQL supports the following literal coercions: **STRING literal** -> `DATE
> DATETIME TIME TIMESTAMP` ... for example, if function func() takes a DATE argument,
> then the expression func("2014-09-27") is valid because the string literal
> "2014-09-27" is coerced to DATE."
> "**Note: String literals don't coerce to numeric types.**"

**What this establishes (load-bearing for the sweep):**
1. A STRING **column** has an EMPTY "Coerce to" set. Coercion applies to *literals and
   query parameters ONLY*. So `WHERE string_date_col >= DATE('2026-01-01')` does not
   silently work -- STRING and DATE share no supertype, so it is a type error, not a
   silent wrong answer. **In SQL the defect is LOUD.** It is only in **Python** that the
   defect goes silent (Python compares `str` to `str` happily and `isinstance` just
   returns False). That asymmetry is the whole reason 82.0's bug survived: the SQL layer
   would have screamed; the Python layer whispered.
2. String literals **do not** coerce to numeric types at all -- so the two numeric-named
   STRING columns (`paper_positions.stop_advanced_at_R`, `calendar_events.confidence`)
   have no free lunch even at the literal level.
3. `SAFE_CAST` is the documented protection: "When using CAST, a query can fail if
   GoogleSQL is unable to perform the cast. If you want to protect your queries from
   these types of errors, you can use SAFE_CAST." This is exactly why
   `cycle_health.py::_STRING_DATE_TIMESTAMP_COLS` exists.

---

### S2-S7 (read in full via WebFetch, 2026-08-05)

| # | URL | Tier | Read in full | What it establishes |
|---|---|---|---|---|
| S1 | https://cloud.google.com/bigquery/docs/reference/standard-sql/conversion_rules | official docs | YES (curl+strip) | STRING columns have an EMPTY coerce-to set; coercion is literal/parameter-only; "String literals don't coerce to numeric types"; SAFE_CAST is the documented protection. -> **In SQL the defect is LOUD; only Python makes it silent.** |
| S2 | https://learn.microsoft.com/en-us/dotnet/fundamentals/code-analysis/quality-rules/ca1508 | official docs | YES | The state-of-the-art always-true/false analyzer. "A method has conditional code that always evaluates to `true` or `false` at runtime. This leads to dead code in the `false` branch." Crucially: it is **"Enabled by default in .NET 10: No"** and "performs an expensive dataflow analysis of non-constant values". -> even the best-in-class tool is off by default and works only on typed, intraprocedural values. |
| S3 | https://learn.adacore.com/courses/SPARK_for_the_MISRA_C_Developer/chapters/08_unreachable_and_dead_code.html | authoritative (AdaCore/MISRA) | YES | Canonical framing. MISRA C **Rule 2.1** "A project shall not contain unreachable code"; **Rule 2.2** "There shall not be dead code". Both are "actively harmful, as they might confuse programmers and lead to errors during maintenance". And the key epistemic limit: "code reported as not being executed is not necessarily unreachable (it could simply reflect gaps in the test suite)". |
| S4 | https://docs.getdbt.com/docs/mesh/govern/model-contracts | official docs | YES | The data-contract answer to schema drift. Contracts enforce "Column data types match exactly"; dbt runs "a 'preflight' check to ensure that the model's query will return a set of columns with names and data types matching the ones you have defined" and otherwise "it will fail to build". -> the industry pattern is **declare + verify at build time**, not defensive isinstance at read time. |
| S5 | https://coverage.readthedocs.io/en/latest/branch.html | official docs | YES | "Where a line in your program could jump to more than one next line, coverage.py tracks which of those destinations are actually visited, and flags lines that haven't visited all of their possible destinations." A vacuous guard shows as a **partial branch**: the `if` line is covered, the taken-branch destination never is. -> **branch coverage catches this class; line coverage does not.** |
| S6 | https://dusted.codes/guard-clauses-without-test-coverage-a-common-tdd-pitfall | authoritative blog | YES | Names the human mechanism: "It is very tempting to write a bit more code if you already know what the desired end result should look like." Guard clauses get written ahead of any test that requires them, so nothing ever proves they fire. Exactly the phase-25.D7 story. |
| S7 | https://mutmut.readthedocs.io/en/latest/ | official docs | YES | Mutation-testing mechanics for criterion 4. "Integer literals are changed by adding 1... `<` is changed to `<=`. `break` is changed to `continue` and vice versa." A **surviving mutant** = the suite cannot tell the mutated code from the original = the guard is not actually tested. Notably the page does NOT cover equivalent mutants -- see the arXiv snippet source for that. |

---

## 3. Snippet-only sources (evaluated, NOT read in full -- do not count toward the gate)

| URL | Kind | Why not read in full |
|---|---|---|
| https://arxiv.org/abs/1303.2784 (Using State Infection Conditions to Detect Equivalent Mutants) | peer-reviewed | Abstract page only; per the gate rules an abstract is not a full read. Establishes the concept used below: proving an infection condition **unsatisfiable** proves mutant equivalence -- i.e. the formal statement of "this guard cannot be killed". |
| https://arxiv.org/pdf/2212.13933 (Coding Guidelines and Undecidability) | peer-reviewed | Binary PDF; not needed once MISRA framing was obtained from S3. |
| https://arxiv.org/pdf/1612.05675 (Targeting Infeasibility Questions on Obfuscated Codes) | peer-reviewed | Source of the "if both branches of a predicate are UNSAT the predicate is dead" formulation. |
| https://arxiv.org/pdf/1204.6719 (Design and Algorithms of a Verification Condition Generator) | peer-reviewed | "a dead command is a program whose VC computed by the weakest precondition method is unsatisfiable". |
| https://arxiv.org/pdf/2112.14151 (Cerebro: Static Subsuming Mutant Selection) | peer-reviewed | Mutant-selection theory; beyond scope. |
| https://arxiv.org/pdf/1803.07901 (Selecting Fault Revealing Mutants) | peer-reviewed | ditto. |
| https://arxiv.org/pdf/2104.11767 (Mutation Coverage vs Branch Coverage, Industrial Setting) | peer-reviewed | Directly relevant comparison; snippet sufficed. |
| https://www.cs.cornell.edu/courses/cs5154/2021sp/resources/MutationTesting.pdf | academic course | Binary PDF. |
| https://docs.greatexpectations.io/docs/core/introduction/try_gx/ | official docs | **Attempted WebFetch; returned overview only, no type-expectation list.** Recorded as an honest failed full-read, not padded. |
| https://tai-e.pascal-lab.net/en/pa3.html (A3: Dead Code Detection) | academic course | Dataflow-based dead-code assignment; snippet sufficed. |
| https://www.sciencedirect.com/topics/computer-science/unreachable-code | reference | paywalled/aggregated. |
| https://vfunction.com/blog/dead-code/ | vendor blog | low tier. |
| https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.table.RowIterator | official docs | **Superseded by direct measurement** -- I measured the actual BQ->Python mapping against the live table (5.B), which is stronger evidence than the doc page. |
| https://cloud.google.com/python/docs/reference/bigquery/3.14.0/google.cloud.bigquery.table.RowIterator | official docs | version-pinned duplicate |
| https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client | official docs | context |
| https://medium.com/@manik.ruet08/strategies-for-detecting-schema-drift-in-data-pipelines-3e49569d4ffc | community | low tier |
| https://totalshiftleft.ai/blog/api-schema-validation-catching-drift | vendor blog | low tier |
| https://streamkap.com/resources-and-guides/schema-drift-detection | vendor blog | low tier |
| https://www.databahn.ai/blog/maintaining-99-ocsf-compliance-at-enterprise-scale-the-schema-drift-challenge | vendor blog | low tier |
| https://blog.anomalyarmor.ai/data-pipeline-monitoring-how-to-stop-silent-failures-before-they-hit-production/ | vendor blog | low tier |
| https://www.thedataops.org/schema-drift/ | community | low tier |
| https://www.hst.ie/blog/how-to-diagnose-and-fix-data-flow-failures-in-production-systems-before-they-impact-revenue/ | vendor blog | low tier |
| https://www.cs.odu.edu/~cs252/Book/branchcov.html | academic course | branch-coverage definition; S5 is authoritative |
| https://wiki.c2.com/?GuardClause | community wiki | the original guard-clause pattern name |
| https://danielnouri.org/notes/2025/11/03/modern-python-ci-with-coverage-in-2025/ | blog | 2025 recency hit; see section 4 |
| https://github.com/rust-lang/rust/issues/124118 | issue tracker | branch-coverage instrumentation limits |
| https://softengbook.org/articles/mutation-testing | textbook site | equivalent-mutant undecidability |
| https://rareskills.io/post/solidity-mutation-testing | blog | cross-domain (Solidity) |
| https://www.augmentcode.com/guides/mutation-testing-ai-generated-code | vendor blog | 2025/26 recency hit; see section 4 |
| https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8745578 | patent | "Eliminating false-positive reports resulting from static analysis" |
| https://greatexpectations.io/expectations | official docs | GX expectation gallery (pointer only) |

**Unique URLs collected across all five searches + direct fetches: 39** (7 read in full + 32 snippet-only, as enumerated in the two tables above).

---

## 4. Recency scan (last 2 years, 2024-2026)

**Performed.** Q4 (`...2026`) and Q5 (`...2025`) were run explicitly for this section.

**Result: 3 findings in the 2024-2026 window. NONE supersede the canonical sources; two
complement them, and one is a cautionary framing I am rejecting.**

1. **Schema drift is now framed as the dominant silent-failure mode in data pipelines
   (2026 practitioner consensus).** The recurring claim across Q4's results is that
   schema drift "allows partially valid data to flow through, creating silent corruption
   downstream" -- contrasted with infrastructure failures that stop execution. This is a
   verbatim description of the 82.0 bug and of the new 5.K defect, and it is the reason
   this is a P1 rather than a tidy-up. One frequently repeated statistic -- "67% of data
   pipeline failures originate from schema changes deployed without backward
   compatibility validation" -- appears only in vendor blogs with **no primary citation**;
   I am recording it as UNVERIFIED and it must not be quoted in the contract.
2. **Branch-coverage tooling in Python matured through 2025** (Q5; coverage.py 7.x,
   `danielnouri.org` 2025-11-03 modern-CI writeup). Relevant because S5's partial-branch
   report is the cheapest existing detector for a vacuous guard, and it is available today
   with no new dependency.
3. **Mutation testing is being re-promoted for AI-generated code (2025-2026).** Directly
   on point for criterion 4: the concern is code that "looks like" a guard and passes a
   suite that never exercises it. Nothing here changes the classical mutmut mechanics in
   S7.

**Nothing in the window supersedes MISRA C Rule 2.1/2.2 (S3) or CA1508 (S2) as the
canonical statement of the defect class**, and nothing in the window offers a static
technique that would have caught the pyfinagent instance -- because, as measured in 5.L,
the decisive fact lives in the BigQuery schema, not in the source.

---

## 5. Internal findings

### 5.A DERIVED SCHEMA SCOPE (criterion 1) -- FROM LIVE TABLES, not a hand-written list

**How measured.** Script at
`/private/tmp/claude-501/.../scratchpad/derive_schema.py` (copy the logic into the
repo for the real step -- see 6.A). It calls `google.cloud.bigquery.Client.list_tables()`
+ `.get_table().schema` for 4 datasets and classifies every field by declared
`field_type` + `mode`. It does NOT read any hand-written column list.

```
DATASETS SCANNED: financial_reports(14 tables, us-central1), pyfinagent_data(10),
                  pyfinagent_pms(7), pyfinagent_hdw(2)   [33 tables total]
NAME-MATCHED date/number-suggestive columns TOTAL: 213
OF WHICH DECLARED STRING:                          17
```

**THE 17 FLAGGED COLUMNS (declared STRING, name suggests date/number):**

| Dataset | Table | Column | Type | Mode |
|---|---|---|---|---|
| financial_reports | historical_fundamentals | filing_date | STRING | NULLABLE |
| financial_reports | historical_fundamentals | report_date | STRING | REQUIRED |
| financial_reports | historical_fx_rates | date | STRING | REQUIRED |
| financial_reports | **historical_macro** | **date** | STRING | REQUIRED | <- the 82.0 instance |
| financial_reports | historical_prices | date | STRING | REQUIRED |
| financial_reports | outcome_tracking | analysis_date | STRING | REQUIRED |
| financial_reports | outcome_tracking | evaluated_at | STRING | NULLABLE |
| financial_reports | paper_portfolio | inception_date | STRING | REQUIRED |
| financial_reports | paper_portfolio | updated_at | STRING | NULLABLE |
| financial_reports | paper_portfolio_snapshots | snapshot_date | STRING | NULLABLE |
| financial_reports | paper_positions | entry_date | STRING | REQUIRED |
| financial_reports | paper_positions | last_analysis_date | STRING | NULLABLE |
| financial_reports | paper_positions | **stop_advanced_at_R** | STRING | NULLABLE | <- NUMERIC-named, STRING-typed |
| financial_reports | paper_trades | created_at | STRING | REQUIRED |
| financial_reports | signals_log | exit_date | STRING | NULLABLE |
| financial_reports | signals_log | signal_date | STRING | NULLABLE |
| pyfinagent_data | **calendar_events** | **confidence** | STRING | REQUIRED | <- NUMERIC-named, STRING-typed |

**RECALL AUDIT of the heuristic (do NOT skip this -- it is a trap, see 6.B).**
Same script, second pass: 477 columns across 33 tables.
`TYPE HISTOGRAM: STRING 189, FLOAT 182, INTEGER 48, TIMESTAMP 31, DATE 13, BOOLEAN 6,
JSON 5, RECORD 3`. The regex matched 213 columns by NAME and flagged 17. But the regex
IS ITSELF A HAND-WRITTEN LIST (of name tokens), which is precisely what criterion 1
forbids. Columns it would MISS if their semantics are numeric/temporal:
`calendar_events.window`, `analysis_results.overall_reliability`,
`strategy_decisions.decay_attribution`, `alt_finra_short_volume.raw_row`. See 6.B for
the two-sided derivation that removes the name heuristic from the gate path.

### 5.B MEASURED PROOF of the BQ->Python type mapping (the sweep's whole premise)

**How measured.** Live query against `financial_reports.historical_macro` through
`google.cloud.bigquery.Client`, printing `type(v).__name__` for each cell of
`dict(row)`. Verbatim output:

```
historical_macro row: {'date': '2026-07-01', 'series_id': 'FEDFUNDS', 'value': 3.63}
  python types: {'date': 'str', 'series_id': 'str', 'value': 'float'}
native DATE/TIMESTAMP row: {'realtime_start': datetime.date(2026, 8, 5),
                            'ingested_at': datetime.datetime(2026, 8, 5, 12, 10, 0, ..., tzinfo=utc)}
  python types: {'realtime_start': 'date', 'ingested_at': 'datetime'}
```

So, on the SAME table, measured on 2026-08-05:
| Declared BQ type | Python type from `dict(row)` |
|---|---|
| STRING (`date`) | `str` |
| FLOAT (`value`) | `float` |
| DATE (`realtime_start`) | `datetime.date` |
| TIMESTAMP (`ingested_at`) | `datetime.datetime` (tz-aware, UTC) |

This is the empirical basis for "`isinstance(rd, datetime.date)` is always False for
`historical_macro.date`". It is now MEASURED against the live table, not merely inferred
from the docs. **Corollary the implementer must not miss:** `datetime.datetime` is a
SUBCLASS of `datetime.date`, so `isinstance(ts, date)` is TRUE for a TIMESTAMP column --
the mirror-image bug (a "is this a plain date?" test that silently accepts datetimes) is
real and is why `wash_sale_filter.py:30` writes `isinstance(d, date) and not
isinstance(d, datetime)`.

Side observation while measuring: `historical_macro` now holds `T10Y2Y` at `2026-08-03`
and `FEDFUNDS` at `2026-07-01` -- i.e. the 82.0 macro ingestion repair is live and the
table is no longer 212 days stale.

**Immediate observation:** every STRING-typed date column lives in `financial_reports`
(the us-central1 dataset). `pyfinagent_data` / `_pms` / `_hdw` are clean on dates --
their only flagged hit is `calendar_events.confidence`. That is a real structural
finding: the STRING-date convention is a `financial_reports`-local legacy, and the
2 non-date hits (`stop_advanced_at_R`, `confidence`) are a DIFFERENT sub-class
(numeric-semantics-in-a-STRING) that lexical comparison does NOT rescue.

---

### 5.C GUARD CLASSIFICATION (criterion 2) -- every hit, with file:line

**How measured.** `grep -rn "isinstance(" backend/ --include="*.py" | grep -v "^backend/tests/"`
= **534** total isinstance calls in non-test backend code; filtering to date/datetime type
tests (`grep -Ei "datetime\.date|datetime\.datetime|, *date\)|, *datetime\)|\(date,"`)
= **8** hits, of which 3 are comments and **5 are live code**. A second pass filtering to
numeric type tests (`float|int)|int,|(int|Number|Decimal`) = **53** hits.

| # | file:line | Guard | Column read | Declared type | CLASS | Proof / note |
|---|---|---|---|---|---|---|
| 1 | `backend/backtest/cache.py:382-391` | `_coerce_date`: `isinstance(v,_dt)` / `isinstance(v,_date)` / `isinstance(v,str)` -> `_date.fromisoformat(v[:10])` | `historical_macro.date` | STRING | **CORRECT (the 82.0 fix)** | str branch present; measured live value `'2026-07-01'` is `str` and hits the third branch |
| 2 | `backend/backtest/cache.py:136-147` | `_effective_vintage._coerce` -- **byte-identical body to #1** | `historical_macro.date` + `realtime_start` | STRING + DATE | **CORRECT but DUPLICATE** | handles both; `realtime_start` is genuinely DATE (measured -> `datetime.date`), so the same helper correctly serves a STRING col and a DATE col |
| 3 | `backend/services/wash_sale_filter.py:29-37` `_to_date` | `isinstance(d,date) and not isinstance(d,datetime)` / `isinstance(d,datetime)` / `isinstance(d,str)` / else `raise TypeError` | paper_trades dates | STRING | **CORRECT -- BEST-IN-REPO PATTERN** | the `and not isinstance(d, datetime)` clause is the datetime-is-a-date-subclass fix; **and it RAISES on an unhandled type instead of returning None**, so a future type change fails LOUD |
| 4 | `backend/services/paper_round_trips.py:36-44` `_parse_ts` | `isinstance(s,datetime)` then `datetime.fromisoformat(str(s)...)` | `paper_trades.created_at` | STRING | **CORRECT** | the isinstance branch is never taken in prod (col is STRING) but the fallthrough `str(s)` path is the real one and is right. Guard is *inert*, not *vacuous-and-load-bearing* -- no safety decision depends on it |
| 5 | `backend/services/reconciliation.py:38-46` `_parse_ts` | identical to #4 -- **THIRD near-duplicate** | `paper_trades.created_at` | STRING | **CORRECT** | same reasoning as #4 |
| 6 | `backend/services/outcome_tracker.py:100-104` | `_ad = report["analysis_date"]; if isinstance(_ad, datetime): ... else: fromisoformat(str(_ad))` | `analysis_results.analysis_date` via `get_recent_reports` (`bigquery_client.py:277-302`) | **TIMESTAMP** (measured) | **CORRECT** | here the isinstance branch IS the live path -- measured `analysis_results.analysis_date` = TIMESTAMP -> `datetime.datetime`. The in-code comment at :96-99 is accurate. |

**No VACUOUS date-guard remains in non-test backend code.** That is the honest headline:
the 82.0 fix appears to have closed the only live instance of the exact pattern. The
value of 82.12 is therefore (a) PROVING that by construction rather than by grep luck,
(b) the LATENT hazards below, and (c) the standing checker so the next one cannot land.

### 5.D KNOWN-GOOD PATTERN -- do NOT report these as defects (anti-cry-wolf)

`backend/backtest/cache.py:231-241` (`preload_prices`):
```
WHERE ticker IN UNNEST(@tickers) AND date >= @start AND date <= @end
... bigquery.ScalarQueryParameter("start", "STRING", start_date)
```
`historical_prices.date` is STRING and the parameter is bound as **STRING**, so this is a
STRING-vs-STRING **lexical** comparison. For zero-padded ISO-8601 `YYYY-MM-DD`, lexical
order == chronological order, so this is **CORRECT BY CONSTRUCTION**. Immediately after,
`cache.py:249` does `df["date"] = pd.to_datetime(df["date"])` -- an explicit coercion.
Both are correct; a naive "STRING column used in a date range" grep would flag them.
The sweep MUST carry this exclusion or it will produce a wall of false positives.

**The exclusion has a sharp boundary the implementer must encode:** lexical == numeric
order holds for zero-padded fixed-width ISO dates ONLY. It does **not** hold for
numbers-in-strings (`'9' > '10'` lexically) and it does not hold for mixed-width or
non-padded dates. So the rule is: *ISO-8601 date STRING compared to ISO-8601 date STRING
= OK; STRING holding a number compared to anything = NOT OK.*

### 5.E THE TWO NUMERIC-NAMED STRING COLUMNS -- both are FALSE POSITIVES (measured)

- `paper_positions.stop_advanced_at_R` (STRING). The name reads "numeric R-multiple",
  but `backend/services/paper_trader.py:747` writes `updates["stop_advanced_at_R"] =
  advance_iso` -- an **ISO timestamp string**. The only read is
  `paper_trader.py:1425` `if pos.get("stop_advanced_at_R"):` -- a pure truthiness test,
  no numeric or temporal semantics applied. **CORRECT.** (Measured: 4 non-test hits, all
  listed.) The `_at_R` suffix means "at the R threshold", not "value in R".
- `pyfinagent_data.calendar_events.confidence` (STRING). Written by
  `backend/econ_calendar/watcher.py` / read by `backend/services/pead_signal.py`; the
  values are categorical labels, not numbers. Needs a 1-line confirm during GENERATE, but
  the name-heuristic hit is almost certainly a false positive.

### 5.F LATENT HAZARD 1 -- `_STRING_DATE_TIMESTAMP_COLS` is a hand-written 2-entry set

`backend/services/cycle_health.py:436-439`:
```python
_STRING_DATE_TIMESTAMP_COLS = {
    ("paper_trades", "created_at"),                  # STRING (RFC3339)
    ("paper_portfolio_snapshots", "snapshot_date"),  # STRING (YYYY-MM-DD)
}
```
Consumed at `cycle_health.py:464` to choose `SAFE.TIMESTAMP(MAX(col))` vs bare `MAX(col)`.

**Measured coverage.** `grep -rn "_bq_max_event_age(" --include="*.py" .` -> the 6
production call sites are `cycle_health.py:502,503,505,506,507,508`:

| call site | table.column | live declared type | in the set? | verdict |
|---|---|---|---|---|
| :502 | paper_trades.created_at | STRING | yes | correct |
| :503 | paper_portfolio_snapshots.snapshot_date | STRING | yes | correct |
| :505 | historical_prices.ingested_at | TIMESTAMP | no | correct |
| :506 | historical_fundamentals.ingested_at | TIMESTAMP | no | correct |
| :507 | historical_macro.ingested_at | TIMESTAMP | no | correct |
| :508 | signals_log.recorded_at | TIMESTAMP | no | correct |

**Verdict: CORRECT TODAY, 6/6.** I am NOT reporting this as a defect. It is a
**latent-drift hazard**: the set is hand-maintained against a live schema with 16 STRING
date columns, and the failure mode is already documented in-file (`cycle_health.py:451-460`:
"`SAFE.TIMESTAMP(MAX(...))` returns BQ 400 BadRequest ... the broad except swallowed it,
returning None, and the band stayed 'unknown' indefinitely"). The `except Exception ->
logger.warning -> return None` at `cycle_health.py:479-486` means a future drift degrades
SILENTLY to "unknown", not loudly. This is the natural first CONSUMER of the derived
scope: replace the literal set with a lookup against the derived inventory.

### 5.G LATENT HAZARD 2 -- a drifted type tag in a live test (MEASURED, and BENIGN)

`backend/tests/test_phase_23_2_11_bq_table_freshness.py:38` declares
`("financial_reports", "analysis_results", "analysis_date", "STRING", 24, "us-central1")`.
**Measured live: `analysis_results.analysis_date` is declared `TIMESTAMP`, not STRING.**
The tag is wrong. The test branches on it at :112 and emits
`SELECT MAX(TIMESTAMP(analysis_date)) ...` for the "STRING" branch.

I RAN BOTH BRANCHES against the live table rather than asserting the outcome:
```
STRING-branch  SQL: SELECT MAX(TIMESTAMP(analysis_date)) AS max_ts, COUNT(*) AS n FROM ...
  RESULT: Row((datetime.datetime(2026, 8, 4, 19, 28, 3, 716651, tzinfo=utc), 509))
NATIVE-branch  SQL: SELECT MAX(analysis_date) ...
  RESULT: Row((datetime.datetime(2026, 8, 4, 19, 28, 3, 716651, tzinfo=utc), 509))
```
**Identical.** `TIMESTAMP(timestamp_expr)` is accepted by BigQuery (unlike
`SAFE.TIMESTAMP(timestamp_expr)`, which per `cycle_health.py:454-457` raises 400). So the
drifted tag is currently HARMLESS. Report it as a documentation/drift defect, not a
behaviour defect -- overstating it would be crying wolf. Two aggravating notes: the whole
module is `skipif PYFINAGENT_LIVE_TESTS != "1"` (`:85-92`), so it does not run in CI; and
`except Exception -> pytest.skip` at `:135-136` means that if the drift ever DID produce a
query error, the test would **skip, not fail** -- a guard that cannot fail.

### 5.H THE 82.0 FIX -- completeness review (question D)

`preload_macro` (`cache.py:333-435`) handles `str` / `date` / `datetime` / unparseable
(`:379-391`), `None` (`_coerce_date` returns None -> `:400-402` counts `unparsed`), and a
**BQ `Row` rather than a dict** (`:397` `row = r if isinstance(r, dict) else dict(r)`).
Empty string `""` -> `_date.fromisoformat("")` raises `ValueError` -> caught at `:389`
-> `None`. **All five production shapes are covered.** It also fails CLOSED
(`:410-418` returns 0 when nothing parses) -- which is the correct direction and the
opposite of the original defect.

Second copies: `cache.py:136-147` is a byte-identical `_coerce` and is already correct.
`preload_prices` (`:208`) and `preload_fundamentals` (`:265`) have **NO staleness gate at
all** -- `grep -n "stale\|MAX_AGE\|age_days\|today" backend/backtest/cache.py` returns
hits only inside the macro block. That is *missing guard*, a different (and arguably
larger) defect class than *vacuous guard*; it belongs in its own queued step, not in
82.12's scope. Flag it, do not silently absorb it.

### 5.I EXISTING TEST PRECEDENT (question E) -- the fixture pattern already exists

`backend/tests/test_phase_82_0_macro_ingestion.py`:
- `:186` docstring: "(BQ schema ('date','STRING','REQUIRED')); live rows come back as ..."
- `:193` `return {"series_id": series_id, "value": 1.0, "date": d.isoformat()}`
  -- the fixture builder emits the **PRODUCTION type (`str`)**, not a `date`.
- `:206` `assert all(isinstance(r["date"], str) for r in rows), (...)`
  -- **a precondition assertion on the fixture itself.** This is exactly criterion 4's
  "a fixture that cannot represent the production failure cannot pass". COPY THIS.
- `:282` `rows = [{"series_id": "GDP", "value": 1.0, "date": "not-a-date"}]` -- the
  unparseable-string case.

There is **no repo-wide helper that returns realistic BQ `Row` objects**
(`grep -rln "SchemaField" backend/ scripts/` returns only migrations +
`backend/backtest/learning_schema.py`, `learning_logger.py`, `paper_metrics_v2.py`).
Every fixture is a hand-built dict. That matters: `cache.py:397` explicitly handles a
non-dict Row, but **no test exercises that branch with a real `Row`**.

### 5.K *** NEW P1 DEFECT FOUND BY THE SWEEP (round 3) -- MEASURED, REPRODUCIBLE ***

`backend/slack_bot/jobs/_production_fns.py:216-235` (`make_ledger_fetch_fn`, the
`nightly_outcome_rebuild` ledger fetch) issues:

```sql
SELECT trade_id, ticker, action, price, quantity, timestamp,
       SAFE_CAST(realized_pnl AS FLOAT64) AS pnl
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND realized_pnl IS NOT NULL
LIMIT 1000
```

**Measured live `paper_trades` schema (18 columns):**
```
action STRING, analysis_id STRING, capture_ratio FLOAT, created_at STRING,
holding_days INTEGER, mae_pct FLOAT, mfe_pct FLOAT, price FLOAT, quantity FLOAT,
realized_pnl_pct FLOAT, reason STRING, risk_judge_decision STRING,
round_trip_id STRING, signals STRING, ticker STRING, total_value FLOAT,
trade_id STRING, transaction_cost FLOAT
```
`'timestamp' present? **False**`   `'realized_pnl' present? **False**`

**I ran the production SQL verbatim against the live table:**
```
PRODUCTION SQL FAILS: BadRequest
  400 Unrecognized name: timestamp at [5:27]; reason: invalidQuery
```

The real columns are `created_at` (STRING) and `realized_pnl_pct` (FLOAT). The fetch is
wrapped in `except Exception -> logger.warning("... BQ fetch fail-open: %r") -> return []`
(`_production_fns.py:231-233`), so **`nightly_outcome_rebuild` has been silently
rebuilding outcomes from ZERO trades**, every night, with only a WARNING line.

This is the SAME defect family as the 82.0 macro bug -- a wrong assumption about a BQ
column, swallowed by a broad `except`, presenting as "working" -- differing only in that
the assumption is about a column's NAME rather than its TYPE. Two extra ironies worth
recording: (a) `SAFE_CAST(realized_pnl AS FLOAT64)` is a *defensive* cast written for a
STRING-typed pnl column that does not exist -- and the column that does exist
(`realized_pnl_pct`) is already FLOAT, so no cast is needed; (b) the fail-open comment
literally says "fail-open", i.e. the silence is intentional at the mechanism level and
accidental at the outcome level.

**This should be its own queued follow-up step** (criterion 3 allows fix-or-queue). It is
out of the literal wording of 82.12's scope (which says "declared STRING but consumed as
a date/number"), and it argues for WIDENING the checker: see 6.C.

### 5.L MEASURED: what static analysis actually catches here (external scope (b))

I did not want to assert this, so I measured it. Probe file with four cases
(`pyright_probe.py` in the scratchpad): (A) `row: dict` untyped -> `row.get("date")` ->
`isinstance(v, date)`; (B) `v: Any`; (C) `v: str` (provably disjoint from `date`);
(D) `row: dict[str, str]`.

- **pyright 1.1.411, default settings:** `errorCount: 0, warningCount: 0`. **Catches
  nothing**, including case C.
- **pyright 1.1.411, `typeCheckingMode: strict` + `reportUnnecessaryIsInstance: error`:**
  5 diagnostics, ALL of them about the untyped dict --
  `reportUnknownParameterType` (line 4), `reportMissingTypeArgument` (line 4),
  `reportUnknownVariableType` (line 5), `reportUnknownMemberType` (line 5),
  `reportUnusedImport` (line 1). **Zero diagnostics on any of the isinstance lines,
  including case C.**

**Conclusion (this is the answer to "what a type-checker WOULD have caught"):** a type
checker would NOT have flagged the vacuous `isinstance`. What it flags is the *upstream
enabler* -- that the row is untyped, so the value is `Unknown`. Static analysis cannot
solve this class, because the fact that makes the predicate always-false lives in the
**BigQuery schema**, not in the Python type system. This is the AdaCore point restated:
"simple cases of unreachable code can be detected by static analysis (typically if a
condition in an if statement can be determined to be always true or false)" -- but here
it *cannot* be so determined without an external schema oracle. **82.12's checker IS that
oracle.** That is the strongest argument for building it, and it should be stated in the
contract.

Repo toolchain note (**correcting my own earlier draft of 5.J**): the repo DOES carry
`pyrightconfig.json` at root and a `.ruff_cache/` directory; my first `ls` used a pattern
list that missed them. `pyrightconfig.json` has **no `typeCheckingMode` key** (so pyright
runs at its default, which measured 0 diagnostics above) and pins
`"venv": ".venv312", "pythonVersion": "3.12"` while the project actually runs Python 3.14
in `.venv` -- itself stale config, though out of 82.12 scope.

### 5.J EXISTING CHECKERS (question F) -- there is NO duplicate to worry about

`ls scripts/housekeeping/` -> `audit_memory.py`, `backfill_handoff_archive.py`,
`quarantine_phantom_archives.py`, `restore_from_quarantine.py`,
`verify_handoff_layout.py`. None touches BQ types. `ls .pre-commit-config.yaml
pyproject.toml setup.cfg .flake8 ruff.toml` at repo root returned **nothing** -- there is
no lint/pre-commit configuration to hang a rule off. So 82.12's checker is greenfield;
build it as a pytest test + a `scripts/housekeeping/` script, matching existing idiom.

---

## 6. Recommendation for the contract

### 6.A THE HEADLINE THE CONTRACT MUST OPEN WITH

**The literal sweep 82.12 describes yields ZERO new defects.** Measured: 534 non-test
`isinstance` calls, 8 date-type ones, 5 live, and all 5 are CORRECT (5.C). Both
numeric-named STRING columns are false positives (5.E). The existing
`_STRING_DATE_TIMESTAMP_COLS` workaround is 6/6 correct (5.F). The 82.0 fix is complete
across all five production shapes (5.H).

A step written as "go find more of the same" will therefore either (a) honestly report
nothing and look like a wasted cycle, or (b) manufacture false positives to look
productive. **Neither is acceptable.** Re-frame the step as:

> Build the **schema oracle** the codebase has never had, prove the current surface is
> clean against it, and wire it as a standing check so the next instance cannot land.

That framing makes "zero remaining vacuous guards" the SUCCESS condition (proved by
construction), not an embarrassment -- and it is honest.

### 6.B THE DESIGN -- two-sided derivation, no name heuristic in the gate path

Criterion 1 says the scope must be "DERIVED FROM LIVE TABLE SCHEMAS rather than from a
hand-written list". **A regex over column NAMES is a hand-written list** (of name tokens)
and satisfies the criterion in letter while violating it in spirit -- and it demonstrably
under-covers (5.A recall audit: it would miss `calendar_events.window`,
`analysis_results.overall_reliability`, `strategy_decisions.decay_attribution`). Build it
two-sided instead:

1. **Schema side (the oracle).** `client.list_tables()` + `get_table().schema` over the 4
   datasets -> `{table: {column: (field_type, mode)}}`. Measured today: **33 tables, 477
   columns, 189 STRING / 182 FLOAT / 48 INTEGER / 31 TIMESTAMP / 13 DATE / 6 BOOLEAN /
   5 JSON / 3 RECORD.** Cache to a checked-in JSON snapshot so CI can run without ADC,
   and add a separate live-refresh job that diffs snapshot-vs-live (that diff IS the
   schema-drift detector, per S4's preflight-check pattern).
2. **Consumer side.** Every read site, no name filter: SQL literals naming a known table,
   and Python guards over a BQ-read value.
3. **Join.** Flag only where a consumer applies date/number semantics to a column the
   oracle says is STRING. **The name heuristic, if used at all, is a REPORTING aid, never
   the gate.**
4. **ASSERT NON-EMPTY at every stage** -- criterion 1 requires it for the scope, and it
   should also cover the instrument. I hit this for real: my first scanner returned
   "0 unknown identifiers", which looked like a clean bill of health but was a relative-
   path bug (cwd resets between calls) plus a non-greedy regex that captured `sunny` from
   `` `sunny-might-477607-p8.financial_reports.paper_trades` `` instead of the table
   name. **A checker that scans nothing reports the same thing as a codebase with no
   defects.** Assert `files_scanned > 0`, `tables_resolved > 0`, `columns_in_oracle > 0`.

### 6.C USE BIGQUERY `dry_run` AS THE INSTRUMENT, NOT A REGEX (measured)

Do not hand-roll SQL parsing. `QueryJobConfig(dry_run=True)` makes BigQuery itself parse
and type-check the query against the live schema, at **$0** (dry runs are not billed). I
verified all five cases live:

```
FAIL | nonexistent column (_production_fns.py:218)
       -> 400 Unrecognized name: timestamp at [1:18]
OK   | fixed equivalent using real columns (created_at, realized_pnl_pct)
FAIL | STRING date >= CURRENT_DATE()
       -> 400 No matching signature for operator >= for argument types: STRING, DATE
OK   | STRING date >= '2026-01-01'          <- the known-good lexical pattern
FAIL | SAFE.TIMESTAMP(MAX(ingested_at)) on a native TIMESTAMP column
       -> 400 SAFE with function timestamp is not supported.
```
Case 3 is the empirical confirmation of S1 (STRING has an empty coerce-to set) and case 5
independently confirms the `cycle_health.py:451-460` comment is ACCURATE -- do not "fix"
that comment. Extract each SQL literal, substitute placeholder params, dry-run it, assert
no `BadRequest`. That single check would have caught 5.K on the day it shipped.

### 6.D CRITERION 4 -- how to make the fixture provably able to fail

The pattern already exists in-repo; copy it rather than inventing one:
`backend/tests/test_phase_82_0_macro_ingestion.py:193` builds rows with
`"date": d.isoformat()` (production type `str`) and `:206` asserts
`all(isinstance(r["date"], str) for r in rows)` -- **a precondition assertion on the
fixture itself.** Generalise to a three-part test per fixed guard:

1. **Precondition:** assert the fixture emits the type the ORACLE declares (not a
   hardcoded `"STRING"` -- read it from the oracle, so schema drift breaks the test).
2. **Positive:** guard fires on bad input (stale date / unparseable / None / `""`).
3. **Negative:** guard does not fire on good input.

Then **mutate**: delete the guard body and assert the suite goes red. Per S6, guard
clauses habitually ship without any test that requires them; per S7, a surviving mutant
is the proof that the suite cannot tell the guard from its absence. Also add
`--cov-branch` for the touched modules -- per S5, a vacuous guard appears as a **partial
branch** (the `if` line covered, one destination never visited), which is the cheapest
possible standing detector and needs no new dependency.

Two gaps to close while here: there is **no helper returning realistic BQ `Row` objects**
in the repo (every fixture is a hand-built dict), yet `cache.py:397` explicitly handles a
non-dict `Row` -- that branch is untested. Build a small `make_bq_row()` helper from
`SchemaField` + `bigquery.Row` so the Row path is exercised.

### 6.E TRAPS -- things that will produce a wrong or cry-wolf result

1. **Do NOT flag lexical ISO-date comparisons.** `cache.py:231-241` binds
   `ScalarQueryParameter("start", "STRING", ...)` against STRING `date` -- correct by
   construction for zero-padded `YYYY-MM-DD`. Encode the boundary: ISO-date-STRING vs
   ISO-date-STRING = OK; STRING-holding-a-number vs anything = NOT OK (`'9' > '10'`).
2. **`datetime` is a subclass of `date`.** `isinstance(ts, date)` is TRUE for a TIMESTAMP
   column. The correct "plain date only" test is
   `isinstance(d, date) and not isinstance(d, datetime)` (`wash_sale_filter.py:30`).
   A sweep that ignores this will mis-classify both directions.
3. **Distinguish INERT from VACUOUS.** `paper_round_trips.py:39` and
   `reconciliation.py:41` have isinstance branches that never execute in production --
   but they are *fallthrough optimisations*, not safety decisions, and the `str` path
   below them is correct. Only call it VACUOUS when a **safety decision** depends on the
   dead branch. Report inert branches as a separate, lower severity.
4. **`_STRING_DATE_TIMESTAMP_COLS` is correct today (6/6).** Do not write it up as a live
   defect. It is a latent-drift hazard whose failure mode is silent
   (`cycle_health.py:479-486` swallows to `return None` -> band "unknown"). Convert it to
   an oracle lookup; that is a refactor, not a bug fix.
5. **The drifted test tag is benign.** `test_phase_23_2_11...py:38` mislabels
   `analysis_results.analysis_date` as STRING when it is TIMESTAMP, but I ran both
   branches and they return byte-identical results. Report as drift, not breakage.
   (Separately: `except Exception -> pytest.skip` at `:135-136` means that test cannot
   fail on a query error -- that IS a guard-that-cannot-fire, in the test layer.)
6. **`stop_advanced_at_R` is a TIMESTAMP string, not an R-multiple** (`paper_trader.py:747`
   writes `advance_iso`; `:1425` only truthiness-tests it). Name-based scoping mis-reads it.
7. **Freeze the tree during EVALUATE.** Standing lesson; the P1 in 5.K is tempting to fix
   inline -- queue it instead.

### 6.F WHERE THE STEP DESCRIPTION IS WRONG OR STALE (high-value, say it plainly)

1. **"other call sites may carry the same wrong assumption" -- MEASURED FALSE for the
   date/isinstance form.** Zero remaining. The step's expected yield is empty.
2. **The scope wording is TOO NARROW and would EXCLUDE the only real defect found.**
   Criterion 1 scopes to "BQ columns declared STRING that are consumed as dates/numbers".
   The P1 in 5.K -- `_production_fns.py:218-227` selecting `timestamp` and `realized_pnl`,
   columns that **do not exist** on `paper_trades` -- is the same *root cause* (an
   unverified assumption about a BQ column, silently swallowed) but a different *symptom*
   (wrong NAME, not wrong TYPE). As written, the criteria would force the implementer to
   drop the best finding of the sweep. **Recommend the contract widen the hypothesis to
   "unverified assumptions about BQ columns (name OR type)" while leaving the immutable
   criteria untouched** -- criterion 3 already permits "fixed or has its own queued
   follow-up step", so file 5.K as its own step.
3. **`_STRING_DATE_TIMESTAMP_COLS` is described as "direct evidence that other call sites
   may carry the same wrong assumption".** It is evidence that the schema is mixed; it is
   NOT evidence of existing defects. Measured 6/6 correct.
4. **The "numbers" half of the scope is empty.** Both numeric-named STRING columns are
   false positives (5.E). Do not let an implementer pad the report with them.
5. **"derived from live table schemas rather than from a hand-written list" is satisfiable
   by a name-regex, which is itself a hand-written list.** See 6.B. The contract should
   spell out the two-sided derivation so the letter and the spirit agree.
6. **Out-of-scope neighbours found while sweeping -- queue, do not absorb:**
   `preload_prices`/`preload_fundamentals` (`cache.py:208`, `:265`) have **no staleness
   gate at all** (missing guard, not vacuous guard); `pyrightconfig.json` pins
   `.venv312`/Python 3.12 while the project runs 3.14 in `.venv`, and sets no
   `typeCheckingMode` (measured: 0 diagnostics at default).

---

## 7. JSON gate envelope

Coverage bookkeeping (audit-class, K=2):

| Round | Focus | New read-in-full findings |
|---|---|---|
| R1 | live schema derivation + isinstance sweep + cache.py/cycle_health | many |
| R2 | SQL param type bindings, lint config, freshness probes | new (pyrightconfig, TIMESTAMP params, drifted tag) |
| R3 | pyright measurement + `_production_fns` reproduction | **new P1 (5.K)** |
| R4 | repo-wide SQL-identifier vs live-schema diff | **0 new defects** (methodology lesson only) -> DRY 1 |
| R5 | BigQuery `dry_run` validation of 5 predicted cases | **0 new defects** (all confirmations) -> DRY 2 |
| R6 | `type() is`, hasattr duck-typing, direct `.isoformat()`, numeric casts on row values | **0 new defects** -> DRY 3 |

`dry_rounds = 3 >= K_required = 2` -> `coverage.dry = true`.

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 32,
  "urls_collected": 39,
  "recency_scan_performed": true,
  "internal_files_inspected": 20,
  "coverage": {
    "audit_class": true,
    "rounds": 6,
    "dry_rounds": 3,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Derived the BQ scope from LIVE schemas (33 tables, 477 columns; 189 STRING, 13 DATE, 31 TIMESTAMP) and MEASURED the BQ->Python mapping on the live table: STRING->str, DATE->datetime.date, TIMESTAMP->datetime.datetime, FLOAT->float. 17 STRING columns have date/number-suggestive names; all 16 date ones sit in financial_reports and both numeric-named ones are false positives. Classified all 5 live isinstance date-guards: ALL CORRECT -- the 82.0 fix closed the only vacuous instance, and cycle_health's hand-written _STRING_DATE_TIMESTAMP_COLS is 6/6 correct today. The literal sweep yields zero new defects. But the sweep DID find a new P1 of the same root cause: backend/slack_bot/jobs/_production_fns.py:218-227 selects paper_trades.timestamp and .realized_pnl, which do not exist (400 Unrecognized name, reproduced live), swallowed by a fail-open except -- nightly_outcome_rebuild has been running on zero trades. Measured that pyright catches none of this even in strict mode; the decisive fact lives in the BQ schema, so the checker IS the oracle. Recommend BigQuery dry_run ($0) as the instrument, two-sided derivation with non-empty assertions, and widening the hypothesis to name-OR-type assumptions.",
  "brief_path": "handoff/current/research_brief_82.12.md",
  "gate_passed": true
}
```
