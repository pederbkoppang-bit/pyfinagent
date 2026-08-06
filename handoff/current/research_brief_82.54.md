# Research Brief -- step 82.54 (P1): second live phantom-column defect (`cost_budget_api.py` -> `llm_call_log`)

TIER: moderate | AUDIT_CLASS: true (loop-until-dry, K=2) | Researcher: Layer-3 (Workflow rail)
Started: 2026-08-06. Status: **COMPLETE -- gate_passed: true** (8 sources read in full;
audit-class loop ran 8 rounds, 2 consecutive dry).

**Headline:** the defect is real and $0-reproducible, but TWO of the step description's
premises are REFUTED by measurement -- nothing reads `llm_tokens_today` (so no operator was
ever shown a false `$0`), and the phase-75.5.1 `$25/day` metric is a different function in a
different file. The fix is NOT a rename: the projection choice swings the number by 23x-26x.
A THIRD live defect was found in the alerting rail (`spend.py:115`) and is queued.

## Scope

Q1. Structural enumeration of every column identifier `backend/api/cost_budget_api.py` selects, from every table; check each against live schema; recall-test the derivation (aliases excluded).
Q2. Correct projection for `pyfinagent_data.llm_call_log` (15 cols) -- measure, do not assume a rename.
Q3. Consumers of `llm_tokens_today`; does anything coerce None -> 0 (a materially worse failure)? Is this the phase-75.5.1 "$25/day LLM metric DARK" tile?
Q4. Is `DATE(ts) = CURRENT_DATE()` correct -- what TYPE is `ts`?
Q5. AUDIT-CLASS: every other file with f-string BQ table refs + named columns; check each against live schema; loop until dry.
Q6. Guard design -- what makes each of criteria 1/2/4 vacuous.

## MEASURED FACTS (established before any external reading; all re-derived today 2026-08-06)

### M1. The f-string blindness -- PROVEN first-hand, not inferred

`backend/db/schema_oracle.py:199-208` (`extract_sql_literals`, the `ast.JoinedStr` branch)
reassembles an f-string from its **Constant parts only**:

```python
text = "".join(
    v.value for v in node.values
    if isinstance(v, ast.Constant) and isinstance(v.value, str)
)
```

Ran it live against `backend/api/cost_budget_api.py`:

```
extract_sql_literals -> 0 literals
--- JoinedStr at line 82 reassembled constant-parts ---
'\n  SELECT\n    COALESCE(SUM(input_tokens) + SUM(output_tokens), 0) AS tokens,\n
   COUNT(*) AS calls\n  FROM `.pyfinagent_data.llm_call_log`\n  WHERE DATE(ts) = CURRENT_DATE()\n'
FQ match: None
tables_in_sql: []
```

The interpolated `{project}` collapses to nothing, leaving `` `.pyfinagent_data.llm_call_log` ``.
`_FQ_TABLE_RE` at `schema_oracle.py:63` is `` `([A-Za-z0-9_\-]+)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)` ``
-- the project group requires **one or more** chars, so the empty project kills the match.
`tables_in_sql` (`:211-213`) returns `[]`, so `derive_scope`'s `for table in tables_in_sql(sql)`
loop at `:477` never executes and BOTH the STRING-semantics check AND the
`unknown_columns` name check are skipped. **The file is invisible to the sweep, not clean.**

### M2. The live schema (measured via `client.get_table`, 2026-08-06)

`pyfinagent_data.llm_call_log` -- **5519 rows, 15 columns**, `TimePartitioning(field='ts', type_='DAY')`,
clustered on `['provider','model']`:

| # | column | type | mode |
|---|--------|------|------|
| 1 | `ts` | **TIMESTAMP** | REQUIRED |
| 2 | `provider` | STRING | REQUIRED |
| 3 | `model` | STRING | REQUIRED |
| 4 | `agent` | STRING | NULLABLE |
| 5 | `latency_ms` | FLOAT | REQUIRED |
| 6 | `ttft_ms` | FLOAT | NULLABLE |
| 7 | **`input_tok`** | INTEGER | NULLABLE |
| 8 | **`output_tok`** | INTEGER | NULLABLE |
| 9 | `request_id` | STRING | NULLABLE |
| 10 | `ok` | BOOLEAN | REQUIRED |
| 11 | `ticker` | STRING | NULLABLE |
| 12 | `cycle_id` | STRING | NULLABLE |
| 13 | `session_cost_usd` | FLOAT | NULLABLE |
| 14 | `cache_creation_tok` | INTEGER | NULLABLE |
| 15 | `cache_read_tok` | INTEGER | NULLABLE |

`input_tokens` / `output_tokens` are **absent**. Confirmed present=False.

### M3. The dry run reproduces the defect exactly ($0)

```
CURRENT PRODUCTION SQL dry-run -> 400 POST https://bigquery.googleapis.com/bigquery/v2/
projects/sunny-might-477607-p8/jobs?prettyPrint=false:
Unrecognized name: input_tokens; Did you mean input_tok? at [3:26]
```

BigQuery's own error even names the fix. `schema_oracle.dry_run` (`:550-566`) already exists
and returns `str(exc).splitlines()[0]` for `BadRequest`, `None` for valid -- REUSE it.

### M4. Column data (measured over all 5519 rows)

| metric | value |
|--------|-------|
| `COUNTIF(input_tok IS NULL)` | **0** |
| `COUNTIF(output_tok IS NULL)` | **0** |
| `COUNTIF(cache_creation_tok IS NULL)` | 0 |
| `COUNTIF(cache_read_tok IS NULL)` | 0 |
| `SUM(input_tok)` | 1,887,400 |
| `SUM(output_tok)` | 6,129,864 |
| `SUM(cache_creation_tok)` | **40,495,744** |
| `SUM(cache_read_tok)` | **41,674,518** |
| `MIN(ts)` .. `MAX(ts)` | 2026-05-16T14:39:15Z .. 2026-08-05T19:34:29Z |
| distinct days | 54 |

**No NULLs anywhere in the four token columns.** The cache columns are ~10x the size of
`input_tok + output_tok` combined (82.2M vs 8.0M) -- a projection decision, not a rounding error.

### M5. TODAY HAS ZERO ROWS -- this breaks the obvious criterion-2 fixture

`CURRENT_DATE()` = **2026-08-06**; the most recent day with rows is **2026-08-05**.

| date | calls | input+output tokens |
|------|-------|---------------------|
| 2026-08-05 | 154 | 353,896 |
| 2026-08-04 | 202 | 496,186 |
| 2026-08-03 | 182 | 434,097 |
| 2026-07-31 | 78 | 158,237 |
| 2026-07-26 | **1** | **0** |

2026-07-26 is the counter-example that kills a naive `> 0` assertion pinned to "the last day
with rows": one call, zero tokens.

### M6. Q1 -- the full derived column set for `cost_budget_api.py` (structural, non-empty, recall-tested)

The file contains **exactly one** SQL literal (the f-string at `cost_budget_api.py:82-88`).
Derivation method: AST walk, `ast.JoinedStr` reassembled with `__INTERP__` **substituted** for
each `FormattedValue` (instead of dropped), split on `\bFROM\b`, strip `AS <alias>`, drop
`_SQL_KEYWORDS` minus `_TYPE_NAMES`, drop `_SQL_NOISE`.

| identifier | where | in live schema? | type |
|---|---|---|---|
| `input_tokens` | SELECT list | **False -- PHANTOM** | -- |
| `output_tokens` | SELECT list | **False -- PHANTOM** | -- |
| `ts` | WHERE clause | True | TIMESTAMP |

**Non-emptiness asserted:** the extractor returned 2 literal nodes (the JoinedStr plus its
inner Constant, which itself spans `SELECT ... FROM`); `assert lits` passes. A dedupe by
`lineno` is required or the site double-counts -- note this for the guard.

**Recall test (the aliases that MUST be excluded):** the query invents two names,
`AS tokens` and `AS calls`. The derivation reports
`ALIASES (invented names, must be EXCLUDED): ['calls', 'tokens']` and neither appears in the
identifier set above. This is the 82.39 false-positive class (`pnl` in the `_production_fns`
query) and it is handled. `COALESCE`/`SUM`/`COUNT`/`SELECT`/`FROM`/`WHERE`/`CURRENT_DATE`
are excluded by `_SQL_KEYWORDS`; `DATE` and `TIMESTAMP` stay eligible via `_TYPE_NAMES`
(`schema_oracle.py:529-541`), which is why `ts` survives and is correctly checked.

**No other table is touched by this file.** The `daily_usd`/`monthly_usd` half of the tile
comes from `backend/services/observability/spend.py::fetch_spend` (imported at
`cost_budget_api.py:25`), which queries `INFORMATION_SCHEMA.JOBS_BY_PROJECT` -- a system
view, not a user table, and not in the oracle.

### M7. Q5 AUDIT -- the invisible surface, measured (rounds 1-3)

Scanned 934 `.py` files across `backend/` + `scripts/` + `tests/`. A site is **invisible**
iff its `lineno` is NOT returned by `schema_oracle.extract_sql_literals` yet the SQL
references a table in the oracle.

**25 invisible f-string SQL sites** exist. Round-2's identifier regex flagged 10 as
carrying phantom columns; **round-3 dry-run proves 9 of those 10 are FALSE POSITIVES**:

| false-positive class | examples |
|---|---|
| string literal mistaken for a column | `features.py:105` `'buy'/'sell'/'purchase'/'sale'`; `spend.py:213` `'cc_rail'/'claude'/'code'` (from `'claude-code'`); `sector_calendars.py:194` `'earnings'`; `cleanup_phase_23_*.py` `'default'`; `signal_reliability_test.py:46` `'publish'` |
| CTE name | `paper_trading.py:1182` `combined`/`ranked`/`tickers` |
| INTERVAL unit after interpolation | `sovereign_api.py:153` and `:257` -- `INTERVAL {days} DAY` |

**Exactly ONE real defect in the whole invisible surface** -- the one this step targets:

```
FAIL backend/api/cost_budget_api.py:82
     400 ... Unrecognized name: input_tokens; Did you mean input_tok? at [3:26]
```

The other 4 dry-run failures are **artifacts of my own harness**, not defects, and this is
the single most important methodological warning for the guard design: I materialized every
interpolation as the literal `1`, which produced
`Could not cast literal "1" to type TIMESTAMP` (`features.py:105`, `:154`) and
`WHERE clause should return type BOOL, but returns INT64` (`performance_api.py:72`,
`scripts/away_ops/metered_spend.py:121`). A dry-run guard that fabricates interpolation
values manufactures failures that look exactly like real ones. Discriminate on the error
CLASS (`Unrecognized name:` / `Name ... not found`) or, better, dry-run the **actual
production string** (see Q6).

### M8. The correct sibling already exists -- `fetch_llm_spend`

`backend/services/observability/spend.py:194-250` (`fetch_llm_spend`, phase-75.5.1) queries
the SAME table and gets every column right (`input_tok`, `output_tok`, `cache_creation_tok`,
`cache_read_tok`), and its dry run is **OK**. Its `WHERE` (`spend.py:225-230`) carries three
exclusions that `_fetch_llm_tokens_today` has none of:

```sql
ts >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)
AND ok
AND provider != 'claude-code'
AND (agent IS NULL OR (agent != 'cc_rail' AND agent NOT LIKE 'cc_rail:%'))
```

Per its own module docstring (`spend.py:23-38`), those exclude the **flat-fee Claude-Code
rail** rows -- three shapes, with the bare `agent='cc_rail'` shape DOMINANT (measured
2026-07-25: 2,549 bare rows / ~4.87M tokens vs 7 in the colon shape). It also documents
(`spend.py:39-42`) that `session_cost_usd` is a **per-cycle cumulative GAUGE -- never sum
it** (phase-66.3), and that `llm_call_log` has **no per-call cost column**. That settles the
double-count half of Q2 from inside the codebase.

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://www.cs.ucdavis.edu/~su/publications/icse.pdf | 2026-08-06 | peer-reviewed (ICSE; Gould, Su, Devanbu, UC Davis) | WebFetch -> binary -> `pdfplumber` (51,419 chars) | THE canonical prior art for this exact defect. "the Java type system does little to check for possible errors in the dynamically generated SQL query strings... such defects must be rooted out through careful testing, or (worse) might be found by customers at runtime." And: "because these queries are dynamically generated, errors are only discovered at runtime. It would be desirable to catch these errors statically in the source code." Their analysis is **sound**: "if it does not find any errors, then such errors do not occur at runtime." |
| https://aclanthology.org/2025.emnlp-main.51.pdf | 2026-08-06 | peer-reviewed (EMNLP 2025 Main, pp. 977-991) | WebFetch -> binary -> `pdfplumber` (56,049 chars) | LinkAlign. "Schema linking is a critical bottleneck"; Challenge 2 = "Schema Item Grounding: how to precisely identify the relevant tables and columns within complex and often redundant schemas". SOTA on Spider2.0-Lite is **33.09%** -- i.e. real-world multi-DB schema grounding fails ~2/3 of the time even at SOTA. |
| https://arxiv.org/abs/2606.28387 | 2026-08-06 | preprint (arXiv, submitted 2026-06-23) | WebFetch (abs page, full abstract + results) | Schema-First Retrieval. "Enterprise text-to-SQL systems often fail before SQL is generated: the model receives the wrong schema context." Indexing **catalog metadata** (tables, columns, metrics, relationships, query history) cut BIRD SQL execution errors **15.6% -> 6.2%, a 2.5x reduction**. Direct external validation of the schema-oracle design. |
| https://engineering.fb.com/2022/11/30/data-infrastructure/static-analysis-sql-queries/ | 2026-08-06 | official engineering blog (Meta) | WebFetch (full page) | "we must understand programmatically what happens in SQL queries before they are executed against our query engines -- a task called static analysis." UPM parses SQL into a **semantic tree**, not text, and catches unit errors and cross-system `user_id` JOIN errors so it can "catch the error before the query reaches the query engine." |
| https://docs.cloud.google.com/bigquery/docs/best-practices-costs | 2026-08-06 | official docs (Google Cloud) | `curl` + tag-strip (38,554 chars; page is JS-rendered -- per `feedback_gcloud_docs_fetch`) | "Before running queries, preview them to estimate costs... Perform a dry run for queries." The validator surfaces exactly our error shape: "Not found: Table myProject:myDataset.myTable was not found in location US". `bq --dry_run` returns "Query successfully validated." API path: "submit a query job with `dryRun` set to true in the `JobConfiguration` type." |
| https://docs.python.org/3/library/ast.html | 2026-08-06 | official docs (Python) | WebFetch (full page) | Root cause, from the spec: "`ast.JoinedStr(values)` -- An f-string, comprising a series of `FormattedValue` and `Constant` nodes." The `values` list **interleaves** them, and `FormattedValue.value` "is any expression node". Keeping only `Constant`s therefore drops every interpolation by design. Also: `ast.walk` "Recursively yield all descendant nodes... **in no specified order**" and DOES yield a JoinedStr's child Constants -- which is why a naive extractor double-counts one f-string (measured: 2 hits at line 82 for 1 query). |
| https://prometheus.io/docs/prometheus/latest/querying/functions/ | 2026-08-06 | official docs (Prometheus) | WebFetch (full page) | The null-vs-zero countermeasure. `absent()` "returns... a 1-element vector with the value 1 if the vector passed to it has no elements." The doc's point: an alert on a metric that has vanished returns an empty result set and therefore **cannot fire** -- "silence is indistinguishable from 'everything is fine.'" You must invert the logic to alert on absence. |
| https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-06 | official book (Google SRE, ch. 6) | WebFetch (full chapter) | **[HONEST NEGATIVE]** "Your monitoring system should address two questions: what's broken, and why?" -- symptom-vs-cause is covered, and "it's better to spend much more effort on catching symptoms than causes". But the chapter does **NOT** cover alerts that never fire, silently-broken monitoring, or distinguishing absent data from a zero value. The canonical SRE text does not solve our Q3; Prometheus `absent()` does. Recorded as a gap, not padded. |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.analysis-tools.dev/tag/sql | directory | Index page; enumerates SQL linters (incl. SafeQL ESLint plugin for type-safe SQL) -- useful as evidence of prior art, no primary content |
| https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/StaticAnalysis.pdf | paper (MSR) | Same defect class as the ICSE paper already read in full; marginal added value |
| https://learn.microsoft.com/en-us/sql/ssdt/overview-of-extensibility-for-database-code-analysis-rules | official docs | SQL-Server-specific (`TSqlModel` = a schema oracle for T-SQL); confirms the oracle pattern is standard, not BigQuery-applicable |
| https://github.com/ashleyglee/TSqlRules | community repo | T-SQL rule set; community tier |
| https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-dry-run | official docs | Code sample only; superseded by the best-practices page read in full |
| https://docs.cloud.google.com/bigquery/docs/running-queries | official docs | General run-a-query guide; dry-run content duplicated above |
| https://medium.com/autotrader-engineering/dry-running-our-data-warehouse-using-bigquery-and-dbt-12ccae0209f1 | industry blog | Autotrader dbt+BQ dry-run CI; practitioner tier, corroborates the CI-gate pattern |
| https://conalldalydev.medium.com/why-i-built-the-python-bigquery-validator-package-3f2b32e9bc5b | community blog | `python-bigquery-validator` pkg -- dry-run-as-unit-test; community tier |
| https://github.com/google/adk-python/issues/2949 | issue tracker | "[Feat] Add BQ Dry run check in Bigquery Toolset" -- 2026 evidence the pattern is still being adopted |
| https://glama.ai/mcp/servers/caron14/mcp-bigquery/tools/bq_validate_sql | tool doc | `bq_validate_sql` MCP tool = dry run; community tier |
| https://arxiv.org/pdf/2604.25149 | preprint | "Semantic Layers for Reliable LLM-Powered Data Analytics"; `/html/` 404'd, superseded by 2606.28387 read in full |
| https://arxiv.org/pdf/2604.16493 | preprint | NL2SQLBench modular benchmarking framework (2026) |
| https://arxiv.org/pdf/2605.29670 | preprint | EviLink multi-path schema linking (2026) |
| https://arxiv.org/pdf/2505.18363 | preprint | SchemaGraphSQL pathfinding schema linking (2025) |
| https://arxiv.org/html/2510.09014v1 | preprint | LitE-SQL vector schema linking + execution-guided self-correction |
| https://link.springer.com/chapter/10.1007/978-981-92-3444-8_24 | book chapter | Graph-structured schema linking; paywalled |

## Search-query composition (three-variant discipline, per `.claude/rules/research-gate.md`)

| variant | query | yield |
|---|---|---|
| current-year frontier (2026) | `text-to-SQL schema linking column hallucination benchmark 2026` | arXiv 2606.28387, 2604.25149, 2604.16493, 2605.29670 |
| last-2-year window | (same pass; results span 2025-2026) | LinkAlign EMNLP **2025**, SchemaGraphSQL 2025, LitE-SQL |
| year-less canonical | `static analysis embedded SQL column reference validation against database schema` | Gould/Su/Devanbu ICSE (the founding paper), Meta UPM, MSR framework, SafeQL/TSqlRules |
| year-less canonical | `BigQuery dry run validate query syntax column names cost free` | GCP best-practices-costs, dry-run sample, dbt/Autotrader, bigquery-validator |

## Recency scan (2024-2026)

**Performed.** Result: **3 new findings in the window that COMPLEMENT (do not supersede) the
canonical 2004 static-analysis result.**

1. **arXiv 2606.28387 (2026-06-23), Schema-First Retrieval** -- quantifies the payoff of a
   catalog/oracle: BIRD SQL execution errors **15.6% -> 6.2% (2.5x)** when the schema
   catalog is indexed as a first-class object rather than inferred. External validation
   that `schema_oracle`'s snapshot approach is the current best practice, not local
   invention.
2. **LinkAlign, EMNLP 2025 (peer-reviewed)** -- SOTA on real-world multi-database schema
   grounding is **33.09%**. Naming the right column against a large schema is an open
   research problem; a codebase that names them by hand with no oracle should expect
   defects at a measurable rate, which is exactly what 82.39 + 82.54 found.
3. **google/adk-python issue #2949 (2026), "[Feat] Add BQ Dry run check"** -- the
   dry-run-as-validator pattern is still being newly adopted in 2026 tooling; it is
   current practice, not legacy advice.

**Nothing in the window supersedes Gould/Su/Devanbu.** The 2004 result (sound static
checking of dynamically-generated query strings) remains the correct framing, and its
central complaint -- errors in string-built SQL "are only discovered at runtime" -- is
verbatim what happened here. The 22-year gap is itself the finding: this is a
well-characterised, long-known defect class, not a novel one.

## Key findings

1. **The defect is real, live, and reproducible for $0.** Dry run against production SQL:
   `Unrecognized name: input_tokens; Did you mean input_tok? at [3:26]`. BigQuery names the
   fix. (Measured 2026-08-06; `backend/db/schema_oracle.py:550-566`.)
2. **The blindness is a documented property of `ast.JoinedStr`, not a subtle bug.** Python's
   own spec says `values` interleaves `FormattedValue` and `Constant`; keeping only the
   latter discards every interpolation. (https://docs.python.org/3/library/ast.html)
   Measured consequence at `schema_oracle.py:199-208`: `FROM \`{project}.pyfinagent_data.llm_call_log\``
   -> `` FROM `.pyfinagent_data.llm_call_log` `` -> `_FQ_TABLE_RE` (`:63`) fails -> 0 tables.
3. **This is the canonical dynamic-SQL failure mode, characterised in 2004.** "because these
   queries are dynamically generated, errors are only discovered at runtime"
   (Gould, Su & Devanbu, ICSE, https://www.cs.ucdavis.edu/~su/publications/icse.pdf).
   Our `except Exception` removes even the runtime discovery, which is strictly worse than
   the 2004 baseline.
4. **A schema catalog is the state-of-the-art fix and it measurably works.** Schema-first
   catalog context cut BIRD execution errors 2.5x (arXiv:2606.28387, 2026-06-23). Meta
   reached the same conclusion structurally: parse SQL into a semantic tree so you can
   "catch the error before the query reaches the query engine"
   (https://engineering.fb.com/2022/11/30/data-infrastructure/static-analysis-sql-queries/).
5. **Absence must be alerted on explicitly -- silence never fires.** Prometheus ships
   `absent()` precisely because "an alert cannot fire on an empty result set" and
   "silence is indistinguishable from 'everything is fine'"
   (https://prometheus.io/docs/prometheus/latest/querying/functions/). This is the exact
   shape of `llm_tokens_today = None` for months with nobody paged. The Google SRE book
   chapter 6 does **not** cover this case (checked; honest negative).
6. **Getting column names right is an unsolved problem even at SOTA.** 33.09% on
   Spider2.0-Lite (LinkAlign, EMNLP 2025). Treat every hand-written column list as
   defect-bearing until an oracle says otherwise.

## Answers to the contract's questions

### Q2 -- the CORRECT projection. It is NOT a rename. (MEASURED)

`llm_call_log` carries **four** token columns, no per-row total, and no per-call cost column.
Measured for 2026-08-05 (the most recent day with rows):

| projection | calls | tokens | ratio vs naive |
|---|---|---|---|
| naive rename `SUM(input_tok)+SUM(output_tok)`, all rows | 154 | **353,896** | 1x |
| + cache columns (`+cache_creation_tok+cache_read_tok`) | 154 | **9,159,745** | **26x** |
| METERED only (`spend.py`'s rail + `ok` exclusions), no cache | 30 | **15,409** | **0.04x (23x smaller)** |

Three independent decisions, each worth an order of magnitude:

1. **Cache tokens.** Over the whole table, `cache_creation_tok` (40.5M) + `cache_read_tok`
   (41.7M) = 82.2M vs `input_tok`+`output_tok` = 8.0M. A field literally named
   `llm_tokens_today` that omits 91% of the tokens moved is arguably wrong; a field that
   includes them is not comparable to the `cost_per_llm_call_usd` sibling. **The step must
   pick one and say so in the field's docstring.**
2. **Flat-fee CC-rail rows.** `spend.py:23-38` documents three rail row shapes carrying
   tokens whose real cost is ~$0, with the bare `agent='cc_rail'` shape DOMINANT. Including
   them inflates "tokens today" 23x (154 vs 30 calls). `_fetch_llm_tokens_today` currently
   has **no** such exclusion.
3. **Failed calls.** `spend.py:227` filters `AND ok`; `_fetch_llm_tokens_today` does not.

**Double-counting risk answered:** there is NO per-row total column and NO cost column, so a
sum of `input_tok`+`output_tok` cannot double-count. `session_cost_usd` (FLOAT) **must not be
summed** -- `spend.py:39-42` records it as a per-cycle cumulative GAUGE (phase-66.3).
**NULL risk answered:** `COUNTIF(<col> IS NULL) = 0` for all four token columns across all
5519 rows. The `COALESCE(...,0)` is therefore redundant against NULL columns -- but see Q6,
it is NOT redundant against an empty result and that is what makes criterion 2 vacuous.

### Q3 -- who reads `llm_tokens_today`? **NOBODY.** (This REFRAMES the P1.)

| candidate consumer | measured result |
|---|---|
| frontend tile | **ZERO references.** `rg -ic "costbudget\|cost-budget\|llm_tokens" frontend/src` returns nothing. No `getCostBudgetToday` in `frontend/src/lib/api.ts` (1015 lines), no `CostBudgetToday` in `types.ts`. The phase-15.1 tile described in `harness_log.md:10294-10300` is **gone from the frontend.** |
| Slack digest | none -- no call site |
| budget gate / kill switch | none. The hard-block reads `fetch_spend`/`fetch_llm_spend`, never this endpoint |
| `tests/api/test_observability.py` | the only code reader -- asserts the rollup field exists |
| masterplan immutable verification commands | `curl .../api/cost-budget/status` (operator/CI only) |
| `docs/architecture/api-route-audit-2026-04-26.md` | already records `/api/cost-budget/status`: **"Zero callers anywhere."** |

**So nothing coerces `None` -> `0`.** The step description's worst case ("the operator has
been shown a $0 LLM spend") does **not** occur -- there is no surface showing it. The actual
failure is *quieter*: a `response_model` field that is structurally always `null`, on an
endpoint with no UI, whose only automated check asserts the field is *present*, not
*populated*. **Recommend the contract drop the "materially worse failure" framing** rather
than assert an unmeasured harm.

**Is this the phase-75.5.1 "$25/day LLM metric DARK" tile? NO -- different metric, different
file, different mechanism.** That metric is `fetch_llm_spend` (`spend.py:194-250`) consumed
by the breaker at `backend/agents/llm_client.py:435-440`, gated by
`settings.cost_budget_use_llm_spend_enabled` (**default OFF** -- that is what "DARK" means;
it awaits the operator flip token). `fetch_llm_spend` **dry-runs clean** and uses the right
columns. `llm_tokens_today` is an unrelated, unread observability field.
**This defect is NOT why 75.5.1 looks dark.** Premise refuted.

### Q4 -- `DATE(ts) = CURRENT_DATE()` is CORRECT. No STRING-date trap here.

`ts` is **TIMESTAMP REQUIRED** (not STRING), and the table is `TimePartitioning(field='ts',
type_='DAY')`. Unlike 82.39's `created_at` (STRING, needed `SAFE.TIMESTAMP`) and 82.21's
`report_date` (STRING, lexicographic `MIN()`), there is nothing to fix in the WHERE clause.
Dry-run `total_bytes_processed`, measured:

| query | bytes processed |
|---|---|
| repaired + `WHERE DATE(ts)=CURRENT_DATE()` | **0** (partition pruned) |
| repaired, no WHERE | 88,304 |
| repaired + explicit `ts >= TIMESTAMP(CURRENT_DATE()) AND ts < ...` | **0** |

BigQuery prunes the DAY partition through `DATE()` on the partitioning column, so the
"rewrite it as a half-open TIMESTAMP range for pruning" optimisation is **unnecessary** --
do not add it as scope creep.

### Q6 -- what makes each guard vacuous. Three specific traps, one of them ALREADY REALISED.

**Criterion 1 (a dry run that FAILS against the current projection).**
Vacuous if the test embeds its own copy of the SQL string. The whole defect is that the
production string is unreachable to static tooling; a guard that hand-copies it proves only
that the *copy* is broken, and it will keep passing after someone fixes the copy and not the
source -- or, worse, keep FAILING after the real fix lands. **Non-vacuous design:** obtain
the string the production function actually builds, by injecting a fake `bigquery.Client`
whose `.query(sql, ...)` captures its first argument, calling `_fetch_llm_tokens_today()`,
and dry-running the captured string. `_fetch_llm_tokens_today` imports
`from google.cloud import bigquery` **function-locally** (`cost_budget_api.py:79`), so
`monkeypatch.setattr("google.cloud.bigquery.Client", ...)` is a viable seam. Mutation test:
revert the column names in the SOURCE and the guard must go red.
*Second trap, measured on my own harness:* a dry-run scanner that materialises unknown
interpolations with a dummy value manufactures false failures --
`Could not cast literal "1" to type TIMESTAMP` and
`WHERE clause should return type BOOL, but returns INT64` (4 of my 5 "failures" were this).
If any generalised sweep is built, it must key on the error CLASS (`Unrecognized name:`).

**Criterion 2 (a fixture proving the repaired query returns a NON-NULL total).**
**Vacuous as literally worded, by construction.** The query is an aggregate with no
`GROUP BY`, so BigQuery *always* returns exactly one row; `COALESCE(SUM(...), 0)` then makes
the value *always* non-NULL. `if not rows: return None, None` (`cost_budget_api.py:90-91`) is
therefore **dead code**. Proof by measurement: today (2026-08-06) has **zero rows**, and the
repaired query returns `calls=0, tokens=0` -- non-NULL, and the assertion passes on a day
where the metric measured nothing. The assertion also passes on a table where every token
column is NULL. **Non-vacuous design:** assert a **strictly positive** total over a window
that provably contains rows, and assert the row-count precondition in the same test (the
`a_green_suite_can_be_blind` lesson: a guard must assert its own preconditions took effect).
Do NOT pin the window to "yesterday" -- 2026-07-26 has 1 call and **0** tokens, and 2026-08-06
has none at all. Pin to a fixed historical range and assert `calls > 0` first.

**Criterion 4 (a failed call emits an operator-visible signal).**
**This exact guard already exists one module over and is DEAD -- measured, not inferred.**
`backend/services/observability/spend.py:115-125` calls
`raise_cron_alert_sync(source=..., error_type=..., severity="P2", title=..., detail=...)`
but the signature is
`raise_cron_alert_sync(source, error_type, severity, title, details)`
(`backend/services/observability/alerting.py:253-259`). Live reproduction:

```
TypeError: raise_cron_alert_sync() got an unexpected keyword argument 'detail'. Did you mean 'details'?
```

It is the **only** malformed call site out of 15 audited repo-wide. It is swallowed by
`except Exception ... logger.debug(...)` at `spend.py:126-127`, so the alert that exists
specifically to announce "the cost-budget guard is fail-open" has **never fired**.
Compounding it: `severity="P2"` and `settings.slack_webhook_url` is **empty** (measured), and
`alerting.py:209-222` only reaches `_bot_token_fallback` for `_CRITICAL_SEVERITIES =
{"P0","P1","critical","CRITICAL"}` (`alerting.py:54`) -- so even a well-formed P2 would be
logged and dropped. Two independent reasons the signal cannot reach an operator.
**Consequences for 82.54:** (a) use `details=` and `severity="P1"`; (b) the guard must assert
the alert *was delivered*, not that the code path was entered -- assert on the captured
kwargs of a patched `raise_cron_alert_sync` **and** that the call does not raise; (c)
`spend.py:115` is a **separate live defect out of this step's scope** -- queue it.

## Internal code inventory

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `backend/api/cost_budget_api.py` | :71-96 `_fetch_llm_tokens_today`; SQL f-string :82-88; `except` :94-96 | THE defect. Selects `input_tokens`/`output_tokens`; 400s every call; returns `(None, None)` | **BROKEN (P1, this step)** |
| `backend/api/cost_budget_api.py` | :59-68 `CostBudgetToday`; `llm_tokens_today` :67 | Response model. Field permanently `null` | affected, **no readers** |
| `backend/api/cost_budget_api.py` | :142-145 | `cost_per_call = daily / calls` -- `daily` is **BigQuery bytes-billed dollars**, `calls` is an **LLM** call count. Cross-unit ratio; permanently `None` today because `calls` is `None` | latent nonsense metric; note, do not silently fix |
| `backend/db/schema_oracle.py` | :199-208 `extract_sql_literals` JoinedStr branch; `_FQ_TABLE_RE` :63; `tables_in_sql` :211-213; `derive_scope` :453-526 | The sweep. Blind to every interpolated table ref | **BLIND (82.55 scope)** |
| `backend/db/schema_oracle.py` | :550-566 `dry_run` | $0 validator, already written. REUSE | good |
| `backend/db/schema_oracle.py` | :501-517 alias-stripping + `_SQL_KEYWORDS`/`_TYPE_NAMES` :529-541 | Alias exclusion the recall test depends on | good |
| `backend/services/observability/spend.py` | :194-250 `fetch_llm_spend` | The CORRECT sibling over the same table. Dry-runs clean. Source of truth for the projection | good, **reuse** |
| `backend/services/observability/spend.py` | :115-125 `detail=` + `severity="P2"` | Alert that cannot fire (TypeError + P2 with empty webhook) | **BROKEN -- queue as a new step** |
| `backend/services/observability/alerting.py` | :253-259 signature; :54 `_CRITICAL_SEVERITIES`; :209-222 webhook fallback | Alert rail. P2 + empty webhook = logged and dropped | good (correctly built) |
| `backend/agents/llm_client.py` | :435-440 | The real $25/day breaker; reads `fetch_llm_spend` behind `cost_budget_use_llm_spend_enabled` (default OFF) | unrelated to this defect |
| `backend/tests/test_phase_82_39_outcome_rebuild_query.py`, `..._82_48_outcome_write_schema.py` | -- | Guard idioms to mirror | reuse |
| `tests/api/test_observability.py` | -- | Only code reader of the endpoint; asserts field presence, not population | insufficient |

## Application to pyfinagent

**The fix (3 lines of production change, plus the deliberate projection decision):**

1. `cost_budget_api.py:82-88` -- make the SQL a **plain string constant** with the table
   fully qualified as a literal (the 82.39/82.48 idiom), so `extract_sql_literals` can
   resolve it and the sweep gains permanent coverage of this file. `PROJECT` is already a
   module constant at `schema_oracle.py:53`; hardcoding `sunny-might-477607-p8` in the SQL
   matches how `_production_fns` was repaired. If the `GCP_PROJECT_ID` override must be
   preserved, keep the f-string **and** add the site to the dry-run guard -- but then the
   sweep still cannot see it, which is exactly why 82.55 exists.
2. `input_tokens` -> `input_tok`, `output_tokens` -> `output_tok`.
3. Decide + document the projection (Q2). A field named `llm_tokens_today` that omits 91% of
   the tokens moved is arguably wrong; one that includes cache tokens is not comparable to
   its `cost_per_llm_call_usd` sibling. Cheapest defensible option: keep it a raw **all-rows,
   input+output** count and state in the docstring that it is NOT a spend proxy and NOT
   comparable to `fetch_llm_spend`. Whatever is chosen, assert that exact number in the
   fixture.
4. `except Exception` at `:94-96` -- add `raise_cron_alert_sync(..., severity="P1",
   details=...)` imported **function-locally** from
   `backend.services.observability.alerting` (so the patch target is the alerting module),
   inside its own `try/except` so alerting can never break the endpoint.

**Do NOT:** rewrite the WHERE clause (Q4: already correct, already pruning to 0 bytes);
"fix" `cost_per_llm_call_usd` (cross-unit, out of scope -- queue it); or claim an operator
was shown a false `$0` (Q3: there is no surface).

**Queue as separate steps (per `feedback_queue_discovered_defects_in_masterplan`):**

| # | defect | anchor | evidence |
|---|---|---|---|
| A | `raise_cron_alert_sync(detail=...)` -- wrong kwarg, `TypeError`, swallowed by `except -> logger.debug`. Only bad site of 15. The "guard is fail-open" alert has never fired. Also `severity="P2"` with an empty `slack_webhook_url` = logged and dropped | `backend/services/observability/spend.py:115-127`; sig at `alerting.py:253-259`; `_CRITICAL_SEVERITIES` at `alerting.py:54` | live `TypeError: ... unexpected keyword argument 'detail'. Did you mean 'details'?` |
| B | `cost_per_llm_call_usd` divides **BigQuery bytes-billed USD** by an **LLM call count** | `cost_budget_api.py:143-145` | cross-unit by inspection; currently masked because `calls` is always `None` |
| C | the phase-15.1 cost-budget **frontend tile no longer exists**; the endpoint has zero UI readers while `/status` + `/today` stay live and public | `frontend/src` (0 hits); `docs/architecture/api-route-audit-2026-04-26.md` "Zero callers anywhere" | measured `rg -ic` = empty |
| D | (already scoped as 82.55) the sweep's f-string blindness -- 25 invisible sites; only 2 tables / 14 literals visible | `schema_oracle.py:199-208`, `:63` | `derive_scope` -> `tables_resolved: 2` |

## Consensus vs debate (external)

**Consensus (strong, 22 years wide):** dynamically-assembled SQL cannot be checked by the
host language's type system, so name/type errors surface only at runtime -- and the fix is
either (a) statically model the generated string (Gould/Su/Devanbu 2004; Meta UPM 2022) or
(b) hand the string to the engine itself before execution (BigQuery dry run; the
dbt/Autotrader and `python-bigquery-validator` practitioner tier). Nobody argues for
regex-over-source.

**Debate:** *static string analysis* vs *engine-side validation*. The academic line buys
**soundness** -- "if it does not find any errors, then such errors do not occur at runtime" --
at the cost of a whole-program automaton analysis, and only over strings it can reconstruct.
Engine-side dry run buys **zero false positives and zero schema drift** (BigQuery is the
oracle) but only covers strings you can actually produce at test time. **For pyfinagent the
debate resolves in favour of dry run**, and the measurement here is why: my static
reconstruction had a 90% false-positive rate on the phantom-name check (9 of 10); the dry run
had none. The static half's remaining job is *enumeration* -- finding the 25 sites -- not
*adjudication*.

**Contradiction worth recording:** the Google SRE Book (the obvious authority for "why did
nobody notice") does **not** address absent-vs-zero signals at all; the operational answer
comes from Prometheus's `absent()` instead. Do not cite the SRE book for that claim.

## Pitfalls (from literature + measured here)

1. **"The sweep is clean" is not "the code is clean."** `derive_scope` resolves **2** tables
   of 33 in the oracle. A clean report over 6% of the surface is false assurance -- precisely
   the emptiness-assertion failure `schema_oracle.py:38-43` warns about, recurring one level
   up.
2. **A fabricated interpolation manufactures failures.** 4 of my 5 dry-run "failures" were my
   own `1` substitution (`Could not cast literal "1" to type TIMESTAMP`; `WHERE clause should
   return type BOOL, but returns INT64`). Capture the production string; do not synthesise one.
3. **Regex over a SELECT list cannot tell a column from a string literal, a CTE name, or an
   INTERVAL unit.** Measured false positives: `'buy'/'sell'`, `'claude-code'`, `'earnings'`,
   `'default'`, `combined`/`ranked`, `INTERVAL {days} DAY`. Aliases were the *known* trap
   (82.39) and are handled; these five classes were not.
4. **`COALESCE` + a no-GROUP-BY aggregate makes "non-NULL" unfalsifiable.** Always exactly one
   row, always non-NULL, even on an empty partition. Assert positivity plus a row-count
   precondition.
5. **Fail-open `except Exception` converts a loud 400 into silence.** Gould et al.'s baseline
   was that errors "are only discovered at runtime"; swallowing removes even that. Every
   fail-open needs an `absent()`-style positive signal.
6. **An alert seam is not an alert.** `spend.py:115` proves a hand-written alert call can be
   permanently dead in two independent ways at once. Assert delivery, not entry.
7. **Schema linking is hard even at SOTA (33.09%, LinkAlign EMNLP 2025).** Treat every
   hand-written column list as suspect; the oracle is the countermeasure, and catalog-first
   context cut BIRD execution errors 2.5x (arXiv:2606.28387).

## Adaptive-coverage log (audit-class, K=2)

| round | probe | new read-in-full findings |
|---|---|---|
| 1 | f-string sites over oracle tables, `backend/` | **NEW** -- 4 sites |
| 2 | widened resolver (doubly-interpolated refs) + `scripts/` + `tests/`; identifier check | **NEW** -- 25 sites, 10 flagged |
| 3 | BigQuery dry run on all 25 | **NEW** -- 1 real defect; 9 regex FPs; 4 harness artifacts |
| 4 | other blindness classes: `.format()` (0), `%` (0), concat (1 FP), unquoted refs (3, one drill script, clean), non-`.py` carriers (2 FP) | dry |
| 5 | `derive_scope` over the VISIBLE half: `sql_literals=14`, `tables_resolved=2`, `unknown_columns=[]` | dry |
| 6 | consumer trace + alerting-rail audit | **NEW** -- `spend.py:115` `detail=` TypeError |
| 7 | generalised: every kwarg of every call to 12 alerting/observability/oracle functions vs live `inspect.signature` | dry (1 hit = round 6's; no new) |
| 8 | WRITE-side phantom columns for `llm_call_log` (the 82.48 class); writer at `api_call_log.py:279` matches **15/15** | dry (0) |

`rounds=8`, `dry_rounds=2` (consecutive: 7, 8), `K_required=2`,
`new_findings_last_round=0` => **`coverage.dry = true`**.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **8** (2 peer-reviewed, 1 preprint,
      4 official docs, 1 official engineering blog)
- [x] 10+ unique URLs total -- **24** (8 full + 16 snippet-only)
- [x] Recency scan (2024-2026) performed + reported -- 3 in-window findings, none superseding
- [x] Full papers/pages read, not abstracts -- 2 PDFs text-extracted with `pdfplumber`
      (51,419 + 56,049 chars); 1 JS-rendered GCP page via `curl` + tag-strip (38,554 chars)
- [x] file:line anchors for every internal claim
- [x] Three-variant search discipline (current-year / last-2-year / year-less canonical)
- [x] Audit-class loop-until-dry: `dry_rounds=2 >= K_required=2`

Soft checks:
- [x] Internal exploration covered every module named in the prompt, plus `alerting.py`,
      `llm_client.py`, `api_call_log.py`, and the frontend (measured absent)
- [x] Contradictions noted (static analysis vs dry run; SRE-book honest negative; the Q3
      "shown $0" premise and the phase-75.5.1 "same tile" premise both REFUTED)
- [x] Claims cited per-claim
- [ ] **Gap:** the 934-file structural scan is a research-time script in the scratchpad, not a
      checked-in guard. Productionising it is 82.55's job, not this step's.

## JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 16,
  "urls_collected": 24,
  "recency_scan_performed": true,
  "internal_files_inspected": 13,
  "coverage": {
    "audit_class": true,
    "rounds": 8,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "brief_path": "handoff/current/research_brief_82.54.md",
  "gate_passed": true
}
```
