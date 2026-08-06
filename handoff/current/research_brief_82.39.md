# Research Brief — step 82.39 (nightly_outcome_rebuild BQ fetch on phantom columns)

**Tier:** moderate | **Audit-class:** TRUE (loop-until-dry, K=2) | **Started:** 2026-08-06
**Status:** IN PROGRESS (write-first skeleton; sections appended incrementally)

## Scope
- FETCH half (82.39): `make_ledger_fetch_fn` selects `timestamp` / `realized_pnl` which do not exist.
- WRITE half (82.48 sibling): `make_outcome_write_fn` — reported here so Main does not close 82.39
  believing the job works.

## Sections
- [ ] Q1 BigQuery dry run: free? catches unknown column?
- [ ] Q2 30-day window over STRING `created_at`
- [ ] Q3 MEASURED contents of paper_trades
- [ ] Q4 Downstream consequence + duration (measured lower bound only)
- [ ] Q5 derive_scope audit + recall test
- [ ] Q6 fail-open-but-loud shape + pytest vacuity traps
- [ ] Read-in-full source table (>=5)
- [ ] Snippet-only table
- [ ] Recency scan 2024–2026
- [ ] Internal code inventory (file:line)
- [ ] Gate checklist + JSON envelope

_(appended below as work proceeds)_

---
## MEASURED FACTS (run 2026-08-06, live BQ + repo)

### Live schemas (re-measured, matches the step description exactly)
`financial_reports.paper_trades` — **65 rows, 18 columns**. NO `timestamp`, NO `realized_pnl`.
```
trade_id STRING REQUIRED | ticker STRING REQUIRED | action STRING REQUIRED
quantity FLOAT REQUIRED | price FLOAT REQUIRED | total_value FLOAT NULLABLE
transaction_cost FLOAT NULLABLE | reason STRING NULLABLE | analysis_id STRING NULLABLE
risk_judge_decision STRING NULLABLE | created_at STRING REQUIRED | round_trip_id STRING NULLABLE
holding_days INTEGER NULLABLE | realized_pnl_pct FLOAT NULLABLE | mfe_pct FLOAT NULLABLE
mae_pct FLOAT NULLABLE | capture_ratio FLOAT NULLABLE | signals STRING NULLABLE
```
`financial_reports.outcome_tracking` — **0 rows, 9 columns**:
`ticker*`, `analysis_date*` (STRING), `recommendation*`, `price_at_recommendation`,
`current_price`, `return_pct`, `holding_days`, `beat_benchmark` (BOOLEAN),
`evaluated_at` (STRING NULLABLE). (`*` = REQUIRED.)

### The defective query — VERBATIM, `backend/slack_bot/jobs/_production_fns.py:220-227`
```sql
SELECT trade_id, ticker, action, price, quantity, timestamp,
       SAFE_CAST(realized_pnl AS FLOAT64) AS pnl
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND realized_pnl IS NOT NULL
LIMIT 1000
```
Anchors (RE-DERIVED 2026-08-06): `make_ledger_fetch_fn` **:209-234**; SQL literal **:220-227**;
`rows = list(client.query(...))` **:228**; the swallowing `except Exception` + `logger.warning`
+ `return []` at **:230-232**. `make_outcome_write_fn` **:237-261** (write half, step 82.48).

### Q1 — MEASURED dry-run behaviour (live, `schema_oracle.dry_run`, 2026-08-06)
| SQL under test | dry-run verdict |
|---|---|
| CURRENT defective query | **ERROR** `400 ... Unrecognized name: timestamp at [5:27]` |
| Repaired w/ `SAFE.TIMESTAMP(created_at)` | **VALID** |
| Repaired w/ lexical `created_at >= FORMAT_TIMESTAMP(...)` | **VALID** |
| Type error `created_at >= TIMESTAMP_SUB(...)` (STRING vs TIMESTAMP) | **ERROR** `No matching signature for operator >= for argument types: STRING, TIMESTAMP` |
| Missing table | **`google.api_core.exceptions.NotFound` (404) RAISED — NOT returned** |

**Dry-run job stats:** the dry run completes without a job id being billed; `total_bytes_billed`
is reported by the API alongside `total_bytes_processed` (estimate only — no slots consumed).

**INSTRUMENT DEFECT (new, not in the step description):** `schema_oracle.dry_run`
(`backend/db/schema_oracle.py:550-566`) catches **only `BadRequest`**. A missing TABLE raises
`NotFound` which propagates. A criterion-1 test that calls `dry_run()` expecting an error
STRING would crash instead of failing cleanly on the table-not-found class. Contract should
either widen the except to `GoogleAPICallError` or state the limit explicitly.

### Q2 — the 30-day window over a STRING `created_at`
`created_at` is `STRING REQUIRED` holding RFC3339 with microseconds + `+00:00` offset
(measured min `2026-04-26T21:12:28.351207+00:00`). **All 65 rows parse**: `COUNTIF(SAFE.TIMESTAMP(created_at) IS NULL) = 0`.

RECOMMENDED (matches the repo idiom):
```sql
WHERE SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND realized_pnl_pct IS NOT NULL
```
Repo precedent: `backend/services/cycle_health.py:436-439` `_STRING_DATE_TIMESTAMP_COLS`
contains **`("paper_trades", "created_at")` verbatim** — the exact column — and
`:442-468` branches `SAFE.TIMESTAMP(MAX(col))` vs bare `MAX(col)` on membership.

**PORTABILITY TRAP (measured live, not inferred):** `SAFE.TIMESTAMP()` applied to an
already-`TIMESTAMP` column returns `400 SAFE with function timestamp is not supported.`
(probe: `SELECT SAFE.TIMESTAMP(created_at) FROM financial_reports.agent_memories`).
Documented at `cycle_health.py:451-460`. 31 native-TIMESTAMP columns exist in the oracle,
so the wrapper is NOT a blanket-safe idiom — it is per-column.
Lexical alternative also dry-runs VALID but depends on a fixed offset/format in the stored
string; prefer SAFE.TIMESTAMP for parity with the existing repo treatment.

### Q3 — MEASURED contents of paper_trades (criterion 2 fixture material)
| action | rows | rows with `realized_pnl_pct` | created_at min | created_at max |
|---|---|---|---|---|
| BUY | 33 | **0** | 2026-04-26T21:12:28 | 2026-07-31T18:47:37 |
| SELL | 32 | **32** | 2026-05-14T18:02:54 | 2026-07-27T18:05:27 |

Step description's "32/32 SELLs, 0/33 BUYs" **CONFIRMED**.
SELLs by month: 2026-05 = **8**, 2026-06 = **20**, 2026-07 = **4** (all with pnl).

**CRITERION-2 TIME-BOMB (must be in the contract):** the rolling 30-day window returns
**3 rows today (2026-08-06)** and the newest SELL is 2026-07-27 — so after **2026-08-26**
the same repaired query returns **0 rows** and any fixture asserting "returns rows" over
the LIVE rolling window flips red with no code change. Pin a FIXED window. Recommended
anchor: `2026-06-01 <= created_at < 2026-07-01` → **20 SELL rows, 20 non-null pnl** (the
densest month, and stable forever).

### Q4 — how long dead, and what is actually frozen
**Dead since the file's first commit.** `git log -S "SAFE_CAST(realized_pnl AS FLOAT64)"`
returns exactly ONE commit, `2301b977` (2026-05-11, "phase-23.6"), which is also
`git log --reverse` head for the file. The query was **never correct at any point** — this
is not a regression from a schema rename. Lower bound on the outage: **2026-05-11 →
2026-08-06 = 87 days**. Upper bound is the same (the file did not exist before).
NOT MEASURED: whether the scheduler actually fired every night for 87 days —
`heartbeat()` (`backend/slack_bot/job_runtime.py:66-114`) sinks to `logger.info` by
default and `IdempotencyStore` (`:26-39`) is an **in-memory `set()`**, so there is **no
durable receipt on disk or in BQ**. State this as a lower bound; do not claim "87 nightly
runs".

**The "froze the learning loop" causal claim is FALSE — verified again 2026-08-06.**
- `backend/services/outcome_tracker.py` contains **0** references to `outcome_tracking`.
- The real writer is `bigquery_client.py:47` (`self.outcomes_table`) → `:415`
  `insert_rows_json` → read back at `:489`, driven from
  `autonomous_loop.py:3041-3096` (phase-35.1), gated by
  `settings.py:34 paper_learn_loop_enabled` whose default is **`False`** (DARK flag).
- `outcome_tracking` = **0 rows**, but that is over-determined: the nightly job's writer
  is broken AND the learn-loop flag is off. Repairing the fetch unfreezes nothing on its
  own. The contract must not promise a learning-loop restoration.

---
## EXTERNAL RESEARCH

### Read in full (counts toward the gate)
| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|---|---|---|---|---|
| 1 | https://cloud.google.com/bigquery/docs/running-queries | 2026-08-06 | official doc | curl + tag-strip (JS-rendered; WebFetch returns nav only — see memory `gcloud-docs-fetch`) | Verbatim: *"A dry run in BigQuery provides the following information: estimate of charges in on-demand mode; **validation of your query**; approximate bytes processed by your query in capacity mode"* and *"**Dry runs don't use query slots, and you are not charged for performing a dry run.**"* Caveat found: *"A dry run of a federated query that uses an external data source might report a lower bound of 0 bytes"* — not applicable to `paper_trades` (native table). |
| 2 | https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-reference | 2026-08-06 | official doc | curl + tag-strip | SAFE. prefix semantics + **exclusions**: *"If you begin a function with the SAFE. prefix, it will return NULL instead of an error. The SAFE. prefix only prevents errors from the prefixed function itself... The CAST and EXTRACT functions don't support the SAFE. prefix"*, and *"Operators, such as + and =, don't support the SAFE. prefix."* → **a `SAFE.` wrapper cannot rescue the `>=` comparison itself**; it must wrap the coercion. Confirms why `SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(...)` is the correct shape and why `SAFE_CAST` (not `SAFE.CAST`) is the cast spelling. |
| 3 | https://github.com/autotraderuk/dbt-dry-run (README) | 2026-08-06 | industry tooling | curl (raw markdown, full) | Independent corroboration that BQ dry run is used as a pre-execution **validator** in production CI, and that the failure it surfaces is exactly a `BadRequest` with a column-level message: *"Node ... failed with exception: 400 POST ... Column d in USING clause not found on left side of join at [6:88] ... BadRequest : ERROR"*, exit code 1. Motivating sentence: dbt *"doesn't check the validity of SQL queries before it executes your project. This dry runner uses BigQuery's dry run capability to allow you to check that SQL queries are valid before trying to execute them."* |
| 4 | https://users.encs.concordia.ca/~shang/pubs/ICPC_gui.pdf (Padua & Shang, ICPC 2017) | 2026-08-06 | peer-reviewed | curl + pdfplumber (24,075 chars extracted) | Names and quantifies this exact defect class. Five anti-patterns are prevalent *"in median detected in over 20% of the catch blocks or throws statements in the subject systems"*: **Unhandled Exceptions 40.8%, Catch Generic 31.9%, Unreachable Handler 28.0%, Over-catch 24.6%, Destructive Wrapping 22.3%**. Also: *"Generic exceptions ... is an anti-pattern by itself, while some other anti-patterns, e.g., Dummy Handler, may be related to Generic catch blocks."* `_production_fns.py:230-232` is **Catch Generic + Dummy Handler + Return-Null(empty-list)** stacked. |
| 5 | https://docs.python.org/3/library/unittest.mock.html#where-to-patch | 2026-08-06 | official doc | WebFetch | *"You must patch where an object is looked up, which is not necessarily the same place as where it is defined."* If the module does `from a import SomeClass`, patch `b.SomeClass`; if it does `import a`, patch `a.SomeClass`. Directly governs the criterion-3 vacuity trap: `autoresearch_health` imports `raise_cron_alert_sync` **function-locally**, so the name never exists at module scope and `patch("...autoresearch_health.raise_cron_alert_sync")` would `AttributeError` (or silently pass with `create=True`, which the docs warn *"allows tests to pass against APIs that don't actually exist"*). |

| 6 | https://sre.google/sre-book/monitoring-distributed-systems/ (Beyer et al., *Site Reliability Engineering*, ch. 6) | 2026-08-06 | canonical book (year-less query) | WebFetch | Governs criterion 3's severity choice. *"Errors: The rate of requests that fail, either explicitly (e.g., HTTP 500s), **implicitly (for example, an HTTP 200 success response, but coupled with the wrong content)**, or by policy."* — a job returning `rebuilt: 0` after a swallowed 400 is exactly the implicit-error case. *"Every page should be actionable."* *"Email alerts are of very limited value and tend to easily become overrun with noise."* Test: *"Does this rule detect an otherwise undetected condition that is urgent, actionable, and actively or imminently user-visible?"* And the anti-cry-wolf constraint: *"Data collection, aggregation, and alerting configuration that is rarely exercised ... should be up for removal."* |

**Deeper read of source 3 (same URL, "Capabilities and Limitations" section) — the direct answer to Q1:**
> **Things this can catch** — *"anything the BigQuery planner can identify before the query has run"*: 1. typos in SQL keywords; **2. "Typos in columns names: `orders.produts` instead of `orders.products`"**; 3. *"Problems with incompatible data types: Trying to execute `"4" + 4`"*; 4/5. *"Incompatible schema changes to models / to sources: Third party modifies schema of source tables without your knowledge"*; 6. *"Permission errors ... dry run queries need table read permissions just like the real query"*.
> **Things this can't catch** — *"1. Queries that run but do not return intended/correct result. This is checked using tests. 2. NULL values in ARRAY_AGG. 3. Bad query performance..."*

**⇒ Q1 ANSWER.** Yes on both halves, with one caveat.
- **Free:** guaranteed verbatim by Google — *"Dry runs don't use query slots, and you are not charged for performing a dry run."*
- **Catches a bad column name:** YES — proven live on OUR query (`Unrecognized name: timestamp at [5:27]`) and corroborated by dbt-dry-run's item 2.
- **Catches a bad type coercion:** YES — proven live (`No matching signature for operator >= for argument types: STRING, TIMESTAMP`) and corroborated by item 3.
- **Catches a missing table:** YES at the API level (404 `Not found: Table ... was not found`) but our wrapper `schema_oracle.dry_run` **only catches `BadRequest`, so `NotFound` escapes as an exception.**
- **Does NOT catch:** that the query returns the *right rows*. That is precisely why criteria 1 and 2 must remain two separate tests — criterion 1 (dry run) cannot subsume criterion 2 (returns rows), by the tool's own documented limits.

| 7 | https://www.dqlabs.ai/blog/data-pipeline-monitoring-and-anomaly-detection/ | 2026-08-06 | industry (pub. **2026-07-24**) | WebFetch | RECENCY hit. *"A 40% drop in records may not trigger a job failure, but it will produce deeply unreliable analytics."* *"Undetected schema drift is one of the most common causes of silent pipeline breakage."* *"Static threshold alerts generate hundreds of notifications daily, with 60–80% being noise."* → argues for a **row-count / zero-row** control in addition to an exception alert, and warns against an always-on alert. |

### Snippet-only (context; does NOT count toward the gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-dry-run | official sample | Same content as #1's API section |
| https://oneuptime.com/blog/post/2026-02-17-how-to-estimate-bigquery-query-costs-before-running-with-dry-run/view | blog (2026-02) | Cost-estimation angle only; #1 is authoritative on billing |
| https://medium.com/autotrader-engineering/dry-running-our-data-warehouse-using-bigquery-and-dbt-12ccae0209f1 | eng blog | Paywalled/duplicative of #3 |
| https://github.com/n8n-io/n8n/issues/15102 | issue tracker | Vendor-specific node bug, not BQ semantics |
| https://link.springer.com/article/10.1186/s13173-019-0095-5 | peer-reviewed | Longitudinal follow-on to #4; #4 supplies the prevalence numbers |
| https://www.researchgate.net/publication/315783485_Studying_the_Prevalence_of_Exception_Handling_Anti-Patterns | mirror | Same paper as #4 |
| https://blog.anomalyarmor.ai/data-pipeline-monitoring-how-to-stop-silent-failures-before-they-hit-production/ | vendor blog | Lower tier than #7 |
| https://www.databricks.com/blog/data-pipeline-best-practices | vendor blog | General architecture, no zero-row control |
| https://www.digna.ai/data-pipeline-best-practices | vendor blog | Listicle |
| https://solvaria.com/news-insight/why-data-pipelines-fail-etl-and-data-integration-issues/ | vendor blog | Listicle |
| https://cloud.google.com/bigquery/docs/dry-run-queries | official doc | Redirect target of #1's `[bq-dry-run]` link; same guarantee |
| https://itblackbelt.wordpress.com/2006/04/17/exception-handling-antipatterns-by-tim-mccune/ | blog (2006, year-less canonical) | Prior-art for the anti-pattern names; #4 is the peer-reviewed quantification |

### Search-query composition (three-variant discipline)
1. **Current-year frontier:** "silent data pipeline failures observability **2026** fail-open alerting best practice" → surfaced #7 (pub. 2026-07-24).
2. **Last-2-year window:** same query's 2026/2025 results + `oneuptime` 2026-02 + `medium/@jooramos` Jun-2026 (snippet).
3. **Year-less canonical:** "BigQuery dry run query validation not charged unrecognized column" and "exception handling anti-pattern 'log and return null' catch generic empirical study" → surfaced #1, #3, #4 (ICPC 2017) and the 2006 McCune prior-art.

### Recency scan (2024–2026) — REPORTED
Searched the 2024–2026 window on (a) BigQuery dry-run semantics and (b) silent
pipeline-failure alerting. **Result: 1 new finding that COMPLEMENTS (does not supersede)
the canonical sources.** dqlabs 2026-07-24 (#7) adds the *volume/zero-row* control that
neither the 2017 anti-patterns paper nor the SRE book names explicitly: an exception alert
alone still misses the case where the query is valid and simply returns nothing.
**Nothing in the 2024–2026 window contradicts** Google's dry-run billing guarantee, the
`SAFE.` prefix exclusions, or the Padua & Shang anti-pattern taxonomy. The `SAFE.` prefix
exclusion list and the dry-run billing sentence are both live on the current doc pages as
of 2026-08-06.

---
## Q5 — CRITERION-4 AUDIT: `derive_scope` run over the repo

**RUN, not asserted.** `schema_oracle.derive_scope(load_snapshot())`, 2026-08-06:
```
files_scanned      = 296     sql_literals    = 13
tables_resolved    = 1       columns_in_oracle = 477   (33 tables in snapshot)
scope              = []            <-- LENGTH 0
unknown_columns    = [ 2 members ] <-- NON-EMPTY
```
**`unknown_columns` — full list, every member classified:**
| # | table | identifier | file:line | Real defect or FP? | Why |
|---|---|---|---|---|---|
| 1 | financial_reports.paper_trades | `timestamp` | `backend/slack_bot/jobs/_production_fns.py:220` | **REAL** | Column absent from the 18-col live schema; live dry run returns `Unrecognized name: timestamp at [5:27]`. Not an alias (no `AS timestamp`), not a CTE column (no `WITH`), not a struct field. |
| 2 | financial_reports.paper_trades | `realized_pnl` | `backend/slack_bot/jobs/_production_fns.py:220` | **REAL** | Same query; absent; the real column is `realized_pnl_pct`. Appears inside `SAFE_CAST(realized_pnl AS FLOAT64)` — the extractor correctly strips the `AS pnl` **alias** but keeps the inner identifier. |

**Zero false positives.** Both members are the same defect this step exists to fix.
Note `pnl` is correctly EXCLUDED as an alias (`schema_oracle.py:501-508` strips `AS <alias>`),
and `timestamp` is deliberately NOT exempted by the keyword filter
(`_TYPE_NAMES`, `:538-541`) — the module was written with this exact defect as its test case.

### RECALL TEST of the instrument (the loud part)
**Does `derive_scope` SEE this defect? YES** — 2/2 identified with correct file:line. So the
criterion-4 instrument is not blind to the bug it is meant to police.
**But its recall envelope is NARROW, measured 2026-08-06:**
- `extract_sql_literals` requires a **literal backticked fully-qualified** `` `project.dataset.table` `` and only joins the `ast.Constant` parts of an f-string. **20 backend files** (non-test) contain `` FROM `{ `` — an interpolated table ref — and are therefore **invisible** to the sweep (`backend/backtest/cache.py`, `backend/metrics/sortino.py`, `backend/autoresearch/slot_accounting.py`, `backend/backtest/data_ingestion.py`, `backend/agents/skill_optimizer.py`, …).
- **89 files repo-wide** contain `FROM` + a known table token; **50** are under `backend/`; the extractor resolves SQL in **13**.
- `tables_resolved = 1` of 33 tables in the oracle.
- `iter_python_files` (`schema_oracle.py:174-180`) roots at `_REPO_ROOT/"backend"` and excludes `/tests/`, so **`scripts/` is never scanned at all** (12+ files with table refs, incl. `scripts/diagnostics/funnel_report.py`, `scripts/away_ops/metered_spend.py`, `scripts/migrations/*`).

**⇒ The correct wording for the 82.39 close is "0 remaining unknown columns WITHIN THE
MEASURED RECALL ENVELOPE (13 SQL literals / 1 table / 296 backend files)", never
"the repo is clean."** A clean report from this instrument is NOT evidence of a clean repo.

### CRITERION-4 DESIGN TRAP — read this before writing the contract
Criterion 4 says *"its derived scope asserted NON-EMPTY"*. `derive_scope` returns **two**
lists and they behave oppositely:
- `scope` (STRING column + date/number semantics) is **`[]` TODAY, before any fix.** An
  assertion `len(result["scope"]) > 0` is **UNSATISFIABLE** — the step would be
  structurally uncloseable (cf. the 81.0 precedent).
- `unknown_columns` is `[2]` today and becomes **`[]` the moment 82.39's fix lands.** An
  assertion `len(result["unknown_columns"]) > 0` **fails after the fix.**

The only satisfiable-and-meaningful reading is the one the module's own docstring states at
`schema_oracle.py:39-43` — *"EVERY STAGE ASSERTS NON-EMPTY ... A checker that scans nothing
and a codebase with no defects produce the identical output; only an emptiness assertion
tells them apart."* i.e. assert the **INPUT surface** is non-empty:
`files_scanned > 0 AND sql_literals > 0 AND tables_resolved > 0 AND columns_in_oracle > 0`,
then enumerate `unknown_columns` and dispose of each member. Main should pin this
interpretation verbatim in the contract with the reasoning above, because the naive reading
of either list produces a test that cannot be green.

---
## Q6 — CRITERION 3: making the swallowed fail-open audible

**REUSE, do not build a third notifier.** The phase-82.11 pattern shipped 6 days ago:
`backend/services/autoresearch_health.py` imports
`backend.services.observability.alerting.raise_cron_alert_sync`
**function-locally at `:321-325`**, chooses `severity = "P0" if escalated else "P1"` at `:328`,
and calls it at `:340-356`. Signature (`alerting.py:253-259`):
`raise_cron_alert_sync(source, error_type, severity, title, details) -> bool`;
it is *"Always fail-open: never raises out"* (`:266`) — so calling it inside the `except`
cannot itself crash the scheduler. **20+ production call sites already exist.**

**Never `P2`:** `autoresearch_health.py:23-26` records that P2 is *logged and dropped*
(`alerting.py:219-224`) while `P0`/`P1` reach `_bot_token_fallback`. A P2 alert here would
reproduce the very silence the criterion is trying to remove.

**What separates "fail-open but loud" from today's "fail-open and silent":**
| | today (`_production_fns.py:230-232`) | required |
|---|---|---|
| exception | `logger.warning` only | `logger.warning` **+** `raise_cron_alert_sync(..., severity="P1")` |
| return value | `[]` — indistinguishable from "no trades" | `[]` is fine, but the caller must be able to tell the two apart |
| scheduler | not crashed (good) | **keep** the broad `except` — SRE ch. 6 pages must be actionable, not fatal |
| zero-row case | invisible | dqlabs #7: a valid query returning 0 rows is *also* a silent failure — consider a distinct signal |

Recommended shape: keep `except Exception`, add the alert INSIDE it, wrap the alert call in
its own `try/except` (or rely on `raise_cron_alert_sync`'s documented never-raises
contract), and make the failure distinguishable from an empty result (e.g. return a
sentinel/raise-to-caller-with-flag, or emit `n_rows` on the heartbeat).

### pytest vacuity traps for THIS shape (all four have live precedent in-repo)
1. **Patching the function-locally imported symbol.** If the fix imports
   `raise_cron_alert_sync` inside the `except` block, the name is NOT at module scope, so
   `patch("backend.slack_bot.jobs._production_fns.raise_cron_alert_sync")` patches nothing
   and every assertion passes vacuously. Per the Python docs (source #5): *"You must patch
   where an object is looked up."* **Correct target:**
   `backend.services.observability.alerting.raise_cron_alert_sync`. Pin it with a
   `test_wrong_patch_target_does_not_exist`-style guard — the exact countermeasure at
   `backend/tests/test_phase_82_11_autoresearch_failure_paging.py:65` + `:502-506`.
2. **Asserting on `caplog` instead of the emitted payload.** `logger.warning` ALREADY fires
   today, so a caplog assertion passes against the UNFIXED code. Assert on the captured
   **call args** of the alert (`severity`, `source`, `error_type`, and that `details`
   contains the BQ error text) — `assert_called_once_with` / `call_args.kwargs` per source #5.
3. **A guard that passes because the alert always fires.** Needs the negative case: a
   SUCCESSFUL fetch must emit **no** alert. Without it, an unconditional `raise_cron_alert_sync`
   at the top of `_fetch` would pass. (This is the SRE ch. 6 "rarely exercised ... up for
   removal" / dqlabs 60–80%-noise constraint expressed as a test.)
4. **No `autospec=True`.** Without it, a renamed kwarg on `raise_cron_alert_sync` is silently
   recorded and the assertion still passes — 82.11 point 4 uses `autospec=True` on every patch.
5. **Fixture with no precondition assertion.** The failing fixture must assert that the query
   really did 400 (not that the client was merely unreachable), else a "no rows" pass can come
   from a broken fixture rather than from the code under test (82.11 point 3).

| 8 | https://cloud.google.com/bigquery/docs/streaming-data-into-bigquery ("Success HTTP response codes") | 2026-08-06 | official doc | curl + tag-strip | Settles the WRITE half. Verbatim: *"**Even if you receive a success HTTP response code, you'll need to check the `insertErrors` property of the response to determine whether the row insertions were successful** because it's possible that BigQuery was only partially successful at inserting the rows."* And: *"If BigQuery encounters a **schema mismatch** on individual rows in the request, **none of the rows are inserted** and an `insertErrors` entry is returned for each row, even the rows that did not have a schema mismatch. Rows that did not have a schema mismatch have an error with the `reason` property set to `stopped`."* |

---
## THE WRITE HALF (sibling step 82.48) — report, do NOT fix here

`make_outcome_write_fn`, `backend/slack_bot/jobs/_production_fns.py:237-261`.
It builds `records = [{**o, "recorded_at": now_iso} for o in outcomes]` (`:248`) where each
`o` comes from `_compute_outcomes` = `{trade_id, ticker, pnl, outcome}`
(`nightly_outcome_rebuild.py:39-47`). So the payload keys are
`{trade_id, ticker, pnl, outcome, recorded_at}`.

Against the **measured 9-column** `outcome_tracking` schema:
- **Shared keys: exactly ONE** (`ticker`).
- **4 of 5 payload keys do not exist** on the table (`trade_id`, `pnl`, `outcome`, `recorded_at`).
- **2 REQUIRED columns are never supplied**: `analysis_date`, `recommendation`.

Per source #8, that is a schema mismatch → **none of the rows insert**, `insert_rows_json`
**returns** an error list rather than raising, and `:253-255` logs a warning and returns `0`.
The docstring at `:240-241` documents a schema
(`trade_id STRING, ticker STRING, pnl FLOAT, outcome STRING, recorded_at TIMESTAMP`) that
**has never existed**. **A fetch-only repair ships a job that still writes 0 rows** — Main
must not close 82.39 believing the job works. This is 82.48's scope.

**Secondary defect in the same job (also 82.48, but it BLOCKS a naive 82.39 widening):**
`_compute_outcomes` (`nightly_outcome_rebuild.py:37-47`) uses `t.get("pnl", 0.0)`, which
returns the default only when the KEY is **absent**, not when the VALUE is `None`. A NULL
`pnl` therefore reaches `None > 0` → `TypeError`. It is called at `:27`, **OUTSIDE** the
`try` that wraps the write at `:28-32`. It IS inside the `heartbeat()` context manager,
which catches and marks the run failed (`job_runtime.py:105-108`) — so it degrades to
another silent `status: failed` rather than a scheduler crash, but the run produces nothing
and emits no operator signal. **Consequence for 82.39:** if the repaired query drops
`realized_pnl_pct IS NOT NULL`, every BUY row (33 of 65, all NULL pnl) triggers this.
**Keep the `IS NOT NULL` predicate**, or fix `_compute_outcomes` first.

---
## BLAST RADIUS — closing 82.39 BREAKS TWO CURRENTLY-GREEN TESTS

Measured 2026-08-06: `pytest backend/tests/test_phase_82_12_string_column_guards.py -k "unknown or oracle or scope"` → **7 passed**. Both of the following pass *because the defect is still there*.

1. **`test_query_selecting_nonexistent_columns_is_detected`** (`backend/tests/test_phase_82_12_string_column_guards.py:403-422`) asserts the defect **IS** flagged:
   ```python
   assert ("timestamp", "backend/slack_bot/jobs/_production_fns.py") in flagged
   assert ("realized_pnl", "backend/slack_bot/jobs/_production_fns.py") in flagged
   ```
   The 82.39 fix removes both from `unknown_columns` → **this test turns RED**.

2. **`test_the_nonexistent_column_defect_is_queued_as_its_own_step`** (`:425-456`) requires at least one masterplan step with `status not in {done, dropped, superseded}` whose **name** contains all four of `("_production_fns", "paper_trades", "timestamp", "realized_pnl")`. **MEASURED over all 1115 steps: 82.39 is the ONLY match** (82.48's name does NOT contain all four tokens). **Flipping 82.39 to `done` turns this test RED.**

**⇒ The contract MUST include updating both guards in the same change** (invert #1 to assert the defect is GONE while keeping the recall test alive via a synthetic fixture; and for #2, either re-point the signature at 82.48 or re-word 82.48's name to carry all four tokens). Otherwise 82.39 ships green on its own criteria and leaves the repo red — the "audit the COMMIT, not your diff" failure mode.

## Criterion 1: LIVE BigQuery or offline snapshot?
- `schema_oracle.dry_run` (`:550-566`) constructs a real `bigquery.Client` and issues an API call. **It CANNOT run offline** — criterion 1's dry-run test requires ADC + network.
- `schema_oracle.load_snapshot()` reads `backend/db/_schema_snapshot.json` (42,653 bytes, mtime 2026-08-05, 33 tables / 477 columns) so **criterion 4's `derive_scope` half runs fully offline** — `test_phase_82_12_string_column_guards.py:32` already does `ORACLE = so.load_snapshot()` at module scope.
- **TRAPDOOR WARNING:** the repo's only live-BQ gating idiom is a single `@pytest.mark.skipif` at `backend/tests/test_phase_23_2_11_bq_table_freshness.py:86`. If criterion 1's test is written with a skip-on-no-credentials guard, it **silently becomes a no-op in CI and the verification command still exits 0** — the exact trapdoor recorded in `feedback_guards_stop_one_seam_short`. Either make the dry-run test hard-fail without creds, or pair it with an offline test that asserts the query text against the snapshot columns so something real always runs.
- The criterion says *"validated ... by a BigQuery DRY RUN (not billed) and reported valid, asserted by a test that FAILS against the current query."* The "FAILS against the current query" half is satisfiable OFFLINE too (assert the current SQL's identifiers are absent from the snapshot), which is the safer construction to pair with the live dry run.

---
## AUDIT-CLASS FINDING: A SECOND LIVE DEFECT OF THE IDENTICAL CLASS

Found in audit round 8 by extending the sweep **past `derive_scope`'s measured recall
envelope** (f-string table refs). `derive_scope` does NOT see it.

**`backend/api/cost_budget_api.py:80-86`** (`_fetch_llm_tokens_today`, def at `:71`):
```sql
SELECT COALESCE(SUM(input_tokens) + SUM(output_tokens), 0) AS tokens, COUNT(*) AS calls
FROM `{project}.pyfinagent_data.llm_call_log`
WHERE DATE(ts) = CURRENT_DATE()
```
`pyfinagent_data.llm_call_log`'s real columns are **`input_tok`** and **`output_tok`**
(15 columns; `input_tokens`/`output_tokens` do not exist).

**Live dry run, 2026-08-06:**
```
CURRENT  -> 400 Unrecognized name: input_tokens; Did you mean input_tok? at [2:23]
REPAIRED -> VALID
```
**Same fail-open swallow** at `:94-96`: `except Exception` → `logger.warning` → `return None, None`.
**Operator-visible consequence:** `cost_budget_api.py:142` calls it and `:154` sets
`llm_tokens_today=tokens`, so the cost-budget status tile has been reporting
`llm_tokens_today: null` permanently — a *third* silent fail-open, on the endpoint that
exists to police LLM spend.

**⇒ Criterion 4 disposition:** this is a "remaining member" in spirit. It is NOT in
`derive_scope`'s output (the instrument cannot see it), so the criterion-4 sweep would
report a **clean bill of health on a repo that demonstrably still has this bug**. Per
criterion 4's own wording (*"every remaining member fixed or given its own queued
follow-up step"*), Main should **queue `cost_budget_api._fetch_llm_tokens_today` as its own
step** (do not absorb it into 82.39 — different file, different table, different consumer),
and should record the recall caveat in the close.

---
## Internal code inventory (all anchors re-derived 2026-08-06)
| File | Lines | Role | Status |
|---|---|---|---|
| `backend/slack_bot/jobs/_production_fns.py` | 369 | `make_ledger_fetch_fn` **:209-234** (SQL **:220-227**, swallow **:230-232**); `make_outcome_write_fn` **:237-261** (insert **:252**, error-swallow **:253-255**, except **:257-259**) | **BROKEN both halves** |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | 58 | `run()` :13-34 (fetch :26 OUTSIDE try; `_compute_outcomes` :27 OUTSIDE try; write :28-32 inside try); `_compute_outcomes` :37-47 (`t.get("pnl", 0.0)` :43-44 — None-unsafe) | latent `TypeError` on NULL pnl |
| `backend/slack_bot/job_runtime.py` | 118 | `IdempotencyStore` :26-39 = **in-memory `set()`** (no durable receipt); `heartbeat` :66-114, default sink = `logger.info` :83, catches+marks failed :105-108 | no on-disk evidence of past runs |
| `backend/slack_bot/scheduler.py` | — | registers the job :1086, wires prod fns :1136-1138, schedule **:1173-1174 `cron hour=4 UTC`, misfire_grace 3600, coalesce** | job IS scheduled; `com.pyfinagent.slack-bot` pid 658 alive |
| `backend/db/schema_oracle.py` | 574 | `derive_scope` **:453-526**; `iter_python_files` :174-180 (**roots at `backend/`, excludes `/tests/` ⇒ `scripts/` never scanned**); `extract_sql_literals` :183-208; alias strip :501-508; `_TYPE_NAMES` :538-541; `dry_run` **:550-566 (catches only `BadRequest`)**; `load_snapshot` :130-133 | instrument works but LOW recall |
| `backend/db/_schema_snapshot.json` | 42,653 B | offline oracle, 33 tables / 477 cols, mtime 2026-08-05 | current |
| `backend/services/cycle_health.py` | 602 | `_STRING_DATE_TIMESTAMP_COLS` **:436-439** contains `("paper_trades","created_at")`; SAFE.TIMESTAMP branch **:442-468**; portability warning **:451-460** | the idiom to copy |
| `backend/services/autoresearch_health.py` | 387 | phase-82.11 template: severity doctrine :17-30, function-local import **:321-325**, `severity = "P0" if escalated else "P1"` **:328** | REUSE this |
| `backend/services/observability/alerting.py` | — | `_CRITICAL_SEVERITIES` **:54** = `{P0,P1,critical,CRITICAL}`; `_bot_token_fallback` :136 invoked :217-218 (webhook is empty on this machine); `raise_cron_alert_sync` **:253-287**, never raises :266 | **P1 DOES reach the operator** |
| `backend/tests/test_phase_82_11_autoresearch_failure_paging.py` | — | `ALERT_TARGET` **:65**; wrong-patch-target guard **:502-506**; 4-point anti-vacuity doctrine :1-35 | test template |
| `backend/tests/test_phase_82_12_string_column_guards.py` | — | `ORACLE = load_snapshot()` :32; **:403-422 asserts the defect IS flagged**; **:425-456 requires an OPEN step w/ the 4-token signature**; alias-FP guard :459-464 | **BOTH turn RED on close** |
| `backend/services/autonomous_loop.py` | — | real `outcome_tracking` writer :3041-3096 (phase-35.1), gated by `settings.py:34 paper_learn_loop_enabled=False` | DARK — separate cause |
| `backend/api/cost_budget_api.py` | — | `_fetch_llm_tokens_today` :71-96, bad SQL **:80-86**, swallow **:94-96**, consumed :142/:154 | **NEW 2nd defect** |

## Application to pyfinagent — what the contract should say
1. **Repair the SQL** at `_production_fns.py:220-227` to
   `SELECT trade_id, ticker, action, price, quantity, created_at, realized_pnl_pct AS pnl`
   with `WHERE SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) AND realized_pnl_pct IS NOT NULL`. Both dry-run VALID (measured). **Keep `IS NOT NULL`** — dropping it detonates `_compute_outcomes`.
2. **Units caveat:** `realized_pnl_pct` is a PERCENT. Aliasing it `AS pnl` preserves the win/loss SIGN (all `_compute_outcomes` needs) but changes UNITS for any future consumer. State it in the contract.
3. **Criterion 2 fixture:** pin `2026-06-01 <= created_at < 2026-07-01` (**20 SELL rows, 20 non-null pnl**). Do NOT assert against the live rolling 30-day window — it yields 3 rows today and **0 after 2026-08-26**.
4. **Criterion 1:** dry run needs live BQ/ADC; do not let it `pytest.skip` into a silent no-op. Pair with an offline snapshot assertion so something always runs. Note `dry_run` only catches `BadRequest`.
5. **Criterion 3:** `raise_cron_alert_sync(source=..., error_type=..., severity="P1", ...)` inside the existing `except`. Never P2. Patch `backend.services.observability.alerting.raise_cron_alert_sync`, `autospec=True`, assert `call_args`, and add the negative (success ⇒ no alert).
6. **Criterion 4:** assert the INPUT surface non-empty (`files_scanned/sql_literals/tables_resolved/columns_in_oracle > 0`), not the findings list. Queue `cost_budget_api._fetch_llm_tokens_today` as its own step.
7. **Same-change repairs:** invert `test_phase_82_12_string_column_guards.py:403-422` and re-point `:443`'s signature (or re-word 82.48's name), or the repo goes red on close.
8. **Do NOT promise** a learning-loop restoration (Q4) or a working job (write half is 82.48).

## Consensus vs debate (external)
**Consensus:** dry runs are free and are a genuine planner-level validator (#1, #3); broad
catch + log + return-empty is a named, quantified anti-pattern (#4, 31.9% Catch Generic);
alerts must be actionable and rare (#6).
**Debate / tension:** #6 (SRE) warns that rarely-exercised alerting *"should be up for
removal"*, while #7 (2026) argues for MORE controls (row-count/volume) because
job-level success hides silent failures. Resolution for this step: one P1 on the exception
path (rare, actionable, currently 100% firing), and treat the zero-row case as a
*heartbeat field* rather than a second pager — satisfying #7's detection need without
#6's noise cost.

## Pitfalls (from literature + measurement)
- A valid query is not a correct query (#3 "Things this can't catch") ⇒ criteria 1 and 2 cannot be merged.
- `SAFE.` does not apply to operators (#2) and breaks on native TIMESTAMP columns (measured).
- A success HTTP code does not mean rows landed (#8) ⇒ the write half needs its own check.
- Patch where looked up, not where defined (#5) ⇒ function-local imports make naive patches vacuous.
- A checker that scans nothing looks identical to a clean repo (`schema_oracle.py:39-43`).

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL — **8** (2 curl+tag-strip on JS-rendered Google docs, 1 curl raw markdown, 1 curl+pdfplumber PDF, 1 curl+tag-strip, 3 WebFetch/curl)
- [x] 10+ unique URLs total — **20** (8 read-in-full + 12 snippet-only)
- [x] Recency scan (2024–2026) performed + reported
- [x] Full pages/papers read, not abstracts (ICPC 2017 = 24,075 chars via pdfplumber)
- [x] file:line anchors for every internal claim (re-derived, not recalled)

Soft checks:
- [x] Internal exploration covered every module the caller named, plus `scheduler.py`, `job_runtime.py`, `alerting.py`, `autonomous_loop.py`, the 82.11/82.12 test files, and `.claude/masterplan.json`
- [x] Contradictions noted (SRE ch.6 vs dqlabs 2026 on alert count)
- [x] Claims cited per-claim

## Adaptive coverage (audit-class)
| Round | Focus | New read-in-full findings |
|---|---|---|
| 1 | Live schemas, dry runs, git history, sources #1–#5 | many |
| 2 | SRE ch.6, dqlabs 2026, dry-run limitations (#6,#7) | yes |
| 3 | `insert_rows_json` / insertAll semantics (#8) | yes |
| 4 | SQL static-analysis / schema-linter literature | **0 — DRY** |
| 5 | Offline-vs-live; discovered the existing 82.12 guards | yes |
| 6 | Masterplan blast radius (1115 steps) | yes |
| 7 | Other `realized_pnl` consumers; runtime state | **0 — DRY** (all `unrealized_pnl`) |
| 8 | Widened sweep past recall envelope | **yes — 2nd live defect** |
| 9 | Disciplined widened sweep (9 survivors) | **0 — DRY** (2 de-dup, 7 FPs classified) |
| 10 | Alert-channel reachability (`_CRITICAL_SEVERITIES`) | **0 — DRY** |
`rounds=10, dry_rounds=2 (consecutive, R9+R10), K_required=2 ⇒ coverage.dry = true`

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 12,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 18,
  "coverage": {"audit_class": true, "rounds": 10, "dry_rounds": 2,
               "K_required": 2, "new_findings_last_round": 0, "dry": true},
  "brief_path": "handoff/current/research_brief_82.39.md",
  "gate_passed": true
}
```
