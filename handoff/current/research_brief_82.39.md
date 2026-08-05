# Research Brief — masterplan step 82.39

**Tier:** moderate (caller-specified). **Audit-class:** false.
**Started:** 2026-08-05. **Status:** IN PROGRESS (write-first; this file grows incrementally).

## Topic

`backend/slack_bot/jobs/_production_fns.py` builds the `nightly_outcome_rebuild`
fetch closure and SELECTs `timestamp` + `realized_pnl` from
`sunny-might-477607-p8.financial_reports.paper_trades`. Neither column exists
(measured 2026-08-05: 18 columns; real names `created_at` STRING and
`realized_pnl_pct` FLOAT). BigQuery returns 400 Unrecognized name; the bare
`except Exception` logs a warning and returns `[]`, so the job has been
running on ZERO trades and looks like a successful no-op.

External research scope: (a) BigQuery dry-run billing + validation semantics;
(b) the fail-open / silent-empty-result anti-pattern (SRE graceful degradation
vs silent failure); (c) prior art on validating SQL against a schema in CI
(dbt compile, sqlfluff, BQ dry-run-in-CI).

---

## Search queries run (three-variant discipline)

| Variant | Query |
|---|---|
| **Year-less canonical** | `BigQuery dry run query not billed validates schema dryRun` |
| **Year-less canonical** | `silent failure anti-pattern catching exception returning empty list data pipeline` |
| **Year-less canonical** | `graceful degradation versus silent failure SRE observability empty result not the same as no data` |
| **Current-year (2026)** | `dbt dry run BigQuery CI validate SQL against schema before merge 2026` |
| **Last-2-year (2025)** | `"silent data loss" data pipeline observability 2025 detecting jobs that succeed with zero rows` |

---

## Sources read IN FULL (counts toward the gate) — 7

| # | URL | Accessed | Tier | Fetched how | What it establishes |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://cloud.google.com/bigquery/docs/running-queries | 2026-08-05 | official docs | `curl` + tag-strip (58,448 chars; JS-rendered so WebFetch returns nav only — see `feedback_gcloud_docs_fetch`) | **The load-bearing citation for criterion 1.** Verbatim: *"A dry run in BigQuery provides the following information: estimate of charges in on-demand mode; **validation of your query**; approximate bytes processed by your query in capacity mode."* and *"**Dry runs don't use query slots, and you are not charged for performing a dry run.**"* Also the one caveat that matters: a dry run of a **federated** query over an external data source *"might report a lower bound of 0 bytes of data, even if rows are returned"* — irrelevant here (`paper_trades` is native). |
| 2 | https://cloud.google.com/bigquery/docs/best-practices-costs | 2026-08-05 | official docs | `curl` + tag-strip | The validator's error shape is a **name-resolution** error, e.g. *"Not found: Table myProject:myDataset.myTable was not found in location US"*; valid → *"Query successfully validated. Assuming the tables are not modified, running this query will process 10918 bytes of data."* Caveat: *"The estimate of the number of bytes that is billed for a query is an upper bound"* — so don't assert an exact byte count in a test. |
| 3 | https://arxiv.org/abs/1704.00778 (PDF via `pdfplumber`, 24,075 chars) | 2026-08-05 | **peer-reviewed** (Padua & Shang, ICPC 2017, Concordia Univ.) | `curl` → `pdfplumber` (per research-gate.md step 3; ar5iv 307-redirected to `/abs/`) | Names the exact defect. Table I: **"Dummy Handler — The handler only display or logs some information"**; **"Log and Return Null — Besides being a dummy handler, the handler return null"**; **"Catch Generic — The handler catches a generic exception type (e.g. Exception)"**. Measured prevalence: only five anti-patterns are prevalent — *"Unhandled Exceptions, Catch Generic, Unreachable Handler, Over-catch and Destructive Wrapping, are detected in over 20% (40.8%, 31.9%, 28.0%, 24.6%, 22.3%, respectively) of the catch blocks or throws statements in median."* Argument: *"Generic catch is a sign of developers' lack of knowledge on the possible exception(s)"* — a `except Exception` cannot distinguish a transient BQ quota error from a permanent schema error, which is precisely why this one survived 86 days. |
| 4 | https://github.com/autotraderuk/dbt-dry-run/blob/main/README.md | 2026-08-05 | practitioner (open-source tool) | WebFetch | The canonical enumeration of what a BQ dry run catches, verbatim: *"Typos in SQL keywords"*, **"Typos in columns names: `orders.produts` instead of `orders.products`"**, *"Problems with incompatible data types"*, *"Incompatible schema changes to models"*, *"Incompatible schema changes to sources"*, *"Permission errors"*. Second row is exactly this defect. What it **cannot** catch: *"Queries producing incorrect results (requires dbt tests)"* — **this is why criterion 2 exists and why criterion 1 alone is insufficient.** |
| 5 | https://medium.com/autotrader-engineering/dry-running-our-data-warehouse-using-bigquery-and-dbt-12ccae0209f1 | 2026-08-05 | authoritative blog (Autotrader Eng.) | WebFetch | The motivating failure mode is ours: *"The dry run can catch typos and SQL syntax errors ... which would otherwise be compiled by dbt without error **until the overnight run failed**."* Cost/latency: 1000+ models validated *"in under 30 seconds."* Establishes dry-run-in-CI as accepted practice, not an invention. |
| 6 | https://sre.google/sre-book/addressing-cascading-failures/ | 2026-08-05 | official docs (Google SRE Book) | WebFetch | The criterion-3 principle: *"**Monitor and alert when too many servers enter these modes**"* (degraded/fallback modes). And *"Graceful degradation shouldn't trigger very often—usually in cases of a capacity planning failure or unexpected load shift"* — a fail-open that fires **every single night for 86 days** is not graceful degradation, it is an outage wearing its costume. Also notes operators gain minimal experience with rarely-exercised code paths. |
| 7 | https://robertsahlin.substack.com/p/your-pipeline-succeeded-your-data (pub. 2026-03-11) | 2026-08-05 | practitioner (Robert Sahlin, data eng.) | WebFetch | Titled for this exact bug. Verbatim: *"**A pipeline can succeed while writing zero rows.** An upstream system can silently halve its output."* Detection design: zero-volume windows must be materialised via LEFT JOIN onto a dense grid so *"windows with no matching data get `actual_rows = 0` instead of being dropped"* — i.e. **zero is a value that must be emitted, not an absence that is skipped.** Directly supports emitting a signal on a 0-row fetch. |

_(An 8th page, https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-dry-run, was also WebFetched in full but is a thin code-sample page that explicitly lacked the billing statement; not counted toward the gate.)_

---

## Snippet-only sources (context; does NOT count toward the gate) — 18

| URL | Tier | Why not read in full |
|-----|------|----------------------|
| https://docs.cloud.google.com/bigquery/docs/samples/bigquery-query-dry-run | official docs | Fetched, but code-samples only; superseded by source 1 |
| https://github.com/n8n-io/n8n/issues/15102 | community | Bug report about a wrapper's dry-run handling; not authoritative on BQ semantics |
| https://oneuptime.com/blog/post/2026-02-17-how-to-estimate-bigquery-query-costs-before-running-with-dry-run/view | community blog | Restates source 1; lower tier |
| https://deepwiki.com/takuya0206/bigquery-mcp-server/3.2-dry-run-query-tool | community | Third-party MCP wrapper docs |
| https://glama.ai/mcp/servers/caron14/mcp-bigquery | community | MCP validator listing |
| https://pypi.org/project/dbt-dry-run/ | practitioner | Package metadata; README (source 4) is richer |
| https://github.com/autotraderuk/dbt-dry-run/blob/main/CHANGES.md | practitioner | Changelog only |
| https://github.com/dbt-labs/dbt-core/discussions/4456 | community | Feature discussion; the tool in source 4 is the outcome |
| https://medium.com/towards-data-engineering/a-guide-to-dbt-dry-runs-safe-simulation-for-data-engineers-7e480ce5dcf7 | community blog | Derivative of source 5 |
| https://aipatternbook.com/silent-failure | community | Pattern catalogue; source 3 is the peer-reviewed equivalent |
| https://sobolevn.me/2019/02/python-exceptions-considered-an-antipattern | authoritative blog | Python-specific opinion piece; source 3 covers the claim empirically |
| https://ericvruder.dk/20190902/exceptions-catch-everything-handle-nothing/ | community blog | Same ground as source 3 |
| https://github.com/charlax/professional-programming/blob/master/antipatterns/error-handling-antipatterns.md | community | Curated link list |
| https://arxiv.org/pdf/2604.17587 (AIRA: AI-Induced Risk Audit) | peer-reviewed preprint | 2026; relevant ("failure-untruthful" AI-generated code returning success regardless of outcome) but tangential to the fix design — noted in the recency scan |
| https://www.frugaltesting.com/blog/how-to-detect-silent-failures-in-microservices-using-advanced-observability-techniques | community | Microservice framing, not batch-job framing |
| https://sreschool.com/blog/comprehensive-tutorial-on-graceful-degradation-in-site-reliability-engineering/ | community | Derivative of source 6 |
| https://blog.anomalyarmor.ai/data-pipeline-monitoring-how-to-stop-silent-failures-before-they-hit-production/ | vendor blog | Vendor content; source 7 is the neutral treatment |
| https://airbyte.com/data-engineering-resources/data-pipeline-observability | vendor blog | Vendor content |
| https://medium.com/@jooramos_37651/catching-silent-failures-in-data-pipelines-with-forecasting-metadata-and-an-llm-d316e1666bb6 | community blog | 2026-06; LLM-based anomaly detection, over-engineered for this step |
| https://zylos.ai/en/research/2026-02-20-graceful-degradation-ai-agent-systems/ | community | Agent-systems framing |

---

## Recency scan (last 2 years, 2024-2026) — PERFORMED

Queries: `dbt dry run BigQuery CI validate SQL against schema before merge 2026` and
`"silent data loss" data pipeline observability 2025 detecting jobs that succeed with zero rows`.

**Result: 2 new findings that COMPLEMENT (do not supersede) the canonical sources.**

1. **"Your Pipeline Succeeded. Your Data Didn't."** (2026-03-11, source 7) is the current
   articulation of exactly this failure class, and it moves the state of the art from
   "log the error" to "**emit the zero as a value**". It also documents BigQuery's own
   `AI.DETECT_ANOMALIES` / `WRITE_API_TIMELINE` metadata route for volume monitoring —
   **out of scope for 82.39** (that is a monitoring platform, not a one-job repair) but
   worth a future step if row-volume regression detection is ever wanted repo-wide.
2. **AIRA (arXiv:2604.17587, 2026)** independently names the class in AI-generated code:
   *"returning success status regardless of whether an operation succeeded or failed"*,
   which it terms **"failure-untruthful"**. Relevant here because the fail-open closure
   family in `_production_fns.py` was itself LLM-authored (phase-23.6, 2026-05-11) and
   the same shape recurs in five other closures in the same file.

**Nothing in the 2024-2026 window contradicts** the 2017 anti-pattern taxonomy (source 3)
or the Google SRE guidance (source 6); both remain the canonical statements. The BigQuery
dry-run semantics in sources 1-2 are current documentation as of the 2026-08-05 fetch.

---

## Internal code inventory

### A. The broken closure — `backend/slack_bot/jobs/_production_fns.py:209-234`

Function name is **`make_ledger_fetch_fn()`** (a factory). It returns the closure
`_fetch() -> list[dict]` defined at `:217`. Verbatim SQL at `:220-227`:

```sql
SELECT trade_id, ticker, action, price, quantity, timestamp,
       SAFE_CAST(realized_pnl AS FLOAT64) AS pnl
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE TIMESTAMP_TRUNC(timestamp, DAY) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND realized_pnl IS NOT NULL
LIMIT 1000
```

`except Exception` at `:230` → `logger.warning(...)` → `return []` at `:232`.
Query runs with `location="us-central1"` and `timeout=30` (`:228`) — both correct.

### B. Live schema — MEASURED 2026-08-05

Command: `python3 -c "from google.cloud import bigquery; c=bigquery.Client(project='sunny-might-477607-p8'); t=c.get_table('sunny-might-477607-p8.financial_reports.paper_trades'); print(len(t.schema)); [print(f.name,f.field_type,f.mode) for f in t.schema]"`

**18 columns, 65 rows, location `us-central1`:**

| # | column | type | mode |
|---|--------|------|------|
| 1 | trade_id | STRING | REQUIRED |
| 2 | ticker | STRING | REQUIRED |
| 3 | action | STRING | REQUIRED |
| 4 | quantity | FLOAT | REQUIRED |
| 5 | price | FLOAT | REQUIRED |
| 6 | total_value | FLOAT | NULLABLE |
| 7 | transaction_cost | FLOAT | NULLABLE |
| 8 | reason | STRING | NULLABLE |
| 9 | analysis_id | STRING | NULLABLE |
| 10 | risk_judge_decision | STRING | NULLABLE |
| 11 | **created_at** | **STRING** | REQUIRED |
| 12 | round_trip_id | STRING | NULLABLE |
| 13 | holding_days | INTEGER | NULLABLE |
| 14 | **realized_pnl_pct** | **FLOAT** | NULLABLE |
| 15 | mfe_pct | FLOAT | NULLABLE |
| 16 | mae_pct | FLOAT | NULLABLE |
| 17 | capture_ratio | FLOAT | NULLABLE |
| 18 | signals | STRING | NULLABLE |

`timestamp` present=**False**. `realized_pnl` present=**False**. Confirms the caller's premise.
Note `total_value` (FLOAT) exists — this is the notional the derivation needs (see traps).

### C. The consumer — `backend/slack_bot/jobs/nightly_outcome_rebuild.py`

`run()` at `:13-34`; `_compute_outcomes()` at `:37-47`. The consumer reads **exactly three
keys** off each row: `trade_id`, `ticker`, `pnl` (`:41-43`). `action`, `price`, `quantity`,
`timestamp` are SELECTed but **never read** — dead projection.
Classification at `:44`: `"win" if t.get("pnl", 0.0) > 0 else "loss"`. Only the **SIGN**
of `pnl` reaches the output; the magnitude is passed through verbatim into the written row.

### D. THE WRITE SIDE IS ALSO BROKEN — new finding, NOT in the step description

Command: `c.get_table('sunny-might-477607-p8.financial_reports.outcome_tracking')`

The table **exists** (`us-central1`, **0 rows**, last modified **2026-03-18**) but its
schema bears no relation to what `make_outcome_write_fn` writes.

| Written key (`_production_fns.py:244-252` + `nightly_outcome_rebuild.py:40-46`) | Exists on table? |
|---|---|
| `trade_id` | **NO** |
| `ticker` | yes (STRING REQUIRED) |
| `pnl` | **NO** |
| `outcome` | **NO** |
| `recorded_at` | **NO** |

Actual columns: `ticker`, `analysis_date` (STRING **REQUIRED**), `recommendation`
(STRING **REQUIRED**), `price_at_recommendation`, `current_price`, `return_pct`,
`holding_days`, `beat_benchmark`, `evaluated_at`. Two REQUIRED columns are never supplied.

The docstring at `_production_fns.py:240-241` asserts
`outcome_tracking(trade_id STRING, ticker STRING, pnl FLOAT, outcome STRING, recorded_at TIMESTAMP)`
— **that schema is a fiction**; it has never matched the live table.
`insert_rows_json` does not raise on unknown fields, it **returns** an error list →
`:253-255` logs a warning and returns 0. So the write is a **second, independent**
fail-open on the same job. **Fixing only the fetch leaves the job writing 0 rows.**

### E. How long has it been dead — MEASURED

Command: `git log -S "realized_pnl IS NOT NULL" --format="%h %ad %s" -- backend/slack_bot/jobs/_production_fns.py`

Single hit: **`2301b977`, 2026-05-11, "phase-23.6: harness MAS cycles 23.6.0-23.6.3"** —
the commit that first created the file (`git log --reverse` on the same path returns
`2301b977` as commit #1). The phantom columns have been there since **day one**; this is
not a regression from a schema rename. Elapsed: **2026-05-11 → 2026-08-05 = 86 days.**
Corroborating measurement: `outcome_tracking.num_rows == 0` and `modified == 2026-03-18`
— the destination has **never** received a row.

### F. Live $0 dry runs — the decisive evidence

Command: `bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)`, `location="us-central1"`.

| Variant | Result |
|---|---|
| **Current query verbatim** | `400 BadRequest: Unrecognized name: timestamp at [5:39]` |
| **Naive rename** (`timestamp`→`created_at`, `realized_pnl`→`realized_pnl_pct`, keeps `TIMESTAMP_TRUNC`) | `400 BadRequest: No matching signature for function TIMESTAMP_TRUNC` |
| **SAFE.TIMESTAMP form** | **VALID**, 7257 bytes, schema `[trade_id, ticker, action, price, quantity, total_value, created_at, realized_pnl_pct, pnl]` |

The middle row is the important one: **the obvious fix is still broken.** `created_at` is
STRING, so `TIMESTAMP_TRUNC(created_at, DAY)` has no matching signature. The working
predicate is the `cycle_health` idiom:

```sql
WHERE SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
```

Established precedent: `backend/services/cycle_health.py:436-439` registers
`("paper_trades", "created_at")` in `_STRING_DATE_TIMESTAMP_COLS`, and `:462-474`
selects `SAFE.TIMESTAMP(MAX(col))` vs bare `MAX(col)` on that membership. The docstring
at `:451-460` records the counter-trap: `SAFE.TIMESTAMP` **breaks** on a native TIMESTAMP
column (`SAFE with function timestamp is not supported`). Test shape at
`backend/tests/test_phase_43_dod5_freshness.py:46-74` (both directions asserted).

### G. Does the repaired query actually return rows? — MEASURED

Command: `SELECT COUNT(*), COUNTIF(realized_pnl_pct IS NOT NULL), COUNTIF(SAFE.TIMESTAMP(created_at) IS NULL), MIN/MAX(created_at), COUNTIF(last 30d), COUNTIF(last 30d AND pnl NOT NULL) FROM paper_trades`

```
total=65  with_pnl=32  unparseable_created_at=0
min_ts=2026-04-26T21:12:28Z  max_ts=2026-07-31T18:47:37Z
last30=7  last30_with_pnl=3
```

Per-action: `BUY n=33 with_pnl=0`; `SELL n=32 with_pnl=32, min=-14.4625, max=116.4029`.

Three consequences:
1. The repaired query returns **3 rows today** — non-vacuous, but only just.
2. `realized_pnl_pct` is populated **only on SELL legs** (realized P&L is a round-trip
   concept). Every BUY is NULL, so the `IS NOT NULL` predicate is doing real work.
3. The value range **-14.46 to +116.40** is unambiguously **percent**, not dollars.

---

### H. Consequence — the step's "frozen reflection corpus" claim is WRONG about the cause

Commands: `grep -n "outcome_tracking" backend/services/outcome_tracker.py` → **zero hits**.
`grep -n "bq_dataset_outcomes|bq_table_outcomes" backend/config/settings.py` → `:62-63`
(`financial_reports.outcome_tracking`). BQ: `financial_reports.agent_memories` rows=**0**,
modified **2026-03-18**, cols `[agent_type, ticker, situation, lesson, created_at]`.

- `backend/services/outcome_tracker.py` never touches `outcome_tracking` by name; it goes
  through `bigquery_client.py:47` (`self.outcomes_table`), written at `:415` and read at
  `:489`. That row shape MATCHES the live 9-column table.
- So the live `outcome_tracking` schema belongs to the **OutcomeTracker / learn-loop**
  path. `nightly_outcome_rebuild` was written against an imagined trade-ledger-shaped
  table that has never existed.
- The reflection corpus IS frozen (`agent_memories` 0 rows) but the cause is
  `settings.py:34` `paper_learn_loop_enabled: bool = Field(False, ...)` — the phase-35.1
  learn-loop flag is **DARK by default**. **Fixing 82.39 will NOT unfreeze it.** Two
  independent causes; do not let the contract claim otherwise.

### I. The paging path (criterion 3) — use the established shape

`backend/services/observability/alerting.py:253-259`:
`raise_cron_alert_sync(source, error_type, severity, title, details) -> bool`.
Docstring `:260-267`: schedules on a running loop (returns True optimistically,
fire-and-forget) else `asyncio.run`; **"Always fail-open: never raises out."**
Precedent from phase-82.10: `backend/services/freshness_cron.py:128-132` imports it and
`:165` passes `severity="P1"`. The `:24` comment records that a **P1 bypasses the
consecutive-occurrence deduper**, i.e. P2/P3 would be silently dropped for a one-shot.
Use `severity="P1"`; do not invent a new channel. NOTE the closure runs on an APScheduler
executor thread with the slack-bot loop on another thread — `raise_cron_alert_sync`
already handles the running-loop case, so it is the correct bridge (the file's own
`_post_slack_sync` at `:267-289` is the older `run_coroutine_threadsafe` idiom).

### J. The sweep (criterion 4) — re-run, MEASURED

Command: `python3 -c "from backend.db import schema_oracle as so; print(so.derive_scope(so.load_snapshot()))"`

```
files_scanned=294  sql_literals=13  tables_resolved=1  columns_in_oracle=477
scope (STRING + date/num semantics) = 0
unknown_columns = 2
  {'table': 'financial_reports.paper_trades', 'identifier': 'realized_pnl',
   'file': 'backend/slack_bot/jobs/_production_fns.py', 'line': 220}
  {'table': 'financial_reports.paper_trades', 'identifier': 'timestamp',
   'file': 'backend/slack_bot/jobs/_production_fns.py', 'line': 220}
```

Derived scope is **non-empty (2)**, and **both members are this one file** — so criterion 4
is satisfiable by fixing `_production_fns.py` alone, with **no follow-up step owed**.

**BUT the sweep's recall is low and the contract must say so.** `schema_oracle.py:174-180`
`iter_python_files()` scans `backend/` only and excludes `/tests/`;
`:183-208` `extract_sql_literals()` requires a **literal backtick-quoted fully-qualified
`project.dataset.table`** inside the string, and for f-strings it joins only the
`ast.Constant` parts — so `FROM \`{table}\`` (the dominant idiom here) resolves nothing.
Measured blind spot: `grep -rl 'FROM \`{' --include="*.py" backend/ | grep -v /tests/ | wc -l`
= **19 files** invisible to the sweep (e.g. `backend/db/bigquery_client.py:489`,
`backend/services/cycle_health.py:469-473`, `backend/metrics/sortino.py:108`,
`backend/agents/skill_optimizer.py:199,214`). `tables_resolved=1` out of 477 oracle
columns across 4 datasets is the same fact stated another way.
**"0 remaining members" means "0 that this scanner can see", not "0 in the repo."**

### K. Test-shape precedent

`backend/tests/test_phase_82_12_string_column_guards.py` (557 lines) is the established
shape. Directly reusable: `test_derived_scope_counters_are_all_non_empty` (`:73`),
`test_derived_scope_is_non_empty` (`:89`), `test_query_selecting_nonexistent_columns_is_detected`
(`:403`), `test_the_nonexistent_column_defect_is_queued_as_its_own_step` (`:425` — this
test asserts 82.39 exists in the masterplan and **will need updating when 82.39 closes**),
`test_alias_is_not_reported_as_a_missing_column` (`:459`),
`test_recall_envelope_is_measured_not_assumed` (`:208`, parametrized).
The verification command names a NEW file: `backend/tests/test_phase_82_39_outcome_rebuild_query.py`.

### L. Wiring / liveness

`backend/slack_bot/scheduler.py:1086` (job id list), `:1136-1138` (injects
`ledger_fetch_fn` + `outcome_write_fn` from `pf`), `:1173` (`cron`). Schedule is
`cron[hour='4']` per `backend/api/cron_dashboard_api.py:102` and
`tests/verify_phase_23_6_2.py:40`. The job IS registered and IS firing — it just fetches
nothing. `schema_oracle.dry_run()` (`:550-567`) passes **no `location=`**; tested live
against the `us-central1` `paper_trades` table and it works (BQ infers location), so no
trap there — `dry_run(valid)` → `None`, `dry_run(broken)` → the 400 string.

---

## Recommendation for the contract

### The repair (dry-run-verified today)

```sql
SELECT trade_id, ticker, action, price, quantity, total_value, created_at,
       SAFE_CAST(realized_pnl_pct AS FLOAT64) AS pnl_pct
FROM `sunny-might-477607-p8.financial_reports.paper_trades`
WHERE SAFE.TIMESTAMP(created_at) >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND realized_pnl_pct IS NOT NULL
LIMIT 1000
```

Keep `location="us-central1"` and `timeout=30` (`_production_fns.py:228`) — both already correct.

### The semantic decision the step asks you to make deliberately

`realized_pnl` (an amount) does **not** exist and never has. `realized_pnl_pct` is a
**percent** (measured range -14.46 to +116.40). The consumer
(`nightly_outcome_rebuild.py:37-47`) uses `pnl` for **exactly two things**: the win/loss
sign test, and a pass-through into the written row.

**Recommendation: do NOT silently alias `realized_pnl_pct AS pnl`.** The sign is
identical, so win/loss classification is unaffected — but the pass-through would write a
percent into a field every reader will interpret as dollars. Two honest options:

- **(a) Rename the contract.** Alias `AS pnl_pct`, rename the consumer's key to `pnl_pct`,
  and name the written column `return_pct` (which is what the destination table actually
  calls it — see below). Smallest change, no fabricated numbers. **Preferred.**
- **(b) Derive a notional.** `total_value` (FLOAT) is on the table and is the SELL-leg
  notional; a notional P&L would be `total_value * realized_pnl_pct / 100` — but that
  formula is an **assumption about how `realized_pnl_pct` is defined** that this brief did
  not verify against the writer. If the contract wants a notional, it must first prove the
  definition, or it will ship a plausible-looking wrong number. Do not do this blind.

### THE FIX IS BIGGER THAN THE STEP DESCRIPTION SAYS — the write is broken too

Repairing only the fetch produces a job that reads 3 rows and writes **0**, still silently.
`make_outcome_write_fn` (`_production_fns.py:237-261`) writes
`{trade_id, ticker, pnl, outcome, recorded_at}` to `financial_reports.outcome_tracking`,
whose real columns are `{ticker, analysis_date*, recommendation*, price_at_recommendation,
current_price, return_pct, holding_days, beat_benchmark, evaluated_at}` (* = REQUIRED).
`trade_id`, `pnl`, `outcome`, `recorded_at` do not exist; two REQUIRED columns are never
supplied. `insert_rows_json` **returns** an error list rather than raising, so `:253-255`
swallows it exactly like the fetch. The contract must cover both halves or the step ships
a job that is still, end-to-end, a no-op.

Note there is no dry-run equivalent for the streaming `insertAll` path — the write must be
validated by comparing the record keys to `client.get_table(...).schema` (cheap, offline
against a snapshot) plus one real insert in the fixture, or by switching the write to a
query job so `dry_run` covers it too.

### Criterion 3 — the operator signal

Use `raise_cron_alert_sync(source, error_type, severity, title, details)` from
`backend/services/observability/alerting.py:253`, `severity="P1"` (P2/P3 are dropped by the
deduper for a one-shot — `freshness_cron.py:24`), following the `freshness_cron.py:128-132,
:165` precedent. **Do not delete the `except Exception`** — a nightly job must not kill the
scheduler. Fire the alert on **two** conditions, not one:

1. the exception path (`:230`), and
2. **the zero-row path** — a fetch that succeeds and returns `[]` is the state this bug
   spent 86 days in, and it is indistinguishable from the exception path to any observer
   (source 7: *"A pipeline can succeed while writing zero rows"*). A test that only covers
   the exception branch leaves the actual historical failure mode unasserted.

`raise_cron_alert_sync` is itself fail-open by contract (*"Always fail-open: never raises
out"*), so the test must capture the **call**, not a side effect — monkeypatch the symbol
in the module under test and assert on the recorded args.

### Criterion 4 — what the sweep can and cannot claim

Re-run gives `unknown_columns = 2`, **both in `_production_fns.py:220`** — so the derived
scope is non-empty and there is **no follow-up step owed**. But the contract must state the
recall bound honestly: the scanner resolved **1 table** and **13 SQL literals** across 294
files because it requires a literal backticked FQ table name, while **19 backend non-test
files** build the table name by interpolation and are invisible to it. Claim
"0 remaining members **within the sweep's measured recall envelope**", not "the repo is
clean". (`test_recall_envelope_is_measured_not_assumed` at
`test_phase_82_12_string_column_guards.py:208` is the existing precedent for saying this.)

Also: `test_the_nonexistent_column_defect_is_queued_as_its_own_step`
(`test_phase_82_12_string_column_guards.py:425`) asserts 82.39 exists in the masterplan.
Closing 82.39 may flip that test — check it in the same cycle.

### Traps

1. **The naive rename still 400s.** `TIMESTAMP_TRUNC(created_at, DAY)` on a STRING column →
   `No matching signature for function TIMESTAMP_TRUNC` (dry-run reproduced above). Use
   `SAFE.TIMESTAMP(created_at)`. Counter-trap: `SAFE.TIMESTAMP` **breaks** on a native
   TIMESTAMP column (`cycle_health.py:451-460`), so this is not a blanket rule.
2. **`_compute_outcomes` crashes on a NULL pnl.** MEASURED:
   `_compute_outcomes([{'trade_id':'t1','ticker':'AMD','pnl':None}])` →
   `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
   (`.get("pnl", 0.0)` returns the default only when the KEY IS ABSENT, not when the value
   is None). And `nightly_outcome_rebuild.py:27` calls it **outside** the try at `:28`, so
   it would propagate out of `run()`. If anyone widens the query by dropping
   `IS NOT NULL`, the job goes from silent-no-op to hard crash. Guard the consumer.
3. **A "last 30 days" fixture is time-fragile.** Measured today: `last30=7`,
   `last30_with_pnl=3`, and the newest trade is `2026-07-31`. On 2026-08-31 that fixture
   returns 0 rows and passes vacuously. Criterion 2 must pin a window that provably
   contains trades (`2026-04-26 .. 2026-07-31`, 32 rows with pnl) or assert
   `row_count > 0` against a measured floor — never assert only "no exception".
4. **A dry run does not prove the query returns data.** Source 4's cannot-catch list is
   explicit: *"Queries producing incorrect results (requires dbt tests)."* Criterion 1 and
   criterion 2 are genuinely two different checks; do not let one test claim both.
5. **The mutation test must break on the CURRENT query.** Criterion 1 says "asserted by a
   test that fails on the current query" — pin the pre-fix SQL as a fixture constant and
   assert `dry_run(BROKEN_SQL) is not None` **and** that the message names `timestamp`.
   Asserting only `is not None` would also pass if the credentials were wrong.
6. **Don't assert an exact byte count.** Source 2: the dry-run byte estimate is *"an upper
   bound"* and drifts as rows land.
7. **Live-BQ tests need a skip guard.** The existing suite runs offline against
   `_schema_snapshot.json`; a dry-run test needs real ADC. Gate it
   (`pytest.mark.skipif` on credential availability) or the suite goes red in any
   credential-less context — but then also assert the guard itself didn't skip everything.
8. **Five sibling closures share the shape.** `grep -c "except Exception"
   backend/slack_bot/jobs/_production_fns.py` = **10**; six are the job closures
   (`:69, :82, :124, :161, :199, :230, :257`). Fixing 82.39 does not fix them. If the
   contract wants them covered, that is a separate queued step (per
   `feedback_queue_discovered_defects_in_masterplan`).

### Where the masterplan step description is WRONG or STALE

| Step says | Measured reality |
|---|---|
| *"outcome tracking feeds agent memories (BM25) and the learning loop, so a long-dead rebuild may mean the reflection corpus has been frozen"* | **Wrong causal link.** `grep -n "outcome_tracking" backend/services/outcome_tracker.py` → **0 hits**. `nightly_outcome_rebuild` is not on the agent-memories path at all. `agent_memories` IS empty (0 rows, modified 2026-03-18) but because `settings.py:34 paper_learn_loop_enabled = False` (phase-35.1 DARK flag). **Fixing 82.39 will not unfreeze the reflection corpus.** The contract should say so explicitly rather than implying a benefit it cannot deliver. |
| *"around line 218-227"* | Correct as of 2026-08-05: factory `:209`, closure `:217`, SQL `:220-227`, `except` `:230`. |
| Scope limited to *"repair the query"* | **Incomplete.** The WRITE (`:237-261`) targets a table whose schema shares exactly one column (`ticker`) with what it writes, and its docstring `:240-241` documents a schema that has never existed. Fetch-only repair leaves the job writing 0 rows. |
| *"MEASURE how long by checking the job's own receipts/logs and the freshest row in whatever table it writes"* | Done: `outcome_tracking` has **0 rows, modified 2026-03-18** (never written). `git log -S` dates the phantom columns to the file's **first commit, `2301b977`, 2026-05-11** — **86 days**, and never correct at any point. |
| *"run it, assert the derived set non-empty, and fix or queue every member"* | Satisfiable: 2 members, both in this file, **nothing to queue** — but only within a recall envelope that misses 19 files. |
| Criterion 1 premise "a BigQuery dry run (which is not billed)" | **Confirmed** verbatim by source 1. `schema_oracle.dry_run()` (`:550`) works without `location=` on the `us-central1` table (tested live). |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL (7: 2 official Google docs via curl,
      1 peer-reviewed ICPC paper via pdfplumber, 1 Google SRE Book chapter, 3 practitioner)
- [x] 10+ unique URLs total (25: 7 read-in-full + 18 snippet-only)
- [x] Recency scan (2024-2026) performed + reported (2 complementary findings, 0 superseding)
- [x] Full pages/papers read, not abstracts (arXiv PDF text-extracted to 24,075 chars;
      cloud.google.com pages curl'd to 58,448 chars because they are JS-rendered)
- [x] file:line anchors for every internal claim, each with the command that measured it

Soft checks:
- [x] Internal exploration covered the closure, the job module, the scheduler wiring, the
      live schemas of both source and destination tables, the paging path, and the sweep
- [x] Contradictions noted (dry run validates names BUT cannot prove rows return — sources
      1/4 in tension, which is exactly why criteria 1 and 2 are separate)
- [x] Claims cited per-claim
- [ ] NOT covered: whether `realized_pnl_pct` is defined as a fraction or a percent at its
      WRITER (needed only if the contract chooses derivation option (b)); the five sibling
      fail-open closures were counted but not individually audited

---

## JSON gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 18,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Confirmed live: paper_trades has 18 columns, no `timestamp`, no `realized_pnl`; the current query 400s on a $0 dry run. Two NEW findings beyond the step description. (1) The WRITE side is broken too -- outcome_tracking's real 9-column schema shares only `ticker` with what make_outcome_write_fn emits, and its docstring documents a schema that never existed, so a fetch-only repair still writes 0 rows. (2) The step's claim that this froze the agent-memories reflection corpus is WRONG: outcome_tracker.py never references outcome_tracking; the corpus is empty because paper_learn_loop_enabled defaults False. Dead since 2026-05-11 (the file's first commit, git log -S) = 86 days; destination table has 0 rows. The naive rename STILL 400s (TIMESTAMP_TRUNC on a STRING column); SAFE.TIMESTAMP(created_at) dry-runs VALID. realized_pnl_pct is a PERCENT (-14.5..116.4), so aliasing it AS pnl silently changes units. _compute_outcomes CRASHES on a NULL pnl. Sweep returns 2 unknown columns, both in this file, but its recall misses 19 interpolated-SQL files.",
  "brief_path": "handoff/current/research_brief_82.39.md",
  "gate_passed": true
}
```
