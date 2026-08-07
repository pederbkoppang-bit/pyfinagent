# Research Brief -- Step 83.0 (tier: moderate)

**Status: IN PROGRESS** (write-first discipline; this file grows incrementally)
**Started:** 2026-08-07
**Researcher:** Layer-3 researcher (Opus 5)

## Topic
News corpus persistence: amend + run `scripts/migrations/add_news_sentiment_schema.py`,
make the fail-open BQ writer observable, make the corpus writer source-agnostic,
disclose the measured absence of FINNHUB/BENZINGA keys.

## Plan
1. Internal audit (backend/news/**, migration, screener:334, settings, tests, observability)
2. Live BQ confirmation that `news_articles` / `news_sentiment` are absent (read-only)
3. External research: BQ DDL idempotency, insert_rows_json semantics, fail-open observability,
   Python adapter registries, PIT news corpus schema design
4. Recency scan 2024-2026
5. Envelope

---

## Internal code inventory (running)

| File | Lines | Role | Status |
|---|---|---|---|
| `backend/news/registry.py` | 98 | **PEP 544 Protocol + `@register(name)` decorator; `_REGISTRY: dict[str, NewsSource]`; `get_sources(names)`; `clear_registry()`** | **ALREADY EXISTS -- criterion (c) needs NO new registry** |
| `backend/news/bq_writer.py` | 221 | 3 writers + `_insert_rows` fail-open | needs observability + `ingested_at` rename |
| `backend/news/fetcher.py` | 270 | `run_once()`, `_normalize()`, `StubSource` (`@register("stub")`) | emits `fetched_at` at :103; needs `ingested_at` + `provenance` |
| `backend/news/normalize.py` | 76 | `canonical_url` / `body_hash` (stdlib only) | unchanged |
| `backend/news/dedup.py` | 173 | intra-batch dedup | unchanged |
| `backend/news/sources/__init__.py` | 11 | side-effect imports finnhub, benzinga, alpaca | **import-time coupling risk** |
| `backend/news/sources/finnhub.py` | 160 | `@register("finnhub")`; key-gated `fetch()` returns `[]` when key empty | byte-unchanged (criterion d) |
| `backend/news/sources/benzinga.py` | 165 | `@register("benzinga")` | byte-unchanged (criterion d) |
| `backend/news/sources/alpaca.py` | 165 | `@register("alpaca")` | byte-unchanged |
| `scripts/migrations/add_news_sentiment_schema.py` | 147 | `CREATE TABLE IF NOT EXISTS` DDL x2 + `--dry-run` | needs amendment |
| `backend/tools/screener.py` | ~334 | `apply_news_to_score` live overlay | read-only reference |

### KEY INTERNAL FINDING 1 -- the registry already exists (criterion c is mostly satisfied)

`backend/news/registry.py:44` `_REGISTRY: dict[str, NewsSource] = {}` with a
`@register(name)` decorator (`registry.py:47-83`) and a `runtime_checkable`
`Protocol` (`registry.py:30-41`). Three adapters already register:
`finnhub.py:62`, `benzinga.py`, `alpaca.py`, plus `StubSource` at
`fetcher.py:185`. So "at least two distinct registered source adapters"
is ALREADY structurally true. The real work for (c) is the **import-chain
non-coupling assertion**, not building a registry.

### KEY INTERNAL FINDING 2 -- `fetched_at` appears in FOUR places, not one

The rename `fetched_at -> ingested_at` is NOT a single-line migration edit:
1. `scripts/migrations/add_news_sentiment_schema.py:9` (docstring) and `:68` (DDL column)
2. `backend/news/bq_writer.py:117` `_serialize_article` emits `"fetched_at"`
3. `backend/news/fetcher.py:100` `_normalize()` sets `fetched_at=_now_iso()`
4. `backend/news/fetcher.py:61` `NormalizedArticle` TypedDict declares `fetched_at: str`
   plus `fetcher.py:255` smoke assertion `assert a["fetched_at"]`

NOTE: `calendar_events` ALSO has a `fetched_at` column
(`bq_writer.py:200`, `_serialize_calendar_event`). That one must NOT be
renamed -- `calendar_events` is the one table that DOES exist live. A
blind repo-wide `fetched_at -> ingested_at` sed WOULD BREAK the live
calendar_events writer. This is the primary mutation/blast-radius trap.

### KEY INTERNAL FINDING 3 -- `insert_rows_json` and unknown columns

`bq_writer.py:89` calls `client.insert_rows_json(table_ref, list(rows))`.
The module docstring (`bq_writer.py:23-25`) claims "let BQ ignore unknown
keys ... tolerated on permissive". This is FALSE by default -- see the
external research section (`ignore_unknown_values` defaults to False, so
an unknown column returns a per-row error, which `_insert_rows:90-92`
swallows into `return 0`). This is exactly the silent-failure mode the
step describes and it means an ordering error (writer renamed before the
migration runs, or vice versa) produces ZERO observable difference today.

### KEY INTERNAL FINDING 4 -- LIVE BQ CONFIRMATION (read-only, 2026-08-07)

`python -c "from google.cloud import bigquery; ... list_tables('pyfinagent_data')"`
returns exactly **10** tables:

```
alt_13f_holdings, alt_congress_trades, alt_finra_short_volume,
calendar_events, llm_call_log, risk_intervention_log,
scraper_audit_log, sla_alerts, strategy_decisions, unified_sar_log
```

- `news_articles`: **ABSENT** (confirms the step premise)
- `news_sentiment`: **ABSENT** (confirms the step premise)
- `calendar_events`: PRESENT
- **`api_call_log`: ABSENT** -- NEW FINDING, out of scope for 83.0.
  `backend/services/observability/api_call_log.py:148` writes to
  `{project}.{dataset}.api_call_log` and `finnhub.py:129` calls
  `log_api_call(...)` on EVERY fetch. That writer has the identical
  fail-open swallow at `api_call_log.py:150-153`. So the observability
  table used to audit the news adapters is ITSELF missing. Recommend
  filing as its own masterplan step (standing rule: every discovered
  defect gets its own research-gated step).

### KEY INTERNAL FINDING 5 -- the observability idiom is a module-level buffer + accessor, NOT a metrics library

There is **no Prometheus/OpenTelemetry/`Counter` facility in this repo**.
`grep -rn "Counter\b|_COUNTERS|counters\[" backend/services/` returns only
`collections.Counter` used locally at `autonomous_loop.py:1575`. The
established idiom (`api_call_log.py`) is:

- module-level state + `threading.Lock` (`api_call_log.py:58-60`)
- a **test-observable accessor** `buffer_size()` (`:164-167`)
- a **test reset helper** `reset_buffer_for_test()` (`:170-183`)
- a warn-once latch `_WARNED_BQ_ABSENT` (`:41`, `:131-138`)
- an env guard `PYFINAGENT_TEST_NO_BQ == "1"` (`:125`, set by
  `backend/tests/conftest.py`) that hard-stops real BQ writes in tests

The 83.0 counter should follow THIS shape, not introduce a new dependency.

---

## External findings

### EXTERNAL FINDING 1 (DECISIVE) -- you CANNOT add a REQUIRED column to an existing BQ table

Google Cloud, *Modifying table schemas*, accessed 2026-08-07, verbatim:

> "If you add new columns to an existing table schema, the columns must be
> **NULLABLE or REPEATED**. **You cannot add a REQUIRED column to an existing
> table schema.** Adding a REQUIRED column to an existing table schema in the
> API or bq command-line tool causes an error. However, you can create a
> nested REQUIRED column as part of a new RECORD field."
>
> "**REQUIRED columns can be added only when you create a table while loading
> data, or when you create an empty table with a schema definition.**"

And on mode changes:

> "The only supported modification you can make to a column's mode is changing
> it from REQUIRED to NULLABLE... **You can't change a column's mode from
> NULLABLE to REQUIRED**... you must recreate the table with the updated
> column modes."

**Why this is decisive for 83.0:** criterion 2 demands a REQUIRED `provenance`
STRING column. That is ONLY achievable because `news_articles` /
`news_sentiment` do **not exist yet** (verified live above). The migration gets
exactly ONE shot: the REQUIRED column must be in the `CREATE TABLE` DDL. If
anyone creates the table first without `provenance NOT NULL`, the only remedy
is to DROP and recreate -- which needs owner approval per CLAUDE.md BQ rule 4.

**Implication for the migration design:** the "amend the DDL + re-run" strategy
is CORRECT *for this step only* because the tables are absent. But the amended
migration must NOT pretend to be a general schema-evolution tool -- see
Finding 2.

### EXTERNAL FINDING 2 -- `CREATE TABLE IF NOT EXISTS` is a NO-OP on an existing table

The current migration uses `CREATE TABLE IF NOT EXISTS`
(`add_news_sentiment_schema.py:65`, `:90`). Per the BigQuery DDL reference
(accessed 2026-08-07): `CREATE TABLE IF NOT EXISTS` "creates a table only if
the table doesn't currently exist in the specified dataset. **If the table
name exists in the dataset, no error is returned, and no action is taken.**"

So amending the DDL text alone is INERT if the table already exists. Today the
tables are absent so the amended DDL applies; but a re-run after creation will
silently do nothing and the migration will still print "OK: ... ready." That
is the same class of silent success the step is complaining about. The
migration should therefore **verify the post-condition** (read the schema back
and assert `ingested_at` + `provenance` present with the right mode) rather
than trust the DDL's exit code.

### EXTERNAL FINDING 3 -- `ADD COLUMN` / `RENAME COLUMN` cannot touch partition or cluster columns

BigQuery DDL reference (accessed 2026-08-07), `ALTER TABLE ADD COLUMN`:

> "You cannot use this statement to create: **Partitioned columns. Clustered
> columns.** Nested columns inside existing RECORD fields."
> "**You cannot add a REQUIRED column to an existing table schema.**"

`ALTER TABLE RENAME COLUMN`:

> "You cannot use this statement to rename the following: Subfields... **Partitioning
> columns**, **Clustering columns**, Fields that are part of primary key
> constraints..."
> "After one or more columns in a table are renamed, you cannot... Query the table
> with legacy SQL. Query the table as a wildcard table."
> "`RENAME COLUMN` cannot be used with other ALTER TABLE actions in one statement."

**Mapped to our DDL** (`add_news_sentiment_schema.py:81-82`): `published_at` is
the PARTITION column and `source`, `ticker` are the CLUSTER columns.
`fetched_at` is neither, so a `RENAME COLUMN fetched_at TO ingested_at` WOULD be
legal if the table existed. It does not, so the clean path is to amend the
`CREATE TABLE` body directly. Record this anyway: it is the fallback if someone
creates the table before the amended migration lands.

### EXTERNAL FINDING 4 (REFUTES A CODE COMMENT) -- `insert_rows_json` does NOT ignore unknown columns by default

Read in full from the **installed** SDK (`.venv/.../google/cloud/bigquery/client.py`,
`Client.insert_rows_json` docstring + body) and cross-checked against the
`tabledata.insertAll` REST reference (accessed 2026-08-07):

> `ignore_unknown_values (Optional[bool])`: "Accept rows that contain values that
> do not match the schema. The unknown values are ignored. **Default is `False`,
> which treats unknown values as errors.**"
> `skip_invalid_rows (Optional[bool])`: "Insert all valid rows of a request, even
> if invalid rows exist. **The default value is `False`, which causes the entire
> request to fail if any invalid rows exist.**"

The body confirms both keys are omitted from the request payload unless
explicitly passed (`if skip_invalid_rows is not None: data[...]`).

`bq_writer.py:89` passes NEITHER, so both are `False`. Therefore:
- a writer emitting `fetched_at` against a schema with `ingested_at` -> **unknown
  value -> error**
- a writer omitting REQUIRED `provenance` -> **invalid row -> WHOLE BATCH fails**

**This REFUTES the module docstring at `bq_writer.py:23-25`**, which claims "let
BQ ignore unknown keys (BQ streaming inserts warn on unknown columns in strict
mode; tolerated on permissive)". There is no "permissive mode" in play -- the
default is strict, and the failure is silent because `_insert_rows:90-92`
converts the returned errors into `return 0` + a WARNING log.

Also from the REST reference, on `rows[].insertId`:

> "Insertion ID for best-effort deduplication. **This feature is not recommended**,
> and users seeking stronger insertion semantics are encouraged to use other
> mechanisms such as the BigQuery Write API."

Note the SDK auto-generates a uuid4 `insertId` per row by default
(`AutoRowIDs.GENERATE_UUID`), which is what makes the call retry-safe.

### EXTERNAL FINDING 5 -- silent failure: returning `0` is indistinguishable from success

*Encyclopedia of Agentic Coding Patterns*, "Silent Failure" (accessed 2026-08-07):

> A silent failure is "**a defect that produces no signal at the moment it
> occurs**."
> Returning empty or zero-valued defaults on failure is dangerous because it
> becomes "**indistinguishable from no result**" -- "A function returns `[]` for
> 'no rows matched' and `[]` for 'the database connection failed.'"

Remediation named: **absence alerts** ("expected things not happening" rather
than only error rates) and **invariant checks** ("converting silent failures into
loud ones through assertions").

This is `_insert_rows` exactly: `return 0` when `rows` is empty
(`bq_writer.py:82-83`), `return 0` when the client is absent (`:85-86`),
`return 0` on insert errors (`:90-92`), and `return 0` on any exception
(`:94-96`). **Four distinct meanings, one return value.** A counter is the
minimum fix; the counter must be keyed so "nothing to write" is not conflated
with "write rejected".

### EXTERNAL FINDING 6 -- registration decorators create import-time coupling

*Don't Use This Code*, "Decorators: Registration Pattern" (2024-05-22, accessed
2026-08-07):

> "**We can't make assumptions about the sequence, the ordering, or the timing of
> that execution**" across module imports.
> "If you are in app.py, and you see f, and you're curious how it works, you'll
> track it back to its definition by looking for the import line. This will take
> you to lib.py. **However, the behavior that you're observing is from
> otherlib.py.**"

The corroborating search-tier consensus (DEV/RealPython/Medium registry-pattern
posts): "decorators only run when the file containing them is imported... which
forces you to import modules you aren't explicitly using just to trigger the
side-effect of registration."

**Mapped to us:** `backend/news/sources/__init__.py:7-9` unconditionally imports
finnhub, benzinga, alpaca; `backend/news/__init__.py:34` imports that package.
So `import backend.news` transitively imports all three adapters. Today none of
them import Alpha Vantage, so criterion 5's import assertion passes -- but it
passes by ACCIDENT, not by construction. See the contract notes for how to make
it hold by construction.

### EXTERNAL FINDING 7 -- provenance = the record of how an artifact came to be

ACL NLP4PI 2025, *Dataset of News Articles with Provenance Metadata*
(read in full via pdfplumber, 14pp; accessed 2026-08-07). **Relevance caveat,
stated honestly: this paper is about IMAGE provenance for misinformation
detection, NOT financial corpus design.** Its usable contribution is the working
definition:

> "provenance metadata -- **a record of a file's existence from its creation
> through edits to distribution**"

and the task framing that the useful question is not "is this relevant?" but
"**was this captured at a time and place that is relevant?**" -- which is exactly
the live-vs-backfill discrimination criterion 2 demands.

---

## Recency scan (2024-2026) -- PERFORMED

Queries run (the mandated three variants, per topic):
- current-year frontier: `... 2026` (BQ DDL idempotency; insert_rows_json; python registry)
- last-2-year window: `... 2024 2025 2026` (BQ schema migration idempotency + INFORMATION_SCHEMA.COLUMNS)
- year-less canonical: `W3C PROV data provenance wasGeneratedBy ingestion time versus event time`;
  `observability swallowed exception counter fail-open anti-pattern`

**Result: 3 new findings in the 2024-2026 window that COMPLEMENT (none that
supersede) the canonical vendor docs.**

1. **`INFORMATION_SCHEMA.COLUMNS` as the post-condition oracle** (2026 sources):
   "The INFORMATION_SCHEMA.COLUMNS view contains one row for each column (field)
   in a table... can be used to query and verify schema changes after migration
   to ensure post-conditions are met." This is the 2024-2026 consensus answer to
   "how do I make a migration re-runnable AND provable". It directly supplies the
   missing half of our migration (Finding 2).
2. **Parametric look-ahead bias** (arXiv:2601.13770 *Look-Ahead-Bench*, submitted
   2026-01-20; abstract page read, full text not fetched -- HTML 404 on v1/v2/v3,
   counted SNIPPET-ONLY). Relevant as context: even a perfectly point-in-time
   corpus does not neutralise bias that "lives inside the model's weights". Our
   `news_sentiment.scorer_model` + `scorer_version` columns
   (`add_news_sentiment_schema.py:91-93`) already capture what is needed to
   audit this later -- a point worth preserving, not a change request.
3. **Look-ahead-bias acknowledgment is rare**: of 164 papers reviewed 2023-2025,
   "only 26.8% acknowledge look-ahead bias" (search-tier finding). Supports
   making `provenance` REQUIRED rather than NULLABLE-with-convention: a nullable
   column will be left null.

**No 2024-2026 source contradicts** the two load-bearing vendor facts (cannot add
a REQUIRED column to an existing table; `ignore_unknown_values` defaults to
False). Both were re-verified against current official docs today.

---

## Consensus vs debate

- **Consensus (strong):** `CREATE TABLE IF NOT EXISTS` is not a schema-evolution
  tool; `ALTER TABLE ... IF [NOT] EXISTS` is the idempotent primitive; verify via
  `INFORMATION_SCHEMA.COLUMNS` / `get_table().schema`. Vendor docs and all
  practitioner sources agree.
- **Consensus (strong):** swallowing an error into a sentinel return is an
  anti-pattern precisely because the sentinel is ambiguous.
- **Debate (mild):** registry-by-decorator vs entry-points. Entry-points decouple
  packages but "scales badly with the number of installed distributions and can
  be very slow"; decorators are fast but import-coupled. **For 83.0 this debate is
  moot** -- a working decorator registry already exists and the criterion only
  requires two registered adapters plus an import-chain assertion. Do not migrate
  to entry points in this step.

## Pitfalls (from literature + code)

1. Amending DDL text without a post-condition read-back = silent no-op on re-run.
2. Adding `provenance` as REQUIRED **after** the table exists is impossible
   without a drop/recreate (owner-approval gated).
3. A counter that only proves "the attribute exists" is not mutation-resistant --
   the criterion explicitly demands a strict numeric increase.
4. A `MagicMock` BQ client that returns `[]` makes a broken writer look healthy
   (this repo already learned this: `test_phase_82_48_outcome_write_schema.py:15-20`).
5. A repo-wide `fetched_at -> ingested_at` rename breaks the LIVE `calendar_events`
   writer.

## Application to pyfinagent

### SCOPE FINDING (READ BEFORE WRITING THE CONTRACT) -- the live path and the corpus path are DISJOINT

The step's framing -- "news influences live trades but NO news corpus is
persisted" -- is TRUE, but the two halves live in **different modules that never
touch each other**. Measured by grep over `backend/` + `scripts/`:

**Live path (moves the real book):**
`autonomous_loop.py:476` (`if getattr(settings, "news_screen_enabled", False)`)
-> `backend/services/news_screen.py::fetch_news_signals` -> `screener.py:334`
`apply_news_to_score`. That module fetches **RSS feeds** (`news_screen.py:90`
`_parse_rss`, `:126` `_fetch_all_feeds`), classifies with an LLM
(`:179` `_build_batch_prompt`, model `news_screen_model` = `claude-haiku-4-5`,
`settings.py:410`), and caches to a **LOCAL FILE** (`:202` `_cache_path`,
`:207` `_load_cache`, `:222` `_save_cache`). **It never imports
`backend.news`, never calls `bq_writer`, and writes NOTHING to BigQuery.**

**Corpus path (what 83.0 fixes):** `backend/news/` (registry + fetcher +
bq_writer + sentiment). Its ONLY consumers outside the package are
`backend/tests/test_bq_writer.py:16`, `backend/tests/test_sentiment_ladder.py:19`
and `scripts/smoketest/phase6_e2e.py:106,131,148,167,195`. **It is not scheduled
and not called by `autonomous_loop`.**

**Consequence Main must plan around:** completing 83.0 as written creates the
tables and makes the corpus writer observable + source-agnostic, but does **NOT**
capture the headlines the live screener actually acted on. The un-audited input
moving the book stays un-audited. 83.0 remains correct and necessary (it is the
schema prerequisite), but the contract should state this explicitly rather than
imply the defect is closed, and the "wire `news_screen` into the corpus" work
should be filed as its **own** research-gated step (standing rule:
`feedback_queue_discovered_defects_in_masterplan`).

### Blast radius of the `fetched_at` -> `ingested_at` rename -- MEASURED: 16 sites in 8 files

Derivation rule (stated so it is reproducible, per
`feedback_normalization_rule_must_be_stated_with_the_ratio`):
`grep -rn "fetched_at" --include="*.py" backend scripts`, run 2026-08-07,
then partitioned by which TABLE the occurrence serves. 16 hits total:
**9 must change, 7 must NOT.**

MUST CHANGE (the `news_articles` path) -- 9 sites:
| File:line | What |
|---|---|
| `scripts/migrations/add_news_sentiment_schema.py:68` | DDL column |
| `scripts/migrations/add_news_sentiment_schema.py:9` | docstring |
| `backend/news/bq_writer.py:117` | `_serialize_article` -- BOTH the emitted key and the `article.get(...)` lookup are on this line |
| `backend/news/fetcher.py:62` | `NormalizedArticle` TypedDict field |
| `backend/news/fetcher.py:100` | `_normalize()` assignment |
| `backend/news/fetcher.py:255` | inline smoke assertion `assert a["fetched_at"]` |
| `backend/tests/test_bq_writer.py:28` | `_NEWS_ARTICLES_FIELDS` set |
| `backend/tests/test_bq_writer.py:65` | article fixture in `test_write_news_articles_fail_open_no_bq_auth` |
| `backend/tests/test_bq_writer.py:91` | article fixture in `test_serialize_article_produces_expected_fields` |

**Two existing tests go RED** if the test file is missed:
`test_serialize_article_produces_expected_fields` (`:105`) and
`test_serialize_article_handles_missing_fields` (`:115`) both assert
`set(row.keys()) == _NEWS_ARTICLES_FIELDS` **exactly** -- an equality on a
set, so an added `provenance` key ALSO breaks them even without the rename.
Both changes must land in the same commit as the writer change.

MUST NOT CHANGE (the `calendar_events` path -- **the trap**) -- 7 sites:
`backend/news/bq_writer.py:200`, `backend/tests/test_bq_writer.py:40`,
`backend/tests/test_bq_writer.py:192`, `backend/tests/test_calendar_watcher.py:245`,
`backend/econ_calendar/watcher.py:51`, `backend/econ_calendar/watcher.py:92`,
`scripts/migrations/add_calendar_events_schema.py:49`.

`calendar_events` is the one table that **DOES exist live**. A blind
`sed -i s/fetched_at/ingested_at/` breaks its writer AND the econ-calendar
watcher, and because `_insert_rows` swallows the error it breaks **silently**.
Recommend the contract require a post-edit assertion that exactly these 7
lines are unchanged (`git diff` scoped to those paths).

### Registry + import-coupling: MEASURED, not assumed

`grep -rn "@register("` returns four live registrations:
`fetcher.py:185` (`stub`), `sources/finnhub.py:62` (`finnhub`),
`sources/benzinga.py` (`benzinga`), `sources/alpaca.py` (`alpaca`).
(A fifth hit, `registry.py:7`, is inside the module docstring, not a
registration.) So criterion 5's "at least two distinct registered source
adapters" is satisfied by existing code.

`grep -rni "alphavantage|alpha_vantage" backend/news/ scripts/migrations/add_news_sentiment_schema.py`
returns **NONE** -- the import-chain assertion holds today.

`calendar_events` is the one table that **DOES exist live**. A blind
`sed -i s/fetched_at/ingested_at/` across the repo breaks its writer, and
because `_insert_rows` swallows the error, it would break **silently**.

### Measured key absence (criterion 6 disclosure)

```
FINNHUB_API_KEY:      ABSENT_FROM_ENV
BENZINGA_API_KEY:     ABSENT_FROM_ENV
ALPHAVANTAGE_API_KEY: SET (len=16)
ALPACA_API_KEY:       SET (len=26)
```
(`backend/.env`, checked 2026-08-07 by `grep -q "^KEY="`.) Both adapters
degrade to an empty iterator when the key is empty (`finnhub.py:68-71`:
`key = settings.finnhub_api_key or ""` then `if not key: ... return`), so
they are reachable-but-inert, not import-time failures. Note for the artifact:
`settings.py:129-130` defaults both to `""`, so the adapters never raise --
this is why the dead code has been invisible.

### NEW DEFECT discovered (out of scope -- file as its own step)

`api_call_log` is **ABSENT** from `pyfinagent_data` (live listing above), yet
`finnhub.py:129` calls `log_api_call(...)` on every fetch and
`api_call_log.py:150-153` swallows the resulting insert errors into
`return 0` + a WARNING. The telemetry table meant to audit the news adapters
does not exist. Same defect class as 83.0, different table.

---

## Contract-ready implementation notes

### 1. Migration amendment strategy (`scripts/migrations/add_news_sentiment_schema.py`)

Because the tables are **absent** (verified live), amend the `CREATE TABLE` body
directly -- this is the only window in which a REQUIRED column can be created.

```
-- news_articles, replacing line 68
  ingested_at TIMESTAMP NOT NULL,       -- renamed from fetched_at
  provenance STRING NOT NULL,           -- {live, backfill}
```
Add the same `provenance STRING NOT NULL` to `news_sentiment` only if the
contract wants it there; criterion 2 names `news_articles` schema, and the
live_check names both tables' schemas -- **Main should decide explicitly and
state it**, because it cannot be added later (Finding 1).

Keep `CREATE TABLE IF NOT EXISTS` but make the migration **prove its
post-condition** rather than trust the DDL:

```python
def _verify(client, fq_table, required: dict[str, str]) -> None:
    schema = {f.name: f.mode for f in client.get_table(fq_table).schema}
    assert schema, f"{fq_table}: empty schema read"      # never a vacuous oracle
    for col, mode in required.items():
        got = schema.get(col)
        if got != mode:
            raise SystemExit(
                f"MIGRATION POST-CONDITION FAILED {fq_table}.{col}: "
                f"want mode={mode}, got {got!r}. "
                "CREATE TABLE IF NOT EXISTS is a NO-OP on an existing table; "
                "a REQUIRED column cannot be added afterwards -- the table "
                "must be dropped and recreated (owner approval required)."
            )
```
Call `_verify` for both tables after the DDL loop AND in `--dry-run` skip it.
Required map for `news_articles`:
`{"published_at": "REQUIRED", "ingested_at": "REQUIRED", "provenance": "REQUIRED"}`.
This is what makes the migration honestly re-runnable: re-running on a correct
table is a green no-op; re-running on a drifted table FAILS LOUD instead of
printing "OK: ... ready."

Do NOT add `ALTER TABLE ADD COLUMN` for `provenance` -- it cannot create a
REQUIRED column (Finding 3) and would silently produce a NULLABLE one.

### 2. Counter design (mutation-resistant, matches the house idiom)

Follow `api_call_log.py`'s module-state + accessor + reset shape. Put it in
`bq_writer.py` so the failure branch and the counter cannot drift apart:

```python
_WRITE_FAILURES: dict[str, int] = {}          # table -> count
_write_lock = threading.Lock()

def write_failure_count(table: str | None = None) -> int:
    with _write_lock:
        return _WRITE_FAILURES.get(table, 0) if table else sum(_WRITE_FAILURES.values())

def reset_write_failures_for_test() -> None:
    with _write_lock:
        _WRITE_FAILURES.clear()

def _record_failure(table: str, reason: str, detail: str) -> None:
    with _write_lock:
        _WRITE_FAILURES[table] = _WRITE_FAILURES.get(table, 0) + 1
        n = _WRITE_FAILURES[table]
    logger.warning(
        "bq_writer write FAILED table=%s reason=%s failures=%d detail=%s",
        table, reason, n, detail,          # ASCII only -- .claude/rules/security.md
    )
```
Call `_record_failure` at **all three** real failure branches in `_insert_rows`
(client-absent `:85-86`, insert-errors `:90-92`, exception `:94-96`) with
distinct `reason` values (`client_absent` / `insert_errors` / `exception`), and
**NOT** at the empty-input early return `:82-83` -- "nothing to write" is not a
failure, and conflating them re-creates the ambiguity Finding 5 names.

Mutation targets to name in the contract (each must turn the test RED):
- delete the `_WRITE_FAILURES[table] = ... + 1` line -> counter test fails
- delete the `logger.warning(...)` -> caplog test fails
- change `+ 1` to `+ 0` -> strict-increase assert fails
- move `_record_failure` into the empty-input branch -> negative control fails

### 3. Test-fixture strategy (how to fake the BQ client)

New file: `backend/tests/test_phase_83_0_news_corpus_persistence.py` (name fixed
by the verification command). `backend/tests/conftest.py:21` already sets
`PYFINAGENT_TEST_NO_BQ=1`, so nothing reaches real BQ by default.

Fake the client at the `_get_client` seam, not at `google.cloud.bigquery`:

```python
class _FakeClient:
    def __init__(self, errors): self.errors = errors; self.calls = []
    def insert_rows_json(self, table_ref, rows):
        self.calls.append((table_ref, rows))
        return self.errors        # [] == success; non-empty == BQ rejected

def test_swallowed_write_increments_counter_and_logs(caplog):
    bq_writer.reset_write_failures_for_test()
    fake = _FakeClient(errors=[{"index": 0, "errors": [{"reason": "invalid"}]}])
    before = bq_writer.write_failure_count("news_articles")
    with patch.object(bq_writer, "_get_client", return_value=fake), \
         caplog.at_level(logging.WARNING, logger="backend.news.bq_writer"):
        n = bq_writer.write_news_articles([_ROW])      # must NOT raise
    assert n == 0                                       # still fail-open
    assert bq_writer.write_failure_count("news_articles") > before   # STRICT increase
    assert any("write FAILED" in r.message or "write FAILED" in r.getMessage()
               for r in caplog.records)                 # BOTH halves, one test
```
Criterion 3 requires both halves **in the same test** -- keep them in one
function. Add a **negative control** (`_FakeClient(errors=[])` -> counter
unchanged) so the guard is not vacuously true.

For the schema assertions (criteria 1 + 2) prefer reading the LIVE schema with a
checked-in snapshot fallback, exactly as
`test_phase_82_48_outcome_write_schema.py:64-90` does -- and assert the resolved
schema map is **non-empty** first, or an empty oracle makes every shape valid.

### 4. Provenance plumbing (criterion 2 needs two DIFFERENT values)

The criterion fails if "both paths emit the same value". `_normalize`
(`fetcher.py:94`) currently has no provenance concept. Minimal change:

```python
def _normalize(raw, source_name, provenance: str = "live") -> NormalizedArticle:
    ...
    provenance=provenance,
```
and thread a `provenance: str = "live"` parameter through
`run_once(...)`, with the backfill entry point passing `"backfill"`.
`_serialize_article` (`bq_writer.py:114-129`) must emit the key -- and because
`provenance` is REQUIRED, a missing value fails the WHOLE batch (Finding 4), so
default it defensively: `article.get("provenance") or "live"`. Consider a
module-level `_VALID_PROVENANCE = frozenset({"live", "backfill"})` guard, since
BigQuery has no CHECK constraint to enforce the enum.

### 5. Source-agnostic assertion (criterion 5) -- make it hold by CONSTRUCTION

The registry already exists (`registry.py:44-93`); do NOT build a second one.
Two assertions:

(a) two distinct registered adapters accepted by the writer -- use
`clear_registry()` (`registry.py:96`) then register two throwaway stub classes,
run `run_once([...], dry_run=True)`, and assert rows from both reach
`_serialize_article` with distinct `source` values. Note `register()` raises
`ValueError` on duplicate names under a *different* class (`registry.py:69-73`),
so `clear_registry()` must run in a fixture with teardown that re-imports
`backend.news.sources` -- otherwise the wiped registry leaks into later tests.

(b) the import-chain assertion. Do it **structurally**, not by string-matching
the source (a source-text grep is the "guards stop one seam short" failure mode):

```python
def test_migration_and_writer_do_not_require_alphavantage(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    for name in list(sys.modules):
        if name.startswith(("backend.news.bq_writer", "scripts.migrations")):
            sys.modules.pop(name, None)
    importlib.import_module("backend.news.bq_writer")
    mod = importlib.import_module("scripts.migrations.add_news_sentiment_schema")
    reached = {m for m in sys.modules if "alphavantage" in m.lower()}
    assert not reached, f"alphavantage module pulled in: {reached}"
```
Caveat to state in the contract: today this passes trivially (no adapter imports
Alpha Vantage). Its value is as a **regression tripwire** for the operator's
pending licence decision. Note the migration currently imports
`backend.config.settings` at module scope (`add_news_sentiment_schema.py:61`),
which is fine -- `settings.py:122` merely *declares* `alphavantage_api_key`; it
does not require it to be set.

### 6. live_check evidence to capture

The live_check demands BEFORE and AFTER. The BEFORE capture is **perishable** --
once the migration runs it cannot be reproduced. Capture it FIRST:
```
bq show --schema sunny-might-477607-p8:pyfinagent_data.news_articles   # expect: Not found
bq show --schema sunny-might-477607-p8:pyfinagent_data.news_sentiment  # expect: Not found
```
The read-only listing in this brief (Finding 4) is a valid BEFORE artifact if
`bq` CLI output is captured in the same session.

---

## Read in full (8; gate floor is 5)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://cloud.google.com/bigquery/docs/managing-table-schemas | 2026-08-07 | official doc | curl + tag-strip (117KB) | "You cannot add a REQUIRED column to an existing table schema"; NULLABLE->REQUIRED impossible |
| 2 | https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language | 2026-08-07 | official doc | curl + tag-strip (331KB) | `ADD COLUMN IF NOT EXISTS` / `RENAME COLUMN IF EXISTS` semantics; cannot touch partition/cluster cols |
| 3 | https://cloud.google.com/bigquery/docs/reference/rest/v2/tabledata/insertAll | 2026-08-07 | official API ref | curl + tag-strip (37KB) | `ignoreUnknownValues`/`skipInvalidRows` default false; insertId "not recommended" |
| 4 | google-cloud-bigquery installed SDK, `Client.insert_rows_json` (`.venv/.../bigquery/client.py`) | 2026-08-07 | vendor source | `inspect.getsource` | Confirms both flags omitted unless passed -> strict by default; REFUTES `bq_writer.py:23-25` |
| 5 | https://aipatternbook.com/silent-failure | 2026-08-07 | practitioner ref | WebFetch | Zero/empty default return "indistinguishable from no result"; remedy = absence alerts + invariant checks |
| 6 | https://www.dontusethiscode.com/blog/2024-05-22_registration-decorators.html | 2026-08-07 | authoritative blog | WebFetch | Registration decorators: "can't make assumptions about the sequence, the ordering, or the timing" |
| 7 | https://aclanthology.org/2025.nlp4pi-1.10.pdf | 2026-08-07 | peer-reviewed (ACL NLP4PI 2025) | curl + **pdfplumber** (14pp) | Provenance = "record of a file's existence from its creation through edits to distribution". *Image-domain; relevance caveat stated* |
| 8 | https://oneuptime.com/blog/post/2026-02-17-how-to-handle-schema-evolution-in-bigquery-when-source-schemas-change-frequently/view | 2026-08-07 | practitioner blog | WebFetch | Defensive patterns (COALESCE/SAFE_CAST); does NOT answer the IF-NOT-EXISTS question -- recorded as a negative result |

## Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2601.13770 | preprint | `/html/` 404 on v1/v2/v3; abstract page read only. LLM-weight bias, not corpus schema |
| https://arxiv.org/html/2605.24564 | preprint | Look-ahead bias mitigation via LLMs; off-axis for schema design |
| https://arxiv.org/html/2606.12210 | preprint | Zero-shot financial NLP limits; off-axis |
| https://arxiv.org/pdf/2601.19191 | preprint | Clinical NLP provenance/datasheets; adjacent domain, budget |
| https://arxiv.org/pdf/1605.01229 | preprint | Provenance-metadata lifecycle; canonical but superseded by W3C PROV for our need |
| https://dvcs.w3.org/hg/prov/.../prov-dm/overview/index.html | standard | W3C PROV overview; concept confirmed via search tier |
| https://cloud.google.com/bigquery/docs/information-schema-columns | official doc | Post-condition oracle; behaviour confirmed via #2 + search tier |
| https://cloud.google.com/bigquery/docs/migration/schema-data-overview | official doc | Migration overview; general |
| https://github.com/medjed/bigquery_migration | code | Ruby migration tool; not our stack |
| https://github.com/sonots/bigquery_migration | code | ditto |
| https://github.com/googleapis/google-cloud-python/issues/7294 | issue | KeyError on missing nullable field; superseded by #4 |
| https://realpython.com/lessons/registering-plugins-decorators/ | video lesson | Registry pattern; covered by #6 |
| https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm | blog | ditto |
| https://dev.to/borisuu/python-entry-points-1idk | blog | entry-points alternative; out of scope for 83.0 |
| https://github.com/FelixSchwarz/puzzle-plugin-system | code | setuptools entry-point plugin system |
| https://pypi.org/project/reentry/1.0 | package | entry-point perf workaround |
| https://medium.com/@abohievanson/exception-swallowing-kills-observability-... | blog | Community tier; covered by #5 |
| https://www.frugaltesting.com/blog/how-to-detect-silent-failures-in-microservices-... | blog | ditto |
| https://lightrun.com/blog/mttd-mean-time-to-detect/ | blog | MTTD; tangential |
| https://oneuptime.com/blog/post/2026-02-17-how-to-troubleshoot-bigquery-streaming-insert-rows-not-appearing-in-table-queries/view | blog | Streaming-buffer visibility; not load-bearing here |
| https://www.owox.com/blog/articles/bigquery-modify-tables-data-definition-language | blog | Vendor-adjacent; superseded by #2 |
| https://www.owox.com/blog/articles/bigquery-create-drop-tables-data-definition-language | blog | ditto |
| https://thisisjayakumar.medium.com/guidance-to-schema-migration-while-on-boarding-to-big-query-... | blog | Recency-scan hit |
| https://dev.to/plugarut/schema-changes-and-data-migration-in-bigquery-24gh | blog | Recency-scan hit |
| https://reintech.io/blog/google-bigquery-schema-design-best-practices | blog | Recency-scan hit |
| https://www.sentinelone.com/cybersecurity-101/data-and-ai/data-provenance/ | vendor | Data-provenance primer |
| https://arxiv.org/pdf/2107.09966 | preprint | Provenance + anonymisation formalism; off-axis |

**Total unique URLs collected: 35** (8 read in full + 27 snippet-only).

### Internal-only sources cited (not counted as external)
- `handoff/current/research_prompt_market_news.md` -- the 2026-08-04 phase-83
  research prompt; confirms the licence/source work is already settled and that
  83.1 (not 83.0) owns theme representation. Cited, not re-derived.
- `handoff/current/phase83_research_raw/{research,synthesis,verdicts}.json` --
  the 2026-08-04 licence findings (Alpha Vantage non-commercial ToS + 25 req/day
  + the >=19x Jan-2025 vs Jan-2026 density discontinuity; GDELT / SEC EDGAR /
  GPR as the measured-clean alternatives). Cited per instruction, not re-derived.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL -- **8** (2 via curl+tag-strip per the gcloud-docs rule, 1 via pdfplumber per the PDF chain, 1 via installed source, 4 via WebFetch)
- [x] 10+ unique URLs total -- **35**
- [x] Recency scan (2024-2026) performed + reported -- 3 complementary findings, 0 superseding
- [x] Full pages read (not abstracts) for the read-in-full set -- the one abstract-only source (arXiv:2601.13770) is recorded as SNIPPET-ONLY and excluded from the count
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module named in the spawn prompt, plus `news_screen.py`, `autonomous_loop.py:476`, `test_bq_writer.py`, `conftest.py`
- [x] Contradictions noted -- Finding 4 REFUTES the `bq_writer.py:23-25` docstring; the SCOPE FINDING qualifies the step's own framing
- [x] Claims cited per-claim with URL + access date or file:line

Three-variant query discipline: current-year (`...2026`), last-2-year
(`...2024 2025 2026`), and year-less canonical (`W3C PROV ... ingestion time
versus event time`; `observability swallowed exception counter fail-open
anti-pattern`) all run; the source tables mix all three.

---

## Envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 27,
  "urls_collected": 35,
  "recency_scan_performed": true,
  "internal_files_inspected": 16,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Live BQ listing confirms news_articles + news_sentiment ABSENT (10 tables in pyfinagent_data). Two decisive vendor facts: BigQuery CANNOT add a REQUIRED column to an existing table, so the absent tables are the ONLY window to create provenance NOT NULL; and insert_rows_json defaults ignore_unknown_values=False AND skip_invalid_rows=False, so a column mismatch fails the WHOLE batch -- refuting the bq_writer.py:23-25 docstring. CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so the migration must verify its post-condition via get_table().schema. SCOPE: the live news path (news_screen.py, RSS + LLM + local file cache, autonomous_loop.py:476 -> screener.py:334) is DISJOINT from backend/news/ -- fixing bq_writer does not capture what moved the book. registry.py already provides the adapter registry. The fetched_at rename touches 5 sites and MUST NOT hit calendar_events (the one live table). FINNHUB/BENZINGA keys absent; ALPHAVANTAGE set. New defect: api_call_log table also absent.",
  "brief_path": "handoff/current/research_brief_83.0.md",
  "gate_passed": true
}
```
