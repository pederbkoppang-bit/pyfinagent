# Dataform / DB Orchestrator -- Feasibility Study

**Phase:** 23.0
**Date:** 2026-04-26
**Operator question:** "Would we benefit from Dataform or a DB orchestrator
in BigQuery to orchestrate our data?"
**Status:** Decision document. NO IMPLEMENTATION.
**Research brief:** `handoff/current/phase-23.0-research-brief.md` (16 KB,
6 external sources read in full, 16 URLs, 8 internal files audited).

---

## Recommendation

**DON'T ADOPT** -- you don't have the pain points these tools solve.

pyfinagent has no SQL transformation layer. All derived data is computed
in Python and written to BigQuery as already-derived scalar fields. The
cron orchestration is Python-native (APScheduler) and works correctly.
Adding Dataform / dbt / Airflow would introduce a second tool to
maintain without replacing any existing one, and would not eliminate any
current pain.

Reconsider in the future if specific signals surface (see "Early signals"
section below).

---

## TL;DR

| Question | Answer |
|----------|--------|
| Are derived BQ columns currently computed in SQL? | NO -- all in Python (`outcome_tracker.py:51-70` etc.) |
| Are there any BigQuery scheduled queries? | NO |
| How many migrations exist today? | 17 in `scripts/migrations/`, idempotent CREATE TABLE IF NOT EXISTS pattern |
| Is there a multi-developer onboarding problem? | NO -- solo developer + 1 Claude Code session |
| Does Dataform replace any current tool? | NO -- you'd add it alongside APScheduler |
| Is Airflow/Composer warranted? | NO -- definitively overkill for single-stack Python+BQ |
| Cost if you adopted Dataform? | $0 licensing + BQ compute (already paid) -- the cost is engineering time, not dollars |

---

## Why each option doesn't fit today

### Dataform

**What it offers:** SQLX (SQL with `ref()` for DAG dependencies), version-controlled Git
workflows, Gemini AI integration baked in, BQ-native scheduling, IAM via BQ.

**Why not yet:** You have zero SQLX-eligible work. Every "derived" column today
(`return_pct`, `holding_days`, `directionally_correct`, `beat_benchmark` in
`outcome_tracker.py:51-70`) is computed in Python before the BQ write. There's
nothing to migrate INTO Dataform.

**When it would make sense:** Once you have 10+ derived columns that are pure
SQL transformations (e.g., rolling 30-day Sharpe over `paper_trades`,
sector-rotation buckets, drawdown calculation). At that point you'd be
duplicating SQL logic across multiple Python files and a single source of
truth in `.sqlx` would pay off.

### dbt (dbt-bigquery + dbt Cloud)

**What it offers:** Same SQL transformation model as Dataform, multi-warehouse
portability (Snowflake, Redshift, etc.), broader community.

**Why not over Dataform:** You will never run anywhere except BigQuery (cost
discipline; project memory `project_local_only_deployment.md`). dbt's
multi-warehouse advantage is irrelevant. dbt Cloud is ~$100/user/month;
local dbt-core is free but adds another binary + lockfile to manage.
Dataform's BQ-native + Gemini integration + $0 licensing wins for
your stack IF you ever cross the threshold above.

### Apache Airflow / Cloud Composer

**What it offers:** Multi-system DAGs (GCS sensor -> BQ load -> external API
call -> Slack notification), retry policies, fancier monitoring.

**Why not:** You have one Python process + BQ. Composer adds ~$300/month base
cost for the smallest environment. You'd be running it for jobs that
APScheduler handles in 200 lines. Definitively overkill.

### BigQuery scheduled queries

**What it offers:** The simplest possible option -- write a SQL query, set a
cron schedule, BQ runs it for you.

**Why not yet:** You have zero standalone SQL queries that need recurring
execution. Every cron job involves Python (LLM calls, external API calls,
state-machine logic). When you DO add a "compute weekly Sharpe in SQL"
job, scheduled queries are the right starting point -- and graduating
from there to Dataform is the natural path if you need DAG dependencies.

---

## Current data-flow audit (what you actually have)

| Layer | How it works | Pain? |
|-------|--------------|-------|
| Layer-1 enrichment writes | `orchestrator.py:444+` -> `bigquery_client.py:251::insert_rows_json` | None |
| Outcome tracking + return computation | `outcome_tracker.py:51-70` Python, then BQ write | None -- single function, easy to test |
| Daily cron jobs (price/FRED/MDA/etc.) | APScheduler @ `slack_bot/scheduler.py` | None -- 7 jobs, all observable |
| Meta-evolution weekly | APScheduler @ `meta_evolution/cron.py` (Sunday 02:00 ET) | None |
| Schema migrations | `scripts/migrations/*.py` (17 files), idempotent + version-controlled | Mild -- manual `--apply` step. NOT a Dataform-shaped problem. |
| Backfills | One deferred (`delisted_at` column) | One occurrence in 2 years -- not a recurring pain |
| Lineage / "where did this row come from?" | Implicit in Python file paths | NOT yet a debugging burden (operator hasn't asked) |

**Pain score: very low.** Nothing on this list points to "we need a SQL transformation framework."

---

## Honest tradeoffs

If you adopted Dataform anyway:

- **Engineering cost:** ~3 days to set up workspace + workflow configs + migrate the first transformation (rolling Sharpe, say). After that, ~0.5 day per derived column.
- **Operational cost:** New scheduling paradigm to monitor (Dataform run history) alongside APScheduler. New permissions surface (Dataform service account).
- **Cognitive cost:** Two places to look when "how is X computed?" -- Python OR SQLX. Worse than today's "always in Python."
- **Reversibility:** Easy to remove if abandoned (delete `dataform/` dir + workflow configs).

The work is genuinely **light** -- but the value is currently zero. Don't pay the cognitive tax for nothing.

---

## Early signals that should flip the recommendation

Revisit this decision when **any** of the following becomes true:

1. **>10 derived columns** are computed in Python that could be pure SQL transformations on existing BQ tables.
2. **A lineage debugging incident** -- "which script wrote this row and why?" takes >30min to answer.
3. **A second developer joins** and needs to understand data flows without reading `bigquery_client.py`.
4. **`outcome_tracker.py` evaluation windows change frequently** enough to warrant versioned SQL models (e.g., switching from 7/30/90 to 5/14/60 day windows triggers a Python deploy + BQ migration today; Dataform would be 1 SQLX file edit + workflow trigger).
5. **You introduce ANY BigQuery scheduled query** -- Dataform is the natural graduation path from there (gives you DAG, version control, lineage in addition to the schedule you already need).
6. **You start writing SQL inside Python triple-strings** as derived-column logic -- that's a Dataform smell.

---

## What we just did instead

The directive_versions table that was the long-pending operator action
(task #54) was applied successfully via the existing migration script:

```
$ python scripts/migrations/create_directive_versions_table.py --apply
[apply] PASS: table created or already exists at sunny-might-477607-p8.pyfinagent_pms.directive_versions
```

Idempotent CREATE TABLE IF NOT EXISTS. This is the canonical pattern for
the project today and works fine for the schema-change cadence (one new
table every 1-2 weeks at most).

---

## Decision

**DON'T ADOPT** Dataform / dbt / Airflow / Composer.

Keep:
- Python-computed derived columns
- APScheduler for cron
- `scripts/migrations/*.py` for schema (idempotent, version-controlled)
- Direct `bigquery_client.insert_rows_json` for writes

Re-audit this decision in 6 months OR when any "Early signal" above fires,
whichever comes first.

---

## Cross-references

- `handoff/current/phase-23.0-research-brief.md` (full audit)
- `backend/clients/bigquery_client.py` (single BQ write surface today)
- `backend/services/outcome_tracker.py:51-70` (the most "derived columns in
  Python" example -- still small enough not to warrant SQL)
- `backend/slack_bot/scheduler.py` (7 APScheduler cron jobs)
- `backend/meta_evolution/cron.py` (weekly meta-evolution)
- `scripts/migrations/` (17 files, idempotent pattern)
- Dataform docs: https://docs.cloud.google.com/dataform/docs/overview
- dbt vs Dataform 2026: https://valiotti.com/blog/dataform-vs-dbt-2026/
