---
name: news-corpus-persistence-83-0
description: Step 83.0 research -- BQ cannot add a REQUIRED column to an existing table (one-shot window); insert_rows_json is STRICT by default; the live news path is DISJOINT from backend/news/; the fetched_at rename has a live-table trap
metadata:
  type: project
---

Step 83.0 (news corpus persistence), researched 2026-08-07.

**Two BigQuery facts that are load-bearing and easy to get wrong:**

1. **You CANNOT add a REQUIRED column to an existing BQ table.** Vendor-verbatim
   (cloud.google.com/bigquery/docs/managing-table-schemas): "REQUIRED columns can
   be added only when you create a table while loading data, or when you create an
   empty table with a schema definition." And NULLABLE->REQUIRED is impossible
   ("you must recreate the table"). So a `provenance STRING NOT NULL` requirement
   is a **one-shot window** that exists only while the table is absent. Verify
   absence live BEFORE planning the migration; if the table already exists the
   only remedy is drop+recreate (owner-approval gated).
2. **`insert_rows_json` is STRICT by default.** `ignore_unknown_values=False`
   ("treats unknown values as errors") and `skip_invalid_rows=False` ("causes the
   ENTIRE request to fail if any invalid rows exist") -- confirmed in the INSTALLED
   SDK source, not just docs. So an unknown column or a missing REQUIRED column
   rejects the WHOLE batch. `bq_writer.py:23-25`'s docstring claiming BQ "ignores
   unknown keys ... tolerated on permissive" is FALSE; there is no permissive mode.
   (Same class as the 82.48 finding "insertAll rejects WHOLE batch".)

**`CREATE TABLE IF NOT EXISTS` is a no-op on an existing table** -- amending DDL
text is INERT on re-run. A migration is only honestly re-runnable if it reads the
schema back (`get_table().schema` or `INFORMATION_SCHEMA.COLUMNS`) and fails LOUD
on drift. Assert the schema map is non-empty first or the oracle is vacuous.

**Why: the step framing said "amend + run the migration" as if DDL text were the
whole job. It is half the job; the post-condition read-back is the other half.**

**How to apply:** any future step that adds a REQUIRED/NOT NULL column to BQ --
check live existence FIRST, and treat "table absent" as a perishable asset.

---

**SCOPE TRAP -- two news systems that never touch each other.** The live path that
moves the book is `autonomous_loop.py:476` -> `backend/services/news_screen.py`
(RSS feeds + LLM classifier + a LOCAL FILE cache) -> `screener.py:334`. The corpus
path being fixed is `backend/news/` (registry/fetcher/bq_writer), whose only
consumers are `backend/tests/test_bq_writer.py` and
`scripts/smoketest/phase6_e2e.py` -- it is NOT scheduled. Fixing bq_writer does
NOT capture what the live screener acted on. Always grep the CONSUMERS before
accepting a step's "X influences trades but isn't persisted" framing: both halves
can be true while referring to different modules.

**RENAME TRAP:** `fetched_at` appears 16x in 8 files; 9 serve `news_articles`
(rename) and 7 serve `calendar_events` (do NOT rename -- it is the one table that
exists live, and its writer swallows errors, so breakage is SILENT). A repo-wide
sed is the failure mode. Also `test_bq_writer.py` asserts
`set(row.keys()) == _NEWS_ARTICLES_FIELDS` **exactly**, so ADDING `provenance`
breaks those tests even without the rename.

**Also measured:** `backend/news/registry.py` ALREADY implements a PEP 544
Protocol + `@register` decorator registry with 4 live registrations -- do not
build a second one. `api_call_log` table is ALSO absent from BQ (same defect
class, separate step). FINNHUB/BENZINGA keys absent from `backend/.env`;
ALPHAVANTAGE set.

Related: [[project_outcome_write_82_48]], [[reference_vacuous_type_guards_on_bq_string_columns]],
[[feedback_guards_stop_one_seam_short]], [[project_phase83_market_news]]
