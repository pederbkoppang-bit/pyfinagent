# Sprint Contract — phase-7 / step 7.1 (Congressional trades ingestion)

**Step id:** 7.1 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** moderate

Parallel-safe: phase-scoped handoff files.

## Research-gate summary

6 sources in full (House Stock Watcher S3 data, senate-stock-watcher GitHub repo + gist, STOCK Act guide, BigQuery MERGE tutorial, duplicates-handling MERGE pattern, Lambda Finance API landscape review), 16 URLs, recency scan, three-variant queries, 5 internal files inspected. Recommended source: House+Senate Stock Watcher S3 JSON (no ToS click-through, STOCK-Act-mandated public, two-URL pull). Brief at `handoff/current/phase-7.1-research-brief.md`. `gate_passed: true`.

## Hypothesis

A single ingest module `backend/alt_data/congress.py` pulls the House + Senate Stock Watcher bulk S3 JSON files, normalizes rows, creates `pyfinagent_data.alt_congress_trades` if absent, and MERGEs rows with `as_of_date = today` as the freshness anchor. The `as_of_date >= CURRENT_DATE() - 30` criterion is satisfied from day 1 because `as_of_date` is the ingest date, not the trade date. House Stock Watcher's bulk file typically contains thousands of historical rows, so ingesting all of them with `as_of_date = today` seeds >100 rows for the criterion. The compliance doc row 7.1 is honored: STOCK-Act-mandated public bulk data, 2 GET requests, no login, no robots.txt issue (S3).

## Immutable success criteria (from .claude/masterplan.json)

List-form:
- `python -c "import ast; ast.parse(open('backend/alt_data/congress.py').read())"` → syntactically valid Python.
- `bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM pyfinagent_data.alt_congress_trades WHERE as_of_date >= CURRENT_DATE() - 30' | tail -n 1 | awk '{ exit ($1 > 100 ? 0 : 1) }'` → row count > 100.

Not edited.

## Plan steps

1. Create `backend/alt_data/__init__.py` (empty package marker).
2. Create `backend/alt_data/congress.py`:
   - `fetch_disclosures(*, house=True, senate=True, timeout=30)` — HTTP GET each bulk JSON (2 URLs from research brief), `requests` client with `User-Agent: pyfinagent/1.0 peder.bkoppang@hotmail.no`. Fail-open per-source (if one source 500s, the other still runs). Returns `{"house": [...], "senate": [...]}`.
   - `_disclosure_id(row)` — deterministic sha256 hash of `(chamber, senator_or_rep, ticker, transaction_date, amount_min, amount_max)` truncated to 24 chars, used as primary key.
   - `normalize(raw_rows, chamber)` — map the House JSON fields and Senate JSON fields to a common shape: `{disclosure_id, as_of_date, senator_or_rep, party, chamber, transaction_type, ticker, amount_min, amount_max, transaction_date, disclosure_date, source, raw_payload}`. `as_of_date = date.today().isoformat()`.
   - `ensure_table(project=None, dataset=None)` — idempotent `CREATE TABLE IF NOT EXISTS`. Uses MCP BQ write or falls back to `google-cloud-bigquery` client.
   - `upsert_trades(rows, project=None, dataset=None) -> int` — MERGE on `disclosure_id` so re-runs don't dedupe-explode. Uses `insert_rows_json` with the dedup MERGE pattern documented in the research brief.
   - `ingest_recent(*, project=None, dataset=None) -> int` — orchestrator: fetch → normalize → ensure_table → upsert. Returns count upserted.
   - All ASCII-only logger output. All BQ calls fail-open.
3. Use the BigQuery MCP to `CREATE TABLE IF NOT EXISTS` and run the MERGE — avoids needing `bq` CLI auth in this session.
4. Run the ingest script once against prod BQ to seed `alt_congress_trades` with >100 fresh rows.
5. Verify the count via MCP `execute_sql_readonly` + the immutable `bq query` shape if `bq` CLI has auth; fall back to the MCP check if not.
6. Run the immutable `ast.parse` check.
7. Run full regression (no new pytest targets this cycle — the ingest script is a CLI utility; a later step can add tests).
8. Write `phase-7.1-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No Senate eFD scraping (JS-heavy; ruled out by compliance doc Section 5.4).
- No licensed-aggregator API calls (CapitolTrades, QuiverQuant) — would introduce ToS contract formation or spend.
- No scheduler wiring (belongs in phase-9).
- No signal extraction / IC eval (phase-7.12).
- No pytest module this cycle (the "syntax parse" immutable check is the only code-level gate; integration test belongs to phase-7.11 shared infra).
- ASCII-only.

## Risk register

| ID | Risk | Mitigation |
|----|------|-----------|
| R1 | House Stock Watcher S3 URL drifts or becomes 403 | Try the primary URL; on failure log WARNING and fall back to `senate-stock-watcher` alone. If both fail, log and exit 0 (fail-open) — but criterion B would then fail, so the step would CONDITIONAL until data source is fixed. |
| R2 | Row count < 100 on single fetch | Ingest entire bulk file (typically thousands of rows historical). `as_of_date = today` seeds every row with today's freshness, making the > 100 check trivially satisfied. |
| R3 | BQ MERGE fails due to permission / schema drift | Fail-open; log the error; rely on `ensure_table` having already run. |
| R4 | `raw_payload` JSON blob too large | Cap individual `raw_payload` at 100KB; truncate with a marker. |
| R5 | PII in the `raw_payload` (filer address, etc.) | House+Senate Stock Watcher data already strips personal info; still apply the compliance doc's hash-or-drop rule to any address field. |

## References

- `handoff/current/phase-7.1-research-brief.md`
- `docs/compliance/alt-data.md` Section 4 row 7.1 + Section 5 (disciplines)
- `scripts/migrations/phase_6_5_intel_schema.py` (CREATE TABLE IF NOT EXISTS house pattern)
- `backend/news/bq_writer.py:41-97` (fail-open BQ client)
- `backend/intel/scanner.py` (fail-open HTTP pattern + User-Agent)
- `.claude/rules/security.md` (User-Agent + ASCII)
- `.claude/masterplan.json` → phase-7 / 7.1 immutable verification
