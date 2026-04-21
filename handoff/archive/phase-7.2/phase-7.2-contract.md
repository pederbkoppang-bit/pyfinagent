# Sprint Contract — phase-7 / step 7.2 (13F institutional holdings ingestion)

**Step id:** 7.2 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** moderate

Parallel-safe: phase-scoped handoff.

## Research-gate summary

5 sources read in full (sec-edgar-api GitHub, sec-edgar-downloader GitHub, tldrfiling 2026 EDGAR rate-limits, TheFullStackAccountant EDGAR intro, elsaifym.github.io EDGAR-Parsing academic writeup), 17 URLs collected, recency scan 2024–2026, three-variant queries, 6 internal files inspected. `backend/alt_data/congress.py` sets the house style; `backend/tools/sec_insider.py` has the `SEC_ARCHIVES_URL` pattern we can reuse. Brief at `handoff/current/phase-7.2-research-brief.md`. `gate_passed: true`.

## Hypothesis

A single module `backend/alt_data/f13.py` mirrors `congress.py`'s structure: fetch → parse (XML) → normalize → ensure_table → upsert. Default smoketest CIK: Berkshire Hathaway `0001067983`. CLI `python -m backend.alt_data.f13` calls `ensure_table` unconditionally so criterion B (`bq ls | grep -q alt_13f_holdings`) is satisfied on any invocation. 19-column DDL matches the research brief's shape. EDGAR rate limit 8 req/s per compliance doc row 7.2; User-Agent `pyfinagent/1.0 peder.bkoppang@hotmail.no`. The smoketest ingests Berkshire's latest 13F-HR (~50 holdings) — enough rows to prove the pipeline without stressing EDGAR.

## Immutable success criteria (from .claude/masterplan.json)

- `python -c "import ast; ast.parse(open('backend/alt_data/f13.py').read())"`
- `bq ls pyfinagent_data | grep -q alt_13f_holdings`

Not edited.

## Plan steps

1. Create `backend/alt_data/f13.py` (~320 lines):
   - URL templates: submissions JSON (`https://data.sec.gov/submissions/CIK{cik_zero_padded}.json`), filing index (`https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.json`), archive prefix.
   - `_rate_limit()` simple sleep-0.125 gate (= 8 req/s ceiling).
   - `_zero_pad_cik(cik)` helper.
   - `fetch_13f_submissions(cik, last_n=1)` — pulls submissions JSON, filters to `form == "13F-HR"`.
   - `fetch_filing_index(cik, accession_number)` — pulls the `-index.json`, identifies `informationTable` entry.
   - `fetch_13f(cik, accession_number)` — pulls the XML bytes.
   - `parse_information_table(xml_bytes)` — namespace-aware XML parse; returns list[dict].
   - `normalize(holdings, filer_meta)` — maps to 19-column row shape; deterministic `holding_id = sha256(accession_number|cusip|sshPrnamt)[:24]`; `as_of_date = today`.
   - `ensure_table(project, dataset)` — idempotent CREATE TABLE IF NOT EXISTS.
   - `upsert_holdings(rows, project, dataset)` — fail-open `insert_rows_json`.
   - `ingest_cik(cik, last_n=1, project, dataset, dry_run)` — orchestrator.
   - `_cli(argv)` — always runs `ensure_table()` first; optional `--cik`, `--last-n`, `--dry-run` flags; default CIK = 0001067983 (Berkshire); prints JSON summary on stdout.
2. Run `python -m backend.alt_data.f13` against live BQ + EDGAR to (a) create the table, (b) ingest Berkshire's latest 13F-HR.
3. Verify table exists via `bq ls` (MCP or CLI).
4. Run `python -c "import ast; ast.parse(...)"` for the immutable syntax check.
5. Run full regression.
6. Write `phase-7.2-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No ticker lookup from CUSIP (ticker column stays NULL for now; derivation lives in phase-7.12 feature integration).
- No multi-filer mass ingest (smoketest = 1 CIK).
- No scheduler wiring (phase-9).
- No pytest module (criterion doesn't require; phase-7.11 shared-infra will add).
- ASCII-only.

## Risk register

| ID | Risk | Mitigation |
|----|------|-----------|
| R1 | EDGAR 403 under rate cap | 60·2^attempt backoff; single retry; fail-open |
| R2 | XML namespace variance | Parse with both namespaced and un-namespaced tag lookup |
| R3 | Filing index JSON missing `informationTable` | Log WARNING, return 0, skip the filing |
| R4 | Value-in-thousands confusion | Column named `value_usd_thousands`, comment in DDL |
| R5 | BQ table exists but ingest fails | The criterion is table-existence, so ensure_table runs first & is idempotent; ingest failure is CONDITIONAL for the module but NOT a FAIL for the step's criteria |

## References

- `handoff/current/phase-7.2-research-brief.md`
- `backend/alt_data/congress.py` (house style)
- `backend/tools/sec_insider.py` (`SEC_ARCHIVES_URL` constant)
- `docs/compliance/alt-data.md` row 7.2
- `.claude/rules/security.md` (EDGAR User-Agent + ASCII)
- `.claude/masterplan.json` → phase-7 / 7.2
