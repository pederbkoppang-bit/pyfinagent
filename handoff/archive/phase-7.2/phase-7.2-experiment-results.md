# Experiment Results — phase-7 / step 7.2 (13F institutional holdings ingestion)

**Step:** 7.2 — second live-data step of phase-7.
**Date:** 2026-04-19.
**Cycle:** 1.

## What was built

One new file + one new BQ table with 110 live rows.

`backend/alt_data/f13.py` (~320 lines):
- `_zero_pad_cik`, `_rate_limit` (8 req/s), `_holding_id` (sha256[:24]), `_safe_payload`, `_normalize_date` helpers.
- `_http_get(url)` with EDGAR-compliant `User-Agent: pyfinagent/1.0 peder.bkoppang@hotmail.no` and 60·2^attempt backoff on 403, 5·2^attempt on 5xx.
- `fetch_13f_submissions(cik, last_n)` — pulls `data.sec.gov/submissions/CIK{padded}.json`, filters `form == "13F-HR"`.
- `fetch_filing_index(cik, accession_number)` — pulls `/index.json` (not the `-index.json` suffix the research brief initially guessed — caught mid-cycle, see below).
- `_find_information_table_filename(index_json)` — walks `directory.item` looking for the `.xml` file that isn't `primary_doc.xml`.
- `fetch_13f(cik, accession_number)` → bytes.
- `parse_information_table(xml_bytes)` — namespace-tolerant XML walk; extracts 11 fields per holding.
- `normalize(holdings, filer_meta)` — maps to 19-column row shape; `as_of_date = today`.
- `ensure_table`, `upsert_holdings` (same fail-open pattern as `congress.py`).
- `ingest_cik` orchestrator + `_cli` that always runs `ensure_table` first so criterion B passes on any invocation.

BQ table: `pyfinagent_data.alt_13f_holdings` — 19 columns, partition `as_of_date`, cluster `cik, cusip`.

## File list

Created: 1 (`backend/alt_data/f13.py`).
BQ live change: 1 new table + 110 rows ingested (Berkshire Hathaway latest 13F-HR, accession 0001193125-26-054580, filed 2026-02-17, 42 unique CUSIPs).
Modified: 0.

## Mid-cycle fix caught

First-pass URL template was `https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession_nodash}-index.json` (the prefix-dashes variant commonly documented). Returned 404 on the canonical Berkshire accession. Diagnosed by curling both variants; the correct URL is `.../index.json` (no `{accession_nodash}-` prefix). Fixed the template; live ingest then pulled 110 holdings cleanly.

Two `DeprecationWarning`s from `element or element` truth-test also caught mid-cycle; rewrote as explicit `is None` checks. Disclosed for auditability (anti-rubber-stamp artifact).

## Verification command output

### Immutable (A) — Python syntax

```
$ python -c "import ast; ast.parse(open('backend/alt_data/f13.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

### Immutable (B) — bq ls grep

```
$ bq ls --project_id=sunny-might-477607-p8 pyfinagent_data | grep alt_13f_holdings
  alt_13f_holdings        TABLE                                                                                      DAY (field: as_of_date)        cik, cusip
$ bq ls --project_id=sunny-might-477607-p8 pyfinagent_data | grep -q alt_13f_holdings && echo "GREP EXIT=0"
GREP EXIT=0
```

Partition + cluster columns are as specified in the DDL.

### Cross-check via MCP

```
MCP execute_sql_readonly:
  SELECT COUNT(*), COUNT(DISTINCT cusip), MAX(as_of_date), ANY_VALUE(filer_name)
  FROM `sunny-might-477607-p8.pyfinagent_data.alt_13f_holdings`
  -> rows=110, unique_cusips=42, latest=2026-04-19, filer="BERKSHIRE HATHAWAY INC"
```

### Ingest CLI

```
$ python -m backend.alt_data.f13
{"ts":"2026-04-19T20:45:08.822097+00:00","cik":"0001067983","last_n":1,"dry_run":false,"table_ready":true,"ingested":110}
```

### Full regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 15.05s
```

Unchanged green baseline.

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `python -c "import ast; ast.parse(open('backend/alt_data/f13.py').read())"` | PASS | Exit 0, `SYNTAX OK`. |
| 2 | `bq ls pyfinagent_data | grep -q alt_13f_holdings` | PASS | Table listed with correct partition/cluster. |

## Known caveats (transparency)

1. **Single-filer smoketest.** Only Berkshire Hathaway's latest 13F-HR was ingested. The module's `--cik` flag accepts any CIK; a phase-7.11 scheduler should enumerate top-N filers and fan out.
2. **`ticker` column is NULL.** CUSIP→ticker mapping is deferred to phase-7.12 feature integration (no CUSIP lookup service wired in this repo yet).
3. **Stream-insert, not MERGE** — same advisory as `congress.py` (adv_71_docstring_merge). Carried forward.
4. **EDGAR informationTable namespace is `http://www.sec.gov/edgar/document/thirteenf/informationtable`.** Parser also falls back to un-namespaced lookups for older (pre-2013-Q2) filings or unusual variants.
5. **URL-template surprise** — `.../{acc}-index.json` is a common documented pattern in tutorials but 404s for current Berkshire filings; actual path is `.../index.json`. Mid-cycle fix disclosed above.
6. **ASCII-only.** Module decodes as ASCII.

## Pre-Q/A self-check

- Immutable (A) exit 0.
- Immutable (B) grep exit 0.
- BQ table new, partition/cluster match DDL, 110 rows from Berkshire's latest 13F-HR with 42 unique CUSIPs.
- Regression 152 passed / 1 skipped unchanged.
- `git status --short` shows only `backend/alt_data/f13.py` + handoff files.
- Handoff phase-scoped.
- Masterplan NOT flipped yet.
