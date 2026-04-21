# Q/A Critique -- phase-7 / 7.2 "13F institutional holdings ingestion"

**Cycle:** 1  **Q/A id:** qa_72_v1  **Date:** 2026-04-19  **Verdict:** PASS

---

## 5-item protocol audit

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Research gate (brief + `gate_passed: true`, >=5 sources in full, three-variant queries, recency scan) | PASS | `phase-7.2-research-brief.md` present (22,335 bytes, mtime 22:41). 5 sources read in full (sec-edgar-api, sec-edgar-downloader, tldrfiling 2026, TheFullStackAccountant, elsaifym.github.io). 17 URLs collected. Three-variant queries #1/#2/#3 explicitly listed. 6 internal files inspected. Brief opens contract's research-gate summary line. |
| 2 | Contract written BEFORE GENERATE (mtime ordering) | PASS | contract mtime `22:43:04` < experiment-results mtime `22:46:29` (diff +3m25s). research-brief `22:41:44` < contract -- correct ordering. |
| 3 | Experiment results verbatim + mid-cycle fixes disclosed | PASS | Results explicitly call out (a) URL-template surprise `.../{acc}-index.json` -> 404, fixed to `.../index.json`; (b) `element or element` DeprecationWarning rewritten as `is None`. Both disclosures match current code. |
| 4 | Log-last -- harness_log.md last block is phase-7.1, NOT 7.2 | PASS | Last cycle in `handoff/harness_log.md` is `phase=7.1 result=PASS` (22:35 UTC). No 7.2 cycle appended yet -- correct per log-last discipline. |
| 5 | First Q/A on 7.2 (no verdict-shopping) | PASS | No prior `phase-7.2-evaluator-critique*.md` exists in handoff/current or archive. This is qa_72_v1. |

---

## Deterministic A--I

| ID | Check | Result | Evidence |
|----|-------|--------|----------|
| A | `python -c "import ast; ast.parse(open('backend/alt_data/f13.py').read()); print('SYNTAX OK')"` | PASS | `SYNTAX OK`, exit 0. Immutable criterion (1) satisfied. |
| B | `bq ls --project_id=sunny-might-477607-p8 pyfinagent_data \| grep -q alt_13f_holdings` | PASS | Row present: `alt_13f_holdings  TABLE  DAY (field: as_of_date)  cik, cusip`. Immutable criterion (2) satisfied. |
| C | `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` | PASS | `152 passed, 1 skipped, 1 warning in 12.96s`. Regression baseline unchanged from phase-7.1. |
| D | File existence: `backend/alt_data/f13.py` + handoff trio | PASS | f13.py 17,004 bytes. contract 4,561 B. experiment-results 5,266 B. research-brief 22,335 B. |
| E | Scope: `git status --short` shows only backend/alt_data + phase-7.2 handoff | PASS | Untracked: `backend/alt_data/` (contains `__init__.py` + `congress.py` from 7.1 + new `f13.py`) and three `handoff/current/phase-7.2-*.md` files. No unrelated modifications. |
| F | Schema match INFORMATION_SCHEMA.COLUMNS | PASS | 20 columns in DDL order: `holding_id, as_of_date, cik, filer_name, accession_number, period_of_report, filed_on, ticker, cusip, nameOfIssuer, titleOfClass, value_usd_thousands, sshPrnamt, sshPrnamtType, putCall, investmentDiscretion, votingAuthority_sole, votingAuthority_shared, votingAuthority_none, raw_payload`. 19 named in research brief + `raw_payload JSON` = 20, matches DDL exactly. Zero drift. |
| G | Row shape sanity | PASS | `SELECT cik, accession_number, COUNT(*), COUNT(DISTINCT cusip), COUNT(DISTINCT nameOfIssuer)` -> 1 row: `cik=0001067983, accession=0001193125-26-054580, n=110, unique_cusips=42, unique_issuers=39`. Matches brief's ~110/42. (39 issuers vs 42 cusips is normal -- multiple share classes of same issuer, e.g. Google A+C, Liberty Media tracking stocks.) |
| H | URL-fix reality-check (independent curl) | PASS | `https://www.sec.gov/.../0001193125-26-054580-index.json` -> `404`. `https://www.sec.gov/.../index.json` -> `200`. Confirms Main's mid-cycle disclosure is truthful; `_FILING_INDEX_URL` in code is the `/index.json` form. |
| I | ASCII discipline | PASS | `open('backend/alt_data/f13.py','rb').read().decode('ascii')` -> no decode error. |

---

## LLM judgment

### Rate-limit + User-Agent compliance

- `_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"` (line 38) -- matches SEC EDGAR's documented `Sample Company Name AdminContact@<sample>.com` convention and `.claude/rules/security.md` ("SEC EDGAR requires custom User-Agent `FirstName LastName email@domain.com`"). Format is acceptable (product/version + contact email).
- `_RATE_INTERVAL_S = 1.0 / 8.0` = 125ms sleep (line 40) = 8 req/s ceiling, safely below EDGAR's 10 req/s.
- `_rate_limit()` is invoked before **every** outbound GET: line 150 (submissions), line 187 (filing index), line 228 (XML archive). 3 GETs per filing x 125ms = 375ms minimum spacing = ~2.7 req/s per filing pipeline -- well below the 8 req/s ceiling even if multiple filings are processed in parallel (which the current code does not do).
- `_http_get` (line 121) applies `headers={"User-Agent": _USER_AGENT, ...}` on every request, with `60*2^attempt` backoff on 403 (line 134-136) and `5*2^attempt` on 5xx (line 137-139). Both attempt caps at `max_attempts=2` preventing runaway retries. PASS.

### Namespace-tolerant XML parse

- `_find_text` (line 232-242) tries `ns:{local_name}` first via `_EDGAR_NS = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}`, then falls back to un-namespaced `el.find(local_name)`. PASS.
- `parse_information_table` (line 266) concatenates namespaced + un-namespaced `infoTable` elements so both pre-2013-Q2 and current filings parse. PASS.
- `_find_int` safely handles None + comma stripping + ValueError. Fail-open logged via `logger.warning` only, never raises. PASS.

### `ticker` column NULL by design

- `normalize()` line 321: `"ticker": None`. Documented in contract out-of-scope (line 45) + experiment-results caveat #2 + brief row mapping. Phase-7.2 compliance-doc row 7.2 does NOT promise ticker at this step. No violation.

### Anti-rubber-stamp -- both mid-cycle fixes present in current code

- **Fix 1 (URL template):** `_FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"` at line 47 -- uses `/index.json` form, not `/{acc}-index.json`. Verified truthful by check H (independent curl).
- **Fix 2 (DeprecationWarning):** No `element or element` truth-test pattern found in the module; all wrappers use explicit `if X is None: X = fallback` pattern (lines 272-273, 278-279). PASS.

### Scope honesty

- `git status --short` confirms only `backend/alt_data/` (new dir containing phase-7.1 and phase-7.2 artifacts) + three `handoff/current/phase-7.2-*.md` files are untracked. No modifications outside scope. Regression 152/1 unchanged. PASS.

### Research-gate compliance in contract

- Contract line 9 opens with a full research-gate summary naming the 5 sources read in full + URL count + three-variant queries + internal files inspected + `gate_passed: true`. References section (line 61-68) cites the brief. PASS.

---

## Advisories (non-blocking; carry forward)

| ID | Issue | Where | Action |
|----|-------|-------|--------|
| `adv_72_ticker_null` | `ticker` column always NULL until phase-7.12 CUSIP->ticker resolver lands. | `normalize()` line 321. | Already documented in experiment-results caveat #2 + contract out-of-scope. No action required this phase. |
| `adv_72_single_filer` | Smoketest ingests only Berkshire (CIK 0001067983, 110 rows). Top-N filer enumeration deferred to phase-7.11 scheduler. | `_DEFAULT_CIK` line 50. | Carry forward to phase-7.11. |
| `adv_72_stream_insert_not_merge` | `upsert_holdings` is stream `insert_rows_json`, not MERGE (same pattern as congress.py `adv_71_docstring_merge`). | `upsert_holdings` line 387-408. | Carry forward to phase-7.11 shared-infra dedup pass. |
| `adv_72_house_style_tests_missing` | No pytest module for `f13.py`. Criteria B doesn't require it, but shared-infra in phase-7.11 should add mock-based tests for `parse_information_table` + `_find_information_table_filename`. | No test file exists. | Carry forward to phase-7.11. |
| `adv_72_backend_alt_data_untracked` | Whole `backend/alt_data/` dir shows as `??` in `git status` because it hasn't been committed since 7.1. Not a 7.2 issue but worth flagging for the next commit boundary. | `git status`. | Peder approves commit cadence; flag for next push. |

---

## Decision

**PASS -- qa_72_v1.** All 5 protocol items PASS. All 9 deterministic A--I checks PASS. Both immutable criteria satisfied. LLM judgment: rate-limit + UA compliant, namespace-tolerant parse verified, mid-cycle URL + DeprecationWarning fixes both reflected in current code, scope clean, research gate cited in contract. Zero blocking findings. Advisories are carry-forward only, not retry triggers.

**Next step order (reminder):** append `harness_log.md` cycle block for phase-7.2 BEFORE flipping status pending -> done in `.claude/masterplan.json`.
