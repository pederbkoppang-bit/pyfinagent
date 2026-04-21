# Research Brief — phase-7.2: 13F Institutional Holdings Ingestion

**Tier:** moderate  
**Date:** 2026-04-19  
**Researcher:** Researcher agent (combined external + internal exploration)

---

## Objective

Design `backend/alt_data/f13.py` — a syntactically valid Python module that:
1. Fetches 13F-HR XML filings from SEC EDGAR for a configurable set of CIKs.
2. Parses the `informationTable.xml` document into a flat holdings list.
3. Normalises rows to the `alt_13f_holdings` BigQuery schema.
4. Provides `ensure_table` (idempotent DDL) and `upsert_holdings` (streaming insert).
5. When run as `python -m backend.alt_data.f13`, calls `ensure_table` so `bq ls pyfinagent_data | grep -q alt_13f_holdings` passes.

Both immutable criteria are purely structural: (a) module must parse with `ast.parse`, (b) `alt_13f_holdings` must appear in `bq ls pyfinagent_data`. No row-count requirement.

---

## Queries Run

| # | Variant | Query string |
|---|---------|-------------|
| 1 | Current-year frontier | `SEC EDGAR 13F-HR XML informationTable schema filing format 2026` |
| 2 | Last-2-year window | `sec-edgar-api Python library jadchaar 13F filings fetch 2025` |
| 3 | Year-less canonical | `SEC EDGAR 13F-HR XML informationTable schema filing format` |
| 4 | Canonical | `EDGAR submissions JSON API data.sec.gov CIK 13F-HR accession number filing documents 2025` |
| 5 | Recency/compliance | `SEC EDGAR best practices data access User-Agent rate limit 10 requests per second 2024` |
| 6 | Parse/Python | `SEC EDGAR 13F-HR parse informationTable XML Python requests cusip nameOfIssuer value shrsOrPrnAmt 2024 2025` |

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://github.com/jadchaar/sec-edgar-api | 2026-04-19 | Official library README (GitHub) | WebFetch | `EdgarClient(user_agent=...)`, `get_submissions(cik, handle_pagination=True)` returns filing history with `form`, `filingDate`, `accessionNumber`; auto-paginates; enforces 10 req/s internally |
| https://github.com/jadchaar/sec-edgar-downloader | 2026-04-19 | Official library README (GitHub) | WebFetch | `Downloader("name","email")`, `dl.get("13F-HR", "0001067983", limit=1)` pattern; `download_details=True`; User-Agent composed automatically |
| https://tldrfiling.com/blog/sec-edgar-api-rate-limits-best-practices | 2026-04-19 | Authoritative blog (2026) | WebFetch | 10 req/s hard ceiling; target 8 req/s for safety margin; exponential backoff (60s/120s/240s) on 403; User-Agent `"CompanyName email"` is the primary gating mechanism; no auth/API key needed |
| https://www.thefullstackaccountant.com/blog/intro-to-edgar | 2026-04-19 | Authoritative blog/tutorial | WebFetch | Submissions endpoint `https://data.sec.gov/submissions/CIK{cik}.json`; company concept/facts endpoints; confirms no auth needed |
| https://elsaifym.github.io/EDGAR-Parsing/ | 2026-04-19 | Academic/practitioner doc | WebFetch | Post-2013-Q2: XML format; pre-2013: heterogeneous (fixed-width, CSV, TSV); CUSIP universe approach; 50% accuracy threshold; confirms `informationTable` is the canonical XML segment |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.sec.gov/edgar/filer-information/specifications/form13fxmltechspec-draft | Official spec | 403 from SEC.gov |
| https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/frequently-asked-questions-about-form-13f | Official FAQ | 403 from SEC.gov |
| https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets | Official data page | 403 from SEC.gov |
| https://www.sec.gov/files/form_13f.pdf | Official PDF schema | 403 from SEC.gov |
| https://sec-edgar-api.readthedocs.io/ | Official docs | 403 |
| https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data | Official guide | 403 |
| https://www.sec.gov/files/edgar/filer-information/api-overview.pdf | Official PDF | 403 |
| https://edgartools.readthedocs.io/en/stable/13f-filings/ | Library docs | 403 |
| https://edgartools.readthedocs.io/en/stable/resources/sec-compliance/ | Library docs | 403 |
| https://irdirect.net/scripts/xslt/13F/EDGAR%20Form%2013%20F%20XML%20Technical%20Specification.pdf | Third-party spec mirror | Not attempted; spec v1.2 (2015) — snippet search results sufficient |
| https://sec-api.io/resources/download-xml-files-from-sec-edgar-with-python | Third-party tutorial | WebFetched; paid service, limited EDGAR-native value |
| https://github.com/CodeWritingCow/sec-web-scraper-13f | Community GitHub | Fetched; only metadata rendered, no code |

---

## Recency Scan (2024-2026)

Searched for 2024-2026 literature on SEC EDGAR 13F ingestion, rate limits, and Python tooling.

**Findings:**
- **2026**: tldrfiling.com published updated best-practices article confirming the 10 req/s ceiling is unchanged and still enforced without API keys; recommends 8 req/s target. No new authentication requirement introduced. (Source: tldrfiling.com, accessed 2026-04-19)
- **2025**: `sec-edgar-downloader` confirmed to have a Berkshire 13F-HR filing from February 17, 2026 (period of report December 31, 2025), confirming the library is actively maintained for Q4 2025 filings.
- **2024**: The `edgartools` library gained explicit 13F-HR support; `get_filings(form="13F-HR")` returns `.obj().holdings` as a DataFrame. Not used in our design (dependency too heavy) but confirms the field names are stable.
- **No structural changes** to the 13F-HR XML schema (still on v1.2/v1.6 draft namespace `http://www.sec.gov/edgar/document/thirteenf/informationtable`) in the 2024-2026 window. The SEC's draft v1.6 spec page was inaccessible (403) but search snippets confirm it retains the same 13-column structure.

---

## Key Findings

1. **EDGAR submissions endpoint** — `https://data.sec.gov/submissions/CIK{cik_zero_padded}.json` returns all filing history with `form`, `filingDate`, `reportDate`, `accessionNumber` fields. CIK must be zero-padded to 10 digits. (Source: EDGAR API overview snippet, data.sec.gov)

2. **Filing document URL pattern** — `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_nodash}/{filename}.xml` where `accession_no_nodash` strips hyphens from the accession number (e.g., `0001067983-24-000009` becomes `0001067983240000090`). Filing index at `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&dateb=&owner=include&count=10&search_text=` — or more reliably via the submissions JSON. (Source: sec_insider.py line 19: `SEC_ARCHIVES_URL` pattern already in codebase)

3. **informationTable XML schema** — Namespace: `http://www.sec.gov/edgar/document/thirteenf/informationtable`. Root element: `<informationTable>`. Each holding is an `<infoTable>` element with children: `nameOfIssuer`, `titleOfClass`, `cusip` (9 chars), `value` (integer, in thousands of USD), `shrsOrPrnAmt` (contains `sshPrnamt` integer + `sshPrnamtType` SH|PRN), `putCall` (optional, Put|Call), `investmentDiscretion` (SOLE|DFND|OTR), `otherManager` (optional), `votingAuthority` (contains `Sole`, `Shared`, `None` integers). (Source: SEC search snippets, EDGAR XML spec v1.2/v1.6)

4. **Rate limit** — 10 req/s hard ceiling across all EDGAR domains; pyfinagent compliance doc specifies 8 req/s. `sec-edgar-api` auto-enforces 10 req/s internally. Since `f13.py` will use `requests` directly (matching congress.py pattern), we must add our own throttle or use `time.sleep(0.125)` between requests. (Source: tldrfiling.com 2026; compliance doc row 7.2)

5. **Canonical smoketest CIKs** — Berkshire Hathaway `0001067983` is the most stable and most widely used 13F smoketest filer: large filing, well-known, quarterly filings never missed. BlackRock `0001364742` is an alternative; its 13F is enormous (thousands of holdings) and may be slow for a smoke test. Vanguard `0000102909` sometimes files 13F-NT (notice; no holdings). Recommended: **Berkshire Hathaway `0001067983` as primary smoketest CIK**; optionally Bridgewater `0001350694` as a second (smaller filing). (Source: sec-edgar-downloader README examples; search snippets)

6. **`sec-edgar-api` is NOT in `backend/requirements.txt`** — only `requests>=2.31.0` is present. Since congress.py uses `requests` directly and the criterion only requires a scaffold that creates the table, implementing with raw `requests` (matching congress.py style) avoids adding a new dependency. If the full fetch is implemented, add `sec-edgar-api>=0.1.5` to requirements. (Source: internal audit of `/Users/ford/.openclaw/workspace/pyfinagent/backend/requirements.txt` line 20)

7. **Existing EDGAR URL pattern in codebase** — `backend/tools/sec_insider.py:19` already defines `SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{filer_cik}/{accession}/{doc}"` using `httpx` for Form 4 fetches. The 13F ingester can reuse this URL template with `requests` (sync, consistent with congress.py).

8. **Dedup anchor** — `(accession_number, cusip, sshPrnamt)` is deterministic: one filer's 13F for one quarter should have exactly one row per `(accession_number, cusip)` pair; `sshPrnamt` breaks ties if the same CUSIP appears under both SH and PRN types. A SHA-256 of those three fields (matching congress.py's `_disclosure_id` pattern) yields a stable 24-char `holding_id`.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/alt_data/congress.py` | 322 | phase-7.1 Congressional trades ingester — canonical house style | Active; model for f13.py |
| `backend/alt_data/__init__.py` | 1 | Package init | Minimal |
| `backend/tools/sec_insider.py` | ~120 | Form 4 EDGAR fetcher (httpx, async) | Active; provides SEC URL patterns |
| `backend/requirements.txt` | ~60 | Production deps; pointer to `backend/requirements.txt` | `requests>=2.31.0` present; no sec-edgar-api |
| `docs/compliance/alt-data.md` | 287 | Legal compliance framework | Row 7.2 commits to EDGAR API + User-Agent + 8 req/s |
| `scripts/audit/gap_analysis.py:49` | — | Tags 13F/13D as a gap; recommends expanding sec_edgar tool | Informational; pre-dates phase-7 |

**No existing `backend/alt_data/f13.py`** — must be created.

**No `informationTable` references** anywhere in the codebase (grep confirms zero hits on `informationTable`). First occurrence will be in `f13.py`.

---

## Consensus vs Debate

- **Consensus**: EDGAR submissions JSON (`data.sec.gov/submissions/CIK.json`) is the stable, no-auth entry point for finding 13F-HR filings by CIK. Not disputed.
- **Consensus**: The `informationTable.xml` file is the XML document containing actual holdings; the `primaryDoc.xml` / cover sheet is separate. Parse the `informationTable` file, not the full submission package.
- **Debate**: Library choice. `sec-edgar-api` (jadchaar) vs `edgartools` vs raw `requests`. For pyfinagent style consistency with congress.py, raw `requests` + minimal helpers is preferred; avoids new hard dependencies.
- **Debate**: XML namespace handling. Some older filings have the namespace absent or use a different prefix. Use `xml.etree.ElementTree` with explicit namespace dict, with a fallback namespace-stripped parse.

---

## Pitfalls (from literature)

- **CIK zero-padding**: must be `str(cik).zfill(10)` — the submissions endpoint 404s without leading zeros.
- **Accession number format**: the submissions JSON returns `XXXXXXXXXX-YY-ZZZZZZ` format; the Archives URL needs it with hyphens removed: `XXXXXXXXXXYYZZZZZ`.
- **Finding the informationTable filename**: the filing index JSON at `https://data.sec.gov/submissions/CIK.json` gives `accessionNumber` but NOT the filename of the XML. Must fetch the filing index at `https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.json` and find the entry with `type == "INFORMATION TABLE"` or filename matching `*infotable*` / `*informationTable*`.
- **13F-NT filings**: some filers (e.g. Vanguard sub-entities) file 13F-NT (notice; no XML holdings). Must filter to `form == "13F-HR"` only.
- **Value is in thousands**: `<value>` in the XML is in thousands of USD. Store as-is (integer) and document in column comment; avoid premature multiplication.
- **`putCall` is optional**: many holdings have no put/call; field must be nullable.
- **Large filers**: BlackRock and Vanguard have 4000-5000 holdings per quarter. For a smoketest, Berkshire Hathaway (~50 holdings) is far faster.

---

## Application to pyfinagent

| Finding | Maps to | File:Line |
|---------|---------|-----------|
| congress.py `_resolve_target` / `_get_bq_client` / `ensure_table` / `upsert_*` / `ingest_*` pattern | f13.py should mirror exactly | `backend/alt_data/congress.py:202-307` |
| congress.py SHA-256 `_disclosure_id` 24-char dedup key | f13.py `_holding_id(accession, cusip, sshPrnamt)` | `backend/alt_data/congress.py:77-89` |
| congress.py `_CREATE_TABLE_SQL` DDL constant | f13.py `_CREATE_TABLE_SQL` for `alt_13f_holdings` | `backend/alt_data/congress.py:53-74` |
| congress.py `_safe_payload` + `_RAW_PAYLOAD_CAP_BYTES` | reuse for raw_payload cap | `backend/alt_data/congress.py:121-129` |
| sec_insider.py EDGAR URL pattern `SEC_ARCHIVES_URL` | f13.py `_EDGAR_ARCHIVES_URL` | `backend/tools/sec_insider.py:19` |
| sec_insider.py XML parse with `xml.etree.ElementTree` | f13.py `parse_information_table()` uses same stdlib | `backend/tools/sec_insider.py:50-55` |
| compliance doc row 7.2 User-Agent + 8 req/s | `_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"` + `time.sleep(0.125)` | `docs/compliance/alt-data.md:155` |

---

## Concrete Design Proposal for `backend/alt_data/f13.py`

### DDL — `alt_13f_holdings`

```sql
CREATE TABLE IF NOT EXISTS `{project}.{dataset}.alt_13f_holdings` (
  holding_id       STRING NOT NULL,          -- sha256[:24] of (accession_number|cusip|sshPrnamt)
  as_of_date       DATE NOT NULL,            -- ingest run date (date.today())
  cik              STRING NOT NULL,           -- zero-padded 10-digit CIK
  filer_name       STRING,                   -- from submissions JSON entityName
  accession_number STRING NOT NULL,           -- e.g. "0001067983-24-000009"
  period_of_report DATE,                     -- from submissions JSON reportDate
  filed_on         DATE,                     -- from submissions JSON filingDate
  ticker           STRING,                   -- nullable; not in 13F XML, derived separately if needed
  cusip            STRING NOT NULL,           -- 9-char CUSIP
  nameOfIssuer     STRING,
  titleOfClass     STRING,
  value_usd_thousands INT64,                 -- <value> in thousands of USD (as filed)
  sshPrnamt        INT64,                    -- share or principal amount
  sshPrnamtType    STRING,                   -- SH or PRN
  putCall          STRING,                   -- Put, Call, or NULL
  investmentDiscretion STRING,               -- SOLE, DFND, OTR
  votingAuthority_sole   INT64,
  votingAuthority_shared INT64,
  votingAuthority_none   INT64,
  raw_payload      JSON                      -- full infoTable XML element as JSON
)
PARTITION BY as_of_date
CLUSTER BY cik, cusip
OPTIONS (
  description = "phase-7.2 SEC 13F-HR institutional holdings from EDGAR"
)
```

### Function Signatures

```python
_USER_AGENT = "pyfinagent/1.0 peder.bkoppang@hotmail.no"
_TABLE = "alt_13f_holdings"
_EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession_nodash}-index.json"
_EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"
_DEFAULT_CIKS = ["0001067983"]  # Berkshire Hathaway; add "0001350694" (Bridgewater) optionally

def _holding_id(accession_number: str, cusip: str, sshPrnamt: int | str) -> str:
    """SHA-256[:24] of pipe-joined key fields. Deterministic dedup anchor."""

def _accession_nodash(accession_number: str) -> str:
    """'0001067983-24-000009' -> '000106798324000009'"""

def fetch_13f_submissions(cik: str, *, last_n: int = 1, timeout: int = 30) -> list[dict]:
    """
    GET data.sec.gov/submissions/CIK{cik}.json.
    Returns list of last_n 13F-HR filing metadata dicts:
    {accession_number, filingDate, reportDate, entityName}.
    Fail-open; returns [] on error.
    """

def fetch_filing_index(cik: str, accession_number: str, *, timeout: int = 30) -> dict:
    """
    GET the filing index JSON to find the informationTable XML filename.
    Returns {filename: str} or {} on error.
    """

def fetch_13f(cik: str, accession_number: str, *, timeout: int = 30) -> bytes:
    """
    Fetches the informationTable XML bytes for a given CIK + accession_number.
    Uses fetch_filing_index() to locate the XML filename, then fetches it.
    Returns raw bytes; returns b"" on error (fail-open).
    """

def parse_information_table(xml_bytes: bytes) -> list[dict]:
    """
    Parse informationTable XML -> list of raw holding dicts.
    Handles namespace both present and absent.
    Returns [] on parse failure (fail-open).
    Fields returned per row:
      cusip, nameOfIssuer, titleOfClass, value, sshPrnamt, sshPrnamtType,
      putCall, investmentDiscretion, votingAuthority_sole,
      votingAuthority_shared, votingAuthority_none, raw_payload
    """

def normalize(
    holdings: list[dict],
    filer_meta: dict,
) -> list[dict]:
    """
    Map raw parse dicts to alt_13f_holdings row shape.
    filer_meta: {cik, filer_name, accession_number, period_of_report, filed_on}
    Computes holding_id; sets as_of_date = date.today().
    """

def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    """Mirrors congress.py:202 exactly."""

def _get_bq_client(project: str) -> Any:
    """Mirrors congress.py:221 exactly."""

def ensure_table(*, project: str | None = None, dataset: str | None = None) -> bool:
    """Idempotent CREATE TABLE IF NOT EXISTS. Returns True on success."""

def upsert_holdings(
    rows: list[dict],
    *,
    project: str | None = None,
    dataset: str | None = None,
) -> int:
    """Streaming insert into alt_13f_holdings. Returns count inserted."""

def ingest_cik(
    cik: str,
    *,
    last_n: int = 1,
    project: str | None = None,
    dataset: str | None = None,
    dry_run: bool = False,
) -> int:
    """
    Full pipeline for one CIK:
      fetch_13f_submissions -> fetch_13f -> parse_information_table
      -> normalize -> dedup -> ensure_table -> upsert_holdings.
    Returns count upserted (or count that would be upserted on dry_run).
    """

def _cli(argv: list[str] | None = None) -> int:
    """
    --dry-run, --cik (repeatable), --last-n
    Calls ensure_table() unconditionally (satisfies bq ls criterion even with --dry-run).
    """
```

### `__main__` strategy to satisfy `bq ls | grep -q alt_13f_holdings`

When `python -m backend.alt_data.f13` runs, `_cli()` always calls `ensure_table()` before any data operations. This means the table is created after a single CLI invocation regardless of whether data is actually fetched. The `--dry-run` flag skips the streaming insert but NOT the DDL. Criterion is met after any one CLI run with BQ ambient auth active.

### Smoketest CIKs (recommended)

| CIK | Filer | Holdings per quarter | Notes |
|-----|-------|---------------------|-------|
| `0001067983` | Berkshire Hathaway | ~50 | Most stable, small-to-medium filing, widely tested in community examples |
| `0001350694` | Bridgewater Associates | ~200 | Moderate size, also widely used |

Default `_DEFAULT_CIKS = ["0001067983"]` — single CIK covers the smoketest criterion without adding latency.

---

## Risk Register

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|-----------|--------|-----------|
| R1 | SEC 403 on Archives URL due to missing/wrong User-Agent | medium | high | Use `_USER_AGENT` in every request; add `Accept: application/json` or `Accept: text/xml` as appropriate |
| R2 | `informationTable` XML filename varies across filers | medium | medium | Fetch filing index JSON; look for `type == "INFORMATION TABLE"` or filename containing `infotable` (case-insensitive) |
| R3 | Accession number format mismatch (hyphens vs no-hyphens) | low | high | `_accession_nodash()` helper; unit-test with a real Berkshire accession |
| R4 | SEC rate-limit (10 req/s) tripped during bulk ingest | low (1-2 CIKs for smoketest) | medium | `time.sleep(0.125)` between requests (= 8 req/s); exponential backoff on 403 |
| R5 | BQ auth not ambient during CI | medium | medium | `ensure_table` is fail-open (returns False, logs warning); module still parses with `ast.parse` |
| R6 | 13F-NT filer returns no informationTable | low | low | Filter submissions to `form == "13F-HR"` only |
| R7 | XML namespace absent on pre-2014 filings | low | medium | Dual-namespace parse: try with namespace dict, fall back to stripping namespace with regex |

---

## Research Gate Checklist

**Hard blockers:**

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched: jadchaar/sec-edgar-api, jadchaar/sec-edgar-downloader, tldrfiling.com 2026, thefullstackaccountant.com, elsaifym.github.io/EDGAR-Parsing/)
- [x] 10+ unique URLs total including snippet-only (12 snippet-only + 5 full = 17 total)
- [x] Recency scan (last 2 years) performed and reported (findings: 2026 rate-limit guidance unchanged; 2025 Berkshire 13F confirmed current; no schema changes)
- [x] Full pages/papers read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (congress.py:77, :202, :221, :53, :121; sec_insider.py:19; requirements.txt:20; alt-data.md:155)

**Soft checks:**

- [x] Internal exploration covered every relevant module (alt_data/, sec_insider.py, requirements.txt, compliance doc)
- [x] Contradictions/consensus noted (library choice debate; namespace handling)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "report_md": "handoff/current/phase-7.2-research-brief.md",
  "gate_passed": true
}
```
