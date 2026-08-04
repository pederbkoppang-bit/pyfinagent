# Contract -- phase-82.7

**Step**: SECURITY DEFECT -- credentials written to logs in plaintext via URL query strings.
**Status at write time**: pending, P0. **Depends on**: none.

## 0. SECURITY HANDLING FOR THIS STEP

No artifact produced by this step -- brief, contract, results, test, commit message --
contains a real credential VALUE. Keys are referred to by ENV VAR NAME only. Tests use
obviously-synthetic placeholders. This contract does not instruct anyone to grep logs for
a key value.

## 1. Research gate -- PASSED

`handoff/current/research_brief_82.7.md` (684 lines), Workflow rail, task `wthmmnx2n`.

```
gate_passed=true   tier=complex          external_sources_read_in_full=10
snippet_only=33    urls_collected=43     recency_scan_performed=true
internal_files_inspected=45
coverage: audit_class=true  rounds=4  dry_rounds=3  K_required=2  dry=TRUE
          16 query shapes (A..P)
```

**The structured return FAILED** -- the subagent finished without calling
StructuredOutput, so the Workflow returned an error and there was no envelope. The gate
survived only because the researcher follows WRITE-FIRST: the brief was on disk,
complete, with its envelope as the final section. This is precisely the failure that
`feedback_researcher_write_first` exists to prevent, and it is the second independent
justification for step **36.27** (a checked-in `research-gate.js`): an inline script
authored per-spawn has no retry path when the structured return drops.

This step is **audit-class** -- criterion 2 is an unknown-denominator sweep -- so the
loop-until-dry gate binds and was satisfied at 3 dry rounds against K=2.

## 2. What the sweep found -- 8 leaking sites, not the 7 I grepped

| # | file:line | Param | Provider | Transport |
|---|---|---|---|---|
| 1 | `backend/backtest/data_ingestion.py:371` | `api_key` | FRED | f-string URL, httpx |
| 2 | `backend/tools/fred_data.py:40` | `api_key` | FRED | f-string URL, httpx |
| 3 | `backend/services/fx_rates.py:147` | `api_key` | FRED | `requests.get(params=)` |
| 4 | `backend/econ_calendar/sources/fred_releases.py:53` | `api_key` | FRED | `requests.get(params=)` |
| 5 | `backend/tools/alphavantage.py:44` | `apikey` | Alpha Vantage | f-string URL, httpx |
| 6 | `backend/tools/social_sentiment.py:44` | `apikey` | Alpha Vantage | f-string URL, httpx |
| 7 | `backend/news/sources/finnhub.py:71` | `token` | Finnhub | `client.get(params=)` |
| 8 | `backend/econ_calendar/sources/finnhub_earnings.py:53` | `token` | Finnhub | `requests.get(params=)` |

**#3 was MISSED by my own grep and I want the reason recorded**, because it generalises:
my pattern looked for a credential adjacent to an f-string URL, and `fx_rates.py` passes
the key through a `params={...}` dict. `requests` still puts it in the URL. A single
query shape cannot close an unknown-denominator sweep -- which is exactly why the
adaptive-coverage rule marks steps like this audit-class. I verified #3 by reading the
file; it is real.

The brief also catalogues **6 already-safe header-auth sites**, including
`backend/news/sources/benzinga.py:66-68` -- a SIBLING MODULE of leaking site #7. The
header pattern is already an accepted idiom in this repo, in the same package.

## 3. Root cause -- CONFIRMED, and worse than I measured

The redaction facility exists and is correct: `SecretRedactionFilter` in
`backend/services/observability/log_redaction.py`. It is attached at
`backend/main.py:110-111`, inside `setup_logging()`, whose **only non-test call site is
`backend/main.py:151`** -- the first statement of the FastAPI `lifespan`.

Measured blast radius, re-derived by me because the brief's file count was off:
`logging.basicConfig(` appears **54 times in 54 files** across `backend/ scripts/
functions/` (the brief says "54 times ... in 40 distinct files" -- the call count is
right, the file count is 54). Every one is an independent bootstrap with NO filter.

**A correction to my own earlier reading, from the brief:** the *daily macro cron* is NOT
in the unprotected set -- `register_macro_ingest_cron` is invoked from
`backend/main.py:333`, inside the lifespan, so it inherits the filter. The 2026-08-03
operator-visible leak therefore came from a **manual/CLI backfill**, which is the
unprotected class. The fix must not be scoped to `macro_cron.py`.

## 4. Immutable success criteria -- copied VERBATIM from `.claude/masterplan.json`

Command: `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_7_credential_logging.py -q`

1. a test asserts that performing a FRED macro fetch with httpx logging at INFO produces no log record containing the api key value
2. the repository is swept for other outbound calls that interpolate a credential into a URL query string, and the sweep result is recorded with file:line for every hit found
3. every hit found by that sweep is either fixed or has its own queued follow-up step; a test asserts the fixed set produces no credential-bearing log records

## 5. Plan

1. **[P0, load-bearing] `install_secret_redaction()`** -- a new IDEMPOTENT helper in
   `log_redaction.py` that attaches the filter to every root handler and is safe to call
   twice. `backend/main.py` delegates to it so there is ONE attach path. Called from the
   entry points that can reach a credential URL, AFTER their `basicConfig()`.
2. **[P0] Close gap G3** -- the filter rewrites only `record.msg`/`record.args`
   (verified at `log_redaction.py:41-44`); `record.exc_text` and the rendered exception
   are untouched, so an `httpx.HTTPStatusError` repr carrying the full URL still leaks.
3. **[P1] Header auth where the provider allows it** -- Finnhub #7/#8 to
   `X-Finnhub-Token` (no endpoint change). Alpha Vantage #5/#6 is **impossible**; those
   depend permanently on step 1.
4. **[P1] Regression in the EXISTING harness** -- `scripts/harness/secret_leak_regression.py`
   already exists, exits non-zero on FAIL and writes a JSON receipt. Extend it; do not
   build a new one.

## 6. Explicit NON-scope, and why

- **FRED `Authorization: Bearer` is NOT attempted here.** It requires migrating to the
  FRED **v2** endpoints, and the brief flags its own sourcing as INCOMPLETE: the vendor
  page returned 403 to WebFetch and outbound curl is sandbox-blocked, so the v2 header
  format is second-hand. Acting on an unverified vendor contract in a security fix is
  how you ship a broken ingest. Queued as its own step instead.
- **`FRED_API_KEY` rotation is operator-owed and is NOT this step's job.** Shipping the
  code fix does not un-disclose the leaked value.
- **Not** `logging.getLogger("httpx").setLevel(WARNING)` -- that suppresses the evidence,
  not the secret, and is trivially reverted.
- **Not** calling `setup_logging()` from scripts. `.claude/masterplan.json:18101` records
  it as single-call-only and DESTRUCTIVE: it does `root.handlers.clear()` and re-wraps
  `sys.stderr.buffer` in a fresh `TextIOWrapper`, observed live as
  `ValueError('I/O operation on closed file.')`, and the blanket clear destroys pytest's
  log capture. That is why step 1 is a new idempotent helper, not a reuse.

## 7. Mutation-test requirement, stated up front

Per `feedback_mutation_test_guards_and_fixtures` and the brief's own warning: the
criterion-1 test must FAIL when the filter is **removed from the handler**, not merely
when `redact_secrets()` is called directly. A test that exercises the function while
leaving the REACHABILITY bug untested re-pins the thing that already worked and misses
the actual defect. The mutation matrix must name that mutant explicitly.

## 8. References

- `handoff/current/research_brief_82.7.md` -- 10 sources read in full, 43 URLs
- OWASP Logging Cheat Sheet; Python `logging` docs on handler- vs logger-level filters
- `backend/services/observability/log_redaction.py`; `backend/main.py:87,110-111,151,333`
- `scripts/harness/secret_leak_regression.py`; `backend/tests/test_phase_60_4_observability.py:246-254`
- NOTE: the raw gate envelope could not be archived under `qa_returns/` because the
  structured return errored; the envelope of record is the final section of the brief.
