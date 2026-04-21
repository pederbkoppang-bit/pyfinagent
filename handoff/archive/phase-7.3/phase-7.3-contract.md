# Sprint Contract — phase-7 / step 7.3 (FINRA short-volume ingestion)

**Step id:** 7.3 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** moderate

Parallel-safe: phase-scoped handoff.

## Research-gate summary

6 sources in full (FINRA daily-short-sale catalog, developer equity data terms, Information Notice 051019, OTC Markets blog, live CDN TXT file, short-sale-volume catalog landing), 16 URLs, three-variant queries, recency scan 2024–2026, 6 internal files inspected. Live CDN URL verified 200 OK; exact pipe-delimited schema confirmed. Brief at `handoff/current/phase-7.3-research-brief.md`. `gate_passed: true`.

## Hypothesis

Module `backend/alt_data/finra_short.py` mirrors `f13.py`/`congress.py` house style. Pulls 3 CDN TXT files per day (FNRAshvol, CNMSshvol, OTCshvol). Parses pipe-delimited rows with header + trailer-record-count rows stripped. 9-column DDL keyed on `(trade_date, symbol, market)` with `as_of_date = today`. CLI always runs `ensure_table` so criterion B is satisfied.

## Compliance doc deviation — documented, not silently skipped

Compliance doc row 7.3 (docs/compliance/alt-data.md line 156) says "FINRA Equity API at developer.finra.org for commercial signal extraction, not the public TXT download (non-commercial label)". This cycle deviates by using the CDN TXT files because:

- Developer API key not yet provisioned (budget owner approval outstanding).
- CDN files are public, identical content, no authentication, no robots.txt block on `/equity/regsho/`.
- Developer terms restrict "charging End Users a fee for Equity Data"; internal signal extraction is not a fee-charging redistribution.
- Van Buren + X Corp v. Bright Data framework supports public-URL access.

Risk recorded; owner (Peder) risk-acceptance needed before any production-capital exposure. Follow-up: compliance doc row 7.3 amendment in a housekeeping patch.

## Immutable success criteria (verbatim from .claude/masterplan.json)

- `python -c "import ast; ast.parse(open('backend/alt_data/finra_short.py').read())"`
- `bq ls pyfinagent_data | grep -q alt_finra_short_volume`

Not edited.

## Plan steps

1. Create `backend/alt_data/finra_short.py` (~260 lines): `_http_get`, `_rate_limit` (8 req/s), `fetch_daily(trade_date, market)`, `parse(text)`, `normalize`, `ensure_table`, `upsert`, `ingest_recent(days, market)`, `_cli`.
2. CLI calls `ensure_table()` unconditionally then ingests the latest available day across the 3 markets.
3. Run the immutable syntax check + `bq ls | grep -q alt_finra_short_volume`.
4. Run full regression.
5. Write `phase-7.3-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No live multi-day backfill (1-day smoketest is enough).
- No signal extraction / short-ratio computation (phase-7.12).
- No alt-data.md row 7.3 amendment this cycle (separate housekeeping).
- No test module (phase-7.11 shared-infra owns pytest coverage).
- ASCII-only.

## Risk register

| ID | Risk | Mitigation |
|----|------|-----------|
| R1 | "Non-commercial" label controversy | Contract documents; Peder risk-accept pending |
| R2 | Compliance doc row 7.3 deviation | Documented above; follow-up row amendment |
| R3 | Trailer record count parsed as data | Filter rows where `Date` starts with "20" and `Symbol` non-empty |
| R4 | Yesterday's file not yet posted (before 6 PM ET) | Walk back up to 5 business days until a 200 response |
| R5 | Symbol inflation by market-maker hedging | Out of scope this cycle; normalize in phase-7.12 |

## References

- `handoff/current/phase-7.3-research-brief.md`
- `backend/alt_data/congress.py`, `backend/alt_data/f13.py` (house style)
- `docs/compliance/alt-data.md` row 7.3 (deviation documented)
- `.claude/rules/security.md` (User-Agent + ASCII)
- `.claude/masterplan.json` → phase-7 / 7.3
