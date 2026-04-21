# Experiment Results — phase-7 / step 7.4 (ETF flows scaffold)

**Step:** 7.4 **Date:** 2026-04-19 **Cycle:** 1.

## What was built

One new scaffold file.

`backend/alt_data/etf_flows.py` (~220 lines): scaffold for ETF flows ingestion. Mirrors `finra_short.py` house style. Stub implementations that return sensible empties (no `NotImplementedError`). DDL constant + `_STARTER_TICKERS` tuple baked in. Live implementation deferred to phase-7.12.

Functions: `fetch_issuer_page(ticker)`, `derive_flow(shares_out_t, shares_out_prev, nav)`, `ensure_table`, `upsert`, `ingest_tickers(tickers, dry_run)`, `_cli`.

## File list

Created: 1 (`backend/alt_data/etf_flows.py`).
Modified: 0.

## Protocol-discipline artifact

The research-gate spawn produced the scaffold prematurely (before Main's contract was on disk). Main deleted the researcher-authored file and re-owned the GENERATE step so contract → generate mtime ordering holds. The current file on disk was written AFTER `phase-7.4-contract.md`. Recorded here for auditability; flagged to the researcher prompt for future cycles (research briefs should stop at brief + design proposal, never write production code).

## Verification command output

### Immutable — Python syntax

```
$ python -c "import ast; ast.parse(open('backend/alt_data/etf_flows.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

### Dry-run CLI

```
$ python -m backend.alt_data.etf_flows --dry-run
{"ts": "2026-04-19T21:12:47.481787+00:00", "dry_run": true, "ingested": 0, "scaffold_only": true}
```

`ingested=0` by design — `fetch_issuer_page` is the scaffold that returns `{}`.

### Full regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 13.61s
```

Unchanged.

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `ast.parse` | PASS | Exit 0, "SYNTAX OK". |

## Known caveats

1. **Scaffold only.** `fetch_issuer_page` returns `{}`; no live HTTP, no parsing. Implementation lives in phase-7.12.
2. **No BQ table created.** Criterion doesn't require. Would be created by `ensure_table()` when live implementation lands.
3. **Pre-Q/A self-check:** syntax OK, dry-run CLI runs, regression unchanged, `git status --short` shows only the new file + handoff.
