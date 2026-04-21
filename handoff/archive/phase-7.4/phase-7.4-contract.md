# Sprint Contract — phase-7 / step 7.4 (ETF flows ingestion scaffold)

**Step id:** 7.4 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** simple

Parallel-safe: phase-scoped handoff.

## Research-gate summary

5 sources in full (ThisMatter ETF valuation, Apify hiQ legal analysis, etf-scraper PyPI, Invesco NAV doc, Twelve Data ETF API), 13 URLs collected, recency scan 2024–2026, three-variant queries, 4 internal files inspected. Brief at `handoff/current/phase-7.4-research-brief.md`. `gate_passed: true`.

**Protocol note:** researcher prematurely wrote the scaffold module during research-gate. Main deleted it and re-owns the GENERATE to keep the contract→generate ordering intact. This contract is written BEFORE Main's scaffold is committed (log-last-style mtime discipline).

## Hypothesis

The only immutable criterion is `ast.parse` on `backend/alt_data/etf_flows.py`. A scaffold module with valid-but-stub implementations satisfies it. Live ingest, issuer-specific parsers, and CUSIP/AUM reconciliation are deferred to phase-7.12. The scaffold establishes function signatures + module docstring + DDL constant so a future cycle can implement quickly without redesign.

## Immutable success criteria (verbatim from .claude/masterplan.json)

- `python -c "import ast; ast.parse(open('backend/alt_data/etf_flows.py').read())"`

Not edited.

## Plan steps

1. Write `backend/alt_data/etf_flows.py` (scaffold):
   - Module docstring cites the compliance doc row 7.4 + the top-20 ticker starter set.
   - `_USER_AGENT`, `_TABLE = "alt_etf_flows"`, `_RATE_INTERVAL_S = 1.0 / 0.5` (1 req/2s per compliance doc).
   - `_CREATE_TABLE_SQL` constant: 10-column DDL (`flow_id, as_of_date, ticker, issuer, nav, shares_out, shares_out_prev, flow_usd, source, raw_payload`), partition `as_of_date`, cluster `ticker, issuer`.
   - Function signatures (stubs that return sensible empty values — not `NotImplementedError`, so criterion A also holds at runtime if anyone calls them):
     - `fetch_issuer_page(ticker) -> dict`
     - `derive_flow(shares_out_t, shares_out_t_1, nav) -> float`
     - `ensure_table(project, dataset) -> bool`
     - `upsert(rows, project, dataset) -> int`
     - `ingest_tickers(tickers, project, dataset, dry_run) -> int`
     - `_cli(argv) -> int`
   - Scaffold comments flag each stub with `# scaffold - live implementation deferred to phase-7.12`.
2. Run the immutable `ast.parse` check.
3. Run full regression.
4. Write `phase-7.4-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No live fetch.
- No BQ table creation (criterion doesn't require; will come in phase-7.12).
- No pytest.
- ASCII-only.

## References

- `handoff/current/phase-7.4-research-brief.md`
- `backend/alt_data/finra_short.py` (house style scaffold to mirror)
- `docs/compliance/alt-data.md` row 7.4
- `.claude/masterplan.json` → phase-7 / 7.4
