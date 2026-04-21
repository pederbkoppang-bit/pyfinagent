# Sprint Contract — phase-7 / 7.10 (Hiring signals scaffold — LinkUp)

**Step id:** 7.10 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Research-gate summary

6 sources in full (LinkUp Raw product, Datarade LinkUp listing, JoE paper in J. Financial Markets 2023, BrightData provider comparison, Paragon Intel human-capital providers, JobBoardDoctor 2025), 11 URLs, recency scan, three-variant queries, 5 internal files inspected. `gate_passed: true`. Brief at `handoff/current/phase-7.10-research-brief.md`.

## Hypothesis

Scaffold-only module mirroring `twitter.py` + `google_trends.py`. Stubs for `fetch_postings` (defer LinkUp REST + MSA to phase-7.12). Real impls: `_posting_id` sha256 surrogate, `normalize`, `ensure_table`, `upsert`. 12-col DDL for `alt_hiring_signals`, partition `as_of_date`, cluster `ticker, department`.

## Immutable criterion

- `python -c "import ast; ast.parse(open('backend/alt_data/hiring.py').read())"`

## Plan

1. Write `backend/alt_data/hiring.py` scaffold (~200 lines).
2. Verify syntax + dry-run + regression.
3. Q/A. Log. Flip.

## Out of scope

- No live LinkUp API call. No MSA execution. No `LINKUP_API_KEY` read at import.
- No per-vendor license doc this cycle (not required by criterion; future step if/when LinkUp MSA is signed — 7.5 + 7.7 template is available).
- ASCII-only.

## References

- `handoff/current/phase-7.10-research-brief.md`
- `backend/alt_data/twitter.py`, `backend/alt_data/google_trends.py` (house style)
- `docs/compliance/alt-data.md` row 7.10
- `.claude/masterplan.json` → phase-7 / 7.10
