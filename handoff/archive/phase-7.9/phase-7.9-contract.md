# Sprint Contract — phase-7 / 7.9 (Google Trends scaffold)

**Step id:** 7.9 **Cycle:** 1 **Date:** 2026-04-19 **Tier:** simple

## Research-gate summary

6 sources in full, 14 URLs, three-variant queries, recency scan 2024–2026. Key finding: pytrends was archived 2025-04-17; scaffold references `pytrends-modern>=0.2.5` (drop-in replacement, v0.2.5 released 2026-03-05). 12s sleep between keyword fetches; 6 starter keywords × 12s = 72s/run within compliance doc's 5 req/min. `gate_passed: true`. Brief at `handoff/current/phase-7.9-research-brief.md`.

## Hypothesis

Scaffold-only module mirroring `twitter.py`. `fetch_trend` stubs out; live pytrends-modern call deferred to phase-7.12. DDL for `alt_google_trends` baked in. Immutable criterion: `ast.parse` exits 0.

## Immutable criterion

- `python -c "import ast; ast.parse(open('backend/alt_data/google_trends.py').read())"`

## Plan

1. Write `backend/alt_data/google_trends.py` (~180 lines): `fetch_trend` stub, `ensure_table`, `upsert`, `ingest_keywords`, `_cli`.
2. Verify + regression. Q/A. Log. Flip.

## Out of scope

- No live pytrends-modern call. No Google Trends API key (alpha-restricted).
- No normalization (raw 0–100 stored, rolling-z in phase-7.12 per research brief).
- ASCII-only.

## References

- `handoff/current/phase-7.9-research-brief.md`
- `backend/alt_data/twitter.py` (house style)
- `docs/compliance/alt-data.md` row 7.9
