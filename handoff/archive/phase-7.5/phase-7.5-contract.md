# Sprint Contract — phase-7 / 7.5 (Reddit WSB sentiment scaffold + license doc)

**Step id:** 7.5 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** moderate

## Research-gate summary

8 sources in full (Reddit API wiki, Arctic Shift GitHub, Redaccs guide, NerdForTech WSB NLP blog, data365 limits, zuplo guide, PainOnSocial rate-limits, TechCrunch 2024 Public Content Policy), 18 URLs, three-variant queries, recency scan 2024–2026 (Responsible Builder Policy, PRAW 7.8.1, Arctic Shift 2026_03, Reddit v Perplexity pending). 5 internal files inspected. `gate_passed: true`. Brief at `handoff/current/phase-7.5-research-brief.md`.

## Hypothesis

Two deliverables mirror the phase-7.6 Twitter pattern plus a first-of-its-kind per-vendor license doc. Scaffold module uses Reddit-specific User-Agent, 100 QPM rate cap, 2-char cashtag floor, script-app OAuth convention, no env-var reads at import time. License doc establishes the template for future per-vendor docs (Revelio 7.7, LinkUp 7.10) with 8-section structure.

## Immutable criteria

- `python -c "import ast; ast.parse(open('backend/alt_data/reddit_wsb.py').read())"`
- `test -f docs/compliance/reddit-license.md`

## Plan

1. Write `backend/alt_data/reddit_wsb.py` (~210 lines, mirrors `twitter.py` with Reddit-specific deltas).
2. Write `docs/compliance/reddit-license.md` (8 sections per brief).
3. Verify both immutable + syntax + ASCII + regression.
4. Q/A. Log. Flip.

## Scaffold discipline

- `fetch_wsb_posts` returns `[]` (stub). `score_sentiment` returns `(0.0, "neutral")`. `extract_cashtags` + `_hash_author` are REAL.
- No `import praw` / `import os.getenv` / `os.environ` at module top. PRAW + env vars deferred to phase-7.12 live impl.
- DDL `alt_reddit_sentiment` (14 columns) baked in.

## Out of scope

- No live PRAW call. No app registration. No RBP submission. No BQ table creation this cycle (criterion is syntax+file, not table existence).
- No sentiment model load.
- No Arctic Shift historical backfill (phase-7.12).
- ASCII-only.

## References

- `handoff/current/phase-7.5-research-brief.md`
- `backend/alt_data/twitter.py` (house style, deltas noted in brief)
- `docs/compliance/alt-data.md` row 7.5 + Sec. 2.2 + Sec. 5 + Sec. 6.2
- `.claude/masterplan.json` → phase-7 / 7.5
