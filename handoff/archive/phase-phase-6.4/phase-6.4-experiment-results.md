# Experiment Results -- phase-6.4

## Research-gate process note (disclosed)
Initial researcher_64 brief cited 3 sources but fetched only 1 in
full. Peder flagged this mid-cycle. Saved durable feedback memory
(`memory/feedback_research_gate_min_three_sources.md`) + spawned
`researcher_64_supplement` which fetched 4 additional sources in
full (Python hashlib docs, Wikipedia URI normalization, RFC 3986
§6, Transloadit SHA-256 dedup article). Supplementary section
appended to the research brief. Gate now passes at the CLAUDE.md
floor (≥3 sources read in full, each cited per claim). All 4
supplementary sources independently validate the shipped design.

## What was built
New module `backend/news/dedup.py` with:
- `DedupReport` dataclass (`n_in`, `n_kept`, `n_dropped_url`, `n_dropped_hash`, `reasons`).
- `dedup_intra_batch(articles)` -- in-memory set filter. Drops an article when its canonical_url OR body_hash was seen earlier. Empty anchors are NOT treated as dedup keys (so two empty-URL articles do not collide).
- `dedup_against_bq(articles, bq_client=None, ...)` -- cross-batch filter against `news_articles`. No-op when `bq_client=None` (dry-run / unit-test safe). When a real client is passed, runs a parameterised UNNEST query over a lookback window and drops matches.

Wired into `fetcher.run_once(dedup=True)` (default True). `FetchReport`
gained `n_deduped`, `dedup_dropped_url`, `dedup_dropped_hash`.
Re-exported from `backend/news/__init__.py`.

## Files changed
- NEW: `backend/news/dedup.py` (173 lines)
- EDIT: `backend/news/fetcher.py` (+10 lines: FetchReport fields + dedup wiring in run_once)
- EDIT: `backend/news/__init__.py` (+2 lines: import + __all__)

## Verbatim verification

Dedup smoke (`backend/news/dedup.py` inline `__main__`):
```
$ python backend/news/dedup.py
phase-6.4 dedup smoke: OK
  intra_batch: n_in=5 n_kept=3 dropped_url=1 dropped_hash=1
```

phase-6.2 regression (stub still yields 3 kept because each
article has a unique canonical_url + body_hash):
```
$ python backend/news/fetcher.py
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
```

End-to-end dedup via `run_once`: synthesized a `DupTestSource` that
emits 4 articles (2 URL dup via tracking-param stripping, 2 body
dup via whitespace/case variation):
```
n_articles kept: 2
n_deduped: 2
dropped_url: 1 dropped_hash: 1
E2E dedup via run_once: OK
```

Syntax check on all 3 touched files: OK.

## Soft-gate coverage
| Gate | Status |
|------|--------|
| `dedup.py` exists with both helpers + DedupReport | MET |
| `run_once(dedup=True)` wires intra-batch dedup | MET |
| Intra-batch 5-article / 2-dup test returns 3 kept | MET |
| Empty anchors NOT treated as dedup anchors | MET |
| `dedup_against_bq(bq_client=None)` is no-op | MET |
| phase-6.2 stub smoke regression still passes | MET |
| Syntax OK | MET |

## Scope honesty
- `dedup_against_bq` NOT exercised against live BQ (no `bq_client`
  passed in dry-run; wiring complete, execution in phase-6.8 smoketest).
- Port stripping + fragment stripping + percent-encoding normalization
  are LOW-risk open audit items per supplementary research --
  deferred to phase-6.8.
- Near-dedup (MinHash/LSH) is explicit non-goal per prior research.

## References
- Contract (pre-commit): `handoff/current/phase-6.4-contract.md`
- Research: `handoff/current/phase-6.4-research-brief.md`
  (incl. supplementary section from researcher_64_supplement)
- phase-6.2 foundation: `backend/news/{normalize,fetcher}.py`
- Peder's gate-breach flag: see `memory/feedback_research_gate_min_three_sources.md`
