# Sprint Contract -- phase-6.4
Step: Dedup layer with canonical URL + body hash

## Research Gate
researcher_64 (tier=simple) initial brief cited 3 sources but only
fetched 1 in full -- Peder flagged this as a research-gate breach
mid-cycle. Saved feedback memory
(`memory/feedback_research_gate_min_three_sources.md`), then spawned
**researcher_64_supplement** which fetched 4 more sources **in full**:

- Python hashlib docs (SHA-256 in `algorithms_guaranteed`, 64-char hex, collision-resistant)
- Wikipedia URI normalization (host-lowercase + tracking-param removal + query-sort are the standard dedup heuristics)
- RFC 3986 §6 (URI comparison ladder; level-2 syntax-based normalization is the target)
- Transloadit SHA-256 dedup engineering post (independent production validation of two-phase design; MD5 explicitly rejected)

Gate now passes with 4 sources read in full. All 4 independently
validate the shipped design. Full brief:
`handoff/current/phase-6.4-research-brief.md` (supplementary section
appended).

Key:
- Two-phase dedup pattern (intra-batch set + cross-batch BQ lookup) is canonical for exact-match SHA-256 dedup.
- Empty `canonical_url`/`body_hash` must be skipped as dedup anchors (all empty strings would collide).
- Drop on EITHER anchor match, not both.
- BQ streaming buffer caveat: same-run inserts may not be visible -- acceptable at this phase, document.
- Open audits for phase-6.8: port-stripping completeness, fragment stripping, percent-encoding normalization (all LOW risk for news URLs in practice).

## Hypothesis
Creating `backend/news/dedup.py` with `dedup_intra_batch(articles) -> (kept, DedupReport)` and `dedup_against_bq(articles, bq_client=None, ...)` (no-op when `bq_client is None`), plus wiring `dedup: bool = True` into `run_once`, satisfies phase-6.4. A new `DedupReport` dataclass carries counts (`n_in`, `n_kept`, `n_dropped_url`, `n_dropped_hash`). `FetchReport` gains a `n_deduped` field sourced from the report.

## Success Criteria (soft)
1. `backend/news/dedup.py` exists with the two helpers + DedupReport dataclass.
2. `run_once(dedup=True)` calls intra-batch dedup after assembly, before the BQ-write guard.
3. Intra-batch test: 5-article list with 2 duplicates (1 URL match, 1 hash match) returns 3 kept.
4. Empty `canonical_url` / `body_hash` are NOT treated as dedup anchors.
5. `dedup_against_bq(..., bq_client=None)` returns input unchanged.
6. Regression: phase-6.2 stub smoke still exits 0 (`run_once(["stub"], dry_run=True)` still gets 3 articles because the stub emits 3 unique URLs/bodies).
7. Syntax OK.

## Plan (PRE-commit)
1. Write `backend/news/dedup.py` with:
   - `@dataclass DedupReport(n_in, n_kept, n_dropped_url, n_dropped_hash, reasons)`.
   - `dedup_intra_batch(articles) -> (list[NormalizedArticle], DedupReport)`.
   - `dedup_against_bq(articles, bq_client=None, dataset='pyfinagent_data', lookback_days=7, dry_run=True) -> list[NormalizedArticle]`. With `bq_client=None` OR `dry_run=True`, no-op return all.
2. Add `n_deduped: int = 0` to `FetchReport` in `backend/news/fetcher.py`.
3. Wire `dedup: bool = True` parameter on `run_once`. After batch assembly, call `dedup_intra_batch`; set `report.n_deduped = len(batch) - len(kept)`; replace `report.articles = kept`. Only call `dedup_against_bq` when not dry_run AND a bq_client is explicitly passed (deferred to phase-6.8).
4. Extend `_smoke()` or add a dedup-specific smoke block that seeds 5 articles (3 unique + 1 URL dup + 1 hash dup), asserts 3 kept.
5. Re-export `dedup_intra_batch`, `DedupReport` from `backend/news/__init__.py`.

## Scope out
- Cross-batch BQ dedup executed against live BQ -- wired but no-op in this cycle.
- Near-dedup (MinHash/LSH) -- explicit non-goal.

## References
- Research brief: `handoff/current/phase-6.4-research-brief.md`
- phase-6.2 foundation: `backend/news/{normalize,fetcher}.py`
