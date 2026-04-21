# Q/A Critique -- phase-6.4 (qa_64)

## Harness audit
1. Researcher spawn: PRESENT. `phase-6.4-research-brief.md` has supplementary section from `researcher_64_supplement`. Verified **4 sources fetched in full** (hashlib docs, Wikipedia URI normalization, RFC 3986 §6, Transloadit SHA-256 article; plus url-normalize PyPI as 5th). Supplement is **substantive**: each source has distinct production/standards content (FIPS 180-4 collision bound, RFC §6 comparison ladder, two-phase pattern confirmation, port/fragment audit items for phase-6.8) -- not a re-grep of the original Medium blog. Meets >=3 read-in-full floor.
2. Contract PRE-commit: contract mtime 08:55:05 <= experiment-results mtime 08:55:28. PASS.
3. experiment_results.md: verbatim verification present; research-gate slip + repair openly disclosed.
4. Log-last: masterplan phase-6.4 status=pending, harness_log not yet appended. Correct ordering.
5. Fresh qa_64 spawn confirmed.

## Code checks
- Syntax OK on all 3 touched files.
- `python backend/news/dedup.py` -> "phase-6.4 dedup smoke: OK" (n_in=5 kept=3).
- `python backend/news/fetcher.py` regression -> "phase-6.2 smoke: OK" (n_articles=3).
- Empty-anchor test (2 empty -> 2 kept): PASS.
- `dedup_against_bq(sample, bq_client=None)` returns full sample: PASS (returns list).
- FetchReport has `n_deduped`, `dedup_dropped_url`, `dedup_dropped_hash`: PASS.
- `run_once(dedup=True)` wires intra-batch dedup at fetcher.py:157-163, BEFORE `_write_batch_to_bq` guard at :167: PASS.
- E2E negative test (4 articles, 2 UTM-url dups + 2 whitespace body dups) -> kept=2, n_deduped=2, dropped_url=1, dropped_hash=1: PASS.

## LLM judgment
Research repair substantive and cited per claim. Implementation faithful: two-phase pattern correct, empty-anchor skip correct (`if url and url in seen_urls`), drop-on-EITHER-match correct (separate URL/hash branches with independent `continue`).

**Verdict: PASS**
