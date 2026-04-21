# Sprint Contract — phase-6.5 / step 6.5.7 (Novelty client + prompt-patch queue)

**Step id:** 6.5.7
**Phase:** phase-6.5 Global Intelligence Directive (Path D)
**Cycle:** 1
**Date:** 2026-04-19
**Tier:** moderate

Parallel-safe: all handoff artifacts phase-scoped.

## Research-gate summary

6 sources fetched in full (Voyage docs, Gemini embeddings docs, Gemini GA announcement, fallback/circuit-breaker production guide, 5-min fallback tutorial, BigQuery deduplication pattern piece), 14 URLs collected, recency scan present (2024–2026), three-variant queries visible, 7 internal files inspected. Risks surfaced: R1 `text-embedding-004` deprecated 2026-01-14 (use `gemini-embedding-001`), R2 Voyage=1024-dim vs Gemini default=3072-dim (force `output_dimensionality=1024`), R5 append-only queue requires `latest-status-per-patch_id` read pattern (naive `WHERE status='pending'` returns stale rows). Brief at `handoff/current/phase-6.5.7-research-brief.md`. `gate_passed: true`.

## Hypothesis

A single `novelty_client` module wraps both providers: Voyage primary, Gemini fallback. Both forced to 1024 dimensions. `novelty_score(text, candidate_embeddings) = 1 - max_cosine_similarity`. The `prompt_patch_queue` is append-only over `intel_prompt_patches`; status transitions insert new rows; reads use `ROW_NUMBER() OVER (PARTITION BY patch_id ORDER BY created_at DESC)` or equivalent to get the latest status per patch. Tests use deterministic stub embedding (sha256 → 1024 floats in [-1, 1]) so zero live API calls.

## Empty-pipe acceptance (addresses path_decision.open_issue)

Per phase-6.5 `path_decision.open_issue` (recorded 20:52 UTC), option (b) is selected: **accept empty-pipe at launch**. At ship time the prompt-patch queue is empty in production because no source-specific extractor is writing to `intel_prompt_patches`. The tests assert the code is correct under both populated and empty cases. A follow-up phase-7-integration step will wire extractor outputs into the queue; that step is explicitly out of scope for 6.5.7.

## Immutable success criteria (copied verbatim from .claude/masterplan.json)

- `voyage_primary_gemini_fallback_smoke_ok`
- `novelty_score_distinguishes_duplicate_vs_novel`
- `prompt_patch_queue_persists_and_dedupes`
- `tests_green`

Interpretation:
- `voyage_primary_gemini_fallback_smoke_ok` — `test_voyage_primary_gemini_fallback_smoke_ok` monkeypatches `_embed_voyage` to raise and `_embed_gemini` to return a stub 1024-vector; `embed(text)` must return the Gemini stub vector. Smoke = "at least one provider responds."
- `novelty_score_distinguishes_duplicate_vs_novel` — `test_novelty_score_distinguishes_duplicate_vs_novel` computes score against same-text embedding (expect < 0.1) and against different-text embedding (expect > 0.5).
- `prompt_patch_queue_persists_and_dedupes` — `test_queue_enqueue_then_latest_status_pending` + `test_dedup_in_memory_collapses_duplicates` + `test_enqueue_skips_existing_patch_id` assert the persist + dedup contract.
- `tests_green` — `pytest backend/tests/test_intel_novelty_client.py backend/tests/test_prompt_patch_queue.py -q` exit 0.

## Plan steps

1. Create `backend/intel/novelty_client.py`:
   - `_stub_embed(text)` utility (sha256 → 1024 floats in [-1, 1]) used by tests and as last-resort fallback
   - `_embed_voyage(text, *, model=None)`, `_embed_gemini(text, *, model=None)` — import at call-time, fail-open to raise
   - `_PROVIDERS` sequence; `embed(text, *, model=None)` iterates until success, raises only if all fail
   - `_cosine(a, b)` pure math
   - `novelty_score(chunk_text, candidate_embeddings) -> (score, nn_index)` — returns `(1.0, -1)` when candidates empty
   - `score_chunks_and_write(chunks, *, candidate_embeddings=None, project=None, dataset=None) -> int` — fail-open BQ write
2. Create `backend/intel/prompt_patch_queue.py`:
   - `_patch_id(patch_type, patch_text, chunk_id)` deterministic hash
   - `enqueue_patch(...)`, `get_pending(limit)`, `mark_approved(patch_id, reviewed_by)`, `mark_rejected(patch_id, reason)`, `dedup(patches)` pure in-memory
   - All BQ writes fail-open; reads use latest-status-per-patch_id SQL
3. Create `backend/tests/test_intel_novelty_client.py` (≥7 tests):
   - `voyage_primary_gemini_fallback_smoke_ok`
   - `novelty_score_distinguishes_duplicate_vs_novel`
   - `novelty_score_empty_candidates_returns_one`
   - `embed_returns_1024_dim`
   - `embed_both_providers_fail_raises_runtime_error`
   - `score_chunks_and_write_fail_open_no_bq`
   - `stub_embed_is_deterministic`
   - `module_is_ascii_only`
4. Create `backend/tests/test_prompt_patch_queue.py` (≥7 tests):
   - `dedup_in_memory_collapses_duplicates`
   - `patch_id_is_deterministic_for_same_inputs`
   - `patch_id_differs_for_different_inputs`
   - `enqueue_patch_fail_open_no_bq`
   - `get_pending_fail_open_no_bq`
   - `mark_approved_fail_open_no_bq`
   - `mark_rejected_fail_open_no_bq`
   - `queue_persists_and_dedupes_end_to_end` (monkeypatch the BQ layer; cover persist + dedup)
   - `module_is_ascii_only`
5. Run immutable verification + full regression.
6. Write `phase-6.5.7-experiment-results.md`, spawn Q/A, log-last, flip.

## Out of scope

- No live Voyage or Gemini calls — all embedding in tests goes through the stub + monkeypatch.
- No integration with phase-7 extractors (deferred to the follow-up phase-7-integration step documented in path_decision.open_issue).
- No Slack/dashboard surfacing (dropped under Path D).
- No capital-allocation side-effect (hard boundary from phase goal).
- No MERGE DML; queue is append-only via `insert_rows_json`.

## References

- `handoff/current/phase-6.5.7-research-brief.md`
- `handoff/current/phase-6.5-decision-contract.md` (open_issue option (b))
- `scripts/migrations/phase_6_5_intel_schema.py` (tables `intel_chunks`, `intel_novelty_scores`, `intel_prompt_patches`)
- `backend/intel/source_registry.py`, `backend/intel/scanner.py` (established module style from 6.5.2)
- `backend/news/bq_writer.py:41-97` (fail-open BQ client pattern)
- `.claude/rules/security.md` (ASCII-only logger)
- `.claude/masterplan.json` → phase-6.5 / 6.5.7 (immutable verification)
