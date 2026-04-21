# Experiment Results — phase-6.5 / step 6.5.7 (Novelty client + prompt-patch queue)

**Step:** 6.5.7 — third executable step under Path D.
**Date:** 2026-04-19.
**Cycle:** 1.

## What was built

Four new files; zero existing code changed.

1. `backend/intel/novelty_client.py` (~220 lines) — Voyage primary / Gemini fallback embedding wrapper. Both providers forced to 1024-dim (research R2). `_stub_embed` produces deterministic sha256-based 1024-float vectors for tests + last-resort fallback. `novelty_score(text, candidates)` returns `(1 - max_cosine_sim, nearest_idx)`; empty candidates → `(1.0, -1)`. `score_chunks_and_write` writes to `intel_novelty_scores` with fail-open BQ.
2. `backend/intel/prompt_patch_queue.py` (~200 lines) — append-only queue over `intel_prompt_patches`. Deterministic `_patch_id = sha256(patch_type:patch_text:chunk_id)[:16]`. `enqueue_patch` always returns the pid (idempotent). `get_pending` uses `ROW_NUMBER() OVER (PARTITION BY patch_id ORDER BY created_at DESC)` SQL pattern to return latest-status-per-patch_id (research R5 mitigation). `mark_approved` / `mark_rejected` insert new status rows, no UPDATE.
3. `backend/tests/test_intel_novelty_client.py` — 11 tests including the immutable criterion targets.
4. `backend/tests/test_prompt_patch_queue.py` — 11 tests including the end-to-end persist-and-dedupe test.

## File list

Created: 4. Modified: 0.

## Verification command output

### Immutable (masterplan 6.5.7)

```
$ source .venv/bin/activate && pytest backend/tests/test_intel_novelty_client.py backend/tests/test_prompt_patch_queue.py -q
......................                                                   [100%]
22 passed in 7.26s
EXIT=0
```

### Full regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
152 passed, 1 skipped, 1 warning in 14.72s
```

Baseline before: 130 passed / 1 skipped (6.5.2 close). Delta = +22 (11 novelty + 11 queue). Zero regressions on pre-existing surface.

## Contract criterion check

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `voyage_primary_gemini_fallback_smoke_ok` | PASS | `test_voyage_primary_gemini_fallback_smoke_ok` forces Voyage to raise, Gemini to return stub, and `embed()` returns a 1024-element list[float]. |
| 2 | `novelty_score_distinguishes_duplicate_vs_novel` | PASS | `test_novelty_score_distinguishes_duplicate_vs_novel` asserts `score_dup < 0.1` and `score_novel > 0.5` using `_stub_embed`. |
| 3 | `prompt_patch_queue_persists_and_dedupes` | PASS | `test_queue_persists_and_dedupes_end_to_end` monkeypatches `_insert` with a captive store that rejects re-pending a same-pid row; enqueuing the same patch twice collapses to 1 persisted row with identical pid. Plus `test_dedup_in_memory_collapses_duplicates` and `test_dedup_preserves_first_occurrence` for the pure-function path. |
| 4 | `tests_green` | PASS | 22/22, exit 0. |

## Mid-cycle bug caught

Original `enqueue_patch` returned `None` when `_insert` returned 0 (which happens both on BQ failure AND when the fake-store skipped a re-pending row). The end-to-end dedup test then failed with `pid1 != pid2` because the second call returned None instead of the deterministic patch_id. Fixed: `enqueue_patch` now always returns the deterministic pid — BQ failures and dedup-skips are logged and swallowed. Callers that need insert-confirmation call `get_pending`. Documented in the module docstring. This is the cycle's anti-rubber-stamp artifact (no rubber-stamp "green on first try" claim).

## Known caveats (transparency)

1. **No live Voyage or Gemini calls this cycle.** All `embed()` paths are exercised via monkeypatch + `_stub_embed`. A future phase-7-integration step or smoketest should do one live round-trip with real API keys before production.
2. **`_stub_embed` is 1024-dim tiled-sha256, not cryptographically random.** It's deterministic on input by design (so tests reproduce). That also means two texts with the same sha256 prefix will collide; trivially unlikely for normal inputs.
3. **Empty-pipe acceptance at ship time.** Per phase-6.5 `path_decision.open_issue`, the prompt-patch queue ships with no source-specific extractor writing to it. The modules are correct under both populated and empty cases; a follow-up phase-7-integration step will wire extractors in.
4. **Cross-batch dedup is read-side, not write-side.** The test's fake-store enforces "skip if latest status is pending" because BQ streaming has no UPSERT. Real production uses `get_pending`'s latest-per-pid SQL to hide the stale pending rows. A duplicate pending row is wasted storage, not a correctness issue.
5. **`score_chunks_and_write` does not embed candidates.** It accepts `candidate_embeddings` pre-computed. Future callers who need "dedup against the whole corpus" must fetch candidate embeddings first (via `intel_chunks.embedding` in BQ). That's phase-7-integration scope.

## Pre-Q/A self-check

- Immutable pytest exit 0 (22/22).
- Full regression 152 passed (baseline 130 + 22 new; zero broken).
- Python AST parses cleanly on all 4 new files.
- `git status --short` shows only new files under `backend/intel/`, `backend/tests/`, and handoff — no production backend module outside the intel package modified.
- Handoff files phase-scoped: `phase-6.5.7-{contract,experiment-results,research-brief}.md`.
- Masterplan NOT flipped yet; log-last discipline preserved.
