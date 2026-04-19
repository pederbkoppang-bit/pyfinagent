# Experiment Results -- phase-11.4 Remove Vertex Deprecation

**Step:** 11.4 (final phase-11 step — phase-11 CLOSES with this commit).
**Date:** 2026-04-19.

## What was built

Researcher-flagged finding: `backend/requirements.txt` has NO `vertexai` line to remove — the `vertexai` namespace is a submodule shipped inside `google-cloud-aiplatform` (still pinned for BigQuery + Vertex AI Search non-generative). The step's immutable verification passes trivially.

The REAL work this cycle was migrating ONE remaining live `vertexai` import that phase-11.0 inventory missed: `backend/tools/nlp_sentiment.py:14-15` used `vertexai.language_models.TextEmbeddingModel`, ALSO deprecated on the same 2026-06-24 deadline as `vertexai.generative_models`.

**`backend/tools/nlp_sentiment.py`:**
- Removed `import vertexai` + `from vertexai.language_models import TextEmbeddingModel`.
- Replaced with the google-genai shim: `from backend.agents._genai_client import get_genai_client`.
- `TextEmbeddingModel.from_pretrained("text-embedding-005")` → `client.models.embed_content(model="gemini-embedding-001", contents=...)`.
- `model.get_embeddings(all_texts)` → `client.models.embed_content(model=..., contents=all_texts).embeddings`.
- Response shape preserved: `.embeddings[i].values` is still the raw float list, so the downstream `np.array(e.values)` cosine-similarity path is unchanged.
- Fail-open guard: shim returning None → raises `RuntimeError("google-genai client unavailable ...")` which the module's existing `except Exception` block converts to a rules-based lexicon fallback (preserves the "4.6.3 at-least-8 non-ERROR" criterion from phase-4.6 contract).

**Model choice: `gemini-embedding-001`** (replaces `text-embedding-005`; it's the google-genai SDK's canonical general-purpose embedding model).

**`backend/requirements.txt`**: NOT modified — `vertexai` was never a top-level pin; removing `google-cloud-aiplatform` would break BigQuery. Researcher's recommendation followed.

## File list

Modified:
- `backend/tools/nlp_sentiment.py` (~15 lines changed; core swap + guard)

NOT modified: `backend/requirements.txt` (no vertexai line to remove); any other file.

## Verification command output

### Immutable (from masterplan contract)

```
$ grep -iE "^vertexai|vertexai==" backend/requirements.txt | wc -l
0
```

Exit 0 (no `vertexai` line). Immutable grep-based verification trivially satisfied.

### Zero live vertexai imports

```
$ grep -rn "^import vertexai\|^from vertexai\b\|^from vertexai\." backend/ --include="*.py" | grep -v __pycache__
(empty)
```

Zero live imports. The only remaining matches in the tree are comments describing the migration history.

### Syntax

```
$ python -c "import ast; ast.parse(open('backend/tools/nlp_sentiment.py').read()); print('SYNTAX OK')"
SYNTAX OK
```

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
79 passed, 1 skipped, 1 warning in 5.51s
```

Zero regressions. Same 79p/1s baseline across all phase-3 + phase-6 + phase-11 tests.

### DeprecationWarning absence

```
$ python -W error::DeprecationWarning -c "from backend.tools.nlp_sentiment import get_nlp_sentiment"
```

Exit 0. (Previously would have triggered the Vertex DeprecationWarning on the language_models import.)

## Contract criterion check (immutable verification from masterplan)

| # | Criterion | Status |
|---|-----------|--------|
| — | `grep ^vertexai requirements.txt` count == 0 | PASS (no line to remove) |
| — | google-genai remains the only Google GenAI SDK pinned (as google-genai==1.73.1; google-cloud-aiplatform stays for BQ/Search) | PASS |
| — | Zero DeprecationWarning from vertexai.* in pytest | PASS (tested; `_UnionGenericAlias` warning is upstream from google-genai types.py, not our code) |

## Known caveats

1. **`gemini-embedding-001` vs `text-embedding-005`** — different model. Both are Google's embedding families; the downstream cosine-similarity math is dimension-agnostic. Production behavior should be equivalent for sentiment polarity, but embedding dimensions may differ (researcher did not deep-dive into dimension parity — documented risk). If a live A/B shows score drift, operator can override via a new settings key in a future cycle.
2. **Live embedding call not exercised in-session** — same caveat as phase-11.1-3. No live GCP auth; test path exercises the rules-based fallback via the `except` block. Real-API smoke will happen in phase-12.4 Rainbow cutover.
3. **`text-embedding-005` is NOT in google-genai's embedding catalog** — using `gemini-embedding-001` per SDK docs. If operator prefers text-embedding-005 specifically, they'd need to route through the older `google-cloud-aiplatform` embedding endpoint (still available post-deprecation) — but that reintroduces the deprecated surface, defeating this step's purpose.
4. **Pre-Q/A self-check**: ran the inventory grep + regression before submitting. Found the nlp_sentiment.py import that the phase-11.0 inventory missed (researcher caught it in phase-11.4 gate). Closing phase-11 with a genuinely zero-live-vertexai tree.

## Phase-11 closure summary

With this step's merge:
- **5/5 phase-11 steps done.**
- Zero live `vertexai.*` imports in `backend/` / `scripts/` (inventory grep).
- Zero DeprecationWarning from Vertex on all migrated import paths (evaluator_agent, skill_optimizer, orchestrator, risk_debate, nlp_sentiment).
- `google-genai==1.73.1` shim consolidates the new SDK surface.
- `google-cloud-aiplatform==1.142.0` preserved for BigQuery + Vertex AI Search non-generative APIs.
- ThinkingConfig silent-breakage closed at the SDK boundary.
- Issue #699 `default`-keys response_schema rejection worked around at SDK boundary.
- 79 tests passing; 0 regressions across the full phase-11 rollout.
