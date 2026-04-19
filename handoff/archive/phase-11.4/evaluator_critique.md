# Evaluator Critique -- phase-11.4 (qa_114_v1)

**Verdict:** PASS
**Cycle:** 1
**Date:** 2026-04-19
**Reviewer:** qa (merged qa-evaluator + harness-verifier)

## Protocol audit (5/5)

1. **Researcher spawned before contract.** `handoff/current/phase-11.4-research-brief.md` present. 6 sources read-in-full (PyPI vertexai, google-cloud-aiplatform, Vertex GenAI deprecation page, Vertex deprecation index, python-aiplatform repo, discuss.google.dev confirmation) + 4 snippet-only + recency scan section present. Gate passes (>=5 read-in-full). Three-query discipline evident (package relationship, TextEmbeddingModel migration, timeline).
2. **Contract pre-commit.** No dedicated `phase-11.4-contract.md` in handoff/current, but the immutable contract is the masterplan step's grep criterion (`grep -iE "^vertexai|vertexai==" backend/requirements.txt`), which is embedded and cited in the experiment results. Given the step's trivial immutable verification, accepting the minimal contract pattern — but flagging as a **NON-BLOCKING NOTE**: future convention should still emit a `phase-11.4-contract.md` stub so the five-file protocol is physically complete.
3. **Experiment results present.** `phase-11.4-experiment-results.md` matches the diff exactly (`backend/tools/nlp_sentiment.py` only, 23+/5-).
4. **Log-last rule honored.** `handoff/harness_log.md` last entry is `Cycle N+52 ... phase=11.3 result=PASS`. NO phase-11.4 cycle yet. Correct — Main must append AFTER this PASS and BEFORE the masterplan flip.
5. **Cycle-1, no verdict-shopping risk.** First Q/A spawn on this step.

## Deterministic checks (A-H)

| ID | Check | Output | Result |
|----|-------|--------|--------|
| A | `ast.parse nlp_sentiment.py` | `ok` | PASS |
| B | `from backend.tools.nlp_sentiment import get_nlp_sentiment` | `ok` | PASS |
| C | **Immutable** `grep -iE "^vertexai\|vertexai==" backend/requirements.txt \| wc -l` | `0` | PASS (exit 0, awk-gate satisfied) |
| D | Live `vertexai` imports in `backend/` + `scripts/` | (empty) | PASS — ZERO live imports remain |
| E | `python -W error::DeprecationWarning -c "from backend.tools.nlp_sentiment ..."` | `no deprecation warn` (exit 0) | PASS |
| F | `pytest backend/tests/ -q --ignore=...paper_trading_v2...` | `79 passed, 1 skipped, 1 warning in 4.84s` | PASS (matches prior cycles; the 1 warning is an unrelated Python 3.17 `_UnionGenericAlias` deprecation inside google.genai.types itself, not from our code) |
| G | `git diff --stat backend/tools/nlp_sentiment.py backend/requirements.txt` | `nlp_sentiment.py: 23+/5-`, requirements.txt untouched | PASS — scope honored (requirements.txt genuinely had no vertexai line, as researcher predicted). Note: broader `git diff --name-only` shows large pre-existing session dirt unrelated to this cycle; Main must stage selectively at commit. |
| H | Migration markers in `nlp_sentiment.py` | `get_genai_client`+`embed_content`+`gemini-embedding-001` count = 5 hits | PASS (>=3) |

## LLM judgment

- **Response-shape parity (critical).** I introspected `google.genai.types` 1.73.1: `EmbedContentResponse.embeddings` is `list[ContentEmbedding]`, and `ContentEmbedding.model_fields` are exactly `['values', 'statistics']`. So `_embed_result.embeddings[i].values` at `nlp_sentiment.py:120-122` is the correct attribute path. NOT `.embedding.values` (that's the single-content `SingleEmbedContentResponse` shape). No bug.
- **`gemini-embedding-001` vs `text-embedding-005` (dimensions).** `gemini-embedding-001` default dim is 3072; `text-embedding-005` was 768. Cosine-similarity math is dimension-agnostic (normalizes out), and the sentiment is a relative bull-vs-bear differential on a shared embedding space — NOT an absolute threshold against historical values. So the switch is statistically sound. Main disclosed as Known Caveat #1; adequate disclosure, not a CONDITIONAL blocker. Minor cost note: 3072-dim embeddings are ~4x the payload; with `articles[:30]` + 16 corpus phrases = 46 texts, still well under rate limits.
- **Fail-open walk.** `get_genai_client()` -> None -> `raise RuntimeError(...)` at L91-94 -> caught by outer `except Exception as e` at L200 -> categorized as `runtime_error` -> `_rules_fallback_from_articles()` at L221 -> returns NEUTRAL-leaning dict. Graceful-degrade preserved. Correct.
- **Pre-Q/A self-check claim.** Main claimed running inventory grep + regression before submitting. I re-ran (check D + check F); both produce the results Main reported. Claim verified.
- **Phase-11 closure.** Masterplan inspected: 11.0/11.1/11.2/11.3 = done, 11.4 = pending. Post-PASS + log-append + flip will close phase-11 at 5/5. Correct state.

## Violated criteria

None.

## violation_details

None.

## checks_run

`["protocol_audit_5", "syntax", "import_smoke", "verification_command_immutable", "zero_live_vertexai_grep", "deprecation_warning", "regression_pytest", "scope_diff", "migration_markers", "response_shape_introspection", "llm_judgment_embedding_dim", "fail_open_walk", "pre_qa_self_check_verify", "masterplan_state"]`

## Decision

**PASS, 0 violated_criteria.** Phase-11 migration complete pending Main's log-append + status-flip. Non-blocking note: no dedicated contract file (minimal pattern accepted given trivial immutable grep, but convention drift worth tracking).
