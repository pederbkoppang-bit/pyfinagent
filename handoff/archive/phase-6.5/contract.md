# Sprint Contract -- phase-6.5 Sentiment Scorer Ladder

**Written:** 2026-04-19 (PRE-commit, before any generator work).
**Step id:** `phase-6.5` in `.claude/masterplan.json` phase-6 (News & Sentiment Cron).
**Supersedes rolling contract.md** (was from phase-2.12 autonomous harness cycle 1).

## Research-gate summary

Researcher agent spawned fresh today per `.claude/rules/research-gate.md`. Envelope:
`{tier: moderate, external_sources_read_in_full: 7, snippet_only_sources: 10, urls_collected: 17, recency_scan_performed: true, internal_files_inspected: 9, gate_passed: true}`. Brief at `handoff/current/research_brief.md` (246 lines). Recency scan returned 4 new 2024-2026 findings that changed the ladder design vs the prior 2026-04-18 brief.

Key research decisions (with empirical backing):
- **ProsusAI/finbert over yiyanghkust/finbert-tone** for general news (nosible 2024 benchmark: 69% vs 53%).
- **No chain-of-thought on Haiku scorer** (arxiv 2506.04574v1, 2025: CoT hurts financial sentiment accuracy at every ambiguity level).
- **Haiku 4.5 prompt cache minimum is 4096 tokens** (Anthropic pricing + prompt caching docs, 2026-04-19) -- silent failure below that threshold.
- **Threshold 0.7** for escalation (WASSA 2024 cascade study operating point).
- **Gemini Flash tier-4 is opt-in** (default OFF via settings flag) -- keeps the 4-tier architecture at design level without mandatory dual-LLM cost.

## Hypothesis

A cost-escalating cascade `VADER (free) -> ProsusAI/finbert (local) -> Haiku 4.5 (cached)` will score ~90%+ of news articles using the two local tiers at near-zero marginal cost, escalating only the ~28% ambiguous residual to Haiku. At 10K articles/day projected volume this keeps daily LLM spend at $1.54-$2.33. Gemini Flash tier-4 is plumbed as opt-in for phase-6.9+ calibration.

## Success criteria

NOTE: `.claude/masterplan.json` phase-6.5 has `verification: null` and `contract: null`, so no immutable verification command is inherited. Defining success criteria here per contract discipline:

**Functional:**
1. `backend/news/sentiment.py` module exists and exports:
   - `ScorerResult` dataclass with fields (scorer_model, scorer_version, scored_at, sentiment_score, sentiment_label, confidence, latency_ms, cost_usd, raw_output) matching `news_sentiment` BQ schema (`scripts/migrations/add_news_sentiment_schema.py:24-36`).
   - `score_ladder(article, *, min_confidence=0.7, use_gemini_flash=False) -> ScorerResult` entry point.
   - `VaderScorer`, `FinBertScorer`, `HaikuScorer`, `GeminiFlashScorer` tier classes.
2. VADER rung returns `ScorerResult(scorer_model="vader", ...)` with `score = compound`, `confidence = abs(compound)`, `label` per thresholds (>=0.05 bullish, <=-0.05 bearish, else neutral). Scores title + body[:200] only.
3. FinBERT rung uses `ProsusAI/finbert` (not yiyanghkust), module-level lazy init, truncates to 400 tokens, returns softmax-based score/confidence.
4. Haiku rung uses `anthropic.Anthropic()` directly (NOT `ClaudeClient.generate_content()` which injects a generic prefix -- `llm_client.py:630`), model `claude-haiku-4-5-20251001`, forced `tool_choice={"type":"tool","name":"classify_sentiment"}`, NO `thinking` parameter, system prompt >= 4096 tokens with `cache_control={"type":"ephemeral","ttl":"1h"}`.
5. Escalation logic: `abs(vader.compound) < 0.7` -> FinBERT; `finbert.softmax_max < 0.7` -> Haiku. Early-return at first rung that meets confidence.
6. New settings keys in `backend/config/settings.py`: `sentiment_min_confidence: float = 0.7`, `sentiment_use_gemini_flash: bool = False`, `sentiment_haiku_batch_mode: bool = False`.
7. Gemini Flash tier-4 class exists but raises `NotImplementedError` if called when flag is off; when on, is plumbed but deferred in body to phase-6.9.
8. Fail-open discipline: any tier exception returns `ScorerResult(confidence=0.0, sentiment_label="neutral", sentiment_score=0.0, raw_output=<exception>)` -- never raises.
9. `scorer_model` values match the migration enum exactly: `vader`, `finbert`, `claude-haiku-4-5`, `gemini-2.0-flash`.

**Correctness verification commands:**
- `python -c "import ast; ast.parse(open('backend/news/sentiment.py').read())"` -> exit 0
- `python -c "from backend.news.sentiment import score_ladder, ScorerResult, VaderScorer, FinBertScorer, HaikuScorer, GeminiFlashScorer; print('ok')"` -> stdout `ok`
- `python -c "from backend.news.sentiment import VaderScorer; r = VaderScorer().score({'title':'Company raises guidance on strong Q4 results', 'body':'', 'article_id':'t1'}); assert r.sentiment_label in ('bullish','bearish','neutral'); assert -1.0 <= r.sentiment_score <= 1.0; print('vader ok', r.sentiment_label, round(r.sentiment_score,2), round(r.confidence,2))"` -> runs without error, prints `vader ok ...`
- `pytest backend/tests/test_sentiment_ladder.py -x -q` -> all tests pass

**Non-goals (explicit scope):**
- NOT wiring sentiment writes to BigQuery (phase-6.8 smoketest handles that).
- NOT wiring sentiment into the 15-step analysis pipeline.
- NOT implementing actual Gemini Flash tier-4 logic beyond the stub.
- NOT running FinBERT at import time (must be lazy).
- NOT touching existing news fetcher / dedup / normalize files.

## Plan steps

1. Add 3 settings keys to `backend/config/settings.py`.
2. Create `backend/news/sentiment.py` with dataclasses + 4 scorer classes + `score_ladder()`.
3. Construct the 4096+ token Haiku system prompt and assert at class init.
4. Guard optional deps (vaderSentiment, transformers/torch) with import-time try/except so module loads even without them; fail-open at call site.
5. Write `backend/tests/test_sentiment_ladder.py` (>=5 tests).
6. Export from `backend/news/__init__.py`.
7. Run verification commands, capture output into `experiment_results.md`.

## References

- `handoff/current/research_brief.md` (canonical, 246 lines)
- `scripts/migrations/add_news_sentiment_schema.py:24-36` (news_sentiment schema = source of truth)
- `backend/news/fetcher.py:94-112` (NormalizedArticle shape)
- `backend/agents/llm_client.py:630,657,649` (what NOT to use / cache_control / 4096 threshold)
- `backend/tools/earnings_tone.py:429-443` (cache_control ttl:1h convention)
- `backend/slack_bot/assistant_handler.py:380-459` (forced tool_choice pattern)
- External (read in full): Anthropic pricing + prompt-caching docs, Google Gemini 2.5 Flash pricing, arxiv 2506.04574v1, nosible.ghost.io 2024 benchmark, ProsusAI + yiyanghkust model cards.

## Researcher agent id

`a5d6fc4bbe8238c01`
