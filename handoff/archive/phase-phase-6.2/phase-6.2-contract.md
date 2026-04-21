# Sprint Contract -- phase-6.2
Step: Source registry and fetcher core

## Research Gate
researcher_62 (tier=moderate) gate_passed=true. Brief: `handoff/current/phase-6.2-research-brief.md`.
Key findings:
- Decorator-based Python registry pattern (dev.to / dontusethiscode 2024 / PEP 544 Protocol) is the canonical shape.
- `url-normalize` vs hand-rolled: hand-rolled `urllib.parse`-only function avoids a new dep and is sufficient for UTM/fbclid/gclid/ref/source/session_id stripping.
- `hashlib.sha256` on `normalize_text(body).encode("utf-8")` for exact-match dedup; near-dedup (MinHash/LSH) is out of scope.
- No existing `backend/news/` package; greenfield.
- `bigquery_client.py` is sync -> async fetchers must wrap BQ inserts in `asyncio.to_thread` per backend-api.md.
- `ARRAY<STRING>` + `JSON` BQ types: pass `list[str]` / dict directly, NOT JSON strings.

## Hypothesis
Creating `backend/news/` package with three modules (`registry.py`,
`normalize.py`, `fetcher.py`) + a stub test source satisfies phase-6.2.
The fetcher orchestrates: iterate registered sources → call `.fetch()` →
normalize canonical_url + body_hash → assemble article dicts matching
the phase-6.1 schema → return a `FetchReport`. No real API calls; no
live BQ writes this step (phase-6.3 adds real adapters, phase-6.4
adds dedup, phase-6.8 adds end-to-end run).

## Success Criteria (masterplan has verification=None; soft gates)
1. `backend/news/{__init__,registry,normalize,fetcher}.py` exist.
2. `from backend.news.registry import register, get_sources` importable.
3. `NewsSource` Protocol with `.fetch() -> Iterable[RawArticle]`.
4. `canonical_url("https://x.com/a?utm_source=foo&id=1")` == `"https://x.com/a?id=1"`.
5. `body_hash` stable + same input → same output; different input → different output.
6. `run_once([stub_source_name], dry_run=True)` returns a FetchReport with N article dicts each carrying article_id (uuid4), canonical_url, body_hash, fetched_at, source, + pass-through fields.
7. Each normalized article matches the phase-6.1 `news_articles` column set.
8. Syntax + import-smoke clean.

## Plan (PRE-commit)
1. Create package skeleton: `backend/news/__init__.py` imports from the sub-modules.
2. `registry.py`:
   - `NewsSource` Protocol with `name: str` and `fetch() -> Iterable[dict]`.
   - `_REGISTRY: dict[str, NewsSource]` module-level.
   - `@register(name)` decorator + `get_sources(names=None) -> dict[str, NewsSource]`.
3. `normalize.py`:
   - `canonical_url(url) -> str` strips a known UTM/tracking set, lowercases scheme+host, drops trailing slash, sorts remaining query params.
   - `normalize_text(text) -> str` strips HTML tags (simple regex), collapses whitespace, lowercases.
   - `body_hash(text) -> str` returns `sha256(normalize_text(text).encode("utf-8")).hexdigest()`.
4. `fetcher.py`:
   - `RawArticle` + `NormalizedArticle` TypedDicts (match phase-6.1 schema).
   - `FetchReport` dataclass (n_sources, n_articles, per_source_counts, errors).
   - `run_once(source_names=None, dry_run=False) -> FetchReport`.
   - Registers `StubSource` (in `fetcher.py` or a sibling `sources/stub.py`) so tests have something to iterate.
5. Tests as inline `if __name__ == "__main__":` smoke (no pytest dep required for the verification): run `run_once(["stub"], dry_run=True)` + assert 3 articles + check a known canonical_url case.

## Scope out-of-scope
- Real API adapters (Finnhub/Benzinga/Alpaca) → phase-6.3.
- Dedup at the BQ layer → phase-6.4.
- Live BQ writes → wired in phase-6.8 smoketest.
- Near-dedup (MinHash/LSH) → explicit non-goal per the research brief.

## References
- Research brief: `handoff/current/phase-6.2-research-brief.md`
- phase-6.1 schema: `scripts/migrations/add_news_sentiment_schema.py`
- Cite precedents: `backend/tools/alphavantage.py` async pattern, `backend/slack_bot/scheduler.py` job registration shape.
