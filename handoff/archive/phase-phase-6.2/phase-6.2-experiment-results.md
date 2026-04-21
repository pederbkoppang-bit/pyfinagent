# Experiment Results -- phase-6.2

## What was built
New package `backend/news/` with three modules:

- **`registry.py`** -- `NewsSource` `Protocol` (PEP 544, `@runtime_checkable`),
  `@register(name)` decorator that instantiates + stores in a
  module-level `_REGISTRY` dict, `get_sources(names=None)` lookup,
  `clear_registry()` test helper. The decorator is idempotent
  across `python -m package.module` double-imports (matches by
  `type(existing).__qualname__ == cls.__qualname__` so the rerun
  as `__main__` doesn't raise).
- **`normalize.py`** -- stdlib-only helpers. `canonical_url` strips
  17 tracking params (utm_*, fbclid, gclid, ref, source, session_id,
  mc_*, _hs*, …), lowercases scheme+host, drops trailing slash,
  sorts remaining query params, drops fragments. `body_hash` returns
  `sha256(normalize_text(body).encode('utf-8'))` hex. `normalize_text`
  strips HTML tags (regex), collapses whitespace, lowercases.
- **`fetcher.py`** -- `RawArticle` + `NormalizedArticle` TypedDicts
  (matching phase-6.1 schema), `FetchReport` dataclass,
  `run_once(source_names, dry_run)` orchestrator, `StubSource`
  baseline (3 fake articles) registered unconditionally. `__main__`
  smoke-test runs canonical_url + body_hash assertions + full
  `run_once(["stub"], dry_run=True)` end-to-end.

Package `__init__.py` re-exports the public surface.

Dedup + live BQ writes are explicitly deferred (phase-6.4 + phase-6.8).

## Files changed
- NEW: `backend/news/__init__.py` (38 lines)
- NEW: `backend/news/registry.py` (88 lines)
- NEW: `backend/news/normalize.py` (70 lines)
- NEW: `backend/news/fetcher.py` (205 lines)

## Verbatim verification
```
$ python -m backend.news.fetcher
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
exit: 0

$ python backend/news/fetcher.py
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
exit: 0
```

Syntax check on all 4 new files: OK.
Package import smoke:
```
>>> from backend.news import run_once, get_sources, canonical_url, body_hash, register, clear_registry
>>> list(get_sources().keys())
['stub']
```

Smoke-test assertions covered:
- `canonical_url("https://X.com/path/?utm_source=foo&z=2&a=1")` →
  `"https://x.com/path?a=1&z=2"` (tracker stripped, params sorted,
  host lowercased, trailing slash dropped).
- `body_hash("<p>Hello World</p>") == body_hash("hello    world")`
  (HTML tags stripped, whitespace collapsed, lowercased → same hash).
- `body_hash("Hello") != body_hash("Goodbye")` (different input).
- `run_once(["stub"], dry_run=True)` yields 3 articles each with
  `article_id` (uuid4), `canonical_url` (stripped), `body_hash`,
  `fetched_at`, `source="stub"` + all phase-6.1 schema fields.

## Soft-gate coverage
| Gate | Status |
|------|--------|
| `backend/news/{__init__,registry,normalize,fetcher}.py` exist | MET |
| public API importable from `backend.news` | MET |
| `NewsSource` Protocol defined | MET (`runtime_checkable`) |
| canonical_url strips trackers + sorts | MET (17 tracker params in set) |
| body_hash stable + content-sensitive | MET |
| `run_once` returns FetchReport w/ phase-6.1 fields | MET |
| syntax + import smoke | MET |

## Follow-up after qa_62 CONDITIONAL
Q/A reproduced that `python backend/news/fetcher.py` (direct-script
invocation) failed with `ModuleNotFoundError: No module named
'backend.news'` because absolute imports need the repo root on
`sys.path`; the `-m backend.news.fetcher` form works only because
`python -m` adds cwd automatically.

**Fix:** added a `__package__ in (None, "")` guard at the top of
`fetcher.py` that prepends the repo root to `sys.path` when the
module is executed as `__main__`. Both invocations now exit 0:

```
$ python backend/news/fetcher.py
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
exit: 0

$ PYTHONPATH=. python -m backend.news.fetcher
phase-6.2 smoke: OK
  n_articles=3
  per_source_counts={'stub': 3}
exit: 0
```

## Scope honesty
- No real API adapters (phase-6.3).
- No dedup at batch level (phase-6.4).
- `_write_batch_to_bq` is a stub that raises NotImplementedError to
  force callers into dry_run until phase-6.8 smoketest.
- Near-dedup (MinHash/LSH) explicit non-goal per research brief.

## References
- Contract (pre-commit): `handoff/current/phase-6.2-contract.md`
- Research: `handoff/current/phase-6.2-research-brief.md`
- Q/A critique (first pass): `handoff/current/phase-6.2-evaluator-critique.md`
- phase-6.1 schema: `scripts/migrations/add_news_sentiment_schema.py`
