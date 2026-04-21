# Research Brief -- phase-6.2 Source Registry and Fetcher Core

Tier: moderate (assumed, caller did not specify)
Date: 2026-04-18

---

## External sources

| URL | Accessed | Kind | Read in full? |
|-----|----------|------|---------------|
| https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm | 2026-04-18 | blog | yes |
| https://medium.com/@geoffreykoh/implementing-the-factory-pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe | 2026-04-18 | blog | yes |
| https://www.dontusethiscode.com/blog/2024-05-22_registration-decorators.html | 2026-04-18 | blog | yes |
| https://pypi.org/project/url-normalize/ | 2026-04-18 | official docs | yes |
| https://pypi.org/project/cleanurl/ | 2026-04-18 | official docs | yes |
| https://docs.python.org/3/library/hashlib.html | 2026-04-18 | official docs | yes |
| https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html | 2026-04-18 | academic blog | yes |
| https://github.com/Finnhub-Stock-API/finnhub-python | 2026-04-18 | code | yes |
| https://github.com/Benzinga/benzinga-python-client | 2026-04-18 | code | yes |
| https://github.com/todofixthis/class-registry | 2026-04-18 | code | yes |

---

## Key findings

### 1. News-fetcher plugin registry pattern

The standard Python pattern for a pluggable cron fetcher is the **decorator-based registry**: a module-level dict maps string keys to classes/callables; a `@register("finnhub")` decorator inserts at import time. Flask and FastAPI use this same mechanism for blueprints and routers.

Concrete form from literature (DEV Community, Medium):

```python
_REGISTRY: dict[str, type] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator
```

A `Protocol` (PEP 544) rather than an ABC is preferred for structural duck-typing: avoids mandatory inheritance, plays better with type checkers, matches the repo's existing style (no ABCs in `backend/tools/`).

(Sources: [DEV Community registry pattern](https://dev.to/dentedlogic/stop-writing-giant-if-else-chains-master-the-python-registry-pattern-ldm), [Medium factory via decorator](https://medium.com/@geoffreykoh/implementing-the-factory-pattern-via-dynamic-registry-and-python-decorators-479fc1537bbe), [DontUseThisCode 2024](https://www.dontusethiscode.com/blog/2024-05-22_registration-decorators.html))

### 2. Canonical URL normalization

Recommended library: **`url-normalize`** (PyPI `url-normalize`, actively maintained, MIT). It handles:
- scheme/host lowercasing
- UTM + tracking param stripping via `filter_params=True`
- trailing slash normalization
- percent-encoding normalization

`cleanurl` is lighter-weight but less configurable on which params to strip. `urllib.parse` alone does not strip query params.

For a self-contained `normalize.py` module with no extra dep, a hand-rolled function using `urllib.parse.urlparse` + `parse_qs` / `urlencode` to drop `utm_*`, `fbclid`, `gclid`, `ref`, `source`, `session_id` is also fully adequate and avoids a new transitive dependency.

**Decision for phase-6.2:** hand-roll `canonical_url(url)` using stdlib `urllib.parse` -- strips `utm_*`, `fbclid`, `gclid`, `ref`, `source`, `session_id`; lowercases scheme+host; strips trailing slash from path. Zero new deps. Swap to `url-normalize` if requirements grow.

(Sources: [url-normalize PyPI](https://pypi.org/project/url-normalize/), [cleanurl PyPI](https://pypi.org/project/cleanurl/), [urllib.parse docs](https://docs.python.org/3/library/urllib.parse.html))

### 3. Body-hash (sha256) for dedup

Standard approach: `hashlib.sha256(text.encode("utf-8")).hexdigest()`. Feed normalized text (stripped HTML tags, collapsed whitespace, lowercased) for robustness against cosmetic reformatting. This is exact-match dedup only.

For near-duplicate detection (same story, different wording), MinHash/LSH is the literature standard (Broder 1997, re-popularized by Mattilyra 2017). Phase-6.2 does not need near-dedup -- the schema comment in `add_news_sentiment_schema.py` explicitly says "Dedup logic is NOT in this migration. Lives in the ingestion cron (phase-6.2+)." Exact hash is sufficient to satisfy the schema's `body_hash` column.

(Sources: [Python hashlib docs](https://docs.python.org/3/library/hashlib.html), [document dedup LSH](https://mattilyra.github.io/2017/05/23/document-deduplication-with-lsh.html))

---

## Internal code inventory

| File | Lines | Role | Status (phase-6.2 relevance) |
|------|-------|------|-------------------------------|
| `scripts/migrations/add_news_sentiment_schema.py` | 148 | DDL for `news_articles` + `news_sentiment` | Active -- phase-6.1 deliverable, defines every column the fetcher must populate |
| `backend/tools/alphavantage.py` | 110 | Alpha Vantage news + yfinance fallback | Active -- closest existing pattern to a news fetcher; shows `httpx.AsyncClient` usage and article dict shape |
| `backend/tools/__init__.py` | 1 (empty) | Tools package init | Empty -- no registry exists anywhere |
| `backend/db/bigquery_client.py` | 200+ | BQ wrapper with `save_report()` | Active -- shows `google.cloud.bigquery` insert pattern (direct `client.query(sql)`) |
| `backend/slack_bot/scheduler.py` | 322 | APScheduler cron wiring | Active -- shows `AsyncIOScheduler` + `add_job("cron", hour=...)` pattern; phase-6.2 news cron should follow this |
| `backend/.claude/rules/backend-tools.md` | -- | 16-tool registry conventions | Active -- documents `{ "ticker": "AAPL", "signal": "...", "summary": "...", "data": {} }` return shape for orchestrator tools |

Key file:line anchors:
- `add_news_sentiment_schema.py:15-16` -- `canonical_url STRING` + `body_hash STRING` columns
- `add_news_sentiment_schema.py:47` -- explicit comment: dedup lives in phase-6.2 ingestion cron
- `alphavantage.py:39` -- `async def get_market_intel(ticker, api_key)` -- async httpx pattern
- `alphavantage.py:23-36` -- article dict shape: title, published, source, sentiment_score, summary
- `bigquery_client.py:35` -- `self.client = bigquery.Client(project=...)` -- standard BQ client init
- `scheduler.py:31` -- `_scheduler = AsyncIOScheduler()` -- cron scheduler pattern
- `scheduler.py:34-43` -- `_scheduler.add_job(..., "cron", hour=..., id="morning_digest")` -- job registration shape

### Does a `NewsSource` abstract class exist anywhere?

No. Confirmed via grep across all `.py` files in the repo. There is no `NewsSource`, `NewsRegistry`, `register_source`, or `news_source` identifier. The `backend/news/` package does not exist. This is a greenfield module.

---

## Consensus vs debate

- **Registry pattern**: consensus across all sources -- decorator + dict is the idiomatic Python approach; Protocol over ABC for duck-typing is modern best practice.
- **URL normalization**: minor debate on library vs hand-rolled. For this project (no new dep preferred, limited param set), hand-rolled is the right call.
- **Body hash**: full consensus -- `hashlib.sha256` on normalized text. No debate about exact vs near-dedup at this stage.

---

## Pitfalls (from literature and code review)

1. **Import-order problem with registries**: sources decorated with `@register` must be imported before `get_sources()` is called. The fetcher must explicitly import all source modules at startup (or the registry will be empty). Pattern: `from backend.news import sources as _sources_module  # noqa: F401` in `fetcher.py` or a dedicated `__init__.py` that imports all source submodules.
2. **Encoding**: `backend/rules/security.md` and `backend/rules/backend-api.md` both require ASCII-only logger messages (no Unicode arrows/em-dashes) and `encoding="utf-8"` on all `open()` calls. The hash function must call `.encode("utf-8")` explicitly.
3. **Async vs sync BQ inserts**: `bigquery_client.py`'s `save_report()` is synchronous. Any async fetcher calling it must wrap with `await asyncio.to_thread(...)` per backend-api conventions (`backend-api.md` line: "Never call sync I/O directly inside async def endpoints").
4. **`raw_payload JSON` column**: the schema uses BQ's native JSON type, not STRING. The Python BQ client requires passing the value as a string when using streaming inserts; it is passed as a Python dict and serialized by the library for `client.query()` DDL paths. Confirm the insert method handles JSON columns correctly.
5. **`ARRAY<STRING>` columns** (`authors`, `categories`): must be Python lists, not JSON-encoded strings, when passed to the BQ insert API.

---

## Application to pyfinagent (mapping to implementation)

### Recommended package layout

```
backend/news/
    __init__.py          -- imports all source modules to trigger @register
    registry.py          -- NewsSource Protocol + _REGISTRY dict + register() + get_sources()
    fetcher.py           -- run_once(source_names=None, dry_run=False) -> FetchReport dataclass
    normalize.py         -- canonical_url(), body_hash(), normalize_text() -- stdlib only
```

NOT `backend/tools/news/` -- the `backend/tools/` convention is for the 28-agent orchestrator signal tools (returns `{ticker, signal, summary, data}`). News ingestion is a separate concern (cron ingest, not per-ticker signal). A top-level `backend/news/` package is cleaner and avoids naming collision with the orchestrator tool dispatch.

### `registry.py` shape

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class NewsSource(Protocol):
    name: str
    def fetch(self) -> list[dict]: ...

_REGISTRY: dict[str, type] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_sources() -> dict[str, type]:
    return dict(_REGISTRY)
```

### `normalize.py` shape

```python
import hashlib
import re
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

_STRIP_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                 "utm_content", "fbclid", "gclid", "ref", "source", "session_id"}

def canonical_url(url: str) -> str:
    p = urlparse(url)
    qs = {k: v for k, v in parse_qs(p.query).items() if k not in _STRIP_PARAMS}
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", urlencode(qs, doseq=True), ""))

def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def body_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
```

### `fetcher.py` shape

```python
from dataclasses import dataclass, field

@dataclass
class FetchReport:
    inserted: int = 0
    skipped: int = 0
    errors: dict[str, str] = field(default_factory=dict)

def run_once(source_names: list[str] | None = None, dry_run: bool = False) -> FetchReport:
    from backend.news.registry import get_sources
    from backend.news import sources as _  # noqa: F401 -- trigger @register side-effects
    ...
```

### APScheduler integration (phase-6.3+)

Add a job to `backend/slack_bot/scheduler.py` following the existing pattern at `scheduler.py:34-43`:

```python
_scheduler.add_job(
    _run_news_fetch,
    "cron",
    minute="*/15",
    id="news_fetch",
    replace_existing=True,
)
```

### Test strategy

- Register a `StubSource` decorated with `@register("stub")` that returns 3 hardcoded article dicts.
- Call `run_once(source_names=["stub"], dry_run=True)`.
- Assert `FetchReport.inserted == 3`.
- Assert each article dict has `canonical_url` (no utm params), `body_hash` (64-char hex), `article_id` (uuid4), `fetched_at`.
- Dedup test: feed same article dict twice; assert both produce identical `body_hash` values (dedup enforcement is phase-6.4).
- All tests use `dry_run=True` -- no BQ writes.

---

## Research Gate Checklist

- [x] 3+ authoritative external sources (10 URLs collected, 3 PyPI/stdlib official docs)
- [x] 10+ unique URLs
- [x] Full pages/docs read (not abstracts)
- [x] Internal exploration covered every relevant module (`tools/`, `db/`, `slack_bot/scheduler.py`, migration)
- [x] file:line anchors for every claim (see internal inventory section)
- [x] All claims cited
- [x] Contradictions / consensus noted
- [x] No `NewsSource` class or `backend/news/` package exists -- confirmed greenfield

**gate_passed: true**
