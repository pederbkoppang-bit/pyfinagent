"""phase-6 News & Sentiment Cron -- package entry.

Three concerns:
- `registry` -- Protocol + decorator-based source registration.
- `normalize` -- canonical URL + body_hash helpers (stdlib only).
- `fetcher` -- run_once orchestrator + FetchReport dataclass.

Real streaming adapters land in phase-6.3 (Finnhub/Benzinga/Alpaca).
Dedup in phase-6.4. BQ writes wired in phase-6.8 smoketest.
"""
from backend.news.registry import (
    NewsSource,
    register,
    get_sources,
    clear_registry,
)
from backend.news.normalize import canonical_url, body_hash, normalize_text
from backend.news.fetcher import (
    FetchReport,
    NormalizedArticle,
    RawArticle,
    run_once,
)
from backend.news.sentiment import (
    ScorerResult,
    VaderScorer,
    FinBertScorer,
    HaikuScorer,
    GeminiFlashScorer,
    score_ladder,
)

# phase-6.3: side-effect import registers finnhub/benzinga/alpaca.
from backend.news import sources as _sources  # noqa: F401

__all__ = [
    "NewsSource",
    "register",
    "get_sources",
    "clear_registry",
    "canonical_url",
    "body_hash",
    "normalize_text",
    "FetchReport",
    "NormalizedArticle",
    "RawArticle",
    "run_once",
    "ScorerResult",
    "VaderScorer",
    "FinBertScorer",
    "HaikuScorer",
    "GeminiFlashScorer",
    "score_ladder",
]
