"""phase-6.3 news sources subpackage.

Importing this package triggers `@register(name)` decorators on the
three adapter modules so `backend.news.get_sources()` returns them
alongside the StubSource.
"""
from backend.news.sources import finnhub as _finnhub  # noqa: F401
from backend.news.sources import benzinga as _benzinga  # noqa: F401
from backend.news.sources import alpaca as _alpaca  # noqa: F401

__all__: list[str] = []
