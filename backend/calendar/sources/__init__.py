"""phase-6.6 calendar source registry -- side-effect imports.

Each submodule calls `register(...)` at import time so that
`backend.calendar.get_sources()` returns the complete list after
`backend.calendar.sources` is imported.
"""
from backend.calendar.sources import (  # noqa: F401
    finnhub_earnings,
    fed_scrape,
    fred_releases,
)
