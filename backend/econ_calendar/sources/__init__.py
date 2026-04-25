"""phase-6.6 calendar source registry -- side-effect imports.

Each submodule calls `register(...)` at import time so that
`backend.econ_calendar.get_sources()` returns the complete list after
`backend.econ_calendar.sources` is imported.
"""
from backend.econ_calendar.sources import (  # noqa: F401
    finnhub_earnings,
    fed_scrape,
    fred_releases,
)
