"""phase-6.7 observability primitives: rate limits, retry, alerting, API call log.

See individual modules for design notes. Intended use: import primitives
here and use them as decorators / context managers inside source adapters
(`backend/news/sources/*.py`, `backend/econ_calendar/sources/*.py`) without
invasive refactors.
"""
from backend.services.observability.rate_limit import (
    get_rate_limiter,
    reset_registry,
)
from backend.services.observability.retry import (
    retry_with_backoff,
    retry_with_backoff_async,
    RetryExhausted,
)
from backend.services.observability.alerting import (
    AlertDeduper,
    raise_cron_alert,
    reset_default_deduper,
)
from backend.services.observability.api_call_log import (
    log_api_call,
    flush,
    buffer_size,
    reset_buffer_for_test,
    log_llm_call,
    flush_llm,
    llm_buffer_size,
)

__all__ = [
    "get_rate_limiter",
    "reset_registry",
    "retry_with_backoff",
    "retry_with_backoff_async",
    "RetryExhausted",
    "AlertDeduper",
    "raise_cron_alert",
    "reset_default_deduper",
    "log_api_call",
    "flush",
    "buffer_size",
    "reset_buffer_for_test",
    "log_llm_call",
    "flush_llm",
    "llm_buffer_size",
]

# phase-75.5 (arch-04): public home for the cloud-spend fetch, promoted out of
# the private slack_bot.jobs.cost_budget_watcher._default_fetch_spend symbol.
from backend.services.observability.spend import (  # noqa: E402
    fetch_spend,
    reset_spend_guard_status,
    spend_guard_status,
)

# phase-75.5 (arch-04, cycle 6): register the promoted symbols in __all__. They were
# re-exported here but omitted from __all__ for five cycles, so `import *` did not bind
# them and the module's public surface disagreed with itself -- while all 15 other
# re-exports WERE listed. The unused-import suppression that used to sit on the import
# above (ruff code F401) was the linter correctly reporting exactly this; with the names
# registered it is no longer needed. arch-04's whole purpose was to give the money guard a PUBLIC home, so leaving
# it out of the public surface undercut the fix. This is the same
# added-without-registering class that masterplan 75.5.7 was queued to generalise --
# found here in the module this step created.
__all__ = [*__all__, "fetch_spend", "spend_guard_status", "reset_spend_guard_status"]
