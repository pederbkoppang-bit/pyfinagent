"""phase-6.7 observability primitives: rate limits, retry, alerting, API call log.

See individual modules for design notes. Intended use: import primitives
here and use them as decorators / context managers inside source adapters
(`backend/news/sources/*.py`, `backend/calendar/sources/*.py`) without
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
