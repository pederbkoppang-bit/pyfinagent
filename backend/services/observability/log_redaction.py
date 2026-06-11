"""phase-60.4 (criterion 5): logging-layer secret redaction.

The away week left 2,101 plaintext `api_key=...` lines in backend.log --
the FRED key rides in request-URL query strings (fred_data.py) and the
httpx LIBRARY logger prints the full URL. Redaction therefore attaches to
the ROOT HANDLER, not the root logger: per the Python logging docs,
logger-level filters do NOT apply to records emitted by descendant loggers
(httpx, urllib3, ...) -- handler-level filters apply to everything the
handler emits. (OWASP Logging Cheat Sheet: sanitize secrets at the logging
layer, not at every call site.)
"""
from __future__ import annotations

import logging
import re

# Secret-bearing query parameters / kwargs. Values must be 8+ chars so
# innocuous short tokens (key=1) are untouched.
_SECRET_PARAM_RE = re.compile(
    r"(?i)\b(api_key|apikey|api-key|access_token|auth_token|token|secret|client_secret)"
    r"(=|:\s*|%3D)([A-Za-z0-9._%\-]{8,})"
)


def redact_secrets(text: str) -> str:
    """Replace secret-bearing parameter values with a redaction marker."""
    return _SECRET_PARAM_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", text)


class SecretRedactionFilter(logging.Filter):
    """Handler-level filter: rewrites the fully-rendered message in place.

    Returns True always (never drops records -- it sanitizes them).
    Fail-open: any internal error leaves the record unmodified rather than
    losing the log line.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            msg = record.getMessage()
            redacted = redact_secrets(msg)
            if redacted != msg:
                record.msg = redacted
                record.args = ()
        except Exception:
            pass
        return True
