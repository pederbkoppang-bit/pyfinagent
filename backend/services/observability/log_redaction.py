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


# phase-82.7: the libraries that print a full request URL. `httpx` logs
# "HTTP Request: GET <url> ..." at INFO, which is how the FRED key reached an
# operator-visible console on 2026-08-03. Filtering these BY NAME makes the
# redaction independent of whether any handler happens to be installed yet.
_EMITTING_LIBRARY_LOGGERS = ("httpx", "httpcore", "urllib3", "requests")


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
            # phase-82.7 (gap G3): the rendered EXCEPTION is a second channel
            # out. `record.getMessage()` covers only the format string; an
            # httpx.HTTPStatusError repr carries the full request URL -- key
            # included -- and reaches the handler through exc_text/exc_info,
            # which the phase-60.4 filter never touched. So a call that logged
            # nothing sensitive itself would still leak the moment it FAILED,
            # which is exactly when it gets logged at ERROR.
            if record.exc_info:
                # PRE-RENDER the traceback here, scrub it, and cache it on the
                # record. `logging.Formatter.format` uses `record.exc_text`
                # when it is already set and only renders the traceback itself
                # when it is empty -- so caching a scrubbed copy is what stops
                # the handler re-rendering the raw exception downstream.
                #
                # Deliberately NOT `type(exc)(cleaned)`: reconstructing the
                # exception fails for any type whose __init__ takes required
                # kwargs. httpx.HTTPStatusError needs request= and response=,
                # so that approach raised, hit the fail-open except below, and
                # silently left the credential intact -- a "fix" that could not
                # fail loudly. Caching exc_text needs no constructor.
                if not record.exc_text:
                    record.exc_text = logging.Formatter().formatException(record.exc_info)
                record.exc_text = redact_secrets(record.exc_text)
        except Exception:
            pass
        return True


def install_secret_redaction(logger: logging.Logger | None = None) -> int:
    """Attach the redaction filter to every handler on `logger` (default root).

    IDEMPOTENT: a handler that already carries a `SecretRedactionFilter` is
    skipped, so calling this twice in one process is safe. Returns the number
    of handlers newly filtered.

    phase-82.7. Why this exists rather than "just call setup_logging()":
    `backend.main.setup_logging` is single-call-only and DESTRUCTIVE -- it does
    `root.handlers.clear()` and re-wraps `sys.stderr.buffer` in a fresh
    TextIOWrapper, observed live as `ValueError: I/O operation on closed file`,
    and the blanket clear also destroys pytest's log capture. It was also only
    ever invoked from the FastAPI lifespan (`backend/main.py:151`), so the 54
    `logging.basicConfig()` bootstraps across backend/, scripts/ and functions/
    each installed a root handler with NO redaction. The facility was correct
    and simply unreachable from every non-FastAPI entry point.

    Call this AFTER `basicConfig()` in a script's entry point. It attaches to
    HANDLERS, not to the logger, because logger-level filters do not apply to
    records emitted by descendant loggers such as `httpx` (Python logging docs)
    -- and httpx logging the full request URL is the leak being closed.
    """
    target = logger if logger is not None else logging.getLogger()
    added = 0
    for handler in list(target.handlers):
        if any(isinstance(f, SecretRedactionFilter) for f in handler.filters):
            continue
        handler.addFilter(SecretRedactionFilter())
        added += 1

    # Named-logger attachment as a second leg. NOTE ITS LIMIT, because getting
    # this wrong is what the 82.7 Q/A caught: a logger-level filter applies ONLY
    # to records logged through THAT logger, never to its descendants. The
    # actual emitters are descendants -- `urllib3.connectionpool`, not
    # `urllib3`; `httpcore.http11`, not `httpcore` -- so this leg alone leaks,
    # measured. It is kept because it costs nothing and covers the parent-named
    # emitters, but the load-bearing coverage is the handler leg above plus the
    # addHandler hook below.
    for name in _EMITTING_LIBRARY_LOGGERS:
        lib = logging.getLogger(name)
        if not any(isinstance(f, SecretRedactionFilter) for f in lib.filters):
            lib.addFilter(SecretRedactionFilter())
            added += 1

    # THE DURABLE LEG. Handlers see every record that PROPAGATES to them, from
    # any logger at any depth -- so a filtered handler covers descendants that
    # named-logger filters cannot. The only hole is a handler installed AFTER
    # this call, which is exactly the script shape that defeated the previous
    # design: `install()` then `logging.basicConfig()` creates a fresh,
    # unfiltered root handler and nothing re-attaches. Hooking `addHandler`
    # closes that hole for every handler added from now on, at zero per-record
    # cost.
    _install_addhandler_hook()
    return added


_HOOK_INSTALLED = False


def _install_addhandler_hook() -> None:
    """Ensure handlers added LATER are filtered too. Idempotent, process-wide.

    Wraps `logging.Logger.addHandler` so any handler attached after
    `install_secret_redaction()` -- notably by a script's `logging.basicConfig()`
    -- carries the filter. Without this, redaction depends on call ORDER, and
    the checked-in `scripts/migrations/extend_historical_data.py` has exactly
    the losing order: it imports a leak site (installing while root has no
    handlers) and only then calls `basicConfig`.
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    original = logging.Logger.addHandler

    def _patched(self, hdlr):  # noqa: ANN001
        original(self, hdlr)
        try:
            if not any(isinstance(f, SecretRedactionFilter) for f in hdlr.filters):
                hdlr.addFilter(SecretRedactionFilter())
        except Exception:
            # Fail-open: never let redaction bookkeeping break logging setup.
            pass

    logging.Logger.addHandler = _patched  # type: ignore[method-assign]
    _HOOK_INSTALLED = True
