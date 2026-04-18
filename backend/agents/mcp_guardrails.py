"""phase-3.7 step 3.7.6 guardrails for MCP tool dispatch.

Two defensive layers wrap every in-process MCP tool call:

1. `sliding_window_debounce(max_calls, window_s)` -- O(1) deque-based
   sliding-window rate limiter; raises `DebounceExceeded` when the
   same (tool_name, args-digest) tuple is called more than
   `max_calls` times in `window_s` seconds. Prevents tool-call
   storms (AWS failure-modes post-mortem).

2. `cap_output_size(result, max_bytes)` -- truncates a tool response
   whose JSON-serialized size exceeds `max_bytes` (default 100KB,
   below the ~25K-token Claude Code subprocess ceiling). Records a
   `_truncated: True` flag on the result so the evaluator can audit.

Import patterns:

    from backend.agents.mcp_guardrails import (
        DebounceExceeded,
        sliding_window_debounce,
        cap_output_size,
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from functools import wraps
from typing import Any, Awaitable, Callable, Deque

logger = logging.getLogger(__name__)


DEFAULT_MAX_OUTPUT_BYTES = 100_000  # ~25k tokens at ~4 bytes/token
DEFAULT_DEBOUNCE_MAX_CALLS = 3
DEFAULT_DEBOUNCE_WINDOW_S = 10.0


class DebounceExceeded(RuntimeError):
    """Raised when the sliding-window debounce guard trips."""
    pass


def _args_digest(args: tuple, kwargs: dict) -> str:
    """Deterministic digest of (args, kwargs) for keying the debounce."""
    try:
        payload = json.dumps([args, kwargs], sort_keys=True, default=repr)
    except Exception:
        payload = repr((args, kwargs))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def sliding_window_debounce(
    max_calls: int = DEFAULT_DEBOUNCE_MAX_CALLS,
    window_s: float = DEFAULT_DEBOUNCE_WINDOW_S,
    clock: Callable[[], float] = time.monotonic,
):
    """Decorator: raises DebounceExceeded when the same (fn, args) tuple
    is called more than `max_calls` times in `window_s` seconds.

    Sync + async compatible. Clock is injectable so tests can drive
    time deterministically.
    """
    histories: dict[str, Deque[float]] = {}

    def decorator(fn: Callable):
        key_prefix = getattr(fn, "__qualname__", fn.__name__)

        if hasattr(fn, "__call__") and _is_coroutine_fn(fn):
            @wraps(fn)
            async def awrapper(*args, **kwargs):
                _check_or_raise(key_prefix, args, kwargs,
                                  histories, max_calls, window_s, clock)
                return await fn(*args, **kwargs)
            return awrapper

        @wraps(fn)
        def wrapper(*args, **kwargs):
            _check_or_raise(key_prefix, args, kwargs,
                              histories, max_calls, window_s, clock)
            return fn(*args, **kwargs)
        return wrapper

    return decorator


def _is_coroutine_fn(fn) -> bool:
    try:
        import inspect
        return inspect.iscoroutinefunction(fn)
    except Exception:
        return False


def _check_or_raise(
    key_prefix: str,
    args: tuple,
    kwargs: dict,
    histories: dict[str, Deque[float]],
    max_calls: int,
    window_s: float,
    clock: Callable[[], float],
) -> None:
    now = clock()
    key = f"{key_prefix}:{_args_digest(args, kwargs)}"
    dq = histories.setdefault(key, deque())
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= max_calls:
        raise DebounceExceeded(
            f"{key_prefix} called {len(dq) + 1}x within {window_s}s "
            f"(max={max_calls}); tool-call storm suppressed"
        )
    dq.append(now)


def cap_output_size(
    result: Any,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> Any:
    """If `result` serialized as JSON exceeds `max_bytes`, truncate
    any list/dict values down so the result fits. Records
    `_truncated: True` on the returned dict. Non-dict results are
    wrapped in a carrier dict before truncation."""
    try:
        raw = json.dumps(result, default=repr)
    except Exception:
        raw = repr(result)

    if len(raw.encode("utf-8")) <= max_bytes:
        return result

    # Wrap non-dict results in a carrier so we can annotate.
    if not isinstance(result, dict):
        result = {"value": result}

    result = dict(result)  # shallow copy
    result["_truncated"] = True
    result["_original_size_bytes"] = len(raw.encode("utf-8"))
    result["_max_bytes"] = max_bytes

    # Walk top-level keys; truncate first over-large list/string.
    for k, v in list(result.items()):
        if k.startswith("_"):
            continue
        if isinstance(v, list) and len(v) > 0:
            # Halve the list until under budget (or empty).
            while True:
                blob = json.dumps({**result, k: v}, default=repr)
                if len(blob.encode("utf-8")) <= max_bytes or not v:
                    break
                v = v[: max(len(v) // 2, 1) if len(v) > 1 else 0]
            result[k] = v
            result["_truncated_field"] = k
            break
        if isinstance(v, str) and len(v.encode("utf-8")) > 1024:
            # Truncate string to half until under budget.
            result[k] = v[: max_bytes // 4]
            result["_truncated_field"] = k
            break

    logger.warning("cap_output_size truncated: %s bytes -> %s bytes",
                    result.get("_original_size_bytes"), max_bytes)
    return result
