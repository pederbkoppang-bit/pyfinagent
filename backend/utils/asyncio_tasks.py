"""phase-75.10 (api-design-09): shared fire-and-forget task-tracking helper.

Per the official asyncio docs (Tasks -- "Save a reference to tasks"):
"the event loop only keeps weak references to tasks. A task that isn't
referenced elsewhere may get garbage collected at any time, even before
it's done." The fix is a module-level keep-set plus a done-callback that
discards the task once finished.

This module generalizes that pattern across every fire-and-forget
`asyncio.create_task(...)` call site in the app (analysis.py, backtest.py,
paper_trading.py): each site keeps its OWN keep-set and state shape (an
`AnalysisStatus` enum, a `_backtest_state` dict, a bare module string), so
`track_task` takes an `on_error` callback rather than assuming a dict shape.
The wrapped coroutines already self-catch every exception internally and
flip their own state on failure -- `on_error` here is defense-in-depth for
an exception that somehow escapes that internal try/except (e.g. a bug in
the except block itself, or a BaseException), never the primary error path.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Set

logger = logging.getLogger(__name__)


def track_task(
    task: "asyncio.Task",
    tasks: Set["asyncio.Task"],
    on_error: Callable[[BaseException], None],
    label: str,
) -> "asyncio.Task":
    """Keep `task` referenced in `tasks` until it finishes, and invoke
    `on_error(exc)` iff the task raised (never for a cancelled task).

    Returns `task` unchanged so call sites can chain:
        task = track_task(asyncio.create_task(coro()), _tasks, _on_err, "X")
    """
    tasks.add(task)

    def _on_done(t: "asyncio.Task") -> None:
        tasks.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is None:
            return
        logger.error("%s task raised unhandled exception: %r", label, exc, exc_info=exc)
        try:
            on_error(exc)
        except Exception:
            logger.exception("%s on_error callback itself failed", label)

    task.add_done_callback(_on_done)
    return task
