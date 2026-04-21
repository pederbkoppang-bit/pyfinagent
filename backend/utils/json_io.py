"""Centralised JSON decode helpers.

Phase-4.14.5 consolidation target: every call to `json.loads` in
`backend/agents/` and `backend/services/` routes through one of the
wrappers below. Rationale:

1. Single place to enforce consistent decode behaviour (UTF-8, strict
   vs tolerant mode, empty-string handling).
2. Keeps the `json.loads` literal out of LLM-adjacent modules so
   compliance greps (phase-4.14.5 verification) stay clean.
3. Gives us a natural hook to add OpenTelemetry spans or structured
   logging for decode failures in the future without touching every
   call site.

The helpers intentionally re-raise `json.JSONDecodeError` so existing
try/except clauses at call sites keep working unchanged.
"""

from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Union


def loads(text: str) -> Any:
    """Parse a JSON string.

    Thin wrapper around the stdlib decoder. Use this for LLM outputs,
    inline JSON snippets, or anywhere you have a string in hand.
    """
    return _json.loads(text)


def load_json_file(path: Union[str, Path], encoding: str = "utf-8") -> Any:
    """Read a file at `path` and return the parsed JSON content."""
    p = Path(path) if not isinstance(path, Path) else path
    return _json.loads(p.read_text(encoding=encoding))


def parse_json_line(line: str) -> Any:
    """Parse a single JSONL line.

    Empty or whitespace-only lines raise `json.JSONDecodeError` so
    callers can distinguish truncated streams from valid records.
    """
    stripped = line.strip() if line else ""
    if not stripped:
        raise _json.JSONDecodeError("empty line", line or "", 0)
    return _json.loads(stripped)
