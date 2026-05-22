#!/usr/bin/env python3
"""phase-38.5: ASCII-only logger audit (CI guard).

Closes closure_roadmap.md section 3 OPEN-14 + .claude/rules/security.md
"ASCII-only logger messages": Windows cp1252 / non-UTF-8 stderr crashes
on non-ASCII chars in logger.*() string literals. Today the rule is
enforced by convention only; this script makes it a runnable check.

AST-walks backend/ + scripts/ for `<logger-name>.<method>()` calls where
the method is in LOGGER_METHODS and inspects every string-literal argument
(plus the literal parts of f-strings) for non-ASCII characters.

Exit codes:
  0 -- no violations found
  1 -- one or more violations
  2 -- internal error (SyntaxError, IOError, etc. -- emit and exit)

Usage:
  python scripts/qa/ascii_logger_check.py [--roots PATH ...] [--names NAME ...] [--json] [--quiet]

Defaults:
  roots: backend scripts
  logger names: logger
  output: text (one violation per line: `path:line:col -- U+XXXX in "..."`)

Stdlib-only (no project deps). Python 3.10+.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


LOGGER_METHODS = frozenset({
    "debug", "info", "warning", "warn", "error",
    "critical", "exception", "log",
})

SKIP_DIRS = frozenset({
    ".venv", ".git", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", "build", "dist",
})


@dataclass(frozen=True)
class Violation:
    path: str
    lineno: int
    col_offset: int
    method: str
    codepoint: int
    excerpt: str


def _is_logger_call(node: ast.Call, allowed_names: frozenset[str]) -> str | None:
    """Return the method name if node is `<allowed>.<METHOD>()`, else None."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in LOGGER_METHODS:
        return None
    value = func.value
    if not isinstance(value, ast.Name):
        return None
    if value.id not in allowed_names:
        return None
    return func.attr


def _iter_string_literal_parts(arg: ast.AST) -> Iterator[tuple[str, int, int]]:
    """Yield (literal_text, lineno, col_offset) for each string-literal
    inside arg. Handles ast.Constant(str) + ast.JoinedStr (yields only the
    Constant parts -- skips FormattedValue interpolations because their
    text comes from runtime values, not source).
    """
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        yield arg.value, arg.lineno, arg.col_offset
        return
    if isinstance(arg, ast.JoinedStr):
        for v in arg.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                yield v.value, getattr(v, "lineno", arg.lineno), getattr(v, "col_offset", arg.col_offset)


def _find_non_ascii(text: str) -> list[tuple[int, int]]:
    """Return [(index_in_text, codepoint), ...] for each non-ASCII char."""
    out = []
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp > 0x7F:
            out.append((i, cp))
    return out


def check_file(
    path: Path,
    allowed_names: frozenset[str] = frozenset({"logger"}),
) -> tuple[list[Violation], int, int]:
    """Return (violations, file_count=1, logger_call_count).

    On SyntaxError or IOError, emit a stderr warning and return ([], 0, 0)."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"WARN: skipped {path} ({type(e).__name__}: {e})", file=sys.stderr)
        return [], 0, 0
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        print(f"WARN: skipped {path} (SyntaxError at line {e.lineno})", file=sys.stderr)
        return [], 0, 0

    violations: list[Violation] = []
    logger_calls = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        method = _is_logger_call(node, allowed_names)
        if method is None:
            continue
        logger_calls += 1
        for arg in node.args:
            for literal_text, lit_line, lit_col in _iter_string_literal_parts(arg):
                hits = _find_non_ascii(literal_text)
                if not hits:
                    continue
                excerpt = literal_text.replace("\n", "\\n")
                if len(excerpt) > 80:
                    excerpt = excerpt[:77] + "..."
                for _idx, cp in hits:
                    violations.append(Violation(
                        path=str(path),
                        lineno=lit_line,
                        col_offset=lit_col,
                        method=method,
                        codepoint=cp,
                        excerpt=excerpt,
                    ))
    return violations, 1, logger_calls


def scan_roots(
    roots: list[Path],
    allowed_names: frozenset[str] = frozenset({"logger"}),
) -> tuple[list[Violation], int, int]:
    """rglob *.py under each root, skipping SKIP_DIRS. Returns (sorted
    violations, scanned_files, total_logger_calls)."""
    violations: list[Violation] = []
    file_count = 0
    call_count = 0
    for root in roots:
        if not root.exists():
            print(f"WARN: root {root} does not exist; skipping", file=sys.stderr)
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            v, n_files, n_calls = check_file(path, allowed_names)
            violations.extend(v)
            file_count += n_files
            call_count += n_calls
    violations.sort(key=lambda x: (x.path, x.lineno, x.col_offset))
    return violations, file_count, call_count


def format_violation(v: Violation, fmt: str = "text") -> str:
    if fmt == "json":
        return json.dumps({
            "path": v.path,
            "line": v.lineno,
            "col": v.col_offset,
            "method": v.method,
            "codepoint": f"U+{v.codepoint:04X}",
            "char": chr(v.codepoint),
            "excerpt": v.excerpt,
        })
    return (
        f"{v.path}:{v.lineno}:{v.col_offset}: logger.{v.method}() "
        f"contains U+{v.codepoint:04X} ({chr(v.codepoint)!r}) -- "
        f'"{v.excerpt}"'
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--roots", nargs="+", default=["backend", "scripts"],
                    help="Directories to scan (default: backend scripts)")
    ap.add_argument("--names", nargs="+", default=["logger"],
                    help="Logger variable names (default: logger). E.g. logger logger_v2")
    ap.add_argument("--json", action="store_true", help="Emit JSON-line output")
    ap.add_argument("--quiet", action="store_true", help="Only print summary + violations")
    args = ap.parse_args(argv)

    roots = [Path(r) for r in args.roots]
    allowed_names = frozenset(args.names)

    try:
        violations, file_count, call_count = scan_roots(roots, allowed_names)
    except Exception as exc:
        print(f"ERROR: scan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    fmt = "json" if args.json else "text"
    for v in violations:
        print(format_violation(v, fmt))

    summary = (
        f"{'OK' if not violations else 'FAIL'}: "
        f"{file_count} files, {call_count} logger calls, "
        f"{len(violations)} violations"
    )
    if args.quiet and not violations:
        pass
    else:
        print(summary, file=sys.stderr)

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
