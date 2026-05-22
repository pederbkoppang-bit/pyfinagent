#!/usr/bin/env python3
"""phase-40.6: .env file syntax guard.

Closes closure_roadmap.md section 3 OPEN-31 (audit-basis 23.5.19-F4): no
static check on .env syntax. A malformed line (missing '=', unescaped
quote, embedded newline, etc.) breaks pydantic-settings parsing at startup
-- and the failure mode is often a silent fallback to Field defaults
(phase-34.1e history: DEEP_THINK_MODEL silently regressing to claude-opus-4-7).

Mirrors the dotenv-linter rule taxonomy (https://github.com/dotenv-linter)
so developers searching the canonical labels find familiar diagnostics:
LeadingCharacter, IncorrectDelimiter, KeyWithoutValue, QuoteCharacter,
LowercaseKey, WindowsLineEnding, TrailingWhitespace, DuplicatedKey.

Exit codes:
  0 -- no violations
  1 -- one or more violations
  2 -- usage / IO error

Stdlib-only (no dotenv import; deliberately so this can run pre-commit
before the venv may be active). Python 3.10+. SECURITY: never echoes
values -- only KEYs are reported. SecretStr exfil risk is the point of
.env files being gitignored.

Usage:
  python scripts/qa/env_syntax_check.py PATH [PATH ...]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path


KEY_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
COMMENT_OR_BLANK_RE = re.compile(r"^\s*(#.*)?$")
LOWERCASE_KEY_RE = re.compile(r"^(?:export\s+)?[a-z]")


@dataclass(frozen=True)
class Violation:
    path: str
    line_no: int
    rule: str
    severity: str
    message: str
    key: str  # value intentionally omitted -- SecretStr exfil risk


def check_line(path: str, line_no: int, raw_line: str) -> list[Violation]:
    """Per-line rules. Returns empty list when clean."""
    out: list[Violation] = []
    line = raw_line.rstrip("\n")

    # Windows CRLF detection
    if line.endswith("\r"):
        out.append(Violation(
            path=path, line_no=line_no,
            rule="WindowsLineEnding", severity="warning",
            message="CRLF line ending detected (use LF)", key="",
        ))
        line = line.rstrip("\r")

    # Blank or comment lines are fine
    if COMMENT_OR_BLANK_RE.match(line):
        return out

    # Leading character: line must start with [A-Za-z_] or "export "
    if not re.match(r"^(?:export\s+)?[A-Za-z_]", line):
        out.append(Violation(
            path=path, line_no=line_no,
            rule="LeadingCharacter", severity="error",
            message="Key must start with letter or underscore",
            key="",
        ))
        return out

    # Match KEY=...  (with optional `export ` prefix)
    m = KEY_RE.match(line)
    if not m:
        # No `=` found = KeyWithoutValue (or IncorrectDelimiter if `:` etc.)
        # Detect a Python-dict-style colon
        if re.match(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*:", line):
            out.append(Violation(
                path=path, line_no=line_no,
                rule="IncorrectDelimiter", severity="error",
                message="Use '=' as key-value delimiter, not ':'",
                key="",
            ))
        else:
            out.append(Violation(
                path=path, line_no=line_no,
                rule="KeyWithoutValue", severity="error",
                message="Missing '=' separator",
                key="",
            ))
        return out

    key = m.group(1)

    # LowercaseKey -- convention is UPPER_SNAKE_CASE
    if LOWERCASE_KEY_RE.match(line):
        out.append(Violation(
            path=path, line_no=line_no,
            rule="LowercaseKey", severity="warning",
            message=f"Key {key!r} should be UPPER_SNAKE_CASE",
            key=key,
        ))

    # Quote balancing on the value portion
    value_part = line[m.end():]
    # Track quote balance ignoring escaped quotes
    in_single = False
    in_double = False
    i = 0
    while i < len(value_part):
        ch = value_part[i]
        if ch == "\\" and i + 1 < len(value_part):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        i += 1
    if in_single or in_double:
        out.append(Violation(
            path=path, line_no=line_no,
            rule="QuoteCharacter", severity="error",
            message=f"Unmatched {'single' if in_single else 'double'} quote in value",
            key=key,
        ))

    # TrailingWhitespace (on the original line BEFORE strip)
    if raw_line.rstrip("\r\n") != raw_line.rstrip("\r\n").rstrip():
        out.append(Violation(
            path=path, line_no=line_no,
            rule="TrailingWhitespace", severity="warning",
            message=f"Key {key!r} has trailing whitespace",
            key=key,
        ))

    return out


def check_file(path: Path) -> list[Violation]:
    """Per-file rules: per-line + DuplicatedKey."""
    out: list[Violation] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"ERROR: {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return out

    seen_keys: dict[str, int] = {}
    for line_no, raw_line in enumerate(raw.splitlines(keepends=True), start=1):
        line_violations = check_line(str(path), line_no, raw_line)
        out.extend(line_violations)
        # Track keys for duplicate detection
        line = raw_line.rstrip("\n\r")
        if COMMENT_OR_BLANK_RE.match(line):
            continue
        m = KEY_RE.match(line)
        if m:
            key = m.group(1)
            if key in seen_keys:
                out.append(Violation(
                    path=str(path), line_no=line_no,
                    rule="DuplicatedKey", severity="error",
                    message=f"Key {key!r} already defined at line {seen_keys[key]}",
                    key=key,
                ))
            else:
                seen_keys[key] = line_no
    return out


def format_violation(v: Violation) -> str:
    """Stable format: path:line: severity: rule: message"""
    return f"{v.path}:{v.line_no}: {v.severity}: {v.rule}: {v.message}"


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "Usage: env_syntax_check.py PATH [PATH ...]\n"
            "Exits 0 clean, 1 violations, 2 usage error.",
            file=sys.stderr,
        )
        return 2

    all_violations: list[Violation] = []
    for arg in args:
        p = Path(arg)
        if not p.exists():
            print(f"ERROR: {p}: file not found", file=sys.stderr)
            return 2
        if not p.is_file():
            print(f"ERROR: {p}: not a regular file", file=sys.stderr)
            return 2
        all_violations.extend(check_file(p))

    # Sort by path, line, rule for stable output
    all_violations.sort(key=lambda v: (v.path, v.line_no, v.rule))
    for v in all_violations:
        print(format_violation(v))

    n_errors = sum(1 for v in all_violations if v.severity == "error")
    n_warnings = sum(1 for v in all_violations if v.severity == "warning")
    summary = (
        f"{'OK' if not n_errors else 'FAIL'}: "
        f"{len(args)} file(s), "
        f"{n_errors} error(s), {n_warnings} warning(s)"
    )
    print(summary, file=sys.stderr)

    return 0 if not n_errors else 1


if __name__ == "__main__":
    sys.exit(main())
