"""Validator for `.env` file syntax (phase-23.6.0).

Catches the specific bug class that has crashed `com.pyfinagent.autoresearch`
nightly since 2026-04-24:

    KEY= value      <- bash `set -a; . file` reads `KEY=""` then tries to
                       execute `value` as a command, exiting 127 (or 1 if
                       the python entrypoint also chokes on the empty key).

`python-dotenv`'s `dotenv_values()` SILENTLY STRIPS the leading space and
returns `KEY=value`, so it can't catch this bug. Pure regex over the file
contents is the only reliable check. Researcher confirmed empirically in
phase-23.6.0 brief.

Usage::

    python3 scripts/validators/check_dotenv_syntax.py backend/.env
    python3 scripts/validators/check_dotenv_syntax.py --strict backend/.env

Exit semantics:
    0 — clean (or `--strict` not set and only WARNING/INFO findings)
    1 — at least one CRITICAL finding (will-crash-bash)
    2 — file unreadable / argument error
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

# Severity:
#   CRITICAL  — will crash `set -a; . file; set +a`
#   WARNING   — silently broken value (e.g. trailing whitespace, inline comment)
#   INFO      — hygiene
#
# Each rule is a (severity, label, regex, description) tuple. The regex matches
# a SINGLE LINE; we apply it line-by-line and keep the 1-based line number.
RULES = (
    (
        "CRITICAL",
        "leading_space_after_eq",
        re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\s+\S"),
        "Leading space after '=' (KEY= value) — bash sources this as KEY=\"\" + run `value` as command",
    ),
    (
        "CRITICAL",
        "leading_space_before_key",
        re.compile(r"^\s+[A-Za-z_]"),
        "Leading whitespace before key (' KEY=value') — bash skips this line silently",
    ),
    (
        "WARNING",
        "trailing_whitespace_unquoted",
        # Match unquoted value (no leading quote) ending in whitespace.
        re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\"'\n]*\S\s+$"),
        "Trailing whitespace in unquoted value — bash includes the whitespace in the value",
    ),
    (
        "WARNING",
        "inline_comment_unquoted",
        # Match unquoted value followed by space + # (inline comment).
        # Skip lines whose value starts with a quote.
        re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\"'\n]+\s+#"),
        "Inline comment after unquoted value — bash includes the '#' in the value",
    ),
    # The "missing trailing newline" rule is checked separately at file scope.
)


Finding = tuple[str, str, int, str, str]
# (severity, label, lineno_1based, raw_line_stripped_of_newline, description)


def scan_text(text: str) -> list[Finding]:
    """Run every rule against the lines of `text`. Returns ordered findings.

    The function is pure: no I/O, deterministic for a given input.
    """
    findings: list[Finding] = []
    lines = text.split("\n")
    # If the file ends with `\n`, `split("\n")` produces a trailing "" which we
    # strip so the line numbers reflect the user's view.
    if lines and lines[-1] == "":
        had_trailing_newline = True
        lines = lines[:-1]
    else:
        had_trailing_newline = False

    for idx, line in enumerate(lines, start=1):
        # Skip blank lines + pure comments.
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        for severity, label, regex, desc in RULES:
            if regex.match(line):
                findings.append((severity, label, idx, line.rstrip("\r"), desc))

    # File-scope: missing trailing newline.
    if text and not had_trailing_newline:
        findings.append(
            ("INFO", "missing_trailing_newline", len(lines) or 1, "<EOF>",
             "File does not end with a newline — last variable may be silently dropped by some shells"),
        )

    return findings


def format_findings(findings: Iterable[Finding], path: str) -> str:
    out: list[str] = []
    for severity, label, lineno, raw, desc in findings:
        out.append(f"{severity:8s} {path}:{lineno}: [{label}] {desc}")
        out.append(f"         | {raw}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate .env file syntax (phase-23.6.0)",
    )
    parser.add_argument(
        "envfile",
        type=Path,
        help="Path to .env file to validate (e.g. backend/.env)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on WARNING findings too (default: only CRITICAL)",
    )
    args = parser.parse_args(argv)

    try:
        text = args.envfile.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError) as exc:
        print(f"ERROR: cannot read {args.envfile}: {exc}", file=sys.stderr)
        return 2

    findings = scan_text(text)
    if not findings:
        print(f"OK {args.envfile} (clean)")
        return 0

    print(format_findings(findings, str(args.envfile)))
    print()

    n_crit = sum(1 for f in findings if f[0] == "CRITICAL")
    n_warn = sum(1 for f in findings if f[0] == "WARNING")
    n_info = sum(1 for f in findings if f[0] == "INFO")
    print(f"summary: {n_crit} CRITICAL, {n_warn} WARNING, {n_info} INFO")

    if n_crit > 0:
        return 1
    if args.strict and n_warn > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
