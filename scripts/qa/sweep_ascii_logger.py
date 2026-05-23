#!/usr/bin/env python3
"""phase-38.5.1: sweep the 151 ASCII-logger violations identified by 38.5.

Strategy:
1. Read ascii_logger_check.py --json output to enumerate violations.
2. For each affected file, apply a character-level replacement using a
   curated ASCII map (emojis -> bracketed labels; arrows -> ->; em-dash -> --).
3. Verify each file is still ast.parse() green after the substitution.
4. Re-run ascii_logger_check at the end. Exit 0 only if clean.

Only touches lines containing logger.*() calls — NOT docstrings, NOT
comments outside logger contexts. We use a regex restricted to lines that
contain `logger.` (any method) so we don't accidentally rewrite docstring
quotes or unrelated emoji-in-comment usage.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ASCII equivalents for the top codepoints in the 151-violation inventory.
# Choose labels that preserve semantic meaning of the original log line.
REPLACEMENTS = {
    "✅": "[OK]",        # white heavy check mark
    "❌": "[FAIL]",      # cross mark
    "️": "",            # variation selector-16 (drops to next char)
    "→": "->",          # right arrow
    "⚠": "[WARN]",      # warning sign
    "—": "--",          # em-dash
    "\U0001F504": "[RETRY]", # counterclockwise arrows
    "\U0001F52A": "[KILL]",  # kitchen knife (used in stuck_task_reaper)
    "\U0001F4CB": "[QUEUE]", # clipboard
    "\U0001F50D": "[SCAN]",  # magnifying glass
    "\U0001F517": "[LINK]",  # link emoji
    "\U0001F914": "[?]",     # thinking face
    "\U0001F4DD": "[NOTE]",  # memo
    "\U0001F4CA": "[CHART]", # bar chart
    "\U0001F4C8": "[CHART]", # chart with upwards trend
    "\U0001F4C9": "[CHART]", # chart with downwards trend
    "\U0001F4B0": "[$]",     # money bag
    "\U0001F4B5": "[$]",     # dollar banknote
    "\U0001F4B8": "[$]",     # money with wings
    "\U0001F680": "[GO]",    # rocket
    "\U0001F6A8": "[ALERT]", # police car light
    "\U0001F6D1": "[STOP]",  # stop sign
    "✨": "[NEW]",       # sparkles
    "⚙": "[CFG]",       # gear
    "⛔": "[BLOCK]",     # no entry
    "⏰": "[CLOCK]",     # alarm clock
    "⏱": "[TIMER]",     # stopwatch
    "⏳": "[WAIT]",      # hourglass not done
    "✓": "[OK]",         # check mark (lighter variant)
    "✗": "[FAIL]",       # ballot x
    "✘": "[FAIL]",       # heavy ballot x
    "ℹ": "[i]",          # information source
    "⭐": "[STAR]",       # white medium star
    "⛪": "[FX]",         # not commonly used; placeholder
    # Box-drawing and bullets sometimes used:
    "•": "*",             # bullet
    "–": "-",             # en-dash
    "…": "...",           # ellipsis
    "‘": "'",             # left single quote
    "’": "'",             # right single quote (apostrophe)
    "“": '"',             # left double quote
    "”": '"',             # right double quote
}


def _replace_non_ascii_on_logger_lines(text: str) -> tuple[str, int]:
    """Apply REPLACEMENTS only to lines that contain `logger.`. Returns
    (new_text, num_lines_changed).
    """
    lines = text.splitlines(keepends=True)
    changed = 0
    for i, line in enumerate(lines):
        if "logger." not in line:
            continue
        orig = line
        for src, dst in REPLACEMENTS.items():
            if src in line:
                line = line.replace(src, dst)
        # Catch-all: any remaining non-ASCII in this logger line gets
        # replaced with a `?` placeholder. We DON'T want to silently leave
        # them; they'd fail the next ascii_logger_check anyway.
        if any(ord(c) > 0x7F for c in line):
            line = "".join(c if ord(c) <= 0x7F else "?" for c in line)
        if line != orig:
            lines[i] = line
            changed += 1
    return "".join(lines), changed


def _files_with_violations() -> list[Path]:
    """Run ascii_logger_check.py to get the affected file list."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qa" / "ascii_logger_check.py"),
         "--roots", "backend", "scripts", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return []
    files = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line)
            files.add(REPO_ROOT / v["path"])
        except (json.JSONDecodeError, KeyError):
            continue
    return sorted(files)


def main() -> int:
    files = _files_with_violations()
    print(f"phase-38.5.1: {len(files)} files with violations", file=sys.stderr)
    if not files:
        print("phase-38.5.1: nothing to sweep (already clean)", file=sys.stderr)
        return 0

    total_files_changed = 0
    total_lines_changed = 0
    skipped_syntax_break: list[str] = []

    for f in files:
        try:
            orig_text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            print(f"SKIP {f.relative_to(REPO_ROOT)}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        new_text, n_lines = _replace_non_ascii_on_logger_lines(orig_text)
        if n_lines == 0:
            continue
        # Verify syntax BEFORE writing
        try:
            ast.parse(new_text, filename=str(f))
        except SyntaxError as e:
            skipped_syntax_break.append(f"{f.relative_to(REPO_ROOT)}: SyntaxError line {e.lineno}")
            continue
        f.write_text(new_text, encoding="utf-8")
        total_files_changed += 1
        total_lines_changed += n_lines
        print(f"  swept {f.relative_to(REPO_ROOT)}: {n_lines} line(s)", file=sys.stderr)

    print(
        f"phase-38.5.1: swept {total_files_changed} files, "
        f"{total_lines_changed} lines changed",
        file=sys.stderr,
    )
    if skipped_syntax_break:
        print("phase-38.5.1: SKIPPED (syntax would break):", file=sys.stderr)
        for s in skipped_syntax_break:
            print(f"  {s}", file=sys.stderr)

    # Re-run the check
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qa" / "ascii_logger_check.py"),
         "--roots", "backend", "scripts"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("phase-38.5.1: CLEAN", file=sys.stderr)
        return 0
    print(
        f"phase-38.5.1: still {result.stdout.count(chr(10))} violations remaining "
        f"(non-logger-line OR new codepoints needing a REPLACEMENTS entry)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
