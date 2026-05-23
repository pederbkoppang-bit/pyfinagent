#!/usr/bin/env python3
"""phase-38.5.1 v2: targeted line-by-line sweep using JSON output."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

REPLACEMENTS = {
    "✅": "[OK]",
    "❌": "[FAIL]",
    "️": "",
    "→": "->",
    "⚠": "[WARN]",
    "—": "--",
    "\U0001F504": "[RETRY]",
    "\U0001F52A": "[KILL]",
    "\U0001F4CB": "[QUEUE]",
    "\U0001F50D": "[SCAN]",
    "\U0001F517": "[LINK]",
    "\U0001F914": "[?]",
    "\U0001F4DD": "[NOTE]",
    "\U0001F4CA": "[CHART]",
    "\U0001F4C8": "[CHART]",
    "\U0001F4C9": "[CHART]",
    "\U0001F4B0": "[$]",
    "\U0001F4B5": "[$]",
    "\U0001F4B8": "[$]",
    "\U0001F680": "[GO]",
    "\U0001F6A8": "[ALERT]",
    "\U0001F6D1": "[STOP]",
    "✨": "[NEW]",
    "⚙": "[CFG]",
    "⛔": "[BLOCK]",
    "⏰": "[CLOCK]",
    "⏱": "[TIMER]",
    "⏳": "[WAIT]",
    "✓": "[OK]",
    "✗": "[FAIL]",
    "✘": "[FAIL]",
    "ℹ": "[i]",
    "⭐": "[STAR]",
    "•": "*",
    "–": "-",
    "…": "...",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
}


def main() -> int:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qa" / "ascii_logger_check.py"),
         "--roots", "backend", "scripts", "--json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("v2: nothing to sweep", file=sys.stderr)
        return 0

    by_path: dict[str, set[int]] = defaultdict(set)
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            continue
        by_path[v["path"]].add(v["line"])

    files_changed = 0
    lines_changed = 0

    for rel_path, line_nos in by_path.items():
        f = REPO_ROOT / rel_path
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        changed = 0
        for ln in sorted(line_nos):
            # Scan from the reported line forward up to 5 lines
            for idx in range(max(0, ln - 1), min(len(lines), ln + 5)):
                line = lines[idx]
                if not any(ord(c) > 0x7F for c in line):
                    continue
                orig = line
                for src, dst in REPLACEMENTS.items():
                    if src in line:
                        line = line.replace(src, dst)
                if any(ord(c) > 0x7F for c in line):
                    line = "".join(c if ord(c) <= 0x7F else "?" for c in line)
                if line != orig:
                    lines[idx] = line
                    changed += 1
        if changed == 0:
            continue
        new_text = "".join(lines)
        try:
            ast.parse(new_text, filename=str(f))
        except SyntaxError as e:
            print(f"SYNTAX BREAK {rel_path}: line {e.lineno}", file=sys.stderr)
            continue
        f.write_text(new_text, encoding="utf-8")
        files_changed += 1
        lines_changed += changed
        print(f"  swept {rel_path}: {changed} line(s)", file=sys.stderr)

    print(f"v2 sweep: {files_changed} files, {lines_changed} lines", file=sys.stderr)

    final = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "qa" / "ascii_logger_check.py"),
         "--roots", "backend", "scripts"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if final.returncode == 0:
        print("v2: CLEAN", file=sys.stderr)
        return 0
    remaining = len([ln for ln in final.stdout.splitlines() if ln.strip()])
    print(f"v2: still {remaining} violations", file=sys.stderr)
    print(final.stdout[:1500], file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
