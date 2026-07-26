#!/usr/bin/env python3
"""Audit the Claude auto-memory tree. Exit 1 on any defect.

WHY THIS EXISTS: on 2026-07-26 an audit found MEMORY.md pointing at
`feedback_full_codebase_audit_before_changes.md` with NO FILE BEHIND IT -- an
operator lesson had been unreachable for weeks, and nothing would ever have said
so. Two more files had no pointer at all, making them invisible at session start
(MEMORY.md is what gets loaded into context, not the directory listing).

Memory rot is silent by construction: a missing memory produces no error, it just
lets an agent repeat a mistake the operator already paid for. Run this whenever
memories are touched.

    python scripts/housekeeping/audit_memory.py [--dir PATH]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

DEFAULT = pathlib.Path.home() / (
    ".claude/projects/-Users-ford--openclaw-workspace-pyfinagent/memory"
)


def audit(root: pathlib.Path) -> int:
    idx_path = root / "MEMORY.md"
    if not idx_path.is_dir() and not idx_path.exists():
        print(f"FAIL: no MEMORY.md in {root}")
        return 1
    files = sorted(p for p in root.glob("*.md") if p.name != "MEMORY.md")
    names = {p.name for p in files}
    idx = idx_path.read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)]+\.md)\)", idx))

    problems: list[str] = []

    # A pointer with no file = a lesson the index promises and cannot deliver.
    for bad in sorted(linked - names):
        problems.append(f"DANGLING POINTER: MEMORY.md links {bad} but the file is gone")

    # A file with no pointer is not loaded at session start -> effectively invisible.
    for orphan in sorted(names - linked):
        problems.append(f"NO POINTER: {orphan} exists but MEMORY.md never links it")

    for p in files:
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---") or "type:" not in text.split("---")[1]:
            problems.append(f"MALFORMED FRONTMATTER: {p.name}")
        for link in sorted(set(re.findall(r"\[\[([^\]]+)\]\]", text))):
            if f"{link}.md" not in names:
                problems.append(f"BROKEN WIKILINK: {p.name} -> [[{link}]]")

    print(f"memory files: {len(files)}   pointers: {len(linked)}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for x in problems:
            print(f"  - {x}")
        return 1
    print("OK -- every file has a pointer, every pointer has a file, links resolve")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=DEFAULT)
    sys.exit(audit(ap.parse_args().dir))
