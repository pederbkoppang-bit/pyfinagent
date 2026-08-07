#!/usr/bin/env python3
"""Audit the Claude auto-memory tree. Exit 1 on unresolvable links or index rot.

WHY THIS EXISTS: on 2026-07-26 an audit found MEMORY.md pointing at
`feedback_full_codebase_audit_before_changes.md` with NO FILE BEHIND IT -- an
operator lesson had been unreachable for weeks, and nothing would ever have said
so. Two more files had no pointer at all, making them invisible at session start
(MEMORY.md is what gets loaded into context, not the directory listing).

phase-84.1: [[wikilink]] resolution reconciled with how the memory-writing
agents actually link. The agents follow the auto-memory guidance and link by
the target's frontmatter `name:` slug (hyphens), while files are named with
underscores -- the old exact-filename check therefore reported false breakage
on the majority of real links (measured 2026-08-07: of 406 links across the
three corpora, 127 resolved once normalized and only 12 were genuinely
unresolvable). Resolution now walks an ordered ladder so the reported class is
deterministic:

    1. exact:       f"{link}.md" is a file in this corpus        (silent)
    2. normalized:  norm(link) matches a same-corpus file's stem
                    or frontmatter name                          (report, PASS)
    3. cross-corpus: norm(link) matches in a sibling corpus      (report, PASS)
    4. unresolvable                                              (report, FAIL)

Normalization folds case (casefold -- real names contain unicode), treats '-'
and '_' as equivalent (folded BEFORE prefix-stripping so "feedback-foo" and
"feedback_foo" both strip), and strips AT MOST ONE leading type prefix.
CONSEQUENCE, stated deliberately: `feedback_x` and `project_x` normalize to
the SAME key; a link matching more than one file is reported as AMBIGUOUS
naming every candidate and does NOT fail the run -- the target demonstrably
exists, and treating ties as broken would re-create the false-breakage defect
this change removes (established tools resolve ambiguity by preference, never
by erroring). Measured at ship time: zero colliding keys in all three corpora.

Frontmatter `name:` is extracted with a LINE-ANCHORED REGEX, never
yaml.safe_load -- two real main-corpus files carry unquoted `description:`
values with ": " that make YAML raise, and the auditor crashing on the live
corpus is worse than any mis-file.

Exit semantics (criterion 3 + the 2026-07-26 lesson): exit 1 on any
UNRESOLVABLE LINK or NO POINTER -- and, unchanged from the tool's founding
purpose, on DANGLING POINTER or MALFORMED FRONTMATTER. NORMALIZED,
CROSS-CORPUS and AMBIGUOUS findings are reported without failing.

The auditor performs NO WRITES of any kind.

    python scripts/housekeeping/audit_memory.py [--dir PATH] [--sibling PATH ...]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

DEFAULT = pathlib.Path.home() / (
    ".claude/projects/-Users-ford--openclaw-workspace-pyfinagent/memory"
)
_REPO = pathlib.Path(__file__).resolve().parents[2]
_KNOWN_CORPORA = (
    DEFAULT,
    _REPO / ".claude/agent-memory/qa",
    _REPO / ".claude/agent-memory/researcher",
)

_TYPE_PREFIXES = ("feedback_", "project_", "reference_", "user_")

# Bounded link pattern: no ], [, newline, |, or # inside; length-capped so a
# greedy multi-line match can never swallow a paragraph. Obsidian's own
# invalid-link characters informed the class.
_LINK_RE = re.compile(r"\[\[([^\[\]\n|#]{1,120})\]\]")
_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_NAME_RE = re.compile(r"(?m)^name:\s*(\S.*)$")


def _norm(s: str) -> str:
    s = s.strip().casefold().replace("-", "_")
    for p in _TYPE_PREFIXES:
        if s.startswith(p):
            return s[len(p):]  # at most one prefix stripped
    return s


def _frontmatter_name(text: str) -> str | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---")
    if len(parts) < 3:
        return None
    m = _NAME_RE.search(parts[1])
    return m.group(1).strip() if m else None


def _index_corpus(root: pathlib.Path) -> dict[str, set[str]]:
    """norm-key -> {filename} over stems AND frontmatter names."""
    idx: dict[str, set[str]] = {}
    if not root.is_dir():
        return idx
    for p in sorted(root.glob("*.md")):
        if p.name == "MEMORY.md":
            continue
        keys = {_norm(p.stem)}
        try:
            fm = _frontmatter_name(p.read_text(encoding="utf-8"))
        except Exception:
            fm = None
        if fm:
            keys.add(_norm(fm))
        for k in keys:
            idx.setdefault(k, set()).add(p.name)
    return idx


def audit(root: pathlib.Path, siblings: list[pathlib.Path]) -> int:
    idx_path = root / "MEMORY.md"
    if not idx_path.exists() or idx_path.is_dir():
        print(f"FAIL: no MEMORY.md in {root}")
        return 1
    files = sorted(p for p in root.glob("*.md") if p.name != "MEMORY.md")
    names = {p.name for p in files}
    idx = idx_path.read_text(encoding="utf-8")
    linked = set(re.findall(r"\]\(([^)]+\.md)\)", idx))

    own_index = _index_corpus(root)
    sibling_indexes = [(s, _index_corpus(s)) for s in siblings if s != root]

    problems: list[str] = []   # these FAIL the run
    findings: list[str] = []   # these are reported and PASS
    census = {"exact": 0, "normalized": 0, "cross": 0, "unresolvable": 0}

    for bad in sorted(linked - names):
        problems.append(f"DANGLING POINTER: MEMORY.md links {bad} but the file is gone")
    for orphan in sorted(names - linked):
        problems.append(f"NO POINTER: {orphan} exists but MEMORY.md never links it")

    for p in files:
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---") or "type:" not in text.split("---")[1]:
            problems.append(f"MALFORMED FRONTMATTER: {p.name}")
        # strip code before matching -- a ```fence``` or `span` is not a link
        scannable = _INLINE_CODE_RE.sub("", _FENCED_RE.sub("", text))
        for link in sorted(set(_LINK_RE.findall(scannable))):
            if f"{link}.md" in names:
                census["exact"] += 1
                continue
            key = _norm(link)
            own = sorted(own_index.get(key, set()) - {p.name})
            if own:
                census["normalized"] += 1
                if len(own) > 1:
                    findings.append(
                        f"AMBIGUOUS LINK: {p.name} -> [[{link}]] matches "
                        f"{{{', '.join(own)}}} (same directory)")
                else:
                    findings.append(
                        f"NORMALIZED LINK: {p.name} -> [[{link}]] resolves to "
                        f"{own[0]} (same directory)")
                continue
            hit = None
            for s_root, s_idx in sibling_indexes:
                cands = sorted(s_idx.get(key, set()))
                if cands:
                    hit = (s_root, cands)
                    break
            if hit:
                census["cross"] += 1
                s_root, cands = hit
                if len(cands) > 1:
                    findings.append(
                        f"AMBIGUOUS LINK: {p.name} -> [[{link}]] matches "
                        f"{{{', '.join(cands)}}} in {s_root}")
                else:
                    findings.append(
                        f"CROSS-CORPUS LINK: {p.name} -> [[{link}]] resolves to "
                        f"{cands[0]} in {s_root}")
                continue
            census["unresolvable"] += 1
            problems.append(f"UNRESOLVABLE LINK: {p.name} -> [[{link}]]")

    print(f"memory files: {len(files)}   pointers: {len(linked)}")
    for x in findings:
        print(f"  ~ {x}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for x in problems:
            print(f"  - {x}")
    print(f"links: {census['exact']} exact, {census['normalized']} normalized, "
          f"{census['cross']} cross-corpus, {census['unresolvable']} unresolvable")
    if problems:
        return 1
    print("OK -- every file has a pointer, every pointer has a file, links resolve")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=DEFAULT)
    ap.add_argument("--sibling", type=pathlib.Path, action="append", default=None,
                    help="sibling corpus for cross-corpus resolution "
                         "(repeatable; default: the known corpora minus --dir)")
    args = ap.parse_args()
    sibs = args.sibling if args.sibling is not None else [
        c for c in _KNOWN_CORPORA if c != args.dir
    ]
    sys.exit(audit(args.dir, sibs))
