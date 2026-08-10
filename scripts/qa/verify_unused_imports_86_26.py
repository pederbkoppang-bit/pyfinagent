#!/usr/bin/env python3
"""phase-86.26 -- prove each F401 name is safe to delete BEFORE deleting it.

An "unused import" is not always unused. Three ways ruff's F401 can be right
about the file and wrong about the change:

  * **re-export** -- another module does `from this_module import Name`, so the
    import IS the public surface even though this file never reads it;
  * **`__all__`** -- the name is declared as public API;
  * **side effect** -- the import is there for what it REGISTERS, not for a
    name (`import backend.something_that_registers_a_handler`).

So this script does not trust the linter's verdict on its own. For every F401
finding it asks the repository whether anyone imports that name FROM that
module, checks `__all__`, and flags any import whose module is plausibly
side-effecting. Only names that clear all three are reported SAFE.

Re-derivation, not transcription: the findings come from running ruff, and the
consumer search is an AST scan of every `from X import Y` in the repo -- not a
grep for the bare name, which would match the definition site and every
unrelated use.

    source .venv/bin/activate
    python scripts/qa/verify_unused_imports_86_26.py
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCOPE = [
    "backend/services/outcome_tracker.py",
    "backend/agents/memory.py",
    "backend/agents/bias_detector.py",
    "backend/agents/conflict_detector.py",
    "backend/agents/skill_optimizer.py",
    "backend/api/portfolio.py",
    "backend/slack_bot/formatters.py",
    "backend/services/recommendation_vocab.py",
]

#: Modules whose import is plausibly for a SIDE EFFECT rather than a name.
#: Deliberately conservative -- a false positive here costs a manual check,
#: a false negative costs a silently unregistered handler.
_SIDE_EFFECT_HINTS = ("register", "signal", "hook", "plugin", "patch", "codec")


def ruff_findings() -> list[dict]:
    proc = subprocess.run(
        ["uvx", "ruff", "check", "--select", "F401", "--output-format", "json",
         *SCOPE],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        raise SystemExit(f"could not parse ruff output:\n{proc.stdout[:500]}\n"
                         f"{proc.stderr[:500]}")


def module_path_of(rel: str) -> str:
    return rel[:-3].replace("/", ".")


def consumers_of(module: str, name: str) -> list[str]:
    """Every `from <module> import <name>` anywhere in the repo, by AST.

    A bare grep for the name would match its definition and every unrelated
    use; only the import statement establishes a re-export.
    """
    hits = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == module:
                for alias in node.names:
                    if alias.name == name:
                        hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


def declared_in_all(rel: str, name: str) -> bool:
    src = (REPO_ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    try:
                        return name in ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return True          # unparseable -> assume public
    return False


def validate_method() -> int:
    """Prove the consumer-finder CAN find consumers before trusting a clean run.

    Every name below reports "no consumers", and that is only meaningful if the
    search would have found them had they existed. `is_buy_intent` is the
    known-positive: phase-86.22 wired it into seven modules, so a working
    scanner must return several hits. A scanner that returns zero for
    EVERYTHING -- because of a typo in the module path, say -- would otherwise
    declare every import safe to delete.
    """
    probes = [
        ("backend.services.recommendation_vocab", "is_buy_intent", True),
        ("backend.services.recommendation_vocab", "canonical_recommendation", True),
        ("backend.services.recommendation_vocab", "no_such_name_xyzzy", False),
        ("backend.does.not.exist", "is_buy_intent", False),
    ]
    print("METHOD VALIDATION -- can the consumer-finder actually find consumers?")
    bad = 0
    for mod, name, expect_hits in probes:
        hits = consumers_of(mod, name)
        ok = bool(hits) == expect_hits
        bad += (not ok)
        print(f"  {'OK  ' if ok else 'FAIL'}  {mod}.{name:<26} "
              f"hits={len(hits):<3} expected={'some' if expect_hits else 'none'}")
    if bad:
        print(f"\n  METHOD REJECTED -- {bad} probe(s) wrong; a clean report would "
              f"be meaningless.")
        return 1
    print("  Method validated in both directions.\n")
    return 0


def main() -> int:
    if validate_method():
        return 1
    findings = ruff_findings()
    print(f"phase-86.26 -- {len(findings)} F401 finding(s), RE-DERIVED by ruff")
    print(f"scope: {len(SCOPE)} files\n")
    if not findings:
        print("nothing to verify.")
        return 0

    hdr = f"{'file':<44}{'line':>5}  {'name':<34}{'re-exported?':<26}verdict"
    print(hdr); print("-" * len(hdr))
    unsafe = 0
    for f in findings:
        rel = str(Path(f["filename"]).relative_to(REPO_ROOT))
        msg = f["message"]
        # "`X` imported but unused" -> X  (may be dotted: a.b.C)
        name = msg.split("`")[1]
        leaf = name.rsplit(".", 1)[-1]
        mod = module_path_of(rel)
        cons = [c for c in consumers_of(mod, leaf) if not c.startswith(rel)]
        in_all = declared_in_all(rel, leaf)
        side = any(h in name.lower() for h in _SIDE_EFFECT_HINTS)

        if cons:
            verdict, note = "KEEP", f"{len(cons)} consumer(s)"
            unsafe += 1
        elif in_all:
            verdict, note = "KEEP", "in __all__"
            unsafe += 1
        elif side:
            verdict, note = "CHECK BY HAND", "possible side-effect import"
            unsafe += 1
        else:
            verdict, note = "safe to remove", "none"
        print(f"{rel:<44}{f['location']['row']:>5}  {leaf:<34}{note:<26}{verdict}")
        for c in cons:
            print(f"{'':85}<- {c}")

    print(f"\n{len(findings) - unsafe} safe / {unsafe} needing a decision")
    if unsafe:
        print("A name with a consumer is a RE-EXPORT: removing the import breaks "
              "that caller even though this file never reads it.")
    print("\nNOTE the method's limit, stated rather than left implicit: it finds "
          "STATIC `from X import Y`. A dynamic `getattr(module, 'name')` or an "
          "importlib lookup would not appear here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
