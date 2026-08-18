#!/usr/bin/env python3
"""phase-86.110 criterion 4: enumerate every site that can leak the heartbeat.

## Why this is keyed on REACHABILITY, not on the patch table

The obvious survey -- "find tests that patch `_HISTORY_PATH` but not
`_HEARTBEAT_PATH`" -- is wrong in BOTH directions, and the step's own research
gate proved it:

  * it **over-reports**, twice. `test_cycle_heartbeat_alarm.py` patches the
    history constant but its code path never reaches a writer. So does
    `scripts/smoketest_stages_5_through_13.py`, which additionally swaps the
    constant by direct assignment outside pytest entirely -- invisible to any
    monkeypatch-shaped survey. Both are measured here, not assumed: grepping
    each for the writer names returns nothing.
  * it **under-reports** by construction: a site that reaches a writer without
    ever touching `_HISTORY_PATH` is invisible to it. `autonomous_loop.py` is
    exactly that shape -- though in its case writing the real file is correct.

So the population is defined as **"can this call site reach a writer of
`_HEARTBEAT_PATH`?"** and the patch table is used only as a cross-check.

*(The step's research gate framed the smoketest as an UNDER-report. Re-derived
here, it is an over-report for THIS question: it patches the history constant
and reaches no writer, so it cannot leak the heartbeat. The gate's framing was
about a different property -- that a monkeypatch survey misses a direct
assignment -- which is true and is why the cross-check is printed rather than
trusted.)*

## The writer surface is larger than the filing said

The filing names `record_cycle_end`. The heartbeat is also written by
`record_cycle_start` (`cycle_health.py:426`); both go through
`_write_heartbeat` (`:555`), which writes `_HEARTBEAT_PATH` at `:558`. An
enumeration keyed only on `record_cycle_end` would itself under-report -- the
same error one level down.

## What "reaches" means here, stated so the result is auditable

A file is IN the population if it calls any name in `WRITERS` (directly, or via
`get_log()`/`CycleHealthLog()`). This is a static approximation: it can include
a call site guarded by a branch that never executes, and it cannot follow a
call through an alias. It is deliberately over-inclusive -- for a leak sweep,
a false positive costs a read and a false negative costs a polluted repo file.
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

# Every public entry point that transitively writes _HEARTBEAT_PATH.
WRITERS = ("record_cycle_end", "record_cycle_start", "_write_heartbeat")

def _isolates(src: str, const: str) -> bool:
    """Does this source ACTUALLY isolate `const`, as an executable statement?

    **A substring search is not enough, and this function exists because a
    substring search silently failed here.** The first version of this scanner
    used `re.search(r"_HEARTBEAT_PATH", src)`. When the fix was reverted as a
    positive control, the scanner STILL reported zero leaks -- because the
    explanatory comments added by the fix mention `_HEARTBEAT_PATH`, and a
    comment satisfied the check. The guard was matching its own documentation.

    So this walks the AST and counts only two executable shapes:
      * `monkeypatch.setattr(<mod>, "<const>", ...)` -- a string argument, and
      * `<anything>.<const> = ...` -- a direct attribute assignment.
    Comments and docstrings are invisible to `ast.parse`, so they cannot
    satisfy it by construction rather than by a rule someone must remember.
    """
    import ast as _ast

    try:
        tree = _ast.parse(src)
    except SyntaxError:
        return False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, _ast.Attribute) else getattr(fn, "id", None)
            if name == "setattr":
                for a in node.args:
                    if isinstance(a, _ast.Constant) and a.value == const:
                        return True
        elif isinstance(node, _ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, _ast.Attribute) and tgt.attr == const:
                    return True
    return False

SEARCH_ROOTS = ("backend", "scripts", "tests")
SELF = pathlib.Path(__file__).name


def scan() -> tuple[list[dict], list[str]]:
    rows, notes = [], []
    for root in SEARCH_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if p.name == SELF:
                continue
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # pragma: no cover
                notes.append(f"UNREADABLE {p}: {exc!r}")
                continue
            if "cycle_health" not in src and "CycleHealthLog" not in src:
                continue
            hit_writers = [w for w in WRITERS if re.search(rf"\b{w}\s*\(", src)]
            if not hit_writers:
                continue
            # The module that DEFINES the writers is not a caller.
            if p.resolve() == (REPO / "backend/services/cycle_health.py").resolve():
                continue
            rows.append({
                "file": str(p.relative_to(REPO)),
                "writers_called": hit_writers,
                "patches_history": _isolates(src, "_HISTORY_PATH"),
                "patches_heartbeat": _isolates(src, "_HEARTBEAT_PATH"),
                "is_test": p.name.startswith("test_") or "/tests/" in str(p),
            })
    return rows, notes


def main() -> int:
    rows, notes = scan()
    print("POPULATION RULE: a file that calls any of "
          f"{WRITERS} and is not cycle_health.py itself.")
    print(f"SEARCH ROOTS: {SEARCH_ROOTS}\n")
    if not rows:
        print("EMPTY POPULATION -- that is a RED FLAG, not a clean result: this "
              "repo is known to contain at least the two phase-66.1 tests. "
              "Treat an empty scan as a broken scanner.")
        return 2

    # PRODUCTION code is SUPPOSED to write the real heartbeat -- that is the
    # file's whole purpose. Only a site that writes it as a SIDE EFFECT of a
    # test or dev script is a leak. Classifying the production writer as a leak
    # would be the mirror of the naive survey's error: a rule that flags the
    # correct behaviour as a defect.
    leaks, safe, legit = [], [], []
    for r in rows:
        if not r["is_test"] and not r["file"].startswith("scripts/"):
            legit.append(r)
        elif r["patches_heartbeat"]:
            safe.append(r)
        else:
            leaks.append(r)

    marks = {id(r): "LEAK" for r in leaks}
    marks.update({id(r): "prod" for r in legit})
    print(f"{'':4s} {'file':61s} {'writers':38s} hist hb")
    print("-" * 118)
    for r in sorted(rows, key=lambda x: (x["file"])):
        mark = marks.get(id(r), "    ")
        print(f"{mark:4s} {r['file']:61s} {','.join(r['writers_called'])[:37]:38s} "
              f"{'Y' if r['patches_history'] else '-':4s} "
              f"{'Y' if r['patches_heartbeat'] else '-'}")

    print(f"\nPOPULATION: {len(rows)}   ISOLATED TESTS: {len(safe)}   "
          f"LEAKING: {len(leaks)}   LEGITIMATE PRODUCTION WRITERS: {len(legit)}")
    if legit:
        print("  production writers (expected to write the real file): "
              + ", ".join(r["file"] for r in legit))

    if leaks:
        print("\nSITES THAT REACH A WRITER WITHOUT ISOLATING _HEARTBEAT_PATH:")
        for r in leaks:
            kind = "TEST" if r["is_test"] else "SCRIPT (not a pytest run -- "\
                                              "reported, not auto-fixed)"
            print(f"  {r['file']}  [{kind}]  writers={','.join(r['writers_called'])}")

    print("\nCROSS-CHECK against the naive patch-table survey "
          "('patches _HISTORY_PATH but not _HEARTBEAT_PATH'):")
    naive = []
    for root in SEARCH_ROOTS:
        base = REPO / root
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            if p.name == SELF or p.resolve() == (REPO / "backend/services/cycle_health.py").resolve():
                continue
            src = p.read_text(encoding="utf-8", errors="replace")
            if _isolates(src, "_HISTORY_PATH") and not _isolates(src, "_HEARTBEAT_PATH"):
                naive.append(str(p.relative_to(REPO)))
    reach = {r["file"] for r in rows}
    over = [f for f in naive if f not in reach]
    under = [r["file"] for r in leaks if r["file"] not in naive]
    print(f"  naive survey would flag: {len(naive)}")
    print(f"  OVER-reported by naive (patches the constant, reaches no writer): {over or 'none'}")
    print(f"  UNDER-reported by naive (reaches a writer, no _HISTORY_PATH patch): {under or 'none'}")

    for n in notes:
        print("NOTE:", n)
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
