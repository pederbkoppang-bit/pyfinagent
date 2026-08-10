#!/usr/bin/env python3
"""phase-86.6 criterion 1 -- DERIVE the population of in-process kill-switch writers.

WHY A STATIC METHOD, AND NOT A RUNTIME DETECTOR
------------------------------------------------
Criterion 1 names one test as the recall probe:
`backend/tests/test_book_safety_69.py::test_peak_reset_dark_by_default`. A
method that reports it clean is rejected regardless of what else it finds.

That test is the probe precisely because a RUNTIME detector cannot see it. It
calls `reset_peak(...)`, which returns early at `kill_switch.py` BEFORE taking
the lock or appending, because `settings.kill_switch_peak_reset_enabled` is
False. So at runtime it writes nothing, and any "did the live journal change?"
detector reports it CLEAN -- while the call site sits there waiting for the
already-APPROVED KS-PEAK-RESET token to arm it.

So the population is derived from the CALL, not from the WRITE. This script
AST-scans the test trees for invocations of the kill-switch methods that can
reach `_append_audit`, and reports, per test function:

  * which mutating methods it invokes,
  * whether it redirects `_AUDIT_PATH` (the established isolation idiom),
  * whether it would be INVISIBLE to a runtime write-detector.

RECALL IS VALIDATED, NOT ASSUMED. The script exits non-zero if the named probe
is not flagged. That check runs BEFORE the report is usable, so a method that
regresses into missing it cannot quietly ship.

    source .venv/bin/activate
    python scripts/qa/derive_live_state_writers_86_6.py
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOTS = [REPO_ROOT / "backend" / "tests", REPO_ROOT / "tests"]
KILL_SWITCH = REPO_ROOT / "backend" / "services" / "kill_switch.py"

#: The recall probe named by criterion 1. Non-negotiable.
RECALL_PROBE = ("test_book_safety_69.py", "test_peak_reset_dark_by_default")


def mutating_api() -> set[str]:
    """Derive the mutating surface FROM kill_switch.py, never a hand-list.

    A hand-written list is the same defect class this step exists to close: it
    goes stale the moment a new auditing method is added. Anything whose body
    reaches `_append_audit` counts.
    """
    tree = ast.parse(KILL_SWITCH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    called = getattr(fn, "attr", None) or getattr(fn, "id", None)
                    if called == "_append_audit":
                        names.add(node.name)
                        break
    names.discard("_append_audit")
    return names


def _calls_in(node: ast.AST) -> list[tuple[str, int]]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if name:
                out.append((name, getattr(sub, "lineno", 0)))
    return out


def _redirects_audit_path(node: ast.AST) -> bool:
    """True if this scope monkeypatches a module-level *_AUDIT_PATH."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if getattr(fn, "attr", None) == "setattr":
                for arg in sub.args:
                    if isinstance(arg, ast.Constant) and arg.value == "_AUDIT_PATH":
                        return True
    return False


def scan(mutators: set[str]) -> list[dict]:
    rows: list[dict] = []
    for root in TEST_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("test_*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as exc:  # report, never skip silently
                rows.append({"file": path.name, "test": "(unparseable)",
                             "calls": [f"SyntaxError: {exc}"], "redirect": False})
                continue

            # Module-scope redirects (autouse fixtures) protect every test here.
            module_redirect = any(
                _redirects_audit_path(n) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name.startswith("_")
            )
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.name.startswith("test_"):
                    continue
                hits = sorted({n for n, _ in _calls_in(node) if n in mutators})
                if not hits:
                    continue
                rows.append({
                    "file": path.name,
                    "test": node.name,
                    "calls": hits,
                    "redirect": _redirects_audit_path(node) or module_redirect,
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    mutators = mutating_api()
    print("phase-86.6 criterion 1 -- in-process kill-switch writer population")
    print(f"mutating surface DERIVED from {KILL_SWITCH.relative_to(REPO_ROOT)} "
          f"({len(mutators)}): {', '.join(sorted(mutators))}\n")

    rows = scan(mutators)
    if not args.quiet:
        hdr = f"{'file':<52}{'test':<58}{'redirects':<11}calls"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['file']:<52}{r['test'][:56]:<58}"
                  f"{('yes' if r['redirect'] else 'NO'):<11}{', '.join(r['calls'])}")

    print(f"\npopulation: {len(rows)} test(s) invoking a mutating kill-switch API")
    unprotected = [r for r in rows if not r["redirect"]]
    print(f"WITHOUT an _AUDIT_PATH redirect: {len(unprotected)}")
    for r in unprotected:
        print(f"  !! {r['file']}::{r['test']}  ({', '.join(r['calls'])})")

    # ── RECALL VALIDATION (criterion 1) ────────────────────────────────────
    # This gate runs on the METHOD, not on the codebase. If the probe is not
    # flagged, the method is rejected -- no matter how good the rest looks.
    probe_file, probe_test = RECALL_PROBE
    flagged = [r for r in rows if r["file"] == probe_file and r["test"] == probe_test]
    print("\nRECALL VALIDATION (criterion 1)")
    print(f"  probe: {probe_file}::{probe_test}")
    if flagged:
        r = flagged[0]
        print(f"  FLAGGED -- calls {', '.join(r['calls'])}; redirects={r['redirect']}")
        print("  A runtime write-detector would report this test CLEAN: reset_peak")
        print("  returns early while kill_switch_peak_reset_enabled is False, so no")
        print("  bytes reach the journal. Deriving from the CALL is what catches it.")
    else:
        print("  NOT FLAGGED -- the derivation method is REJECTED (criterion 1).")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
