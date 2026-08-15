#!/usr/bin/env python3
"""phase-86.85 C8 -- DERIVE guard-shaped mutation coverage from the writer's AST.

SCOPE, STATED FIRST BECAUSE AN EARLIER VERSION OF THIS DOCSTRING OVERCLAIMED.
This module was titled "make completeness a DERIVATION, not a claim". Measured
known-member recall against the three prior FAILs that motivated it is **1 of
4**: dropping the cell for the ordering guard, the step_id-in-key guard, or the
cycle-fallback guard all leave this gate GREEN; only the fail-loud-I/O guard is
demanded. So this checker closes the GUARD-SHAPED half of the failure class --
a `raise LedgerError` or a refusing `if` with no cell, which is how it found
`main`'s uncovered CLI validation -- and does NOT close the BEHAVIOURAL half.
Extending the enumeration to behavioural guards is step 86.89.

    python scripts/qa/verify_matrix_coverage_86_85.py

WHY THIS EXISTS. Three consecutive FAILs on step 86.85 were ONE class: a new
guard shipped with no mutation cell (ordering; then fail-loud-I/O + step_id-in-
key; then cycle-fallback). Each time the guard list was written BY HAND, and
each time the Q/A found the one that was missed. A hand-list cannot be audited
by the person who wrote it -- its omissions are invisible from the inside.

WHAT THIS DOES INSTEAD. It enumerates the writer's guards FROM SOURCE by AST,
then measures -- per cell -- which guards that cell actually touches. **No cell
declares what it covers.** A declaration would be a hand-list wearing a
different hat. Coverage is the union of what the mutations measurably change;
any enumerated guard that no cell touches makes this checker exit non-zero, and
the matrix that calls it goes RED.

WHAT A MATRIX LICENSES, restated because it is the rule this step keeps
relearning: a matrix licenses exactly "these N mutations were killed" -- never a
global claim of completeness. This checker does not upgrade that licence. It
adds a SEPARATE, narrower guarantee: *every guard the enumeration can see has at
least one cell aiming at it.* The enumeration's own blind spots remain, which is
why the enumeration rule is written down below and why the checker proves it can
FAIL before any result of it is believed.

THE ENUMERATION RULE, stated so it can be argued with rather than trusted:

  GUARD-RAISE  -- every `raise LedgerError(...)`. `LedgerError` is the module's
                  own refusal type, so this is a rule about semantics, not a
                  hand-picked list. `raise SystemExit(main())` is deliberately
                  NOT a guard: it is the CLI dispatch, and the rule (exception
                  type is LedgerError) excludes it without anyone naming it.
  GUARD-BRANCH -- every `ast.If` that refuses DIRECTLY: its own body (or else)
                  reaches a GUARD-RAISE or a refusal `return` **without passing
                  through a nested `ast.If`**. Directness is load-bearing, not
                  stylistic: under transitive containment every enclosing
                  dispatch branch is promoted to a guard -- `if
                  args.emit_sequence:` would qualify merely because a nested
                  `if not args.step:` refuses inside it -- and the checker would
                  then demand a mutation cell for a mode selector. The guard is
                  the branch that refuses; its ancestors merely route.

A guard is COVERED by a cell when either is measurably true:

  (a) STRUCTURAL -- the guard's fingerprint is present in the original source
      and absent (or altered) in that cell's mutated source. This catches
      deletion and predicate-weakening.
  (b) TEXTUAL -- the cell's `find` anchor overlaps the guard's own source span
      **or the span of an enclosing `ast.If`** (`If` ONLY -- see the measured
      note in `spans_with_ancestors`; including `ast.Try` over-credited coverage
      and hid a real gap). Both halves are needed
      and each was added because the other produced a wrong answer. Own-span
      alone is required because some mutations INSERT code that makes a guard
      unreachable while leaving its AST node intact (an early `return` before a
      `raise`), where (a) scores the cell as covering nothing. ANCESTOR spans
      are required because a cell that neuters `if key in existing_keys(path):`
      kills the `raise` nested inside it while touching neither that raise's
      text nor its AST node -- scoring by own-span alone reported that guard as
      UNCOVERED, a FALSE gap, on this checker's first run.

SELF-CONTROL (the part that makes a clean result mean anything). Before
reporting, the checker injects a synthetic guard into a COPY of the writer and
requires itself to report that guard as UNCOVERED. If the checker cannot detect
a guard it planted itself, it exits non-zero and reports FAILED GATE rather than
a clean bill -- a scan that cannot find its own known members is a failed gate,
not a pass.

ZERO REPO WRITES. Every mutated source exists only as an in-memory string. The
writer's sha256 is printed before and after.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "verdict_ledger_write.py"

sys.path.insert(0, str(HERE))
from mutation_matrix_86_85 import CELLS  # noqa: E402  (cells are the subject)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    out: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            out[child] = parent
    return out


def _enclosing_fn(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parents.get(cur)
    return "<module>"


def _is_guard_raise(node: ast.AST) -> bool:
    """A refusal, by the module's own vocabulary -- not by a curated list."""
    if not isinstance(node, ast.Raise) or node.exc is None:
        return False
    exc = node.exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "LedgerError"


def _direct_refusal(if_node: ast.If) -> bool:
    """Does THIS branch refuse, without delegating to a nested branch?

    Deliberately does NOT descend into a nested `ast.If`. Transitive containment
    would promote every enclosing dispatch branch into a 'guard' -- e.g.
    `if args.emit_sequence:` would count merely because a nested
    `if not args.step:` refuses inside it. That over-enumerates and would demand
    a mutation cell for a mode selector. The guard is the branch that refuses;
    its ancestors merely route.
    """
    def scan(stmts) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                continue  # a nested branch owns its own refusal
            for sub in ast.walk(stmt):
                if _is_guard_raise(sub):
                    return True
                if isinstance(sub, ast.Return) and sub.value is not None:
                    txt = ast.unparse(sub.value)
                    # A refusal returns a FAILURE code. `return EXIT_OK` is a
                    # success path: on this checker's first run the plain
                    # `"EXIT_" in txt` test promoted `if args.emit_sequence:`
                    # -- a mode selector whose body ends `return EXIT_OK` --
                    # into a guard, and demanded a mutation cell for it.
                    if "EXIT_" in txt and "EXIT_OK" not in txt:
                        return True
        return False

    return scan(if_node.body) or scan(if_node.orelse)


def enumerate_guards(src: str) -> dict[str, list[tuple[int, int]]]:
    """Fingerprint -> [(start_offset, end_offset), ...] spans in `src`.

    The fingerprint is line-number-free on purpose: mutations shift lines, and a
    line-keyed identity would report every guard below an insertion as 'changed'.
    """
    tree = ast.parse(src)
    parents = _parents(tree)
    lines = src.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))

    def span(node: ast.AST) -> tuple[int, int]:
        s = starts[node.lineno - 1] + node.col_offset
        end_lineno = getattr(node, "end_lineno", node.lineno)
        end_col = getattr(node, "end_col_offset", 0)
        e = starts[end_lineno - 1] + end_col
        return (s, e)

    def spans_with_ancestors(node: ast.AST) -> list[tuple[int, int]]:
        """The guard's own span PLUS every enclosing branch/try span.

        WHY ANCESTORS. A cell that neuters `if key in existing_keys(path):`
        kills the `raise` nested inside it, but that raise's own AST node is
        untouched and its own span does not overlap the anchor. Scoring by the
        guard's own span alone reported such a guard as UNCOVERED -- a FALSE
        gap. Containment is the honest rule: mutating the branch that reaches a
        refusal is mutating that refusal.
        """
        out = [span(node)]
        cur: ast.AST | None = parents.get(node)
        while cur is not None and not isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            # `ast.If` ONLY -- deliberately NOT Try/With/For/While.
            #
            # WHY, measured. The first version included `ast.Try`, and `main`
            # wraps its whole body in one `try:`. Every anchor inside that body
            # therefore "overlapped" every guard in it, and the checker credited
            # cell M13 with covering a raise in a completely different branch.
            # The tell: removing cell M14 left the gate GREEN. Over-crediting is
            # the DANGEROUS direction for a coverage checker -- it hides real
            # gaps -- and it was invisible until the gate was tested for its
            # ability to go RED. An `If` genuinely CONDITIONS the refusal below
            # it; a `try` merely wraps it.
            if isinstance(cur, ast.If):
                out.append(span(cur))
            cur = parents.get(cur)
        return out

    guards: dict[str, list[tuple[int, int]]] = {}
    for node in ast.walk(tree):
        if _is_guard_raise(node):
            fp = f"RAISE::{_enclosing_fn(node, parents)}::{ast.unparse(node)[:120]}"
            guards[fp] = spans_with_ancestors(node)
        elif isinstance(node, ast.If) and _direct_refusal(node):
            fp = (f"BRANCH::{_enclosing_fn(node, parents)}::"
                  f"{ast.unparse(node.test)[:120]}")
            guards[fp] = spans_with_ancestors(node)
    return guards


def apply_cell(src: str, find: str, replace: str) -> str | None:
    """None when the anchor is not uniquely present -- never a silent skip."""
    if src.count(find) != 1:
        return None
    return src.replace(find, replace, 1)


def coverage(src: str, guards: dict[str, list[tuple[int, int]]]):
    """Per-cell measured coverage. Cells declare nothing."""
    covered: dict[str, list[str]] = {fp: [] for fp in guards}
    problems: list[str] = []

    for cell, _desc, find, replace in CELLS:
        mutated = apply_cell(src, find, replace)
        if mutated is None:
            problems.append(
                f"cell {cell}: anchor is not uniquely present "
                f"(count={src.count(find)}) -- cell cannot be scored"
            )
            continue
        try:
            mutated_guards = set(enumerate_guards(mutated))
        except SyntaxError as exc:
            problems.append(f"cell {cell}: mutated source does not parse ({exc})")
            continue

        # (a) structural: guard gone or altered
        for fp in guards:
            if fp not in mutated_guards:
                covered[fp].append(cell)

        # (b) textual: the anchor overlaps the guard's own span
        a_start = src.index(find)
        a_end = a_start + len(find)
        for fp, spans in guards.items():
            if cell in covered[fp]:
                continue
            if any(a_start < g_end and g_start < a_end for g_start, g_end in spans):
                covered[fp].append(cell)

    return covered, problems


SYNTHETIC = '''

def _synthetic_guard_for_the_self_control(value):
    """Planted by verify_matrix_coverage_86_85.py's self-control. Never shipped."""
    if value is None:
        raise LedgerError("synthetic guard -- no cell should cover this", EXIT_INVALID)
    return value
'''


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    sha_before = sha(src)
    print("phase-86.85 C8 -- mutation-matrix coverage, DERIVED from source")
    print("=" * 72)
    print(f"subject      : {TARGET.name}")
    print(f"sha256 before: {sha_before}")
    print(f"cells        : {len(CELLS)}")
    print()

    guards = enumerate_guards(src)
    print(f"guards enumerated FROM SOURCE (rule in the docstring): {len(guards)}")
    n_raise = sum(1 for fp in guards if fp.startswith("RAISE::"))
    n_branch = sum(1 for fp in guards if fp.startswith("BRANCH::"))
    print(f"  GUARD-RAISE : {n_raise}")
    print(f"  GUARD-BRANCH: {n_branch}")
    print()

    # ---- SELF-CONTROL FIRST -------------------------------------------
    # A clean coverage report is worthless if the checker cannot see a guard it
    # planted itself.
    planted_src = src + SYNTHETIC
    planted_guards = enumerate_guards(planted_src)
    new_fps = set(planted_guards) - set(guards)
    print("=== SELF-CONTROL (must pass before any result below is believed) ===")
    if not new_fps:
        print("  FAILED GATE: the enumerator did not see a guard planted into a copy")
        print("  -- a scan that cannot find its own known members is a FAILED gate.")
        return 2
    print(f"  enumerator DETECTED the planted guard ({len(new_fps)} new fingerprint)")

    planted_cov, _ = coverage(planted_src, planted_guards)
    planted_uncovered = [fp for fp in new_fps if not planted_cov.get(fp)]
    if not planted_uncovered:
        print("  FAILED GATE: the planted guard was reported as COVERED, so this")
        print("  checker cannot distinguish covered from uncovered.")
        return 2
    print("  planted guard correctly reported UNCOVERED -- the checker can go RED")
    print()

    # ---- THE REAL AUDIT -----------------------------------------------
    covered, problems = coverage(src, guards)
    uncovered = sorted(fp for fp, cells in covered.items() if not cells)

    print("=== COVERAGE (measured per cell; no cell declares what it covers) ===")
    for fp in sorted(covered):
        cells = covered[fp]
        mark = "OK  " if cells else "GAP "
        print(f"  {mark} {fp[:96]}")
        print(f"        cells: {', '.join(cells) if cells else '<NONE>'}")
    print()

    if problems:
        print("=== CELL PROBLEMS ===")
        for p in problems:
            print(f"  {p}")
        print()

    sha_after = sha(TARGET.read_text(encoding="utf-8"))
    print(f"sha256 after : {sha_after}")
    print(f"UNCHANGED    : {sha_before == sha_after}  (all mutation was in memory)")
    print()

    print("=" * 72)
    print(f"guards: {len(guards)}   covered: {len(guards) - len(uncovered)}   "
          f"uncovered: {len(uncovered)}   cell problems: {len(problems)}")
    if sha_before != sha_after:
        print("\nRESULT: FAILED -- the subject file changed on disk.")
        return 3
    if uncovered or problems:
        print("\nRESULT: FAILED -- a guard has no cell aiming at it:")
        for fp in uncovered:
            print(f"  UNCOVERED  {fp}")
        for p in problems:
            print(f"  PROBLEM    {p}")
        print("\nAdd a cell that measurably touches it. Do NOT add it to a list;")
        print("the coverage above is derived, so a list would not change this result.")
        return 1
    print("\nRESULT: OK -- every enumerated guard is touched by at least one cell.")
    print("This licenses ONLY: 'no guard the enumeration can see is unmutated'.")
    print("It is NOT a claim that the guard set itself is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
