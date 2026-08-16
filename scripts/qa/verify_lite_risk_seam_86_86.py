#!/usr/bin/env python3
"""phase-86.86 (D6) -- re-runnable checker for the lite risk-judge INGRESS seam.

WHAT THIS DEFENDS. `backend/services/autonomous_loop.py` built the persisted
lite `risk_assessment` with::

    float(risk_dict.get("recommended_position_pct")
          or _LITE_RISK_DEFAULT["recommended_position_pct"])

`0.0` is falsy, so an explicit 0% verdict -- the strongest risk signal the judge
can issue -- was rewritten to 3.0 at construction, UPSTREAM of every guard
phase-86.74 built. Measured pre-fix: judge 0.0 -> 3.0 and judge ABSENT -> 3.0,
i.e. indistinguishable.

WHY AN AST WALK AND NOT A GREP. Criterion 3 requires the CLASS to be enumerated
FROM SOURCE rather than hand-listed. A grep over source text matches its own
documentation (see `feedback_a_probe_can_match_its_own_documentation`); the
docstrings in this repo quote the very idiom being hunted. An `ast.BoolOp`
walk sees expressions, not prose, so comments and docstrings cannot poison it.

WHY THE POSITIVE CONTROL IS NOT BUILT FROM THE PRODUCTION FILE. A control
assembled from the same pattern the scanner looks for proves nothing (see
`feedback_a_control_built_from_your_own_pattern_tests_nothing`). This checker
runs its scanner against a synthetic module that CONTAINS the idiom and against
one that does NOT, and requires the scanner to separate them. A scanner that
cannot find a member it is handed is a FAILED gate, not a clean result.

Exit 0 = all checks pass. Exit 1 = at least one failed. Prints every check.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "backend" / "services" / "autonomous_loop.py"

#: The key whose falsy value (0.0) is DECISION-INVERTING -- measured by driving
#: the real `decide_trades`: 0.0 collapsed to 3.0 turns "no order" into
#: "BUY $719.93" on NAV 23,997.71. This key must never sit behind an `or`.
FORBIDDEN_KEY = "recommended_position_pct"

#: Keys whose `or _LITE_RISK_DEFAULT[...]` idiom is RETAINED deliberately. Their
#: falsy triggers are "" / {} and were measured NOT to change any order:
#: an empty `decision` and the substituted APPROVE_REDUCED both produce the same
#: BUY (only an exact REJECT blocks), and `risk_level` is read zero times by
#: portfolio_manager. They fabricate the persisted AUDIT TRAIL, which is real
#: and is filed as step 86.87 -- not silently fixed here.
RETAINED_KEYS = {"decision", "reasoning", "risk_level", "risk_limits"}

#: phase-86.88 (criteria 4 + 5). The WHOLE-DICT COPY routes -- the shape the
#: `<whole-dict>` branch above could never see, because `dict(_LITE_RISK_DEFAULT)`
#: is an `ast.Call` and the branch only inspected `BoolOp` operands.
#:
#: CLASSIFICATION, per criterion 5. These four are the JUDGE-FAILURE routes: the
#: no-JSON and exception handlers of both lite paths. They hand the producer a
#: complete copy of the default, so before phase-86.88 the seam resolved
#: SIZE(3.0) and a judge that produced NOTHING was persisted as a judge that
#: chose 3%. Same number, destroyed provenance -- which is why no
#: number-asserting test saw it.
#:
#: They are now BENIGN, and benign for a stated reason rather than by assertion:
#: `_lite_position_pct` detects the whole-default by VALUE EQUALITY and resolves
#: ABSENT. The sizing is deliberately unchanged (ABSENT yields the same 3.0), so
#: no order outcome moves; only the recorded provenance is corrected.
#:
#: They are enumerated here so that a FIFTH route added later shows up as
#: unclassified and fails this checker, exactly as `<whole-dict-copy>` itself did
#: on the run that introduced it.
WHOLE_DICT_COPY = "<whole-dict-copy>"


def or_default_sites(tree: ast.AST) -> list[tuple[int, str]]:
    """Every `<x> or _LITE_RISK_DEFAULT[<key>]` in an AST, as (lineno, key)."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)):
            continue
        for operand in node.values:
            if (isinstance(operand, ast.Subscript)
                    and isinstance(operand.value, ast.Name)
                    and operand.value.id == "_LITE_RISK_DEFAULT"):
                key = getattr(operand.slice, "value", "<dynamic>")
                out.append((operand.lineno, key))
            elif isinstance(operand, ast.Name) and operand.id == "_LITE_RISK_DEFAULT":
                out.append((operand.lineno, "<whole-dict>"))
    # phase-86.88 (criterion 4). WIDENED NODE SHAPES -- and the reason matters,
    # because the step that filed this called the branch above DEAD and it is
    # not. Measured: it FIRES on `x or _LITE_RISK_DEFAULT` (a bare Name operand)
    # and is blind ONLY to the `dict(_LITE_RISK_DEFAULT)` **Call** shape, which
    # is the shape this codebase actually uses -- so it was unreachable on the
    # shipped file while being perfectly capable in general.
    #
    # "Made LIVE" therefore means widening what counts as a reference, not
    # resurrecting a corpse. A whole-dict COPY is exactly as capable of carrying
    # the default into the seam as a bare reference is -- more so, since all
    # four production routes take that form.
    # phase-86.88 cycle 2 (Q/A finding 4). The cycle-1 widening saw only
    # BARE-NAME calls -- `dict(X)` and `deepcopy(X)` -- while the artifact claimed
    # coverage of "dict(), copy() and deepcopy() call shapes". MEASURED by the
    # Q/A against 7 shapes, these were still INVISIBLE: `copy.deepcopy(X)`,
    # `copy.copy(X)`, `dict(**X)`, `X.copy()` and `{**X}`. A stated bound that is
    # narrower than the tool's real blindness is worse than no bound, so the
    # shapes are covered rather than the sentence softened.
    def _is_default(n) -> bool:
        return isinstance(n, ast.Name) and n.id == "_LITE_RISK_DEFAULT"

    for node in ast.walk(tree):
        # dict(X) / copy(X) / deepcopy(X)  AND  copy.copy(X) / copy.deepcopy(X)
        if isinstance(node, ast.Call):
            fn = node.func
            bare = isinstance(fn, ast.Name) and fn.id in ("dict", "copy", "deepcopy")
            dotted = isinstance(fn, ast.Attribute) and fn.attr in ("copy", "deepcopy")
            # X.copy()  -- the receiver IS the default
            receiver = isinstance(fn, ast.Attribute) and fn.attr == "copy" and _is_default(fn.value)
            if receiver:
                out.append((fn.value.lineno, "<whole-dict-copy>"))
                continue
            if bare or dotted:
                for arg in node.args:
                    if _is_default(arg):
                        out.append((arg.lineno, "<whole-dict-copy>"))
                # dict(**X)
                for kw in node.keywords:
                    if kw.arg is None and _is_default(kw.value):
                        out.append((kw.value.lineno, "<whole-dict-copy>"))
        # {**X}
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None and _is_default(v):
                    out.append((v.lineno, "<whole-dict-copy>"))
    return sorted(out)


def default_pct_readers(tree: ast.AST) -> list[tuple[int, str]]:
    """Every function that can reach `_LITE_RISK_DEFAULT[FORBIDDEN_KEY]`.

    Returned as (lineno, enclosing function name). Criterion 2 requires this to
    be exactly one place.
    """
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == "_LITE_RISK_DEFAULT"
                and getattr(node.slice, "value", None) == FORBIDDEN_KEY):
            cur: ast.AST | None = node
            fname = "<module>"
            while cur is not None:
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    fname = cur.name
                    break
                cur = parents.get(cur)
            hits.append((node.lineno, fname))
    return sorted(hits)


def calls_to(tree: ast.AST, name: str) -> list[int]:
    return sorted(
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
    )


def defs_of(tree: ast.AST, name: str) -> list[int]:
    return sorted(
        n.lineno for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


# --------------------------------------------------------------------------
# Synthetic fixtures for the positive/negative control. Deliberately NOT read
# from the production file: a control drawn from the thing under test cannot
# discriminate.
# --------------------------------------------------------------------------
_CONTROL_POSITIVE = '''
_LITE_RISK_DEFAULT = {"recommended_position_pct": 3.0, "risk_level": "MODERATE"}

def producer(risk_dict):
    return {
        "recommended_position_pct": float(
            risk_dict.get("recommended_position_pct")
            or _LITE_RISK_DEFAULT["recommended_position_pct"]
        ),
        "risk_level": str(risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]),
    }
'''

_CONTROL_NEGATIVE = '''
_LITE_RISK_DEFAULT = {"recommended_position_pct": 3.0, "risk_level": "MODERATE"}

# A docstring/comment that QUOTES the idiom must NOT be counted:
#   float(risk_dict.get("recommended_position_pct")
#         or _LITE_RISK_DEFAULT["recommended_position_pct"])
def producer(risk_dict):
    """Mentions `or _LITE_RISK_DEFAULT["recommended_position_pct"]` in prose."""
    verdict = resolve(risk_dict)
    if verdict.kind == "SIZE":
        return float(verdict.pct)
    return float(_LITE_RISK_DEFAULT["recommended_position_pct"])
'''


def main() -> int:
    failures: list[str] = []
    checks: list[str] = []

    def ok(msg: str) -> None:
        checks.append(f"  PASS  {msg}")

    def bad(msg: str) -> None:
        checks.append(f"  FAIL  {msg}")
        failures.append(msg)

    # ---- CONTROL FIRST -------------------------------------------------
    # If the scanner cannot separate a module that HAS the idiom from one that
    # only mentions it in prose, every result below is meaningless.
    pos = or_default_sites(ast.parse(_CONTROL_POSITIVE))
    neg = or_default_sites(ast.parse(_CONTROL_NEGATIVE))
    pos_keys = {k for _, k in pos}

    if FORBIDDEN_KEY in pos_keys and "risk_level" in pos_keys:
        ok(f"control(+): scanner FOUND the idiom it hunts ({sorted(pos_keys)})")
    else:
        bad(f"control(+): scanner MISSED a known member -- found {sorted(pos_keys)}; "
            "a scan that cannot find its own known members is a FAILED gate")

    if not neg:
        ok("control(-): prose/comments quoting the idiom did NOT register (AST, not grep)")
    else:
        bad(f"control(-): scanner matched prose at {neg} -- it is grepping, not parsing")

    # ---- THE REAL TARGET ------------------------------------------------
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    sites = or_default_sites(tree)
    keys = {k for _, k in sites}

    checks.append("")
    checks.append(f"  Enumerated `or _LITE_RISK_DEFAULT[...]` sites in {TARGET.name}: {len(sites)}")
    for lineno, key in sites:
        checks.append(f"    line {lineno:5d}  key={key!r}")
    checks.append("")

    # criterion 3 / the fix itself
    if FORBIDDEN_KEY not in keys:
        ok(f"{FORBIDDEN_KEY!r} appears in ZERO `or _LITE_RISK_DEFAULT[...]` nodes "
           "(the decision-inverting member is gone from the class)")
    else:
        offending = [ln for ln, k in sites if k == FORBIDDEN_KEY]
        bad(f"{FORBIDDEN_KEY!r} is back behind an `or` at line(s) {offending} -- "
            "an explicit 0.0 verdict is being destroyed at ingress again")

    # phase-86.88: the whole-dict copies are a CLASSIFIED member, not an
    # unexpected one. Asserted separately below so their count is pinned rather
    # than merely tolerated.
    copies = sorted(ln for ln, k in sites if k == WHOLE_DICT_COPY)
    if len(copies) == 4:
        ok(f"the 4 judge-failure whole-dict routes are SEEN by the scanner at {copies} "
           "(phase-86.88: the branch now fires on a real match, not only on a control)")
    else:
        bad(f"expected 4 whole-dict copy routes, found {len(copies)} at {copies} -- "
            "a route was added or removed; classify it before shipping")

    # phase-86.88 cycle 4 (Q/A NOTE). The cycle-3 artifact and commit message both
    # credited both-path safety to "the literal count ASSERTED == 2 rather than
    # assumed". A grep returned exactly 2 hits -- BOTH production lines, zero test
    # or checker assertions. Read as "I measured it once" the sentence was true;
    # read as a shipped guard it was not, and that is precisely what left the
    # Gemini half unprotected when the persisted provenance was added. So the
    # count is asserted HERE, where it runs on every invocation.
    prov = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and n.value == "risk_assessment_provenance"]
    if len(prov) == 2:
        ok(f"the persisted provenance block is present on BOTH lite paths at {sorted(prov)} "
           "(phase-86.88: asserted every run, not measured once)")
    else:
        bad(f"expected the persisted provenance on exactly 2 lite paths, found {len(prov)} "
            f"at {sorted(prov)} -- one path would persist judge-failed and judge-said-3% "
            "byte-identically")

    unexpected = keys - RETAINED_KEYS - {WHOLE_DICT_COPY}
    if not unexpected:
        ok(f"remaining members are exactly the retained set {sorted(RETAINED_KEYS)} "
           f"plus the classified {WHOLE_DICT_COPY} routes")
    else:
        bad(f"unexpected member(s) in the class: {sorted(unexpected)} -- "
            "classify them before shipping (criterion 3)")

    # criterion 2: exactly one place can reach the pct default
    readers = default_pct_readers(tree)
    fn_names = sorted({f for _, f in readers})
    if len(fn_names) == 1 and fn_names[0] == "_lite_position_pct":
        ok(f"exactly ONE function can reach _LITE_RISK_DEFAULT[{FORBIDDEN_KEY!r}]: "
           f"{fn_names[0]} (at line(s) {[ln for ln, _ in readers]})")
    else:
        bad(f"the {FORBIDDEN_KEY!r} default is reachable from {len(fn_names)} "
            f"function(s): {fn_names} -- criterion 2 requires exactly one "
            "(_lite_position_pct)")

    # one producer, and BOTH lite paths route through it
    n_defs = defs_of(tree, "_build_lite_risk_assessment")
    n_calls = calls_to(tree, "_build_lite_risk_assessment")
    if len(n_defs) == 1:
        ok(f"_build_lite_risk_assessment defined exactly once (line {n_defs[0]})")
    else:
        bad(f"_build_lite_risk_assessment defined {len(n_defs)} times at {n_defs}")

    if len(n_calls) == 2:
        ok(f"BOTH lite paths route through the one producer (call sites {n_calls})")
    else:
        bad(f"expected 2 call sites for _build_lite_risk_assessment (Claude + "
            f"Gemini lite), found {len(n_calls)} at {n_calls} -- a path may have "
            "drifted off the shared seam")

    n_seam_defs = defs_of(tree, "_lite_position_pct")
    n_seam_calls = calls_to(tree, "_lite_position_pct")
    if len(n_seam_defs) == 1 and len(n_seam_calls) == 1:
        ok(f"_lite_position_pct: 1 definition (line {n_seam_defs[0]}), "
           f"1 call site (line {n_seam_calls[0]}) -- no second parallel idiom")
    else:
        bad(f"_lite_position_pct has {len(n_seam_defs)} definition(s) at "
            f"{n_seam_defs} and {len(n_seam_calls)} call site(s) at {n_seam_calls}")

    print("phase-86.86 -- lite risk-judge INGRESS seam checker")
    print("=" * 70)
    print("\n".join(checks))
    print("=" * 70)
    n_pass = sum(1 for c in checks if c.startswith("  PASS"))
    n_fail = sum(1 for c in checks if c.startswith("  FAIL"))
    print(f"checks emitted: {n_pass + n_fail}  (PASS {n_pass} / FAIL {n_fail})")
    if failures:
        print("\nRESULT: FAILED")
        return 1
    print("\nRESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
