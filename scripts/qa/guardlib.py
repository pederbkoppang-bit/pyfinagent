#!/usr/bin/env python3
"""Make RED-FIRST and SELF-ADVERSARIAL mechanical instead of remembered.

WHY THIS EXISTS -- the measurement, not a preference
----------------------------------------------------
Over one session the Layer-3 Q/A rail spent **1,511,690 tokens across 7
evaluations on 6 steps and returned ZERO product defects**. Every capping
finding was about *my* evidence, and five of them were the same bug wearing
different clothes -- a guard that could not fail:

* a check on a *value* where the defect lived in a *definition*;
* a check on the *right* property read off the *wrong* variable;
* a bound that no input could violate (``count >= 0`` on a count);
* an ``A or B`` whose ``B`` was true on the control, so the guard was green
  no matter what ``A`` did;
* a fixture that had been emptied, leaving the assertion looping over nothing.

Every one of those would have been caught in seconds by asking a single
question: *show me an input where this guard says NO.* None of them was asked,
because asking was manual. This module makes it structural.

WHAT IT ENFORCES
----------------
1. ``Guards.ok()`` **cannot be called without a known-bad fixture.** You pass a
   predicate plus at least one ``falsified_by`` value, and the library re-proves
   on EVERY RUN that the predicate returns ``False`` there. A guard whose red
   state cannot be exhibited raises :class:`VacuousGuard` and is not counted.
   The red state is therefore re-measured continuously, not asserted once in a
   commit message.

2. **Compound predicates need one counterexample per ``and``-clause.** A single
   counterexample against ``A and B`` licenses nothing about the clause it did
   not falsify -- measured the hard way when a mutation killed a whole return
   statement and the matrix read 9/9 while one ``startswith`` clause had no
   falsifying fixture at all. :func:`Guards.ok` splits the predicate's AST and
   requires each ``and``-clause to be individually falsified.

3. :func:`census` fails when a registered guard has **no mutation cell**. It is
   an AST census, never a grep: a grep over source matches the guard's own
   documentation, which is how a probe once certified itself. Docstrings are
   structurally excluded because only *data* positions (container literals and
   call arguments) are collected.

4. :class:`MutationMatrix` is the multi-target, signal-safe runner: the control
   is observed GREEN first, ``pytest`` exit 5 ("no tests collected") scores as
   UNSCORABLE rather than a kill, the mutant must collect the SAME number of
   tests as the control, the NAMED test must be among the failures, restore is
   verified by SHA-256, and SIGTERM/SIGINT/SIGHUP restore **every** target
   (``try/finally`` does not run on a signal -- a 2-minute timeout stranded a
   mutant on disk during step 86.59).

HAZARD -- this module writes to your working tree
-------------------------------------------------
:class:`MutationMatrix` edits real files in place and restores them. **Never
wire it into a hook or any automation that runs during ordinary edits**: a
disk-mutating checker fired from a PostToolUse hook produced 14 truncated
states in a single run. It refuses to start from a target that already carries
a ``# MUTANT`` marker, which converts "a previous run was killed mid-cell" from a
silent poisoned baseline into a loud stop.

Usage::

    import sys; sys.path.insert(0, "scripts/qa")
    from guardlib import Guards, census, Cell, Target, MutationMatrix, pytest_runner

    g = Guards()
    g.ok("dup_keys_are_present", lambda d: d["dup_keys"] > 0, measured,
         falsified_by={"dup_keys": 0})

The red-first proof for this module is :mod:`guardlib_selftest`, which feeds it
known-bad guards and asserts it rejects them::

    python scripts/qa/guardlib_selftest.py            # watch the guards go red
    python scripts/qa/guardlib_selftest.py --mutate   # guardlib mutates itself
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "VacuousGuard",
    "GuardFailed",
    "CounterexampleError",
    "Guards",
    "GuardRecord",
    "census",
    "CensusResult",
    "Target",
    "Cell",
    "MutationMatrix",
    "RunResult",
    "pytest_runner",
    "script_runner",
]

# The token every mutation must carry, so a stranded one is DETECTABLE. It is
# the commented form rather than the bare word on purpose: the bare word occurs
# legitimately in prose and in this module's own source, and a stranded-marker
# check that fires on its own documentation refuses to run against a clean tree.
# The convention is not merely hoped for -- `MutationMatrix` REFUSES any cell
# whose replacement text omits it, so every mutation this runner can write is
# one it can also detect.
MUTANT_MARKER = "# MUTANT"


# ---------------------------------------------------------------------------
# exceptions -- deliberately distinct, because they mean opposite things
# ---------------------------------------------------------------------------


class VacuousGuard(AssertionError):
    """The guard could not be shown to fail. It is decoration, not evidence."""


class GuardFailed(AssertionError):
    """The guard fired: the invariant does not hold on the measured subject."""


class CounterexampleError(AssertionError):
    """A ``falsified_by`` fixture is unusable (wrong shape, or it raised)."""


# ---------------------------------------------------------------------------
# strict truth -- "truthy" is how a non-answer passes for an answer
# ---------------------------------------------------------------------------


def _strict_bool(value: Any, where: str) -> bool:
    """Return a real bool, or raise.

    A predicate that returns ``[1]``, ``1`` or ``"yes"`` is *truthy*, and every
    truthy non-bool is a place where a guard says YES without having been asked
    a yes/no question. numpy/pandas scalars are accepted via ``.item()`` because
    this repo's measurements are pandas-shaped; a numpy ARRAY raises from
    ``.item()`` and is therefore rejected, which is correct -- an array has no
    single truth value.
    """
    if value is True or value is False:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            inner = item()
        except Exception:  # pragma: no cover - shape-dependent
            inner = None
        if inner is True or inner is False:
            return inner
    raise VacuousGuard(
        f"{where}: predicate returned {value!r} of type "
        f"{type(value).__name__}, not a bool. A truthy non-bool means the "
        f"guard was never asked a yes/no question."
    )


# ---------------------------------------------------------------------------
# clause analysis -- so a compound guard cannot hide an unfalsified clause
# ---------------------------------------------------------------------------


def _predicate_lambda_node(predicate: Callable) -> tuple[ast.AST | None, str]:
    """Return ``(node, reason)`` where node is the Lambda/FunctionDef body expr.

    ``reason`` is empty on success and carries a human-readable cause when the
    source could not be recovered. Failure is DEGRADATION, never silence: the
    caller records it and :meth:`Guards.summary` reports how many guards had no
    clause census, so an unenforced predicate is visible rather than assumed
    safe.
    """
    try:
        src = inspect.getsource(predicate)
    except (OSError, TypeError) as exc:
        return None, f"source unavailable ({exc.__class__.__name__})"
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        # A lambda extracted mid-expression can be an unparseable fragment.
        try:
            tree = ast.parse("(" + textwrap.dedent(src).rstrip().rstrip(",") + ")")
        except SyntaxError as exc:
            return None, f"source did not parse ({exc.msg})"

    lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    name = getattr(predicate, "__name__", "")
    if name == "<lambda>":
        if len(lambdas) != 1:
            return None, (
                f"{len(lambdas)} lambdas on the extracted source line -- which "
                f"one is the predicate is ambiguous"
            )
        return lambdas[0], ""

    defs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    if len(defs) != 1:
        return None, f"{len(defs)} defs named {name!r} in the extracted source"
    body = defs[0].body
    body = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None, "predicate is not a single `return <expr>` -- clause census skipped"
    return defs[0], ""


def _body_expr(node: ast.AST) -> ast.expr:
    if isinstance(node, ast.Lambda):
        return node.body
    stmts = [
        s
        for s in node.body  # type: ignore[attr-defined]
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return stmts[0].value  # type: ignore[union-attr,return-value]


def _and_clauses(node: ast.expr) -> list[ast.expr]:
    """Flatten nested ``and`` into its individual clauses.

    ``or`` is deliberately NOT split. Any value that falsifies ``A or B`` has
    already falsified both clauses, so an ``or`` needs no per-clause fixture --
    its failure mode is the opposite one (a clause that is ALWAYS true), and
    that shows up as "no counterexample falsifies this predicate at all", which
    :meth:`Guards.ok` already rejects.
    """
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out: list[ast.expr] = []
        for value in node.values:
            out.extend(_and_clauses(value))
        return out
    return [node]


def _clause_globals(predicate: Callable) -> dict[str, Any]:
    """Globals for an isolated clause, WITH the predicate's closure rebound.

    Without this, a predicate that closes over a local (``lambda d: d["a"] >
    limit``) compiles fine and then raises ``NameError`` when the isolated
    clause is called -- which an earlier revision scored as "this clause was
    never shown false", i.e. it REJECTED a perfectly good guard. A false
    positive here is worse than a miss: it teaches the author that the library
    is wrong, and the next lesson learned is to stop using it.
    """
    scope = dict(getattr(predicate, "__globals__", {}))
    code = getattr(predicate, "__code__", None)
    cells = getattr(predicate, "__closure__", None)
    if code is not None and cells:
        for name, cell in zip(code.co_freevars, cells):
            try:
                scope[name] = cell.cell_contents
            except ValueError:  # pragma: no cover - unbound cell
                pass
    return scope


def _compile_clause(clause: ast.expr, node: ast.AST, predicate: Callable):
    """Build a callable for one ``and``-clause, or return ``None``."""
    args = node.args  # type: ignore[attr-defined]
    lam = ast.Lambda(args=args, body=clause)
    expr = ast.Expression(body=lam)
    ast.fix_missing_locations(expr)
    try:
        code = compile(expr, "<guardlib-clause>", "eval")
        return eval(code, _clause_globals(predicate))  # noqa: S307
    except Exception:
        # The caller degrades to the count-only requirement and says so out loud.
        return None


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@dataclass
class GuardRecord:
    name: str
    detail: str
    counterexamples: int
    and_clauses: int
    clause_census: str  # "ENFORCED" | "PARTIAL:<why>" | "UNAVAILABLE:<why>"


class Guards:
    """A registry of invariants that each carry their own proof of falsifiability.

    Every :meth:`ok` call performs BOTH halves of red-first, in this order:

    1. RED -- the predicate is applied to each ``falsified_by`` fixture and must
       return ``False``. If it does not, the guard is vacuous and raises.
    2. GREEN -- the predicate is applied to the measured subject and must return
       ``True``. If it does not, the invariant genuinely failed and raises
       :class:`GuardFailed`, which is a different outcome and says so.
    """

    def __init__(self, *, label: str = "") -> None:
        self.label = label
        self.records: list[GuardRecord] = []

    # `ok` is the short name used at call sites; `guard` is an alias so the
    # census can be pointed at either spelling.
    _UNSET = object()

    def ok(
        self,
        name: str,
        predicate: Callable[[Any], Any],
        subject: Any,
        *,
        falsified_by: Any = _UNSET,
        falsified_by_each: Any = _UNSET,
        detail: str = "",
    ) -> None:
        """Register one invariant, red half first.

        Exactly one of ``falsified_by`` (ONE known-bad value, never unpacked --
        so a list or dict fixture passes through intact) or
        ``falsified_by_each`` (a sequence of known-bad values, one per
        ``and``-clause) must be supplied. The split exists because the obvious
        single-parameter design is silently ambiguous: ``falsified_by=[1, 2,
        3]`` could mean one bad list or three bad ints, and guessing wrong
        changes what was proved without saying so.
        """
        if not isinstance(name, str) or not name.strip():
            raise VacuousGuard("a guard must have a non-empty name")
        if any(r.name == name for r in self.records):
            raise VacuousGuard(
                f"{name!r} is registered twice -- two guards sharing a name make "
                f"the mutation census ambiguous about which one has a cell"
            )
        if not callable(predicate):
            raise VacuousGuard(
                f"{name}: predicate must be a CALLABLE, not a precomputed "
                f"{type(predicate).__name__}. A precomputed boolean cannot be "
                f"re-applied to a known-bad input, so its red state can never "
                f"be shown."
            )

        fixtures = self._normalise_fixtures(name, falsified_by, falsified_by_each)

        # ---- half 1: RED -------------------------------------------------
        for i, bad in enumerate(fixtures):
            try:
                result = predicate(bad)
            except Exception as exc:
                raise CounterexampleError(
                    f"{name}: falsified_by[{i}]={bad!r} made the predicate "
                    f"raise {exc.__class__.__name__}({exc}). A counterexample "
                    f"must be a value the guard can EVALUATE and reject, not "
                    f"one it chokes on -- otherwise the red state proves only "
                    f"that the fixture is the wrong shape."
                ) from exc
            if _strict_bool(result, f"{name} (on falsified_by[{i}])"):
                raise VacuousGuard(
                    f"{name}: falsified_by[{i}]={bad!r} was supposed to be "
                    f"REJECTED and the predicate accepted it. This guard has "
                    f"no demonstrated red state, so a green run of it is not "
                    f"evidence of anything. {detail}"
                )

        and_clauses, census_state = self._clause_census(name, predicate, fixtures)

        # ---- half 2: GREEN -----------------------------------------------
        try:
            live = predicate(subject)
        except Exception as exc:
            raise GuardFailed(
                f"{name}: predicate raised {exc.__class__.__name__}({exc}) on "
                f"the measured subject. {detail}"
            ) from exc
        if not _strict_bool(live, f"{name} (on the measured subject)"):
            raise GuardFailed(f"INVARIANT FAILED: {name} -- {detail}")

        self.records.append(
            GuardRecord(
                name=name,
                detail=detail,
                counterexamples=len(fixtures),
                and_clauses=and_clauses,
                clause_census=census_state,
            )
        )

    guard = ok

    # -- internals ---------------------------------------------------------

    @classmethod
    def _normalise_fixtures(cls, name: str, one: Any, each: Any) -> list[Any]:
        has_one = one is not cls._UNSET
        has_each = each is not cls._UNSET
        if has_one and has_each:
            raise VacuousGuard(
                f"{name}: pass falsified_by OR falsified_by_each, not both -- "
                f"which set was actually proved would be ambiguous"
            )
        if not has_one and not has_each:
            raise VacuousGuard(
                f"{name}: no known-bad fixture. A guard with no counterexample "
                f"is decoration: nothing about it has been shown to fail."
            )
        if has_one:
            fixtures = [one]
        else:
            if not isinstance(each, (list, tuple)):
                raise VacuousGuard(
                    f"{name}: falsified_by_each must be a list or tuple, got "
                    f"{type(each).__name__} -- use falsified_by for a single value"
                )
            fixtures = list(each)
        if not fixtures:
            raise VacuousGuard(
                f"{name}: falsified_by is EMPTY. A guard with no known-bad "
                f"fixture is decoration -- an empty fixture is exactly how an "
                f"assertion ends up looping over nothing and reporting green."
            )
        seen: list[str] = []
        for bad in fixtures:
            key = repr(bad)
            if key in seen:
                raise VacuousGuard(
                    f"{name}: falsified_by contains {bad!r} twice. A repeated "
                    f"counterexample proves nothing the first one did not."
                )
            seen.append(key)
        return fixtures

    def _clause_census(
        self, name: str, predicate: Callable, fixtures: Sequence[Any]
    ) -> tuple[int, str]:
        node, reason = _predicate_lambda_node(predicate)
        if node is None:
            return 1, f"UNAVAILABLE:{reason}"
        clauses = _and_clauses(_body_expr(node))
        if len(clauses) <= 1:
            return 1, "ENFORCED"

        if len(fixtures) < len(clauses):
            raise VacuousGuard(
                f"{name}: the predicate has {len(clauses)} `and`-clauses but "
                f"only {len(fixtures)} counterexample(s). One counterexample "
                f"against `A and B` licenses NOTHING about the clause it did "
                f"not falsify -- supply one fixture per clause. Clauses: "
                + "; ".join(ast.unparse(c) for c in clauses)
            )

        unfalsified: list[str] = []
        uncheckable = 0
        for clause in clauses:
            fn = _compile_clause(clause, node, predicate)
            if fn is None:
                uncheckable += 1
                continue
            evaluated = False
            shown_false = False
            for bad in fixtures:
                try:
                    value = _strict_bool(fn(bad), f"{name} (clause)")
                except Exception:
                    continue  # inconclusive for this clause on this fixture
                evaluated = True
                if not value:
                    shown_false = True
                    break
            if not evaluated:
                # NEVER SUCCESSFULLY EVALUATED is not the same as NEVER FALSE.
                # Conflating them rejects good guards, so it degrades loudly
                # instead of accusing.
                uncheckable += 1
            elif not shown_false:
                unfalsified.append(ast.unparse(clause))
        if unfalsified:
            raise VacuousGuard(
                f"{name}: no counterexample falsifies "
                + "; ".join(repr(c) for c in unfalsified)
                + ". That clause is unfalsified by this guard's own fixtures, "
                "so a green run says nothing about it."
            )
        if uncheckable:
            return len(clauses), (
                f"PARTIAL:{uncheckable}/{len(clauses)} clauses could not be "
                f"isolated (closure variables); count-only requirement applied"
            )
        return len(clauses), "ENFORCED"

    # -- reporting ---------------------------------------------------------

    def names(self) -> list[str]:
        return [r.name for r in self.records]

    def summary(self) -> str:
        total = len(self.records)
        partial = [r for r in self.records if r.clause_census.startswith("PARTIAL")]
        unavailable = [r for r in self.records if r.clause_census.startswith("UNAVAILABLE")]
        fixtures = sum(r.counterexamples for r in self.records)
        head = (
            f"{self.label + ' ' if self.label else ''}guards: {total} passed, "
            f"each with a demonstrated red state ({fixtures} known-bad fixtures "
            f"re-proved this run)"
        )
        lines = [head]
        if partial:
            lines.append(
                f"  clause census PARTIAL on {len(partial)}: "
                + ", ".join(r.name for r in partial)
            )
        if unavailable:
            lines.append(
                f"  clause census UNAVAILABLE on {len(unavailable)}: "
                + ", ".join(r.name for r in unavailable)
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# census -- every guard must have a mutation cell
# ---------------------------------------------------------------------------


@dataclass
class CensusResult:
    guards: dict[str, list[str]] = field(default_factory=dict)
    cell_strings: set[str] = field(default_factory=set)
    uncelled: list[str] = field(default_factory=list)
    dynamic_names: list[str] = field(default_factory=list)
    never_executed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.uncelled or self.dynamic_names or self.problems)

    def report(self) -> str:
        lines = [
            f"guards registered : {len(self.guards)}",
            f"cell strings found: {len(self.cell_strings)}",
        ]
        for label, items in (
            ("UNCELLED (no mutation cell names this guard)", self.uncelled),
            ("DYNAMIC NAME (census cannot see this guard)", self.dynamic_names),
            ("NEVER EXECUTED (in source, absent at runtime)", self.never_executed),
            ("PROBLEM", self.problems),
        ):
            for item in items:
                lines.append(f"  {label}: {item}")
        lines.append("CENSUS " + ("OK" if self.ok else "FAILED"))
        return "\n".join(lines)


def _data_strings(tree: ast.AST) -> set[str]:
    """String constants in DATA positions only.

    Container literals, dict keys/values and call arguments -- never a bare
    ``Expr`` statement, which is what a docstring is. This is the whole reason
    the census is an AST walk and not a grep: 121 lines of prose in a module
    docstring will satisfy an unanchored grep for a guard name, and did.
    """
    found: set[str] = set()

    def add(node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)

    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                add(element)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                add(key)
            for value in node.values:
                add(value)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                add(arg)
            for kw in node.keywords:
                add(kw.value)
    return found


def census(
    guard_sources: Iterable[Path | str],
    cell_sources: Iterable[Path | str],
    *,
    callees: Sequence[str] = ("ok", "guard", "_ok"),
    runtime_names: Sequence[str] | None = None,
    exempt: Sequence[str] = (),
) -> CensusResult:
    """Fail when a registered guard has no mutation cell naming it.

    ``guard_sources`` are the files that CALL ``ok()``; ``cell_sources`` are the
    files that declare mutation cells. A guard is covered when its name appears
    as a string constant in a data position of some cell source.

    Passing ``runtime_names`` (from :meth:`Guards.names`) additionally reports
    guards that exist in source but never executed -- the "loop over an empty
    glob" class, where the assertion is real and simply never runs.
    """
    result = CensusResult()

    for src in guard_sources:
        path = Path(src)
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError) as exc:
            result.problems.append(f"{path}: unreadable ({exc})")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            fname = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if fname not in callees or not node.args:
                continue
            first = node.args[0]
            where = f"{path}:{node.lineno}"
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                result.guards.setdefault(first.value, []).append(where)
            else:
                result.dynamic_names.append(
                    f"{where} -- first argument is {type(first).__name__}, not a "
                    f"string literal, so no census can see this guard's name"
                )

    for src in cell_sources:
        path = Path(src)
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError) as exc:
            result.problems.append(f"{path}: unreadable ({exc})")
            continue
        result.cell_strings |= _data_strings(tree)

    # Positive control. A census that matched nothing must not report success:
    # an empty result is what a mis-pointed path looks like, and it is
    # indistinguishable from full coverage unless it is called out.
    if not result.guards:
        result.problems.append(
            "no guards found in guard_sources -- an empty census is not "
            "evidence of coverage, it is evidence the paths are wrong"
        )
    if not result.cell_strings:
        result.problems.append(
            "no cell strings found in cell_sources -- same positive control, "
            "on the other side"
        )

    exempt_set = set(exempt)
    for name, sites in result.guards.items():
        if name in exempt_set:
            continue
        if name not in result.cell_strings:
            result.uncelled.append(f"{name} (registered at {', '.join(sites)})")

    if runtime_names is not None:
        live = set(runtime_names)
        for name in result.guards:
            if name not in live and name not in exempt_set:
                result.never_executed.append(name)

    return result


# ---------------------------------------------------------------------------
# multi-target, signal-safe mutation runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    rc: int
    collected: int  # -1 when the runner has no meaningful collected count
    output: str


@dataclass
class Target:
    """One mutatable file plus how to exercise it.

    ``score_rc`` decides whether a given exit code is even ELIGIBLE to be a
    kill. It is per-target because the rule is runner-specific: ``pytest``
    exits 5 on "no tests collected", so a typo in ``-k`` would otherwise be
    scored as a successful kill, while a plain script legitimately exits with
    whatever it likes.
    """

    path: Path
    runner: Callable[[], RunResult]
    name: str = ""
    score_rc: Callable[[int], bool] = lambda rc: rc == 1
    compare_collected: bool = True
    # How many `# MUTANT` markers the CLEAN file legitimately contains. Almost
    # always 0. Set it to None only for a file that defines or documents the
    # marker itself -- guardlib.py cannot pattern-scan for a token declared in
    # its own source without refusing to run against a spotless tree, which is
    # exactly what it did on the first self-mutation attempt. Disabling the
    # broad net does NOT disable stranded-detection: the exact check below
    # looks for this matrix's own replacement texts and is not pattern-based.
    baseline_markers: int | None = 0

    def label(self) -> str:
        return self.name or self.path.name


@dataclass
class Cell:
    name: str
    target: Path
    old: str
    new: str
    named: str  # the test / invariant that MUST appear among the failures


@dataclass
class CellResult:
    name: str
    verdict: str  # KILLED | SURVIVED | UNSCORABLE
    detail: str


def pytest_runner(suite: str, cwd: Path, extra: Sequence[str] = ()) -> Callable[[], RunResult]:
    def run() -> RunResult:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-q", "--no-header", *extra],
            capture_output=True,
            text=True,
            cwd=str(cwd),
        )
        out = proc.stdout + proc.stderr
        passed = re.search(r"(\d+) passed", out)
        failed = re.search(r"(\d+) failed", out)
        collected = (int(passed.group(1)) if passed else 0) + (
            int(failed.group(1)) if failed else 0
        )
        return RunResult(proc.returncode, collected, out)

    return run


def script_runner(argv: Sequence[str], cwd: Path) -> Callable[[], RunResult]:
    def run() -> RunResult:
        proc = subprocess.run(
            [sys.executable, *argv], capture_output=True, text=True, cwd=str(cwd)
        )
        return RunResult(proc.returncode, -1, proc.stdout + proc.stderr)

    return run


class MutationMatrix:
    """Score mutation cells across several targets without ever stranding one."""

    def __init__(
        self,
        targets: Sequence[Target],
        cells: Sequence[Cell],
        *,
        equivalent_by_design: Sequence[tuple[str, str]] = (),
        title: str = "MUTATION MATRIX",
    ) -> None:
        self.targets = {Path(t.path).resolve(): t for t in targets}
        self.cells = list(cells)
        self.equivalent_by_design = list(equivalent_by_design)
        self.title = title
        self.results: list[CellResult] = []
        for cell in self.cells:
            if Path(cell.target).resolve() not in self.targets:
                raise ValueError(
                    f"cell {cell.name!r} mutates {cell.target}, which is not a "
                    f"declared target -- an undeclared target would never be "
                    f"restored on a signal"
                )
            if MUTANT_MARKER not in cell.new:
                raise ValueError(
                    f"cell {cell.name!r} writes a replacement that does not "
                    f"carry the {MUTANT_MARKER!r} marker. The stranded-mutation "
                    f"check scans for that marker, so an unmarked mutation is "
                    f"one this runner could leave on disk and then fail to "
                    f"notice on the next run."
                )

    # -- lifecycle ---------------------------------------------------------

    def _install_signal_restore(self, originals: dict[Path, bytes]) -> None:
        def restore_and_die(signum, _frame):  # pragma: no cover - signal path
            for path, data in originals.items():
                try:
                    path.write_bytes(data)
                except OSError:
                    pass
            print(f"\nsignal {signum} -- all targets restored, matrix aborted")
            raise SystemExit(130)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(sig, restore_and_die)
            except (ValueError, OSError):
                pass

    def run(self) -> int:
        originals = {p: p.read_bytes() for p in self.targets}
        shas = {p: hashlib.sha256(b).hexdigest() for p, b in originals.items()}

        stranded: list[str] = []
        for path, target in self.targets.items():
            text = originals[path].decode(errors="replace")
            # Exact check: a mutation this matrix could have written is sitting
            # on disk. No false positives, no dependence on a marker convention.
            for cell in self.cells:
                if Path(cell.target).resolve() == path and cell.new in text:
                    stranded.append(
                        f"{target.label()} already contains the replacement text "
                        f"of cell {cell.name!r}"
                    )
                    break
            # Broad net: catches a stranding from a DIFFERENT cell set than the
            # one declared here, which the exact check by construction cannot.
            if target.baseline_markers is not None:
                count = text.count(MUTANT_MARKER)
                if count > target.baseline_markers:
                    stranded.append(
                        f"{target.label()} carries {count} {MUTANT_MARKER!r} "
                        f"marker(s), expected {target.baseline_markers}"
                    )
        if stranded:
            print("REFUSING TO RUN: a matrix run from a poisoned baseline is not")
            print("a measurement. Restore these before scoring anything:")
            for item in stranded:
                print(f"  - {item}")
            return 2

        self._install_signal_restore(originals)

        print("=" * 78)
        print(self.title)
        print("=" * 78)
        for path, target in self.targets.items():
            print(f"target : {target.label()}  sha256 {shas[path][:16]}...")
        print()

        controls: dict[Path, RunResult] = {}
        for path, target in self.targets.items():
            res = target.runner()
            controls[path] = res
            state = "GREEN" if res.rc == 0 else "RED"
            print(
                f"control {target.label():<28} rc={res.rc} "
                f"collected={res.collected} {state}"
            )
            if res.rc != 0:
                print(f"  CONTROL IS NOT GREEN -- every cell on this target is UNSCORABLE.")
                print("  " + res.output[-800:].replace("\n", "\n  "))
        print()

        for cell in self.cells:
            path = Path(cell.target).resolve()
            target = self.targets[path]
            if controls[path].rc != 0:
                self._record(cell.name, "UNSCORABLE", "control was not green")
                continue

            text = originals[path].decode()
            occurrences = text.count(cell.old)
            if occurrences != 1:
                self._record(
                    cell.name,
                    "UNSCORABLE",
                    f"anchor appears {occurrences}x in {target.label()}, expected 1",
                )
                continue

            path.write_text(text.replace(cell.old, cell.new, 1))
            try:
                mutant = target.runner()
            finally:
                path.write_bytes(originals[path])
                restored = hashlib.sha256(path.read_bytes()).hexdigest()
                if restored != shas[path]:
                    raise RuntimeError(
                        f"RESTORE FAILED on {target.label()} -- the tree is dirty"
                    )
            self._score(cell, target, controls[path], mutant)

        for name, why in self.equivalent_by_design:
            print(f"[EQUIVALENT-BY-DESIGN] {name}")
            for i in range(0, len(why), 72):
                print(f"           {why[i:i + 72]}")

        return self._summarise(shas)

    # -- scoring -----------------------------------------------------------

    def _score(
        self, cell: Cell, target: Target, control: RunResult, mutant: RunResult
    ) -> None:
        if mutant.rc == 0:
            self._record(cell.name, "SURVIVED", f"{target.label()} stayed green")
            return
        if not target.score_rc(mutant.rc):
            self._record(
                cell.name,
                "UNSCORABLE",
                f"exited {mutant.rc}, which this target does not accept as a "
                f"kill (pytest exit 5 is 'no tests collected' and must never "
                f"score as a success)",
            )
            return
        if (
            target.compare_collected
            and control.collected >= 0
            and mutant.collected != control.collected
        ):
            self._record(
                cell.name,
                "UNSCORABLE",
                f"collected {mutant.collected} vs {control.collected} in "
                f"control -- the mutant changed WHAT RUNS, so the comparison is "
                f"not like-for-like",
            )
            return
        if cell.named not in mutant.output:
            self._record(
                cell.name,
                "UNSCORABLE",
                f"went red but `{cell.named}` is not among the failures -- "
                f"something else broke and this guard was never shown to fire",
            )
            return
        self._record(
            cell.name,
            "KILLED",
            f"`{cell.named}` failed (rc={mutant.rc}, collected={mutant.collected})",
        )

    def _record(self, name: str, verdict: str, detail: str) -> None:
        self.results.append(CellResult(name, verdict, detail))
        print(f"[{verdict}] {name}\n           {detail}")

    def _summarise(self, shas: dict[Path, str]) -> int:
        print()
        print("-" * 78)
        killed = [r for r in self.results if r.verdict == "KILLED"]
        survived = [r for r in self.results if r.verdict == "SURVIVED"]
        unscorable = [r for r in self.results if r.verdict == "UNSCORABLE"]
        print(
            f"KILLED {len(killed)} / {len(self.results)}   "
            f"SURVIVED {len(survived)}   UNSCORABLE {len(unscorable)}   "
            f"EQUIVALENT-BY-DESIGN {len(self.equivalent_by_design)} (not scored)"
        )
        for r in survived:
            print(f"  SURVIVED: {r.name}")
        for r in unscorable:
            print(f"  UNSCORABLE: {r.name} -- {r.detail}")
        for path in self.targets:
            now = hashlib.sha256(path.read_bytes()).hexdigest()
            state = "verified" if now == shas[path] else "DIRTY"
            print(f"restore {state}: {path.name} {now[:16]}...")
        # A matrix with NO cells has nothing to say and must not report success.
        if not self.results:
            print("NO CELLS RAN -- an empty matrix is not a passing matrix.")
            return 1
        return 0 if not survived and not unscorable else 1


if __name__ == "__main__":  # pragma: no cover
    print(__doc__)
    print("This is a library. Its red-first proof is:")
    print("  python scripts/qa/guardlib_selftest.py")
