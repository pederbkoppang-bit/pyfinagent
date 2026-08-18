#!/usr/bin/env python3
"""RED-FIRST proof for :mod:`guardlib` -- watch every guard in it fail.

A guard whose red state was never observed is decoration that reads like
evidence. That rule applies to the library that enforces the rule, so this file
exists to make :mod:`guardlib` earn its own keep: **every case below is a
known-bad input, and the case passes only when guardlib REJECTS it.**

    python scripts/qa/guardlib_selftest.py           # observe the red states
    python scripts/qa/guardlib_selftest.py --mutate  # guardlib mutates itself

The outer harness here is deliberately dumb -- ``expect_raises`` /
``expect_value`` and nothing else. It does NOT use :class:`guardlib.Guards`,
because a library that certifies itself through its own machinery proves only
that the machinery is self-consistent. The fixtures are the evidence.

``--mutate`` closes the last gap: it disables each check inside guardlib.py one
at a time, using guardlib's OWN :class:`~guardlib.MutationMatrix`, and requires
the NAMED case below to go red. A check that survives its own removal was never
doing anything.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardlib import (  # noqa: E402
    Cell,
    CounterexampleError,
    GuardFailed,
    Guards,
    MutationMatrix,
    RunResult,
    Target,
    VacuousGuard,
    census,
    script_runner,
)

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def _pass(case: str, note: str) -> None:
    PASSED.append(case)
    print(f"[ok]   {case}\n         {note}")


def _fail(case: str, note: str) -> None:
    FAILED.append((case, note))
    print(f"[FAIL] {case}\n         {note}")


def expect_raises(case: str, exc, fn, *, containing: str = "") -> None:
    """The case passes only when guardlib refuses the known-bad input."""
    try:
        fn()
    except exc as err:
        message = str(err)
        if containing and containing not in message:
            _fail(case, f"raised {exc.__name__} but not about {containing!r}: {message[:160]}")
        else:
            _pass(case, f"rejected with {exc.__name__}: {message[:130]}")
    except Exception as err:  # noqa: BLE001
        _fail(case, f"raised {type(err).__name__}, expected {exc.__name__}: {err}")
    else:
        _fail(case, f"did NOT raise -- guardlib ACCEPTED a known-bad guard")


def expect_ok(case: str, fn, note: str = "") -> None:
    try:
        fn()
    except Exception as err:  # noqa: BLE001
        _fail(case, f"a legitimate guard was rejected: {type(err).__name__}: {err}")
    else:
        _pass(case, note or "accepted, as it should be")


def expect_value(case: str, actual, expected, note: str = "") -> None:
    if actual == expected:
        _pass(case, note or f"{actual!r}")
    else:
        _fail(case, f"expected {expected!r}, got {actual!r}. {note}")


# ===========================================================================
# PART 1 -- Guards.ok() must refuse a guard that cannot fail
# ===========================================================================

# The "right check on the WRONG variable" fixture: this module-level value is
# what the broken predicate reads instead of its own argument.
MEASURED_ELSEWHERE = {"n": 5}


def part1_guards() -> None:
    print("\n--- PART 1: Guards.ok() rejects guards with no red state ---\n")

    expect_ok(
        "A-genuine-guard-is-accepted",
        lambda: Guards().ok("n_is_positive", lambda n: n > 0, 7, falsified_by=0),
        "a real predicate with a real counterexample passes both halves",
    )

    expect_raises(
        "B-always-true-predicate",
        VacuousGuard,
        lambda: Guards().ok("always_true", lambda n: True, 7, falsified_by=0),
        containing="supposed to be REJECTED",
    )

    expect_raises(
        "C-predicate-ignores-its-argument",
        VacuousGuard,
        lambda: Guards().ok(
            "reads_the_wrong_variable",
            lambda d: MEASURED_ELSEWHERE["n"] > 0,  # noqa: ARG005 -- the defect
            {"n": 5},
            falsified_by={"n": 0},
        ),
        containing="supposed to be REJECTED",
    )

    expect_raises(
        "D-truthy-non-bool-is-not-an-answer",
        VacuousGuard,
        lambda: Guards().ok(
            "returns_a_list", lambda xs: [x for x in xs if x], [1, 2], falsified_by=[]
        ),
        containing="not a bool",
    )

    expect_raises(
        "E-empty-fixture-list",
        VacuousGuard,
        lambda: Guards().ok("no_fixture", lambda n: n > 0, 7, falsified_by_each=[]),
        containing="EMPTY",
    )

    expect_raises(
        "F-precomputed-boolean-instead-of-predicate",
        VacuousGuard,
        lambda: Guards().ok("precomputed", 5 > 0, 7, falsified_by=0),
        containing="CALLABLE",
    )

    expect_raises(
        "G-repeated-counterexample",
        VacuousGuard,
        lambda: Guards().ok("dup_fixture", lambda n: n > 0, 7, falsified_by_each=[0, 0]),
        containing="twice",
    )

    expect_raises(
        "H-compound-guard-with-too-few-fixtures",
        VacuousGuard,
        lambda: Guards().ok(
            "both_positive",
            lambda d: d["a"] > 0 and d["b"] > 0,
            {"a": 1, "b": 1},
            falsified_by={"a": 0, "b": 1},
        ),
        containing="licenses NOTHING",
    )

    expect_raises(
        "I-both-fixtures-falsify-the-SAME-clause",
        VacuousGuard,
        lambda: Guards().ok(
            "both_positive",
            lambda d: d["a"] > 0 and d["b"] > 0,
            {"a": 1, "b": 1},
            falsified_by_each=[{"a": 0, "b": 1}, {"a": -1, "b": 1}],
        ),
        containing="no counterexample falsifies",
    )

    expect_ok(
        "J-compound-guard-with-one-fixture-per-clause",
        lambda: Guards().ok(
            "both_positive",
            lambda d: d["a"] > 0 and d["b"] > 0,
            {"a": 1, "b": 1},
            falsified_by_each=[{"a": 0, "b": 1}, {"a": 1, "b": 0}],
        ),
        "each and-clause has a fixture that falsifies it individually",
    )

    expect_raises(
        "K-fixture-of-the-wrong-shape",
        CounterexampleError,
        lambda: Guards().ok("needs_a_dict", lambda d: d["n"] > 0, {"n": 1}, falsified_by=None),
        containing="wrong shape",
    )

    expect_raises(
        "L-real-failure-is-GuardFailed-not-VacuousGuard",
        GuardFailed,
        lambda: Guards().ok("n_is_positive", lambda n: n > 0, 0, falsified_by=-1),
        containing="INVARIANT FAILED",
    )

    def _dup() -> None:
        g = Guards()
        g.ok("same_name", lambda n: n > 0, 1, falsified_by=0)
        g.ok("same_name", lambda n: n < 9, 1, falsified_by=99)

    expect_raises("M-duplicate-guard-name", VacuousGuard, _dup, containing="registered twice")

    expect_raises(
        "N-or-clause-that-is-always-true",
        VacuousGuard,
        lambda: Guards().ok(
            "or_always_true", lambda n: n > 0 or True, 7, falsified_by=-1
        ),
        containing="supposed to be REJECTED",
    )

    expect_raises(
        "O1-neither-fixture-parameter-supplied",
        VacuousGuard,
        lambda: Guards().ok("no_kwarg", lambda n: n > 0, 7),
        containing="no known-bad fixture",
    )

    expect_raises(
        "O2-both-fixture-parameters-supplied",
        VacuousGuard,
        lambda: Guards().ok(
            "both_kwargs", lambda n: n > 0, 7, falsified_by=0, falsified_by_each=[-1]
        ),
        containing="not both",
    )

    # A list-valued counterexample must survive as ONE fixture. This is the case
    # that exposed the original single-parameter ambiguity: `falsified_by=[]`
    # was silently read as "zero fixtures" instead of "the empty list is the
    # known-bad value", so a guard about emptiness could not be written at all.
    expect_ok(
        "O3-list-valued-fixture-is-not-unpacked",
        lambda: Guards().ok(
            "nonempty_list", lambda xs: len(xs) > 0, [1, 2], falsified_by=[]
        ),
        "the empty list arrives as ONE fixture and correctly falsifies the "
        "predicate; under the old single-parameter API this same call raised "
        "'falsified_by is EMPTY', which is the ambiguity being removed",
    )

    # The summary must not overstate: a closure-bound predicate cannot have its
    # clauses isolated, and that has to be VISIBLE rather than assumed enforced.
    def _closure_case() -> str:
        limit = 3
        g = Guards()
        g.ok(
            "closure_bound",
            lambda d: d["a"] > limit and d["b"] > limit,
            {"a": 9, "b": 9},
            falsified_by_each=[{"a": 0, "b": 9}, {"a": 9, "b": 0}],
        )
        return g.records[0].clause_census

    state = _closure_case()
    if state.startswith(("PARTIAL", "ENFORCED")):
        _pass("O-closure-degradation-is-reported", f"clause_census={state}")
    else:
        _fail("O-closure-degradation-is-reported", f"clause_census={state!r}")


# ===========================================================================
# PART 2 -- census() must find a guard that has no mutation cell
# ===========================================================================

GUARD_SRC = '''\
"""A fixture module that registers two guards."""
from guardlib import Guards

g = Guards()
g.ok("alpha_is_positive", lambda n: n > 0, 1, falsified_by=-1)
g.ok("beta_is_bounded", lambda n: n < 10, 1, falsified_by=99)
'''

# The docstring names beta_is_bounded. A grep would call it covered. The census
# is an AST walk over DATA positions only, so it must still report it uncelled.
#
# `_note` carries a docstring that is EXACTLY the guard name, and that is the
# load-bearing part of this fixture. `cell_strings` is an exact set-membership
# check, so prose containing the name as a substring cannot match under EITHER
# the correct code or the mutation -- a fixture with only the prose mention
# gives the same answer both ways and proves nothing. The mutation matrix
# caught exactly that (cell G8 SURVIVED) before this line existed.
CELL_SRC_PARTIAL = '''\
"""Mutation cells. This docstring mentions beta_is_bounded on purpose."""

CELLS = [
    ("M1 alpha is disarmed", "old", "new", "alpha_is_positive"),
]


def _note():
    """beta_is_bounded"""
'''

CELL_SRC_FULL = '''\
"""Mutation cells."""

CELLS = [
    ("M1 alpha is disarmed", "old", "new", "alpha_is_positive"),
    ("M2 beta is disarmed", "old", "new", "beta_is_bounded"),
]
'''

GUARD_SRC_DYNAMIC = '''\
from guardlib import Guards

NAME = "computed_at_runtime"
g = Guards()
g.ok(NAME, lambda n: n > 0, 1, falsified_by=-1)
'''


def part2_census(tmp: Path) -> None:
    print("\n--- PART 2: census() finds guards with no mutation cell ---\n")

    guard_file = tmp / "fixture_guards.py"
    partial = tmp / "fixture_cells_partial.py"
    full = tmp / "fixture_cells_full.py"
    empty = tmp / "fixture_empty.py"
    dynamic = tmp / "fixture_dynamic.py"
    guard_file.write_text(GUARD_SRC)
    partial.write_text(CELL_SRC_PARTIAL)
    full.write_text(CELL_SRC_FULL)
    empty.write_text("# no guards here at all\n")
    dynamic.write_text(GUARD_SRC_DYNAMIC)

    res = census([guard_file], [partial])
    expect_value(
        "P-uncelled-guard-is-reported",
        [u.split(" (")[0] for u in res.uncelled],
        ["beta_is_bounded"],
        "alpha has a cell, beta does not",
    )
    # The discriminating half: a grep WOULD have passed this file, and so would
    # a census that collected docstrings -- `_note.__doc__` is the guard name
    # verbatim. Only a census restricted to DATA positions reports it uncelled.
    grep_would_match = "beta_is_bounded" in partial.read_text()
    if grep_would_match and any(u.startswith("beta_is_bounded") for u in res.uncelled):
        _pass(
            "Q-docstring-prose-does-not-count-as-a-cell",
            "the name appears twice in the file (prose, and a docstring that is "
            "exactly the name) and the AST census still reports it uncelled -- "
            "grep says covered, the census says not, so the answers differ",
        )
    else:
        _fail(
            "Q-docstring-prose-does-not-count-as-a-cell",
            f"grep_would_match={grep_would_match} uncelled={res.uncelled}",
        )

    expect_value("R-census-ok-flag-is-false-when-uncelled", res.ok, False)

    full_res = census([guard_file], [full])
    expect_value(
        "S-fully-celled-census-passes",
        (full_res.ok, sorted(full_res.guards)),
        (True, ["alpha_is_positive", "beta_is_bounded"]),
    )

    blank = census([empty], [full])
    if any("not evidence of coverage" in p for p in blank.problems) and not blank.ok:
        _pass("T-empty-census-is-not-a-pass", "positive control fires on zero guards found")
    else:
        _fail("T-empty-census-is-not-a-pass", f"problems={blank.problems} ok={blank.ok}")

    no_cells = census([guard_file], [empty])
    if any("other side" in p for p in no_cells.problems):
        _pass("U-empty-cell-source-is-not-a-pass", "positive control fires on zero cells found")
    else:
        _fail("U-empty-cell-source-is-not-a-pass", f"problems={no_cells.problems}")

    dyn = census([dynamic], [full])
    if dyn.dynamic_names and not dyn.ok:
        _pass("V-non-literal-guard-name-is-flagged", dyn.dynamic_names[0][-90:])
    else:
        _fail("V-non-literal-guard-name-is-flagged", f"dynamic_names={dyn.dynamic_names}")

    # -- parameterised guard names -----------------------------------------
    # A guard named f"window_len_{d}" has a different name every iteration, so
    # no exact match can cover it. Real evidence scripts do this (86.59 has
    # three), and a census that calls them all uncoverable is wrong about a
    # matrix that DOES name their prefix.
    param = tmp / "fixture_param.py"
    param.write_text(
        "from guardlib import Guards\n"
        "g = Guards()\n"
        "for d in ('a', 'b'):\n"
        "    g.ok(f'window_len_{d}', lambda n: n > 0, 1, falsified_by=-1)\n"
        "    g.ok(f'{d}_leading_interp', lambda n: n > 0, 1, falsified_by=-1)\n"
    )
    param_cells = tmp / "fixture_param_cells.py"
    param_cells.write_text('CELLS = [("M1", "old", "new", "window_len_")]\n')

    pres = census([param], [param_cells])
    if pres.prefix_matches.get("window_len_") == "window_len_":
        _pass(
            "X1-parameterised-guard-covered-by-its-prefix",
            "f'window_len_{d}' matched the cell string 'window_len_', and the "
            "match is recorded so a reader can audit it",
        )
    else:
        _fail("X1-parameterised-guard-covered-by-its-prefix", f"{pres.prefix_matches}")

    # The fixture has exactly two f-string call sites: one with a usable literal
    # prefix and one that starts with an interpolation. Exactly one must be
    # reported unattributable, and it must say the prefix was empty.
    if len(pres.dynamic_names) == 1 and "prefix ''" in pres.dynamic_names[0]:
        _pass(
            "X2-name-starting-with-an-interpolation-stays-unattributable",
            "f'{d}_leading_interp' yields an empty literal prefix, so the census "
            "reports it rather than guessing: " + pres.dynamic_names[0][-70:],
        )
    else:
        _fail(
            "X2-name-starting-with-an-interpolation-stays-unattributable",
            f"dynamic_names={pres.dynamic_names}",
        )

    no_prefix_cell = tmp / "fixture_param_nocell.py"
    no_prefix_cell.write_text('CELLS = [("M1", "old", "new", "something_else")]\n')
    miss = census([param], [no_prefix_cell])
    if any(u.startswith("window_len_*") for u in miss.uncelled):
        _pass(
            "X3-parameterised-guard-with-no-matching-cell-is-uncelled",
            "prefix coverage is a match, not a free pass",
        )
    else:
        _fail("X3-parameterised-guard-with-no-matching-cell-is-uncelled", f"{miss.uncelled}")

    never = census([guard_file], [full], runtime_names=["alpha_is_positive"])
    expect_value(
        "W-guard-in-source-but-never-executed",
        never.never_executed,
        ["beta_is_bounded"],
        "the 'loop over an empty glob' class: real assertion, never runs",
    )


# ===========================================================================
# PART 3 -- MutationMatrix scoring and the restore contract
# ===========================================================================

PROBE_GREEN = '''\
VALUE = 1
if VALUE != 1:
    print("[probe] probe_value_check failed")
    raise SystemExit(1)
print("[probe] ok")
'''

PROBE_RED = '''\
print("[probe] this control is red on purpose")
raise SystemExit(1)
'''

PROBE_EXIT5 = '''\
VALUE = 1
if VALUE != 1:
    print("[probe] probe_value_check failed")
raise SystemExit(5 if VALUE != 1 else 0)
'''

PROBE_SILENT = '''\
VALUE = 1
if VALUE != 1:
    raise SystemExit(1)
print("[probe] ok")
'''


def part3_matrix(tmp: Path) -> None:
    print("\n--- PART 3: MutationMatrix scoring + restore ---\n")

    def probe(name: str, body: str) -> Path:
        p = tmp / name
        p.write_text(body)
        return p

    # -- a genuine kill ----------------------------------------------------
    green = probe("probe_green.py", PROBE_GREEN)
    before = green.read_bytes()
    matrix = MutationMatrix(
        [Target(green, script_runner([str(green)], tmp), name="probe_green")],
        [
            Cell(
                "X1 the checked constant is changed",
                green,
                "VALUE = 1",
                "VALUE = 2  # MUTANT",
                "probe_value_check failed",
            )
        ],
        title="selftest: genuine kill",
    )
    rc = matrix.run()
    expect_value(
        "X-genuine-mutation-is-KILLED",
        [(r.verdict) for r in matrix.results],
        ["KILLED"],
        f"matrix rc={rc}",
    )
    expect_value(
        "Y-target-is-restored-byte-for-byte",
        green.read_bytes(),
        before,
        "the file on disk is identical after the matrix ran",
    )

    # -- exit 5 must never score as a kill ---------------------------------
    five = probe("probe_five.py", PROBE_EXIT5)
    m5 = MutationMatrix(
        [Target(five, script_runner([str(five)], tmp), name="probe_five")],
        [
            Cell(
                "X2 mutation makes the probe exit 5, not 1",
                five,
                "VALUE = 1",
                "VALUE = 2  # MUTANT",
                "probe_value_check failed",
            )
        ],
        title="selftest: exit-5 is not a kill",
    )
    m5.run()
    expect_value(
        "Z-exit-5-scores-UNSCORABLE-not-KILLED",
        [r.verdict for r in m5.results],
        ["UNSCORABLE"],
        "pytest exits 5 on 'no tests collected'; a typo'd -k must not read as success",
    )

    # -- red control -------------------------------------------------------
    red = probe("probe_red.py", PROBE_RED)
    mred = MutationMatrix(
        [Target(red, script_runner([str(red)], tmp), name="probe_red")],
        [Cell("X3 anything at all", red, "on purpose", "on purpose  # MUTANT", "whatever")],
        title="selftest: red control",
    )
    mred.run()
    expect_value(
        "AA-red-control-makes-cells-UNSCORABLE",
        [r.verdict for r in mred.results],
        ["UNSCORABLE"],
    )

    # -- the named test must actually be among the failures ----------------
    silent = probe("probe_silent.py", PROBE_SILENT)
    msil = MutationMatrix(
        [Target(silent, script_runner([str(silent)], tmp), name="probe_silent")],
        [
            Cell(
                "X4 goes red without naming the guard",
                silent,
                "VALUE = 1",
                "VALUE = 2  # MUTANT",
                "probe_value_check failed",
            )
        ],
        title="selftest: something-else-broke",
    )
    msil.run()
    expect_value(
        "AB-red-without-the-NAMED-guard-is-UNSCORABLE",
        [r.verdict for r in msil.results],
        ["UNSCORABLE"],
        "'something went red' is not evidence that THIS guard fired",
    )

    # -- collected-count parity -------------------------------------------
    calls = {"n": 0}

    def uneven_runner() -> RunResult:
        calls["n"] += 1
        if calls["n"] == 1:
            return RunResult(0, 4, "4 passed")
        return RunResult(1, 3, "probe_value_check failed\n3 passed 1 failed")

    parity = probe("probe_parity.py", PROBE_GREEN)
    mpar = MutationMatrix(
        [Target(parity, uneven_runner, name="probe_parity")],
        [
            Cell(
                "X5 mutant collects fewer tests than the control",
                parity,
                "VALUE = 1",
                "VALUE = 2  # MUTANT",
                "probe_value_check failed",
            )
        ],
        title="selftest: collected parity",
    )
    mpar.run()
    expect_value(
        "AC-collected-count-mismatch-is-UNSCORABLE",
        [r.verdict for r in mpar.results],
        ["UNSCORABLE"],
        "a mutant that changes WHAT RUNS is not a like-for-like comparison",
    )

    # -- non-unique anchor -------------------------------------------------
    anchor = probe("probe_anchor.py", PROBE_GREEN)
    manc = MutationMatrix(
        [Target(anchor, script_runner([str(anchor)], tmp), name="probe_anchor")],
        [Cell("X6 ambiguous anchor", anchor, "VALUE", "VALUE2  # MUTANT", "probe_value_check")],
        title="selftest: ambiguous anchor",
    )
    manc.run()
    expect_value(
        "AD-non-unique-anchor-is-UNSCORABLE",
        [r.verdict for r in manc.results],
        ["UNSCORABLE"],
    )

    # -- stranded MUTANT marker -------------------------------------------
    poisoned = probe("probe_poisoned.py", PROBE_GREEN + "\n# MUTANT left behind\n")
    mpois = MutationMatrix(
        [Target(poisoned, script_runner([str(poisoned)], tmp), name="probe_poisoned")],
        [Cell("X7 never reached", poisoned, "VALUE = 1", "VALUE = 2  # MUTANT", "x")],
        title="selftest: poisoned baseline",
    )
    expect_value(
        "AE-stranded-MUTANT-marker-refuses-to-run",
        (mpois.run(), len(mpois.results)),
        (2, 0),
        "a matrix run from a poisoned baseline is not a measurement",
    )

    # -- an empty matrix is not a passing matrix ---------------------------
    empty_t = probe("probe_empty.py", PROBE_GREEN)
    mempty = MutationMatrix(
        [Target(empty_t, script_runner([str(empty_t)], tmp), name="probe_empty")],
        [],
        title="selftest: empty matrix",
    )
    expect_value("AF-empty-matrix-returns-nonzero", mempty.run(), 1)

    # -- a cell whose target was never declared ----------------------------
    # Everything else about X8 is well-formed (it carries the marker) so the
    # case isolates the property it claims to test rather than tripping on a
    # second defect and calling that a pass.
    undeclared = probe("probe_undeclared.py", PROBE_GREEN)
    expect_raises(
        "AG-undeclared-target-is-refused",
        ValueError,
        lambda: MutationMatrix(
            [Target(empty_t, script_runner([str(empty_t)], tmp))],
            [Cell("X8", undeclared, "a", "b  # MUTANT", "c")],
        ),
        containing="never be restored on a signal",
    )

    # -- a replacement text already on disk, with the broad net DISABLED ---
    # baseline_markers=None turns off the pattern scan, so this case can only
    # pass via the exact check -- otherwise it would be indistinguishable from
    # AE and would certify a mechanism it never exercised.
    already = probe("probe_already.py", PROBE_GREEN.replace("VALUE = 1", "VALUE = 2  # MUTANT"))
    malready = MutationMatrix(
        [
            Target(
                already,
                script_runner([str(already)], tmp),
                name="probe_already",
                baseline_markers=None,
            )
        ],
        [Cell("X10 already applied", already, "VALUE = 1", "VALUE = 2  # MUTANT", "x")],
        title="selftest: replacement already on disk",
    )
    expect_value(
        "AJ-replacement-already-on-disk-is-refused",
        (malready.run(), len(malready.results)),
        (2, 0),
        "the exact check fires with the marker scan switched off",
    )

    # -- an unmarked mutation could strand undetected ----------------------
    expect_raises(
        "AI-unmarked-mutation-is-refused",
        ValueError,
        lambda: MutationMatrix(
            [Target(empty_t, script_runner([str(empty_t)], tmp))],
            [Cell("X9 no marker", empty_t, "VALUE = 1", "VALUE = 2", "x")],
        ),
        containing="marker",
    )


# ===========================================================================
# PART 4 -- the signal path, driven for real
# ===========================================================================

SIGNAL_DRIVER = '''\
import sys, time
from pathlib import Path
sys.path.insert(0, {qa!r})
from guardlib import Cell, MutationMatrix, RunResult, Target

target = Path({target!r})

def runner():
    # Green instantly on the control; hang once the mutant is on disk, so the
    # parent can signal us mid-cell -- exactly the 2-minute-timeout shape that
    # stranded a mutant during step 86.59.
    if "MUTANT" in target.read_text():
        time.sleep(120)
    return RunResult(0, -1, "control ok")

MutationMatrix(
    [Target(target, runner, name="slow")],
    [Cell("S1 hang under mutation", target, "VALUE = 1", "VALUE = 2  # MUTANT", "x")],
    title="selftest: signal restore",
).run()
'''


def part4_signal(tmp: Path) -> None:
    print("\n--- PART 4: SIGTERM restores every target (driven for real) ---\n")
    target = tmp / "probe_signal.py"
    original = "VALUE = 1\nprint('ok')\n"
    target.write_text(original)
    driver = tmp / "signal_driver.py"
    driver.write_text(
        SIGNAL_DRIVER.format(qa=str(Path(__file__).resolve().parent), target=str(target))
    )

    proc = subprocess.Popen(
        [sys.executable, str(driver)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(tmp),
    )
    deadline = time.time() + 30
    saw_mutant = False
    while time.time() < deadline:
        if "MUTANT" in target.read_text():
            saw_mutant = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.05)

    if not saw_mutant:
        proc.kill()
        proc.wait(timeout=10)
        _fail(
            "AH-SIGTERM-restores-the-target",
            "the mutant never appeared on disk within 30s, so the signal path "
            "was never actually exercised -- reporting UNSCORABLE rather than a "
            "pass, because a check that never reached its subject is not green",
        )
        return

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
        _fail("AH-SIGTERM-restores-the-target", "the driver ignored SIGTERM")
        return

    after = target.read_text()
    if after == original and "MUTANT" not in after:
        _pass(
            "AH-SIGTERM-restores-the-target",
            f"mutant was observed on disk, SIGTERM sent, file restored "
            f"byte-for-byte (driver rc={proc.returncode})",
        )
    else:
        _fail(
            "AH-SIGTERM-restores-the-target",
            f"target left as {after!r} -- a mutant is stranded on disk",
        )


# ===========================================================================
# --mutate : guardlib mutation-tests itself with its own matrix
# ===========================================================================

# (cell, anchor, replacement, the selftest case that MUST go red)
SELF_CELLS: list[tuple[str, str, str, str]] = [
    (
        "G1 the RED half is skipped -- ok() stops proving the counterexample fails",
        "            if _strict_bool(result, f\"{name} (on falsified_by[{i}])\"):",
        "            if False:  # MUTANT",
        "[FAIL] B-always-true-predicate",
    ),
    (
        "G2 truthy values are accepted as answers",
        "    raise VacuousGuard(\n        f\"{where}: predicate returned {value!r} of type \"",
        "    return bool(value)  # MUTANT\n    raise VacuousGuard(\n        f\"{where}: predicate returned {value!r} of type \"",
        "[FAIL] D-truthy-non-bool-is-not-an-answer",
    ),
    (
        "G3 an empty fixture list is allowed through",
        "        if not fixtures:",
        "        if False:  # MUTANT",
        "[FAIL] E-empty-fixture-list",
    ),
    (
        "G4 a precomputed boolean is accepted instead of a predicate",
        "        if not callable(predicate):",
        "        if False:  # MUTANT",
        "[FAIL] F-precomputed-boolean-instead-of-predicate",
    ),
    (
        "G5 the per-clause fixture count is not enforced",
        "        if len(fixtures) < len(clauses):",
        "        if False:  # MUTANT",
        "[FAIL] H-compound-guard-with-too-few-fixtures",
    ),
    (
        "G6 clauses are no longer individually falsified",
        "        if unfalsified:",
        "        if False:  # MUTANT",
        "[FAIL] I-both-fixtures-falsify-the-SAME-clause",
    ),
    (
        "G7 duplicate counterexamples are allowed",
        "                raise VacuousGuard(\n                    f\"{name}: falsified_by contains {bad!r} twice.",
        "                pass  # MUTANT\n            if False:\n                raise VacuousGuard(\n                    f\"{name}: falsified_by contains {bad!r} twice.",
        "[FAIL] G-repeated-counterexample",
    ),
    (
        "G8 the census collects docstrings too -- prose starts counting as a cell",
        "    for node in ast.walk(tree):\n        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):",
        "    for node in ast.walk(tree):\n        if isinstance(node, ast.Expr):  # MUTANT\n            add(node.value)\n        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):",
        "[FAIL] Q-docstring-prose-does-not-count-as-a-cell",
    ),
    (
        "G9 the empty-census positive control is removed",
        "    if not result.guards and not result.prefix_guards:",
        "    if False:  # MUTANT",
        "[FAIL] T-empty-census-is-not-a-pass",
    ),
    (
        "G10 any non-zero exit scores as a kill again",
        "        if not target.score_rc(mutant.rc):",
        "        if False:  # MUTANT",
        "[FAIL] Z-exit-5-scores-UNSCORABLE-not-KILLED",
    ),
    (
        "G11 the collected-count parity check is dropped",
        "            and mutant.collected != control.collected",
        "            and False  # MUTANT",
        "[FAIL] AC-collected-count-mismatch-is-UNSCORABLE",
    ),
    (
        "G12 'something went red' counts as the named guard firing",
        "        if cell.named not in mutant.output:",
        "        if False:  # MUTANT",
        "[FAIL] AB-red-without-the-NAMED-guard-is-UNSCORABLE",
    ),
    (
        "G13 a poisoned baseline is no longer refused",
        "        if stranded:",
        "        if False:  # MUTANT",
        "[FAIL] AE-stranded-MUTANT-marker-refuses-to-run",
    ),
    (
        "G14 an empty matrix reports success",
        "        if not self.results:",
        "        if False:  # MUTANT",
        "[FAIL] AF-empty-matrix-returns-nonzero",
    ),
    (
        "G15 the signal handlers are never installed -- a mutant strands on disk",
        "        self._install_signal_restore(originals)",
        "        pass  # MUTANT",
        "[FAIL] AH-SIGTERM-restores-the-target",
    ),
    (
        "G16 unmarked mutations are allowed -- stranded-detection loses its teeth",
        "            if MUTANT_MARKER not in cell.new:",
        "            if False:  # MUTANT",
        "[FAIL] AI-unmarked-mutation-is-refused",
    ),
    (
        "G21 a name starting with an interpolation is silently given a prefix",
        "        for value in node.values:\n            if isinstance(value, ast.Constant) and isinstance(value.value, str):\n                parts.append(value.value)\n            else:\n                break",
        "        for value in node.values:  # MUTANT\n            if isinstance(value, ast.Constant) and isinstance(value.value, str):\n                parts.append(value.value)",
        "[FAIL] X2-name-starting-with-an-interpolation-stays-unattributable",
    ),
    (
        "G19 parameterised guards lose their prefix coverage",
        "            if len(prefix) >= MIN_CENSUS_PREFIX:",
        "            if False:  # MUTANT",
        "[FAIL] X1-parameterised-guard-covered-by-its-prefix",
    ),
    (
        "G20 any cell string covers any prefix -- coverage becomes a coincidence",
        "        matched = next(\n            (s for s in sorted(result.cell_strings) if s.startswith(prefix)), None\n        )",
        "        matched = next(iter(sorted(result.cell_strings)), None)  # MUTANT",
        "[FAIL] X3-parameterised-guard-with-no-matching-cell-is-uncelled",
    ),
    (
        "G17 the exact stranded-replacement check is disarmed",
        "                if Path(cell.target).resolve() == path and cell.new in text:",
        "                if False:  # MUTANT",
        "[FAIL] AJ-replacement-already-on-disk-is-refused",
    ),
    (
        "G18 the broad marker-count net is disarmed",
        "                if count > target.baseline_markers:",
        "                if False:  # MUTANT",
        "[FAIL] AE-stranded-MUTANT-marker-refuses-to-run",
    ),
]


def run_mutate() -> int:
    guardlib_py = Path(__file__).resolve().parent / "guardlib.py"
    selftest_py = Path(__file__).resolve()
    matrix = MutationMatrix(
        [
            Target(
                guardlib_py,
                script_runner([str(selftest_py)], REPO),
                name="guardlib.py",
                compare_collected=False,
                # guardlib.py declares and documents the marker, so the broad
                # pattern net would fire on its own source. The exact
                # replacement-text check still covers every cell below.
                baseline_markers=None,
            )
        ],
        [
            Cell(name, guardlib_py, old, new, named)
            for name, old, new, named in SELF_CELLS
        ],
        title="guardlib mutates ITSELF -- every check must go red when removed",
    )
    return matrix.run()


# ===========================================================================


def main() -> int:
    if "--mutate" in sys.argv:
        return run_mutate()

    print("=" * 78)
    print("guardlib RED-FIRST selftest -- every case is a known-bad input")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="guardlib-selftest-") as td:
        tmp = Path(td)
        part1_guards()
        part2_census(tmp)
        part3_matrix(tmp)
        if "--no-signal" not in sys.argv:
            part4_signal(tmp)

    print()
    print("-" * 78)
    print(f"cases passed: {len(PASSED)}   FAILED: {len(FAILED)}")
    for case, note in FAILED:
        print(f"  FAILED {case}: {note}")
    if not PASSED:
        print("NO CASES RAN -- an empty selftest is not a passing selftest.")
        return 1
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
