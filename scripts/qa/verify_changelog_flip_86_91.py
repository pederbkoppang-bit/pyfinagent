#!/usr/bin/env python3
"""phase-86.91 -- the changelog flip detector's THREE-STATE membership test.

WHAT THIS DEFENDS. `.claude/hooks/post-commit-changelog.sh::_flip_magnitude`
decided a version bump from the masterplan `id -> status` diff with::

    newly_done = [sid for sid, st in after.items()
                  if st == "done" and before.get(sid) not in (None, "done")]

`before.get(sid)` is None for a step that DID NOT EXIST at HEAD~1, so the None
exclusion silently dropped every step FILED AND CLOSED IN THE SAME COMMIT -- the
file-it-and-fix-it workflow this project runs on. MEASURED on commit e4f2e844:
`86.86 before: None -> after: done`, `newly_done == []`, no bump, while the
version header sat frozen at v6.93.222 (2026-08-14) and Recent-Activity rows kept
landing.

WHY THIS DRIVES THE REAL HOOK. The detector lives inside a python heredoc inside
a bash script. This checker EXTRACTS that heredoc, parses it with `ast`, and
execs the shipped `_flip_magnitude` / `_ABSENT` / `_FLIP_DECISION` definitions
with `subprocess.run` stubbed. A re-implemented copy would stay green while
production drifted -- which is exactly how the sibling replay harness came to
mirror the same defect at its own line 54.

WHY THE CONTROL RUNS FIRST. ISSTA '24 (arXiv:2408.01760) puts equivalent mutants
at 4-39% of real-world mutants and notes equivalence is undecidable. A cell whose
CONTROL was never observed green is unscorable, not passed.

Read-only: stubs git, writes nothing, spawns nothing, touches no live state.

    python scripts/qa/verify_changelog_flip_86_91.py
    -> exit 0 all green, exit 1 with the failing case named
"""
from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "post-commit-changelog.sh"

PASSED: list[str] = []
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        FAILURES.append(f"{name}{' -- ' + detail if detail else ''}")
        print(f"  FAIL {name}{' -- ' + detail if detail else ''}")


# ---------------------------------------------------------------------------
# Extract the SHIPPED detector from the hook's heredoc.
# ---------------------------------------------------------------------------
def heredoc_python(source: str) -> str:
    start = source.index("<< 'PYEOF'")
    start = source.index("\n", start) + 1
    end = source.index("\nPYEOF", start)
    return source[start:end]


# CYCLE-4 (Q/A finding Q1). `_log_decision` was NOT in this tuple, so it was never
# extracted, never exec'd and never driven -- and every [2] assertion read the
# in-memory _FLIP_DECISION dict that FEEDS the file rather than the file itself.
# The Q/A deleted the entire write and this checker stayed ALL GREEN 34/0.
# Criterion 4 names the hook's OWN OUTPUT as the mechanism, so the output is what
# has to be asserted.
NEEDED = ("_ABSENT", "_FLIP_DECISION", "_flip_magnitude", "_log_decision")


def detector_source(py: str) -> str:
    """The shipped definitions of the three names, sliced from the real file."""
    tree = ast.parse(py)
    lines = py.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
        if any(n in NEEDED for n in names):
            chunks.append("".join(lines[node.lineno - 1:node.end_lineno]))
    return "\n".join(chunks)


def load_detector(src: str):
    """exec the extracted source into a fresh namespace with the globals the
    shipped code closes over."""
    ns: dict = {"json": json, "re": re, "sys": sys, "repo_root": REPO}
    exec(compile(src, "<shipped-detector>", "exec"), ns)  # noqa: S102
    missing = [n for n in NEEDED if n not in ns]
    if missing:
        raise RuntimeError(f"extraction lost {missing} -- the hook's shape changed")
    return ns


def masterplan_blob(statuses: dict[str, str]) -> str:
    return json.dumps({"phases": [{"steps": [{"id": k, "status": v} for k, v in statuses.items()]}]})


def run_detector(ns: dict, before, after, *, explode: bool = False) -> tuple[str, dict, str]:
    """Drive the shipped detector against a synthetic history.

    `before=None` models "masterplan unreadable at HEAD~1" (the first-commit
    branch). `explode=True` makes the git call raise, for the never-raises test.
    Returns (bump, decision_dict_copy, captured_stderr).
    """
    def fake_run(cmd, *a, **kw):
        if explode:
            raise OSError("injected fault: git is unavailable")
        ref = cmd[2].split(":")[0]
        payload = before if ref.endswith("~1") else after
        if payload is None:
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: path does not exist")
        return subprocess.CompletedProcess(cmd, 0, masterplan_blob(payload), "")

    ns["_FLIP_DECISION"].clear()
    err = io.StringIO()
    with mock.patch.object(subprocess, "run", fake_run), redirect_stderr(err):
        bump = ns["_flip_magnitude"]()
    return bump, dict(ns["_FLIP_DECISION"]), err.getvalue()


HOOK_SRC = HOOK.read_text(encoding="utf-8")
SHIPPED = detector_source(heredoc_python(HOOK_SRC))

print("phase-86.91 -- changelog flip detector, three-state membership test\n")
print(f"  (driving the SHIPPED detector, {len(SHIPPED.splitlines())} lines extracted from {HOOK.name})\n")

ns = load_detector(SHIPPED)

# ── [0] CONTROL FIRST ───────────────────────────────────────────────────────
# Every claim below is of the form "this input produces that bump". They are
# worthless if the detector returns "none" for everything, so the two behaviours
# that must have been TRUE BEFORE this step are established first.
print("[0] CONTROL -- behaviour that was already correct must still hold\n")
# The pending sibling is LOAD-BEARING, not decoration: without it every step of
# top-level phase 86 is done and the "whole phase shipped" rule returns major,
# so a fixture that omits it silently tests a different branch. Caught by this
# checker's own first run.
b, d, _ = run_detector(ns, {"86.1": "pending", "86.7": "pending"}, {"86.1": "done", "86.7": "pending"})
check("[0] a NORMAL transition pending->done still bumps", b == "patch", f"got {b!r}")
check("[0] and it is recorded as a transition", d.get("reason") == "flip_transitioned", f"reason={d.get('reason')!r}")

b, d, _ = run_detector(ns, {"86.1": "done"}, {"86.1": "done"})
check("[0] an ALREADY-done step does NOT bump", b == "none", f"got {b!r}")
check("[0] and the 'none' is explained as no_flip", d.get("reason") == "no_flip", f"reason={d.get('reason')!r}")

b, _, _ = run_detector(ns, {"86.1": "pending"}, {"86.1": "pending"})
check("[0] a chore commit that moves nothing does NOT bump", b == "none", f"got {b!r}")

# ── [1] THE DEFECT, and the fix (criteria 1, 2) ─────────────────────────────
print("\n[1] THE CLASS -- a step created AND closed in one commit (criteria 1, 2)\n")
b, d, _ = run_detector(ns, {"86.1": "done", "86.7": "pending"}, {"86.1": "done", "86.7": "pending", "86.86": "done"})
check("[1] created-and-closed BUMPS", b == "patch", f"got {b!r} -- the None exclusion is back")
check("[1] the created step is NAMED in the decision", d.get("created_done") == ["86.86"], f"decision={d}")
check("[1] the reason distinguishes created from transitioned",
      d.get("reason") == "flip_created", f"reason={d.get('reason')!r}")

# The rule must be about the CLASS, not about one step id.
b1, _, _ = run_detector(ns, {"9.1": "done", "9.5": "pending"}, {"9.1": "done", "9.5": "pending", "9.99": "done"})
b2, _, _ = run_detector(ns, {"9.1": "done", "12.5": "pending"}, {"9.1": "done", "12.5": "pending", "12.7": "done"})
check("[1] the rule is about the CLASS, not a hardcoded id", b1 == "patch" and b2 == "patch",
      f"got {b1!r} / {b2!r}")

# Magnitude rules are untouched by this step.
b, _, _ = run_detector(ns, {"77.1": "pending"}, {"77.0": "done", "77.1": "pending"})
check("[1] magnitude: a created X.0 kickoff is minor, not patch", b == "minor", f"got {b!r}")
b, _, _ = run_detector(ns, {"78.1": "pending"}, {"78.1": "done"})
check("[1] magnitude: closing the last step of a phase is major", b == "major", f"got {b!r}")

# ── [2] EVERY 'none' IS EXPLAINED (criterion 4) ─────────────────────────────
print("\n[2] NO UNEXPLAINED 'none' -- the silent-swallow class (criterion 4)\n")
# phase-86.91 cycle-2 (Q/A finding W3): this list used to hold two entries while
# the assertion below was NAMED "EVERY branch". Deleting the
# masterplan_unreadable_at_HEAD assignment left the guard fully green -- 3 of 4
# branches driven, on the very criterion that demands the CLASS be closed. The
# denominator is now DERIVED FROM THE SHIPPED SOURCE rather than hand-listed, so
# a future 5th branch fails this check instead of slipping past it.
CASES = [
    ("no_flip", {"86.1": "done"}, {"86.1": "done"}),
    ("first_commit", None, {"86.1": "done"}),
    ("masterplan_unreadable_at_HEAD", {"86.1": "pending"}, None),
]
for expected, before, after in CASES:
    b, d, _ = run_detector(ns, before, after)
    check(f"[2] '{expected}' is reported as its own reason", b == "none" and d.get("reason") == expected,
          f"bump={b!r} reason={d.get('reason')!r}")
b, d, err = run_detector(ns, {"a": "pending"}, {"a": "done"}, explode=True)
check("[2] an internal error is reported as detector_error",
      b == "none" and str(d.get("reason", "")).startswith("detector_error:"),
      f"bump={b!r} reason={d.get('reason')!r}")
# KNOWN-MEMBER RECALL, with the member list taken from the code rather than from
# me: count the `return "none"` sites inside the shipped _flip_magnitude and
# require that many DISTINCT reasons to have been observed.
_none_sites = 0
for _n in ast.walk(ast.parse(SHIPPED)):
    if isinstance(_n, ast.FunctionDef) and _n.name == "_flip_magnitude":
        _none_sites = sum(1 for _r in ast.walk(_n)
                          if isinstance(_r, ast.Return)
                          and isinstance(_r.value, ast.Constant) and _r.value.value == "none")
_observed = {run_detector(ns, bef, aft)[1].get("reason") for _, bef, aft in CASES}
_observed.add(run_detector(ns, {"a": "pending"}, {"a": "done"}, explode=True)[1].get("reason", "").split(":")[0] or None)
_observed.discard(None)
check("[2] EVERY branch that returns 'none' sets a reason -- none is left unrecorded",
      all(run_detector(ns, bef, aft)[1].get("reason") for _, bef, aft in CASES), "a branch returned 'none' silently")
check(f"[2] known-member RECALL: all {_none_sites} none-returning branches are DRIVEN "
      f"(denominator derived from source, not hand-listed)",
      _none_sites > 0 and len(_observed) >= _none_sites,
      f"{len(_observed)} distinct reasons observed for {_none_sites} branches: {sorted(_observed)}")

# ── [3] NEVER RAISES (criterion 7), by fault injection ──────────────────────
print("\n[3] NEVER RAISES -- proven by injecting a fault, not by reading the source\n")
raised = None
try:
    b, d, err = run_detector(ns, {"a": "pending"}, {"a": "done"}, explode=True)
except BaseException as exc:                                   # noqa: BLE001
    raised = exc
check("[3] the injected fault does NOT propagate", raised is None, f"propagated: {raised!r}")
check("[3] a fault bumps NOTHING", raised is None and b == "none", f"got {b!r}")
check("[3] the FAILED marker still reaches stderr", "flip-detect FAILED" in err, f"stderr={err.strip()[:120]!r}")

# ── [4] MUTATION (criterion 6) ──────────────────────────────────────────────
# Anchor uniqueness is asserted FIRST: a no-match replace is indistinguishable
# from a successful one and would score a phantom kill.
print("\n[4] MUTATION -- each cell must turn a check above RED (criterion 6)\n")
MUTANTS = [
    {
        "id": "restore-the-None-exclusion",
        "from": 'if st == "done" and before.get(sid, _ABSENT) is _ABSENT]',
        "to": 'if st == "done" and before.get(sid) not in (None, "done") and False]',
        # KILLED iff the created-and-closed case stops bumping.
        "probe": lambda m: run_detector(m, {"86.1": "done", "86.7": "pending"}, {"86.1": "done", "86.7": "pending", "86.86": "done"})[0] == "none",
        "why": "check [1] must go RED when created steps are excluded again",
    },
    {
        "id": "over-credit-already-done",
        "from": 'and before.get(sid) != "done"]',
        "to": "]",
        # KILLED iff an ALREADY-done step starts bumping (the dangerous direction).
        "probe": lambda m: run_detector(m, {"86.1": "done"}, {"86.1": "done"})[0] != "none",
        "why": "check [0] must go RED when a no-op commit starts bumping",
    },
    {
        "id": "drop-the-unreadable-reason",
        "from": '_FLIP_DECISION["reason"] = "masterplan_unreadable_at_HEAD"',
        "to": "pass",
        # This cell SURVIVED in cycle 1 -- the branch was never driven.
        "probe": lambda m: not run_detector(m, {"86.1": "pending"}, None)[1].get("reason"),
        "why": "check [2] must go RED when the masterplan-unreadable branch stops explaining itself",
    },
    {
        "id": "drop-the-reason",
        "from": '_FLIP_DECISION["reason"] = "no_flip"',
        "to": "pass",
        # KILLED iff a 'none' becomes unexplained again.
        "probe": lambda m: not run_detector(m, {"86.1": "done"}, {"86.1": "done"})[1].get("reason"),
        "why": "check [2] must go RED when a 'none' stops explaining itself",
    },
]
for mut in MUTANTS:
    n = SHIPPED.count(mut["from"])
    if n != 1:
        check(f"[4] {mut['id']}: anchor is unique", False,
              f"found {n} occurrences -- a no-match replace looks identical to success")
        continue
    mutated_ns = load_detector(SHIPPED.replace(mut["from"], mut["to"]))
    try:
        killed = bool(mut["probe"](mutated_ns))
    except Exception as exc:                                   # noqa: BLE001
        killed = True   # a mutant that cannot even run is dead
        _ = exc
    check(f"[4] {mut['id']}: KILLED ({mut['why']})", killed,
          "the mutant SURVIVED -- this guard is not the one doing the work")

# ── [5] THE SIBLING HARNESS MUST NOT MIRROR THE DEFECT ──────────────────────
# Research finding I5: replay_changelog_rule_86_68.py carried a byte-copy of the
# same predicate. Fixing only the hook would leave the replay measuring the OLD
# rule, so criterion 3's three numbers would compare against a stale baseline.
print("\n[5] THE SIBLING REPLAY HARNESS -- guarded BEHAVIOURALLY, not by substring\n")
# phase-86.91 cycle-2 (Q/A finding W2). This section used to be three substring
# scans, and the cycle-1 Q/A killed it with two mutants that both SURVIVED at
# 24/24 green: one made newly_done_ids ignore `count_created` while KEEPING the
# literal in the file, the other restored the None exclusion REWORDED as
# `not in ("done", None)`. A scan cannot see either. The replay is the instrument
# that produces criterion 3's three numbers, so it is now DRIVEN: the predicate
# is extracted from the shipped file and its two arms must actually DISAGREE.
REPLAY = REPO / "scripts" / "qa" / "replay_changelog_rule_86_68.py"
REPLAY_NEEDED = ("_ABSENT", "newly_done_ids")


def replay_predicate(src_text: str):
    """Extract and exec ONLY the predicate -- importing the module would run the
    whole multi-minute replay and shell out to git."""
    tree = ast.parse(src_text)
    lines = src_text.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(n in REPLAY_NEEDED for n in names):
            chunks.append("".join(lines[node.lineno - 1:node.end_lineno]))
    ns_r: dict = {}
    exec(compile("\n".join(chunks), "<shipped-replay>", "exec"), ns_r)  # noqa: S102
    return ns_r["newly_done_ids"]


REPLAY_SRC = REPLAY.read_text(encoding="utf-8")
# CYCLE-3 (Q/A finding QA-C2-6): the fixture used the single id 86.86, so
# narrowing the shipped predicate to `... and s == "86.86"` left every [5]
# assertion green -- including "the two arms genuinely DISAGREE". A single-id
# fixture cannot distinguish the CLASS from the instance, which is exactly what
# criterion 2 forbids. TWO unrelated created ids, in different top-level phases.
# CYCLE-4 (Q/A findings Q4/Q2b). The two-id fixture MOVED the bound rather than
# closing it: a whitelist matching exactly the fixture's ids SURVIVED on both the
# replay and the hook. Adding exemplars can never win, because an N-id fixture is
# defeated by an N-id whitelist. The closing move is an id that exists in NO
# SOURCE LITERAL -- derived at runtime from HEAD, so it changes as the repo moves
# and cannot be written into a whitelist in advance.
#
# THE BOUND, STATED (this step's own doctrine, which cycle 3 failed to apply to
# itself): a whitelist containing _RUNTIME_ID would still survive. What this
# closes is the AUTHORABLE special-case -- one a developer could write while
# looking at the checker. It does not, and cannot, prove the predicate is
# id-agnostic for every id.
_HEAD_SHA = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=REPO, timeout=30).stdout.strip()
_RUNTIME_ID = f"{700 + int(_HEAD_SHA[:4], 16) % 200}.{1 + int(_HEAD_SHA[4:8], 16) % 90}"
AFTER_R = {"86.1": "done", "86.7": "pending", "86.86": "done", "12.7": "done", _RUNTIME_ID: "done"}
BEFORE_R = {"86.1": "done", "86.7": "pending"}
try:
    pred = replay_predicate(REPLAY_SRC)
    fixed = pred(AFTER_R, BEFORE_R, count_created=True)
    shipped = pred(AFTER_R, BEFORE_R, count_created=False)
except Exception as exc:                                       # noqa: BLE001
    fixed = shipped = None
    check("[5] the replay predicate is extractable and runnable", False, repr(exc))
else:
    check("[5] the replay predicate is extractable and runnable", True)
    check("[5] count_created=True COUNTS created-and-closed steps in UNRELATED phases",
          sorted(fixed) == sorted(["12.7", "86.86", _RUNTIME_ID]),
          f"got {fixed!r} -- an N-id fixture is defeated by an N-id whitelist")
    check(f"[5] and it counts a RUNTIME-DERIVED id ({_RUNTIME_ID}) present in no source literal",
          _RUNTIME_ID in fixed,
          "an id no whitelist could have been authored for was still dropped")
    check("[5] count_created=False reproduces the SHIPPED (defective) result", shipped == [], f"got {shipped!r}")
    # The whole point: the two arms must DISAGREE, or the replay's three numbers
    # are three names for one measurement.
    check("[5] the two arms genuinely DISAGREE (not two names for one number)",
          fixed != shipped, f"both arms returned {fixed!r}")

# CYCLE-3 (Q/A finding QA-C2-1). This was a SUBSTRING SCAN, and the Q/A killed it:
# replacing the single line `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)` with
# `pass` keeps every scanned literal, leaves this checker ALL GREEN, and
# measurably unpins the corpus (707 -> 712) -- while the printed header still says
# `= 8dc70502`, because the endpoint is rev-parse'd from the CONSTANT and never
# compared to the commits actually selected. That is the same vacuity shape the
# cycle-1 W2 finding killed one guard over; I fixed it there and left it here.
#
# So the pin is now DRIVEN: build the corpus the way the shipped script does and
# require the NEWEST SELECTED COMMIT to be the pinned endpoint. A scan cannot see
# the difference; this can.
def corpus_head(src_text: str) -> str | None:
    """DRIVE the shipped corpus selection and return the newest selected sha.

    CYCLE-3 SECOND REPAIR, and the reason it is written this way. The first
    attempt re-implemented the selection -- it re-read CORPUS_SINCE/CORPUS_UNTIL
    and built its own argv with its own `if CORPUS_UNTIL: append(...)`. Mutating
    the SHIPPED append line therefore changed nothing my probe could see, and the
    QA-C2-1 cell scored SURVIVED. That is the same defect the Q/A had just charged
    me with, one level down: a re-implementation cannot detect a mutation of the
    original.

    So the shipped lines are EXECUTED. The block from `CORPUS_SINCE =` through
    `rc, out = sh(*_log_args)` is sliced out and exec'd with `sh` stubbed to
    CAPTURE the argv the shipped code actually assembles; git is then run with
    that captured argv.
    """
    lines = src_text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if ln.startswith("CORPUS_SINCE =")), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("rc, out = sh(*_log_args)")), None)
    if start is None or end is None:
        return None
    captured: list[list[str]] = []

    def _sh(*a):
        captured.append(list(a))
        return 0, ""

    ns_c: dict = {"sh": _sh, "os": __import__("os")}
    try:
        exec(compile("".join(lines[start:end + 1]), "<shipped-corpus>", "exec"), ns_c)  # noqa: S102
    except Exception as exc:                                   # noqa: BLE001
        # CYCLE-4: RAISE rather than return None. Returning None made a mutant
        # that could not BUILD score as DETECTED at the call site (`mh is None or
        # mh != pin`), i.e. a build failure counted as a kill -- the same defect
        # the cycle-3 repair fixed in the sibling [6] branch and left here.
        raise RuntimeError(f"the sliced corpus block did not run: {exc}") from exc
    if not captured:
        raise RuntimeError("the sliced corpus block ran but never called sh() -- "
                           "the slice boundary no longer covers the selection")
    out = subprocess.run(captured[0], capture_output=True, text=True, cwd=REPO, timeout=120).stdout
    recs = [r for r in out.split("\x1e") if r.strip()]
    if not recs:
        return None
    return recs[0].strip("\n").split("\x1f")[0]


def resolve(rev: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", rev], capture_output=True, text=True, cwd=REPO, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


_pin = None
for _line in REPLAY_SRC.splitlines():
    if _line.startswith("CORPUS_UNTIL ="):
        _pin = eval(_line.split("=", 1)[1].strip(), {"os": __import__("os")})  # noqa: S307
        break
_head = corpus_head(REPLAY_SRC)
check("[5] the corpus UPPER bound is pinned BEHAVIOURALLY (newest selected commit == the pin)",
      bool(_pin) and _head is not None and _head == resolve(_pin),
      f"pin={_pin!r} newest_selected={_head!r} resolved_pin={resolve(_pin) if _pin else None!r} "
      f"-- if these differ the corpus still floats with HEAD")
check("[5] the corpus LOWER bound is an explicit timestamp, not a bare date",
      # phase-86.94: this is a SUBSTRING test, so it still matches the
      # TZ-qualified "2026-08-11T00:00:00Z" the replay now pins. That is why
      # this checker stayed green through the TZ fix -- but note it would
      # ALSO stay green if the Z were removed again, so it is a pin-presence
      # check and not a TZ check. The TZ property is asserted by
      # scripts/qa/verify_no_sliding_windows_86_94.py, which resolves the
      # constant and classifies its VALUE.
      "2026-08-11T00:00:00" in REPLAY_SRC,
      "a bare --since date is applied at the CURRENT time of day and slides")

print("\n[6] MUTATION of the REPLAY predicate -- the cycle-1 survivors\n")
REPLAY_MUTANTS = [
    {
        "id": "QA-11 ignore-count_created (literal kept, behaviour stripped)",
        "from": "    return (created + transitioned) if count_created else transitioned",
        "to": "    return transitioned  # count_created ignored",
        "probe": lambda f: f(AFTER_R, BEFORE_R, count_created=True) == f(AFTER_R, BEFORE_R, count_created=False),
        "why": "a scan for the word 'count_created' cannot see this; the drive can",
    },
    {
        "id": "QA-12 reworded None exclusion",
        "from": '               if st=="done" and before.get(s, _ABSENT) is _ABSENT]',
        "to": '               if st=="done" and before.get(s) not in ("done", None) and False]',
        "probe": lambda f: f(AFTER_R, BEFORE_R, count_created=True) == [],
        "why": "the defect reworded is invisible to a literal scan",
    },
]
REPLAY_MUTANTS.append({
    # The cycle-3 Q/A's mutant, which SURVIVED the two-id fixture. It is scored
    # against the RUNTIME id, which the whitelist cannot contain.
    "id": "Q4 whitelist matching the fixture's authored ids",
    "from": '    return (created + transitioned) if count_created else transitioned',
    "to": ('    created = [s for s in created if s in ("86.86", "12.7")]\n'
           '    return (created + transitioned) if count_created else transitioned'),
    "probe": lambda f: _RUNTIME_ID not in f(AFTER_R, BEFORE_R, count_created=True),
    "why": "an N-id whitelist defeats an N-id fixture; a runtime-derived id defeats the whitelist",
})
REPLAY_MUTANTS.append({
    # The cycle-2 Q/A's own mutant. With the single-id fixture it SURVIVED every
    # [5] assertion, including "the two arms genuinely DISAGREE". It is a cell now
    # so the two-id fixture's class-coverage is PROVEN rather than asserted.
    "id": "QA-C2-6 special-case a single step id (the shape criterion 2 forbids)",
    "from": '               if st=="done" and before.get(s, _ABSENT) is _ABSENT]',
    "to": '               if st=="done" and before.get(s, _ABSENT) is _ABSENT and s=="86.86"]',
    "probe": lambda f: sorted(f(AFTER_R, BEFORE_R, count_created=True)) != ["12.7", "86.86"],
    "why": "a single-id fixture cannot tell the CLASS from the instance",
})
REPLAY_MUTANTS.append({
    "id": "QA-C2-1 unpin the upper bound (literals all kept)",
    "from": "if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)",
    "to": "pass  # upper bound unpinned",
    "probe": None,   # scored against corpus_head, not against the predicate
    "why": "a substring scan cannot see this; the behavioural pin check can",
})

# CYCLE-3: THREE outcomes, not two. `except -> killed = True` scores a mutant
# that never ran as a kill -- the shape the cycle-2 Q/A flagged as latent here and
# proved live in the sibling checker, where a SyntaxError mutant was counted
# KILLED for a whole cycle. A mutant that does not BUILD is UNSCORABLE, which
# FAILS the check rather than passing it.
for mut in REPLAY_MUTANTS:
    n = REPLAY_SRC.count(mut["from"])
    if n != 1:
        check(f"[6] {mut['id']}: anchor is unique", False,
              f"found {n} occurrences -- a no-match replace looks identical to success")
        continue
    mutated_src = REPLAY_SRC.replace(mut["from"], mut["to"])
    if mut["probe"] is None:
        # Scored against the CORPUS SELECTION, not the predicate.
        try:
            mh = corpus_head(mutated_src)
        except Exception as exc:                               # noqa: BLE001
            outcome = f"UNSCORABLE: the mutant did not build ({exc})"
        else:
            outcome = "DETECTED" if mh != resolve(_pin) else "SURVIVED"
    else:
        try:
            mpred = replay_predicate(mutated_src)
        except Exception as exc:                               # noqa: BLE001
            outcome = f"UNSCORABLE: mutant did not build ({exc})"
        else:
            try:
                outcome = "DETECTED" if mut["probe"](mpred) else "SURVIVED"
            except Exception as exc:                           # noqa: BLE001
                outcome = f"UNSCORABLE: probe raised ({exc})"
    check(f"[6] {mut['id']}: KILLED ({mut['why']})", outcome == "DETECTED",
          f"{outcome} -- a mutant that did not run has not been tested by anything")

# ── [7] THE DECISION LOG IS DRIVEN, AND THE FILE IS READ BACK ──────────────
# Criterion 4's mechanism is a FILE. Asserting the dict that feeds it is vacuity
# shape #1: it cannot fail when the write is removed. This section points
# repo_root at a temp tree, calls the SHIPPED _log_decision, and reads the line.
print("\n[7] THE DECISION LOG -- the hook's own OUTPUT, read back from disk\n")


def drive_log(ns_l, bump: str, decision: dict, root: Path) -> str | None:
    ns_l["repo_root"] = root
    ns_l["_FLIP_DECISION"].clear()
    ns_l["_FLIP_DECISION"].update(decision)
    ns_l["_log_decision"](bump)
    f = root / "handoff" / "logs" / "changelog-decisions.log"
    return f.read_text(encoding="utf-8").strip() if f.exists() else None


with tempfile.TemporaryDirectory() as _td:
    _root = Path(_td)
    _line = drive_log(load_detector(SHIPPED), "patch",
                      {"reason": "flip_created", "created_done": ["86.86"], "transitioned_done": []}, _root)
    check("[7] a decision WRITES a line to changelog-decisions.log", bool(_line), f"got {_line!r}")
    check("[7] the line carries the bump", bool(_line) and "bump=patch" in _line, f"got {_line!r}")
    check("[7] the line carries the REASON, which is the whole point of criterion 4",
          bool(_line) and "reason=flip_created" in _line, f"got {_line!r}")
    check("[7] the line names the created step", bool(_line) and "created_done=86.86" in _line, f"got {_line!r}")

with tempfile.TemporaryDirectory() as _td:
    _line = drive_log(load_detector(SHIPPED), "none",
                      {"reason": "no_flip", "created_done": [], "transitioned_done": []}, Path(_td))
    check("[7] a 'none' decision is ALSO written -- an unexplained none is the defect",
          bool(_line) and "bump=none" in _line and "reason=no_flip" in _line, f"got {_line!r}")

# MUTATION: delete the write. This is the mutant that SURVIVED at 34/0 in cycle 3.
# The anchor is the WRITE ITSELF, replaced by a no-op of the same indentation.
# A first attempt prefixed `if False:` to the `with`, which left the `with` at the
# original indent and produced an IndentationError -- the harness correctly scored
# that UNSCORABLE rather than KILLED, which is the cycle-3 repair doing its job on
# its own author.
_MUT_WRITE = ('        with open(log_dir / "changelog-decisions.log", "a", encoding="utf-8") as _fh:\n'
              '            _fh.write(f"{stamp} {sha} bump={bump} reason={reason} "\n'
              '                      f"created_done={created} transitioned_done={transitioned}\\n")')
if SHIPPED.count(_MUT_WRITE) != 1:
    check("[7] mutate-delete-the-write: anchor is unique", False, f"found {SHIPPED.count(_MUT_WRITE)}")
else:
    _mut = SHIPPED.replace(_MUT_WRITE, '        pass  # the write is deleted')
    try:
        with tempfile.TemporaryDirectory() as _td:
            _ml = drive_log(load_detector(_mut), "patch", {"reason": "flip_created"}, Path(_td))
        _out = "DETECTED" if not _ml else "SURVIVED"
    except Exception as _e:                                    # noqa: BLE001
        _out = f"UNSCORABLE: the mutant did not build ({_e})"
    check("[7] delete-the-decision-log-write: KILLED (this mutant SURVIVED all 34 assertions in cycle 3)",
          _out == "DETECTED", _out)

print()
if FAILURES:
    print(f"FAILED: {len(PASSED)} passed, {len(FAILURES)} failed")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL GREEN: {len(PASSED)} passed, 0 failed")
