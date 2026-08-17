#!/usr/bin/env python3
"""phase-86.97 -- the changelog decision log: coverage of the paths that never
reach the detector, and a guard that can actually see a DELETED CALL.

WHY THIS IS A SEPARATE FILE FROM verify_changelog_flip_86_91.py
---------------------------------------------------------------
The 86.91 checker extracts the shipped detector with `detector_source()`, which
walks `tree.body` collecting `FunctionDef` / `Assign` / `AnnAssign` -- all
DEFINITION classes that bind a name. The production invocation is a bare
`_log_decision(bump_type)`: an `ast.Expr(Call)`, binding nothing. It can never
match, so **enlarging the NEEDED tuple cannot help**.

That is not a weak assertion, it is structural blindness, and the difference
matters. MEASURED: deleting the production call leaves the extracted source
BYTE-IDENTICAL, while an edit *inside* the definition changes it (+24 B). So the
extraction is live -- it is specifically the call it cannot see.

THE FIGURE IS PINNED TO A COMMIT, because it moves. At 52358053 (this step's
parent) the extraction was 7,597 B / sha1 f7458a6ab1f5fe96; at HEAD it is
8,617 B / sha1 072056e58af2befa. What changed it was THIS STEP's own criterion-5
docstring correction inside `_log_decision` -- one of the four extracted names --
which rode in the same commit that first stated the number. The PROPERTY is
invariant (call-deleted is byte-identical to base at both commits); only the
byte count is not. A measured figure without the commit it was measured at is a
claim that quietly expires.

The mutant is therefore INVISIBLE, not merely surviving: no assertion added to
that file, however clever, could ever kill it.

The literature name for a method whose removal breaks no test is
**pseudo-tested** (Vera-Perez et al.; Niedermayr et al. found 291/2041, and 14
of 25 manually inspected were side-effect methods -- `_log_decision`'s exact
shape). The documented remedy is to assert on the OBSERVABLE EFFECT, which here
means driving the real hook and reading the log FILE it writes.

Driving the whole hook is also the only way to reach the three pre-detector
`exit 0` paths at all: they live in bash, OUTSIDE the heredoc, so every
Python-side test in the 86.91 checker is structurally unable to execute them.

Sections:
  [1] PRECONDITIONS for the lexical classification rule (asserted, not assumed)
  [2] ENUMERATION of every exit path FROM SOURCE, by a written-down rule that
      self-tests, plus a classification keyed on guard CONDITION text
  [3] END-TO-END: drive the REAL hook in a temp git repo, assert on the FILE
  [4] MUTATION of the production guards: control GREEN first, then kill the
      delete-the-call mutant; a mutant that does not BUILD is UNSCORABLE and FAILS
  [5] MUTATION of the [1]/[2] guards themselves -- criterion 6 says EVERY new
      guard, and cycles 1-2 covered only [3]/[4]

Run:  python scripts/qa/verify_decision_log_86_97.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "post-commit-changelog.sh"

_pass = 0
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    global _pass
    if cond:
        _pass += 1
        print(f"  ok   {name}")
    else:
        _failures.append(f"{name}{' -- ' + detail if detail else ''}")
        print(f"  FAIL {name}{' -- ' + detail if detail else ''}")
    return bool(cond)


HOOK_SRC = HOOK.read_text(encoding="utf-8")
LINES = HOOK_SRC.splitlines()

# ── [1] PRECONDITIONS ───────────────────────────────────────────────────────
#
# The classification rule below is LEXICAL: "an exit before the detector cannot
# reach it". That is only sound while bash's execution order matches source
# order. Three things would break it, so all three are asserted rather than
# assumed -- and if any ever becomes false, this section goes red and says why,
# instead of the classification silently becoming wrong.
FUNC_RE = re.compile(r"^\s*(function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{")
REORDER_RE = re.compile(r"^\s*(trap|source|\.|eval)\s")
HEREDOC_RE = re.compile(r"<<\s*'?([A-Za-z_][A-Za-z0-9_]*)'?\s*$")
RULE = re.compile(r"(?:^|;|&&|\|\|)\s*exit\b")
DUMB = re.compile(r"\bexit\b")


# CLASSIFICATION.
#
# Keyed on the guard's CONDITION TEXT, never on a line number. Line numbers are
# exactly the fixture that rots (phase-86.92): one inserted line above and a
# pinned table silently describes the wrong path. Condition text moves with the
# code it guards.
#
# A pre-detector exit that matches NO entry here is a FAILURE, not a default.
# The new member gets classified deliberately; it is never waved through --
# which is the same discipline as "classify the new member, never loosen the
# gate to get green".
PRE_DETECTOR_CLASSIFICATION = [
    (r"chore: \(auto-changelog\|changelog drift\)", "LEGITIMATELY-SILENT",
     ("The recursion guard. The hook is re-entering itself: the commit it is "
     "looking at is the one IT created. Such a commit is by construction not a "
     "bump candidate, so there is no decision to explain. Logging it would "
     "double the log with entries about the logger. This is a BOUND to state, "
     "not a defect to fix -- and it accounts for essentially the whole "
     "commits-vs-lines gap (re-derived in [3]).")),
    (r"! -f \"\$CHANGELOG\"", "MUST-LOG",
     ("The CHANGELOG is missing entirely. That is machinery breakage, not a "
     "routine skip: every subsequent commit will silently produce nothing, and "
     "the decision log is the only place an operator would find out.")),
    (r"### Recent Activity", "MUST-LOG",
     ("The CHANGELOG exists but has lost its anchor section. Same class as "
     "above -- silent structural breakage that looks identical to 'nothing to "
     "do' from outside.")),
]


def analyse(src: str) -> dict:
    """Everything sections [1] and [2] assert on, as data, from ONE source of truth.

    THIS IS A FUNCTION SO THE SECTION-[1]/[2] GUARDS CAN BE MUTATION-TESTED.
    Cycle 2 computed all of this inline at module scope, which made those guards
    unreachable by any mutant: the only way to re-run them was to re-run the
    whole file against the real hook. Section [5] now feeds mutated sources
    through THIS function -- the same code path the shipped assertions use, not
    a re-implementation, which would test a copy instead of the guard.
    """
    lines = src.splitlines()
    heredocs = [(i + 1, m.group(1)) for i, l in enumerate(lines)
                if (m := HEREDOC_RE.search(l))]
    start = heredocs[0][0] if heredocs else 10**9
    term = heredocs[0][1] if heredocs else "PYEOF"
    term_lines = [i + 1 for i, l in enumerate(lines) if l.rstrip() == term]
    end = term_lines[0] if len(term_lines) == 1 else 10**9

    def outside(n: int) -> bool:
        return not (start < n < end)

    rule_hits = [(i + 1, l) for i, l in enumerate(lines)
                 if RULE.search(l) and outside(i + 1)]
    dumb_hits = [(i + 1, l) for i, l in enumerate(lines)
                 if DUMB.search(l) and outside(i + 1)]
    missed = [(n, l) for (n, l) in dumb_hits if (n, l) not in rule_hits]
    pre = [(n, l) for (n, l) in rule_hits if n < start]
    unclassified = []
    for n, _l in pre:
        cond = guard_condition(lines, n)
        if not any(re.search(pat, cond) for (pat, _k, _w) in PRE_DETECTOR_CLASSIFICATION):
            unclassified.append((n, cond))
    return {
        "lines": lines,
        "funcs": [i + 1 for i, l in enumerate(lines) if FUNC_RE.match(l)],
        "reorder": [i + 1 for i, l in enumerate(lines) if REORDER_RE.match(l)],
        "heredocs": heredocs, "term": term, "term_lines": term_lines,
        "start": start, "end": end,
        "rule_hits": rule_hits,
        "unexplained": [(n, l) for (n, l) in missed if not l.lstrip().startswith("#")],
        "pre": pre,
        "post": [(n, l) for (n, l) in rule_hits if n > end],
        "unclassified": unclassified,
    }


def guard_condition(lines: list[str], lineno: int) -> str:
    """The nearest preceding `if`/`||` line -- the condition this exit serves."""
    for j in range(lineno - 1, max(0, lineno - 6), -1):
        text = lines[j - 1]
        if re.search(r"^\s*(if|elif)\b", text) or "||" in text:
            return text.strip()
    return lines[lineno - 1].strip()


print("\n[1] PRECONDITIONS for the lexical rule (criterion 2)\n")
A = analyse(HOOK_SRC)
check("[1] the hook defines NO bash functions (so lexical order == execution order)",
      not A["funcs"], f"functions at lines {A['funcs']} -- a lexical rule is no longer sound")
check("[1] no trap / source / . / eval (nothing reorders execution)",
      not A["reorder"], f"found at lines {A['reorder']}")
check("[1] exactly ONE heredoc -- the detector", len(A["heredocs"]) == 1,
      f"found {len(A['heredocs'])}: {A['heredocs']}")
check("[1] the heredoc terminator is found exactly once", len(A["term_lines"]) == 1,
      f"'{A['term']}' at {A['term_lines']}")
check("[1] the detector region is non-empty and ordered",
      A["start"] < A["end"], f"{A['start']}..{A['end']}")
DETECTOR_START, DETECTOR_END = A["start"], A["end"]
print(f"       detector heredoc: lines {DETECTOR_START}..{DETECTOR_END} (terminator {A['term']!r})")

# ── [2] ENUMERATION FROM SOURCE, BY A WRITTEN-DOWN RULE ─────────────────────
#
# THE RULE (this is the "written-down rule" criterion 2 requires):
#   An EXIT PATH is any line, outside the detector heredoc, on which the token
#   `exit` appears as a command -- i.e. matching  (^|;|&&|\|\|)\s*exit\b .
#
# THE SELF-TEST: a scan that quietly matches nothing looks identical to a clean
# bill of health, so the rule is cross-checked against a deliberately DUMBER
# one (every line containing the substring "exit" at all). Every line the dumb
# scan finds and the rule does not must be explainable as a comment or prose;
# anything else means the rule is under-matching and the gate FAILS.
print("\n[2] ENUMERATION of exit paths from source (criterion 2)\n")

check("[2] the rule finds a non-zero number of exit paths (a scan that matches "
      "nothing is not a clean bill of health)", len(A["rule_hits"]) > 0,
      "the rule matched NOTHING -- it is broken, not the hook")

check("[2] every line the dumber scan finds but the rule does not is a comment "
      "(so the rule is not under-matching)", not A["unexplained"],
      f"unexplained at {[n for n, _ in A['unexplained']]}")


pre, post = A["pre"], A["post"]
check("[2] at least one PRE-detector exit path exists (else this step's premise "
      "is void)", len(pre) > 0)
check("[2] at least one POST-detector exit path exists", len(post) > 0)


print(f"       {len(pre)} pre-detector exit path(s), {len(post)} post-detector\n")
classified = 0
for n, _l in pre:
    cond = guard_condition(A["lines"], n)
    hit = next(((pat, kind, why) for (pat, kind, why) in PRE_DETECTOR_CLASSIFICATION
                if re.search(pat, cond)), None)
    if hit is None:
        check(f"[2] pre-detector exit at :{n} is classified", False,
              f"UNCLASSIFIED guard {cond!r} -- a new early-exit path was added and "
              "nobody decided whether it must log. Classify it; do not widen the rule.")
        continue
    classified += 1
    print(f"       :{n}  {hit[1]:<22} {cond[:64]}")
check("[2] every pre-detector exit path is classified MUST-LOG or "
      "LEGITIMATELY-SILENT", classified == len(pre),
      f"{classified}/{len(pre)} classified")

# Post-detector exits are a different question and are recorded as such: by the
# time they run the detector has already executed and written its line, so they
# cannot cause a MISSING decision. They are in scope for enumeration, not for
# the must-log judgement.
for n, _l in post:
    print(f"       :{n}  POST-DETECTOR          (decision already written)")

# ── [3] END-TO-END: DRIVE THE REAL HOOK ─────────────────────────────────────
print("\n[3] END-TO-END -- drive the REAL hook in a temp git repo (criterion 4)\n")

CHANGELOG_SEED = (
    "# Changelog\n\n## v1.0.0\n\n### Recent Activity\n\n"
    "| Date | Commit | Description |\n|---|---|---|\n"
)


EMPTY_MP = '{"phases": []}'


def make_repo(tmp: Path, masterplan: str = EMPTY_MP) -> None:
    (tmp / "CHANGELOG.md").write_text(CHANGELOG_SEED, encoding="utf-8")
    (tmp / ".claude").mkdir(exist_ok=True)
    (tmp / ".claude" / "masterplan.json").write_text(masterplan, encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "-A"],
                 ["commit", "-q", "-m", "feat: seed commit"]):
        subprocess.run(["git", *args], cwd=tmp, env=env, check=True,
                       capture_output=True)


def drive(hook_src: str, subject: str = "feat: a real change",
          before_mp: str = EMPTY_MP, after_mp: str | None = None) -> tuple[int, str, str]:
    """Run the hook end-to-end. Returns (rc, decision-log text, stderr).

    `before_mp` / `after_mp` are the masterplan at HEAD~1 and HEAD. The detector
    diffs those two revisions, so a flip scenario is expressed by making them
    differ. Cycle 3 hardcoded `{"phases": []}` for both, which is why the driver
    could only ever produce ONE of the nine reason states.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        make_repo(tmp, before_mp)
        env = {**os.environ, "CLAUDE_PROJECT_DIR": str(tmp),
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        (tmp / "f.txt").write_text(subject, encoding="utf-8")
        if after_mp is not None:
            (tmp / ".claude" / "masterplan.json").write_text(after_mp, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp, env=env, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", subject], cwd=tmp, env=env,
                       check=True, capture_output=True)
        hp = tmp / "hook.sh"
        hp.write_text(hook_src, encoding="utf-8")
        hp.chmod(0o755)
        r = subprocess.run(["bash", str(hp)], cwd=tmp, env=env, check=False,
                           capture_output=True, text=True, timeout=120)
        log = tmp / "handoff" / "logs" / "changelog-decisions.log"
        return r.returncode, (log.read_text(encoding="utf-8") if log.exists() else ""), r.stderr


DECISION_RE = re.compile(
    r"bump=(?P<bump>\S+)\s+reason=(?P<reason>\S+)\s+"
    r"created_done=(?P<created>\S+)\s+transitioned_done=(?P<transitioned>\S+)")


def parse_decision(log_text: str) -> dict | None:
    """The decision as DATA. Returns None when no line was written at all.

    Cycle 3 asserted `"reason=" in log_text`. That token is emitted from a
    LITERAL format string at post-commit-changelog.sh:271, so it is true for
    every non-empty line the writer can produce -- strictly subsumed by the
    "a line was written" check above, and unable to distinguish any decision
    from any other. Parsing makes the decision assertable.
    """
    last = [ln for ln in log_text.strip().splitlines() if ln.strip()]
    if not last:
        return None
    m = DECISION_RE.search(last[-1])
    return dict(m.groupdict()) if m else {}


# ISOLATION ASSERTION. The driver writes into a temp dir, but "it should be
# isolated" is a claim, and a driver that quietly wrote to the real log would
# corrupt the very evidence this step reasons about. So: snapshot the real log
# and require it byte-identical afterwards.
real_log = REPO / "handoff" / "logs" / "changelog-decisions.log"
real_before = real_log.read_bytes() if real_log.exists() else b""

rc, log_text, err = drive(HOOK_SRC)
check("[3] the real hook runs to completion in a temp repo", rc == 0,
      f"rc={rc} stderr={err[-200:]!r}")
check("[3] a decision line is WRITTEN TO THE FILE (the observable effect, not an "
      "extracted namespace)", log_text.strip() != "",
      "no decision-log file was produced -- this is the assertion the 86.91 "
      "extraction structurally could not make")
# ── [3a] THE DECISION ITSELF, PER SCENARIO (phase-86.97 cycle 4) ────────────
#
# WHAT THIS REPLACES, and why it was not merely weak but VACUOUS. The assertion
# here was:
#
#     check("[3] the decision line carries a reason", "reason=" in log_text)
#
# `reason=` is a LITERAL in the writer's format string
# (post-commit-changelog.sh:271), so it is present in EVERY non-empty line the
# writer can emit. The check was therefore strictly subsumed by the
# "a line was written" assertion immediately above it, and could not tell any
# decision from any other. MEASURED: deleting `bump_type = _flip_magnitude()`
# (hook :214) yields rc=0 and writes `bump=minor reason=unrecorded` instead of
# `bump=none reason=no_flip` -- a SPURIOUS minor bump (the thing 86.68 exists to
# prevent) carrying an unexplained reason (the thing 86.91 criterion 4 exists to
# close) -- and every assertion in this file stayed green.
#
# THE EXPECTED TABLE IS DERIVED FROM BRANCH STRUCTURE, NOT FROM OBSERVED OUTPUT.
# That ordering is the point: an oracle written after looking at a run drifts to
# whatever the run produced (arXiv:2410.21136, arXiv:2402.11041 on oracle
# improvement). The nine reason states and their sites, read out of the hook
# before any scenario was driven:
#
#   :160 masterplan_unreadable_at_HEAD   :193 flip_created
#   :163 first_commit                    :194 flip_transitioned
#   :190 no_flip                         :195 flip_created_and_transitioned
#   :207 detector_error:<Type>           :216 subject_forced_major  <- set OUTSIDE
#   :267 unrecorded  <- the `.get` default: nobody sets it. It is the signature
#                       of a detector that never ran, i.e. the :214 mutant.
#
# Cycle 3's driver seeded `{"phases": []}` for both revisions, so it exercised
# exactly ONE of the nine (no_flip) and pinned ZERO values.
print("\n[3a] THE DECISION'S CONTENT, per scenario (criterion 4)\n")

_MP_PENDING = ('{"phases": [{"steps": ['
               '{"id": "99.1", "status": "pending"}, '
               '{"id": "99.2", "status": "pending"}]}]}')
_MP_FLIPPED = ('{"phases": [{"steps": ['
               '{"id": "99.1", "status": "done"}, '
               '{"id": "99.2", "status": "pending"}]}]}')
_MP_CREATED = ('{"phases": [{"steps": ['
               '{"id": "99.1", "status": "pending"}, '
               '{"id": "99.2", "status": "pending"}, '
               '{"id": "98.0", "status": "done"}, '
               '{"id": "98.1", "status": "pending"}]}]}')

# `bump` IS PINNED TOO, and it was not in cycle 4's first draft. The cycle-4 Q/A
# executed `return "minor"` -> `return "patch"` in _flip_magnitude()'s kickoff
# branch and it SURVIVED at 48/0: DECISION_RE captured bump, parse_decision
# returned it, and no assertion read it. A field that is parsed but never
# asserted is decoration, and the artifacts had already described the tuple as
# "compared by exact equality". The magnitudes below are derived from the
# documented rule (major if the flip emptied a whole top-level phase, minor if
# the step is the phase kickoff `X.0`, patch otherwise; an explicit `!` subject
# forces major on its own authority) and each seed is chosen to land on exactly
# one of them.
SCENARIOS = [
    # (label, subject, before_mp, after_mp, expected bump, reason, created,
    #  transitioned)
    ("no_flip -- masterplan unchanged",
     "feat: a real change", _MP_PENDING, _MP_PENDING, "none", "no_flip", "-", "-"),
    # 99.1 closes while sibling 99.2 stays pending -> phase not emptied, and
    # 99.1 is not a `.0` kickoff -> patch.
    ("flip_transitioned -- 99.1 pending -> done",
     "phase-99.1: close it", _MP_PENDING, _MP_FLIPPED, "patch", "flip_transitioned", "-", "99.1"),
    # 98.0 IS a kickoff, and sibling 98.1 stays pending -> minor, not major.
    ("flip_created -- 98.0 appears already done",
     "phase-98.0: file and close", _MP_PENDING, _MP_CREATED, "minor", "flip_created", "98.0", "-"),
    # The ONLY reason set outside _flip_magnitude() (:216). It is the scenario
    # the :214 mutant must NOT disturb, which is what makes the pair
    # discriminating rather than a single tripwire.
    ("subject_forced_major -- a `!` subject",
     "feat!: breaking change", _MP_PENDING, _MP_PENDING, "major", "subject_forced_major", "-", "-"),
]

_observed: dict[str, dict] = {}
for (_label, _subj, _bmp, _amp, _exp_bump, _exp_reason, _exp_created,
     _exp_trans) in SCENARIOS:
    _rc, _log, _err = drive(HOOK_SRC, _subj, _bmp, _amp)
    _d = parse_decision(_log)
    _observed[_label] = {"rc": _rc, "decision": _d}
    print(f"       {_label}\n         -> {_d}")
    check(f"[3a] {_label}: a parseable decision line was written",
          isinstance(_d, dict) and _d.get("reason") is not None,
          f"rc={_rc} log={_log.strip()[:120]!r} stderr={_err[-160:]!r}")
    if isinstance(_d, dict) and _d.get("reason") is not None:
        check(f"[3a] {_label}: reason == {_exp_reason!r} (the DECISION, not the "
              "presence of a line)",
              _d.get("reason") == _exp_reason,
              f"expected {_exp_reason!r}, got {_d.get('reason')!r} in {_d!r}")
        check(f"[3a] {_label}: bump == {_exp_bump!r} (the MAGNITUDE, which cycle 4 "
              "parsed and then asserted nowhere -- a `minor`->`patch` mutant "
              "survived at 48/0)",
              _d.get("bump") == _exp_bump,
              f"expected bump={_exp_bump!r}, got {_d.get('bump')!r} in {_d!r}")
        check(f"[3a] {_label}: created_done == {_exp_created!r} and "
              f"transitioned_done == {_exp_trans!r}",
              _d.get("created") == _exp_created and _d.get("transitioned") == _exp_trans,
              f"expected created={_exp_created!r} transitioned={_exp_trans!r}, got "
              f"created={_d.get('created')!r} transitioned={_d.get('transitioned')!r}")

# STANDING NEGATIVE CONTROL. `unrecorded` is the `.get` default at :267 and no
# branch ever assigns it, so its appearance means the detector did not run. It
# must never be observed in a healthy run of ANY scenario.
_unrecorded = [lbl for lbl, o in _observed.items()
               if isinstance(o["decision"], dict)
               and o["decision"].get("reason") == "unrecorded"]
check("[3a] NEGATIVE CONTROL: no scenario yields reason='unrecorded' (nobody "
      "assigns it -- it is the signature of a detector that never ran)",
      not _unrecorded,
      f"scenarios reporting an unrecorded decision: {_unrecorded}")

# DISCRIMINATION CONTROL. If every scenario produced the same reason, the four
# assertions above would be one assertion repeated and the table would prove
# nothing about the detector's branches.
_reasons = {o["decision"].get("reason") for o in _observed.values()
            if isinstance(o["decision"], dict)}
check("[3a] the scenarios DISCRIMINATE -- they do not all produce one reason",
      len(_reasons) >= 3,
      f"only {len(_reasons)} distinct reason(s) observed across "
      f"{len(SCENARIOS)} scenarios: {_reasons} -- the table cannot detect a "
      "detector that collapsed to a single branch")

def isolation_holds(snapshot: bytes) -> bool:
    """Pure predicate: is the real log still byte-identical to `snapshot`?

    Separated from the assertion so section [5] can probe it with a deliberately
    corrupted snapshot WITHOUT emitting a spurious FAIL line and without the
    caller hand-patching the pass/fail counters. A mutation cell that has to
    edit the scoreboard to run is a cell nobody will trust.
    """
    now = real_log.read_bytes() if real_log.exists() else b""
    return snapshot == now


def assert_isolated(where: str) -> None:
    """Re-check after EVERY drive, not just the first.

    Cycle 1 snapshotted once and checked once, so the recursion-guard drive and
    both mutant drives ran with no isolation assertion at all -- while the
    artifact claimed the property broadly. The check is cheap; the claim has to
    be true for every drive that exists, not for the first one.
    """
    check(f"[3] ISOLATION after {where}: the real repo's decision log is untouched",
          isolation_holds(real_before),
          "the driver wrote into the real repo -- it is contaminating its own evidence")


assert_isolated("the baseline drive")

# The recursion guard, driven for real: an auto-changelog subject must produce
# NO decision line. This is criterion 3's evidence, executed rather than argued.
rc2, log2, _ = drive(HOOK_SRC, subject="chore: auto-changelog hook entry for abc1234")
check("[3] recursion guard: an auto-changelog commit exits 0", rc2 == 0, f"rc={rc2}")
check("[3] recursion guard: and writes NO decision line (the BOUND, measured)",
      log2.strip() == "", f"unexpectedly logged: {log2.strip()[:120]!r}")
assert_isolated("the recursion-guard drive")

# Re-derive the commits-vs-lines gap AT EXECUTION TIME. Never pin the figure:
# the step was filed as "10 commits vs 5 lines" and that snapshot is already
# stale. The window is anchored to the log's own first timestamp -- a bare date
# would slide with the clock.
#
# FAIL LOUDLY IF THE INPUT IS ABSENT. Cycle 1 wrapped this whole block --
# including its check() -- in `if real_before:`, so a missing decision log made
# the re-derivation SILENTLY VANISH rather than go red. A guard whose input can
# disappear must say so; a skipped check is indistinguishable from a passing one
# in the summary line.
check("[3] the real decision log exists, so the gap CAN be re-derived",
      bool(real_before),
      "handoff/logs/changelog-decisions.log is missing -- the re-derivation below "
      "is SKIPPED, and that is reported rather than passed over in silence")
if real_before:
    first_stamp = real_before.decode("utf-8", "replace").splitlines()[0].split()[0]
    n_lines = len(real_before.decode("utf-8", "replace").strip().splitlines())
    out = subprocess.run(["git", "log", f"--since={first_stamp}", "--format=%s"],
                         cwd=REPO, capture_output=True, text=True, timeout=30,
                         check=False).stdout
    subjects = [s for s in out.splitlines() if s.strip()]
    recursion = [s for s in subjects
                 if re.match(r"^chore: (auto-changelog|changelog drift)", s, re.IGNORECASE)]
    gap = len(subjects) - n_lines
    print(f"\n       RE-DERIVED at execution time (window pinned to {first_stamp}):")
    print(f"         commits={len(subjects)}  decision lines={n_lines}  gap={gap}")
    print(f"         commits matching the recursion guard={len(recursion)}")
    check("[3] the gap is explained by the recursion guard (criterion 3: a BOUND, "
          "not an unexplained loss)", abs(gap - len(recursion)) <= 2,
          f"gap={gap} but recursion-guard commits={len(recursion)} -- the "
          "difference is NOT accounted for and needs investigating")

# ── [4] MUTATION ────────────────────────────────────────────────────────────
print("\n[4] MUTATION -- the guard must SEE a deleted call (criteria 4, 6)\n")


def heredoc_body(src: str) -> str | None:
    """The Python inside the detector heredoc, or None if it cannot be located."""
    ls = src.splitlines()
    try:
        s = next(i for i, l in enumerate(ls) if "<< 'PYEOF'" in l)
        e = next(i for i, l in enumerate(ls) if l.rstrip() == "PYEOF")
    except StopIteration:
        return None
    return "\n".join(ls[s + 1:e]) if e > s else None


def _bash_parses(src: str) -> bool:
    """The bash half of the buildability oracle, named so a failure can say which
    leg rejected the mutant rather than always blaming `bash -n`."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.sh"
        p.write_text(src, encoding="utf-8")
        return subprocess.run(["bash", "-n", str(p)], capture_output=True,
                              check=False).returncode == 0


def buildable(src: str) -> bool:
    """A mutant that does not BUILD is UNSCORABLE, never a kill.

    BOTH HALVES ARE CHECKED, AND THE SECOND HALF IS THE WHOLE POINT.
    The first version of this oracle was `bash -n` alone. That is structurally
    blind to the mutants it gates: `bash -n` does NOT parse inside a QUOTED
    heredoc (`<< 'PYEOF'`), and every mutation this file performs is a
    Python-side edit inside exactly that heredoc.

    MEASURED, with this file's own helper, before the fix: the mutant
    `_log_decision(bump_type` (unbalanced paren) gave bash -n rc=0 -- "buildable"
    -- while `compile()` raised `SyntaxError: '(' was never closed`. Driven, it
    produced rc=1 and an EMPTY log, and the cell's `m_log.strip() == ""` scoring
    rule therefore recorded it as KILLED. A crash was being counted as a kill.

    That is the same category of blindness this step exists to close -- an oracle
    that cannot observe the failure mode it is nominally guarding -- reproduced
    inside the guard itself. Found by the phase-86.97 cycle-1 Q/A.
    """
    if not _bash_parses(src):
        return False
    body = heredoc_body(src)
    if body is None:
        return False          # cannot locate it -> cannot vouch for it
    try:
        compile(body, "<heredoc>", "exec")
    except SyntaxError:
        return False
    return True


MUTANTS = [
    ("delete-the-production-call", "\n_log_decision(bump_type)\n", "\n",
     "the call the 86.91 extraction is structurally blind to"),
    # THE SECOND CELL'S MUTATION WAS WRONG IN CYCLE 1 AND THE FIX IS INSTRUCTIVE.
    # It redirected the write to `os.devnull` -- but `os` is imported ZERO times
    # inside the heredoc, so it actually killed via
    # `NameError: name 'os' is not defined`, swallowed by _log_decision's broad
    # `except Exception`. The cell passed, for a mechanism nobody intended, and
    # two artifacts repeated the wrong explanation. A mutation that kills by
    # accident is not evidence about the thing it names.
    # It now retargets the FILENAME: every name stays defined, the write still
    # succeeds, no exception is raised anywhere -- the only thing that changes is
    # that the bytes land somewhere the guard does not read. That isolates the
    # property under test: the guard is bound to the FILE, not to call text.
    ("retarget-the-log-write", 'log_dir / "changelog-decisions.log"',
     'log_dir / "changelog-decisions.log.RETARGETED"',
     "the write's destination, with no name left undefined and no exception raised"),
]

# ORACLE SELF-TEST. buildable() gates every cell below, so a buildable() that
# cannot say NO would silently disarm criterion 6's UNSCORABLE arm. Both
# directions are asserted -- a one-sided control proves nothing.
check("[4] ORACLE: buildable() says YES to the unmutated hook", buildable(HOOK_SRC),
      "the oracle rejects the real hook -- every cell below would be skipped")
_syntax_mutant = HOOK_SRC.replace("\n_log_decision(bump_type)\n", "\n_log_decision(bump_type\n")
check("[4] ORACLE: buildable() says NO to a Python SyntaxError INSIDE the heredoc",
      _syntax_mutant != HOOK_SRC and not buildable(_syntax_mutant),
      "bash -n alone cannot see inside a quoted heredoc; without the compile() leg "
      "a crashing mutant scores as a KILL")

# CONTROL FIRST: the unmutated hook must produce a line, or every 'kill' below
# would be measuring a driver that never worked.
control_ok = check("[4] CONTROL: the UNMUTATED hook writes a decision line",
                   log_text.strip() != "",
                   "control is dead -- no kill below can be believed")

for mid, frm, to, why in MUTANTS:
    n = HOOK_SRC.count(frm)
    if n != 1:
        check(f"[4] {mid}: anchor is unique", False,
              f"found {n} occurrences -- a no-match replace looks identical to success")
        continue
    mutated = HOOK_SRC.replace(frm, to)
    if mutated == HOOK_SRC:
        check(f"[4] {mid}: the mutation changed bytes", False, "no-op replace")
        continue
    if not buildable(mutated):
        _bash_ok = _bash_parses(mutated)
        _leg = "bash -n" if not _bash_ok else "the heredoc compile() leg"
        check(f"[4] {mid}: UNSCORABLE -- the mutant does not build, so it cannot "
              "be scored as a kill", False, f"{_leg} rejected the mutant")
        continue
    if not control_ok:
        continue
    m_rc, m_log, m_err = drive(mutated)
    # THE MUTANT MUST STILL RUN CLEANLY. Without this, ANY mutant that crashes
    # the hook produces an empty log and scores as a kill -- so the cell would
    # be measuring "did I break the hook", not "is this guard load-bearing".
    check(f"[4] {mid}: the mutant still runs cleanly (rc=0), so an empty log means "
          "the GUARD caught it and not that the hook crashed", m_rc == 0,
          f"rc={m_rc} stderr={m_err.strip()[-160:]!r}")
    _detail = (f"the mutant exited rc={m_rc}, so the empty log says nothing about "
               "the guard -- see the rc assertion above"
               if m_rc != 0 else
               f"the mutant STILL produced a decision line ({m_log.strip()[:90]!r}) "
               "-- this guard is not the one doing the work")
    check(f"[4] {mid}: KILLED -- removing {why} makes the guard RED",
          m_rc == 0 and m_log.strip() == "", _detail)

assert_isolated("all mutant drives")

# ── [5] MUTATION OF THE SECTION-[1]/[2] GUARDS ──────────────────────────────
#
# WHY THIS SECTION EXISTS. Criterion 6 says "EVERY new guard is mutation-tested".
# Cycles 1 and 2 mutation-tested only the [3]/[4] guards, and the [1]/[2] guards
# -- preconditions, enumeration recall, classification keying -- had no cell at
# all. The cycle-2 Q/A ran four such mutations itself, found zero survivors, and
# correctly recorded the gap as coverage-and-disclosure rather than vacuity.
# Zero survivors is not the same as tested, so the cells are now SHIPPED, and
# they run through `analyse()` -- the same function the shipped assertions use,
# never a re-implementation.
#
# Each cell is a DIFFERENTIAL: the property must hold on the real hook and FAIL
# on the mutant. A cell that only checks the mutant proves nothing about whether
# the guard was ever satisfied to begin with.
print("\n[5] MUTATION of the [1]/[2] guards (criterion 6, 'every new guard')\n")

BASE = analyse(HOOK_SRC)
check("[5] CONTROL: the real hook satisfies all four [1]/[2] properties",
      not BASE["funcs"] and not BASE["reorder"] and not BASE["unexplained"]
      and not BASE["unclassified"],
      "the control is already violated -- no differential below can be believed")

_ins = HOOK_SRC.index("\nNEW_ROW=")   # a stable pre-detector insertion point


def _mutate(text: str) -> str:
    return HOOK_SRC[:_ins] + "\n" + text + HOOK_SRC[_ins:]


SECTION12_MUTANTS = [
    ("bash-function-defined", 'helper() {\n  echo hi\n}',
     "funcs", "the lexical rule stops being sound and [1] must say so"),
    ("trap-reorders-execution", 'trap "echo bye" EXIT',
     "reorder", "execution order no longer matches source order"),
    ("exit-the-RULE-under-matches", 'if true; then exit 0; fi   # same-line exit',
     "unexplained", "the dumber scan sees an exit the rule missed"),
    ("unclassified-pre-detector-exit",
     'if [ -n "${SOME_NEW_FLAG:-}" ]; then\n    exit 0\nfi',
     "unclassified", "a fourth early exit nobody has classified"),
]

for mid, injected, key, why in SECTION12_MUTANTS:
    mutated = _mutate(injected)
    if mutated == HOOK_SRC:
        check(f"[5] {mid}: the mutation changed bytes", False, "no-op insert")
        continue
    if not buildable(mutated):
        check(f"[5] {mid}: UNSCORABLE -- the mutant does not build", False,
              "cannot score a mutant that does not build")
        continue
    found = analyse(mutated)[key]
    check(f"[5] {mid}: KILLED -- {why}", bool(found),
          f"analyse()['{key}'] is still empty -- this guard is not doing the work")

# The classification must key on CONDITION TEXT, not position. Reword the
# recursion guard's condition and the entry must stop matching -- which is what
# makes "an unclassified exit FAILS" a real gate rather than a line-number
# coincidence.
_reworded = HOOK_SRC.replace('^chore: (auto-changelog|changelog drift)',
                             '^chore: (SOMETHING-ELSE-ENTIRELY)')
if _reworded == HOOK_SRC:
    check("[5] classification-keying: anchor is unique", False, "condition text not found")
else:
    check("[5] classification-keys-on-condition-text: KILLED -- rewording the "
          "recursion guard's condition makes it UNCLASSIFIED",
          bool(analyse(_reworded)["unclassified"]),
          "the entry still matched after its condition changed -- the table is "
          "keyed on something other than the condition text")

# And the isolation check itself: probe it with a snapshot that cannot match.
check("[5] isolation-check: KILLED -- a corrupted snapshot makes it report FALSE",
      not isolation_holds(b"__CORRUPTED_SNAPSHOT__"),
      "isolation_holds() returned True against a snapshot that cannot match -- "
      "it is not comparing what it claims to compare")
check("[5] isolation-check CONTROL: the true snapshot still reports TRUE",
      isolation_holds(real_before),
      "the control is broken, so the kill above proves nothing")

print()
if _failures:
    print(f"FAILED: {_pass} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL GREEN: {_pass} passed, 0 failed")
