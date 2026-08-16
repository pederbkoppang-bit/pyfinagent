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
BYTE-IDENTICAL (7,597 B, sha1 f7458a6ab1f5fe96), while an edit *inside* the
definition changes it (+24 B). So the extraction is live -- it is specifically
the call it cannot see. The mutant is therefore INVISIBLE, not merely surviving:
no assertion added to that file, however clever, could ever kill it.

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
  [4] MUTATION: control GREEN first, then kill the delete-the-call mutant;
      a mutant that does not BUILD is UNSCORABLE and FAILS

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
print("\n[1] PRECONDITIONS for the lexical rule (criterion 2)\n")

FUNC_RE = re.compile(r"^\s*(function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\(\)\s*\{")
funcs = [i + 1 for i, l in enumerate(LINES) if FUNC_RE.match(l)]
check("[1] the hook defines NO bash functions (so lexical order == execution order)",
      not funcs, f"functions at lines {funcs} -- a lexical rule is no longer sound")

REORDER_RE = re.compile(r"^\s*(trap|source|\.|eval)\s")
reorder = [i + 1 for i, l in enumerate(LINES) if REORDER_RE.match(l)]
check("[1] no trap / source / . / eval (nothing reorders execution)",
      not reorder, f"found at lines {reorder}")

HEREDOC_RE = re.compile(r"<<\s*'?([A-Za-z_][A-Za-z0-9_]*)'?\s*$")
heredocs = [(i + 1, m.group(1)) for i, l in enumerate(LINES) if (m := HEREDOC_RE.search(l))]
check("[1] exactly ONE heredoc -- the detector", len(heredocs) == 1,
      f"found {len(heredocs)}: {heredocs}")
DETECTOR_START = heredocs[0][0] if heredocs else 10**9
TERM = heredocs[0][1] if heredocs else "PYEOF"
term_lines = [i + 1 for i, l in enumerate(LINES) if l.rstrip() == TERM]
check("[1] the heredoc terminator is found exactly once", len(term_lines) == 1,
      f"'{TERM}' at {term_lines}")
DETECTOR_END = term_lines[0] if len(term_lines) == 1 else 10**9
check("[1] the detector region is non-empty and ordered",
      DETECTOR_START < DETECTOR_END, f"{DETECTOR_START}..{DETECTOR_END}")
print(f"       detector heredoc: lines {DETECTOR_START}..{DETECTOR_END} (terminator {TERM!r})")

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

RULE = re.compile(r"(?:^|;|&&|\|\|)\s*exit\b")
DUMB = re.compile(r"\bexit\b")


def in_detector(lineno: int) -> bool:
    return DETECTOR_START < lineno < DETECTOR_END


rule_hits = [(i + 1, l) for i, l in enumerate(LINES)
             if RULE.search(l) and not in_detector(i + 1)]
dumb_hits = [(i + 1, l) for i, l in enumerate(LINES)
             if DUMB.search(l) and not in_detector(i + 1)]

check("[2] the rule finds a non-zero number of exit paths (a scan that matches "
      "nothing is not a clean bill of health)", len(rule_hits) > 0,
      "the rule matched NOTHING -- it is broken, not the hook")

missed = [(n, l) for (n, l) in dumb_hits if (n, l) not in rule_hits]
unexplained = [(n, l) for (n, l) in missed if not l.lstrip().startswith("#")]
check("[2] every line the dumber scan finds but the rule does not is a comment "
      "(so the rule is not under-matching)", not unexplained,
      f"unexplained at {[n for n, _ in unexplained]}")

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
     "The recursion guard. The hook is re-entering itself: the commit it is "
     "looking at is the one IT created. Such a commit is by construction not a "
     "bump candidate, so there is no decision to explain. Logging it would "
     "double the log with entries about the logger. This is a BOUND to state, "
     "not a defect to fix -- and it accounts for essentially the whole "
     "commits-vs-lines gap (re-derived in [3])."),
    (r"! -f \"\$CHANGELOG\"", "MUST-LOG",
     "The CHANGELOG is missing entirely. That is machinery breakage, not a "
     "routine skip: every subsequent commit will silently produce nothing, and "
     "the decision log is the only place an operator would find out."),
    (r"### Recent Activity", "MUST-LOG",
     "The CHANGELOG exists but has lost its anchor section. Same class as "
     "above -- silent structural breakage that looks identical to 'nothing to "
     "do' from outside."),
]

pre = [(n, l) for (n, l) in rule_hits if n < DETECTOR_START]
post = [(n, l) for (n, l) in rule_hits if n > DETECTOR_END]
check("[2] at least one PRE-detector exit path exists (else this step's premise "
      "is void)", len(pre) > 0)
check("[2] at least one POST-detector exit path exists", len(post) > 0)


def guard_condition(lineno: int) -> str:
    """The nearest preceding `if`/`||` line -- the condition this exit serves."""
    for j in range(lineno - 1, max(0, lineno - 6), -1):
        text = LINES[j - 1]
        if re.search(r"^\s*(if|elif)\b", text) or "||" in text:
            return text.strip()
    return LINES[lineno - 1].strip()


print(f"       {len(pre)} pre-detector exit path(s), {len(post)} post-detector\n")
classified = 0
for n, _l in pre:
    cond = guard_condition(n)
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


def make_repo(tmp: Path) -> None:
    (tmp / "CHANGELOG.md").write_text(CHANGELOG_SEED, encoding="utf-8")
    (tmp / ".claude").mkdir(exist_ok=True)
    (tmp / ".claude" / "masterplan.json").write_text('{"phases": []}', encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "-A"],
                 ["commit", "-q", "-m", "feat: seed commit"]):
        subprocess.run(["git", *args], cwd=tmp, env=env, check=True,
                       capture_output=True)


def drive(hook_src: str, subject: str = "feat: a real change") -> tuple[int, str, str]:
    """Run the hook end-to-end. Returns (rc, decision-log text, stderr)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        make_repo(tmp)
        env = {**os.environ, "CLAUDE_PROJECT_DIR": str(tmp),
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        (tmp / "f.txt").write_text(subject, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=tmp, env=env, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", subject], cwd=tmp, env=env,
                       check=True, capture_output=True)
        hp = tmp / "hook.sh"
        hp.write_text(hook_src, encoding="utf-8")
        hp.chmod(0o755)
        r = subprocess.run(["bash", str(hp)], cwd=tmp, env=env,
                           capture_output=True, text=True, timeout=120)
        log = tmp / "handoff" / "logs" / "changelog-decisions.log"
        return r.returncode, (log.read_text(encoding="utf-8") if log.exists() else ""), r.stderr


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
check("[3] the decision line carries a reason", "reason=" in log_text,
      f"line: {log_text.strip()[:120]!r}")

real_after = real_log.read_bytes() if real_log.exists() else b""
check("[3] ISOLATION: the real repo's decision log is untouched by this driver",
      real_before == real_after,
      "the driver wrote into the real repo -- it is contaminating its own evidence")

# The recursion guard, driven for real: an auto-changelog subject must produce
# NO decision line. This is criterion 3's evidence, executed rather than argued.
rc2, log2, _ = drive(HOOK_SRC, subject="chore: auto-changelog hook entry for abc1234")
check("[3] recursion guard: an auto-changelog commit exits 0", rc2 == 0, f"rc={rc2}")
check("[3] recursion guard: and writes NO decision line (the BOUND, measured)",
      log2.strip() == "", f"unexpectedly logged: {log2.strip()[:120]!r}")

# Re-derive the commits-vs-lines gap AT EXECUTION TIME. Never pin the figure:
# the step was filed as "10 commits vs 5 lines" and that snapshot is already
# stale. The window is anchored to the log's own first timestamp -- a bare date
# would slide with the clock.
if real_before:
    first_stamp = real_before.decode("utf-8", "replace").splitlines()[0].split()[0]
    n_lines = len(real_before.decode("utf-8", "replace").strip().splitlines())
    out = subprocess.run(["git", "log", f"--since={first_stamp}", "--format=%s"],
                         cwd=REPO, capture_output=True, text=True, timeout=30).stdout
    subjects = [s for s in out.splitlines() if s.strip()]
    recursion = [s for s in subjects
                 if re.match(r"^chore: (auto-changelog|changelog drift)", s, re.I)]
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


def buildable(src: str) -> bool:
    """A mutant that does not BUILD is UNSCORABLE, never a kill."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.sh"
        p.write_text(src, encoding="utf-8")
        return subprocess.run(["bash", "-n", str(p)],
                              capture_output=True).returncode == 0


MUTANTS = [
    ("delete-the-production-call", "\n_log_decision(bump_type)\n", "\n",
     "the call the 86.91 extraction is structurally blind to"),
    ("neuter-the-log-write", 'with open(log_dir / "changelog-decisions.log", "a", encoding="utf-8") as _fh:',
     'with open(os.devnull, "a", encoding="utf-8") as _fh:',
     "the write itself, so the effect disappears without the call moving"),
]

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
        check(f"[4] {mid}: UNSCORABLE -- the mutant does not build, so it cannot "
              "be scored as a kill", False, "bash -n rejected the mutant")
        continue
    if not control_ok:
        continue
    m_rc, m_log, m_err = drive(mutated)
    check(f"[4] {mid}: KILLED -- removing {why} makes the guard RED",
          m_log.strip() == "",
          f"the mutant STILL produced a decision line ({m_log.strip()[:90]!r}) -- "
          "this guard is not the one doing the work")

print("")
if _failures:
    print(f"FAILED: {_pass} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"ALL GREEN: {_pass} passed, 0 failed")
