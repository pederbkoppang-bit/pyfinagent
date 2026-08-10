#!/usr/bin/env python3
"""phase-86.31 -- PROVE the Q/A write separation by DRIVING the real hook.

The step's demand is "read-only-on-the-work must survive intact, and that
separation is the whole difficulty, so prove it, do not assert it." A source
scan cannot do that: the hook is a bash script wrapping an embedded python
program, and the interesting behaviour is its EXIT CODE under a real payload
(2 = block, 0 = allow).

So this executes `.claude/hooks/qa-write-guard.sh` as a subprocess, feeds it
PreToolUse JSON on stdin, and asserts the exit code.

TWO THINGS IT CHECKS THAT AN OBVIOUS VERSION WOULD MISS
------------------------------------------------------
1. **LIVENESS FIRST.** The hook is FAIL-OPEN by design, and its python lives
   inside a bash single-quoted block -- one apostrophe anywhere makes it a
   SyntaxError, at which point it allows EVERYTHING while every deny-case
   assertion below still "passes" for the wrong reason. On 2026-08-10 that
   exact break happened. So the first cell compiles the embedded source, and
   a dead guard fails the run before any decision is graded.
2. **FALSE POSITIVES.** A guard that blocks everything would pass every deny
   case. Main and the researcher rail must remain able to write -- write-first
   is mandatory for the researcher, so over-blocking would break the other
   Layer-3 rail.

Exit 0 = separation proven; 1 = a case failed or the guard is not alive.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude/hooks/qa-write-guard.sh"

ALLOW, BLOCK = 0, 2

# (label, agent_type, tool, file_path, expected_exit, why)
CASES = [
    # ---- THE PERMIT: the verdict file, and only it ----------------------
    ("verdict file (the whole point of 86.31)", "qa", "Write",
     ".claude/agent-memory/qa/verdicts/verdict_wip_86.34.md", ALLOW,
     "a dropped Q/A must leave its completed evaluation on disk"),
    ("qa memory curation", "qa", "Write",
     ".claude/agent-memory/qa/MEMORY.md", ALLOW,
     "the frontmatter memory injection this hook exists to serve"),

    # ---- THE REFUSALS: read-only on the WORK ----------------------------
    ("production source", "qa", "Write",
     "backend/services/kill_switch.py", BLOCK,
     "an evaluator that edits the subject is not an evaluator"),
    ("the critique Main is scribe for", "qa", "Write",
     "handoff/current/evaluator_critique.md", BLOCK,
     "MEASURED historical breach: qa-80-2 wrote this file"),
    ("frontend source", "qa", "Edit",
     "frontend/src/lib/api.ts", BLOCK,
     "MEASURED historical breach: qa-80-2-c2 edited this exact path"),
    ("the experiment_results it is grading", "qa", "Edit",
     "handoff/current/experiment_results_86.34.md", BLOCK,
     "grading your own edited evidence"),

    # ---- THE IDENTITY CLASS (cycle-2 fix: name, not literal type) -------
    ("named spawn qa-86-34-c2", "qa-86-34-c2", "Write",
     "backend/services/kill_switch.py", BLOCK,
     "agent_type carries the SPAWN NAME; 27 such identities were unguarded"),
    ("named spawn qa_85_5_c3 (underscore)", "qa_85_5_c3", "Edit",
     "backend/main.py", BLOCK, "underscore variant of the same class"),
    ("named spawn QA-Upper (case)", "QA-Upper", "Write",
     "backend/main.py", BLOCK, "predicate lowercases before matching"),
    ("named spawn writes its OWN verdict", "qa-86-34-c2", "Write",
     ".claude/agent-memory/qa/verdicts/verdict_wip_86.34.md", ALLOW,
     "the permit must follow the ROLE, not only the literal name"),

    # ---- TRAVERSAL ------------------------------------------------------
    ("traversal out of the memory dir", "qa", "Write",
     ".claude/agent-memory/qa/../../../backend/services/kill_switch.py", BLOCK,
     "normpath collapses ../ so the segment check cannot be escaped"),

    # ---- FALSE POSITIVES: these MUST still be allowed -------------------
    ("Main (no agent_type)", "", "Write",
     "backend/services/kill_switch.py", ALLOW,
     "Main owns the fix; blocking it would brick every step"),
    ("researcher writing its brief", "researcher", "Write",
     "handoff/current/research_brief_86.38.md", ALLOW,
     "write-first is MANDATORY for the researcher rail"),
    ("qa READING (not Write/Edit)", "qa", "Read",
     "backend/services/kill_switch.py", ALLOW,
     "read-only means read is allowed"),
    ("an agent merely named qa-adjacent", "quality-auditor", "Write",
     "backend/main.py", ALLOW,
     "the predicate must not swallow unrelated agents"),
]


def drive(agent_type: str, tool: str, path: str) -> int:
    payload = json.dumps({
        "agent_type": agent_type,
        "tool_name": tool,
        "tool_input": {"file_path": path},
    })
    p = subprocess.run(["bash", str(HOOK)], input=payload, text=True,
                       capture_output=True, cwd=REPO,
                       env={"CLAUDE_PROJECT_DIR": str(REPO), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    return p.returncode


def guard_is_alive() -> tuple[bool, str]:
    """Compile the python embedded in the bash single-quoted block."""
    src = HOOK.read_text()
    m = re.search(r"python3 -c '(.*?)'\s*2>>", src, re.S)
    if not m:
        return False, "could not locate the embedded python block"
    body = m.group(1)
    if "'" in body:
        return False, "an apostrophe inside the single-quoted block would truncate it"
    try:
        compile(body, "<qa-write-guard embedded>", "exec")
    except SyntaxError as e:
        return False, f"embedded python does not compile: {e}"
    return True, "embedded python compiles and contains no apostrophe"


def main() -> int:
    print(f"hook: {HOOK.relative_to(REPO)}")

    alive, why = guard_is_alive()
    print(f"\n[LIVENESS] {'OK' if alive else 'DEAD'} -- {why}")
    if not alive:
        print("\nFAIL -- the guard is fail-open, so a dead guard ALLOWS EVERYTHING "
              "while every deny assertion below would pass for the wrong reason.")
        return 1

    # The liveness check must itself be able to fail, or it is decoration.
    broken = HOOK.read_text().replace("def is_qa_role(name):",
                                      "def is_qa_role(name):  # it's broken")
    m = re.search(r"python3 -c '(.*?)'\s*2>>", broken, re.S)
    if m and "'" not in m.group(1):
        print("[LIVENESS-CONTROL] FAILED -- injecting an apostrophe did not trip "
              "the check; the liveness probe is decorative")
        return 1
    print("[LIVENESS-CONTROL] OK -- an injected apostrophe DOES trip it")

    print(f"\n{'result':<8} {'exit':<5} {'expect':<7} case")
    print("-" * 96)
    failures = []
    for label, at, tool, path, expected, why in CASES:
        rc = drive(at, tool, path)
        ok = rc == expected
        if not ok:
            failures.append((label, rc, expected, why))
        verb = "ALLOW" if expected == ALLOW else "BLOCK"
        print(f"{'ok' if ok else 'FAIL':<8} {rc:<5} {verb:<7} {label}")
        if not ok:
            print(f"{'':<22} why it matters: {why}")

    allows = sum(1 for c in CASES if c[4] == ALLOW)
    blocks = len(CASES) - allows
    print("-" * 96)
    print(f"{len(CASES)} cases: {blocks} must BLOCK, {allows} must ALLOW "
          f"(both directions exercised, so neither always-allow nor "
          f"always-block can pass)")

    if failures:
        print(f"\nFAIL -- {len(failures)} case(s) wrong: {[f[0] for f in failures]}")
        return 1
    print("\nOK -- the Q/A can write its verdict file and NOTHING ELSE, across "
          "the literal type, named spawns, case and underscore variants, and "
          "path traversal; Main and the researcher rail are unaffected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
