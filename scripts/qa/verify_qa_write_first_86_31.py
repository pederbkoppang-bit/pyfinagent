#!/usr/bin/env python3
"""phase-86.31 -- re-runnable checker for the Q/A write-first verdict rail.

Answers, by EXECUTION rather than by reading source:

  [1] Does the guard ALLOW the Q/A's verdict-WIP path and DENY the four named
      classes -- for the qa ROLE, not merely the literal name "qa"?
  [2] Are Main and the RESEARCHER rail unaffected?
  [3] GUARD HEALTH -- does the hook still actually WORK? It is fail-open, so a
      syntax error in its embedded python presents as "allow everything".
  [4] Completion-marker semantics: is an INCOMPLETE artifact impossible to
      mistake for a COMPLETE one, and is NEITHER a verdict?
  [5] IDENTITY: can a PREVIOUS cycle's artifact be mistaken for this one's?
  [6] Prose anchors -- and they are checked for an intact IMPERATIVE, not for
      bare token presence.
  [7] `qa_wip.MEMORY_DIR` still agrees with the hook's own constant.
  [8] The `verdicts/` subdir is out of audit_memory.py's non-recursive glob.
  [9] BEHAVIOURAL: did the live rail actually produce WIP artifacts?

CYCLE-2 HISTORY, kept because it explains why [3] and [6] look the way they do.
The cycle-1 Q/A killed two of this file's own guards:

  * Section [1] drove the hook only with `agent_type == "qa"`, so it could not
    see that the hook's predicate was an EXACT match while `agent_type` carries
    the SPAWN NAME -- 27 qa-* identities had been writing past it, two of them
    into production frontend source and into evaluator_critique.md itself.
  * Section [6] asserted bare token presence, so replacing the directive with an
    ANTI-directive that PRESERVED every scanned literal left the checker ALL
    GREEN while the mechanism was dead (vacuity shapes #2/#3).

And fixing the first of those broke the hook: an apostrophe in a docstring
terminated the bash single-quoted block, python died, and the hook -- fail-open
BY DESIGN -- answered "allow" to everything. Nothing in this file noticed,
because every assertion was about DENY decisions that a broken hook cannot make
and an ALLOW decision it makes for the wrong reason. Section [3] exists so that
never passes silently again.

`--guard PATH` lets `mutation_matrix_86_31.py` drive a MUTATED COPY. Nothing
here mutates the repo; writes go to a `mkdtemp` sandbox torn down on exit.

    $ python scripts/qa/verify_qa_write_first_86_31.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
GUARD = REPO / ".claude" / "hooks" / "qa-write-guard.sh"

_spec = importlib.util.spec_from_file_location(
    "qa_wip_86_31", pathlib.Path(__file__).with_name("qa_wip.py"))
qa_wip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qa_wip)


def drive_guard(guard: pathlib.Path, agent_type: str, tool: str, file_path: str):
    """Run the PreToolUse hook exactly as Claude Code does: JSON on stdin."""
    payload = json.dumps({
        "agent_type": agent_type,
        "tool_name": tool,
        "tool_input": {"file_path": file_path},
    })
    proc = subprocess.run(
        ["bash", str(guard)], input=payload, text=True,
        capture_output=True, timeout=30,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin",
             "CLAUDE_PROJECT_DIR": str(REPO)},
    )
    return proc.returncode, proc.stdout, proc.stderr


def section(text: str, start: str, end: str) -> str:
    """The span [start, next `end` AFTER start) -- '' when start is missing.

    The end marker is searched FROM `start`, not from position 0. Searching
    from 0 made every runbook section unlocatable, because `\\n### ` occurs
    long before the block being extracted -- and an unlocatable section then
    failed six assertions that had nothing to do with the real content.
    """
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:j] if j > i else text[i:]


#: Words that turn a directive into its opposite while leaving every scanned
#: literal in place. The cycle-1 Q/A's mutant was exactly this shape:
#: "RETIRED 2026-08-11. Do NOT create ...verdict_wip_<step_id>.md ... Write
#: nothing at all." -- all four scanned tokens present, mechanism dead.
#: NOTE the omission of a bare "do not write": a CORRECT section says exactly
#: that about every OTHER path ("Do not write anything else. Not production
#: code, not tests..."), so blacklisting it flagged the real directive as its
#: own inversion. The list must name phrases that negate the WIP instruction
#: specifically, never phrases a faithful directive also uses.
ANTI_DIRECTIVE = [
    "do not create", "don't create", "write nothing at all", "do not write a wip",
    "retired", "deprecated", "obsolete", "no longer required",
    "no longer applies", "rescinded", "superseded", "ignore this section",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", type=pathlib.Path, default=GUARD)
    ns = ap.parse_args()
    guard = ns.guard

    failures: list[str] = []
    lines: list[str] = []
    npass = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal npass
        lines.append(("  PASS  " if ok else "  FAIL  ") + label
                     + (f"\n          {detail}" if detail else ""))
        if ok:
            npass += 1
        else:
            failures.append(label)

    if not guard.exists():
        print(f"FAILED -- guard not found: {guard}")
        return 1

    wip = qa_wip.resolve_wip_path("86.31")
    wip_rel = wip.relative_to(REPO)

    # ---- [3] GUARD HEALTH (run FIRST -- everything below assumes it works) --
    lines.append("\n[3] GUARD HEALTH -- the hook is FAIL-OPEN, so a broken hook 'allows'")
    gsrc = guard.read_text(encoding="utf-8")
    rc, _o, err = drive_guard(guard, "qa", "Write", str(REPO / "backend/main.py"))
    check("Traceback" not in err and "SyntaxError" not in err and "command not found" not in err,
          "the hook runs without interpreter or shell errors",
          f"stderr tail: {err.strip().splitlines()[-1] if err.strip() else '(clean)'}")
    # The embedded python lives inside a bash SINGLE-QUOTED block. One
    # apostrophe anywhere in it ends the string; python then gets a truncated
    # script, dies, and the hook answers 'allow' to everything. This exact
    # break was introduced and caught here on 2026-08-10.
    if "python3 -c '" in gsrc and "\n' 2>>" in gsrc:
        pyblock = gsrc.split("python3 -c '", 1)[1].split("\n' 2>>", 1)[0]
        offenders = [ln.strip()[:60] for ln in pyblock.splitlines() if "'" in ln]
        check(not offenders,
              "no apostrophe inside the single-quoted python block (it would fail the hook OPEN)",
              f"offending lines: {offenders}" if offenders else "")
        try:
            compile(pyblock, "<guard>", "exec")
            check(True, "the embedded python COMPILES")
        except SyntaxError as exc:
            check(False, "the embedded python COMPILES", f"{exc}")
    else:
        check(False, "the guard has the expected `python3 -c '...'` shape")
    check(subprocess.run(["bash", "-n", str(guard)], capture_output=True).returncode == 0,
          "bash -n parses the hook")

    # ---- [1] guard decisions ------------------------------------------------
    lines.append("\n[1] GUARD DECISIONS -- the qa ROLE, not just the literal name (verbatim)")
    lines.append(f"    guard: {guard}")

    # agent_type carries the SPAWN NAME. These are REAL names taken from
    # handoff/logs/qa_write_guard.log, not invented ones.
    QA_IDENTITIES = ["qa", "qa-80-2-c2", "qa-85-5-c3", "qa-36-12-cycle6", "qa_86_31", "QA-80-2"]

    ALLOWED = [
        ("verdict WIP for the step under evaluation", str(wip)),
        ("the WIP for ANOTHER step (same dir -- see the DISCLOSED RESIDUAL below)",
         str(qa_wip.resolve_wip_path("86.24"))),
        ("a Q/A memory file (curation must keep working -- regression)",
         str(REPO / ".claude/agent-memory/qa/feedback_probe_self_contamination.md")),
    ]
    DENIED = [
        ("production code", str(REPO / "backend/paper_trading/paper_trading_service.py")),
        ("production FRONTEND source (qa-80-2-c2 really did Edit this one)",
         str(REPO / "frontend/src/lib/api.ts")),
        ("a test file", str(REPO / "backend/tests/test_paper_trading_service.py")),
        ("the masterplan", str(REPO / ".claude/masterplan.json")),
        ("another step's handoff artifact",
         str(REPO / "handoff/current/experiment_results_86.24.md")),
        ("evaluator_critique.md (qa-80-2 really did Write this one -- Main is its scribe)",
         str(REPO / "handoff/current/evaluator_critique.md")),
        ("the guard itself (a guard that can rewrite itself is not a guard)", str(guard)),
        ("the Q/A's own operating instructions", str(REPO / ".claude/agents/qa.md")),
        ("traversal OUT of the memory dir (tests the normpath leg)",
         str(REPO / ".claude/agent-memory/qa/verdicts/../../../../backend/main.py")),
    ]

    for ident in QA_IDENTITIES:
        for label, path in ALLOWED:
            rc, out, err = drive_guard(guard, ident, "Write", path)
            check(rc == 0, f"ALLOW  [{ident}] {label}", f"exit={rc}")
    for ident in QA_IDENTITIES:
        for label, path in DENIED:
            for tool in ("Write", "Edit"):
                rc, out, err = drive_guard(guard, ident, tool, path)
                check(rc == 2, f"DENY   [{ident}] {tool:5s} {label}",
                      f"exit={rc} stderr={err.strip()!r}" if ident == "qa" else f"exit={rc}")

    # ---- [2] everyone else is unaffected ------------------------------------
    lines.append("\n[2] MAIN AND THE RESEARCHER RAIL ARE UNAFFECTED")
    for label, path in DENIED[:4]:
        rc, _o, _e = drive_guard(guard, "", "Write", path)
        check(rc == 0, f"Main (no agent_type) may still Write {label}", f"exit={rc}")
    for ident in ("researcher", "research-82-0", "res-78-1", "researcher-80-4-death"):
        rc, _o, _e = drive_guard(guard, ident, "Write",
                                 str(REPO / "handoff/current/research_brief_86.31.md"))
        check(rc == 0, f"the researcher rail [{ident}] still writes its brief", f"exit={rc}")

    # ---- [4] completion-marker semantics ------------------------------------
    lines.append("\n[4] COMPLETION-MARKER SEMANTICS (an INCOMPLETE artifact is inert)")
    sandbox = pathlib.Path(tempfile.mkdtemp(prefix="qa_wip_86_31_"))
    try:
        (sandbox / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR).mkdir(parents=True)
        target = qa_wip.resolve_wip_path("99.99", repo=sandbox)
        hdr = "\nSTEP: 99.99\nWRITTEN: 2026-08-10T12:00:00Z\n"

        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["status"] == "ABSENT" and not rep["recoverable"], "ABSENT before anything is written")

        target.write_text(qa_wip.INCOMPLETE_MARKER + hdr + "\n## findings\n- M4 survived\n")
        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["status"] == "INCOMPLETE", "born INCOMPLETE on first write")
        check(rep["recoverable"] is True, "an INCOMPLETE artifact IS recoverable as evidence")
        check(rep["is_verdict"] is False and "verdict" not in rep,
              "not a verdict, and carries no key to scrape as one", f"keys={sorted(rep)}")

        target.write_text(qa_wip.INCOMPLETE_MARKER + hdr + "\n## findings\n- M4 surv")
        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["status"] == "INCOMPLETE",
              "a TRUNCATED artifact stays INCOMPLETE (cannot be mistaken for COMPLETE)")

        target.write_text(qa_wip.COMPLETE_MARKER + hdr + "COMPLETED: 2026-08-10T12:09:00Z\n\n- done\n")
        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["status"] == "COMPLETE", "final act flips the marker to COMPLETE")
        check(rep["is_verdict"] is False and "verdict" not in rep,
              "even a COMPLETE artifact is NOT a verdict (crash-only)")
        check("NEVER PASS" in rep["guidance"], "the report restates NO VERDICT, NEVER PASS")

        target.write_text("PASS -- looks like a verdict\n")
        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["status"] == "UNMARKED", "a file with no marker is UNMARKED, not degraded to INCOMPLETE")

        for hostile in ("../../backend/main", "86.31/../../x", "..", "86.31; rm -rf /"):
            try:
                qa_wip.resolve_wip_path(hostile, repo=sandbox)
                check(False, f"hostile step id refused: {hostile!r}", "resolver RETURNED a path")
            except qa_wip.BadStepId:
                check(True, f"hostile step id refused: {hostile!r}")

        # ---- [5] IDENTITY: a prior cycle's artifact must not read as current
        lines.append("\n[5] IDENTITY -- a PREVIOUS cycle's artifact cannot pose as this one's")
        target.write_text(qa_wip.COMPLETE_MARKER
                          + "\nSTEP: 99.99\nWRITTEN: 2026-08-10T10:00:00Z\n"
                          + "COMPLETED: 2026-08-10T10:11:00Z\n\n- cycle-1 findings, PRE-FIX\n")
        rep = qa_wip.report("99.99", repo=sandbox, spawned_at="2026-08-10T12:00:00Z")
        check(rep["status"] == "STALE",
              "a COMPLETE artifact written BEFORE the spawn is STALE, not current",
              f"written_at={rep['written_at']} spawned_at=2026-08-10T12:00:00Z -> {rep['status']}")
        check(rep["recoverable"] is False, "a STALE artifact is NOT offered as recoverable evidence")
        check("PREVIOUS cycle" in rep["guidance"], "the guidance names it as a previous cycle's")

        rep = qa_wip.report("99.99", repo=sandbox, spawned_at="2026-08-10T09:00:00Z")
        check(rep["status"] == "COMPLETE",
              "the SAME artifact is COMPLETE when the spawn predates it (no false positive)")

        target.write_text(qa_wip.COMPLETE_MARKER + "\nSTEP: 99.99\n\n- no stamp\n")
        rep = qa_wip.report("99.99", repo=sandbox, spawned_at="2026-08-10T12:00:00Z")
        check(rep["status"] == "IDENTITY_UNKNOWN" and rep["recoverable"] is False,
              "an artifact with NO written stamp is IDENTITY_UNKNOWN, not silently trusted")

        rep = qa_wip.report("99.99", repo=sandbox)
        check(rep["identity_checked"] is False,
              "omitting spawned_at is REPORTED as identity_checked=false, never assumed safe")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    # ---- [6] prose anchors: the IMPERATIVE, not bare tokens -----------------
    lines.append("\n[6] PROSE ANCHORS -- an intact IMPERATIVE, not bare token presence")
    qa_md = (REPO / ".claude/agents/qa.md").read_text(encoding="utf-8")
    js = (REPO / ".claude/workflows/qa-verdict.js").read_text(encoding="utf-8")
    runbook = (REPO / "docs/runbooks/per-step-protocol.md").read_text(encoding="utf-8")

    SECTIONS = [
        (".claude/agents/qa.md :: write-first section",
         section(qa_md, "## Write-first for your VERDICT FILE ONLY",
                 "## Verification order (deterministic FIRST)"),
         1200,
         ["verdict_wip_", ".claude/agent-memory/qa/verdicts/", "STATUS: INCOMPLETE",
          "STATUS: COMPLETE", "Append findings", "final act",
          # phase-86.36: the RUN STAMP. Without this needle the directive can
          # revert to the fixed per-step filename and every anchor still passes
          # -- which is exactly what happened to the qa-verdict.js copy below
          # and was caught by a Q/A, not by this checker.
          "__<STAMP>", "%Y%m%dT%H%M%SZ"],
         [r"\*\*Create\*\*|^\s*1\.\s+\*\*Create\*\*"]),
        (".claude/workflows/qa-verdict.js :: STEP 0b",
         section(js, "'STEP 0b (binding, phase-86.31", "  'You are INDEPENDENT of the author"),
         900,
         ["verdict_wip_", ".claude/agent-memory/qa/verdicts/", "STATUS: INCOMPLETE",
          "STATUS: COMPLETE", "FINAL act",
          # phase-86.36 blocker B1: THIS copy drifted. qa.md was updated to the
          # stamped path and this one was not, so the PRIMARY launch path kept
          # injecting the destructive fixed filename plus the falsified premise
          # "the path is FIXED per step". Every anchor above passed on that
          # stale text. These two needles are the guard that was missing.
          "__<STAMP>", "%Y%m%dT%H%M%SZ"],
         [r"create\s*$|verdict_wip_'"]),
        ("docs/runbooks/per-step-protocol.md :: recovery contract",
         section(runbook, "**RECOVERY AFTER A DROPPED Q/A", "\n### "),
         900,
         ["verdict_wip_", "scripts/qa/qa_wip.py", "NO VERDICT, NEVER PASS", "--spawned-at"],
         [r"EVIDENCE FOR THE NEXT SPAWN, NEVER A VERDICT"]),
    ]
    for name, body, floor, needles, patterns in SECTIONS:
        check(bool(body), f"{name}: the section is locatable")
        check(len(body) >= floor,
              f"{name}: section is >= {floor} chars (an anti-directive stub is short)",
              f"actual={len(body)}")
        for n in needles:
            check(n in body, f"{name}: carries {n!r}")
        for pat in patterns:
            check(re.search(pat, body, re.M | re.I) is not None,
                  f"{name}: the IMPERATIVE survives -- /{pat}/ matches")
        hits = [w for w in ANTI_DIRECTIVE if w in body.lower()]
        check(not hits,
              f"{name}: contains NO anti-directive language",
              f"found: {hits}" if hits else "")

    # ---- [7] the duplicated constant still agrees ---------------------------
    lines.append("\n[7] qa_wip.MEMORY_DIR AGREES WITH THE HOOK'S OWN CONSTANT")
    check(f'MEMORY_DIR = "{qa_wip.MEMORY_DIR}"' in gsrc,
          f"the hook defines MEMORY_DIR = {qa_wip.MEMORY_DIR!r}")

    # ---- [8] subdir invisible to the memory auditor -------------------------
    lines.append("\n[8] THE verdicts/ SUBDIR IS OUT OF audit_memory.py's NON-RECURSIVE GLOB")
    am = (REPO / "scripts/housekeeping/audit_memory.py").read_text(encoding="utf-8")
    check('root.glob("*.md")' in am,
          "audit_memory.py globs *.md NON-recursively, so one level down is invisible",
          "a top-level WIP adds NO POINTER + MALFORMED FRONTMATTER (measured 2026-08-10)")
    check(str(wip_rel).startswith(f"{qa_wip.MEMORY_DIR}{qa_wip.WIP_SUBDIR}/"),
          f"the resolved WIP path sits under {qa_wip.MEMORY_DIR}{qa_wip.WIP_SUBDIR}/",
          f"resolved: {wip_rel}")

    # ---- [9] BEHAVIOURAL: did the live rail actually write anything? --------
    lines.append("\n[9] BEHAVIOURAL EVIDENCE -- what the LIVE rail actually produced")
    live = sorted((REPO / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR).glob("verdict_wip_*.md")) \
        if (REPO / qa_wip.MEMORY_DIR / qa_wip.WIP_SUBDIR).is_dir() else []
    # THE FLOOR IS THE WHOLE POINT (cycle-2 finding A). This section used to
    # loop over whatever artifacts happened to exist and assert nothing when
    # there were none -- so in the EXACT state it is offered to detect (the
    # directive silently disabled, therefore no artifact produced) it passed
    # green, while emitting zero assertions. Reproduced hermetically: deleting
    # every artifact left the checker at exit=0 with 0 red. That is qa.md
    # section 4c shape #5 applied to the mitigation itself.
    #
    # A RED [9] ON A TREE WHERE NO Q/A HAS EVER RUN IS CORRECT, NOT A BUG: it
    # says the rail has not yet demonstrated write-first, which is exactly what
    # a reader needs to know. There is deliberately NO opt-out flag -- an
    # escape hatch here would re-create the vacuity it exists to remove.
    check(len(live) >= 1,
          "the LIVE rail has produced at least one WIP artifact",
          f"found {len(live)} under {qa_wip.MEMORY_DIR}{qa_wip.WIP_SUBDIR}/ -- if zero, either no "
          "Q/A has run since the directive shipped, or the directive is not reaching the agent. "
          "Section [6] is a text scan and cannot tell those apart; this can.")
    stamps = []
    for p in live:
        txt = p.read_text(encoding="utf-8", errors="replace")
        st = qa_wip.classify(txt)
        h = qa_wip.headers(txt)
        stamps.append(h.get("WRITTEN", ""))
        check(st in ("COMPLETE", "INCOMPLETE"),
              f"live artifact {p.name} carries a valid marker",
              f"status={st} written={h.get('WRITTEN','(none)')} bytes={len(txt)}")
    # WHAT [9] DOES **NOT** PROVE, said plainly rather than left for a reader to
    # discover: the floor above is satisfied by ANY artifact, including a stale
    # one. So [9] proves the rail HAS produced write-first artifacts; it does
    # NOT prove it produced one for the step currently under evaluation. It
    # therefore catches "the directive never reached the agent" but not "the
    # directive stopped reaching the agent while old artifacts linger". The
    # newest stamp is printed so that gap is visible rather than implicit.
    newest = max([s for s in stamps if s], default="")
    lines.append(f"  NOTE  newest WRITTEN stamp on disk: {newest or '(none carry one)'}")
    lines.append("  NOTE  [9] proves the rail HAS produced artifacts, NOT that it produced one")
    lines.append("        for THIS step -- a lingering old artifact satisfies the floor.")

    print("\n".join(lines))
    print("\nDISCLOSED RESIDUALS (stated rather than smoothed over):")
    print("  R1 The guard is a DIRECTORY allowlist, not a single-path one. It denies every")
    print("     class criterion 1 names and the whole tree outside the memory dir, but the")
    print("     Q/A can still write other files INSIDE its own memory dir. That is the")
    print("     pre-existing memory-curation grant; narrowing it would kill curation.")
    print("     Nothing in .claude/hooks/ reads that directory, so no write there has authority.")
    print("  R2 agent_type values 'workflow-subagent' (80 events) and 'general-purpose' (22)")
    print("     are NOT matched by the qa-role predicate and remain unguarded. They are")
    print("     indistinguishable from LEGITIMATE writers (the researcher rail launches under")
    print("     such names and write-first is mandatory for it). Covering control: Main's")
    print("     post-verdict `git status` rule. Queued as its own masterplan step.")
    print("  R3 Section [6] is still a TEXT SCAN. It now kills the reword-inversion class the")
    print("     cycle-1 Q/A demonstrated, but no scan is proof against every rewrite. The only")
    print("     non-circular evidence that the directive reaches the agent is section [9].")

    print(f"\n{'ALL GREEN' if not failures else 'FAILED'} -- {npass} passed, {len(failures)} failed")
    for f in failures:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
