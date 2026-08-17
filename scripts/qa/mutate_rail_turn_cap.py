#!/usr/bin/env python3
"""Mutation matrix for `scripts/qa/rail_turn_cap.py`. Re-runnable, zero tree writes.

WHY THIS FILE EXISTS RATHER THAN A PARAGRAPH IN A HANDOFF NOTE
--------------------------------------------------------------
phase-86.84 cycle-2 Q/A, finding V-1: the cycle-1 mutation matrix existed only as
three lines of commit-message prose -- no control observation, no per-cell record,
no restore evidence, and nothing anyone could re-run. Criterion 8 of the step
requires mutation-testing every new guard with the control observed GREEN first
and survivors REPORTED rather than dropped. A matrix that cannot be re-executed
cannot satisfy that, so the matrix is now code.

    python3 scripts/qa/mutate_rail_turn_cap.py           # run the matrix
    python3 scripts/qa/mutate_rail_turn_cap.py --verify  # exit 1 on a real survivor

TWO PROBE DEFECTS THIS HARNESS DELIBERATELY AVOIDS, BOTH MEASURED
------------------------------------------------------------------
1. STALE-DATA SURVIVORS. The cap is resolved at collect() time, not analyse()
   time. The cycle-2 Q/A's first pass reused one cached `data` across cells and
   every timeline cell falsely SURVIVED. Each cell here re-runs collect() from
   scratch. A harness that reuses the corpus cannot test the corpus reader.
2. VACUOUS KILLS. `killed = rc != 0` scores a typo'd selector as a KILL (the
   project's pytest-exit-5 lesson). Here a cell is KILLED only when verify()
   returns False AND at least one problem string is produced, and the control is
   asserted GREEN before any mutant runs. A cell that errors out is recorded as
   ERROR, never as a kill.

WHAT A SURVIVOR MEANS
---------------------
Not every survivor is a defect. An EQUIVALENT mutant changes source without
changing behaviour (e.g. moving the cap boundary past a corpus that already
entirely precedes it). Those are labelled and explained. A NON-EQUIVALENT
survivor is a real hole and `--verify` fails on it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TARGET = REPO / "scripts" / "qa" / "rail_turn_cap.py"
AGENTS = REPO / ".claude" / "agents"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fresh_module():
    """Load rail_turn_cap.py into a NEW module object every time.

    Never cache: the cap is baked into each spawn record at collect() time, so a
    reused module or a reused corpus silently makes mutants look survivable.
    """
    spec = importlib.util.spec_from_file_location("rtc_mut", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mirror_agents() -> Path:
    """A throwaway copy of .claude/agents (and, since cycle 10, the workflow
    scripts) that a cell may freely mutate.

    The workflow files joined the mirror when verify() gained the
    orphan-classifier coupling pin, which reads them through mod.REPO -- a
    mirror without them made the CONTROL itself red (harness environment, not
    subject). The mirror is a faithful mini-.claude for everything the
    subject reads via REPO."""
    tmp = Path(tempfile.mkdtemp(prefix="rtc_mut_"))
    (tmp / ".claude" / "agents").mkdir(parents=True)
    for md in AGENTS.glob("*.md"):
        shutil.copy(md, tmp / ".claude" / "agents" / md.name)
    (tmp / ".claude" / "workflows").mkdir(parents=True)
    for js in (REPO / ".claude" / "workflows").glob("*.js"):
        shutil.copy(js, tmp / ".claude" / "workflows" / js.name)
    return tmp


def run_cell(mutate) -> tuple[bool, list[str], dict]:
    """Apply `mutate(mod, agents_dir)`, then collect+analyse+verify from scratch."""
    mod = fresh_module()
    tmp = mirror_agents()
    try:
        mod.REPO = tmp
        if mutate is not None:
            mutate(mod, tmp / ".claude" / "agents")
        analysis = mod.analyse(mod.collect())
        ok, problems = mod.verify(analysis)
        return ok, problems, analysis
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ── phase-86.84 cycle-5: SOURCE-level cells over the cycle-4 additions ──────
# The cycle-4 Q/A proved the 22 attribute-level cells above never touch
# analyse()'s new code (per-spawn era-correct caps, post_removal_turns, _q).
# These cells mutate the SOURCE TEXT of a temp copy -- the real tree is never
# written -- so the new code paths themselves are under the matrix.
def _inject_synthetic(mod, kind: str) -> None:
    """Wrap collect() to append ONE synthetic post-removal qa spawn.

    kind='non_emitter': a COMPLETED spawn that never emitted StructuredOutput
    -- the true new-loss signal. The live corpus has zero of these, so a
    reporter that hardcodes zero is EQUIVALENT ON THE CORPUS and only an
    injected known truth can distinguish it (fixture-must-break-the-symmetry).

    kind='errored': a completed-status spawn whose agent entry carries a
    server-side error (the 529 shape that turned this step's own immutable
    command red at cycle 7). The floor must stay GREEN on it -- cell S12.

    kind='killed': a spawn from a KILLED run -- an operator abort at 12 turns,
    nowhere near any cap. phase-86.84 cycle-5 proved the first non-emitter
    floor conflated this with a genuine loss (the F4 class, recommitted); the
    fixed floor must stay GREEN on it, and cell S11 pins that.
    """
    if kind == "wf_comment_evasion":
        # cycle-11 (cycle-10 Q/A F1, the executed MUT-A): rename the EMITTED
        # header on every non-comment line of the MIRRORED qa-verdict.js while
        # retaining the retired literal in a // comment -- house style already
        # quotes retired text in comments (criterion 5), so this is the
        # realistic drift path. The non-comment-line pin must REDDEN; the old
        # bytes pin stayed green here, which was the survivor.
        wfp = Path(mod.REPO) / ".claude" / "workflows" / "qa-verdict.js"
        s = wfp.read_text(encoding="utf-8")
        lines = s.splitlines()
        n_replaced = 0
        for i, ln in enumerate(lines):
            if ("IMMUTABLE SUCCESS CRITERIA" in ln
                    and not ln.lstrip().startswith(("//", "*", "/*"))):
                lines[i] = ln.replace("IMMUTABLE SUCCESS CRITERIA",
                                      "BINDING ACCEPTANCE TESTS")
                n_replaced += 1
        assert n_replaced >= 1, "evasion mutation matched nothing -- fixture dead"
        lines.append("// IMMUTABLE SUCCESS CRITERIA -- retired header retained "
                     "in a comment (the MUT-A evasion, quoted per house style)")
        wfp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    orig = mod.collect
    killed = kind == "killed"
    errored = kind == "errored"

    if kind == "unclassified_orphan":
        # cycle-11 (S18's first run): the post-removal unclassified count is
        # genuinely 0 on today's corpus, so hardcoding 0 was EQUIVALENT ON
        # THE CORPUS -- the same symmetry class this docstring already names
        # for non_emitter. Plant ONE post-removal role=None orphan so the
        # recount wants 1 and a hardcoded 0 misreports. The same plant makes
        # S17 corpus-independent (the 41 real pre-removal orphans age out
        # ~30d, after which S17 unplanted would degrade to equivalent -- the
        # S14 fragility, closed here by construction instead of by a note).
        def wrapped_orphan():
            data = orig()
            data.setdefault("erased_transcripts", []).append({
                "run_id": "wf_synthetic-orphan", "role": None,
                "post_removal": True, "agent_type": None,
                "session_started_at": "2026-08-17T00:00:00Z",
                "non_emitter": True,
            })
            return data

        mod.collect = wrapped_orphan
        return

    def wrapped():
        data = orig()
        data["spawns"].append({
            "run_id": "wf_synthetic-cell",
            "status": "killed" if killed else "completed",
            "errored": errored,
            "dropped": False, "completed": not killed, "killed": killed,
            "agent_type": "qa", "cap": None, "post_removal": True,
            "session_started_at": "2026-08-17T00:00:00Z", "model": "synthetic",
            "turns": 12 if killed else 35,
            "assistant_lines": 12 if killed else 35, "tool_calls": None,
            "tokens": 0, "structured_output": False,
            "timestamp": "2026-08-17T00:00:00Z", "workflow": "synthetic",
        })
        return data

    mod.collect = wrapped


def run_source_cell(old: str | None, new: str | None,
                    inject: str | None = None) -> tuple[bool, list[str], dict]:
    """collect+analyse+verify a SOURCE-mutated temp copy of rail_turn_cap.py.

    A missing anchor raises (a no-match replace that silently grades the
    unmutated source is the operations-that-cannot-fail-loudly trap).
    `old is None` runs the unmutated source (used by the injection control).
    """
    src = TARGET.read_text(encoding="utf-8")
    if old is not None:
        if old not in src:
            raise AssertionError(f"source anchor not found: {old[:70]!r}")
        src = src.replace(old, new, 1)
    tmpdir = Path(tempfile.mkdtemp(prefix="rtc_src_"))
    agents_tmp = None
    try:
        mfile = tmpdir / "rail_turn_cap_mut.py"
        mfile.write_text(src, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("rtc_srcmut", mfile)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agents_tmp = mirror_agents()
        mod.REPO = agents_tmp
        if inject:
            _inject_synthetic(mod, inject)
        analysis = mod.analyse(mod.collect())
        ok, problems = mod.verify(analysis)
        return ok, problems, analysis
    finally:
        if agents_tmp is not None:
            shutil.rmtree(agents_tmp, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)


#: (id, description, old_anchor, new_text, inject, kill_mode, expect)
#: kill_mode: "VERIFY" -- killed iff verify() goes red with a reason;
#:            "ORACLE" -- killed iff post_removal_turns differs from the
#:                        CONTROL's over the same immutable corpus (report
#:                        fidelity: the published numbers must be the ones the
#:                        unmutated computation produces);
#:            "INJECTED_TRUTH" -- killed iff the report shows the injected
#:                        non-emitter (a reporter that hides a known-planted
#:                        signal is the mutant; showing it is the kill).
SOURCE_CELLS = [
    ("S1", "D1 revert: row cap taken from group[0] again",
     "caps_present[0]\n                    if len(caps_present) == 1\n                    else (caps_present or None)",
     'group[0]["cap"]', False, "VERIFY", "KILL"),
    ("S2", "non-emitter counter floods (counts every post-removal spawn)",
     '"non_emitters": sum(\n                    1 for s in g\n                    if not s["structured_output"]\n                    and not s["killed"] and not s["dropped"]\n                    and not s.get("errored")\n                ),',
     '"non_emitters": len(g),', None, "VERIFY", "KILL"),
    ("S3", "_q zeroed -- percentiles all report 0",
     "return sorted_vals[int(frac * (len(sorted_vals) - 1))]",
     "return 0", False, "VERIFY", "KILL"),
    ("S4", "post-removal role filter broken -- the sample empties",
     'g = [s for s in post_removal_spawns if s["agent_type"] == role]',
     'g = [s for s in post_removal_spawns if s["agent_type"] == role + "_x"]',
     False, "VERIFY", "KILL"),
    ("S5", "past_old_cap comparison inverted (> becomes <)",
     '"past_old_cap": sum(1 for t in turns if t > old_cap),',
     '"past_old_cap": sum(1 for t in turns if t < old_cap),',
     False, "ORACLE", "KILL"),
    ("S6", "INJECTION CONTROL: one synthetic post-removal non-emitter, source unmutated -- the new floor must fire",
     None, None, "non_emitter", "VERIFY", "KILL"),
    ("S7", "injection + non_emitters hardcoded to 0 -- the report hides a planted signal",
     '"non_emitters": sum(\n                    1 for s in g\n                    if not s["structured_output"]\n                    and not s["killed"] and not s["dropped"]\n                    and not s.get("errored")\n                ),',
     '"non_emitters": 0,', "non_emitter", "INJECTED_TRUTH", "KILL"),
    # ── cycle-6 cells (the cycle-5 FAIL's named gaps) ───────────────────────
    ("S8", "percentile reversed (frac -> 1-frac) -- p90 becomes p10",
     "return sorted_vals[int(frac * (len(sorted_vals) - 1))]",
     "return sorted_vals[int((1 - frac) * (len(sorted_vals) - 1))]",
     None, "VERIFY", "KILL"),
    ("S9", "median reported as the maximum -- the monotone floor must catch it",
     '"median_turns": _q(turns, 0.5),',
     '"median_turns": turns[-1] if turns else 0,',
     None, "VERIFY", "KILL"),
    ("S10", "non-emitter narrowed to dropped-only -- a planted completed "
            "non-emitter is hidden (the third hiding shape, after flood and "
            "hardcode)",
     '''if not s["structured_output"]
                    and not s["killed"] and not s["dropped"]''',
     '''if not s["structured_output"]
                    and not s["killed"] and s["dropped"]''',
     "non_emitter", "INJECTED_TRUTH", "KILL"),
    # cycle-5 Invalid_Precondition, pinned as a NEGATIVE control: an operator
    # abort (killed run, 12 turns) must NOT redden the immutable command. The
    # first floor conflated it; the fixed floor names killed separately.
    ("S11", "KILLED-RUN INJECTION CONTROL: source unmutated, verify must STAY "
            "green -- an operator abort is not a new loss mechanism",
     None, None, "killed", "MUST_STAY_GREEN", "KILL"),
    # ── cycle-8 cells (the cycle-7 FAIL: 529-errored spawns counted as
    # non-emitters; the exclusion list must cover the CLASS) ────────────────
    ("S12", "ERRORED-SPAWN INJECTION CONTROL: source unmutated, verify must "
            "STAY green -- a server-side API error is not a new loss mechanism",
     None, None, "errored", "MUST_STAY_GREEN", "KILL"),
    ("S15", "role-marker constant drifted -- the classifier and its pin move "
            "together (single-sourced), so verify must redden against the "
            "workflow files that still emit the undrifted literal",
     '"qa": ("IMMUTABLE SUCCESS CRITERIA", ".claude/workflows/qa-verdict.js")',
     '"qa": ("IMMUTABLE SUCCESS CRITERIA DRIFTED", ".claude/workflows/qa-verdict.js")',
     None, "VERIFY", "KILL"),
    # ── cycle-11 (cycle-10 Q/A): the two executed survivors, kept permanent.
    ("S16", "emitted header renamed while the retired literal survives ONLY "
            "in // comments (the cycle-10 Q/A's executed MUT-A) -- the "
            "non-comment-line pin must redden where the bytes pin stayed green",
     None, None, "wf_comment_evasion", "VERIFY", "KILL"),
    ("S17", "erased_unclassified hardcoded to 0 (the cycle-10 Q/A's executed "
            "MUT-C) -- the verify() recount from erased_transcripts must "
            "catch the misreport; a synthetic orphan is planted so the kill "
            "does not depend on the 41 real pre-removal orphans that age out",
     '"erased_unclassified": sum(\n            1 for e in data.get("erased_transcripts", [])\n            if e.get("role") is None),',
     '"erased_unclassified": 0,',
     "unclassified_orphan", "VERIFY", "KILL"),
    ("S18", "erased_unclassified_post_removal hardcoded to 0 -- same class, "
            "the post-removal half. On the bare corpus this value IS 0, so "
            "the first run of this cell SURVIVED as equivalent-on-corpus "
            "(fixture-must-break-the-symmetry); the planted post-removal "
            "role=None orphan breaks the symmetry",
     '"erased_unclassified_post_removal": sum(\n            1 for e in data.get("erased_transcripts", [])\n            if e.get("role") is None and e.get("post_removal")),',
     '"erased_unclassified_post_removal": 0,',
     "unclassified_orphan", "VERIFY", "KILL"),
    ("S18b", "paired control for the orphan plant -- UNMUTATED source with "
             "the same planted orphan: stored aggregates and recount both "
             "see the plant, so verify must stay green (proves S17/S18 kill "
             "the MUTATION, not the plant)",
     None, None, "unclassified_orphan", "MUST_STAY_GREEN", "KILL"),
    ("S14", "erased-transcript accounting neutered -- the orphan glob stops "
            "seeing re-dispatch-erased attempts. ORACLE kill against a REAL "
            "present signal: the live corpus carries the two 529-killed "
            "evaluator attempts of this very step (qa erased_n=2), so the "
            "mutant's remediation visibly loses them. NOTE this cell's "
            "discriminating signal lives in the rotating corpus; when those "
            "transcripts age out (~30d) the cell degrades to equivalent and "
            "must be re-pointed or retired -- stated here so the label cannot "
            "silently outlive the run (the M14 lesson).",
     "        for tr in tdir.glob(\"agent-*.jsonl\"):",
     "        for tr in []:",
     None, "ORACLE", "KILL"),
    ("S13", "errored exclusion removed + errored spawn planted -- the cycle-7 "
            "defect reproduced as a permanent cell; verify must go RED under "
            "the mutant (the false positive is the visible malfunction)",
     '                    if not s["structured_output"]\n                    and not s["killed"] and not s["dropped"]\n                    and not s.get("errored")',
     '                    if not s["structured_output"]\n                    and not s["killed"] and not s["dropped"]',
     "errored", "VERIFY", "KILL"),
]


# ── cells ───────────────────────────────────────────────────────────────────
def _write_pin(agents: Path, role: str, line: str) -> None:
    f = agents / f"{role}.md"
    t = f.read_text()
    assert "model: opus\n" in t, f"anchor missing in {role}.md"
    f.write_text(t.replace("model: opus\n", f"model: opus\n{line}\n", 1))


CELLS = [
    # (id, description, mutate, expectation)
    ("M4r", "qa pin restored, bare `maxTurns: 30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 30"), "KILL"),
    ("M5r", "qa pin restored at a different value, 60",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 60"), "KILL"),
    ("M9", "researcher pin restored alone, 40",
     lambda m, a: _write_pin(a, "researcher", "maxTurns: 40"), "KILL"),
    ("M8", "no space after the colon, `maxTurns:30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns:30"), "KILL"),
    ("M7c", "space before the colon, `maxTurns : 30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns : 30"), "KILL"),
    # V-5: the two shapes that SURVIVED the regex guard in cycle 2.
    ("M7b", "pin with a trailing YAML comment, `maxTurns: 30  # restored`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 30  # restored"), "KILL"),
    ("M7", "quoted scalar, `maxTurns: \"30\"`",
     lambda m, a: _write_pin(a, "qa", 'maxTurns: "30"'), "KILL"),
    # F-C (cycle 3): these read as UNCAPPED on the regex fallback, which is the
    # branch the SHIPPED command takes -- bare `python3` here has no PyYAML.
    # They must be killed on BOTH paths, so they are permanent cells now.
    ("M15", "YAML tag, `maxTurns: !!int 30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: !!int 30"), "KILL"),
    ("M16", "anchor, `maxTurns: &cap 30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: &cap 30"), "KILL"),
    ("M17", "hex, `maxTurns: 0x1e`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 0x1e"), "KILL"),
    ("M18", "tab separator, `maxTurns:\\t30`",
     lambda m, a: _write_pin(a, "qa", "maxTurns:\t30"), "KILL"),
    ("M19", "float, `maxTurns: 30.0`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 30.0"), "KILL"),
    ("M20", "zero is still a pin, `maxTurns: 0`",
     lambda m, a: _write_pin(a, "qa", "maxTurns: 0"), "KILL"),
    # Timeline constants must be load-bearing. phase-86.84 F-E retargeted these
    # from the retired CAP_REMOVED_AT to CAP_EDIT_AT, the constant the disk-derived
    # boundary actually reads. Leaving them on the old name would have made all
    # three cells INERT -- setattr would have created an attribute nothing reads,
    # and an inert mutation proves nothing about the code that replaced it.
    ("M11", "CAP_EDIT_AT moved before the corpus (2026-01-01)",
     lambda m, a: setattr(m, "CAP_EDIT_AT", "2026-01-01T00:00:00Z"), "KILL"),
    ("M11b", "CAP_EDIT_AT moved mid-corpus (2026-08-01)",
     lambda m, a: setattr(m, "CAP_EDIT_AT", "2026-08-01T00:00:00Z"), "KILL"),
    ("M12", "HISTORICAL_CAPS qa 30 -> 31 (off by one, high)",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("qa", 31), "KILL"),
    ("M12b", "HISTORICAL_CAPS qa 30 -> 29 (off by one, low)",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("qa", 29), "KILL"),
    ("M13", "HISTORICAL_CAPS researcher 40 -> 41",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("researcher", 41), "KILL"),
    # phase-86.84 cycle-5 (Q/A finding): this cell was annotated EQUIVALENT while
    # the whole corpus predated the boundary. Since the first post-removal run
    # landed on disk (2026-08-14T19:35Z) that is no longer true -- moving the
    # boundary to 2027 reclassifies 59 post-removal spawns as capped-era, C2
    # fires ("capped spawns exceed their cap"), and the cell KILLS. The
    # annotation now tracks the outcome; the summary below counts OUTCOMES.
    ("M14", "CAP_EDIT_AT moved far future (2027)",
     lambda m, a: setattr(m, "CAP_EDIT_AT", "2027-01-01T00:00:00Z"), "KILL"),
    # F-E: the boundary is now DERIVED FROM DISK, so the derivation itself must be
    # load-bearing too -- a constant-only matrix would not notice if
    # session_is_post_removal stopped discriminating. Forcing it True reclassifies
    # the whole corpus as uncapped, which must collapse capped_n and kill the run.
    ("M21", "session_is_post_removal forced True (every session reads as uncapped)",
     lambda m, a: setattr(m, "session_is_post_removal", lambda _d: True), "KILL"),
    # Absent-subject vacuity, reported honestly rather than hidden.
    ("M6", "qa.md deleted entirely",
     lambda m, a: (a / "qa.md").unlink(), "SURVIVE_KNOWN_GAP"),
    ("M6b", "both agent files deleted",
     lambda m, a: [(a / f"{r}.md").unlink() for r in ("qa", "researcher")],
     "SURVIVE_KNOWN_GAP"),
]

EXPLANATIONS = {
    "M6": "KNOWN GAP, accepted. A missing agent file makes the role read as "
          "uncapped. Lower severity than a restored pin: a vanished qa.md breaks "
          "the Agent-tool roster loudly and immediately, so this guard is not the "
          "thing that would notice. Recorded rather than dropped.",
    "M6b": "KNOWN GAP, same class as M6.",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="exit 1 if a KILL-expected cell survives")
    args = ap.parse_args()

    before = {p: md5(p) for p in [TARGET, AGENTS / "qa.md", AGENTS / "researcher.md"]}

    print("MUTATION MATRIX -- scripts/qa/rail_turn_cap.py")
    print("=" * 78)
    print("CONTROL (unmutated) must be GREEN before any mutant is scored.")
    ok, problems, analysis = run_cell(None)
    print(f"  control verify_ok={ok}  live_caps={analysis['remediation']['live_caps']}")
    if not ok:
        print("  CONTROL IS RED -- the matrix is meaningless. Fix the subject first.")
        for p in problems:
            print(f"    - {p}")
        return 1
    print()

    # Oracle for the report-fidelity cells: the CONTROL's own re-measurement
    # over the same immutable corpus. A source mutant whose published numbers
    # deviate from it is behaviourally non-equivalent by construction.
    control_prt = analysis["remediation"].get("post_removal_turns")

    rows, real_survivors = [], []
    for cid, desc, mutate, expect in CELLS:
        try:
            ok, problems, cell_analysis = run_cell(mutate)
            # A cell counts as KILLED only if verify said False AND said why.
            killed = (not ok) and bool(problems)
            outcome = "KILLED" if killed else "SURVIVED"
            first = problems[0][:88] if problems else ""
        except Exception as exc:  # a broken cell is ERROR, never a kill
            outcome, first = "ERROR", f"{type(exc).__name__}: {exc}"[:88]
        rows.append((cid, desc, expect, outcome, first))
        if expect == "KILL" and outcome != "KILLED":
            real_survivors.append((cid, desc))

    # phase-86.84 cycle-5: source-level cells over the cycle-4 additions.
    for cid, desc, old, new, inject, kill_mode, expect in SOURCE_CELLS:
        try:
            ok, problems, cell_analysis = run_source_cell(old, new, inject=inject)
            prt = cell_analysis["remediation"].get("post_removal_turns")
            if kill_mode == "VERIFY":
                killed = (not ok) and bool(problems)
                first = problems[0][:88] if problems else ""
            elif kill_mode == "ORACLE":
                killed = prt != control_prt
                first = ("report deviates from the control oracle"
                         if killed else "report identical to control")
            elif kill_mode == "INJECTED_TRUTH":
                # The harness planted ONE non-emitter; the guard under test is
                # the injected-truth assertion itself: a report that hides a
                # known-planted signal is caught, so hiding == KILLED. (S6 is
                # the paired control proving the plant is visible unmutated.)
                qa_row = next((r for r in (prt or [])
                               if r["agent_type"] == "qa"), None)
                shown = bool(qa_row) and qa_row["non_emitters"] >= 1
                killed = not shown
                first = ("mutant hides the planted non-emitter -- caught by the "
                         "injected-truth assertion" if killed else
                         "planted signal still visible (mutation ineffective)")
            elif kill_mode == "MUST_STAY_GREEN":
                # Not a mutant: a negative control. The cell "kills" (passes)
                # iff verify stays green under the injection -- a red here is
                # the cycle-5 false-positive regressing.
                killed = ok
                first = ("no false positive on an operator abort" if ok else
                         "FALSE POSITIVE: " + (problems[0][:70] if problems else ""))
            else:  # pragma: no cover
                raise AssertionError(f"unknown kill_mode {kill_mode}")
            outcome = "KILLED" if killed else "SURVIVED"
            # cycle-6 (Q/A Overgeneralization finding): the kill MODE travels
            # with the outcome so VERIFY, ORACLE, INJECTED_TRUTH and
            # MUST_STAY_GREEN kills are never pooled into one undifferentiated
            # count -- an ORACLE kill is a harness detection, not a shipped
            # guard, and the reader must be able to tell.
            outcome = f"{outcome}[{kill_mode}]" if killed else outcome
        except Exception as exc:
            outcome, first = "ERROR", f"{type(exc).__name__}: {exc}"[:88]
        rows.append((cid, desc, expect, outcome, first))
        if expect == "KILL" and not outcome.startswith("KILLED"):
            real_survivors.append((cid, desc))

    w = max(len(d) for _, d, _, _, _ in rows)
    for cid, desc, expect, outcome, first in rows:
        flag = ("  <-- REAL SURVIVOR"
                if (expect == "KILL" and not outcome.startswith("KILLED"))
                else "")
        print(f"  {cid:<5} {desc:<{w}}  {outcome:<8}{flag}")
        if first:
            print(f"        {first}")
        if cid in EXPLANATIONS and outcome == "SURVIVED":
            print(f"        {EXPLANATIONS[cid]}")
    print()

    after = {p: md5(p) for p in before}
    unchanged = all(before[p] == after[p] for p in before)
    print("BYTE-IDENTICAL RESTORE (md5 before == after, real tree never written):")
    for p in before:
        mark = "ok " if before[p] == after[p] else "CHANGED "
        print(f"  {mark}{p.relative_to(REPO)}  {after[p]}")
    print()

    # phase-86.84 cycle-5 (Q/A finding): the summary previously counted LABELS
    # (cells whose expectation was not KILL) and presented that as a survivor
    # count. Count OUTCOMES, and treat a stale annotation as a failure in its
    # own right -- an expected-survivor that actually KILLED means the
    # explanation no longer describes the run.
    survived_known = [(cid, desc) for cid, desc, expect, outcome, _ in rows
                      if expect != "KILL" and outcome == "SURVIVED"]
    annotation_mismatch = [(cid, expect, outcome)
                           for cid, _, expect, outcome, _ in rows
                           if expect != "KILL" and outcome != "SURVIVED"]
    mode_counts = {}
    for _cid, _d, _e, outcome, _f in rows:
        if outcome.startswith("KILLED["):
            mode = outcome[len("KILLED["):-1]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
        elif outcome == "KILLED":
            mode_counts["VERIFY"] = mode_counts.get("VERIFY", 0) + 1
    print(f"kills by mode (never pooled): {mode_counts}")
    errors = [(cid, desc) for cid, desc, _, outcome, _ in rows
              if outcome == "ERROR"]
    print(f"cells={len(rows)}  real survivors={len(real_survivors)}  "
          f"known/equivalent survivors (BY OUTCOME)={len(survived_known)}  "
          f"errors={len(errors)}")
    for cid, desc in real_survivors:
        print(f"  REAL SURVIVOR {cid}: {desc}")
    for cid, expect, outcome in annotation_mismatch:
        print(f"  ANNOTATION MISMATCH {cid}: expected {expect}, observed {outcome}"
              f" -- the cell's explanation no longer describes the run")

    if args.verify:
        if not unchanged:
            print("\nVERIFY: FAIL -- the matrix modified the real tree.")
            return 1
        if real_survivors:
            print("\nVERIFY: FAIL -- a guard that should have caught a mutant did not.")
            return 1
        if annotation_mismatch:
            print("\nVERIFY: FAIL -- a cell annotation contradicts its observed outcome.")
            return 1
        if errors:
            print("\nVERIFY: FAIL -- a cell errored; an unscored cell is not evidence.")
            return 1
        print("\nVERIFY: PASS -- control green, 0 real survivors, outcomes match "
              "annotations, tree unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
