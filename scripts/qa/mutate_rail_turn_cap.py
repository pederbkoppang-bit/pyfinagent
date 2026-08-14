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
    """A throwaway copy of .claude/agents that a cell may freely mutate."""
    tmp = Path(tempfile.mkdtemp(prefix="rtc_mut_"))
    (tmp / ".claude" / "agents").mkdir(parents=True)
    for md in AGENTS.glob("*.md"):
        shutil.copy(md, tmp / ".claude" / "agents" / md.name)
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
    # Timeline constants must be load-bearing.
    ("M11", "CAP_REMOVED_AT moved before the corpus (2026-01-01)",
     lambda m, a: setattr(m, "CAP_REMOVED_AT", "2026-01-01T00:00:00Z"), "KILL"),
    ("M11b", "CAP_REMOVED_AT moved mid-corpus (2026-08-01)",
     lambda m, a: setattr(m, "CAP_REMOVED_AT", "2026-08-01T00:00:00Z"), "KILL"),
    ("M12", "HISTORICAL_CAPS qa 30 -> 31 (off by one, high)",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("qa", 31), "KILL"),
    ("M12b", "HISTORICAL_CAPS qa 30 -> 29 (off by one, low)",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("qa", 29), "KILL"),
    ("M13", "HISTORICAL_CAPS researcher 40 -> 41",
     lambda m, a: m.HISTORICAL_CAPS.__setitem__("researcher", 41), "KILL"),
    # Known-equivalent: the whole corpus already precedes any later boundary.
    ("M14", "CAP_REMOVED_AT moved far future (2027)",
     lambda m, a: setattr(m, "CAP_REMOVED_AT", "2027-01-01T00:00:00Z"),
     "SURVIVE_EQUIVALENT"),
    # Absent-subject vacuity, reported honestly rather than hidden.
    ("M6", "qa.md deleted entirely",
     lambda m, a: (a / "qa.md").unlink(), "SURVIVE_KNOWN_GAP"),
    ("M6b", "both agent files deleted",
     lambda m, a: [(a / f"{r}.md").unlink() for r in ("qa", "researcher")],
     "SURVIVE_KNOWN_GAP"),
]

EXPLANATIONS = {
    "M14": "EQUIVALENT. Every run in the corpus already precedes the boundary, so "
           "moving it later changes no row. Not a hole -- there is nothing on the "
           "far side of the boundary to misclassify yet.",
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

    rows, real_survivors = [], []
    for cid, desc, mutate, expect in CELLS:
        try:
            ok, problems, analysis = run_cell(mutate)
            # A cell counts as KILLED only if verify said False AND said why.
            killed = (not ok) and bool(problems)
            outcome = "KILLED" if killed else "SURVIVED"
            first = problems[0][:88] if problems else ""
        except Exception as exc:  # a broken cell is ERROR, never a kill
            outcome, first = "ERROR", f"{type(exc).__name__}: {exc}"[:88]
            analysis = None
        rows.append((cid, desc, expect, outcome, first))
        if expect == "KILL" and outcome != "KILLED":
            real_survivors.append((cid, desc))

    w = max(len(d) for _, d, _, _, _ in rows)
    for cid, desc, expect, outcome, first in rows:
        flag = "  <-- REAL SURVIVOR" if (expect == "KILL" and outcome != "KILLED") else ""
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

    expected_survivors = [c for c, _, _, e, _ in
                          [(r[0], r[1], r[2], r[2], r[3]) for r in rows]
                          if e != "KILL"]
    print(f"cells={len(rows)}  real survivors={len(real_survivors)}  "
          f"known/equivalent survivors={len(expected_survivors)}")
    for cid, desc in real_survivors:
        print(f"  REAL SURVIVOR {cid}: {desc}")

    if args.verify:
        if not unchanged:
            print("\nVERIFY: FAIL -- the matrix modified the real tree.")
            return 1
        if real_survivors:
            print("\nVERIFY: FAIL -- a guard that should have caught a mutant did not.")
            return 1
        print("\nVERIFY: PASS -- control green, 0 real survivors, tree unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
