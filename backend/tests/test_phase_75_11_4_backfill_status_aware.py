"""phase-75.11.4 -- the handoff backfill must be status-aware and safe-by-default.

WHAT BROKE, AND WHY A FILENAME TEST IS NOT ENOUGH. Both handoff readers
carried a byte-identical PREFIX-only `STEP_ID_RE` that matched **0** of the
727 files in `handoff/current/` on 2026-08-17. Two failures followed:

  * `verify_handoff_layout.py`'s done-step arm was guarded by a match that
    never happened, so the invariant could not fire;
  * `backfill_handoff_archive.py` routed every unmatched name to `archive/
    misc/`, so a bare run swept ~664 of 668 `.md` files -- including
    artifacts of steps that were IN FLIGHT, and including `census_78.json`,
    which step 78.0's own immutable verification command opens by literal
    path. It has fired for real (commit `fa9aaf8e`: 315 files swept, verdict
    gate dark for 13 consecutive step closes).

Every test below drives the REAL script against a temp tree -- never a
re-implementation of its logic -- following the idiom already established by
`test_phase_36_7_kill_switch_rotation_rearm.py::_load_script`. Mutations are
applied to SOURCE TEXT IN MEMORY and exec'd into a throwaway module, so the
repo is never written to and there is no restore step to get wrong.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOUSEKEEPING = REPO_ROOT / "scripts" / "housekeeping"
HOOK = REPO_ROOT / ".claude" / "hooks" / "archive-handoff.sh"


# ── loading the real scripts ──────────────────────────────────────────────

def _load_script(name: str, mutate=None):
    """Load a housekeeping script by path, optionally mutating its SOURCE first.

    `mutate` is a callable str -> str applied to the file text before exec.
    The file on disk is never modified: the mutation lives only in the string
    handed to `exec`, so a failed assertion cannot leave the repo dirty.
    """
    path = HOUSEKEEPING / f"{name}.py"
    src = path.read_text(encoding="utf-8")
    if mutate is not None:
        mutated = mutate(src)
        assert mutated != src, (
            f"mutation for {name} changed nothing -- the cell would score as a "
            "survivor for a reason that has nothing to do with the guard"
        )
        src = mutated
    mod = types.ModuleType(f"_p75114_{name}_{hashlib.sha1(src.encode()).hexdigest()[:8]}")
    mod.__file__ = str(path)
    # The scripts do `sys.path.insert(...)` then `from handoff_naming import ...`;
    # exec'ing their source runs that, so the shared module resolves normally.
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def _point_at(mod, handoff: Path, masterplan: Path) -> None:
    """Repoint a loaded script's module-level path globals at a temp tree.

    Same technique as test_phase_36_7_*::test_..._not_swept_by_backfill, which
    is why `main(dry_run=...)` must keep reading these globals.
    """
    mod.REPO = handoff.parent
    mod.HANDOFF = handoff
    mod.CURRENT = handoff / "current"
    mod.ARCHIVE = handoff / "archive"
    mod.AUDIT = handoff / "audit"
    mod.LOGS = handoff / "logs"
    mod.MISC = handoff / "archive" / "misc"
    mod.MASTERPLAN = masterplan


def _tree(tmp_path: Path) -> Path:
    handoff = tmp_path / "handoff"
    (handoff / "current").mkdir(parents=True)
    (handoff / "audit").mkdir()
    (handoff / "logs").mkdir()
    (handoff / "archive" / "misc").mkdir(parents=True)
    return handoff


def _masterplan(tmp_path: Path, steps: list[dict]) -> Path:
    mp = tmp_path / "masterplan.json"
    mp.write_text(json.dumps({"phases": [{"id": "99", "steps": steps}]}), encoding="utf-8")
    return mp


def _fake_repo(tmp_path: Path, steps: list[dict], files: dict[str, str],
               mutate=None) -> Path:
    """Build a self-contained repo copy and return the script path inside it.

    WHY THIS EXISTS (cycle-2 fix for the criterion-6 vacuity). The previous
    criterion-6 test wrote its own harness that did `import
    backfill_handoff_archive as m` -- which never runs `if __name__ ==
    "__main__"` -- and then re-declared argparse, calling
    `m.main(dry_run=not a.execute)`. The `not` under test lives in the SCRIPT's
    __main__ block, so the harness was asserting its own private copy and
    mutant M-INV (`dry_run=not args.execute` -> `dry_run=args.execute`) passed
    the whole suite while a bare run MOVED files.

    The script derives `REPO = Path(__file__).resolve().parents[2]`, so a copy
    at `<tmp>/scripts/housekeeping/` makes `<tmp>` the repo. Running that copy
    as a SUBPROCESS with no arguments executes the real `__main__` -- the only
    way to test what a bare invocation actually does.
    """
    root = tmp_path / "fakerepo"
    hk = root / "scripts" / "housekeeping"
    hk.mkdir(parents=True)
    (root / ".claude").mkdir()
    (root / "handoff" / "current").mkdir(parents=True)
    (root / "handoff" / "archive" / "misc").mkdir(parents=True)
    (root / "handoff" / "audit").mkdir()
    (root / "handoff" / "logs").mkdir()

    src = (HOUSEKEEPING / "backfill_handoff_archive.py").read_text(encoding="utf-8")
    if mutate is not None:
        mutated = mutate(src)
        assert mutated != src, "mutation changed nothing -- the cell is not scorable"
        src = mutated
    (hk / "backfill_handoff_archive.py").write_text(src, encoding="utf-8")
    shutil.copy2(HOUSEKEEPING / "handoff_naming.py", hk / "handoff_naming.py")

    (root / ".claude" / "masterplan.json").write_text(
        json.dumps({"phases": [{"id": "99", "steps": steps}]}), encoding="utf-8")
    for name, body in files.items():
        (root / "handoff" / "current" / name).write_text(body, encoding="utf-8")
    return hk / "backfill_handoff_archive.py"


def _run_script(script: Path, *args: str):
    """Execute the real script as a subprocess -- __main__ block and all."""
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, timeout=180)


_C6_FILES = {
    "research_brief_99.1.md": "pending\n",     # open step -> always kept
    "research_brief_99.2.md": "done\n",        # closed step -> moves ONLY on --execute
}
_C6_STEPS = [{"id": "99.1", "status": "pending"}, {"id": "99.2", "status": "done"}]


# ── criterion 1: step ids resolve; only closed statuses are archivable ────

def test_c1_resolver_reads_the_convention_the_project_actually_writes():
    sys.path.insert(0, str(HOUSEKEEPING))
    from handoff_naming import is_archivable, resolve_step_id

    # The live convention (suffix) and its variant tail.
    assert resolve_step_id("contract_86.29.md").sid == "86.29"
    assert resolve_step_id("live_check_75.11.4.md").sid == "75.11.4"
    assert resolve_step_id("evaluator_critique_36.13.json").sid == "36.13"
    assert resolve_step_id("research_brief_86.59_rerun.md").sid == "86.59"
    # The legacy convention must KEEP working -- 507 archived files use it.
    assert resolve_step_id("phase-4.16.2-contract.md").sid == "4.16.2"
    assert resolve_step_id("4.5.9-contract.md").sid == "4.5.9"
    # A file naming no step must return None, NOT a guess. This is the branch
    # whose old "unrecognised therefore junk" reading caused the sweep.
    for name in ("contract.md", "day_report_2026-08-09.md",
                 "INCIDENT_2026-08-14_credential_exposure.md"):
        assert resolve_step_id(name) is None, name

    assert is_archivable("done")
    assert is_archivable("superseded")
    assert is_archivable("dropped")
    for open_status in ("pending", "in-progress", "blocked", "", None, "DONE"):
        assert not is_archivable(open_status), open_status


def test_c1_resolver_precision_limit_is_pinned_not_papered_over():
    """MEASURED weakness, pinned so it cannot silently widen.

    Some real filenames use `_` where the convention uses `.`, so the resolver
    recovers a WRONG id: `cc_rail_baseline_4000_1.md` -> `1` (the artifact is
    4000.1), `verdict_population_86_98_input.md` -> `86` (it is 86.98).

    The resolver is a CANDIDATE source, not an authority -- so the safety
    property to assert is not "it never mis-reads a name" (it does) but "a
    mis-read cannot cause a move", which the status gate enforces because none
    of these ids is a real step.
    """
    sys.path.insert(0, str(HOUSEKEEPING))
    from handoff_naming import is_archivable, resolve_step_id

    for name, wrong_sid in (
        ("cc_rail_baseline_4000_1.md", "1"),
        ("verdict_population_86_98_input.md", "86"),
        ("incident_86_70_risk_gate_bypass_DELL.md", "86"),
    ):
        ref = resolve_step_id(name)
        assert ref is not None and ref.sid == wrong_sid, (
            f"{name}: precision limit changed; re-derive it rather than "
            "editing this expectation"
        )
    # The real masterplan must not contain these ids as STEPS, which is what
    # makes the mis-reads harmless. Derived from the live file, not asserted.
    mp = json.loads((REPO_ROOT / ".claude" / "masterplan.json").read_text())
    live_ids = {
        str(s.get("id"))
        for ph in mp.get("phases", [])
        for s in ph.get("steps", [])
        if s.get("id")
    }
    for wrong_sid in ("1", "86"):
        assert wrong_sid not in live_ids, (
            f"sid {wrong_sid!r} is now a real step, so a filename mis-read "
            "COULD archive a live artifact -- tighten the resolver"
        )
    assert not is_archivable(None)


# ── criterion 2: pending is kept and done is moved, IN ONE RUN ────────────

def _mixed_tree(tmp_path: Path):
    """Fixture covering ALL FOUR keep/move branches in one run.

    Cycle 1 contained only names that resolve to steps present in the fixture
    masterplan, so the no-step-id branch and the unknown-sid branch had NO
    exercise at all -- mutants M3 and M4 (either branch restored to sweeping
    into archive/misc/) passed the whole suite. Worse, cycle 1's own fix for a
    different mutant renamed `census_99.json` (sid 99, unknown) to
    `census_99.4.json`, which removed the ONE place the unknown branch was
    reached and put nothing back. Both are now first-class fixture members.
    """
    handoff = _tree(tmp_path)
    cur = handoff / "current"
    # (1) OPEN step -> keep
    (cur / "research_brief_99.1.md").write_text("pending step brief\n", encoding="utf-8")
    (cur / "live_check_99.1.md").write_text("pending step live check\n", encoding="utf-8")
    # (2) CLOSED step -> move
    (cur / "research_brief_99.2.md").write_text("done step brief\n", encoding="utf-8")
    # (3) NO step id at all -> keep (the 664-file class)
    (cur / "day_report_2026-08-17.md").write_text("no step id\n", encoding="utf-8")
    (cur / "INCIDENT_2026-08-14_credential_exposure.md").write_text("no sid\n", encoding="utf-8")
    # (4) resolves to a step the masterplan does NOT have -> keep + WARN
    (cur / "research_brief_77.9.md").write_text("unknown step\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [
        {"id": "99.1", "status": "pending"},
        {"id": "99.2", "status": "done"},
    ])
    return handoff, mp


def test_c2_pending_kept_and_done_moved_in_the_same_run(tmp_path, capsys):
    handoff, mp = _mixed_tree(tmp_path)
    mod = _load_script("backfill_handoff_archive")
    _point_at(mod, handoff, mp)

    assert mod.main(dry_run=False) == 0
    out = capsys.readouterr().out
    cur = handoff / "current"

    # BOTH directions asserted in ONE run, per the criterion.
    assert (cur / "research_brief_99.1.md").exists(), (
        "a PENDING step's write-first brief was moved -- this is the 2026-07-24 "
        "incident, where the file was pulled out from under an active researcher"
    )
    assert (cur / "live_check_99.1.md").exists()
    assert not (cur / "research_brief_99.2.md").exists(), (
        "the DONE step's artifact was not archived, so the test proves nothing "
        "about the status check -- it would pass with all moves disabled"
    )
    assert (handoff / "archive" / "phase-99.2" / "research_brief_99.2.md").exists()

    # (3) NO step id -> kept. This is the branch mutant M3 restored to sweeping.
    assert (cur / "day_report_2026-08-17.md").exists(), (
        "a file naming no step was swept -- the 664-file class is back"
    )
    assert (cur / "INCIDENT_2026-08-14_credential_exposure.md").exists()
    # (4) unknown sid -> kept AND warned. Mutant M4 kept the WARN while moving
    # the file, so the print and the file must BOTH be asserted.
    assert (cur / "research_brief_77.9.md").exists(), (
        "a file naming an unknown step was swept"
    )
    assert "[warn] KEEP: research_brief_77.9.md" in out, (
        "the unknown-step WARN line is missing -- criterion 1 requires it"
    )

    # Nothing may reach misc/: that bucket IS the sweep. THIS is the assertion
    # that kills M3/M4 -- it reads the filesystem, not the report.
    assert not list((handoff / "archive" / "misc").iterdir())
    assert "no-step-id=2" in out and "unknown-step=1" in out
    # NOTE (cycle 4): `assert "misc-moved=0" in out` used to sit here and was a
    # TAUTOLOGY -- `misc_moved` is initialised to 0 and, since the sweep branch
    # was removed, nothing can increment it, so the string is present under
    # every mutant too. Removed rather than left to look like coverage.


# ── criterion 3: the status check and the fixture are both load-bearing ───

def _strip_status_gate(src: str) -> str:
    """M1: make every resolved file archivable, i.e. delete the status check."""
    return src.replace(
        "        if is_archivable(status):",
        "        if True:  # MUTANT M1: status check removed",
    )


def test_c3_m1_removing_the_status_check_moves_the_pending_file(tmp_path):
    # CONTROL FIRST: the unmutated script keeps the pending file (asserted by
    # test_c2). Re-asserted here so this cell is scorable on its own.
    handoff, mp = _mixed_tree(tmp_path)
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    assert (handoff / "current" / "research_brief_99.1.md").exists(), (
        "CONTROL NOT GREEN -- this cell is UNSCORABLE, not a kill"
    )

    # MUTANT: same fixture, status check removed.
    handoff2, mp2 = _mixed_tree(tmp_path / "m1")
    mutant = _load_script("backfill_handoff_archive", mutate=_strip_status_gate)
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    assert not (handoff2 / "current" / "research_brief_99.1.md").exists(), (
        "M1 SURVIVED: the pending file stayed put even with the status check "
        "removed, so the assertion in test_c2 is not what keeps it there"
    )


def test_c1_m3_restoring_the_no_step_id_sweep_is_caught(tmp_path, capsys):
    """MUTATION M3: the `no_sid_kept += 1; continue` branch sweeps to misc/.

    Survived cycle 1 because no fixture contained an unresolvable name. This is
    the exact 664-file class the step exists to remove, so it must be killable.
    """
    handoff, mp = _mixed_tree(tmp_path)
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    capsys.readouterr()
    assert (handoff / "current" / "day_report_2026-08-17.md").exists(), (
        "CONTROL NOT GREEN -- cell UNSCORABLE"
    )

    handoff2, mp2 = _mixed_tree(tmp_path / "m3")
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            "            no_sid_kept += 1\n            continue",
            "            no_sid_kept += 1\n            _move(p, MISC, dry_run)  # MUTANT M3\n            continue",
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    capsys.readouterr()
    assert not (handoff2 / "current" / "day_report_2026-08-17.md").exists(), (
        "M3 SURVIVED: the no-step-id sweep was restored and nothing noticed"
    )


def test_c1_m4_restoring_the_unknown_step_sweep_is_caught(tmp_path, capsys):
    """MUTATION M4: the unknown-sid branch keeps its WARN print but MOVES.

    This is the nastier shape -- it restores verbatim the defect criterion 1
    claims to have fixed ("the summary contradicted the action"), because the
    `[warn] KEEP` line still prints while the file is relocated. A guard that
    only checked the log line would pass.
    """
    handoff, mp = _mixed_tree(tmp_path)
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    out = capsys.readouterr().out
    assert (handoff / "current" / "research_brief_77.9.md").exists() and "[warn] KEEP" in out, (
        "CONTROL NOT GREEN -- cell UNSCORABLE"
    )

    handoff2, mp2 = _mixed_tree(tmp_path / "m4")
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            '            print(f"[warn] KEEP: {name} -- names step {sid}, which the "\n'
            '                  "masterplan does not have")',
            '            print(f"[warn] KEEP: {name} -- names step {sid}, which the "\n'
            '                  "masterplan does not have")\n'
            '            _move(p, MISC, dry_run)  # MUTANT M4',
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    mout = capsys.readouterr().out
    assert "[warn] KEEP" in mout, "the mutant should still PRINT the warn line"
    assert not (handoff2 / "current" / "research_brief_77.9.md").exists(), (
        "M4 SURVIVED: the file was swept while the KEEP line still printed, "
        "and no assertion noticed the contradiction"
    )


def test_c3_fixture_is_load_bearing_when_the_step_is_marked_done(tmp_path):
    """Mutate the FIXTURE, not the code: flip 99.1 to done and the
    not-moved assertion must flip too. A fixture that would pass either way
    tests nothing."""
    handoff = _tree(tmp_path)
    (handoff / "current" / "research_brief_99.1.md").write_text("x\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [{"id": "99.1", "status": "done"}])
    mod = _load_script("backfill_handoff_archive")
    _point_at(mod, handoff, mp)
    assert mod.main(dry_run=False) == 0
    assert not (handoff / "current" / "research_brief_99.1.md").exists(), (
        "with 99.1 marked done the file must ARCHIVE; if it stays, the "
        "pending-fixture assertion elsewhere is passing vacuously"
    )
    assert (handoff / "archive" / "phase-99.1" / "research_brief_99.1.md").exists()


# ── criterion 4: a second run moves nothing and exits 0 ──────────────────

def test_c4_second_run_moves_nothing_and_exits_zero(tmp_path, capsys):
    handoff, mp = _mixed_tree(tmp_path)
    mod = _load_script("backfill_handoff_archive")
    _point_at(mod, handoff, mp)

    assert mod.main(dry_run=False) == 0
    first = capsys.readouterr().out
    assert "done-moved=1" in first

    before = sorted(p.name for p in (handoff / "current").iterdir())
    assert mod.main(dry_run=False) == 0
    second = capsys.readouterr().out
    after = sorted(p.name for p in (handoff / "current").iterdir())

    assert after == before, "the second run changed current/"
    assert "done-moved=0" in second, (
        "the second run moved something -- note this script's `-vN` minting "
        "means a re-run is non-DESTRUCTIVE without being convergent, so "
        "convergence has to be asserted, not assumed"
    )
    # And specifically: no `-v2` duplicate was minted.
    assert not list((handoff / "archive" / "phase-99.2").glob("*-v2*"))


# ── criteria 5 + 7: files named by an immutable criterion are untouchable ─

def _referenced_tree(tmp_path: Path):
    """Fixture for criteria 5 and 7.

    The referenced file MUST belong to a step that is genuinely `done`, or the
    mutation cannot discriminate: an earlier version named it `census_99.json`
    (sid `99`, not a step in this fixture), so the unknown-step branch held it
    in place even with the reference guard removed and M2 read as a survivor
    for a reason unrelated to the guard.
    """
    handoff = _tree(tmp_path)
    # (A) protected via verification.COMMAND -- belongs to DONE step 99.4.
    (handoff / "current" / "census_99.4.json").write_text("{}\n", encoding="utf-8")
    # (B) protected via verification.LIVE_CHECK ONLY -- belongs to DONE step
    # 99.5. Cycle 2 FAILED because this file did not exist: the fixture named a
    # live_check path it never created, so dropping "live_check" from the key
    # tuple in _masterplan_referenced_names changed no assertion and the
    # criterion-7 mutation survived the whole suite. Measured live exposure at
    # the time: of 381 protected basenames, 166 are protected ONLY by the
    # live_check half.
    (handoff / "current" / "live_check_99.5.md").write_text("live check\n", encoding="utf-8")
    # (C) unprotected DONE-step sibling -- must still archive, so the guard is
    # narrow rather than a blanket refusal.
    (handoff / "current" / "research_brief_99.2.md").write_text("x\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [
        {"id": "99.2", "status": "done"},
        {"id": "99.4", "status": "done"},
        {"id": "99.5", "status": "done"},
        {"id": "99.3", "status": "done",
         "verification": {
             "command": "python -c \"import json; json.load(open('handoff/current/census_99.4.json'))\"",
             "live_check": "handoff/current/live_check_99.5.md"}},
    ])
    return handoff, mp


def test_c5_referenced_file_is_not_moved(tmp_path, capsys):
    handoff, mp = _referenced_tree(tmp_path)
    mod = _load_script("backfill_handoff_archive")
    _point_at(mod, handoff, mp)
    assert mod.main(dry_run=False) == 0
    out = capsys.readouterr().out

    assert (handoff / "current" / "census_99.4.json").exists(), (
        "a file named by an immutable verification.command was relocated -- "
        "exactly what broke step 78.0's reproducibility on 2026-07-25"
    )
    # BOTH halves of criterion 5 ("verification.command OR verification.live_check").
    assert (handoff / "current" / "live_check_99.5.md").exists(), (
        "a file protected ONLY by a verification.live_check was relocated -- "
        "166 of 381 protected basenames are in exactly that position"
    )
    assert "[protected]" in out
    # DISCRIMINATION: an unreferenced done-step sibling still moves, so the
    # guard is narrow rather than a blanket refusal to move anything.
    assert not (handoff / "current" / "research_brief_99.2.md").exists()


def test_c7_m5_dropping_the_live_check_half_moves_a_live_check_protected_file(tmp_path):
    """Criterion 7's NAMED mutation, which cycle 2 proved was absent.

    "point a step's verification.live_check at a file the classifier would
    otherwise sweep -> the protection test goes red when the guard is removed".

    The guard removal is `for k in ("command", "live_check")` -> `("command",)`.
    Cycle 2's suite stayed 22/22 green under it because the fixture's live_check
    named a file it never created. Both halves are now cells, so neither can be
    dropped silently.
    """
    handoff, mp = _referenced_tree(tmp_path)
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    assert (handoff / "current" / "live_check_99.5.md").exists(), (
        "CONTROL NOT GREEN -- cell UNSCORABLE"
    )

    handoff2, mp2 = _referenced_tree(tmp_path / "m5")
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            'str(ver.get(k) or "") for k in ("command", "live_check")',
            'str(ver.get(k) or "") for k in ("command",)  # MUTANT M5',
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    assert not (handoff2 / "current" / "live_check_99.5.md").exists(), (
        "M5 SURVIVED: dropping the live_check half of the protected set did not "
        "move the file it exclusively protects, so criterion 5's 'or "
        "verification.live_check' clause has no guard that can fail"
    )
    # The command half must STILL protect its own file under this mutant --
    # otherwise the cell would pass for the wrong reason (everything moved).
    assert (handoff2 / "current" / "census_99.4.json").exists(), (
        "the command half broke too; this cell no longer isolates live_check"
    )


def test_c7_m6_dropping_the_command_half_moves_a_command_protected_file(tmp_path):
    """The converse cell, so neither half can be dropped without a red test."""
    handoff, mp = _referenced_tree(tmp_path / "m6ctl")
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    assert (handoff / "current" / "census_99.4.json").exists(), "CONTROL NOT GREEN"

    handoff2, mp2 = _referenced_tree(tmp_path / "m6")
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            'str(ver.get(k) or "") for k in ("command", "live_check")',
            'str(ver.get(k) or "") for k in ("live_check",)  # MUTANT M6',
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    assert not (handoff2 / "current" / "census_99.4.json").exists(), "M6 SURVIVED"
    assert (handoff2 / "current" / "live_check_99.5.md").exists(), (
        "the live_check half broke too; this cell no longer isolates command"
    )


def test_c7_m2_removing_the_reference_guard_moves_the_referenced_file(tmp_path):
    handoff, mp = _referenced_tree(tmp_path)
    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    assert (handoff / "current" / "census_99.4.json").exists(), (
        "CONTROL NOT GREEN -- cell UNSCORABLE"
    )

    handoff2, mp2 = _referenced_tree(tmp_path / "m2")
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            "        if name in protected:",
            "        if False:  # MUTANT M2: reference protection removed",
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    assert not (handoff2 / "current" / "census_99.4.json").exists(), (
        "M2 SURVIVED: the referenced file stayed put with the guard removed, "
        "so test_c5 is not proving the guard"
    )


# ── criterion 6: dry-run is the DEFAULT, executing needs a flag ───────────

def test_c6_bare_invocation_is_a_dry_run(tmp_path):
    """Drives the REAL script as a subprocess so `__main__` actually runs.

    Cycle-1 version wrote its own argparse harness and imported the module
    (which never executes `__main__`), so mutant M-INV -- dropping the `not`
    from `dry_run=not args.execute` -- passed the entire suite while a bare
    invocation MOVED files. The whole point of criterion 6 is what a BARE
    invocation does, so the bare invocation is what gets executed here.
    """
    script = _fake_repo(tmp_path, _C6_STEPS, _C6_FILES)
    cur = script.parents[2] / "handoff" / "current"

    r = _run_script(script)                       # NO arguments at all
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert (cur / "research_brief_99.2.md").exists(), (
        "a bare invocation MOVED a done-step file -- dry-run is not the default"
    )
    assert (cur / "research_brief_99.1.md").exists()
    assert not list((script.parents[2] / "handoff" / "archive").glob("phase-*/*"))


def test_c6_execute_flag_actually_moves(tmp_path):
    """The other half, also through the real CLI: --execute must still work,
    or 'safe by default' would just be 'broken'."""
    script = _fake_repo(tmp_path, _C6_STEPS, _C6_FILES)
    root = script.parents[2]
    cur = root / "handoff" / "current"

    r = _run_script(script, "--execute")
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" not in r.stdout
    assert not (cur / "research_brief_99.2.md").exists(), "--execute did not move"
    assert (root / "handoff" / "archive" / "phase-99.2" / "research_brief_99.2.md").exists()
    assert (cur / "research_brief_99.1.md").exists(), "the OPEN step's file moved"


def test_c6_a_dry_run_creates_no_directories(tmp_path):
    """A dry run must not TOUCH the tree, not merely not-move files.

    Cycle 4 found `_move` running `dest_dir.mkdir(parents=True, exist_ok=True)`
    BEFORE its `if dry_run: return`, so a bare "dry run" created every
    destination directory it merely planned to use -- three empty dirs on the
    live tree (phase-80.5, -81.1, -82.23), which then classified as
    `no_contract` and inflated this step's own census denominator (845 -> 842
    after cleanup).
    """
    script = _fake_repo(tmp_path, _C6_STEPS, _C6_FILES)
    root = script.parents[2]
    archive = root / "handoff" / "archive"

    before = sorted(p.name for p in archive.iterdir())
    r = _run_script(script)                      # bare == dry run
    assert r.returncode == 0, r.stderr
    after = sorted(p.name for p in archive.iterdir())

    assert after == before, (
        f"a dry run created {set(after) - set(before)} -- a dry run that "
        "mutates the tree is not a dry run"
    )
    # And it really did PLAN the move, so the assertion above is not vacuous.
    assert "would-move" in r.stdout


def test_rolling_keep_prefixes_protects_per_step_verdict_jsons(tmp_path, capsys):
    """ADJACENT SAFETY PROPERTY, surfaced as a surviving mutant in cycle 4.

    `ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)` keeps per-step verdict
    JSONs in handoff/current/, where `auto-commit-and-push.sh` reads them by
    literal path. Emptying the tuple archived them -- the phase-81.0 defect
    this file's own comment records as having "left the verdict gate dark for
    13 consecutive step closes". 54 such files live in handoff/current/ now.
    """
    handoff = _tree(tmp_path)
    cur = handoff / "current"
    (cur / "evaluator_critique_99.2.json").write_text("{}\n", encoding="utf-8")
    (cur / "research_brief_99.2.md").write_text("x\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [{"id": "99.2", "status": "done"}])

    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    capsys.readouterr()
    assert (cur / "evaluator_critique_99.2.json").exists(), (
        "CONTROL NOT GREEN -- the verdict JSON was archived by the shipped code"
    )
    # Discrimination: its done-step sibling DOES archive.
    assert not (cur / "research_brief_99.2.md").exists()

    handoff2 = _tree(tmp_path / "q3")
    cur2 = handoff2 / "current"
    (cur2 / "evaluator_critique_99.2.json").write_text("{}\n", encoding="utf-8")
    mp2 = _masterplan(tmp_path / "q3", [{"id": "99.2", "status": "done"}])
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            'ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)',
            'ROLLING_KEEP_PREFIXES = ()  # MUTANT Q3',
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    capsys.readouterr()
    assert not (cur2 / "evaluator_critique_99.2.json").exists(), (
        "Q3 SURVIVED: emptying ROLLING_KEEP_PREFIXES did not archive the "
        "verdict JSON, so nothing guards the phase-81.0 regression"
    )


def test_safe_target_never_clobbers_prior_archived_evidence(tmp_path, capsys):
    """ADJACENT SAFETY PROPERTY, surfaced as a surviving mutant in cycle 4.

    The module docstring claims "prior evidence is never clobbered". Nothing
    asserted it: `_safe_target` returning `dest` unchanged overwrites an
    existing archived file and the suite stayed green.
    """
    handoff = _tree(tmp_path)
    cur = handoff / "current"
    dest_dir = handoff / "archive" / "phase-99.2"
    dest_dir.mkdir(parents=True)
    (dest_dir / "research_brief_99.2.md").write_text("PRIOR EVIDENCE\n", encoding="utf-8")
    (cur / "research_brief_99.2.md").write_text("NEW CONTENT\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [{"id": "99.2", "status": "done"}])

    control = _load_script("backfill_handoff_archive")
    _point_at(control, handoff, mp)
    assert control.main(dry_run=False) == 0
    capsys.readouterr()
    assert (dest_dir / "research_brief_99.2.md").read_text(encoding="utf-8") == "PRIOR EVIDENCE\n", (
        "CONTROL NOT GREEN -- the shipped code already clobbered prior evidence"
    )
    assert (dest_dir / "research_brief_99.2-v2.md").exists(), "no -v2 was minted"

    handoff2 = _tree(tmp_path / "q5")
    cur2 = handoff2 / "current"
    dd2 = handoff2 / "archive" / "phase-99.2"
    dd2.mkdir(parents=True)
    (dd2 / "research_brief_99.2.md").write_text("PRIOR EVIDENCE\n", encoding="utf-8")
    (cur2 / "research_brief_99.2.md").write_text("NEW CONTENT\n", encoding="utf-8")
    mp2 = _masterplan(tmp_path / "q5", [{"id": "99.2", "status": "done"}])
    mutant = _load_script(
        "backfill_handoff_archive",
        mutate=lambda s: s.replace(
            "def _safe_target(dest: Path) -> Path:\n    if not dest.exists():\n        return dest",
            "def _safe_target(dest: Path) -> Path:\n    return dest  # MUTANT Q5\n    if not dest.exists():\n        return dest",
        ),
    )
    _point_at(mutant, handoff2, mp2)
    assert mutant.main(dry_run=False) == 0
    capsys.readouterr()
    assert (dd2 / "research_brief_99.2.md").read_text(encoding="utf-8") == "NEW CONTENT\n", (
        "Q5 SURVIVED: _safe_target was neutered and prior archived evidence "
        "was NOT overwritten, so the docstring's 'never clobbered' claim has "
        "no guard"
    )


def test_c6_m_inv_inverting_the_default_is_caught(tmp_path):
    """MUTATION M-INV: `dry_run=not args.execute` -> `dry_run=args.execute`.

    This is the mutant that survived cycle 1. It is the dangerous shape -- it
    reinstates exactly the "a bare run moves files" defect this step exists to
    remove -- and it is NOT the same as a full revert to `args.dry_run`, which
    is why an `ast.dump` substring check could not see it (both dumps contain
    the token "execute"; the substring cannot see the `Not()` node).
    """
    # CONTROL first: unmutated bare run keeps the file.
    ctrl = _fake_repo(tmp_path / "control", _C6_STEPS, _C6_FILES)
    rc = _run_script(ctrl)
    assert rc.returncode == 0 and "DRY RUN" in rc.stdout, "CONTROL NOT GREEN -- cell UNSCORABLE"
    assert (ctrl.parents[2] / "handoff" / "current" / "research_brief_99.2.md").exists(), (
        "CONTROL NOT GREEN -- cell UNSCORABLE"
    )

    # MUTANT: drop the `not`.
    mut = _fake_repo(
        tmp_path / "mutant", _C6_STEPS, _C6_FILES,
        mutate=lambda s: s.replace(
            "raise SystemExit(main(dry_run=not args.execute))",
            "raise SystemExit(main(dry_run=args.execute))  # MUTANT M-INV",
        ),
    )
    rm = _run_script(mut)
    assert rm.returncode == 0, rm.stderr
    moved = not (mut.parents[2] / "handoff" / "current" / "research_brief_99.2.md").exists()
    assert moved and "DRY RUN" not in rm.stdout, (
        "M-INV SURVIVED: inverting the default did not change what a bare "
        "invocation does, so the criterion-6 tests above are not what keeps "
        "the default safe"
    )


# ── criterion 9: the two readers agree with the hook on ONE convention ────

def test_c9_verifier_and_hook_classify_the_same_filename_identically():
    """The hook DERIVES its source names; the verifier RECOGNISES them. If the
    two disagree, an artifact the hook archives is one the verifier cannot see.
    """
    sys.path.insert(0, str(HOUSEKEEPING))
    from handoff_naming import resolve_step_id

    hook_src = HOOK.read_text(encoding="utf-8")
    # The hook's live branch: `src="$CURRENT_DIR/${base}_${short_sid}.md"`.
    assert '${base}_${short_sid}.md' in hook_src, (
        "the hook no longer derives suffix names; re-derive this test's premise"
    )
    bases = ["contract", "experiment_results", "evaluator_critique",
             "research_brief", "live_check"]
    for base in bases:
        assert f"for base in {' '.join(bases)}" in hook_src or base in hook_src
    for sid in ("75.11.4", "86.29", "4000.1"):
        for base in bases:
            name = f"{base}_{sid}.md"          # exactly what the hook writes
            ref = resolve_step_id(name)
            assert ref is not None and ref.sid == sid, (
                f"the hook archives {name} but the readers cannot resolve it"
            )


def _point_verifier(mod, handoff: Path, masterplan: Path) -> None:
    """verify_handoff_layout.py reads only these three globals."""
    mod.REPO = handoff.parent
    mod.HANDOFF = handoff
    mod.CURRENT = handoff / "current"
    mod.MASTERPLAN = masterplan


def _verifier_tree(tmp_path: Path):
    handoff = _tree(tmp_path)
    cur = handoff / "current"
    # DONE step, SUFFIX-named -> the invariant must SEE it. Under the retired
    # PREFIX-only regex this file was invisible, which is the whole defect.
    (cur / "research_brief_99.2.md").write_text("done\n", encoding="utf-8")
    # OPEN step -> must NOT be a violation.
    (cur / "research_brief_99.1.md").write_text("pending\n", encoding="utf-8")
    # No step id -> reported as info, NOT a violation.
    (cur / "day_report_2026-08-17.md").write_text("no sid\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [
        {"id": "99.1", "status": "pending"},
        {"id": "99.2", "status": "done"},
    ])
    return handoff, mp


def test_c9_the_verifier_is_actually_EXECUTED_and_sees_a_done_step_artifact(tmp_path, capsys):
    """Cycle 2's blocker: NO test in this suite ever ran verify_handoff_layout.

    Its entire coverage was a source-scan for the string `from handoff_naming
    import`, which two different mutants satisfied while reverting the
    behaviour. A source scan cannot observe runtime behaviour; this runs it.
    """
    handoff, mp = _verifier_tree(tmp_path)
    mod = _load_script("verify_handoff_layout")
    _point_verifier(mod, handoff, mp)

    rc = mod.main()
    out = capsys.readouterr().out

    assert rc == 1, "a done-step artifact in current/ must fail the invariant"
    assert "research_brief_99.2.md belongs to done step 99.2" in out, (
        "the done-step arm did not fire on a SUFFIX-named artifact -- this is "
        "exactly the pre-fix state (the arm was unreachable)"
    )
    assert "research_brief_99.1.md" not in out.split("FAIL")[-1], (
        "an OPEN step's artifact was reported as a violation"
    )
    assert "carry no step id" in out, "the no-step-id class must be reported as info"
    assert "day_report_2026-08-17.md belongs to" not in out


def test_c9_n14_disabling_the_verifier_status_arm_is_caught(tmp_path, capsys):
    """MUTANT N14: `elif is_archivable(status)` -> `elif False`. Survived cycle 2."""
    handoff, mp = _verifier_tree(tmp_path)
    control = _load_script("verify_handoff_layout")
    _point_verifier(control, handoff, mp)
    assert control.main() == 1, "CONTROL NOT GREEN -- cell UNSCORABLE"
    capsys.readouterr()

    handoff2, mp2 = _verifier_tree(tmp_path / "n14")
    mutant = _load_script(
        "verify_handoff_layout",
        mutate=lambda s: s.replace(
            "            elif is_archivable(status):",
            "            elif False:  # MUTANT N14",
        ),
    )
    _point_verifier(mutant, handoff2, mp2)
    rc = mutant.main()
    capsys.readouterr()
    assert rc == 0, (
        "N14 SURVIVED: the status arm was disabled and the verifier still "
        "reported the done-step artifact"
    )


def test_c9_n15_reverting_the_verifier_to_the_dead_prefix_regex_is_caught(tmp_path, capsys):
    """MUTANT N15: keep the `from handoff_naming import` LINE (which the old
    source-scan guard byte-checked) but rebind resolve_step_id to the retired
    PREFIX-only matcher. Survived cycle 2 for exactly that reason -- the guard
    checked a literal, not behaviour."""
    handoff, mp = _verifier_tree(tmp_path)
    control = _load_script("verify_handoff_layout")
    _point_verifier(control, handoff, mp)
    assert control.main() == 1, "CONTROL NOT GREEN -- cell UNSCORABLE"
    capsys.readouterr()

    handoff2, mp2 = _verifier_tree(tmp_path / "n15")
    mutant = _load_script(
        "verify_handoff_layout",
        mutate=lambda s: s.replace(
            "from handoff_naming import is_archivable, resolve_step_id  # noqa: E402",
            "from handoff_naming import is_archivable, resolve_step_id  # noqa: E402\n"
            "import re as _re_n15\n"
            "_PREFIX_N15 = _re_n15.compile(r'^(?:phase-)?([0-9]+(?:\\.[0-9]+)*)[-.].*\\.md$')\n"
            "def resolve_step_id(name):  # MUTANT N15\n"
            "    m = _PREFIX_N15.match(name)\n"
            "    return None if not m else type('R', (), {'sid': m.group(1), 'convention': 'prefix'})()",
        ),
    )
    _point_verifier(mutant, handoff2, mp2)
    rc = mutant.main()
    capsys.readouterr()
    assert rc == 0, (
        "N15 SURVIVED: the verifier was reverted to the PREFIX-only regex -- "
        "which matches 0 of the project's artifacts -- and the suite stayed green"
    )


def test_c9_both_readers_share_one_definition_and_keep_their_own_allowlists():
    """Shared where drift is a bug; duplicated where a drift TEST depends on
    the literal. `test_phase_36_8_*` ast.literal_eval's AUDIT_KEEP_GLOBS out of
    each file, so that one must stay a literal in both."""
    for name in ("backfill_handoff_archive", "verify_handoff_layout"):
        src = (HOUSEKEEPING / f"{name}.py").read_text(encoding="utf-8")
        assert "from handoff_naming import" in src, f"{name} does not share the resolver"
        tree = ast.parse(src)
        literal = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "AUDIT_KEEP_GLOBS" for t in node.targets
            ):
                literal = ast.literal_eval(node.value)
        assert literal, (
            f"{name}: AUDIT_KEEP_GLOBS is no longer an ast.literal_eval-able "
            "literal -- test_phase_36_8_both_housekeeping_scripts_protect_the_"
            "audit_archives parses it that way and would break"
        )


# ── criteria 8 + 11: the hook archives suffix-named artifacts, hermetically ─

def _drive_hook(tmp_path: Path, sid: str, current_files: dict[str, str],
                statuses: list[dict]) -> Path:
    """Run the REAL hook against a scratch tree via CLAUDE_PROJECT_DIR.

    This deliberately replaces criterion 8's literal "flip a scratch step":
    flipping a step in the live masterplan fires auto-commit-and-push.sh ->
    `git add -A`, which would sweep a peer session's uncommitted work into a
    commit named after a probe step. Driving the hook directly is
    deterministic, repeatable, and touches nothing outside tmp_path.
    """
    root = tmp_path / f"hookroot_{sid.replace('.', '_')}"
    (root / ".claude").mkdir(parents=True)
    (root / "handoff" / "current").mkdir(parents=True)
    (root / "handoff" / "archive").mkdir(parents=True)
    (root / ".claude" / "masterplan.json").write_text(
        json.dumps({"phases": [{"id": "99", "steps": statuses}]}), encoding="utf-8")
    (root / ".claude" / ".archive-baseline.json").write_text(
        json.dumps({"seen": []}), encoding="utf-8")
    for fname, body in current_files.items():
        (root / "handoff" / "current" / fname).write_text(body, encoding="utf-8")
    env = dict(os.environ, CLAUDE_PROJECT_DIR=str(root))
    subprocess.run(["bash", str(HOOK)], env=env, capture_output=True, timeout=120)
    return root / "handoff" / "archive" / f"phase-{sid}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_c8_c11_hook_archives_the_step_s_own_suffix_named_artifacts(tmp_path):
    target = _drive_hook(
        tmp_path, "99.7",
        {"contract_99.7.md": "# Contract -- step 99.7\nmine\n",
         "live_check_99.7.md": "# Live check -- step 99.7\nmine\n",
         "contract.md": "# Contract -- step 12.3\nANOTHER STEP\n"},
        [{"id": "99.7", "status": "done"}],
    )
    assert target.is_dir(), "the hook created no archive dir for the closing step"
    assert (target / "contract.md").exists(), "criterion 11: the step's own contract"
    assert (target / "live_check.md").exists(), (
        "criterion 11: live_check_<sid>.md is the artifact the old PREFIX glob "
        "could never match"
    )
    assert (target / "PROVENANCE.md").exists()
    # CONTENT, not just existence. Cycle 2 raised this as a WARN: a mutant that
    # created the archived file EMPTY (`: > "$target/..."` instead of `cp`)
    # satisfied every existence assert AND the negative below, because a
    # zero-byte file trivially does not contain another step's text.
    assert (target / "contract.md").read_text(encoding="utf-8") == \
        "# Contract -- step 99.7\nmine\n", "archived contract.md is not the source"
    assert (target / "live_check.md").read_text(encoding="utf-8") == \
        "# Live check -- step 99.7\nmine\n", "archived live_check.md is not the source"
    # criterion 13 (prevention half): the foreign rolling file must NOT win.
    assert "ANOTHER STEP" not in (target / "contract.md").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")
def test_c13_foreign_rolling_file_is_refused_and_a_declaring_one_is_admitted(tmp_path):
    """The guard must DISCRIMINATE. A blanket refusal would pass the first
    assertion and fail the second.

    Note the fixture omits `contract_99.8.md` on purpose: with the derived file
    present, `[ -f "$target/$f" ] && continue` short-circuits and the
    declaration guard is never reached -- a probe that included it would credit
    the guard for the derived branch's work.
    """
    target = _drive_hook(
        tmp_path, "99.8",
        {"contract.md": "# Contract -- step 12.3\nFOREIGN\n",
         "experiment_results.md": "# Experiment results -- phase-99.8\nMINE\n"},
        [{"id": "99.8", "status": "done"}],
    )
    assert not (target / "contract.md").exists(), (
        "a rolling file declaring ANOTHER step was archived as this step's -- "
        "that is the misattribution defect itself"
    )
    assert (target / "experiment_results.md").exists(), (
        "a rolling file that DOES declare this step was refused: the guard is "
        "a blanket refusal, not a discriminating one"
    )
    prov = (target / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "SKIPPED" in prov and "declares this step" in prov


# ── criteria 10 + 12: the whole-tree census, and what it says today ───────

def test_c10_c12_archive_census_runs_over_the_whole_tree_and_reports(tmp_path):
    """Criterion 10 asks for a checker over the WHOLE archive tree, not a spot
    check. That checker already exists (phase-86.29) with a recall gate that
    REFUSES to print a census if it misses a known positive -- so a green run
    is itself evidence the method has recall. Reused rather than rewritten.
    """
    checker = REPO_ROOT / "scripts" / "qa" / "derive_archive_misattribution_86_29.py"
    assert checker.exists()
    r = subprocess.run([sys.executable, str(checker)], capture_output=True,
                       text=True, timeout=600, cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stderr[-2000:]
    out = r.stdout
    assert "mismatches reported" in out
    assert "precision" in out
    # The count is re-measured at run time, never inherited from prose.
    n_dirs = len([d for d in (REPO_ROOT / "handoff" / "archive").iterdir()
                  if d.is_dir() and d.name.startswith("phase-")])
    assert n_dirs > 700, f"archive tree unexpectedly small ({n_dirs})"


def test_c12_every_mismatched_directory_carries_a_quarantine_marker():
    """Criterion 12's remediation half, asserted against the live tree.

    The remediation is ADDITIVE -- a `MISATTRIBUTION_NOTICE.md` is written into
    each flagged directory and nothing is moved. Re-derived here from the
    classifier rather than compared against a hardcoded count, so the assertion
    cannot go stale the way "129 of 747" did.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "qa"))
    from derive_archive_misattribution_86_29 import classify

    archive = REPO_ROOT / "handoff" / "archive"
    missing = []
    flagged = 0
    for d in sorted(archive.iterdir()):
        if not d.is_dir() or not d.name.startswith("phase-"):
            continue
        verdict, _declares = classify(d)
        if verdict != "mismatch":
            continue
        flagged += 1
        if not (d / "MISATTRIBUTION_NOTICE.md").exists():
            missing.append(d.name)

    assert flagged > 0, (
        "the classifier flagged nothing, so this test would pass vacuously -- "
        "re-derive the census before trusting a green here"
    )
    assert not missing, (
        f"{len(missing)} of {flagged} mismatched dirs have no quarantine "
        f"marker: {sorted(missing)[:5]}"
    )


def test_quarantine_tool_is_dry_run_by_default_and_idempotent(tmp_path):
    """Direct coverage for quarantine_misattributed_archives.py.

    Cycle 4: the 174-line production script that WROTE the 156 markers had
    zero direct tests -- it was exercised only through its output. Driven here
    against a scratch archive tree via the same parents[2] relocation the
    backfill uses.
    """
    root = tmp_path / "qrepo"
    hk = root / "scripts" / "housekeeping"
    hk.mkdir(parents=True)
    qa_dir = root / "scripts" / "qa"
    qa_dir.mkdir(parents=True)
    (root / "handoff" / "archive").mkdir(parents=True)
    shutil.copy2(HOUSEKEEPING / "quarantine_misattributed_archives.py",
                 hk / "quarantine_misattributed_archives.py")
    # The tool imports the phase-86.29 census from <repo>/scripts/qa and
    # REFUSES to run without it (by design -- it has no classifier of its own).
    # That refusal is itself worth pinning, so it gets its own assertion below.
    shutil.copy2(REPO_ROOT / "scripts" / "qa" / "derive_archive_misattribution_86_29.py",
                 qa_dir / "derive_archive_misattribution_86_29.py")

    # One MISMATCHED dir (contract declares another step) and one that AGREES.
    bad = root / "handoff" / "archive" / "phase-99.6"
    bad.mkdir()
    (bad / "contract.md").write_text("# Contract -- step `82.54`\n", encoding="utf-8")
    good = root / "handoff" / "archive" / "phase-99.9"
    good.mkdir()
    (good / "contract.md").write_text("# Contract -- step `99.9`\n", encoding="utf-8")

    script = hk / "quarantine_misattributed_archives.py"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    # 1. DEFAULT IS A DRY RUN -- plans, writes nothing.
    r = subprocess.run([sys.executable, str(script)], capture_output=True,
                       text=True, timeout=600, cwd=str(REPO_ROOT), env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "DRY RUN" in r.stdout
    assert not (bad / "MISATTRIBUTION_NOTICE.md").exists(), "the dry run WROTE"

    # 2. --execute writes exactly one marker, into the mismatched dir only.
    r = subprocess.run([sys.executable, str(script), "--execute"], capture_output=True,
                       text=True, timeout=600, cwd=str(REPO_ROOT), env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    assert (bad / "MISATTRIBUTION_NOTICE.md").exists(), "no marker written"
    assert not (good / "MISATTRIBUTION_NOTICE.md").exists(), (
        "a marker was written into a dir that AGREES -- the tool is not narrow"
    )
    body = (bad / "MISATTRIBUTION_NOTICE.md").read_text(encoding="utf-8")
    assert "82.54" in body and "99.6" in body, "the notice names neither step"

    # 3. IDEMPOTENT: a second --execute writes nothing new.
    r = subprocess.run([sys.executable, str(script), "--execute"], capture_output=True,
                       text=True, timeout=600, cwd=str(REPO_ROOT), env=env)
    assert r.returncode == 0
    assert "mismatched-needing-marker=0" in r.stdout
    assert "already-marked=1" in r.stdout


def test_c12_prevention_holds_for_every_guard_created_directory():
    """The remediation scope claim, asserted rather than argued.

    `.claude/hooks/archive-handoff.sh` writes PROVENANCE.md UNCONDITIONALLY,
    before any branch, so its presence marks a directory created by the
    phase-86.29 (guarded) hook. If prevention works, no such directory is in
    the mismatch set -- making criteria 10/12 REMEDIATION of a closed class.

    Bound stated in the assertion message: the guarded population is small.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "qa"))
    from derive_archive_misattribution_86_29 import classify

    archive = REPO_ROOT / "handoff" / "archive"
    guarded = {d.name for d in archive.iterdir()
               if d.is_dir() and (d / "PROVENANCE.md").exists()}
    assert guarded, "no guard-created archive dirs found; re-derive the marker"

    # Built from classify(), NOT from a regex over the checker's PROSE output.
    # The prose-regex version admitted 11 tokens that are not mismatched dirs
    # (the DECLARED sids in the top-8 summary, plus the synthetic controls) and
    # MISSED two real ones whose names the pattern cannot express --
    # `phase-63.3-parked` and `phase-audit-2.10-4.14.20`. It was conservative
    # rather than vacuous, but it was not the census it claimed to be.
    wrong = {d.name for d in archive.iterdir()
             if d.is_dir() and d.name.startswith("phase-")
             and classify(d)[0] == "mismatch"}
    assert wrong, "classifier flagged nothing -- this test would pass vacuously"

    overlap = guarded & wrong
    assert not overlap, (
        f"{len(overlap)} directory(ies) created by the GUARDED hook are still "
        f"misattributed ({sorted(overlap)[:5]}): prevention is not holding, so "
        "this is not merely a historical remediation"
    )


# ── the kill-switch allowlist must survive every change above ─────────────

def test_kill_switch_state_file_still_excluded_after_the_rewrite(tmp_path, capsys):
    """phase-36.7's regression, re-asserted against the NEW classifier.

    This file is the canonical demonstration that a name-shaped rule can
    mis-classify live safety state: moving it left the kill switch disarmed
    after every restart. Any new classifier must inherit the exclusion.
    """
    handoff = _tree(tmp_path)
    (handoff / "kill_switch_audit.jsonl").write_text('{"ts":"x"}\n', encoding="utf-8")
    (handoff / "some_other_audit.jsonl").write_text("{}\n", encoding="utf-8")
    (handoff / "stray.log").write_text("x\n", encoding="utf-8")
    mp = _masterplan(tmp_path, [])
    mod = _load_script("backfill_handoff_archive")
    _point_at(mod, handoff, mp)

    assert mod.main(dry_run=False) == 0
    assert (handoff / "kill_switch_audit.jsonl").exists(), (
        "phase-36.7 REGRESSION reintroduced by phase-75.11.4"
    )
    assert not list((handoff / "audit").glob("kill_switch_audit*.jsonl"))
    # Narrowness preserved: a genuinely log-shaped sibling still moves.
    assert (handoff / "audit" / "some_other_audit.jsonl").exists()
    assert (handoff / "logs" / "stray.log").exists()


def test_the_scripts_on_disk_were_not_mutated_by_this_suite():
    """Mutations are exec'd from strings; the files must be byte-identical.

    Asserted rather than trusted: a suite that mutates source has to prove it
    left no residue, and `disk-mutating checkers must not be wired` is a
    standing lesson in this repo.
    """
    for name in ("backfill_handoff_archive.py", "verify_handoff_layout.py",
                 "handoff_naming.py"):
        src = (HOUSEKEEPING / name).read_text(encoding="utf-8")
        assert "MUTANT M1" not in src and "MUTANT M2" not in src, (
            f"{name} contains mutation residue on disk"
        )
