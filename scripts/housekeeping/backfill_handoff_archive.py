"""phase-4.16.2 one-time backfill for handoff folder layout.

Reads .claude/masterplan.json. For each step with status=done, moves
any matching `handoff/current/phase-<sid>-*.md` (and `<sid>-*.md`)
into `handoff/archive/phase-<sid>/`. Non-conforming files (no step-id
prefix) go to `handoff/archive/misc/`. Root-level audit JSON + log
files move to `handoff/audit/` + `handoff/logs/`.

"Idempotent" here means NON-DESTRUCTIVE, not convergent: if the target
path exists, `_safe_target` appends `-v2`, `-v3`, ... so prior evidence is
never clobbered (mirrors `archive-handoff.sh`). Read that precisely -- a
re-run does NOT converge to a fixed point by itself, and mistaking the two
is how `kill_switch_audit.jsonl` reached `-v3` and `-v4`. Convergence comes
from the status gate below: once a file is archived it is gone from
`current/`, and what remains is either open, unknown, protected, or
step-less, all of which are KEPT.

phase-75.11.4 -- SAFE BY DEFAULT. A bare invocation is a DRY RUN. It used
to execute, and it swept ~664 of 668 `.md` files including artifacts of
steps that were still in flight.

Usage:
    python scripts/housekeeping/backfill_handoff_archive.py            # plan only
    python scripts/housekeeping/backfill_handoff_archive.py --execute  # apply
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

# phase-75.11.4: the step-id recogniser is now SHARED with
# verify_handoff_layout.py so the two readers cannot drift. Explicit sys.path
# insert because this file is also loaded by absolute path via importlib
# (backend/tests/test_phase_36_7_*.py::_load_script), where the containing
# directory is NOT on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_naming import is_archivable, resolve_step_id  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HANDOFF = REPO / "handoff"
CURRENT = HANDOFF / "current"
ARCHIVE = HANDOFF / "archive"
AUDIT = HANDOFF / "audit"
LOGS = HANDOFF / "logs"
MISC = ARCHIVE / "misc"
MASTERPLAN = REPO / ".claude" / "masterplan.json"

ROLLING_KEEP = {
    "contract.md",
    "experiment_results.md",
    "evaluator_critique.md",
    "research_brief.md",
    "research.md",
    "research_plan.md",
    "harness_log.md",
    # phase-81.0: the MACHINE-READABLE verdict, not just the prose critique.
    # `.claude/hooks/auto-commit-and-push.sh` reads this file by literal path
    # to decide whether a CONDITIONAL/FAIL verdict should hold a push. It was
    # absent from this allowlist, so the 2026-07-24 run swept it to
    # archive/misc/ and the verdict gate went dark for 13 consecutive step
    # closes. This script is documented idempotent and safe to re-run, so
    # without this entry any restoration is undone on the next invocation.
    "evaluator_critique.json",
}

# phase-81.0: per-step verdict JSONs are the CURRENT convention (six were
# written 2026-07-26). STEP_ID_RE only matches `.md`, so these never took the
# step-move path -- they fall through to the misc bucket and get swept away
# from the one directory the hook reads. Keep them in place by prefix.
ROLLING_KEEP_PREFIXES = ("evaluator_critique_",)


def _is_rolling_keep(name: str) -> bool:
    """True if `name` must stay in handoff/current/ (a live hook reads it)."""
    if name in ROLLING_KEEP:
        return True
    return name.startswith(ROLLING_KEEP_PREFIXES) and name.endswith(".json")

STEP_ID_RE = re.compile(r"^(?:phase-)?([0-9]+(?:\.[0-9]+)*)[-.].*\.md$")

# phase-36.7: LIVE PRODUCTION STATE FILES that merely LOOK like audit output.
#
# `kill_switch_audit.jsonl` is not a log -- it is the kill switch's only
# persistence. `backend/services/kill_switch.py::_load_from_audit` replays it at
# every process start to restore `sod_nav` / `peak_nav`. Sweeping it into
# handoff/audit/ with a `-vN` suffix left the switch DISARMED after every
# restart: a 50% drawdown returned any_breached=False (measured 2026-07-26).
# Git shows this shipped twice already (fa9aaf8e -> -v3, 77bc7db5 -> -v4), so
# it is a recurring, self-perpetuating defect, not a one-off.
#
# Keep this set byte-identical to `verify_handoff_layout.py::HANDOFF_ROOT_KEEP`
# -- if the verifier still demands the move, the next housekeeping run undoes
# this exclusion.
HANDOFF_ROOT_KEEP = {
    "kill_switch_audit.jsonl",
}

# phase-36.8: the audit ARCHIVES are safety-relevant and MUST NOT be pruned.
# Keep this set byte-identical between the two housekeeping scripts; the test
# backend/tests/test_phase_36_8_kill_switch_archive_merge_authority.py::
# test_phase_36_8_both_housekeeping_scripts_protect_the_audit_archives parses
# both by AST and fails if they drift.
#
# WHY NO CAP / NO OLDEST-FIRST PRUNING (criterion 3, decided on measurement):
# `kill_switch._load_from_audit` merges these files on every boot to restore the
# trailing high-water mark, and 100% of the live book's baselines come from them
# today. The TRUE peak (24666.57) lives in the OLDEST file, so an
# oldest-first cap would delete the row the kill switch depends on. Measured
# 2026-07-26: 897 rows across 5 files, boot cost 0.95 ms total (1.06 us/row) --
# so growth is accepted as a bounded cost rather than capped. All five files are
# git-tracked, which is the recoverability backstop.
AUDIT_KEEP_GLOBS = (
    "kill_switch_audit*.jsonl",
)


def _step_statuses() -> dict[str, str]:
    with MASTERPLAN.open() as f:
        mp = json.load(f)
    out: dict[str, str] = {}
    for p in mp.get("phases", []):
        for s in p.get("steps", []):
            sid = s.get("id")
            if sid:
                out[str(sid)] = str(s.get("status") or "pending")
    return out


def _masterplan_referenced_names() -> set[str]:
    """Basenames any masterplan step names in verification.command/live_check.

    phase-75.11.4, criterion 5. Measured 2026-08-17 with THIS function's own
    rule (the regex below, over every verification.command + live_check):
    **577** handoff-path references across **386** distinct paths, **415** of
    them into handoff/current/, **381** distinct basenames.

    (An earlier revision of this docstring said 557/373/395. That triple was
    carried over from the research brief and never re-derived here; it does not
    reproduce under any operationalization tried, and the masterplan file was
    untouched during the work. Corrected rather than annotated -- a number in a
    docstring is read as measured.)
    Every one is an IMMUTABLE criterion the project forbids editing, so
    relocating the file silently breaks a step's reproducibility with nothing
    to warn you. That has already happened: the 2026-07-25 sweep moved
    `census_78.json`, which step 78.0's own verification command opens by
    literal path, so re-running it would have raised FileNotFoundError.

    Matched by BASENAME, not by full path, and that is deliberate: the file
    is protected wherever the classifier would send it, and a reference
    written with a different prefix (`./handoff/...`, `handoff/current/...`)
    still protects the same artifact. The cost is over-protection, which is
    the safe direction for a mover.
    """
    try:
        with MASTERPLAN.open() as f:
            mp = json.load(f)
    except (OSError, ValueError):
        # Fail SAFE, not open: if the masterplan cannot be read we cannot know
        # what is referenced, so protect nothing-by-moving-nothing is wrong
        # (it would block all work) -- instead the caller treats an empty set
        # as "no protection available" and the status gate still applies.
        return set()

    names: set[str] = set()
    pat = re.compile(r"handoff/[A-Za-z0-9_./*-]+")

    def walk(node) -> None:
        if isinstance(node, dict):
            ver = node.get("verification")
            if isinstance(ver, dict):
                blob = " ".join(
                    str(ver.get(k) or "") for k in ("command", "live_check")
                )
                for m in pat.finditer(blob):
                    names.add(Path(m.group(0)).name)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(mp)
    return names


def _git_mv(src: Path, dest: Path) -> bool:
    """Move with `git mv` so rename history survives; False if git declines.

    Mirrors `.claude/hooks/archive-handoff.sh:279`, which already prefers
    `git mv` and falls back to a plain move. This script used bare
    `shutil.move` for every relocation, so every backfill lost rename
    tracking for the files it touched.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "mv", str(src), str(dest)],
            capture_output=True,
            timeout=30,
        )
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _safe_target(dest: Path) -> Path:
    if not dest.exists():
        return dest
    n = 2
    while True:
        alt = dest.with_name(f"{dest.stem}-v{n}{dest.suffix}")
        if not alt.exists():
            return alt
        n += 1


def _move(src: Path, dest_dir: Path, dry_run: bool) -> tuple[str, Path]:
    # phase-75.11.4 cycle 5: the mkdir used to run BEFORE this check, so a
    # "dry run" CREATED every destination directory it merely planned to use.
    # Measured: a single dry run on the live tree minted three empty dirs
    # (handoff/archive/phase-80.5, -81.1, -82.23), which then classified as
    # `no_contract` and inflated the very census this step reports. A dry run
    # that mutates the tree is not a dry run.
    dest = _safe_target(dest_dir / src.name)
    if dry_run:
        return ("would-move", dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # phase-75.11.4: prefer `git mv` so rename history survives; fall back to
    # shutil.move for untracked files and non-repo (test) trees.
    if not _git_mv(src, dest):
        shutil.move(str(src), str(dest))
    return ("moved", dest)


def main(dry_run: bool) -> int:
    if not CURRENT.exists():
        print("no handoff/current/ -- nothing to do")
        return 0
    statuses = _step_statuses()
    ARCHIVE.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    MISC.mkdir(exist_ok=True)

    done_moved = 0
    # phase-75.11.4: STRUCTURALLY ZERO. The misc sweep branch is gone, so
    # nothing increments this. It is kept only so the summary line still
    # states the invariant explicitly ("we moved nothing to misc/"). Do NOT
    # write a test that asserts `misc-moved=0` in the output -- that string is
    # present under every mutant and is not coverage; assert that
    # `handoff/archive/misc/` is empty instead.
    misc_moved = 0
    ambiguous: list[str] = []
    protected_skipped = 0
    open_step_kept = 0
    no_sid_kept = 0

    protected = _masterplan_referenced_names()

    for p in sorted(CURRENT.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        if _is_rolling_keep(name) or name.startswith("."):
            continue

        # phase-75.11.4 (criterion 5) -- checked BEFORE any classification, so
        # no later branch can route a referenced artifact anywhere.
        if name in protected:
            print(f"[protected] KEEP: {name} -- named by a masterplan "
                  "verification.command/live_check")
            protected_skipped += 1
            continue

        ref = resolve_step_id(name)
        if ref is None:
            # phase-75.11.4 -- THE SWEEP IS GONE. This branch used to
            # `_move(p, MISC)` every unrecognised name, which is how a bare run
            # relocated ~664 of 668 .md files including in-flight steps'
            # artifacts. A file that names no step belongs to no step and is
            # therefore not this script's to move: report and keep.
            no_sid_kept += 1
            continue

        sid = ref.sid
        # Masterplan step ids are inconsistent: some buckets store the
        # bare `4.14.1`, others the prefixed `phase-6.1`. Try both.
        status = statuses.get(sid) or statuses.get(f"phase-{sid}")
        if is_archivable(status):
            dest_dir = ARCHIVE / f"phase-{sid}"
            verb, dest = _move(p, dest_dir, dry_run)
            print(f"[{sid}] {verb}: {name} -> {dest.relative_to(REPO)}")
            done_moved += 1
        elif status is None:
            # phase-75.11.4 -- UNKNOWN id: keep, and say so. Previously this
            # branch MOVED the file to misc/ while the summary line claimed it
            # was "left in current/ for manual review" -- the report
            # contradicted the action. Now the report is true.
            ambiguous.append(f"{name} -- sid={sid} ({ref.convention}) status=unknown")
            print(f"[warn] KEEP: {name} -- names step {sid}, which the "
                  "masterplan does not have")
        else:
            # A known but OPEN status (pending / in-progress / blocked).
            open_step_kept += 1

    audit_moved = 0
    log_moved = 0
    kept = 0
    for p in sorted(HANDOFF.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        # phase-36.7: never sweep a live production state file (see
        # HANDOFF_ROOT_KEEP). Print it so the exclusion is visible, not implicit.
        if name in HANDOFF_ROOT_KEEP:
            print(f"[root] KEEP (live state file, phase-36.7): {name}")
            kept += 1
            continue
        if name.endswith(".log"):
            # phase-36.7: this loop moved files SILENTLY -- unlike the
            # `current/` loop above it printed nothing, so not even --dry-run
            # named the kill-switch state file it was about to relocate. An
            # operator could not see it. Print every root move.
            verb, dest = _move(p, LOGS, dry_run)
            print(f"[root] {verb}: {name} -> {dest.relative_to(REPO)}")
            log_moved += 1
        elif name.endswith("_audit.json") or name.endswith("_audit.jsonl"):
            verb, dest = _move(p, AUDIT, dry_run)
            print(f"[root] {verb}: {name} -> {dest.relative_to(REPO)}")
            audit_moved += 1

    print()
    print(
        f"Summary: done-moved={done_moved} misc-moved={misc_moved} "
        f"audit-moved={audit_moved} log-moved={log_moved} "
        f"root-kept={kept} ambiguous={len(ambiguous)}"
    )
    # phase-75.11.4: report what was DELIBERATELY kept. A mover that prints
    # only what it moved reads as "nothing else was here".
    print(
        f"Kept in current/: protected={protected_skipped} "
        f"open-step={open_step_kept} no-step-id={no_sid_kept} "
        f"unknown-step={len(ambiguous)}"
    )
    if ambiguous:
        print("Unknown step id (left in current/ for manual review):")
        for a in ambiguous:
            print(f"  - {a}")
    if dry_run:
        print()
        print("DRY RUN -- nothing was moved. Re-run with --execute to apply.")
    return 0


if __name__ == "__main__":
    # phase-75.11.4 (criterion 6) -- SAFE BY DEFAULT. `--dry-run` used to be
    # OPT-IN, so a bare invocation executed shutil.move over every
    # unrecognised file. That is not a hypothetical: commit fa9aaf8e swept 315
    # files and left the verdict gate dark for 13 consecutive step closes, and
    # the module docstring advertised the bare form as normal usage.
    #
    # THE INVERSION IS CONFINED TO ARGUMENT PARSING ON PURPOSE.
    # `main(dry_run: bool)` keeps its exact signature and keeps reading the
    # module-level path globals, because
    # backend/tests/test_phase_36_7_kill_switch_rotation_rearm.py loads this
    # file by importlib, monkeypatches REPO/HANDOFF/CURRENT/... and calls
    # `mod.main(dry_run=False)`. Changing the function contract would break
    # the regression test that protects the kill switch's state file.
    ap = argparse.ArgumentParser(
        description="Archive done-step handoff artifacts. Dry-run by default."
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="actually move files (default: print the plan and change nothing)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="accepted for backward compatibility; dry-run is now the default",
    )
    args = ap.parse_args()
    if args.dry_run and args.execute:
        ap.error("--dry-run and --execute are contradictory")
    raise SystemExit(main(dry_run=not args.execute))
