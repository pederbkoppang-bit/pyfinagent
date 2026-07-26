"""phase-4.16.2 one-time backfill for handoff folder layout.

Reads .claude/masterplan.json. For each step with status=done, moves
any matching `handoff/current/phase-<sid>-*.md` (and `<sid>-*.md`)
into `handoff/archive/phase-<sid>/`. Non-conforming files (no step-id
prefix) go to `handoff/archive/misc/`. Root-level audit JSON + log
files move to `handoff/audit/` + `handoff/logs/`.

Idempotent: if target path exists, appends `-v2`, `-v3`, ... suffix
so prior evidence is never clobbered (mirrors `archive-handoff.sh`).

Usage:
    python scripts/housekeeping/backfill_handoff_archive.py --dry-run
    python scripts/housekeeping/backfill_handoff_archive.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

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
}

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
# backend/tests/test_phase_36_8_archive_merge_authority.py::
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
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _safe_target(dest_dir / src.name)
    if dry_run:
        return ("would-move", dest)
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
    misc_moved = 0
    ambiguous: list[str] = []

    for p in sorted(CURRENT.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        if name in ROLLING_KEEP or name.startswith("."):
            continue
        m = STEP_ID_RE.match(name)
        sid = m.group(1) if m else None
        if sid is None:
            verb, dest = _move(p, MISC, dry_run)
            print(f"[misc] {verb}: {name} -> {dest.relative_to(REPO)}")
            misc_moved += 1
            continue
        # Masterplan step ids are inconsistent: some buckets store the
        # bare `4.14.1`, others the prefixed `phase-6.1`. Try both.
        status = statuses.get(sid) or statuses.get(f"phase-{sid}")
        if status == "done":
            dest_dir = ARCHIVE / f"phase-{sid}"
            verb, dest = _move(p, dest_dir, dry_run)
            print(f"[{sid}] {verb}: {name} -> {dest.relative_to(REPO)}")
            done_moved += 1
        elif status in ("pending", "in-progress", "blocked"):
            continue
        else:
            # Unknown / parent-phase id -- route to misc (flagged).
            ambiguous.append(f"{name} -- sid={sid} status={status!r}")
            verb, dest = _move(p, MISC, dry_run)
            print(f"[misc:ambig] {verb}: {name} -> {dest.relative_to(REPO)}")
            misc_moved += 1

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
    if ambiguous:
        print("Ambiguous (left in current/ for manual review):")
        for a in ambiguous:
            print(f"  - {a}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
