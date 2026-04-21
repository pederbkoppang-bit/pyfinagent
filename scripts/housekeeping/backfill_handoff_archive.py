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
import sys
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
    for p in sorted(HANDOFF.iterdir()):
        if p.is_dir():
            continue
        name = p.name
        if name.endswith(".log"):
            _move(p, LOGS, dry_run)
            log_moved += 1
        elif name.endswith("_audit.json") or name.endswith("_audit.jsonl"):
            _move(p, AUDIT, dry_run)
            audit_moved += 1

    print()
    print(
        f"Summary: done-moved={done_moved} misc-moved={misc_moved} "
        f"audit-moved={audit_moved} log-moved={log_moved} "
        f"ambiguous={len(ambiguous)}"
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
