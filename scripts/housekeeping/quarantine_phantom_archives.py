"""phase-10.10 Non-destructive quarantine of phantom `-vN` archive dirs.

The archive-handoff hook race left 12,784 phantom directories under
`handoff/archive/` (pattern `phase-X.Y-vN` for N>=2), each a byte-identical
duplicate of the canonical `phase-X.Y` dir. This script moves them (NOT
deletes) into a timestamped `_quarantine_*/` subdir, emitting a JSONL
manifest so the move is fully reversible via `restore_from_quarantine.py`.

Default is `--dry-run`: no file system mutation until the operator passes
`--no-dry-run`. Canonical dirs are regex-gated and never touched.

Safety invariants:
  1. Canonical dirs (pattern `^phase-\\d+(?:\\.\\d+)+$`) are never moved.
  2. Manifest entry is written (and flushed) BEFORE the move -- crash mid-run
     leaves a recoverable partial manifest.
  3. `shutil.move()` on same APFS volume = atomic `os.rename()` -- no data copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# PHANTOM covers two bug-generated name shapes:
#   phase-X.Y-vN         (standard hook race)
#   phase-phase-X.Y-vN   (older double-prefix bug)
PHANTOM_RE = re.compile(r"^phase-(?:phase-)?\d+(?:\.\d+)+-v\d+$")
CANONICAL_RE = re.compile(r"^phase-\d+(?:\.\d+)+$")

_DEFAULT_QUARANTINE_SUBDIR = "_quarantine_2026-04-21"


def _dir_sha256(path: Path) -> tuple[str, int, int]:
    """SHA-256 of a directory tree's file contents.

    Walks files in sorted relative-path order, hashes each file's bytes,
    combines into one digest. Returns (hexdigest, size_bytes, file_count).
    Directories, symlinks, and other non-regular files are skipped.
    """
    hasher = hashlib.sha256()
    total_size = 0
    file_count = 0
    root = path.resolve()
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            files.append(p)
    files.sort(key=lambda p: str(p.relative_to(root)))
    for f in files:
        rel = str(f.relative_to(root)).encode("utf-8")
        hasher.update(rel + b"\0")  # path-in-digest so rename detected
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        total_size += size
        file_count += 1
        try:
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    hasher.update(chunk)
        except OSError as exc:
            logger.warning("quarantine: read skipped %s: %r", f, exc)
    return hasher.hexdigest(), total_size, file_count


def quarantine_phantom_dirs(
    *,
    archive_root: Path,
    quarantine_subdir: str = _DEFAULT_QUARANTINE_SUBDIR,
    dry_run: bool = True,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Move phantom `-vN` dirs under `archive_root` into `quarantine_subdir/`.

    Returns `{moved, skipped_canonical, manifest, reversible, dry_run}`.
    """
    archive_root = Path(archive_root).resolve()
    if not archive_root.is_dir():
        raise ValueError(f"archive_root is not a directory: {archive_root}")

    quarantine = archive_root / quarantine_subdir
    mpath = Path(manifest_path) if manifest_path is not None else quarantine / "MANIFEST.jsonl"

    moved = 0
    skipped_canonical = 0
    to_move: list[Path] = []

    for entry in sorted(archive_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name == quarantine_subdir:
            continue
        if CANONICAL_RE.match(name):
            skipped_canonical += 1
            continue
        if PHANTOM_RE.match(name):
            to_move.append(entry)
        # dirs that match neither (e.g., "_templates", "_quarantine_*") are left alone

    if dry_run:
        return {
            "moved": 0,
            "skipped_canonical": skipped_canonical,
            "would_move": len(to_move),
            "manifest": str(mpath),
            "reversible": True,
            "dry_run": True,
        }

    quarantine.mkdir(parents=True, exist_ok=True)
    mpath.parent.mkdir(parents=True, exist_ok=True)

    with mpath.open("a", encoding="utf-8") as mf:
        for phantom in to_move:
            dest = quarantine / phantom.name
            if dest.exists():
                logger.warning("quarantine: dest exists, skipping: %s", dest)
                continue
            try:
                digest, size, count = _dir_sha256(phantom)
            except Exception as exc:
                logger.warning("quarantine: hash failed for %s: %r", phantom, exc)
                continue

            entry: dict[str, Any] = {
                "original_path": str(phantom),
                "quarantine_path": str(dest),
                "dir_sha256": digest,
                "size_bytes": size,
                "file_count": count,
                "moved_at_iso": datetime.now(timezone.utc).isoformat(),
            }
            # Manifest-BEFORE-move: if we crash mid-move, the entry documents intent
            # and the phantom is still at original_path. If the move completes, the
            # entry is the recovery key. Verified by
            # `test_manifest_written_before_move_crash_resilience`.
            mf.write(json.dumps(entry, sort_keys=True) + "\n")
            mf.flush()

            try:
                shutil.move(str(phantom), str(dest))
                moved += 1
            except Exception as exc:
                logger.warning("quarantine: move failed for %s -> %s: %r", phantom, dest, exc)

    return {
        "moved": moved,
        "skipped_canonical": skipped_canonical,
        "would_move": len(to_move),
        "manifest": str(mpath),
        "reversible": True,
        "dry_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Non-destructively quarantine phantom -vN archive dirs."
    )
    ap.add_argument(
        "--archive-root",
        default="handoff/archive",
        help="Root directory containing phase-X.Y* subdirs (default: handoff/archive)",
    )
    ap.add_argument(
        "--quarantine-subdir",
        default=_DEFAULT_QUARANTINE_SUBDIR,
        help=f"Name of the quarantine subdir (default: {_DEFAULT_QUARANTINE_SUBDIR})",
    )
    ap.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually move dirs. Without this flag, only counts what would move.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Manifest JSONL path (default: <quarantine>/MANIFEST.jsonl)",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = quarantine_phantom_dirs(
        archive_root=Path(args.archive_root),
        quarantine_subdir=args.quarantine_subdir,
        dry_run=not args.no_dry_run,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "quarantine_phantom_dirs",
    "PHANTOM_RE",
    "CANONICAL_RE",
]


if __name__ == "__main__":
    raise SystemExit(main())
