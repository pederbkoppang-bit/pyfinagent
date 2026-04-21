"""phase-10.10 Restore-from-quarantine companion script.

Reads a JSONL manifest produced by `quarantine_phantom_archives.py` and
reverses every move: quarantine_path -> original_path. Verifies post-restore
`dir_sha256` matches the manifest value; mismatches are reported but do not
abort (forensic value preserved).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.housekeeping.quarantine_phantom_archives import _dir_sha256  # noqa: E402

logger = logging.getLogger(__name__)


def restore_from_manifest(
    *,
    manifest_path: Path,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Restore each entry in the manifest to its original path.

    Returns `{restored, mismatches, skipped, dry_run}`.
    """
    mpath = Path(manifest_path)
    if not mpath.is_file():
        raise ValueError(f"manifest not found: {mpath}")

    restored = 0
    mismatches: list[str] = []
    skipped: list[str] = []

    with mpath.open("r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception as exc:
                logger.warning("restore: bad manifest line: %r", exc)
                skipped.append(line[:80])
                continue

            src = Path(entry["quarantine_path"])
            dst = Path(entry["original_path"])
            expected_sha = entry["dir_sha256"]

            if not src.exists():
                skipped.append(f"source missing: {src}")
                continue
            if dst.exists():
                skipped.append(f"destination occupied: {dst}")
                continue

            if dry_run:
                restored += 1  # count as would-restore
                continue

            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            except Exception as exc:
                logger.warning("restore: move failed %s -> %s: %r", src, dst, exc)
                skipped.append(f"move failed: {src}")
                continue

            try:
                actual_sha, _, _ = _dir_sha256(dst)
                if actual_sha != expected_sha:
                    mismatches.append(
                        f"{dst}: expected {expected_sha[:12]}... got {actual_sha[:12]}..."
                    )
            except Exception as exc:
                logger.warning("restore: hash after move failed for %s: %r", dst, exc)

            restored += 1

    return {
        "restored": restored,
        "mismatches": mismatches,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Restore phantom dirs from a quarantine manifest.")
    ap.add_argument("--manifest", required=True, help="Path to MANIFEST.jsonl")
    ap.add_argument("--no-dry-run", action="store_true", help="Actually restore (default: dry-run)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = restore_from_manifest(
        manifest_path=Path(args.manifest),
        dry_run=not args.no_dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["restore_from_manifest"]


if __name__ == "__main__":
    raise SystemExit(main())
