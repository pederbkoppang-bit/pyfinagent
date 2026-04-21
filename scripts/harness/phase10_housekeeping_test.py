"""phase-10.10 verification CLI: non-destructive housekeeping quarantine.

Four cases matching the masterplan success_criteria:
  1. phantom_dirs_moved_not_deleted
  2. canonical_dirs_untouched
  3. manifest_written_with_sha256_per_dir
  4. quarantine_is_reversible

Each case runs in a tempfile.TemporaryDirectory with a synthetic archive
tree. The real handoff/archive/ is NEVER touched.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.housekeeping.quarantine_phantom_archives import (
    quarantine_phantom_dirs,
    PHANTOM_RE,
    CANONICAL_RE,
)
from scripts.housekeeping.restore_from_quarantine import restore_from_manifest


def _build_synthetic_archive(root: Path) -> None:
    """3 canonical + 5 phantom dirs, each with 4 rolling files."""
    canonicals = ["phase-10.0", "phase-8.5.1", "phase-9.3"]
    phantoms = [
        "phase-10.0-v2",
        "phase-10.0-v3",
        "phase-10.0-v4",
        "phase-8.5.1-v2",
        "phase-9.3-v5",
    ]
    rolling_files = [
        "contract.md",
        "evaluator_critique.md",
        "experiment_results.md",
        "research_brief.md",
    ]
    for name in canonicals + phantoms:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for fn in rolling_files:
            (d / fn).write_text(f"content of {name}/{fn}\n", encoding="utf-8")


def case_phantom_dirs_moved_not_deleted() -> bool:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_synthetic_archive(root)
        before = sorted(d.name for d in root.iterdir() if d.is_dir())

        result = quarantine_phantom_dirs(archive_root=root, dry_run=False)
        quarantine = root / "_quarantine_2026-04-21"

        after_root = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name != "_quarantine_2026-04-21"
        )
        quarantined = sorted(d.name for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else []

        phantoms_gone_from_root = not any(PHANTOM_RE.match(n) for n in after_root)
        ok = (
            result["moved"] == 5
            and quarantine.exists()
            and len(quarantined) == 5
            and phantoms_gone_from_root
            and all(PHANTOM_RE.match(n) for n in quarantined)
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] phantom_dirs_moved_not_deleted  "
            f"(moved={result['moved']}, quarantined={len(quarantined)}, root_phantoms_gone={phantoms_gone_from_root})"
        )
        return ok


def case_canonical_dirs_untouched() -> bool:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_synthetic_archive(root)
        # Snapshot canonical contents BEFORE
        canonical_files_before = {
            d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
            for d in root.iterdir()
            if d.is_dir() and CANONICAL_RE.match(d.name)
        }

        result = quarantine_phantom_dirs(archive_root=root, dry_run=False)

        canonical_files_after = {
            d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
            for d in root.iterdir()
            if d.is_dir() and CANONICAL_RE.match(d.name)
        }

        ok = (
            result["skipped_canonical"] == 3
            and canonical_files_before == canonical_files_after
            and set(canonical_files_after.keys())
                == {"phase-10.0", "phase-8.5.1", "phase-9.3"}
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] canonical_dirs_untouched  "
            f"(skipped_canonical={result['skipped_canonical']}, content_identical={canonical_files_before == canonical_files_after})"
        )
        return ok


def case_manifest_written_with_sha256_per_dir() -> bool:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_synthetic_archive(root)

        result = quarantine_phantom_dirs(archive_root=root, dry_run=False)

        manifest = Path(result["manifest"])
        if not manifest.exists():
            print("[FAIL] manifest_written_with_sha256_per_dir  (manifest not found)")
            return False

        entries = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        sha_re = re.compile(r"^[0-9a-f]{64}$")

        all_have_sha = all(sha_re.match(e.get("dir_sha256", "")) for e in entries)
        required_keys = {"original_path", "quarantine_path", "dir_sha256", "size_bytes", "file_count", "moved_at_iso"}
        all_keys_present = all(required_keys.issubset(e.keys()) for e in entries)

        ok = len(entries) == 5 and all_have_sha and all_keys_present
        print(
            f"[{'PASS' if ok else 'FAIL'}] manifest_written_with_sha256_per_dir  "
            f"(entries={len(entries)}, all_sha={all_have_sha}, all_keys={all_keys_present})"
        )
        return ok


def case_quarantine_is_reversible() -> bool:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _build_synthetic_archive(root)

        # Snapshot phantom contents BEFORE quarantine
        phantoms_before = {
            d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
            for d in root.iterdir()
            if d.is_dir() and PHANTOM_RE.match(d.name)
        }

        q_result = quarantine_phantom_dirs(archive_root=root, dry_run=False)
        r_result = restore_from_manifest(manifest_path=Path(q_result["manifest"]), dry_run=False)

        # After restore, phantoms back at original paths
        phantoms_after = {
            d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
            for d in root.iterdir()
            if d.is_dir() and PHANTOM_RE.match(d.name)
        }

        ok = (
            r_result["restored"] == 5
            and len(r_result["mismatches"]) == 0
            and phantoms_before == phantoms_after
        )
        print(
            f"[{'PASS' if ok else 'FAIL'}] quarantine_is_reversible  "
            f"(restored={r_result['restored']}, mismatches={len(r_result['mismatches'])}, identical={phantoms_before == phantoms_after})"
        )
        return ok


def main() -> int:
    results = [
        case_phantom_dirs_moved_not_deleted(),
        case_canonical_dirs_untouched(),
        case_manifest_written_with_sha256_per_dir(),
        case_quarantine_is_reversible(),
    ]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'FAILED'}  ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
