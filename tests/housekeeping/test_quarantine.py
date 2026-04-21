"""phase-10.10 pytest companion to phase10_housekeeping_test.py."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.housekeeping.quarantine_phantom_archives import (
    quarantine_phantom_dirs,
    PHANTOM_RE,
    CANONICAL_RE,
    _dir_sha256,
)
from scripts.housekeeping.restore_from_quarantine import restore_from_manifest


def _build_synthetic(root: Path, canonicals: list[str], phantoms: list[str]) -> None:
    for name in canonicals + phantoms:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("contract.md", "evaluator_critique.md", "experiment_results.md", "research_brief.md"):
            (d / fn).write_text(f"content of {name}/{fn}\n", encoding="utf-8")


def test_dry_run_default_moves_nothing(tmp_path):
    _build_synthetic(tmp_path, ["phase-10.0"], ["phase-10.0-v2"])
    result = quarantine_phantom_dirs(archive_root=tmp_path)
    assert result["dry_run"] is True
    assert result["moved"] == 0
    assert result["would_move"] == 1
    # phantom still at original path
    assert (tmp_path / "phase-10.0-v2").is_dir()


def test_phantoms_moved_canonicals_untouched(tmp_path):
    _build_synthetic(
        tmp_path,
        ["phase-10.0", "phase-8.5.1"],
        ["phase-10.0-v2", "phase-10.0-v3", "phase-8.5.1-v5"],
    )
    result = quarantine_phantom_dirs(archive_root=tmp_path, dry_run=False)
    assert result["moved"] == 3
    assert result["skipped_canonical"] == 2
    assert (tmp_path / "phase-10.0").is_dir()
    assert (tmp_path / "phase-8.5.1").is_dir()
    assert not (tmp_path / "phase-10.0-v2").exists()
    assert (tmp_path / "_quarantine_2026-04-21" / "phase-10.0-v2").is_dir()


def test_manifest_has_required_keys_and_valid_sha(tmp_path):
    _build_synthetic(tmp_path, ["phase-10.0"], ["phase-10.0-v2", "phase-10.0-v3"])
    result = quarantine_phantom_dirs(archive_root=tmp_path, dry_run=False)
    manifest = Path(result["manifest"])
    entries = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(entries) == 2
    sha_re = re.compile(r"^[0-9a-f]{64}$")
    for e in entries:
        assert sha_re.match(e["dir_sha256"])
        assert {"original_path", "quarantine_path", "size_bytes", "file_count", "moved_at_iso"}.issubset(e.keys())
        assert e["file_count"] == 4
        assert e["size_bytes"] > 0


def test_restore_reverses_quarantine(tmp_path):
    _build_synthetic(tmp_path, [], ["phase-10.0-v2", "phase-10.0-v3"])
    # Snapshot contents
    before = {
        d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
        for d in tmp_path.iterdir()
        if d.is_dir() and PHANTOM_RE.match(d.name)
    }
    q = quarantine_phantom_dirs(archive_root=tmp_path, dry_run=False)
    r = restore_from_manifest(manifest_path=Path(q["manifest"]), dry_run=False)
    assert r["restored"] == 2
    assert r["mismatches"] == []

    after = {
        d.name: sorted((f.name, f.read_text(encoding="utf-8")) for f in d.iterdir())
        for d in tmp_path.iterdir()
        if d.is_dir() and PHANTOM_RE.match(d.name)
    }
    assert before == after


def test_regex_gates():
    assert CANONICAL_RE.match("phase-10.0")
    assert CANONICAL_RE.match("phase-8.5.1")
    assert not CANONICAL_RE.match("phase-10.0-v2")
    assert PHANTOM_RE.match("phase-10.0-v2")
    assert PHANTOM_RE.match("phase-8.5.10-v16")
    assert not PHANTOM_RE.match("phase-10.0")
    # Neither matches random non-phase dirs
    assert not PHANTOM_RE.match("_templates")
    assert not CANONICAL_RE.match("_quarantine_2026-04-21")


def test_dir_sha256_deterministic_and_unique(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    (a / "f.md").write_text("same content")
    (b / "f.md").write_text("same content")
    (c / "f.md").write_text("different content")
    ha, _, _ = _dir_sha256(a)
    hb, _, _ = _dir_sha256(b)
    hc, _, _ = _dir_sha256(c)
    # Same content at same relative path -> same hash
    assert ha == hb
    # Different content -> different hash
    assert ha != hc


def test_skipped_canonical_counter(tmp_path):
    _build_synthetic(
        tmp_path,
        ["phase-10.0", "phase-8.5.1", "phase-9.3"],
        ["phase-10.0-v2"],
    )
    result = quarantine_phantom_dirs(archive_root=tmp_path, dry_run=False)
    assert result["skipped_canonical"] == 3


def test_missing_archive_root_raises(tmp_path):
    import pytest
    nonexistent = tmp_path / "not_a_dir"
    with pytest.raises(ValueError, match="not a directory"):
        quarantine_phantom_dirs(archive_root=nonexistent, dry_run=False)


def test_manifest_written_before_move_crash_resilience(tmp_path, monkeypatch):
    """qa_1010_v1 cycle-2 fix: manifest entry must be on disk BEFORE the move.

    If `shutil.move` crashes mid-run, every phantom that was ABOUT to move
    still has a manifest entry -- making the crashed work recoverable. Without
    this test, flipping the write order to AFTER the move is undetectable
    (the "crash-recovery invariant" stated in the code's safety comment would
    be asserted but not tested — qa_1010_v1 M3 gap).
    """
    import shutil as _shutil
    import json as _json

    _build_synthetic(tmp_path, [], ["phase-10.0-v2", "phase-10.0-v3", "phase-10.0-v4"])

    call_count = {"n": 0}
    real_move = _shutil.move

    def flaky_move(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-run crash")
        return real_move(src, dst)

    import scripts.housekeeping.quarantine_phantom_archives as qpa
    monkeypatch.setattr(qpa.shutil, "move", flaky_move)

    result = quarantine_phantom_dirs(archive_root=tmp_path, dry_run=False)

    manifest = Path(result["manifest"])
    entries = [_json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]

    # CRITICAL INVARIANT: 3 manifest entries for 3 phantoms (1 crashed + 2 ok).
    # Only possible if the manifest write happens BEFORE the move attempt.
    # Flipping the order would leave the crashed phantom unrecoverable (no entry)
    # -> len(entries) would be 2 instead of 3.
    assert len(entries) == 3, (
        f"manifest-before-move invariant violated: {len(entries)} entries after "
        "2 successful moves + 1 crash; crashed phantom would be unrecoverable"
    )

    # Exactly 2 phantoms physically moved (v2 success, v3 crash, v4 success).
    quarantine_dir = tmp_path / "_quarantine_2026-04-21"
    moved_dirs = [d for d in quarantine_dir.iterdir() if d.is_dir() and PHANTOM_RE.match(d.name)]
    assert len(moved_dirs) == 2

    # All 3 phantom original_paths appear in the manifest -- including the
    # crashed one, which is the whole point of manifest-before-move.
    manifest_sources = {Path(e["original_path"]).name for e in entries}
    assert manifest_sources == {"phase-10.0-v2", "phase-10.0-v3", "phase-10.0-v4"}
