"""phase-6.5.2 tests for backend.intel.scanner."""
from __future__ import annotations

from pathlib import Path

from backend.intel.scanner import (
    BaseScanner,
    DocumentCandidate,
    _REQUIRED_CANDIDATE_KEYS,
    _canonicalize,
    _content_hash,
)
from backend.intel.source_registry import SourceRow


def _make_source(**overrides) -> SourceRow:
    defaults = dict(
        source_id="t_src",
        source_name="Test Source",
        source_type="http",
        kill_switch=False,
        rate_limit_per_day=100,
        metadata={"url": "https://stub.example.com/doc"},
    )
    defaults.update(overrides)
    return SourceRow(**defaults)


def test_dry_run_returns_stub_candidate():
    """Immutable criterion: scanner_dry_run_returns_candidates."""
    scanner = BaseScanner(_make_source())
    cands = scanner.scan(dry_run=True)
    assert isinstance(cands, list)
    assert len(cands) == 1
    cand = cands[0]
    # Every required key for an intel_documents row is present.
    for key in _REQUIRED_CANDIDATE_KEYS:
        assert key in cand, key
    assert cand["source_id"] == "t_src"
    assert cand["source_type"] == "http"
    assert cand["doc_type"] == "stub"
    assert cand["content_hash"]
    assert cand["canonical_url"].startswith("https://stub.example.com/")


def test_dry_run_is_deterministic_for_hash():
    src = _make_source()
    h1 = BaseScanner(src).scan(dry_run=True)[0]["content_hash"]
    h2 = BaseScanner(src).scan(dry_run=True)[0]["content_hash"]
    assert h1 == h2


def test_scan_fail_open_on_network_error(monkeypatch):
    scanner = BaseScanner(_make_source())
    monkeypatch.setattr(
        scanner, "_do_scan", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = scanner.scan(dry_run=False)
    assert result == []


def test_scan_returns_empty_when_source_missing_url():
    scanner = BaseScanner(_make_source(metadata={}))
    result = scanner.scan(dry_run=False)
    assert result == []


def test_document_candidate_fields_align_with_schema():
    """Ensure the required keys match the intel_documents column set."""
    needed = {
        "doc_id", "source_id", "source_type", "ingested_at",
        "url", "canonical_url", "content_hash",
    }
    assert _REQUIRED_CANDIDATE_KEYS >= needed


def test_canonicalize_strips_fragment_and_trailing_slash():
    assert _canonicalize("https://x.com/a/#frag") == "https://x.com/a"
    assert _canonicalize("https://x.com/b/") == "https://x.com/b"
    assert _canonicalize("https://x.com/c") == "https://x.com/c"


def test_content_hash_normalizes_whitespace():
    assert _content_hash("hello   world") == _content_hash("hello world")
    assert _content_hash("a") != _content_hash("b")


def test_intra_batch_dedup_removes_duplicate_canonical_url(monkeypatch):
    scanner = BaseScanner(_make_source())
    dup = DocumentCandidate(
        doc_id="1", source_id="x", source_type="http",
        ingested_at="t", url="u", canonical_url="u", content_hash="h",
    )
    monkeypatch.setattr(scanner, "_do_scan", lambda: [dict(dup), dict(dup), dict(dup)])
    result = scanner.scan(dry_run=False)
    assert len(result) == 1


def test_scanner_module_is_ascii_only():
    """ASCII-only discipline per .claude/rules/security.md (logger cp1252 protection)."""
    mod_path = Path(__file__).resolve().parents[1] / "intel" / "scanner.py"
    mod_path.read_bytes().decode("ascii")


def test_registry_module_is_ascii_only():
    mod_path = Path(__file__).resolve().parents[1] / "intel" / "source_registry.py"
    mod_path.read_bytes().decode("ascii")
