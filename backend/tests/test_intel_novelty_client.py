"""phase-6.5.7 tests for backend.intel.novelty_client."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.intel import novelty_client as nc


def test_stub_embed_is_deterministic():
    a = nc._stub_embed("same text")
    b = nc._stub_embed("same text")
    assert a == b
    assert len(a) == 1024
    assert all(isinstance(x, float) for x in a)
    assert all(-1.0 <= x <= 1.0 for x in a)


def test_stub_embed_distinguishes_text():
    a = nc._stub_embed("alpha")
    b = nc._stub_embed("beta")
    assert a != b


def test_voyage_primary_gemini_fallback_smoke_ok(monkeypatch):
    """Immutable criterion: voyage_primary_gemini_fallback_smoke_ok.

    Voyage fails; Gemini succeeds (with stub). `embed()` returns the Gemini vector.
    """
    def _raise(text, *, model=None):
        raise RuntimeError("voyage unavailable")

    monkeypatch.setattr(nc, "_embed_voyage", _raise)
    monkeypatch.setattr(
        nc, "_embed_gemini",
        lambda text, *, model=None: nc._stub_embed(text),
    )
    # Refresh the provider tuple to point at the patched funcs.
    monkeypatch.setattr(nc, "_PROVIDERS", (nc._embed_voyage, nc._embed_gemini))

    result = nc.embed("any smoke text")
    assert len(result) == 1024
    assert all(isinstance(v, float) for v in result)


def test_embed_both_providers_fail_raises_runtime_error(monkeypatch):
    def _raise(text, *, model=None):
        raise RuntimeError("unreachable")

    monkeypatch.setattr(nc, "_embed_voyage", _raise)
    monkeypatch.setattr(nc, "_embed_gemini", _raise)
    monkeypatch.setattr(nc, "_PROVIDERS", (nc._embed_voyage, nc._embed_gemini))

    with pytest.raises(RuntimeError) as exc:
        nc.embed("x")
    assert "all embedding providers failed" in str(exc.value)


def test_novelty_score_distinguishes_duplicate_vs_novel(monkeypatch):
    """Immutable criterion: novelty_score_distinguishes_duplicate_vs_novel."""
    monkeypatch.setattr(
        nc, "_embed_voyage",
        lambda text, *, model=None: nc._stub_embed(text),
    )
    monkeypatch.setattr(nc, "_PROVIDERS", (nc._embed_voyage,))

    text = "AAPL earnings beat estimate"
    same = nc._stub_embed(text)
    diff = nc._stub_embed("completely unrelated string about weather")

    dup_score, _ = nc.novelty_score(text, [same])
    novel_score, _ = nc.novelty_score(text, [diff])

    assert dup_score < 0.1, f"duplicate score too high: {dup_score}"
    assert novel_score > 0.5, f"novel score too low: {novel_score}"


def test_novelty_score_empty_candidates_returns_one():
    score, nn = nc.novelty_score("anything", [], embedder=nc._stub_embed)
    assert score == 1.0
    assert nn == -1


def test_novelty_score_nearest_neighbor_index(monkeypatch):
    monkeypatch.setattr(
        nc, "_embed_voyage",
        lambda text, *, model=None: nc._stub_embed(text),
    )
    monkeypatch.setattr(nc, "_PROVIDERS", (nc._embed_voyage,))

    target = "the target string"
    cands = [
        nc._stub_embed("unrelated a"),
        nc._stub_embed(target),  # index 1 is the nearest
        nc._stub_embed("unrelated b"),
    ]
    score, nn = nc.novelty_score(target, cands)
    assert nn == 1
    assert score < 0.1


def test_cosine_basic():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert abs(nc._cosine(a, b) - 1.0) < 1e-9
    assert abs(nc._cosine(a, c) - 0.0) < 1e-9


def test_cosine_mismatch_returns_zero():
    assert nc._cosine([1.0, 2.0], [1.0]) == 0.0
    assert nc._cosine([], []) == 0.0


def test_score_chunks_and_write_fail_open_no_bq(monkeypatch):
    """Writing with a bad BQ project must return 0 and never raise."""
    chunks = [{"chunk_id": "c1", "chunk_text": "hello"}]
    result = nc.score_chunks_and_write(
        chunks,
        candidate_embeddings=[],
        project="nonexistent-fail-open-test",
        dataset="nx",
        embedder=nc._stub_embed,
    )
    assert result == 0


def test_module_is_ascii_only():
    mod_path = Path(__file__).resolve().parents[1] / "intel" / "novelty_client.py"
    mod_path.read_bytes().decode("ascii")
