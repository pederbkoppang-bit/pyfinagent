"""phase-6.5.7 novelty client (Voyage primary, Gemini fallback).

Provides:
- `embed(text)` -> list[float]  -- tries Voyage first, falls back to Gemini
- `novelty_score(chunk_text, candidates)` -> (float, int)
    Score = 1 - max_cosine_similarity against a set of candidate embeddings.
    Empty candidates => fully novel (1.0).
- `score_chunks_and_write(chunks, ...)` -> int  -- writes novelty rows to BQ.

Both providers are forced to 1024 dimensions. Voyage-finance-2 is natively
1024-dim; Gemini `gemini-embedding-001` is 3072 by default and requires
`output_dimensionality=1024` (research brief R2).

The stub embedder (`_stub_embed`) produces deterministic 1024-float vectors
from sha256(text) and is used by tests + as a last-resort fallback in case
both live providers are unreachable (e.g. air-gapped dev run).

ASCII-only logger messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

_NOVELTY_SCORES_TABLE = "intel_novelty_scores"
_EMBED_DIM = 1024
_VOYAGE_DEFAULT_MODEL = "voyage-finance-2"
_GEMINI_DEFAULT_MODEL = "gemini-embedding-001"


def _stub_embed(text: str, *, model: str | None = None) -> list[float]:
    """Deterministic pseudo-embedding: sha256(text) tiled to 1024 floats in [-1, 1]."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = list(digest) * ((_EMBED_DIM // len(digest)) + 1)
    return [(b / 127.5) - 1.0 for b in raw[:_EMBED_DIM]]


def _embed_voyage(text: str, *, model: str | None = None) -> list[float]:
    """Voyage AI embedding. Raises on any failure."""
    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY absent")
    try:
        import voyageai  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(f"voyageai import failed: {exc!r}")
    client = voyageai.Client()
    use_model = model or os.environ.get("NOVELTY_EMBED_MODEL", _VOYAGE_DEFAULT_MODEL)
    result = client.embed([text], model=use_model, input_type="document")
    vec = result.embeddings[0]
    if len(vec) != _EMBED_DIM:
        raise RuntimeError(f"voyage returned dim={len(vec)}, want {_EMBED_DIM}")
    return list(vec)


def _embed_gemini(text: str, *, model: str | None = None) -> list[float]:
    """Gemini embedding fallback. Forces output_dimensionality=1024."""
    try:
        from google import genai  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(f"google.genai import failed: {exc!r}")
    client = genai.Client()
    use_model = model or _GEMINI_DEFAULT_MODEL
    resp = client.models.embed_content(
        model=use_model,
        contents=text,
        config={"output_dimensionality": _EMBED_DIM},
    )
    vec = resp.embeddings[0].values
    if len(vec) != _EMBED_DIM:
        raise RuntimeError(f"gemini returned dim={len(vec)}, want {_EMBED_DIM}")
    return list(vec)


_PROVIDERS: tuple[Callable[..., list[float]], ...] = (_embed_voyage, _embed_gemini)


def embed(text: str, *, model: str | None = None) -> list[float]:
    """Try Voyage, fall back to Gemini. Raise only if BOTH fail."""
    errors: list[tuple[str, str]] = []
    for provider in _PROVIDERS:
        try:
            return provider(text, model=model)
        except Exception as exc:
            errors.append((provider.__name__, repr(exc)))
            logger.warning(
                "novelty_client: provider %s failed: %r", provider.__name__, exc
            )
    raise RuntimeError(f"all embedding providers failed: {errors}")


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def novelty_score(
    chunk_text: str,
    candidate_embeddings: list[list[float]],
    *,
    embedder: Callable[..., list[float]] = embed,
) -> tuple[float, int]:
    """Novelty = 1 - max cosine similarity. Empty candidates => 1.0 (fully novel)."""
    if not candidate_embeddings:
        return 1.0, -1
    q = embedder(chunk_text)
    best_sim = -1.0
    best_idx = -1
    for i, cand in enumerate(candidate_embeddings):
        sim = _cosine(q, cand)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return (1.0 - best_sim), best_idx


def _resolve_target(project: str | None, dataset: str | None) -> tuple[str, str]:
    proj = project
    ds = dataset
    if proj is None or ds is None:
        try:
            from backend.config.settings import get_settings

            s = get_settings()
            if proj is None:
                proj = s.gcp_project_id
            if ds is None:
                ds = getattr(s, "bq_dataset_observability", None) or "pyfinagent_data"
        except Exception as exc:  # pragma: no cover
            logger.warning("novelty_client: settings load failed: %r", exc)
            proj = proj or ""
            ds = ds or "pyfinagent_data"
    return proj or "", ds or "pyfinagent_data"


def _get_client(project: str) -> Any:
    try:
        from google.cloud import bigquery  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("novelty_client: google-cloud-bigquery absent (%r)", exc)
        return None
    try:
        return bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as exc:
        logger.warning("novelty_client: bigquery.Client() init failed (%r)", exc)
        return None


def score_chunks_and_write(
    chunks: Iterable[dict[str, Any]],
    *,
    candidate_embeddings: list[list[float]] | None = None,
    project: str | None = None,
    dataset: str | None = None,
    embedder: Callable[..., list[float]] = embed,
    scorer_model: str = _VOYAGE_DEFAULT_MODEL,
    scorer_version: str = "v1",
) -> int:
    """Score each chunk, write novelty rows to BQ. Fail-open."""
    rows: list[dict[str, Any]] = []
    candidates = list(candidate_embeddings or [])
    now_iso = datetime.now(timezone.utc).isoformat()
    for c in chunks:
        chunk_id = c.get("chunk_id")
        chunk_text = c.get("chunk_text", "")
        if not chunk_id:
            continue
        try:
            score, nn = novelty_score(chunk_text, candidates, embedder=embedder)
        except Exception as exc:
            logger.warning("novelty_client score fail-open: %r", exc)
            continue
        nn_chunk_id = None
        nn_distance = None
        if nn >= 0:
            nn_chunk_id = f"candidate_{nn}"
            nn_distance = 1.0 - score
        rows.append(
            {
                "chunk_id": chunk_id,
                "scorer_model": scorer_model,
                "scorer_version": scorer_version,
                "scored_at": now_iso,
                "novelty_score": score,
                "nearest_neighbor_chunk_id": nn_chunk_id,
                "nearest_neighbor_distance": nn_distance,
                "latency_ms": None,
                "cost_usd": None,
            }
        )
    if not rows:
        return 0
    proj, ds = _resolve_target(project, dataset)
    client = _get_client(proj)
    if client is None:
        return 0
    try:
        table_ref = f"{proj}.{ds}.{_NOVELTY_SCORES_TABLE}" if proj else f"{ds}.{_NOVELTY_SCORES_TABLE}"
        errors = client.insert_rows_json(table_ref, rows)
        if errors:
            logger.warning("novelty_client insert errors: %s", errors)
            return 0
        return len(rows)
    except Exception as exc:
        logger.warning("novelty_client write fail-open: %r", exc)
        return 0
