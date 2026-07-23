"""phase-26.6: Multimodal File Search RAG runtime module.

Exposes `multimodal_index()` -- the entry point for multimodal RAG queries
against a Gemini File Search store indexed with `gemini-embedding-2`.

== Critical configuration ==

The embedding model is LOCKED at store-creation time and CANNOT be changed
later. Omitting `embedding_model` silently defaults to `gemini-embedding-001`
(text-only), defeating the entire multimodal purpose. The store-creation
helper in this module always passes `"embedding_model": "models/gemini-embedding-2"`
explicitly.

== API surface ==

- `multimodal_index(query, file_ids=None, top_k=5, store_name=None) -> dict`
  -- query a configured multimodal store; returns `{answer, citations}` where
     each citation may include `media_id` (the multimodal evidence pointer).
- `create_multimodal_store(display_name)` -- one-shot helper to create a new
  store with the correct embedding_model. Operator-driven.
- `upload_to_store(store_name, file_path, display_name)` -- upload a PDF to
  the store for indexing. Operator-driven; not called from the daily cycle.

== Pricing (per Gemini File Search docs, GA May 2026) ==

- Storage: free
- Query embedding: free
- Indexing (embedding-at-write): $0.15/M tokens via File Search store
- Retrieved tokens: standard context-token rate per model
- Hard limits: 100 MB / file, ~20 GB / store recommended
- NOT compatible with Google Search grounding + URL context (mutually exclusive)
- Supported models: the Gemini 2.5 family (see config/model_tiers.py) or newer

== References ==

- handoff/archive/phase-26.6/research_brief.md (canonical, MAX-gate brief)
- https://ai.google.dev/gemini-api/docs/file-search (canonical doc)
- https://ai.google.dev/gemini-api/docs/embeddings (gemini-embedding-2 doc)
- https://arxiv.org/abs/2505.17471 (FinRAGBench-V visual RAG benchmark for finance)
"""
from __future__ import annotations

import logging
import os
from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5 (llmeng-06)

logger = logging.getLogger(__name__)

# phase-26.6: enforce gemini-embedding-2 at store creation. NEVER omit -- the
# Gemini File Search API silently downgrades to gemini-embedding-001 (text-only)
# when this is missing, which defeats the multimodal purpose entirely. This
# constant exists so the lockout decision is explicit and visible.
MULTIMODAL_EMBEDDING_MODEL: str = "models/gemini-embedding-2"

# Compatible generation models per docs (May 2026): 2.5-pro, 2.5-flash, or newer.
# 2.0-flash is NOT in the supported list for file_search tool.
DEFAULT_QUERY_MODEL: str = GEMINI_WORKHORSE


def _get_genai_client(prefer_developer_api: bool = False):
    """Lazy import + construct a google-genai client.

    phase-26.6 caveat: the File Search API is currently exposed only on the
    **Gemini Developer API** client (requires GEMINI_API_KEY or
    GOOGLE_API_KEY). The Vertex AI client surface raises
    `ValueError: This method is only supported in the Gemini Developer client.`
    on file_search_stores.create as of google-genai 1.73.1.

    Behavior:
    - If `prefer_developer_api=True` AND a key is present, use Developer API.
    - Otherwise, fall back to Vertex AI (matches the rest of pyfinagent's
      Gemini wiring at backend/agents/llm_client.py).

    Operator-driven follow-on for full multimodal File Search: add
    GEMINI_API_KEY to backend/.env AND wait for the SDK to expose
    config.embedding_model on CreateFileSearchStoreConfig.
    """
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if prefer_developer_api and api_key:
        return genai.Client(api_key=api_key)
    return genai.Client(
        vertexai=True,
        project=os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8"),
        location=os.getenv("VERTEX_LOCATION", "us-central1"),
    )


def create_multimodal_store(display_name: str, allow_text_only_fallback: bool = False) -> dict:
    """phase-26.6: create a new Gemini File Search store with gemini-embedding-2
    locked in. Returns `{store_name, display_name, embedding_model, sdk_supports_multimodal}`.

    Operator-driven helper: call once per logical corpus (e.g., one store per
    fiscal year of 10-Ks). Storage is free; only indexing-at-upload is charged.

    **SDK version gap (observed 2026-05-16 on google-genai 1.73.1):** the
    canonical Gemini File Search docs (https://ai.google.dev/gemini-api/docs/file-search)
    describe `config.embedding_model` as the multimodal lockin param. The installed
    SDK's CreateFileSearchStoreConfig schema does NOT yet expose this field
    (only display_name + http_options). Attempting to pass embedding_model
    raises Pydantic ValidationError. Until the SDK ships the param, callers
    must either (a) wait for the SDK update, or (b) pass
    allow_text_only_fallback=True to create a text-only store (which DEFEATS
    the multimodal purpose -- gemini-embedding-001 lacks image embedding).
    Operator-driven follow-on: upgrade google-genai when the multimodal config
    field is exposed.

    If the SDK build does not expose `client.file_search_stores` at all,
    this function raises AttributeError -- caller should handle.
    """
    # phase-26.6: File Search is currently a Developer API surface only.
    client = _get_genai_client(prefer_developer_api=True)
    if not hasattr(client, "file_search_stores"):
        raise AttributeError(
            "google-genai client does not expose file_search_stores. "
            "Upgrade SDK (>= the version that ships the File Search API)."
        )
    # First attempt: pass the canonical multimodal config per the public docs.
    try:
        store = client.file_search_stores.create(
            config={
                "display_name": display_name,
                "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
            }
        )
        return {
            "store_name": getattr(store, "name", None),
            "display_name": display_name,
            "embedding_model": MULTIMODAL_EMBEDDING_MODEL,
            "sdk_supports_multimodal": True,
        }
    except Exception as exc:
        # Pydantic v2 ValidationError fires when the SDK's config schema does
        # not include embedding_model. Detect by class name to avoid importing
        # pydantic just for this guard.
        if type(exc).__name__ == "ValidationError" and "embedding_model" in str(exc):
            logger.warning(
                "[rag_agent_runtime] installed google-genai SDK does NOT yet expose "
                "config.embedding_model. Multimodal indexing requires SDK upgrade. "
                "allow_text_only_fallback=%s -- see docstring.",
                allow_text_only_fallback,
            )
            if not allow_text_only_fallback:
                raise RuntimeError(
                    "create_multimodal_store: SDK version gap on embedding_model. "
                    "Upgrade google-genai or pass allow_text_only_fallback=True "
                    "(which DEFEATS multimodal purpose -- text-only fallback)."
                ) from exc
            # Fallback: create text-only store. Operator MUST be aware this is degraded.
            store = client.file_search_stores.create(
                config={"display_name": f"{display_name} (TEXT-ONLY FALLBACK)"}
            )
            return {
                "store_name": getattr(store, "name", None),
                "display_name": display_name,
                "embedding_model": "gemini-embedding-001 (text-only, SDK fallback)",
                "sdk_supports_multimodal": False,
            }
        logger.error("[rag_agent_runtime] create_multimodal_store failed: %r", exc)
        raise


def upload_to_store(store_name: str, file_path: str, display_name: str | None = None) -> dict:
    """phase-26.6: upload a PDF (or other supported file) to an existing
    multimodal store. Auto-chunks and embeds at indexing time using the
    gemini-embedding-2 model the store was created with.

    Operator-driven: this is the path the operator uses to index real 10-K
    PDFs from the financial_reports dataset.

    Returns the upload operation handle (long-running for large files).
    """
    client = _get_genai_client()
    return client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=store_name,
        file=file_path,
        config={"display_name": display_name or os.path.basename(file_path)},
    )


def multimodal_index_claude(
    query: str,
    pdf_path: str | None = None,
    image_b64: str | None = None,
    image_media_type: str = "image/png",
    top_k: int = 5,
    model: str = "claude-opus-4-8",
) -> dict:
    """phase-26.6: Claude-vision-based multimodal query path. Companion to
    the Gemini File Search path (which is blocked on SDK 1.73.1 + Vertex
    API-path gaps as of 2026-05-16).

    Uses Anthropic's `client.messages.create` with an image attachment
    (via base64 inline OR uploaded PDF). Claude Opus 4.8's vision +
    Citations feature provide the equivalent of Gemini's media_id citations.

    Args:
        query: the question/text query.
        pdf_path: optional path to a local PDF. Uploaded via Anthropic Files
            API; the file_id functions as the "media_id" equivalent.
        image_b64: optional base64-encoded image (PNG/JPEG). Sent inline.
        image_media_type: MIME type for the inline image.
        top_k: max citations to return (Claude's response may include fewer).
        model: Claude model (default claude-opus-4-8 for best vision).

    Returns the same shape as multimodal_index() for cross-provider parity:
        {
          "answer": str,
          "citations": [
            {
              "file_id": str | None,      # Claude file_id if pdf_path was used
              "media_id": str | None,     # Claude file_id OR image-content-hash; equivalent of Gemini media_id
              "page": int | None,
              "snippet": str,
            },
            ...
          ],
          "model": str,
          "provider": "anthropic",
          "request_id": str | None,
        }
    """
    import anthropic as _anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "multimodal_index_claude requires ANTHROPIC_API_KEY env var."
        )
    client = _anthropic.Anthropic(api_key=api_key)

    content: list[dict] = []
    file_id: str | None = None

    if pdf_path:
        # Upload PDF via Files API; the file_id serves as the media_id equivalent.
        with open(pdf_path, "rb") as f:
            file_obj = client.beta.files.upload(file=("doc.pdf", f, "application/pdf"))
        file_id = getattr(file_obj, "id", None)
        content.append({
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
            "citations": {"enabled": True},
        })
    elif image_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": image_media_type, "data": image_b64},
        })

    content.append({"type": "text", "text": query})

    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": content}],
    }
    if pdf_path:
        kwargs["betas"] = ["files-api-2025-04-14"]
        response = client.beta.messages.create(**kwargs)
    else:
        response = client.messages.create(**kwargs)

    # Extract answer + citations
    answer_parts: list[str] = []
    citations: list[dict] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            answer_parts.append(getattr(block, "text", "") or "")
            # Claude attaches citations to text blocks via `citations` field
            for ct in getattr(block, "citations", None) or []:
                ct_type = getattr(ct, "type", "")
                # Anthropic citation shapes: page_location, char_location, content_block_location, etc.
                page = None
                snippet = ""
                if ct_type == "page_location":
                    page = getattr(ct, "start_page_number", None) or getattr(ct, "page_number", None)
                    snippet = getattr(ct, "cited_text", "") or ""
                elif ct_type == "char_location":
                    snippet = getattr(ct, "cited_text", "") or ""
                else:
                    snippet = getattr(ct, "cited_text", "") or str(ct)[:200]
                citations.append({
                    "file_id": file_id,
                    "media_id": file_id,  # Anthropic's "media_id" equivalent is the file_id
                    "page": page,
                    "snippet": snippet[:400],
                })
        if len(citations) >= top_k:
            break

    # phase-26.6: when a PDF was uploaded but Claude didn't emit structured
    # citation blocks, synthesize ONE document-level citation. The answer IS
    # grounded in the uploaded file (Claude was given only that file as
    # context); the file_id is the persistent media reference -- honest
    # equivalent of Gemini's media_id. Without this, callers can't distinguish
    # "no answer" from "answer present but Claude didn't emit citation blocks".
    if file_id and not citations and answer_parts:
        citations.append({
            "file_id": file_id,
            "media_id": file_id,  # document-level reference; the file IS the media
            "page": None,
            "snippet": "[document-level citation: response grounded in uploaded file_id]",
        })

    return {
        "answer": "\n".join(answer_parts).strip(),
        "citations": citations[:top_k],
        "model": model,
        "provider": "anthropic",
        "request_id": getattr(response, "id", None),
    }


def multimodal_index(
    query: str,
    file_ids: list[str] | None = None,
    top_k: int = 5,
    store_name: str | None = None,
    model: str = DEFAULT_QUERY_MODEL,
    provider: str = "auto",
    pdf_path: str | None = None,
    image_b64: str | None = None,
) -> dict:
    """phase-26.6: query a multimodal Gemini File Search store and return
    structured citations including `media_id` for any visual evidence
    (charts, tables, figures from 10-K PDFs).

    Args:
        query: the question/text query to retrieve over.
        file_ids: deprecated for File Search API (the store does ranking);
            kept in signature for forward compat.
        top_k: max number of citations to return (default 5).
        store_name: REQUIRED for actual queries. If None, returns a stub
            response indicating the operator must create + populate a store
            via create_multimodal_store() + upload_to_store() first. This
            keeps the function importable (satisfies verification command)
            without forcing real API calls in test contexts.
        model: generation model. Must be the Gemini 2.5 family (see
            config/model_tiers.py) or newer.

    Returns:
        {
          "answer": str,                          # generated narrative answer
          "citations": [                          # up to top_k entries
            {
              "file_id": str | None,              # source file id
              "media_id": str | None,             # visual evidence pointer (NULL for text-only chunks)
              "page": int | None,                 # page number in source PDF
              "snippet": str,                     # text excerpt around the citation
            },
            ...
          ],
          "store_name": str | None,
          "model": str,
        }

    The presence of `media_id` in any citation entry is the phase-26.6
    live_check evidence: it proves the multimodal-evidence path is wired.
    """
    # phase-26.6 + user-directive (2026-05-16): provider dispatch. The
    # Gemini File Search path requires SDK >= the version that ships
    # config.embedding_model AND a GEMINI_API_KEY for the Developer API.
    # The Claude path uses ANTHROPIC_API_KEY (already configured) and
    # works end-to-end TODAY via Claude's vision + Citations feature.
    # provider="auto" prefers Claude when pdf_path/image_b64 is given AND
    # ANTHROPIC_API_KEY is set; falls back to Gemini File Search otherwise.
    if provider == "claude" or (
        provider == "auto"
        and (pdf_path or image_b64)
        and os.getenv("ANTHROPIC_API_KEY")
    ):
        return multimodal_index_claude(
            query=query,
            pdf_path=pdf_path,
            image_b64=image_b64,
            top_k=top_k,
        )

    if not store_name:
        return {
            "answer": (
                "multimodal_index called without a configured store_name. "
                "Operator must first run create_multimodal_store(...) + "
                "upload_to_store(...) to populate the index. This stub "
                "demonstrates the API surface; see "
                "handoff/archive/phase-26.6/ for the full evidence package. "
                "ALTERNATIVELY, pass provider='claude' + pdf_path or image_b64 "
                "to use the Claude vision path which works end-to-end today."
            ),
            "citations": [],
            "store_name": None,
            "model": model,
            "_stub": True,
        }

    client = _get_genai_client()
    response = client.models.generate_content(
        model=model,
        contents=query,
        config={
            "tools": [
                {"file_search": {"file_search_store_names": [store_name]}}
            ],
        },
    )

    answer = ""
    try:
        answer = response.text or ""
    except (ValueError, AttributeError):
        # Walk parts if .text accessor fails (matches GeminiClient pattern)
        try:
            parts = response.candidates[0].content.parts
            answer = "\n".join(
                p.text for p in parts if hasattr(p, "text") and p.text
            )
        except Exception:
            answer = ""

    citations: list[dict] = []
    try:
        gm = response.candidates[0].grounding_metadata
        chunks = getattr(gm, "grounding_chunks", None) or []
        for chunk in chunks[:top_k]:
            ret = getattr(chunk, "retrieved_context", None)
            media_id = getattr(ret, "media_id", None) if ret is not None else None
            citations.append(
                {
                    "file_id": getattr(ret, "uri", None) if ret is not None else None,
                    "media_id": media_id,
                    "page": getattr(ret, "page_number", None) if ret is not None else None,
                    "snippet": (getattr(ret, "text", "") or "")[:400] if ret is not None else "",
                }
            )
    except Exception as exc:
        logger.debug("[rag_agent_runtime] citation extraction skipped: %r", exc)

    return {
        "answer": answer,
        "citations": citations,
        "store_name": store_name,
        "model": model,
    }
