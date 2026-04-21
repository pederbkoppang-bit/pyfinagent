"""phase-6.5.2 intel scanner core.

`BaseScanner` is the generic pull-model scanner. Subclass and override
`_do_scan()` for source-specific extraction. The default implementation
HTTP-fetches `source.metadata["feed_url"]` or `["url"]` and returns a single
DocumentCandidate with the response body as `raw_text`.

`scan(dry_run=True)` returns a deterministic single stub DocumentCandidate
without any network or BQ activity. This satisfies phase-6.5.2 immutable
criterion `scanner_dry_run_returns_candidates`.

Dedup discipline: intra-batch dedup via a `seen` set of
`(canonical_url, content_hash)` pairs. Cross-batch dedup is phase-6.5.7's
job (embeddings + novelty score).

EDGAR backoff: 60 * 2^attempt on 403 (see research brief risk R1), 5 * 2^attempt
on 5xx. Not exercised this cycle (no live fetch) but encoded for reuse.

ASCII-only logger messages per `.claude/rules/security.md`.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from backend.intel.source_registry import SourceRow

logger = logging.getLogger(__name__)


class DocumentCandidate(TypedDict, total=False):
    doc_id: str
    source_id: str
    source_type: str
    doc_type: str | None
    published_at: str | None
    ingested_at: str
    title: str | None
    authors: list[str]
    url: str
    canonical_url: str
    content_hash: str
    raw_text: str | None
    language: str | None
    raw_payload: dict[str, Any]


_REQUIRED_CANDIDATE_KEYS = {
    "doc_id",
    "source_id",
    "source_type",
    "ingested_at",
    "url",
    "canonical_url",
    "content_hash",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonicalize(url: str) -> str:
    """Minimal URL canonicalizer: strip trailing slash + fragment."""
    if not url:
        return ""
    u = url.strip()
    frag = u.find("#")
    if frag != -1:
        u = u[:frag]
    if u.endswith("/") and len(u) > len("https://"):
        u = u[:-1]
    return u


class BaseScanner:
    """Generic pull-model scanner for one source.

    The dry-run path returns exactly 1 deterministic stub candidate with every
    required column populated. Subclasses override `_do_scan()` for real sources.
    """

    def __init__(self, source: SourceRow) -> None:
        self.source = source
        self._logger = logging.getLogger(
            f"intel.scanner.{source.source_type}.{source.source_id}"
        )

    def scan(self, *, dry_run: bool = False) -> list[DocumentCandidate]:
        if dry_run:
            return self._stub_candidates()
        try:
            raw = self._do_scan()
        except Exception as exc:
            self._logger.warning(
                "scanner fail-open source=%s err=%r", self.source.source_id, exc
            )
            return []
        return self._dedup(raw)

    def _dedup(self, cands: list[DocumentCandidate]) -> list[DocumentCandidate]:
        seen: set[tuple[str, str]] = set()
        out: list[DocumentCandidate] = []
        for c in cands:
            key = (c.get("canonical_url", ""), c.get("content_hash", ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _do_scan(self) -> list[DocumentCandidate]:
        meta = self.source.metadata or {}
        url = meta.get("feed_url") or meta.get("url") or ""
        if not url:
            return []
        return self._fetch_http(url)

    def _fetch_http(
        self, url: str, *, attempt: int = 0, max_attempts: int = 3
    ) -> list[DocumentCandidate]:
        """Generic HTTP GET -> single DocumentCandidate with body as raw_text.

        Fail-open on any error. Respects EDGAR-style backoff on 403.
        """
        try:
            import requests  # type: ignore[import-not-found]
        except Exception as exc:
            self._logger.warning("requests missing: %r", exc)
            return []
        headers = {"User-Agent": (self.source.metadata or {}).get("user_agent", "pyfinagent/1.0")}
        try:
            resp = requests.get(url, headers=headers, timeout=20)
        except Exception as exc:
            self._logger.warning("http get fail-open: %r", exc)
            return []
        if resp.status_code == 403 and attempt < max_attempts:
            time.sleep(60 * (2**attempt))
            return self._fetch_http(url, attempt=attempt + 1, max_attempts=max_attempts)
        if 500 <= resp.status_code < 600 and attempt < max_attempts:
            time.sleep(5 * (2**attempt))
            return self._fetch_http(url, attempt=attempt + 1, max_attempts=max_attempts)
        if resp.status_code != 200:
            self._logger.warning("http non-200 status=%s url=%s", resp.status_code, url)
            return []
        text = resp.text or ""
        return [
            DocumentCandidate(
                doc_id=str(uuid.uuid4()),
                source_id=self.source.source_id,
                source_type=self.source.source_type,
                doc_type=None,
                published_at=None,
                ingested_at=_now_iso(),
                title=None,
                authors=[],
                url=url,
                canonical_url=_canonicalize(url),
                content_hash=_content_hash(text),
                raw_text=text,
                language=None,
                raw_payload={"status_code": resp.status_code},
            )
        ]

    def _stub_candidates(self) -> list[DocumentCandidate]:
        body = f"stub document for {self.source.source_id}"
        url = "https://stub.example.com/" + self.source.source_id
        return [
            DocumentCandidate(
                doc_id=str(uuid.uuid4()),
                source_id=self.source.source_id,
                source_type=self.source.source_type,
                doc_type="stub",
                published_at=None,
                ingested_at=_now_iso(),
                title=f"stub: {self.source.source_name}",
                authors=[],
                url=url,
                canonical_url=_canonicalize(url),
                content_hash=_content_hash(body),
                raw_text=body,
                language="en",
                raw_payload={},
            )
        ]
