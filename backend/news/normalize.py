"""phase-6.2 canonical URL + body-hash normalization (stdlib only).

`canonical_url` strips tracking params (utm_*, fbclid, gclid, ref,
source, session_id), lowercases scheme + host, removes trailing slash
from path, and sorts remaining query params for deterministic output.

`body_hash` returns `sha256(normalize_text(body).encode("utf-8"))` hex.
Normalization collapses whitespace, strips HTML tags (simple regex),
and lowercases, so cosmetic changes do not produce a different hash.
Exact-match only; near-dedup (MinHash/LSH) is an explicit non-goal.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_referrer", "utm_social",
        "utm_brand", "utm_creative",
        "fbclid", "gclid", "msclkid", "dclid",
        "ref", "ref_src", "ref_url", "source",
        "session_id", "sid", "s_cid", "s_kwcid",
        "mc_cid", "mc_eid",
        "_hsenc", "_hsmi",
    }
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def canonical_url(url: str) -> str:
    """Return a dedup-friendly canonical form of `url`.

    - scheme + host lowercased
    - tracking params removed (`utm_*`, `fbclid`, `gclid`, `ref`,
      `source`, `session_id`, etc.)
    - trailing slash stripped from the path (unless path is `/`)
    - remaining query params sorted
    - fragment (#...) dropped
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "http").lower()
    host = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ]
    kept.sort()
    query = urlencode(kept, doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def normalize_text(text: str) -> str:
    """Collapse whitespace, strip HTML tags, lowercase."""
    if not text:
        return ""
    stripped = _HTML_TAG_RE.sub(" ", text)
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    return collapsed.lower()


def body_hash(text: str) -> str:
    """Return sha256 hex of the normalized body text."""
    norm = normalize_text(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()
