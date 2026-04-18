"""phase-3.7 step 3.7.7 capability tokens + PII filter for MCP input.

Two defensive layers wrapped around every `@mcp.tool` entry point:

1. **Capability tokens** (HMAC-SHA256, stdlib only). Agents receive a
   short-lived (<=30 min) signed token binding:

       (session_id, role, scopes, expires_at)

   Roles map to fixed scope sets; a researcher token cannot invoke
   `trading.write` even if the agent is prompt-injected.

2. **PII filter** on inbound args. Emails, phone numbers, JWT-ish
   tokens, and provider API keys (sk-ant-*, sk-*) are regex-scrubbed
   from the args dict before the tool body sees them; a WARNING is
   logged on every redaction.

Design decisions and evidence: see handoff/research_3.7.7.md.

Import patterns:

    from backend.agents.mcp_capabilities import (
        issue_token, verify_token,
        CapabilityError, TokenExpiredError, TokenInvalidError,
        ScopeViolationError,
        scrub_args, ROLE_SCOPES,
    )
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from typing import Any, Iterable

logger = logging.getLogger(__name__)


TOKEN_TTL_SECONDS = 1800  # 30 minutes (NIST SP 800-63B-4 guidance)

_DEFAULT_DEV_SECRET = "dev-only-mcp-cap-secret-CHANGE-IN-PROD"


def _secret() -> bytes:
    val = os.getenv("MCP_CAPABILITY_SECRET", _DEFAULT_DEV_SECRET)
    return val.encode("utf-8")


ROLE_SCOPES: dict[str, frozenset[str]] = {
    "researcher":   frozenset({"data.read", "signals.read", "backtest.read"}),
    "strategy":     frozenset({"data.read", "signals.read",
                               "signals.write", "backtest.read"}),
    "risk":         frozenset({"data.read", "signals.read",
                               "risk.read", "risk.write"}),
    "evaluator":    frozenset({"data.read", "signals.read",
                               "backtest.read", "risk.read"}),
    "orchestrator": frozenset({"data.read", "signals.read",
                               "signals.write", "backtest.read",
                               "risk.read", "risk.write"}),
    "paper_trader": frozenset({"data.read", "signals.read",
                               "trading.write", "risk.read"}),
}


class CapabilityError(PermissionError):
    """Base class for all capability-token failures."""


class TokenExpiredError(CapabilityError):
    """Token's expires_at has passed."""


class TokenInvalidError(CapabilityError):
    """Token signature did not verify, or token is malformed."""


class ScopeViolationError(CapabilityError):
    """Token is valid but lacks the required scope for this tool."""


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64u_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def issue_token(
    session_id: str,
    role: str,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
    *,
    now: float | None = None,
) -> str:
    """Mint an HMAC-signed capability token for a session.

    Format: <b64u(payload_json)>.<b64u(hmac_sha256)>

    Token is opaque to callers; only `verify_token` decodes it.
    """
    if role not in ROLE_SCOPES:
        raise ValueError(f"unknown role: {role!r}")
    iat = float(now) if now is not None else time.time()
    payload = {
        "sid": session_id,
        "role": role,
        "scopes": sorted(ROLE_SCOPES[role]),
        "iat": iat,
        "exp": iat + ttl_seconds,
        "nonce": secrets.token_hex(8),
    }
    payload_b = json.dumps(payload, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    sig = hmac.new(_secret(), payload_b, hashlib.sha256).digest()
    return f"{_b64u(payload_b)}.{_b64u(sig)}"


def verify_token(
    token: str | None,
    required_scope: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify signature, TTL, and scope. Return the decoded payload.

    Raises the appropriate CapabilityError subclass on failure.
    """
    if not token:
        raise CapabilityError("missing capability token")
    try:
        payload_b64, sig_b64 = token.rsplit(".", 1)
        payload_b = _b64u_decode(payload_b64)
        sig = _b64u_decode(sig_b64)
    except (ValueError, Exception) as e:
        raise TokenInvalidError(f"malformed token: {e}") from None

    expected = hmac.new(_secret(), payload_b, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise TokenInvalidError("signature mismatch")

    try:
        payload = json.loads(payload_b)
    except json.JSONDecodeError as e:
        raise TokenInvalidError(f"payload not JSON: {e}") from None

    t = float(now) if now is not None else time.time()
    if t >= float(payload.get("exp", 0)):
        raise TokenExpiredError(
            f"token expired at {payload.get('exp')}, now {t}")

    scopes = set(payload.get("scopes") or [])
    if required_scope not in scopes:
        raise ScopeViolationError(
            f"role={payload.get('role')!r} lacks scope "
            f"{required_scope!r}; has {sorted(scopes)}"
        )
    return payload


# ---- PII filter ------------------------------------------------------

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email",    re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone",    re.compile(
        r"(?:\+\d[\d\s\-.()]{8,}\d"
        r"|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}"
        r"|\b\d{3}[-.]\d{3}[-.]\d{4}\b)")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai_key",    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("jwt",      re.compile(
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    ("ssn",      re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

_REDACTED = "[REDACTED]"


def _scrub_string(s: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    out = s
    for kind, pat in _PII_PATTERNS:
        if pat.search(out):
            out = pat.sub(_REDACTED, out)
            hits.append(kind)
    return out, hits


def scrub_args(args: Any) -> tuple[Any, list[str]]:
    """Deep-scrub PII from a dict/list/scalar arg tree.

    Returns (scrubbed_copy, list_of_redaction_kinds_fired). Callers
    should log a WARNING when the returned list is non-empty.

    Clean args pass through byte-identical (same nested object shape,
    same strings). The returned tree is a shallow copy at any node
    where a redaction occurred.
    """
    all_hits: list[str] = []

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            out, hits = _scrub_string(v)
            all_hits.extend(hits)
            return out
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, tuple):
            return tuple(walk(x) for x in v)
        return v

    scrubbed = walk(args)
    if all_hits:
        logger.warning(
            "mcp_capabilities.scrub_args redacted PII kinds=%s",
            sorted(set(all_hits)),
        )
    return scrubbed, all_hits


def has_scope(role: str, scope: str) -> bool:
    return scope in ROLE_SCOPES.get(role, frozenset())


def enforce(required_scope: str):
    """Decorator: verify capability token + scrub PII before tool body.

    The wrapped tool must accept `_cap_token: str | None = None` as a
    kwarg; this is the session's signed capability token.
    """
    def decorator(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = kwargs.pop("_cap_token", None)
            verify_token(token, required_scope)
            scrubbed_kwargs, _ = scrub_args(kwargs)
            scrubbed_args, _ = scrub_args(args)
            return fn(*scrubbed_args, **scrubbed_kwargs)
        return wrapper
    return decorator


__all__ = [
    "ROLE_SCOPES",
    "TOKEN_TTL_SECONDS",
    "CapabilityError",
    "TokenExpiredError",
    "TokenInvalidError",
    "ScopeViolationError",
    "issue_token",
    "verify_token",
    "scrub_args",
    "has_scope",
    "enforce",
]
