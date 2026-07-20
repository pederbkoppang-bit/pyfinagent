"""Authenticated TestClient helper (phase-75.1).

Step 75.1 pruned `_PUBLIC_PATHS` down to the minimal read-only set, so
suites that exercise `/api/sovereign`, `/api/signals`, `/api/observability`,
`/api/cost-budget` or `/api/harness/monthly-approval` must now present
credentials like any other consumer.

`DEV_LOCALHOST_BYPASS` does NOT cover them: Starlette's TestClient reports
`request.client.host == "testclient"`, so the localhost rail in
`backend.api.auth.get_current_user` never fires for in-process suites.

`authed_test_client(app)` returns a TestClient that sends a real minted
NextAuth-shaped JWE on every request, so the middleware exercises the true
decrypt + allowlist path rather than a bypass. When `AUTH_SECRET` is absent
(clean CI checkout) it falls back to the documented `DEV_DISABLE_AUTH=1`
escape hatch, which only works when there is no secret to verify against.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
from typing import Any

from fastapi.testclient import TestClient

__all__ = ["mint_session_token", "authed_test_client"]


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def mint_session_token(email: str | None = None, ttl_seconds: int = 3600) -> str:
    """Mint a NextAuth v5 JWE (dir / A256CBC-HS512) signed with AUTH_SECRET.

    The email defaults to the first entry of ALLOWED_EMAILS when that
    whitelist is populated, so the token also clears the allowlist leg.
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    from backend.api.auth import _hkdf_derive_key
    from backend.config.settings import get_settings

    settings = get_settings()
    secret = settings.auth_secret.get_secret_value()
    if not secret:
        raise RuntimeError("AUTH_SECRET is empty -- cannot mint a session token")

    if email is None:
        allowed = [e.strip() for e in settings.allowed_emails.split(",") if e.strip()]
        email = allowed[0] if allowed else "pytest@localhost"

    key = _hkdf_derive_key(secret, "authjs.session-token")
    mac_key, enc_key = key[:32], key[32:64]

    header_b64 = _b64u(json.dumps({"alg": "dir", "enc": "A256CBC-HS512"}).encode())
    plaintext = json.dumps({"email": email, "exp": int(time.time()) + ttl_seconds}).encode()
    pad = 16 - (len(plaintext) % 16)
    plaintext += bytes([pad]) * pad

    iv = os.urandom(16)
    encryptor = Cipher(algorithms.AES(enc_key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    aad = header_b64.encode("ascii")
    al = struct.pack(">Q", len(aad) * 8)
    tag = hmac.new(mac_key, aad + iv + ciphertext + al, hashlib.sha512).digest()[:32]

    return f"{header_b64}..{_b64u(iv)}.{_b64u(ciphertext)}.{_b64u(tag)}"


def authed_test_client(app: Any, **kwargs: Any) -> TestClient:
    """TestClient that authenticates on every request."""
    try:
        token = mint_session_token()
    except Exception:
        # No AUTH_SECRET configured: the only supported way through the
        # middleware is the explicit no-secret dev opt-out.
        os.environ.setdefault("DEV_DISABLE_AUTH", "1")
        return TestClient(app, **kwargs)

    headers = dict(kwargs.pop("headers", None) or {})
    headers.setdefault("Authorization", f"Bearer {token}")
    return TestClient(app, headers=headers, **kwargs)
