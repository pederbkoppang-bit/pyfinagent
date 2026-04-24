"""
Backend authentication: decrypt NextAuth.js v5 JWE session tokens.

NextAuth v5 uses JWE (JSON Web Encryption) with:
  - HKDF(AUTH_SECRET, info=b"Auth.js Generated Encryption Key", salt=b"", length=32)
  - A256CBC-HS512 / dir
"""

import hashlib
import hmac
import json
import logging
import os
import struct
import time
from typing import Optional

from fastapi import HTTPException, Request

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """HKDF-Expand (RFC 5869) with SHA-256."""
    hash_len = 32  # SHA-256 output
    n = (length + hash_len - 1) // hash_len
    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + struct.pack("B", i), hashlib.sha256).digest()
        okm += t
    return okm[:length]


def _hkdf_derive_key(secret: str, salt: str = "") -> bytes:
    """
    Derive the 64-byte encryption key from AUTH_SECRET using HKDF.

    NextAuth v5 (5.0.0-beta.x) derives the key with:
      - ikm  = AUTH_SECRET (utf-8)
      - salt = cookie name, e.g. "authjs.session-token" (utf-8)
      - info = "Auth.js Generated Encryption Key ({salt})"
      - length = 64 (A256CBC-HS512: 32 bytes HMAC + 32 bytes AES)

    NextAuth v4 used salt="" and info="Auth.js Generated Encryption
    Key". When `salt` is the empty string, we still follow the
    v5-style info template with an empty parenthetical so v5 tokens
    minted with a fallback empty salt still decrypt correctly; a
    caller wanting v4-bit-compatible behaviour passes salt="".
    """
    ikm = secret.encode("utf-8")
    salt_bytes = salt.encode("utf-8") if salt else b"\x00" * 32
    prk = hmac.new(salt_bytes, ikm, hashlib.sha256).digest()
    info = f"Auth.js Generated Encryption Key ({salt})".encode("utf-8")
    return _hkdf_expand(prk, info, 64)


def _hkdf_derive_key_v4(secret: str) -> bytes:
    """Legacy v4 HKDF (no salt, no salt in info). Kept for a narrow
    compatibility fallback when a v4-format token is still in flight."""
    ikm = secret.encode("utf-8")
    salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    info = b"Auth.js Generated Encryption Key"
    return _hkdf_expand(prk, info, 64)


def _base64url_decode(s: str) -> bytes:
    """Decode base64url without padding."""
    import base64
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def decrypt_jwe(token: str, secret: str, salt: str = "authjs.session-token") -> dict:
    """
    Decrypt a NextAuth v5 JWE token (compact serialization).
    Format: header.encryptedKey.iv.ciphertext.tag
    Algorithm: dir + A256CBC-HS512

    `salt` is the NextAuth cookie name (without `__Secure-` prefix).
    NextAuth v5 derives the key with HKDF(salt=cookie_name,
    info=f"Auth.js Generated Encryption Key ({cookie_name})").
    """
    parts = token.split(".")
    if len(parts) != 5:
        raise ValueError("Invalid JWE format")

    header_b64, enc_key_b64, iv_b64, ciphertext_b64, tag_b64 = parts

    header = json.loads(_base64url_decode(header_b64))
    if header.get("alg") != "dir" or header.get("enc") != "A256CBC-HS512":
        raise ValueError(f"Unsupported JWE algorithm: {header.get('alg')}/{header.get('enc')}")

    iv = _base64url_decode(iv_b64)
    ciphertext = _base64url_decode(ciphertext_b64)
    tag = _base64url_decode(tag_b64)
    aad = header_b64.encode("ascii")
    al = struct.pack(">Q", len(aad) * 8)
    mac_input = aad + iv + ciphertext + al

    # A256CBC-HS512: key is MAC_KEY (first 32) || ENC_KEY (last 32),
    # HMAC is SHA-512 truncated to 256 bits (RFC 7518 section 5.2.5).
    # Try v5 salt first; fall back to v4 for legacy tokens in flight.
    key = None
    enc_key = None
    for key_fn in (lambda: _hkdf_derive_key(secret, salt),
                    lambda: _hkdf_derive_key_v4(secret)):
        candidate = key_fn()
        mac_key_c = candidate[:32]
        computed_tag = hmac.new(mac_key_c, mac_input, hashlib.sha512).digest()[:32]
        if hmac.compare_digest(tag, computed_tag):
            key = candidate
            enc_key = candidate[32:64]
            break
    if key is None:
        raise ValueError("JWE tag verification failed")

    # Decrypt AES-256-CBC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()

    # Remove PKCS7 padding
    pad_len = plaintext[-1]
    plaintext = plaintext[:-pad_len]

    return json.loads(plaintext)


async def get_current_user(request: Request) -> Optional[dict]:
    """
    Extract and validate user from Authorization header.
    Returns the JWT payload dict or None if no auth required.
    Raises HTTPException(401) if token is invalid.
    """
    settings = get_settings()

    # Localhost-only UAT bypass. Fires ONLY when:
    #   (a) DEV_LOCALHOST_BYPASS=1 is set in env, AND
    #   (b) the request.client.host is 127.0.0.1 / ::1 / localhost.
    # Both conditions must hold; the env flag alone is insufficient.
    # Intended for full-app UAT drills (phase-16) where CLI-level
    # requests need to exercise authenticated endpoints. NEVER set
    # this flag in production or on a publicly-reachable host.
    if os.getenv("DEV_LOCALHOST_BYPASS") == "1":
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost"):
            return {"email": "dev@localhost", "localhost_bypass": True}

    # Auth-secret-missing handling. Previously this returned None silently,
    # which created a latent "any request passes" bypass in any deployment
    # that accidentally shipped without AUTH_SECRET set (see phase-4.6.4
    # follow-up). Now the only way to skip auth is an explicit opt-in
    # env var DEV_DISABLE_AUTH=1. Without it, a missing AUTH_SECRET is a
    # hard failure so operators notice at the first unauthenticated call.
    if not settings.auth_secret:
        if os.getenv("DEV_DISABLE_AUTH") == "1":
            return None
        raise HTTPException(
            status_code=401,
            detail="Authentication required (AUTH_SECRET not configured; set DEV_DISABLE_AUTH=1 only in local dev)",
        )

    # Extract token from either the Authorization header OR the
    # NextAuth session cookie. The cookie is HttpOnly, so the frontend
    # cannot read it to attach as a Bearer header; it sends the
    # "session-active" sentinel in that case, and we read the cookie
    # here (cross-origin cookies work because fetch uses
    # credentials:"include" + CORS allow-credentials=true).
    auth_header = request.headers.get("Authorization") or ""
    header_token = auth_header[7:] if auth_header.startswith("Bearer ") else ""

    # Candidates: (cookie_name_as_salt, token_value). NextAuth v5 always
    # uses `authjs.session-token` as the HKDF salt even when the cookie
    # is set with the `__Secure-` prefix in production.
    candidates: list[tuple[str, str]] = []
    if header_token and header_token != "session-active":
        candidates.append(("authjs.session-token", header_token))
    secure_cookie = request.cookies.get("__Secure-authjs.session-token")
    dev_cookie = request.cookies.get("authjs.session-token")
    if secure_cookie:
        candidates.append(("authjs.session-token", secure_cookie))
    if dev_cookie:
        candidates.append(("authjs.session-token", dev_cookie))

    if not candidates:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload: Optional[dict] = None
    last_error: Optional[Exception] = None
    for salt, tok in candidates:
        try:
            payload = decrypt_jwe(tok, settings.auth_secret, salt=salt)
            break
        except Exception as e:
            last_error = e
            continue
    if payload is None:
        logger.warning("auth_failed: invalid token (%s)", last_error)
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check email whitelist
    email = payload.get("email", "")
    if settings.allowed_emails:
        allowed = [e.strip().lower() for e in settings.allowed_emails.split(",") if e.strip()]
        if allowed and email.lower() not in allowed:
            logger.warning(f"auth_denied: email={email}")
            raise HTTPException(status_code=401, detail="Authentication required")

    # Check token expiry
    exp = payload.get("exp")
    if exp and time.time() > exp:
        raise HTTPException(status_code=401, detail="Authentication required")

    return payload
