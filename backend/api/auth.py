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


def _hkdf_derive_key(secret: str) -> bytes:
    """
    Derive the 64-byte encryption key from AUTH_SECRET using HKDF.
    NextAuth uses HKDF with:
      - ikm = AUTH_SECRET (utf-8)
      - salt = "" (empty)
      - info = "Auth.js Generated Encryption Key"
      - length = 64 (for A256CBC-HS512: 32 bytes HMAC + 32 bytes AES)
    """
    ikm = secret.encode("utf-8")
    # HKDF-Extract with empty salt
    salt = b"\x00" * 32  # SHA-256 block size
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    # HKDF-Expand
    info = b"Auth.js Generated Encryption Key"
    return _hkdf_expand(prk, info, 64)


def _base64url_decode(s: str) -> bytes:
    """Decode base64url without padding."""
    import base64
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def decrypt_jwe(token: str, secret: str) -> dict:
    """
    Decrypt a NextAuth v5 JWE token (compact serialization).
    Format: header.encryptedKey.iv.ciphertext.tag
    Algorithm: dir + A256CBC-HS512
    """
    parts = token.split(".")
    if len(parts) != 5:
        raise ValueError("Invalid JWE format")

    header_b64, enc_key_b64, iv_b64, ciphertext_b64, tag_b64 = parts

    header = json.loads(_base64url_decode(header_b64))
    if header.get("alg") != "dir" or header.get("enc") != "A256CBC-HS512":
        raise ValueError(f"Unsupported JWE algorithm: {header.get('alg')}/{header.get('enc')}")

    key = _hkdf_derive_key(secret)
    mac_key = key[:32]
    enc_key = key[32:64]

    iv = _base64url_decode(iv_b64)
    ciphertext = _base64url_decode(ciphertext_b64)
    tag = _base64url_decode(tag_b64)

    # Verify HMAC-SHA-256 tag (A256CBC-HS512 uses HMAC-SHA-256 truncated to 256 bits)
    aad = header_b64.encode("ascii")
    al = struct.pack(">Q", len(aad) * 8)
    mac_input = aad + iv + ciphertext + al
    computed_tag = hmac.new(mac_key, mac_input, hashlib.sha256).digest()

    if not hmac.compare_digest(tag, computed_tag):
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

    # Skip auth if no AUTH_SECRET configured (dev mode)
    if not settings.auth_secret:
        return None

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")

    token = auth_header[7:]  # Strip "Bearer "
    if not token or token == "session-active":
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        payload = decrypt_jwe(token, settings.auth_secret)
    except Exception:
        logger.warning("auth_failed: invalid token")
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
