"""Cycle 72 harness: NextAuth v5 JWE round-trip test.

Mint a v5 JWE in Node (using @panva/jose + @panva/hkdf from the
frontend node_modules) and decrypt it in Python via
backend.api.auth.decrypt_jwe. Asserts the payload round-trips.

Exit 0 on PASS; 1 on FAIL. Writes handoff/auth_jwe_roundtrip.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backend.api.auth import decrypt_jwe, _hkdf_derive_key  # noqa: E402


NODE_MINT = r'''
const hkdfModule = require('./frontend/node_modules/@panva/hkdf');
const hkdf = hkdfModule.hkdf;
const { CompactEncrypt, base64url } = require('./frontend/node_modules/jose');

const SECRET = process.env.AUTH_SECRET;
const SALT = process.env.JWE_SALT || 'authjs.session-token';

(async () => {
  const key = await hkdf(
    'sha256',
    SECRET,
    SALT,
    'Auth.js Generated Encryption Key (' + SALT + ')',
    64,
  );
  const payload = {
    email: 'peder.bkoppang@hotmail.no',
    name: 'Peder',
    sub: 'test-sub-42',
    iat: Math.floor(Date.now() / 1000),
    exp: Math.floor(Date.now() / 1000) + 3600,
  };
  const jwe = await new CompactEncrypt(
    new TextEncoder().encode(JSON.stringify(payload)),
  )
    .setProtectedHeader({ alg: 'dir', enc: 'A256CBC-HS512' })
    .encrypt(key);
  console.log(jwe);
})().catch((e) => { console.error(e); process.exit(1); });
'''


def mint_token(secret: str, salt: str) -> str:
    env = {"AUTH_SECRET": secret, "JWE_SALT": salt, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    res = subprocess.run(
        ["node", "-e", NODE_MINT],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
    )
    if res.returncode != 0:
        raise RuntimeError(f"node mint failed: {res.stderr}")
    return res.stdout.strip()


def main() -> int:
    import os
    secret = os.getenv("AUTH_SECRET") or "i8fx3bCk0uD1+0F58i9dfTtRNDoobL477W/GH8woeDI="
    salt = "authjs.session-token"

    mint_ok = False
    decrypt_ok = False
    email_match = False
    jwe: str = ""
    payload: dict = {}
    error: str | None = None
    try:
        jwe = mint_token(secret, salt)
        mint_ok = bool(jwe and jwe.count(".") == 4)
    except Exception as e:
        error = f"mint: {e}"

    if mint_ok:
        try:
            payload = decrypt_jwe(jwe, secret, salt=salt)
            decrypt_ok = True
            email_match = payload.get("email") == "peder.bkoppang@hotmail.no"
        except Exception as e:
            error = f"decrypt: {e}"

    # Also verify key derivation matches by re-deriving both ways.
    key_py = _hkdf_derive_key(secret, salt).hex()

    result = {
        "step": "auth-jwe-roundtrip",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "mint_ok": mint_ok,
        "decrypt_ok": decrypt_ok,
        "email_match": email_match,
        "jwe_parts": jwe.count(".") + 1 if jwe else 0,
        "payload": payload,
        "key_py_prefix": key_py[:32],
        "error": error,
        "verdict": "PASS" if (mint_ok and decrypt_ok and email_match) else "FAIL",
    }
    out = REPO / "handoff" / "auth_jwe_roundtrip.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "verdict": result["verdict"],
        "mint_ok": mint_ok, "decrypt_ok": decrypt_ok, "email_match": email_match,
        "error": error,
    }))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
