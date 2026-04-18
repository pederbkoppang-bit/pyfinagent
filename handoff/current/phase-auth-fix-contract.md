# Contract -- Cycle 72 / ad-hoc auth-loop fix

**Step**: diagnose + fix "Sign in with Google kicks user out" loop.

## Hypothesis (verified)

Backend `decrypt_jwe` uses the NextAuth v4 HKDF format:
  info = "Auth.js Generated Encryption Key", salt = empty

NextAuth v5 (installed: 5.0.0-beta.30) uses:
  info = "Auth.js Generated Encryption Key ({cookie_name})"
  salt = cookie_name (e.g. "authjs.session-token")

Empirically verified by deriving both keys against the same AUTH_SECRET
+ comparing with `backend.api.auth._hkdf_derive_key`. Keys diverge:

  v4 (backend today):  a3dbffa562da8cc0c714a62a1650ceb3a5ff4d0211f55c4b6...
  v5 (NextAuth live):  02033467325c219a9b7843093f45984f70409c6a7f1b9620a6...

Result: **every** JWE decrypt at the backend fails -> 401 ->
frontend api.ts line 111-114 forces `window.location.href="/login"`
-> infinite kick-back loop after Google sign-in completes.

## Scope

Files modified:

1. **MODIFY** `backend/api/auth.py`:
   - `_hkdf_derive_key(secret, salt)` -- take cookie-name salt;
     do HKDF-Extract with salt bytes, HKDF-Expand with info =
     `"Auth.js Generated Encryption Key ({salt})"`.
   - `decrypt_jwe(token, secret, salt)` -- propagate salt.
   - `get_current_user`: when reading cookie token, try both salts
     in order: `__Secure-authjs.session-token` (prod / https) then
     `authjs.session-token` (dev / http). Use the matching salt for
     each key derivation.
   - Keep v4-style as a documented fallback for legacy tokens, but
     default to v5 format (current production path).

2. **NEW** `scripts/harness/auth_jwe_roundtrip.py`:
   - Calls NextAuth's `@auth/core` JWT encode via `node -e` to
     produce a fresh v5 JWE.
   - Decrypts it from Python using the updated `decrypt_jwe`.
   - Asserts the decoded payload contains the expected email and exp.
   - Exit 0 on success, 1 on failure.

## Immutable success criteria (self-imposed for this fix cycle)

1. `hkdf_key_matches_nextauth_v5`: Python `_hkdf_derive_key(secret,
   "authjs.session-token")` produces the same 64-byte key as
   `@panva/hkdf` in Node with the matching v5 args.
2. `jwe_decrypt_round_trip`: a JWE minted by NextAuth v5 can be
   decrypted in Python and the payload matches.
3. `backend_401_no_longer_fires_on_valid_cookie`: after the fix + a
   real browser sign-in, `GET /api/paper-trading/status` with the
   forwarded cookie returns 200 (not 401).
4. `old_v4_format_still_handles_legacy`: if a legacy v4 token is
   presented, it still decrypts (document in code; don't bake in).

## Verification

    python scripts/harness/auth_jwe_roundtrip.py

Must exit 0.

## References

- NextAuth v5 packages/core/src/jwt.ts `getDerivedEncryptionKey`
  (info = `Auth.js Generated Encryption Key (${salt})`, salt =
  cookie name).
- RFC 5869 HKDF.
- https://authjs.dev/getting-started/deployment (secret derivation)
- frontend/src/lib/auth.config.ts (session strategy jwt).
- backend/api/auth.py:109-158 (broken path).
