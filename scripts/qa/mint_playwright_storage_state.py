#!/usr/bin/env python3
"""Mint a Playwright storageState so UI verification can get behind NextAuth.

WHY THIS EXISTS. `.claude/rules` make Playwright verification BINDING whenever
the operator pastes a UI screenshot or a step claims anything about the UI --
"code reading alone is not UI evidence". On 2026-08-09 that rule could not be
honoured: `browser_navigate` to http://localhost:3000/settings redirected to
/login, because the MCP browser has no session and the app is behind NextAuth
(Google SSO / passkey), which cannot be driven headlessly.

So a claim about the dashboard fell back to an API cross-check -- which is
weaker, and is exactly the substitution the rule forbids. This script removes
the excuse.

WHAT IT DOES. Reuses the ALREADY-PROVEN minter in backend/tests/auth_helper.py
(`mint_session_token`, a real NextAuth v5 JWE: dir / A256CBC-HS512 derived from
AUTH_SECRET via HKDF) and writes a Playwright storageState JSON carrying it as
the `authjs.session-token` cookie -- the exact name frontend/src/lib/api.ts:150
reads. Nothing is stubbed: the middleware runs its true decrypt + allowlist
path, so a page that renders under this cookie really is a page an authorised
operator sees.

    python scripts/qa/mint_playwright_storage_state.py
    -> writes .playwright-mcp/storage-state.json  (gitignored; 1h TTL)

Then point the Playwright MCP at it (see `.claude/rules/frontend.md`,
"Playwright behind the auth wall").

SECURITY. The cookie is a short-lived (default 1h) session token for a
localhost dev app, minted from a secret already on this machine. It is written
ONLY under .playwright-mcp/, which is gitignored -- this script refuses to run
if that is not true, so a session token can never be committed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".playwright-mcp"
OUT = OUT_DIR / "storage-state.json"
COOKIE_NAME = "authjs.session-token"          # frontend/src/lib/api.ts:150
TTL_SECONDS = 3600


def _refuse_if_committable() -> None:
    """A session cookie must never reach git. Verify, do not assume."""
    rel = OUT.relative_to(REPO)
    r = subprocess.run(["git", "check-ignore", "-q", str(rel)], cwd=REPO)
    if r.returncode != 0:
        sys.exit(
            f"REFUSED: {rel} is NOT gitignored, so writing a session token there\n"
            f"could commit it. Add '.playwright-mcp/' to .gitignore first."
        )


def main() -> int:
    _refuse_if_committable()
    sys.path.insert(0, str(REPO))
    try:
        from backend.tests.auth_helper import mint_session_token
    except Exception as e:  # noqa: BLE001 - surfaced to the operator verbatim
        sys.exit(f"REFUSED: cannot import the minter ({e}). Run from the repo root "
                 f"with the venv active.")

    try:
        token = mint_session_token(ttl_seconds=TTL_SECONDS)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"REFUSED: minting failed ({e}). Is AUTH_SECRET set in "
                 f"frontend/.env.local or backend/.env?")

    if not token or len(token) < 32:
        sys.exit(f"REFUSED: minted token looks wrong (len={len(token or '')}).")

    state = {
        "cookies": [{
            "name": COOKIE_NAME,
            "value": token,
            "domain": "localhost",
            "path": "/",
            "expires": int(time.time()) + TTL_SECONDS,
            "httpOnly": True,
            "secure": False,          # localhost dev is http://
            "sameSite": "Lax",
        }],
        "origins": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(state, indent=2))
    os.chmod(OUT, 0o600)

    print(f"  wrote {OUT.relative_to(REPO)}")
    print(f"  cookie   : {COOKIE_NAME} (len={len(token)}, expires in {TTL_SECONDS}s)")
    print(f"  perms    : 600, gitignored (verified)")
    print()
    print("  Point the Playwright MCP at it, then navigate to a protected page and")
    print("  CONFIRM you did not land on /login -- a redirect means the cookie did")
    print("  not take, and a snapshot of /login is not UI evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
