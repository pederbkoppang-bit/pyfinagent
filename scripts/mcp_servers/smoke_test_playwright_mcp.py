"""goal-browser-mcp: smoke test for the pinned Playwright MCP server.

Spawns `@playwright/mcp` via `npx` over stdio (headless + isolated so it runs
unattended and leaves no profile), performs the MCP handshake
(`initialize` -> `notifications/initialized`), then:

  Phase 1 (MCP-level): `tools/list` and asserts the browser-driving tools
    (`browser_navigate`, `browser_click`, `browser_snapshot`) are present.
    This is the in-session evidence that the server ATTACHES and its tools
    are CALLABLE (the live `mcp__playwright__*` dispatch needs a Claude Code
    restart -- see docs/runbooks/browser-mcp.md).

  Phase 2 (live driving): `browser_navigate` to the running dev server's
    public `/login` page (no auth needed), then `browser_snapshot`, and
    asserts the real login DOM is present (specific tokens like "sign in" /
    "passkey"; explicitly REJECTS Playwright error text so a failed browser
    launch can never false-pass).

Phase 2 is SKIPPED (not failed) if `localhost:3000` is unreachable.

BROWSER: @playwright/mcp@0.0.75 defaults to the system `chrome` channel
(`/Applications/Google Chrome.app`), which is NOT installed on this Mac.
This test (and the `.mcp.json` entry) therefore point `--executable-path` at
the Chromium that Playwright already bundles in the ms-playwright cache
(resolved dynamically here so the build number isn't hard-coded). If the
cache is missing, run `npx playwright install chromium`.

Exit 0 = pinned MCP attaches, exposes browser tools, and (if the dev server
is up) drives a real navigation + DOM read end-to-end.
Exit non-zero = a required step failed; server stderr tail is captured.

Usage:
    python scripts/mcp_servers/smoke_test_playwright_mcp.py
    LOGIN_URL=http://localhost:3000/login python scripts/mcp_servers/smoke_test_playwright_mcp.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PACKAGE_SPEC = "@playwright/mcp@0.0.75"
TIMEOUT_S = 90.0
LOGIN_URL = os.environ.get("LOGIN_URL", "http://localhost:3000/login")
REQUIRED_TOOLS = ("browser_navigate", "browser_click", "browser_snapshot")
# SPECIFIC tokens only present on the real NextAuth login page (NOT "google",
# which also appears in the "Google Chrome.app not found" launch error).
LOGIN_TOKENS = ("sign in", "passkey", "ai financial analyst")
# Substrings that mean the browser failed to launch / a tool errored.
ERROR_MARKERS = ("### error", "is not found at", "run \"npx playwright install", "executable doesn't exist")


def _resolve_chromium() -> str | None:
    """Find the Chromium binary Playwright bundles in the ms-playwright cache.

    Globs so the build number (e.g. chromium-1208) is not hard-coded; returns
    the lexicographically-latest match, or None if the cache is missing.
    """
    cache = Path.home() / "Library" / "Caches" / "ms-playwright"
    patterns = (
        "chromium-*/chrome-mac*/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        "chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
    )
    hits: list[Path] = []
    for pat in patterns:
        hits.extend(cache.glob(pat))
    hits = [h for h in hits if h.exists()]
    return str(sorted(hits)[-1]) if hits else None


def _send(proc: subprocess.Popen, payload: dict) -> None:
    proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _recv_until_id(proc: subprocess.Popen, want_id: int, deadline: float) -> dict:
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except Exception:
            continue
        if msg.get("id") == want_id:
            return msg
    raise TimeoutError(f"no response with id={want_id} within {TIMEOUT_S}s")


def _text_of(resp: dict) -> str:
    return "\n".join(
        c.get("text", "")
        for c in resp.get("result", {}).get("content", [])
        if isinstance(c, dict) and c.get("type") == "text"
    )


def _dev_server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as resp:
            return resp.status < 500
    except Exception:
        return False


def main() -> int:
    chromium = _resolve_chromium()
    if chromium is None:
        print(
            "FAIL no bundled Chromium found in ~/Library/Caches/ms-playwright -- "
            "run `npx playwright install chromium` first.",
            file=sys.stderr,
        )
        return 1
    print(f"using bundled Chromium: {chromium}")

    cmd = [
        "npx", "-y", PACKAGE_SPEC,
        "--headless",
        "--isolated",
        "--executable-path", chromium,
    ]
    print(f"spawning: npx -y {PACKAGE_SPEC} --headless --isolated --executable-path <bundled>", flush=True)
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )

    try:
        deadline = time.time() + TIMEOUT_S

        _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", "capabilities": {},
                "clientInfo": {"name": "smoke-test", "version": "1.0"},
            },
        })
        init_resp = _recv_until_id(proc, 1, deadline)
        if "result" not in init_resp:
            print(f"FAIL initialize: {init_resp}", file=sys.stderr)
            return 1
        print(f"OK initialize -- server={init_resp['result'].get('serverInfo', {}).get('name', '?')}")

        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # --- Phase 1: tools/list + required browser tools present ---
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tools_resp = _recv_until_id(proc, 2, deadline)
        if "result" not in tools_resp:
            print(f"FAIL tools/list: {tools_resp}", file=sys.stderr)
            return 1
        tool_names = sorted({t.get("name") for t in tools_resp["result"].get("tools", [])})
        print(f"OK tools/list -- {len(tool_names)} tools; sample: {tool_names[:8]}")
        for required in REQUIRED_TOOLS:
            if required not in tool_names:
                print(f"FAIL missing required tool: {required}", file=sys.stderr)
                return 1
        print(f"OK required browser tools present: {list(REQUIRED_TOOLS)}")

        # --- Phase 2: live drive against the dev server's /login page ---
        if not _dev_server_up(LOGIN_URL):
            print(f"SKIP live-driving leg -- dev server not reachable at {LOGIN_URL} (Phase 1 passed).")
            return 0

        _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "browser_navigate", "arguments": {"url": LOGIN_URL}},
        })
        nav_resp = _recv_until_id(proc, 3, time.time() + TIMEOUT_S)
        nav_text = _text_of(nav_resp).lower()
        if "result" not in nav_resp or nav_resp["result"].get("isError") or any(m in nav_text for m in ERROR_MARKERS):
            print(f"FAIL browser_navigate {LOGIN_URL} (launch/nav error):\n{nav_text[:600]}", file=sys.stderr)
            return 1
        print(f"OK browser_navigate -- {LOGIN_URL}")

        _send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "browser_snapshot", "arguments": {}},
        })
        snap_resp = _recv_until_id(proc, 4, time.time() + TIMEOUT_S)
        if "result" not in snap_resp:
            print(f"FAIL browser_snapshot: {snap_resp}", file=sys.stderr)
            return 1
        blob = _text_of(snap_resp).lower()
        if any(m in blob for m in ERROR_MARKERS):
            print(f"FAIL browser_snapshot returned an error (browser likely failed to launch):\n{blob[:600]}", file=sys.stderr)
            return 1
        hit = next((tok for tok in LOGIN_TOKENS if tok in blob), None)
        if hit is None:
            print(
                f"FAIL browser_snapshot read a page but none of {LOGIN_TOKENS} were present. "
                f"First 600 chars:\n{blob[:600]}",
                file=sys.stderr,
            )
            return 1
        print(f"OK browser_snapshot -- real login DOM read (matched token: {hit!r})")
        print("\nSMOKE PASS: Playwright MCP attaches, exposes browser tools, and drove a "
              "real navigation + DOM read on the live dev server.")
        return 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            err = proc.stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                print("\nserver stderr (tail):\n  " + "\n  ".join(err.strip().splitlines()[-5:]), file=sys.stderr)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
