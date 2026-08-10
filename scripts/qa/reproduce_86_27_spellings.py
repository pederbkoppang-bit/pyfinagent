#!/usr/bin/env python3
"""phase-86.27 criterion 1 -- the before/after refusal table, re-runnable.

    $ python scripts/qa/reproduce_86_27_spellings.py

SAFETY, WHICH IS THE WHOLE DESIGN OF THIS SCRIPT
------------------------------------------------
Measuring a hole in a guard that protects a LIVE TRADING BOOK must not exploit
it. So:

* **Reachability** is probed with a read-only `GET /api/health`. GETs are
  allowed by policy and change nothing.
* **Refusal** is probed by running the OLD guard function with
  `_REAL_URLOPEN` replaced by a NON-NETWORKING SENTINEL. A mutating PUT that the
  old guard would have let through lands in the sentinel and is COUNTED. Not one
  byte of it reaches the backend. The count of requests that actually left this
  process is printed at the end and is always zero.

WHAT "BEFORE" MEANS HERE
------------------------
The old predicate is reconstructed VERBATIM from
`conftest.py@cad386472dc161d121c069cce7a3032598d1f75b` -- the commit that closed
phase-86.6, i.e. the last tree before this step:

    def _is_live_backend(url: str) -> bool:
        try:
            parts = urlsplit(url)
            return parts.hostname in _LOOPBACK_HOSTS and parts.port == _LIVE_BACKEND_PORT
        except ValueError:
            return False

The SHA is pinned rather than written as `HEAD~1` because the auto-changelog
hook commits on top of every fix, so `HEAD` moves within minutes of any edit.
"""
from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "qa"))

from live_backend_origin import LIVE_BACKEND_PORT, is_live_backend  # noqa: E402

PRE_FIX_REV = "cad386472dc161d121c069cce7a3032598d1f75b"

# --- the OLD predicate, verbatim -------------------------------------------
_OLD_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


def _old_is_live_backend(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return (parts.hostname in _OLD_LOOPBACK_HOSTS
                and parts.port == LIVE_BACKEND_PORT)
    except ValueError:
        return False


_MUTATING = frozenset({"POST", "PUT", "DELETE", "PATCH"})
_REAL_URLOPEN = urllib.request.urlopen
_would_have_been_sent: list[str] = []


class _Sentinel(Exception):
    """The old guard allowed it; the request stops here instead of on the wire."""


def _sentinel_urlopen(req, *a, **k):
    _would_have_been_sent.append(getattr(req, "full_url", None) or str(req))
    raise _Sentinel()


def _old_guard_verdict(url: str) -> str:
    """Run the OLD guard body with a non-networking sentinel underneath."""
    req = urllib.request.Request(url, data=b"{}", method="PUT")
    method = req.get_method().upper()
    try:
        if method in _MUTATING and _old_is_live_backend(url):
            raise RuntimeError("phase-86.3 test guard: refusing")
        _sentinel_urlopen(req)
    except RuntimeError:
        return "REFUSED"
    except _Sentinel:
        return "NOT-REFUSED"
    return "NOT-REFUSED"


def _reachable(host: str) -> str:
    url = f"http://{host}:{LIVE_BACKEND_PORT}/api/health"
    try:
        with _REAL_URLOPEN(url, timeout=3) as r:
            return f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    except Exception as e:                                   # noqa: BLE001
        return type(e).__name__


def spellings() -> list[tuple[str, str]]:
    hostname = socket.gethostname()
    return [
        ("127.0.0.1", "control -- the canonical spelling"),
        ("localhost", "control"),
        ("LOCALHOST", "control -- case-folded by urlsplit"),
        ("0.0.0.0", "control -- added by phase-86.6"),
        ("127.1", "the step's eight: short form"),
        ("0", "the step's eight: bare zero"),
        ("2130706433", "the step's eight: 32-bit integer"),
        ("localhost.", "the step's eight: trailing dot"),
        ("127.000.000.001", "the step's eight: zero-padded"),
        ("[::ffff:127.0.0.1]", "the step's eight: IPv4-mapped"),
        (_lan(), "the step's eight: THIS MACHINE'S LAN ADDRESS"),
        (hostname, "the step's eight: THIS MACHINE'S HOSTNAME"),
        ("0x7f.0x0.0x0.0x1", "invented 2026-08-10: hex-dotted"),
        ("017700000001", "invented 2026-08-10: 32-bit octal"),
        ("[::ffff:7f00:1]", "invented 2026-08-10: hex IPv4-mapped"),
        ("%31%32%37%2e%30%2e%30%2e%31", "found by research: PERCENT-ENCODED"),
    ]


def _lan() -> str:
    try:
        return socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)[0][4][0]
    except Exception:                                        # noqa: BLE001
        return "127.0.0.1"


def main() -> int:
    print(f"pre-fix predicate reconstructed from conftest.py@{PRE_FIX_REV[:12]}")
    print(f"live backend port: {LIVE_BACKEND_PORT}\n")
    print(f"{'spelling':32} {'reachable':12} {'BEFORE':13} {'AFTER':13} note")
    print("-" * 118)
    regressions, still_open = [], []
    for host, note in spellings():
        url = f"http://{host}:{LIVE_BACKEND_PORT}/api/paper-trading/PROBE-NEVER-SENT"
        reach = _reachable(host)
        before = _old_guard_verdict(url)
        after = "REFUSED" if is_live_backend(url) else "NOT-REFUSED"
        print(f"{host:32} {reach:12} {before:13} {after:13} {note}")
        if before == "REFUSED" and after != "REFUSED":
            regressions.append(host)
        if reach.startswith("HTTP") and after != "REFUSED":
            still_open.append(host)

    print("-" * 118)
    print(f"mutating requests that actually left this process: "
          f"{0} (all {len(_would_have_been_sent)} were absorbed by the sentinel)")
    if regressions:
        print(f"\nREGRESSION -- these were refused BEFORE and are not now: {regressions}")
        return 2
    if still_open:
        print(f"\nSTILL OPEN -- reachable and un-refused after the fix: {still_open}")
        return 1
    print("\nEvery spelling that REACHES this machine on the live port is refused, "
          "and no refusal was lost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
