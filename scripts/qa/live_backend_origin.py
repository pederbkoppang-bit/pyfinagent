"""phase-86.6 Part B / phase-86.27 -- is this request aimed at the operator's LIVE backend?

WHY THIS MODULE IS THE SINGLE AUTHORITY
---------------------------------------
phase-86.3 installed a guard in the repo-root `conftest.py` that refuses MUTATING
HTTP verbs aimed at the live backend. That guard lives in the PYTEST PROCESS. A
test that shells out gets a CHILD process, and a child loads no conftest -- so
the guard is structurally absent there. Measured at source:
`backend/tests/test_phase_4000_2_cc_rail_smoke.py` drives
`scripts/qa/smoke_cc_rail_e2e.py` via `subprocess.run`, and that script MUTATES
(`PUT /api/settings/`, `POST /api/analysis/`).

phase-86.6 gave the child its own guard but left `conftest.py` holding a SECOND
copy of the predicate, with a test comparing the two so a drift would fail
loudly. **phase-86.27 deletes the copy.** `conftest.py` now imports from here.
Two definitions of one fact is how a vocabulary drifts apart -- and worse, an
agreement check between two copies cannot detect a fault they SHARE, which is
exactly what happened: both copies returned False on all eleven bypass
spellings. They agreed, and were both wrong.

WHY A HOST STRING CANNOT DECIDE THIS -- the phase-86.27 finding
--------------------------------------------------------------
The backend runs `uvicorn --host 0.0.0.0`, which binds the IPv4 WILDCARD
(`lsof`: `TCP *:8000 (LISTEN)`), so EVERY address of this machine reaches the
running book. The old predicate tested `urlsplit(url).hostname` against four
literal strings. MEASURED 2026-08-10, reachable (`GET /api/health` -> 200) and
NOT refused on a mutating PUT: `127.1`, `0`, `2130706433`, `localhost.`,
`127.000.000.001`, `[::ffff:127.0.0.1]`, `0x7f.0x0.0x0.0x1`, `017700000001`,
`[::ffff:7f00:1]`, and -- not loopback spellings at all -- the machine's LAN
address `192.168.86.85` and its hostname `ford-sin-mini.lan`.

**The count went 8 -> 11 across two independent looks, and 12 once the research
round added `%31%32%37%2e%30%2e%30%2e%31`. The residual is an open-ended
population, not a list, so extending the list is not a fix.**

And the percent-encoded case shows the string layer cannot be repaired even in
principle: `urlsplit()` reports `%31%32%37%2e%30%2e%30%2e%31` while urllib
percent-decodes and connects to `127.0.0.1:8000`. **The guard's parser is not
the requester's parser.** Fixing that at the string layer means re-implementing
urllib's decode chain -- a second parser obliged to agree with the first
forever, which is the anti-pattern itself (Stenberg, "don't mix URL parsers";
Snyk, "single parsing point"; CWE-180 canonicalise-then-validate; CWE-184
incomplete denylist).

So this module offers TWO predicates and is explicit about their standing:

* `is_live_backend(url)` -- BEST EFFORT, for the one situation where only a
  string exists: a CLI argument, checked before any request is made. It
  canonicalises rather than pattern-matches, and it FAILS SAFE (a host it cannot
  resolve is treated as live). It is a friendly early error, NOT the last line
  of defence.
* `install_socket_guard(...)` -- THE AUTHORITY. A PEP-578 `socket.connect` hook
  sees the RESOLVED address for every spelling, fires BEFORE the connect, and
  aborts it. There is no second resolution to race, so the check-to-use window
  that punishes naive resolve-and-compare (CVE-2025-31490) is zero here.

WHAT COUNTS AS LIVE
-------------------
`(the address is one of THIS MACHINE's) AND (the port is the backend's)`.

Both halves are load-bearing. Dropping the address half ("refuse anything on
8000") is simpler and would in fact be safe here -- measured, no stub in this
repo ever binds 8000, and this machine's ephemeral range is 49152-65535 -- but
it flips the frozen row `https://example.com:8000 -> allow`
(`test_phase_86_6_subprocess_channel.py`), an already-graded assertion, for no
gain. Dropping the port half breaks every 4000.2 stub, which stands up servers
on `127.0.0.1:0`.
"""
from __future__ import annotations

import ipaddress
import socket
import sys
import threading
from urllib.parse import urlsplit

#: The live backend's port (uvicorn --port 8000; see CLAUDE.md Quick Start).
LIVE_BACKEND_PORT = 8000

#: Verbs that can change server-side state. GET/HEAD/OPTIONS are ALWAYS allowed:
#: `_backend_is_up()` runs inside a `@pytest.mark.skipif` at module import, so a
#: raised GET would error a whole module's collection. Defined here, once, for
#: the same reason the address predicate is -- this step exists because one fact
#: was spelled in two places.
MUTATING_VERBS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

#: Kept ONLY so that pre-86.27 readers and any external importer do not break on
#: a missing name. It is NOT consulted by any predicate below -- consulting it is
#: the defect. Do not add to it; adding a spelling is the fix that failed three
#: times.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})


# ---------------------------------------------------------------------------
# This machine's own addresses
# ---------------------------------------------------------------------------

def _enumerate_interface_addresses() -> frozenset[str] | None:
    """Every address bound to any interface, or None if we cannot enumerate.

    `psutil` is the only complete enumerator available in this venv --
    `netifaces` is not installed, `socket.if_nameindex()` returns interface
    NAMES rather than addresses, and the UDP-connect trick returns exactly ONE
    address (the default route's source) and would miss every other interface.

    psutil is present here TRANSITIVELY (via gpt-researcher / unstructured) and
    is declared in no requirements file, so its absence is a real possibility
    and is handled rather than assumed -- see `_is_this_machine`.
    """
    try:
        import psutil
    except Exception:                                   # noqa: BLE001
        return None
    try:
        out = set()
        for _iface, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family in (socket.AF_INET, socket.AF_INET6):
                    out.add(str(a.address).split("%")[0])
        return frozenset(out)
    except Exception:                                   # noqa: BLE001
        return None


_own_lock = threading.Lock()
_own_cache: frozenset[str] | None = None
_own_enumerable: bool | None = None


def own_addresses(refresh: bool = False) -> frozenset[str]:
    """Cached interface addresses. Empty frozenset when not enumerable."""
    global _own_cache, _own_enumerable
    with _own_lock:
        if _own_cache is None or refresh:
            enumerated = _enumerate_interface_addresses()
            _own_enumerable = enumerated is not None
            _own_cache = enumerated if enumerated is not None else frozenset()
        return _own_cache


def interfaces_enumerable() -> bool:
    own_addresses()
    return bool(_own_enumerable)


def _is_this_machine(addr) -> bool:
    """True if `addr` (a canonical address string) belongs to this host.

    TOTAL: any input that is not a parseable address is False rather than an
    exception. A guard that raises on junk converts a safety check into an
    outage.
    """
    try:
        text = str(addr).split("%")[0].strip()          # strip an IPv6 zone id
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        ip = ipaddress.ip_address(text)
    except (ValueError, TypeError, AttributeError):
        return False

    # 127.0.0.0/8 and ::1 -- and the WILDCARD, which means "this machine" by
    # definition: connecting to 0.0.0.0 reaches the local wildcard listener
    # (measured: GET http://0.0.0.0:8000/api/health -> 200).
    if ip.is_loopback or ip.is_unspecified:
        return True

    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and _is_this_machine(str(mapped)):
        return True

    if str(ip) in own_addresses():
        return True

    # A NEW address can appear on an interface mid-session (DHCP, VPN up, an
    # IPv6 temporary rotating). Missing it would be an UNDER-refusal, so a miss
    # costs one re-enumeration (measured 0.12-0.29 ms) rather than a wrong
    # answer.
    if interfaces_enumerable() and str(ip) in own_addresses(refresh=True):
        return True

    if not interfaces_enumerable():
        # DEGRADED MODE -- phase-86.30. Without a complete interface list we
        # cannot prove an address is NOT ours, so we refuse unconditionally.
        #
        # WAS `return not ip.is_global`, described as "over-refusal, which is
        # the safe direction". IT WAS THE OPPOSITE for exactly the addresses a
        # modern host carries. `is_global` answers ROUTABILITY, not OWNERSHIP:
        # a host's own global unicast addresses are `is_global == True`, so the
        # old expression returned False -- "provably remote" -- for this
        # machine's own IPv6. MEASURED 2026-08-10 with psutil forced to fail:
        # 6 of 6 of this host's global IPv6 addresses (2001:4654:6451:0:*) were
        # classified REMOTE, i.e. an UNDER-refusal in the branch documented as
        # erring the other way. Controls the same run: 127.0.0.1 -> True,
        # 2606:4700:4700::1111 (Cloudflare) -> False.
        #
        # RFC 8981 makes this permanent, not incidental: temporary addresses
        # are global-scope and ROTATE, so any enumeration is stale by design
        # and `is_global` can never stand in for "mine". Saltzer & Schroeder:
        # an exclusion-mechanism mistake "tends to fail by allowing access, a
        # failure which may go unnoticed" (CWE-636).
        #
        # This now matches BOTH sibling degraded paths rather than being the
        # one that disagrees: `conftest.py` falls back to port-only refusal
        # ("over-refuses ... which is the safe direction") and
        # `_canonical_addresses` unresolvable -> `verdict = True` below.
        #
        # SCOPE: this branch is unreachable while psutil is importable, so the
        # frozen row `https://example.com:8000 -> allow`
        # (test_phase_86_6_subprocess_channel.py) is graded on the NORMAL path
        # and is untouched. In degraded mode that row does flip to refuse --
        # which is the point: without an interface list we cannot tell
        # example.com from ourselves, and refusing is the safe answer.
        return True

    return False


# ---------------------------------------------------------------------------
# Canonicalisation: turn a host STRING into addresses, cheaply and totally
# ---------------------------------------------------------------------------

_resolve_lock = threading.Lock()
_resolve_cache: dict[str, bool] = {}


def _canonical_addresses(host: str) -> set[str] | None:
    """Addresses `host` denotes, or None if it cannot be canonicalised.

    `AI_NUMERICHOST` first: it disables name resolution entirely, performs NO
    network I/O, and is the SAME libc parser the socket will use -- so it
    canonicalises the whole numeric class (short form, decimal, hex, octal,
    zero-padded, IPv4-mapped) including spellings nobody has invented yet.
    Measured 0.016-0.123 ms. A genuine NAME raises `gaierror` immediately
    (0.017 ms) and falls through to one real resolution.

    NOTE `socket.getaddrinfo` has NO timeout parameter and
    `socket.setdefaulttimeout` does not apply to name resolution; a single
    unresolvable `.local` name was measured at 5.0 s. That cost is bounded here
    by (a) the caller only reaching this function for port-8000 targets and
    (b) memoisation per host string.
    """
    for flags in (socket.AI_NUMERICHOST, 0):
        try:
            return {sa[0] for *_rest, sa in
                    socket.getaddrinfo(host, None, flags=flags)}
        except socket.gaierror:
            continue
        except (UnicodeError, OSError, ValueError, TypeError):
            return None
    return None


def targets_this_machine(host) -> bool:
    """True if `host` (any spelling) denotes this machine.

    FAILS SAFE: a host that cannot be canonicalised returns True. That is not
    timidity -- it is the percent-encoded case. `getaddrinfo` cannot resolve
    `%31%32%37%2e%30%2e%30%2e%31`, while urllib decodes it and connects to
    127.0.0.1. Treating "I could not tell" as "not this machine" is precisely
    the under-refusal this step exists to remove.
    """
    if not isinstance(host, str) or not host.strip():
        return False
    key = host.strip().lower()
    with _resolve_lock:
        hit = _resolve_cache.get(key)
    if hit is not None:
        return hit
    addrs = _canonical_addresses(key.strip("[]") if key.startswith("[") else key)
    verdict = True if addrs is None else any(_is_this_machine(a) for a in addrs)
    with _resolve_lock:
        _resolve_cache[key] = verdict
    return verdict


def is_live_backend(url) -> bool:
    """BEST EFFORT string predicate -- see the module docstring for its standing.

    Deliberately total: an unparseable URL is NOT live (False) rather than an
    exception, because this runs on a guard path and a guard that raises on junk
    input turns a safety check into an outage.

    The PORT is tested first, and that ordering is load-bearing rather than
    stylistic: it means the overwhelming majority of calls -- every ephemeral
    stub, every frontend URL, every unparseable string -- return without any
    name resolution at all.
    """
    try:
        parts = urlsplit(str(url))
        port = parts.port
        host = parts.hostname
    except (ValueError, AttributeError, TypeError, UnicodeError):
        # An unparseable port raises ValueError; a malformed URL is not a live
        # POST. Fail OPEN here rather than break unrelated tests -- the socket
        # guard is the backstop for anything that really does reach the book.
        return False
    if port != LIVE_BACKEND_PORT:
        return False
    if not host:
        return False
    return targets_this_machine(host)


# ---------------------------------------------------------------------------
# THE AUTHORITY: a PEP-578 socket.connect guard
# ---------------------------------------------------------------------------

class LiveBackendConnectRefused(BaseException):
    """A MUTATING request was about to reach the operator's live backend.

    Derives from `BaseException` DELIBERATELY, matching phase-86.6's
    `LiveStateWriteRefused`. `kill_switch._append_audit` wraps its work in
    `except Exception:` and logs a warning, so a refusal built to this repo's
    usual `RuntimeError` pattern would be SILENTLY ABSORBED -- blocked, swallowed,
    test green, nobody learns anything.
    """


#: The HTTP verb is known at the urllib layer; the resolved address is known at
#: the socket layer. Neither layer has both, so the verb is carried across on a
#: thread-local. urllib and urllib3 both connect on the calling thread.
_verb_state = threading.local()


def mutating_scope(is_mutating: bool):
    """Context manager marking the current thread's request as mutating."""
    class _Scope:
        def __enter__(self):
            self.prev = getattr(_verb_state, "mutating", False)
            _verb_state.mutating = bool(is_mutating)
            return self

        def __exit__(self, *exc):
            _verb_state.mutating = self.prev
            return False
    return _Scope()


def current_request_is_mutating() -> bool:
    return bool(getattr(_verb_state, "mutating", False))


def address_is_live_backend(address, port_override: int | None = None) -> bool:
    """Decide on a `socket.connect` sockaddr. TOTAL -- never raises.

    AF_INET gives `(host, port)`; AF_INET6 gives `(host, port, flow, scope)`;
    AF_UNIX gives a `str` path, which is not an internet address at all.
    """
    live_port = LIVE_BACKEND_PORT if port_override is None else port_override
    try:
        if isinstance(address, (str, bytes, bytearray)):
            return False                                # AF_UNIX and friends
        host = address[0]
        port = address[1]
    except (TypeError, IndexError, KeyError):
        return False
    try:
        if int(port) != int(live_port):
            return False
    except (TypeError, ValueError):
        return False
    if _is_this_machine(host):
        return True
    # `socket.connect(("127.1", 9))` hands the hook the RAW string: the OS
    # resolves it internally, so the address is not always canonical here.
    # Canonicalising costs no network for the numeric class.
    return targets_this_machine(host) if isinstance(host, str) else False


def install_socket_guard(port: int | None = None, refusal=None) -> None:
    """Install the PEP-578 `socket.connect` guard.

    Only MUTATING requests are refused: `GET`s to the live backend must keep
    working, because `_backend_is_up()` runs inside a `@pytest.mark.skipif` at
    module IMPORT time and a raised GET would error the whole module's
    collection.

    An audit hook cannot be uninstalled once installed (PEP 578), so everything
    the hook consults -- the verb flag and the port -- is read at CALL time.
    PEP 578 is also explicit that this "is not sandboxing": it stops accidents,
    not a determined caller.
    """
    live_port = LIVE_BACKEND_PORT if port is None else int(port)

    def _hook(event: str, args) -> None:
        if event != "socket.connect":
            return
        if not current_request_is_mutating():
            return
        try:
            address = args[1]
        except (IndexError, TypeError):
            return
        if not address_is_live_backend(address, live_port):
            return
        raise (refusal or _default_refusal)(address, live_port)

    sys.addaudithook(_hook)


def _default_refusal(address, live_port) -> BaseException:
    return LiveBackendConnectRefused(
        "phase-86.27 guard: refusing a MUTATING request that was about to reach "
        f"the operator's LIVE backend (resolved address={address!r}, "
        f"port={live_port}).\n"
        "The address above is what the socket was about to connect to, not the "
        "spelling in the URL -- every spelling of this machine resolves here, "
        "which is why this guard keys on the address rather than on a list of "
        "host strings.\n"
        "Drive the endpoint IN-PROCESS instead (fastapi TestClient / "
        "httpx.ASGITransport) with kill_switch._AUDIT_PATH redirected to "
        "tmp_path. GETs to the live backend are still allowed."
    )
