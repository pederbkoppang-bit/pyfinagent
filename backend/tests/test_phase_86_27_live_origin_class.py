"""phase-86.27 -- the live-backend guard keys on the ADDRESS, not on a spelling.

WHY THESE TESTS ARE SHAPED LIKE THIS
------------------------------------
phase-86.6 shipped `test_the_two_live_origin_predicates_AGREE`, comparing
conftest's predicate against `scripts/qa/live_backend_origin.py`'s over a table
of URLs. **It was structurally incapable of catching the defect it stood over.**
Both predicates returned `False` on all eleven bypass spellings: they AGREED,
and were BOTH WRONG. An equality oracle over two implementations of one
specification detects DIVERGENCE only; a fault they SHARE is invisible to it by
construction.

So the oracle here is not another implementation and not another table of
strings. **It is REACHABILITY, measured at test time**, against a server this
test stands up itself, bound to the IPv4 wildcard exactly as `uvicorn --host
0.0.0.0` binds -- on an EPHEMERAL port, so the operator's live book on 8000 is
never addressed by anything in this file.

The property asserted is the one that matters:

    every spelling that ACTUALLY REACHES this machine is classified as this
    machine

Because the reachable set is MEASURED rather than enumerated, a spelling nobody
has invented yet is covered the moment it works. That is the difference between
"the guard understands addresses" and "someone added another string".
"""
from __future__ import annotations

import ipaddress
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "qa"))

from live_backend_origin import (  # noqa: E402
    LIVE_BACKEND_PORT,
    address_is_live_backend,
    is_live_backend,
    own_addresses,
    targets_this_machine,
)

PROBE_TIMEOUT = 0.75


# ── the wildcard-bound stand-in for the live backend ────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def _ok(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    do_GET = do_PUT = do_POST = _ok

    def log_message(self, *a):        # keep pytest output clean
        pass


@pytest.fixture(scope="module")
def wildcard_stub():
    """A server bound to 0.0.0.0 on an EPHEMERAL port -- uvicorn's binding shape.

    Never port 8000: this module must not address the operator's book at all,
    and the assertion below makes that structural rather than a promise.
    """
    srv = ThreadingHTTPServer(("0.0.0.0", 0), _Handler)
    port = srv.server_address[1]
    assert port != LIVE_BACKEND_PORT, (
        "the stub was handed the live backend's port; refusing to run"
    )
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    try:
        yield port
    finally:
        srv.shutdown()
        srv.server_close()


def _candidate_spellings() -> list[str]:
    """The eight the step measured, three invented later, plus runtime-derived.

    This list exists to give the reachability probe something to chew on. It is
    NOT the specification -- the specification is whatever turns out to reach the
    stub, which is why nothing here is asserted directly.
    """
    hostname = socket.gethostname()
    return [
        "127.0.0.1", "localhost", "LOCALHOST", "0.0.0.0",
        "127.1", "0", "2130706433", "localhost.", "127.000.000.001",
        "[::ffff:127.0.0.1]", hostname,
        "0x7f.0x0.0x0.0x1", "017700000001", "[::ffff:7f00:1]",
        "%31%32%37%2e%30%2e%30%2e%31",
    ] + [str(int(ipaddress.ip_address(a)))            # integer form of each
         for a in sorted(own_addresses())            # real interface address
         if ipaddress.ip_address(a).version == 4]


def _reaches(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=PROBE_TIMEOUT) as r:
            return r.status == 200
    except Exception:                                 # noqa: BLE001
        return False


# ── criterion 6 / criterion 2: the GROUND TRUTH ─────────────────────────────

def test_every_spelling_that_actually_reaches_this_machine_is_classified_as_local(
        wildcard_stub):
    """THE CENTRAL TEST. Reachability is the oracle; no list is trusted.

    A spelling that reaches a wildcard-bound server on this machine IS this
    machine, by definition. If the predicate calls any such spelling remote,
    a mutating request wearing that spelling would sail past the guard.
    """
    port = wildcard_stub
    reachable = [h for h in _candidate_spellings() if _reaches(h, port)]

    # A probe that reaches nothing would make the assertion below vacuous.
    assert len(reachable) >= 8, (
        f"only {len(reachable)} spellings reached the stub -- the probe itself "
        f"is broken, so this test would pass without testing anything: {reachable}"
    )

    misclassified = [h for h in reachable
                     if not targets_this_machine(h.strip("[]"))]
    assert not misclassified, (
        "these spellings REACH a wildcard-bound server on this machine but the "
        f"predicate calls them remote, so a mutating request using them would "
        f"not be refused: {misclassified}"
    )


def test_the_predicate_is_not_a_blanket_true(wildcard_stub):
    """Positive control for the test above.

    Without this, a predicate hard-wired to `return True` would pass the ground
    truth perfectly. These are numeric, so `AI_NUMERICHOST` classifies them with
    no DNS and the control is hermetic.
    """
    for remote in ("8.8.8.8", "1.1.1.1", "203.0.113.7", "2001:4860:4860::8888"):
        assert targets_this_machine(remote) is False, remote
        assert _reaches(remote, wildcard_stub) is False, (
            f"{remote} unexpectedly reached the stub; the control is invalid"
        )


# ── criterion 2: judged on a spelling that is in NO list in the repo ─────────

def _novel_spellings() -> list[str]:
    """Derived from this machine's interfaces AT RUNTIME, never hard-coded.

    LOOPBACK IS DELIBERATELY EXCLUDED. Its integer form is `2130706433`, which is
    one of the eight spellings the step already names and appears several places
    in this repo -- so it proves nothing about the CLASS. The first draft
    included it and failed on exactly that, which was the assertion working
    rather than a bug. The addresses used here are this machine's non-loopback
    interface addresses: nobody could have enumerated them in advance, because
    they are properties of this host rather than of IPv4.

    THE FAMILY IS UNBOUNDED, AND THAT IS A CORRECTION MADE AFTER 86.27 CLOSED.
    The first version emitted exactly THREE fixed renderings per address, and it
    went red within the hour -- not because the guard broke, but because the Q/A
    independently derived the same three spellings, put them in its verdict, and
    Main transcribed that verdict verbatim into `evaluator_critique_86.27.md`.
    All three were then "in the repo", the `>= 3` floor could never be met again,
    and the test was SELF-DEFEATING BY CONSTRUCTION: any honest audit trail that
    named the spellings killed it.

    The repair is NOT to relax the absence check. It is to draw from a family
    that recording cannot exhaust: `inet_aton` accepts arbitrary leading zeros,
    so `0xc0a85655`, `0x0c0a85655`, `0x00c0a85655` ... all resolve to the same
    address (measured: 16 of 16 widths tried). The probe widens the padding until
    it finds unused spellings, so the absence requirement stays at its strictest
    -- absent from the WHOLE tracked tree, records included.

    That the family is infinite is the step's own thesis demonstrated: no
    enumeration closes a set with an unbounded number of members.
    """
    out = []
    for a in sorted(own_addresses()):
        ip = ipaddress.ip_address(a)
        if ip.version != 4 or ip.is_loopback or ip.is_link_local:
            continue
        n = int(ip)
        out.append(str(n))                              # 32-bit integer form
        for width in range(8, 24):
            out.append("0x%0*x" % (width, n))           # hex, widening padding
        for width in range(11, 26):
            out.append("0%0*o" % (width, n))            # octal, widening padding
    return out


def _absent_from_repo(spelling: str) -> bool:
    """True when `spelling` appears in NO tracked file. returncode 1 == no match."""
    return subprocess.run(
        ["git", "grep", "-I", "--fixed-strings", "-c", "--", spelling],
        cwd=REPO, capture_output=True, text=True).returncode == 1


def test_a_spelling_absent_from_the_entire_REPO_is_still_refused():
    """An allowlist extension cannot pass this.

    Every spelling is computed from the interface table at runtime and then
    PROVEN absent from the tracked tree before it is used. **Pasting these
    strings into a list to make a future failure go away does not work**: a
    pasted spelling stops being absent, drops out of the probe set, and the
    `>= 3` floor below turns red. The only way to keep this green is for the
    predicate to understand addresses.
    """
    candidates = _novel_spellings()
    assert candidates, "no non-loopback IPv4 interface address; cannot derive one"

    novel = [s for s in candidates if _absent_from_repo(s)]
    assert len(novel) >= 3, (
        "fewer than 3 of the runtime-derived spellings are absent from the repo "
        f"({len(novel)} of {len(candidates)}). Either the machine has no "
        "non-loopback IPv4 address, or someone has pasted these literals into "
        f"the tree -- which is the failure mode this floor exists to catch: {candidates}"
    )

    for spelling in novel:
        assert targets_this_machine(spelling) is True, (
            f"{spelling!r} is a valid spelling of one of THIS machine's own "
            f"addresses and was classified remote"
        )
        assert is_live_backend(f"http://{spelling}:{LIVE_BACKEND_PORT}/x") is True
        assert address_is_live_backend((spelling, LIVE_BACKEND_PORT)) is True


# ── criterion 3: the ephemeral-stub precision the 4000.2 tests depend on ─────

def test_ephemeral_ports_are_still_allowed():
    """Keying on the address alone would break every 4000.2 stub."""
    for port in (59999, 49152, 65535, 3000, 80, 443):
        assert is_live_backend(f"http://127.0.0.1:{port}/x") is False, port


def test_no_stub_in_this_repo_can_ever_bind_the_live_port():
    """Criterion 3's measurement, as an assertion rather than a claim.

    The ephemeral range is what the OS hands out for `bind(("127.0.0.1", 0))`.
    If 8000 were inside it, a stub could randomly collide with the live backend
    and this whole design would need re-thinking.
    """
    lo = int(subprocess.run(["sysctl", "-n", "net.inet.ip.portrange.first"],
                            capture_output=True, text=True).stdout.strip())
    hi = int(subprocess.run(["sysctl", "-n", "net.inet.ip.portrange.last"],
                            capture_output=True, text=True).stdout.strip())
    assert lo <= hi
    assert not (lo <= LIVE_BACKEND_PORT <= hi), (
        f"the live port {LIVE_BACKEND_PORT} is INSIDE this machine's ephemeral "
        f"range {lo}-{hi}; an ephemeral stub could collide with the live backend"
    )


# ── criterion 4: TOTAL on junk. A guard that raises is an outage. ────────────

JUNK = [None, 12345, 3.14, b"http://127.0.0.1:8000", bytearray(b"x"), object(),
        "", "   ", "://", "http://[::1", "http://127.0.0.1:notaport/",
        "not a url", "http://" + "a" * 300 + ":8000/", "%%%",
        ("http://127.0.0.1:8000",), {"url": "x"}, [1, 2, 3]]


@pytest.mark.parametrize("junk", JUNK, ids=lambda j: repr(j)[:28])
def test_is_live_backend_is_total_on_junk(junk):
    assert is_live_backend(junk) in (True, False)


@pytest.mark.parametrize("junk", JUNK + [(), ("x",), ("x", "y"), (None, None)],
                         ids=lambda j: repr(j)[:28])
def test_address_predicate_is_total_on_junk(junk):
    """The socket hook hands this whatever `socket.connect` was called with --
    including an AF_UNIX path string, and duck-typed objects from test doubles."""
    assert address_is_live_backend(junk) in (True, False)


def test_the_OLD_predicate_actually_RAISED_on_two_of_these():
    """Criterion 4 was FAILING before this step, in conftest's copy.

    Measured pre-fix: `urlsplit(12345)` raises `AttributeError: 'int' object has
    no attribute 'decode'`, and the old `except ValueError` did not catch it. So
    this is a fixed defect, not a new assertion about unchanged behaviour --
    recorded here so the criterion is not read as vacuous.
    """
    import conftest as root_conftest
    for junk in (12345, object(), None, b"x"):
        assert root_conftest._is_live_backend(junk) in (True, False)


# ── criterion 5: the guard path introduces no unbounded wait ────────────────

def test_an_unresolvable_host_neither_hangs_nor_raises():
    """`getaddrinfo` has NO timeout parameter and `setdefaulttimeout` does not
    apply to name resolution, so an unbounded wait is the real risk. A `.local`
    name was measured at 5.0 s; `.invalid` is the bounded, RFC-2606-reserved
    case. Whatever it costs, it must not raise and must be memoised."""
    host = "definitely-not-a-real-host-86-27.invalid"
    t0 = time.perf_counter()
    first = targets_this_machine(host)
    mid = time.perf_counter()
    second = targets_this_machine(host)
    end = time.perf_counter()

    assert first is True, "an unresolvable host must FAIL SAFE (treated as live)"
    assert second is first
    assert (end - mid) < (mid - t0) + 0.05, "the second call was not memoised"
    assert (mid - t0) < 10.0, f"first resolution took {mid - t0:.2f}s"


def test_the_socket_hook_path_does_no_name_resolution_for_a_numeric_address(
        monkeypatch):
    """The hook must read an ALREADY-resolved tuple, not resolve again.

    Proving the absence of a call is exactly the kind of claim that goes stale,
    so it is asserted by making a resolving call impossible rather than by
    reading the source.
    """
    import live_backend_origin as lbo

    calls = []
    real = socket.getaddrinfo

    def _spy(host, port, *a, **k):
        if k.get("flags", 0) != socket.AI_NUMERICHOST:
            calls.append(host)
        return real(host, port, *a, **k)

    monkeypatch.setattr(lbo.socket, "getaddrinfo", _spy)
    lbo._resolve_cache.clear()
    assert lbo.address_is_live_backend(("127.0.0.1", LIVE_BACKEND_PORT)) is True
    assert lbo.address_is_live_backend(("192.0.2.55", LIVE_BACKEND_PORT)) is False
    assert calls == [], f"the hook performed a NAME resolution: {calls}"


# ── criterion 6: what replaces the drift alarm, stated ──────────────────────

def test_conftest_no_longer_carries_a_second_copy_of_the_predicate():
    """The drift alarm compared two copies. There is now ONE.

    That makes `test_the_two_live_origin_predicates_AGREE` trivially true, which
    is why it is no longer the coverage -- the ground-truth test above is. This
    asserts the deletion actually happened rather than trusting the comment.
    """
    import conftest as root_conftest
    import live_backend_origin as lbo

    assert root_conftest._is_live_backend("http://127.0.0.1:8000") is True
    # The delegation is the point: same function object reached from both sides.
    assert root_conftest._shared_is_live_backend is lbo.is_live_backend
    src = (REPO / "conftest.py").read_text()
    assert "_LOOPBACK_HOSTS = frozenset" not in src, (
        "conftest re-grew its own host allowlist; the second parser is back"
    )


# ── the INSTALLED guard, end to end ─────────────────────────────────────────
# Everything above tests the PREDICATE. The predicate is not the guard. This
# runs the whole path -- urlopen -> verb flag -> socket.connect hook -> refusal
# -- against a wildcard-bound stub on an ephemeral port, in a CHILD process,
# because a PEP-578 audit hook cannot be uninstalled once installed and must not
# leak into the rest of the session. The live backend is never addressed.

_E2E_CHILD = r'''
import json, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, sys.argv[1])
from live_backend_origin import (install_socket_guard, mutating_scope,
                                 LiveBackendConnectRefused)

class H(BaseHTTPRequestHandler):
    def _ok(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    do_GET = do_PUT = do_POST = _ok
    def log_message(self, *a): pass

srv = ThreadingHTTPServer(("0.0.0.0", 0), H)
PORT = srv.server_address[1]
assert PORT != 8000
threading.Thread(target=srv.serve_forever, daemon=True).start()
time.sleep(0.2)

if sys.argv[2] == "armed":
    install_socket_guard(PORT)

def attempt(host, method):
    data = b"{}" if method != "GET" else None
    req = urllib.request.Request(f"http://{host}:{PORT}/x", data=data, method=method)
    try:
        with mutating_scope(method != "GET"):
            with urllib.request.urlopen(req, timeout=1.5) as r:
                return f"SENT:{r.status}"
    except LiveBackendConnectRefused:
        return "REFUSED"
    except urllib.error.HTTPError as e:
        return f"SENT:{e.code}"
    except Exception as e:
        return f"{type(e).__name__}"

import socket as _s
HOSTS = ["127.0.0.1", "localhost", "0.0.0.0", "127.1", "0", "2130706433",
         "localhost.", "127.000.000.001", "[::ffff:127.0.0.1]",
         "0x7f.0x0.0x0.0x1", "017700000001", "[::ffff:7f00:1]",
         "%31%32%37%2e%30%2e%30%2e%31", _s.gethostname()]
print(json.dumps({"port": PORT,
                  "put": {h: attempt(h, "PUT") for h in HOSTS},
                  "get": {h: attempt(h, "GET") for h in HOSTS}}))
'''


def _run_e2e(mode: str) -> dict:
    import json
    proc = subprocess.run(
        [sys.executable, "-c", _E2E_CHILD, str(REPO / "scripts" / "qa"), mode],
        capture_output=True, text=True, timeout=180, cwd=str(REPO))
    assert proc.returncode == 0, f"child failed: {proc.stderr[-1500:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def e2e_armed():
    return _run_e2e("armed")


@pytest.fixture(scope="module")
def e2e_disarmed():
    return _run_e2e("disarmed")


def test_e2e_every_reachable_spelling_is_REFUSED_on_a_mutating_verb(
        e2e_armed, e2e_disarmed):
    """The mutation IS the disarmed run: same code, guard not installed.

    Only spellings that actually reached the stub when DISARMED are judged --
    an unreachable spelling proves nothing, and demanding a refusal for it would
    let the test pass on a machine where nothing works at all.
    """
    reached = [h for h, v in e2e_disarmed["get"].items() if v.startswith("SENT")]
    assert len(reached) >= 8, f"probe broken; only {reached} reached the stub"

    # CONTROL: with the guard off, these mutating PUTs go through.
    unguarded = [h for h in reached if e2e_disarmed["put"][h].startswith("SENT")]
    assert unguarded == reached, (
        "the DISARMED control did not send every mutating PUT, so an ARMED "
        f"refusal would not prove the guard did it: {e2e_disarmed['put']}"
    )

    # ARMED: every one of them is refused.
    leaked = {h: e2e_armed["put"][h] for h in reached
              if e2e_armed["put"][h] != "REFUSED"}
    assert not leaked, f"mutating PUTs reached this machine un-refused: {leaked}"


def test_e2e_GETs_still_pass_with_the_guard_armed(e2e_armed):
    """Constraint 1 from conftest's docstring: `_backend_is_up()` runs inside a
    `@pytest.mark.skipif` at module import, so a raised GET would error a whole
    module's collection. A guard that blocks GETs is a worse defect than the one
    it fixes."""
    blocked = {h: v for h, v in e2e_armed["get"].items() if v == "REFUSED"}
    assert not blocked, f"the guard refused read-only GETs: {blocked}"
    assert sum(v.startswith("SENT") for v in e2e_armed["get"].values()) >= 8


def test_a_mapped_NON_loopback_address_is_recognised():
    """The gap the mutation matrix found, pinned as a test.

    CPython already reports `IPv6Address('::ffff:127.0.0.1').is_loopback ==
    True`, so the `.ipv4_mapped` unwrapping looks redundant when you probe it
    with mapped LOOPBACK -- and the first mutation cell SURVIVED for exactly
    that reason. It is not redundant: `::ffff:<this machine's LAN address>` is
    neither loopback nor unspecified, and its literal string is not in the
    interface table (which holds the bare IPv4 form). Without the unwrapping it
    reads as a remote host.
    """
    lan = [a for a in sorted(own_addresses())
           if ":" not in a and not a.startswith("127.")]
    if not lan:
        pytest.skip("no non-loopback IPv4 interface address on this host")
    mapped = f"::ffff:{lan[0]}"
    assert ipaddress.ip_address(mapped).is_loopback is False
    assert mapped not in own_addresses(), (
        "the interface table holds the mapped form directly, so this test would "
        "pass without exercising the unwrapping"
    )
    assert targets_this_machine(mapped) is True
    assert address_is_live_backend((mapped, LIVE_BACKEND_PORT)) is True
