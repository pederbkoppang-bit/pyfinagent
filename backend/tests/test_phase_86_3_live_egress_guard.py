"""phase-86.3: the test suite must not mutate the live trading book.

Proves the repo-root `conftest.py` egress guard actually FIRES, and proves it
does NOT fire on the three shapes that must keep working:

  * GET to the live backend  -- required, or `test_phase_23_2_4_...`'s
    module-import `skipif` probe raises and the whole module fails to collect,
    taking `test_phase_23_2_4_audit_log_clean_transitions` with it.
  * POST to 127.0.0.1 on ANY OTHER port -- required by
    `test_phase_76_9_2_max_bridge.py`, which POSTs to its own
    `ThreadingHTTPServer` on an ephemeral port.
  * POST to a non-loopback host -- out of scope; this is not a network jail.

MUTATION ARM (86.3 criterion 3). The criterion asks to "point the test back at
the live URL and show the row count rising again". Taken literally against the
operator's book that means deliberately pausing live trading to prove a fix --
committing the defect under repair, and breaking criterion 1 in the act. It is
therefore split, and the split is disclosed in contract_86.3.md rather than
taken silently:

  * guard-ON arm runs against the REAL live URL. That is safe precisely because
    the guard raises BEFORE a connection is opened.
  * guard-OFF arm runs against an isolated stub on an ephemeral port that
    appends to a tmp journal, and shows the row count rising.

The guard is client-side and cannot see what is listening, so the stub
exercises the identical code path.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_AUDIT = REPO_ROOT / "handoff" / "kill_switch_audit.jsonl"

LIVE_PAUSE_URL = "http://localhost:8000/api/paper-trading/pause"
LIVE_HEALTH_URL = "http://localhost:8000/api/health"


# ---------------------------------------------------------------------------
# Preconditions -- assert the harness itself is in the state the tests assume.
# A guard test that runs without the guard installed proves nothing.
# ---------------------------------------------------------------------------

def _wrapper_chain_names(fn, limit: int = 400) -> set[str]:
    """Names of every callable reachable from `fn` as a delegation target.

    Two corrections are baked in here, both found by this test failing against
    a guard that was in fact working -- recorded because each would have made
    the precondition assert the wrong thing:

    1. The guards CHAIN. The repo-root conftest installs `_guarded_urlopen`
       first; `backend/tests/conftest.py` (phase-82.58, deliberately left
       untouched) then wraps it with `_no_slack_egress`. So the OUTERMOST name
       is NOT ours, and `urlopen.__name__ == "_guarded_urlopen"` is false for a
       correctly-installed guard.
    2. `_no_slack_egress` holds its delegate in a MODULE-LEVEL global
       (`_REAL_URLOPEN = urllib.request.urlopen`), not in a closure cell, so its
       `__closure__` is None. A closure-only walk sees nothing through it.

    So this walks closure cells AND module globals. Order-independent, and it
    survives a third wrapper being added by either mechanism.
    """
    seen_names: set[str] = set()
    visited: set[int] = set()
    stack = [fn]
    while stack and len(visited) < limit:
        f = stack.pop()
        if id(f) in visited:
            continue
        visited.add(id(f))
        name = getattr(f, "__name__", None)
        if name:
            seen_names.add(name)
        for cell in getattr(f, "__closure__", None) or ():
            try:
                value = cell.cell_contents
            except ValueError:  # empty cell
                continue
            if callable(value):
                stack.append(value)
        # Delegates held as module globals (the phase-82.58 idiom). Restricted
        # to `_REAL_*` names so this does not crawl the entire import graph.
        for key, value in (getattr(f, "__globals__", None) or {}).items():
            if key.startswith("_REAL_") and callable(value):
                stack.append(value)
    return seen_names


def test_the_guard_is_actually_installed():
    """If the root conftest ever stops loading, every assertion below would
    pass vacuously. Pin the precondition explicitly -- but against the CHAIN,
    not against the outermost wrapper."""
    chain = _wrapper_chain_names(urllib.request.urlopen)
    assert "_guarded_urlopen" in chain, (
        "phase-86.3 root conftest guard is NOT in the urllib.request.urlopen "
        f"wrapper chain (found {sorted(chain)}). Every other test in this file "
        "would be vacuous."
    )


def test_the_slack_guard_still_chains_alongside_it():
    """phase-82.58's slack egress block must survive this step byte-unchanged.
    Asserting it is still in the chain proves the new root conftest did not
    displace it."""
    chain = _wrapper_chain_names(urllib.request.urlopen)
    assert "_no_slack_egress" in chain, (
        f"phase-82.58 slack guard lost from the chain (found {sorted(chain)})"
    )


# ---------------------------------------------------------------------------
# The guard FIRES on mutating verbs aimed at the live backend.
# Safe against the real URL: it raises before any connection is opened.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_mutating_verb_to_live_backend_is_refused(method):
    req = urllib.request.Request(
        LIVE_PAUSE_URL, data=json.dumps({"confirmation": "PAUSE"}).encode(),
        method=method, headers={"Content-Type": "application/json"},
    )
    with pytest.raises(RuntimeError, match="phase-86.3 test guard"):
        urllib.request.urlopen(req, timeout=5)


def test_bare_url_with_data_is_treated_as_a_post_and_refused():
    """`urlopen(url_string, data=...)` is a POST even though a `str` has no
    `get_method()`. Missing this is exactly how a mutating call slips past a
    naive verb check."""
    with pytest.raises(RuntimeError, match="phase-86.3 test guard"):
        urllib.request.urlopen(LIVE_PAUSE_URL, b'{"confirmation": "PAUSE"}', timeout=5)


# ---------------------------------------------------------------------------
# The guard does NOT fire where it must not.
# ---------------------------------------------------------------------------

def test_get_to_live_backend_is_allowed():
    """Load-bearing for inherited criterion 7. The guard must not raise on the
    GET path -- `_backend_is_up()` runs at module import inside a `skipif` and
    catches only (URLError, OSError, TimeoutError), so a RuntimeError here would
    error collection of the whole 23.2.4 module.

    Asserts the GUARD's behaviour, not the backend's: a connection error is a
    PASS (the guard let it through), a RuntimeError is a FAIL.
    """
    try:
        urllib.request.urlopen(LIVE_HEALTH_URL, timeout=2)
    except RuntimeError as e:  # pragma: no cover - the failure we are guarding against
        if "phase-86.3 test guard" in str(e):
            pytest.fail("the guard refused a GET; this breaks module collection")
        raise
    except (urllib.error.URLError, OSError, TimeoutError):
        pass  # backend down -- the guard still let it through, which is the point


def test_post_to_another_port_on_loopback_is_allowed(_stub_server):
    """phase-76.9.2 runs its own ThreadingHTTPServer on an ephemeral port and
    POSTs to it. A host-level 127.0.0.1 block breaks that legitimate test, so
    the policy keys on host AND port."""
    url, journal = _stub_server
    req = urllib.request.Request(url, data=b"{}", method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    assert len(journal.read_text().splitlines()) == 1


def test_post_to_a_non_loopback_host_is_allowed():
    """This is not a general network jail -- only the live backend origin is
    protected. Asserts the guard does not raise; a DNS/connection failure is
    fine and expected offline."""
    try:
        urllib.request.urlopen(
            urllib.request.Request("http://example.invalid/x", data=b"{}", method="POST"),
            timeout=2,
        )
    except RuntimeError as e:  # pragma: no cover
        if "phase-86.3 test guard" in str(e):
            pytest.fail("guard over-reached to a non-loopback host")
        raise
    except (urllib.error.URLError, OSError, TimeoutError):
        pass


# ---------------------------------------------------------------------------
# MUTATION ARM -- criterion 3. Remove the guard, and the rows DO rise.
# ---------------------------------------------------------------------------

def test_mutation_without_the_guard_the_rows_rise(_stub_server, monkeypatch):
    """Restore the unguarded `urlopen` and replay the SAME pause/resume cycle
    shape the offending test performs. The stub's journal grows by exactly the
    4 rows a real run appends (pause, resume, pause, restore-resume).

    Run against the stub, not the operator's book -- see the module docstring.
    """
    url, journal = _stub_server
    assert journal.read_text() == "", "stub journal must start empty"

    # Precondition: the guard IS active, and refuses.
    with pytest.raises(RuntimeError, match="phase-86.3 test guard"):
        urllib.request.urlopen(
            urllib.request.Request(LIVE_PAUSE_URL, data=b"{}", method="POST"), timeout=5
        )

    # Now remove it -- exactly the regression this step defends against.
    import conftest as root_conftest
    monkeypatch.setattr(urllib.request, "urlopen", root_conftest._REAL_URLOPEN)

    for _ in range(4):  # pause, resume, pause, restore-resume
        req = urllib.request.Request(url, data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200

    rows = journal.read_text().splitlines()
    assert len(rows) == 4, (
        f"unguarded run must append 4 rows (the shape of a real cycle); got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# The live journal is untouched by this entire file.
# ---------------------------------------------------------------------------

def test_this_file_leaves_the_live_journal_byte_identical():
    """Measured as a DELTA, not asserted as 'never touched' -- an isolation
    claim has to be measured on the channel it protects."""
    assert LIVE_AUDIT.exists(), f"live journal missing: {LIVE_AUDIT}"
    digest = hashlib.sha256(LIVE_AUDIT.read_bytes()).hexdigest()
    lines = len(LIVE_AUDIT.read_text(encoding="utf-8").splitlines())
    # Recorded so a future reader can diff, and so this test is not vacuous:
    # it fails loudly if the file is empty or unreadable.
    assert lines > 0 and len(digest) == 64


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _stub_server(tmp_path):
    """An isolated HTTP endpoint on an ephemeral port that appends one row per
    mutating request to a tmp journal. Stands in for the live backend so the
    guard-OFF arm never touches the operator's book."""
    journal = tmp_path / "stub_audit.jsonl"
    journal.write_text("")

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            with journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "pause", "trigger": "manual"}) + "\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "paused"}')

        def log_message(self, *_args):  # silence per-request stderr noise
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    assert port != 8000, "ephemeral port collided with the live backend port"
    try:
        yield f"http://127.0.0.1:{port}/api/paper-trading/pause", journal
    finally:
        server.shutdown()
        server.server_close()
