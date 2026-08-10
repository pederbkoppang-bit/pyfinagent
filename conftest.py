"""Repo-root pytest conftest — test-session egress policy (phase-86.3).

WHY THIS FILE EXISTS
--------------------
`backend/tests/test_phase_23_2_4_pause_resume_no_deadlock_live.py` POSTs a real
`pause -> resume -> pause` cycle (plus a 4th restore `resume` when the book was
unpaused) to `http://localhost:8000` whenever `GET /api/health` answers 200.
Measured 2026-08-09: 8 rows appended to the LIVE
`handoff/kill_switch_audit.jsonl` across two full-suite runs, and the operator's
live ARMED trading book was paused four times by a test run.

A detached git worktree does NOT contain this. A worktree relocates
`Path(__file__).parents[N]`; it does not relocate a TCP connection.

THE POLICY, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------
Refuse a request when BOTH hold:
  * the verb is MUTATING (POST / PUT / DELETE / PATCH), and
  * the target host:port is the live backend origin.

Two constraints make anything simpler wrong, and both were measured:

1. **GETs must pass.** `_backend_is_up()` is called inside the module's
   `@pytest.mark.skipif(...)` decorator, so it runs at module IMPORT — before
   any fixture of any scope, session included, can intervene. It catches only
   `(urllib.error.URLError, OSError, TimeoutError)`, so raising on the GET probe
   would escape, error the whole module's collection, and take
   `test_phase_23_2_4_audit_log_clean_transitions` down with it. That test must
   keep passing with its trigger allowlist byte-unchanged (phase-36.21
   criterion, inherited by 86.3).

2. **The match must key on host AND port, never host alone.**
   `backend/tests/test_phase_76_9_2_max_bridge.py` runs its own
   `ThreadingHTTPServer` and POSTs to it on an EPHEMERAL port
   (`s.bind(("127.0.0.1", 0))` at :130). A host-level `127.0.0.1` block --
   which is what `pytest-socket --allow-hosts` would force -- breaks that
   legitimate test. This is the single easiest way to get this fix wrong.

WHY IMPORT TIME, AND WHY THE REPO ROOT
--------------------------------------
Import time, not an autouse fixture, because collection is what has to be
covered (constraint 1). The repo root, because `pytest.ini` lives there, so
rootdir is the repo root and this file is loaded for BOTH test trees --
including `tests/`, which has no conftest.py at all. That gives the property
phase-36.21 asked for: a NEW test module is protected by EXISTING, not by
remembering to opt in.

`backend/tests/conftest.py`'s phase-82.58 slack.com guard is deliberately left
byte-untouched. This file is imported first, so that wrapper simply chains on
top of this one; both policies apply.

KNOWN, BOUNDED GAPS -- stated rather than papered over
------------------------------------------------------
* `httpx` is NOT covered. httpx rides `httpcore`, not urllib3.
* Raw `socket` use is NOT covered.
* The FILESYSTEM channel is NOT covered: a test that calls a mutating
  `kill_switch` method in-process while `_AUDIT_PATH` still points at the live
  file writes directly, and no network guard can see that. That is a different
  channel with its own blast radius; it is queued as its own research-gated
  step. The one KNOWN in-process writer is step 86.1's `reset_peak` landmine,
  inert today only because `kill_switch_peak_reset_enabled` is False.
"""

from __future__ import annotations

import logging
import urllib.request
from urllib.parse import urlsplit

_log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Policy constants
# --------------------------------------------------------------------------

#: Verbs that can change server-side state. GET/HEAD/OPTIONS are always allowed
#: -- see constraint 1 in the module docstring.
_MUTATING_VERBS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

#: Loopback spellings that reach the operator's running backend.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
#: "0.0.0.0" added phase-86.6. It was MISSING, and that was a live hole,
#: not a tidiness issue: the backend runs `uvicorn --host 0.0.0.0`, so
#: `curl http://0.0.0.0:8000/api/health` answers 200 on this machine, and a
#: MUTATING PUT to that spelling passed straight through this guard while
#: the identical PUT to 127.0.0.1:8000 was refused. Found by the phase-86.6
#: Q/A because 86.6's own predicate listed the host and this one did not --
#: the two definitions disagreed, and the disagreement WAS the bug.

#: The live backend's port. Keyed explicitly so that an ephemeral-port stub on
#: the SAME host stays allowed (constraint 2).
_LIVE_BACKEND_PORT = 8000

_REAL_URLOPEN = urllib.request.urlopen


def _resolve_url(req) -> str:
    """A urlopen target is either a `Request` (has .full_url) or a bare str."""
    return getattr(req, "full_url", None) or str(req)


def _resolve_method(req, args: tuple, kwargs: dict) -> str:
    """Determine the HTTP verb the way urllib itself will.

    A `Request` knows its own method. A BARE URL STRING does not -- and
    `urlopen(url, data=b"...")` is a POST even though the target is a plain
    `str`. Missing that is exactly the class of gap that lets a mutating call
    slip past a verb check, so `data` is inspected explicitly.

    urlopen signature: urlopen(url, data=None, timeout=..., ...)
    """
    get_method = getattr(req, "get_method", None)
    if callable(get_method):
        try:
            return str(get_method()).upper()
        except (AttributeError, TypeError, ValueError) as exc:
            # A duck-typed request object must not break an unrelated test.
            # Fall through to the `data` inference below, which is the
            # conservative reading (data present => treated as a POST).
            _log.debug("phase-86.3: get_method() failed on %r: %s", req, exc)
    data = kwargs.get("data", args[0] if args else None)
    return "POST" if data is not None else "GET"


def _is_live_backend(url: str) -> bool:
    """True only for the live backend ORIGIN -- host AND port both matching."""
    try:
        parts = urlsplit(url)
        return parts.hostname in _LOOPBACK_HOSTS and parts.port == _LIVE_BACKEND_PORT
    except ValueError:
        # An unparseable port raises; fail OPEN rather than break unrelated
        # tests on a malformed URL. A malformed URL is not a live POST.
        return False


def _refusal(method: str, url: str) -> RuntimeError:
    return RuntimeError(
        "phase-86.3 test guard: refusing a MUTATING request from the test suite "
        f"to the live backend (method={method}, url={url!r}).\n"
        "A full backend/tests run must not pause, resume or otherwise mutate the "
        "operator's live paper-trading book -- measured 2026-08-09: 8 rows were "
        "appended to handoff/kill_switch_audit.jsonl and the live armed book was "
        "paused 4 times by a test run.\n"
        "Drive the endpoint IN-PROCESS instead (fastapi TestClient / "
        "httpx.ASGITransport) with kill_switch._AUDIT_PATH redirected to tmp_path "
        "-- see tests/api/test_pause_resume_timeout.py for the in-repo idiom. "
        "GETs to the live backend are still allowed."
    )


def _guarded_urlopen(req, *args, **kwargs):
    """Import-time replacement for `urllib.request.urlopen`.

    Raises BEFORE a connection is opened, so a blocked call never reaches the
    network at all.
    """
    url = _resolve_url(req)
    method = _resolve_method(req, args, kwargs)
    if method in _MUTATING_VERBS and _is_live_backend(url):
        raise _refusal(method, url)
    return _REAL_URLOPEN(req, *args, **kwargs)


urllib.request.urlopen = _guarded_urlopen


# --------------------------------------------------------------------------
# Second client family: urllib3 (covers `requests`; does NOT cover httpx).
# --------------------------------------------------------------------------
# stdlib urllib.request does NOT pass through urllib3, which is why the guard
# above is the load-bearing one for this defect -- the widely-cited urllib3
# conftest recipe would have missed it entirely. This block is defence in depth
# for any FUTURE test that reaches for `requests`.
try:  # pragma: no cover - exercised only when urllib3 is installed
    import urllib3.connectionpool as _urllib3_connectionpool

    _REAL_POOL_URLOPEN = _urllib3_connectionpool.HTTPConnectionPool.urlopen

    def _guarded_pool_urlopen(self, method, url, *args, **kwargs):
        if (
            str(method).upper() in _MUTATING_VERBS
            and getattr(self, "host", None) in _LOOPBACK_HOSTS
            and getattr(self, "port", None) == _LIVE_BACKEND_PORT
        ):
            raise _refusal(str(method).upper(), f"{self.host}:{self.port}{url}")
        return _REAL_POOL_URLOPEN(self, method, url, *args, **kwargs)

    _urllib3_connectionpool.HTTPConnectionPool.urlopen = _guarded_pool_urlopen
except (ImportError, AttributeError) as _exc:
    # urllib3 absent (ImportError) or its internals moved (AttributeError).
    # The stdlib urllib guard above is the load-bearing one for this defect and
    # stands regardless; this block is defence in depth for `requests`.
    _log.debug("phase-86.3: urllib3 guard not installed: %s", _exc)


# ==========================================================================
# phase-86.6 -- THE FILESYSTEM CHANNEL: from DETECTED to PREVENTED
# ==========================================================================
# The module docstring above declares this gap explicitly: "The FILESYSTEM
# channel is NOT covered ... it is queued as its own research-gated step."
# This is that step.
#
# WHY A `BaseException`, AND WHY THAT IS THE WHOLE POINT
# ------------------------------------------------------
# MEASURED in the project venv (pytest 9.0.3):
#     Failed MRO: ['Failed', 'OutcomeException', 'BaseException', 'object']
#     kill_switch shape: logger.warning swallowed: RuntimeError
#     kill_switch shape: logger.warning swallowed: PermissionError
# `kill_switch._append_audit` wraps its write in `except Exception:` and logs a
# warning. BOTH existing guards in this repo raise `RuntimeError`, which IS an
# `Exception`. So a filesystem refusal built to the established in-repo pattern
# would be SILENTLY ABSORBED: the write is blocked, the test goes GREEN, and
# nobody learns anything. `assert` fails the same way -- `AssertionError` is an
# `Exception`. The refusal below therefore derives from `BaseException`, which
# is the opposite of what this repo's own precedent would suggest.
#
# WHY `sys.addaudithook` (PEP 578)
# --------------------------------
# It is the only seam that sees `open`, `subprocess.Popen`, `socket.connect`
# and `urllib.Request` at once, and it fires BEFORE the operation. Measured
# overhead ~1.05x. Two honest limits, stated rather than sold around:
#   * PEP 578 says plainly it "is not sandboxing" -- this stops ACCIDENTS, not
#     a determined caller who could remove the predicate.
#   * Audit hooks "cannot be removed or replaced" once installed. That is why
#     the mode is read from the environment at CALL time, not baked in at
#     install time: an installed hook that could never be quietened would be a
#     worse trade than the defect.
#
# READS ARE ALWAYS ALLOWED. `test_phase_23_2_4_audit_log_clean_transitions`
# reads the LIVE journal deliberately and must keep doing so; criterion 4 of
# this step calls a blanket redirect an explicit failure, not a trade-off.
# --------------------------------------------------------------------------
import os as _os
import sys as _sys
from pathlib import Path as _Path


class LiveStateWriteRefused(BaseException):
    """A test tried to WRITE live operator state.

    Derives from `BaseException` DELIBERATELY -- see the note above. If this
    ever becomes an `Exception` subclass, `kill_switch._append_audit`'s
    `except Exception` will swallow it and the guard becomes decorative.
    """


#: The live-state tree. Everything the operator's running system reads or writes
#: lives under here, including UNTRACKED files -- `handoff/.autonomous_loop.lock`
#: is gitignored, which is exactly why a git-tracked digest sweep missed it.
_LIVE_STATE_ROOT = _Path(__file__).resolve().parent / "handoff"

#: BLOCKED set -- the kill-switch safety journal and its DERIVED archive dir.
#: Scoped deliberately, and the scope is MEASURED rather than guessed. Running
#: the full backend/tests/ tree with the whole `handoff/` tree blocking turned
#: 21 tests RED, and NONE of them was a kill-switch write: they write
#: `handoff/.autonomous_loop.lock` (the cycle lock), `handoff/.cycle_heartbeat.json`
#: and one probe file under `handoff/logs/`. Those are real live-state writes and
#: they are REPORTED below -- but blocking them is a behaviour change to 21
#: tests, which is its own step, not something to land at the end of a night
#: inside a step whose criteria are specifically about the kill-switch journal.
#: The archive dir is DERIVED from the journal path, mirroring
#: `kill_switch._audit_archive_dir()`, so a rotation cannot slip past.
_BLOCKED_PATHS = (
    _LIVE_STATE_ROOT / "kill_switch_audit.jsonl",
    _LIVE_STATE_ROOT / "audit",
)

#: Write intent, by open() mode string and by os.open() flags.
_WRITE_MODE_CHARS = frozenset("wax+")
_WRITE_FLAGS = (
    _os.O_WRONLY | _os.O_RDWR | _os.O_APPEND | _os.O_CREAT | _os.O_TRUNC
)

#: block (default) | report | off. Read at CALL time -- see the note above.
_GUARD_ENV = "PYFINAGENT_LIVE_STATE_GUARD"


def _guard_mode() -> str:
    return (_os.environ.get(_GUARD_ENV) or "block").strip().lower()


def _is_write_intent(mode, flags) -> bool:
    if isinstance(mode, str) and (set(mode) & _WRITE_MODE_CHARS):
        return True
    if isinstance(flags, int) and (flags & _WRITE_FLAGS):
        return True
    return False


def _classify(path):
    """Return "block" | "report" | None for a write-intent path.

    Two tiers on purpose. `block` is the kill-switch journal, which is what
    this step's criteria are about and where a stray write silently loosens a
    safety limit. `report` is the rest of the live-state tree: real writes,
    surfaced loudly, but turning them into failures is a 21-test behaviour
    change that belongs to its own step rather than to this one.
    """
    try:
        p = _Path(_os.fsdecode(path)).resolve()
    except (TypeError, ValueError, OSError):
        return None
    for blocked in _BLOCKED_PATHS:
        if p == blocked or blocked in p.parents:
            return "block"
    try:
        p.relative_to(_LIVE_STATE_ROOT)
        return "report"
    except ValueError:
        return None


def _current_test() -> str:
    return _os.environ.get("PYTEST_CURRENT_TEST", "(outside a test)")


def _live_state_audit_hook(event: str, args) -> None:
    if event != "open":
        return
    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None
    if not _is_write_intent(mode, flags):
        return                      # reads always pass
    path = args[0] if args else None
    tier = _classify(path)
    if tier is None:
        return
    m = _guard_mode()
    if m == "off":
        return
    if tier == "report":
        # DELIBERATELY A NO-OP, and this is a correction rather than laziness.
        # The first revision wrote a line to stderr here. MEASURED: a full
        # backend/tests run emitted ZERO visible lines, because pytest captures
        # stderr per test and discards it for PASSING tests -- which is every
        # test that merely writes live state without failing. A "report" tier
        # that reports nothing is the same silent-failure class this step
        # exists to close, so it is removed rather than left looking useful.
        #
        # The finding it was meant to carry is MEASURED and RECORDED instead:
        # blocking the whole `handoff/` tree turns +7 tests RED (14 -> 21
        # against a 14-failure pre-existing baseline), writing
        # `handoff/.autonomous_loop.lock`, `handoff/.cycle_heartbeat.json` and
        # one probe file under `handoff/logs/`. Closing those is a behaviour
        # change to real tests and is filed as its own step.
        return
    msg = (
        f"phase-86.6: REFUSED a WRITE to live operator state.\n"
        f"  path : {path}\n"
        f"  test : {_current_test()}\n"
        f"  mode : {mode!r} flags={flags!r}\n"
        "Redirect the module-level path to tmp_path (e.g. "
        "monkeypatch.setattr(ks, '_AUDIT_PATH', tmp_path / '...')), or set "
        f"{_GUARD_ENV}=off for a run that genuinely must touch live state.\n"
        "This refusal derives from BaseException on purpose: an Exception "
        "would be swallowed by kill_switch._append_audit's `except Exception`."
    )
    if m == "report":
        _sys.stderr.write("[86.6 REPORT-ONLY] " + msg + "\n")
        return
    raise LiveStateWriteRefused(msg)


_sys.addaudithook(_live_state_audit_hook)
