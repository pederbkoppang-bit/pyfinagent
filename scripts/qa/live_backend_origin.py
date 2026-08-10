"""phase-86.6 Part B -- is this URL the operator's LIVE backend?

WHY THIS FILE EXISTS SEPARATELY FROM conftest.py
------------------------------------------------
phase-86.3 installed a guard in the repo-root `conftest.py` that refuses
MUTATING HTTP verbs aimed at the live backend. That guard lives in the PYTEST
PROCESS. A test that shells out gets a CHILD process, and a child process loads
no conftest -- so the guard is structurally absent there. Measured at source:
`backend/tests/test_phase_4000_2_cc_rail_smoke.py` drives
`scripts/qa/smoke_cc_rail_e2e.py` via `subprocess.run`, and that script MUTATES
(`PUT /api/settings/`, `POST /api/analysis/`).

The child therefore needs its own guard, and the predicate it guards on must not
be a SECOND, drifting definition of "the live backend". That is precisely the
defect phase-86.22 spent two cycles removing from the recommendation vocabulary:
two normalisers that disagree are one bug wearing two hats.

So the constants live here, once. The script imports them. `conftest.py` keeps
its own copy for now -- rewiring a guard that has already been graded is a risk
this step does not need to take -- and
`test_phase_86_6_subprocess_channel.py::test_the_two_live_origin_predicates_agree`
compares the two across a table of URLs, so a drift becomes a test failure
rather than a silent divergence. That is a deliberate trade, stated rather than
hidden: one authority for the NEW code, an alarm on the OLD one.

WHAT COUNTS AS LIVE
-------------------
A loopback host on the backend's port. The port is the discriminator on
purpose: the 4000.2 tests stand up stub servers on `127.0.0.1:0` (an ephemeral
high port), and those must keep working. Keying on the host alone would break
every one of them; keying on the port is what separates "the operator's running
book" from "a throwaway server this test just created".
"""
from __future__ import annotations

from urllib.parse import urlsplit

#: Loopback spellings that reach the operator's running backend.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

#: The live backend's port (uvicorn --port 8000; see CLAUDE.md Quick Start).
LIVE_BACKEND_PORT = 8000


def is_live_backend(url: str) -> bool:
    """True if `url` points at the operator's running backend.

    Deliberately total: an unparseable URL is NOT live (False) rather than an
    exception, because this predicate is called on a guard path and a guard
    that raises on junk input turns a safety check into an outage.
    """
    try:
        parts = urlsplit(str(url))
    except (ValueError, AttributeError):
        return False
    host = (parts.hostname or "").lower()
    if host not in LOOPBACK_HOSTS:
        return False
    port = parts.port
    if port is None:
        # No explicit port: http -> 80, https -> 443. Neither is the backend.
        return False
    return port == LIVE_BACKEND_PORT
