"""phase-86.6 Part B -- the SUBPROCESS channel.

A guard installed at conftest import time exists only in the PYTEST PROCESS.
A test that shells out gets a CHILD, and a child loads no conftest, so
phase-86.3's HTTP guard is structurally absent there. Measured at source:
`test_phase_4000_2_cc_rail_smoke.py` drives `scripts/qa/smoke_cc_rail_e2e.py`
via `subprocess.run`, and that script MUTATES -- `PUT /api/settings/` and
`POST /api/analysis/`. Its `--backend-url` used to default to
`http://localhost:8000`, the operator's live backend.

The risk was LATENT, not active: all twelve existing call sites pass an explicit
stub URL. **The defect is the future thirteenth**, which is why criterion 7
requires the seam to reject a NEW unguarded call site rather than a census of
the current ones -- a census is regressed by someone simply writing another
test.

WHY THE SEAM IS IN THE SCRIPT AND NOT AT THE CALL SITES
------------------------------------------------------
A convention ("always pass --backend-url") cannot fail a build. An argument with
no default can. The guard therefore sits at the one place every invocation must
pass through, and this module proves it by ADDING the offending call site here
and asserting it is rejected.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "qa" / "smoke_cc_rail_e2e.py"

sys.path.insert(0, str(REPO / "scripts" / "qa"))
from live_backend_origin import is_live_backend  # noqa: E402


def run_smoke(*argv: str) -> subprocess.CompletedProcess:
    """Deliberately mirrors test_phase_4000_2_cc_rail_smoke.run_smoke -- this
    module must exercise the SAME invocation shape a real caller uses."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *argv],
        capture_output=True, text=True, timeout=120, cwd=str(REPO),
    )


# ── criterion 7: a NEW unguarded call site must be REJECTED ────────────────


def test_a_new_call_site_that_OMITS_backend_url_is_REJECTED():
    """THIS IS THE CRITERION-7 PROOF.

    The call below is exactly the mistake a future test would make. Before this
    step it would have run happily against `http://localhost:8000` and PUT a
    setting on the operator's live backend. It must now fail loudly.
    """
    proc = run_smoke("--dry")                    # <-- the offending call site

    assert proc.returncode != 0, (
        "a run_smoke invocation with NO --backend-url was ACCEPTED; the "
        "subprocess channel is still open"
    )
    assert "--backend-url is REQUIRED" in proc.stderr, (
        f"rejected, but not for the right reason: {proc.stderr[:400]!r}"
    )
    # The message must name the consequence, not just the missing flag --
    # a future reader has to understand why it is required.
    assert "mutates" in proc.stderr.lower()


def test_explicitly_targeting_the_LIVE_backend_is_REJECTED_without_optin():
    """Requiring the flag closes OMISSION. It does not close a call site that
    passes the live URL explicitly -- which is the same mistake with more
    typing. That is the class, so it is guarded too."""
    proc = run_smoke("--dry", "--backend-url", "http://localhost:8000")

    assert proc.returncode != 0, (
        "an explicit live-backend target was accepted without --allow-live-backend"
    )
    assert "refusing to target the LIVE backend" in proc.stderr


@pytest.mark.parametrize(
    "url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"]
)
def test_every_loopback_spelling_of_the_live_backend_is_refused(url):
    """One spelling guarded is an instance; the set is the class."""
    proc = run_smoke("--dry", "--backend-url", url)
    assert proc.returncode != 0, f"{url} was accepted"
    assert "refusing to target the LIVE backend" in proc.stderr


# ── positive controls: the guard must not be a blanket refusal ─────────────


def test_the_optin_IS_honoured_so_a_real_window_remains_possible():
    """Without this, the refusal tests above would pass against a script that
    rejects everything -- and phase-4000.3 needs a real live window."""
    proc = run_smoke("--dry", "--backend-url", "http://localhost:8000",
                     "--allow-live-backend")
    assert "refusing to target the LIVE backend" not in proc.stderr, (
        "--allow-live-backend did not lift the refusal"
    )
    assert "--backend-url is REQUIRED" not in proc.stderr


def test_an_ephemeral_stub_url_is_NOT_treated_as_live():
    """The twelve existing 4000.2 call sites stand up stubs on 127.0.0.1:0, an
    ephemeral high port. Keying the guard on the HOST would have broken every
    one of them; it keys on the PORT for exactly this reason."""
    proc = run_smoke("--dry", "--backend-url", "http://127.0.0.1:59999")
    assert "refusing to target the LIVE backend" not in proc.stderr
    assert "--backend-url is REQUIRED" not in proc.stderr


# ── the predicate itself, and the drift alarm ──────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8000", True),
        ("http://127.0.0.1:8000", True),
        ("http://0.0.0.0:8000", True),
        ("http://localhost:8000/api/settings/", True),
        ("http://127.0.0.1:59999", False),      # ephemeral stub
        ("http://127.0.0.1:3000", False),       # the frontend
        ("http://localhost", False),            # port 80, not the backend
        ("https://example.com:8000", False),    # right port, wrong host
        ("", False),
        ("not a url", False),
    ],
)
def test_is_live_backend_table(url, expected):
    assert is_live_backend(url) is expected, url


def test_the_two_live_origin_predicates_AGREE():
    """phase-86.22's lesson, applied here before it can bite.

    `conftest.py` carries its own copy of this predicate (phase-86.3, already
    graded; rewiring it is a risk this step does not need to take). Two
    definitions of the same fact are how a vocabulary drifts apart, so they are
    compared here across a table. A disagreement fails THIS test rather than
    silently opening one of the two channels.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_conftest_for_drift_check", REPO / "conftest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    urls = [
        "http://localhost:8000", "http://127.0.0.1:8000",
        "http://localhost:8000/api/settings/", "http://127.0.0.1:59999",
        "http://127.0.0.1:3000", "http://localhost", "https://example.com:8000",
    ]
    disagreements = [
        (u, mod._is_live_backend(u), is_live_backend(u))
        for u in urls
        if mod._is_live_backend(u) != is_live_backend(u)
    ]
    assert not disagreements, (
        "conftest's live-origin predicate and scripts/qa/live_backend_origin.py "
        f"DISAGREE -- one of the two channels is mis-guarded: {disagreements}"
    )
