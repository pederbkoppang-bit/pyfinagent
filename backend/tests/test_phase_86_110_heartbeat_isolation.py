"""phase-86.110 -- a passing unit test wrote into a real, git-tracked file.

`cycle_health` exposes TWO module-level path constants and its public
`record_cycle_end` / `record_cycle_start` write BOTH. Isolating one is
therefore silently partial, and that is exactly what
`test_phase_66_1_rail_guard.py` did: it monkeypatched `_HISTORY_PATH` to a
`tmp_path`, passed, and wrote `cycle_id="c2"` into the real
`handoff/.cycle_heartbeat.json` the dashboard reads as a liveness signal.

The tests here are deliberately of three different kinds, because no one kind
covers this:

  - **behavioural** -- run the previously-leaking tests in a subprocess against
    a snapshot of the real file and assert it did not move;
  - **structural** -- the reachability sweep, whose own positive control is
    asserted (a scanner that cannot report a leak is worse than none);
  - **the runtime guard itself** -- prove the conftest fixture fires AND
    repairs, since a guard that only reports leaves most of the damage.

Every test that touches the real heartbeat snapshots and restores it, so this
file cannot become the thing it is testing for.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
HEARTBEAT = REPO / "handoff" / ".cycle_heartbeat.json"
HISTORY = REPO / "handoff" / "cycle_history.jsonl"
RAIL_GUARD = "backend/tests/test_phase_66_1_rail_guard.py"
SWEEP = "scripts/qa/heartbeat_leak_sweep_86_110.py"
PYBIN = str(REPO / ".venv" / "bin" / "python")


def _sha(p: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


# ── 1. the leak is gone, asserted by RUNNING the tests that caused it ──


def test_the_previously_leaking_tests_no_longer_touch_the_real_heartbeat():
    """Criterion 3's post-fix half, driven in a SUBPROCESS.

    A subprocess is required, not stylistic: the conftest guard would repair
    the file in-process and mask the very thing being measured.
    """
    before = HEARTBEAT.read_bytes()
    try:
        r = subprocess.run(
            [PYBIN, "-m", "pytest", RAIL_GUARD, "-q", "-p", "no:cacheprovider"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stdout[-3000:]
        assert HEARTBEAT.read_bytes() == before, (
            "running the rail-guard suite still mutates the real heartbeat"
        )
    finally:
        HEARTBEAT.write_bytes(before)


def test_both_leaking_sites_isolate_the_heartbeat_constant():
    """Criterion 2, checked as an executable statement rather than a substring.

    A `"_HEARTBEAT_PATH" in src` check would be satisfied by the explanatory
    comment the fix itself added -- the trap this step hit while writing the
    sweep. Uses the sweep's own AST helper so there is one definition of
    "isolates", not two.
    """
    sys.path.insert(0, str(REPO / "scripts" / "qa"))
    try:
        from heartbeat_leak_sweep_86_110 import _isolates
    finally:
        sys.path.pop(0)

    src = (REPO / RAIL_GUARD).read_text(encoding="utf-8")
    assert _isolates(src, "_HEARTBEAT_PATH"), "the heartbeat constant is not isolated"
    assert _isolates(src, "_HISTORY_PATH"), "the history constant is not isolated"
    # Anti-vacuity: the helper must be able to say NO.
    assert not _isolates("# monkeypatch.setattr(ch, '_HEARTBEAT_PATH', x)", "_HEARTBEAT_PATH")
    assert not _isolates('"""_HEARTBEAT_PATH mentioned in a docstring"""', "_HEARTBEAT_PATH")


# ── 2. the sweep, and its positive control ─────────────────────────


def test_the_reachability_sweep_is_clean():
    r = subprocess.run([PYBIN, SWEEP], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert "LEAKING: 0" in r.stdout, r.stdout


def test_the_sweep_can_actually_detect_a_leak(tmp_path):
    """POSITIVE CONTROL. The sweep reporting zero is only meaningful if it can
    report non-zero -- and the first version of it could NOT: its substring
    check was satisfied by the fix's own comments, so it stayed green with the
    fix reverted. Reverting the fix must now flip it to exit 1.
    """
    target = REPO / RAIL_GUARD
    orig = target.read_bytes()
    hb = HEARTBEAT.read_bytes()
    try:
        stripped = orig.decode().replace(
            '    monkeypatch.setattr(ch, "_HEARTBEAT_PATH", tmp_path / ".cycle_heartbeat.json")\n',
            "", 2,
        )
        assert stripped != orig.decode(), "the fix lines were not found to remove"
        target.write_bytes(stripped.encode())
        r = subprocess.run([PYBIN, SWEEP], cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 1, f"the sweep did NOT detect the reverted fix:\n{r.stdout}"
        assert "LEAKING: 1" in r.stdout, r.stdout
        assert "test_phase_66_1_rail_guard.py" in r.stdout
    finally:
        target.write_bytes(orig)
        HEARTBEAT.write_bytes(hb)
        assert _sha(target) == hashlib.sha256(orig).hexdigest()


def test_the_sweep_treats_the_production_writer_as_legitimate():
    """`autonomous_loop.py` writes the real heartbeat on purpose. A sweep that
    called that a leak would be the mirror of the naive survey's error."""
    r = subprocess.run([PYBIN, SWEEP], cwd=REPO, capture_output=True, text=True)
    assert "LEGITIMATE PRODUCTION WRITERS: 1" in r.stdout, r.stdout
    assert "backend/services/autonomous_loop.py" in r.stdout


# ── 3. the runtime guard: it must FIRE and it must REPAIR ──────────


def test_the_conftest_guard_fires_and_repairs(tmp_path):
    """Drives the guard by running a throwaway test that writes the real file.

    Subprocess again: an in-process write would be caught by this very test's
    own guard invocation and confuse the measurement.
    """
    probe = tmp_path / "test_probe_leak.py"
    probe.write_text(
        "import pathlib\n"
        "def test_writes_a_tracked_file():\n"
        f"    p = pathlib.Path({str(HEARTBEAT)!r})\n"
        '    p.write_text(\'{"cycle_id": "PROBE", "event": "end", "updated_at": "x"}\')\n'
        "    assert True\n",
        encoding="utf-8",
    )
    before = HEARTBEAT.read_bytes()
    try:
        r = subprocess.run(
            [PYBIN, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider",
             "-p", "backend.tests.conftest"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert r.returncode != 0, f"the guard did not fire:\n{r.stdout[-3000:]}"
        assert "phase-86.110 test guard" in r.stdout, r.stdout[-3000:]
        # REPAIRED, not merely reported.
        assert HEARTBEAT.read_bytes() == before, (
            "the guard reported the leak but left the file dirty"
        )
    finally:
        HEARTBEAT.write_bytes(before)


def test_the_guard_does_not_fire_on_a_clean_test(tmp_path):
    """Anti-vacuity for the guard: it must PASS a test that writes nothing,
    otherwise the previous test proves only that pytest can fail."""
    probe = tmp_path / "test_probe_clean.py"
    probe.write_text("def test_touches_nothing():\n    assert True\n", encoding="utf-8")
    r = subprocess.run(
        [PYBIN, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider",
         "-p", "backend.tests.conftest"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout[-3000:]


# ── 4. criterion 5 -- the file's content is DERIVED from the ledger ──


def test_the_heartbeat_value_exists_in_the_real_cycle_ledger():
    """The disposition, asserted rather than described: whatever the heartbeat
    holds must be a cycle the ledger actually recorded. `c1`/`c2` are fixture
    values with zero ledger rows -- that is what made the polluted state a lie
    rather than merely stale."""
    hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    cid = hb.get("cycle_id")
    assert cid, hb
    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = sum(1 for r in rows if r.get("cycle_id") == cid)
    assert n > 0, (
        f"heartbeat cycle_id={cid!r} appears in 0 of {len(rows)} ledger rows -- "
        "the dashboard would be reporting liveness off a value no cycle produced"
    )
    # And the fixture values must NOT be there.
    for fake in ("c1", "c2"):
        assert sum(1 for r in rows if r.get("cycle_id") == fake) == 0


@pytest.mark.parametrize("const", ["_HISTORY_PATH", "_HEARTBEAT_PATH"])
def test_both_constants_still_exist_and_differ(const):
    """If a refactor collapses the two constants this suite's premise changes,
    and these tests should be re-read rather than silently kept green."""
    import backend.services.cycle_health as ch

    assert hasattr(ch, const)
    assert ch._HISTORY_PATH != ch._HEARTBEAT_PATH


# ── 5. the guard must not DESTROY a legitimate concurrent write ────
#
# Raised by the cycle-1 evaluator and it is the sharpest finding in this step:
# pyfinagent is local-only, the live autonomous_loop writes these exact files
# from this machine, and a full suite run takes ~8 minutes. A guard that
# blindly restored its snapshot would DELETE a real cycle row and then blame an
# innocent test -- strictly worse than the leak it was built to stop.


def test_a_real_APPEND_to_the_cycle_ledger_is_not_reverted():
    """`cycle_history.jsonl` is append-only. A concurrent real append leaves
    the snapshot intact as a prefix, and must be left alone."""
    from backend.tests.conftest import _is_legitimate_concurrent_write as legit

    before = b'{"cycle_id": "aaa", "status": "completed"}\n'
    appended = before + b'{"cycle_id": "bbb", "status": "completed"}\n'
    assert legit("handoff/cycle_history.jsonl", before, appended) is True

    # A REWRITE is not an append and must still be caught.
    rewritten = b'{"cycle_id": "zzz"}\n'
    assert legit("handoff/cycle_history.jsonl", before, rewritten) is False
    # Truncation is the dangerous case and must be caught too.
    assert legit("handoff/cycle_history.jsonl", before, b"") is False
    # A no-growth "append" is not an append.
    assert legit("handoff/cycle_history.jsonl", before, before) is False

    # THE CASE THAT MAKES THE `startswith` HALF LOAD-BEARING: a rewrite that is
    # LONGER than the snapshot. Without this, all three cases above are
    # length-only discriminable (rewrite 20B and truncation 0B are both SHORTER
    # than the 42B snapshot, and no-growth is equal), so dropping `startswith`
    # entirely leaves the suite green -- a Q/A executed exactly that mutant and
    # it SURVIVED 13/13. A guard whose only tested half is the redundant one is
    # not a guard.
    longer_rewrite = (b'{"cycle_id": "z"}\n{"cycle_id": "y"}\n'
                      b'{"cycle_id": "x"}\n{"cycle_id": "w"}\n')
    assert len(longer_rewrite) > len(before), "the fixture must be LONGER to bite"
    assert not longer_rewrite.startswith(before)
    assert legit("handoff/cycle_history.jsonl", before, longer_rewrite) is False, (
        "a LONGER rewrite was accepted as an append -- the prefix check is dead, "
        "and a wholesale ledger rewrite would be silently tolerated"
    )


def test_a_real_cycle_heartbeat_write_is_not_reverted_but_a_fixture_value_is():
    """The heartbeat is last-writer-wins, so "did it change" cannot decide.
    What decides is whether the cycle_id EXISTS in the real ledger -- the same
    property that made `c2` a lie rather than merely stale."""
    from backend.tests.conftest import _is_legitimate_concurrent_write as legit

    rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_id = next(r["cycle_id"] for r in rows if r.get("cycle_id"))

    before = b'{"cycle_id": "old", "event": "end", "updated_at": "x"}'
    real = json.dumps({"cycle_id": real_id, "event": "end", "updated_at": "y"}).encode()
    fake = b'{"cycle_id": "c2", "event": "end", "updated_at": "y"}'

    assert legit("handoff/.cycle_heartbeat.json", before, real) is True, (
        "a write naming a REAL cycle was treated as a leak -- the guard would "
        "revert live data"
    )
    assert legit("handoff/.cycle_heartbeat.json", before, fake) is False, (
        "a fixture cycle_id was treated as legitimate -- the guard is now blind"
    )
    # Unparseable content is not evidence of legitimacy.
    assert legit("handoff/.cycle_heartbeat.json", before, b"not json") is False


def test_the_guard_STILL_catches_a_test_leak_after_the_concurrency_fix(tmp_path):
    """Anti-regression for the fix above: making the guard tolerant must not
    make it blind. A probe writing a fixture cycle_id must still fail."""
    probe = tmp_path / "test_probe_leak2.py"
    probe.write_text(
        "import pathlib\n"
        "def test_writes_a_fixture_cycle_id():\n"
        f"    pathlib.Path({str(HEARTBEAT)!r}).write_text("
        "'{\"cycle_id\": \"c2\", \"event\": \"end\", \"updated_at\": \"x\"}')\n"
        "    assert True\n",
        encoding="utf-8",
    )
    before = HEARTBEAT.read_bytes()
    try:
        r = subprocess.run(
            [PYBIN, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider",
             "-p", "backend.tests.conftest"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert r.returncode != 0, f"the guard went blind:\n{r.stdout[-3000:]}"
        assert "phase-86.110 test guard" in r.stdout
        assert HEARTBEAT.read_bytes() == before
    finally:
        HEARTBEAT.write_bytes(before)
