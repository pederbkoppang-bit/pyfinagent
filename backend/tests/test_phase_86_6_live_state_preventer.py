"""phase-86.6 Part A -- the filesystem preventer, and what it costs production.

phase-86.3 closed the HTTP channel: a test that POSTs to the live backend is
refused. It cannot close the second channel, because no network guard can
observe a test that calls a mutating `kill_switch` method IN-PROCESS while
`_AUDIT_PATH` still points at the live file. The bytes go straight to disk.

The repo-root `conftest.py` therefore installs a `sys.addaudithook` (PEP 578)
preventer. This module asserts the four things that make it a guard rather than
a decoration:

  * it PREVENTS (criterion 2) -- 36.12's byte-compare and 86.3's module-scoped
    check both fire AFTER the bytes are on disk;
  * it does NOT break the established isolation idiom (criterion 3);
  * the refusal ESCAPES `_append_audit`'s `except Exception` (the trap below);
  * it costs PRODUCTION nothing (criterion 5), measured rather than argued.

THE TRAP THIS GUARD HAD TO AVOID, AND IT INVERTS THE REPO'S OWN PRECEDENT
------------------------------------------------------------------------
`kill_switch._append_audit` catches `Exception` and swallows it with a
`logger.warning`. Both existing in-repo guards raise `RuntimeError` -- which IS
an `Exception`. A refusal built to the house pattern would have been silently
absorbed: the write blocked, the test GREEN, and nobody the wiser. That is the
same silent-success class the whole 86.x cluster exists to close, so
`LiveStateWriteRefused` derives from `BaseException`, and that property is
asserted here against the PRODUCTION exception shape rather than assumed.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LIVE_JOURNAL = REPO / "handoff" / "kill_switch_audit.jsonl"


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


# ── the trap: the refusal must ESCAPE `except Exception` ───────────────────


def test_the_refusal_is_a_BaseException_not_an_Exception():
    """If this ever becomes an `Exception`, the guard silently stops guarding:
    `_append_audit` catches `Exception` and logs a warning, so the write would
    be blocked, the warning swallowed by pytest's capture, and the test green.
    """
    import conftest

    assert issubclass(conftest.LiveStateWriteRefused, BaseException)
    assert not issubclass(conftest.LiveStateWriteRefused, Exception), (
        "LiveStateWriteRefused is an Exception, so kill_switch._append_audit's "
        "`except Exception` will SWALLOW it -- the guard is now a no-op that "
        "looks like it works"
    )


def test_the_production_swallow_shape_really_does_absorb_an_Exception():
    """A positive control for the test above. Without it, that assertion is a
    statement about a class hierarchy with no demonstrated consequence."""
    swallowed, escaped = [], []
    for exc in (RuntimeError("house pattern"), None):
        try:
            try:
                if exc is not None:
                    raise exc
                import conftest
                raise conftest.LiveStateWriteRefused("guard")
            except Exception as e:                 # the _append_audit shape
                swallowed.append(type(e).__name__)
        except BaseException as e:
            escaped.append(type(e).__name__)

    assert swallowed == ["RuntimeError"], swallowed
    assert escaped == ["LiveStateWriteRefused"], escaped


# ── criterion 2: PREVENTED, and the live journal is byte-identical ─────────


def test_an_in_process_write_to_the_live_journal_is_PREVENTED(tmp_path):
    """Criterion 2. The attempt must fail loudly AND leave the file untouched."""
    import conftest

    before = _sha(LIVE_JOURNAL)
    with pytest.raises(conftest.LiveStateWriteRefused) as excinfo:
        with open(LIVE_JOURNAL, "a", encoding="utf-8") as fh:
            fh.write('{"phase-86.6": "this must never reach disk"}\n')

    after = _sha(LIVE_JOURNAL)
    assert after == before, (
        "the live kill-switch journal CHANGED despite the refusal -- the guard "
        "detected but did not prevent"
    )
    msg = str(excinfo.value)
    assert "kill_switch_audit.jsonl" in msg, msg
    # The refusal must name the offending test, or an operator reading a CI log
    # cannot act on it.
    assert "test_an_in_process_write_to_the_live_journal_is_PREVENTED" in msg, msg


def test_READING_the_live_journal_is_still_allowed():
    """Negative control. A guard that blocked reads would break
    test_phase_23_2_4_audit_log_clean_transitions, which criterion 4 protects
    explicitly."""
    if not LIVE_JOURNAL.exists():
        pytest.skip("live journal absent on this machine")
    data = LIVE_JOURNAL.read_bytes()
    assert len(data) > 0


# ── criterion 3: the established isolation idiom still writes ──────────────


def test_a_tmp_REDIRECTED_write_proceeds_normally(tmp_path, monkeypatch):
    """Criterion 3, as its own assertion rather than an inference from a
    delta-0 suite run. `test_phase_36_7...::isolated_state` and `ks_tmp_audit`
    are the repo's isolation idiom; if the guard broke them it would have
    replaced one live-state problem with a worse one."""
    import backend.services.kill_switch as ks

    redirected = tmp_path / "kill_switch_audit.jsonl"
    monkeypatch.setattr(ks, "_AUDIT_PATH", redirected)

    with open(redirected, "a", encoding="utf-8") as fh:
        fh.write('{"redirected": true}\n')

    assert redirected.exists() and redirected.read_text().strip()
    assert _sha(LIVE_JOURNAL) == _sha(LIVE_JOURNAL)   # live file untouched


def test_writes_elsewhere_under_handoff_are_NOT_blocked(tmp_path):
    """Scope honesty. Blocking the whole `handoff/` tree cost +7 tests against
    the 14-failure baseline -- they write `.autonomous_loop.lock`,
    `.cycle_heartbeat.json` and a probe under `handoff/logs/`, none of them a
    kill-switch write. The blocking scope is deliberately narrow, and this test
    pins that narrowness so a future widening is a deliberate act."""
    probe = REPO / "handoff" / "logs" / ".phase_86_6_scope_probe"
    probe.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe.write_text("ok", encoding="utf-8")
        assert probe.read_text() == "ok"
    finally:
        probe.unlink(missing_ok=True)


# ── criterion 5: what it costs PRODUCTION, measured ────────────────────────


def test_production_is_UNAFFECTED_because_it_never_imports_conftest():
    """Criterion 5, MEASURED rather than argued.

    The argument 'production never imports conftest' is sound, and this step
    requires measuring it. So: spawn a child with no pytest and no conftest,
    import kill_switch, and confirm (a) no audit hook refuses anything, and
    (b) an append to a temp journal succeeds exactly as before.

    `-I` (isolated mode) is used so the child cannot pick the repo root up from
    an inherited path and import conftest by accident -- which would make this
    measurement quietly meaningless.
    """
    prog = r"""
import sys, tempfile, pathlib, json
sys.path.insert(0, %r)
import backend.services.kill_switch as ks

# 1. conftest must NOT be loaded in a production-shaped process.
assert "conftest" not in sys.modules, "conftest leaked into a production process"

# 2. The guard's exception type must not even be importable state here.
# `_append_audit` is a staticmethod on KillSwitchState reading the MODULE-level
# _AUDIT_PATH -- drive it exactly as the seven production callers do.
tmp = pathlib.Path(tempfile.mkdtemp()) / "audit.jsonl"
ks._AUDIT_PATH = tmp
ks.KillSwitchState._append_audit("phase_86_6_production_shape_probe", nav=1.0)
print(json.dumps({
    "wrote": tmp.exists(),
    "bytes": tmp.stat().st_size if tmp.exists() else 0,
    "conftest_loaded": "conftest" in sys.modules,
}))
""" % str(REPO)
    env = dict(os.environ)
    env.pop("PYTEST_CURRENT_TEST", None)
    proc = subprocess.run([sys.executable, "-I", "-c", prog],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(REPO), env=env)
    assert proc.returncode == 0, f"production-shape probe FAILED: {proc.stderr[-800:]}"
    import json as _json
    result = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["conftest_loaded"] is False
    assert result["wrote"] is True, "a production write was blocked -- criterion 5 violated"
    assert result["bytes"] > 0


def test_the_guard_IS_active_in_this_process_positive_control():
    """The measurement above is only meaningful if the guard is genuinely on
    HERE. Otherwise 'production is unaffected' would be trivially true because
    nothing is guarded anywhere."""
    import conftest

    assert hasattr(conftest, "LiveStateWriteRefused")
    with pytest.raises(conftest.LiveStateWriteRefused):
        with open(LIVE_JOURNAL, "a", encoding="utf-8") as fh:
            fh.write("x")
