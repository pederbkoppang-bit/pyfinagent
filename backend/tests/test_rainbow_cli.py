"""phase-12.2 tests for promote.py + rollback.py Rainbow CLIs.

All tests monkeypatch subprocess.run so kubectl is never invoked.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Scripts live outside the importable backend/ tree; load via importlib.
_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _ROOT / "scripts" / "deploy" / "rainbow"


def _load_cli(name: str):
    spec = importlib.util.spec_from_file_location(
        f"rainbow_{name}", _SCRIPTS_DIR / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


promote = _load_cli("promote")
rollback = _load_cli("rollback")


# ---------- build_patch_json ----------


def test_promote_build_patch_json_shape():
    j = promote.build_patch_json("green")
    d = json.loads(j)
    assert d == {"spec": {"selector": {"color": "green"}}}


def test_rollback_build_patch_json_shape():
    j = rollback.build_patch_json("blue")
    d = json.loads(j)
    assert d == {"spec": {"selector": {"color": "blue"}}}


# ---------- build_kubectl_cmd ----------


def test_build_kubectl_cmd_has_service_and_patch():
    cmd = promote.build_kubectl_cmd("pyfinagent-backend", '{"x":1}')
    assert cmd[0] == "kubectl"
    assert cmd[1] == "patch"
    assert "pyfinagent-backend" in cmd
    assert '{"x":1}' in cmd


# ---------- promote dry-run ----------


def test_promote_dry_run_exits_zero(capsys):
    rc = promote.main(["--dry-run", "--to", "green"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "color=green" in out or '"color":"green"' in out or '"color": "green"' in out


def test_promote_dry_run_does_not_invoke_kubectl():
    # If --dry-run code path called subprocess.run, the test would raise
    # an uncaught error (since real kubectl isn't on CI path). If it
    # returns 0 cleanly, it confirmed the dry-run short-circuited.
    with patch("subprocess.run") as m:
        rc = promote.main(["--dry-run", "--to", "green"])
    assert rc == 0
    m.assert_not_called()


# ---------- promote live ----------


def test_promote_live_calls_kubectl_with_right_args():
    fake = MagicMock(returncode=0, stdout="service/pyfinagent-backend patched", stderr="")
    with patch("subprocess.run", return_value=fake) as m:
        rc = promote.main(["--to", "green"])
    assert rc == 0
    args, kwargs = m.call_args
    cmd = args[0]
    assert cmd[0:3] == ["kubectl", "patch", "service"]
    assert "pyfinagent-backend" in cmd


def test_promote_invalid_color_returns_2(capsys):
    rc = promote.main(["--to", "bad color!"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "invalid" in err.lower()


# ---------- rollback ----------


def test_rollback_dry_run_without_cluster_defaults_to_blue(capsys):
    # No cluster → read_current_color returns None → fall back to blue.
    with patch("subprocess.run", side_effect=FileNotFoundError):
        rc = rollback.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "blue" in out
    assert "DRY-RUN" in out


def test_rollback_explicit_to_overrides_detection(capsys):
    rc = rollback.main(["--dry-run", "--to", "indigo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "indigo" in out


def test_rollback_live_reads_current_then_patches():
    # First subprocess call: kubectl get service ... → stdout "green"
    # Second subprocess call: kubectl patch service ... → returncode 0
    call_log = []

    def fake_run(cmd, *a, **kw):
        call_log.append(cmd)
        if cmd[0:3] == ["kubectl", "get", "service"]:
            return MagicMock(returncode=0, stdout="green", stderr="")
        if cmd[0:3] == ["kubectl", "patch", "service"]:
            return MagicMock(returncode=0, stdout="service patched", stderr="")
        raise AssertionError(f"unexpected cmd: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        rc = rollback.main([])
    assert rc == 0
    # Two kubectl calls: one get, one patch.
    assert len(call_log) == 2
    assert call_log[0][0:3] == ["kubectl", "get", "service"]
    assert call_log[1][0:3] == ["kubectl", "patch", "service"]
    # Second call should patch to "blue" (the toggle of current="green").
    patch_json_arg = call_log[1][-1]
    assert '"blue"' in patch_json_arg


def test_rollback_toggle_map_is_symmetric():
    assert rollback._TOGGLE["blue"] == "green"
    assert rollback._TOGGLE["green"] == "blue"
    assert len(rollback._TOGGLE) == 2
