"""phase-62.0: unit tests for the away-ops PreToolUse danger-hook patterns.

Invokes .claude/hooks/pre-tool-use-danger.sh as a subprocess with synthetic
CLAUDE_TOOL_NAME / CLAUDE_TOOL_INPUT payloads (the hook's documented env
interface) and asserts block (exit 2) / allow (exit 0) semantics.

The backend/.env token-cursor gate is exercised against a TEMP project dir
(CLAUDE_PROJECT_DIR) so no real cursor or .env state is touched.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "pre-tool-use-danger.sh"


def run_hook(tool: str, tool_input: dict, project_dir: Path | None = None,
             extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE_")}
    env["CLAUDE_TOOL_NAME"] = tool
    env["CLAUDE_TOOL_INPUT"] = json.dumps(tool_input)
    env["CLAUDE_PROJECT_DIR"] = str(project_dir or REPO)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(HOOK)], env=env, capture_output=True, text=True, timeout=30
    )


def bash(cmd: str, project_dir: Path | None = None) -> subprocess.CompletedProcess:
    return run_hook("Bash", {"command": cmd}, project_dir=project_dir)


# ── force-push variants (rail 3) ──────────────────────────────────────

@pytest.mark.parametrize("cmd", [
    "git push origin main --force",          # position-free flag (gap closed)
    "git push --force origin main",          # legacy prefix form
    "git push -f",                           # bare -f
    "git push origin -f main",               # -f mid-args
    "git push origin +main",                 # +refspec forces with no flag
    "git add -A && git push origin main --force-with-lease",
])
def test_force_push_variants_blocked(cmd):
    r = bash(cmd)
    assert r.returncode == 2, (cmd, r.stderr)
    assert "force" in r.stderr.lower()


@pytest.mark.parametrize("cmd", [
    "git push origin main",
    "git push",
    "git commit -m 'feat: x' && git push origin main",
])
def test_normal_push_allowed(cmd):
    r = bash(cmd)
    assert r.returncode == 0, (cmd, r.stderr)


# ── launchctl removal verbs on pyfinagent labels (rail 9) ─────────────

@pytest.mark.parametrize("verb", ["bootout", "unload", "remove", "disable"])
def test_launchctl_removal_blocked(verb):
    r = bash(f"launchctl {verb} gui/501/com.pyfinagent.backend")
    assert r.returncode == 2, (verb, r.stderr)
    assert "rail 9" in r.stderr


def test_launchctl_kickstart_allowed():
    r = bash("launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend")
    assert r.returncode == 0, r.stderr


def test_launchctl_removal_other_label_allowed():
    r = bash("launchctl bootout gui/501/com.example.other")
    assert r.returncode == 0, r.stderr


# ── backend/.env write tripwire + token-cursor gate (rail 1) ──────────

@pytest.fixture()
def temp_project(tmp_path: Path) -> Path:
    (tmp_path / "handoff" / "away_ops").mkdir(parents=True)
    (tmp_path / "handoff" / "audit").mkdir(parents=True)
    return tmp_path


@pytest.mark.parametrize("cmd", [
    "printf 'PAPER_SWAP_CHURN_FIX_ENABLED=false\\n' >> backend/.env",
    "echo 'PAPER_DATA_INTEGRITY_ENABLED=false' > backend/.env",
    "sed -i '' 's/PAPER_RISK_JUDGE_REJECT_BINDING=true/PAPER_RISK_JUDGE_REJECT_BINDING=false/' backend/.env",
    "echo PAPER_X=1 | tee -a backend/.env",
    "perl -pi -e 's/true/false/' backend/.env",
])
def test_env_write_blocked_without_cursor(cmd, temp_project):
    r = bash(cmd, project_dir=temp_project)
    assert r.returncode == 2, (cmd, r.stderr)
    assert "rail 1" in r.stderr
    assert "pending_tokens" in r.stderr  # block message prescribes the ask path


def test_env_write_allowed_with_fresh_cursor(temp_project):
    cursor = temp_project / "handoff" / "away_ops" / "tokens_cursor"
    cursor.write_text("42\n")  # fresh mtime = token just applied
    r = bash("printf 'PAPER_SWAP_CHURN_FIX_ENABLED=true\\n' >> backend/.env",
             project_dir=temp_project)
    assert r.returncode == 0, r.stderr


def test_env_write_blocked_with_stale_cursor(temp_project):
    cursor = temp_project / "handoff" / "away_ops" / "tokens_cursor"
    cursor.write_text("42\n")
    stale = time.time() - 7 * 3600  # 7h > the 6h freshness window
    os.utime(cursor, (stale, stale))
    r = bash("echo PAPER_X=1 >> backend/.env", project_dir=temp_project)
    assert r.returncode == 2, r.stderr


def test_other_env_file_write_allowed(temp_project):
    r = bash("echo FOO=1 >> frontend/.env.local", project_dir=temp_project)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("tool", ["Edit", "Write"])
def test_edit_write_tools_on_env_blocked(tool, temp_project):
    r = run_hook(tool, {"file_path": str(REPO / "backend" / ".env"),
                        "old_string": "x", "new_string": "y"},
                 project_dir=temp_project)
    assert r.returncode == 2, (tool, r.stderr)
    assert "rail 1" in r.stderr


def test_edit_tool_on_other_file_allowed(temp_project):
    r = run_hook("Edit", {"file_path": str(REPO / "backend" / "main.py"),
                          "old_string": "x", "new_string": "y"},
                 project_dir=temp_project)
    assert r.returncode == 0, r.stderr


def test_edit_on_env_allowed_with_fresh_cursor(temp_project):
    cursor = temp_project / "handoff" / "away_ops" / "tokens_cursor"
    cursor.write_text("42\n")
    r = run_hook("Edit", {"file_path": str(REPO / "backend" / ".env"),
                          "old_string": "x", "new_string": "y"},
                 project_dir=temp_project)
    assert r.returncode == 0, r.stderr


# ── pre-existing guards still intact (regression) ─────────────────────

def test_rm_rf_dangerous_target_still_blocked():
    r = bash("rm -rf /")
    assert r.returncode == 2


def test_rm_rf_scoped_target_still_allowed():
    r = bash("rm -rf node_modules")
    assert r.returncode == 0, r.stderr


def test_escape_hatch_still_works(temp_project):
    r = run_hook("Bash", {"command": "echo PAPER_X=1 >> backend/.env"},
                 project_dir=temp_project, extra_env={"CLAUDE_ALLOW_DANGER": "1"})
    assert r.returncode == 0
