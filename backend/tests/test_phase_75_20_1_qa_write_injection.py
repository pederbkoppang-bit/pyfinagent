"""phase-75.20.1 -- the subagent loader injects Write+Edit past the qa
agent's frontmatter allowlist.

Root cause (documented upstream behavior, not a bug): qa.md `memory:
project` auto-enables Read/Write/Edit for memory-file management
(sub-agents doc), so the tools allowlist cannot exclude Write/Edit
without killing Q/A memory curation. Enforcement is therefore the
path-aware `qa-write-guard.sh` PreToolUse hook: deny Write/Edit when the
acting agent is `qa` unless the target is under .claude/agent-memory/qa/.

These tests drive the REAL hook script via subprocess with the
documented stdin-JSON hook payload shape. The hook is fail-open by
design: only an explicit qa-outside-memory match may block.

See handoff/current/contract.md (step 75.20.1) and
docs/runbooks/per-step-protocol.md section 4 POST-VERDICT CLEANLINESS
(the Bash-write path the hook structurally cannot intercept).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "qa-write-guard.sh"


def _run_hook(payload) -> subprocess.CompletedProcess:
    """Run the real hook with `payload` on stdin (str passed through,
    anything else JSON-encoded), from the repo root like Claude Code does."""
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        ["bash", str(HOOK)],
        input=data, capture_output=True, text=True, timeout=15,
        cwd=REPO, env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                       "CLAUDE_PROJECT_DIR": str(REPO)},
    )


def test_t1_qa_write_outside_memory_blocks():
    r = _run_hook({"agent_type": "qa", "tool_name": "Write",
                   "tool_input": {"file_path": "/tmp/evil.md"}})
    assert r.returncode == 2, f"expected block (2), got {r.returncode}: {r.stderr!r}"
    assert "read-only" in r.stderr


def test_t2_qa_edit_outside_memory_blocks():
    r = _run_hook({"agent_type": "qa", "tool_name": "Edit",
                   "tool_input": {"file_path": str(REPO / "backend" / "main.py")}})
    assert r.returncode == 2, f"expected block (2), got {r.returncode}: {r.stderr!r}"


def test_t3_qa_write_inside_memory_allowed():
    r = _run_hook({"agent_type": "qa", "tool_name": "Write",
                   "tool_input": {"file_path":
                                  str(REPO / ".claude" / "agent-memory" / "qa" / "MEMORY.md")}})
    assert r.returncode == 0, f"memory curation must stay allowed: {r.stderr!r}"


def test_t3b_qa_traversal_escape_still_blocks():
    r = _run_hook({"agent_type": "qa", "tool_name": "Write",
                   "tool_input": {"file_path":
                                  ".claude/agent-memory/qa/../../../etc/x"}})
    assert r.returncode == 2, "normpath traversal escape must not defeat the memory-dir check"


def test_t4_main_no_agent_type_allowed():
    r = _run_hook({"tool_name": "Write", "tool_input": {"file_path": "/tmp/x.md"}})
    assert r.returncode == 0, f"Main (no agent_type) must never be blocked: {r.stderr!r}"


def test_t5_other_agent_type_allowed():
    r = _run_hook({"agent_type": "researcher", "tool_name": "Write",
                   "tool_input": {"file_path": "/tmp/x.md"}})
    assert r.returncode == 0, "only the qa agent type is guarded"


def test_t6_malformed_payload_fails_open():
    r = _run_hook("this is not json {")
    assert r.returncode == 0, "fail-open on malformed payload is a design requirement"


def test_t6b_empty_payload_fails_open():
    r = _run_hook("")
    assert r.returncode == 0


def test_t7_hook_registered_in_settings():
    s = json.loads((REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))
    entries = s["hooks"]["PreToolUse"]
    guard = [e for e in entries if "qa-write-guard" in json.dumps(e)]
    assert len(guard) == 1, "qa-write-guard must be registered exactly once under PreToolUse"
    assert guard[0].get("matcher") == "Write|Edit", (
        "guard must be scoped to Write|Edit -- an unmatched entry would run "
        "on every tool call"
    )


def test_t8_runbook_carries_cleanliness_rule():
    raw = (REPO / "docs" / "runbooks" / "per-step-protocol.md").read_text(encoding="utf-8")
    # Whitespace-normalized so hard-wrapped prose can't split a phrase.
    text = " ".join(raw.split())
    assert "POST-VERDICT CLEANLINESS" in text
    assert "git status --short" in text
    assert "INADMISSIBLE" in text
    assert "fresh Q/A" in text
    # The rule must explain WHY it coexists with the hook (Bash-write gap).
    assert "Bash subprocess writes" in text


def test_t9_qa_memory_pin():
    """qa.md `memory: project` is the documented ROOT CAUSE of the
    Write/Edit injection (sub-agents doc: memory auto-enables
    Read/Write/Edit). If this pin ever fails, the injection premise
    changed -- re-run .claude/workflows/probe-qa-tool-surface.js and
    re-evaluate whether qa-write-guard.sh is still needed before
    deleting either."""
    qa = (REPO / ".claude" / "agents" / "qa.md").read_text(encoding="utf-8")
    frontmatter = qa.split("---")[1]
    assert "memory: project" in frontmatter
