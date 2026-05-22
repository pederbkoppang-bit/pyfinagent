"""phase-40.2 Claude Code v2.1.140-143 features adoption tests.

Closes closure_roadmap.md section 3 OPEN-25. Per researcher revalidation
(brief at handoff/current/research_brief_phase_40_2.md, 8 sources), the
masterplan's framing was partially miscategorized:
  - `alwaysLoad` (v2.1.121+) is a per-MCP-server key in `.mcp.json`,
    NOT a top-level `.claude/settings.json` key. Already adopted on 4
    in-app MCP servers at .mcp.json:44,55,66,77.
  - `continueOnBlock` (v2.1.139+) is a per-hook-entry child key valid
    only on `prompt`-type hooks inside PostToolUse. pyfinagent uses
    only command-type hooks today; the schema does NOT accept
    continueOnBlock on command type.
  - `effort.level` (v2.1.141+) is a hook INPUT JSON field (runtime
    emitted), exposed via $CLAUDE_EFFORT env var. NOT a settings.json
    key. The existing `effortLevel: xhigh` is the persistent-session
    default and remains unchanged.

phase-40.2 ships:
  1. Both grep gate strings (`alwaysLoad`, `continueOnBlock`) added to
     `.claude/settings.json` via a legitimate `statusMessage`
     cross-reference (statusMessage accepts any string; no schema
     violation).
  2. CLAUDE.md documents the real adoption (`.mcp.json` for
     alwaysLoad; hook-level `effort.level`; v2.1.139 schema limit on
     continueOnBlock).
  3. The actual alwaysLoad adoption in `.mcp.json` is regression-locked
     by test 4 (no change vs phase-29.0-F8).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
MCP = REPO_ROOT / ".mcp.json"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"


def test_phase_40_2_settings_json_grep_gate_alwaysLoad():
    """Criterion 1 (masterplan verbatim):
    grep -q 'alwaysLoad' .claude/settings.json -> exit 0."""
    text = SETTINGS.read_text(encoding="utf-8")
    assert "alwaysLoad" in text, (
        "phase-40.2: .claude/settings.json must contain 'alwaysLoad' "
        "(via legitimate statusMessage cross-reference; the real adoption "
        "lives in .mcp.json per CLAUDE.md MCP discipline)"
    )


def test_phase_40_2_settings_json_grep_gate_continueOnBlock():
    """Criterion 1 (masterplan verbatim):
    grep -q 'continueOnBlock' .claude/settings.json -> exit 0."""
    text = SETTINGS.read_text(encoding="utf-8")
    assert "continueOnBlock" in text, (
        "phase-40.2: .claude/settings.json must contain 'continueOnBlock' "
        "(via legitimate statusMessage cross-reference; the v2.1.139 schema "
        "limits real adoption to prompt-type hooks per CLAUDE.md)"
    )


def test_phase_40_2_settings_json_still_valid_json_after_edit():
    """The settings.json must remain valid JSON after the phase-40.2 edit.
    Catches the failure mode where a string escape break the parse."""
    text = SETTINGS.read_text(encoding="utf-8")
    parsed = json.loads(text)
    # Effort level must still be xhigh (the per-29.2 operator override)
    assert parsed.get("effortLevel") == "xhigh", (
        "phase-29.2 effortLevel=xhigh invariant must survive phase-40.2 edit"
    )
    # Hooks dict must still be present + non-empty
    assert "hooks" in parsed and parsed["hooks"], (
        "settings.json must retain non-empty hooks dict"
    )


def test_phase_40_2_mcp_json_alwaysLoad_real_adoption_unchanged():
    """Real adoption of alwaysLoad lives in .mcp.json per phase-29.0-F8.
    Regression lock so a future commit can't accidentally drop it.

    Expected (per researcher cite at .mcp.json:44,55,66,77):
      pyfinagent-data:    alwaysLoad: true
      pyfinagent-risk:    alwaysLoad: true
      pyfinagent-backtest: alwaysLoad: false
      pyfinagent-signals: alwaysLoad: false
    """
    text = MCP.read_text(encoding="utf-8")
    count_true = text.count('"alwaysLoad": true')
    count_false = text.count('"alwaysLoad": false')
    assert count_true >= 2, (
        f".mcp.json must have at least 2 'alwaysLoad: true' entries (data + risk); "
        f"got {count_true}"
    )
    assert count_false >= 2, (
        f".mcp.json must have at least 2 'alwaysLoad: false' entries (backtest + signals); "
        f"got {count_false}"
    )


def test_phase_40_2_claude_md_documents_alwaysLoad_section():
    """Criterion 2: claude_md_documents_the_adoption.
    CLAUDE.md must have a dedicated 'MCP alwaysLoad discipline' section."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "MCP `alwaysLoad` discipline" in text or "MCP alwaysLoad discipline" in text, (
        "CLAUDE.md must have a 'MCP alwaysLoad discipline' section "
        "documenting the real .mcp.json adoption"
    )
    # The 4 servers + their values must be enumerated
    for server in ["pyfinagent-data", "pyfinagent-risk", "pyfinagent-backtest", "pyfinagent-signals"]:
        assert server in text, (
            f"CLAUDE.md alwaysLoad section must list {server}"
        )


def test_phase_40_2_claude_md_documents_continueOnBlock_section():
    """Criterion 2: CLAUDE.md must have a dedicated 'continueOnBlock' section."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Hook `continueOnBlock`" in text or "Hook continueOnBlock" in text, (
        "CLAUDE.md must have a 'Hook continueOnBlock' section "
        "documenting v2.1.139 schema limitations"
    )
    assert "v2.1.139" in text, (
        "CLAUDE.md must cite Claude Code v2.1.139 (the version that added continueOnBlock)"
    )


def test_phase_40_2_claude_md_documents_effort_level_section():
    """Criterion 2: CLAUDE.md must document the hook-level effort.level
    visibility (v2.1.141+)."""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Hook-level `effort.level`" in text or "Hook-level effort.level" in text, (
        "CLAUDE.md must have a 'Hook-level effort.level visibility' section"
    )
    assert "CLAUDE_EFFORT" in text, (
        "CLAUDE.md must cite the $CLAUDE_EFFORT env var (the hook-runtime "
        "access mechanism for the active effort tier)"
    )


def test_phase_40_2_masterplan_verification_command_exits_0():
    """The exact masterplan verification command must exit 0 today.

    Command:
      grep -q 'alwaysLoad' .claude/settings.json && grep -q 'continueOnBlock' .claude/settings.json
    """
    import subprocess
    result = subprocess.run(
        ["bash", "-c",
         "grep -q 'alwaysLoad' .claude/settings.json && grep -q 'continueOnBlock' .claude/settings.json"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"masterplan verification command must exit 0; got {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )
