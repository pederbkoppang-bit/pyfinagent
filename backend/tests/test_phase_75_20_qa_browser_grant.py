"""phase-75.20: Q/A live-UI gate enforceability + read-only primary path.

Config-content step: the SUBJECT of every guard here IS configuration text
(qa.md frontmatter/prose, settings.json deny list, .mcp.json args,
qa-verdict.js launch options), so content assertions are the correct guard
shape -- the qa.md 4c source-scan caution applies to scans posing as
BEHAVIORAL evidence, not to pins whose subject is the source itself. Each
pin is mutation-tested against the real config (matrix + verbatim results
in live_check_75.20.md), including a harness-stub mutation of this suite's
own tools-line extractor.

Pinned facts (probe-derived, wf_9277ada4-390 / wf_78b46633-fdd):
  - general-purpose Workflow agents carry Edit/Write + the full deferred
    MCP surface -> qa-verdict.js must launch agentType 'qa'.
  - deny rules bind immediately and session-wide (the two RCE playwright
    tools disappeared from the live tool surface on settings.json write).
  - The R11 vacuity: the step's IMMUTABLE verification command accepts
    `--user-data-dir` via a token-level substring accident, so THIS suite
    carries the non-vacuous isolation assert instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QA_MD = REPO_ROOT / ".claude/agents/qa.md"
SETTINGS = REPO_ROOT / ".claude/settings.json"
MCP_JSON = REPO_ROOT / ".mcp.json"
QA_VERDICT = REPO_ROOT / ".claude/workflows/qa-verdict.js"

GRANTED_BROWSER = (
    "mcp__playwright__browser_navigate",
    "mcp__playwright__browser_snapshot",
    "mcp__playwright__browser_take_screenshot",
    "mcp__playwright__browser_console_messages",
)
# Criterion 1's "at most" envelope: the grant may not exceed this set.
ALLOWED_BROWSER_SUPERSET = set(GRANTED_BROWSER) | {
    "mcp__playwright__browser_network_requests",
    "mcp__playwright__browser_resize",
}
MUTATION_TOOLS = (
    "browser_run_code_unsafe",
    "browser_evaluate",
    "browser_click",
    "browser_type",
    "browser_fill_form",
)
DENY_ENTRIES = (
    "mcp__playwright__browser_run_code_unsafe",
    "mcp__playwright__browser_evaluate",
)


def _tools_line(text: str) -> str:
    """The qa.md frontmatter tools allowlist -- the single line every grant
    assertion keys off (mutating this helper is part of the 4c matrix)."""
    for line in text.splitlines():
        if line.startswith("tools:"):
            return line
    raise AssertionError("qa.md has no frontmatter tools: line")


# -- C1: the grant ---------------------------------------------------------

def test_tools_line_grants_the_read_only_browser_subset():
    line = _tools_line(QA_MD.read_text(encoding="utf-8"))
    for tool in GRANTED_BROWSER:
        assert tool in line, f"{tool} missing from qa.md tools grant"


@pytest.mark.parametrize("tool", MUTATION_TOOLS)
def test_tools_line_grants_no_mutation_tool(tool):
    line = _tools_line(QA_MD.read_text(encoding="utf-8"))
    assert tool not in line, f"mutation tool {tool} granted in qa.md tools line"


def test_browser_grant_is_within_the_allowed_superset():
    line = _tools_line(QA_MD.read_text(encoding="utf-8"))
    granted = set(re.findall(r"mcp__playwright__[a-z_]+", line))
    excess = granted - ALLOWED_BROWSER_SUPERSET
    assert not excess, f"grant exceeds criterion-1's 'at most' envelope: {excess}"


# -- C2: session-wide deny (bypass-proof, binds the Workflow path too) -----

def test_settings_denies_the_rce_tools_exactly():
    deny = json.loads(SETTINGS.read_text(encoding="utf-8"))["permissions"]["deny"]
    for entry in DENY_ENTRIES:
        assert entry in deny, f"deny list missing exact entry {entry}"


# -- C4: isolation, the NON-vacuous form (R11 replacement) -----------------

def test_playwright_server_is_isolated_and_sheds_the_profile_pin():
    args = json.loads(MCP_JSON.read_text(encoding="utf-8"))["mcpServers"]["playwright"]["args"]
    assert "--isolated" in args, "--isolated missing from playwright args"
    # The immutable command's assert #3 is satisfiable by the '--user-data-dir'
    # TOKEN itself (R11) -- this is the assert that can actually fail:
    assert not any("user-data-dir" in str(a) for a in args), (
        "fixed --user-data-dir profile pin still present alongside/instead of --isolated")


# -- C3: the primary path is constrained by configuration ------------------

def test_qa_verdict_launches_the_restricted_agent_type():
    src = QA_VERDICT.read_text(encoding="utf-8")
    assert re.search(r"agentType:\s*'qa'", src), "qa-verdict.js does not launch agentType 'qa'"
    assert not re.search(r"agentType:\s*'general-purpose'", src), (
        "qa-verdict.js still launches the unrestricted general-purpose type")


# -- C5/C6: the 1c prose contract ------------------------------------------

def test_1c_capture_taken_by_evaluator_with_degraded_fallback_named():
    qa = QA_MD.read_text(encoding="utf-8")
    assert "taken BY YOU" in qa, "1c does not put the capture on the evaluator"
    assert "EXPLICITLY-DEGRADED fallback" in qa, (
        "1c does not name the Main-produced capture as the degraded fallback")


def test_1c_instructs_the_deterministic_select_form():
    qa = QA_MD.read_text(encoding="utf-8")
    assert "select:mcp__playwright__browser_navigate" in qa, (
        "1c does not instruct the deterministic ToolSearch select: form")


def test_1c_keeps_dev_server_lifecycle_with_main():
    qa = QA_MD.read_text(encoding="utf-8")
    assert "NEVER start or kill a server" in qa, (
        "1c does not reserve dev-server lifecycle to Main")
