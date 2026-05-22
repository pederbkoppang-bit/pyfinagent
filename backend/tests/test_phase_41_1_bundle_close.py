"""phase-41.1 Phase-29.9 P3 bundle close -- trace-link regression-lock.

Mirror of test_phase_41_0_bundle_close.py for the P3 bundle. Closes
closure_roadmap.md section 3 OPEN-33 as a TRACE-LINK CLOSURE.

Tests:
  1. Masterplan invariant: phase-29.9 absent OR done.
  2. SUBSTANTIVE caveat (load-bearing): phase-40.3 (stress-test doctrine)
     remains independently visible as a separate step ID. Catches future
     drift where someone tidies up 40.3 alongside the 41.1 flip.
  3. Engineered-done sub-items persisted (researcher multi-subagent fork
     doc + Q/A cycle-2-flow surfacing) -- verifies agent prompts retain
     the substantive content.
  4. ADR exists at docs/decisions/phase-41-1-bundle-close.md.
  5. ADR follows project naming convention.

Per researcher (handoff/current/research_brief_phase_41_1.md): 10 P3
sub-items in 4 buckets: 2 engineered-done in agent prompts; 2 vendor-
released (owner-only adoption); 4 sandbox-blocked / future; 1 absorbed
into phase-40.3; 1 independently pending (phase-40.3 itself).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERPLAN = REPO_ROOT / ".claude" / "masterplan.json"
ADR = REPO_ROOT / "docs" / "decisions" / "phase-41-1-bundle-close.md"
RESEARCHER_PROMPT = REPO_ROOT / ".claude" / "agents" / "researcher.md"
QA_PROMPT = REPO_ROOT / ".claude" / "agents" / "qa.md"


def test_phase_41_1_masterplan_invariant_29_9_absent_or_done():
    """Criterion 1 verbatim from masterplan: phase-29.9 absent OR done."""
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    ps = [p for p in d["phases"] if p["id"] == "phase-29.9"]
    if ps:
        assert ps[0]["status"] == "done", (
            f"phase-29.9 present but status={ps[0]['status']!r}; must be 'done' or absent"
        )


def test_phase_41_1_residual_40_3_remains_visible_separately():
    """SUBSTANTIVE caveat (load-bearing). phase-40.3 (stress-test doctrine
    harness-free Opus 4.7 cycle, OPEN-26) remains INDEPENDENTLY pending.
    This test catches future drift where someone removes 40.3 from the
    masterplan alongside the 41.1 flip (which would silently hide the
    unresolved stress-test work)."""
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    found_40_3 = False
    for phase in d["phases"]:
        for step in phase.get("steps", []):
            if step.get("id") == "40.3":
                found_40_3 = True
                break
    assert found_40_3, (
        "phase-40.3 (stress-test doctrine harness-free Opus 4.7 cycle, OPEN-26) "
        "must remain tracked as a separate masterplan step -- it was NOT closed "
        "by phase-41.1 trace-link closure."
    )


def test_phase_41_1_engineered_done_sub_items_persisted():
    """Sub-items 1+2 of the P3 bundle were engineered-done in agent prompts
    (researcher.md multi-subagent fork doc + qa.md cycle-2-flow surfacing).
    This test asserts the substantive content remains in those files so a
    future commit can't accidentally regress the prompt while leaving 41.1
    marked done."""
    if RESEARCHER_PROMPT.exists():
        researcher_text = RESEARCHER_PROMPT.read_text(encoding="utf-8")
        # The phrase should appear (some variation OK): researcher prompt
        # mentions multi-subagent / sub-agent / fork patterns
        assert any(
            phrase.lower() in researcher_text.lower()
            for phrase in ["multi-subagent", "subagent fork", "spawn researcher", "researcher must", "researcher subagent"]
        ), "researcher.md must document the multi-subagent / spawn discipline (phase-41.1 sub-item #1)"
    if QA_PROMPT.exists():
        qa_text = QA_PROMPT.read_text(encoding="utf-8")
        # Q/A prompt should mention cycle-2 flow / second-opinion / verdict-shopping
        assert any(
            phrase.lower() in qa_text.lower()
            for phrase in ["cycle-2", "cycle 2", "second-opinion", "second opinion", "verdict-shopping", "verdict shopping"]
        ), "qa.md must document the cycle-2-flow / no-second-opinion-shopping discipline (phase-41.1 sub-item #2)"


def test_phase_41_1_adr_documents_the_trace_link_closure():
    """ADR file must exist with Nygard 5-section structure + residual table."""
    assert ADR.exists(), f"ADR file missing: {ADR}"
    text = ADR.read_text(encoding="utf-8")
    for section in ["Context", "Decision", "Status", "Consequences"]:
        assert f"## {section}" in text, (
            f"ADR missing Nygard section '## {section}'"
        )
    # Must explicitly note the residual caveat
    assert "phase-40.3" in text, (
        "ADR must reference phase-40.3 (the independently-tracked residual)"
    )
    assert "trace-link" in text.lower(), (
        "ADR must explicitly frame closure as 'trace-link' (not engineered)"
    )
    # Must enumerate the bucket counts honestly
    assert "10" in text or "ten" in text.lower(), (
        "ADR must reference the 10 P3 sub-items (the count from research brief)"
    )


def test_phase_41_1_decisions_directory_structure():
    """ADR filename follows phase-41-1- convention (mirrors phase-41-0-)."""
    assert ADR.name.startswith("phase-41-1-"), (
        f"ADR filename {ADR.name!r} must start with 'phase-41-1-'"
    )
    # The phase-41.0 ADR also exists (cycle 26 prior)
    sibling = REPO_ROOT / "docs" / "decisions" / "phase-41-0-bundle-close.md"
    assert sibling.exists(), (
        "phase-41-0-bundle-close.md ADR sibling must exist (cycle 26 prior closure)"
    )
