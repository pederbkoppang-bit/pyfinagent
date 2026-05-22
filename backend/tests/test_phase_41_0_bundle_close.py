"""phase-41.0 Phase-29.8 P2 bundle close -- trace-link regression-lock.

Closes closure_roadmap.md section 3 OPEN-32 as a TRACE-LINK CLOSURE:
phase-29.8 was a planning-time bundle that mapped to phase-37.3 +
phase-40.1 + phase-40.2 (closure_roadmap.md section 1 verdict table).
Phase-29.8 was dropped from `.claude/masterplan.json` during phase-45.0
closure re-audit (cycle 12); the phase-41.0 verification command was
relaxed to "absent OR done".

Tests:
  1. Masterplan invariant: phase-29.8 absent OR status=done (mirrors
     verbatim masterplan verification command).
  2. SUBSTANTIVE caveat (load-bearing): phase-37.3 + phase-40.1
     remain independently visible as separate step IDs in their
     parent phases. Catches future drift where someone "tidies up"
     37.3 + 40.1 alongside the 41.0 flip.
  3. ADR exists at docs/decisions/phase-41-0-bundle-close.md.

Per researcher (handoff/current/research_brief_phase_41_0.md): the
trace-link closure is mechanical; the substantive work for 37.3 + 40.1
is tracked independently. This test file enforces both.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MASTERPLAN = REPO_ROOT / ".claude" / "masterplan.json"
ADR = REPO_ROOT / "docs" / "decisions" / "phase-41-0-bundle-close.md"


def test_phase_41_0_masterplan_invariant_29_8_absent_or_done():
    """Criterion 1 verbatim from masterplan: phase-29.8 absent OR done.
    Mirrors `python -c "import json; ...; assert (not ps) or ps[0]['status']=='done'"`."""
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    ps = [p for p in d["phases"] if p["id"] == "phase-29.8"]
    if ps:
        assert ps[0]["status"] == "done", (
            f"phase-29.8 present but status={ps[0]['status']!r}; must be 'done' or absent"
        )
    # If ps is empty (absent), the invariant is satisfied.


def test_phase_41_0_phase_29_9_invariant_also_absent_or_done():
    """phase-41.1 sibling: phase-29.9 mirror invariant (the P3 bundle).
    Test here too so the file covers both 41.0 + 41.1 dependencies."""
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    ps = [p for p in d["phases"] if p["id"] == "phase-29.9"]
    if ps:
        assert ps[0]["status"] == "done", (
            f"phase-29.9 present but status={ps[0]['status']!r}; must be 'done' or absent"
        )


def test_phase_41_0_residuals_37_3_and_40_1_remain_visible_separately():
    """SUBSTANTIVE caveat (load-bearing). Researcher flagged: 41.0 PASS
    is mechanical trace-link closure, NOT engineered closure of all 9
    sub-items. 2 sub-items (phase-37.3 budget_tokens + phase-40.1 OpenAlex)
    remain INDEPENDENTLY tracked. This test catches future drift where
    someone deletes 37.3 + 40.1 from the masterplan alongside the 41.0 flip
    (which would silently hide unresolved work)."""
    d = json.loads(MASTERPLAN.read_text(encoding="utf-8"))
    found_37_3 = False
    found_40_1 = False
    for phase in d["phases"]:
        for step in phase.get("steps", []):
            if step.get("id") == "37.3":
                found_37_3 = True
            if step.get("id") == "40.1":
                found_40_1 = True
    assert found_37_3, (
        "phase-37.3 (budget_tokens deprecation, OPEN-18) must remain "
        "tracked as a separate masterplan step -- it was NOT closed by "
        "phase-41.0 trace-link closure."
    )
    assert found_40_1, (
        "phase-40.1 (OpenAlex .env.example, OPEN-24) must remain tracked "
        "as a separate masterplan step -- it was NOT closed by phase-41.0 "
        "trace-link closure."
    )


def test_phase_41_0_adr_documents_the_trace_link_closure():
    """ADR file must exist with required Nygard 5-section structure."""
    assert ADR.exists(), f"ADR file missing: {ADR}"
    text = ADR.read_text(encoding="utf-8")
    # Nygard ADR sections
    for section in ["Context", "Decision", "Status", "Consequences"]:
        assert f"## {section}" in text, (
            f"ADR missing Nygard section '## {section}'"
        )
    # Must explicitly note the residual caveat
    assert "phase-37.3" in text and "phase-40.1" in text, (
        "ADR must enumerate phase-37.3 + phase-40.1 as remaining residuals"
    )
    assert "trace-link" in text.lower(), (
        "ADR must explicitly frame closure as 'trace-link' (not engineered)"
    )


def test_phase_41_0_decisions_directory_structure():
    """The docs/decisions/ pattern is the project's canonical ADR location.
    Verify it exists + the new ADR is named per convention."""
    decisions_dir = REPO_ROOT / "docs" / "decisions"
    assert decisions_dir.exists() and decisions_dir.is_dir(), (
        f"docs/decisions/ must exist as a directory: {decisions_dir}"
    )
    # New ADR filename convention: phase-<id>-<slug>.md
    assert ADR.name.startswith("phase-41-0-"), (
        f"ADR filename {ADR.name!r} must start with 'phase-41-0-'"
    )
