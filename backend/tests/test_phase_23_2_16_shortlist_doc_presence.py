"""phase-23.2.16 (P2) verification: deferred items triage shortlist exists.

Per researcher (handoff/current/research_brief_phase_23_2_16.md, 10 sources):
shortlist of 3 highest-leverage items extracted from Section H of phase-23.2.0
audit. Frameworks: WSJF + RICE hybrid; sources Intercom + SAFe + ProductPlan.

This test enforces the shortlist doc presence + structural invariants
(8 source cycles enumerated; 3 shortlist items; each with leverage score;
ASCII-only).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# phase-56.2: the archive-handoff hook moved the shortlist doc out of
# handoff/current/ when its step closed; the canonical location is the archive.
SHORTLIST_DOC = REPO_ROOT / "handoff" / "archive" / "phase-23.2.16" / "phase-23.2.16-shortlist.md"


def test_phase_23_2_16_shortlist_doc_exists():
    """The shortlist deliverable file must exist."""
    assert SHORTLIST_DOC.exists(), (
        f"phase-23.2.16 shortlist doc missing: {SHORTLIST_DOC}"
    )


def test_phase_23_2_16_doc_has_8_deferred_items_table():
    """The shortlist doc must reference all 8 source cycles per researcher
    (23.1.13, 23.1.14, 23.1.15, 23.1.16, 23.1.17, 23.1.18, 23.1.19, 23.1.22)."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    expected_cycles = ["23.1.13", "23.1.14", "23.1.15", "23.1.16", "23.1.17", "23.1.18", "23.1.19", "23.1.22"]
    missing = [c for c in expected_cycles if c not in text]
    assert not missing, (
        f"shortlist doc missing source cycles: {missing}"
    )


def test_phase_23_2_16_doc_has_3_item_shortlist():
    """The doc must contain a 3-item shortlist (numbered #1, #2, #3)."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    # Match "### #1 --", "### #2 --", "### #3 --"
    pattern = re.compile(r"^###\s+#[123]\s+--", re.MULTILINE)
    matches = pattern.findall(text)
    assert len(matches) >= 3, (
        f"shortlist doc must have 3 numbered items (### #1/#2/#3); got {len(matches)}"
    )


def test_phase_23_2_16_each_shortlist_item_has_leverage_score():
    """Each shortlist item must cite a numeric Leverage score."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    # Match "**Leverage:** NNN" or "Leverage: NNN" with optional decimal
    pattern = re.compile(r"\*\*Leverage:\*\*\s*([\d.]+)", re.MULTILINE)
    matches = pattern.findall(text)
    assert len(matches) >= 3, (
        f"shortlist doc must have >=3 explicit Leverage scores; got {len(matches)}: {matches}"
    )
    # All should be numeric
    for m in matches:
        float(m)  # raises if not


def test_phase_23_2_16_doc_ascii_only():
    """Per CLAUDE.md no-emoji rule: shortlist doc must be ASCII-only."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    non_ascii = [(i, c, ord(c)) for i, c in enumerate(text) if ord(c) > 0x7F]
    assert not non_ascii, (
        f"shortlist doc has {len(non_ascii)} non-ASCII chars; "
        f"first 5: {non_ascii[:5]}"
    )


def test_phase_23_2_16_doc_references_research_brief():
    """The shortlist doc must cite research_brief_phase_23_2_16.md as
    its source (audit-trail discipline)."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    assert "research_brief_phase_23_2_16.md" in text, (
        "shortlist doc must cite the research_brief as its source"
    )


def test_phase_23_2_16_doc_cross_references_8_new_tickets():
    """The shortlist doc must also acknowledge the 8 new tickets surfaced
    by the verification sweep (not silently drop them)."""
    text = SHORTLIST_DOC.read_text(encoding="utf-8")
    expected_tickets = [
        "23.2.6.1", "23.2.11.1", "23.2.11.2", "23.2.12.1",
        "23.2.12.2", "23.2.13.1", "23.2.15.1", "23.2.15.2",
    ]
    missing = [t for t in expected_tickets if t not in text]
    assert not missing, (
        f"shortlist doc must cross-reference all 8 new tickets; missing: {missing}"
    )
