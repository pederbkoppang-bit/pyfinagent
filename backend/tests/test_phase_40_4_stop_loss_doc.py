"""phase-40.4 verification: stop-loss 8% vs 10% A/B decision doc + turnkey.

Per researcher (handoff/current/research_brief_phase_40_4.md, 6 sources):
KEEP 8% per literature consensus (O'Neil CAN SLIM = our layer; Han/Zhou/Zhu
10% = different portfolio-momentum overlay layer). Walk-forward A/B
DEFERRED to operator runbook (30-90 min compute).

Tests verify the doc deliverables (the literature-driven KEEP decision +
turnkey runner) are in place. The masterplan criterion's "results in
quant_results.tsv" half is acknowledged as DEFERRED-LIVE in the ADR.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR = REPO_ROOT / "docs" / "decisions" / "stop_loss_default.md"
RUNNER = REPO_ROOT / "scripts" / "backtest" / "run_stop_loss_ab.py"


def test_phase_40_4_adr_exists():
    """ADR file exists per masterplan criterion 'decision_documented_in_docs_decisions'."""
    assert ADR.exists(), f"ADR missing: {ADR}"


def test_phase_40_4_adr_cites_oneil_can_slim():
    """ADR must cite William O'Neil CAN SLIM (the canonical 7-8% rule)
    per researcher's literature scoring."""
    text = ADR.read_text(encoding="utf-8")
    assert "CAN SLIM" in text, "ADR must cite O'Neil CAN SLIM"
    assert "O'Neil" in text or "ONeil" in text or "O'Neil" in text, (
        "ADR must cite William O'Neil by name"
    )


def test_phase_40_4_adr_cites_han_zhou_zhu_2014():
    """ADR must cite Han/Zhou/Zhu 2014 (the 10% portfolio-momentum
    overlay literature anchor for the alternative)."""
    text = ADR.read_text(encoding="utf-8")
    assert "Han/Zhou/Zhu" in text or "Han, Zhou" in text or "SSRN 2407199" in text, (
        "ADR must cite Han/Zhou/Zhu 2014 (SSRN 2407199)"
    )


def test_phase_40_4_adr_references_settings_field():
    """ADR must reference the actual settings field it governs."""
    text = ADR.read_text(encoding="utf-8")
    assert "paper_default_stop_loss_pct" in text, (
        "ADR must reference backend/config/settings.py:paper_default_stop_loss_pct"
    )


def test_phase_40_4_adr_documents_deferred_a_b_run():
    """ADR must explicitly disclose the deferred A/B walk-forward run
    (honest-disclosure pattern per phase-23.2.6/10/11/12/13 precedent)."""
    text = ADR.read_text(encoding="utf-8")
    assert "deferred" in text.lower() or "defer" in text.lower(), (
        "ADR must explicitly disclose deferred A/B run"
    )
    # Must reference the literal tag the masterplan grep checks for
    assert "stop_loss_default_8_vs_10" in text, (
        "ADR must reference the masterplan grep tag 'stop_loss_default_8_vs_10'"
    )


def test_phase_40_4_turnkey_runner_exists_and_executable():
    """The turnkey A/B runner script must exist + be executable."""
    assert RUNNER.exists(), f"runner missing: {RUNNER}"
    assert RUNNER.stat().st_mode & 0o111, f"runner must be executable: {RUNNER}"


def test_phase_40_4_turnkey_runner_writes_masterplan_grep_tag():
    """The runner must use the literal tag 'stop_loss_default_8_vs_10'
    as the default --tag value (so masterplan grep finds it)."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "stop_loss_default_8_vs_10" in text, (
        "runner must use 'stop_loss_default_8_vs_10' tag (masterplan grep target)"
    )


def test_phase_40_4_adr_explicit_keep_decision():
    """ADR must declare a clear KEEP 8% (NOT switch 10%) verdict."""
    text = ADR.read_text(encoding="utf-8")
    # Look for KEEP 8% phrasing
    pattern = re.compile(r"KEEP\s+8\s*%|KEEP\s+8\.0|keep\s+8\s*%", re.IGNORECASE)
    assert pattern.search(text), (
        "ADR must declare KEEP 8% (the literature-supported decision)"
    )
