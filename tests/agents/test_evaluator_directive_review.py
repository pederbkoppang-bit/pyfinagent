"""phase-10.7.7 unit tests for the evaluator review gate on directive diffs.

Mock-injection pattern via `llm_call_override` (mirrors
`directive_rewriter.rewrite_directive` test pattern). No live LLM calls;
no API cost. Fail-CLOSED semantics: any LLM failure / missing field /
out-of-range score MUST return REJECT (not ACCEPT).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.directive_rewriter import DirectiveVersion  # noqa: E402
from backend.meta_evolution.directive_review import (  # noqa: E402
    ACCEPT_THRESHOLD,
    RUBRIC_DIMENSIONS,
    ReviewResult,
    review_directive_diff,
)


CURRENT_TEXT = (
    "# Researcher\n\nResearch directive: read 5 sources in full. Recency scan required."
)


def _proposal(
    *,
    proposed_text: str = "# Researcher\n\nResearch directive: read 5 sources in full. Recency scan required. Cite per claim.",
    diff_summary: str = "added 'Cite per claim' clause",
    judge_score: float | None = 0.85,
) -> DirectiveVersion:
    """Build a DirectiveVersion with sensible defaults for tests."""
    return DirectiveVersion(
        version_id="rev-test-001",
        parent_version_id=None,
        proposed_text=proposed_text,
        diff_summary=diff_summary,
        diff_size_bytes=abs(len(proposed_text.encode("utf-8")) - len(CURRENT_TEXT.encode("utf-8"))),
        judge_score=judge_score,
        proposed_at=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )


# ----------------------
# Verdict tests
# ----------------------

def test_accept_on_high_scores():
    """All 5 dims >= 0.75 -> aggregate >= 0.75 >= 0.70 -> ACCEPT."""
    high = {
        "clarity": 0.85,
        "alignment": 0.80,
        "safety": 0.90,
        "proportionality": 0.75,
        "factuality": 0.80,
        "reason": "tight, targeted change preserving guardrails",
    }
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: high
    )
    assert out.verdict == "ACCEPT"
    assert out.aggregate_score >= ACCEPT_THRESHOLD


def test_reject_on_low_aggregate():
    """Mean = 0.55 -> REJECT."""
    low = {
        "clarity": 0.55,
        "alignment": 0.55,
        "safety": 0.55,
        "proportionality": 0.55,
        "factuality": 0.55,
        "reason": "lukewarm",
    }
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: low
    )
    assert out.verdict == "REJECT"
    assert out.aggregate_score == pytest.approx(0.55)


def test_accept_threshold_boundary_exact():
    """Mean = 0.70 -> ACCEPT; mean = 0.699 -> REJECT."""
    on_threshold = {dim: 0.70 for dim in RUBRIC_DIMENSIONS}
    on_threshold["reason"] = "exactly at threshold"
    out_eq = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: on_threshold
    )
    assert out_eq.verdict == "ACCEPT"
    assert out_eq.aggregate_score == pytest.approx(0.70)

    just_under = {dim: 0.699 for dim in RUBRIC_DIMENSIONS}
    just_under["reason"] = "fractionally below"
    out_lt = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: just_under
    )
    assert out_lt.verdict == "REJECT"


# ----------------------
# Fail-CLOSED tests (the key safety guarantee)
# ----------------------

def test_reject_on_llm_none_fail_closed():
    """LLM returns None -> REJECT, all scores 0.0."""
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: None
    )
    assert out.verdict == "REJECT"
    assert out.reason == "llm_error_fail_closed"
    assert out.aggregate_score == 0.0
    assert out.clarity_score == 0.0


def test_reject_on_invalid_json_fail_closed():
    """LLM returns garbage (non-dict) -> REJECT (fail-closed)."""
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: "not a dict"
    )
    assert out.verdict == "REJECT"
    assert out.reason == "llm_error_fail_closed"


def test_reject_on_override_exception_fail_closed():
    """If the override raises, the gate must fail-closed (NOT propagate)."""

    def boom(p):
        raise RuntimeError("simulated llm crash")

    out = review_directive_diff(_proposal(), CURRENT_TEXT, llm_call_override=boom)
    assert out.verdict == "REJECT"
    assert out.reason == "llm_error_fail_closed"


def test_reject_on_missing_dimension():
    """LLM omits one rubric dimension -> REJECT, score 0.0."""
    incomplete = {
        "clarity": 0.9,
        "alignment": 0.9,
        "safety": 0.9,
        "proportionality": 0.9,
        # 'factuality' missing
        "reason": "missing one dim",
    }
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: incomplete
    )
    assert out.verdict == "REJECT"
    assert "factuality" in out.reason


def test_reject_on_out_of_range_score():
    """Score > 1.0 is rejected as invalid (fail-closed)."""
    bad = {dim: 0.8 for dim in RUBRIC_DIMENSIONS}
    bad["safety"] = 1.5  # out of range
    bad["reason"] = "bogus"
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: bad
    )
    assert out.verdict == "REJECT"
    assert "safety" in out.reason


def test_missing_proposed_text_reject():
    """Empty proposed_text -> REJECT before any LLM call."""
    spy_called = {"n": 0}

    def spy(p):
        spy_called["n"] += 1
        return {dim: 1.0 for dim in RUBRIC_DIMENSIONS}

    out = review_directive_diff(
        _proposal(proposed_text=""), CURRENT_TEXT, llm_call_override=spy
    )
    assert out.verdict == "REJECT"
    assert out.reason == "empty_proposed_text"
    assert spy_called["n"] == 0  # short-circuited before LLM


# ----------------------
# Anti-rubber-stamp / prompt-discipline tests
# ----------------------

def test_proposer_self_score_stripped_from_prompt():
    """The proposer's `judge_score` numeric value MUST NOT appear in the judge prompt."""
    captured: dict = {}

    def capture(prompt: str):
        captured["prompt"] = prompt
        return {dim: 0.8 for dim in RUBRIC_DIMENSIONS} | {"reason": "ok"}

    proposal = _proposal(judge_score=0.95)
    review_directive_diff(proposal, CURRENT_TEXT, llm_call_override=capture)

    # The literal "0.95" (the proposer's self-score) must not appear in
    # the judge's prompt -- otherwise the judge could anchor on it.
    assert "0.95" not in captured["prompt"]
    assert "judge_score" not in captured["prompt"]


def test_current_text_present_in_prompt():
    """The CURRENT directive text must be included so the judge has the diff context."""
    captured: dict = {}

    def capture(prompt: str):
        captured["prompt"] = prompt
        return {dim: 0.8 for dim in RUBRIC_DIMENSIONS} | {"reason": "ok"}

    review_directive_diff(_proposal(), CURRENT_TEXT, llm_call_override=capture)
    assert CURRENT_TEXT in captured["prompt"]
    assert _proposal().proposed_text in captured["prompt"]
    assert _proposal().diff_summary in captured["prompt"]


# ----------------------
# Determinism / data-shape tests
# ----------------------

def test_idempotent_same_proposal_same_verdict():
    """Same proposal + same override = identical ReviewResult."""
    payload = {dim: 0.80 for dim in RUBRIC_DIMENSIONS} | {"reason": "deterministic"}
    a = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: dict(payload)
    )
    b = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: dict(payload)
    )
    assert a == b
    assert a.verdict == "ACCEPT"


def test_review_result_stores_raw_llm_response():
    """raw_llm_response must be the parsed dict for forensic logging."""
    payload = {dim: 0.80 for dim in RUBRIC_DIMENSIONS} | {"reason": "trace this"}
    out = review_directive_diff(
        _proposal(), CURRENT_TEXT, llm_call_override=lambda p: dict(payload)
    )
    assert isinstance(out.raw_llm_response, dict)
    assert out.raw_llm_response["reason"] == "trace this"
    assert out.raw_llm_response["clarity"] == 0.80
