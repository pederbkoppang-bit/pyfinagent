"""phase-3.4 tests for the SkillOptimizer.

Focus on the deterministic helpers and constants; DO NOT instantiate
`SkillOptimizer` (its __init__ calls BigQueryClient + OutcomeTracker +
model loading which require auth). The autoresearch loop body is covered
by integration tests elsewhere; these tests lock down the small pure
units that hold the loop's invariants.

Coverage:
 1. `passes_simplicity_criterion` -- simplifications always pass when
    delta >= 0; added lines require proportional improvement.
 2. `_extract_json` isolates JSON from `` ```json ``` `` code fences.
 3. `_extract_json` isolates raw JSON embedded in prose.
 4. `_extract_json` returns None when no JSON is present.
 5. `iteration_counter` round-robins modulo N.
 6. `OPTIMIZABLE_AGENTS` non-empty + unique.
 7. `TSV_HEADER` matches the column order the writer uses.
 8. `_get_short_hash` returns "no-git" on git failure (fail-open).
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from backend.agents.skill_optimizer import (
    OPTIMIZABLE_AGENTS,
    SkillOptimizer,
    TSV_HEADER,
    _extract_json,
    _get_short_hash,
    iteration_counter,
)


# ---------- 1. Simplicity criterion ----------


def test_simplicity_criterion_simplification_passes_when_delta_nonnegative():
    # 5 old lines -> 2 new lines; delta=0.0 (no harm). Simplifications always pass.
    proposal = {"old_text": "a\nb\nc\nd\ne", "new_text": "a\nb"}
    assert SkillOptimizer.passes_simplicity_criterion(proposal, 0.0) is True


def test_simplicity_criterion_added_lines_require_proportional_delta():
    # 1 old -> 21 new (+20 lines). Required delta = 0.005 * 20/10 = 0.01.
    proposal = {"old_text": "a", "new_text": "a\n" * 20 + "b"}
    assert SkillOptimizer.passes_simplicity_criterion(proposal, 0.009) is False  # below gate
    assert SkillOptimizer.passes_simplicity_criterion(proposal, 0.010) is True   # at gate
    assert SkillOptimizer.passes_simplicity_criterion(proposal, 0.050) is True   # well above


def test_simplicity_criterion_simplification_with_negative_delta_fails():
    # Fewer lines AND negative delta -- no free lunch; drop the change.
    proposal = {"old_text": "a\nb\nc\nd\ne", "new_text": "a"}
    assert SkillOptimizer.passes_simplicity_criterion(proposal, -0.01) is False


# ---------- 2-4. _extract_json ----------


def test_extract_json_from_fenced_code_block():
    text = 'Thinking...\n\n```json\n{"a": 1, "b": [2, 3]}\n```\nDone.'
    out = _extract_json(text)
    assert out is not None
    assert '"a"' in out
    assert '"b"' in out


def test_extract_json_from_raw_prose():
    text = 'Here is the payload: {"proposal": "raise ma_short to 10"} end.'
    out = _extract_json(text)
    assert out is not None
    assert out.startswith("{")
    assert out.endswith("}")


def test_extract_json_returns_none_on_prose_only():
    assert _extract_json("just some thoughts, no JSON here at all") is None


def test_extract_json_array_form():
    text = "Result: [1, 2, 3]"
    out = _extract_json(text)
    assert out == "[1, 2, 3]"


# ---------- 5. iteration_counter round-robin ----------


def test_iteration_counter_round_robins():
    import backend.agents.skill_optimizer as so

    # Reset the module-level counter for determinism.
    so._iteration_counter = 0
    mod = 5
    seen = [iteration_counter(mod) for _ in range(12)]
    # With mod=5 we expect 0,1,2,3,4,0,1,2,3,4,0,1
    assert seen == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


# ---------- 6-7. Constants ----------


def test_optimizable_agents_non_empty_and_unique():
    assert len(OPTIMIZABLE_AGENTS) > 0
    assert len(set(OPTIMIZABLE_AGENTS)) == len(OPTIMIZABLE_AGENTS)
    # Every entry is a non-empty string
    for a in OPTIMIZABLE_AGENTS:
        assert isinstance(a, str) and a


def test_tsv_header_has_expected_columns_in_order():
    assert TSV_HEADER == [
        "timestamp", "commit", "agent", "metric_before", "metric_after",
        "delta", "status", "description",
    ]


# ---------- 8. _get_short_hash fail-open ----------


def test_get_short_hash_fail_open_on_git_failure():
    def _fail(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=1, stdout="", stderr="not a git repo"
        )

    with patch("subprocess.run", side_effect=_fail):
        # Our subprocess.run stub returns a CompletedProcess with nonzero
        # returncode; _git should raise RuntimeError and _get_short_hash
        # should catch it, returning "no-git".
        result = _get_short_hash()
    assert result == "no-git"
