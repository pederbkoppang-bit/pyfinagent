"""phase-75.4.2 -- skill_optimizer post-write delivery invariant.

phase-75.4 (test_phase_75_skill_delivery.py) closed the loader-side truncation
bug for the skill files that exist TODAY. This step closes the write side: the
autonomous `SkillOptimizer.apply_modification` can re-introduce that exact
truncation (e.g. promoting a `### ` heading back to `## `) at any future
autonomous run, and its ONLY post-write check before this step was "does
`load_skill()` raise" -- which a truncated-but-still-parseable template
passes cleanly. `apply_modification` now also asserts a DELIVERY invariant:
the delivered template (what `load_skill` returns, matching what the model
actually receives) must not drop a `{{placeholder}}` and must not shrink
below 80% of its pre-write length. A violation reverts the file byte-exactly
and the write is never committed.

TEST DOCTRINE (phase-75 durable rule, harness_log.md Cycle 130 + carried
forward by 75.4's test suite): a guard that cannot fail does not count. Every
assertion here drives the REAL `SkillOptimizer.apply_modification` against a
REAL temporary copy of `quant_model_agent.md` -- never a string stub -- and
every fixture asserts `content.count(old_text) == 1` immediately before use,
because the pre-existing occurs-exactly-once guard (skill_optimizer.py:465)
would otherwise reject an ambiguous `old_text` for the WRONG reason and mask
whether the new invariant ever ran (the exact vacuous-test trap the
research brief flags for `{{quant_model_data}}`, which occurs twice in this
file -- once in `## Data Inputs` prose, once in the template).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from backend.agents import skill_optimizer
from backend.agents.skill_optimizer import SkillOptimizer
from backend.config import prompts

REPO = Path(__file__).resolve().parents[2]
FIXTURE_SKILL = REPO / "backend/agents/skills/quant_model_agent.md"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def optimizer_env(tmp_path, monkeypatch):
    """Point `apply_modification` + `load_skill` at an isolated temp copy of
    the REAL `quant_model_agent.md` skill file (never a string stub).

    `apply_modification` reads/writes via `skill_optimizer.SKILLS_DIR`; the
    delivery check goes through `prompts.load_skill`, which reads
    `SKILLS_DIR` from ITS OWN module namespace (`backend.config.prompts`,
    where it is defined) -- both must be patched or the two halves of
    apply_modification would look at different directories.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skill_path = skills_dir / "quant_model_agent.md"
    shutil.copy(FIXTURE_SKILL, skill_path)

    monkeypatch.setattr(skill_optimizer, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(prompts, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(prompts, "_SKILL_FILE_ID_CACHE_PATH", tmp_path / ".skill_file_ids.json")
    monkeypatch.setattr(skill_optimizer, "_git", lambda *a, **k: "")

    # Hermetic cache: `prompts._skill_cache` is a module-level dict keyed only
    # by agent_name (not by path), so a leftover entry from an earlier test
    # could in principle be served here. reload_skills() is a pure cache
    # clear with no args -- safe to call in a fixture.
    prompts.reload_skills()

    optimizer = SkillOptimizer.__new__(SkillOptimizer)

    def call(old_text: str, new_text: str) -> bool:
        current = skill_path.read_text(encoding="utf-8")
        assert current.count(old_text) == 1, (
            f"fixture drift: old_text must occur exactly once in the skill "
            f"file, occurs {current.count(old_text)} times -- pick a unique "
            f"anchor or the pre-existing occurs-once guard "
            f"(skill_optimizer.py) will reject this for the WRONG reason "
            f"and the test would prove nothing about the delivery invariant"
        )
        return optimizer.apply_modification(
            "quant_model_agent",
            {"old_text": old_text, "new_text": new_text, "description": "test"},
        )

    return skill_path, call


# ── criterion 1: heading promotion is reverted ───────────────────────────────

def test_heading_promotion_reverted_and_file_byte_identical(optimizer_env):
    """Promoting the in-template '### Quant Model Data' heading back to
    '## Quant Model Data' is the exact gap5-01 regression: `load_skill`'s
    region regex (prompts.py:196-197) stops at the first subsequent '## '
    heading, so this trips BOTH invariant guards (drops {{quant_model_data}}
    AND truncates ~2500 chars of delivered body). apply_modification must
    return False and leave the file untouched."""
    skill_path, call = optimizer_env
    before_bytes = skill_path.read_bytes()
    before_sha = _sha256(before_bytes)

    result = call("### Quant Model Data", "## Quant Model Data")

    assert result is False, "heading-promotion truncation was not reverted"
    after_bytes = skill_path.read_bytes()
    assert after_bytes == before_bytes, "file bytes changed despite a reverted write"
    after_sha = _sha256(after_bytes)
    assert after_sha == before_sha, (
        f"sha256 mismatch after revert: before={before_sha} after={after_sha}"
    )


# ── criterion 2: placeholder drop is reverted, byte-identical ───────────────

def test_placeholder_drop_reverted_byte_identical(optimizer_env):
    """`{{quant_model_data}}` occurs TWICE in the raw file (once as
    documentation in '## Data Inputs', once as the real template
    placeholder in '## Prompt Template'). A bare `old_text` of just the
    placeholder would be rejected by the PRE-EXISTING occurs-once guard
    (skill_optimizer.py, not the new invariant) -- a vacuous test that
    would stay green even if the new invariant were deleted. Use the
    unique 2-line anchor that includes the heading, so the occurs-once
    guard passes and only the delivery invariant can catch the drop."""
    skill_path, call = optimizer_env
    before_bytes = skill_path.read_bytes()
    before_sha = _sha256(before_bytes)

    old_text = "### Quant Model Data\n{{quant_model_data}}"
    new_text = "### Quant Model Data\nthe provided factor data"

    result = call(old_text, new_text)

    assert result is False, "placeholder-dropping write was not reverted"
    after_bytes = skill_path.read_bytes()
    assert after_bytes == before_bytes, "file bytes changed despite a reverted write"
    after_sha = _sha256(after_bytes)
    assert after_sha == before_sha, (
        f"sha256 mismatch after revert: before={before_sha} after={after_sha}"
    )


# ── criterion 3: negative control -- legitimate prose edit is accepted ──────

def test_prose_only_edit_is_accepted_negative_control(optimizer_env):
    """The guard must not blanket-refuse every write. A same-shape prose
    edit inside '### Instructions' that touches no heading and no
    placeholder must be ACCEPTED, written to disk, and still deliver every
    placeholder the original template had."""
    skill_path, call = optimizer_env
    before_bytes = skill_path.read_bytes()
    before_sha = _sha256(before_bytes)

    old_text = "1. Assess the overall factor signal (score direction and magnitude)"
    new_text = "1. Assess the overall factor signal, covering both its direction and its magnitude"

    result = call(old_text, new_text)

    assert result is True, "legitimate prose-only edit was rejected (blanket-refusal)"
    after_bytes = skill_path.read_bytes()
    assert after_bytes != before_bytes, "file was not actually written on an accepted edit"
    assert _sha256(after_bytes) != before_sha

    delivered = prompts.load_skill("quant_model_agent")
    assert new_text in delivered
    assert "{{quant_model_data}}" in delivered, "guard silently dropped a surviving placeholder"
    assert "{{fact_ledger_section}}" in delivered
    assert "{{ticker}}" in delivered


# ── length-retention guard, isolated from the placeholder guard ─────────────

def test_length_only_truncation_reverted(optimizer_env):
    """Truncate a large placeholder-free stretch (the '### Code Execution
    Tasks' body, ~919 chars of the ~2739-char delivered template) down to a
    one-line stub while keeping every {{placeholder}} untouched. This must
    trip ONLY the length-retention guard (delivered length drops to ~70% of
    its prior length, below DELIVERY_MIN_RETAIN_RATIO=0.80) -- pinning that
    guard independently of the placeholder-subset guard exercised by T1/T2."""
    skill_path, call = optimizer_env
    before_bytes = skill_path.read_bytes()
    before_sha = _sha256(before_bytes)

    delivered_before = prompts.load_skill("quant_model_agent")
    assert len(delivered_before) > 2500, (
        f"fixture drift: expected a large delivered template, got "
        f"{len(delivered_before)} chars -- re-derive the truncation size"
    )

    old_text = (
        "### Code Execution Tasks (phase-26.3)\n\n"
        "When the Gemini `code_execution` tool is available (it is wired on "
        "`quant_exec_client`), USE IT to verify your arithmetic. Specifically, "
        "before finalizing the signal:\n\n"
        "1. **Verify composite score.** Recompute the composite from input "
        "factor weights and confirm it matches `quant_model_data.score` "
        "within float tolerance of 1e-6. If they diverge, surface the "
        "discrepancy.\n"
        "2. **Verify Sharpe arithmetic.** When the input includes "
        "`mean_return` and `std_return`, compute "
        "`sharpe = mean_return / std_return` (or "
        "`(mean_return - rf) / std_return` if `rf` is given) and confirm "
        "against any provided Sharpe value.\n"
        "3. **Position-sizing bound check.** Compute "
        "`assert 0.0 <= position_size_pct <= 100.0`. Flag violations "
        "explicitly.\n\n"
        "These checks ELIMINATE silent arithmetic drift (the \"model says "
        "0.42 when the math is 0.24\" class of bug). Do NOT freestyle the "
        "math; run code to verify it."
    )
    new_text = "### Code Execution Tasks (phase-26.3)\n\nVerify arithmetic before finalizing the signal."
    assert "{{" not in old_text and "{{" not in new_text, (
        "this fixture must stay placeholder-free to isolate the length guard"
    )

    result = call(old_text, new_text)

    assert result is False, "length-only truncation was not reverted"
    after_bytes = skill_path.read_bytes()
    assert after_bytes == before_bytes, "file bytes changed despite a reverted write"
    assert _sha256(after_bytes) == before_sha
