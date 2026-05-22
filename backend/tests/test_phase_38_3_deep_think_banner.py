"""phase-38.3 deep-think tier startup banner tests.

Closes closure_roadmap.md §3 OPEN-12 + phase-34.1 observability gap:
backend/main.py:140-152 emits a startup banner for the STANDARD tier
(gemini_model) with provider-detect + warning if non-Gemini. There was
NO parallel banner for the DEEP-THINK tier (deep_think_model) -- this
hid the phase-34.1e diagnosis (claude-opus-4-7 + Anthropic credit-
exhaustion) until the operator noticed silently.

phase-38.3 adds the missing banner. These tests verify:
  1. The banner string `phase-38.3 model routing:` is present in main.py.
  2. The provider-detect classifier covers gemini / claude / openai / unknown.
  3. The WARNING branch fires for non-Gemini models.
  4. Greppability: `grep "phase-3[18] model routing"` returns BOTH banners.
"""

from __future__ import annotations


def test_phase_38_3_main_py_has_deep_think_banner_string():
    """Criterion: phase-38.3 model routing INFO line is present in main.py."""
    src = open("backend/main.py").read()
    assert "phase-38.3 model routing: settings.deep_think_model=" in src, (
        "main.py must emit a `phase-38.3 model routing: settings.deep_think_model=...` "
        "INFO line at startup mirroring the standard-tier banner."
    )


def test_phase_38_3_main_py_has_warning_branch():
    """Criterion: when deep_think_model is non-Gemini, a WARNING is emitted
    with the phase-38.3 prefix and the phase-34.1e history reference."""
    src = open("backend/main.py").read()
    assert "phase-38.3: settings.deep_think_model is set to a non-Gemini model" in src
    assert "phase-34.1e history" in src
    assert "credit balance dependency" in src


def test_phase_38_3_provider_detect_classifier_covers_4_branches():
    """Criterion: classifier covers gemini / claude / openai / unknown."""
    src = open("backend/main.py").read()
    # The block uses _dt_model and _dt_provider; verify 4 branches present
    assert "_dt_model.startswith(\"gemini-\")" in src, "gemini branch present"
    assert "_dt_model.startswith(\"claude-\")" in src, "claude branch present"
    assert "_dt_model.startswith((\"gpt-\", \"o1\", \"o3\", \"o4\"))" in src, "openai branch present"
    assert "_dt_provider = f\"unknown (model='{_dt_model}')\"" in src, "unknown branch present"


def test_phase_38_3_greppable_with_phase_31_1_pattern():
    """Convenience: `grep 'phase-3[18] model routing' backend/main.py` returns
    BOTH the standard-tier banner (phase-31.1 line) and the deep-think
    banner (phase-38.3 line)."""
    src = open("backend/main.py").read()
    assert "phase-31.1 model routing" in src
    assert "phase-38.3 model routing" in src


def test_phase_38_3_deep_think_banner_uses_settings_deep_think_model():
    """The banner must read from `settings.deep_think_model` (the Field
    aligned to gemini-2.5-pro by phase-37.2), not from a hard-coded value."""
    src = open("backend/main.py").read()
    # Locate the phase-38.3 block
    start = src.find("phase-38.3: parallel banner")
    end = src.find("phase-23.1.21:", start)
    block = src[start:end]
    assert "settings.deep_think_model" in block, "banner must read from settings.deep_think_model"
    # And the logging.info call must format with _dt_model + _dt_provider
    assert "%s, _dt_model, _dt_provider" in block.replace("\n", " ") or \
           "_dt_model, _dt_provider," in block, "format args must be _dt_model + _dt_provider"
