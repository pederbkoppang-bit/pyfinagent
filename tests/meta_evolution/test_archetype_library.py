"""phase-10.7.3 unit tests for the Algorithm Discovery archetype seed library.

7 cases:
1. ARCHETYPES tuple length is exactly 6 (matches immutable verification)
2. All strategy_ids are unique
3. Required string fields (name, description, directive_template) non-empty
4. default_params has >= 2 keys for every archetype
5. Every is_implemented=True archetype's strategy_id is in IMPLEMENTED_STRATEGY_IDS
6. expected_regime values are all in ALLOWED_REGIMES
7. The forward-declaration archetype (sentiment_event_driven) has is_implemented=False

No external deps; pure Python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.meta_evolution.archetype_library import (  # noqa: E402
    ALLOWED_REGIMES,
    ARCHETYPES,
    IMPLEMENTED_STRATEGY_IDS,
    Archetype,
    get_archetype,
)


def test_archetypes_count():
    assert len(ARCHETYPES) == 6, (
        f"Expected exactly 6 archetypes (matches immutable verification "
        f"command), got {len(ARCHETYPES)}"
    )


def test_strategy_ids_unique():
    ids = [a.strategy_id for a in ARCHETYPES]
    assert len(ids) == len(set(ids)), f"Duplicate strategy_id values: {ids}"


def test_required_fields_non_empty():
    for arch in ARCHETYPES:
        assert arch.name and arch.name.strip(), f"empty name for {arch.strategy_id}"
        assert arch.description and arch.description.strip(), (
            f"empty description for {arch.strategy_id}"
        )
        assert arch.directive_template and arch.directive_template.strip(), (
            f"empty directive_template for {arch.strategy_id}"
        )
        # directive_template MUST contain a placeholder so the 10.7.2 rewriter
        # can substitute name or strategy_id
        assert (
            "{name}" in arch.directive_template
            or "{strategy_id}" in arch.directive_template
        ), (
            f"directive_template for {arch.strategy_id} missing placeholder"
        )


def test_default_params_non_empty():
    for arch in ARCHETYPES:
        assert len(arch.default_params) >= 2, (
            f"{arch.strategy_id} has only {len(arch.default_params)} "
            f"default_params keys; QuantEvolve seed-quality requires >= 2"
        )


def test_implemented_ids_in_registry():
    for arch in ARCHETYPES:
        if arch.is_implemented:
            assert arch.strategy_id in IMPLEMENTED_STRATEGY_IDS, (
                f"{arch.strategy_id} flagged is_implemented=True but not in "
                f"IMPLEMENTED_STRATEGY_IDS={sorted(IMPLEMENTED_STRATEGY_IDS)}; "
                f"this would cause silent fallback to triple_barrier"
            )


def test_expected_regime_valid():
    for arch in ARCHETYPES:
        assert arch.expected_regime in ALLOWED_REGIMES, (
            f"{arch.strategy_id} has expected_regime={arch.expected_regime!r} "
            f"not in ALLOWED_REGIMES={sorted(ALLOWED_REGIMES)}"
        )


def test_sixth_archetype_is_forward_declaration():
    """The 6th archetype is sentiment_event_driven, a forward-declaration."""
    sixth = ARCHETYPES[5]
    assert sixth.strategy_id == "sentiment_event_driven", (
        f"Expected 6th archetype to be sentiment_event_driven, got "
        f"{sixth.strategy_id}"
    )
    assert sixth.is_implemented is False, (
        "sentiment_event_driven must be flagged is_implemented=False; "
        "the engine has no label method for it yet"
    )


# Bonus: validate constructor guards (these are not in the 7-test plan but
# verify __post_init__ catches bad seeds at import time -- pytest collection
# itself proves the live ARCHETYPES tuple has zero validation errors).
def test_constructor_rejects_empty_strategy_id():
    with pytest.raises(ValueError, match="strategy_id"):
        Archetype(
            strategy_id="",
            name="x",
            description="x",
            default_params={"a": 1, "b": 2},
            expected_regime="ALL",
            directive_template="x {name}",
        )


def test_constructor_rejects_implemented_unknown_id():
    with pytest.raises(ValueError, match="not in IMPLEMENTED_STRATEGY_IDS"):
        Archetype(
            strategy_id="bogus_strategy",
            name="Bogus",
            description="x",
            default_params={"a": 1, "b": 2},
            expected_regime="ALL",
            directive_template="x {strategy_id}",
            is_implemented=True,
        )


def test_get_archetype_lookup():
    arch = get_archetype("triple_barrier")
    assert arch is not None
    assert arch.strategy_id == "triple_barrier"
    assert get_archetype("does_not_exist") is None
