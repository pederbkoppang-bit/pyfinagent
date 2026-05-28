"""phase-47.3: regression guard for Claude Opus 4.8 cost-tracking pricing.

Commit 8ecc9efe bumped model_tiers 4.7->4.8 but the pricing tables were not
updated, so claude-opus-4-8 fell through to _DEFAULT_PRICING (0.10, 0.40) -- a
~50x input / ~62.5x output understatement that corrupts the Compute term of
Net System Alpha. These guards fail if the 4.8 entry is dropped or mis-priced.
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.agents.cost_tracker import (
    MODEL_PRICING,
    _DEFAULT_PRICING,
    CostTracker,
)


def test_opus_4_8_priced_same_as_4_7_and_not_default():
    o8 = MODEL_PRICING.get("claude-opus-4-8")
    assert o8 is not None, "claude-opus-4-8 missing from MODEL_PRICING"
    assert o8 == (5.00, 25.00), f"4.8 pricing should be (5.00, 25.00), got {o8}"
    assert o8 == MODEL_PRICING["claude-opus-4-7"], "4.8 should match 4.7 pricing"
    assert o8 != _DEFAULT_PRICING, "4.8 must NOT fall through to _DEFAULT_PRICING"


def _fake_response(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=input_tokens + output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
    )


def test_opus_4_8_recorded_cost_uses_real_rate_not_default():
    """Behavioral: record() must cost a 4.8 call at $5/$25, not the 50x-low default."""
    tracker = CostTracker()
    entry = tracker.record(
        agent_name="Test Agent",
        model="claude-opus-4-8",
        response=_fake_response(1_000_000, 1_000_000),
    )
    assert entry is not None, "record() returned None for a valid usage_metadata response"
    # (1M in * $5 + 1M out * $25) / 1e6 = $30.00
    assert round(entry.cost_usd, 2) == 30.00, f"expected 30.00, got {entry.cost_usd}"
    # The 50x-too-low default would have produced $0.50 -- prove we are NOT on it.
    default_cost = (1_000_000 * _DEFAULT_PRICING[0] + 1_000_000 * _DEFAULT_PRICING[1]) / 1_000_000
    assert round(entry.cost_usd, 2) != round(default_cost, 2), \
        "4.8 cost must NOT equal the default-priced cost"
