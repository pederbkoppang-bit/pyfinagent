"""phase-66.3 tests: cost-truth -- writer gauge-guard + token-derived metered spend.

Immutable verification command:
`python -m pytest backend/tests/test_phase_66_3_cost_truth.py -q`

Covers:
- criterion 1: failed 0-token calls never get the cumulative session-cost
  gauge stamped (they log 0.0); explicit values and billed paths untouched.
- criterion 2: metered figure derives from tokens x pinned prices over
  metered providers only; flat-fee rail/CLI rows excluded; unpriced metered
  models fail-visible; rail failure counts surfaced.
All BQ-free: buffer inspection + pure-function fixtures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "away_ops"))

import backend.services.observability.api_call_log as acl  # noqa: E402
from metered_spend import compute_metered, is_flat_fee  # noqa: E402


# ── criterion 1: writer gauge-guard ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_buffer(monkeypatch):
    # keep the lazy gauge-fill deterministic: simulate an active cycle at $0.50
    import backend.services.autonomous_loop as al

    monkeypatch.setattr(al, "_session_cost", 0.50, raising=False)
    monkeypatch.setattr(al, "_current_cycle_id", "cycle-t", raising=False)
    # phase-75.9: reset_llm_buffer_for_test() also re-arms
    # _llm_last_flush_ts -- a bare `_llm_buffer.clear()` left the
    # time-based auto-flush window stale, so a full-suite run older than
    # _FLUSH_SECONDS (60s) made the first log_llm_call() below trigger an
    # immediate flush that drained the row before _last_row() could read
    # it back (order-dependent: passed alone, failed in the full suite).
    acl.reset_llm_buffer_for_test()
    yield
    acl.reset_llm_buffer_for_test()


def _last_row():
    with acl._llm_lock:
        return acl._llm_buffer[-1]


def test_failed_zero_token_call_logs_zero_cost_not_gauge():
    acl.log_llm_call(provider="anthropic", model="claude-sonnet-4-6",
                     agent="cc_rail", ok=False)
    row = _last_row()
    assert row["session_cost_usd"] == 0.0  # NOT the 0.50 gauge
    assert row["cycle_id"] == "cycle-t"     # cycle context still stamped


def test_successful_call_still_carries_the_gauge():
    acl.log_llm_call(provider="gemini", model="gemini-2.5-flash",
                     agent="synthesis", ok=True, input_tok=100, output_tok=50)
    assert _last_row()["session_cost_usd"] == 0.50


def test_failed_call_with_tokens_keeps_gauge_midstream_may_bill():
    acl.log_llm_call(provider="gemini", model="gemini-2.5-flash",
                     agent="synthesis", ok=False, input_tok=500, output_tok=0)
    assert _last_row()["session_cost_usd"] == 0.50


def test_explicit_cost_always_wins():
    acl.log_llm_call(provider="anthropic", model="claude-sonnet-4-6",
                     agent="cc_rail", ok=False, session_cost_usd=0.42)
    assert _last_row()["session_cost_usd"] == 0.42


# ── criterion 2: token-derived metered figure ────────────────────────────


def _row(**kw):
    base = {"provider": "gemini", "model": "gemini-2.5-flash", "agent": "synthesis",
            "ok": True, "input_tok": 0, "output_tok": 0,
            "cache_creation_tok": 0, "cache_read_tok": 0}
    base.update(kw)
    return base


def test_gemini_tokens_price_correctly():
    # 1M in @ $0.30 + 1M out @ $2.50 = $2.80
    res = compute_metered([_row(input_tok=1_000_000, output_tok=1_000_000)])
    assert res["metered_llm_usd"] == pytest.approx(2.80)
    assert res["warnings"] == []


def test_rail_and_cli_rows_are_flat_fee_excluded():
    rows = [
        _row(provider="anthropic", model="claude-sonnet-4-6",
             agent="cc_rail:lite_trader", input_tok=1_000_000, output_tok=1_000_000),
        _row(provider="claude-code", model="claude-code-cli",
             agent=None, input_tok=1_000_000, output_tok=500_000),
    ]
    assert all(is_flat_fee(r) for r in rows)
    assert compute_metered(rows)["metered_llm_usd"] == 0.0


def test_non_rail_anthropic_api_rows_are_metered_with_cache_pricing():
    # haiku dated id prices via prefix match: 1M in @ $1 + 1M out @ $5
    # + 1M cache_read @ $0.10 + 1M cache_creation @ $1.25 = $7.35
    res = compute_metered([_row(provider="anthropic",
                                model="claude-haiku-4-5-20251001",
                                agent="ticket_agent",
                                input_tok=1_000_000, output_tok=1_000_000,
                                cache_read_tok=1_000_000,
                                cache_creation_tok=1_000_000)])
    assert res["metered_llm_usd"] == pytest.approx(7.35)


def test_unpriced_metered_model_counts_zero_and_warns():
    res = compute_metered([_row(model="mystery-model-9", input_tok=2_000_000)])
    assert res["metered_llm_usd"] == 0.0
    assert res["unpriced_models"] == ["mystery-model-9"]
    assert any("unpriced" in w for w in res["warnings"])


def test_rail_failures_counted_as_first_class_signal():
    rows = [_row(provider="anthropic", model="claude-sonnet-4-6",
                 agent="cc_rail", ok=False) for _ in range(25)]
    res = compute_metered(rows)
    assert res["rail_failures"] == 25
    assert res["metered_llm_usd"] == 0.0


def test_zero_movement_rows_bill_nothing():
    res = compute_metered([_row(ok=False)])  # failed, 0 tokens
    assert res["metered_llm_usd"] == 0.0 and res["warnings"] == []
