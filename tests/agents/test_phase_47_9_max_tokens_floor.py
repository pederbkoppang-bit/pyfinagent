"""phase-47.9: guards for the Opus-4.8 max_tokens-at-xhigh floor + driver-pin finish.

Priority-3 completion. Two fixes:

1. **Adaptive max_tokens floor.** On the Opus-4.8/4.7 adaptive thinking path
   (`multi_agent_orchestrator.py:1061` IF-branch), `max_tokens` is a HARD ceiling
   on thinking + visible text COMBINED (Anthropic adaptive-thinking doc). Layer-2
   agents run at `effort=max` but the call set `max_tokens=agent_config.max_tokens
   + 2048` (~2548-5048 for the 500-3000 configured agents) -> adaptive thinking
   could exhaust that and silently starve the answer. `_adaptive_max_tokens` now
   floors the ceiling so thinking always has headroom; the non-adaptive ELSE
   branch (manual budget_tokens=2048, thinking bounded) is left unchanged.

2. **Driver-pin finish.** Three stale `claude-opus-4-6` pins the backend-only
   47.8 sweep didn't reach are bumped to 4-8; PlannerAgent (which the driver puts
   on 4-8) has its fragile `response.content[0].text` hardened to tolerate a
   leading thinking block via `_first_text`.

These guards fail if the floor regresses, the floored ceiling stops being used at
the create/retry sites, a driver pin reverts to 4-6, or the planner parse stops
tolerating a thinking-block-first response.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Criterion 1 -- adaptive max_tokens floor (BEHAVIORAL on the pure helper)
# --------------------------------------------------------------------------
def test_adaptive_max_tokens_floors_low_and_respects_high():
    from backend.agents.multi_agent_orchestrator import (
        _OPUS_ADAPTIVE_MIN_MAX_TOKENS,
        _adaptive_max_tokens,
    )

    assert _OPUS_ADAPTIVE_MIN_MAX_TOKENS == 16384
    # small configured budgets are raised to the floor (the starvation fix)
    assert _adaptive_max_tokens(500) == 16384
    assert _adaptive_max_tokens(3000) == 16384
    assert _adaptive_max_tokens(4096) == 16384  # largest intended (Synthesis)
    # boundary: configured+2048 == floor
    assert _adaptive_max_tokens(14336) == 16384
    # large configured budgets are respected (+2048 thinking headroom)
    assert _adaptive_max_tokens(30000) == 32048
    # it is always a CEILING >= the configured output budget
    for c in (500, 3000, 4096, 30000):
        assert _adaptive_max_tokens(c) >= c


def test_orchestrator_uses_the_floor_on_adaptive_branch_only():
    src = (REPO / "backend/agents/multi_agent_orchestrator.py").read_text(encoding="utf-8")
    # adaptive (Opus) branch floors via the helper
    assert "_max_tokens = _adaptive_max_tokens(agent_config.max_tokens)" in src
    # the create call uses the floored variable, not the old inline +2048
    assert "max_tokens=_max_tokens," in src
    assert "max_tokens=agent_config.max_tokens + 2048," not in src
    # the non-adaptive ELSE branch is unchanged (manual budget bounded)
    assert "_max_tokens = agent_config.max_tokens + 2048" in src
    # the tool_use retry is based on the floored ceiling, above the floor
    assert "_retry_max = min(_max_tokens * 2, 32768)" in src


# --------------------------------------------------------------------------
# Criterion 2 -- the three 4-6 driver pins bumped to 4-8
# --------------------------------------------------------------------------
def test_run_autonomous_loop_planner_pin_is_4_8():
    src = (REPO / "scripts/harness/run_autonomous_loop.py").read_text(encoding="utf-8")
    assert 'planner_model="claude-opus-4-8"' in src
    assert "claude-opus-4-6" not in src


def test_run_cycle_sh_model_flag_is_4_8():
    src = (REPO / "scripts/mas_harness/run_cycle.sh").read_text(encoding="utf-8")
    assert "--model claude-opus-4-8" in src
    assert "claude-opus-4-6" not in src


# --------------------------------------------------------------------------
# Criterion 3 -- PlannerAgent parse is thinking-block tolerant
# --------------------------------------------------------------------------
def test_first_text_skips_leading_thinking_block():
    from backend.agents.planner_agent import _first_text

    # Opus 4.8 can emit a thinking block BEFORE the text block -> content[0]
    # is not the text. _first_text must still return the text.
    resp = SimpleNamespace(content=[
        SimpleNamespace(type="thinking", thinking="...internal reasoning..."),
        SimpleNamespace(type="text", text="the answer"),
    ])
    assert _first_text(resp) == "the answer"


def test_first_text_joins_multiple_text_blocks():
    from backend.agents.planner_agent import _first_text

    resp = SimpleNamespace(content=[
        SimpleNamespace(type="text", text="foo"),
        SimpleNamespace(type="text", text="bar"),
    ])
    assert _first_text(resp) == "foobar"


def test_first_text_fallback_to_content0_when_untyped():
    from backend.agents.planner_agent import _first_text

    # single-block response without a typed text block -> content[0].text fallback
    resp = SimpleNamespace(content=[SimpleNamespace(text="legacy")])
    assert _first_text(resp) == "legacy"


def test_first_text_empty_content_is_safe():
    from backend.agents.planner_agent import _first_text

    assert _first_text(SimpleNamespace(content=[])) == ""
    assert _first_text(SimpleNamespace(content=None)) == ""
