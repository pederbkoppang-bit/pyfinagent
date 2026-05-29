"""phase-47.8: regression guards for the Opus-4.8 stale-pin sweep.

Commit 8ecc9efe bumped the *canonical* pins 4-7->4-8 (model_tiers, cost_tracker
pricing, llm_client accept-lists, settings_api) but missed ~8 OPERATIVE files.
The most dangerous miss was multi_agent_orchestrator.py:1061 -- a
``startswith("claude-opus-4-7")`` predicate that is **False** for the now-4-8
pin, so a 4-8 agent fell into the ELSE branch that sets a manual
``budget_tokens`` + ``temperature=1``. Opus 4.8 **rejects** manual budget_tokens
+ sampling params with a **400** (it inherits 4.7's adaptive-thinking-only /
no-sampling constraint). The bug was silent because the canonical pins looked
done.

These guards fail if:
  * any operative default reverts to 4-7,
  * the 4-8 context-window / dropdown entries are dropped,
  * the orchestrator thinking branch narrows back to a 4-7-only predicate, OR
  * a *legit* 4-7 compat entry (pricing / effort fallback / accept-list) is
    purged (4-7 remains a valid legacy fallback).

Wherever the symbol is cleanly importable the assertion is BEHAVIORAL (it
exercises the real lookup / signature), not a source grep -- see
``test_harness_memory_*`` which proves an unknown model gets the 128K default,
so a missing 4-8 entry would have silently truncated the window.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Criterion 2 -- missing-4-8 map entries ADDED (4-7 kept)
# --------------------------------------------------------------------------
def test_harness_memory_context_window_4_8_is_1m_and_4_7_kept():
    """BEHAVIORAL: the lookup that should_reset_context's default feeds.

    Pre-fix, get_context_window('claude-opus-4-8') fell to the 128K default ->
    premature context resets / over-aggressive masking on a 1M model.
    """
    from backend.agents.harness_memory import (
        MODEL_CONTEXT_WINDOWS,
        get_context_window,
    )

    assert get_context_window("claude-opus-4-8") == 1_000_000
    assert get_context_window("claude-opus-4-7") == 1_000_000  # compat kept
    # Proves the 4-8 assertion is meaningful (not a tautology): an unknown
    # model resolves to the 128K default, so a dropped 4-8 entry would silently
    # truncate the window from 1M to 128K.
    assert get_context_window("zzz-nonexistent-model") == 128_000
    assert MODEL_CONTEXT_WINDOWS.get("claude-opus-4-8") == 1_000_000


def test_app_home_dropdown_offers_4_8_first_and_keeps_4_7():
    from backend.slack_bot.app_home import AVAILABLE_MODELS

    assert "claude-opus-4-8" in AVAILABLE_MODELS, "operator cannot select 4-8"
    assert AVAILABLE_MODELS[0] == "claude-opus-4-8", "4-8 should be the preferred (first) option"
    assert "claude-opus-4-7" in AVAILABLE_MODELS, "legacy 4-7 should remain selectable"


# --------------------------------------------------------------------------
# Criterion 3 -- operative stale 4-7 DEFAULT pins bumped to 4-8 (BEHAVIORAL)
# --------------------------------------------------------------------------
def test_planner_agent_defaults_to_4_8():
    from backend.agents.planner_agent import PlannerAgent, get_planner_agent

    assert inspect.signature(PlannerAgent.__init__).parameters["model"].default == "claude-opus-4-8"
    assert inspect.signature(get_planner_agent).parameters["model"].default == "claude-opus-4-8"


def test_autonomous_loop_planner_model_defaults_to_4_8():
    from backend.autonomous_loop import AutonomousLoopOrchestrator

    default = inspect.signature(AutonomousLoopOrchestrator.__init__).parameters["planner_model"].default
    assert default == "claude-opus-4-8"


def test_multi_agent_should_reset_context_defaults_to_4_8():
    from backend.agents.multi_agent_orchestrator import MultiAgentOrchestrator

    default = inspect.signature(MultiAgentOrchestrator.should_reset_context).parameters["model"].default
    assert default == "claude-opus-4-8"


def test_rag_vision_model_defaults_to_4_8():
    from backend.agents.rag_agent_runtime import multimodal_index_claude

    assert inspect.signature(multimodal_index_claude).parameters["model"].default == "claude-opus-4-8"


def test_openclaw_overrides_main_qa_4_8_research_unchanged():
    from backend.agents.openclaw_client import AGENT_MODEL_OVERRIDES

    assert AGENT_MODEL_OVERRIDES["main"] == "anthropic/claude-opus-4-8"
    assert AGENT_MODEL_OVERRIDES["qa"] == "anthropic/claude-opus-4-8"
    # research deliberately stays on the cost-efficient Sonnet tier
    assert AGENT_MODEL_OVERRIDES["research"] == "anthropic/claude-sonnet-4-6"


def _extract_agent_model_map() -> dict:
    """AST-extract the local agent_model_map literal (it lives inside a method,
    so it is not importable as a symbol)."""
    src = (REPO / "backend/services/ticket_queue_processor.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "agent_model_map":
                    return ast.literal_eval(node.value)
    raise AssertionError("agent_model_map assignment not found")


def test_ticket_queue_agent_model_map_and_default_4_8():
    m = _extract_agent_model_map()
    assert m["main"] == "claude-opus-4-8"
    assert m["q-and-a"] == "claude-opus-4-8"
    assert m["research"] == "claude-sonnet-4-6"  # unchanged
    # the .get() default also bumped (operative fallback when agent_id unknown)
    src = (REPO / "backend/services/ticket_queue_processor.py").read_text(encoding="utf-8")
    assert 'agent_model_map.get(agent_id, "claude-opus-4-8")' in src
    assert 'agent_model_map.get(agent_id, "claude-opus-4-7")' not in src


# --------------------------------------------------------------------------
# Criterion 1 (CRITICAL) -- orchestrator thinking/sampling branch widened
# --------------------------------------------------------------------------
def test_orchestrator_thinking_branch_includes_4_8():
    """The :1061 predicate must route 4-8 down the adaptive-only path.

    A narrow ``startswith("claude-opus-4-7")`` would send a 4-8 agent into the
    ELSE branch (manual budget_tokens + temperature=1) -> Anthropic 400.
    """
    src = (REPO / "backend/agents/multi_agent_orchestrator.py").read_text(encoding="utf-8")
    assert 'startswith(("claude-opus-4-8", "claude-opus-4-7"))' in src, (
        "thinking/sampling branch must include claude-opus-4-8"
    )
    # the narrow single-string form would silently re-break 4-8
    assert 'startswith("claude-opus-4-7")' not in src
    # masker model default bumped too (drives the context-window calc)
    assert 'model_name="claude-opus-4-8"' in src


# --------------------------------------------------------------------------
# Compat -- legit 4-7 entries PRESERVED (4-7 is a valid legacy fallback)
# --------------------------------------------------------------------------
def test_4_7_compat_entries_preserved():
    from backend.agents.cost_tracker import MODEL_PRICING
    from backend.config.model_tiers import (
        EFFORT_SUPPORTED_MODELS,
        MODEL_EFFORT_FALLBACK,
    )

    # pricing: legacy 4-7 row kept (and 4-8 also present, guarded by 47.3 test)
    assert MODEL_PRICING.get("claude-opus-4-7") == (5.00, 25.00)
    assert MODEL_PRICING.get("claude-opus-4-8") == (5.00, 25.00)

    # effort: both versions accepted for xhigh
    assert "claude-opus-4-8" in EFFORT_SUPPORTED_MODELS
    assert "claude-opus-4-7" in EFFORT_SUPPORTED_MODELS
    fallback = dict(MODEL_EFFORT_FALLBACK)
    assert fallback["claude-opus-4-8"] == "xhigh"
    assert fallback["claude-opus-4-7"] == "xhigh"


def test_llm_client_accept_lists_keep_both_versions():
    """llm_client's effort/xhigh predicates must accept BOTH 4-8 and 4-7."""
    src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
    # the startswith gates that whitelist Opus for xhigh/effort
    assert '("claude-opus-4-8", "claude-opus-4-7")' in src
    # provider-map still routes the legacy id
    assert '"claude-opus-4-7":   "anthropic/claude-opus-4-7"' in src
    assert '"claude-opus-4-8":   "anthropic/claude-opus-4-8"' in src
