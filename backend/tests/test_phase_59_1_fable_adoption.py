"""phase-59.1 (operator-directed 2026-06-11): Fable 5 adoption regression tests.

Pins the quality-first rare-event adoption decision:
- mas_main + autoresearch_strategic -> claude-fable-5 (rare-event roles);
- mas_qa + every per-ticker/metered role keeps its existing model
  (cost discipline -- $10/$50 per Mtok is 2x Opus 4.8);
- claude-fable-5 is effort-SUPPORTED (without the EFFORT_SUPPORTED_MODELS +
  MODEL_EFFORT_FALLBACK entries, llm_client silently DROPS the effort param);
- the ticket agent map repins main/q-and-a to Fable (research stays Sonnet);
- the Layer-3 agent files pin `model: fable` with the raised turn caps.

All offline (config + file assertions). No network/LLM.
"""
from __future__ import annotations

from pathlib import Path

from backend.config import model_tiers as mt

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Layer-2: rare-event roles on Fable; metered roles unchanged ──────
def test_rare_event_roles_resolve_to_fable():
    assert mt.resolve_model("mas_main") == "claude-fable-5"
    assert mt.resolve_model("autoresearch_strategic") == "claude-fable-5"


def test_metered_roles_unchanged():
    """Cost discipline: NO per-ticker/per-analysis role moves to Fable."""
    assert mt.resolve_model("mas_qa") == "claude-opus-4-8"          # per-ticker analyst
    assert mt.resolve_model("mas_communication") == "claude-sonnet-4-6"
    assert mt.resolve_model("mas_research") == "claude-sonnet-4-6"
    assert mt.resolve_model("autoresearch_fast") == "claude-haiku-4-5"
    assert mt.resolve_model("autoresearch_smart") == "claude-sonnet-4-6"
    # Gemini-locked pipeline roles must never move to a Claude model.
    assert mt.resolve_model("gemini_enrichment").startswith("gemini-")
    assert mt.resolve_model("gemini_deep_think").startswith("gemini-")


def test_fable_is_effort_supported_not_silently_dropped():
    """The 59.1 researcher's trap: without EFFORT_SUPPORTED_MODELS membership,
    model_supports_effort() returns False and llm_client drops the effort
    param for fable-pinned roles."""
    assert mt.model_supports_effort("claude-fable-5") is True
    assert mt.resolve_effort_by_model("claude-fable-5") == "xhigh"
    # role-level override still wins (mas_main runs max per EFFORT_DEFAULTS)
    assert mt.resolve_effort("mas_main") == "max"


def test_ticket_agent_map_pins():
    """Ticket agents are operator-paced (~$0.18/day on Fable): main + q-and-a
    repinned; research deliberately stays on cost-efficient Sonnet."""
    src = (REPO_ROOT / "backend/services/ticket_queue_processor.py").read_text(encoding="utf-8")
    assert '"main": "claude-fable-5"' in src
    assert '"q-and-a": "claude-fable-5"' in src
    assert '"research": "claude-sonnet-4-6"' in src


# ── Layer-3: harness agent frontmatter ───────────────────────────────
def _frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return text.split("---", 2)[1]


def test_layer3_agents_pin_fable_with_raised_caps():
    researcher = _frontmatter(REPO_ROOT / ".claude/agents/researcher.md")
    qa = _frontmatter(REPO_ROOT / ".claude/agents/qa.md")
    assert "model: fable" in researcher
    assert "model: fable" in qa
    assert "maxTurns: 40" in researcher   # was 30; stalled twice on complex briefs
    assert "maxTurns: 30" in qa           # was 12; FIVE mid-evaluation stalls 2026-06-10/11
    assert "effort: max" in researcher    # retained (documented over-spec)
    assert "effort: max" in qa
    # the economics + restart caveats must be recorded in the files
    for fm in (researcher, qa):
        assert "2026-06-23" in fm or "June" in fm or "USAGE CREDITS" in fm
        assert "verify_qa_roster_live" in fm


def test_claude_md_effort_policy_updated_additively():
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "claude-fable-5" in text
    assert "$10/$50" in text
    # the Opus 4.8 history must NOT be deleted (additive change)
    assert "phase-29.2" in text
    assert "Introducing Claude Opus 4.8" in text
