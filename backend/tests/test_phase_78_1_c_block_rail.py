"""phase-78.1: the six signal-overlay services must obtain their LLM client
through `make_client`, so PAPER_USE_CLAUDE_CODE_ROUTE governs them.

Before this step all six constructed `ClaudeClient(...)` DIRECTLY, so the
CC-rail flag could never apply to them -- the phase-72 rail-bypass class
implicated in the 97%-cash incident: with Anthropic credits dead the overlays
fail while the flag claims the system is on the flat-fee Max rail.

WHY THESE TESTS EXIST AT ALL (measured, 2026-07-25): before this file, the six
services had **zero behavioural coverage of the client-construction path**. The
9 tests that matched their names are source-scans (model-pin literals, BQ
timeouts) plus two observability state-line tests. Proven by instrumentation:
injecting `raise RuntimeError('SENTINEL')` immediately before the client
construction in meta_scorer.py and running its whole suite left all 3 tests
GREEN -- the sentinel was never tripped. So a green suite was NOT evidence the
rewire worked, and the research gate's prediction that "three meta_scorer tests
break" did not materialise for the same reason: those tests never reach the code.

Two complementary legs, because neither alone is sufficient:
  1. STRUCTURAL (AST, not grep): each module must contain a `make_client(...)`
     call and NO `ClaudeClient(...)` construction. An AST check cannot be
     satisfied by a mention in a comment or a docstring.
  2. BEHAVIOURAL: for each service's OWN configured model, flipping
     `paper_use_claude_code_route` must change the CLIENT TYPE `make_client`
     returns. That is the criterion's "the flag governs them" half.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# service module -> the settings attribute holding its model id
C_BLOCK = {
    "meta_scorer": "meta_scorer_model",
    "news_screen": "news_screen_model",
    "macro_regime": "macro_regime_model",
    "pead_signal": "pead_signal_model",
    "analyst_narrative_scorer": "analyst_narrative_model",
    "call_transcript_gpr": "call_transcript_gpr_model",
}


def _module_path(mod: str) -> Path:
    return REPO_ROOT / "backend" / "services" / f"{mod}.py"


def _call_names(tree: ast.AST) -> set[str]:
    """Every simple callee name invoked anywhere in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


# ── Leg 1: structural (AST) ───────────────────────────────────────────

@pytest.mark.parametrize("mod", sorted(C_BLOCK))
def test_service_constructs_no_direct_claude_client(mod):
    """No `ClaudeClient(...)` CALL node may survive in the C-block.

    AST rather than grep on purpose: a grep is satisfied by the name appearing
    in a comment or docstring, and this file's own explanatory prose mentions
    `ClaudeClient` repeatedly -- a grep-based guard would flag itself.
    """
    tree = ast.parse(_module_path(mod).read_text(encoding="utf-8"))
    assert "ClaudeClient" not in _call_names(tree), (
        f"{mod}.py still CONSTRUCTS ClaudeClient directly -- "
        f"PAPER_USE_CLAUDE_CODE_ROUTE cannot govern it (phase-72 rail-bypass class)"
    )


@pytest.mark.parametrize("mod", sorted(C_BLOCK))
def test_service_obtains_its_client_via_make_client(mod):
    tree = ast.parse(_module_path(mod).read_text(encoding="utf-8"))
    assert "make_client" in _call_names(tree), (
        f"{mod}.py does not call make_client -- it cannot see the CC rail"
    )


@pytest.mark.parametrize("mod", sorted(C_BLOCK))
def test_service_key_guard_is_rail_aware(mod):
    """D-KEY: the empty-key guard must not disable the service when the rail is
    up. The rail needs NO Anthropic key -- claude_code_invoke SCRUBS
    ANTHROPIC_API_KEY from the subprocess env -- so a key-only guard would keep
    all six dark in exactly the dead-credits scenario this step exists to fix.

    The guard is NOT deleted: with neither a key nor the rail, make_client
    raises (llm_client.py:2163), so the short-circuit must remain.
    """
    src = _module_path(mod).read_text(encoding="utf-8")
    assert "paper_use_claude_code_route" in src, (
        f"{mod}.py's key guard is not rail-aware; it will still disable the "
        f"service when the key is absent even though the rail could serve it"
    )
    assert "not _rail_on" in src, (
        f"{mod}.py references the rail flag but does not use it in the guard"
    )


# ── Leg 2: behavioural — the flag actually changes the client TYPE ────

@pytest.mark.parametrize("mod,attr", sorted(C_BLOCK.items()))
def test_flag_flip_changes_client_type_for_this_services_model(mod, attr, monkeypatch):
    """The criterion's core: flip PAPER_USE_CLAUDE_CODE_ROUTE and the client
    TYPE must change -- asserted against each service's OWN configured model,
    not a hardcoded one, so a model re-pin cannot silently escape the rail.
    """
    from backend.agents.llm_client import ClaudeClient, make_client
    from backend.agents.claude_code_client import ClaudeCodeClient
    from backend.config.settings import get_settings

    settings = get_settings()
    model = getattr(settings, attr, None)
    assert model, f"{mod}: settings.{attr} is unset; cannot test its routing"

    class _S:
        """Minimal settings stand-in: a real key AND a togglable rail flag, so
        the two branches differ ONLY by the flag under test."""
        anthropic_api_key = "sk-ant-api-test-not-real"
        github_token = ""
        gemini_api_key = ""
        openai_api_key = ""
        claude_code_timeout_s = 60

    on, off = _S(), _S()
    on.paper_use_claude_code_route = True
    off.paper_use_claude_code_route = False

    railed = make_client(model, None, on)
    metered = make_client(model, None, off)

    assert isinstance(railed, ClaudeCodeClient), (
        f"{mod}: model {model!r} did NOT route to the CC rail with the flag ON "
        f"(got {type(railed).__name__}) -- the rewire is cosmetic"
    )
    assert isinstance(metered, ClaudeClient), (
        f"{mod}: model {model!r} did not fall back to the metered client with "
        f"the flag OFF (got {type(metered).__name__}) -- one-flag revert is broken"
    )
    assert type(railed) is not type(metered), (
        f"{mod}: the flag did not change the client type at all"
    )
