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


def _guard_tests_are_rail_aware(tree: ast.AST) -> bool:
    """True iff some `if` statement's TEST EXPRESSION is a conjunction that
    negates both the key and the rail flag -- i.e. `not <key> and not <rail>`.

    Inspecting the `if` TEST, not the file text, is the whole point: the
    previous version of this guard scanned for the substrings
    "paper_use_claude_code_route" and "not _rail_on" anywhere in the module,
    and the 78.1 Q/A killed it by reverting the guard to key-only while leaving
    both literals sitting in ordinary code -- all three guards stayed GREEN.
    That is vacuity shape #5 (a check that cannot fail when its subject breaks).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)):
            continue
        negated = [v.operand for v in test.values
                   if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.Not)]
        if len(negated) < 2:
            continue
        names = {n.id for n in negated if isinstance(n, ast.Name)}
        # one operand must be the rail flag, another the key
        if any("rail" in nm for nm in names) and any("key" in nm for nm in names):
            return True
    return False


@pytest.mark.parametrize("mod", sorted(C_BLOCK))
def test_service_key_guard_is_rail_aware(mod):
    """D-KEY: the empty-key guard must not disable the service when the rail is
    up. The rail needs NO Anthropic key -- claude_code_invoke SCRUBS
    ANTHROPIC_API_KEY from the subprocess env -- so a key-only guard would keep
    all six dark in exactly the dead-credits scenario this step exists to fix.

    The guard is NOT deleted: with neither a key nor the rail, make_client
    raises (llm_client.py:2163), so the short-circuit must remain.

    Asserted on the `if` TEST EXPRESSION via AST -- see _guard_tests_are_rail_aware
    for why a substring scan was insufficient (it was killed by the 78.1 Q/A).
    """
    tree = ast.parse(_module_path(mod).read_text(encoding="utf-8"))
    assert _guard_tests_are_rail_aware(tree), (
        f"{mod}.py has no `if not <key> and not <rail_flag>` guard -- its key "
        f"guard is not rail-aware, so it still disables the service when the "
        f"key is absent even though the CC rail could serve it"
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


# ── Regression guard: the house system prompt must reach the rail ──────

@pytest.mark.parametrize("mod", sorted(C_BLOCK))
def test_service_passes_house_system_prompt(mod):
    """phase-78.1 REGRESSION GUARD (found by the 78.1 Q/A, not by me).

    `ClaudeClient.generate_content` sets `system_prompt = _HOUSE_INSTRUCTIONS`
    UNCONDITIONALLY (llm_client.py:1453) and never reads `config["system"]`, so
    before the rewire all six services always received the 19,026-char house
    framing. `ClaudeCodeClient.generate_content` instead reads
    `config.get("system")` (claude_code_client.py:524) -- which was None -- so
    the rewire SILENTLY DROPPED the house prompt from all six on the rail.

    The research gate had asserted the opposite ("the house prompt is already
    absent from every rail call, pre-existing"); that premise was false for
    these six, because their prompt came from inside ClaudeClient rather than
    from a caller-supplied config key.

    Asserted on the AST of the config dict actually passed to generate_content,
    so a mention in a comment cannot satisfy it.
    """
    tree = ast.parse(_module_path(mod).read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
        # the six call it as asyncio.to_thread(client.generate_content, prompt, {...})
        args = list(node.args)
        if name not in ("generate_content", "to_thread") and not any(
            isinstance(a, ast.Attribute) and a.attr == "generate_content" for a in args
        ):
            continue
        for a in args:
            if isinstance(a, ast.Dict):
                keys = [k.value for k in a.keys if isinstance(k, ast.Constant)]
                if "system" in keys:
                    idx = keys.index("system")
                    val = a.values[idx]
                    assert isinstance(val, ast.Name) and val.id == "_HOUSE_INSTRUCTIONS", (
                        f"{mod}.py passes config['system'] but not _HOUSE_INSTRUCTIONS "
                        f"(got {ast.dump(val)[:80]}) -- the rail would get the wrong framing"
                    )
                    found = True
    assert found, (
        f"{mod}.py does not pass config['system'] to generate_content -- on the CC "
        f"rail its 19,026-char house prompt is DROPPED (ClaudeCodeClient reads "
        f"config['system']; ClaudeClient applied _HOUSE_INSTRUCTIONS internally)"
    )
