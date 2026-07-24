"""phase-75.5.2 -- route the remaining hardcoded gemini-2.5 BEHAVIOURAL pins
through backend/config/model_tiers.py::GEMINI_WORKHORSE (+ the family-guard
prefix constant). Boundary: literals become constants ONLY -- NO tier pin
VALUE changed anywhere.

Census (research_brief_75.5.2.md, re-derived tree-wide, 5 coverage rounds,
2 consecutive dry): 9 behavioural pins, ALL resolving to `gemini-2.5-flash`
-> GEMINI_WORKHORSE. NONE routes to GEMINI_DEEP_THINK.

  1. backend/meta_evolution/directive_review.py    (Vertex call, model=)
  2. backend/meta_evolution/directive_rewriter.py   (Vertex call, model=)
  3. backend/news/sentiment.py                      (module const)
  4. backend/agents/harness_memory.py (ObservationMasker.__init__ default)
  5. backend/agents/harness_memory.py (create_masker default)
  6. backend/services/autonomous_loop.py (_model_for_block fallback)
  7. backend/services/autonomous_loop.py (model_name fallback)
  8. backend/api/agent_map.py           (live_model fallback)
  9. scripts/harness/run_autonomous_loop.py (evaluator_model= kwarg)

Plus one FAMILY GUARD (behavioural but NOT a pin -- llm_client.py:985
`startswith("gemini-2.5")`), routed through a new
`GEMINI_2_5_FAMILY_PREFIX` constant.

TEST DOCTRINE (phase-75 durable rule; harness_log Cycle 131, carried
forward from test_phase_75_llm_rail.py): a guard that cannot fail does
not count. Guards here assert BEHAVIOUR (the actual kwarg captured, the
actual resolved default, the actual AST reference) rather than the
presence of a substring in source, EXCEPT criterion 1, which explicitly
demands a tree-wide scan.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]


# ══════════════════════════════════════════════════════════════════════════
# Criterion 1 -- tree-wide scan: ZERO behavioural gemini-2.5 literals remain
# outside backend/config/model_tiers.py (docstring prose included).
# ══════════════════════════════════════════════════════════════════════════

# Each excluded file is a genuine non-behavioural DATA/catalog home --
# justified individually so an auditor can see none of them is a blind
# spot for a real pin:
#   - model_tiers.py   -- the constants HOME itself (criterion: "outside
#                          model_tiers.py").
#   - cost_tracker.py  -- MODEL_PRICING lookup-table KEYS, not selectors.
#   - settings_api.py  -- _VALID_MODELS whitelist + UI pricing dropdown
#                          (input-validation/display data).
EXCLUDE_FILES = {
    "backend/config/model_tiers.py",
    "backend/agents/cost_tracker.py",
    "backend/api/settings_api.py",
}

# The 9-site census (file-level; some files carry >1 site) -- used only to
# prove the derived scan is non-vacuous, i.e. it actually reaches every
# known pin file. This is NOT the fix's source of truth (the parametrized
# scan below is) -- it is a floor the scan must clear.
KNOWN_PIN_FILES = {
    "backend/meta_evolution/directive_review.py",
    "backend/meta_evolution/directive_rewriter.py",
    "backend/news/sentiment.py",
    "backend/agents/harness_memory.py",
    "backend/services/autonomous_loop.py",
    "backend/api/agent_map.py",
    "backend/agents/llm_client.py",  # family guard, not a resolved-value pin
    "scripts/harness/run_autonomous_loop.py",
}


def _derive_in_scope_files() -> list[str]:
    files: set[str] = set()
    for p in (REPO / "backend").rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith("backend/tests/"):
            continue
        if rel in EXCLUDE_FILES:
            continue
        files.add(rel)
    # scope note (research brief §6): criterion 1 says "outside
    # model_tiers.py", not "outside backend/" -- a strict reading includes
    # scripts/. Only ONE scripts/ file carries a behavioural pin; the rest
    # of scripts/ is pricing/migration-text/debug DATA and stays untouched.
    files.add("scripts/harness/run_autonomous_loop.py")
    return sorted(files)


IN_SCOPE_FILES = _derive_in_scope_files()


def test_scan_is_non_vacuous():
    """Anti-vacuous-guard family: a scan that cannot locate its own known
    members fails (75.5.8 doctrine). Catches a glob that silently returns
    nothing OR an over-broad EXCLUDE_FILES set (mutation M3)."""
    assert len(IN_SCOPE_FILES) > 50, (
        f"IN_SCOPE_FILES suspiciously small ({len(IN_SCOPE_FILES)}) -- "
        "the glob or exclusion set is probably broken"
    )
    missing = KNOWN_PIN_FILES - set(IN_SCOPE_FILES)
    assert not missing, (
        f"the derived scan excludes known pin file(s) {missing} -- an "
        "over-broad EXCLUDE_FILES entry would hide a real pin"
    )


@pytest.mark.parametrize("rel", IN_SCOPE_FILES)
def test_zero_gemini_25_literals_outside_model_tiers(rel):
    """Per-file, so a regression names the offending file. Raw substring,
    docstrings/comments included -- criterion 1 says "read strictly, not
    reinterpreted" (75.5 precedent: test_phase_75_llm_rail.py:396-399)."""
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "gemini-2.5" not in text, f"{rel} still hardcodes a gemini-2.5 literal"


# ══════════════════════════════════════════════════════════════════════════
# Criterion 2 -- each newly-routed site resolves to the SAME model string
# it resolved to before the change (no tier pin VALUE changed).
# ══════════════════════════════════════════════════════════════════════════

def test_gemini_workhorse_and_deep_think_are_the_deliberate_migration_tripwire():
    """VALUE-PIN (also satisfies the mutation-matrix "changed VALUE fails"
    requirement). This is INTENTIONAL: when the 2.5 family retires
    2026-10-16 this test goes red on purpose, so the migration cannot slip
    by silently the way the 2.0-flash retirement did (9 silent days). Do
    NOT loosen this assertion to "fix" a future failure -- update the
    constants in model_tiers.py instead."""
    from backend.config.model_tiers import GEMINI_DEEP_THINK, GEMINI_WORKHORSE

    assert GEMINI_WORKHORSE == "gemini-2.5-flash"
    assert GEMINI_DEEP_THINK == "gemini-2.5-pro"


# ---- site 3: sentiment.py module constant ---------------------------------

def test_sentiment_scorer_const_resolves_to_workhorse():
    from backend.config.model_tiers import GEMINI_WORKHORSE
    from backend.news.sentiment import SCORER_MODEL_GEMINI_FLASH

    assert SCORER_MODEL_GEMINI_FLASH == GEMINI_WORKHORSE == "gemini-2.5-flash"


# ---- sites 4 & 5: harness_memory.py param defaults -------------------------

def test_harness_memory_masker_defaults_resolve_to_workhorse():
    from backend.agents.harness_memory import ObservationMasker, create_masker
    from backend.config.model_tiers import GEMINI_WORKHORSE

    assert (
        inspect.signature(ObservationMasker.__init__).parameters["model_name"].default
        == GEMINI_WORKHORSE
    )
    assert (
        inspect.signature(create_masker).parameters["model_name"].default
        == GEMINI_WORKHORSE
    )


def test_harness_memory_context_window_table_is_value_preserving():
    """The co-located MODEL_CONTEXT_WINDOWS keys (finding 4 / §6 of the
    brief) were also routed through the constants -- confirm the resolved
    context-window VALUES are unchanged (both were 1_048_576 before)."""
    from backend.agents.harness_memory import MODEL_CONTEXT_WINDOWS, get_context_window
    from backend.config.model_tiers import GEMINI_DEEP_THINK, GEMINI_WORKHORSE

    assert MODEL_CONTEXT_WINDOWS[GEMINI_WORKHORSE] == 1_048_576
    assert MODEL_CONTEXT_WINDOWS[GEMINI_DEEP_THINK] == 1_048_576
    assert get_context_window(GEMINI_WORKHORSE) == 1_048_576


# ---- site 8: agent_map.py locked-node fallback (behavioural capture) ------

def test_agent_map_locked_node_fallback_resolves_to_workhorse():
    from backend.api.agent_map import _inject_live_model
    from backend.config.model_tiers import GEMINI_WORKHORSE

    # No "model" key -> falls through to the OR-fallback under test.
    out = _inject_live_model({"id": "rag_agent", "gemini_locked": True})
    assert out["live_model"] == GEMINI_WORKHORSE == "gemini-2.5-flash"


# ---- sites 1 & 2: directive_review / directive_rewriter Gemini fallback ---
# GOLD: force the Anthropic leg to skip (empty api key) and capture the
# `model=` kwarg the Gemini fallback actually sends.

def _fake_settings_no_anthropic_key():
    return SimpleNamespace(
        anthropic_api_key=SimpleNamespace(get_secret_value=lambda: ""),
        gcp_project_id="test-project",
        gcp_location="us-central1",
    )


def test_directive_review_gemini_fallback_calls_with_workhorse(monkeypatch):
    from backend.config.model_tiers import GEMINI_WORKHORSE
    from backend.meta_evolution import directive_review

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.config.settings.get_settings", _fake_settings_no_anthropic_key
    )

    captured: dict = {}

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="{}")

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content

    with patch("google.genai.Client", return_value=fake_client):
        directive_review._call_llm_for_review("prompt")

    assert captured.get("model") == GEMINI_WORKHORSE == "gemini-2.5-flash", (
        "directive_review's Gemini fallback no longer routes through GEMINI_WORKHORSE"
    )


def test_directive_rewriter_gemini_fallback_calls_with_workhorse(monkeypatch):
    from backend.config.model_tiers import GEMINI_WORKHORSE
    from backend.meta_evolution import directive_rewriter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "backend.config.settings.get_settings", _fake_settings_no_anthropic_key
    )

    captured: dict = {}

    def fake_generate_content(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(text="{}")

    fake_client = MagicMock()
    fake_client.models.generate_content = fake_generate_content

    with patch("google.genai.Client", return_value=fake_client):
        directive_rewriter._call_llm_for_rewrite("prompt")

    assert captured.get("model") == GEMINI_WORKHORSE == "gemini-2.5-flash", (
        "directive_rewriter's Gemini fallback no longer routes through GEMINI_WORKHORSE"
    )


# ---- sites 6 & 7: autonomous_loop.py deep fallbacks (AST + MISROUTE guard) -

def _function_node(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _assign_rhs_names(func_node: ast.AST, target_name: str) -> list[set[str]]:
    """For every `target_name = <expr>` inside func_node, return the set of
    bare Name ids referenced anywhere in <expr>."""
    hits: list[set[str]] = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if target_name in ids:
                hits.append({n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)})
    return hits


@pytest.mark.parametrize("target", ["_model_for_block", "model_name"])
def test_autonomous_loop_gemini_fallback_sites_reference_workhorse_not_deep_think(target):
    """AST-based Name-reference check, scoped to `_run_gemini_analysis` by
    FUNCTION identity (not line number -- the same variable names are also
    assigned, with a *different* Claude fallback, inside the sibling
    `_run_claude_analysis`). Doubles as the MISROUTE guard (mutation M4):
    proves the site references GEMINI_WORKHORSE specifically, not
    GEMINI_DEEP_THINK -- independent of criterion 1's plain-substring scan."""
    tree = ast.parse(
        (REPO / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    )
    func = _function_node(tree, "_run_gemini_analysis")
    assert func is not None, "could not locate _run_gemini_analysis in autonomous_loop.py"

    hits = _assign_rhs_names(func, target)
    assert hits, f"no assignment to {target!r} found inside _run_gemini_analysis"
    for names in hits:
        assert "GEMINI_WORKHORSE" in names, (
            f"{target} inside _run_gemini_analysis no longer references GEMINI_WORKHORSE"
        )
        assert "GEMINI_DEEP_THINK" not in names, (
            f"{target} inside _run_gemini_analysis MISROUTED to GEMINI_DEEP_THINK"
        )


def test_autonomous_loop_model_tiers_import_is_alias_proof():
    """Alias-proofing for sites 6,7 (found by the 75.5.2 cycle-1 Q/A: the
    name-reference guard above is defeated by an aliased import
    `GEMINI_DEEP_THINK as GEMINI_WORKHORSE`, which binds the wrong value
    while every name check stays green -- the same shape the mutation
    matrix caught at site 9). Every model_tiers import in the module must
    bind GEMINI_WORKHORSE un-aliased and must not import GEMINI_DEEP_THINK
    at all."""
    tree = ast.parse(
        (REPO / "backend/services/autonomous_loop.py").read_text(encoding="utf-8")
    )
    tiers_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module and "model_tiers" in node.module
    ]
    assert tiers_imports, "autonomous_loop.py no longer imports from model_tiers"
    for node in tiers_imports:
        imported = {(a.name, a.asname) for a in node.names}
        for name, asname in imported:
            assert asname is None, (
                f"aliased model_tiers import ({name} as {asname}) -- the "
                "alias-misroute shape the cycle-1 Q/A flagged"
            )
        assert not any(name == "GEMINI_DEEP_THINK" for name, _ in imported), (
            "GEMINI_DEEP_THINK imported in autonomous_loop.py -- misroute risk "
            "for the _run_gemini_analysis fallback sites"
        )


# ---- site 9: scripts/harness/run_autonomous_loop.py (AST + MISROUTE guard) -

def test_run_autonomous_loop_script_evaluator_model_references_workhorse():
    tree = ast.parse(
        (REPO / "scripts/harness/run_autonomous_loop.py").read_text(encoding="utf-8")
    )
    call_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AutonomousLoopOrchestrator"
        ):
            call_node = node
            break
    assert call_node is not None, "could not locate AutonomousLoopOrchestrator(...) call"

    kwargs = {kw.arg: kw.value for kw in call_node.keywords}
    assert "evaluator_model" in kwargs
    val = kwargs["evaluator_model"]
    assert isinstance(val, ast.Name) and val.id == "GEMINI_WORKHORSE", (
        "evaluator_model no longer references GEMINI_WORKHORSE by name -- either a "
        "literal was restored (fails C1 too) or it was misrouted to GEMINI_DEEP_THINK"
    )

    # Alias-proofing (found by the 75.5.2 mutation matrix: an aliased import
    # `GEMINI_DEEP_THINK as GEMINI_WORKHORSE` defeats a call-site name check
    # while binding the wrong value). The import itself must bind the real
    # name, un-aliased, and DEEP_THINK must not be imported at all.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "model_tiers" in node.module:
            imported = {(a.name, a.asname) for a in node.names}
            assert ("GEMINI_WORKHORSE", None) in imported, (
                "GEMINI_WORKHORSE must be imported un-aliased from model_tiers"
            )
            assert not any(name == "GEMINI_DEEP_THINK" for name, _ in imported), (
                "GEMINI_DEEP_THINK imported in the script -- aliased misroute"
            )


# ---- family guard (llm_client.py:985) -- behavioural, not a resolved pin --

def test_family_guard_disables_thinking_for_workhorse_via_prefix_constant():
    """Behavioural (mirrors test_phase_75_llm_rail.py's Gemini-path pattern).
    bundle.model_name == GEMINI_WORKHORSE (2.5-flash, non-pro) must still
    disable default thinking (budget=0) after the startswith() literal was
    routed through GEMINI_2_5_FAMILY_PREFIX -- the exact guard that
    prevented the phase-60.1 step-timeout regression."""
    from backend.agents.llm_client import GeminiClient
    from backend.config.model_tiers import GEMINI_WORKHORSE

    fake_resp = SimpleNamespace(
        text='{"a": 1}',
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"),
                                    content=None, grounding_metadata=None)],
        usage_metadata=SimpleNamespace(prompt_token_count=10,
                                       candidates_token_count=5,
                                       total_token_count=15),
    )
    bundle = SimpleNamespace(
        client=MagicMock(), model_name=GEMINI_WORKHORSE, tools=[], base_config={}
    )
    bundle.client.models.generate_content.return_value = fake_resp
    client = GeminiClient(bundle, GEMINI_WORKHORSE)

    with patch("backend.agents.llm_client._check_cost_budget", lambda: None):
        client.generate_content("x", {"max_output_tokens": 128})

    call_kwargs = bundle.client.models.generate_content.call_args.kwargs
    config = call_kwargs.get("config")
    assert config is not None and getattr(config, "thinking_config", None) is not None, (
        "no thinking_config was assembled -- the flash-family guard did not fire at all"
    )
    assert config.thinking_config.thinking_budget == 0, (
        "the flash-family thinking-disable guard did not fire with budget=0 -- "
        "GEMINI_2_5_FAMILY_PREFIX routing regressed"
    )


def test_family_prefix_constant_value_is_unchanged():
    from backend.config.model_tiers import GEMINI_2_5_FAMILY_PREFIX

    assert GEMINI_2_5_FAMILY_PREFIX == "gemini-2.5"


# ══════════════════════════════════════════════════════════════════════════
# Criterion 3 -- gemini_retirement_warning fires for every routed site's
# resolved model under a frozen >=2026-09-15 date, silent before that date
# and for a non-2.5 model (two negative controls).
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("model_const_name", ["GEMINI_WORKHORSE", "GEMINI_DEEP_THINK"])
def test_retirement_warning_fires_for_both_constants_on_and_after_warn_date(model_const_name):
    from backend.config import model_tiers

    model = getattr(model_tiers, model_const_name)
    warn = model_tiers.gemini_retirement_warning(model, date(2026, 9, 15))
    assert warn, f"{model_const_name} ({model!r}) did not trigger a retirement warning"
    assert "2026-10-16" in warn


@pytest.mark.parametrize("model_const_name", ["GEMINI_WORKHORSE", "GEMINI_DEEP_THINK"])
def test_retirement_warning_is_silent_before_the_warn_date(model_const_name):
    """Negative control 1 -- a warning that always fires is noise, not a guard."""
    from backend.config import model_tiers

    model = getattr(model_tiers, model_const_name)
    assert model_tiers.gemini_retirement_warning(model, date(2026, 9, 14)) is None


def test_retirement_warning_is_silent_for_an_off_family_successor():
    """Negative control 2 -- a non-2.5 model must never trip the tripwire,
    even on/after the warn date."""
    from backend.config.model_tiers import gemini_retirement_warning

    assert gemini_retirement_warning("gemini-3.6-flash", date(2026, 9, 15)) is None
