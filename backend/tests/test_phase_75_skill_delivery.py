"""phase-75.4 -- skill-prompt delivery integrity.

Five silent-failure defects, all of the same family: content that a skill file was
written to say never reached the model, and nothing crashed or logged an error.

  gap5-01  `load_skill` stops at the next H2, so `quant_model_agent.md`'s in-template
           `## Quant Model Data` / `## Instructions` headings truncated the delivered
           prompt to 190 of 7532 chars -- with `{{quant_model_data}}` outside the
           delivered region, so the agent was asked to interpret a factor score and
           handed no score.
  gap5-02  The phase-4.14.26 uncertainty/empty-bracket sections (8 files) and the
           phase-26.3 code-execution sections (3 files) sat past the heading that
           terminates the extracted region, so they were never delivered.
  gap5-03  The critic is instructed to echo a full corrected report (budgeted 4096)
           through a 2048-token output cap, and an unparseable verdict was silently
           upgraded to PASS -- the quality gate disappeared rather than failed.
  gap5-06  `_skill_gen_config` carried no output cap on either path, so the Claude
           rail used its own 2048 default instead of the documented 1024.
  gap5-10  `_skill_gen_config("sector_agent")` named a file that has never existed
           (`sector_analysis_agent.md`), silently losing the Files-API token saving.

TEST DOCTRINE (phase-75 durable rule, harness_log.md Cycle 130): a guard that cannot
fail does not count. Every assertion here goes through the REAL `load_skill()`, never
a string stub -- asserting a canary against `Path.read_text()` would prove nothing,
because every canary phrase is already in every affected file today and always was.
The mutation matrix in the step contract (M1-M11) must kill each guard below,
including M10, which mutates this test harness itself.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import pytest

from backend.config.prompts import SKILLS_DIR, format_skill, load_skill

REPO = Path(__file__).resolve().parents[2]
ORCHESTRATOR_SRC = REPO / "backend/agents/orchestrator.py"

# Files the phase-4.14.26 uncertainty sections must be delivered for, each mapped to a
# VERBATIM line of that file's own section BODY. Per-file rather than one shared phrase
# because the bodies genuinely differ (bias_detector's is bias-specific, and the
# code-execution tasks are different in all three files) -- a single shared canary
# would have silently skipped whatever it did not cover. Asserting the heading alone
# would not prove delivery of the content that matters.
UNCERTAINTY_BODY_CANARY = {
    "bias_detector": 'If the evidence does not clearly show bias -- say "I don\'t know" and',
    "critic_agent": '- Say "I don\'t know" rather than forcing a guess.',
    "deep_dive_agent": '- Say "I don\'t know" rather than forcing a guess.',
    "moderator_agent": '- Say "I don\'t know" rather than forcing a guess.',
    "quant_model_agent": '- Say "I don\'t know" rather than forcing a guess.',
    "risk_judge": '- Say "I don\'t know" rather than forcing a guess.',
    "scenario_agent": '- Say "I don\'t know" rather than forcing a guess.',
    "synthesis_agent": '- Say "I don\'t know" rather than forcing a guess.',
}

# Files the phase-26.3 code-execution section must be delivered for.
CODE_EXEC_BODY_CANARY = {
    "enhanced_macro_agent": "USE IT for the deterministic arithmetic before generating your",
    "quant_model_agent": "USE IT to verify your arithmetic. Specifically, before finalizing the signal:",
    "scenario_agent": "USE IT to verify the Monte Carlo result coherence before producing your risk_profile:",
}

UNCERTAINTY_FILES = sorted(UNCERTAINTY_BODY_CANARY)
CODE_EXEC_FILES = sorted(CODE_EXEC_BODY_CANARY)

# Skill .md files that are deliberately NOT on the load_skill path.
#   quant_strategy   -- an OPTIMIZER skill with no '## Prompt Template' section at all;
#                       read WHOLE at quant_optimizer.py:488, so its own phase-26.3
#                       section already reaches the model. load_skill() raises
#                       ValueError on it. Relocating anything in it would change what
#                       the optimizer sees.
#   SKILL_TEMPLATE   -- the scaffold for new skills, never loaded.
# Excluded by NAME and asserted, never swallowed by a try/except -- a bare except
# around load_skill() would silently absorb a real regression in any other file.
NON_TEMPLATE_SKILLS = {"quant_strategy", "SKILL_TEMPLATE"}


# ── criterion 1: gap5-01, the loader truncation ──────────────────────────────

def test_quant_model_template_ships_its_data_placeholder_and_instructions():
    """The delivered prompt must contain the runtime data placeholder and the
    instruction block that the in-template H2 headings used to cut off."""
    delivered = load_skill("quant_model_agent")

    assert "{{quant_model_data}}" in delivered, (
        "the runtime factor-score placeholder is outside the delivered region -- "
        "the agent is being asked to interpret a score it is never given"
    )
    # A line of the former '## Instructions' block, verbatim.
    assert "3. Check for factor contradictions" in delivered

    # The truncation was not subtle: 190 chars of a 7532-char file. Pin the floor so a
    # regression that re-truncates cannot pass by keeping the placeholder alone.
    assert len(delivered) > 2000, f"delivered template collapsed to {len(delivered)} chars"


def test_no_skill_has_an_unintended_h2_inside_its_template_region():
    """gap5-01 was one instance of a class. Assert the class is empty: for every
    load_skill-backed file, no placeholder present in the raw file may be lost from
    the delivered template."""
    losses: dict[str, list[str]] = {}
    for path in sorted(SKILLS_DIR.glob("*.md")):
        if path.stem in NON_TEMPLATE_SKILLS:
            continue
        raw = path.read_text(encoding="utf-8")
        delivered = load_skill(path.stem)
        raw_ph = set(re.findall(r"\{\{(\w+)\}\}", raw))
        delivered_ph = set(re.findall(r"\{\{(\w+)\}\}", delivered))
        # Placeholders documented in '## Data Inputs' prose are not real template
        # placeholders; only count a loss if a BUILDER could actually pass it.
        lost = {p for p in raw_ph - delivered_ph if f"{{{{{p}}}}}\n" in raw}
        if lost:
            losses[path.stem] = sorted(lost)
    assert losses == {}, f"placeholders lost to template truncation: {losses}"


# ── criterion 2: gap5-02, the undelivered sections ───────────────────────────

@pytest.mark.parametrize("stem", UNCERTAINTY_FILES)
def test_uncertainty_permission_is_delivered(stem: str):
    """Per-file, NOT any()-shaped: each of the 8 files is its own test case, so
    moving a single file's section back out fails exactly one case (mutation M3)."""
    delivered = load_skill(stem)
    assert "Uncertainty Permission" in delivered
    # A heading is not delivery -- assert this file's own BODY line.
    assert UNCERTAINTY_BODY_CANARY[stem] in delivered
    # The empty-bracket sibling section relocates with it.
    assert "An empty bracket marker `[]` or an omitted field is an acceptable" in delivered


def test_uncertainty_permission_covers_every_file_that_has_one():
    """The 8-file list must not silently shrink: if a file gains the section, it must
    gain a test case too."""
    have_it = {
        p.stem for p in SKILLS_DIR.glob("*.md")
        if p.stem not in NON_TEMPLATE_SKILLS
        and "Uncertainty Permission" in p.read_text(encoding="utf-8")
    }
    assert have_it == set(UNCERTAINTY_BODY_CANARY), (
        f"untested files carrying the section: {have_it - set(UNCERTAINTY_BODY_CANARY)}"
    )


@pytest.mark.parametrize("stem", CODE_EXEC_FILES)
def test_code_execution_section_is_delivered(stem: str):
    delivered = load_skill(stem)
    assert "Code Execution" in delivered
    assert CODE_EXEC_BODY_CANARY[stem] in delivered


def test_non_template_skills_are_excluded_by_name_not_by_swallowing_errors():
    """Assert the exclusion list is exactly what we think it is. If a file stops
    raising (someone adds a '## Prompt Template' to quant_strategy), this fails and
    forces a decision -- rather than a try/except quietly widening the exclusion."""
    actually_unloadable = set()
    for path in sorted(SKILLS_DIR.glob("*.md")):
        try:
            load_skill(path.stem)
        except (ValueError, FileNotFoundError):
            actually_unloadable.add(path.stem)

    assert actually_unloadable == {"quant_strategy"}, (
        f"the set of skill files without a '## Prompt Template' changed: "
        f"{actually_unloadable}. SKILL_TEMPLATE.md is excluded separately (it is a "
        f"scaffold, not a skill)."
    )
    # quant_strategy's own code-execution section reaches the model by the raw-read
    # path, so it must NOT have been relocated by the phase-75.4 migration.
    raw = (SKILLS_DIR / "quant_strategy.md").read_text(encoding="utf-8")
    assert "## Code Execution Tasks (phase-26.3)" in raw, (
        "quant_strategy.md was modified -- it is read whole by quant_optimizer.py "
        "and must be left alone"
    )


# ── criterion 3: format_skill warning + sector stem + startup assertion ──────

def test_format_skill_warns_when_a_kwarg_has_no_placeholder(caplog):
    template = "Hello {{name}}."
    with caplog.at_level(logging.WARNING, logger="backend.config.prompts"):
        out = format_skill(template, name="world", orphan="discarded")

    assert out == "Hello world."
    assert "orphan" in caplog.text
    assert "no matching placeholder" in caplog.text


def test_format_skill_does_not_warn_when_every_kwarg_matches(caplog):
    """The warning must be specific enough to stay silent on the happy path,
    otherwise it is noise that will be tuned out."""
    with caplog.at_level(logging.WARNING, logger="backend.config.prompts"):
        format_skill("Hello {{name}}.", name="world")
    assert "no matching placeholder" not in caplog.text


def test_sector_call_site_uses_the_real_skill_stem():
    src = ORCHESTRATOR_SRC.read_text(encoding="utf-8")
    assert '_skill_gen_config("sector_analysis_agent")' in src
    assert '_skill_gen_config("sector_agent")' not in src


def test_every_skill_gen_config_call_site_resolves_to_a_real_file():
    """The startup assertion guards the registry; this guards the registry against
    the CALL SITES, which is the drift that produced gap5-10 in the first place."""
    from backend.agents.orchestrator import _SKILL_GEN_STEMS, _assert_skill_stems_exist

    src = ORCHESTRATOR_SRC.read_text(encoding="utf-8")
    call_site_stems = set(re.findall(r'_skill_gen_config\(\s*"([^"]+)"\s*\)', src))

    assert call_site_stems == set(_SKILL_GEN_STEMS), (
        f"registry/call-site drift -- only in call sites: "
        f"{call_site_stems - set(_SKILL_GEN_STEMS)}; only in registry: "
        f"{set(_SKILL_GEN_STEMS) - call_site_stems}"
    )
    for stem in call_site_stems:
        assert (SKILLS_DIR / f"{stem}.md").is_file(), f"no skill file for stem {stem!r}"

    # The assertion must be wired to run at import, not merely defined.
    tree = ast.parse(src)
    called_at_module_level = any(
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "id", None) == "_assert_skill_stems_exist"
        for node in tree.body
    )
    assert called_at_module_level, "_assert_skill_stems_exist is defined but never called"
    _assert_skill_stems_exist()  # must not raise against the real skills dir


# ── criterion 4: gap5-03, the critic budget and the fail-open branch ─────────

def test_critic_output_budget_fits_the_report_it_must_echo():
    from backend.agents.orchestrator import (
        _CRITIC_STRUCTURED_CONFIG,
        _SYNTHESIS_STRUCTURED_CONFIG,
        _THINKING_CRITIC_CONFIG,
    )

    assert _CRITIC_STRUCTURED_CONFIG["max_output_tokens"] >= 6144
    # The dead twin must not silently reintroduce the truncation if it is ever adopted.
    assert _THINKING_CRITIC_CONFIG["max_output_tokens"] >= 6144
    # 1.5x the report it is instructed to echo, per Anthropic's sizing guidance.
    assert (
        _CRITIC_STRUCTURED_CONFIG["max_output_tokens"]
        >= 1.5 * _SYNTHESIS_STRUCTURED_CONFIG["max_output_tokens"]
    )


def test_unparseable_critic_verdict_is_not_silently_treated_as_pass():
    """Criterion 4's literal-string clause. Kept as a source scan because the criterion
    is worded that way -- but it proves only the ABSENCE of the old string. The
    behavioral proof that something correct replaced it lives in the two tests below,
    which actually run the loop."""
    src = ORCHESTRATOR_SRC.read_text(encoding="utf-8")
    assert "treating as PASS with draft" not in src, (
        "the fail-OPEN branch is back: an unparseable critic verdict must never be "
        "upgraded to PASS -- the quality gate would disappear rather than fail"
    )


def _run_pipeline_with_critic_text(monkeypatch, critic_payloads: list[str]):
    """Drive the real run_synthesis_pipeline with a stubbed LLM, feeding a scripted
    sequence of Critic responses. Returns (result_dict, agent_call_names).

    Behavioral, not textual: a substring scan for 'Critic-Retry' is satisfied by a
    comment, so it survived deleting the entire retry block. This exercises the code.
    """
    from backend.agents import orchestrator as orch

    class _Settings:
        max_synthesis_iterations = 2

    class _Client:
        model_name = "gemini-2.5-flash"

    orc = orch.AnalysisOrchestrator.__new__(orch.AnalysisOrchestrator)
    orc.settings = _Settings()
    orc.synthesis_client = _Client()
    orc.deep_think_client = _Client()
    calls: list[str] = []
    draft = '{"recommendation": "HOLD", "summary": "draft"}'
    remaining = list(critic_payloads)

    def fake_generate(client, prompt, agent_name, **kwargs):
        calls.append(agent_name)
        if agent_name.startswith("Critic"):
            return remaining.pop(0) if remaining else "not json at all"
        return draft

    monkeypatch.setattr(orc, "_generate_with_retry", fake_generate, raising=False)
    monkeypatch.setattr(orch, "_extract_text", lambda r: r)
    monkeypatch.setattr(orch, "_clean_json_output", lambda t: t)
    monkeypatch.setattr(
        orch.prompts, "get_critic_prompt",
        lambda *a, **k: "critic prompt", raising=False,
    )
    monkeypatch.setattr(
        orch.prompts, "get_synthesis_prompt",
        lambda *a, **k: "synthesis prompt", raising=False,
    )
    result = orc.run_synthesis_pipeline("TEST", {"ticker": "TEST"})
    return result, calls


def test_unparseable_critic_triggers_exactly_one_retry_then_flags_degraded(monkeypatch):
    """The behavioral guard for criterion 4's replacement. Both critic responses are
    unparseable: the loop must retry ONCE (a second Critic call), then return the draft
    flagged critic_degraded=True rather than silently blessed."""
    # NO try/except-skip here on purpose: a skip is a guard that cannot fail, which is
    # the exact defect class this suite exists to prevent. If the pipeline stops being
    # drivable in isolation, this test must go RED and be fixed, not silently skipped.
    result, calls = _run_pipeline_with_critic_text(
        monkeypatch, ["<<garbage not json>>", "<<still garbage>>"]
    )

    critic_calls = [c for c in calls if c.startswith("Critic")]
    assert "Critic-Retry" in critic_calls, (
        f"no retry was attempted on an unparseable verdict; calls={calls}"
    )
    assert len(critic_calls) == 2, (
        f"expected exactly one retry (2 critic calls), got {critic_calls}"
    )
    assert result.get("critic_degraded") is True, (
        "an unparseable critic verdict must leave the report FLAGGED -- "
        f"got critic_degraded={result.get('critic_degraded')!r}"
    )


def test_parseable_critic_verdict_does_not_retry_and_is_not_flagged(monkeypatch):
    """Negative control: the happy path must NOT retry and must NOT flag degraded,
    or the flag is meaningless."""
    result, calls = _run_pipeline_with_critic_text(
        monkeypatch, ['{"verdict": "PASS", "issues": []}']
    )

    critic_calls = [c for c in calls if c.startswith("Critic")]
    assert "Critic-Retry" not in critic_calls, f"retried on a VALID verdict; calls={calls}"
    assert result.get("critic_degraded") is False


def _returns_missing_critic_degraded(func: ast.FunctionDef) -> list[int]:
    """Return the line numbers of every value-returning `return` in `func` that does
    NOT carry `critic_degraded`.

    Structural, not textual. For a dict-literal return the key must be in the literal;
    for a `return <name>` the enclosing statement list must contain a preceding
    `<name>["critic_degraded"] = ...` assignment. Counting substring occurrences (the
    pre-cycle-2 version of this guard) could not distinguish a real attachment from a
    comment, and survived five separate mutations that stripped the flag from one, two,
    three, or all four return paths.
    """
    missing: list[int] = []

    def key_names(d: ast.Dict) -> set[str]:
        return {k.value for k in d.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}

    def attached_before(body: list[ast.stmt], idx: int, name: str) -> bool:
        for prev in body[:idx]:
            for node in ast.walk(prev):
                if (
                    isinstance(node, ast.Assign)
                    and any(
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == name
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == "critic_degraded"
                        for t in node.targets
                    )
                ):
                    return True
        return False

    def walk_block(body: list[ast.stmt]):
        for i, stmt in enumerate(body):
            if isinstance(stmt, ast.Return) and stmt.value is not None:
                val = stmt.value
                if isinstance(val, ast.Dict):
                    if "critic_degraded" not in key_names(val):
                        missing.append(stmt.lineno)
                elif isinstance(val, ast.Name):
                    if not attached_before(body, i, val.id):
                        missing.append(stmt.lineno)
                elif not isinstance(val, ast.Constant):
                    missing.append(stmt.lineno)
            for attr in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, attr, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    walk_block(inner)
            for handler in getattr(stmt, "handlers", []) or []:
                walk_block(handler.body)

    walk_block(func.body)
    return missing


def test_critic_degraded_flag_is_present_on_every_return_path():
    """A flag that only SOME return paths set is worse than none -- a consumer reading
    `report.get("critic_degraded")` would see None on the unset paths and read it as
    False, i.e. 'the gate ran and passed'.

    This asserts per-return-path structurally. Stripping the flag from ANY single
    return path fails this test (the count-based predecessor survived stripping three
    of four)."""
    src = ORCHESTRATOR_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, node) or ""
            if "critic_degraded = False" in seg:
                target = node
                break
    assert target is not None, "could not locate the function that owns critic_degraded"

    value_returns = [
        n for n in ast.walk(target)
        if isinstance(n, ast.Return) and n.value is not None
        and not isinstance(n.value, ast.Constant)
    ]
    assert len(value_returns) >= 4, (
        f"expected >=4 value returns in {target.name}, found {len(value_returns)}"
    )

    missing = _returns_missing_critic_degraded(target)
    assert missing == [], (
        f"{len(missing)} return path(s) in {target.name} do not attach critic_degraded "
        f"(orchestrator.py lines {missing}) -- a consumer would read the absent key as "
        f"False, i.e. as 'the critic gate ran and passed'"
    )


# ── criterion 5: gap5-06, the enrichment cap on BOTH paths ──────────────────

def _stub_orchestrator():
    """Build an orchestrator without running __init__ (which needs live clients)."""
    from backend.agents.orchestrator import AnalysisOrchestrator

    return AnalysisOrchestrator.__new__(AnalysisOrchestrator)


def test_enrichment_cap_present_on_the_file_id_path():
    stub = _stub_orchestrator()
    stub._skill_file_ids = {"insider_agent": "file_xyz_123"}
    config = stub._skill_gen_config("insider_agent")
    assert config["max_output_tokens"] == 1024
    assert config["skill_file_id"] == "file_xyz_123"


def test_enrichment_cap_present_on_the_no_file_id_paths():
    """The path that used to return None. Both flavours: empty map, and a map that
    simply lacks this stem."""
    stub = _stub_orchestrator()

    stub._skill_file_ids = {}
    assert stub._skill_gen_config("insider_agent") == {"max_output_tokens": 1024}

    stub._skill_file_ids = {"insider_agent": "file_xyz_123"}
    assert stub._skill_gen_config("nonexistent_agent") == {"max_output_tokens": 1024}

    # Attribute entirely absent (pre-25.D9 orchestrators).
    bare = _stub_orchestrator()
    assert bare._skill_gen_config("insider_agent") == {"max_output_tokens": 1024}


def test_enrichment_cap_has_one_source_of_truth():
    """The Gemini bundle and the helper must read the same constant, or the two rails
    drift back apart the next time one is edited."""
    from backend.agents.orchestrator import _ENRICHMENT_MAX_OUTPUT_TOKENS

    assert _ENRICHMENT_MAX_OUTPUT_TOKENS == 1024
    src = ORCHESTRATOR_SRC.read_text(encoding="utf-8")
    assert '"max_output_tokens": _ENRICHMENT_MAX_OUTPUT_TOKENS' in src
    assert '"max_output_tokens": 1024' not in src, (
        "a literal 1024 reintroduces the drift the constant exists to prevent"
    )
