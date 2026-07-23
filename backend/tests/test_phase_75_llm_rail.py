"""phase-75.5 -- LLM rail: schema enforcement, metered-bypass guards, model
retirement, and cost correctness.

Seven defects, all SILENT -- nothing crashed, everything under-reported or under-guarded:

  llmeng-01  ClaudeCodeClient dropped Pydantic model CLASSES via an isinstance(dict)
             gate, so --json-schema was dead code on the entire Layer-1 pipeline path
             (all 9 response_schema values there are classes) while the sibling
             direct-API client enforced the schema.
  llmeng-03  advisor_call -- a live metered call site -- skipped BOTH the cost-budget
             hard-block and the Claude-Code routing-breach guard, and built a raw
             Anthropic client from os.getenv.
  llmeng-04  LLMResponse had no stop_reason, so truncation was invisible above the
             client layer: a max_tokens cut-off looked like ordinary malformed output.
  llmeng-06  5 hardcoded gemini-2.5 pins with no constant and no retirement tripwire
             (the family shuts down 2026-10-16; the 2.0-flash class already cost this
             project 9 silent days).
  llmeng-10  cost_tracker double-subtracted cache tokens, under-reporting input cost
             by 66.7% on every cached call, silently, behind a max(0, ...) clamp.
  llmeng-11  OpenAIClient wrote no llm_call_log row at all.
  arch-04    The money guard was reached via a PRIVATE symbol in a Slack job.

TEST DOCTRINE (phase-75 durable rule; harness_log Cycle 131): a guard that cannot fail
does not count, and a mutation matrix licenses only "these N mutations were killed" --
never the global "this suite has no vacuous guards". Guards here assert BEHAVIOUR
(argv actually built, guard actually raised, price actually computed) rather than the
presence of a substring in source, except where a criterion explicitly demands a scan.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[2]


# ── criterion 1: llmeng-01, Pydantic CLASS -> --json-schema ──────────────────

def test_pydantic_class_schema_yields_json_schema_argv_with_additional_properties_false():
    """A config carrying a Pydantic model CLASS must produce a --json-schema argv
    whose JSON sets additionalProperties:false. Argv construction only -- no CLI
    is spawned (subprocess is mocked at the boundary)."""
    from backend.agents.claude_code_client import ClaudeCodeClient
    from backend.agents.schemas import CriticVerdict

    captured: dict = {}

    def fake_invoke(prompt, **kwargs):
        captured["json_schema"] = kwargs.get("json_schema")
        return {
            "subtype": "success", "result": "{}", "is_error": False,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "stop_reason": "end_turn",
        }

    client = ClaudeCodeClient(model_name="claude-opus-4-8")
    with patch("backend.agents.claude_code_client.claude_code_invoke", side_effect=fake_invoke):
        client.generate_content(
            "x",
            {"response_mime_type": "application/json", "response_schema": CriticVerdict},
        )

    schema = captured.get("json_schema")
    assert schema is not None, (
        "a Pydantic model CLASS was dropped -- --json-schema would never be emitted, "
        "leaving the CC rail unconstrained on the whole pipeline path"
    )
    assert schema.get("additionalProperties") is False

    # Every nested object node too -- Anthropic's validator rejects any object node
    # missing the field, not just the root.
    for name, node in (schema.get("$defs") or {}).items():
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, f"$defs.{name} not sealed"

    # And it must be JSON-serialisable, since the argv builder json.dumps() it.
    json.dumps(schema)


def test_dict_schema_path_is_preserved_not_replaced():
    """Six production services (meta_scorer, news_screen, pead_signal, macro_regime,
    analyst_narrative_scorer, call_transcript_gpr) pass pre-cleaned DICT schemas.
    The class fix must ADD a branch, never replace the dict one."""
    from backend.agents.claude_code_client import ClaudeCodeClient

    captured: dict = {}

    def fake_invoke(prompt, **kwargs):
        captured["json_schema"] = kwargs.get("json_schema")
        return {"subtype": "success", "result": "{}", "is_error": False,
                "usage": {}, "stop_reason": "end_turn"}

    client = ClaudeCodeClient(model_name="claude-opus-4-8")
    dict_schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with patch("backend.agents.claude_code_client.claude_code_invoke", side_effect=fake_invoke):
        client.generate_content(
            "x",
            {"response_mime_type": "application/json", "response_schema": dict_schema},
        )

    assert captured.get("json_schema") is not None, "the live dict path regressed"
    assert captured["json_schema"]["additionalProperties"] is False


def test_json_schema_argv_flag_is_actually_emitted():
    """Guard the argv builder itself: the schema must reach the CLI as --json-schema."""
    from backend.agents import claude_code_client as ccc

    with patch.object(ccc.subprocess, "run") as mock_run:
        mock_run.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"subtype": "success", "result": "{}", "is_error": False,
                               "usage": {}, "stop_reason": "end_turn"}),
            stderr="",
        )
        ccc.claude_code_invoke("hello", json_schema={"type": "object",
                                                     "additionalProperties": False})
        argv = mock_run.call_args[0][0]

    assert "--json-schema" in argv
    payload = json.loads(argv[argv.index("--json-schema") + 1])
    assert payload["additionalProperties"] is False


# ── criterion 2: llmeng-03, advisor_call spend + routing guards ──────────────

def test_advisor_call_raises_routing_breach_when_cc_rail_active(monkeypatch):
    """With the Max-subscription rail on, a direct-API advisor call would silently
    bill api.anthropic.com. It must fail loudly instead."""
    from backend.agents import llm_client

    monkeypatch.setattr(llm_client, "_check_cost_budget", lambda: None)
    monkeypatch.setattr(
        "backend.config.settings.get_settings",
        lambda: SimpleNamespace(paper_use_claude_code_route=True, anthropic_api_key="k"),
    )
    with pytest.raises(ValueError, match="Routing breach"):
        llm_client.advisor_call("prompt")


def test_advisor_call_invokes_cost_budget_guard(monkeypatch):
    """The cost-budget hard-block must run on this path. Proven by making it raise
    and observing the raise escape -- not by scanning source for the call."""
    from backend.agents import llm_client

    calls: list[str] = []

    def boom():
        calls.append("checked")
        raise RuntimeError("budget tripped")

    monkeypatch.setattr(llm_client, "_check_cost_budget", boom)
    monkeypatch.setattr(
        "backend.config.settings.get_settings",
        lambda: SimpleNamespace(paper_use_claude_code_route=False, anthropic_api_key="k"),
    )
    with pytest.raises(RuntimeError, match="budget tripped"):
        llm_client.advisor_call("prompt")
    assert calls == ["checked"]


def test_advisor_call_no_longer_reads_the_api_key_from_os_getenv():
    """Criterion 2 explicitly requires the raw os.getenv client build to be gone."""
    src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
    body = src[src.index("def advisor_call("):]
    body = body[: body.index("\ndef ")] if "\ndef " in body[10:] else body
    assert '_os.getenv("ANTHROPIC_API_KEY")' not in body
    assert "unwrap_secret" in body, (
        "the key must resolve via unwrap_secret -- a non-empty SecretStr is TRUTHY, so "
        "`settings.x or ''` returns the WRAPPER (this exact bug silently disabled 4 "
        "alpha overlays for ~3 weeks)"
    )


# ── criterion 3: llmeng-04, stop_reason + degraded WITHOUT a retry ───────────

@pytest.mark.parametrize("raw,expected", [
    ("max_tokens", True),    # Claude, lowercase
    ("MAX_TOKENS", True),    # Gemini, uppercase
    ("end_turn", False),
    ("STOP", False),
    (None, False),
])
def test_llmresponse_is_truncated_is_provider_and_case_insensitive(raw, expected):
    from backend.agents.llm_client import LLMResponse
    assert LLMResponse(text="x", stop_reason=raw).is_truncated() is expected


def test_stop_reason_is_populated_by_the_claude_code_path():
    from backend.agents.claude_code_client import ClaudeCodeClient

    client = ClaudeCodeClient(model_name="claude-opus-4-8")
    with patch(
        "backend.agents.claude_code_client.claude_code_invoke",
        return_value={"subtype": "success", "result": "{}", "is_error": False,
                      "usage": {"input_tokens": 1, "output_tokens": 1},
                      "stop_reason": "max_tokens"},
    ):
        resp = client.generate_content("x", {})
    assert resp.stop_reason == "max_tokens"
    assert resp.is_truncated() is True


def test_stop_reason_is_populated_by_the_claude_path():
    """BEHAVIOURAL (cycle 2). The cycle-1 version of this guard was a disk source
    scan, which the Q/A proved comment-satisfiable: replacing the real wiring with
    `stop_reason=None,  # was: stop_reason=getattr(...)` left it green. Drive the
    real client against a mocked SDK response instead."""
    from backend.agents.llm_client import ClaudeClient

    client = ClaudeClient(model_name="claude-opus-4-8", api_key="k")
    fake_sdk = MagicMock()
    fake_sdk.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"a": 1}', citations=None)],
        stop_reason="max_tokens",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5,
                              cache_read_input_tokens=0,
                              cache_creation_input_tokens=0),
        id="msg_1",
    )
    with patch.object(client, "_client", fake_sdk, create=True), \
         patch.object(ClaudeClient, "_get_client", return_value=fake_sdk, create=True), \
         patch("backend.agents.llm_client._check_cost_budget", lambda: None):
        resp = client.generate_content("x", {"max_output_tokens": 128})

    assert resp.stop_reason == "max_tokens", (
        f"Claude path did not surface stop_reason (got {resp.stop_reason!r}) -- "
        "truncation is invisible to callers again"
    )
    assert resp.is_truncated() is True


def test_stop_reason_is_populated_by_the_gemini_path():
    """BEHAVIOURAL (cycle 2). Gemini reports UPPERCASE finishReason; it must reach
    LLMResponse.stop_reason and be recognised by is_truncated()."""
    from backend.agents.llm_client import GeminiClient

    fake_resp = SimpleNamespace(
        text='{"a": 1}',
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"),
                                    content=None, grounding_metadata=None)],
        usage_metadata=SimpleNamespace(prompt_token_count=10,
                                       candidates_token_count=5,
                                       total_token_count=15),
    )
    bundle = SimpleNamespace(client=MagicMock(), model_name="gemini-2.5-flash",
                             tools=[], base_config={})
    bundle.client.models.generate_content.return_value = fake_resp
    client = GeminiClient(bundle, "gemini-2.5-flash")

    with patch("backend.agents.llm_client._check_cost_budget", lambda: None):
        resp = client.generate_content("x", {"max_output_tokens": 128})

    assert resp.stop_reason == "MAX_TOKENS", (
        f"Gemini path did not surface finish_reason (got {resp.stop_reason!r})"
    )
    assert resp.is_truncated() is True, "UPPERCASE MAX_TOKENS must be recognised"


def test_parse_helper_flags_truncation_without_retrying(monkeypatch):
    """Criterion 3's no-double-retry clause, guarded so it can actually FAIL.

    The cycle-1..7 version of this test watched a `generate_content` method bolted onto
    the RESPONSE object and asserted it was never called. That could not fail:
    parse_llm_json's signature is `(response, label, *, text=None)` -- it receives no
    client and calls nothing on `response` except `is_truncated()`. A Q/A proved it by
    running a mutant that issued a REAL re-request through a module-level client while
    the assertion still held.

    This version arms a SENTINEL on every client constructor and transport a retry could
    reach, so any re-request -- however it is routed -- raises instead of passing.
    """
    from backend.agents import llm_client as _llm
    from backend.agents.llm_client import LLMResponse
    from backend.agents.llm_parse import parse_llm_json

    fired: list[str] = []

    def _tripwire(name):
        def _boom(*a, **k):
            fired.append(name)
            raise AssertionError(
                f"parse_llm_json issued a re-request via {name} -- FORBIDDEN. The sole "
                f"max_tokens re-request owner is ClaudeClient.generate_content's "
                f"MF-26/27 dispatch."
            )
        return _boom

    # Every seam a retry could plausibly take from inside this module.
    for cls_name in ("ClaudeClient", "GeminiClient", "OpenAIClient"):
        cls = getattr(_llm, cls_name, None)
        if cls is not None:
            monkeypatch.setattr(cls, "generate_content", _tripwire(f"{cls_name}.generate_content"),
                                raising=False)
    monkeypatch.setattr(_llm, "make_client", _tripwire("make_client"), raising=False)
    monkeypatch.setattr(_llm, "advisor_call", _tripwire("advisor_call"), raising=False)

    resp = LLMResponse(text='{"a": 1}', stop_reason="max_tokens")
    data, degraded = parse_llm_json(resp, "T")

    assert data == {"a": 1}
    assert degraded is True, "truncation must be reported"
    assert resp.degraded is True, "the flag must land on the response object"
    assert fired == [], f"a re-request was issued via {fired}"


def test_parse_layer_cannot_reach_a_client_by_dependency_direction():
    """Structural backstop for the same clause, at the IMPORT level.

    The previous backstop scanned for three literal substrings
    (`messages.create`, `generate_content(`, `chat.completions.create`). A Q/A evaded it
    with `getattr(_m, 'create')(...)` -- a real retry that matches none of them. Names
    can be indirected; imports cannot be hidden as easily, so assert the DEPENDENCY
    DIRECTION instead: the parse layer must not import anything capable of constructing
    or invoking an LLM client.
    """
    import ast

    src = (REPO / "backend/agents/llm_parse.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(f"{node.module or ''}.{a.name}" for a in node.names)

    forbidden = ("llm_client", "claude_code_client", "anthropic", "openai",
                 "google.genai", "make_client")
    offenders = sorted(m for m in imported if any(f in m for f in forbidden))
    assert offenders == [], (
        f"llm_parse imports client-capable modules {offenders} -- it must only OBSERVE. "
        f"Any retry belongs to ClaudeClient.generate_content's MF-26/27 dispatch."
    )

    # The owner must still be named in-module so the next reader knows where retries live.
    assert "MF-26/27" in src


def test_parse_helper_marks_truncated_but_still_parseable_json_as_degraded():
    """The dangerous case: JSON that parses but is missing its tail. A helper that
    only flagged parse FAILURES would wave this through."""
    from backend.agents.llm_client import LLMResponse
    from backend.agents.llm_parse import parse_llm_json

    data, degraded = parse_llm_json(
        LLMResponse(text='{"complete": "looking"}', stop_reason="MAX_TOKENS"), "T"
    )
    assert data is not None and degraded is True


def test_parse_helper_does_not_flag_healthy_responses():
    """Negative control -- a flag that always fires carries no information."""
    from backend.agents.llm_client import LLMResponse
    from backend.agents.llm_parse import parse_llm_json

    data, degraded = parse_llm_json(LLMResponse(text='{"a": 1}', stop_reason="end_turn"), "T")
    assert data == {"a": 1} and degraded is False


def test_no_double_retry_the_parse_layer_contains_no_re_request():
    """Static backstop for the retry-ownership rule, and the contract's requirement
    that the owning layer be named explicitly in-code."""
    src = (REPO / "backend/agents/llm_parse.py").read_text(encoding="utf-8")
    for forbidden in ("messages.create", "generate_content(", "chat.completions.create"):
        assert forbidden not in src, f"llm_parse issues its own request via {forbidden}"
    assert "MF-26/27" in src, "the single retry owner must be named in the module"


def test_sole_max_tokens_retry_owner_is_the_claude_client_layer():
    """The owning layer must still exist and still be single-shot at min(x*2, 8192)."""
    src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
    assert 'retry_kwargs["max_tokens"] = min(max_tokens * 2, 8192)' in src


# ── criterion 4: llmeng-06, pins routed + retirement tripwire ────────────────

CRITERION_4_FILES = [
    "backend/config/settings.py",
    "backend/agents/evaluator_agent.py",
    "backend/agents/rag_agent_runtime.py",
    "backend/agents/skill_modification_review.py",
    "backend/autonomous_loop.py",
]


@pytest.mark.parametrize("rel", CRITERION_4_FILES)
def test_zero_gemini_25_literals_outside_model_tiers(rel):
    """Per-file, so a regression names the offending file."""
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "gemini-2.5" not in text, f"{rel} still hardcodes a gemini-2.5 literal"


def test_pins_resolve_through_the_model_tiers_constants():
    from backend.agents.evaluator_agent import EvaluatorAgent
    from backend.agents.rag_agent_runtime import DEFAULT_QUERY_MODEL
    from backend.config.model_tiers import GEMINI_DEEP_THINK, GEMINI_WORKHORSE

    assert DEFAULT_QUERY_MODEL == GEMINI_WORKHORSE
    import inspect
    assert inspect.signature(EvaluatorAgent.__init__).parameters["model_name"].default == GEMINI_WORKHORSE
    assert GEMINI_DEEP_THINK  # the constant must exist; it did not before 75.5


def test_retirement_warning_fires_on_and_after_the_warn_date():
    from backend.config.model_tiers import gemini_retirement_warning

    warn = gemini_retirement_warning("gemini-2.5-pro", date(2026, 9, 15))
    assert warn and "2026-10-16" in warn


def test_retirement_warning_is_silent_before_the_warn_date_and_off_family():
    """Two negative controls -- a warning that always fires is noise, not a guard."""
    from backend.config.model_tiers import gemini_retirement_warning

    assert gemini_retirement_warning("gemini-2.5-pro", date(2026, 9, 14)) is None
    assert gemini_retirement_warning("gemini-3.5-flash", date(2026, 9, 15)) is None


# ── criterion 5: llmeng-10 cost math + llmeng-11 OpenAI telemetry ────────────

def _usage(**kw):
    """Build the real UsageMeta/response shape record() consumes.

    NOTE ON THE CRITERION WORDING: criterion 5 says `UsageMeta(input=1000,
    cache_read=5000)`. Those kwargs do not exist -- the real fields are
    prompt_token_count / cache_read_input_tokens, and record() takes a RESPONSE
    object, not a UsageMeta. The criterion is IMMUTABLE and is NOT amended here; its
    kwargs are read as descriptive shorthand and its INTENT (1000 uncached input
    tokens must be priced, not clamped to zero) is what these tests enforce.
    """
    from backend.agents.llm_client import UsageMeta
    return SimpleNamespace(usage_metadata=UsageMeta(**kw))


@pytest.mark.parametrize("uncached,cache_read,cache_create", [
    (1000, 5000, 0),      # criterion 5's case
    (250, 20000, 0),      # cache-heavy: the old clamp hid the error hardest here
    (3000, 1000, 4000),   # cache WRITE in play too (2.0x, 1h TTL)
    (0, 8000, 0),         # fully cached -- must NOT price negative or clamp wrongly
])
def test_cached_call_prices_all_uncached_input_tokens_no_double_subtraction(
    uncached, cache_read, cache_create
):
    """THE MONEY TEST. input_tokens already EXCLUDES both cache buckets (Anthropic:
    'The input_tokens field represents only the tokens that come after the last cache
    breakpoint'). Subtracting them again under-reported input cost by 66.7%, and the
    max(0, ...) clamp made it silent.

    PARAMETRIZED deliberately: a single fixed case can be satisfied by a stub that
    returns one hardcoded number (mutation M14 proved exactly that against the
    single-case version of this test). Four different token mixes cannot be.
    """
    from backend.agents.cost_tracker import MODEL_PRICING, CostTracker

    model = "claude-opus-4-8"
    tracker = CostTracker()

    # ANTI-STUB CLAUSE (mutation M14): bind this test to the REAL implementation.
    # Without it, replacing CostTracker with a stub that returns the expected value
    # leaves the suite green while the money math goes unverified -- the 75.4 M10
    # failure shape, one module over.
    assert type(tracker).__module__ == "backend.agents.cost_tracker", (
        "CostTracker was replaced by a stub -- this test would prove nothing"
    )
    assert type(tracker).__name__ == "CostTracker"

    entry = tracker.record(
        "agent", model,
        _usage(prompt_token_count=uncached, candidates_token_count=0,
               total_token_count=uncached + cache_read + cache_create,
               cache_read_input_tokens=cache_read,
               cache_creation_input_tokens=cache_create),
    )
    in_price = MODEL_PRICING[model][0]
    expected = (
        uncached * in_price
        + cache_read * in_price * 0.1
        + cache_create * in_price * 2.0
    ) / 1_000_000

    assert entry is not None
    assert entry.cost_usd == pytest.approx(round(expected, 6)), (
        "uncached input tokens were not priced at full rate -- the double-subtraction "
        "is back"
    )
    # Pin the exact old failure mode: it clamped to the cache costs alone whenever
    # uncached < cache_read + cache_create.
    if uncached and uncached < cache_read + cache_create:
        old_buggy = (
            cache_read * in_price * 0.1 + cache_create * in_price * 2.0
        ) / 1_000_000
        assert entry.cost_usd != pytest.approx(round(old_buggy, 6)), (
            "cost equals the old clamped value -- uncached tokens are being lost"
        )


def test_gemini_shape_safety_is_pinned_not_incidental():
    """record() reads Gemini's prompt_token_count but Anthropic's cache field names,
    and Gemini's semantics are the OPPOSITE (its count INCLUDES cached tokens).
    `regular_input = input_tokens` is correct only because GeminiClient never
    populates the cache fields. Pin that, so a future Gemini caching change fails
    HERE instead of silently over-counting cost."""
    src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
    gemini = src[src.index("class GeminiClient"): src.index("class OpenAIClient")]
    assert "cache_read_input_tokens=" not in gemini, (
        "GeminiClient now populates cache fields -- cost_tracker.record() will "
        "OVER-count, because Gemini's prompt_token_count already includes cached "
        "tokens. Split the provider paths before shipping this."
    )


def test_openai_client_writes_an_llm_call_log_row():
    from backend.agents.llm_client import OpenAIClient

    client = OpenAIClient(model_name="gpt-4o", api_key="k")
    fake = MagicMock()
    fake.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"a":1}'),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        id="resp_1",
    )
    with patch.object(OpenAIClient, "_client", create=True, new=fake), \
         patch("backend.services.observability.log_llm_call") as mock_log, \
         patch.object(client, "_get_client", return_value=fake, create=True):
        # NO try/except-skip: a skip is a guard that cannot fail, which is the
        # defect class this suite exists to prevent (harness_log Cycle 131). If the
        # client stops being drivable offline, this must go RED and be fixed.
        client.generate_content("x", {})
    assert mock_log.called, "OpenAIClient still writes no telemetry row"


def test_openai_telemetry_provider_is_conditional_not_hardcoded():
    """One class serves BOTH direct OpenAI and GitHub Models (routed via base_url);
    a hardcoded provider would mis-attribute every GitHub Models call."""
    src = (REPO / "backend/agents/llm_client.py").read_text(encoding="utf-8")
    assert 'provider="github_models" if self._base_url else "openai"' in src


# ── criterion 6: arch-04, public fetch_spend + degradation counter ───────────

def test_fetch_spend_is_importable_from_observability():
    from backend.services.observability import fetch_spend
    assert callable(fetch_spend)


@pytest.mark.parametrize("rel", [
    "backend/agents/llm_client.py",
    "backend/api/cost_budget_api.py",
    "backend/slack_bot/jobs/cost_budget_watcher.py",
])
def test_consumers_resolve_fetch_spend_from_observability(rel):
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "from backend.services.observability import fetch_spend" in text, (
        f"{rel} does not resolve the spend fetch from its public home"
    )
    assert "from backend.slack_bot.jobs.cost_budget_watcher import _default_fetch_spend" not in text


def test_back_compat_alias_survives_for_the_out_of_scope_test():
    """tests/slack_bot/test_scheduler_wiring_phase991.py:150 monkeypatches this exact
    attribute and lives OUTSIDE backend/tests/ -- i.e. outside this step's verification
    command -- so dropping the name would break it SILENTLY."""
    from backend.slack_bot.jobs import cost_budget_watcher
    assert callable(cost_budget_watcher._default_fetch_spend)


def test_spend_guard_degradation_is_counted_and_alerted_not_just_logged():
    """A fail-open returning (0.0, 0.0) is indistinguishable from 'no spend', so the
    budget guard silently opens. That degradation must be observable."""
    from backend.services.observability import spend as spend_mod

    spend_mod.reset_spend_guard_status()
    with patch.object(spend_mod, "_record_degradation", wraps=spend_mod._record_degradation) as spy, \
         patch("google.cloud.bigquery.Client", side_effect=RuntimeError("bq down")):
        daily, monthly = spend_mod.fetch_spend()

    assert (daily, monthly) == (0.0, 0.0), "fail-open semantics must be preserved"
    assert spy.called
    status = spend_mod.spend_guard_status()
    assert status["degraded_count"] == 1
    assert status["alerted"] is True
    spend_mod.reset_spend_guard_status()


def test_spend_guard_counter_stays_zero_on_the_happy_path():
    """Negative control for the counter."""
    from backend.services.observability import spend as spend_mod

    spend_mod.reset_spend_guard_status()
    assert spend_mod.spend_guard_status()["degraded_count"] == 0
