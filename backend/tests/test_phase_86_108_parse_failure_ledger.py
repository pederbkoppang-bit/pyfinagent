"""phase-86.108 -- the parse-failure ledger and the gated-flag read path.

Every test here DRIVES THE REAL FUNCTION. None of them greps source text, and
none asserts on a fixture it built itself: the step this file belongs to was
filed after three consecutive evaluations blocked on guard vacuity, where the
shipped code was correct and the guard could not have failed.

Two properties get the most attention because they are the ones that would do
damage if they broke silently:

  - **recording must never change control flow.** The emit sites sit inside
    `except` blocks on the money path. Tests 4-6 assert the return value of the
    REAL `_parse_json` functions, not of a stand-in.
  - **the gated-flag payload must never carry a str-typed field.** That is what
    keeps an API key out of an observability response, and it is asserted
    against the live `Settings` model rather than a denylist.
"""

from __future__ import annotations

import json
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.agents import parse_failure_ledger as pfl


@pytest.fixture(autouse=True)
def _clean_ledger():
    pfl.reset_for_tests()
    yield
    pfl.reset_for_tests()


# ── 1. the record carries the fields the log line never had ────────


def _settings_stub(monkeypatch, *, cc_route: bool, failforward: bool = False):
    """Replace `get_settings` so the rail can be exercised at both flag states.

    `resolve_rail` imports `get_settings` inside the call, so patching the
    attribute on the settings module is what the real code will see.
    """
    import backend.config.settings as cs

    stub = types.SimpleNamespace(
        paper_use_claude_code_route=cc_route,
        paper_rail_failforward_enabled=failforward,
    )
    monkeypatch.setattr(cs, "get_settings", lambda: stub)


def test_record_carries_agent_kind_site_and_rail(monkeypatch):
    _settings_stub(monkeypatch, cc_route=True)
    rec = pfl.record_parse_failure(
        "Conservative Analyst", pfl.KIND_PARSE_FAILED, site="unit:test",
        ticker="NTAP", model_name="claude-opus-4-8",
    )
    assert rec is not None
    assert rec["agent"] == "Conservative Analyst"
    assert rec["kind"] == pfl.KIND_PARSE_FAILED
    assert rec["site"] == "unit:test"
    assert rec["ticker"] == "NTAP"
    assert rec["model_name"] == "claude-opus-4-8"
    # The rail is the field the whole log corpus lacks. Assert the VALUE, not
    # membership in the vocabulary: a membership assertion passes for every
    # wrong answer, which is exactly how the first revision of this guard let
    # an inverted attribution survive.
    assert rec["rail"] == pfl.RAIL_CLAUDE_CODE
    assert rec["rail_basis"].startswith("measured:")


# ── 1b. the rail is MEASURED from the model, not inherited from a flag ──


@pytest.mark.parametrize(
    "model_name, cc_route, expected",
    [
        ("claude-opus-4-8", True, pfl.RAIL_CLAUDE_CODE),
        # The case the flag-only rule got WRONG: CC route on, Gemini model.
        ("gemini-2.5-flash", True, pfl.RAIL_DIRECT),
        ("claude-opus-4-8", False, pfl.RAIL_DIRECT),
        ("gemini-2.5-flash", False, pfl.RAIL_DIRECT),
        ("gpt-4o", True, pfl.RAIL_DIRECT),
    ],
)
def test_resolve_rail_truth_table(monkeypatch, model_name, cc_route, expected):
    _settings_stub(monkeypatch, cc_route=cc_route)
    rail, basis = pfl.resolve_rail(model_name)
    assert rail == expected, f"{model_name} @ cc_route={cc_route} -> {rail} ({basis})"


def test_resolve_rail_disagrees_with_the_flag_only_rule(monkeypatch):
    """The discriminating guard. It fails the moment anyone reverts to reading
    `paper_use_claude_code_route` on its own.

    The client enters the CC rail only when the model is a `claude-` variant
    AND the route flag is on. A flag-only rule stamps every Gemini-served call
    `claude_code`, and the call log shows Gemini traffic outnumbering
    claude-code-tagged traffic by roughly 20x -- so the misattribution would be
    the common case, not an edge case.
    """
    _settings_stub(monkeypatch, cc_route=True)
    flag_only = pfl.RAIL_CLAUDE_CODE  # what the flag alone would say, for both
    assert pfl.resolve_rail("claude-opus-4-8")[0] == flag_only     # agrees here
    assert pfl.resolve_rail("gemini-2.5-flash")[0] != flag_only    # and MUST differ here


def test_resolve_rail_is_unknown_with_a_stated_basis_when_no_model():
    rail, basis = pfl.resolve_rail(None)
    assert rail == pfl.RAIL_UNKNOWN
    assert "no_model_in_scope" in basis
    assert pfl.resolve_rail("")[0] == pfl.RAIL_UNKNOWN


def test_resolve_rail_is_unknown_when_failforward_can_substitute(monkeypatch):
    """With fail-forward armed, a claude- model under an enabled CC route can
    still be served by the Vertex-Gemini workhorse. The flags cannot tell which
    happened, so the record must not claim one."""
    _settings_stub(monkeypatch, cc_route=True, failforward=True)
    rail, basis = pfl.resolve_rail("claude-opus-4-8")
    assert rail == pfl.RAIL_UNKNOWN
    assert "failforward" in basis


def test_resolve_rail_is_unknown_when_settings_are_unavailable(monkeypatch):
    import backend.config.settings as cs

    def _explode():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(cs, "get_settings", _explode)
    rail, basis = pfl.resolve_rail("claude-opus-4-8")
    assert rail == pfl.RAIL_UNKNOWN
    assert basis == "settings_unavailable"


def test_emit_sites_pass_the_model_through_to_the_record(monkeypatch):
    """The threading is real, not just an accepted kwarg."""
    from backend.agents.debate import _parse_json as debate_parse
    from backend.agents.risk_debate import _parse_json as risk_parse
    from backend.agents.orchestrator import _parse_json_with_fallback as orch_parse

    _settings_stub(monkeypatch, cc_route=True)
    assert debate_parse("nope", "Moderator", model_name="gemini-2.5-flash") is None
    assert risk_parse("nope", "Risk Judge", model_name="claude-opus-4-8") is None
    assert orch_parse("}{", "Critic", model_name=None) is None

    by_model = {r["agent"]: (r["model_name"], r["rail"]) for r in pfl.snapshot()["recent"]}
    assert by_model["Moderator"] == ("gemini-2.5-flash", pfl.RAIL_DIRECT)
    assert by_model["Risk Judge"] == ("claude-opus-4-8", pfl.RAIL_CLAUDE_CODE)
    assert by_model["Critic"] == (None, pfl.RAIL_UNKNOWN)


def test_agent_phrase_is_preserved_verbatim_not_collapsed():
    """The original filing's "Analyst 926" was three distinct agents folded into
    one bucket by a match rule. The ledger must not repeat that fold."""
    for phrase in ("Conservative Analyst", "Neutral Analyst", "Aggressive Analyst"):
        pfl.record_parse_failure(phrase, pfl.KIND_PARSE_FAILED, site="unit:test")
    snap = pfl.snapshot()
    keys = set(snap["by_agent_kind"])
    assert keys == {
        "Conservative Analyst|parse_failed",
        "Neutral Analyst|parse_failed",
        "Aggressive Analyst|parse_failed",
    }, keys
    assert all(v == 1 for v in snap["by_agent_kind"].values())


# ── 2. the kinds stay distinct ─────────────────────────────────────


def test_the_three_kinds_are_counted_separately():
    pfl.record_parse_failure("A", pfl.KIND_PARSE_FAILED, site="s")
    pfl.record_parse_failure("A", pfl.KIND_TRUNCATED, site="s")
    pfl.record_parse_failure("A", pfl.KIND_SCHEMA_VALID_BUT_REJECTED, site="s")
    snap = pfl.snapshot()
    assert snap["by_agent_kind"] == {
        "A|parse_failed": 1,
        "A|truncated": 1,
        "A|schema_valid_but_rejected_downstream": 1,
    }


def test_an_unrecognised_kind_does_not_invent_a_bucket():
    rec = pfl.record_parse_failure("A", "typo_kind", site="s")
    assert rec["kind"] == pfl.KIND_UNKNOWN
    assert "A|typo_kind" not in pfl.snapshot()["by_agent_kind"]


# ── 3. recording never changes control flow (the REAL functions) ───


def test_debate_parse_json_return_values_are_unchanged_by_recording():
    from backend.agents.debate import _parse_json

    assert _parse_json('{"a": 1}', "Moderator") == {"a": 1}
    assert _parse_json("not json at all", "Moderator") is None
    assert _parse_json("[1, 2, 3]", "Moderator") is None  # valid JSON, not a dict
    snap = pfl.snapshot()
    assert snap["by_agent_kind"] == {
        "Moderator|parse_failed": 1,
        "Moderator|schema_valid_but_rejected_downstream": 1,
    }
    assert snap["by_site"] == {"debate.py:_parse_json": 2}


def test_risk_debate_parse_json_return_values_are_unchanged_by_recording():
    from backend.agents.risk_debate import _parse_json

    assert _parse_json('{"verdict": "REJECT"}', "Risk Judge") == {"verdict": "REJECT"}
    assert _parse_json("<<garbled>>", "Risk Judge") is None
    assert _parse_json('"a bare string"', "Risk Judge") is None
    snap = pfl.snapshot()
    assert snap["by_site"] == {"risk_debate.py:_parse_json": 2}
    assert snap["by_agent_kind"] == {
        "Risk Judge|parse_failed": 1,
        "Risk Judge|schema_valid_but_rejected_downstream": 1,
    }


def test_orchestrator_parse_json_with_fallback_records_and_returns_none():
    from backend.agents.orchestrator import _parse_json_with_fallback

    assert _parse_json_with_fallback('{"x": 2}', "Synthesis-Final") == {"x": 2}
    assert _parse_json_with_fallback("}{", "Synthesis-Final") is None
    snap = pfl.snapshot()
    assert snap["by_site"] == {"orchestrator.py:_parse_json_with_fallback": 1}
    assert snap["by_agent_kind"] == {"Synthesis-Final|parse_failed": 1}


def test_llm_parse_separates_truncated_from_malformed():
    from backend.agents.llm_parse import parse_llm_json

    class _Resp:
        def __init__(self, text, truncated):
            self.text = text
            self.stop_reason = "max_tokens" if truncated else "end_turn"
            self._t = truncated

        def is_truncated(self):
            return self._t

    ok, degraded = parse_llm_json(_Resp('{"k": 1}', False), "Critic")
    assert ok == {"k": 1} and degraded is False

    bad, degraded = parse_llm_json(_Resp("nope", False), "Critic")
    assert bad is None and degraded is False

    cut, degraded = parse_llm_json(_Resp('{"k": ', True), "Critic")
    assert cut is None and degraded is True

    non_dict, _ = parse_llm_json(_Resp("[1,2]", False), "Critic")
    assert non_dict is None

    snap = pfl.snapshot()
    assert snap["by_agent_kind"] == {
        "Critic|parse_failed": 1,
        "Critic|truncated": 1,
        "Critic|schema_valid_but_rejected_downstream": 1,
    }, snap["by_agent_kind"]
    assert snap["by_site"] == {"llm_parse.py:parse_llm_json": 3}


# ── 4. counter vs gauge, and the fail-open guard is observable ─────


def test_records_seen_is_a_counter_and_reconciles_after_eviction():
    small = pfl._Ledger(maxlen=3)
    for i in range(10):
        small.record({"agent": "A", "kind": "parse_failed", "site": "s",
                      "rail": "unknown", "ts": "", "detail": "", "ticker": None})
    snap = small.snapshot()
    assert snap["records_seen"] == 10          # counter: every event
    assert snap["records_retained"] == 3       # gauge: ring occupancy
    assert snap["evicted"] == 7
    assert snap["reconciles"] is True
    assert snap["by_agent_kind"]["A|parse_failed"] == 10  # counter, not pruned


def test_recorder_failure_is_counted_and_does_not_propagate(monkeypatch):
    """A fail-open guard that hides its own breakage is worse than no guard."""
    def _boom(*_a, **_k):
        raise RuntimeError("ledger is wedged")

    monkeypatch.setattr(pfl._LEDGER, "record", _boom)
    rec = pfl.record_parse_failure("A", pfl.KIND_PARSE_FAILED, site="s")
    assert rec is None                                   # did not raise
    assert pfl.snapshot()["recorder_errors"] == 1        # and did not hide


def test_a_wedged_recorder_still_does_not_break_the_parse_path(monkeypatch):
    from backend.agents.debate import _parse_json

    monkeypatch.setattr(
        pfl._LEDGER, "record",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("wedged")),
    )
    assert _parse_json("not json", "Moderator") is None   # unchanged behaviour
    assert _parse_json('{"a": 1}', "Moderator") == {"a": 1}


# ── 5. the read-only routes (driven through the real handlers) ─────


def test_parse_failures_route_returns_the_live_ledger():
    from backend.api.observability_api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    pfl.record_parse_failure("Devil's Advocate", pfl.KIND_PARSE_FAILED, site="unit:route")
    body = client.get("/api/observability/parse-failures").json()
    assert body["records_seen"] == 1
    assert body["by_agent_kind"] == {"Devil's Advocate|parse_failed": 1}
    assert body["recent"][0]["agent"] == "Devil's Advocate"
    assert body["reconciles"] is True


def test_flags_route_reports_in_force_and_env_for_the_named_dark_flags():
    from backend.api.settings_api import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    named = [
        "paper_synthesis_integrity_enabled",
        "paper_position_recommendation_fix_enabled",
        "paper_risk_judge_shape_fix_enabled",
        "claude_code_timeout_s",
        "claude_code_empty_retry_max",
        "paper_soft_sector_diversity_enabled",
        "paper_soft_sector_diversity_w",
    ]
    r = client.get("/api/settings/flags", params={"only": ",".join(named)})
    assert r.status_code == 200
    body = r.json()
    assert set(body["flags"]) == set(named), body["flags"].keys()
    for name in named:
        row = body["flags"][name]
        assert "in_force" in row and "env_file" in row and "divergent" in row
    # The step was filed because GET /api/settings/ exposes none of these.
    full = client.get("/api/settings/").json()
    assert not (set(named) & set(full)), "a named dark flag is already in FullSettings"


def test_flags_route_reports_an_unknown_name_instead_of_dropping_it():
    from backend.api.settings_api import router

    app = FastAPI()
    app.include_router(router)
    body = TestClient(app).get(
        "/api/settings/flags", params={"only": "paper_swap_enabled,not_a_real_flag"}
    ).json()
    assert body["requested_but_unknown"] == ["not_a_real_flag"]
    assert set(body["flags"]) == {"paper_swap_enabled"}


# ── 6. the population rule, and the secret guard ───────────────────


def test_population_is_derived_and_excludes_what_fullsettings_exposes():
    from backend.api.settings_api import FullSettings
    from backend.config.gated_flags import gated_scalar_names
    from backend.config.settings import Settings

    names = set(gated_scalar_names())
    exposed = set(FullSettings.model_fields)
    assert not (names & exposed), "a field is reported as hidden AND exposed"
    # Recompute the rule independently of the implementation.
    expected = {
        n for n, f in Settings.model_fields.items()
        if n not in exposed and f.annotation in (bool, int, float)
    }
    assert names == expected
    # The step named seven; the real population is far larger, which is the
    # finding. Pin the direction, not a brittle exact count.
    assert len(names) > 100, len(names)
    assert "paper_synthesis_integrity_enabled" in names


def test_no_string_typed_field_can_enter_the_payload():
    """The guard that keeps an API key out of an observability response.

    Asserted against the live `Settings` model, so a newly added secret is
    covered without anyone remembering to extend a denylist.
    """
    from backend.config.gated_flags import gated_flag_report
    from backend.config.settings import Settings

    reported = set(gated_flag_report()["flags"])
    non_scalar = {
        n for n, f in Settings.model_fields.items()
        if f.annotation not in (bool, int, float)
    }
    leaked = reported & non_scalar
    assert not leaked, f"non-scalar field(s) reached the payload: {sorted(leaked)}"
    # And prove the guard has something to exclude -- a vacuous assertion over
    # an empty exclusion set would pass no matter what.
    assert len(non_scalar) > 20, len(non_scalar)


def test_divergence_between_env_and_the_running_process_is_detected(tmp_path, monkeypatch):
    """The committed-is-not-in-force case, driven end to end."""
    from backend.config import gated_flags

    env = tmp_path / ".env"
    # in_force for paper_swap_enabled defaults True; write the opposite.
    env.write_text("PAPER_SWAP_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setattr(gated_flags, "_ENV_FILE", env)

    body = gated_flags.gated_flag_report(only=["paper_swap_enabled"])
    row = body["flags"]["paper_swap_enabled"]
    assert row["env_file_present"] is True
    assert row["env_file"] is False
    assert row["in_force"] is True
    assert row["divergent"] is True
    assert body["divergent"] == ["paper_swap_enabled"]


def test_env_reader_returns_only_requested_keys_and_ignores_comments(tmp_path, monkeypatch):
    """It must not be usable as a file-dump primitive."""
    from backend.config import gated_flags

    env = tmp_path / ".env"
    env.write_text(
        "ANTHROPIC_API_KEY=sk-super-secret\n"
        "# PAPER_SWAP_ENABLED=true\n"
        "PAPER_SWAP_MAX_PER_CYCLE=7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gated_flags, "_ENV_FILE", env)

    raw = gated_flags._read_env_raw({"PAPER_SWAP_MAX_PER_CYCLE"})
    assert raw == {"PAPER_SWAP_MAX_PER_CYCLE": "7"}
    assert "ANTHROPIC_API_KEY" not in raw
    # A commented-out line is not a value.
    assert gated_flags._read_env_raw({"PAPER_SWAP_ENABLED"}) == {}


def test_report_is_json_serialisable():
    """An observability route that cannot be serialised is not observable."""
    from backend.config.gated_flags import gated_flag_report

    json.dumps(gated_flag_report())
    json.dumps(pfl.snapshot())


# ── 7. the PRODUCTION CALL SITES, not just the record seam ─────────
#
# Cycle 2 fixed the rail at the record seam and built every guard there. A Q/A
# then executed a mutant that hardcoded `_effective_model_name` to a Claude
# model: it SURVIVED 29/29, reinstating the original misattribution one call
# frame upstream of the fix. The lesson is the project's own -- a guard that
# stops one seam short of the defect is not a guard -- so this section drives
# the real production entry points and, separately, checks the call sites for
# completeness.


class _FakeLLMClient:
    """Minimal stand-in for LLMClient: returns text that cannot parse."""

    supports_thinking = False

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_content(self, prompt, generation_config=None):
        return types.SimpleNamespace(text="NOT JSON AT ALL", thoughts="", usage_metadata=None)


def test_run_risk_debate_records_the_real_client_model_on_every_agent(monkeypatch):
    """Drives the REAL `run_risk_debate`. Four agents, four records, each
    carrying the model the fake client actually reported."""
    from backend.agents.risk_debate import run_risk_debate

    _settings_stub(monkeypatch, cc_route=True)
    run_risk_debate(
        model=_FakeLLMClient("gemini-2.5-flash"), ticker="NTAP",
        synthesis_result={"recommendation": "Hold", "final_score": 5},
        enrichment_signals={}, max_risk_rounds=1,
    )
    recs = {r["agent"]: r for r in pfl.snapshot()["recent"]}
    assert set(recs) == {
        "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst", "Risk Judge",
    }, sorted(recs)
    for agent, rec in recs.items():
        # The CC-route flag is ON. A hardcoded or flag-derived model would read
        # claude_code here; the real client's Gemini name must win.
        assert rec["model_name"] == "gemini-2.5-flash", (agent, rec)
        assert rec["rail"] == pfl.RAIL_DIRECT, (agent, rec)


def test_run_debate_records_the_real_client_model(monkeypatch):
    """Drives the REAL `run_debate` (the other production entry point)."""
    from backend.agents.debate import run_debate

    _settings_stub(monkeypatch, cc_route=True)
    run_debate(
        model=_FakeLLMClient("gemini-2.5-flash"), ticker="NTAP",
        enrichment_signals={}, trace_summary={}, max_debate_rounds=1,
    )
    recs = [r for r in pfl.snapshot()["recent"] if r["site"] == "debate.py:_parse_json"]
    assert recs, "run_debate produced no ledger record at all"
    for rec in recs:
        assert rec["model_name"] == "gemini-2.5-flash", rec
        assert rec["rail"] == pfl.RAIL_DIRECT, rec


@pytest.mark.parametrize(
    "declared, client_name, expected",
    [
        ("gemini-2.5-flash", "claude-opus-4-8", "gemini-2.5-flash"),  # declared wins
        ("", "claude-opus-4-8", "claude-opus-4-8"),                   # falls back
        ("", "", None),                                              # neither -> None
        ("", None, None),
    ],
)
def test_effective_model_name_resolution(declared, client_name, expected):
    """Unit-guards the helper itself. A mutant that returns a constant -- the
    one the Q/A executed -- fails the first two rows."""
    from backend.agents.debate import _effective_model_name as dbg
    from backend.agents.risk_debate import _effective_model_name as rdg

    client = types.SimpleNamespace(model_name=client_name)
    assert dbg(declared, client) == expected
    assert rdg(declared, client) == expected


_RESOLVERS = {"_effective_model_name", "_client_model_name"}


def _accepted_model_name_arg(node) -> tuple[bool, str]:
    """WHITELIST the accepted shapes for a `model_name=` argument.

    A first version of this guard BLACKLISTED one shape -- an `ast.Constant` --
    and a Q/A immediately executed two AST-legal mutants that survived it:
    `_client_model_name(None)`, which degrades every record to a FALSE
    `unknown` while a model is in scope, and `_client_model_name(c) or
    "claude-opus-4-8"`, which actively misattributes. Both are Calls, not
    Constants, so a blacklist could not see them.

    The lesson is the general one: rejecting one syntactic form does not cover
    a semantic class. This function instead names what IS acceptable -- a call
    to one of the two resolvers, with at least one argument and no argument
    that is a bare `None` -- and rejects everything else, so a mutation has to
    look like the real thing to pass.
    """
    import ast as _ast

    if not isinstance(node, _ast.Call):
        shape = type(node).__name__
        val = f" {node.value!r}" if isinstance(node, _ast.Constant) else ""
        return False, f"model_name= is a {shape}{val}, not a resolver call"
    fn = node.func
    name = fn.id if isinstance(fn, _ast.Name) else getattr(fn, "attr", None)
    if name not in _RESOLVERS:
        return False, f"model_name= calls {name!r}, not one of {sorted(_RESOLVERS)}"
    if not node.args:
        return False, f"{name}() is called with no argument"
    for a in node.args:
        if isinstance(a, _ast.Constant) and a.value is None:
            return False, f"{name}(None) -- a model IS in scope; this fakes 'unknown'"
    return True, ""


def test_every_parse_call_site_forwards_a_model_name():
    """COMPLETENESS guard, and deliberately a different kind from the two above.

    The drivers prove the six `debate` / `risk_debate` call sites behave. They
    cannot prove anything about a call site that does not exist yet, and they do
    not reach `orchestrator.py`'s three sites, which sit inside the synthesis
    loop of `run_full_analysis` and are not drivable without the whole
    orchestrator. This is a static check over the AST -- it observes no
    behaviour and is not offered as a behavioural guard -- but it is the only
    thing that catches a NEW call site added without threading the model, which
    is how this defect would return.

    The acceptance rule is a WHITELIST (`_accepted_model_name_arg`), not a
    blacklist. See that function for the two AST-legal mutants that defeated
    the blacklist version.
    """
    import ast
    import pathlib

    targets = {"_parse_json", "_parse_json_with_fallback"}
    missing = []
    for rel in ("backend/agents/debate.py", "backend/agents/risk_debate.py",
                "backend/agents/orchestrator.py"):
        tree = ast.parse(pathlib.Path(rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name not in targets:
                continue
            kw = {k.arg: k.value for k in node.keywords}
            if "model_name" not in kw:
                missing.append(f"{rel}:{node.lineno} (no model_name=)")
            else:
                ok, why = _accepted_model_name_arg(kw["model_name"])
                if not ok:
                    missing.append(f"{rel}:{node.lineno} ({why})")

    assert not missing, f"parse call site(s) not forwarding a real model_name: {missing}"
    # Anti-vacuity: the scan must actually have found call sites. An empty
    # population would make the assertion above pass for the wrong reason.
    found = 0
    for rel in ("backend/agents/debate.py", "backend/agents/risk_debate.py",
                "backend/agents/orchestrator.py"):
        tree = ast.parse(pathlib.Path(rel).read_text(encoding="utf-8"))
        found += sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)) in targets
        )
    assert found >= 9, f"expected >=9 parse call sites across the three modules, found {found}"


def test_client_model_name_unit():
    """The orchestrator's three call sites go through this named unit."""
    from backend.agents.orchestrator import _client_model_name

    assert _client_model_name(types.SimpleNamespace(model_name="claude-opus-4-8")) == "claude-opus-4-8"
    assert _client_model_name(types.SimpleNamespace(model_name="")) is None
    assert _client_model_name(types.SimpleNamespace()) is None
    assert _client_model_name(None) is None
