"""phase-78.16: `make_client` must not silently discard a caller's explicit
prompt-caching intent.

WHY THIS FILE EXISTS. Step 78.1 rewired six signal-overlay services from
`ClaudeClient(..., enable_prompt_caching=False)` onto `make_client(...)`.
`make_client` constructed `ClaudeClient(model_name=..., api_key=...)` with no
such kwarg, against a constructor default of `True` (llm_client.py:1345). So the
documented ONE-FLAG REVERT -- `PAPER_USE_CLAUDE_CODE_ROUTE=false`, which is also
78.1's own criterion 5 -- stopped restoring pre-78.1 behaviour: on the metered
path the `system` field changes from a plain 19,075-char `str` into a 1-block
list carrying `cache_control {"type":"ephemeral","ttl":"1h"}`. A different
request shape on the money path, reached by a flag the operator is told is safe.

The blast radius is exactly the revert path: `ClaudeCodeClient` has no caching
notion at all (it shells out to the `claude` CLI, which manages its own), so this
parameter only ever bites the metered/direct branch.

DESIGN NOTE -- why the per-service test reads the source instead of hardcoding
`False`. A test that simply called `make_client(..., enable_prompt_caching=False)`
would prove `make_client` forwards the kwarg but would stay GREEN if a service
stopped passing it -- the exact regression 78.1 shipped. So
`_service_caching_kwarg()` EXTRACTS the literal each service actually passes and
feeds that through the real construction path.

Precisely which assertion fires, measured rather than assumed (the first version
of this docstring got the causal story wrong; Q/A finding N2):

- **A service drops the kwarg** -> the extracted value is `None` -> the test dies
  at the `assert intent is False` guard, BEFORE `_wire_kwargs` is ever called.
  The wire-shape leg is not reached in that scenario. It is nonetheless
  independently red there: deleting the kwarg and passing `None` through
  `make_client` yields the cached-list shape, which `_assert_uncached_shape`
  rejects.
- **`make_client` stops forwarding** -> the intent extracts as `False` fine, and
  the test dies in the wire-shape leg instead.

So both failure directions are covered, by two different assertions, and neither
can be satisfied by a mention in a comment or a docstring.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Hermetic: the documented hatch for the cost gate in generate_content.
os.environ.setdefault("COST_BUDGET_HARD_BLOCK_DISABLED", "1")

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


def _service_caching_kwarg(mod: str):
    """The literal `enable_prompt_caching=` value the service passes to
    `make_client`, or `None` if it passes none (which is the defect shape).

    Reads the AST of the actual call node, so a mention in a comment or a
    docstring cannot satisfy it -- and this module's own prose mentions the
    kwarg repeatedly, so a substring scan would flag itself.
    """
    tree = ast.parse(_module_path(mod).read_text(encoding="utf-8"))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (getattr(n.func, "id", None) == "make_client"
             or getattr(n.func, "attr", None) == "make_client")
    ]
    assert calls, f"{mod}.py does not call make_client at all"
    assert len(calls) == 1, (
        f"{mod}.py has {len(calls)} make_client calls; this test assumes one "
        f"-- update it deliberately rather than loosening the assertion"
    )
    for kw in calls[0].keywords:
        if kw.arg == "enable_prompt_caching":
            assert isinstance(kw.value, ast.Constant), (
                f"{mod}.py passes a non-literal enable_prompt_caching; this test "
                f"can only evaluate literals"
            )
            return kw.value.value
    return None


# ── SDK-boundary fake (same shape as test_claude_request_shapes.py:36-65) ──

def _fake_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="{}", citations=None)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        model="stub",
        _request_id="req_test",
    )


class _CaptureMessages:
    def __init__(self, captured: list):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.append(kwargs)
        return _fake_response()


def _fake_client(captured: list):
    return SimpleNamespace(
        messages=_CaptureMessages(captured),
        beta=SimpleNamespace(messages=_CaptureMessages(captured)),
    )


class _RailOffSettings:
    """A real key AND the rail flag OFF -- i.e. the state the operator lands in
    after the documented one-flag revert. Mirrors the settings stand-in at
    test_phase_78_1_c_block_rail.py:157-168 so the two files agree on what
    'metered path' means."""
    anthropic_api_key = "sk-ant-api-test-not-real"
    github_token = ""
    gemini_api_key = ""
    openai_api_key = ""
    claude_code_timeout_s = 60
    paper_use_claude_code_route = False


class _RailOnSettings(_RailOffSettings):
    paper_use_claude_code_route = True


def _wire_kwargs(monkeypatch, model: str, **make_client_kwargs) -> dict:
    """Drive the REAL make_client -> ClaudeClient -> request-assembly path and
    return the kwargs handed to the SDK. Only the SDK boundary is faked."""
    from backend.agents.llm_client import ClaudeClient, make_client

    captured: list = []
    monkeypatch.setattr(
        ClaudeClient, "_get_client", lambda self: _fake_client(captured)
    )
    import backend.services.observability as obs
    monkeypatch.setattr(obs, "log_llm_call", lambda *a, **k: None, raising=False)

    client = make_client(model, None, _RailOffSettings(), **make_client_kwargs)
    assert isinstance(client, ClaudeClient), (
        f"expected the metered client with the rail flag OFF, got "
        f"{type(client).__name__}"
    )
    # The config shape the six actually pass: a pre-cleaned DICT schema (not a
    # pydantic class), which is why their cached block is byte-identical.
    client.generate_content(
        "PROBE",
        {
            "response_schema": {"type": "object", "properties": {}},
            "response_mime_type": "application/json",
            "max_output_tokens": 512,
        },
    )
    assert captured, "no request captured at the SDK boundary"
    return captured[0]


def _assert_uncached_shape(k: dict, why: str):
    system = k["system"]
    assert isinstance(system, str), (
        f"{why}: `system` is a {type(system).__name__}, not a plain str -- "
        f"prompt caching is ON where it must be OFF"
    )
    assert "cache_control" not in system, (
        f"{why}: cache_control leaked into the system STRING"
    )


def _assert_cached_shape(k: dict, why: str):
    system = k["system"]
    assert isinstance(system, list) and len(system) == 1, (
        f"{why}: expected a 1-block list, got {type(system).__name__}"
    )
    assert system[0].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}, (
        f"{why}: cache_control missing or wrong: {system[0].get('cache_control')!r}"
    )


# ── Criterion 1 + 2: the revert path reproduces the pre-78.1 request shape ──

@pytest.mark.parametrize("mod,attr", sorted(C_BLOCK.items()))
def test_revert_path_restores_pre_78_1_request_shape(mod, attr, monkeypatch):
    """For each of the six: take the caching intent the SERVICE ACTUALLY PASSES,
    push it through the real make_client path with the rail flag OFF, and assert
    the wire shape matches pre-78.1 -- `system` a plain str, no cache_control.

    Fails if the service stops passing the kwarg (extracted None -> class
    default True) OR if make_client stops forwarding it.
    """
    from backend.config.settings import get_settings

    model = getattr(get_settings(), attr, None)
    assert model, f"{mod}: settings.{attr} is unset; cannot test its routing"

    intent = _service_caching_kwarg(mod)
    assert intent is False, (
        f"{mod}.py no longer passes enable_prompt_caching=False to make_client "
        f"(got {intent!r}). Before 78.1 it constructed "
        f"ClaudeClient(..., enable_prompt_caching=False) explicitly; dropping "
        f"that intent is what broke the one-flag revert in the first place."
    )

    k = _wire_kwargs(monkeypatch, model, enable_prompt_caching=intent)
    _assert_uncached_shape(k, f"{mod} on the revert path")


# ── The parameter is honoured in BOTH directions ──────────────────────────

def test_make_client_forwards_caching_true(monkeypatch):
    """Guards the other direction: if make_client ignored the parameter, the
    revert test above could pass for the wrong reason (e.g. someone flipped the
    ClaudeClient class default to False)."""
    k = _wire_kwargs(monkeypatch, "claude-haiku-4-5", enable_prompt_caching=True)
    _assert_cached_shape(k, "enable_prompt_caching=True through make_client")


def test_make_client_default_leaves_class_default_untouched(monkeypatch):
    """BLAST-RADIUS GUARD. The 7 make_client callers outside the C-block pass no
    caching argument. `None` must mean 'no opinion' and leave the ClaudeClient
    class default (True) in place -- NOT be coerced to False. If this goes red,
    phase-78.16 changed behaviour for callers it was never scoped to touch."""
    k = _wire_kwargs(monkeypatch, "claude-haiku-4-5")
    _assert_cached_shape(k, "make_client called with no caching argument")


# ── The rail path is unaffected ───────────────────────────────────────────

def test_rail_path_ignores_the_parameter():
    """ClaudeCodeClient has no caching notion (the CLI manages its own), so the
    parameter must be accepted and ignored, not raise, when the rail is ON."""
    from backend.agents.claude_code_client import ClaudeCodeClient
    from backend.agents.llm_client import make_client

    for intent in (True, False, None):
        client = make_client(
            "claude-haiku-4-5", None, _RailOnSettings(),
            enable_prompt_caching=intent,
        )
        assert isinstance(client, ClaudeCodeClient), (
            f"rail flag ON with enable_prompt_caching={intent!r} did not return "
            f"the rail client (got {type(client).__name__})"
        )
        assert not hasattr(client, "enable_prompt_caching"), (
            "ClaudeCodeClient grew an enable_prompt_caching attribute; if the "
            "rail really can control caching now, this test and the make_client "
            "docstring both need revisiting rather than silently passing"
        )
