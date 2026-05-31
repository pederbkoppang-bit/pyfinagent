"""phase-51.1: SecretStr unwrap -- the SDK boundary must receive a plain str.

$0, no network: ClaudeClient/OpenAIClient/BatchClient.__init__ do NOT instantiate
the provider SDK (that is deferred to _get_client), so constructing them with a
fake key is free. Pins the root-cause fix: a non-empty pydantic SecretStr is
TRUTHY, so `getattr(settings,'k','') or ''` returned the WRAPPER, which the
Anthropic SDK put straight into the X-Api-Key header -> httpx raised
'Header value must be str or bytes, not SecretStr' -> the 4 overlay services
silently fell back. The fix MUST unwrap via .get_secret_value(), NEVER str()
(pydantic renders str(SecretStr('x')) as '**********' WITHOUT erroring -> a
silent 401; pydantic #4217).
"""
from pydantic import SecretStr

from backend.agents.llm_client import (
    unwrap_secret,
    ClaudeClient,
    OpenAIClient,
    BatchClient,
)

_MASK = "**********"  # what str(SecretStr(...)) would wrongly produce


def test_unwrap_secret_on_secretstr_uses_real_value_not_mask():
    assert unwrap_secret(SecretStr("sk-ant-test")) == "sk-ant-test"
    assert unwrap_secret(SecretStr("sk-ant-test")) != _MASK  # str() footgun guard


def test_unwrap_secret_passthrough_and_empty():
    assert unwrap_secret("already-a-str") == "already-a-str"  # no double-unwrap
    assert unwrap_secret("") == ""
    assert unwrap_secret(None) == ""


def test_secretstr_is_truthy_proving_the_original_bug():
    # documents WHY `getattr(...) or ''` failed: a non-empty SecretStr is truthy,
    # so `or ''` returned the wrapper unchanged.
    assert bool(SecretStr("x")) is True
    assert (SecretStr("x") or "") != "x"  # the wrapper survives the `or ''`


def test_claude_client_self_unwraps_secretstr():
    c = ClaudeClient(model_name="claude-haiku-4-5", api_key=SecretStr("sk-ant-test"))
    assert c._api_key == "sk-ant-test"
    assert isinstance(c._api_key, str)
    assert not hasattr(c._api_key, "get_secret_value")  # genuinely a str, not a wrapper
    assert c._api_key != _MASK


def test_claude_client_plain_str_no_double_unwrap():
    c = ClaudeClient(model_name="claude-haiku-4-5", api_key="sk-ant-test")
    assert c._api_key == "sk-ant-test"  # str passes through unchanged
    assert isinstance(c._api_key, str)


def test_sibling_clients_self_unwrap():
    o = OpenAIClient(model_name="gpt-4", api_key=SecretStr("sk-openai"))
    assert o._api_key == "sk-openai" and isinstance(o._api_key, str)
    b = BatchClient(model_name="claude-haiku-4-5", api_key=SecretStr("sk-batch"))
    assert b._api_key == "sk-batch" and isinstance(b._api_key, str)


def test_overlay_services_pass_str_to_claudeclient(monkeypatch):
    """Criterion #1: the 4 overlay services resolve a plain str api_key from a
    SecretStr setting. Mirror their exact idiom (unwrap_secret(getattr(...))) and
    assert the result is a str the SDK header will accept."""
    import backend.agents.llm_client as llm

    class _FakeSettings:
        anthropic_api_key = SecretStr("sk-ant-live")

    resolved = llm.unwrap_secret(getattr(_FakeSettings(), "anthropic_api_key", ""))
    assert resolved == "sk-ant-live"
    assert isinstance(resolved, str) and resolved != _MASK
    # and that key flows through ClaudeClient as a str (what every overlay does next)
    client = ClaudeClient(model_name="claude-haiku-4-5", api_key=resolved)
    assert isinstance(client._api_key, str) and client._api_key == "sk-ant-live"
