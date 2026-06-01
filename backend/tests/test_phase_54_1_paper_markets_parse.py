"""phase-54.1: paper_markets must parse identically on EVERY load path.

Root cause (operator-away cron audit): the autoresearch + ablation launchd
wrappers `set -a; . backend/.env; set +a` bash-source the env. The multi-market
go-live set PAPER_MARKETS=["US","EU","KR"] (JSON). Bash strips the quotes when
sourcing -> the OS env holds the literal `[US,EU,KR]`, which pydantic-settings'
built-in complex-field JSON decoder rejected -> SettingsError at get_settings(),
crashing both nightly cron jobs (launchctl last-exit=1). uvicorn was unaffected
(native dotenv, no shell).

Fix: Annotated[list[str], NoDecode] + a field_validator(mode="before") that
accepts JSON / bracket-mangled / plain-comma / real-list forms. This test pins
that every form resolves to the same list AND that the live JSON path is
byte-identical (DO-NO-HARM).
"""
import importlib
import os

import pytest


def _settings_with_env(monkeypatch, raw):
    """Construct a fresh Settings() with PAPER_MARKETS set to `raw` in the OS env
    (which overrides the .env file in pydantic-settings priority), exercising the
    real EnvSettingsSource path -- the exact path that crashed."""
    from backend.config import settings as settings_mod
    if raw is None:
        monkeypatch.delenv("PAPER_MARKETS", raising=False)
    else:
        monkeypatch.setenv("PAPER_MARKETS", raw)
    # Settings() reads .env for the other required fields; OS env wins for ours.
    return settings_mod.Settings()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('["US","EU","KR"]', ["US", "EU", "KR"]),   # native dotenv / JSON (DO-NO-HARM)
        ("[US,EU,KR]", ["US", "EU", "KR"]),          # bash-sourced .env (the bug)
        ("US,EU,KR", ["US", "EU", "KR"]),            # plain comma
        ('["US"]', ["US"]),                          # JSON single
        ("US", ["US"]),                              # bare single
        ("[US, EU , KR]", ["US", "EU", "KR"]),       # mangled + whitespace
        ('[ "US" , "EU" ]', ["US", "EU"]),           # spaced JSON-ish
    ],
)
def test_paper_markets_parses_every_form(monkeypatch, raw, expected):
    s = _settings_with_env(monkeypatch, raw)
    assert s.paper_markets == expected


def test_paper_markets_default_factory_is_us():
    """The field default (when NEITHER OS env NOR .env file supplies a value) is
    ['US'] -- asserted on the field default_factory directly so the repo's own
    .env (which DOES set PAPER_MARKETS) can't mask it."""
    from backend.config.settings import Settings
    assert Settings.model_fields["paper_markets"].default_factory() == ["US"]


def test_paper_markets_empty_string_defaults_to_us(monkeypatch):
    s = _settings_with_env(monkeypatch, "")
    assert s.paper_markets == ["US"]


def test_live_json_path_is_byte_identical(monkeypatch):
    """DO-NO-HARM: the value the live engine resolves from the .env JSON form is
    exactly what it was before the fix -- ['US','EU','KR']."""
    s = _settings_with_env(monkeypatch, '["US","EU","KR"]')
    assert s.paper_markets == ["US", "EU", "KR"]
    assert all(isinstance(m, str) for m in s.paper_markets)


def test_bash_sourced_form_no_longer_raises(monkeypatch):
    """The exact crash repro: the bash-mangled [US,EU,KR] must now load cleanly
    (previously raised pydantic_settings SettingsError)."""
    s = _settings_with_env(monkeypatch, "[US,EU,KR]")
    assert s.paper_markets == ["US", "EU", "KR"]
