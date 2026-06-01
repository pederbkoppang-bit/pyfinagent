"""phase-50.6: paper_markets is exposed + round-trips through the settings API.

The settings UI writes settings to .env via settings_api; a list field must
serialize as CSV (the PUT loop) and parse back through the settings.py
field_validator (the 54.1 fix). These tests prove the read exposure + the
CSV<->list round-trip WITHOUT writing the real .env (no live PUT call).
"""
from __future__ import annotations


def test_fullsettings_exposes_paper_markets():
    from backend.api.settings_api import FullSettings
    assert "paper_markets" in FullSettings.model_fields
    # default is US-only (byte-identical to today) -- asserted on the field default
    # (FullSettings has other required fields, so don't construct it bare)
    assert FullSettings.model_fields["paper_markets"].default == ["US"]


def test_settingsupdate_accepts_paper_markets():
    from backend.api.settings_api import SettingsUpdate
    body = SettingsUpdate(paper_markets=["US", "EU", "KR"])
    assert body.paper_markets == ["US", "EU", "KR"]
    # optional: omitted -> None (excluded from the PUT diff)
    assert SettingsUpdate().paper_markets is None


def test_settings_to_full_reads_paper_markets():
    from backend.api.settings_api import _settings_to_full
    from backend.config.settings import get_settings
    full = _settings_to_full(get_settings())
    # whatever the live .env holds, it is a non-empty list of market codes
    assert isinstance(full.paper_markets, list)
    assert all(isinstance(m, str) and m for m in full.paper_markets)


def test_csv_serialization_round_trips_through_validator():
    """The PUT loop serializes a list as CSV; settings.py's _parse_paper_markets
    parses it back. Prove the exact round-trip the UI relies on."""
    from backend.config.settings import Settings
    value = ["US", "EU", "KR"]
    # mirror the PUT loop's list branch:
    env_value = ",".join(str(x) for x in value)
    assert env_value == "US,EU,KR"
    # mirror the read side (settings.py field_validator, mode="before"):
    parsed = Settings._parse_paper_markets(env_value)
    assert parsed == ["US", "EU", "KR"]


def test_csv_round_trip_single_market():
    from backend.config.settings import Settings
    env_value = ",".join(["US"])
    assert Settings._parse_paper_markets(env_value) == ["US"]
