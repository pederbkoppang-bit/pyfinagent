"""phase-6.5.2 tests for backend.intel.source_registry."""
from __future__ import annotations

from pathlib import Path

from backend.intel.source_registry import (
    SourceRow,
    load_active_sources,
    load_from_yaml,
    upsert_sources,
)


FIXTURE = Path(__file__).parent / "fixtures" / "intel_sources.yaml"


def test_load_from_yaml_returns_all_sources():
    """Immutable criterion: registry_loads_all_configured_sources."""
    rows = load_from_yaml(FIXTURE)
    assert len(rows) == 3
    ids = {r.source_id for r in rows}
    assert ids == {"stub_http", "stub_rss", "stub_disabled"}
    # Kill-switched row IS present in the parsed list; filtering happens downstream.
    killed = [r for r in rows if r.kill_switch]
    assert len(killed) == 1
    assert killed[0].source_id == "stub_disabled"


def test_load_from_yaml_missing_file_returns_empty():
    rows = load_from_yaml(Path("/nonexistent/does/not/exist.yaml"))
    assert rows == []


def test_source_row_dataclass_fields():
    r = SourceRow(
        source_id="a",
        source_name="A",
        source_type="http",
        kill_switch=False,
        rate_limit_per_day=100,
        metadata={"k": "v"},
    )
    assert r.source_id == "a"
    assert r.kill_switch is False
    assert r.metadata == {"k": "v"}


def test_load_from_yaml_preserves_metadata():
    rows = load_from_yaml(FIXTURE)
    rss = next(r for r in rows if r.source_id == "stub_rss")
    assert rss.metadata.get("feed_url") == "https://feeds.example.com/finance.rss"


def test_upsert_fail_open_no_bq_auth():
    """Non-empty input with a guaranteed-bad project must return 0 and never raise."""
    rows = load_from_yaml(FIXTURE)
    assert rows
    result = upsert_sources(rows, project="nonexistent-fail-open-test", dataset="nx")
    assert result == 0


def test_upsert_empty_input_returns_zero():
    assert upsert_sources([]) == 0


def test_load_active_fail_open_no_bq_auth():
    result = load_active_sources(project="nonexistent-fail-open-test", dataset="nx")
    assert result == []


def test_yaml_root_not_mapping_raises_value_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    try:
        load_from_yaml(bad)
    except ValueError as exc:
        assert "mapping" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_yaml_sources_key_must_be_list(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text("sources:\n  not: a list\n", encoding="utf-8")
    try:
        load_from_yaml(bad)
    except ValueError as exc:
        assert "list" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
