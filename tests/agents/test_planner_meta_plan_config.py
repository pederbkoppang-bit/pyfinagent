"""phase-23.8.0 (R-4): PlannerAgent reads META_PLAN from meta_plan.json.

Asserts:
  1. meta_plan.json exists at the canonical path.
  2. The JSON has all 7 required numeric keys.
  3. _load_meta_plan_text() returns a string containing all 7 values
     formatted into the STRATEGIC GOAL block.
  4. PlannerAgent.__init__() populates self.meta_plan_text from the
     JSON (no Anthropic API call needed for this check).
  5. Editing the JSON propagates through to the rendered prompt
     (round-trip).

Audit basis: docs/audits/dev-mas-2026-05-11/04-remediation.md R-4.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.agents.planner_agent import (
    _META_PLAN_JSON_PATH,
    _load_meta_plan_text,
)


REQUIRED_KEYS = (
    "sharpe_target",
    "annual_return_min_pct",
    "annual_return_max_pct",
    "max_drawdown_pct",
    "max_trades_per_month",
    "sector_concentration_max_pct",
    "cost_stress_multiple",
)


def test_meta_plan_json_exists_at_canonical_path():
    assert _META_PLAN_JSON_PATH.exists(), (
        f"meta_plan.json must exist at {_META_PLAN_JSON_PATH}"
    )


def test_meta_plan_json_has_seven_required_keys():
    data = json.loads(_META_PLAN_JSON_PATH.read_text(encoding="utf-8"))
    for k in REQUIRED_KEYS:
        assert k in data, f"meta_plan.json missing key: {k}"
        assert isinstance(data[k], (int, float)), (
            f"{k} must be numeric, got {type(data[k]).__name__}"
        )


def test_load_meta_plan_text_renders_all_values():
    text = _load_meta_plan_text()
    assert "STRATEGIC GOAL" in text
    data = json.loads(_META_PLAN_JSON_PATH.read_text(encoding="utf-8"))
    assert f"> {data['sharpe_target']}" in text
    assert f"{data['annual_return_min_pct']}-{data['annual_return_max_pct']}%" in text
    assert f"< {data['max_drawdown_pct']}%" in text
    assert f"<{data['max_trades_per_month']}/month" in text
    assert f"> {data['sector_concentration_max_pct']}%" in text
    assert f"{data['cost_stress_multiple']}×" in text


def test_load_meta_plan_text_uses_overridden_path(tmp_path):
    """Round-trip: edit the JSON, confirm the text reflects the edit."""
    custom_json = tmp_path / "meta_plan_test.json"
    custom_json.write_text(json.dumps({
        "sharpe_target": 2.5,
        "annual_return_min_pct": 11,
        "annual_return_max_pct": 22,
        "max_drawdown_pct": 7,
        "max_trades_per_month": 13,
        "sector_concentration_max_pct": 17,
        "cost_stress_multiple": 3,
    }))
    text = _load_meta_plan_text(custom_json)
    assert "> 2.5" in text
    assert "11-22%" in text
    assert "< 7%" in text
    assert "<13/month" in text
    assert "> 17%" in text
    assert "3×" in text


def test_planner_agent_init_loads_meta_plan_text(monkeypatch):
    """PlannerAgent.__init__() populates self.meta_plan_text from JSON.

    We patch Anthropic() to avoid real API client construction, which
    is unrelated to the R-4 contract.
    """
    from backend.agents import planner_agent as pa

    class _StubAnthropic:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(pa, "Anthropic", _StubAnthropic)
    agent = pa.PlannerAgent()
    assert hasattr(agent, "meta_plan_text")
    assert isinstance(agent.meta_plan_text, str)
    assert "STRATEGIC GOAL" in agent.meta_plan_text


def test_no_hardcoded_meta_plan_string_remains():
    """phase-23.8.0 success criterion: the old hardcoded constant is gone."""
    src = (_META_PLAN_JSON_PATH.parents[3] / "backend" / "agents" / "planner_agent.py").read_text(
        encoding="utf-8"
    )
    # The OLD hardcoded constant assignment pattern - must not exist
    # anywhere in the module after R-4. (The numeric values may still
    # appear in test fixtures, but never as a `META_PLAN = """...` assignment.)
    assert "META_PLAN = \"\"\"" not in src and "META_PLAN = '''" not in src, (
        "Hardcoded META_PLAN triple-quoted constant must be removed; "
        "values now live in meta_plan.json (R-4)."
    )
