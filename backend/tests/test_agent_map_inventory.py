"""phase-18.1 tests for the agent topology inventory + GET /api/agent-map.

8 tests covering schema validity, node-count floor, required fields,
parent-child consistency, and endpoint round-trip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.agent_map import (  # noqa: E402
    INVENTORY_PATH,
    _derive_edges,
    get_agent_map,
)

VALID_LAYERS = {1, 2, 3, 4}
VALID_PROVIDERS = {"anthropic", "google", "openai", "github_models", "none"}
VALID_KINDS = {"harness", "in_app", "skill", "service", "meta_evolution"}
REQUIRED_NODE_FIELDS = {
    "id", "name", "layer", "model", "provider", "role", "file",
    "parents", "children", "kind",
}


@pytest.fixture(scope="module")
def inventory() -> dict:
    """Load the static inventory once per module."""
    assert INVENTORY_PATH.exists(), f"missing inventory file: {INVENTORY_PATH}"
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


# ----------------------
# Schema-level checks
# ----------------------

def test_inventory_json_loads(inventory):
    """Inventory file is valid JSON with version + nodes."""
    assert "version" in inventory
    assert isinstance(inventory["version"], int)
    assert "nodes" in inventory
    assert isinstance(inventory["nodes"], list)


def test_node_count_minimum(inventory):
    """At least 30 nodes (sanity floor; ~54 expected)."""
    n = len(inventory["nodes"])
    assert n >= 30, f"too few nodes: {n}"


def test_every_node_has_required_fields(inventory):
    for node in inventory["nodes"]:
        missing = REQUIRED_NODE_FIELDS - set(node.keys())
        assert not missing, f"node {node.get('id')!r} missing fields: {missing}"


def test_layer_values_in_range(inventory):
    for node in inventory["nodes"]:
        assert node["layer"] in VALID_LAYERS, (
            f"node {node['id']!r} has invalid layer={node['layer']}"
        )


def test_provider_values_valid(inventory):
    for node in inventory["nodes"]:
        assert node["provider"] in VALID_PROVIDERS, (
            f"node {node['id']!r} has invalid provider={node['provider']!r}"
        )


def test_kind_values_valid(inventory):
    for node in inventory["nodes"]:
        assert node["kind"] in VALID_KINDS, (
            f"node {node['id']!r} has invalid kind={node['kind']!r}"
        )


# ----------------------
# Cross-node consistency
# ----------------------

def test_node_ids_unique(inventory):
    ids = [n["id"] for n in inventory["nodes"]]
    assert len(ids) == len(set(ids)), "duplicate node ids found"


def test_parent_child_consistency(inventory):
    """If A.parents has B, then B.children should reference A (or A's id should be transitively reachable). We do a soft check: every parent reference points to an existing node id."""
    by_id = {n["id"] for n in inventory["nodes"]}
    for node in inventory["nodes"]:
        for parent in node.get("parents") or []:
            assert parent in by_id, (
                f"node {node['id']!r} references unknown parent {parent!r}"
            )
        for child in node.get("children") or []:
            assert child in by_id, (
                f"node {node['id']!r} references unknown child {child!r}"
            )


# ----------------------
# Endpoint round-trip
# ----------------------

def test_get_agent_map_endpoint_returns_inventory():
    """The endpoint function returns the inventory + derived edges."""
    out = get_agent_map()
    assert "nodes" in out
    assert "edges" in out
    assert isinstance(out["edges"], list)
    # Edges are derived; should be > 0 for a connected graph
    assert len(out["edges"]) > 0


def test_no_orphan_ids_in_edges():
    """Every derived edge must reference existing node ids."""
    out = get_agent_map()
    by_id = {n["id"] for n in out["nodes"]}
    for edge in out["edges"]:
        assert edge["from"] in by_id, f"edge from-id missing: {edge}"
        assert edge["to"] in by_id, f"edge to-id missing: {edge}"


def test_derive_edges_dedups():
    """If both A.children=[B] and B.parents=[A], emit one edge not two."""
    nodes = [
        {"id": "a", "children": ["b"], "parents": []},
        {"id": "b", "children": [], "parents": ["a"]},
    ]
    edges = _derive_edges(nodes)
    assert len(edges) == 1
    assert edges[0] == {"from": "a", "to": "b"}


# ----------------------
# phase-20.1 -- workflow data (production daily-cycle)
# ----------------------

def test_inventory_version_supports_workflow(inventory):
    """v2+ schema added workflow_steps + workflow_edges in phase-20.1.
    phase-22.1 bumped version to 3; the workflow keys remain present."""
    assert inventory["version"] >= 2


def test_workflow_steps_present(inventory):
    """8 production steps from autonomous_loop.run_daily_cycle docstring."""
    steps = inventory.get("workflow_steps")
    assert isinstance(steps, list)
    assert len(steps) == 8
    # Each step has required fields
    for s in steps:
        for key in ("step", "name", "agent_id", "kind"):
            assert key in s, f"workflow_step missing {key}: {s}"
    # Step numbers are 1-8 unique
    nums = [s["step"] for s in steps]
    assert sorted(nums) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_workflow_edges_reference_existing_nodes(inventory):
    """Every workflow_edge from/to id must reference a real node."""
    by_id = {n["id"] for n in inventory["nodes"]}
    edges = inventory.get("workflow_edges") or []
    assert len(edges) >= 10
    for e in edges:
        assert e["from"] in by_id, f"workflow_edge from-id {e['from']!r} not in nodes"
        assert e["to"] in by_id, f"workflow_edge to-id {e['to']!r} not in nodes"


def test_workflow_has_loop_back(inventory):
    """At least one edge marks the daily cycle-back loop."""
    edges = inventory.get("workflow_edges") or []
    loops = [e for e in edges if e.get("loop") is True]
    assert len(loops) >= 1, "expected at least one workflow_edge with loop=true"


def test_workflow_step_numbers_in_range(inventory):
    """workflow_edge.step values are floats between 1 and 10 (room for sub-steps + loop)."""
    edges = inventory.get("workflow_edges") or []
    for e in edges:
        if "step" in e:
            v = float(e["step"])
            assert 1 <= v <= 10, f"workflow_edge.step out of range: {v} ({e})"


# ----------------------
# phase-22.1 -- per-node Gemini-lock granularity (v3 schema)
# ----------------------

def test_inventory_version_3(inventory):
    """v3 schema introduced in phase-22.1 with gemini_locked + grounding_dependent fields."""
    assert inventory["version"] == 3


def test_locked_count_is_one(inventory):
    """Exactly 1 node is hard-locked to Gemini (RAGAgent -- Vertex AI Search dep)."""
    locked = [n for n in inventory["nodes"] if n.get("gemini_locked")]
    assert len(locked) == 1
    assert locked[0]["id"] == "rag_agent"


def test_grounding_dependent_count_is_four(inventory):
    """4 nodes lose live web-search citations on Claude but still produce text."""
    deps = [n for n in inventory["nodes"] if n.get("grounding_dependent")]
    ids = {n["id"] for n in deps}
    assert ids == {"market_agent", "competitor_agent", "deep_dive_agent", "enhanced_macro_agent"}


def test_locked_node_has_lock_reason(inventory):
    """The 1 locked node should have a non-empty lock_reason."""
    locked = [n for n in inventory["nodes"] if n.get("gemini_locked")]
    for n in locked:
        assert n.get("lock_reason"), f"locked node {n['id']} missing lock_reason"
        assert len(n["lock_reason"]) > 10


def test_lock_flags_default_false(inventory):
    """Most nodes (52 - 1 locked - 4 grounding = 47) have neither flag set."""
    flagged = [n for n in inventory["nodes"]
               if n.get("gemini_locked") or n.get("grounding_dependent")]
    assert len(flagged) == 5  # 1 locked + 4 grounding
    other = [n for n in inventory["nodes"] if n["id"] not in {n2["id"] for n2 in flagged}]
    for n in other:
        assert not n.get("gemini_locked")
        assert not n.get("grounding_dependent")
