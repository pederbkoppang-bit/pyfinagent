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
