"""phase-3.5 step 3.5.0: MCP surface inventory (read-only).

Walks .mcp.json (registered external servers) + backend/mcp/
(legacy in-repo FastMCP stubs) + backend/agents/mcp_servers/
(authoritative in-process servers) and emits JSON to stdout.

Secret safety: only env-var KEY names are printed, never values.

Usage:
    python scripts/audit/mcp_inventory.py --json > handoff/mcp_inventory.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MCP_JSON = REPO / ".mcp.json"
LEGACY_STUB_DIR = REPO / "backend" / "mcp"
AUTHORITATIVE_DIR = REPO / "backend" / "agents" / "mcp_servers"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN PRIVATE KEY-----"),
]
ENV_TEMPLATE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _read_mcp_json() -> list[dict]:
    if not MCP_JSON.exists():
        return []
    try:
        data = json.loads(MCP_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []
    servers = []
    for name, cfg in (data.get("mcpServers") or {}).items():
        env_vars = sorted(
            set(ENV_TEMPLATE.findall(json.dumps(cfg.get("env", {}))))
        )
        servers.append({
            "name": name,
            "source_type": "external_registered",
            "source_path": ".mcp.json",
            "transport": cfg.get("type", "stdio"),
            "command": cfg.get("command"),
            "args": cfg.get("args", []),
            "env_var_keys": env_vars,
            "stub": False,
        })
    return servers


def _scan_server_dir(dir_path: Path, source_type: str,
                     stub: bool) -> list[dict]:
    """Walk a Python directory, detect FastMCP @mcp.resource/tool decorators,
    and summarize each server module."""
    out: list[dict] = []
    if not dir_path.exists():
        return out
    for py in sorted(dir_path.glob("*.py")):
        if py.name.startswith("__"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        tools: list[str] = []
        resources: list[str] = []
        has_factory = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("create_") and node.name.endswith("_server"):
                    has_factory = True
                for dec in node.decorator_list:
                    src = ast.unparse(dec)
                    if "@mcp.tool" in ("@" + src) or src.endswith(".tool"):
                        tools.append(node.name)
                    elif src.startswith("mcp.resource") or ".resource(" in src:
                        resources.append(node.name)
        out.append({
            "name": py.stem,
            "source_type": source_type,
            "source_path": str(py.relative_to(REPO)),
            "has_factory": has_factory,
            "tool_count": len(tools),
            "resource_count": len(resources),
            "tools_sample": tools[:5],
            "resources_sample": resources[:5],
            "env_var_keys": [],
            "stub": stub,
        })
    return out


def build_inventory() -> dict:
    servers: list[dict] = []
    servers.extend(_read_mcp_json())
    servers.extend(_scan_server_dir(AUTHORITATIVE_DIR,
                                     "in_process_authoritative", stub=False))
    servers.extend(_scan_server_dir(LEGACY_STUB_DIR,
                                     "in_process_legacy_stub", stub=True))
    return {
        "schema_version": 1,
        "counts": {
            "external_registered": sum(1 for s in servers if s["source_type"] == "external_registered"),
            "in_process_authoritative": sum(1 for s in servers if s["source_type"] == "in_process_authoritative"),
            "in_process_legacy_stub": sum(1 for s in servers if s["source_type"] == "in_process_legacy_stub"),
        },
        "servers": servers,
    }


def _assert_no_secrets(blob: str) -> list[str]:
    return [p.pattern for p in SECRET_PATTERNS if p.search(blob)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="emit JSON to stdout")
    ap.add_argument("--output", default=None,
                    help="optional output file path")
    args = ap.parse_args()

    inventory = build_inventory()
    blob = json.dumps(inventory, indent=2)

    leaks = _assert_no_secrets(blob)
    if leaks:
        print(json.dumps({"error": "secret_leak_detected",
                          "patterns": leaks}), file=sys.stderr)
        return 2

    if args.json or args.output is None:
        print(blob)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(blob + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
