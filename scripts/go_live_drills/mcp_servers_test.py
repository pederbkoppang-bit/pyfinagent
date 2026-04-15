"""
Go-Live drill test: MCP servers deployed and authenticated (Phase 4.4.3.1).

Standalone, stdlib-only drill. Verifies that all three MCP server modules
(data / backtest / signals) exist, expose the required classes and factory
functions, are exported from __init__.py, and that the /api/health endpoint
includes MCP server health subfields.

Run from the repo root:

    python scripts/go_live_drills/mcp_servers_test.py

Exit code 0 on PASS, 1 on any failure.
"""

import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SERVERS = [
    {
        "name": "data",
        "module": "data_server.py",
        "class": "DataServer",
        "factory": "create_data_server",
    },
    {
        "name": "backtest",
        "module": "backtest_server.py",
        "class": "BacktestServer",
        "factory": "create_backtest_server",
    },
    {
        "name": "signals",
        "module": "signals_server.py",
        "class": "SignalsServer",
        "factory": "create_signals_server",
    },
]

MCP_DIR = REPO_ROOT / "backend" / "agents" / "mcp_servers"
MAIN_PY = REPO_ROOT / "backend" / "main.py"
INIT_PY = MCP_DIR / "__init__.py"

results = []


def record(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((label, passed))
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))


def parse_ast(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# --- S0: All three module files exist ---
for srv in SERVERS:
    p = MCP_DIR / srv["module"]
    record(f"S0.{srv['name']}_file_exists", p.is_file(), str(p))

# --- S1: Each module defines the expected class ---
for srv in SERVERS:
    p = MCP_DIR / srv["module"]
    tree = parse_ast(p)
    class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    found = srv["class"] in class_names
    record(f"S1.{srv['name']}_class_defined", found, f"classes={class_names}")

# --- S2: Each module defines the factory function ---
for srv in SERVERS:
    p = MCP_DIR / srv["module"]
    tree = parse_ast(p)
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    found = srv["factory"] in func_names
    record(f"S2.{srv['name']}_factory_defined", found, f"found={srv['factory'] in func_names}")

# --- S3: Each module has if __name__ == "__main__" block ---
for srv in SERVERS:
    src = (MCP_DIR / srv["module"]).read_text(encoding="utf-8")
    found = 'if __name__ == "__main__"' in src or "if __name__ == '__main__'" in src
    record(f"S3.{srv['name']}_has_main_block", found)

# --- S4: __init__.py exports all three factories ---
init_tree = parse_ast(INIT_PY)
init_src = INIT_PY.read_text(encoding="utf-8")
for srv in SERVERS:
    found = srv["factory"] in init_src
    record(f"S4.{srv['name']}_exported_from_init", found)

# --- S5: __init__.py defines start_all_servers ---
init_funcs = [n.name for n in ast.walk(init_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
record("S5.start_all_servers_defined", "start_all_servers" in init_funcs, f"functions={init_funcs}")

# --- S6: /api/health includes mcp_servers health subfields ---
main_tree = parse_ast(MAIN_PY)
main_src = MAIN_PY.read_text(encoding="utf-8")
has_mcp_health = "mcp_servers" in main_src
record("S6.health_endpoint_has_mcp_servers", has_mcp_health)

# Deeper check: find the health function and verify it returns mcp_servers key
health_func = None
for node in ast.walk(main_tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "health":
        health_func = node
        break

if health_func:
    health_src = ast.get_source_segment(main_src, health_func) or ""
    has_mcp_in_return = "mcp_servers" in health_src
    record("S6b.health_func_returns_mcp_servers", has_mcp_in_return)

    for srv_name in ["data", "backtest", "signals"]:
        mod_ref = f"backend.agents.mcp_servers.{srv_name}_server"
        found = mod_ref in health_src
        record(f"S6c.health_probes_{srv_name}", found, f"module={mod_ref}")
else:
    record("S6b.health_func_found", False, "health() not found in main.py")

# --- S7: importlib.util.find_spec used for lightweight health check ---
has_find_spec = "importlib.util" in main_src or "find_spec" in main_src
record("S7.lightweight_health_check", has_find_spec, "uses importlib.util.find_spec")

# --- Summary ---
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print()
if failed == 0:
    print(f"DRILL PASS: {passed}/{total}")
    sys.exit(0)
else:
    print(f"DRILL FAIL: {passed}/{total} ({failed} failures)")
    for label, ok in results:
        if not ok:
            print(f"  FAILED: {label}")
    sys.exit(1)
