"""phase-23.2.21: immutable verification.

Asserts:
1. .mcp.json contains the bigquery server entry pinning
   mcp-server-bigquery==0.3.2 with --project + --location args.
2. .claude/settings.json deny list contains mcp__bigquery__execute-query
   (hyphenated, matches LucasHild) and NO LONGER contains the obsolete
   mcp__bigquery__execute_sql (underscored, matches nothing real).
3. CLAUDE.md "BigQuery Access (MCP)" section references the actual tool
   names exposed by LucasHild and drops the "harness-injected" myth.
4. scripts/mcp_servers/smoke_test_bigquery_mcp.py exists and exits 0
   end-to-end against the user's ADC.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def check_mcp_json():
    rel = ".mcp.json"
    data = json.loads(_read(rel))
    servers = data.get("mcpServers", {})
    assert "bigquery" in servers, "bigquery server entry missing"
    bq = servers["bigquery"]
    assert bq.get("type") == "stdio", "bigquery server must be stdio"
    assert bq.get("command") == "uvx", "bigquery server must launch via uvx"
    args = bq.get("args", [])
    assert "mcp-server-bigquery==0.3.2" in args, \
        "bigquery server must pin v0.3.2"
    assert "--project" in args and "sunny-might-477607-p8" in args, \
        "bigquery server must pass --project sunny-might-477607-p8"
    assert "--location" in args and "US" in args, \
        "bigquery server must pass --location US"
    # alpaca must remain intact
    assert "alpaca" in servers, "alpaca server must remain pinned"
    return f"OK {rel}"


def check_settings_deny():
    rel = ".claude/settings.json"
    data = json.loads(_read(rel))
    deny = data.get("permissions", {}).get("deny", [])
    assert "mcp__bigquery__execute-query" in deny, \
        "deny rule mcp__bigquery__execute-query (hyphenated) missing — write SQL would not require approval"
    assert "mcp__bigquery__execute_sql" not in deny, \
        "obsolete deny rule mcp__bigquery__execute_sql (underscored) must be removed — it matches nothing real"
    return f"OK {rel}"


def check_claude_md():
    rel = "CLAUDE.md"
    text = _read(rel)
    # New tool names (LucasHild's actual surface)
    for name in ("list-tables", "describe-table", "execute-query"):
        assert name in text, f"CLAUDE.md must document the actual tool name `{name}`"
    # The harness-injected myth must be retired
    assert "harness-injected" not in text, \
        "CLAUDE.md still claims BQ MCP is harness-injected — must reflect the .mcp.json pin"
    # Old tool names should no longer be load-bearing in the BQ section
    assert "execute_sql_readonly" not in text, \
        "CLAUDE.md still references execute_sql_readonly — that tool does not exist on LucasHild"
    return f"OK {rel}"


def check_smoke_test_script():
    rel = "scripts/mcp_servers/smoke_test_bigquery_mcp.py"
    p = ROOT / rel
    assert p.exists(), f"smoke test script missing: {rel}"
    text = p.read_text()
    assert "mcp-server-bigquery==0.3.2" in text, \
        "smoke test must use the same pinned version"
    assert "list-tables" in text, "smoke test must exercise list-tables"
    return f"OK {rel}"


def check_smoke_test_runs():
    """Run the smoke test end-to-end. Slowest check; do it last."""
    rel = "scripts/mcp_servers/smoke_test_bigquery_mcp.py"
    proc = subprocess.run(
        [sys.executable, str(ROOT / rel)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        last_stderr = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise AssertionError(
            f"smoke test failed (exit {proc.returncode}). stderr tail:\n{last_stderr}"
        )
    return f"OK {rel} -- end-to-end"


def main() -> int:
    checks = [
        check_mcp_json,
        check_settings_deny,
        check_claude_md,
        check_smoke_test_script,
        check_smoke_test_runs,
    ]
    failed = 0
    for fn in checks:
        try:
            print(fn())
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            failed += 1
    if failed:
        print(f"\n{failed} verification(s) failed")
        return 1
    print("\nphase-23.2.21 verification: ALL PASS (5/5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
