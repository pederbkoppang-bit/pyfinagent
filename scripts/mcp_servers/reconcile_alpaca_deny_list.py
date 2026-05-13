"""phase-25.A10: reconcile .claude/settings.json deny list vs canonical
Alpaca MCP V2 write-class tool surface.

Reads `.claude/settings.json::permissions.deny[]`. For each canonical
write-class Alpaca tool (place_*_order, cancel_*_order, replace_order_by_id,
close_position, close_all_positions, exercise_*_position, update_account_config),
asserts the `mcp__alpaca__<tool>` entry is present in `deny[]`.

Exit 0 if reconciled. Exit 1 with a diff if any are missing.

Usage:
    python scripts/mcp_servers/reconcile_alpaca_deny_list.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / ".claude/settings.json"

# Canonical Alpaca MCP V2.0.1 write-class trading tools.
# Sourced from handoff/current/alpaca-mcp-research-brief.md (V2 inventory,
# Trading/Orders + Positions categories) + the audit consensus on which
# tools mutate trading state.
CANONICAL_WRITE_TOOLS = [
    "place_stock_order",
    "place_crypto_order",
    "place_option_order",
    "cancel_order_by_id",
    "cancel_all_orders",
    "replace_order_by_id",
    "close_position",
    "close_all_positions",
    "exercise_options_position",
    "do_not_exercise_options_position",
    "update_account_config",
]


def main() -> int:
    if not SETTINGS.exists():
        print(f"FAIL .claude/settings.json not found at {SETTINGS}", file=sys.stderr)
        return 1

    data = json.loads(SETTINGS.read_text(encoding="utf-8"))
    deny_list = data.get("permissions", {}).get("deny", [])
    if not isinstance(deny_list, list):
        print(
            f"FAIL permissions.deny is not a list (got {type(deny_list).__name__})",
            file=sys.stderr,
        )
        return 1

    deny_set = set(deny_list)
    missing = []
    for tool in CANONICAL_WRITE_TOOLS:
        entry = f"mcp__alpaca__{tool}"
        if entry not in deny_set:
            missing.append(entry)

    if missing:
        print("FAIL deny list missing canonical Alpaca V2 write tools:", file=sys.stderr)
        for m in missing:
            print(f"  -- {m}", file=sys.stderr)
        print(
            "\nFix: add the missing entries to permissions.deny[] in .claude/settings.json",
            file=sys.stderr,
        )
        return 1

    print(f"OK deny list covers all {len(CANONICAL_WRITE_TOOLS)} canonical write tools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
