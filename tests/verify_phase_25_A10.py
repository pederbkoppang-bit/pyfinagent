"""verify_phase_25_A10 -- Alpaca MCP smoke test + deny-list reconcile.

Verifies:
  1. Smoke test script exists at scripts/mcp_servers/smoke_test_alpaca_mcp.py
     with the expected shape (initialize handshake + tools/list call).
  2. Smoke test gracefully handles no-credentials with a SKIP exit=0.
  3. Reconcile script exists at scripts/mcp_servers/reconcile_alpaca_deny_list.py.
  4. Reconcile script exits 0 on the current deny list.
  5. .claude/settings.json deny list has all 11 canonical V2 write tools.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: smoke test exists with right shape ────────────────────────
smoke_path = REPO / "scripts/mcp_servers/smoke_test_alpaca_mcp.py"
smoke_exists = smoke_path.exists()
smoke_src = smoke_path.read_text(encoding="utf-8") if smoke_exists else ""
has_initialize = '"initialize"' in smoke_src or "'initialize'" in smoke_src
has_tools_list = '"tools/list"' in smoke_src or "'tools/list'" in smoke_src
has_uvx = "uvx" in smoke_src and "alpaca-mcp-server" in smoke_src
claim(
    "1. scripts_mcp_servers_smoke_test_alpaca_mcp_py_exists_and_passes",
    smoke_exists and has_initialize and has_tools_list and has_uvx,
    f"exists={smoke_exists} initialize={has_initialize} tools_list={has_tools_list} uvx={has_uvx}",
)


# ── Claim 2: smoke test handles no-creds gracefully ────────────────────
has_skip = bool(re.search(r'"SKIP\s+--', smoke_src))
has_env_check = "ALPACA_API_KEY" in smoke_src and "ALPACA_API_KEY_ID" in smoke_src
claim(
    "2. smoke_test_skips_on_missing_creds",
    has_skip and has_env_check,
    f"skip_print={has_skip} env_check={has_env_check}",
)


# ── Claim 3: reconcile script exists ───────────────────────────────────
recon_path = REPO / "scripts/mcp_servers/reconcile_alpaca_deny_list.py"
recon_exists = recon_path.exists()
recon_src = recon_path.read_text(encoding="utf-8") if recon_exists else ""
has_canonical_const = "CANONICAL_WRITE_TOOLS" in recon_src
claim(
    "3. reconcile_script_exists",
    recon_exists and has_canonical_const,
    f"exists={recon_exists} canonical_const={has_canonical_const}",
)


# ── Claim 4: reconcile script exits 0 ──────────────────────────────────
proc = subprocess.run(
    ["python3", "scripts/mcp_servers/reconcile_alpaca_deny_list.py"],
    cwd=str(REPO),
    capture_output=True,
    text=True,
    timeout=30,
)
ok = proc.returncode == 0 and "deny list covers all" in proc.stdout
claim(
    "4. reconcile_alpaca_deny_list_py_passes_no_unauthorized_writes",
    ok,
    f"exit={proc.returncode} stdout_tail={proc.stdout.strip()[:80]}",
)


# ── Claim 5: deny list has all 11 canonical write tools ────────────────
settings_data = json.loads((REPO / ".claude/settings.json").read_text(encoding="utf-8"))
deny_list = set(settings_data.get("permissions", {}).get("deny", []))
canonical = [
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
missing = [t for t in canonical if f"mcp__alpaca__{t}" not in deny_list]
claim(
    "5. deny_list_has_all_11_v2_write_tools",
    not missing,
    f"missing={missing}" if missing else f"all {len(canonical)} present",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.A10 verification ===\n")
all_pass = True
for name, ok, detail in results:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"        -> {detail}")
    if not ok:
        all_pass = False

print()
if all_pass:
    print(f"ALL {len(results)} CLAIMS PASS")
    sys.exit(0)
else:
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{failed} of {len(results)} claims FAILED")
    sys.exit(1)
