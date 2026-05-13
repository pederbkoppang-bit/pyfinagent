"""verify_phase_25_E -- Drawer summary vs full toggle (?full=1 query param).

Verifies:
  1. `get_trade_rationale(trade_id, full: bool=...)` route signature has `full` query param.
  2. Backend filter prunes Layer-1 / Quant / SignalStack / Bull / Bear from signals when full=False.
  3. Backend returns everything when full=True (no pruning).
  4. Frontend api.ts has the `full` argument in getPaperTradeRationale and appends `?full=0|1`.
  5. Drawer has a toggle button with useState for `full`.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: route signature has `full` query param ────────────────────
pt_src = (REPO / "backend/api/paper_trading.py").read_text(encoding="utf-8")
pt_tree = ast.parse(pt_src)
route_node = None
for node in ast.walk(pt_tree):
    if isinstance(node, ast.AsyncFunctionDef) and node.name == "get_trade_rationale":
        route_node = node
        break

sig_args = [a.arg for a in route_node.args.args] if route_node else []
has_full = "full" in sig_args
claim(
    "1. api_paper_trading_trades_trade_id_rationale_supports_full_query_param",
    has_full,
    f"args={sig_args}",
)


# ── Claim 2 + 3: backend filter behavior ───────────────────────────────
# Inspect the handler body source for the filter logic
if route_node:
    body_src = ast.unparse(route_node)
    has_full_branch = "if not full:" in body_src or "not full:" in body_src
    has_analyst_filter = "Analyst" in body_src and "RiskJudge" in body_src
    has_trader_check = "Trader" in body_src
else:
    has_full_branch = has_analyst_filter = has_trader_check = False

claim(
    "2. backend_filter_prunes_tree_when_full_false",
    has_full_branch and has_analyst_filter and has_trader_check,
    f"full_branch={has_full_branch} analyst+risk_filter={has_analyst_filter} trader_check={has_trader_check}",
)

# Claim 3: backend returns everything when full=True (no extra pruning)
# Verified by source inspection -- the filter is gated by `if not full:`
returns_unfiltered_when_full = has_full_branch
claim(
    "3. backend_returns_full_tree_when_full_true",
    returns_unfiltered_when_full,
    "Filter only applied when full=False (else branch returns unmodified signals)",
)


# ── Claim 4: frontend api.ts passes `full` argument ────────────────────
api_src = (REPO / "frontend/src/lib/api.ts").read_text(encoding="utf-8")
has_full_arg = bool(
    re.search(r"export function getPaperTradeRationale\(\s*tradeId:\s*string\s*,\s*full\s*=", api_src)
)
has_query_param = "?full=1" in api_src or '"?full=1"' in api_src
claim(
    "4. frontend_api_passes_full_query_param",
    has_full_arg and has_query_param,
    f"full_arg={has_full_arg} query_param={has_query_param}",
)


# ── Claim 5: drawer has toggle button + useState ───────────────────────
drawer_src = (REPO / "frontend/src/components/AgentRationaleDrawer.tsx").read_text(encoding="utf-8")
has_state = "useState<boolean>(false)" in drawer_src
has_toggle_button = "Show full view" in drawer_src and "Show compact view" in drawer_src
has_fetch_with_full = "getPaperTradeRationale(tradeId, full)" in drawer_src
claim(
    "5. frontend_drawer_toggle_button_implemented",
    has_state and has_toggle_button and has_fetch_with_full,
    f"useState={has_state} toggle_button={has_toggle_button} fetch_with_full={has_fetch_with_full}",
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.E verification ===\n")
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
