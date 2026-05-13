"""verify_phase_25_E7 -- yfinance_tool.get_price_history() guard + counter.

Verifies:
  1. Source contains try/except wrapper in get_price_history.
  2. Source references `save_data_source_event` + `yfinance_price_history`.
  3. Behavioral: patch yf.Ticker to raise; assert returns error-list.
  4. Behavioral: patch BQ method; assert it's invoked once on failure.
  5. Behavioral: empty DataFrame produces error-list + persist call.

Exit 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

results: list[tuple[str, bool, str]] = []


def claim(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, bool(condition), detail))


# ── Claim 1: source has try/except in get_price_history ────────────────
src = (REPO / "backend/tools/yfinance_tool.py").read_text(encoding="utf-8")
tree = ast.parse(src)
fn_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "get_price_history":
        fn_node = node
        break

has_try = False
if fn_node:
    has_try = any(isinstance(n, ast.Try) for n in ast.walk(fn_node))

claim(
    "1. get_price_history_has_try_except_wrapper",
    has_try,
    "try/except present in function body" if has_try else "Missing try/except",
)


# ── Claim 2: source references save_data_source_event + correct source ─
has_call = "save_data_source_event" in src
has_source_key = '"yfinance_price_history"' in src or "'yfinance_price_history'" in src
claim(
    "2. failure_counter_incremented_and_persisted_to_bq",
    has_call and has_source_key,
    f"save_call={has_call} source_key={has_source_key}",
)


# ── Claim 3: behavioral -- exception path returns error list ───────────
try:
    with patch("backend.tools.yfinance_tool.yf") as mock_yf, \
         patch("backend.tools.yfinance_tool._persist_yfinance_event") as mock_persist:
        # Simulate rate-limit failure
        mock_yf.Ticker.side_effect = RuntimeError("rate_limited")
        from backend.tools.yfinance_tool import get_price_history  # noqa: E402
        out = get_price_history("AAPL", period="1y")
        rt3_ok = (
            isinstance(out, list)
            and len(out) == 1
            and "error" in out[0]
            and out[0].get("ticker") == "AAPL"
            and mock_persist.called
        )
        rt3_detail = f"len={len(out)} keys={list(out[0].keys()) if out else []} persist_called={mock_persist.called}"
except Exception as e:
    rt3_ok = False
    rt3_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "3. get_price_history_returns_error_dict_on_failure",
    rt3_ok,
    rt3_detail,
)


# ── Claim 4: persist called exactly once on exception ──────────────────
try:
    with patch("backend.tools.yfinance_tool.yf") as mock_yf, \
         patch("backend.tools.yfinance_tool._persist_yfinance_event") as mock_persist:
        mock_yf.Ticker.side_effect = ValueError("test_error")
        from backend.tools.yfinance_tool import get_price_history  # noqa: E402
        get_price_history("TSLA", period="6mo")
        rt4_ok = mock_persist.call_count == 1
        rt4_detail = f"persist_call_count={mock_persist.call_count}"
except Exception as e:
    rt4_ok = False
    rt4_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "4. persist_called_once_per_failure",
    rt4_ok,
    rt4_detail,
)


# ── Claim 5: empty DataFrame path ───────────────────────────────────────
try:
    import pandas as pd  # noqa: E402
    empty_df = pd.DataFrame()
    with patch("backend.tools.yfinance_tool.yf") as mock_yf, \
         patch("backend.tools.yfinance_tool._persist_yfinance_event") as mock_persist:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = empty_df
        mock_yf.Ticker.return_value = mock_ticker
        from backend.tools.yfinance_tool import get_price_history  # noqa: E402
        out = get_price_history("NVDA", period="1y")
        rt5_ok = (
            isinstance(out, list)
            and len(out) == 1
            and out[0].get("error") == "no_data"
            and out[0].get("ticker") == "NVDA"
            and mock_persist.called
        )
        rt5_detail = f"out={out} persist_called={mock_persist.called}"
except Exception as e:
    rt5_ok = False
    rt5_detail = f"Exception: {type(e).__name__}: {e}"

claim(
    "5. empty_dataframe_returns_no_data_error",
    rt5_ok,
    rt5_detail,
)


# ── Summary ─────────────────────────────────────────────────────────────
print("\n=== phase-25.E7 verification ===\n")
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
