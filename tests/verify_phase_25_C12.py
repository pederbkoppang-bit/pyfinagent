"""phase-25.C12 verifier -- cross-tab Sharpe KPI reconciliation.

Closes phase-24.12 F-4 (home page local kpiSharpe diverges from
/paper-trading API perf.sharpe_ratio by ~0.16 Sharpe units at 4% RFR
because the local formula skips the risk-free-rate subtraction).

Run: source .venv/bin/activate && python3 tests/verify_phase_25_C12.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
PAPER_TRADING_API = REPO / "backend" / "api" / "paper_trading.py"
TYPES_TS = REPO / "frontend" / "src" / "lib" / "types.ts"
API_TS = REPO / "frontend" / "src" / "lib" / "api.ts"
PAGE_TSX = REPO / "frontend" / "src" / "app" / "page.tsx"
KPI_METRICS = REPO / "frontend" / "src" / "lib" / "kpiMetrics.ts"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PAPER_TRADING_API, TYPES_TS, API_TS, PAGE_TSX, KPI_METRICS):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    pt_src = PAPER_TRADING_API.read_text(encoding="utf-8")
    types_src = TYPES_TS.read_text(encoding="utf-8")
    api_src = API_TS.read_text(encoding="utf-8")
    page_src = PAGE_TSX.read_text(encoding="utf-8")
    kpi_src = KPI_METRICS.read_text(encoding="utf-8")

    # ---- Claim 1: get_portfolio body contains sharpe_ratio insertion + canonical helper call.
    # Extract get_portfolio function body.
    fn_match = re.search(
        r"async def get_portfolio\(.*?\)(.*?)(?=\n@router\.|\nasync def |\ndef |\Z)",
        pt_src,
        re.DOTALL,
    )
    has_assignment = False
    has_helper_call = False
    if fn_match:
        body = fn_match.group(1)
        has_assignment = 'portfolio["sharpe_ratio"]' in body
        has_helper_call = "compute_sharpe_from_snapshots" in body
    results.append((
        "PASS" if has_assignment and has_helper_call else "FAIL",
        "paper_trading_portfolio_endpoint_returns_sharpe_ratio_field",
        "get_portfolio must assign portfolio['sharpe_ratio'] via compute_sharpe_from_snapshots",
    ))

    # ---- Claim 2: PaperPortfolio interface includes sharpe_ratio?: number | null.
    iface_match = re.search(
        r"export interface PaperPortfolio\s*\{[^}]*?sharpe_ratio\?:\s*number\s*\|\s*null",
        types_src,
        re.DOTALL,
    )
    results.append((
        "PASS" if iface_match else "FAIL",
        "paper_portfolio_interface_extended_with_sharpe_ratio",
        "PaperPortfolio interface must declare sharpe_ratio?: number | null",
    ))

    # ---- Claim 3: api.ts getPaperPortfolio returns sharpe_ratio implicitly via PaperPortfolio.
    # The api.ts wrapper returns { portfolio: PaperPortfolio; positions: PaperPosition[] }, so
    # sharpe_ratio is reachable as portfolio.sharpe_ratio via the interface extension.
    # We grep for the unchanged signature and confirm PaperPortfolio is the reference type.
    api_sig = re.search(
        r"export\s+function\s+getPaperPortfolio\(\)\s*:\s*Promise<\{[^}]*?portfolio:\s*PaperPortfolio",
        api_src,
        re.DOTALL,
    )
    results.append((
        "PASS" if api_sig else "FAIL",
        "api_ts_getpaperportfolio_returns_paper_portfolio_with_sharpe",
        "getPaperPortfolio return type must include portfolio: PaperPortfolio (which now carries sharpe_ratio)",
    ))

    # ---- Claim 4: page.tsx declares apiSharpe state.
    state_decl = re.search(
        r"const\s+\[\s*apiSharpe\s*,\s*setApiSharpe\s*\]\s*=\s*useState<\s*number\s*\|\s*null\s*>\(\s*null\s*\)",
        page_src,
    )
    results.append((
        "PASS" if state_decl else "FAIL",
        "home_page_declares_apisharpe_state",
        "page.tsx must declare const [apiSharpe, setApiSharpe] = useState<number | null>(null)",
    ))

    # ---- Claim 5: page.tsx swap -- apiSharpe ?? kpiSharpe(navSeries).
    swap_match = re.search(
        r"const\s+sharpe90\s*=\s*apiSharpe\s*\?\?\s*kpiSharpe\(navSeries\)",
        page_src,
    )
    # Also confirm setApiSharpe call inside the portfolio fetch handler.
    capture_match = re.search(
        r"setApiSharpe\(\s*portfolio\.value\.portfolio\?\.sharpe_ratio\s*\?\?\s*null\s*\)",
        page_src,
    )
    results.append((
        "PASS" if swap_match and capture_match else "FAIL",
        "home_page_uses_api_sharpe_ratio_not_local_kpisharpe",
        "page.tsx must (a) capture sharpe_ratio from portfolio fetch and (b) use apiSharpe ?? kpiSharpe(navSeries)",
    ))

    # ---- Claim 6: kpiMetrics.ts::sharpe is preceded by @deprecated JSDoc.
    deprecated_block = re.search(
        r"/\*\*[\s\S]*?@deprecated[\s\S]*?phase-25\.C12[\s\S]*?\*/\s*export\s+function\s+sharpe\b",
        kpi_src,
    )
    results.append((
        "PASS" if deprecated_block else "FAIL",
        "deprecation_marker_on_kpisharpe_function",
        "kpiMetrics.ts::sharpe must be preceded by JSDoc with @deprecated + phase-25.C12 tag",
    ))

    # ---- Behavioral fixtures.
    sys.path.insert(0, str(REPO))

    # ---- Claim 7: BEHAVIORAL get_portfolio happy path -- sharpe_ratio is computed.
    happy_ok = False
    happy_err = ""
    try:
        sys.modules.pop("backend.api.paper_trading", None)
        from backend.api import paper_trading as pt_mod  # type: ignore

        import asyncio
        from backend.services.api_cache import get_api_cache as _g
        cache = _g()
        if hasattr(cache, "_store"):
            cache._store.clear()

        # Build synthetic snapshots with variance so the Sharpe formula yields finite.
        import random
        rng = random.Random(42)
        snapshots = []
        nav = 10000.0
        for _ in range(60):
            nav = nav * (1.0 + 0.003 + rng.gauss(0.0, 0.01))
            snapshots.append({"total_nav": round(nav, 2)})

        fake_bq = MagicMock()
        fake_portfolio = {
            "portfolio_id": "default",
            "starting_capital": 10000.0,
            "current_cash": 1000.0,
            "total_nav": 12000.0,
            "total_pnl_pct": 20.0,
            "benchmark_return_pct": 5.0,
            "inception_date": "2026-01-01",
            "updated_at": "2026-05-13T00:00:00+00:00",
        }
        fake_bq.get_paper_portfolio.return_value = fake_portfolio
        fake_bq.get_paper_snapshots.return_value = snapshots

        fake_trader = MagicMock()
        fake_trader.get_positions.return_value = []

        with patch(
            "backend.api.paper_trading.BigQueryClient",
            return_value=fake_bq,
        ), patch(
            "backend.api.paper_trading.PaperTrader",
            return_value=fake_trader,
        ), patch(
            "backend.api.paper_trading._fetch_ticker_meta",
            return_value={"meta": {}},
        ):
            result = asyncio.run(pt_mod.get_portfolio())

        portfolio_returned = result.get("portfolio") or {}
        sharpe_value = portfolio_returned.get("sharpe_ratio")
        if sharpe_value is None:
            happy_err = "portfolio.sharpe_ratio is None despite 60 snapshots with variance"
        elif not isinstance(sharpe_value, (int, float)):
            happy_err = f"sharpe_ratio type wrong: {type(sharpe_value).__name__}"
        else:
            happy_ok = True
    except Exception as e:
        happy_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if happy_ok else "FAIL",
        "behavioral_get_portfolio_returns_sharpe_ratio_for_valid_snapshots",
        f"happy-path get_portfolio with 60 noisy snapshots must yield numeric sharpe_ratio ({happy_err})",
    ))

    # ---- Claim 8: BEHAVIORAL no-data path -- empty snapshots -> sharpe_ratio is 0.0 or None.
    nodata_ok = False
    nodata_err = ""
    try:
        cache_obj = _g()
        if hasattr(cache_obj, "_store"):
            cache_obj._store.clear()

        fake_bq2 = MagicMock()
        fake_bq2.get_paper_portfolio.return_value = dict(fake_portfolio)
        fake_bq2.get_paper_snapshots.return_value = []
        fake_trader2 = MagicMock()
        fake_trader2.get_positions.return_value = []

        with patch(
            "backend.api.paper_trading.BigQueryClient",
            return_value=fake_bq2,
        ), patch(
            "backend.api.paper_trading.PaperTrader",
            return_value=fake_trader2,
        ), patch(
            "backend.api.paper_trading._fetch_ticker_meta",
            return_value={"meta": {}},
        ):
            r2 = asyncio.run(pt_mod.get_portfolio())

        pv = r2.get("portfolio") or {}
        sv = pv.get("sharpe_ratio")
        if sv not in (None, 0.0):
            nodata_err = f"empty snapshots yielded {sv!r}; expected None or 0.0"
        elif "sharpe_ratio" not in pv:
            nodata_err = "sharpe_ratio key not present at all"
        else:
            nodata_ok = True
    except Exception as e:
        nodata_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if nodata_ok else "FAIL",
        "behavioral_get_portfolio_empty_snapshots_sharpe_none_or_zero",
        f"empty snapshots must yield sharpe_ratio=None or 0.0 (graceful) ({nodata_err})",
    ))

    # ---- Claim 9: BEHAVIORAL fail-open path -- get_paper_snapshots raises.
    fo_ok = False
    fo_err = ""
    try:
        cache_obj = _g()
        if hasattr(cache_obj, "_store"):
            cache_obj._store.clear()

        fake_bq3 = MagicMock()
        fake_bq3.get_paper_portfolio.return_value = dict(fake_portfolio)
        fake_bq3.get_paper_snapshots.side_effect = RuntimeError("BQ down")
        fake_trader3 = MagicMock()
        fake_trader3.get_positions.return_value = []

        with patch(
            "backend.api.paper_trading.BigQueryClient",
            return_value=fake_bq3,
        ), patch(
            "backend.api.paper_trading.PaperTrader",
            return_value=fake_trader3,
        ), patch(
            "backend.api.paper_trading._fetch_ticker_meta",
            return_value={"meta": {}},
        ):
            r3 = asyncio.run(pt_mod.get_portfolio())

        if not isinstance(r3, dict):
            fo_err = f"result not dict: {type(r3)}"
        else:
            pv3 = r3.get("portfolio") or {}
            sv3 = pv3.get("sharpe_ratio")
            if sv3 is not None:
                fo_err = f"sharpe_ratio={sv3!r}, expected None on fail-open"
            elif r3.get("positions") is None:
                fo_err = "positions missing from result"
            else:
                fo_ok = True
    except Exception as e:
        fo_err = f"caller crashed: {type(e).__name__}: {e}"

    results.append((
        "PASS" if fo_ok else "FAIL",
        "behavioral_get_portfolio_snapshot_failure_fails_open",
        f"snapshot fetch failure must yield sharpe_ratio=None + rest of response intact ({fo_err})",
    ))

    # ---- Claim 10: grep-level no regression -- response still has portfolio/positions/sector_breakdown.
    has_all_keys = (
        '"portfolio": portfolio' in pt_src
        and '"positions": positions' in pt_src
        and '"sector_breakdown": sector_breakdown' in pt_src
    )
    results.append((
        "PASS" if has_all_keys else "FAIL",
        "no_regression_response_keys_preserved",
        "result dict must still contain portfolio + positions + sector_breakdown keys",
    ))

    # ---- Claim 11: phase-25.C12 attribution comment in get_portfolio.
    attribution = "phase-25.C12" in pt_src
    results.append((
        "PASS" if attribution else "FAIL",
        "phase_25_c12_attribution_in_source",
        "paper_trading.py must include phase-25.C12 attribution comment in get_portfolio",
    ))

    # ---- Print results.
    n_pass = sum(1 for r in results if r[0] == "PASS")
    n_fail = len(results) - n_pass
    for verdict, claim, detail in results:
        print(f"{verdict}: {claim}")
        if verdict == "FAIL":
            print(f"      {detail}")

    print(f"\n{n_pass}/{len(results)} claims PASS, {n_fail} FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
