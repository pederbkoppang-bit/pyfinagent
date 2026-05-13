"""phase-25.S verifier -- daily P&L attribution report per ticker.

Closes phase-24.13 F-6 (SHARP arxiv finding: attribution is load-bearing;
no per-ticker pnl_per_cost_usd today). First-mover per-ticker variant of
25.Q's aggregate profit_per_llm_dollar metric.

Run: source .venv/bin/activate && python3 tests/verify_phase_25_S.py
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO = Path(__file__).resolve().parents[1]
PT = REPO / "backend" / "api" / "paper_trading.py"
CACHE = REPO / "backend" / "services" / "api_cache.py"
AUTO = REPO / "backend" / "services" / "autonomous_loop.py"


def main() -> int:
    results: list[tuple[str, str, str]] = []

    for p in (PT, CACHE, AUTO):
        if not p.exists():
            print(f"FAIL: required source file missing: {p}")
            return 1

    pt_src = PT.read_text(encoding="utf-8")
    cache_src = CACHE.read_text(encoding="utf-8")
    auto_src = AUTO.read_text(encoding="utf-8")

    # ---- Claim 1: route /attribution registered with canonical query.
    route_match = re.search(
        r'@router\.get\(["\']/attribution["\']\)',
        pt_src,
    )
    sig_match = re.search(
        r"async def get_attribution\(window_days:\s*int\s*=\s*Query\(\s*7\s*,\s*ge=1\s*,\s*le=365\s*\)\s*\)",
        pt_src,
    )
    results.append((
        "PASS" if route_match and sig_match else "FAIL",
        "new_api_paper_trading_attribution_endpoint_returns_per_ticker_data",
        "@router.get('/attribution') + async def get_attribution(window_days: int = Query(7, ge=1, le=365)) required",
    ))

    # ---- Claim 2: _compute_attribution signature exists.
    helper_sig = re.search(
        r"def _compute_attribution\(\s*bq:\s*BigQueryClient\s*,\s*window_days:\s*int\s*\)\s*->\s*dict\s*:",
        pt_src,
    )
    results.append((
        "PASS" if helper_sig else "FAIL",
        "compute_attribution_helper_signature",
        "_compute_attribution(bq: BigQueryClient, window_days: int) -> dict must be declared",
    ))

    # ---- Claim 3: response shape includes per_ticker + totals.
    has_per_ticker = '"per_ticker"' in pt_src
    has_totals = '"totals"' in pt_src
    has_note = '"note"' in pt_src
    results.append((
        "PASS" if has_per_ticker and has_totals and has_note else "FAIL",
        "response_includes_per_ticker_totals_and_note",
        "response dict must include 'per_ticker' list, 'totals' dict, and 'note' string",
    ))

    # ---- Claim 4: ENDPOINT_TTLS has paper:attribution.
    ttl_match = re.search(r'"paper:attribution"\s*:\s*[0-9.]+', cache_src)
    results.append((
        "PASS" if ttl_match else "FAIL",
        "endpoint_ttl_paper_attribution_declared",
        "ENDPOINT_TTLS must include 'paper:attribution'",
    ))

    # ---- Claim 5: autonomous_loop sets attribution_computed in cycle summary.
    cycle_flag = '"attribution_computed": True' in auto_src
    results.append((
        "PASS" if cycle_flag else "FAIL",
        "per_ticker_attribution_computed_at_cycle_completion",
        "autonomous_loop.py must set summary['attribution_computed'] = True on cycle completion",
    ))

    # ---- Behavioral fixtures: load module + mock dependencies.
    sys.path.insert(0, str(REPO))
    sys.modules.pop("backend.api.paper_trading", None)
    from backend.api import paper_trading as pt_mod  # type: ignore

    def _build_fake_bq(trades: list):
        fake_bq = MagicMock()
        fake_bq.get_paper_trades_in_window.return_value = trades
        return fake_bq

    # Helper to wrap pair_round_trips mock.
    def _build_round_trips(rts: list):
        return rts

    # ---- Claim 6: BEHAVIORAL happy path -- 2 tickers + non-zero LLM cost.
    happy_ok = False
    happy_err = ""
    try:
        # 5 trades: 3 for AAPL (yielding 1 round-trip pnl=100), 2 for MSFT
        # (yielding 1 round-trip pnl=50). Total: 5 trades, 2 tickers.
        trades = [
            {"ticker": "AAPL", "action": "BUY", "price": 100, "quantity": 10},
            {"ticker": "AAPL", "action": "BUY", "price": 110, "quantity": 5},
            {"ticker": "AAPL", "action": "SELL", "price": 120, "quantity": 10},
            {"ticker": "MSFT", "action": "BUY", "price": 200, "quantity": 5},
            {"ticker": "MSFT", "action": "SELL", "price": 220, "quantity": 5},
        ]
        rts = [
            {"ticker": "AAPL", "realized_pnl_usd": 200.0},
            {"ticker": "MSFT", "realized_pnl_usd": 100.0},
        ]
        fake_bq = _build_fake_bq(trades)
        with patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=rts,
        ), patch(
            "backend.api.sovereign_api._fetch_llm_cost_by_provider",
            return_value={"anthropic": 1.0, "vertex": 0.3, "openai": 0.2},
        ):
            result = pt_mod._compute_attribution(fake_bq, window_days=7)

        if result.get("window_days") != 7:
            happy_err = f"window_days={result.get('window_days')}"
        elif not isinstance(result.get("per_ticker"), list):
            happy_err = "per_ticker is not a list"
        elif len(result["per_ticker"]) != 2:
            happy_err = f"per_ticker len={len(result['per_ticker'])}, expected 2"
        else:
            by_t = {row["ticker"]: row for row in result["per_ticker"]}
            aapl = by_t.get("AAPL")
            msft = by_t.get("MSFT")
            # Total cost = 1.5, AAPL n_analyses=3, MSFT n_analyses=2, total=5
            # AAPL cost share: 1.5 * 3/5 = 0.9
            # MSFT cost share: 1.5 * 2/5 = 0.6
            if abs(aapl["llm_cost_usd"] - 0.9) > 0.01:
                happy_err = f"AAPL llm_cost_usd={aapl['llm_cost_usd']}, expected ~0.9"
            elif abs(msft["llm_cost_usd"] - 0.6) > 0.01:
                happy_err = f"MSFT llm_cost_usd={msft['llm_cost_usd']}, expected ~0.6"
            elif aapl["realized_pnl_usd"] != 200.0:
                happy_err = f"AAPL pnl={aapl['realized_pnl_usd']}"
            elif aapl["pnl_per_cost_usd"] is None:
                happy_err = "AAPL pnl_per_cost_usd is None despite non-zero cost"
            elif abs(aapl["pnl_per_cost_usd"] - (200.0 / 0.9)) > 0.5:
                happy_err = f"AAPL ratio={aapl['pnl_per_cost_usd']}, expected ~{200.0/0.9:.2f}"
            else:
                happy_ok = True
    except Exception as e:
        happy_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if happy_ok else "FAIL",
        "behavioral_happy_path_proportional_cost_split",
        f"5 trades / 2 tickers / $1.50 cost must yield AAPL=$0.90 (3/5 share) MSFT=$0.60 ({happy_err})",
    ))

    # ---- Claim 7: BEHAVIORAL zero-cost -> pnl_per_cost_usd = None.
    zc_ok = False
    zc_err = ""
    try:
        with patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=[{"ticker": "AAPL", "realized_pnl_usd": 100.0}],
        ), patch(
            "backend.api.sovereign_api._fetch_llm_cost_by_provider",
            return_value={"anthropic": 0.0, "vertex": 0.0, "openai": 0.0},
        ):
            result2 = pt_mod._compute_attribution(
                _build_fake_bq([{"ticker": "AAPL", "action": "BUY", "price": 100, "quantity": 1}]),
                window_days=7,
            )
        aapl2 = result2["per_ticker"][0]
        if aapl2["pnl_per_cost_usd"] is not None:
            zc_err = f"pnl_per_cost_usd={aapl2['pnl_per_cost_usd']}, expected None"
        elif result2["totals"]["pnl_per_cost_usd"] is not None:
            zc_err = f"totals.pnl_per_cost_usd={result2['totals']['pnl_per_cost_usd']}, expected None"
        else:
            zc_ok = True
    except Exception as e:
        zc_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if zc_ok else "FAIL",
        "behavioral_zero_cost_yields_none_ratio",
        f"zero aggregate cost must yield pnl_per_cost_usd=None for per-ticker and totals ({zc_err})",
    ))

    # ---- Claim 8: BEHAVIORAL empty-trades -> per_ticker is [].
    et_ok = False
    et_err = ""
    try:
        with patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=[],
        ), patch(
            "backend.api.sovereign_api._fetch_llm_cost_by_provider",
            return_value={"anthropic": 0.0, "vertex": 0.0, "openai": 0.0},
        ):
            result3 = pt_mod._compute_attribution(_build_fake_bq([]), window_days=7)
        if result3["per_ticker"] != []:
            et_err = f"per_ticker={result3['per_ticker']!r}, expected []"
        elif result3["totals"]["realized_pnl_usd"] != 0.0:
            et_err = f"totals.realized_pnl_usd={result3['totals']['realized_pnl_usd']}, expected 0.0"
        else:
            et_ok = True
    except Exception as e:
        et_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if et_ok else "FAIL",
        "behavioral_empty_trades_path",
        f"empty trades must yield per_ticker=[] + totals.realized_pnl_usd=0.0 ({et_err})",
    ))

    # ---- Claim 9: BEHAVIORAL ratio computation -- pnl=200, cost=0.10 -> 2000.0.
    rc_ok = False
    rc_err = ""
    try:
        # Single ticker AAPL with 1 analysis; cost = 0.10 -> AAPL gets 0.10 (all).
        trades_rc = [{"ticker": "AAPL", "action": "BUY", "price": 100, "quantity": 1}]
        with patch(
            "backend.services.paper_round_trips.pair_round_trips",
            return_value=[{"ticker": "AAPL", "realized_pnl_usd": 200.0}],
        ), patch(
            "backend.api.sovereign_api._fetch_llm_cost_by_provider",
            return_value={"anthropic": 0.10, "vertex": 0.0, "openai": 0.0},
        ):
            result4 = pt_mod._compute_attribution(_build_fake_bq(trades_rc), window_days=7)
        aapl4 = result4["per_ticker"][0]
        if abs(aapl4["llm_cost_usd"] - 0.10) > 0.001:
            rc_err = f"AAPL cost={aapl4['llm_cost_usd']}, expected 0.10"
        elif abs(aapl4["pnl_per_cost_usd"] - 2000.0) > 0.5:
            rc_err = f"AAPL ratio={aapl4['pnl_per_cost_usd']}, expected 2000.0"
        else:
            rc_ok = True
    except Exception as e:
        rc_err = f"{type(e).__name__}: {e}"

    results.append((
        "PASS" if rc_ok else "FAIL",
        "behavioral_ratio_computation_pnl_200_cost_010_yields_2000",
        f"pnl=200, cost=0.10 must yield pnl_per_cost_usd=2000.0 ({rc_err})",
    ))

    # ---- Claim 10: response note documents the approximation.
    response_note_text = (
        "LLM cost split proportionally by analysis count per ticker"
    )
    note_present = response_note_text in pt_src
    results.append((
        "PASS" if note_present else "FAIL",
        "response_note_documents_proportional_approximation",
        "_compute_attribution response 'note' must document the proportional-split approximation",
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
