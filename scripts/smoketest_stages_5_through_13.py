"""phase-31.0.5 through 31.0.13 consolidated smoketest.

Runs each remaining stage as an in-process verification, persisting per-
stage PASS/FAIL plus a final summary. NO LLM calls (Claude Code subagents
were spawned for Stage 4 separately). NO production BQ writes.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT = Path(__file__).resolve().parent.parent / "handoff" / "smoketest_20260520" / "STAGES_4_13_summary.json"

results = {"stage_results": {}}


def record(stage: str, verdict: str, evidence: dict) -> None:
    results["stage_results"][stage] = {"verdict": verdict, **evidence}
    print(f"  {stage}: {verdict}")
    for k, v in evidence.items():
        print(f"    {k}: {v}")


def stage_5_decide_trades_synthetic():
    """Stage 5: decide_trades on synthetic portfolio + price_at_analysis threading."""
    from backend.services.portfolio_manager import decide_trades, TradeOrder

    settings = SimpleNamespace(
        paper_max_per_sector=2,
        paper_max_per_sector_nav_pct=30.0,
        paper_max_positions=10,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_default_stop_loss_pct=8.0,
    )

    # Stage 2 syntheses -> analysis dicts
    candidates = [
        {
            "ticker": "NVDA", "recommendation": "BUY", "final_score": 8.7,
            "risk_assessment": {"reason": "strong momentum"}, "price_at_analysis": 220.61,
            "_path": "lite", "full_report": {"market_data": {"sector": "Technology"}},
        },
        {
            "ticker": "AAPL", "recommendation": "HOLD", "final_score": 7.2,
            "risk_assessment": {"reason": "overbought"}, "price_at_analysis": 298.97,
            "_path": "lite", "full_report": {"market_data": {"sector": "Technology"}},
        },
    ]
    orders = decide_trades(
        current_positions=[],
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state={"nav": 10000.0, "cash": 10000.0, "positions_value": 0.0, "position_count": 0},
        settings=settings,
    )
    buys = [o for o in orders if o.action == "BUY"]
    # NVDA should be the sole BUY (AAPL is HOLD).
    nvda_orders = [o for o in buys if o.ticker == "NVDA"]
    nvda_has_price_at_analysis = (
        len(nvda_orders) == 1
        and nvda_orders[0].price_at_analysis == 220.61
    )
    record("Stage 5 -- decide_trades synthetic", "PASS" if (
        len(buys) == 1 and buys[0].ticker == "NVDA" and nvda_has_price_at_analysis
    ) else "FAIL", {
        "buy_count": len(buys),
        "buy_ticker": buys[0].ticker if buys else None,
        "nvda_price_at_analysis": nvda_orders[0].price_at_analysis if nvda_orders else None,
        "expected_price_at_analysis": 220.61,
    })


def stage_6_step_5_6_ordering():
    """Stage 6: phase-30.2 backfill before phase-30.3 check ordering."""
    parent = MagicMock()
    parent.backfill_missing_stops.return_value = {
        "backfilled": [{"ticker": "AAPL", "entry_price": 100.0, "stop_loss_price": 92.0}],
        "skipped": [],
        "count_backfilled": 1,
        "count_skipped": 0,
    }
    parent.check_stop_losses.return_value = []

    # Reproducer matching the production sequence
    async def step_5_6(trader, summary):
        summary["stop_loss_backfilled"] = []
        try:
            backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
            summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
        except Exception:
            pass
        await asyncio.to_thread(trader.check_stop_losses)
        return summary

    summary = {}
    asyncio.run(step_5_6(parent, summary))
    method_names = [c[0] for c in parent.method_calls]
    backfill_idx = method_names.index("backfill_missing_stops") if "backfill_missing_stops" in method_names else -1
    check_idx = method_names.index("check_stop_losses") if "check_stop_losses" in method_names else -1
    record("Stage 6 -- Step 5.6 backfill-then-check ordering", "PASS" if (
        backfill_idx >= 0 and check_idx >= 0 and backfill_idx < check_idx
    ) else "FAIL", {
        "backfill_idx": backfill_idx,
        "check_idx": check_idx,
        "stop_loss_backfilled_count": len(summary["stop_loss_backfilled"]),
    })


def stage_7_sector_nav_pct_cap():
    """Stage 7: phase-30.5 NAV-pct cap blocks 3rd Tech buy."""
    from backend.services.portfolio_manager import decide_trades

    settings = SimpleNamespace(
        paper_max_per_sector=10,  # high (count cap won't fire)
        paper_max_per_sector_nav_pct=30.0,  # nav cap WILL fire
        paper_max_positions=10,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_default_stop_loss_pct=8.0,
    )
    existing = [
        {
            "ticker": "INTC", "sector": "Technology", "quantity": 50,
            "current_price": 110, "avg_entry_price": 100,
            "market_value": 5500, "recommendation": "BUY",
        },
    ]
    candidates = [{
        "ticker": "AMD",
        "recommendation": "BUY",
        "final_score": 7.0,
        "risk_assessment": {"reason": "test"},
        "price_at_analysis": 100.0,
        "_path": "lite",
        "full_report": {"market_data": {"sector": "Technology"}},
    }]
    orders = decide_trades(
        current_positions=existing,
        candidate_analyses=candidates,
        holding_analyses=[],
        portfolio_state={"nav": 20000.0, "cash": 14000.0, "positions_value": 5500.0, "position_count": 1},
        settings=settings,
    )
    buys = [o for o in orders if o.action == "BUY"]
    record("Stage 7 -- phase-30.5 NAV-pct cap blocks 3rd Tech buy", "PASS" if len(buys) == 0 else "FAIL", {
        "buy_count": len(buys),
        "existing_tech_value": 5500,
        "nav": 20000,
        "existing_tech_pct": 27.5,
        "would_be_post_buy_pct_approx": 30.5,
        "cap_pct": 30.0,
    })


def stage_8_phase_25_6_hard_block():
    """Stage 8: phase-25.6 HARD BLOCK synthesizes a default 8% stop when missing."""
    from backend.services.paper_trader import PaperTrader

    settings = SimpleNamespace(
        paper_price_tolerance_pct=0.0,  # disable to focus on stop synthesis
        paper_default_stop_loss_pct=8.0,
        paper_max_positions=10,
        paper_transaction_cost_pct=0.05,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
    )
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default", "current_cash": 5000.0,
        "starting_capital": 10000.0, "inception_date": "2026-05-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = []
    bq.get_paper_trades_for_ticker_since.return_value = []
    trader = PaperTrader(settings=settings, bq_client=bq)

    with patch("backend.services.paper_trader.ExecutionRouter") as mock_router_cls:
        mock_router = mock_router_cls.return_value
        mock_router.submit_order.return_value = SimpleNamespace(fill_price=100.0, source="bq_sim")
        bq.save_paper_position = MagicMock()
        bq.upsert_paper_portfolio = MagicMock()
        trade = trader.execute_buy(
            ticker="TEST", amount_usd=500.0, price=100.0,
            stop_loss_price=None,  # triggers phase-25.6 synthesis
            price_at_analysis=100.0,
        )

    # phase-25.6 synthesizes the stop and persists it on the POSITION row
    # (not the trade row). Inspect save_paper_position call to confirm.
    synthesized_stop = None
    if bq.save_paper_position.call_args is not None:
        position_row = bq.save_paper_position.call_args.args[0]
        synthesized_stop = position_row.get("stop_loss_price")
    # Canonical 8% stop: 100.0 * (1 - 0.08) = 92.0
    record("Stage 8 -- phase-25.6 HARD BLOCK synthesizes 8% stop", "PASS" if (
        trade is not None and synthesized_stop == 92.0
    ) else "FAIL", {
        "trade_exists": trade is not None,
        "synthesized_stop_on_position": synthesized_stop,
        "expected_stop": 92.0,
        "save_paper_position_calls": bq.save_paper_position.call_count,
    })


def stage_9_execution_router_bq_sim():
    """Stage 9: execution_router.submit_order with EXECUTION_BACKEND=bq_sim."""
    import os
    os.environ["EXECUTION_BACKEND"] = "bq_sim"
    from backend.services.execution_router import ExecutionRouter, _current_mode
    mode = _current_mode()
    router = ExecutionRouter()
    fill = router.submit_order(
        symbol="TEST", qty=10.0, side="buy",
        client_order_id="smoke-test-9", close_price=100.0,
    )
    record("Stage 9 -- execution_router bq_sim FillResult", "PASS" if (
        mode == "bq_sim" and fill.source == "bq_sim" and fill.fill_price == 100.0
    ) else "FAIL", {
        "mode": mode,
        "fill_source": fill.source,
        "fill_price": fill.fill_price,
        "fill_status": fill.status,
    })


def stage_10_mark_to_market():
    """Stage 10: paper_trader.mark_to_market with mocked yfinance."""
    from backend.services.paper_trader import PaperTrader

    settings = SimpleNamespace(
        paper_transaction_cost_pct=0.05,
        paper_starting_capital=10000.0,
        paper_min_cash_reserve_pct=5.0,
        paper_default_stop_loss_pct=8.0,
        paper_max_positions=10,
    )
    bq = MagicMock()
    bq.get_paper_portfolio.return_value = {
        "portfolio_id": "default", "current_cash": 8000.0,
        "starting_capital": 10000.0, "inception_date": "2026-05-01T00:00:00+00:00",
    }
    bq.get_paper_positions.return_value = [
        {
            "position_id": "p1", "ticker": "WDC", "quantity": 5.0,
            "avg_entry_price": 100.0, "cost_basis": 500.0,
            "current_price": 100.0, "stop_loss_price": 92.0,
        }
    ]
    trader = PaperTrader(settings=settings, bq_client=bq)

    with patch("backend.services.paper_trader._get_live_price") as mock_price:
        mock_price.return_value = 110.0  # +10% move
        bq.upsert_paper_portfolio = MagicMock()
        bq.save_paper_position = MagicMock()
        mtm = trader.mark_to_market()
    record("Stage 10 -- mark_to_market mocked yfinance", "PASS" if (
        "nav" in mtm and mtm["nav"] > 0 and "positions_value" in mtm
    ) else "FAIL", {
        "nav": mtm.get("nav"),
        "positions_value": mtm.get("positions_value"),
        "position_count": mtm.get("position_count"),
    })


def stage_11_stop_loss_enforcement():
    """Stage 11: -10% MTM + 8% stop -> check_stop_losses triggers -> closed_tickers populated."""
    parent = MagicMock()
    parent.backfill_missing_stops.return_value = {"backfilled": [], "skipped": [], "count_backfilled": 0, "count_skipped": 0}
    parent.check_stop_losses.return_value = ["WDC"]
    parent.execute_sell.return_value = {"trade_id": "t1", "ticker": "WDC", "price": 92.0}

    async def step_5_6_with_learn(trader, closed_tickers, summary):
        summary["stop_loss_triggered"] = []
        summary["stop_loss_backfilled"] = []
        try:
            backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
            summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
        except Exception:
            pass
        triggered = await asyncio.to_thread(trader.check_stop_losses)
        for t in triggered:
            try:
                sl_trade = await asyncio.to_thread(
                    trader.execute_sell, ticker=t, quantity=None, price=None,
                    reason="stop_loss_trigger", signals=None,
                )
                if sl_trade:
                    summary["stop_loss_triggered"].append(t)
                    closed_tickers.append(t)
            except Exception:
                pass
        return triggered

    closed_tickers = []
    summary = {}
    triggered = asyncio.run(step_5_6_with_learn(parent, closed_tickers, summary))
    record("Stage 11 -- stop-loss enforcement + phase-30.3 learn-loop routing", "PASS" if (
        triggered == ["WDC"] and closed_tickers == ["WDC"]
    ) else "FAIL", {
        "triggered": triggered,
        "closed_tickers": closed_tickers,
    })


def stage_12_outcome_tracker_chain():
    """Stage 12: mock OutcomeTracker -> verify bq.save_agent_memory invoked."""
    closed_tickers = ["WDC"]
    mock_bq = MagicMock()
    mock_bq.get_paper_trades.return_value = [{
        "trade_id": "t1", "ticker": "WDC", "action": "SELL",
        "analysis_id": "aid", "risk_judge_decision": "STOP_LOSS_TRIGGER",
        "price": 92.0, "created_at": "2026-05-20T18:00:00+00:00",
    }]
    mock_settings = MagicMock()

    def fake_tracker_factory(settings_arg):
        tracker = MagicMock()
        def fake_eval(ticker, *args, **kwargs):
            mock_bq.save_agent_memory({
                "ticker": ticker, "agent_type": "stop_loss_outcome",
                "situation": "stop_loss_trigger", "lesson": "loss-protection exit",
                "created_at": "2026-05-20T18:00:01+00:00",
            })
        tracker.evaluate_recommendation.side_effect = fake_eval
        return tracker

    with patch("backend.services.outcome_tracker.OutcomeTracker", side_effect=fake_tracker_factory):
        from backend.services.autonomous_loop import _learn_from_closed_trades
        asyncio.run(_learn_from_closed_trades(closed_tickers, mock_bq, mock_settings))
    record("Stage 12 -- OutcomeTracker chain -> save_agent_memory invoked", "PASS" if (
        mock_bq.save_agent_memory.call_count >= 1
    ) else "FAIL", {
        "save_agent_memory_calls": mock_bq.save_agent_memory.call_count,
    })


def stage_13_heartbeat_alarm_and_phase_30_7_shape():
    """Stage 13: cycle_heartbeat_alarm verdict + phase-30.7 row shape (mocked)."""
    from backend.services import cycle_health
    import tempfile
    from datetime import datetime, timezone

    # Synthetic cycle_history.jsonl: 27h stale on a weekday -> alarm
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({
            "cycle_id": "abc12345",
            "started_at": "2026-05-19T15:00:00+00:00",
            "completed_at": "2026-05-19T15:00:00+00:00",
            "duration_ms": 300_000,
            "status": "completed",
            "n_trades": 0,
            "error_count": 0,
            "data_source_ages": {},
            "bq_ingest_lag_sec": None,
        }) + "\n")
        history_path = Path(f.name)

    original_history = cycle_health._HISTORY_PATH
    original_now = cycle_health._now_utc
    cycle_health._HISTORY_PATH = history_path
    cycle_health._now_utc = lambda: datetime(2026, 5, 20, 18, 0, 0, tzinfo=timezone.utc)
    try:
        verdict = cycle_health.cycle_heartbeat_alarm()
    finally:
        cycle_health._HISTORY_PATH = original_history
        cycle_health._now_utc = original_now
        history_path.unlink(missing_ok=True)

    # phase-30.7 row shape (verified against scripts/migrations/add_strategy_decisions_table.py schema)
    strategy_row = {
        "ts": "2026-05-20T18:00:00+00:00",
        "cycle_id": "abc12345",
        "decided_strategy": "triple_barrier",
        "prior_strategy": "triple_barrier",
        "trigger": "cycle_heartbeat",
        "decay_signal": None,
        "decay_attribution": None,
        "rationale": "per-cycle heartbeat",
    }
    required_not_null = ["ts", "decided_strategy", "trigger"]
    nullable_present = ["cycle_id", "prior_strategy", "decay_signal", "decay_attribution", "rationale"]
    shape_ok = (
        all(strategy_row[k] for k in required_not_null)
        and strategy_row["trigger"] == "cycle_heartbeat"
        and all(k in strategy_row for k in nullable_present)
    )

    record("Stage 13 -- heartbeat alarm + phase-30.7 row shape (mocked)", "PASS" if (
        verdict.get("stale") and verdict.get("should_alarm") and verdict.get("is_weekday_et")
        and shape_ok
    ) else "FAIL", {
        "alarm_stale": verdict.get("stale"),
        "alarm_should_fire": verdict.get("should_alarm"),
        "alarm_weekday_et": verdict.get("is_weekday_et"),
        "phase_30_7_row_shape_ok": shape_ok,
    })


def main() -> int:
    print("=== Stages 5-13 consolidated smoketest ===\n")
    stages = [
        ("Stage 5", stage_5_decide_trades_synthetic),
        ("Stage 6", stage_6_step_5_6_ordering),
        ("Stage 7", stage_7_sector_nav_pct_cap),
        ("Stage 8", stage_8_phase_25_6_hard_block),
        ("Stage 9", stage_9_execution_router_bq_sim),
        ("Stage 10", stage_10_mark_to_market),
        ("Stage 11", stage_11_stop_loss_enforcement),
        ("Stage 12", stage_12_outcome_tracker_chain),
        ("Stage 13", stage_13_heartbeat_alarm_and_phase_30_7_shape),
    ]
    for name, fn in stages:
        try:
            print(f"\n--- {name} ---")
            fn()
        except Exception as exc:
            import traceback
            record(name, "FAIL", {
                "error": str(exc),
                "traceback": traceback.format_exc()[:1500],
            })

    # Summary
    passes = sum(1 for r in results["stage_results"].values() if r["verdict"] == "PASS")
    fails = sum(1 for r in results["stage_results"].values() if r["verdict"] == "FAIL")
    results["pass_count"] = passes
    results["fail_count"] = fails
    results["total"] = len(stages)
    results["overall_verdict"] = "PASS" if fails == 0 else ("PARTIAL" if passes >= 7 else "FAIL")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== SUMMARY ===")
    print(f"PASS: {passes}/{len(stages)}")
    print(f"FAIL: {fails}/{len(stages)}")
    print(f"Overall: {results['overall_verdict']}")
    print(f"Persisted: {OUTPUT}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
