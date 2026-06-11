"""phase-57.1 (55.3 findings F-3/F-8): binding RiskJudge gate regression tests.

The away week executed 3 BUYs at risk_judge_decision='REJECT' -- ALL via the
swap path (HPE 06-02, DELL 06-03, 066570.KS 06-09; net realized -$23.45) --
because the verdict was advisory-only. These tests pin:

1. the binding gate blocks a REJECT candidate on BOTH BUY paths (main + swap)
   when paper_risk_judge_reject_binding=True, and preserves the advisory
   behavior (BUY still emitted) when False/default;
2. default-OFF byte-identity: order lists unchanged on REJECT-free sets, and
   the RiskJudge prompts render as the verbatim pre-57.1 constants;
3. prompt-context correctness with the flag ON (configured cap, no phantom
   10%, live sector-breakdown line);
4. per-cycle single-compute of the sector context (structural: the per-ticker
   analyzers RECEIVE a precomputed portfolio_context and do not fetch
   positions themselves).

All offline; fixtures mirror test_portfolio_swap.py. No LLM/BQ/network.
"""
from __future__ import annotations

import inspect

from backend.config.settings import Settings
from backend.services import autonomous_loop as al
from backend.services.portfolio_manager import decide_trades


def _make_settings(**overrides) -> Settings:
    base = {
        "paper_starting_capital": 10000.0,
        "paper_max_positions": 10,
        "paper_max_per_sector": 2,
        "paper_max_per_sector_nav_pct": 30.0,
        "paper_max_factor_corr": 0.0,
        "paper_min_cash_reserve_pct": 5.0,
        "paper_swap_enabled": True,
        "paper_swap_min_delta_pct": 25.0,
        "paper_swap_max_per_cycle": 2,
    }
    base.update(overrides)
    return Settings(**base)


def _holding(ticker: str, sector: str, market_value: float) -> dict:
    return {
        "ticker": ticker,
        "sector": sector,
        "market_value": market_value,
        "current_price": market_value / 10.0,
        "recommendation": "BUY",
    }


def _holding_analysis(ticker: str, final_score: float) -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-06-11",
        "recommendation": "BUY",
        "final_score": final_score,
    }


def _candidate(ticker: str, sector: str, final_score: float, decision: str = "APPROVE_FULL") -> dict:
    return {
        "ticker": ticker,
        "analysis_date": "2026-06-11",
        "recommendation": "BUY",
        "final_score": final_score,
        "risk_assessment": {
            "decision": decision,
            "recommended_position_pct": 10.0,
            "reasoning": "test fixture",
        },
        "price_at_analysis": 100.0,
        "sector": sector,
        "full_report": {"market_data": {"sector": sector}},
    }


def _portfolio_state(nav: float, cash: float, positions: list[dict]) -> dict:
    return {
        "nav": nav,
        "cash": cash,
        "positions_value": nav - cash,
        "position_count": len(positions),
    }


# ── criterion 1: binding gate, MAIN BUY path ─────────────────────────
def test_reject_binding_main_path_off_emits_on_blocks():
    """Flag OFF (default): a REJECT candidate's BUY IS emitted (advisory,
    pre-57.1 behavior). Flag ON: the BUY is ABSENT and blocked_out records it."""
    positions: list[dict] = []
    cand = [_candidate("REJ1", "Technology", 0.9, decision="REJECT")]
    state = _portfolio_state(10_000.0, 10_000.0, positions)

    # default flag (OFF) -- advisory: BUY emitted
    s_off = _make_settings()
    assert s_off.paper_risk_judge_reject_binding is False  # ships default-OFF
    orders_off = decide_trades(positions, cand, [], state, s_off)
    assert any(o.ticker == "REJ1" and o.action == "BUY" for o in orders_off), (
        f"flag-OFF must preserve the advisory BUY; orders={orders_off}"
    )

    # flag ON -- binding: BUY absent, blocked_out populated
    s_on = _make_settings(paper_risk_judge_reject_binding=True)
    blocked: list[dict] = []
    orders_on = decide_trades(positions, cand, [], state, s_on, blocked_out=blocked)
    assert not any(o.ticker == "REJ1" and o.action == "BUY" for o in orders_on), (
        f"flag-ON must block the REJECT BUY; orders={orders_on}"
    )
    assert len(blocked) == 1 and blocked[0]["ticker"] == "REJ1"
    assert blocked[0]["decision"] == "REJECT"


# ── criterion 1: binding gate, SWAP path (the away-week vulnerability) ──
def _swap_scenario(decision_for_top: str):
    """The test_portfolio_swap zero-buy scenario with the TOP tech candidate
    carrying `decision_for_top`. All 3 real-world REJECT executions were
    swap_buy -- this is the path that must be covered."""
    nav, cash = 10_000.0, 2_000.0
    tech_scores = [0.55, 0.58, 0.60, 0.65, 0.68, 0.70, 0.73, 0.75]
    positions = [
        _holding(f"TECH{i}", "Technology", 1100.0) for i in range(len(tech_scores))
    ]
    positions.append(_holding("INDU1", "Industrials", 1000.0))
    holding_analyses = [
        _holding_analysis(p["ticker"], tech_scores[i] if i < 8 else 0.65)
        for i, p in enumerate(positions)
    ]
    candidate_analyses = [
        _candidate("TECH_NEW1", "Technology", 0.85, decision=decision_for_top),
        _candidate("TECH_NEW2", "Technology", 0.82),
        _candidate("INDU_NEW", "Industrials", 0.70),
    ]
    state = _portfolio_state(nav, cash, positions)
    return positions, candidate_analyses, holding_analyses, state


def test_reject_binding_swap_path_off_emits_on_blocks():
    positions, cands, holds, state = _swap_scenario("REJECT")

    # Flag OFF: TECH_NEW1 (REJECT) still swap-buys -- reproduces the
    # away-week vulnerability (advisory verdict executed via swap_buy).
    orders_off = decide_trades(positions, cands, holds, state, _make_settings())
    swap_buys_off = {o.ticker for o in orders_off if o.reason == "swap_buy"}
    assert "TECH_NEW1" in swap_buys_off, (
        f"flag-OFF must preserve the (vulnerable) swap BUY; swap_buys={swap_buys_off}"
    )
    rejected_order = next(o for o in orders_off if o.ticker == "TECH_NEW1")
    assert rejected_order.risk_judge_decision == "REJECT"

    # Flag ON: TECH_NEW1 emits NO order of any kind; the next-ranked
    # candidate (TECH_NEW2) takes the swap slot (budget reallocation by
    # construction -- the dropped candidate never enters buy_candidates).
    blocked: list[dict] = []
    orders_on = decide_trades(
        positions, cands, holds, state,
        _make_settings(paper_risk_judge_reject_binding=True),
        blocked_out=blocked,
    )
    on_tickers = {o.ticker for o in orders_on if o.action == "BUY"}
    assert "TECH_NEW1" not in on_tickers, (
        f"flag-ON must block the REJECT candidate on the swap path; BUYs={on_tickers}"
    )
    assert "TECH_NEW2" in on_tickers, (
        f"the next-ranked survivor should take the freed slot; BUYs={on_tickers}"
    )
    assert [b["ticker"] for b in blocked] == ["TECH_NEW1"]


# ── criterion 2: default-OFF byte-identity ───────────────────────────
def test_off_identity_orders_no_reject_set():
    """With NO REJECT verdicts, flag ON and flag OFF produce identical order
    lists (the gate only ever touches REJECT candidates)."""
    positions, cands, holds, state = _swap_scenario("APPROVE_FULL")
    orders_off = decide_trades(positions, cands, holds, state, _make_settings())
    orders_on = decide_trades(
        positions, cands, holds, state,
        _make_settings(paper_risk_judge_reject_binding=True),
    )
    key = lambda o: (o.ticker, o.action, o.reason, o.amount_usd, o.quantity)  # noqa: E731
    assert [key(o) for o in orders_off] == [key(o) for o in orders_on]


def test_off_identity_prompts_are_verbatim_constants():
    s_off = _make_settings()
    assert al._build_risk_judge_system(s_off) is al._LITE_RISK_JUDGE_SYSTEM
    assert al._build_risk_judge_template(s_off, "anything") is al._LITE_RISK_JUDGE_TEMPLATE
    # and the rendered template equals a render of the raw constant
    kwargs = dict(ticker="MU", name="Micron", sector="Technology", pe_ratio=45.1,
                  market_cap_b=1076.8, momentum_20d=20.1, momentum_60d=124.2,
                  trader_action="BUY", trader_confidence=75)
    assert (al._build_risk_judge_template(s_off, "ctx").format(**kwargs)
            == al._LITE_RISK_JUDGE_TEMPLATE.format(**kwargs))


# ── criterion 3: prompt-context correctness flag ON ──────────────────
def test_prompt_content_flag_on_real_cap_and_sector_line():
    s_on = _make_settings(paper_risk_judge_reject_binding=True)
    system = al._build_risk_judge_system(s_on)
    assert "exceed 30% of portfolio NAV in one sector" in system
    assert "10% of portfolio in one sector" not in system

    fake_positions = [
        {"ticker": "066570.KS", "sector": "Technology", "quantity": 1.468448,
         "current_price": 248_000.0, "avg_entry_price": 248_000.0},
    ]
    ctx = al._build_portfolio_sector_context(fake_positions)
    assert "Technology 100.0%" in ctx

    kwargs = dict(ticker="MU", name="Micron", sector="Technology", pe_ratio=45.1,
                  market_cap_b=1076.8, momentum_20d=20.1, momentum_60d=124.2,
                  trader_action="BUY", trader_confidence=75)
    rendered = al._build_risk_judge_template(s_on, ctx).format(**kwargs)
    assert "Current portfolio context: invested-book sector weights: Technology 100.0%" in rendered


def test_sector_context_all_cash_and_fallback_price():
    assert al._build_portfolio_sector_context([]) == "no open positions (all cash)"
    # current_price missing -> falls back to avg_entry_price (mark_to_market
    # has not run at the call site)
    ctx = al._build_portfolio_sector_context(
        [{"ticker": "DELL", "sector": "Technology", "quantity": 2.0,
          "current_price": None, "avg_entry_price": 370.7}]
    )
    assert "Technology 100.0%" in ctx


# ── criterion 4: per-cycle single-compute (structural assertions) ────
def test_analyzers_receive_precomputed_context_not_positions_fetch():
    """The per-ticker analyzers RECEIVE portfolio_context as a parameter and
    never fetch positions themselves (a per-ticker get_positions inside the
    concurrent fan-out would be N redundant BQ reads + a race)."""
    for fn in (al._run_single_analysis, al._run_claude_analysis, al._run_gemini_analysis):
        assert "portfolio_context" in inspect.signature(fn).parameters, fn.__name__
        assert "get_positions" not in inspect.getsource(fn), (
            f"{fn.__name__} must not fetch positions per-ticker"
        )
    # the single compute site lives in the cycle (run_daily_cycle):
    cycle_src = inspect.getsource(al.run_daily_cycle)
    assert "_build_portfolio_sector_context(positions)" in cycle_src
