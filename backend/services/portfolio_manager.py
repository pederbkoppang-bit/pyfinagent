"""
Portfolio Manager — decides which trades to execute based on analysis results.

Implements sell-first-then-buy logic with Risk Judge position sizing.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.config.settings import Settings
from backend.services import risk_overrides
from backend.backtest import markets  # phase-50.3: derive market from ticker suffix
from backend.services.signal_attribution import extract_signals_from_analysis, extract_all_signals

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    ticker: str
    action: str  # "BUY" or "SELL"
    amount_usd: Optional[float] = None   # For buys
    quantity: Optional[float] = None      # For sells (None = full exit)
    reason: str = ""
    analysis_id: str = ""
    risk_judge_decision: str = ""
    stop_loss_price: Optional[float] = None
    risk_judge_position_pct: Optional[float] = None
    price: Optional[float] = None
    signals: list[dict] = field(default_factory=list)  # 4.5.5 agent attribution
    sector: str = ""  # phase-23.2.6-fix: persisted to paper_positions.sector at execute_buy
    market: str = "US"  # phase-50.3: derived from the ticker suffix (markets.market_for_symbol); "US" = byte-identical
    # phase-30.6: analysis-time price reference for the price-tolerance gate.
    # Distinct from `price` which historically held the same value but will be
    # overwritten with the LIVE fetch in autonomous_loop Step 7 -- separating
    # the two so execute_buy can reject when live diverges from analysis.
    price_at_analysis: Optional[float] = None
    # phase-40.8.1 (P3): FF3 factor loadings carried from candidate through
    # to in-memory pos_row in execute_buy. None until phase-40.8.1 producer
    # populates. BQ persistence deferred to phase-40.8.2.
    factor_loadings: Optional[dict] = None
    # phase-61.2 (criterion 5): the ANALYSIS recommendation (BUY/STRONG_BUY)
    # behind this order, distinct from `reason` (the trade mechanism, e.g.
    # "new_buy_signal"/"swap_buy"). paper_trader historically wrote `reason`
    # into paper_positions.recommendation, so the signal_downgrade SELL rule
    # (old_rec in _BUY_RECS at :127) could never match -- structurally dead.
    # Consumed by execute_buy only when
    # paper_position_recommendation_fix_enabled is ON.
    analysis_recommendation: str = ""


# Recommendations that imply selling
_SELL_RECS = {"SELL", "STRONG_SELL"}
# Recommendations that indicate downgrade from a prior buy
_DOWNGRADE_RECS = {"HOLD", "SELL", "STRONG_SELL"}
# Recommendations that trigger buying
_BUY_RECS = {"BUY", "STRONG_BUY"}


def decide_trades(
    current_positions: list[dict],
    candidate_analyses: list[dict],
    holding_analyses: list[dict],
    portfolio_state: dict,
    settings: Settings,
    candidates_by_ticker: dict[str, dict] | None = None,
    blocked_out: list[dict] | None = None,
) -> list[TradeOrder]:
    """
    Decide which trades to execute.

    Args:
        current_positions: List of open position dicts from BQ
        candidate_analyses: Analysis results for new candidates (with recommendation, risk_assessment)
        holding_analyses: Analysis results for current holdings (re-evaluation)
        portfolio_state: Dict with keys: nav, cash, positions_value, position_count
        settings: App settings
        blocked_out: phase-57.1 (F-3) optional out-channel -- when the binding
            RiskJudge gate drops a REJECT candidate, a dict per blocked BUY is
            appended here so the cycle summary can surface it. Backward
            compatible: callers that pass nothing lose only observability.

    Returns:
        List of TradeOrders (sells first, then buys)
    """
    orders: list[TradeOrder] = []
    nav = portfolio_state.get("nav", settings.paper_starting_capital)
    cash = portfolio_state.get("cash", nav)
    # phase-49.1: runtime-overridable (operator can tune live, no restart).
    min_cash = nav * (
        risk_overrides.get_effective("paper_min_cash_reserve_pct", settings.paper_min_cash_reserve_pct) / 100.0
    )
    held_tickers = {p["ticker"] for p in current_positions}

    # ── 1. Sell decisions (process first to free up cash) ────────

    # Build lookup: ticker -> re-evaluation result
    holding_lookup = {}
    for analysis in holding_analyses:
        ticker = analysis.get("ticker", "")
        if ticker:
            holding_lookup[ticker] = analysis

    # phase-61.2 (criterion 5 interaction guard): reviving signal_downgrade
    # while synthetic HOLDs can still be fabricated means a transient rail
    # failure on a held ticker's re-eval would SELL a healthy position. The
    # combination is legal (flags are independent operator levers) but loud.
    if getattr(settings, "paper_position_recommendation_fix_enabled", False) and not getattr(
        settings, "paper_synthesis_integrity_enabled", False
    ):
        logger.warning(
            "paper_position_recommendation_fix_enabled is ON while "
            "paper_synthesis_integrity_enabled is OFF -- rail-failure synthetic "
            "HOLDs can trigger signal_downgrade SELLs of healthy positions. "
            "Enable the integrity flag first (phase-61.2 interaction hazard)."
        )

    for pos in current_positions:
        ticker = pos["ticker"]
        analysis = holding_lookup.get(ticker)

        # Stop loss check (already priced in mark-to-market)
        stop = pos.get("stop_loss_price")
        current = pos.get("current_price", 0)
        if stop and current and current <= stop:
            orders.append(TradeOrder(
                ticker=ticker, action="SELL", reason="stop_loss",
                price=current,
            ))
            continue

        # If we have a fresh re-evaluation
        if analysis:
            rec = (analysis.get("recommendation") or "HOLD").upper()  # phase-66.2 review C1: None-safe (lite fallback can return recommendation=None -> None.upper() crashed decide_trades)
            old_rec = (pos.get("recommendation") or "").upper()

            # Explicit sell signal
            if rec in _SELL_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="sell_signal",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                    signals=extract_signals_from_analysis(analysis),
                ))
                continue

            # Downgrade: was a buy, now hold/sell
            if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS:
                orders.append(TradeOrder(
                    ticker=ticker, action="SELL", reason="signal_downgrade",
                    analysis_id=analysis.get("analysis_date", ""),
                    price=pos.get("current_price"),
                    signals=extract_signals_from_analysis(analysis),
                ))
                continue

    # Tickers being sold
    selling_tickers = {o.ticker for o in orders if o.action == "SELL"}

    # Estimate cash freed from sells
    estimated_freed_cash = 0.0
    for pos in current_positions:
        if pos["ticker"] in selling_tickers:
            estimated_freed_cash += pos.get("market_value", 0)

    available_cash = cash + estimated_freed_cash - min_cash

    # ── 2. Buy decisions ─────────────────────────────────────────

    # Count positions after sells
    remaining_positions = len(current_positions) - len(selling_tickers)

    buy_candidates = []
    for analysis in candidate_analyses:
        ticker = analysis.get("ticker", "")
        rec = (analysis.get("recommendation") or "HOLD").upper()  # phase-66.2 review C1: None-safe (lite fallback can return recommendation=None -> None.upper() crashed decide_trades)

        # Skip if already held (and not being sold)
        if ticker in held_tickers and ticker not in selling_tickers:
            continue

        if rec not in _BUY_RECS:
            continue

        # Extract Risk Judge sizing
        risk_assessment = analysis.get("risk_assessment", {})
        # phase-66.2 RJ-shape fix (flag-gated, default OFF): the FULL-path
        # judge verdict is nested under risk_assessment['judge']
        # (risk_debate.py:310); the lite path is flat. Resolve nested-first so
        # sizing, the REJECT gate, and the recorded decision all read the real
        # judge (matching api/analysis.py:158 + tasks/analysis.py:162). OFF ->
        # _rj_view IS risk_assessment == current top-level behavior (lite
        # already flat, so lite is byte-identical either way).
        _rj_view = risk_assessment
        if getattr(settings, "paper_risk_judge_shape_fix_enabled", False):
            _judge = risk_assessment.get("judge")
            if isinstance(_judge, dict):
                _rj_view = _judge
        position_pct = _extract_position_pct(_rj_view, analysis)
        # Respect an explicit 0.0 pct as no-buy (the zero-falsy `if pct:` in
        # _extract_position_pct otherwise falls through to the 10% default).
        if getattr(settings, "paper_risk_judge_shape_fix_enabled", False):
            _raw_pct = _rj_view.get("recommended_position_pct")
            if _raw_pct is not None:
                try:
                    position_pct = float(_raw_pct)
                except (ValueError, TypeError):
                    pass
        stop_loss = _extract_stop_loss(risk_assessment, analysis, settings=settings)
        final_score = analysis.get("final_score", 0)

        # phase-23.1.7: pull the screener candidate dict (contains momentum / RSI /
        # composite_score / conviction / signal-stack tags) so the rationale captures
        # ALL inputs that drove this decision, not just the LLM's verdict.
        screener_candidate = (candidates_by_ticker or {}).get(ticker)
        # phase-23.1.13: capture sector for the per-sector cap. Resolution order:
        # screener candidate (preferred -- enriched in autonomous_loop via
        # _fetch_ticker_meta) -> analysis full_report.market_data.sector ->
        # analysis.sector -> "Unknown" sentinel.
        cand_sector = ""
        if screener_candidate:
            cand_sector = screener_candidate.get("sector") or ""
        if not cand_sector:
            full_report = analysis.get("full_report") or {}
            md = full_report.get("market_data") or {}
            cand_sector = md.get("sector") or analysis.get("sector") or ""

        # phase-57.1 (55.3 finding F-3): BINDING RiskJudge gate. The away week
        # executed 3 REJECT BUYs -- all via the swap path, so the gate sits
        # HERE at the candidate-build chokepoint (the common ancestor feeding
        # both the main BUY-emit loop and, via sector_blocked, the swap path;
        # SEC 15c3-5(d) non-bypassable placement). REJECT-only: REDUCED/HEDGED
        # remain advisory sizing. Budget reallocates by construction -- a
        # dropped candidate never enters buy_candidates, so the next-ranked
        # survivor draws its cash in the emit loop. Flag default-OFF.
        # phase-66.2 RJ-shape fix: read the decision from the resolved judge
        # view (nested-first when the flag is ON) so the binding gate actually
        # sees a full-path REJECT. OFF -> _rj_view is risk_assessment (top-level).
        _rj_decision = (_rj_view.get("decision", "") or "")
        if (
            _rj_decision == "REJECT"
            and getattr(settings, "paper_risk_judge_reject_binding", False)
        ):
            logger.warning(
                "BINDING RiskJudge gate: BLOCKED BUY %s (decision=REJECT, "
                "final_score=%s) -- paper_risk_judge_reject_binding=ON (F-3)",
                ticker, round(float(final_score or 0), 3),
            )
            if blocked_out is not None:
                blocked_out.append({
                    "ticker": ticker,
                    "decision": _rj_decision,
                    "reason": risk_assessment.get("reason")
                    or risk_assessment.get("reasoning") or "",
                    "final_score": final_score,
                })
            continue

        buy_candidates.append({
            "ticker": ticker,
            "recommendation": rec,
            "position_pct": position_pct,
            "stop_loss_price": stop_loss,
            # phase-66.2 RJ-shape fix: record the resolved judge decision so
            # full-path BUYs no longer persist risk_judge_decision='' (66.2
            # criterion-1(a) requires it recorded). OFF -> top-level (lite).
            "risk_judge_decision": _rj_view.get("decision", ""),
            "analysis_id": analysis.get("analysis_date", ""),
            "final_score": final_score,
            "price": analysis.get("price_at_analysis"),
            "sector": cand_sector or "Unknown",
            "signals": extract_all_signals(analysis, candidate=screener_candidate),
        })

        decision = _rj_view.get("decision", "") or ""
        if decision and decision != "APPROVE_FULL":
            logger.info(
                "buy_candidate risk_judge decision=%s ticker=%s position_pct=%s final_score=%s",
                decision, ticker, position_pct, round(float(final_score or 0), 3),
            )

    # Sort by final_score descending (best opportunities first)
    buy_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)

    # phase-23.1.13: build sector_counts from current positions NOT being sold,
    # plus a meta lookup for legacy positions that lack a sector field. The cap
    # is `settings.paper_max_per_sector` (0 disables). We block a candidate's
    # BUY when its sector has already reached the cap; non-blocked sectors go
    # through normally. Increments after each accepted BUY so 3rd+ candidates
    # in the same sector skip cleanly.
    #
    # phase-30.5: ALSO build sector_market_values for the NAV-percentage cap
    # (`settings.paper_max_per_sector_nav_pct`, 0 disables, default 30). The
    # two caps fire independently: count handles "many small positions",
    # NAV-pct handles "one fat position dominating the sector". Sound source:
    # arXiv 2512.02227 Dec 2025 Orchestration Framework `sectorLimit: 0.30`.
    # Edge cases per research_brief.md Section 4: missing market_value -> 0
    # (conservative, no crash); cap=0 disables; candidate self-exceeds gets
    # blocked because existing_sector_value + buy_amount is checked; already-
    # over sector keeps existing semantics (no force-divest).
    # phase-49.1: runtime-overridable via risk_overrides (operator deployment control).
    max_per_sector = int(
        risk_overrides.get_effective("paper_max_per_sector", getattr(settings, "paper_max_per_sector", 0)) or 0
    )
    max_sector_nav_pct = float(
        risk_overrides.get_effective(
            "paper_max_per_sector_nav_pct", getattr(settings, "paper_max_per_sector_nav_pct", 0.0)
        ) or 0.0
    )
    # phase-40.8 (OPEN-5): FF3 factor-correlation cap. Default-OFF (0.0).
    max_factor_corr = float(getattr(settings, "paper_max_factor_corr", 0.0) or 0.0)
    # phase-70.2: optionally exempt the "Unknown" (missing-sector) bucket from the
    # count + NAV-pct caps, so a ticker-meta enrichment failure can't collapse N
    # real sectors into one bucket and freeze the funnel. OFF -> byte-identical.
    unknown_sector_cap_exempt = bool(getattr(settings, "paper_unknown_sector_cap_exempt", False))
    sector_counts: dict[str, int] = {}
    sector_market_values: dict[str, float] = {}
    if max_per_sector > 0 or max_sector_nav_pct > 0:
        for pos in current_positions:
            if pos["ticker"] in selling_tickers:
                continue
            s = (pos.get("sector") or "").strip() or "Unknown"
            sector_counts[s] = sector_counts.get(s, 0) + 1
            sector_market_values[s] = (
                sector_market_values.get(s, 0.0)
                + float(pos.get("market_value", 0) or 0)
            )
    port_factor_loadings: dict[str, float] = {}
    if max_factor_corr > 0:
        from backend.services.factor_correlation import aggregate_portfolio_loadings
        retained = [p for p in current_positions if p["ticker"] not in selling_tickers]
        port_factor_loadings = aggregate_portfolio_loadings(retained)

    # phase-cycle-1 (2026-05-26): collect candidates blocked by sector COUNT cap
    # so the post-loop swap path can consider sell-to-buy-better rebalances.
    sector_blocked: list[dict] = []

    # phase-49.1: runtime-overridable position cap (operator deployment control).
    max_positions = int(
        risk_overrides.get_effective("paper_max_positions", settings.paper_max_positions)
    )
    for cand in buy_candidates:
        if remaining_positions >= max_positions:
            # phase-23.2.22: emit a diagnostic line so 0-trade cycles are
            # diagnosable from logs without forensic analysis. Without this
            # the loop silently `break`s and the cycle reports "Executing 0
            # trades" with no explanation -- making the position cap look
            # like a bug.
            logger.info(
                "Position cap reached: %d held >= %d max -- skipping all BUY candidates",
                remaining_positions, max_positions,
            )
            break

        if available_cash <= 0:
            break

        # phase-23.1.13: per-GICS-sector cap (default 2). 0 disables the check.
        # phase-cycle-1 (2026-05-26): capture blocked candidates so the
        # post-loop swap path can consider them. North star = maximize profit;
        # mandate = default to firing, not gating, when risk caps permit.
        if max_per_sector > 0:
            cand_sector = cand.get("sector") or "Unknown"
            # phase-70.2: skip the count cap for the "Unknown" (missing-data) bucket
            # when exempt -- it is not concentration evidence. OFF -> byte-identical.
            _unk_exempt = unknown_sector_cap_exempt and cand_sector == "Unknown"
            current_in_sector = sector_counts.get(cand_sector, 0)
            if not _unk_exempt and current_in_sector >= max_per_sector:
                logger.info(
                    "Skipping BUY %s: sector %s at cap (%d/%d) -- queued for swap check",
                    cand["ticker"], cand_sector,
                    current_in_sector, max_per_sector,
                )
                sector_blocked.append(cand)
                continue

        # Position size: min(risk_judge_pct * NAV, available_cash)
        # phase-66.2 RJ-shape fix: under the flag, an explicit 0.0 pct means
        # no-buy (falls below the $50 floor below) instead of inverting to the
        # 10% default via `or`. OFF -> `or 10.0` byte-identical. A genuinely
        # absent pct (None, judge didn't specify) still defaults to 10%.
        if getattr(settings, "paper_risk_judge_shape_fix_enabled", False):
            position_pct = cand["position_pct"] if cand["position_pct"] is not None else 10.0
        else:
            position_pct = cand["position_pct"] or 10.0  # Default 10% if Risk Judge didn't specify
        target_amount = nav * (position_pct / 100.0)
        buy_amount = min(target_amount, available_cash)

        # Skip tiny positions (less than $50)
        if buy_amount < 50:
            logger.warning(
                "Skipping BUY %s: buy_amount=%.2f below $50 minimum (nav=%.2f position_pct=%s available_cash=%.2f)",
                cand["ticker"], buy_amount, nav, position_pct, available_cash,
            )
            continue

        # phase-30.5: per-sector NAV-percentage cap. Independent of the
        # count cap above. Cap=0 disables. Check AFTER buy_amount is known
        # so we know exactly how much this BUY would push the sector past.
        if max_sector_nav_pct > 0 and nav > 0:
            cand_sector_nav = cand.get("sector") or "Unknown"
            # phase-70.2: exempt the "Unknown" bucket from the NAV-pct cap too. OFF -> byte-identical.
            _unk_exempt_nav = unknown_sector_cap_exempt and cand_sector_nav == "Unknown"
            existing_sector_value = sector_market_values.get(cand_sector_nav, 0.0)
            projected_sector_pct = (
                (existing_sector_value + buy_amount) / nav * 100.0
            )
            if not _unk_exempt_nav and projected_sector_pct > max_sector_nav_pct:
                logger.info(
                    "Skipping BUY %s: sector %s would hit NAV-pct cap "
                    "(%.2f%% projected > %.2f%% cap; existing=$%.2f buy=$%.2f nav=$%.2f)",
                    cand["ticker"], cand_sector_nav,
                    projected_sector_pct, max_sector_nav_pct,
                    existing_sector_value, buy_amount, nav,
                )
                continue

        # phase-40.8 (OPEN-5): FF3 factor-correlation cap. Catches
        # cross-sector factor crowding that GICS cap misses. Default-OFF:
        # cap=0 short-circuits; cand missing factor_loadings short-circuits.
        if max_factor_corr > 0 and port_factor_loadings:
            cand_loadings = cand.get("factor_loadings")
            if cand_loadings:
                from backend.services.factor_correlation import factor_correlation_score
                corr = factor_correlation_score(cand_loadings, port_factor_loadings)
                if corr > max_factor_corr:
                    logger.info(
                        "Skipping BUY %s: FF3 factor correlation %.3f > cap %.3f "
                        "(cross-sector crowding -- per phase-40.8/OPEN-5)",
                        cand["ticker"], corr, max_factor_corr,
                    )
                    continue

        orders.append(TradeOrder(
            ticker=cand["ticker"],
            action="BUY",
            amount_usd=round(buy_amount, 2),
            reason="new_buy_signal",
            analysis_id=cand["analysis_id"],
            risk_judge_decision=cand["risk_judge_decision"],
            stop_loss_price=cand["stop_loss_price"],
            risk_judge_position_pct=cand["position_pct"],
            price=cand.get("price"),
            # phase-30.6: thread the analysis-time price separately so
            # execute_buy can compare it against the LIVE fetch and
            # reject when divergence > paper_price_tolerance_pct.
            price_at_analysis=cand.get("price"),
            signals=cand.get("signals", []),
            sector=cand.get("sector", ""),  # phase-23.2.6-fix
            market=markets.market_for_symbol(cand["ticker"]),  # phase-50.3: US for bare tickers (byte-identical)
            # phase-40.8.1 (P3): forward FF3 loadings to execute_buy
            # so the in-memory pos_row carries them.
            factor_loadings=cand.get("factor_loadings"),
            # phase-61.2 (criterion 5): the analysis verdict, so positions can
            # persist BUY/STRONG_BUY instead of the trade mechanism string.
            analysis_recommendation=cand.get("recommendation", ""),
        ))
        available_cash -= buy_amount
        remaining_positions += 1
        # phase-23.1.13: increment sector count after a BUY clears the cap
        # phase-30.5: ALSO increment sector_market_values so the next
        # candidate in the same sector sees the updated NAV-pct.
        if max_per_sector > 0 or max_sector_nav_pct > 0:
            cs = cand.get("sector") or "Unknown"
            sector_counts[cs] = sector_counts.get(cs, 0) + 1
            sector_market_values[cs] = sector_market_values.get(cs, 0.0) + buy_amount

    # phase-cycle-1 (2026-05-26): sell-to-buy-better swap path. North star =
    # maximize profit. Testing-phase mandate = do not idle on cash when a
    # higher-conviction candidate is sector-blocked by an existing low-
    # conviction holding. Gated by paper_swap_enabled + delta threshold +
    # max-per-cycle cap. Re-checks all risk gates on the projected post-swap
    # portfolio. Cited literature in handoff/current/contract.md.
    if (
        getattr(settings, "paper_swap_enabled", False)
        and sector_blocked
        and max_per_sector > 0
    ):
        swap_orders = _compute_swap_candidates(
            sector_blocked=sector_blocked,
            current_positions=current_positions,
            holding_lookup=holding_lookup,
            sector_counts=sector_counts,
            sector_market_values=sector_market_values,
            selling_tickers=selling_tickers,
            settings=settings,
            nav=nav,
        )
        orders.extend(swap_orders)

    # phase-cycle-1: enforce the sell-first-then-buy invariant on the final
    # orders list. Both the standard buy-loop and the swap path may have
    # appended BUYs that came before SELLs in append order (swap pairs are
    # internally SELL+BUY, but if a prior iteration's BUY was already in the
    # list, the new swap_SELL lands after it). Python's stable sort preserves
    # relative ordering within each group (signal-SELL, stop-loss-SELL,
    # swap-SELL stay in their original order; same for BUYs). The downstream
    # executor relies on cash being freed by SELLs before BUYs draw from it.
    orders.sort(key=lambda o: 0 if o.action == "SELL" else 1)

    logger.info(f"Trade decisions: {len([o for o in orders if o.action == 'SELL'])} sells, "
                f"{len([o for o in orders if o.action == 'BUY'])} buys")
    return orders


def _compute_swap_candidates(
    sector_blocked: list[dict],
    current_positions: list[dict],
    holding_lookup: dict[str, dict],
    sector_counts: dict[str, int],
    sector_market_values: dict[str, float],
    selling_tickers: set,
    settings: "Settings",
    nav: float,
) -> list[TradeOrder]:
    """phase-cycle-1: sell-to-buy-better swap path.

    For each sector-blocked candidate, find the lowest-conviction existing
    holding IN THE SAME SECTOR that is not already being sold. If the
    candidate's final_score exceeds the holding's by paper_swap_min_delta_pct
    (relative), emit a paired SELL + BUY. Cap at paper_swap_max_per_cycle.

    Risk-gate preservation: each swap is +1 BUY / -1 SELL in the same sector,
    so the sector COUNT remains unchanged. The NAV-pct cap is rechecked on
    the projected post-swap composition. The factor-correlation cap is NOT
    re-checked here (the swap stays within the same sector so loadings move
    in a bounded way; future tightening can revisit if backtest evidence
    supports it).
    """
    swap_orders: list[TradeOrder] = []
    min_delta = float(getattr(settings, "paper_swap_min_delta_pct", 25.0) or 0.0)
    max_per_cycle = int(getattr(settings, "paper_swap_max_per_cycle", 0) or 0)
    if max_per_cycle <= 0:
        return swap_orders

    # Index holdings by sector with their analysis score so we can find the
    # weakest holding per sector in O(1) per candidate.
    _churn_fix_on = bool(getattr(settings, "paper_swap_churn_fix_enabled", False))
    holdings_by_sector: dict[str, list[dict]] = {}
    for pos in current_positions:
        if pos["ticker"] in selling_tickers:
            continue
        sector = (pos.get("sector") or "Unknown").strip() or "Unknown"
        analysis = holding_lookup.get(pos["ticker"], {}) or {}
        score = analysis.get("final_score")
        if score is None:
            if _churn_fix_on:
                # phase-60.2 (AW-5): a missing same-cycle analysis is NOT
                # evidence of zero conviction -- the away week showed every
                # fresh BUY (analyzed at buy time, re-eval gated 3 days)
                # scored as sentinel 0.0 here and swapped out the next day
                # ("swap_for_higher_conviction" against fabricated evidence:
                # MU -6.3%, SNDK/DELL round trips, 81.4% weekly turnover,
                # 10 round trips net -$132). EXCLUDE the holding from
                # displacement entirely: we do not displace what we cannot
                # value on same-cycle evidence. Day-old real scores remain
                # valid for HOLDING (Alpha Decay: signal alpha decays over
                # months), and the re-eval cadence re-scores the position
                # within 3-4 days, restoring displaceability on true
                # evidence. LOCF valuation was considered and rejected for
                # DISPLACEMENT decisions: day-over-day score noise (mean
                # |delta| 1.10 on the 1-10 scale, 59.3 stability table) can
                # cross a 25% relative bar at low scores, re-admitting churn.
                logger.info(
                    "Swap path: %s has no same-cycle analysis -- excluded "
                    "from displacement this cycle (churn fix ON)",
                    pos["ticker"],
                )
                continue
            # Pre-60.2 sentinel (flag OFF, byte-identical): no fresh
            # analysis => treat as worst so swaps against it can fire.
            # KNOWN DEFECT (AW-5): this is the away-week churn engine;
            # enable paper_swap_churn_fix_enabled for the fix.
            score = 0.0
        holdings_by_sector.setdefault(sector, []).append(
            {
                "ticker": pos["ticker"],
                "sector": sector,
                "final_score": float(score),
                "market_value": float(pos.get("market_value", 0) or 0),
                "current_price": pos.get("current_price"),
            }
        )

    # Sort each sector list ascending by score so [0] is the weakest holding.
    for sector_list in holdings_by_sector.values():
        sector_list.sort(key=lambda h: h["final_score"])

    swaps_fired = 0
    swapped_tickers: set = set()

    for cand in sector_blocked:
        if swaps_fired >= max_per_cycle:
            logger.info(
                "Swap path: hit per-cycle cap (%d) -- skipping remaining %d blocked candidates",
                max_per_cycle, len(sector_blocked) - swaps_fired,
            )
            break

        cand_sector = cand.get("sector") or "Unknown"
        cand_score = float(cand.get("final_score") or 0.0)
        sector_holdings = holdings_by_sector.get(cand_sector, [])

        # Skip if no eligible holding to displace (already swapped, or sector
        # empty post-sell). Walk the sorted list and pick the lowest-score
        # holding not already swapped this cycle.
        weakest = None
        for h in sector_holdings:
            if h["ticker"] in swapped_tickers:
                continue
            weakest = h
            break
        if weakest is None:
            continue

        holding_score = weakest["final_score"]
        # phase-60.2 (AW-5) CORRECTED COMMENT: final_score does NOT live in
        # [0,1] -- the lite path emits 1-10 integers and the full path a 0-10
        # weighted score (the away-week rows are 1-10). The old premise drove
        # a 0.01 epsilon denominator, so a sentinel-0.0 holding compared at
        # ~70,000% delta vs the 25% bar and ANY candidate cleared it by scale
        # accident. settings.paper_swap_min_delta_pct's own description has
        # always documented max(abs(holding_score), 1.0).
        # Flag ON: the documented 1.0 clamp on the true 1-10 scale (7-vs-5 =
        # 40%, 6-vs-5 = 20% -- true relative deltas; the 25.0 bar UNCHANGED,
        # widening it would be the 53.1/55.3-rejected band family).
        # Flag OFF: the historical 0.01 epsilon, byte-identical pre-60.2.
        denom = max(abs(holding_score), 1.0 if _churn_fix_on else 0.01)
        delta_pct = ((cand_score - holding_score) / denom) * 100.0

        if delta_pct < min_delta:
            logger.info(
                "Swap skip %s -> %s: delta=%.1f%% below threshold %.1f%% (cand_score=%.3f holding_score=%.3f)",
                weakest["ticker"], cand["ticker"], delta_pct, min_delta,
                cand_score, holding_score,
            )
            continue

        # Projected sector NAV-pct check: removing weakest, adding cand at its
        # target position size. Conservative: use the same position_pct the
        # buy-loop would use, capped at available_cash equivalent.
        #
        # Edge case (north-star aligned): when the sector is ALREADY over the
        # NAV-pct cap (legacy 8-Tech-holding portfolios), block only swaps
        # that WORSEN the exposure. Swaps that REDUCE or hold the exposure
        # constant are strictly an improvement and should be allowed -- per
        # the testing-phase mandate "default to firing, not gating, when risk
        # caps permit". Idling on a worse composition because we can't fully
        # cure the cap in one swap is the wrong default.
        # phase-49.1: runtime-overridable (same key as the buy-loop check above).
        max_sector_nav_pct = float(
            risk_overrides.get_effective(
                "paper_max_per_sector_nav_pct", getattr(settings, "paper_max_per_sector_nav_pct", 0.0)
            ) or 0.0
        )
        if max_sector_nav_pct > 0 and nav > 0:
            position_pct = float(cand.get("position_pct") or 10.0)
            buy_amount = nav * (position_pct / 100.0)
            existing = sector_market_values.get(cand_sector, 0.0)
            projected_sector_value = existing - weakest["market_value"] + buy_amount
            projected_pct = (projected_sector_value / nav) * 100.0
            existing_pct = (existing / nav) * 100.0
            # Block only if projected exceeds cap AND projected exceeds the
            # pre-swap exposure (i.e., the swap actually WORSENS the cap
            # breach). When existing_pct > cap and projected_pct <= existing_pct,
            # the swap is a strict reduction toward compliance.
            if projected_pct > max_sector_nav_pct and projected_pct > existing_pct:
                logger.info(
                    "Swap skip %s -> %s: projected sector %s NAV-pct %.2f%% > cap %.2f%% AND > existing %.2f%% (worsens breach)",
                    weakest["ticker"], cand["ticker"], cand_sector,
                    projected_pct, max_sector_nav_pct, existing_pct,
                )
                continue

        # Emit the swap pair: SELL first, then BUY (sell-first-then-buy invariant).
        swap_orders.append(TradeOrder(
            ticker=weakest["ticker"],
            action="SELL",
            reason="swap_for_higher_conviction",
            price=weakest.get("current_price"),
        ))
        position_pct = cand.get("position_pct") or 10.0
        buy_amount = nav * (float(position_pct) / 100.0)
        swap_orders.append(TradeOrder(
            ticker=cand["ticker"],
            action="BUY",
            amount_usd=round(buy_amount, 2),
            reason="swap_buy",
            analysis_id=cand.get("analysis_id", ""),
            risk_judge_decision=cand.get("risk_judge_decision", ""),
            stop_loss_price=cand.get("stop_loss_price"),
            risk_judge_position_pct=cand.get("position_pct"),
            price=cand.get("price"),
            price_at_analysis=cand.get("price"),
            signals=cand.get("signals", []),
            sector=cand.get("sector", ""),
            market=markets.market_for_symbol(cand["ticker"]),  # phase-50.3
            factor_loadings=cand.get("factor_loadings"),
            # phase-61.2 (criterion 5): swap BUYs carry the analysis verdict too.
            analysis_recommendation=cand.get("recommendation", ""),
        ))
        swapped_tickers.add(weakest["ticker"])
        # Update sector_market_values so subsequent swap-checks see the
        # updated composition.
        sector_market_values[cand_sector] = (
            sector_market_values.get(cand_sector, 0.0)
            - weakest["market_value"] + buy_amount
        )
        swaps_fired += 1
        logger.info(
            "Swap fired (%d/%d): SELL %s (score=%.3f) -> BUY %s (score=%.3f) delta=%.1f%%",
            swaps_fired, max_per_cycle,
            weakest["ticker"], holding_score,
            cand["ticker"], cand_score,
            delta_pct,
        )

    return swap_orders


def _extract_position_pct(risk_assessment: dict, analysis: dict) -> Optional[float]:
    """Extract recommended position % from risk assessment."""
    # Try risk_judge output
    pct = risk_assessment.get("recommended_position_pct")
    if pct:
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    # Fall back to analysis-level field
    pct = analysis.get("risk_judge_position_pct")
    if pct:
        try:
            return float(pct)
        except (ValueError, TypeError):
            pass
    return None


def _extract_stop_loss(
    risk_assessment: dict,
    analysis: dict,
    settings: Optional["Settings"] = None,
) -> Optional[float]:
    """Extract stop loss price from risk assessment.

    Resolution order:
      1. Explicit `risk_assessment.risk_limits.stop_loss` (absolute price)
      2. Encoded `risk_assessment.risk_limits.stop_loss_pct` (% below entry)
      3. phase-23.1.8 fallback: `settings.paper_default_stop_loss_pct` * entry
         price. Activates on lite-Claude-analyzer BUYs where risk_assessment
         is just `{"reason": "..."}` and risk_limits is absent.

    Returns None when no price_at_analysis is available (cannot derive any
    stop without an entry reference).
    """
    limits = risk_assessment.get("risk_limits", {})
    if isinstance(limits, dict):
        stop = limits.get("stop_loss")
        if stop:
            try:
                return float(stop)
            except (ValueError, TypeError):
                pass
    # Maybe encoded as % below entry price
    stop_pct = limits.get("stop_loss_pct") if isinstance(limits, dict) else None
    price = analysis.get("price_at_analysis")
    if stop_pct and price:
        try:
            return float(price) * (1 - float(stop_pct) / 100.0)
        except (ValueError, TypeError):
            pass
    # phase-23.1.8: settings-driven default (Option B from research brief)
    if settings is not None and price:
        default_pct = getattr(settings, "paper_default_stop_loss_pct", None)
        if default_pct is not None:
            try:
                return float(price) * (1 - float(default_pct) / 100.0)
            except (ValueError, TypeError):
                pass
    return None
