"""
Autonomous Paper Trading Loop — daily cycle orchestrator.

Screen → Analyze → Decide → Trade → Snapshot → Learn.
Designed to run as an APScheduler cron job.
"""

import asyncio
import hashlib
import json

from backend.utils import json_io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.agents.meta_coordinator import MetaCoordinator
from backend.agents.orchestrator import AnalysisOrchestrator
from backend.config.settings import Settings, get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.paper_trader import PaperTrader
from backend.services.portfolio_manager import decide_trades
from backend.tools.screener import screen_universe, rank_candidates, get_sp500_tickers, get_russell1000_tickers

logger = logging.getLogger(__name__)

# Path to optimizer best parameters
_OPTIMIZER_BEST_PATH = Path(__file__).parent.parent / "backtest" / "experiments" / "optimizer_best.json"


def load_best_params() -> dict:
    """Load the best backtest parameters from optimizer_best.json."""
    if not _OPTIMIZER_BEST_PATH.exists():
        logger.warning("optimizer_best.json not found, using defaults")
        return {}
    with open(_OPTIMIZER_BEST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    params = data.get("params", data)
    sharpe = data.get("sharpe", "?")
    logger.info(f"Loaded best params (Sharpe {sharpe}): {list(params.keys())}")
    return params


def load_promoted_params(bq: BigQueryClient) -> dict:
    """phase-25.B3: prefer the latest BQ-promoted strategy params over the
    local optimizer_best.json snapshot. Closes phase-24.3 F-6 -- before
    this fix the daily cycle could not pick up newly promoted strategies
    written by `backend/autoresearch/friday_promotion.py`.

    Three-tier fallback:
      1. BQ row found with non-empty params -> return those params + log.
      2. BQ returns None or empty params -> fall back to load_best_params().
      3. BQ raises (network / table missing / etc.) -> fall back to
         load_best_params(). Never raises out of this function.
    """
    try:
        row = bq.get_latest_promoted_strategy()
        if row and row.get("params"):
            logger.info(
                "Loaded promoted params (DSR %s week=%s): %s",
                row.get("dsr", "?"),
                row.get("week_iso", "?"),
                list((row.get("params") or {}).keys()),
            )
            return row["params"]
        logger.info("No active promoted strategy in BQ, falling back to optimizer_best")
    except Exception as exc:
        logger.warning(
            "Promoted strategy BQ unavailable, falling back to optimizer_best: %s",
            exc,
        )
    return load_best_params()


# Module-level state
_running = False
_last_run: Optional[str] = None
_last_result: Optional[dict] = None
_coordinator = MetaCoordinator()

# phase-26.1: per-session LLM cost ceiling (local mirror of Agent SDK's
# max_budget_usd pattern). autonomous_loop drives client.messages.create()
# and llm_client.generate_content() directly, not via Managed Agents or
# Agent SDK sessions, so Anthropic's Task Budgets API is not wirable --
# enforcement must be application-level. Reset to 0 at start of every
# cycle; raises BudgetBreachError when cumulative cost crosses ceiling.
# Env-var override: PYFINAGENT_SESSION_BUDGET_USD=<float>.
_SESSION_BUDGET_USD: float = float(os.getenv("PYFINAGENT_SESSION_BUDGET_USD", "1.0"))
_session_cost: float = 0.0
_current_cycle_id: Optional[str] = None


def _check_session_budget(stage: str = "pre_call") -> None:
    """phase-26.1: raise BudgetBreachError if cumulative session LLM cost
    has reached the per-cycle ceiling. Called before LLM-heavy steps.
    Lazy-imports BudgetBreachError to avoid module-load coupling."""
    if _session_cost >= _SESSION_BUDGET_USD:
        from backend.agents.llm_client import BudgetBreachError
        raise BudgetBreachError(
            f"session_budget_breach: cumulative ${_session_cost:.4f} "
            f">= ceiling ${_SESSION_BUDGET_USD:.4f} (stage={stage}, "
            f"cycle_id={_current_cycle_id})"
        )


def _add_session_cost(usd: float) -> None:
    """phase-26.1: mutate the module-level session cost accumulator."""
    global _session_cost
    _session_cost += float(usd)


def get_current_cycle_id() -> Optional[str]:
    """phase-26.1: exported helper for log_llm_call to stamp BQ rows."""
    return _current_cycle_id


def get_session_cost_usd() -> float:
    """phase-26.1: exported helper for log_llm_call to stamp BQ rows."""
    return _session_cost


async def run_daily_cycle(settings: Optional[Settings] = None, dry_run: bool = False) -> dict:
    """
    Execute one full paper trading cycle:
    1. Screen universe (free)
    2. Analyze top candidates (lite mode)
    3. Re-evaluate holdings due for refresh
    4. Decide trades
    5. Execute trades
    6. Mark to market
    7. Save snapshot
    8. Learn from closed trades

    Returns summary dict.

    dry_run=True short-circuits the cycle: stamps _last_run and returns
    ok without running any LLM / BQ / trade work. Used by the phase-4.6
    smoketest; not for production use.
    """
    global _running, _last_run, _last_result, _session_cost, _current_cycle_id

    # phase-38.6.1: replace in-process _running guard with file-based
    # cycle_lock (handoff/.autonomous_loop.lock). SIGKILL/crash mid-cycle
    # no longer leaves stale state; flock auto-released on process death;
    # next startup's clean_stale_lock cleans pidfile. The in-process
    # _running flag is kept for UI/api status surface (get_loop_status)
    # but the LOCK is the source of truth for re-entrancy.
    from backend.services.cycle_lock import acquire as _cycle_lock_acquire, CycleLockError

    if _running:
        logger.warning("Paper trading cycle already running, skipping")
        return {"status": "skipped", "reason": "already_running"}

    if dry_run:
        _last_run = datetime.now(timezone.utc).isoformat()
        _last_result = {"status": "ok", "dry_run": True, "timestamp": _last_run}
        logger.info("Paper trading dry-run: stamped _last_run, no real work performed")
        return _last_result

    # File-based lock acquire (raises if a live cycle holds it OR if a
    # stale lock cant be cleaned). We acquire BEFORE setting _running so
    # the cross-process guard runs first.
    _cycle_id_for_lock = _current_cycle_id or f"cycle-{int(datetime.now(timezone.utc).timestamp())}"
    try:
        _lock_cm = _cycle_lock_acquire(_cycle_id_for_lock)
        _lock_cm.__enter__()
    except CycleLockError as _lock_exc:
        logger.warning("Paper trading cycle already running (file-lock), skipping: %r", _lock_exc)
        return {"status": "skipped", "reason": "already_running_file_lock"}

    _running = True
    settings = settings or get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)
    total_analysis_cost = 0.0
    trades_executed = 0
    # phase-30.3: hoist closed_tickers to cycle-top so the stop-loss-
    # enforcement step can append to it BEFORE the execute-trades step
    # runs. Without this hoist the variable only exists inside the
    # execute step (the old initialization site), so stop-loss-triggered
    # closes never reach _learn_from_closed_trades.
    # Closes phase-30.0 Stage 12 + P1-3 (empty agent_memories table).
    # Researcher Option A: only timeout-safe init site (the cycle body is
    # wrapped in `async with asyncio.timeout(...)` -- a timeout mid-cycle
    # could otherwise leave closed_tickers undefined at summary-serialize
    # time in the finally block).
    closed_tickers: list[str] = []
    summary = {"status": "running", "steps": []}

    # phase-26.1: reset per-session cost accumulator at cycle start.
    _session_cost = 0.0

    # 4.5.8 cycle health: start-of-cycle heartbeat + history row.
    from backend.services.cycle_health import get_log as _cycle_log
    import uuid as _uuid
    _cycle_id = str(_uuid.uuid4())[:8]
    _cycle_started_at = _cycle_log().record_cycle_start(_cycle_id)
    summary["cycle_id"] = _cycle_id
    summary["started_at"] = _cycle_started_at

    # phase-26.1: propagate cycle_id to module state so log_llm_call can
    # stamp BQ rows with cycle_id + session_cost_usd. Reset to None in
    # the finally block at end of cycle.
    _current_cycle_id = _cycle_id
    summary["session_budget_usd"] = _SESSION_BUDGET_USD

    # phase-25.B3: prefer the latest BQ-promoted strategy params; falls back
    # to optimizer_best.json if BQ has nothing active or is unavailable.
    best_params = load_promoted_params(bq)
    if best_params:
        summary["best_params_sharpe"] = best_params.get("sharpe", "?")
        summary["strategy_params"] = {
            k: best_params[k] for k in ["tp_pct", "sl_pct", "holding_days"]
            if k in best_params
        }

    _cycle_timeout = float(getattr(settings, "paper_cycle_max_seconds", 1800.0))
    try:
        # phase-23.2.18: outer asyncio.timeout ceiling so a stuck
        # asyncio.to_thread (yfinance/BQ blocking call inside a worker
        # thread the asyncio side cannot cancel) cannot hang the cycle
        # indefinitely. On TimeoutError, status is recorded and the
        # operator is alerted in the post-finally block.
        async with asyncio.timeout(_cycle_timeout):
            # ── Step 1: Screen universe (free) ───────────────────────
            logger.info("Paper trading: Step 1 -- Screening universe")
            summary["steps"].append("screening")

            regime = None
            if getattr(settings, "macro_regime_filter_enabled", False):
                try:
                    from backend.services.macro_regime import compute_macro_regime
                    regime = await compute_macro_regime()
                    logger.info(
                        "Macro regime: %s conviction=%.2f mult=%.2f",
                        regime.regime, regime.conviction, regime.conviction_multiplier,
                    )
                    summary["macro_regime"] = regime.regime
                    summary["macro_regime_multiplier"] = regime.conviction_multiplier
                except Exception as e:
                    logger.warning("Macro regime fetch failed (non-fatal): %s", e)

            pead_signals = {}
            if getattr(settings, "pead_signal_enabled", False):
                try:
                    from backend.services.pead_signal import fetch_pead_signals_for_recent_reporters
                    pead_signals = await fetch_pead_signals_for_recent_reporters()
                    logger.info("PEAD signals fetched: %d tickers", len(pead_signals))
                    summary["pead_tickers_scored"] = len(pead_signals)
                except Exception as e:
                    logger.warning("PEAD signal fetch failed (non-fatal): %s", e)

            news_signals = {}
            if getattr(settings, "news_screen_enabled", False):
                try:
                    from backend.services.news_screen import fetch_news_signals
                    news_signals = await fetch_news_signals(
                        max_headlines=getattr(settings, "news_screen_max_headlines", 100),
                    )
                    logger.info("News screen produced %d ticker signals", len(news_signals))
                    summary["news_tickers_scored"] = len(news_signals)
                except Exception as e:
                    logger.warning("News screen failed (non-fatal): %s", e)

            sector_events = {}
            if getattr(settings, "sector_calendars_enabled", False):
                try:
                    from backend.services.sector_calendars import fetch_sector_events
                    sector_events = await fetch_sector_events()
                    logger.info("Sector calendars: %d events", len(sector_events))
                    summary["sector_events"] = len(sector_events)
                except Exception as e:
                    logger.warning("Sector calendars failed (non-fatal): %s", e)

            # phase-28.12: sector-ETF momentum overlay (top-3 rotation boost)
            sector_momentum_ranks = {}
            if getattr(settings, "sector_momentum_enabled", False):
                try:
                    from backend.services.sector_momentum import fetch_sector_momentum_ranks
                    sector_momentum_ranks = await fetch_sector_momentum_ranks(
                        cache_hours=getattr(settings, "sector_momentum_cache_hours", 24),
                        lookback_months=getattr(settings, "sector_momentum_lookback_months", 12),
                        top_n=getattr(settings, "sector_momentum_top_n", 3),
                        boost_top=getattr(settings, "sector_momentum_boost_top", 1.10),
                        boost_leader=getattr(settings, "sector_momentum_boost_leader", 1.15),
                    )
                    logger.info("sector_momentum ranks loaded: %d sectors", len(sector_momentum_ranks))
                    summary["sector_momentum_top"] = [
                        r.sector for r in sector_momentum_ranks.values() if r.rank <= 3
                    ]
                except Exception as e:
                    logger.warning("sector_momentum fetch failed (non-fatal): %s", e)

            # phase-28.5: short-interest exclusion lookup (FINRA bimonthly CSV preferred, yfinance fallback)
            short_interest_lookup: dict[str, float] = {}
            if getattr(settings, "short_interest_filter_enabled", False):
                try:
                    from backend.services.short_interest import fetch_short_interest_lookup
                    short_interest_lookup = await fetch_short_interest_lookup()
                    logger.info(
                        "Short-interest lookup loaded: %d tickers (threshold=%.3f)",
                        len(short_interest_lookup), settings.short_interest_threshold,
                    )
                    summary["short_interest_tickers_loaded"] = len(short_interest_lookup)
                except Exception as e:
                    logger.warning("Short-interest lookup failed (non-fatal): %s", e)

            # phase-28.8: optionally use Russell-1000 universe instead of S&P 500
            # (addresses Sandisk/SNDK spinoff miss). Default OFF.
            if getattr(settings, "russell1000_universe_enabled", False):
                try:
                    universe = get_russell1000_tickers()
                    summary["universe_source"] = "russell1000"
                    summary["universe_size"] = len(universe)
                    logger.info("phase-28.8: using Russell-1000 universe (%d tickers)", len(universe))
                except Exception as e:
                    logger.warning("Russell-1000 fetch failed (%s); falling back to SP500", e)
                    universe = None
            else:
                universe = None

            screen_data = screen_universe(
                tickers=universe,
                period="6mo",
                short_interest_lookup=short_interest_lookup or None,
                short_interest_threshold=getattr(settings, "short_interest_threshold", 0.10),
            )

            # phase-40.8.1 (P3): producer for the dormant FF3 cap.
            # Default-OFF: behavior is byte-identical to today until operator
            # flips settings.enable_factor_loadings AND populates a real FF3
            # cache (phase-40.8.2 follow-up). Stubbed factor returns this
            # cycle so the wiring is tested end-to-end.
            if getattr(settings, "enable_factor_loadings", False) and screen_data:
                try:
                    from backend.services.factor_loadings import compute_candidate_loadings
                    price_histories = {
                        s["ticker"]: s.get("price_history", [])
                        for s in screen_data if s.get("ticker")
                    }
                    compute_candidate_loadings(screen_data, price_histories, window_days=60)
                except Exception as e:
                    logger.warning("phase-40.8.1: factor_loadings producer failed (fail-open): %r", e)

            # phase-28.16: M&A pre-announcement aggregator (Legs 1+2 from 28.9+28.10; Leg 3 stub).
            # Pure compute — no extra fetches; reuses options_surge + insider signals already
            # collected by phase-28.9 + 28.10 when their flags are on. Default OFF.
            ma_preannounce_signals = {}
            if getattr(settings, "ma_preannounce_enabled", False) and screen_data:
                try:
                    from backend.services.ma_preannounce_screen import compute_ma_preannounce_signals
                    cand_tickers = [s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n] if s.get("ticker")]
                    ma_preannounce_signals = compute_ma_preannounce_signals(
                        cand_tickers,
                        options_surge_signals=options_surge_signals or {},
                        insider_signals=insider_signals or {},
                        schedule_13d_signals={},  # Leg 3 stub; phase-28.16-followup
                        strong_boost=getattr(settings, "ma_preannounce_strong_boost", 0.10),
                        moderate_boost=getattr(settings, "ma_preannounce_moderate_boost", 0.05),
                    )
                    summary["ma_preannounce_flagged"] = len(ma_preannounce_signals)
                except Exception as e:
                    logger.warning("ma_preannounce_screen compute failed (non-fatal): %s", e)

            # phase-28.17: peer-correlation laggard catch-up. Fetch analyst+market_cap via
            # yfinance.info for top candidates, compute pure-function signals.
            peer_leadlag_signals = {}
            if getattr(settings, "peer_leadlag_enabled", False) and screen_data:
                try:
                    import yfinance as yf
                    from backend.services.peer_leadlag_screen import compute_peer_leadlag_signals
                    target_tickers = [s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n] if s.get("ticker")]
                    lookup: dict[str, dict] = {}
                    for t in target_tickers:
                        try:
                            info = await asyncio.to_thread(lambda x=t: yf.Ticker(x).info or {})
                            lookup[t.upper()] = {
                                "analyst_count": int(info.get("numberOfAnalystOpinions") or 0),
                                "market_cap": float(info.get("marketCap") or 0),
                            }
                        except Exception:
                            continue
                    peer_leadlag_signals = compute_peer_leadlag_signals(
                        screen_data,
                        lookup,
                        leader_threshold=getattr(settings, "peer_leadlag_leader_threshold", 10.0),
                        laggard_threshold=getattr(settings, "peer_leadlag_laggard_threshold", 2.0),
                        max_analyst_count=getattr(settings, "peer_leadlag_min_analyst_filter", 5),
                        min_market_cap_usd=getattr(settings, "peer_leadlag_min_market_cap_usd", 2_000_000_000.0),
                        boost=getattr(settings, "peer_leadlag_boost", 0.08),
                    )
                    summary["peer_leadlag_qualifying"] = len(peer_leadlag_signals)
                except Exception as e:
                    logger.warning("peer_leadlag fetch/compute failed (non-fatal): %s", e)

            # phase-28.14: defense/war-stocks reference case (GPR + XAR AND-gate, cycle-level).
            # Boost defense-list tickers when both gates fire. Default OFF.
            defense_signal_obj = None
            if getattr(settings, "defense_signal_enabled", False):
                try:
                    from backend.services.defense_signal import fetch_defense_trigger
                    defense_signal_obj = await fetch_defense_trigger(
                        defense_tickers_csv=getattr(settings, "defense_tickers", ""),
                        xar_window_days=getattr(settings, "defense_xar_window_days", 5),
                        xar_min_momentum=getattr(settings, "defense_xar_min_momentum", 0.0),
                        boost=getattr(settings, "defense_boost", 0.05),
                        gpr_quantile=getattr(settings, "gpr_signal_quantile", 0.90),
                        gpr_cache_hours=getattr(settings, "gpr_signal_cache_hours", 24),
                        pledge_keywords_csv=getattr(settings, "defense_budget_pledge_keywords", ""),
                    )
                    summary["defense_signal_triggered"] = bool(defense_signal_obj.triggered)
                    summary["defense_signal_xar_5d"] = defense_signal_obj.xar_5d_momentum
                except Exception as e:
                    logger.warning("defense_signal fetch failed (non-fatal): %s", e)

            # phase-28.15: social media velocity overlay (Alpha Vantage NEWS_SENTIMENT
            # cross-source — Reddit/Twitter/StockTwits/blogs). Pre-rally signal per
            # supplement Gap 2 + DNUT July 2025 case. Default OFF.
            social_velocity_signals = {}
            if getattr(settings, "social_velocity_enabled", False) and screen_data:
                try:
                    from backend.services.social_velocity_screen import fetch_social_velocity_signals
                    candidate_tickers_for_social = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    social_velocity_signals = await fetch_social_velocity_signals(
                        candidate_tickers_for_social,
                        min_threshold=getattr(settings, "social_velocity_min_threshold", 0.10),
                        min_mentions=getattr(settings, "social_velocity_min_mentions", 3),
                        strong_threshold=getattr(settings, "social_velocity_strong_threshold", 0.20),
                        strong_boost=getattr(settings, "social_velocity_strong_boost", 0.06),
                        moderate_boost=getattr(settings, "social_velocity_moderate_boost", 0.03),
                    )
                    logger.info(
                        "social_velocity_screen: %d/%d candidates flagged",
                        len(social_velocity_signals), len(candidate_tickers_for_social),
                    )
                    summary["social_velocity_flagged"] = len(social_velocity_signals)
                except Exception as e:
                    logger.warning("social_velocity_screen fetch failed (non-fatal): %s", e)

            # phase-28.13: firm-level GPR exposure DEFENSIVE filter (Fed 2025 R²=0.23
            # contemporaneous only; NOT forward alpha). LLM-classify per-firm 4-tier
            # from earnings-call transcripts. Default OFF.
            gpr_exposure_signals = {}
            if getattr(settings, "call_transcript_gpr_enabled", False) and screen_data:
                try:
                    from backend.services.call_transcript_gpr import fetch_gpr_exposure_signals
                    candidate_tickers_for_gpr = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    gpr_exposure_signals = await fetch_gpr_exposure_signals(
                        candidate_tickers_for_gpr,
                        model=getattr(settings, "call_transcript_gpr_model", "claude-haiku-4-5"),
                        bucket_name=getattr(settings, "gcs_bucket_name", ""),
                    )
                    logger.info(
                        "call_transcript_gpr: %d/%d candidates classified",
                        len(gpr_exposure_signals), len(candidate_tickers_for_gpr),
                    )
                    summary["call_transcript_gpr_classified"] = len(gpr_exposure_signals)
                except Exception as e:
                    logger.warning("call_transcript_gpr fetch failed (non-fatal): %s", e)

            # phase-28.11: management-outlook narrative overlay (MVP proxy for canonical
            # analyst Strategic Outlook signal — which needs paid data). 8-K Exhibit 99 +
            # Claude Haiku. Default OFF. Per-cycle LLM cost <$0.10 for ~10 recent reporters.
            narrative_signals = {}
            if getattr(settings, "analyst_narrative_enabled", False) and screen_data:
                try:
                    from backend.services.analyst_narrative_scorer import fetch_narrative_signals
                    candidate_tickers_for_narrative = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    narrative_signals = await fetch_narrative_signals(
                        candidate_tickers_for_narrative,
                        model=getattr(settings, "analyst_narrative_model", "claude-haiku-4-5"),
                        strong_threshold=getattr(settings, "analyst_narrative_strong_threshold", 0.70),
                        weak_threshold=getattr(settings, "analyst_narrative_weak_threshold", 0.30),
                        strong_boost=getattr(settings, "analyst_narrative_strong_boost", 0.05),
                        moderate_boost=getattr(settings, "analyst_narrative_moderate_boost", 0.025),
                    )
                    logger.info(
                        "analyst_narrative_scorer: %d/%d candidates scored",
                        len(narrative_signals), len(candidate_tickers_for_narrative),
                    )
                    summary["analyst_narrative_scored"] = len(narrative_signals)
                except Exception as e:
                    logger.warning("analyst_narrative_scorer fetch failed (non-fatal): %s", e)

            # phase-28.10: opportunistic insider-buying overlay. Fetched AFTER first-pass
            # screen so SEC EDGAR cost is bounded by candidate-set size. Default OFF.
            insider_signals = {}
            if getattr(settings, "insider_signal_screen_enabled", False) and screen_data:
                try:
                    from backend.services.insider_signal_screen import fetch_insider_signals
                    candidate_tickers_for_insider = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    insider_signals = await fetch_insider_signals(
                        candidate_tickers_for_insider,
                        lookback_months=getattr(settings, "insider_lookback_history_months", 48),
                        window_days=getattr(settings, "insider_signal_window_days", 30),
                        min_usd=getattr(settings, "insider_signal_min_aggregate_usd", 500_000.0),
                        strong_usd=getattr(settings, "insider_signal_strong_aggregate_usd", 2_000_000.0),
                        strong_boost=getattr(settings, "insider_strong_boost", 0.07),
                        moderate_boost=getattr(settings, "insider_moderate_boost", 0.04),
                    )
                    logger.info(
                        "insider_signal_screen: %d/%d candidates flagged",
                        len(insider_signals), len(candidate_tickers_for_insider),
                    )
                    summary["insider_signals_flagged"] = len(insider_signals)
                except Exception as e:
                    logger.warning("insider_signal_screen fetch failed (non-fatal): %s", e)

            # phase-28.9: options-flow OI-surge overlay. Fetched AFTER first-pass screen
            # so per-ticker yfinance.option_chain cost is bounded by candidate-set size
            # (top 2*paper_screen_top_n ~= 20 tickers), not full S&P 500. Default OFF.
            options_surge_signals = {}
            if getattr(settings, "options_flow_screen_enabled", False) and screen_data:
                try:
                    from backend.services.options_flow_screen import fetch_oi_surge_signals
                    candidate_tickers_for_options = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    options_surge_signals = await fetch_oi_surge_signals(
                        candidate_tickers_for_options,
                        otm_threshold=getattr(settings, "options_otm_threshold", 1.01),
                        dte_min=getattr(settings, "options_dte_min", 2),
                        dte_max=getattr(settings, "options_dte_max", 45),
                        vol_avg_mult=getattr(settings, "options_vol_avg_multiplier", 5.0),
                        vol_oi_mult=getattr(settings, "options_vol_oi_multiplier", 3.0),
                        strong_boost=getattr(settings, "options_strong_boost", 0.06),
                        moderate_boost=getattr(settings, "options_moderate_boost", 0.03),
                    )
                    logger.info(
                        "options_flow_screen signals: %d/%d candidates flagged",
                        len(options_surge_signals), len(candidate_tickers_for_options),
                    )
                    summary["options_surge_flagged"] = len(options_surge_signals)
                except Exception as e:
                    logger.warning("options_flow_screen fetch failed (non-fatal): %s", e)

            # phase-28.1: analyst EPS revision-breadth overlay. Fetched AFTER first-pass
            # screen so per-ticker cost is bounded by candidate-set size (typically <=30),
            # not full S&P 500. Default-OFF; non-fatal failure preserves cycle.
            revision_signals = {}
            if getattr(settings, "analyst_revisions_enabled", False) and screen_data:
                try:
                    from backend.services.analyst_revisions import fetch_revision_signals
                    candidate_tickers = [
                        s["ticker"] for s in screen_data[: 2 * settings.paper_screen_top_n]
                        if s.get("ticker")
                    ]
                    revision_signals = await fetch_revision_signals(
                        candidate_tickers,
                        lookback_days=getattr(settings, "analyst_revisions_lookback_days", 100),
                        min_analysts=getattr(settings, "analyst_revisions_min_analysts", 3),
                    )
                    logger.info(
                        "analyst_revisions signals: %d/%d candidates scored",
                        len(revision_signals), len(candidate_tickers),
                    )
                    summary["analyst_revisions_scored"] = len(revision_signals)
                except Exception as e:
                    logger.warning("analyst_revisions fetch failed (non-fatal): %s", e)

            candidates = rank_candidates(
                screen_data,
                top_n=settings.paper_screen_top_n,
                regime=regime,
                pead_signals=pead_signals or None,
                news_signals=news_signals or None,
                sector_events=sector_events or None,
                revision_signals=revision_signals or None,
                sector_neutral=getattr(settings, "sector_neutral_momentum_enabled", False),
                sector_neutral_min_group_size=getattr(settings, "sector_neutral_min_group_size", 3),
                sector_momentum_ranks=sector_momentum_ranks or None,
                multidim_momentum=getattr(settings, "multidim_momentum_enabled", False),
                multidim_weights={
                    "price":    getattr(settings, "multidim_momentum_weight_price", 0.35),
                    "52w_high": getattr(settings, "multidim_momentum_weight_52w_high", 0.25),
                    "sue":      getattr(settings, "multidim_momentum_weight_sue", 0.20),
                    "sector":   getattr(settings, "multidim_momentum_weight_sector", 0.20),
                },
                options_surge_signals=options_surge_signals or None,
                insider_signals=insider_signals or None,
                narrative_signals=narrative_signals or None,
                gpr_exposure_signals=gpr_exposure_signals or None,
                social_velocity_signals=social_velocity_signals or None,
                defense_signal=defense_signal_obj,
                peer_leadlag_signals=peer_leadlag_signals or None,
                ma_preannounce_signals=ma_preannounce_signals or None,
                gpr_exposure_config={
                    "exempt_sectors_csv": getattr(settings, "call_transcript_gpr_exempt_sectors", "Industrials,Energy"),
                    "high_penalty": getattr(settings, "call_transcript_gpr_high_penalty", 0.97),
                },
            )

            # phase-23.1.13: enrich top-N candidates with GICS sector via the
            # already-cached ticker_meta endpoint (BQ-first / yfinance fallback).
            # `_fetch_ticker_meta` is sync; wrap in to_thread. Cost: at most 10-30
            # tickers; 24h cache per ticker means subsequent cycles incur near zero
            # latency. Without this enrichment, decide_trades sees `sector=None` on
            # every candidate and the new sector cap is a no-op.
            if candidates:
                try:
                    from backend.api.paper_trading import _fetch_ticker_meta
                    top_tickers = [c["ticker"] for c in candidates if c.get("ticker")]
                    meta_response = await asyncio.to_thread(
                        _fetch_ticker_meta, top_tickers, settings, bq,
                    )
                    meta_map = (meta_response or {}).get("meta", {})
                    for c in candidates:
                        info = meta_map.get(c.get("ticker"), {})
                        sector = info.get("sector") or ""
                        if sector:
                            c["sector"] = sector
                        company = info.get("company_name")
                        if company and not c.get("company_name"):
                            c["company_name"] = company
                except Exception as e:
                    logger.warning("Ticker meta enrichment failed (non-fatal): %s", e)

            if getattr(settings, "meta_scorer_enabled", False):
                try:
                    from backend.services.meta_scorer import meta_score_candidates
                    candidates = await meta_score_candidates(candidates, regime=regime)
                    if candidates:
                        summary["meta_scored_top_conviction"] = candidates[0].get("conviction_score")
                    logger.info(
                        "Meta-scorer ranked %d candidates (top conviction=%s)",
                        len(candidates),
                        candidates[0].get("conviction_score") if candidates else None,
                    )
                except Exception as e:
                    logger.warning("Meta-scorer failed (non-fatal): %s", e)
            summary["screened"] = len(screen_data)
            summary["candidates"] = len(candidates)

            # ── Step 2: Filter candidates ────────────────────────────
            # phase-23.1.23: wrap blocking trader.* calls in asyncio.to_thread so
            # they don't freeze the asyncio event loop. mark_to_market in
            # particular does ~14 positions x (yfinance + 2 BQ DML) = 42 blocking
            # network ops which previously blocked /api/health past the watchdog
            # threshold and got the backend kickstart-killed daily.
            positions = await asyncio.to_thread(trader.get_positions)
            held_tickers = {p["ticker"] for p in positions}
            new_candidates = [c for c in candidates if c["ticker"] not in held_tickers]
            analyze_tickers = [c["ticker"] for c in new_candidates[:settings.paper_analyze_top_n]]

            # Determine holdings due for re-evaluation
            reeval_tickers = []
            now = datetime.now(timezone.utc)
            for pos in positions:
                last_date = pos.get("last_analysis_date", "")
                if not last_date:
                    reeval_tickers.append(pos["ticker"])
                    continue
                try:
                    last_dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
                    days_since = (now - last_dt).days
                    if days_since >= settings.paper_reeval_frequency_days:
                        reeval_tickers.append(pos["ticker"])
                except (ValueError, TypeError):
                    reeval_tickers.append(pos["ticker"])

            summary["new_to_analyze"] = len(analyze_tickers)
            summary["reeval_tickers"] = len(reeval_tickers)

            # ── Step 3: Analyze candidates ───────────────────────────
            # phase-23.1.12: removed the hardcoded `settings.lite_mode = True` override.
            # The operator's lite_mode setting is now respected. Cost containment is
            # enforced by `paper_max_daily_cost_usd` cap (the loop break below); the
            # full Gemini orchestrator path is more expensive but the cap remains
            # the circuit breaker.
            logger.info(
                "Paper trading: Step 3 -- Analyzing %d new + %d re-evals (lite_mode=%s)",
                len(analyze_tickers), len(reeval_tickers), settings.lite_mode,
            )
            summary["steps"].append("analyzing")

            # phase-27.5.1 + 27.6.5: parallelize per-ticker analysis with
            # PER-PROVIDER bounded concurrency. Gemini AI Studio paid-tier
            # RPM tolerates 8 concurrent (Gemini cycle #8 confirmed). Claude
            # tier-1 RPM is tighter (~50 input, ~10 output per minute) and
            # cycle #10 hit `HTTP/1.1 429 Too Many Requests` from
            # api.anthropic.com on concurrency=8 — so we cap Claude at 3.
            # Detection: prefix-match the configured standard model.
            _std_model = (settings.gemini_model or "").strip().lower()
            if _std_model.startswith("claude-"):
                _concurrency = 3
            else:
                _concurrency = 8
            logger.info(
                "Paper trading: per-provider concurrency cap = %d (standard=%s)",
                _concurrency, _std_model or "<unset>",
            )
            _analysis_semaphore = asyncio.Semaphore(_concurrency)

            async def _run_and_persist_one(ticker: str, kind: str):
                """Run + persist one ticker under the concurrency cap.

                Budget check runs INSIDE the lock so we don't dispatch new
                LLM calls past the cap. Exceptions are caught and logged so
                one bad ticker doesn't kill the whole gather.
                Returns the analysis dict (or None on failure) for the caller
                to fold into candidate_analyses / holding_analyses.
                """
                nonlocal total_analysis_cost
                async with _analysis_semaphore:
                    try:
                        _check_session_budget(f"pre_analysis_{kind}")
                    except Exception as exc:
                        # BudgetBreachError -- propagate to the cycle-level catch.
                        raise
                    if total_analysis_cost >= settings.paper_max_daily_cost_usd:
                        logger.warning(
                            f"Daily cost cap (${settings.paper_max_daily_cost_usd}) "
                            f"reached during {kind} for {ticker}; skipping"
                        )
                        return None
                    try:
                        analysis = await _run_single_analysis(ticker, settings)
                    except Exception as exc:
                        logger.error(f"Failed to analyze {kind} {ticker}: {exc}")
                        return None
                    if not analysis:
                        return None
                    cost = analysis.get("total_cost_usd", 0.1)
                    total_analysis_cost += cost
                    _add_session_cost(cost)
                    if analysis.get("_path") in ("lite", "full"):
                        try:
                            await _persist_analysis(analysis, bq)
                        except Exception as exc:
                            logger.warning(
                                f"Persist failed for {kind} {ticker} (non-fatal): {exc}"
                            )
                    return analysis

            # Dispatch new candidates concurrently.
            candidate_results = await asyncio.gather(
                *[_run_and_persist_one(t, "new") for t in analyze_tickers],
                return_exceptions=True,
            )
            candidate_analyses = [r for r in candidate_results if isinstance(r, dict)]

            # ── Step 4: Re-evaluate holdings ─────────────────────────
            holding_results = await asyncio.gather(
                *[_run_and_persist_one(t, "reeval") for t in reeval_tickers],
                return_exceptions=True,
            )
            holding_analyses = [r for r in holding_results if isinstance(r, dict)]

            # phase-23.1.12: no longer mutate settings.lite_mode here — operator's
            # value is preserved across the cycle.

            # ── Step 5: Mark to market ───────────────────────────────
            # phase-23.1.23: mark_to_market does ~42 blocking ops (14 pos x 3);
            # offload to threadpool so /api/health stays responsive.
            logger.info("Paper trading: Step 5 -- Mark to market")
            summary["steps"].append("mark_to_market")
            portfolio_state = await asyncio.to_thread(trader.mark_to_market)

            # ── Step 5.4: Scale-out take-profit ladder (phase-36.1) ──
            # Fires partial-close SELLs at MFE >= 2*R (50% close) and
            # MFE >= 3*R (remainder close), where R = paper_default_stop_loss_pct.
            # Gated by settings.paper_scale_out_enabled (default OFF per /goal
            # gate 3). Idempotent via scale_out_levels_hit JSON column on
            # paper_positions. Closes phase-31.0 audit P1.3 (only OPEN code
            # BLOCK on profit-protection per closure_roadmap §2 OPEN-2).
            # MUST run AFTER mark_to_market (fresh MFE) and BEFORE Step 5.6
            # stop-loss enforcement (a 3R close at +24% MFE should fire BEFORE
            # the trail-stop catches up at +trail_pct below the peak).
            try:
                scale_out_fires = await asyncio.to_thread(trader.check_scale_out_fires)
                if scale_out_fires:
                    summary["steps"].append("scale_out")
                    summary["scale_out_fires"] = scale_out_fires
                    logger.info(
                        "phase-36.1: scale-out fired for %d ticker(s) -- %s",
                        len(scale_out_fires),
                        [f"{f['ticker']}/{f['level']}" for f in scale_out_fires],
                    )
            except Exception as so_exc:
                # Fail-open: scale-out is an enhancement, not safety-critical.
                # Stop-loss enforcement at Step 5.6 still provides the floor.
                logger.warning("phase-36.1: scale-out check failed (non-fatal): %r", so_exc)

            # ── Step 5.5: Kill-switch evaluation (4.5.7) ─────────────
            # If a daily-loss or trailing-DD limit is breached, auto-flatten and
            # pause before any new-order decisions. Also short-circuits if the
            # system is already paused from a prior cycle's breach.
            from backend.services.kill_switch import get_state as _ks_state
            ks_check = await asyncio.to_thread(trader.check_and_enforce_kill_switch)
            summary["kill_switch"] = ks_check
            if ks_check.get("triggered") or _ks_state().is_paused():
                logger.warning("Paper trading: kill-switch active -- skipping decide/execute")
                summary["steps"].append("kill_switch_halted")
                summary["halted"] = True
                ks_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                _log_cycle_signals_to_bq(bq, [], ks_today)
                final_state = await asyncio.to_thread(trader.mark_to_market)
                await asyncio.to_thread(
                    trader.save_daily_snapshot,
                    trades_today=0,
                    analysis_cost_today=total_analysis_cost,
                )
                _last_run = datetime.now(timezone.utc).isoformat()
                _last_result = summary
                return summary

            # ── Step 5.6: Stop-loss enforcement (phase-25.1) ─────────
            # Wire check_stop_losses() into the cycle. Closes phase-24.1 audit
            # finding F-1 (orphan check_stop_losses with zero callers; TER held
            # at -12.30%). execute_sell is naturally idempotent: get_position
            # returns None if already sold, so retries are safe.
            #
            # phase-30.2: ALSO call backfill_missing_stops() BEFORE
            # check_stop_losses() so legacy positions with stop_loss_price=NULL
            # get a default stop synthesized from settings.paper_default_stop_loss_pct.
            # Closes phase-30.0 Stage 7 / P1-2 (7-of-11 open positions had NULL
            # stop_loss_price because phase-25.2 backfill helper had zero
            # production callers). Idempotent on subsequent cycles (returns
            # 0 backfilled, N skipped). Fail-open: a backfill exception MUST
            # NOT break check_stop_losses, which is the safety primitive.
            logger.info("Paper trading: Step 5.6 -- Stop-loss enforcement")
            summary["steps"].append("stop_loss_enforcement")
            summary["stop_loss_triggered"] = []
            summary["stop_loss_backfilled"] = []
            try:
                backfill_result = await asyncio.to_thread(trader.backfill_missing_stops)
                summary["stop_loss_backfilled"] = backfill_result.get("backfilled", [])
                if backfill_result.get("count_backfilled", 0) > 0:
                    logger.info(
                        "phase-30.2: backfill_missing_stops synthesized %d stops (skipped %d)",
                        backfill_result.get("count_backfilled", 0),
                        backfill_result.get("count_skipped", 0),
                    )
            except Exception as bf_exc:
                logger.exception(
                    "phase-30.2: backfill_missing_stops failed (non-fatal; check_stop_losses still runs): %s",
                    bf_exc,
                )
            triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
            for sl_ticker in triggered_stops:
                try:
                    sl_trade = await asyncio.to_thread(
                        trader.execute_sell,
                        ticker=sl_ticker,
                        quantity=None,
                        price=None,
                        reason="stop_loss_trigger",
                        signals=None,
                    )
                    if sl_trade:
                        summary["stop_loss_triggered"].append(sl_ticker)
                        closed_tickers.append(sl_ticker)  # phase-30.3: route stop-out exits through the learn loop (audit Stage 12 + P1-3).
                        logger.warning(
                            "Paper trading: stop-loss triggered for %s -- sold at %s",
                            sl_ticker, sl_trade.get("price"),
                        )
                except Exception as sl_exc:
                    logger.exception("Stop-loss execute_sell failed for %s: %s", sl_ticker, sl_exc)

            # phase-32.4: backfill missing company_name on paper_positions
            # (legacy rows opened pre-_fetch_ticker_meta default to ticker).
            # Cosmetic; runs AFTER check_stop_losses so it never blocks the
            # safety-critical stop-loss path. Fail-open: a yfinance hiccup
            # never breaks the cycle.
            summary["company_name_backfilled"] = []
            try:
                cn_result = await asyncio.to_thread(trader.backfill_missing_company_names)
                summary["company_name_backfilled"] = cn_result.get("backfilled", [])
                if cn_result.get("count_backfilled", 0) > 0:
                    logger.info(
                        "phase-32.4: backfill_missing_company_names updated %d rows (skipped %d)",
                        cn_result.get("count_backfilled", 0),
                        cn_result.get("count_skipped", 0),
                    )
            except Exception as cn_exc:
                logger.exception(
                    "phase-32.4: backfill_missing_company_names failed (non-fatal; cosmetic only): %s",
                    cn_exc,
                )

            # ── Step 6: Decide trades ────────────────────────────────
            logger.info("Paper trading: Step 6 -- Deciding trades")
            summary["steps"].append("deciding")
            positions = await asyncio.to_thread(trader.get_positions)  # Refresh after MTM (phase-23.1.23)

            # phase-23.1.14: enrich legacy positions whose `sector` field is empty
            # (BQ paper_positions rows predating the sector column migration).
            # decide_trades reads pos.get("sector") to seed sector_counts; without
            # this enrichment those rows fall into "Unknown" and the sector cap is
            # silently bypassed for tickers whose true GICS sector already exceeds
            # the cap. Same _fetch_ticker_meta + asyncio.to_thread pattern used at
            # the candidate-enrichment site above. Skipped when cap is disabled.
            max_per_sector = int(getattr(settings, "paper_max_per_sector", 0) or 0)
            if max_per_sector > 0 and positions:
                legacy_tickers = [
                    p["ticker"] for p in positions
                    if not (p.get("sector") or "").strip()
                ]
                if legacy_tickers:
                    try:
                        from backend.api.paper_trading import _fetch_ticker_meta
                        pos_meta_response = await asyncio.to_thread(
                            _fetch_ticker_meta, legacy_tickers, settings, bq,
                        )
                        pos_meta_map = (pos_meta_response or {}).get("meta", {})
                        enriched_count = 0
                        for p in positions:
                            if (p.get("sector") or "").strip():
                                continue
                            info = pos_meta_map.get(p["ticker"], {}) or {}
                            sector = info.get("sector") or ""
                            if sector:
                                p["sector"] = sector
                                enriched_count += 1
                        logger.info(
                            "Enriched %d legacy positions with sector (of %d missing)",
                            enriched_count, len(legacy_tickers),
                        )
                    except Exception as e:
                        logger.warning(
                            "Legacy position sector enrichment failed (non-fatal): %s", e,
                        )

            # phase-23.1.7: thread the screener candidate dict through to the buy-side
            # decider so the trade record captures momentum/RSI/composite_score and
            # all signal-stack overlays in the rationale.
            candidates_by_ticker = {c["ticker"]: c for c in candidates if c.get("ticker")}
            orders = decide_trades(
                current_positions=positions,
                candidate_analyses=candidate_analyses,
                holding_analyses=holding_analyses,
                portfolio_state=portfolio_state,
                settings=settings,
                candidates_by_ticker=candidates_by_ticker,
            )

            # ── Step 7: Execute trades ───────────────────────────────
            logger.info(f"Paper trading: Step 7 -- Executing {len(orders)} trades")
            summary["steps"].append("executing")
            # phase-30.3: closed_tickers now lives at cycle-top (line ~169)
            # so Step 5.6 stop-outs can populate it. Re-init here would
            # erase Step 5.6's appends.

            # Sells first
            # phase-23.1.23: execute_sell/execute_buy also do blocking BQ + yfinance
            # + ExecutionRouter ops; offload to threadpool.
            for order in orders:
                if order.action != "SELL":
                    continue
                trade = await asyncio.to_thread(
                    trader.execute_sell,
                    ticker=order.ticker,
                    quantity=order.quantity,
                    price=order.price,
                    reason=order.reason,
                    signals=order.signals,
                )
                if trade:
                    trades_executed += 1
                    closed_tickers.append(order.ticker)

            # Then buys
            # phase-30.6: ALWAYS fetch the live price for fill (was: prefer
            # order.price -- the analysis-time price -- which left the price-
            # tolerance gate with nothing to compare against). The live price
            # becomes the fill reference; order.price_at_analysis (now a
            # distinct TradeOrder field) is passed separately so execute_buy's
            # gate can reject when divergence > paper_price_tolerance_pct.
            for order in orders:
                if order.action != "BUY":
                    continue
                from backend.services.paper_trader import _get_live_price
                live_price = _get_live_price(order.ticker) or 0
                # Fallback: if live fetch failed (network outage), use the
                # analysis-time price so the cycle still progresses. Gate is
                # automatically a no-op in this branch because live == analysis.
                price = live_price if live_price > 0 else (order.price_at_analysis or order.price or 0)
                if price <= 0:
                    logger.warning(f"Dropping BUY for {order.ticker}: price={price} (yfinance + analysis fallback both empty)")
                    continue
                trade = await asyncio.to_thread(
                    trader.execute_buy,
                    ticker=order.ticker,
                    amount_usd=order.amount_usd or 0,
                    price=price,
                    reason=order.reason,
                    analysis_id=order.analysis_id,
                    risk_judge_decision=order.risk_judge_decision,
                    stop_loss_price=order.stop_loss_price,
                    risk_judge_position_pct=order.risk_judge_position_pct,
                    signals=order.signals,
                    sector=order.sector or None,  # phase-23.2.6-fix
                    # phase-30.6: analysis-time reference for the
                    # price-tolerance gate inside execute_buy.
                    price_at_analysis=order.price_at_analysis,
                    # phase-40.8.1 (P3): in-memory FF3 loadings; BQ persist
                    # deferred to phase-40.8.2.
                    factor_loadings=order.factor_loadings,
                )
                if trade:
                    trades_executed += 1

            # ── Step 7.5: Log signals to BQ signals_log ─────────────
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            signals_logged = _log_cycle_signals_to_bq(bq, orders, today_str)
            summary["signals_logged"] = signals_logged

            # ── Step 8: Final mark-to-market + snapshot ──────────────
            # phase-23.1.23: same async wrap as Step 5.
            logger.info("Paper trading: Step 8 -- Final snapshot")
            summary["steps"].append("snapshot")
            final_state = await asyncio.to_thread(trader.mark_to_market)
            snapshot = await asyncio.to_thread(
                trader.save_daily_snapshot,
                trades_today=trades_executed,
                analysis_cost_today=total_analysis_cost,
            )

            # ── Step 9: Learn from closed trades ─────────────────────
            if closed_tickers:
                summary["steps"].append("learning")
                try:
                    await _learn_from_closed_trades(closed_tickers, bq, settings)
                except Exception as e:
                    logger.error(f"Learning step failed (non-fatal): {e}")

            # ── Step 10: MetaCoordinator health check ────────────────
            try:
                snapshots = bq.get_paper_snapshots(limit=60)
                from backend.services.perf_tracker import get_perf_tracker
                health = MetaCoordinator.gather_health(
                    bq_client=bq,
                    perf_tracker=get_perf_tracker(),
                    paper_snapshots=snapshots,
                )
                decision = _coordinator.decide(health)
                summary["coordinator"] = {
                    "action": decision.action,
                    "reason": decision.reason,
                    "target_agents": decision.target_agents,
                    "priority": decision.priority,
                    "health": {
                        "sharpe": round(health.sharpe_ratio, 4),
                        "accuracy": round(health.agent_accuracy, 4),
                        "p95_latency_ms": round(health.p95_latency_ms, 1),
                    },
                }
                logger.info(
                    f"MetaCoordinator decision: {decision.action} "
                    f"(reason={decision.reason})"
                )
            except Exception as e:
                logger.warning(f"MetaCoordinator step failed (non-fatal): {e}")

            # ── Step 10.5: strategy_decisions heartbeat (phase-30.7) ──
            # Emit a per-cycle heartbeat row to `pyfinagent_data.strategy_decisions`
            # so the table is operator-visible-NOT-empty. The phase-26.5
            # migration created the table but no writer was ever wired into
            # the production cycle (audit Stage 3: only 1 row across 36+
            # days of production, a smoke-test). This heartbeat closes the
            # observability gap WITHOUT activating the full Layer-2
            # strategy router (deferred to phase-31). Dead-man's-switch
            # pattern per OneUptime Feb 2026 + arXiv 2509.16707 immutable
            # per-cycle persistence. Fail-open: any BQ exception MUST NOT
            # break the cycle.
            try:
                current_strategy = (best_params.get("strategy", "unknown")
                                    if best_params else "unknown")
                strategy_decisions_row = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "cycle_id": _cycle_id,
                    "decided_strategy": current_strategy,
                    "prior_strategy": current_strategy,
                    "trigger": "cycle_heartbeat",
                    "decay_signal": None,
                    "decay_attribution": None,
                    "rationale": ("per-cycle heartbeat; no regime change detected. "
                                  "Full router activation deferred to phase-31."),
                }
                await asyncio.to_thread(bq.save_strategy_decision, strategy_decisions_row)
                summary["strategy_decision_logged"] = "cycle_heartbeat"
            except Exception as sd_exc:
                logger.warning(
                    "phase-30.7: strategy_decisions heartbeat write failed (non-fatal): %s",
                    sd_exc,
                )

            # ── Done ─────────────────────────────────────────────────
            summary.update({
                "status": "completed",
                "nav": final_state["nav"],
                "pnl_pct": final_state["pnl_pct"],
                "trades_executed": trades_executed,
                "analysis_cost": round(total_analysis_cost, 4),
                "closed_tickers": closed_tickers,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                # phase-25.S: per-ticker attribution is computed on-the-fly
                # by GET /api/paper-trading/attribution. This flag marks
                # the cycle as "attribution-ready" so operators (and Q/A
                # verifiers) can confirm the data is queryable post-cycle.
                # No new BQ table; the endpoint reads existing trades +
                # llm_call_log on demand. Closes phase-24.13 F-6.
                "attribution_computed": True,
            })
            _last_run = summary["timestamp"]
            _last_result = summary
            logger.info(f"Paper trading cycle complete: NAV=${final_state['nav']:.2f}, "
                         f"P&L={final_state['pnl_pct']:.2f}%, trades={trades_executed}, "
                         f"cost=${total_analysis_cost:.4f}")
            return summary

    except asyncio.TimeoutError:
        logger.error("Paper trading cycle TIMED OUT after %.0fs", _cycle_timeout)
        summary.update({"status": "timeout", "error": f"cycle exceeded {_cycle_timeout:.0f}s"})
        _last_result = summary
        return summary
    except Exception as e:
        # phase-25.A8: cost-budget HARD-BLOCK. BudgetBreachError raised by
        # llm_client._check_cost_budget() halts the cycle BEFORE further
        # LLM spend accumulates. Closes phase-24.8 F-4. Catch via
        # name-check to avoid importing the symbol at module load time
        # (keeps llm_client and autonomous_loop loosely coupled).
        if type(e).__name__ == "BudgetBreachError":
            logger.warning("Paper trading cycle HALTED by cost-budget hard-block: %s", e)
            summary.update({
                "status": "budget_breach",
                "error": str(e),
                "budget_tripped": True,
            })
        else:
            logger.error(f"Paper trading cycle failed: {e}", exc_info=True)
            summary.update({"status": "error", "error": str(e)})
        _last_result = summary
        return summary
    finally:
        _running = False
        # phase-38.6.1: release the file-based cycle_lock (unlinks pidfile
        # + releases flock). Idempotent: if already exited, _lock_cm is unset.
        try:
            _lock_cm.__exit__(None, None, None)  # type: ignore[name-defined]
        except (NameError, AttributeError):
            pass  # _lock_cm not set (e.g. dry-run path)
        except Exception as _release_exc:
            logger.warning("phase-38.6.1: cycle_lock release failed (non-fatal): %r", _release_exc)
        # phase-26.1: clear cycle_id so log_llm_call rows OUTSIDE a cycle
        # don't accidentally tag with a stale id from a prior cycle.
        _current_cycle_id = None
        # 4.5.8 cycle health: end-of-cycle row (always, regardless of branch).
        try:
            _cycle_log().record_cycle_end(
                cycle_id=_cycle_id,
                started_at=_cycle_started_at,
                status=summary.get("status", "unknown"),
                n_trades=int(summary.get("trades_executed", trades_executed) or 0),
                error_count=int(summary.get("error_count", 0) or 0),
                data_source_ages=summary.get("data_source_ages") or {},
                bq_ingest_lag_sec=summary.get("bq_ingest_lag_sec"),
            )
        except Exception as _e:
            logger.warning(f"cycle_health record_cycle_end failed: {_e}")

        # phase-23.2.18: operator notification on any non-completed status.
        # Closes the silent-failure gap from 04-30 / 05-01 / 05-04 / 05-05
        # where the cycle hung or was kickstart-killed and the user got
        # no signal. Async-safe: we are still inside the coroutine. Uses
        # the sync wrapper because finally may run during cancellation
        # cleanup where awaiting is not always safe.
        _final_status = summary.get("status", "unknown")
        if _final_status not in ("completed", "skipped"):
            try:
                from backend.services.observability.alerting import raise_cron_alert_sync
                raise_cron_alert_sync(
                    source="autonomous_loop",
                    error_type=f"cycle_{_final_status}",
                    severity="P1",
                    title=f"Autonomous trading cycle {_final_status}",
                    details={
                        "cycle_id": summary.get("cycle_id", "?"),
                        "started_at": summary.get("started_at", "?"),
                        "status": _final_status,
                        "error": str(summary.get("error", ""))[:300],
                        "steps_completed": ",".join(summary.get("steps", [])[-5:]),
                        "trades_executed": summary.get("trades_executed", 0),
                    },
                )
            except Exception as _alert_err:
                logger.warning(f"cycle failure-alert dispatch failed: {_alert_err}")
        elif _final_status == "completed":
            # phase-25.N: emit a P3 cycle-completed summary so operators get a
            # positive signal per cycle (not just on failure). Closes audit
            # bucket 24.5 F-5(e). Dedup key 'cycle_completed_summary' is
            # distinct from the failure path so the two paths never collide.
            try:
                from backend.services.observability.alerting import raise_cron_alert_sync
                _duration_sec = None
                try:
                    if _cycle_started_at:
                        from datetime import datetime as _dt, timezone as _tz
                        _start = _dt.fromisoformat(str(_cycle_started_at).replace("Z", "+00:00")) \
                            if isinstance(_cycle_started_at, str) else _cycle_started_at
                        _now = _dt.now(_tz.utc)
                        _duration_sec = (_now - _start).total_seconds()
                except Exception:
                    pass
                raise_cron_alert_sync(
                    source="autonomous_loop",
                    error_type="cycle_completed_summary",
                    severity="P3",
                    title="Autonomous trading cycle completed",
                    details={
                        "cycle_id": summary.get("cycle_id", "?"),
                        "started_at": str(summary.get("started_at", "?")),
                        "duration_sec": _duration_sec if _duration_sec is not None else "?",
                        "trades_executed": summary.get("trades_executed", 0),
                        "stops_executed": summary.get("stops_executed", 0),
                        "mode": summary.get("mode", "full"),
                        "recommendations_count": summary.get("recommendations_count", 0),
                        "status": _final_status,
                    },
                )
            except Exception as _summary_err:
                logger.warning(f"cycle summary-alert dispatch failed: {_summary_err}")

            # phase-25.L: tiered drawdown alarm. Fetches the latest snapshots
            # and fires P1 Slack alerts at -3%/-5%/-10% drawdown tiers. Fully
            # fail-open. Each tier has a distinct dedup key so AlertDeduper
            # suppresses repeated same-tier alerts.
            try:
                from backend.services.drawdown_alarm import emit_drawdown_alarms
                from backend.db.bigquery_client import BigQueryClient as _BQ
                _bq = _BQ(settings or get_settings())
                _snapshots = _bq.get_paper_snapshots(limit=180)
                emit_drawdown_alarms(_snapshots or [], source="autonomous_loop")
            except Exception as _dd_err:
                logger.warning(f"drawdown_alarm dispatch failed: {_dd_err}")


async def _run_single_analysis(ticker: str, settings: Settings) -> Optional[dict]:
    """Run a single analysis and extract key fields for trade decisions.

    phase-23.1.12: branches on `settings.lite_mode`:
      - lite_mode=True (operator opted into cheap fast analysis) -> 4-field
        Claude lite analyzer using `settings.gemini_model`.
      - lite_mode=False (operator picked Sonnet/Opus and wants the full
        pipeline) -> AnalysisOrchestrator with their `gemini_model` +
        `deep_think_model`. Falls back to lite Claude if the orchestrator
        fails (e.g. transient Vertex/Gemini outage).

    Cost containment is via `paper_max_daily_cost_usd` cap in the calling
    cycle loop -- not via silent forced-lite.
    """
    if settings.lite_mode:
        try:
            # phase-27.3 (C2): dispatch by configured standard model. Was hardcoded
            # to _run_claude_analysis which refused non-Claude models, leaving
            # gemini-* selections with no lite fallback.
            return await _select_lite_analyzer(settings.gemini_model)(ticker, settings)
        except Exception as e:
            logger.warning("Lite analysis failed for %s (lite_mode=True): %s", ticker, e)
            return None

    # Full pipeline path (operator chose lite_mode=False)
    try:
        orchestrator = AnalysisOrchestrator(settings)
        report = await orchestrator.run_full_analysis(ticker)
        if not report:
            raise RuntimeError("orchestrator returned empty report")

        synthesis = report.get("final_synthesis", {})
        rec = synthesis.get("recommendation", {})
        quant = report.get("quant", {})
        risk = synthesis.get("risk_assessment", {})
        cost_summary = report.get("cost_summary", {})

        return {
            "ticker": ticker,
            "recommendation": rec.get("action", "HOLD") if isinstance(rec, dict) else str(rec),
            "final_score": synthesis.get("final_score", 0),
            "risk_assessment": risk,
            "price_at_analysis": quant.get("yf_data", {}).get("valuation", {}).get("currentPrice") if isinstance(quant.get("yf_data"), dict) else None,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "total_cost_usd": cost_summary.get("total_cost_usd", 0.1) if isinstance(cost_summary, dict) else 0.1,
            "full_report": report,
            # phase-25.A2: marker so _persist_analysis guard picks up full-pipeline rows.
            # Closes phase-24.2 audit F-2 (orchestrator.py had zero save_report calls;
            # /reports page empty because full-path runs evaporated without persistence).
            "_path": "full",
        }
    except Exception as e:
        logger.warning(
            "Full orchestrator failed for %s: %s -- falling back to lite Claude analyzer",
            ticker, e,
        )

    # Last-resort fallback: try lite path so the cycle still produces a decision.
    # phase-27.3 (C2): provider-aware via _select_lite_analyzer.
    try:
        return await _select_lite_analyzer(settings.gemini_model)(ticker, settings)
    except Exception as e:
        logger.error("Both full and lite paths failed for %s: %s", ticker, e)
        return None


# phase-25.A: independent Risk Judge for the lite path. The trader and the
# risk judge are TWO distinct LLM calls now; the judge's system prompt forces
# evaluation along volatility/concentration/valuation axes rather than
# rubber-stamping the trader. Pattern grounded in ATLAS arXiv 2510.15949 +
# EvidentlyAI rubric-based judge guidance + Anthropic structured-output
# recommendations. See handoff/archive/phase-25.A/research_brief.md.
_LITE_RISK_JUDGE_SYSTEM = (
    "You are an independent Risk Judge for a paper trading portfolio. "
    "Your role is to evaluate position risk -- NOT to validate the trader's recommendation. "
    "Evaluate the following three axes independently, then size the position:\n"
    "  1. VOLATILITY: Is 20d or 60d momentum extreme (>15% either direction)? High = reduce size.\n"
    "  2. CONCENTRATION: Would adding this position exceed 10% of portfolio in one sector? High = reduce size.\n"
    "  3. VALUATION: Is P/E > 40 or market cap < $2B (micro-cap)? High = reduce size.\n"
    "Derive a recommended_position_pct (1-10) from these axes alone. "
    "Do not simply agree with the trader.\n"
    "Respond ONLY with valid JSON."
)

_LITE_RISK_JUDGE_TEMPLATE = (
    "Stock: {ticker} ({name})\n"
    "Sector: {sector} | P/E: {pe_ratio:.1f} | Market Cap: ${market_cap_b:.1f}B\n"
    "20d momentum: {momentum_20d:+.1f}% | 60d momentum: {momentum_60d:+.1f}%\n"
    "Trader recommendation: {trader_action} (confidence: {trader_confidence})\n\n"
    "Evaluate the three risk axes above. Return JSON:\n"
    "{{\n"
    '  "decision": "APPROVE_FULL" | "APPROVE_REDUCED" | "APPROVE_HEDGED" | "REJECT",\n'
    '  "recommended_position_pct": <float 1-10>,\n'
    '  "risk_level": "LOW" | "MODERATE" | "HIGH" | "EXTREME",\n'
    '  "reasoning": "<one sentence per axis, then position conclusion>",\n'
    '  "risk_limits": {{"stop_loss_pct": <float>, "max_drawdown_pct": <float>}}\n'
    "}}"
)

_LITE_RISK_DEFAULT = {
    "decision": "APPROVE_REDUCED",
    "recommended_position_pct": 3.0,
    "risk_level": "MODERATE",
    "reasoning": "risk-judge parse failed; falling back to conservative default sizing",
    "risk_limits": {"stop_loss_pct": 10.0, "max_drawdown_pct": 15.0},
}


def _select_lite_analyzer(model_name):
    """Factory: pick the lite-analyzer coroutine for the configured standard model.

    phase-27.3 (C2): the lite fallback was hardcoded to Claude only, so
    selecting `gemini-2.5-flash` as the standard model bricked the safety
    net ("standard model … is not a Claude model" raise). The factory
    dispatches by model-name prefix:
      - `gemini-*` -> `_run_gemini_analysis` (direct AI Studio API key)
      - anything else (default `claude-*`) -> `_run_claude_analysis`

    Returns the coroutine FUNCTION (uncalled). Callers do
    `await _select_lite_analyzer(name)(ticker, settings)`.
    """
    name = (model_name or "").strip().lower()
    if name.startswith("gemini-"):
        return _run_gemini_analysis
    return _run_claude_analysis


async def _run_claude_analysis(ticker: str, settings: Settings) -> dict:
    """Lightweight Claude-based analysis for paper trading decisions."""
    import anthropic
    import yfinance as yf

    logger.info(f"Claude analysis: analyzing {ticker}")

    # Fetch current market data via yfinance
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="3mo")

    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    market_cap = info.get("marketCap", 0)
    pe_ratio = info.get("trailingPE", 0)
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    name = info.get("shortName", ticker)

    # Calculate simple momentum
    if len(hist) >= 20:
        price_20d_ago = hist["Close"].iloc[-20]
        momentum_20d = ((current_price - price_20d_ago) / price_20d_ago * 100) if price_20d_ago else 0
    else:
        momentum_20d = 0

    if len(hist) >= 60:
        price_60d_ago = hist["Close"].iloc[-60]
        momentum_60d = ((current_price - price_60d_ago) / price_60d_ago * 100) if price_60d_ago else 0
    else:
        momentum_60d = 0

    # Resolve the standard model from settings (Claude default; Gemini/others
    # selectable from the Settings UI). Field name `gemini_model` is preserved
    # for backward compat; routing layer (make_client) dispatches by prefix.
    model_name = (settings.gemini_model or "claude-sonnet-4-6").strip()

    # Only the direct-Anthropic path is exercised here. Non-Claude model
    # selections flow through _run_single_analysis's Gemini fallback.
    if not model_name.startswith("claude-"):
        raise ValueError(
            f"standard model '{model_name}' is not a Claude model; "
            f"_run_claude_analysis is Claude-only. Gemini/other paths run via the "
            f"AnalysisOrchestrator fallback in _run_single_analysis."
        )

    api_key = settings.anthropic_api_key.get_secret_value() or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("No ANTHROPIC_API_KEY available")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Analyze {ticker} ({name}) for a paper trading portfolio. Be concise.

Stock: {ticker} ({name})
Sector: {sector} | Industry: {industry}
Price: ${current_price:.2f} | Market Cap: ${market_cap/1e9:.1f}B | P/E: {pe_ratio:.1f}
20-day momentum: {momentum_20d:+.1f}% | 60-day momentum: {momentum_60d:+.1f}%

Decision rules (apply in order):
- A portfolio needs positions to generate return; HOLD on ambiguous data, but lean BUY on clear momentum.
- If momentum_20d > 3.0 AND momentum_60d > 5.0 AND market_cap > 5e9, lean BUY unless there is a clear negative signal in the data.
- If momentum_20d < -5.0 AND position is held, lean SELL.
- Otherwise HOLD.

Based on the rules and data above, provide:
1. Action: BUY, SELL, or HOLD
2. Confidence: 0-100
3. Score: 1-10 (overall attractiveness)
4. Key reason (one sentence)

Respond in this exact JSON format:
{{"action": "BUY", "confidence": 75, "score": 7, "reason": "Strong momentum with reasonable valuation"}}"""

    response = await asyncio.to_thread(
        client.messages.create,
        model=model_name,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    text = response.content[0].text.strip()
    # Extract JSON from response
    import re
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        analysis = json_io.loads(json_match.group())
    else:
        analysis = {"action": "HOLD", "confidence": 0, "score": 5, "reason": "Could not parse analysis"}

    logger.info(f"Claude analysis for {ticker}: {analysis['action']} (confidence={analysis['confidence']}, score={analysis['score']})")

    # phase-25.A: SECOND, INDEPENDENT LLM call -- the Risk Judge. Closes
    # phase-24.4 F-1 (the lite path previously aliased the trader's reason
    # into risk_assessment). The risk judge system prompt forces evaluation
    # along volatility/concentration/valuation axes -- it does NOT validate
    # the trader's recommendation. Cost impact: ~$0.003/ticker, already
    # accounted in the existing $0.01/ticker ceiling.
    risk_prompt = _LITE_RISK_JUDGE_TEMPLATE.format(
        ticker=ticker,
        name=name,
        sector=sector,
        pe_ratio=pe_ratio or 0.0,
        market_cap_b=(market_cap or 0) / 1e9,
        momentum_20d=momentum_20d,
        momentum_60d=momentum_60d,
        trader_action=analysis["action"],
        trader_confidence=analysis["confidence"],
    )
    try:
        risk_response = await asyncio.to_thread(
            client.messages.create,
            model=model_name,
            max_tokens=300,
            system=_LITE_RISK_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": risk_prompt}],
        )
        risk_text = risk_response.content[0].text.strip()
        # re.DOTALL so the nested risk_limits object is captured.
        risk_json_match = re.search(r"\{.*\}", risk_text, re.DOTALL)
        if risk_json_match:
            risk_dict = json_io.loads(risk_json_match.group())
        else:
            risk_dict = dict(_LITE_RISK_DEFAULT)
            logger.warning(
                "Lite risk judge for %s: no JSON in response -- using default sizing", ticker,
            )
    except Exception as exc:
        risk_dict = dict(_LITE_RISK_DEFAULT)
        logger.warning("Lite risk judge for %s failed (%s) -- using default sizing", ticker, exc)

    risk_reasoning = str(risk_dict.get("reasoning") or _LITE_RISK_DEFAULT["reasoning"])
    risk_assessment = {
        "decision": str(risk_dict.get("decision") or _LITE_RISK_DEFAULT["decision"]),
        "reasoning": risk_reasoning,
        # Backward-compat alias: bq.save_report at line ~818 reads
        # risk_assessment.get("reason", "") for the summary column.
        "reason": risk_reasoning,
        "recommended_position_pct": float(
            risk_dict.get("recommended_position_pct")
            or _LITE_RISK_DEFAULT["recommended_position_pct"]
        ),
        "risk_level": str(risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]),
        "risk_limits": dict(risk_dict.get("risk_limits") or _LITE_RISK_DEFAULT["risk_limits"]),
    }
    logger.info(
        "Lite risk judge for %s: decision=%s position_pct=%.1f risk_level=%s",
        ticker,
        risk_assessment["decision"],
        risk_assessment["recommended_position_pct"],
        risk_assessment["risk_level"],
    )

    return {
        "ticker": ticker,
        # phase-23.1.12: marker so the cycle loop knows this came from the lite
        # path (and therefore needs explicit persist via _persist_lite_analysis).
        # The full orchestrator path writes its own row directly.
        "_path": "lite",
        "recommendation": analysis["action"],
        "final_score": analysis["score"],
        "risk_assessment": risk_assessment,
        "price_at_analysis": current_price,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": 0.01,
        # phase-23.1.11: full_report.source reflects the actual model_name (was hardcoded
        # "claude-sonnet-4" — wrong since gemini_model can be a Claude variant or Gemini).
        # market_data carries name + industry so the Reports History tab can render company name.
        "full_report": {
            "source": model_name,
            "analysis": analysis,
            "market_data": {
                "name": name,
                "price": current_price,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "sector": sector,
                "industry": industry,
                "momentum_20d": momentum_20d,
                "momentum_60d": momentum_60d,
            },
        },
    }


async def _run_gemini_analysis(ticker: str, settings: Settings) -> dict:
    """Lightweight Gemini-based analysis for paper trading decisions.

    phase-27.3 (C2): mirror of `_run_claude_analysis` for non-Claude standard
    models. Output dict shape IDENTICAL — same keys, `_path: "lite"` marker,
    so `_persist_analysis` and downstream readers don't branch by provider.
    Routes through `make_client` (post-27.1 priority order) which dispatches
    `gemini-*` to a direct AI Studio API key (no Vertex / GCP creds).

    Two-LLM-call pattern preserved: trader prompt + independent risk-judge.
    """
    import re as _re
    import yfinance as yf
    from backend.agents.llm_client import make_client, safe_text

    logger.info(f"Gemini analysis: analyzing {ticker}")

    # 1. Market data via yfinance (parity with Claude path).
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="3mo")

    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    market_cap = info.get("marketCap", 0)
    pe_ratio = info.get("trailingPE", 0)
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    name = info.get("shortName", ticker)

    if len(hist) >= 20:
        price_20d_ago = hist["Close"].iloc[-20]
        momentum_20d = ((current_price - price_20d_ago) / price_20d_ago * 100) if price_20d_ago else 0
    else:
        momentum_20d = 0
    if len(hist) >= 60:
        price_60d_ago = hist["Close"].iloc[-60]
        momentum_60d = ((current_price - price_60d_ago) / price_60d_ago * 100) if price_60d_ago else 0
    else:
        momentum_60d = 0

    model_name = (settings.gemini_model or "gemini-2.5-flash").strip()
    if not model_name.startswith("gemini-"):
        raise ValueError(
            f"standard model '{model_name}' is not a Gemini model; "
            "_run_gemini_analysis is Gemini-only. _select_lite_analyzer should "
            "have routed claude-* to _run_claude_analysis instead."
        )

    # Build a single Gemini client and reuse for trader + risk-judge calls.
    client = make_client(model_name, vertex_model=None, settings=settings)

    trader_prompt = f"""Analyze {ticker} ({name}) for a paper trading portfolio. Be concise.

Stock: {ticker} ({name})
Sector: {sector} | Industry: {industry}
Price: ${current_price:.2f} | Market Cap: ${market_cap/1e9:.1f}B | P/E: {pe_ratio:.1f}
20-day momentum: {momentum_20d:+.1f}% | 60-day momentum: {momentum_60d:+.1f}%

Decision rules (apply in order):
- A portfolio needs positions to generate return; HOLD on ambiguous data, but lean BUY on clear momentum.
- If momentum_20d > 3.0 AND momentum_60d > 5.0 AND market_cap > 5e9, lean BUY unless there is a clear negative signal.
- If momentum_20d < -5.0 AND position is held, lean SELL.
- Otherwise HOLD.

Respond ONLY with valid JSON, no prose. Schema:
{{"action": "BUY"|"SELL"|"HOLD", "confidence": <int 0-100>, "score": <int 1-10>, "reason": "<one sentence>"}}"""

    # Trader call. asyncio.to_thread because GeminiClient.generate_content
    # blocks (it runs concurrent.futures.Future internally; safe to wrap).
    trader_response = await asyncio.to_thread(
        client.generate_content,
        trader_prompt,
        {"max_output_tokens": 200, "temperature": 0.0, "response_mime_type": "application/json"},
    )
    text = safe_text(trader_response.text).strip()
    json_match = _re.search(r"\{[^}]+\}", text, _re.DOTALL)
    if json_match:
        try:
            analysis = json_io.loads(json_match.group())
        except Exception:
            analysis = {"action": "HOLD", "confidence": 0, "score": 5, "reason": "Could not parse trader JSON"}
    else:
        analysis = {"action": "HOLD", "confidence": 0, "score": 5, "reason": "No JSON in trader response"}

    logger.info(
        f"Gemini analysis for {ticker}: {analysis['action']} "
        f"(confidence={analysis['confidence']}, score={analysis['score']})"
    )

    # Risk Judge — independent second call. Same system prompt as Claude path.
    risk_prompt = (
        _LITE_RISK_JUDGE_SYSTEM
        + "\n\n"
        + _LITE_RISK_JUDGE_TEMPLATE.format(
            ticker=ticker,
            name=name,
            sector=sector,
            pe_ratio=pe_ratio or 0.0,
            market_cap_b=(market_cap or 0) / 1e9,
            momentum_20d=momentum_20d,
            momentum_60d=momentum_60d,
            trader_action=analysis["action"],
            trader_confidence=analysis["confidence"],
        )
    )
    try:
        risk_response = await asyncio.to_thread(
            client.generate_content,
            risk_prompt,
            {"max_output_tokens": 300, "temperature": 0.0, "response_mime_type": "application/json"},
        )
        risk_text = safe_text(risk_response.text).strip()
        risk_json_match = _re.search(r"\{.*\}", risk_text, _re.DOTALL)
        if risk_json_match:
            risk_dict = json_io.loads(risk_json_match.group())
        else:
            risk_dict = dict(_LITE_RISK_DEFAULT)
            logger.warning(
                "Gemini lite risk judge for %s: no JSON in response -- using default sizing", ticker,
            )
    except Exception as exc:
        risk_dict = dict(_LITE_RISK_DEFAULT)
        logger.warning(
            "Gemini lite risk judge for %s failed (%s) -- using default sizing", ticker, exc,
        )

    risk_reasoning = str(risk_dict.get("reasoning") or _LITE_RISK_DEFAULT["reasoning"])
    risk_assessment = {
        "decision": str(risk_dict.get("decision") or _LITE_RISK_DEFAULT["decision"]),
        "reasoning": risk_reasoning,
        "reason": risk_reasoning,  # backward-compat alias for bq.save_report
        "recommended_position_pct": float(
            risk_dict.get("recommended_position_pct")
            or _LITE_RISK_DEFAULT["recommended_position_pct"]
        ),
        "risk_level": str(risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]),
        "risk_limits": dict(risk_dict.get("risk_limits") or _LITE_RISK_DEFAULT["risk_limits"]),
    }
    logger.info(
        "Gemini lite risk judge for %s: decision=%s position_pct=%.1f risk_level=%s",
        ticker,
        risk_assessment["decision"],
        risk_assessment["recommended_position_pct"],
        risk_assessment["risk_level"],
    )

    return {
        "ticker": ticker,
        "_path": "lite",
        "recommendation": analysis["action"],
        "final_score": analysis["score"],
        "risk_assessment": risk_assessment,
        "price_at_analysis": current_price,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": 0.005,  # Gemini Flash is ~half Claude Sonnet at this prompt size
        "full_report": {
            "source": model_name,
            "analysis": analysis,
            "market_data": {
                "name": name,
                "price": current_price,
                "market_cap": market_cap,
                "pe_ratio": pe_ratio,
                "sector": sector,
                "industry": industry,
                "momentum_20d": momentum_20d,
                "momentum_60d": momentum_60d,
            },
        },
    }


# phase-23.1.11: persist lite-Claude analyzer rows to analysis_results so the
# Reports page History tab shows paper-trading candidates alongside manual
# analyses. Path A from the research brief — write to existing table; ~14
# fields populated, ~74 columns left NULL (storage-free in BQ columnar
# format; honest signal that the full Gemini pipeline did not run).

async def _persist_analysis(analysis: dict, bq: BigQueryClient) -> None:
    """phase-25.A2: write an analysis row to analysis_results.

    Generalized from `_persist_lite_analysis` to handle BOTH lite and full
    paths (closes phase-24.2 F-2 — full pipeline previously evaporated
    without persistence; /reports page was empty).

    Reads `_path` from the analysis dict for honest source tagging in the
    persisted row (lite vs full). Non-fatal: any BQ error logs a warning
    but the trading cycle continues.
    """
    try:
        ticker = analysis.get("ticker") or ""
        if not ticker:
            return
        full_report = analysis.get("full_report") or {}
        market_data = full_report.get("market_data") or {}
        await asyncio.to_thread(
            bq.save_report,
            ticker=ticker,
            company_name=market_data.get("name") or ticker,
            final_score=float(analysis.get("final_score") or 0.0),
            recommendation=analysis.get("recommendation") or "HOLD",
            summary=(analysis.get("risk_assessment") or {}).get("reason", "") or "",
            full_report=full_report,
            price_at_analysis=analysis.get("price_at_analysis"),
            market_cap=market_data.get("market_cap"),
            pe_ratio=market_data.get("pe_ratio"),
            sector=market_data.get("sector") or "",
            industry=market_data.get("industry") or "",
            recommendation_confidence=(full_report.get("analysis") or {}).get("confidence"),
            total_cost_usd=float(analysis.get("total_cost_usd") or 0.01),
            standard_model=full_report.get("source") or "",
        )
        logger.info("Lite analysis persisted to analysis_results for %s", ticker)
    except Exception as exc:
        logger.warning(
            "Failed to persist lite analysis for %s: %s",
            analysis.get("ticker", "?"), exc,
        )


async def _learn_from_closed_trades(tickers: list[str], bq: BigQueryClient, settings: Settings):
    """Feed closed trades into outcome tracking for reflection generation.

    phase-31.1 fix: previously instantiated `OutcomeTracker(settings)` with
    NO model parameter; `OutcomeTracker._generate_and_persist_reflections`
    is gated on `if self._model:` (outcome_tracker.py:147) so
    `bq.save_agent_memory` never fired in production -> `agent_memories`
    table stayed empty across 36+ days of cycles. Closes phase-30.0 Stage
    12 FAIL (the known separate-step issue disclosed in phase-30.3
    experiment_results.md).

    Resolution: construct a Gemini client via `make_client` and pass it
    to OutcomeTracker. `make_client` routes by model-name prefix:
    `gemini-*` -> Vertex/AI Studio; `claude-*` -> Anthropic; etc. Per
    backend/agents/memory.py::generate_reflection, the model is invoked
    via `model.generate_content(prompt, ...)`. The reflection-write to
    `agent_memories` has a fallback string when the LLM call errors
    (memory.py:248-254), so even Anthropic credit-balance failures still
    result in a non-empty lesson being persisted.

    Fail-open: if `make_client` raises (e.g., misconfigured keys), log
    at WARNING and proceed with `model=None` -- preserves the legacy
    behavior of NOT writing agent_memories rather than crashing the cycle.
    """
    from backend.services.outcome_tracker import OutcomeTracker

    # phase-31.1: try to construct a reflection-model client. Reads
    # `settings.gemini_model` (the misnamed standard-tier model field;
    # routes to Anthropic when set to "claude-*", to Gemini direct or
    # Vertex when set to "gemini-*"). Audit log: see phase-30.7 cycle +
    # phase-31.0.3 critical finding documenting the field misnomer.
    model_client = None
    try:
        from backend.agents.llm_client import make_client
        model_client = make_client(settings.gemini_model, None, settings)
        logger.info(
            "phase-31.1: OutcomeTracker reflection-model constructed "
            "(model=%s, provider routed by make_client)",
            settings.gemini_model,
        )
    except Exception as exc:
        logger.warning(
            "phase-31.1: OutcomeTracker model construction failed "
            "(agent_memories writes will be skipped this cycle): %r",
            exc,
        )

    tracker = OutcomeTracker(settings, model=model_client)

    # Get recent sell trades to find analysis_date, recommendation, and entry price
    recent_trades = bq.get_paper_trades(limit=50)
    sell_by_ticker = {}
    for t in recent_trades:
        if t.get("action") == "SELL" and t.get("ticker") in tickers:
            sell_by_ticker.setdefault(t["ticker"], t)

    learn_loop_enabled = bool(getattr(settings, "paper_learn_loop_enabled", False))

    for ticker in tickers:
        try:
            trade = sell_by_ticker.get(ticker)
            if not trade:
                logger.debug(f"No sell trade found for {ticker}, skipping outcome eval")
                continue
            analysis_date = trade.get("analysis_id") or trade.get("created_at", "")
            if hasattr(analysis_date, "isoformat"):
                analysis_date = analysis_date.isoformat()
            recommendation = trade.get("risk_judge_decision", "HOLD")
            # phase-35.1: stop_loss_trigger SELLs may have empty risk_judge_decision
            # (per BQ-probe B-5 in closure_roadmap §3); coerce empty string to a
            # neutral recommendation so OutcomeTracker doesn't barf downstream.
            if not recommendation or not str(recommendation).strip():
                recommendation = "HOLD"
            price_at_rec = trade.get("price", 0.0)
            outcome = tracker.evaluate_recommendation(
                ticker, str(analysis_date), recommendation, price_at_rec
            )

            # phase-35.1: writer fan-out (gated by paper_learn_loop_enabled).
            # Bug found in closure_roadmap §3 BQ-probe B-1/B-2: even when
            # closed_tickers fired (e.g. cycle c7801712 COHR stop-out 2026-05-22),
            # outcome_tracking and agent_memories stayed schema-empty because:
            #   (a) evaluate_recommendation early-returns None when yfinance
            #       current_price is missing -> NO write
            #   (b) evaluate_recommendation never calls
            #       _generate_and_persist_reflections -> NO agent_memories write
            # Fix: gate behind flag (default OFF per /goal gate 3); when ON,
            # write outcome_tracking via fallback path if evaluate_recommendation
            # returned None, AND call _generate_and_persist_reflections to land
            # agent_memories lesson rows.
            if not learn_loop_enabled:
                continue

            if outcome is None:
                # Fallback: build a minimal outcome dict from trade fields so
                # outcome_tracking gets a row even when yfinance flake or
                # missing analysis_date kills the primary path. Idempotent
                # via the (ticker, analysis_date) composite -- bq.save_outcome
                # is an UPSERT in the existing implementation.
                try:
                    sell_price = float(trade.get("price") or 0.0)
                    pnl_pct = float(trade.get("return_pct") or 0.0)
                    holding_days = int(trade.get("holding_days") or 0)
                    bq.save_outcome(
                        ticker=ticker,
                        analysis_date=str(analysis_date),
                        recommendation=recommendation,
                        price_at_rec=price_at_rec or sell_price,
                        current_price=sell_price,
                        return_pct=pnl_pct,
                        holding_days=holding_days,
                        beat_benchmark=(pnl_pct > 0),
                    )
                    outcome = {
                        "ticker": ticker,
                        "analysis_date": str(analysis_date),
                        "recommendation": recommendation,
                        "return_pct": pnl_pct,
                        "holding_days": holding_days,
                    }
                    logger.info(
                        "phase-35.1: fallback outcome_tracking row written for %s (sell_price=%s, pnl=%.2f%%, hold=%dd)",
                        ticker, sell_price, pnl_pct, holding_days,
                    )
                except Exception as fb_exc:
                    logger.warning(
                        "phase-35.1: fallback outcome_tracking write failed for %s: %r",
                        ticker, fb_exc,
                    )
                    outcome = None

            # agent_memories fan-out (writes one lesson row per
            # REFLECTION_AGENTS entry; fail-open per existing pattern in
            # _generate_and_persist_reflections).
            if outcome is not None:
                try:
                    full_report = {}
                    # Try to enrich the lesson with the original full report
                    # if it exists; pass {} when not found (lesson stays
                    # generic but still lands).
                    try:
                        stored = bq.get_report(ticker, str(analysis_date))
                        if stored and stored.get("full_report_json"):
                            fr = stored["full_report_json"]
                            if isinstance(fr, str):
                                import json as _json
                                full_report = _json.loads(fr) if fr else {}
                            elif isinstance(fr, dict):
                                full_report = fr
                    except Exception as fr_exc:
                        logger.debug(
                            "phase-35.1: full_report fetch failed for %s (using empty dict): %r",
                            ticker, fr_exc,
                        )

                    tracker._generate_and_persist_reflections(outcome, full_report)
                    logger.info(
                        "phase-35.1: agent_memories reflections fan-out fired for %s",
                        ticker,
                    )
                except Exception as ref_exc:
                    logger.warning(
                        "phase-35.1: agent_memories fan-out failed for %s: %r",
                        ticker, ref_exc,
                    )
        except Exception as e:
            logger.debug(f"Outcome evaluation failed for {ticker}: {e}")


def _log_cycle_signals_to_bq(bq, orders, today_str: str) -> int:
    """Write trade orders (or a HOLD heartbeat) to BQ signals_log.

    Ensures every daily cycle produces >= 1 row with event_kind='publish'
    so that the 4.4.2.4 signal reliability drill can verify coverage.
    Best-effort: never raises.
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    records = []

    for order in orders:
        if order.action not in ("BUY", "SELL"):
            continue
        sig_id = hashlib.sha1(
            f"{order.ticker}:{today_str}:{order.action}".encode()
        ).hexdigest()[:16]
        factors = order.signals if order.signals else ([order.reason] if order.reason else [])
        records.append({
            "signal_id": sig_id,
            "ticker": order.ticker,
            "signal_type": order.action,
            "confidence": 0.0,
            "signal_date": today_str,
            "entry_price": order.price or 0.0,
            "factors_json": json.dumps(factors, default=str),
            "created_at": now_iso,
            "outcome": "pending",
            "scored": False,
            "hit": None,
            "exit_price": None,
            "exit_date": None,
            "forward_return_pct": None,
            "holding_days": None,
            "recorded_at": now_iso,
            "event_kind": "publish",
        })

    if not records:
        sig_id = hashlib.sha1(
            f"HOLD:{today_str}:daily_cycle".encode()
        ).hexdigest()[:16]
        records.append({
            "signal_id": sig_id,
            "ticker": "$CYCLE",
            "signal_type": "HOLD",
            "confidence": 0.0,
            "signal_date": today_str,
            "entry_price": None,
            "factors_json": json.dumps(["no_trade_orders"]),
            "created_at": now_iso,
            "outcome": None,
            "scored": False,
            "hit": None,
            "exit_price": None,
            "exit_date": None,
            "forward_return_pct": None,
            "holding_days": None,
            "recorded_at": now_iso,
            "event_kind": "publish",
        })

    written = 0
    for rec in records:
        try:
            bq.save_signal(rec)
            written += 1
        except Exception as e:
            logger.warning(f"signals_log write failed for {rec['ticker']}: {type(e).__name__}")
    if written:
        logger.info(f"Logged {written} signal(s) to BQ signals_log for {today_str}")
    return written


def get_loop_status() -> dict:
    """Return current status of the autonomous loop."""
    return {
        "running": _running,
        "last_run": _last_run,
        "last_result": _last_result,
    }


def get_coordinator() -> MetaCoordinator:
    """Return the module-level MetaCoordinator instance."""
    return _coordinator


# 4.5.5: external callers (e.g. harness verification) import `run_cycle` as the
# canonical entry point. It's an alias for the established run_daily_cycle.
run_cycle = run_daily_cycle
