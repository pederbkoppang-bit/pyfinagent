"""
Paper Trading API — endpoints for autonomous virtual fund management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
from backend.services.api_cache import ENDPOINT_TTLS, get_api_cache
from backend.services.autonomous_loop import run_daily_cycle, get_loop_status
from backend.services.paper_metrics_v2 import compute_metrics_v2, persist_metrics_v2
from backend.services.paper_round_trips import compute_round_trips_response
from backend.services.live_prices import get_live_cache
from backend.services.cycle_health import compute_freshness, get_log as _get_cycle_log
from backend.services.kill_switch import evaluate_breach, get_state as _get_ks_state
from backend.services.paper_go_live_gate import compute_gate
from backend.services.reconciliation import compute_reconciliation
from backend.services.signal_attribution import group_signals_for_drawer, redact_pii
from backend.services.paper_trader import PaperTrader
from backend.services.perf_metrics import compute_sharpe_from_snapshots, compute_alpha

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])

# APScheduler instance (set by lifespan)
_scheduler = None
_scheduler_job_id = "paper_trading_daily"


# ── Models ───────────────────────────────────────────────────────

class StartRequest(BaseModel):
    starting_capital: Optional[float] = None


class KillSwitchActionRequest(BaseModel):
    """Confirmation token required for destructive actions (4.5.7)."""
    confirmation: str  # must equal the action name (e.g. "FLATTEN_ALL")


class DepositRequest(BaseModel):
    """phase-23.1.9: top up the virtual fund. Increments BOTH current_cash AND
    starting_capital so total_pnl_pct denominator stays anchored (a deposit is
    P&L-neutral by definition)."""
    amount: float = Field(..., gt=0, le=1_000_000, description="USD to deposit (0 < x <= $1M)")


class StartResponse(BaseModel):
    status: str
    portfolio_id: str
    starting_capital: float
    scheduler_active: bool


# ── Endpoints ────────────────────────────────────────────────────

@router.post("/start", response_model=StartResponse)
async def start_paper_trading(req: StartRequest = StartRequest()):
    """Initialize paper portfolio and start the autonomous scheduler."""
    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    # Initialize portfolio (idempotent)
    portfolio = await asyncio.to_thread(trader.get_or_create_portfolio)

    # Start scheduler if not already running
    scheduler_active = False
    if _scheduler and not _scheduler.get_job(_scheduler_job_id):
        _add_scheduler_job(settings)
        scheduler_active = True
    elif _scheduler and _scheduler.get_job(_scheduler_job_id):
        scheduler_active = True

    return StartResponse(
        status="active",
        portfolio_id=portfolio["portfolio_id"],
        starting_capital=portfolio["starting_capital"],
        scheduler_active=scheduler_active,
    )


@router.post("/stop")
async def stop_paper_trading():
    """Pause the autonomous trading scheduler."""
    get_api_cache().invalidate("paper:*")
    if _scheduler:
        job = _scheduler.get_job(_scheduler_job_id)
        if job:
            _scheduler.remove_job(_scheduler_job_id)
            return {"status": "stopped", "message": "Scheduler paused"}
    return {"status": "not_running"}


@router.get("/status")
async def get_status():
    """Current paper trading status: NAV, P&L, scheduler info."""
    cache = get_api_cache()
    cache_key = "paper:status"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
    if not portfolio:
        return {"status": "not_initialized", "message": "Call POST /start first"}

    positions = await asyncio.to_thread(trader.get_positions)
    loop_status = get_loop_status()

    scheduler_active = False
    if _scheduler and _scheduler.get_job(_scheduler_job_id):
        scheduler_active = True

    # Get next scheduled run time
    next_run = None
    if _scheduler:
        job = _scheduler.get_job(_scheduler_job_id)
        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

    result = {
        "status": "active" if scheduler_active else "paused",
        "portfolio": {
            "nav": portfolio.get("total_nav"),
            "cash": portfolio.get("current_cash"),
            "starting_capital": portfolio.get("starting_capital"),
            "pnl_pct": portfolio.get("total_pnl_pct"),
            "benchmark_return_pct": portfolio.get("benchmark_return_pct"),
            "inception_date": portfolio.get("inception_date"),
            "updated_at": portfolio.get("updated_at"),
        },
        "position_count": len(positions),
        "scheduler_active": scheduler_active,
        "next_run": next_run,
        "loop": loop_status,
        # Phase-4.6 smoketest alias: expose last_run timestamp at the top
        # level so 4.6.4 verifier can check last_run_ts without diving
        # into loop.*.
        "last_run_ts": loop_status.get("last_run"),
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:status"])
    return result


@router.get("/portfolio")
async def get_portfolio():
    """All open positions with current prices."""
    cache = get_api_cache()
    cache_key = "paper:portfolio"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
    if not portfolio:
        raise HTTPException(404, "Paper portfolio not initialized")

    positions = await asyncio.to_thread(trader.get_positions)

    # phase-23.1.13: backend-authoritative sector breakdown so the Risk Monitor
    # and any future monitoring (Slack, dashboards) read the same numbers.
    # Uses the cached _fetch_ticker_meta helper -- BQ-first / yfinance-fallback,
    # 24h cached -- so this adds near-zero latency to the portfolio endpoint.
    sector_breakdown: dict[str, dict] = {}
    try:
        tickers = [p["ticker"] for p in positions if p.get("ticker")]
        if tickers:
            meta_resp = await asyncio.to_thread(_fetch_ticker_meta, tickers, settings, bq)
            meta_map = (meta_resp or {}).get("meta", {})
            total_value = float(portfolio.get("total_nav") or 0.0)
            if total_value <= 0:
                total_value = sum(float(p.get("market_value") or 0.0) for p in positions) or 1.0
            for p in positions:
                t = p.get("ticker") or ""
                sector = (meta_map.get(t, {}) or {}).get("sector") or "Unknown"
                weight_pct = (float(p.get("market_value") or 0.0) / total_value) * 100.0
                row = sector_breakdown.setdefault(
                    sector, {"count": 0, "weight_pct": 0.0, "tickers": []},
                )
                row["count"] += 1
                row["weight_pct"] += weight_pct
                row["tickers"].append(t)
            for s in sector_breakdown.values():
                s["weight_pct"] = round(s["weight_pct"], 2)
    except Exception as e:
        logger.warning("Sector breakdown computation failed (non-fatal): %s", e)
        sector_breakdown = {}

    # phase-25.C12: backend-authoritative Sharpe so home + paper-trading
    # pages render identical numbers (closes phase-24.12 F-4 where the
    # home page's local kpiSharpe diverges by ~0.16 Sharpe units at 4% RFR
    # because it does NOT subtract the risk-free rate). Same canonical
    # helper as /performance at line 276; fail-open to None on snapshot
    # fetch failure. Field lives INSIDE the `portfolio` dict so frontend
    # `PaperPortfolio` interface stays the single point of extension.
    portfolio_sharpe: Optional[float] = None
    try:
        snapshots_for_sharpe = await asyncio.to_thread(bq.get_paper_snapshots, limit=365)
        if snapshots_for_sharpe:
            portfolio_sharpe = compute_sharpe_from_snapshots(snapshots_for_sharpe)
    except Exception as exc:
        logger.warning("get_portfolio: sharpe_ratio fail-open: %r", exc)
    portfolio["sharpe_ratio"] = portfolio_sharpe

    result = {
        "portfolio": portfolio,
        "positions": positions,
        "sector_breakdown": sector_breakdown,
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:portfolio"])
    return result


@router.get("/trades")
async def get_trades(limit: int = Query(100, ge=1, le=1000)):
    """Trade history."""
    cache = get_api_cache()
    cache_key = f"paper:trades:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    trades = await asyncio.to_thread(bq.get_paper_trades, limit=limit)
    result = {"trades": trades, "count": len(trades)}
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:trades"])
    return result


@router.get("/snapshots")
async def get_snapshots(limit: int = Query(365, ge=1, le=3650)):
    """Daily NAV history for charting."""
    cache = get_api_cache()
    cache_key = f"paper:snapshots:{limit}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    snapshots = await asyncio.to_thread(bq.get_paper_snapshots, limit=limit)
    result = {"snapshots": snapshots, "count": len(snapshots)}
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:snapshots"])
    return result


@router.get("/performance")
async def get_performance():
    """Performance metrics: win rate, avg return, alpha, Sharpe."""
    cache = get_api_cache()
    cache_key = "paper:performance"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)

    portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
    if not portfolio:
        raise HTTPException(404, "Paper portfolio not initialized")

    snapshots = await asyncio.to_thread(bq.get_paper_snapshots, limit=365)
    trades = await asyncio.to_thread(bq.get_paper_trades, limit=1000)

    # Compute metrics from trades
    sell_trades = [t for t in trades if t.get("action") == "SELL"]
    buy_trades = [t for t in trades if t.get("action") == "BUY"]

    # Rough P&L per closed trade (compare sell value to buy cost)
    total_trades = len(sell_trades)
    total_cost = sum(t.get("analysis_cost_today", 0) for t in snapshots)

    # Sharpe from NAV snapshots — delegates to canonical risk-free-adjusted formula
    sharpe = compute_sharpe_from_snapshots(snapshots)

    pnl_pct = portfolio.get("total_pnl_pct", 0) or 0
    bench_pct = portfolio.get("benchmark_return_pct", 0) or 0

    # 4.5.2: include round-trip summary (win_rate, profit_factor, expectancy,
    # median_holding_days, avg_mfe/avg_mae/avg_capture_ratio) inline so the
    # existing /performance caller gets the new fields without a second round trip.
    from backend.services.paper_round_trips import pair_round_trips, summarize
    rts = pair_round_trips(trades)
    rt_summary = summarize(rts)

    result = {
        "nav": portfolio.get("total_nav"),
        "starting_capital": portfolio.get("starting_capital"),
        "pnl_pct": pnl_pct,
        "benchmark_return_pct": bench_pct,
        "alpha_pct": compute_alpha(pnl_pct, bench_pct),
        "sharpe_ratio": sharpe,
        "total_sell_trades": total_trades,
        "total_buy_trades": len(buy_trades),
        "total_analysis_cost": round(total_cost, 2),
        "days_active": len(snapshots),
        "round_trip_summary": rt_summary,
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:performance"])
    return result


# phase-25.S: per-ticker P&L attribution + proportional LLM cost split.
# Closes phase-24.13 F-6 (SHARP arxiv finding: attribution is load-bearing for
# alpha; no published per-ticker pnl_per_cost_usd metric existed -- pyfinagent
# is a first-mover on the per-ticker variant of the 25.Q aggregate metric).
def _compute_attribution(bq: BigQueryClient, window_days: int) -> dict:
    """Aggregate per-ticker realized P&L + proportional LLM cost split.

    LLM cost is split across tickers in proportion to each ticker's share of
    paper_trades rows in the window. This is a first-pass approximation;
    per-call ticker tagging in `llm_call_log` is a follow-up step (25.S.1).

    Returns a dict with `per_ticker` (list), `totals`, `window_days`,
    `computed_at`, and a `note` documenting the approximation.
    """
    from collections import Counter, defaultdict
    from backend.services.paper_round_trips import pair_round_trips
    from backend.api.sovereign_api import _fetch_llm_cost_by_provider

    trades = bq.get_paper_trades_in_window(window_days) or []
    round_trips = pair_round_trips(trades)

    pnl_by_ticker: dict[str, float] = defaultdict(float)
    rt_count: Counter = Counter()
    for rt in round_trips:
        t = rt.get("ticker") or ""
        pnl_by_ticker[t] += float(rt.get("realized_pnl_usd") or 0.0)
        rt_count[t] += 1

    analysis_count: Counter = Counter()
    for tr in trades:
        t = tr.get("ticker") or ""
        analysis_count[t] += 1
    total_analyses = sum(analysis_count.values())

    llm_costs = _fetch_llm_cost_by_provider(window_days)
    total_llm = (
        float(llm_costs.get("anthropic", 0.0))
        + float(llm_costs.get("vertex", 0.0))
        + float(llm_costs.get("openai", 0.0))
    )

    per_ticker: list[dict] = []
    for t in sorted({*pnl_by_ticker.keys(), *analysis_count.keys()}):
        n_an = int(analysis_count.get(t, 0))
        pnl = float(pnl_by_ticker.get(t, 0.0))
        if total_analyses > 0 and total_llm > 0 and n_an > 0:
            ticker_llm = total_llm * (n_an / total_analyses)
        else:
            ticker_llm = 0.0
        ratio: Optional[float] = (pnl / ticker_llm) if ticker_llm > 0 else None
        per_ticker.append({
            "ticker": t,
            "realized_pnl_usd": round(pnl, 4),
            "llm_cost_usd": round(ticker_llm, 4),
            "pnl_per_cost_usd": round(ratio, 4) if ratio is not None else None,
            "n_round_trips": int(rt_count.get(t, 0)),
            "n_analyses": n_an,
        })

    total_pnl = sum(p["realized_pnl_usd"] for p in per_ticker)
    total_ratio: Optional[float] = (total_pnl / total_llm) if total_llm > 0 else None
    return {
        "window_days": window_days,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "per_ticker": per_ticker,
        "totals": {
            "realized_pnl_usd": round(total_pnl, 4),
            "llm_cost_usd": round(total_llm, 4),
            "pnl_per_cost_usd": round(total_ratio, 4) if total_ratio is not None else None,
        },
        "note": (
            "LLM cost split proportionally by analysis count per ticker "
            "(first pass; per-ticker tagging in llm_call_log is a follow-up step)."
        ),
    }


@router.get("/attribution")
async def get_attribution(window_days: int = Query(7, ge=1, le=365)):
    """phase-25.S: per-ticker realized P&L + LLM cost split.

    Closes phase-24.13 F-6 (SHARP attribution gap). Operators can answer
    "which tickers earned the most relative to LLM cost?" -- the per-ticker
    variant of 25.Q's aggregate profit_per_llm_dollar metric.
    """
    cache = get_api_cache()
    cache_key = f"paper:attribution:{window_days}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(_compute_attribution, bq, window_days)
    cache.set(cache_key, result, ENDPOINT_TTLS.get("paper:attribution", 300.0))
    return result


@router.get("/cycles/history")
async def get_cycles_history(limit: int = Query(10, ge=1, le=100)):
    """Return the last N autonomous-loop runs (JSONL tail at handoff/cycle_history.jsonl).
    Oracle-OAC-style fields: cycle_id, started_at, completed_at, duration_ms,
    status, error_count, n_trades, data_source_ages, bq_ingest_lag_sec."""
    rows = await asyncio.to_thread(_get_cycle_log().last_cycles, limit)
    return {"cycles": rows, "count": len(rows)}


@router.get("/freshness")
async def get_freshness():
    """
    CANONICAL freshness route. Signal-freshness strip payload:
    per-source last_tick_age, process heartbeat (dead-man's-switch
    control plane), BQ ingest lag, and the warn/critical ratio
    thresholds. UI drives colors from the `band` field.

    phase-16.22 added a thin alias at `/api/observability/freshness`
    that delegates to the same `compute_freshness` helper. This route
    is the canonical home; the alias exists only because the
    masterplan verification command pinned that prefix. Future
    consumers should hit THIS route. See `.claude/rules/backend-api.md`
    "Dual-route freshness" note for the rationale.
    """
    settings = get_settings()
    bq = BigQueryClient(settings)
    # Approximate cycle interval: the paper-trading scheduler runs daily at
    # paper_trading_hour, so the "normal" interval is 24h. Configurable via
    # settings.paper_cycle_interval_sec if future phases add one.
    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
    return await asyncio.to_thread(compute_freshness, bq, cycle_interval_sec)


@router.get("/kill-switch")
async def get_kill_switch_state():
    """
    Current kill-switch status: pause flag, limit breach booleans, thresholds.
    Read-only; poll this to drive UI badge + banner.
    """
    settings = get_settings()
    bq = BigQueryClient(settings)
    # phase-23.1.20: 5s timeout on the BQ portfolio fetch (same hardening as
    # /resume). On timeout we still return the in-memory pause state and a
    # null breach, so the UI badge stays informative.
    try:
        async with asyncio.timeout(5):
            portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
    except TimeoutError:
        logger.warning("get_kill_switch_state: BQ portfolio fetch timed out after 5s")
        portfolio = None
    nav = float((portfolio or {}).get("total_nav") or (portfolio or {}).get("starting_capital") or 0.0)
    breach = evaluate_breach(
        current_nav=nav,
        daily_loss_limit_pct=settings.paper_daily_loss_limit_pct,
        trailing_dd_limit_pct=settings.paper_trailing_dd_limit_pct,
    )
    state = _get_ks_state().snapshot()
    return {
        "paused": state["paused"],
        "pause_reason": state["pause_reason"],
        "sod_nav": state["sod_nav"],
        # phase-23.2.19: surface the SOD anchor date so the UI can show
        # when daily-loss% was last re-anchored (rolls every UTC midnight).
        "sod_date": state.get("sod_date"),
        "peak_nav": state["peak_nav"],
        "current_nav": nav,
        "breach": breach,
        "thresholds": {
            "daily_loss_limit_pct": settings.paper_daily_loss_limit_pct,
            "trailing_dd_limit_pct": settings.paper_trailing_dd_limit_pct,
        },
    }


@router.post("/pause")
async def pause_trading(req: KillSwitchActionRequest):
    if req.confirmation != "PAUSE":
        raise HTTPException(400, "Confirmation must equal PAUSE")
    get_api_cache().invalidate("paper:*")
    state = _get_ks_state().pause(trigger="manual")
    return {"status": "paused", "state": state}


@router.post("/resume")
async def resume_trading(req: KillSwitchActionRequest):
    if req.confirmation != "RESUME":
        raise HTTPException(400, "Confirmation must equal RESUME")
    # Resume precondition: both limits healthy. Prevents auto-re-entry after a
    # breach on brief recoveries (documented anti-pattern in RESEARCH.md 4.5.7).
    settings = get_settings()
    bq = BigQueryClient(settings)
    # phase-23.1.20: 5s timeout on the BQ breach-check so a slow / hung BQ
    # doesn't strand the user on a 30s frontend AbortController. On timeout
    # we return 503 + Retry-After so the UI can surface a useful error and
    # auto-retry per RFC 9110.
    try:
        async with asyncio.timeout(5):
            portfolio = await asyncio.to_thread(bq.get_paper_portfolio, "default")
    except TimeoutError:
        logger.warning("resume_trading: BQ breach-check timed out after 5s")
        return JSONResponse(
            status_code=503,
            content={
                "detail": "BigQuery breach-check timed out; retry in a few seconds",
                "error": "bq_timeout",
            },
            headers={"Retry-After": "5"},
        )
    nav = float((portfolio or {}).get("total_nav") or 0.0)
    breach = evaluate_breach(
        current_nav=nav,
        daily_loss_limit_pct=settings.paper_daily_loss_limit_pct,
        trailing_dd_limit_pct=settings.paper_trailing_dd_limit_pct,
    )
    if breach["any_breached"]:
        raise HTTPException(
            409,
            f"Cannot resume: limit still breached. "
            f"daily_loss={breach['daily_loss_pct']:.2f}% (limit {breach['daily_loss_limit_pct']}%), "
            f"trailing_dd={breach['trailing_dd_pct']:.2f}% (limit {breach['trailing_dd_limit_pct']}%)"
        )
    get_api_cache().invalidate("paper:*")
    state = _get_ks_state().resume(trigger="manual")
    return {"status": "resumed", "state": state}


@router.post("/flatten-all")
async def flatten_all(req: KillSwitchActionRequest):
    if req.confirmation != "FLATTEN_ALL":
        raise HTTPException(400, "Confirmation must equal FLATTEN_ALL")
    get_api_cache().invalidate("paper:*")
    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)
    result = await asyncio.to_thread(trader.flatten_all, "manual_flatten")
    # Also pause to enter known-quiet state (industry standard per 3forge + NYIF).
    _get_ks_state().pause(trigger="manual_flatten", details=result)
    return {"status": "flattened_and_paused", "result": result}


@router.get("/live-prices")
async def get_live_prices(tickers: str = Query(..., description="Comma-separated tickers")):
    """
    Cached, rate-limited intraday price fetch. Returns `{ticker: price|null}`.
    Drives the no-cycle chart refresh path (4.5.6). Frontend should poll this
    at 60s cadence while the tab is visible.
    """
    raw = [t.strip().upper() for t in (tickers or "").split(",") if t.strip()]
    if not raw:
        raise HTTPException(400, "Provide at least one ticker")
    if len(raw) > 50:
        raise HTTPException(400, "Too many tickers (max 50)")
    cache = get_live_cache()
    prices = await asyncio.to_thread(cache.get_many, raw)
    return {"prices": prices, "ttl_sec": 60, "count": len(prices)}


@router.get("/trades/{trade_id}/rationale")
async def get_trade_rationale(trade_id: str, full: bool = Query(True)):
    """
    Per-trade agent attribution tree for the rationale drawer (4.5.5).
    Parses the JSON `signals` column on the stored trade row and returns the
    progressive-disclosure hierarchy (analyst -> bull/bear -> trader -> risk).

    phase-25.E: `?full=1` (default) returns the full attribution tree. `?full=0`
    returns a compact subset: first Analyst row + Trader + RiskJudge. Other
    layers pruned to []. Closes audit bucket 24.4 F-3.
    """
    # Sanitize trade_id to avoid accidental injection into the parameterized
    # query downstream (also guarded by BQ bind parameters).
    if not all(c.isalnum() or c in ("-", "_") for c in trade_id):
        raise HTTPException(400, "Invalid trade_id")

    settings = get_settings()
    bq = BigQueryClient(settings)

    def _load():
        trades = bq.get_paper_trades(limit=5000) or []
        for t in trades:
            if t.get("trade_id") == trade_id:
                return t
        return None

    trade = await asyncio.to_thread(_load)
    if not trade:
        raise HTTPException(404, f"Trade {trade_id} not found")

    import json as _json
    raw = trade.get("signals")
    signals: list[dict] = []
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                signals = [s for s in parsed if isinstance(s, dict)]
        except Exception as e:
            logger.warning(f"rationale: could not parse signals for {trade_id}: {e}")

    # Defensive redact in case anything slipped through persistence.
    for s in signals:
        if "rationale" in s:
            s["rationale"] = redact_pii(s["rationale"])

    # phase-25.E: compact view returns only Analyst (first) + Trader + RiskJudge.
    # Other layers are pruned to []. Preserves the data shape so frontend doesn't
    # need a separate code path.
    if not full:
        kept_signals: list[dict] = []
        analyst_seen = False
        for s in signals:
            agent = s.get("agent", "")
            if agent == "Analyst" and not analyst_seen:
                kept_signals.append(s)
                analyst_seen = True
            elif agent in ("Trader", "RiskJudge"):
                kept_signals.append(s)
        signals = kept_signals

    tree = group_signals_for_drawer(signals)
    return {
        "trade_id": trade_id,
        "ticker": trade.get("ticker"),
        "action": trade.get("action"),
        "created_at": trade.get("created_at"),
        "reason": trade.get("reason"),
        "signals": signals,
        "tree": tree,
        "full": full,
    }


@router.get("/gate")
async def get_gate():
    """
    Go-Live Gate: 5 deterministic booleans + promote_eligible. See
    backend/services/paper_go_live_gate.py for threshold definitions. The UI
    "Promote to live" button is disabled unless promote_eligible is true.
    """
    cache = get_api_cache()
    cache_key = "paper:gate"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(compute_gate, bq)
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:gate"])
    return result


@router.get("/reconciliation")
async def get_reconciliation():
    """
    Paper-live vs parallel OOS-backtest NAV reconciliation. Returns a per-date
    time series and a summary with `latest_divergence_pct` + an `alert` flag
    when divergence exceeds 5%. See backend/services/reconciliation.py for the
    shadow-backtest definition (frictionless fills on yfinance adj-close).
    """
    cache = get_api_cache()
    cache_key = "paper:reconciliation"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(compute_reconciliation, bq)
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:reconciliation"])
    return result


# phase-25.A11: kill-switch audit JSONL location, mirrored from
# backend/services/kill_switch.py::_AUDIT_PATH. Reading is local-file only,
# no BQ required.
_KILL_SWITCH_AUDIT_PATH = Path(__file__).resolve().parents[2] / "handoff" / "kill_switch_audit.jsonl"


def _compute_learnings(bq: BigQueryClient, window_days: int) -> dict:
    """phase-25.A11: build the VirtualFundLearningsData response shape.

    Three sections, each independent so a failure in one does not nuke the
    other two (component already handles empty arrays gracefully):
      1. reconciliation_divergences -- per round-trip drift between entry
         (sim proxy) and exit (paper fill).
      2. kill_switch_triggers -- aggregate counts from the local audit
         JSONL filtered to event=pause within the window.
      3. regime_buckets -- empty for first pass (no per-trade regime tag
         column in paper_trades today; closes phase-24.11 F-1 without a
         schema migration).
    """
    from collections import Counter
    from backend.services.paper_round_trips import pair_round_trips

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    # 1. Reconciliation divergences via round-trip pairing.
    divergences: list[dict] = []
    try:
        trades = bq.get_paper_trades_in_window(window_days) or []
        round_trips = pair_round_trips(trades)
        for rt in round_trips:
            entry_price = float(rt.get("entry_price") or 0.0)
            exit_price = float(rt.get("exit_price") or 0.0)
            if entry_price <= 0:
                continue
            drift_pct = (exit_price - entry_price) / entry_price * 100.0
            divergences.append({
                "symbol": rt.get("ticker"),
                "side": "sell",
                "paper_fill": round(exit_price, 4),
                "sim_fill": round(entry_price, 4),
                "drift_pct": round(drift_pct, 4),
                "ts": rt.get("exit_date") or "",
            })
        divergences.sort(key=lambda d: abs(d["drift_pct"]), reverse=True)
        divergences = divergences[:100]
    except Exception as e:
        logger.warning("get_learnings: divergence computation failed: %s", e)

    # 2. Kill-switch trigger distribution from the local audit JSONL.
    trigger_counts: Counter = Counter()
    try:
        if _KILL_SWITCH_AUDIT_PATH.exists():
            with _KILL_SWITCH_AUDIT_PATH.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") != "pause":
                        continue
                    ts_raw = row.get("ts") or row.get("timestamp")
                    if not ts_raw:
                        continue
                    try:
                        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        continue
                    trigger_counts[row.get("trigger") or "unknown"] += 1
    except Exception as e:
        logger.warning("get_learnings: kill-switch audit read failed: %s", e)

    kill_switch_triggers = [
        {"reason": reason, "count": count}
        for reason, count in trigger_counts.most_common()
    ]

    # 3. Regime buckets -- per-trade regime tag not persisted today;
    # documented gap, return empty array. The component empty-state
    # already shows "No regime buckets computed yet."
    regime_buckets: list[dict] = []
    logger.info(
        "get_learnings: regime_buckets empty (paper_trades has no per-trade "
        "regime column; populate in a follow-up step)"
    )

    return {
        "reconciliation_divergences": divergences,
        "kill_switch_triggers": kill_switch_triggers,
        "regime_buckets": regime_buckets,
        "window_days": window_days,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/learnings")
async def get_learnings(window_days: int = Query(30, ge=1, le=365)):
    """phase-25.A11: virtual-fund learnings summary for the paper-trading
    learnings page. Three sections: reconciliation drift per round-trip,
    kill-switch trigger distribution, and regime buckets. Closes
    phase-24.11 F-1 (orphan UI page wired to live backend).
    """
    cache = get_api_cache()
    cache_key = f"paper:learnings:{window_days}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(_compute_learnings, bq, window_days)
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:learnings"])
    return result


@router.get("/mfe-mae-scatter")
async def get_mfe_mae_scatter():
    """
    Exit-quality scatter (4.5.9). Per-trade points with MFE, |MAE|, capture_ratio,
    and a leakage flag. Edge-Ratio per-trade (mfe/|mae|) averaged. Leakage rule:
    capture_ratio < 0.4 AND mfe_pct > P75 (see RESEARCH.md 4.5.9).
    """
    cache = get_api_cache()
    cache_key = "paper:mfe_mae_scatter"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    from backend.services.paper_round_trips import pair_round_trips

    settings = get_settings()
    bq = BigQueryClient(settings)
    trades = await asyncio.to_thread(bq.get_paper_trades, 2000)
    trades = trades or []
    rts = pair_round_trips(trades)

    # Per-trade edge_ratio + leakage computation.
    points = []
    edge_ratios: list[float] = []
    mfes: list[float] = []
    for rt in rts:
        mfe = float(rt.get("mfe_pct") or 0.0)
        mae_abs = abs(float(rt.get("mae_pct") or 0.0))
        capture = float(rt.get("capture_ratio") or 0.0)
        if mae_abs > 0:
            edge_ratios.append(mfe / mae_abs)
        mfes.append(mfe)
        points.append({
            "ticker": rt.get("ticker"),
            "entry_date": rt.get("entry_date"),
            "exit_date": rt.get("exit_date"),
            "mfe_pct": round(mfe, 4),
            "mae_pct": round(float(rt.get("mae_pct") or 0.0), 4),
            "mae_abs_pct": round(mae_abs, 4),
            "capture_ratio": round(capture, 4),
            "realized_pnl_pct": float(rt.get("realized_pnl_pct") or 0.0),
            "holding_days": int(rt.get("holding_days") or 0),
            "leakage_flag": False,  # filled below once P75 is known
        })

    n = len(points)
    mfe_p75 = None
    n_leakers = 0
    if n >= 10:
        sorted_mfe = sorted(mfes)
        # P75 index = ceil(0.75 * n) - 1 (inclusive upper-quartile value).
        idx = max(0, int(round(0.75 * (n - 1))))
        mfe_p75 = sorted_mfe[idx]
        for p in points:
            if p["capture_ratio"] < 0.4 and p["mfe_pct"] > mfe_p75:
                p["leakage_flag"] = True
                n_leakers += 1

    edge_ratio = (sum(edge_ratios) / len(edge_ratios)) if edge_ratios else 0.0
    avg_capture = (sum(p["capture_ratio"] for p in points) / n) if n else 0.0

    result = {
        "points": points,
        "summary": {
            "edge_ratio": round(edge_ratio, 4),
            "avg_capture_ratio": round(avg_capture, 4),
            "mfe_p75": round(mfe_p75, 4) if mfe_p75 is not None else None,
            "leakage_threshold_capture": 0.4,
            "n_points": n,
            "n_leakers": n_leakers,
        },
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:mfe_mae_scatter"])
    return result


@router.get("/round-trips")
async def get_round_trips():
    """
    Round-trip (BUY->SELL) performance: win_rate, profit_factor, expectancy,
    median holding days, average MFE/MAE/capture_ratio. Pairs historic trades
    FIFO per ticker (see backend/services/paper_round_trips.py).
    """
    cache = get_api_cache()
    cache_key = "paper:round_trips"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    result = await asyncio.to_thread(compute_round_trips_response, bq)
    cache.set(cache_key, result, ENDPOINT_TTLS["paper:round_trips"])
    return result


@router.get("/metrics-v2")
async def get_metrics_v2():
    """
    Evaluation-grade metrics: PSR, DSR, Sortino, Calmar, and bootstrap 95% CI
    on rolling Sharpe. Sourced from backend.services.perf_metrics (single source
    of truth) via backend.services.paper_metrics_v2 orchestrator.
    """
    cache = get_api_cache()
    cache_key = "paper:metrics_v2"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    settings = get_settings()
    bq = BigQueryClient(settings)
    metrics = await asyncio.to_thread(compute_metrics_v2, bq)
    await asyncio.to_thread(persist_metrics_v2, bq, metrics)

    cache.set(cache_key, metrics, ENDPOINT_TTLS["paper:metrics_v2"])
    return metrics


@router.post("/run-now")
async def run_now(dry_run: bool = False):
    """Manually trigger one daily cycle (for testing).

    dry_run=true is a phase-4.6 smoketest hook: stamps _last_run=now and
    returns without running the real cycle (no LLM calls, no BQ writes,
    no trades). Use it from CI / the smoketest only; never from prod.
    """
    status = get_loop_status()
    if status["running"]:
        raise HTTPException(409, "A trading cycle is already in progress")

    get_api_cache().invalidate("paper:*")
    settings = get_settings()
    if dry_run:
        await run_daily_cycle(settings, dry_run=True)
        return {"status": "ok", "started": True, "dry_run": True,
                "message": "Dry-run cycle completed (no trades)"}
    asyncio.create_task(_run_cycle_background(settings))
    return {"status": "started", "started": True, "message": "Daily cycle triggered"}


_last_cycle_error: Optional[str] = None


# phase-23.1.10 — ticker metadata (company name + sector) for Positions/Trades tables.
# BQ-first lookup against analysis_results (zero rate-limit risk); yfinance fallback for
# tickers we've never analyzed before. 24h cache via paper:ticker_meta TTL.

def _yfinance_ticker_info(ticker: str) -> dict:
    """Fetch company_name + sector from yfinance. Graceful on any error."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or ""
        return {"company_name": name, "sector": sector, "source": "yfinance"}
    except Exception as e:  # network / rate-limit / unknown ticker
        logger.debug("yfinance lookup failed for %s: %s", ticker, e)
        return {"company_name": ticker, "sector": "", "source": "error"}


def _fetch_ticker_meta(tickers: list[str], settings, bq) -> dict:
    """Resolve {ticker -> {company_name, sector, source}}.

    Strategy (phase-32.5 update):
      1. Single BQ query that UNIONs paper_positions (priority 1) and
         analysis_results (priority 2), ranks per ticker, returns the
         highest-priority non-null name. paper_positions wins because
         phase-32.4's backfill_missing_company_names keeps it canonical
         and the autonomous loop refreshes it every cycle; analysis_results
         may be stale or absent for some tickers.
      2. For tickers missing OR missing sector, fall through to yfinance
         (slow but caches for 24h via the route's get_api_cache).
    """
    out: dict[str, dict] = {}
    if not tickers:
        return {"meta": out, "ttl_sec": 86400, "count": 0}

    # Step 1: combined BQ query (paper_positions priority over analysis_results).
    # Skipping NULL or ticker-as-name sentinels at the SQL layer prevents the
    # 'legacy fallback' values from outranking a real name from the secondary
    # source. ROW_NUMBER() picks the lowest priority number per ticker.
    try:
        dataset = f"{settings.gcp_project_id}.{settings.bq_dataset_reports}"
        query = f"""
            WITH combined AS (
              SELECT
                ticker,
                company_name,
                sector,
                1 AS priority
              FROM `{dataset}.paper_positions`
              WHERE ticker IN UNNEST(@tickers)
                AND company_name IS NOT NULL
                AND company_name != ''
                AND company_name != ticker
              UNION ALL
              SELECT
                ticker,
                company_name,
                sector,
                2 AS priority
              FROM `{dataset}.analysis_results`
              WHERE ticker IN UNNEST(@tickers)
                AND company_name IS NOT NULL
                AND company_name != ''
                AND company_name != ticker
            ),
            ranked AS (
              SELECT
                ticker,
                company_name,
                sector,
                priority,
                ROW_NUMBER() OVER (
                  PARTITION BY ticker
                  ORDER BY priority, sector IS NULL
                ) AS rn
              FROM combined
            )
            SELECT ticker, ANY_VALUE(company_name) AS company_name,
                   ANY_VALUE(sector) AS sector, ANY_VALUE(priority) AS priority
            FROM ranked
            WHERE rn = 1
            GROUP BY ticker
        """
        from google.cloud import bigquery as _bq
        job_config = _bq.QueryJobConfig(query_parameters=[
            _bq.ArrayQueryParameter("tickers", "STRING", tickers),
        ])
        rows = list(bq.client.query(query, job_config=job_config).result())
        for r in rows:
            t = r["ticker"]
            name = r["company_name"]
            sect = r["sector"]
            priority = r["priority"]
            if t and name:
                out[t] = {
                    "company_name": name,
                    "sector": sect or "",
                    # phase-32.5: surface which BQ source won so operators
                    # can audit cache hits / source attribution.
                    "source": "paper_positions" if priority == 1 else "analysis_results",
                }
    except Exception as e:
        logger.warning("ticker-meta BQ lookup failed (graceful fallback): %s", e)

    # Step 2: yfinance fallback for tickers missing OR missing sector.
    # phase-23.1.16: parallel via ThreadPoolExecutor (max_workers=5) instead
    # of the previous serial loop with sleep(0.3). yfinance has no batch-info
    # API; per-Ticker is the only path. Per-thread Ticker objects are
    # thread-safe (the bug in #2557 is in yfinance.download, not Ticker.info).
    # Wall-clock for 14 tickers: ~18s -> ~3s.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    tickers_needing_yf = [
        t for t in tickers
        if out.get(t) is None or not out.get(t, {}).get("sector")
    ]
    if tickers_needing_yf:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_yfinance_ticker_info, t): t
                for t in tickers_needing_yf
            }
            for future in as_completed(futures):
                t = futures[future]
                try:
                    info = future.result()
                except Exception as e:
                    logger.warning("yfinance worker failed for %s: %s", t, e)
                    continue
                existing = out.get(t)
                if existing:
                    existing["sector"] = existing.get("sector") or info["sector"]
                    existing["source"] = "bq+yf" if existing["sector"] else "bq"
                else:
                    out[t] = info

    return {"meta": out, "ttl_sec": 86400, "count": len(out)}


@router.get("/ticker-meta")
async def get_ticker_meta(tickers: str = Query(...)):
    """Return {ticker -> {company_name, sector}} for the comma-separated tickers.

    phase-23.1.16: per-ticker cache keys so adding/removing one position
    doesn't bust the whole 24h cache. Each resolved ticker is stored at
    `paper:ticker_meta:single:{TICKER}`. On a request, we look up each
    ticker individually; only the missing subset goes through
    `_fetch_ticker_meta` (BQ-first / parallel-yfinance fallback).

    Cached 24h. Max 50 tickers per call.
    """
    raw = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not raw:
        raise HTTPException(400, "Provide at least one ticker")
    if len(raw) > 50:
        raise HTTPException(400, "Max 50 tickers per request")

    settings = get_settings()
    cache = get_api_cache()

    out_meta: dict[str, dict] = {}
    missing: list[str] = []
    for t in raw:
        cached = cache.get(f"paper:ticker_meta:single:{t}")
        if cached is not None:
            out_meta[t] = cached
        else:
            missing.append(t)

    if missing:
        bq = BigQueryClient(settings)
        result = await asyncio.to_thread(_fetch_ticker_meta, missing, settings, bq)
        ttl = ENDPOINT_TTLS["paper:ticker_meta"]
        for t, info in (result.get("meta") or {}).items():
            out_meta[t] = info
            cache.set(f"paper:ticker_meta:single:{t}", info, ttl)

    return {"meta": out_meta, "ttl_sec": 86400, "count": len(out_meta)}


@router.post("/deposit")
async def deposit_funds(req: DepositRequest):
    """Top up the virtual paper-trading fund.

    phase-23.1.9: increments BOTH `current_cash` AND `starting_capital` so
    `total_pnl_pct = (nav - starting_capital) / starting_capital * 100` stays
    anchored. A deposit is P&L-neutral by definition; without bumping the
    starting_capital the operator would see a fake gain.
    """
    settings = get_settings()
    bq = BigQueryClient(settings)
    trader = PaperTrader(settings, bq)

    portfolio = await asyncio.to_thread(trader.get_or_create_portfolio)
    balance_before = float(portfolio.get("current_cash") or 0.0)
    starting_before = float(portfolio.get("starting_capital") or 0.0)
    nav_before = float(portfolio.get("total_nav") or starting_before)

    new_cash = round(balance_before + req.amount, 2)
    new_starting = round(starting_before + req.amount, 2)
    new_nav = round(nav_before + req.amount, 2)
    new_pnl_pct = (
        round(((new_nav - new_starting) / new_starting) * 100, 4)
        if new_starting > 0 else 0.0
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    updated = {
        **portfolio,
        "current_cash": new_cash,
        "starting_capital": new_starting,
        "total_nav": new_nav,
        "total_pnl_pct": new_pnl_pct,
        "updated_at": now_iso,
    }
    await asyncio.to_thread(bq.upsert_paper_portfolio, updated)
    get_api_cache().invalidate("paper:*")

    logger.info(
        "[paper_trading] deposit accepted: amount=$%.2f portfolio=%s "
        "cash %.2f -> %.2f, starting %.2f -> %.2f, nav %.2f -> %.2f",
        req.amount, portfolio.get("portfolio_id", "default"),
        balance_before, new_cash, starting_before, new_starting, nav_before, new_nav,
    )

    return {
        "status": "deposited",
        "amount": req.amount,
        "new_cash": new_cash,
        "new_starting_capital": new_starting,
        "new_nav": new_nav,
        "new_pnl_pct": new_pnl_pct,
        "deposited_at": now_iso,
    }


async def _run_cycle_background(settings):
    """Run daily cycle as async task on the event loop."""
    global _last_cycle_error
    try:
        result = await run_daily_cycle(settings)
        logger.info(f"Manual paper trading cycle result: {result.get('status')}")
        _last_cycle_error = None
    except Exception as e:
        logger.error(f"Manual cycle failed: {e}", exc_info=True)
        _last_cycle_error = f"{type(e).__name__}: {e}"


# ── Scheduler Integration ────────────────────────────────────────

def init_scheduler(scheduler):
    """Called from app lifespan to wire up APScheduler."""
    global _scheduler
    _scheduler = scheduler
    settings = get_settings()
    if settings.paper_trading_enabled:
        _add_scheduler_job(settings)
        logger.info(f"Paper trading scheduler active: daily at {settings.paper_trading_hour}:00 ET")


def _add_scheduler_job(settings):
    if not _scheduler:
        return
    _scheduler.add_job(
        _scheduled_run,
        "cron",
        hour=settings.paper_trading_hour,
        minute=0,
        day_of_week="mon-fri",
        timezone=ZoneInfo("America/New_York"),
        id=_scheduler_job_id,
        name="Paper trading daily run",  # phase-23.3.1: human-readable label
        replace_existing=True,
        # phase-44.2.X (2026-05-26): default APScheduler misfire_grace_time
        # is 1 second; on 2026-05-25 the cron fired at the right second but
        # event-loop contention from the ticket-queue interval job + polling
        # endpoints pushed dispatch 2.10s late, so APScheduler skipped the
        # run and advanced next_run to tomorrow. A daily job has no harm in
        # running a few seconds (or minutes) late, so we raise the grace
        # window to 1 hour. coalesce=True ensures if multiple windows are
        # missed (e.g. backend down for hours), we run ONCE, not N times.
        misfire_grace_time=3600,
        coalesce=True,
    )


async def _scheduled_run():
    """Wrapper for the scheduler (APScheduler calls this)."""
    settings = get_settings()
    try:
        result = await run_daily_cycle(settings)
        logger.info(f"Scheduled paper trading result: {result.get('status')}")
    except Exception as e:
        logger.error(f"Scheduled paper trading failed: {e}", exc_info=True)
