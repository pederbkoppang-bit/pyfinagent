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
from backend.services.portfolio_manager import (
    decide_trades,
    # phase-86.86 (D6): the lite producer reuses the SAME three-state verdict
    # rule as the full path rather than growing a second parallel idiom.
    SIZE,
    UNPARSEABLE,
    _resolve_position_pct,
)
from backend.tools.screener import screen_universe, rank_candidates, get_sp500_tickers, get_russell1000_tickers
from backend.services.recommendation_vocab import resolve_outcome_recommendation  # phase-86.25

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


# phase-70.4 (G1-C): the EFFECTIVE per-cycle session ceiling. Defaults to the
# hidden $1.00 module const (byte-identical); run_daily_cycle raises it to the
# operator-visible paper_max_daily_cost_usd when paper_session_budget_reconcile_enabled
# is ON. Read by _check_session_budget so the reconcile is honored.
_effective_session_budget: float = _SESSION_BUDGET_USD


def _check_session_budget(stage: str = "pre_call") -> None:
    """phase-26.1: raise BudgetBreachError if cumulative session LLM cost
    has reached the per-cycle ceiling. Called before LLM-heavy steps.
    Lazy-imports BudgetBreachError to avoid module-load coupling."""
    if _session_cost >= _effective_session_budget:
        from backend.agents.llm_client import BudgetBreachError
        # phase-70.4 (G1-A): log the breach BEFORE raising so a cost-truncated cycle
        # is NEVER silent (the raise is swallowed by gather(return_exceptions=True)).
        logger.warning(
            "phase-70.4: SESSION BUDGET BREACH -- cumulative $%.4f >= ceiling $%.4f "
            "(stage=%s, cycle=%s); remaining candidates this cycle are SKIPPED "
            "(cost-truncated, NOT a no-signal cycle)",
            _session_cost, _effective_session_budget, stage, _current_cycle_id,
        )
        raise BudgetBreachError(
            f"session_budget_breach: cumulative ${_session_cost:.4f} "
            f">= ceiling ${_effective_session_budget:.4f} (stage={stage}, "
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


def cycle_halt_reason(ks_check: dict, is_paused: bool) -> Optional[str]:
    """Should this cycle skip decide/execute, and why?

    Extracted from `run_daily_cycle`'s Step 5.5 in phase-36.12 so the wiring that
    converts a kill-switch verdict into "no orders placed" is BEHAVIOURALLY
    testable. It was previously an inline three-term boolean, and the only guard
    on it was a source scan -- which a cycle-1 Q/A defeated by keeping the literal
    `ks_check.get("blocked")` while neutering it (`and False`). A source scan
    cannot see that; this function can be called.

    Precedence matters and is asserted by tests: a real breach outranks a
    lost-history block, which outranks an already-paused state, because the
    reason is what the operator reads in the log line.

    Returns None when the cycle may proceed.
    """
    if ks_check.get("triggered"):
        return "breach"
    if ks_check.get("blocked"):
        # phase-36.12: the third halt reason. A cycle that came up DISARMED on a
        # book with prior history neither breached nor paused, so the original
        # two-term condition let it trade on baselines that had just been
        # silently anchored to today's NAV.
        return str(ks_check.get("block_reason") or "blocked")
    if is_paused:
        return "paused"
    return None


def _min_k_sector_slice(candidates: list[dict], n: int, k: int) -> list[dict]:
    """phase-70.2: pick n candidates that span >=min(k, #distinct-sectors) GICS
    sectors, best-effort, WITHOUT hard-neutralizing the momentum order (S2).

    `candidates` arrive in composite-score-descending order. Bucket by sector
    preserving that order; take the LEADER (highest-scored remaining) of each of
    the k highest-peak distinct sectors first (guarantees the sector spread), then
    fill the remaining slots by pure score, then re-sort the picked set by
    composite_score desc so the analyzer still sees the best names first
    (arXiv 2408.09168 multinomial-blend leader-pick). Graceful when fewer than k
    sectors exist. Callers pass k>0; k=0 keeps the plain top-N slice.
    """
    if n <= 0 or not candidates:
        return candidates[: max(n, 0)]
    from collections import OrderedDict
    buckets: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in candidates:
        sec = (c.get("sector") or "").strip() or "Unknown"
        buckets.setdefault(sec, []).append(c)
    sectors_by_peak = sorted(
        buckets.keys(),
        key=lambda s: buckets[s][0].get("composite_score") or 0.0,
        reverse=True,
    )
    picked: list[dict] = []
    picked_ids: set[int] = set()
    for sec in sectors_by_peak[:k]:
        if len(picked) >= n:
            break
        lead = buckets[sec][0]
        picked.append(lead)
        picked_ids.add(id(lead))
    for c in candidates:
        if len(picked) >= n:
            break
        if id(c) in picked_ids:
            continue
        picked.append(c)
        picked_ids.add(id(c))
    picked.sort(key=lambda c: c.get("composite_score") or 0.0, reverse=True)
    return picked[:n]


def _execute_swap_pair(trader, sell_order, buy_order) -> dict:
    """phase-70.3: execute an atomic swap pair (paper_atomic_swap_enabled). Runs the
    pair BUY-first-with-reserved-cash AFTER a SELL-feasibility pre-check so a SELL can
    NEVER persist while its paired BUY is dropped (the net -1-position bug). The common
    failure (BUY drops for cash/price/max-pos/idempotency) means the SELL is simply not
    attempted -> book unchanged, NO ledger reversal. The SELL-fails-after-BUY branch is
    unreachable given the pre-check (defensive compensation only). Returns
    {sold, bought, reason}. Sync (called via asyncio.to_thread)."""
    from backend.services.paper_trader import _get_live_price, _fx_local_to_usd
    # 1. SELL-feasibility pre-check: position exists + FX resolvable. execute_sell would
    #    return None otherwise -> we must NOT execute the BUY (keep atomicity).
    sell_pos = trader.get_position(sell_order.ticker)
    if not sell_pos or _fx_local_to_usd(sell_pos.get("market")) is None:
        logger.info(
            "phase-70.3 swap %s->%s: SELL leg infeasible (no position / FX) -- dropping BOTH legs",
            sell_order.ticker, buy_order.ticker,
        )
        return {"sold": False, "bought": False, "reason": "sell_infeasible"}
    freed = float(sell_pos.get("market_value", 0.0) or 0.0)
    # 2. BUY-first with the paired SELL's value reserved (live price for the fill).
    live = _get_live_price(buy_order.ticker) or 0
    price = live if live > 0 else (buy_order.price_at_analysis or buy_order.price or 0)
    if price <= 0:
        logger.warning("phase-70.3 swap: no price for BUY %s -- dropping BOTH legs", buy_order.ticker)
        return {"sold": False, "bought": False, "reason": "buy_no_price"}
    buy_trade = trader.execute_buy(
        ticker=buy_order.ticker,
        amount_usd=buy_order.amount_usd or 0,
        price=price,
        reason=buy_order.reason,
        analysis_id=buy_order.analysis_id,
        risk_judge_decision=buy_order.risk_judge_decision,
        stop_loss_price=buy_order.stop_loss_price,
        risk_judge_position_pct=buy_order.risk_judge_position_pct,
        signals=buy_order.signals,
        sector=buy_order.sector or None,
        market=getattr(buy_order, "market", "US"),
        price_at_analysis=buy_order.price_at_analysis,
        factor_loadings=buy_order.factor_loadings,
        analysis_recommendation=getattr(buy_order, "analysis_recommendation", ""),
        reserved_cash=freed,
    )
    if not buy_trade:
        logger.info(
            "phase-70.3 swap %s->%s: BUY dropped -- SELL NOT attempted (atomic, book unchanged)",
            sell_order.ticker, buy_order.ticker,
        )
        return {"sold": False, "bought": False, "reason": "buy_dropped"}
    # 3. SELL (pre-validated feasible).
    sell_trade = trader.execute_sell(
        ticker=sell_order.ticker,
        quantity=sell_order.quantity,
        price=sell_order.price,
        reason=sell_order.reason,
        signals=sell_order.signals,
    )
    if not sell_trade:
        logger.error(
            "phase-70.3 swap %s->%s: SELL failed AFTER BUY despite pre-check -- "
            "compensating by removing the just-created BUY position",
            sell_order.ticker, buy_order.ticker,
        )
        try:
            trader.bq.delete_paper_position(buy_order.ticker)
        except Exception as e:
            logger.error("phase-70.3 swap compensation delete failed for %s: %r", buy_order.ticker, e)
        return {"sold": False, "bought": False, "reason": "sell_failed_compensated"}
    return {"sold": True, "bought": True, "reason": "ok"}


async def dispatch_analyses(
    runner,
    new_tickers: list[str],
    reeval_tickers: list[str],
    *,
    merged: bool = False,
) -> tuple[list, list]:
    """phase-85.4 (criterion 2): the per-ticker analysis fan-out, as a seam.

    `runner` is an `async (ticker, kind) -> result | None` callable; in
    production it is the `_run_and_persist_one` closure, which acquires the
    shared per-provider `asyncio.Semaphore` itself.

    Returns `(candidate_results, holding_results)`, each positionally aligned
    with its input list, exceptions captured in place (`return_exceptions=True`)
    exactly as the two original gathers did.

    THE DEFECT THIS SEAM EXISTS TO EXPOSE (`merged=False`, the legacy path):
    the two gathers share ONE semaphore but are awaited in sequence, so the
    second batch cannot begin until the first has fully drained. A slot freed
    by an early-finishing new-candidate therefore sits idle for the remainder
    of the first batch instead of picking up a re-eval.

    Measured on the 2026-08-07 cycle (6 tickers, semaphore=3, from
    backend.log via scripts/diagnostics/measure_analysis_phase.py):

        20:09:16  dispatch DELL, PANW, CRWD          (3 of 3 slots busy)
        20:45:35  CRWD done  -> dispatch HPE
        20:50:24  DELL done  -> dispatch HUM
        20:52:30  PANW done  -> *** slot idle, NTAP not dispatched ***
        21:24:33  HUM done   -> dispatch NTAP        (batch 1 finally drained)
        22:00:01  cycle TIMED OUT, NTAP never analysed

    The slot idled 1923s and NTAP started 4517s into the analysis phase. With
    `merged=True` NTAP would have been dispatched at 20:52:30 -- 1923s earlier,
    against a cycle that overran its 7200s budget by 1329s.

    `merged=True` is BEHAVIOURAL and therefore DARK by default
    (`settings.paper_merged_analysis_dispatch_enabled`, phase-85.4 criterion 5).
    It changes nothing about order/sizing/risk logic; it changes only WHEN a
    ticker's analysis starts, and -- as a consequence -- which tickers are
    skipped if a cost cap trips mid-fan-out.
    """
    if not merged:
        candidate_results = await asyncio.gather(
            *[runner(t, "new") for t in new_tickers],
            return_exceptions=True,
        )
        # ── Step 4: Re-evaluate holdings ─────────────────────────────
        holding_results = await asyncio.gather(
            *[runner(t, "reeval") for t in reeval_tickers],
            return_exceptions=True,
        )
        return list(candidate_results), list(holding_results)

    # Merged: one gather, one drain. gather() preserves input order, so the
    # split below restores the exact (candidates, holdings) partition the
    # legacy path produced -- same elements, same positions, same types.
    n_new = len(new_tickers)
    combined = await asyncio.gather(
        *[runner(t, "new") for t in new_tickers],
        *[runner(t, "reeval") for t in reeval_tickers],
        return_exceptions=True,
    )
    return list(combined[:n_new]), list(combined[n_new:])


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
    # phase-69.1 (audit item 4b): the file-lock + _running are set ~90 lines BEFORE
    # the main try/finally that releases them. An exception from the unguarded init
    # (notably BigQueryClient construction) would otherwise strand the flock (never
    # __exit__'d) and leave _running=True -> the trading loop is permanently bricked
    # with no alert. Guard the init so a construction failure releases the lock +
    # resets _running (Python contextlib acquire-then-guard).
    try:
        settings = settings or get_settings()
        bq = BigQueryClient(settings)
        trader = PaperTrader(settings, bq)
    except Exception as _init_exc:
        logger.error("Paper trading cycle init failed after lock acquire: %r", _init_exc)
        _running = False
        try:
            _lock_cm.__exit__(type(_init_exc), _init_exc, _init_exc.__traceback__)
        except Exception:
            pass
        return {"status": "error", "reason": "init_failed", "error": str(_init_exc)[:200]}
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
    # phase-70.4 (G1-C): reconcile the per-cycle session ceiling with the
    # operator-visible daily cost cap when the flag is ON (session==daily); else
    # the legacy hidden $1.00 (byte-identical). Cost knob only; NO risk threshold moved.
    global _effective_session_budget
    if getattr(settings, "paper_session_budget_reconcile_enabled", False):
        _effective_session_budget = float(getattr(settings, "paper_max_daily_cost_usd", _SESSION_BUDGET_USD))
    else:
        _effective_session_budget = _SESSION_BUDGET_USD
    summary["session_budget_usd"] = _effective_session_budget

    # phase-56.2 (55.3 finding F-4): claude-CLI rail health probe at cycle
    # start. The 2026-06 away week ran with the OAuth rail silently down --
    # no check distinguished "rail down" from "no work". Free (no tokens),
    # non-blocking, own try/except: a probe bug must never break a cycle.
    if getattr(settings, "paper_use_claude_code_route", False):
        try:
            from backend.agents.claude_code_client import (
                claude_code_health_probe,
                rail_guard_disable,
                rail_guard_reset,
            )
            # phase-66.1: per-cycle breaker window reset BEFORE the probe.
            rail_guard_reset(_cycle_id)
            _rail_ok, _rail_detail = await asyncio.to_thread(claude_code_health_probe)
            summary["claude_rail_healthy"] = _rail_ok
            if not _rail_ok:
                # phase-66.1 (criterion 1): failed probe gates the rail --
                # every ClaudeCodeClient call this cycle returns the empty
                # LLMResponse immediately, zero subprocess spawns. The P1
                # below is the incident's single page (the guard's latch is
                # consumed by rail_guard_disable, so the breaker won't
                # double-page the same rail-down incident).
                rail_guard_disable(_rail_detail)
                logger.warning("claude-code rail health probe FAILED: %s", _rail_detail)
                from backend.services.observability.alerting import raise_cron_alert  # phase-66.1: was backend.services.alerting (module DOES NOT EXIST; ModuleNotFoundError swallowed by the fail-open except -> zero pages all away window)
                await raise_cron_alert(
                    source="claude_code_rail",
                    error_type="rail_down",
                    severity="P1",
                    title="Claude Code CLI rail unhealthy at cycle start",
                    details={
                        "cycle_id": _cycle_id,
                        "probe_detail": _rail_detail,
                        "consequence": "lite analyzer + conviction overlay will run degraded fallbacks",
                        "operator_action": "run `claude auth status` / re-login on the host; do NOT un-scrub ANTHROPIC_API_KEY",
                    },
                )
        except Exception as _probe_exc:
            logger.warning("claude rail probe errored (non-fatal): %s", _probe_exc)

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
            # ── Step 0: Roll the start-of-day anchor (phase-85.6) ────
            # THIS MUST STAY FIRST. The daily-loss anchor used to roll only at
            # paper_trader.py:1298, inside check_and_enforce_kill_switch, which
            # this cycle reaches only at Step 5.5 -- BEHIND the analysis phase.
            # Cycles were dying in `analyzing`, so the roll never ran, the anchor
            # went stale, the daily leg disarmed, POST /resume 409'd, and the book
            # could not be un-paused by any sequence of cycles. Measured: paused
            # since 2026-08-03T09:03:17Z, last sod_snapshot 2026-08-05T19:34:47Z.
            #
            # Step 0 is the point every cycle reaches regardless of where it later
            # dies, which is exactly what criterion 1 of phase-85.6 asks for.
            #
            # It changes no threshold and disarms nothing -- see
            # PaperTrader.roll_daily_anchor for the safety argument on the NAV
            # source. The :1298 roll is unchanged and becomes a same-day no-op via
            # the existing sod_anchor_needs_reroll date guard, so the phase-36.12
            # invariant (the BREACH decision reads a POST-roll state) still holds.
            #
            # Fail-open is safe here and only here: a roll that does not happen
            # leaves the anchor stale, which DISARMS the daily leg and REFUSES to
            # trade. Failing open cannot enable trading.
            summary["steps"].append("sod_anchor_roll")
            _anchor = await asyncio.to_thread(trader.roll_daily_anchor)
            summary["sod_anchor_roll"] = _anchor

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

            # phase-50.3: extend the universe with international markets when
            # settings.paper_markets includes non-US codes. Default ['US'] ->
            # _intl is empty -> universe unchanged -> BYTE-IDENTICAL to today.
            # Symbols are stored yfinance-suffixed (SAP.DE, 005930.KS); market
            # is derived from the suffix at buy-time (markets.market_for_symbol).
            _paper_markets = getattr(settings, "paper_markets", None) or ["US"]
            _intl_markets = [m for m in _paper_markets if m != "US"]
            if _intl_markets:
                from backend.backtest.universe_lists import INTL_UNIVERSE
                # phase-75.10 (perf-01 sibling): to_thread -- yfinance/network fetch,
                # execution-only change (same tickers, same order).
                base = list(universe) if universe is not None else await asyncio.to_thread(get_sp500_tickers)
                intl = [t for m in _intl_markets for t in INTL_UNIVERSE.get(m, [])]
                universe = base + intl
                # phase-50.4: ENTRY calendar gate -- drop a ticker whose market
                # is CLOSED today (market-local date), so we don't screen/buy a
                # closed exchange on stale data. US tickers are NEVER gated (the
                # live loop has never gated US -> keeps it byte-identical). Exits
                # are NOT gated here (a stop-loss must always be able to fire).
                from backend.backtest.markets import is_trading_day, market_for_symbol, get_market_config
                from datetime import datetime as _dt, timezone as _tz
                from zoneinfo import ZoneInfo as _ZI

                def _open_today(sym: str) -> bool:
                    mk = market_for_symbol(sym)
                    if mk == "US":
                        return True  # ungated -> byte-identical with today's US-only behaviour
                    try:
                        market_tz = get_market_config(mk).get("timezone", "UTC")
                        local_date = _dt.now(_tz.utc).astimezone(_ZI(market_tz)).date()
                        return is_trading_day(local_date, mk)
                    except Exception as _e:
                        logger.warning("phase-50.4: calendar gate error for %s (%s); keeping", sym, _e)
                        return True  # fail-open: never drop on a calendar error

                _before = len(universe)
                universe = [t for t in universe if _open_today(t)]
                _dropped = _before - len(universe)
                summary["universe_source"] = "+".join(_paper_markets)
                summary["universe_size"] = len(universe)
                if _dropped:
                    logger.info("phase-50.4: calendar gate dropped %d closed-market tickers", _dropped)
                logger.info(
                    "phase-50.3: multi-market universe %s -> %d tickers (+%d intl, %d calendar-gated)",
                    _paper_markets, len(universe), len(intl), _dropped,
                )

            # phase-51.2: give candidates a sector AT rank time so the (already-wired)
            # sector-neutral lever is functional -- enrichment used to run AFTER ranking
            # (autonomous_loop ~:659), making the within-sector path a silent no-op.
            # GATED on the flag so the OFF-default live path is BYTE-IDENTICAL (no map
            # build, sector_lookup=None -> identical to the prior call). Measured
            # 2026-06-01 (scripts/ablation/sector_neutral_replay.py): HARD sector-neutral
            # HURTS long-only Sharpe (-0.166), so the flag stays OFF; this keeps the lever
            # live-measurable for a future SOFT-tilt variant.
            _sector_lookup = None
            if getattr(settings, "sector_neutral_momentum_enabled", False) or getattr(settings, "multidim_momentum_enabled", False) or getattr(settings, "paper_soft_sector_diversity_enabled", False):
                try:
                    from backend.tools.screener import build_sector_map
                    # phase-75.10 (perf-01 sibling): to_thread -- execution-only,
                    # same universe in -> same sector map out.
                    _sector_lookup = await asyncio.to_thread(build_sector_map, universe)
                except Exception as e:
                    logger.warning("phase-51.2: sector map build failed (%s); sector-aware path falls back to global pool", e)

            # phase-75.10 (perf-01): run_daily_cycle is invoked as a coroutine ON
            # the API event loop by both AsyncIOScheduler instances (main.py:301,
            # :348) -- this yf.download(~500 tickers, 6mo) call genuinely blocked
            # it (the 2026-05-25 misfire-grace incident, paper_trading.py:1301-1311,
            # traced to smaller contention than this). to_thread moves WHERE it
            # runs only; same kwargs in, same screen_data out.
            screen_data = await asyncio.to_thread(screen_universe,
                tickers=universe,
                period="6mo",
                sector_lookup=_sector_lookup,
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

            # phase-69.1: hoist options_surge_signals / insider_signals defaults
            # BEFORE the ma_preannounce aggregator reads them at :474-475. They are
            # otherwise first assigned ~140 lines below (at the options/insider
            # screens), so the read raised UnboundLocalError when ma_preannounce_enabled
            # was on (a pre-existing latent bug + ruff F821 in this file, which 69.1's
            # lock fix already edits). Matches the existing `or {}` fallback; no prod
            # effect (ma_preannounce_enabled defaults OFF; the screens re-init these
            # unconditionally at their own blocks below). Do-no-harm side-fix to clear the gate.
            options_surge_signals: dict = {}
            insider_signals: dict = {}
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
                    # phase-75.10 (perf-10): this was ALREADY to_thread'd (2026-05-18,
                    # commit 6ceeb10ff) -- it does not block the loop today. The
                    # remaining defect is serial LATENCY (N sequential awaits); this
                    # bounds concurrency to 8 in-flight fetches, same per-ticker
                    # try/except/continue-on-failure semantics, same lookup shape.
                    _peer_sem = asyncio.Semaphore(8)

                    async def _fetch_peer_info(x: str):
                        async with _peer_sem:
                            try:
                                info = await asyncio.to_thread(lambda xx=x: yf.Ticker(xx).info or {})
                                return x, {
                                    "analyst_count": int(info.get("numberOfAnalystOpinions") or 0),
                                    "market_cap": float(info.get("marketCap") or 0),
                                }
                            except Exception:
                                return x, None

                    for t, entry in await asyncio.gather(*[_fetch_peer_info(t) for t in target_tickers]):
                        if entry is not None:
                            lookup[t.upper()] = entry
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
                # phase-70.2: soft cross-sector diversity (default OFF -> byte-identical)
                soft_sector_diversity=getattr(settings, "paper_soft_sector_diversity_enabled", False),
                soft_sector_diversity_w=getattr(settings, "paper_soft_sector_diversity_w", 0.0),
                sector_momentum_ranks=sector_momentum_ranks or None,
                multidim_momentum=getattr(settings, "multidim_momentum_enabled", False),
                multidim_weights={
                    "price":    getattr(settings, "multidim_momentum_weight_price", 0.35),
                    "52w_high": getattr(settings, "multidim_momentum_weight_52w_high", 0.25),
                    "sue":      getattr(settings, "multidim_momentum_weight_sue", 0.20),
                    "sector":   getattr(settings, "multidim_momentum_weight_sector", 0.20),
                },
                # phase-52.2: 52wh tilt (default OFF -> byte-identical; enable is operator-gated)
                momentum_52wh_tilt=getattr(settings, "momentum_52wh_tilt_enabled", False),
                momentum_52wh_tilt_k=getattr(settings, "momentum_52wh_tilt_k", 0.5),
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
                    # phase-56.2 (55.3 finding F-7): the away week ran every BUY on
                    # the no-LLM fallback ("conviction 10.00; fallback (LLM
                    # unavailable)") -- the momentum-damping overlay was silently
                    # removed. Make that LOUD. Deliberately observability-only:
                    # the fallback VALUE (composite-derived) stays byte-identical
                    # because conviction_score drives top-K selection and changing
                    # it is a live-behavior change (deferred to the gated phase-57
                    # redesign per the Confidence-Gate abstention analysis).
                    if _all_conviction_fallback(candidates):
                        summary["meta_scorer_degraded"] = True
                        logger.warning(
                            "Meta-scorer ran ENTIRELY on the no-LLM fallback for %d candidates "
                            "(conviction overlay degraded; damping leg inactive)",
                            len(candidates),
                        )
                        from backend.services.observability.alerting import raise_cron_alert  # phase-66.1: was backend.services.alerting (module DOES NOT EXIST; ModuleNotFoundError swallowed by the fail-open except -> zero pages all away window)
                        await raise_cron_alert(
                            source="meta_scorer",
                            error_type="conviction_overlay_degraded",
                            severity="P1",
                            title="Conviction overlay (SignalStack) running on no-LLM fallback",
                            details={
                                "cycle_id": _cycle_id,
                                "candidates": len(candidates),
                                "consequence": "momentum damping inactive; rankings are raw composite scores",
                            },
                        )
                        # phase-61.2 (criterion 4): cross-cycle streak WARN.
                        # The per-cycle P1 above fires every degraded cycle;
                        # the streak alert marks PERSISTENT unavailability
                        # (>=2 consecutive cycles -- the 7-week credit death
                        # went unnoticed partly for lack of exactly this).
                        # P2 is this project's WARN tier. Counter lives in a
                        # state file (module state dies on kickstart restarts).
                        if getattr(settings, "paper_synthesis_integrity_enabled", False):
                            _streak = _bump_conviction_fallback_streak(1)
                            if _streak >= 2:
                                await raise_cron_alert(
                                    source="meta_scorer",
                                    error_type="conviction_fallback_streak",
                                    severity="P2",
                                    title=f"Conviction overlay on no-LLM fallback for {_streak} consecutive cycles",
                                    details={
                                        "cycle_id": _cycle_id,
                                        "streak": _streak,
                                        "root_cause_hint": "direct-API Anthropic credit/key (live_check_66.2.md 5d)",
                                    },
                                )
                    elif getattr(settings, "paper_synthesis_integrity_enabled", False):
                        _bump_conviction_fallback_streak(0)
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
            # phase-70.2: min-K-sector round-robin on the deep-analyze slice (S2).
            # K=0 -> plain top-N slice (byte-identical). Enrichment already ran on
            # the top-N candidates above, so new_candidates carry a sector here.
            _min_k = int(getattr(settings, "paper_min_k_sectors_analyzed", 0) or 0)
            if _min_k > 0:
                _analyze_cands = _min_k_sector_slice(new_candidates, settings.paper_analyze_top_n, _min_k)
            else:
                _analyze_cands = new_candidates[:settings.paper_analyze_top_n]
            analyze_tickers = [c["ticker"] for c in _analyze_cands]

            # phase-57.1 (F-8): compute the RiskJudge sector context ONCE per
            # cycle (positions are identical for every ticker in the fan-out;
            # a per-ticker fetch would be N redundant BQ reads + a race).
            # phase-61.2 (criterion 6): ALSO built when the integrity flag is
            # ON -- the judge receives portfolio context in ADVISORY mode
            # regardless of the binding flag (which stays OFF). Both OFF =
            # byte-identical legacy.
            _rj_portfolio_ctx = ""
            if getattr(settings, "paper_risk_judge_reject_binding", False) or getattr(
                settings, "paper_synthesis_integrity_enabled", False
            ):
                try:
                    _rj_portfolio_ctx = _build_portfolio_sector_context(positions)
                except Exception as _ctx_exc:
                    logger.warning("RiskJudge sector context build failed (non-fatal): %s", _ctx_exc)

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
                    # phase-60.2 (AW-5): with the churn-fix flag ON, compare
                    # hours-precise age so "3 days" means >=72h. The truncated
                    # .days form makes the gate effectively 3-4 days when the
                    # cycle hour drifts (DELL was sold at 2d23h07m unanalyzed
                    # during the away week). Flag OFF: byte-identical .days.
                    if getattr(settings, "paper_swap_churn_fix_enabled", False):
                        days_since = (now - last_dt).total_seconds() / 86400.0
                    else:
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
                        # phase-57.1 (F-8): pass the per-cycle precomputed
                        # sector context (empty when the flag is OFF).
                        analysis = await _run_single_analysis(
                            ticker, settings, portfolio_context=_rj_portfolio_ctx,
                        )
                    except Exception as exc:
                        logger.error(f"Failed to analyze {kind} {ticker}: {exc}")
                        return None
                    if not analysis:
                        return None
                    cost = analysis.get("total_cost_usd", 0.1)
                    total_analysis_cost += cost
                    _add_session_cost(cost)
                    if analysis.get("_path") in ("lite", "full", "degraded"):
                        try:
                            await _persist_analysis(analysis, bq)
                        except Exception as exc:
                            logger.warning(
                                f"Persist failed for {kind} {ticker} (non-fatal): {exc}"
                            )
                    return _fold_degraded_for_trading(analysis)

            # ── Step 3 + Step 4 dispatch (phase-85.4) ────────────────
            # Extracted to a module-level seam so the two-gather barrier can be
            # measured and mutation-tested against the production call site
            # rather than against a copy of it. See dispatch_analyses().
            candidate_results, holding_results = await dispatch_analyses(
                _run_and_persist_one,
                analyze_tickers,
                reeval_tickers,
                merged=bool(
                    getattr(settings, "paper_merged_analysis_dispatch_enabled", False)
                ),
            )
            candidate_analyses = [r for r in candidate_results if isinstance(r, dict)]
            holding_analyses = [r for r in holding_results if isinstance(r, dict)]

            # phase-70.4 (G1-B): detect a swallowed session-budget breach in the raw
            # gather results (return_exceptions=True captures BudgetBreachError as a
            # result element, which the isinstance(dict) filter above silently drops).
            # Surface it so a cost-truncated cycle is never mistaken for a no-signal one.
            _budget_breaches = [
                r for r in (candidate_results + holding_results)
                if type(r).__name__ == "BudgetBreachError"
            ]
            if _budget_breaches:
                _n_skipped = (len(analyze_tickers) + len(reeval_tickers)) - (
                    len(candidate_analyses) + len(holding_analyses)
                )
                summary["session_budget_breach"] = True
                summary["session_budget_ceiling_usd"] = _effective_session_budget
                summary["session_cost_at_breach_usd"] = round(_session_cost, 4)
                summary["analyses_skipped_by_budget"] = _n_skipped
                logger.warning(
                    "phase-70.4: session-budget breach truncated this cycle -- ceiling $%.4f, "
                    "cost $%.4f, ~%d analysis(es) skipped. Set paper_session_budget_reconcile_enabled "
                    "(+ paper_max_daily_cost_usd) to lift the per-cycle ceiling.",
                    _effective_session_budget, _session_cost, _n_skipped,
                )

            # phase-56.2 (55.3 finding F-5): cycle-level degraded-scoring guard.
            # The 2026-05-27 incident wrote 11 rows of 0.0/HOLD (the rail-down
            # fallback) and the digest published them as confident neutrals --
            # a silent failure. Assert the cycle output BEFORE the consumption
            # layer (Write-Audit-Publish): if ALL analyses are degraded, or
            # >= 3 scored 0, alert P1 and stamp the cycle degraded.
            try:
                _fire, _n_degraded, _n_total = _degraded_scoring_check(
                    candidate_analyses + holding_analyses
                )
                if _fire:
                    summary["degraded"] = True
                    summary["degraded_analyses"] = f"{_n_degraded}/{_n_total}"
                    logger.warning(
                        "Degraded-scoring guard fired: %d/%d analyses scored 0/degraded",
                        _n_degraded, _n_total,
                    )
                    from backend.services.observability.alerting import raise_cron_alert  # phase-66.1: was backend.services.alerting (module DOES NOT EXIST; ModuleNotFoundError swallowed by the fail-open except -> zero pages all away window)
                    await raise_cron_alert(
                        source="autonomous_loop",
                        error_type="degraded_scoring",
                        severity="P1",
                        title=f"Cycle scoring degraded -- {_n_degraded}/{_n_total} analyses scored 0",
                        details={
                            "cycle_id": _cycle_id,
                            "degraded": f"{_n_degraded}/{_n_total}",
                            "consequence": "scores are failure artifacts, not real neutrals; digest readers must not trust this cycle's recs",
                        },
                    )
            except Exception as _guard_exc:
                logger.warning("Degraded-scoring guard errored (non-fatal): %s", _guard_exc)

            # phase-60.1 (AW-4): full->lite fallback-rate alarm, wired beside
            # the 56.2 guard (same raise_cron_alert path, distinct error_type)
            # -- NOT a parallel bespoke path. The away week ran 9 days at 100%
            # fallback (retired gemini-2.0-flash pin + KR SEC-CIK aborts) and
            # nothing fired; the operator believed the full skills pipeline
            # was deciding. Per-cycle window per SRE low-traffic guidance
            # (burn-rate windows degenerate at ~1 cycle/day).
            try:
                _fb_threshold = float(getattr(settings, "fallback_alarm_threshold", 0.5))
                _fb_fire, _n_fb, _n_fb_total, _fb_reasons = _fallback_rate_check(
                    candidate_analyses + holding_analyses, _fb_threshold,
                )
                # phase-86.38: RECORD ALWAYS, PAGE ONLY ABOVE THRESHOLD.
                # Before this, these fields were set ONLY inside the `if
                # _fb_fire` branch, so a cycle that degraded BELOW the threshold
                # left no trace anywhere an operator looks. MEASURED on the
                # 2026-08-10 cycle: 3 of the 6 TICKERS analysed fell back to
                # the lite analyser after the 28-agent orchestrator hit 429
                # RESOURCE_EXHAUSTED, the alarm did not fire, and the
                # degradation was therefore invisible outside a grep of
                # backend.log.
                #
                # STATED AT ITS TRUE SIZE: the alarm's own denominator is
                # len(candidate_analyses)+len(holding_analyses), which was NOT
                # measured. `3/6` is the TICKER ratio. The non-firing is
                # consistent with n_total>=6; it is NOT established that the
                # alarm missed by exactly one ticker, and this comment must not
                # be read as saying so.
                #
                # The THRESHOLD AND THE PREDICATE ARE UNCHANGED. This adds
                # observability, it does not re-tune an alarm: paging behaviour
                # is byte-identical, and `_fallback_rate_check` is not touched
                # (its strict `>` is pinned by
                # test_phase_60_1_deep_pipeline.py::test_fallback_alarm_threshold_is_strictly_greater_than
                # and changing it is an operator decision, not this step's).
                summary.update(_degradation_summary_fields(
                    _fb_fire, _n_fb, _n_fb_total, _fb_reasons,
                ))
                if _fb_fire:
                    logger.warning(
                        "Fallback-rate alarm fired: %d/%d analyses fell back full->lite (threshold %.0f%%)",
                        _n_fb, _n_fb_total, _fb_threshold * 100,
                    )
                    from backend.services.observability.alerting import raise_cron_alert  # phase-66.1: was backend.services.alerting (module DOES NOT EXIST; ModuleNotFoundError swallowed by the fail-open except -> zero pages all away window)
                    await raise_cron_alert(
                        source="autonomous_loop",
                        error_type="fallback_rate",
                        severity="P1",
                        title=(
                            f"Full-pipeline fallback rate {_n_fb}/{_n_fb_total} "
                            f"exceeds {_fb_threshold:.0%}"
                        ),
                        details={
                            "cycle_id": _cycle_id,
                            "fallback": f"{_n_fb}/{_n_fb_total}",
                            "per_ticker_reasons": _fb_reasons,
                            "consequence": (
                                "scores came from the 2-call lite momentum wrapper, "
                                "not the full skills pipeline; treat this cycle's "
                                "recommendations as degraded"
                            ),
                        },
                    )
            except Exception as _fb_exc:
                logger.warning("Fallback-rate alarm errored (non-fatal): %s", _fb_exc)

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
            halt_reason = cycle_halt_reason(ks_check, _ks_state().is_paused())
            if halt_reason:
                logger.warning(
                    "Paper trading: kill-switch active (%s) -- skipping decide/execute",
                    halt_reason,
                )
                summary["steps"].append("kill_switch_halted")
                summary["halted"] = True
                # phase-85.4 (criterion 3): STATUS FIDELITY. This early return
                # used to leave summary["status"] at the ":362" initializer's
                # placeholder "running", so the finally block wrote a terminal
                # cycle_history row with status="running" and raised a P1
                # titled "Autonomous trading cycle running" -- a terminal row
                # that claims the cycle is still going, and an alert whose
                # title names no failure. Measured on the 2026-08-05 cycle,
                # the ONE day all six tickers finished: it halted here, logged
                # status "running", and traded nothing.
                #
                # "halted_kill_switch" is a real terminal status: it is not in
                # the finally block's ("completed", "skipped") quiet-list, so
                # the P1 still fires -- but now it names the reason, and the
                # completed-age clock in cycle_health correctly counts this
                # cycle as a NON-completion.
                summary["status"] = "halted_kill_switch"
                summary["halt_reason"] = halt_reason
                ks_today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                _log_cycle_signals_to_bq(bq, [], ks_today)
                final_state = await asyncio.to_thread(trader.mark_to_market)

                # ── phase-36.17: EXIT-ONLY pass on a halted cycle ────────
                # A halt stops NEW ENTRIES; it must not stop PROTECTIVE EXITS.
                # The `paused` and `blocked` paths do NOT flatten -- kill_switch.py
                # line 14: "Pause = halt new entries; existing positions kept" --
                # so without this the book holds full exposure with its stop-losses
                # unenforced, every cycle until the halt clears. `blocked` is
                # NON-LATCHING, so it can recur indefinitely with no operator
                # resume. BEFORE this change check_stop_losses had exactly ONE
                # production caller (the Step 5.6 block below), so a halted cycle
                # had no enforcement layer at all. THIS PASS IS THE SECOND CALLER
                # -- re-derive with `grep -rn check_stop_losses backend --include="*.py"`
                # rather than trusting this comment.
                #
                # This is the standard "liquidation-only" state: NYSE Pillar Risk
                # Controls v4.7 ships "Block only -- accept cancels"; ESMA's
                # Supervisory Briefing on Algorithmic Trading (2026-02) para 93
                # makes cancel/withdraw/kill three distinct rungs; MiFID II RTS 6
                # Art. 12 scopes "kill" to UNEXECUTED orders. Operator decision
                # 2026-08-09 = option (b). Runs AFTER mark_to_market so the stop
                # comparison sees fresh prices, and BEFORE save_daily_snapshot so
                # the snapshot reflects any exit.
                #
                # THREE THINGS THIS DELIBERATELY DOES NOT DO:
                #  1. It does NOT run on the `triggered` (breach) path --
                #     check_and_enforce_kill_switch already called flatten_all, so
                #     a second pass would duplicate exits, fee events and
                #     learn-loop rows over positions that no longer exist.
                #  2. It does NOT call backfill_missing_stops. SYNTHESIZING a stop
                #     level is a NEW risk decision (ESMA para 11(5)); the
                #     synthesized price can land ABOVE the current mark, turning
                #     "this position has no stop" into "sell it at market now" --
                #     a flatten by side effect on exactly the branches that
                #     deliberately do not flatten.
                #  3. It does NOT append to summary["steps"]: two phase-36.12
                #     tests assert summary["steps"][-1] == "kill_switch_halted".
                #
                # SELL-only, and no BUY can leak: execute_buy refuses when paused
                # (paper_trader.py:282-294, fail-closed at :225-233), and on the
                # `blocked` path BUY suppression comes ONLY from the `return
                # summary` below -- so that return MUST stay last.
                if not ks_check.get("triggered"):
                    summary["halt_stop_loss_triggered"] = []
                    try:
                        halt_stops = await asyncio.to_thread(trader.check_stop_losses)
                        for sl_ticker in halt_stops or []:
                            sl_trade = await asyncio.to_thread(
                                trader.execute_sell,
                                ticker=sl_ticker,
                                quantity=None,
                                price=None,
                                reason="stop_loss_trigger",
                                signals=None,
                            )
                            if sl_trade:
                                summary["halt_stop_loss_triggered"].append(sl_ticker)
                                logger.warning(
                                    "phase-36.17: stop-loss enforced on a HALTED "
                                    "cycle (%s) -- sold %s at %s",
                                    halt_reason, sl_ticker, sl_trade.get("price"),
                                )
                    except Exception as halt_sl_exc:
                        # The halt itself must still complete and report: the
                        # phase-85.4 loudness guards depend on the terminal
                        # status set above. Loud, but never at that cost.
                        summary["halt_stop_loss_error"] = repr(halt_sl_exc)
                        logger.exception(
                            "phase-36.17: exit-only stop-loss pass FAILED on a "
                            "halted cycle (%s) -- positions may be holding "
                            "exposure with unenforced stops: %s",
                            halt_reason, halt_sl_exc,
                        )

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
            # phase-57.1 (F-3): blocked-BUY out-channel for the binding
            # RiskJudge gate -- surfaced on the cycle summary for the event
            # study / DoD-7 evidence trail.
            _rj_blocked: list[dict] = []
            orders = decide_trades(
                current_positions=positions,
                candidate_analyses=candidate_analyses,
                holding_analyses=holding_analyses,
                portfolio_state=portfolio_state,
                settings=settings,
                candidates_by_ticker=candidates_by_ticker,
                blocked_out=_rj_blocked,
            )
            if _rj_blocked:
                summary["risk_judge_blocked"] = _rj_blocked
                logger.warning(
                    "BINDING RiskJudge gate blocked %d BUY(s) this cycle: %s",
                    len(_rj_blocked), [b["ticker"] for b in _rj_blocked],
                )

            # ── Step 7: Execute trades ───────────────────────────────
            logger.info(f"Paper trading: Step 7 -- Executing {len(orders)} trades")
            summary["steps"].append("executing")
            # phase-30.3: closed_tickers now lives at cycle-top (line ~169)
            # so Step 5.6 stop-outs can populate it. Re-init here would
            # erase Step 5.6's appends.

            # phase-70.3: atomic swap execution. When ON, execute swap PAIRS (tagged
            # with a shared swap_group_id) as BUY-first-with-reserved-cash units so a
            # SELL can never persist while its paired BUY drops (net -1 position); the
            # handled legs are then removed from the flat loops below. OFF -> every order
            # has swap_group_id=None -> all orders flow the flat loops (byte-identical).
            if getattr(settings, "paper_atomic_swap_enabled", False):
                _swap_gids = [g for g in {getattr(o, "swap_group_id", None) for o in orders} if g]
                if _swap_gids:
                    _handled: set = set()
                    for _gid in _swap_gids:
                        _legs = [o for o in orders if getattr(o, "swap_group_id", None) == _gid]
                        _sell = next((o for o in _legs if o.action == "SELL"), None)
                        _buy = next((o for o in _legs if o.action == "BUY"), None)
                        if not (_sell and _buy):
                            continue
                        _res = await asyncio.to_thread(_execute_swap_pair, trader, _sell, _buy)
                        if _res.get("bought"):
                            trades_executed += 1
                        if _res.get("sold"):
                            trades_executed += 1
                            closed_tickers.append(_sell.ticker)
                        summary.setdefault("swap_pairs", []).append({"group": _gid, **_res})
                        _handled.add(_gid)
                    orders = [o for o in orders if getattr(o, "swap_group_id", None) not in _handled]

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
                    market=getattr(order, "market", "US"),  # phase-50.3: US for bare tickers (byte-identical)
                    # phase-30.6: analysis-time reference for the
                    # price-tolerance gate inside execute_buy.
                    price_at_analysis=order.price_at_analysis,
                    # phase-40.8.1 (P3): in-memory FF3 loadings; BQ persist
                    # deferred to phase-40.8.2.
                    factor_loadings=order.factor_loadings,
                    # phase-61.2 (criterion 5): analysis verdict for the
                    # position row (consumed only when the fix flag is ON).
                    analysis_recommendation=getattr(order, "analysis_recommendation", ""),
                )
                if trade:
                    trades_executed += 1

            # phase-70.4 (G2-A): surface BUY rejections (price-tolerance etc.) to the
            # cycle summary so a 0-trade cycle is attributable, not silently un-counted.
            if getattr(trader, "buy_rejections", None):
                from collections import Counter as _Counter
                summary["buy_rejections"] = list(trader.buy_rejections)
                summary["buy_rejections_by_reason"] = dict(
                    _Counter(r.get("reason", "unknown") for r in trader.buy_rejections)
                )
                logger.warning(
                    "phase-70.4: %d BUY(s) rejected this cycle by reason %s",
                    len(trader.buy_rejections), summary["buy_rejections_by_reason"],
                )

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
        # phase-38.6.1: release the file-based cycle_lock. phase-85.5: this
        # marks the pidfile {state: "released"} in place and THEN releases the
        # flock -- it no longer unlinks (unlink-before-LOCK_UN split the lock
        # across two inodes). Idempotent: if already exited, _lock_cm is unset.
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
            # phase-66.1: rail-guard outcome persisted per cycle so the 66.2
            # funnel diagnosis can separate "rail skipped/tripped" from
            # "gates rejected". Fail-open: guard status must never break
            # the cycle-health write.
            _rail_skipped = False
            _breaker_tripped = False
            try:
                from backend.agents.claude_code_client import rail_guard_status

                _rg = rail_guard_status()
                _rail_skipped = bool(_rg.get("rail_skipped"))
                _breaker_tripped = bool(_rg.get("breaker_tripped"))
                if _rail_skipped or _breaker_tripped:
                    summary["rail_skipped"] = _rail_skipped
                    summary["breaker_tripped"] = _breaker_tripped
                    summary["rail_skipped_calls"] = _rg.get("skipped_calls")
            except Exception:
                pass
            # phase-66.2: persist the per-stage funnel counts that previously
            # lived only in the in-memory summary + backend.log lines (the
            # criterion-b diagnosis needs durable per-cycle stage counts).
            _funnel = {
                k: summary.get(k)
                for k in (
                    "universe_source", "universe_size", "screened",
                    "candidates", "new_to_analyze", "reeval_tickers",
                )
                if summary.get(k) is not None
            }
            # phase-86.38: persist the DEGRADATION facts alongside the funnel.
            # Kept as its own dict rather than folded into `_funnel` because the
            # funnel answers "how many candidates survived each stage" and this
            # answers "was the pipeline that judged them the real one" -- two
            # questions, and conflating them is how the fallback rate ended up
            # with no home. Populated on every cycle, not only when the alarm
            # pages, so a sub-threshold degradation is still durable.
            _degradation = _degradation_record(summary)
            _cycle_log().record_cycle_end(
                cycle_id=_cycle_id,
                started_at=_cycle_started_at,
                status=summary.get("status", "unknown"),
                n_trades=int(summary.get("trades_executed", trades_executed) or 0),
                error_count=int(summary.get("error_count", 0) or 0),
                data_source_ages=summary.get("data_source_ages") or {},
                bq_ingest_lag_sec=summary.get("bq_ingest_lag_sec"),
                meta_scorer_degraded=bool(summary.get("meta_scorer_degraded")),
                rail_skipped=_rail_skipped,
                breaker_tripped=_breaker_tripped,
                funnel=_funnel,
                degradation=_degradation,
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
                # phase-85.4 (criterion 3): NAME THE PHASE IT DIED IN, in the
                # TITLE. `steps_completed` already carried the trailing steps
                # in the details body, but the Slack channel renders ~24 hourly
                # freshness P1s a day and the operator triages on titles -- the
                # 08-04/08-06/08-07 timeout P1s were all delivered=True and all
                # went unread because "Autonomous trading cycle timeout" says
                # nothing about WHERE. The last appended step IS the phase the
                # cycle was in when it died, because every phase appends its
                # name to summary["steps"] on entry.
                _steps = summary.get("steps", []) or []
                _died_in = _steps[-1] if _steps else "before_first_step"
                raise_cron_alert_sync(
                    source="autonomous_loop",
                    error_type=f"cycle_{_final_status}",
                    severity="P1",
                    title=f"Autonomous trading cycle {_final_status} in phase '{_died_in}'",
                    details={
                        "cycle_id": summary.get("cycle_id", "?"),
                        "started_at": summary.get("started_at", "?"),
                        "status": _final_status,
                        "died_in_phase": _died_in,
                        "error": str(summary.get("error", ""))[:300],
                        "steps_completed": ",".join(_steps[-5:]),
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


class SynthesisDegradedError(RuntimeError):
    """phase-61.2 (criterion 1): raised INSIDE _run_single_analysis's full-path
    try when the orchestrator returns a synthesis carrying final_synthesis.error
    (or missing scoring_matrix), so the EXISTING except -> lite-fallback ->
    60.1 fallback-rate-alarm machinery handles it. Before this, the error dict
    returned successfully and was assembled into a synthetic HOLD/0.0 that
    decide_trades could not distinguish from a conviction HOLD (destroyed two
    live BUY/0.62 consensuses on cycle 0725d2aa). Only raised when
    paper_synthesis_integrity_enabled is ON."""


async def _run_single_analysis(
    ticker: str, settings: Settings, portfolio_context: str | None = None,
) -> Optional[dict]:
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

    phase-57.1 (F-8): `portfolio_context` is the per-cycle precomputed sector
    summary threaded into the lite RiskJudge prompt (computed ONCE in the
    cycle, never fetched per ticker). None/"" + flag OFF = byte-identical.
    """
    if settings.lite_mode:
        try:
            # phase-27.3 (C2): dispatch by configured standard model. Was hardcoded
            # to _run_claude_analysis which refused non-Claude models, leaving
            # gemini-* selections with no lite fallback.
            return await _select_lite_analyzer(settings.gemini_model, settings)(
                ticker, settings, portfolio_context=portfolio_context or "",
            )
        except Exception as e:
            logger.warning("Lite analysis failed for %s (lite_mode=True): %s", ticker, e)
            return None

    # Full pipeline path (operator chose lite_mode=False)
    # phase-cycle-8 (38.13, 2026-05-27): rail attribution log. Cycle-7 Q/A
    # misdiagnosed lite-vs-full because BQ rows had no rail tag. This log
    # lets us see, per-ticker, which rail the full orchestrator chose
    # BEFORE diving into the per-agent calls.
    _route = "claude_code" if getattr(settings, "paper_use_claude_code_route", False) else "anthropic_direct"
    logger.info("Orchestrator pre-dispatch ticker=%s rail=%s lite_mode=False model=%s", ticker, _route, settings.gemini_model)
    _fb_reason = "unknown"  # phase-60.1: overwritten by the except below
    try:
        # phase-38.13.1 (cycle 11, 2026-05-27): cure the get_settings()
        # lru_cache desync across uvicorn workers. When the operator flips
        # paper_use_claude_code_route via Settings UI, only one worker
        # clears its cache; cron-spawned cycles using a stale snapshot get
        # rail=False and silently bill against direct Anthropic. Force a
        # fresh read here so the orchestrator constructs with the same
        # rail value the autonomous_loop layer logged at _route.
        from backend.config.settings import get_settings as _get_settings_fresh
        _get_settings_fresh.cache_clear()
        settings = _get_settings_fresh()
        _orch_rail = "claude_code" if getattr(settings, "paper_use_claude_code_route", False) else "anthropic_direct"
        logger.info(
            "AnalysisOrchestrator construction ticker=%s constructor_rail=%s cycle_rail=%s",
            ticker, _orch_rail, _route,
        )
        orchestrator = AnalysisOrchestrator(settings)
        report = await orchestrator.run_full_analysis(ticker)
        if not report:
            raise RuntimeError("orchestrator returned empty report")

        synthesis = report.get("final_synthesis", {})
        # phase-61.2 (criterion 1): a synthesis error dict must never be
        # assembled into a persistable result. Raise into the except below so
        # the lite fallback produces a REAL scored row (first option of the
        # immutable criterion); the raise also stamps _fallback_reason for the
        # 60.1 fallback-rate alarm. Flag OFF = legacy fabrication (covered by
        # the byte-identical regression test).
        if getattr(settings, "paper_synthesis_integrity_enabled", False) and (
            not isinstance(synthesis, dict)
            or synthesis.get("error")
            or "scoring_matrix" not in synthesis
        ):
            _syn_err = (
                synthesis.get("error") if isinstance(synthesis, dict) else None
            ) or "missing scoring_matrix"
            raise SynthesisDegradedError(f"synthesis_error: {_syn_err}")
        rec = synthesis.get("recommendation", {})
        quant = report.get("quant", {})
        risk = synthesis.get("risk_assessment", {})
        cost_summary = report.get("cost_summary", {})

        return {
            "ticker": ticker,
            "recommendation": rec.get("action", "HOLD") if isinstance(rec, dict) else str(rec),
            # phase-71 cycle (2026-05-26): the orchestrator stores the weighted
            # score under "final_weighted_score" (backend/agents/orchestrator.py:2001),
            # not "final_score". Reading the wrong key here cascaded into
            # _persist_analysis writing 0 to analysis_results.final_score for
            # every full-path autonomous cycle since the first clean run on
            # 2026-05-22 (commit 29ab0ff6 phase-34.2). Slack morning digests
            # have been showing "0.0/10" for every ticker as a result.
            # Manual-path tasks/analysis.py:208 has always used the correct key.
            # Defensive: keep the legacy "final_score" as fallback for any
            # future writer that re-introduces the bare key.
            "final_score": synthesis.get(
                "final_weighted_score", synthesis.get("final_score", 0)
            ),
            "risk_assessment": risk,
            "price_at_analysis": quant.get("yf_data", {}).get("valuation", {}).get("currentPrice") if isinstance(quant.get("yf_data"), dict) else None,
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "total_cost_usd": cost_summary.get("total_cost_usd", 0.1) if isinstance(cost_summary, dict) else 0.1,
            # phase-cycle-8 (38.13): populate full_report["source"] with the
            # active standard model so _persist_analysis at line ~1844 writes
            # standard_model into BQ analysis_results. Lite-path already sets
            # this; full-path was silently leaving it NULL, which made cycle-7
            # Q/A misread the BQ signature as lite-fallback.
            "full_report": {**(report if isinstance(report, dict) else {}), "source": settings.gemini_model, "rail": _route},
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
        # phase-60.1 (AW-4): capture the per-ticker failure reason. The away
        # week's 100% fallback was invisible because the reason lived only in
        # this log line -- the alarm predicate needs it on the analysis dict.
        _fb_reason = f"{type(e).__name__}: {e}"

    # Last-resort fallback: try lite path so the cycle still produces a decision.
    # phase-27.3 (C2): provider-aware via _select_lite_analyzer.
    try:
        _lite = await _select_lite_analyzer(settings.gemini_model, settings)(
            ticker, settings, portfolio_context=portfolio_context or "",
        )
        # phase-60.1 (AW-4): tag the INTENDED-full-but-landed-lite analyses.
        # Deliberate lite_mode runs (the branch above) carry NO tag, so the
        # fallback-rate alarm never fires on an operator's lite choice.
        #
        # phase-86.38: `_intended_path = "full"` was REMOVED from here, not
        # wired up. It was WRITE-ONLY -- one write at this site, ZERO reads
        # repo-wide (measured with a search whose recall was validated by
        # returning 13 hits for the sibling `_fallback_reason`), and it was
        # never copied into the persisted `full_report` by `_persist_analysis`,
        # so it reached neither BQ nor any API nor the UI.
        #
        # REMOVED RATHER THAN CONSUMED, deliberately: it is REDUNDANT. The set
        # it marked -- intended-full, landed-lite -- is exactly the set carrying
        # `_fallback_reason`, which `_fallback_rate_check` already keys on and
        # which IS persisted. Wiring a second field for one fact creates two
        # sources of truth that can disagree; the cheaper correctness is one
        # field with a consumer.
        if isinstance(_lite, dict):
            _lite["_fallback_reason"] = _fb_reason[:500]
        return _lite
    except Exception as e:
        logger.error("Both full and lite paths failed for %s: %s", ticker, e)
        # phase-61.2 (criterion 1, second option): with the integrity flag ON,
        # leave an HONEST trace instead of vanishing -- a degraded marker dict
        # that _run_and_persist_one persists as a NULL-score/NULL-rec row with
        # $._degraded, then converts to None so it NEVER enters
        # candidate/holding analyses (decide_trades input not neutralized,
        # not poisoned). Flag OFF = legacy silent None (no row).
        if getattr(settings, "paper_synthesis_integrity_enabled", False):
            return {
                "ticker": ticker,
                "_degraded": True,
                "_degraded_reason": (
                    f"both_paths_failed: full={( _fb_reason or '')[:200]}; "
                    f"lite={type(e).__name__}: {e}"
                )[:500],
                "_path": "degraded",
                "recommendation": None,
                "final_score": None,
                "risk_assessment": {},
                "total_cost_usd": 0.0,
                "analysis_date": datetime.now(timezone.utc).isoformat(),
                "full_report": {},
            }
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


def _lite_judge_produced_no_verdict(risk_dict: dict) -> bool:
    """True when the caller handed the producer the WHOLE default dict.

    phase-86.88 cycle 2, from the cycle-1 Q/A. The cycle-1 fix logged a sentence
    and returned the same number, and the artifacts then claimed the value was
    "resolved as ABSENT". Measured, that was FALSE in the only place it matters:
    the persisted `recommended_position_pct` was still 3.0, and
    `_resolve_position_pct` on that record still returned
    `PositionVerdict(SIZE, 3.0)` -- byte-identical to a judge that really said
    3%. The early return above hands back the default float BEFORE the resolver
    is ever called, so no ABSENT verdict was ever constructed.

    A log line is not provenance: nothing downstream, and no auditor reading the
    persisted row, can see it. The remedy is an ADDITIVE key -- it distinguishes
    judge-failed from judge-said-3% in the record itself while leaving the number
    (and therefore every order outcome) untouched, satisfying criterion 5 and
    criterion 7 at once. The cycle-1 choice to leave the record unchanged was a
    choice, not a constraint.
    """
    return risk_dict == _LITE_RISK_DEFAULT


def _lite_position_pct(risk_dict: dict, ticker: str = "?") -> float:
    """THE single seam at which the LITE judge's position size is resolved.

    phase-86.86 (D6). Both lite paths previously built the persisted
    risk_assessment with::

        float(risk_dict.get("recommended_position_pct")
              or _LITE_RISK_DEFAULT["recommended_position_pct"])

    and **0.0 is falsy**, so the strongest risk signal the judge can issue -- an
    explicit 0% -- was rewritten to the 3.0 default at INGRESS. Measured on the
    shipped tree 2026-08-15: judge 0.0 -> 3.0, judge 3.0 -> 3.0, judge ABSENT ->
    3.0. The second and third rows are the defect: **an explicit zero and a
    silent judge became indistinguishable**, and the zero died UPSTREAM of every
    guard phase-86.74 built, so `PositionVerdict(SIZE, 3.0)` downstream was a
    correct reading of an already-falsified value. Driven through the real
    `decide_trades` that produced a BUY of $719.93 on NAV 23,997.71 where a true
    0.0 produces no order -- in ALL FOUR flag combinations, because the decision
    was APPROVE_REDUCED and `paper_risk_judge_reject_binding` only blocks an
    exact REJECT.

    This routes through `_resolve_position_pct` -- the SAME three-state resolver
    the full path uses -- rather than adding a second idiom, so lite and full
    share one rule. The default is returned for ABSENT and **only** for ABSENT:

    * ``SIZE``        -> the judge's number, **0.0 included**
    * ``UNPARSEABLE`` -> 0.0, fail CLOSED and LOUD (a verdict we could not read
      is not evidence of safety -- mirrors `_sizing_pct`)
    * ``ABSENT``      -> the 3.0 default; nobody expressed an opinion

    Note `analysis={}`: at construction time the raw judge dict is the only
    source that exists, and the second source `_resolve_position_pct` consults
    (`analysis["risk_judge_position_pct"]`) is not populated until later.
    """
    # phase-86.88 (N2). THE JUDGE-FAILURE ROUTE, resolved AT THE SEAM.
    # Four callers -- autonomous_loop.py :3177 / :3182 (Claude lite) and
    # :3411 / :3416 (Gemini lite), the no-JSON and exception handlers -- assign
    # `risk_dict = dict(_LITE_RISK_DEFAULT)` when the judge produced nothing
    # parseable at all. That copy carries `recommended_position_pct: 3.0`, so
    # this seam saw SIZE(3.0) and a JUDGE FAILURE was persisted as though the
    # judge had deliberately specified 3%. SAME NUMBER, DESTROYED PROVENANCE --
    # which is exactly why no number-asserting test could see it, and why
    # phase-86.86's six mutation cells all passed over it.
    #
    # It is the phase-86.74/86.86 collapse one seam over: one value carrying
    # three domain states, with judge-failed and judge-said-3% made
    # indistinguishable.
    #
    # Detected HERE and not at the four call sites, because criterion 5 requires
    # the fix at the seam -- and because a call-site fix would have to be
    # repeated at every future handler that falls back to the default. Value
    # equality (not identity) is the right test: every route reaches us through
    # `dict(...)`, which copies. CERT OBJ06-J's copy-then-validate, applied to a
    # dict we do not own.
    #
    # THE NUMBER IS DELIBERATELY UNCHANGED. ABSENT resolves to the same 3.0
    # default, so no order outcome moves (criterion 7). What changes is that the
    # persisted verdict now says "nobody expressed an opinion" rather than "the
    # judge chose 3%".
    if risk_dict == _LITE_RISK_DEFAULT:
        logger.warning(
            "Lite risk judge for %s: the caller supplied the WHOLE default dict, "
            "i.e. the judge produced nothing parseable -- resolving as ABSENT, "
            "not as an explicit SIZE(%.1f). Sizing is unchanged at the %.1f "
            "default; only the provenance is corrected.",
            ticker,
            _LITE_RISK_DEFAULT["recommended_position_pct"],
            _LITE_RISK_DEFAULT["recommended_position_pct"],
        )
        return float(_LITE_RISK_DEFAULT["recommended_position_pct"])


    verdict = _resolve_position_pct(risk_dict, {})
    if verdict.kind == SIZE:
        # A SIZE with no number is contradictory; fail closed rather than let a
        # corrupted verdict reach the default.
        return float(verdict.pct) if verdict.pct is not None else 0.0
    if verdict.kind == UNPARSEABLE:
        logger.warning(
            "Lite risk judge for %s: recommended_position_pct is present but "
            "UNPARSEABLE (%r) -- sizing at 0.0 (fail closed), NOT at the %.1f "
            "default. A verdict that cannot be read is not evidence of safety.",
            ticker,
            risk_dict.get("recommended_position_pct"),
            _LITE_RISK_DEFAULT["recommended_position_pct"],
        )
        return 0.0
    return float(_LITE_RISK_DEFAULT["recommended_position_pct"])


def _build_lite_risk_assessment(risk_dict: dict, ticker: str = "?") -> dict:
    """Build the persisted lite `risk_assessment` -- the ONE producer.

    phase-86.86 (D6). The Claude lite path (was `:3085-3097`) and the Gemini
    lite path (was `:3333-3343`) carried byte-identical copies of this dict
    literal, so a fix applied to one would silently miss the other. Extracting
    it does three things the in-place edit could not:

    1. there is now **exactly one** place that can reach
       `_LITE_RISK_DEFAULT["recommended_position_pct"]` -- inside
       `_lite_position_pct`, and nowhere else;
    2. a test can DRIVE the real producer, which a dict literal buried in a
       300-line async LLM function cannot be driven; a mutation cell against an
       undriveable site is UNSCORABLE;
    3. the two paths cannot drift apart again.

    **The other four keys are relocated BYTE-IDENTICALLY and deliberately.**
    Their `or` idioms are not decision-inverting -- measured 2026-08-15 by
    driving the real `decide_trades`: an empty `decision` and the substituted
    APPROVE_REDUCED both produce the same BUY (only an exact REJECT blocks), and
    `risk_level` is read zero times by `portfolio_manager`. They DO fabricate the
    persisted audit trail -- the substituted `reasoning` states "risk-judge parse
    failed" when the parse succeeded and only that field was blank -- which is a
    real defect, filed as its own step **86.87** rather than fixed here.
    `risk_limits` is left alone because its substitution INSTALLS a stop where
    none existed, i.e. it is protective.
    """
    risk_reasoning = str(risk_dict.get("reasoning") or _LITE_RISK_DEFAULT["reasoning"])
    return {
        "decision": str(risk_dict.get("decision") or _LITE_RISK_DEFAULT["decision"]),
        "reasoning": risk_reasoning,
        # Backward-compat alias: bq.save_report at line ~818 reads
        # risk_assessment.get("reason", "") for the summary column.
        "reason": risk_reasoning,
        # phase-86.86 (D6): NOT `or _LITE_RISK_DEFAULT[...]` -- see
        # `_lite_position_pct`. An explicit 0.0 must survive this line.
        "recommended_position_pct": _lite_position_pct(risk_dict, ticker),
        "risk_level": str(risk_dict.get("risk_level") or _LITE_RISK_DEFAULT["risk_level"]),
        "risk_limits": dict(risk_dict.get("risk_limits") or _LITE_RISK_DEFAULT["risk_limits"]),
        # phase-86.88 cycle 2: ADDITIVE provenance. Distinguishes "the judge
        # produced nothing parseable and we fell back" from "the judge chose 3%"
        # IN THE RECORD, where a downstream reader or an auditor can see it. The
        # number is untouched, so no order outcome moves (criterion 7); what
        # changes is that the two states stop being byte-identical.
        "judge_verdict_absent": _lite_judge_produced_no_verdict(risk_dict),
    }


def _integrity_market_data(
    name, current_price, market_cap, pe_ratio, sector, industry,
    momentum_20d, momentum_60d, norm, flags,
) -> dict:
    """phase-60.3 (AW-9): lite market_data + ADDITIVE provenance fields.

    The provenance fields (currency/price_usd/market_cap_usd/fx_rate/as_of/
    integrity_flags) are UNGATED observability -- they change no decision and
    make every persisted lite row unit-auditable in BQ (criterion 4)."""
    return {
        "name": name,
        "price": current_price,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "sector": sector,
        "industry": industry,
        "momentum_20d": momentum_20d,
        "momentum_60d": momentum_60d,
        "currency": norm.get("currency"),
        "price_usd": norm.get("price_usd"),
        "market_cap_usd": norm.get("market_cap_usd"),
        "fx_rate": norm.get("fx_rate"),
        "as_of": norm.get("as_of"),
        "integrity_flags": [f["flag"] for f in flags],
    }


def _data_integrity_blocked_analysis(
    ticker, name, sector, industry, model_name, current_price, market_cap,
    pe_ratio, momentum_20d, momentum_60d, norm, flags,
) -> dict:
    """phase-60.3 (AW-9): pre-LLM in-code enforcement of blocking integrity
    flags (GuardAgent chokepoint pattern -- arXiv:2406.09187). The away week
    proved prose-only flagging is ignored: the judge wrote 'physically
    impossible... KRW/USD unit error' and the BUY executed anyway
    (066570.KS 2026-06-09, stopped out -9.7%).

    Returns a lite-shaped HOLD: never enters _BUY_RECS, costs $0 LLM, and
    counts toward the 56.2 degraded-scoring guard (deliberate -- widespread
    integrity blocks SHOULD alarm)."""
    reasons = "; ".join(f"{f['flag']}: {f['detail']}" for f in flags if f.get("blocking"))
    logger.warning("Data-integrity block for %s (no LLM call): %s", ticker, reasons)
    return {
        "ticker": ticker,
        "_path": "lite",
        "recommendation": "HOLD",
        "final_score": 0,
        "_data_integrity_blocked": True,
        "risk_assessment": {
            "decision": "REJECT",
            "reasoning": f"data-integrity block (deterministic pre-check, no LLM): {reasons}",
            "reason": f"data-integrity block: {reasons}",
            "recommended_position_pct": 0.0,
            "risk_level": "EXTREME",
            "risk_limits": dict(_LITE_RISK_DEFAULT["risk_limits"]),
        },
        "price_at_analysis": None,
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": 0.0,
        "full_report": {
            "source": model_name,
            "analysis": {"action": "HOLD", "confidence": 0, "score": 0,
                         "reason": f"data-integrity block: {reasons}"},
            "market_data": _integrity_market_data(
                name, current_price, market_cap, pe_ratio, sector, industry,
                momentum_20d, momentum_60d, norm, flags,
            ),
        },
    }


def _build_risk_judge_system(settings) -> str:
    """phase-57.1 (55.3 finding F-8): the system prompt's CONCENTRATION axis
    hardcoded a phantom '10% of portfolio in one sector' while the configured
    cap is paper_max_per_sector_nav_pct (30.0 default). Behind the SAME flag
    as the binding gate -- never bind on a blind judge (GuardAgent: a guard
    must be given its inputs). Flag OFF returns the verbatim constant so the
    rendered prompt is byte-identical."""
    if not getattr(settings, "paper_risk_judge_reject_binding", False):
        return _LITE_RISK_JUDGE_SYSTEM
    cap = float(getattr(settings, "paper_max_per_sector_nav_pct", 30.0) or 30.0)
    return _LITE_RISK_JUDGE_SYSTEM.replace(
        "exceed 10% of portfolio in one sector",
        f"exceed {cap:.0f}% of portfolio NAV in one sector",
    )


def _build_risk_judge_template(settings, portfolio_context: str = "") -> str:
    """phase-57.1 (F-8): flag ON injects a live portfolio sector-breakdown
    line so the judge stops reasoning blind ('no current portfolio sector
    breakdown was provided' -- every away-week rationale). Flag OFF returns
    the verbatim constant (byte-identity; no new .format key on that path)."""
    if not getattr(settings, "paper_risk_judge_reject_binding", False):
        return _LITE_RISK_JUDGE_TEMPLATE
    context_line = (
        f"Current portfolio context: {portfolio_context}\n" if portfolio_context else ""
    )
    return _LITE_RISK_JUDGE_TEMPLATE.replace(
        "Trader recommendation: {trader_action} (confidence: {trader_confidence})\n\n",
        "Trader recommendation: {trader_action} (confidence: {trader_confidence})\n"
        + context_line.replace("{", "{{").replace("}", "}}")
        + "\n",
    )


def _build_portfolio_sector_context(positions: list[dict]) -> str:
    """phase-57.1 (F-8): compact sector-weight summary computed ONCE per cycle
    (positions are identical for every ticker in the fan-out; per-ticker BQ
    reads would be N redundant calls + a race). Uses the same
    quantity*(current_price or avg_entry_price) fallback idiom as the
    paper_trader marks (mark_to_market has not yet run at the call site)."""
    weights: dict[str, float] = {}
    total = 0.0
    for pos in positions or []:
        try:
            qty = float(pos.get("quantity") or 0)
            px = float(pos.get("current_price") or pos.get("avg_entry_price") or 0)
            val = qty * px
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        sector = (pos.get("sector") or "Unknown").strip() or "Unknown"
        weights[sector] = weights.get(sector, 0.0) + val
        total += val
    if total <= 0:
        return "no open positions (all cash)"
    parts = [
        f"{sector} {val / total * 100:.1f}%"
        for sector, val in sorted(weights.items(), key=lambda kv: -kv[1])
    ]
    return "invested-book sector weights: " + "; ".join(parts)


def _select_lite_analyzer(model_name, settings=None):
    """Factory: pick the lite-analyzer coroutine for the configured standard model.

    phase-27.3 (C2): the lite fallback was hardcoded to Claude only, so
    selecting the Gemini 2.5 Flash workhorse (`GEMINI_WORKHORSE`) as the
    standard model bricked the safety net ("standard model … is not a
    Claude model" raise). The factory
    dispatches by model-name prefix:
      - `gemini-*` -> `_run_gemini_analysis` (direct AI Studio API key)
      - anything else (default `claude-*`) -> `_run_claude_analysis`

    phase-72.0.2 (DARK): with `settings.paper_rail_failforward_enabled` ON, a
    `claude-*` standard model whose cc_rail is dead for the cycle dispatches to
    `_run_failforward_analysis` (Gemini substitute under the quality floor)
    instead of a `_run_claude_analysis` that can only return rail_guard_skipped
    empties. `settings=None` (the pre-72.0.2 signature) keeps today's routing.

    Returns the coroutine FUNCTION (uncalled). Callers do
    `await _select_lite_analyzer(name)(ticker, settings)`.
    """
    name = (model_name or "").strip().lower()
    if name.startswith("gemini-"):
        return _run_gemini_analysis
    if (
        name.startswith("claude-")
        and settings is not None
        and getattr(settings, "paper_rail_failforward_enabled", False)
        and _rail_dead_reason() is not None
    ):
        return _run_failforward_analysis
    return _run_claude_analysis


def _rail_dead_reason():
    """phase-72.0.2: the reason the cc_rail is dead this cycle, else None.

    Strict READER of the public rail_guard_status() -- never touches the
    mutators, so a fail-forward decision can never feed the Claude breaker
    (criterion 2 / Azure resource-differentiation). Fail-open: any error
    reads as rail-healthy so the legacy path serves.
    """
    try:
        from backend.agents.claude_code_client import rail_guard_status

        _rg = rail_guard_status()
        if _rg.get("rail_skipped"):
            return str(_rg.get("disabled_reason") or "probe gate")
        if _rg.get("breaker_tripped"):
            return str(_rg.get("last_error") or "breaker open")
    except Exception:
        pass
    return None


def _failforward_floor_ok(analysis) -> bool:
    """phase-72.0.2 quality floor over the inner trader-analysis dict.

    Two deterministic $0 stages (research_brief_72.0.2.md section 3.4 --
    schema-validity is NOT quality, arXiv:2604.25359, so parse-success alone
    never clears the floor):
      1. structural gate -- dict; action in {BUY,SELL,HOLD}; confidence
         numeric in [0,100] and not None; score numeric in [1,10]; non-empty
         reason (the "hardening rule": a payload missing structure carries no
         semantic score).
      2. degenerate-signature rejection -- the `_parse_failed` marker or
         confidence == 0 (the fabricated-HOLD tell, mirrored from
         _degraded_scoring_check) means this is NOT a real score.
    Floor-fail hands the result to the honest-degraded path; it is never
    fabricated into a tradeable value.
    """
    if not isinstance(analysis, dict):
        return False
    if analysis.get("_parse_failed"):
        return False
    if analysis.get("action") not in ("BUY", "SELL", "HOLD"):
        return False
    conf_raw = analysis.get("confidence")
    score_raw = analysis.get("score")
    if conf_raw is None or score_raw is None:
        return False
    try:
        conf = float(conf_raw)
        score = float(score_raw)
    except (TypeError, ValueError):
        return False
    if not (0.0 <= conf <= 100.0 and 1.0 <= score <= 10.0):
        return False
    if not str(analysis.get("reason") or "").strip():
        return False
    if conf == 0.0:
        return False
    return True


def _build_failforward_client(ff_model):
    """phase-72.0.2 cycle-2 (Q/A F1): the fail-forward's VERTEX-only transport.

    Criterion 1 names Vertex-Gemini. Routing the substitute through
    make_client's normal priority serves the AI-Studio key branch when
    GEMINI_API_KEY is set and GeminiClient(model=None) -- the None-trap --
    when it is not, making the real-score property silently depend on a key
    the contract never named. Build the same in-seam ADC bundle Seam A uses
    instead; the transport is now caller- and key-independent.
    Returns None fail-open when no ADC genai client is available.
    """
    from backend.agents.llm_client import GeminiClient, _build_vertex_bundle

    bundle = _build_vertex_bundle(ff_model)
    if bundle is None:
        return None
    return GeminiClient(model=bundle, model_name=ff_model)


async def _run_failforward_analysis(ticker, settings, portfolio_context=""):
    """phase-72.0.2 (DARK): rail-dead fail-forward for the lite path.

    Serves the lite analysis from the Gemini substitute model under the
    deterministic quality floor, stamping fail-forward provenance so a
    substituted answer is auditable and can never masquerade as a rail-served
    one (the repo-local gen_ai.fallback.* analogue). Floor-fail marks the
    result `_degraded` -- the honest 61.2 path (`_fold_degraded_for_trading`
    drops it from decide_trades under the integrity flag) -- never fabricates.
    Transport is Vertex-only via `_build_failforward_client` (Q/A F1).
    """
    ff_model = str(getattr(settings, "paper_failforward_model", "") or "").strip()
    reason = _rail_dead_reason() or "rail dead"
    if not ff_model.startswith("gemini-"):
        # Misconfigured substitute: fail-open to the legacy claude path,
        # which serves exactly what it serves today on a dead rail.
        logger.warning(
            "phase-72.0.2: paper_failforward_model=%r is not gemini-*; "
            "falling back to the legacy lite path for %s", ff_model, ticker,
        )
        return await _run_claude_analysis(
            ticker, settings, portfolio_context=portfolio_context,
        )
    client = _build_failforward_client(ff_model)
    if client is None:
        logger.warning(
            "phase-72.0.2: no ADC genai client for the fail-forward Vertex "
            "bundle; falling back to the legacy lite path for %s", ticker,
        )
        return await _run_claude_analysis(
            ticker, settings, portfolio_context=portfolio_context,
        )
    result = await _run_gemini_analysis(
        ticker, settings, portfolio_context=portfolio_context,
        model_override=ff_model, client_override=client,
    )
    inner = (result.get("full_report") or {}).get("analysis") if isinstance(result, dict) else None
    if isinstance(result, dict):
        result["_failforward"] = True
        result["_failforward_provider"] = ff_model
        result["_failforward_reason"] = str(reason)[:500]
        if not _failforward_floor_ok(inner):
            result["_degraded"] = True
            result["_failforward_reason"] = f"floor_reject: {str(reason)[:480]}"
            logger.warning(
                "phase-72.0.2: fail-forward result for %s REJECTED by the "
                "quality floor -- honest degraded row, not a fabricated score",
                ticker,
            )
    return result


def _fold_degraded_for_trading(analysis: dict | None) -> dict | None:
    """phase-61.2 (cycle-173 seam extraction): a _degraded analysis leaves its
    honest BQ row but must NEVER enter the trade-decision inputs -- a NULL
    recommendation would crash decide_trades at portfolio_manager:114, and a
    fabricated value is the bug 61.2 exists to fix. Extracted from the
    _run_and_persist_one closure so the guard is BEHAVIOURALLY testable (the
    prior source-scan test was proven vacuous by the cycle-2 Q/A: a
    comment-only module satisfied it)."""
    if analysis and analysis.get("_degraded"):
        return None
    return analysis


def _degraded_scoring_check(analyses: list[dict]) -> tuple[bool, int, int]:
    """phase-56.2 (F-5) pure predicate: (fire, n_degraded, n_total).

    An analysis is degraded when final_score/score == 0, or when
    confidence == 0 with an UPPERCASE recommendation (the rail-down
    fallback's tell, 55.2 F-D). Fire when ALL are degraded or >= 3 are.
    """
    n_total = len(analyses)
    n_degraded = 0
    for a in analyses:
        try:
            score_zero = float(a.get("final_score") or a.get("score") or 0) == 0.0
            rec = str(a.get("recommendation") or a.get("action") or "")
            # NB: `or`-defaulting would turn a REAL confidence of 0 into the
            # default (the falsy-zero trap) -- check None explicitly.
            _conf_raw = a.get("confidence")
            conf_zero_upper = (
                _conf_raw is not None
                and float(_conf_raw) == 0.0
                and rec.isupper()
                and bool(rec)
            )
            # phase-70.4 (G3-B): a lite parse-fail (or an explicitly-degraded row) is
            # degraded even though it carries a fabricated score-5 HOLD -- count it so
            # the P1 degraded-scoring alert fires (this affects only the ALERT, not any trade).
            if score_zero or conf_zero_upper or a.get("_parse_failed") or a.get("_degraded"):
                n_degraded += 1
        except (TypeError, ValueError):
            n_degraded += 1
    fire = n_total > 0 and (n_degraded == n_total or n_degraded >= 3)
    return fire, n_degraded, n_total


def _fallback_rate_check(
    analyses: list[dict], threshold: float = 0.5,
) -> tuple[bool, int, int, dict[str, str]]:
    """phase-60.1 (AW-4) pure predicate: (fire, n_fallback, n_total, reasons).

    Counts analyses that INTENDED the full pipeline but landed on the lite
    fallback -- tagged `_fallback_reason` at the fallback site in
    `_run_single_analysis`. Deliberate lite_mode analyses carry no tag and
    can never fire this. Fires when the fallback fraction strictly EXCEEDS
    `threshold` (default 0.5 per the 60.1 spec): 3/5 fires, 2/4 does not.
    The away-week case (everything fell back, 100%) always fires.
    """
    n_total = len(analyses)
    reasons: dict[str, str] = {}
    for a in analyses:
        if not isinstance(a, dict):
            continue
        if a.get("_fallback_reason"):
            reasons[str(a.get("ticker") or "?")] = str(a["_fallback_reason"])[:200]
    n_fallback = len(reasons)
    fire = n_total > 0 and (n_fallback / n_total) > threshold
    return fire, n_fallback, n_total, reasons


def _degradation_summary_fields(
    fire: bool, n_fallback: int, n_total: int, reasons: dict[str, str],
) -> dict:
    """phase-86.38 pure predicate: what a cycle RECORDS about its own degradation.

    Extracted as a real seam rather than left inline, because the inline version
    could only be guarded by asserting the ORDER of source text -- and a mutation
    that disabled it (`if _n_fb_total:` -> `if False:`) left that order untouched
    and SURVIVED the matrix. A guard that cannot fail when its subject is broken
    does not count, so the subject was moved somewhere a test can execute it.

    Returns `{}` only when the cycle analysed nothing. Otherwise the rate is
    ALWAYS reported, together with whether it paged -- a reader must be able to
    tell "quiet because fine" from "quiet because below threshold", which is the
    distinction the 2026-08-10 cycle -- 3 of its 6 tickers on the lite
    fallback, no page -- had no way to record. (The alarm's own denominator
    was not measured; see the call site.)

    Reporting is not paging: this function never decides whether to alert.
    `_fallback_rate_check` owns that and is untouched.
    """
    if not n_total:
        return {}
    out: dict = {
        "fallback_rate": f"{n_fallback}/{n_total}",
        "fallback_alarm_fired": bool(fire),
    }
    if reasons:
        out["fallback_reasons"] = dict(reasons)
    return out


#: phase-86.38 -- the keys that make a cycle's DEGRADATION legible after the fact.
#: Kept as a module constant so a test can assert the SET, not just the plumbing.
DEGRADATION_RECORD_KEYS = (
    "fallback_rate", "fallback_alarm_fired", "fallback_reasons",
    "degraded", "degraded_analyses", "meta_scorer_degraded",
)


def _degradation_record(summary: dict) -> dict:
    """phase-86.38 cycle 2: the degradation facts persisted on the cycle record.

    EXTRACTED BECAUSE THE WIRING HAD NO GUARD. The cycle-1 Q/A (which dropped
    before returning, but got this far) mutated the call site by deleting
    `degradation=_degradation,` from `record_cycle_end(...)` and the whole suite
    stayed GREEN -- 7 passed. Under that mutant every future cycle persists
    `degradation: {}`, i.e. the exact defect this step exists to remove returns
    silently. Dropping keys from the tuple survived too.

    That is the guards-stop-one-seam-short class: I had guarded
    summary -> `_degradation_summary_fields` and left
    `_degradation` -> `record_cycle_end` uncovered, which is the half that
    actually reaches durable storage.
    """
    return {k: summary.get(k) for k in DEGRADATION_RECORD_KEYS
            if summary.get(k) is not None}


def _all_conviction_fallback(candidates: list[dict]) -> bool:
    """phase-56.2 (F-7) pure predicate: True when EVERY candidate's
    conviction came from the no-LLM fallback (the damping overlay is dead)."""
    return bool(candidates) and all(
        "fallback (LLM unavailable)" in str(c.get("conviction_reason") or "")
        for c in candidates
    )


_CONVICTION_STREAK_PATH = (
    Path(__file__).parent.parent.parent / "handoff" / ".conviction_fallback_streak.json"
)


def _bump_conviction_fallback_streak(delta: int) -> int:
    """phase-61.2 (criterion 4): durable consecutive-all-fallback-cycle
    counter. delta=1 increments, delta=0 resets. File-backed (module state
    dies on the routine kickstart restarts -- same rationale as the cycle
    heartbeat). Never raises; on any IO/parse error returns the in-flight
    value so the caller's alert logic stays fail-open."""
    streak = 0
    try:
        if _CONVICTION_STREAK_PATH.exists():
            streak = int(
                json.loads(_CONVICTION_STREAK_PATH.read_text(encoding="utf-8")).get(
                    "streak", 0
                )
            )
    except Exception:
        streak = 0
    streak = streak + 1 if delta else 0
    try:
        _CONVICTION_STREAK_PATH.write_text(
            json.dumps({"streak": streak}), encoding="utf-8"
        )
    except Exception as exc:  # fail-open: alerting still works this cycle
        logger.warning("conviction-fallback streak write failed (non-fatal): %s", exc)
    return streak


def _log_claude_code_call(
    envelope: dict | None, *, agent: str, ticker: str, ok: bool,
    requested_model: str | None = None,
) -> None:
    """phase-56.2 (55.3 finding F-6): meter the claude-CLI rail into
    llm_call_log. The away week wrote ZERO rows for 6/7 cycles because this
    rail never called log_llm_call (only the SDK clients did) -- the burn/
    skill audit was blind. Fail-open: a metering bug must never break the
    analyzer. cycle_id/session_cost auto-populate from module state."""
    try:
        from backend.services.observability.api_call_log import log_llm_call
        usage = (envelope or {}).get("usage") or {}
        # phase-78.2 (criterion 3, 78.2 Q/A finding): this is the SECOND CC-rail
        # logger -- B1 (lite trader) and B2 (lite risk judge) come through here,
        # NOT through ClaudeCodeClient._log_cc_call. It previously read the
        # envelope's TOP-LEVEL `model` key and fell back to the literal
        # "claude-code-cli", so it never saw modelUsage and could not name the
        # model that actually ran. Same resolver as the other logger, so the two
        # seams cannot drift.
        from backend.agents.claude_code_client import resolve_rail_model
        _fallback_label = str((envelope or {}).get("model") or "claude-code-cli")
        resolved = resolve_rail_model(envelope, requested_model) or _fallback_label
        if requested_model and resolved != requested_model:
            logger.warning(
                "rail model MISMATCH (lite path): requested=%s resolved=%s agent=%s ticker=%s",
                requested_model, resolved, agent, ticker,
            )
        log_llm_call(
            provider="claude-code",
            model=resolved,
            agent=agent,
            latency_ms=float((envelope or {}).get("duration_ms") or 0.0),
            input_tok=int(usage.get("input_tokens") or 0),
            output_tok=int(usage.get("output_tokens") or 0),
            cache_creation_tok=int(usage.get("cache_creation_input_tokens") or 0),
            cache_read_tok=int(usage.get("cache_read_input_tokens") or 0),
            ok=ok,
            ticker=ticker,
        )
    except Exception as exc:
        logger.debug("claude-code llm_call_log metering failed (non-fatal): %s", exc)


def _fetch_yf_market_data(ticker: str):
    """phase-60.4 (AW-1 residual, criterion 3): the yfinance fetch as ONE
    sync unit for asyncio.to_thread. `stock.info` + `.history` are blocking
    network calls; naked inside the async lite analyzers they stalled the
    event loop (the watchdog's away-week ReadTimeouts while a cycle ran).
    Returns (info, hist)."""
    import yfinance as yf

    stock = yf.Ticker(ticker)
    return stock.info, stock.history(period="3mo")


async def _run_claude_analysis(
    ticker: str, settings: Settings, portfolio_context: str = "",
) -> dict:
    """Lightweight Claude-based analysis for paper trading decisions.

    phase-57.1 (F-8): `portfolio_context` is the per-cycle sector summary for
    the RiskJudge prompt; consumed only when paper_risk_judge_reject_binding
    is ON (the prompt builders return verbatim constants when OFF)."""
    import anthropic

    logger.info(f"Claude analysis: analyzing {ticker}")

    # Fetch current market data via yfinance.
    # phase-60.4 (criterion 3): off the event loop -- names preserved so the
    # 60.3 integrity wiring below is untouched.
    info, hist = await asyncio.to_thread(_fetch_yf_market_data, ticker)

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

    # phase-60.3 (AW-9): deterministic decision-input integrity. Normalize +
    # flag ALWAYS (provenance is ungated observability); ENFORCE blocking
    # flags pre-LLM only when paper_data_integrity_enabled (default OFF).
    from backend.services.data_integrity import (
        check_data_integrity, normalize_market_values, render_market_lines,
    )
    _di_enabled = bool(getattr(settings, "paper_data_integrity_enabled", False))
    _di_norm = normalize_market_values(ticker, info)
    _di_flags = check_data_integrity(ticker, info, _di_norm)
    _model_for_block = (settings.gemini_model or "claude-sonnet-4-6").strip()
    if _di_enabled and any(f.get("blocking") for f in _di_flags):
        return _data_integrity_blocked_analysis(
            ticker, name, sector, industry, _model_for_block, current_price,
            market_cap, pe_ratio, momentum_20d, momentum_60d, _di_norm, _di_flags,
        )
    _di_market_lines = render_market_lines(
        ticker, current_price, market_cap, pe_ratio, _di_norm, _di_enabled,
    )
    # USD-true market cap for the risk-judge template (flag ON, non-US);
    # OFF/US byte-identical raw value.
    _di_mcap_b = (
        (_di_norm.get("market_cap_usd") or 0) / 1e9
        if _di_enabled and not _di_norm.get("is_us") and _di_norm.get("market_cap_usd") is not None
        else (market_cap or 0) / 1e9
    )

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

    # phase-cycle-3 (2026-05-26): rail-selection log + Claude Code CLI route.
    # When settings.paper_use_claude_code_route is True, route through the
    # `claude` CLI subprocess on the Max-subscription flat-fee rail instead
    # of api.anthropic.com direct billing. Bypasses credit-exhaustion in
    # the testing phase. Per-rail log per Yin et al. 2026 implementation-
    # risk framework so A/B integrity is preserved when we later compare.
    use_claude_code_route = bool(getattr(settings, "paper_use_claude_code_route", False))
    logger.info(
        "Analysis ticker=%s rail=%s",
        ticker,
        "claude_code" if use_claude_code_route else "anthropic_direct",
    )

    api_key = settings.anthropic_api_key.get_secret_value() or os.getenv("ANTHROPIC_API_KEY", "")
    if not use_claude_code_route and not api_key:
        raise ValueError("No ANTHROPIC_API_KEY available")

    # Only instantiate the direct-API client when actually using that rail.
    # When the CC route is active, the rail call below shells out to
    # `claude` CLI via claude_code_invoke and never touches api.anthropic.com.
    client = anthropic.Anthropic(api_key=api_key) if not use_claude_code_route else None
    prompt = f"""Analyze {ticker} ({name}) for a paper trading portfolio. Be concise.

Stock: {ticker} ({name})
Sector: {sector} | Industry: {industry}
{_di_market_lines}
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

    if use_claude_code_route:
        from backend.agents.claude_code_client import (
            ClaudeCodeError,
            claude_code_invoke,
            extract_result_text,
        )
        try:
            envelope = await asyncio.to_thread(
                claude_code_invoke,
                prompt,
                max_tokens=200,
                timeout_s=120,
                # phase-78.2: pin the SAME model the metered branch below uses
                # (:2392, guarded to be a claude-* id). Without it this rail
                # ran the CLI session default -- measured as claude-opus-5[1m]
                # -- for a live trading decision, while the log said otherwise.
                model=model_name,
            )
            text = extract_result_text(envelope).strip()
            # phase-56.2 (55.3 finding F-6): the away week wrote ZERO llm_call_log
            # rows for 6/7 cycles because this rail never metered. Fail-open.
            _log_claude_code_call(envelope, agent="lite_trader", ticker=ticker, ok=True,
                                  requested_model=model_name)
        except ClaudeCodeError as exc:
            logger.warning(
                "claude_code rail failed for %s: %s -- returning empty text",
                ticker, exc,
            )
            text = ""
            _log_claude_code_call(None, agent="lite_trader", ticker=ticker, ok=False,
                                  requested_model=model_name)
    else:
        response = await asyncio.to_thread(
            client.messages.create,
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    # Extract JSON from response
    import re
    json_match = re.search(r'\{[^}]+\}', text)
    if json_match:
        analysis = json_io.loads(json_match.group())
    else:
        # phase-70.4 (G3-A): a parse failure is NOT a genuine HOLD. Mark it so the
        # degraded guard counts it (G3-B) and, under paper_synthesis_integrity_enabled,
        # so it is dropped from decide_trades input (G3-C) rather than silently
        # suppressing a BUY as a fabricated score-5 neutral.
        analysis = {"action": "HOLD", "confidence": 0, "score": 5,
                    "reason": "Could not parse analysis", "_parse_failed": True}
        logger.warning(
            "phase-70.4: lite analysis parse FAILED for %s -- degraded HOLD placeholder "
            "(not a genuine neutral); response head=%r", ticker, (text or "")[:120],
        )

    if analysis.get("_parse_failed"):
        logger.info("Claude analysis for %s: PARSE-FAILED -> degraded HOLD placeholder", ticker)
    else:
        logger.info(f"Claude analysis for {ticker}: {analysis['action']} (confidence={analysis['confidence']}, score={analysis['score']})")

    # phase-25.A: SECOND, INDEPENDENT LLM call -- the Risk Judge. Closes
    # phase-24.4 F-1 (the lite path previously aliased the trader's reason
    # into risk_assessment). The risk judge system prompt forces evaluation
    # along volatility/concentration/valuation axes -- it does NOT validate
    # the trader's recommendation. Cost impact: ~$0.003/ticker, already
    # accounted in the existing $0.01/ticker ceiling.
    # phase-57.1 (F-8): builders return the verbatim constants when the flag
    # is OFF (byte-identical prompts); flag ON corrects the cap + injects the
    # per-cycle sector context.
    _rj_system = _build_risk_judge_system(settings)
    risk_prompt = _build_risk_judge_template(settings, portfolio_context).format(
        ticker=ticker,
        name=name,
        sector=sector,
        pe_ratio=pe_ratio or 0.0,
        market_cap_b=_di_mcap_b,  # phase-60.3: USD-true when flag ON + non-US
        momentum_20d=momentum_20d,
        momentum_60d=momentum_60d,
        trader_action=analysis["action"],
        trader_confidence=analysis["confidence"],
    )
    try:
        if use_claude_code_route:
            from backend.agents.claude_code_client import (
                ClaudeCodeError,
                claude_code_invoke,
                extract_result_text,
            )
            try:
                risk_envelope = await asyncio.to_thread(
                    claude_code_invoke,
                    risk_prompt,
                    max_tokens=300,
                    system=_rj_system,
                    timeout_s=120,
                    # phase-78.2: same pin as the lite trader above -- the risk
                    # judge is the independent second opinion, so it must not
                    # silently run a different tier than the call it checks.
                    model=model_name,
                )
                risk_text = extract_result_text(risk_envelope).strip()
                # phase-56.2 (F-6): meter the risk-judge leg of the CLI rail.
                _log_claude_code_call(risk_envelope, agent="lite_risk_judge", ticker=ticker, ok=True,
                                      requested_model=model_name)
            except ClaudeCodeError as exc:
                logger.warning(
                    "claude_code risk-judge rail failed for %s: %s",
                    ticker, exc,
                )
                risk_text = ""
                _log_claude_code_call(None, agent="lite_risk_judge", ticker=ticker, ok=False,
                                      requested_model=model_name)
        else:
            risk_response = await asyncio.to_thread(
                client.messages.create,
                model=model_name,
                max_tokens=300,
                system=_rj_system,
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

    # phase-86.86 (D6): ONE producer, shared with the Gemini lite path.
    risk_assessment = _build_lite_risk_assessment(risk_dict, ticker)
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
        # phase-70.4 (G3-A/B): mark a parse failure so the degraded-scoring guard counts
        # it and it is not mistaken for a genuine HOLD. (G3-C) under paper_synthesis_
        # integrity_enabled, ALSO mark it _degraded so the cycle loop (:~1088) drops it
        # from decide_trades input -- fail-safe (removing a spurious neutral can never
        # create a BUY; a genuine parsed HOLD is untouched). OFF -> _degraded absent (legacy).
        "_parse_failed": bool(analysis.get("_parse_failed")),
        **({"_degraded": True} if analysis.get("_parse_failed") and getattr(settings, "paper_synthesis_integrity_enabled", False) else {}),
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
            "market_data": _integrity_market_data(
                name, current_price, market_cap, pe_ratio, sector, industry,
                momentum_20d, momentum_60d, _di_norm, _di_flags,
            ),
        },
    }


def _resolve_lite_gemini_model(settings, model_override=None) -> str:
    """phase-72.0.2: resolve + validate the lite-Gemini model name.

    Extracted from the body of `_run_gemini_analysis` so the override plumb is
    behaviourally testable without any I/O, and so a misconfigured model fails
    BEFORE market-data fetching. `model_override` (the fail-forward substitute)
    wins over `settings.gemini_model`; the Gemini-only guard is preserved
    verbatim from the phase-27.3 seam.
    """
    from backend.config.model_tiers import GEMINI_WORKHORSE  # phase-75.5.2

    model_name = str(
        (model_override or settings.gemini_model) or GEMINI_WORKHORSE
    ).strip()
    if not model_name.startswith("gemini-"):
        raise ValueError(
            f"standard model '{model_name}' is not a Gemini model; "
            "_run_gemini_analysis is Gemini-only. _select_lite_analyzer should "
            "have routed claude-* to _run_claude_analysis instead."
        )
    return model_name


async def _run_gemini_analysis(
    ticker: str, settings: Settings, portfolio_context: str = "",
    model_override: str | None = None, client_override=None,
) -> dict:
    """Lightweight Gemini-based analysis for paper trading decisions.

    phase-27.3 (C2): mirror of `_run_claude_analysis` for non-Claude standard
    models. Output dict shape IDENTICAL — same keys, `_path: "lite"` marker,
    so `_persist_analysis` and downstream readers don't branch by provider.
    By default routes through `make_client` (post-27.1 priority order) which
    dispatches `gemini-*` to a direct AI Studio API key (no Vertex / GCP
    creds).

    phase-72.0.2: `model_override` (default None = today's behaviour) lets the
    rail-dead fail-forward substitute its own model without touching
    `settings.gemini_model`; resolution + the Gemini-only guard live in
    `_resolve_lite_gemini_model` and run before any I/O. `client_override`
    (cycle-2, Q/A F1) lets the fail-forward inject its Vertex-only ADC client
    so the substitute never depends on GEMINI_API_KEY and never hits the
    None-trap; both defaults preserve the legacy path byte-identically.

    Two-LLM-call pattern preserved: trader prompt + independent risk-judge.
    """
    import re as _re
    from backend.agents.llm_client import make_client, safe_text

    model_name = _resolve_lite_gemini_model(settings, model_override)

    logger.info(f"Gemini analysis: analyzing {ticker}")

    # 1. Market data via yfinance (parity with Claude path).
    # phase-60.4 (criterion 3): off the event loop (mirror of the Claude path).
    info, hist = await asyncio.to_thread(_fetch_yf_market_data, ticker)

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

    # phase-60.3 (AW-9): same deterministic integrity wiring as the Claude
    # mirror (output-shape parity preserved).
    from backend.services.data_integrity import (
        check_data_integrity, normalize_market_values, render_market_lines,
    )
    _di_enabled = bool(getattr(settings, "paper_data_integrity_enabled", False))
    _di_norm = normalize_market_values(ticker, info)
    _di_flags = check_data_integrity(ticker, info, _di_norm)
    # phase-72.0.2: stamp the model that actually serves (the override when
    # the fail-forward is active), resolved once at the function top.
    _model_for_block = model_name
    if _di_enabled and any(f.get("blocking") for f in _di_flags):
        return _data_integrity_blocked_analysis(
            ticker, name, sector, industry, _model_for_block, current_price,
            market_cap, pe_ratio, momentum_20d, momentum_60d, _di_norm, _di_flags,
        )
    _di_market_lines = render_market_lines(
        ticker, current_price, market_cap, pe_ratio, _di_norm, _di_enabled,
    )
    _di_mcap_b = (
        (_di_norm.get("market_cap_usd") or 0) / 1e9
        if _di_enabled and not _di_norm.get("is_us") and _di_norm.get("market_cap_usd") is not None
        else (market_cap or 0) / 1e9
    )

    # phase-72.0.2: model resolution + the Gemini-only guard moved to
    # _resolve_lite_gemini_model at the function top (before any I/O).

    # Build a single Gemini client and reuse for trader + risk-judge calls.
    # phase-72.0.2 cycle-2 (Q/A F1): the fail-forward injects its Vertex-only
    # client via client_override; the default (None) path is byte-identical.
    client = (
        client_override
        if client_override is not None
        else make_client(model_name, vertex_model=None, settings=settings)
    )

    trader_prompt = f"""Analyze {ticker} ({name}) for a paper trading portfolio. Be concise.

Stock: {ticker} ({name})
Sector: {sector} | Industry: {industry}
{_di_market_lines}
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
        # phase-56.2 (F-6): _role/_ticker make GeminiClient's existing
        # log_llm_call instrumentation stamp agent+ticker (was NULL/NULL).
        {"max_output_tokens": 200, "temperature": 0.0, "response_mime_type": "application/json",
         "_role": "lite_trader", "_ticker": ticker},
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
    # phase-57.1 (F-8): builders return verbatim constants when the flag is OFF.
    risk_prompt = (
        _build_risk_judge_system(settings)
        + "\n\n"
        + _build_risk_judge_template(settings, portfolio_context).format(
            ticker=ticker,
            name=name,
            sector=sector,
            pe_ratio=pe_ratio or 0.0,
            market_cap_b=_di_mcap_b,  # phase-60.3: USD-true when flag ON + non-US
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
            # phase-56.2 (F-6): tag the risk-judge leg for llm_call_log.
            {"max_output_tokens": 300, "temperature": 0.0, "response_mime_type": "application/json",
             "_role": "lite_risk_judge", "_ticker": ticker},
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

    # phase-86.86 (D6): ONE producer, shared with the Claude lite path.
    risk_assessment = _build_lite_risk_assessment(risk_dict, ticker)
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
            "market_data": _integrity_market_data(
                name, current_price, market_cap, pe_ratio, sector, industry,
                momentum_20d, momentum_60d, _di_norm, _di_flags,
            ),
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
        # phase-60.1 (AW-4): stamp lite/full provenance INTO the persisted
        # JSON so BQ readers (digest, reports API) can distinguish a 2-call
        # lite-wrapper score from a full 28-agent score. The away week wrote
        # 64 lite rows that looked identical to full rows.
        if isinstance(full_report, dict) and analysis.get("_path"):
            full_report = {**full_report, "_path": analysis["_path"]}
            if analysis.get("_fallback_reason"):
                full_report["_fallback_reason"] = str(analysis["_fallback_reason"])[:500]
        # phase-61.2 (criterion 1c): honest-absence passthrough. Marker rows
        # keep NULL score/recommendation instead of being re-fabricated into
        # 0.0/"Hold" by the coercions below (this function was the SECOND
        # fabrication site). $._degraded mirrors the $._path JSON-marker idiom.
        _degraded = bool(analysis.get("_degraded"))
        if _degraded and isinstance(full_report, dict):
            full_report = {
                **full_report,
                "_degraded": True,
                "_degraded_reason": str(analysis.get("_degraded_reason") or "")[:500],
            }
        market_data = full_report.get("market_data") or {}
        # phase-86.74 (criterion 4): persist the RiskJudge verdict into its own
        # columns. This writer -- the one the AUTONOMOUS LOOP actually uses --
        # never passed them, so `risk_judge_decision`, `risk_level` and
        # `recommended_position_pct` were empty on 129 of 129 rows across
        # 2026-07-20..2026-08-13 while `save_report` accepted all three the whole
        # time (bigquery_client.py:119,148,149). The rich write in
        # tasks/analysis.py:273,302,303 is a DIFFERENT path (API-triggered), which
        # is why tracing that one did not explain the empty columns.
        #
        # The verdict itself was never lost -- it sits in the JSON blob at
        # $.final_synthesis.risk_assessment.judge -- but a JSON path is not an
        # auditable column, which is exactly why this incident had to be
        # reconstructed from log timestamps by elimination.
        #
        # Resolution is nested-first, matching portfolio_manager and
        # api/analysis.py:158; `or {}` guards a present-but-null `judge` key,
        # which `.get(k, {})` alone would not.
        _ra = (analysis.get("risk_assessment") or {})
        _judge = (_ra.get("judge") or {}) if isinstance(_ra, dict) else {}
        if not isinstance(_judge, dict):
            _judge = {}
        _rj = _judge or (_ra if isinstance(_ra, dict) else {})
        _rj_pct = _rj.get("recommended_position_pct")
        try:
            _rj_pct = float(_rj_pct) if _rj_pct is not None else None
        except (ValueError, TypeError):
            _rj_pct = None
        await asyncio.to_thread(
            bq.save_report,
            ticker=ticker,
            risk_judge_decision=str(_rj.get("decision") or ""),
            risk_level=str(_rj.get("risk_level") or ""),
            recommended_position_pct=_rj_pct,
            # phase-61.2 (criterion 3, ungated pure fix): full-path reports
            # carry no market_data key (lite-only), so every full-path
            # autonomous row persisted NULL company_name; the quant dict is
            # present on the full path and carries the name.
            company_name=(
                market_data.get("name")
                or (full_report.get("quant") or {}).get("company_name")
                or None
            ),
            final_score=None if _degraded else float(analysis.get("final_score") or 0.0),
            recommendation=None if _degraded else (analysis.get("recommendation") or "Hold"),
            summary=(
                ("DEGRADED: " + str(analysis.get("_degraded_reason") or "")[:400])
                if _degraded
                else (analysis.get("risk_assessment") or {}).get("reason", "") or ""
            ),
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
            # phase-86.25 (S1). WAS: `recommendation = trade.get(
            # "risk_judge_decision", "HOLD")` followed by an empty-string coercion
            # to the literal "HOLD". Two defects in one line.
            #
            # (a) `risk_judge_decision` is an APPROVAL vocabulary -- MEASURED over
            #     the same table this reads: APPROVE_REDUCED 15, REJECT 3,
            #     APPROVE_HEDGED 1, and NOT ONE value on the BUY/HOLD/SELL scale.
            #     Passing it to a parameter typed for an analyst recommendation is
            #     the phase-86.22 defect arriving through a different door.
            # (b) the "HOLD" coercion used an IN-DOMAIN value as the missing-data
            #     marker (PEP 661's named anti-pattern), so a reader could not tell
            #     "the analyst said hold" from "we had nothing".
            #
            # The vocabulary is now resolved AT THE BOUNDARY: hand over a real
            # analyst recommendation or nothing. Nothing is what is available, so
            # this resolves to UNKNOWN today, and UNKNOWN is non-directional by
            # construction -- `directionally_correct` becomes False for an honest
            # reason (no direction was known) instead of a dishonest one (a
            # fabricated hold). The reason NOTHING is available is stated below,
            # and it is not the one this comment used to give.
            # WHY THE (A) BRANCH IS DEAD, CORRECTED cycle 2 (Q/A finding W1). An earlier
            # version of this comment blamed the unreachable ANCHOR -- analysis_id empty on
            # 32/32 SELLs, round_trip_id one-sided 32/32 SELL vs 0/33 BUY. Those numbers are
            # real, but they are NOT the operative cause, and citing them told a future
            # reader that fixing round_trip_id would make this path resolve. IT WOULD NOT.
            #
            # MEASURED: `analyst_recommendation` is not a column of paper_trades at all (18
            # columns), and `_production_fns.LEDGER_FETCH_SQL` selects ten named columns,
            # none of them this one. The lookup below therefore reads a dict key that NO
            # PRODUCER EMITS: the branch is dead BY CONSTRUCTION, not by a missing join.
            #
            # CONSEQUENCE, stated so nobody credits it as coverage: making this resolve
            # needs a PRODUCER change -- some writer must start emitting an analyst
            # recommendation onto the trade -- which is a separate step. The call is kept
            # because it is the correct boundary SHAPE (a caller hands over a real
            # recommendation or nothing), not because it currently does anything.
            recommendation = resolve_outcome_recommendation(
                trade.get("analyst_recommendation")
            )
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
                # missing analysis_date kills the primary path.
                # NOTE (phase-47.7): bq.save_outcome APPENDS -- it is NOT an upsert
                # (corrected from a prior comment); re-running a sell-close could
                # duplicate the (ticker, analysis_date) row. Acceptable for v1
                # (sell-closes are one-shot per cycle); composite dedup is a follow-up.
                try:
                    sell_price = float(trade.get("price") or 0.0)
                    # phase-47.7: paper_trades rows carry `realized_pnl_pct` (written by
                    # paper_trader.execute_sell), NOT `return_pct`. Reading the
                    # non-existent return_pct silently recorded 0.0 return for EVERY
                    # sell-close -- the learn-loop's core value (true realized P&L) was
                    # always zero. Prefer the real field; keep return_pct as a fallback.
                    _rp = trade.get("realized_pnl_pct")
                    if _rp is None:
                        _rp = trade.get("return_pct")
                    pnl_pct = float(_rp or 0.0)
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
