"""
Stock universe screener — quant-only filter for paper trading candidates.
Uses yfinance batch download. Zero LLM cost.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# S&P 500 tickers — updated periodically. Good starting universe.
# In production, this could be loaded from a file or API.
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Sector ETF mapping for relative strength calculation
SECTOR_ETFS = {
    "Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Communication Services": "XLC",
    "Industrials": "XLI", "Consumer Staples": "XLP", "Energy": "XLE",
    "Utilities": "XLU", "Real Estate": "XLRE", "Materials": "XLB",
}


def get_sp500_tickers(as_of: datetime | None = None) -> list[str]:
    """Fetch S&P 500 ticker list.

    When `as_of` is None, returns the CURRENT index composition via
    Wikipedia scrape (survivorship-biased snapshot of today's
    membership). When `as_of` is a datetime, raises
    NotImplementedError -- we do not yet have a historical index-
    membership table, and silently returning today's list would
    reintroduce the survivorship bias this PIT kwarg is meant to
    prevent (phase-4.8.1). Backtest callers that need PIT-correct
    universe membership must either supply a cached historical list
    or wait for the delistings-feed ingestion (queued phase-4.8.x).
    """
    if as_of is not None:
        raise NotImplementedError(
            "point-in-time S&P 500 membership not available yet; "
            "callers must supply a cached historical universe or "
            "wait for the delistings-feed ingestion (phase-4.8.x)."
        )
    try:
        import io
        import urllib.request
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0 pyfinagent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html), header=0)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info(f"Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}")
        return _FALLBACK_TICKERS


def build_sector_map(tickers: Optional[list[str]] = None) -> dict[str, str]:
    """phase-51.2: {ticker: GICS sector} from the Wikipedia S&P 500 table (same
    source + User-Agent as get_sp500_tickers). Gives candidates a sector AT rank
    time so the sector-neutral lever is functional (it was a silent no-op because
    enrichment ran AFTER ranking). Tickers absent from the S&P 500 table (e.g.
    intl .DE/.KS) map to "" -> the global-pool fallback in rank_candidates. Cheap
    (one HTTP request); only called when a sector-aware flag is enabled (default
    OFF). NOTE: a 2026-06-01 replay found HARD sector-neutral HURTS long-only
    Sharpe (-0.166); the flag stays OFF -- this wiring keeps the lever
    live-measurable for a future SOFT-tilt variant."""
    try:
        import io
        import urllib.request
        req = urllib.request.Request(SP500_URL, headers={"User-Agent": "Mozilla/5.0 pyfinagent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        df = pd.read_html(io.StringIO(html), header=0)[0]
        full = {str(r["Symbol"]).strip().replace(".", "-"): str(r["GICS Sector"]).strip()
                for _, r in df.iterrows()}
    except Exception as e:
        logger.warning(f"build_sector_map: Wikipedia fetch failed: {e}")
        full = {}
    if tickers is None:
        return full
    return {t: full.get(t, "") for t in tickers}


def screen_universe(
    tickers: Optional[list[str]] = None,
    min_avg_volume: int = 100_000,
    min_price: float = 5.0,
    period: str = "6mo",
    sector_lookup: Optional[dict] = None,
    short_interest_lookup: Optional[dict[str, float]] = None,
    short_interest_threshold: float = 0.10,
) -> list[dict]:
    """
    Screen a universe of tickers using quant factors.
    Returns raw screening data for each ticker that passes basic filters.

    Cost: $0 (yfinance only, no LLM, no API keys)

    phase-23.1.13: optional `sector_lookup={ticker: sector}` is attached to
    each result dict so downstream rank_candidates / portfolio_manager can
    enforce sector concentration without extra yfinance fetches. Caller
    typically builds this via `_fetch_ticker_meta` (BQ-first / yfinance
    fallback). Backward compatible: when None, results lack the sector field
    and any sector-aware logic falls back to None / "Unknown" handling.

    phase-28.0 (2026-05-17): removed unused `min_market_cap` parameter. The
    parameter was accepted but never applied in the function body (only
    price + volume filters fire). Zero callers passed it. Re-add via a
    separate explicit step if market-cap filtering is needed beyond the
    inherent S&P 500 inclusion floor (currently ~$22.7B per S&P DJI 2024
    methodology update).

    phase-28.5 (2026-05-17): optional short-interest exclusion. When
    `short_interest_lookup={ticker: shortPercentOfFloat}` is provided, any
    ticker whose float-short ratio exceeds `short_interest_threshold`
    (default 0.10 = top-decile for S&P 500) is skipped. Source: Boehmer-
    Jones-Zhang 2008 documents 1.16%/mo underperformance for high-short
    stocks; Oxford RAPS 2022 confirms in 32 countries. Caller (typically
    autonomous_loop) builds the lookup via FINRA bimonthly CSV (preferred)
    or yfinance per-ticker fallback. When None or empty dict, no exclusion
    fires (back-compat).
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    logger.info(f"Screening {len(tickers)} tickers (period={period})")

    # Batch download price history
    try:
        data = yf.download(tickers, period=period, group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
    except Exception as e:
        logger.error(f"yfinance batch download failed: {e}")
        return []

    if data is None or data.empty:
        return []

    results = []
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None

            if ticker_data is None or ticker_data.empty:
                continue

            # phase-50.5: data-quality gate -- no-op for US (byte-identical);
            # drops unambiguous bad intl (.DE/.KS) bars before they feed
            # momentum/RSI/vol signals (the yfinance 11%-deviation /
            # identical-OHLC risk).
            from backend.tools.price_quality import validate_ohlcv
            from backend.backtest.markets import market_for_symbol
            ticker_data, _dq = validate_ohlcv(
                ticker_data, market=market_for_symbol(ticker), ticker=ticker
            )
            if ticker_data is None or ticker_data.empty:
                continue

            close = ticker_data["Close"].dropna()
            volume = ticker_data["Volume"].dropna()

            if len(close) < 20:
                continue

            current_price = float(close.iloc[-1])
            avg_vol = float(volume.tail(20).mean())

            # Basic filters
            if current_price < min_price or avg_vol < min_avg_volume:
                continue

            # phase-28.5: short-interest exclusion (high-short underperforms ~1.16%/mo per Boehmer-Jones-Zhang 2008).
            # Lookup is opt-in; built by caller via FINRA bimonthly CSV (preferred) or yfinance per-ticker fallback.
            # No exclusion fires when lookup is None or empty dict, preserving back-compat.
            if short_interest_lookup:
                short_pct = short_interest_lookup.get(ticker)
                if short_pct is not None and short_pct > short_interest_threshold:
                    logger.debug(
                        "Excluding %s: shortPercentOfFloat=%.3f > %.3f (phase-28.5)",
                        ticker, short_pct, short_interest_threshold,
                    )
                    continue

            # Momentum factors
            momentum_1m = _pct_change(close, 21)
            momentum_3m = _pct_change(close, 63)
            momentum_6m = _pct_change(close, len(close) - 1)

            # RSI (14-day)
            rsi = _compute_rsi(close, 14)

            # Volatility (annualized)
            daily_returns = close.pct_change().dropna()
            volatility = float(daily_returns.std() * (252 ** 0.5)) if len(daily_returns) > 5 else None

            # Mean reversion signal: distance from 50-day SMA
            sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else current_price
            sma_distance = (current_price - sma_50) / sma_50 * 100

            # phase-28.7: 52-week-high proximity (George-Hwang 2004 anchoring effect).
            # current_price / trailing-252d max -> values in (0, 1]; 1.0 means at 52w high.
            try:
                high_52w = float(close.rolling(252, min_periods=20).max().iloc[-1])
                pct_to_52w_high = round(current_price / high_52w, 4) if high_52w > 0 else None
            except Exception:
                pct_to_52w_high = None

            row = {
                "ticker": ticker,
                "current_price": round(current_price, 2),
                "avg_volume_20d": int(avg_vol),
                "momentum_1m": round(momentum_1m, 2) if momentum_1m else None,
                "momentum_3m": round(momentum_3m, 2) if momentum_3m else None,
                "momentum_6m": round(momentum_6m, 2) if momentum_6m else None,
                "rsi_14": round(rsi, 1) if rsi else None,
                "volatility_ann": round(volatility, 4) if volatility else None,
                "sma_50_distance_pct": round(sma_distance, 2),
                "pct_to_52w_high": pct_to_52w_high,  # phase-28.7
            }
            # phase-23.1.13: attach sector when caller provided the lookup.
            # The lookup is built via _fetch_ticker_meta (BQ-first/yfinance fallback)
            # so we pay zero extra cost in the screener inner loop.
            if sector_lookup:
                meta = sector_lookup.get(ticker)
                if isinstance(meta, dict):
                    row["sector"] = meta.get("sector") or ""
                    row["company_name"] = meta.get("company_name") or ticker
                elif isinstance(meta, str):
                    row["sector"] = meta
            results.append(row)
        except Exception as e:
            logger.debug(f"Skipping {ticker}: {e}")
            continue

    logger.info(f"Screening complete: {len(results)}/{len(tickers)} passed basic filters")
    return results


def rank_candidates(
    screen_data: list[dict],
    top_n: int = 10,
    strategy: str = "momentum",
    regime=None,
    pead_signals=None,
    news_signals=None,
    sector_events=None,
    revision_signals=None,
    sector_neutral: bool = False,
    sector_neutral_min_group_size: int = 3,
    soft_sector_diversity: bool = False,
    soft_sector_diversity_w: float = 0.0,
    sector_momentum_ranks=None,
    multidim_momentum: bool = False,
    multidim_weights: Optional[dict[str, float]] = None,
    momentum_52wh_tilt: bool = False,
    momentum_52wh_tilt_k: float = 0.5,
    pead_signals_lookup=None,
    options_surge_signals=None,
    insider_signals=None,
    narrative_signals=None,
    gpr_exposure_signals=None,
    gpr_exposure_config: Optional[dict] = None,
    social_velocity_signals=None,
    defense_signal=None,
    peer_leadlag_signals=None,
    ma_preannounce_signals=None,
) -> list[dict]:
    """
    Rank screened candidates by composite alpha score.

    Strategies:
    - "momentum": Favor strong recent momentum + reasonable RSI (not overbought)
    - "value_momentum": Blend momentum with mean-reversion (SMA distance)

    Returns top_n candidates sorted by composite score (descending).
    """
    if not screen_data:
        return []

    scored = []
    for stock in screen_data:
        mom_1m = stock.get("momentum_1m") or 0
        mom_3m = stock.get("momentum_3m") or 0
        mom_6m = stock.get("momentum_6m") or 0
        rsi = stock.get("rsi_14") or 50
        vol = stock.get("volatility_ann") or 0.3
        sma_dist = stock.get("sma_50_distance_pct") or 0

        if strategy == "momentum":
            # Composite: weight recent momentum more, penalize overbought/oversold
            score = (
                mom_1m * 0.40 +
                mom_3m * 0.35 +
                mom_6m * 0.25
            )
            # RSI penalty: reduce score if extremely overbought (>80) or oversold (<20)
            if rsi > 80:
                score *= 0.7
            elif rsi < 20:
                score *= 0.8
            # Volatility adjustment: slightly penalize very high vol
            if vol > 0.6:
                score *= 0.85
        elif strategy == "value_momentum":
            # Blend: strong 3M momentum but currently pulled back from SMA
            score = mom_3m * 0.5 - abs(sma_dist) * 0.2 + mom_1m * 0.3
        else:
            score = mom_3m

        if regime is not None:
            from backend.services.macro_regime import apply_regime_to_score
            score = apply_regime_to_score(
                score, stock.get("sector"), SECTOR_ETFS, regime,
            )

        if pead_signals:
            from backend.services.pead_signal import apply_pead_to_score
            new_score = apply_pead_to_score(score, stock.get("ticker"), pead_signals)
            if new_score is None:
                continue
            score = new_score

        if news_signals:
            from backend.services.news_screen import apply_news_to_score
            score = apply_news_to_score(score, stock.get("ticker"), news_signals)

        if sector_events:
            from backend.services.sector_calendars import apply_sector_events_to_score
            new_score = apply_sector_events_to_score(
                score, stock.get("ticker"), stock.get("sector"), sector_events,
            )
            if new_score is None:
                continue
            score = new_score

        # phase-28.1: analyst EPS revision-breadth overlay
        # Mill Street Research 19yr backtest: Sharpe~1.60 combined with momentum.
        # score *= (1 + breadth * weight) when |breadth| > threshold; no-op otherwise.
        if revision_signals:
            from backend.services.analyst_revisions import apply_revisions_to_score
            score = apply_revisions_to_score(score, stock.get("ticker"), revision_signals)

        # phase-28.12: sector-ETF momentum overlay (Quantpedia top-3 rotation).
        # Boost candidates in top-N momentum sectors. Identity when ranks dict is None
        # or sector missing/non-top.
        if sector_momentum_ranks:
            from backend.services.sector_momentum import apply_sector_momentum_to_score
            score = apply_sector_momentum_to_score(score, stock.get("sector"), sector_momentum_ranks)

        # phase-28.9: options-flow OI-surge overlay (Wayne State / J. Portfolio Mgmt).
        # Boost when near-expiry OTM call volume surge detected. Identity otherwise.
        if options_surge_signals:
            from backend.services.options_flow_screen import apply_options_surge_to_score
            score = apply_options_surge_to_score(score, stock.get("ticker"), options_surge_signals)

        # phase-28.10: opportunistic insider-buying overlay (Cohen-Malloy-Pomorski).
        # Boost when material opportunistic-buy aggregate detected; identity otherwise.
        if insider_signals:
            from backend.services.insider_signal_screen import apply_insider_signal_to_score
            score = apply_insider_signal_to_score(score, stock.get("ticker"), insider_signals)

        # phase-28.11: management-outlook narrative overlay (MVP proxy for canonical
        # analyst Strategic Outlook signal which needs paid data). Identity otherwise.
        if narrative_signals:
            from backend.services.analyst_narrative_scorer import apply_narrative_signal_to_score
            score = apply_narrative_signal_to_score(score, stock.get("ticker"), narrative_signals)

        # phase-28.13: firm-level GPR exposure DEFENSIVE FILTER (Fed 2025; no forward alpha).
        # Penalize HIGH-exposure firms unless their sector benefits from GPR.
        if gpr_exposure_signals:
            from backend.services.call_transcript_gpr import apply_gpr_exposure_to_score
            cfg = gpr_exposure_config or {}
            score = apply_gpr_exposure_to_score(
                score, stock.get("ticker"), stock.get("sector"), gpr_exposure_signals,
                exempt_sectors_csv=cfg.get("exempt_sectors_csv", "Industrials,Energy"),
                high_penalty=cfg.get("high_penalty", 0.97),
            )

        # phase-28.15: social media velocity overlay (Alpha Vantage cross-source).
        # Boost when velocity >= threshold + sufficient mentions; identity otherwise.
        if social_velocity_signals:
            from backend.services.social_velocity_screen import apply_social_velocity_to_score
            score = apply_social_velocity_to_score(score, stock.get("ticker"), social_velocity_signals)

        # phase-28.14: defense/war-stocks reference-case (GPR + XAR AND-gate).
        # Cycle-level signal; boosts defense-list tickers when triggered.
        if defense_signal:
            from backend.services.defense_signal import apply_defense_boost_to_score
            score = apply_defense_boost_to_score(score, stock.get("ticker"), defense_signal)

        # phase-28.17: peer-correlation laggard catch-up (intra-sector lead-lag).
        # Boost laggards in sectors with strong-momentum leaders + low analyst coverage.
        if peer_leadlag_signals:
            from backend.services.peer_leadlag_screen import apply_peer_leadlag_to_score
            score = apply_peer_leadlag_to_score(score, stock.get("ticker"), peer_leadlag_signals)

        # phase-28.16: M&A pre-announcement aggregator (options + insider + 13D-stub).
        # Boost when multiple legs of the public footprint converge.
        if ma_preannounce_signals:
            from backend.services.ma_preannounce_screen import apply_ma_preannounce_to_score
            score = apply_ma_preannounce_to_score(score, stock.get("ticker"), ma_preannounce_signals)

        scored.append({**stock, "composite_score": round(score, 3)})

    # phase-23.1.3: surface news-only candidates not already in screen_data.
    # These are tickers the news screen flagged that pure-momentum rejected.
    if news_signals:
        existing_tickers = {s.get("ticker") for s in scored}
        for ticker, sig in news_signals.items():
            if ticker in existing_tickers:
                continue
            if sig.confidence == "low" or sig.impact_polarity != "positive":
                continue
            scored.append({
                "ticker": ticker,
                "composite_score": round(5.0 * 1.10, 3),  # mid-tier baseline + positive-news boost
                "source": "news_only",
                "news_event_type": sig.event_type,
                "news_rationale": sig.rationale,
            })

    # phase-28.7: optional multidimensional momentum composite. Replaces composite_score
    # with a z-blended composite of 4 components: existing price-momentum composite,
    # 52w-high proximity, SUE (pead surprise_score), sector momentum boost. Preserves
    # original on composite_score_raw. Stocks missing a component get 0 (mean) for it.
    # Per CFA Institute Dec 2025 + George-Hwang 2004 + Novy-Marx 2014: superior
    # Sharpe + lower crash risk vs price-only momentum.
    if multidim_momentum and scored:
        _apply_multidim_momentum(
            scored,
            weights=multidim_weights or {"price": 0.35, "52w_high": 0.25, "sue": 0.20, "sector": 0.20},
            pead_signals=pead_signals if pead_signals else (pead_signals_lookup or None),
            sector_momentum_ranks=sector_momentum_ranks,
        )

    # phase-28.4: optional sector-neutral re-scoring (within-sector percentile rank).
    # Default OFF. When ON, replaces composite_score with within-sector percentile in
    # [0.0, 1.0]; original composite preserved on composite_score_raw. Groups with
    # fewer than sector_neutral_min_group_size members + missing-sector stocks fall
    # back to a global cross-sector percentile pool. Improves Sharpe + reduces sector
    # concentration per CFA Institute Dec 2025 framework.
    if sector_neutral and scored:
        from collections import defaultdict
        groups: dict[str, list[dict]] = defaultdict(list)
        for s in scored:
            key = (s.get("sector") or "").strip() or "_UNKNOWN_"
            groups[key].append(s)

        # Decide which groups get within-sector percentile vs global pool
        global_pool: list[dict] = []
        for key, members in list(groups.items()):
            if key == "_UNKNOWN_" or len(members) < sector_neutral_min_group_size:
                global_pool.extend(members)
                del groups[key]

        def _apply_pct_rank(members: list[dict]) -> None:
            raws = pd.Series([m.get("composite_score") or 0 for m in members])
            pcts = raws.rank(method="average", pct=True).tolist()
            for m, p in zip(members, pcts):
                m["composite_score_raw"] = m.get("composite_score")
                m["composite_score"] = round(float(p), 4)

        for members in groups.values():
            _apply_pct_rank(members)
        if global_pool:
            _apply_pct_rank(global_pool)

    # phase-52.2: 52-week-high momentum tilt (config-gated, DEFAULT OFF -> byte-identical).
    # A centered multiplicative tilt on composite_score (k=0.5 measured +0.05 ann Sharpe,
    # turnover-neutral, in phase-52.1). Faithful to scripts/ablation/sector_neutral_replay.py
    # ::hi52_tilt_basket so the LIVE ranking == the 52.1-measured ranking. Skipped when OFF.
    if momentum_52wh_tilt and scored:
        _apply_52wh_tilt(scored, momentum_52wh_tilt_k)

    # phase-70.2: SOFT profit-aware cross-sector diversity (S2). Runs LAST, after
    # every score overlay, just before the sort/truncation that structurally
    # discards cross-sector names. w=0 or flag OFF -> byte-identical.
    if soft_sector_diversity and soft_sector_diversity_w > 0 and scored:
        _apply_soft_sector_diversity(scored, soft_sector_diversity_w)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[:top_n]


def _apply_soft_sector_diversity(scored: list[dict], w: float) -> None:
    """phase-70.2: SOFT, profit-aware sector diversification (S2).

    Within each sector, rank candidates by descending RAW composite score; the
    j-th (0-based) same-sector name is shaded by mult=(1-w)^j -- so the sector
    LEADER (j=0) is untouched (across-sector momentum kept) and deeper same-sector
    names are progressively demoted, never zeroed (arXiv 2601.08717; Springer 2026
    prefers bounded multiplicative rank-decay over linear). This is NOT hard
    sector-neutralization (which the 2026-06-01 replay measured at -0.166 long-only
    Sharpe). w=0 -> mult=1 everywhere -> byte-identical.

    Shading uses the canonical SIGN-SAFE multiplier (overlay_math.sign_safe_mult,
    forced enabled) so a penalty always LOWERS rank even for a negative composite
    score (a raw base*mult would raise a negative score toward zero -- the sign
    inversion the phase-69.3 sign-safe work fixed). Sets composite_score_raw and
    mutates composite_score in place.
    """
    from collections import defaultdict
    from backend.services.overlay_math import sign_safe_mult

    order = sorted(
        range(len(scored)),
        key=lambda i: scored[i].get("composite_score") or 0.0,
        reverse=True,
    )
    seen: dict[str, int] = defaultdict(int)
    for i in order:
        s = scored[i]
        sec = (s.get("sector") or "").strip() or "Unknown"
        j = seen[sec]
        seen[sec] += 1
        base = s.get("composite_score") or 0.0
        s["composite_score_raw"] = s.get("composite_score")
        mult = (1.0 - w) ** j
        s["composite_score"] = round(sign_safe_mult(base, mult, enabled=True), 4)


def _zscore(values: list[float]) -> list[float]:
    """Cross-sectional z-score. Missing values (None/NaN) get 0 (mean). Std=0 -> all zeros."""
    cleaned = [(v if isinstance(v, (int, float)) and v == v else 0.0) for v in values]
    if not cleaned:
        return []
    mean = sum(cleaned) / len(cleaned)
    var = sum((v - mean) ** 2 for v in cleaned) / len(cleaned)
    std = var ** 0.5
    if std < 1e-9:
        return [0.0] * len(cleaned)
    return [(v - mean) / std for v in cleaned]


def _apply_52wh_tilt(scored: list[dict], k: float) -> None:
    """phase-52.2: in-place CENTERED multiplicative 52-week-high tilt on composite_score.
    Mirrors scripts/ablation/sector_neutral_replay.py::hi52_tilt_basket EXACTLY so the LIVE
    ranking == the 52.1-measured ranking. Tilts UP names nearer their 52w high, DOWN names
    far below it, centered on the universe mean so the average tilt ~= 1.0 (turnover-neutral
    on average). `pct_to_52w_high` is set on every screen_universe row (screener.py:228); a
    missing/None pct -> tilt 1.0 (no-op for that name). Preserves the pre-tilt score on
    composite_score_raw (which also witnesses that this pass ran)."""
    pcts = [s.get("pct_to_52w_high") for s in scored if s.get("pct_to_52w_high") is not None]
    if not pcts:
        return
    mean_pct = sum(pcts) / len(pcts)
    for s in scored:
        p = s.get("pct_to_52w_high")
        tilt = (1 + k * (p - mean_pct)) if p is not None else 1.0
        s["composite_score_raw"] = s.get("composite_score")
        s["composite_score"] = (s.get("composite_score") or 0.0) * tilt


def _apply_multidim_momentum(
    scored: list[dict],
    weights: dict[str, float],
    pead_signals: Optional[dict] = None,
    sector_momentum_ranks: Optional[dict] = None,
) -> None:
    """phase-28.7: Replace composite_score with z-blended 4-component multidim momentum.

    Components per stock:
        price:     existing composite_score (already includes mom_1m/3m/6m + RSI/vol penalties)
        52w_high:  pct_to_52w_high field (1.0 = at high)
        sue:       pead_signals[ticker].surprise_score if available, else 0
        sector:    sector_momentum_ranks[sector].boost_multiplier - 1.0 if available, else 0

    Each component is z-scored across the universe; final composite_score is the
    weighted sum. Original composite preserved on composite_score_raw. Mutates `scored`
    in place.
    """
    if not scored:
        return
    price_vals: list[float] = []
    high_vals: list[float] = []
    sue_vals: list[float] = []
    sector_vals: list[float] = []
    for s in scored:
        price_vals.append(s.get("composite_score") or 0.0)
        high_vals.append(s.get("pct_to_52w_high") or 0.0)
        if pead_signals:
            sig = pead_signals.get(s.get("ticker"))
            try:
                sue_vals.append(float(getattr(sig, "surprise_score", 0.0)))
            except Exception:
                sue_vals.append(0.0)
        else:
            sue_vals.append(0.0)
        if sector_momentum_ranks:
            sector_entry = sector_momentum_ranks.get(s.get("sector") or "")
            try:
                sector_vals.append(float(getattr(sector_entry, "boost_multiplier", 1.0)) - 1.0)
            except Exception:
                sector_vals.append(0.0)
        else:
            sector_vals.append(0.0)
    z_price = _zscore(price_vals)
    z_high = _zscore(high_vals)
    z_sue = _zscore(sue_vals)
    z_sector = _zscore(sector_vals)
    w_price = float(weights.get("price", 0.35))
    w_high = float(weights.get("52w_high", 0.25))
    w_sue = float(weights.get("sue", 0.20))
    w_sector = float(weights.get("sector", 0.20))
    for i, s in enumerate(scored):
        s["composite_score_raw"] = s.get("composite_score")
        s["composite_score"] = round(
            w_price * z_price[i]
            + w_high * z_high[i]
            + w_sue * z_sue[i]
            + w_sector * z_sector[i],
            4,
        )


def _pct_change(series: pd.Series, periods: int) -> Optional[float]:
    """Calculate percentage change over N periods."""
    if len(series) <= periods:
        return None
    old = float(series.iloc[-periods - 1])
    new = float(series.iloc[-1])
    return ((new - old) / old) * 100 if old != 0 else None


def _compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    """Compute RSI indicator."""
    if len(series) < period + 1:
        return None
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# Fallback tickers if Wikipedia scrape fails
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT",
    "CRM", "ACN", "CSCO", "ABT", "DHR", "ADBE", "NKE", "TXN", "NEE",
    "PM", "UNP", "RTX", "LOW", "HON", "ORCL", "BMY", "QCOM", "UPS",
    "INTC", "AMD", "SBUX", "BA",
]

# phase-28.8: Russell-1000 universe expansion (addresses Sandisk/SNDK reference-case
# miss where the picker's SP500-only universe excluded the spinoff during the early
# rally phase). Russell-1000 ~doubles the universe and includes recent spinoffs +
# mid-caps below SP500's $22.7B inclusion floor.
#
# Data source: iShares IWB ETF CSV holdings. Cached locally for 180 days (FTSE
# Russell does semi-annual reconstitution). Fallback to hardcoded extension list
# (SP500 fallback + 50 well-known mid-caps + reference-case adds like SNDK, WDC, MU).
IWB_HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239707/"
    "ishares-russell-1000-etf/1467271812596.ajax"
    "?fileType=csv&fileName=IWB_holdings&dataType=fund"
)
_RUSSELL_1000_CACHE = Path(__file__).parent.parent / "services" / "_cache" / "russell1000"

# Reference-case + popular mid-caps not in the SP500 fallback above. Used when
# IWB download fails AND Wikipedia scrape fails. Hand-curated for coverage of
# common spinoff / mid-cap names.
_RUSSELL_1000_EXTRA_FALLBACK = [
    "SNDK", "WDC", "MU", "STX", "LITE", "CIEN", "COHR", "AMAT", "LRCX", "KLAC",
    "FIX", "DELL", "GLW", "ON", "MCHP", "MPWR", "NXPI", "ADI", "MRVL", "TER",
    "ZS", "OKTA", "DDOG", "MDB", "NET", "TEAM", "PLTR", "SNOW", "WDAY", "NOW",
    "PANW", "FTNT", "CRWD", "DASH", "ABNB", "UBER", "LYFT", "PINS", "ROKU", "SPOT",
    "OXY", "EOG", "PSX", "MPC", "VLO", "DVN", "PXD", "BKR", "SLB", "HAL",
    "C", "WFC", "USB", "PNC", "TFC", "COF", "SCHW", "BLK", "MS", "GS",
]


def _read_russell_cache() -> Optional[list[str]]:
    """Return cached Russell-1000 ticker list if fresh; else None."""
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    p = _RUSSELL_1000_CACHE / "tickers.txt"
    if not p.exists():
        return None
    try:
        age = _dt.now(_tz.utc) - _dt.fromtimestamp(p.stat().st_mtime, tz=_tz.utc)
        if age > _td(days=180):
            return None
        tickers = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return tickers if len(tickers) >= 500 else None
    except Exception:
        return None


def _write_russell_cache(tickers: list[str]) -> None:
    try:
        _RUSSELL_1000_CACHE.mkdir(parents=True, exist_ok=True)
        (_RUSSELL_1000_CACHE / "tickers.txt").write_text("\n".join(tickers), encoding="utf-8")
    except Exception as e:
        logger.warning("Russell-1000 cache write failed: %s", e)


def get_russell1000_tickers() -> list[str]:
    """Fetch Russell-1000 ticker list via iShares IWB CSV (preferred) with cache.

    Cache TTL: 180 days (matches FTSE Russell semi-annual reconstitution).

    Fallback chain:
        1. Local 180-day cache
        2. iShares IWB CSV download (browser User-Agent)
        3. SP500 list + _RUSSELL_1000_EXTRA_FALLBACK (de-duplicated)
    """
    cached = _read_russell_cache()
    if cached:
        logger.info("Russell-1000 cache hit: %d tickers", len(cached))
        return cached

    import io as _io
    import urllib.request as _urlreq
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.5",
    }
    try:
        req = _urlreq.Request(IWB_HOLDINGS_URL, headers=headers)
        with _urlreq.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        text = "\n".join(line for line in raw.splitlines() if "," in line)
        df = pd.read_csv(_io.StringIO(text), skiprows=9, engine="python", on_bad_lines="skip")
        if "Ticker" in df.columns and "Asset Class" in df.columns:
            df = df[df["Asset Class"].astype(str).str.strip() == "Equity"]
            tickers = (
                df["Ticker"].astype(str).str.strip().str.upper()
                  .str.replace(".", "-", regex=False).dropna().unique().tolist()
            )
            tickers = [t for t in tickers if t and t.replace("-", "").isalnum()]
            if len(tickers) >= 500:
                _write_russell_cache(tickers)
                logger.info("Russell-1000 IWB download succeeded: %d tickers", len(tickers))
                return tickers
        logger.warning("IWB CSV parse: unexpected schema; falling back")
    except Exception as e:
        logger.warning("IWB CSV download failed (%s); falling back to combined SP500+extras", e)

    sp500 = get_sp500_tickers()
    combined = list(dict.fromkeys(sp500 + _RUSSELL_1000_EXTRA_FALLBACK))
    logger.info("Russell-1000 fallback list: %d tickers (SP500 %d + extras %d, deduped)",
                len(combined), len(sp500), len(_RUSSELL_1000_EXTRA_FALLBACK))
    return combined
