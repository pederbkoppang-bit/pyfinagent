"""
Backtest engine — main orchestrator for walk-forward ML backtesting.
Trains GradientBoosting with Triple Barrier labels, sample weights,
fractional differentiation, and meta-labeling position sizing.
Zero LLM cost.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from backend.backtest.historical_data import HistoricalDataProvider
from backend.backtest.candidate_selector import CandidateSelector
from backend.backtest.walk_forward import WalkForwardScheduler, WalkForwardWindow
from backend.backtest.backtest_trader import BacktestTrader
from backend.backtest import cache
from backend.backtest.historical_data import _MACRO_SERIES

logger = logging.getLogger(__name__)

# ── Strategy Registry ────────────────────────────────────────────
# Maps strategy name → label computation method name on BacktestEngine.
# QuantOptimizer rotates among these as a categorical parameter.
# phase-82.16: `quality_momentum` and `factor_model` were DEMOTED out of the
# selection pool. Both classify purely on features AT entry_date and never fetch a
# post-entry price, so the GradientBoostingClassifier is trained to reproduce a
# deterministic threshold rule over its OWN inputs (`momentum_6m` and `quality_score`
# are both in _NUMERIC_FEATURES). Their in-sample accuracy is near-tautological and
# any Sharpe/DSR computed from them is not evidence about the future.
#
# MEASURED on the committed 82.2 fixture (880 rows): quality_momentum produced 880
# labels and ZERO changed when the entire post-entry price path was destroyed;
# factor_model produced 0 labels at all. This is CIRCULAR ANALYSIS (Kriegeskorte et
# al. 2009), not look-ahead bias -- note that a shuffled-label permutation test, the
# reflex check, would PASS both of them.
#
# DEMOTED rather than rewritten, deliberately: giving them a forward-looking label
# would be a NEW strategy under an old name, silently changing what every historical
# `quality_momentum` row in quant_results.tsv means. And quality_momentum would stay
# untrainable regardless -- step 82.21 measured zero `historical_fundamentals` rows
# before 2024-06-30, ~81% of the standard window. Bailey & Lopez de Prado require a
# non-comparable candidate to be REMOVED from the trial pool, not caveated.
#
# The methods are KEPT so the demotion is reversible and nothing referencing them by
# name breaks.
NON_COMPARABLE_STRATEGIES: dict[str, str] = {
    "quality_momentum": (
        "no forward information: classifies on momentum_6m + quality_score at "
        "entry_date, never reads a post-entry price (measured 880 labelled / 0 "
        "changed under post-entry mutation). Also fundamentals-dependent, and "
        "historical_fundamentals has no rows before 2024-06-30 (step 82.21)."
    ),
    "factor_model": (
        "no forward information: same shape as quality_momentum, no forward window "
        "(measured 0 labelled / 0 changed -- the zero-change result is VACUOUS "
        "because it returns None everywhere on that fixture)."
    ),
}

STRATEGY_REGISTRY: dict[str, str] = {
    "triple_barrier": "_compute_triple_barrier_label",
    "mean_reversion": "_compute_mean_reversion_label",
    "meta_label": "_compute_triple_barrier_label",  # uses TB labels; meta-labeling applied in _run_window
    # ── phase-82.2 candidates (overpriced-market lenses) ──────────
    # All three are FORWARD-LOOKING, cost-adjusted and sigma-scaled.
    # Forward-looking is not optional: `quality_momentum` and `factor_model`
    # classify on features AT entry_date and never read a forward price, so a
    # model trained on them learns a deterministic function of its own inputs
    # and their Sharpe/DSR say nothing about the future (see step 82.16).
    "stretch_regime": "_compute_stretch_regime_label",
    "qarp": "_compute_qarp_label",
    "reversion_sigma": "_compute_reversion_sigma_label",
}

def resolve_strategy(name: str) -> tuple[str, bool]:
    """Resolve a requested strategy to the one that will actually RUN.

    Returns `(effective_name, was_demoted)`. Extracted as a module-level function so
    the demotion path is drivable without constructing a BacktestEngine (whose
    __init__ reaches for BigQuery) -- it is the engine's real decision, not a copy.

    phase-82.16: the engine has always coerced an unknown strategy to triple_barrier.
    That was harmless while every name was registered; demoting quality_momentum and
    factor_model makes it REACHABLE, and a run reported as quality_momentum that
    actually ran triple_barrier is worse than the tautological label it replaced --
    the provenance is wrong too. A caller really can ask:
    meta_evolution/archetype_library.py still lists both names in its own
    IMPLEMENTED_STRATEGY_IDS.
    """
    was_demoted = name in NON_COMPARABLE_STRATEGIES
    if not was_demoted and name not in STRATEGY_REGISTRY:
        # phase-82.16 cycle 2: the demoted names are not the only way to reach the
        # silent coercion. ANY unregistered name -- including "blend", which the
        # optimizer still offers and which has no implementation here -- was
        # coerced to triple_barrier with no signal at all, and then SCORED under
        # the requested name. Warn on the whole class, not just the two names this
        # step demoted.
        logger.warning(
            "backtest: strategy %r is not in STRATEGY_REGISTRY -- falling back to "
            "triple_barrier. Results are triple_barrier's, NOT %r's; do not record "
            "them under the requested name.",
            name, name,
        )
    if was_demoted:
        logger.warning(
            "backtest: strategy %r is NON-COMPARABLE and was demoted from "
            "STRATEGY_REGISTRY -- falling back to triple_barrier. Reason: %s. "
            "Results are triple_barrier's, NOT %r's; do not record them under the "
            "requested name.",
            name, NON_COMPARABLE_STRATEGIES[name], name,
        )
    return (name if name in STRATEGY_REGISTRY else "triple_barrier"), was_demoted


# Numeric features used for ML training (excludes categorical: ticker, date, sector, industry)
_NUMERIC_FEATURES = [
    "price_at_analysis", "momentum_1m", "momentum_3m", "momentum_6m", "momentum_12m",
    "rsi_14", "annualized_volatility", "sma_50_distance", "sma_200_distance",
    "var_95_6m", "var_99_6m", "expected_shortfall_6m", "prob_positive_6m",
    "anomaly_count", "amihud_illiquidity", "volume_ratio_20d",
    "pe_ratio", "pb_ratio", "debt_equity", "roe", "profit_margin", "market_cap",
    "total_revenue", "net_income", "total_debt", "total_equity", "total_assets",
    "fcf_yield", "dividend_yield", "quality_score", "revenue_growth_yoy",
    "fed_funds_rate", "cpi_yoy", "unemployment_rate", "yield_curve_spread",
    "consumer_sentiment", "treasury_10y",
]

# Non-stationary features that need fractional differentiation
def compute_matrix_coverage(df: "pd.DataFrame", X: "pd.DataFrame") -> dict:
    """phase-82.43: what the training matrix ACTUALLY received, per window.

    Extracted as a module-level function on purpose. The first version of this
    lived inline in `_build_training_data`, and the only guard that could reach
    it was a source scan for token names -- which a mutation run defeated twice:
    replacing `int(X.shape[1])` with a literal survived (the scan checked the
    key, not the value), and deleting the `macro_series_min` entry survived
    because that same string still appeared in a logger call below. A guard that
    reads source text cannot see either. This function is EXECUTABLE, so the
    guard asserts values instead of spelling.

    `macro_series_count` is a per-row 0-6 integer from
    `historical_data.build_feature_vector`. The MINIMUM and the degraded-row
    count are what distinguish a partially-covered window from a clean one; a
    boolean or a mean would not -- on the real biweekly grid
    2018-01-01..2019-12-31 (n=53) there were 1 empty, 3 partial and 49 full
    cutoffs, and the partial rows are silently median-imputed downstream.
    """
    counts = (
        [int(c) for c in df["macro_series_count"].fillna(0).tolist()]
        if "macro_series_count" in df.columns else []
    )
    return {
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "macro_series_min": min(counts) if counts else None,
        "macro_series_max": max(counts) if counts else None,
        "macro_rows_degraded": sum(1 for c in counts if c < len(_MACRO_SERIES)),
    }


_NON_STATIONARY = {"price_at_analysis", "market_cap", "total_revenue", "total_debt", "total_equity"}

# ── MDA Cache ────────────────────────────────────────────────────
# Persists the latest aggregate MDA weights so that the live quant_model
# tool can read them without re-running a full backtest.
_MDA_CACHE_PATH = Path(__file__).parent / "experiments" / "mda_cache.json"
_latest_mda: dict[str, float] = {}


def get_latest_mda() -> dict[str, float]:
    """Return cached MDA weights. Reads JSON file if module-level cache is empty."""
    global _latest_mda
    if _latest_mda:
        return dict(_latest_mda)
    if _MDA_CACHE_PATH.exists():
        try:
            data = json.loads(_MDA_CACHE_PATH.read_text(encoding="utf-8"))
            _latest_mda = {k: float(v) for k, v in data.items()}
            return dict(_latest_mda)
        except Exception:
            pass
    return {}


def _save_mda_cache(mda: dict[str, float]) -> None:
    """Persist MDA weights to JSON and module-level cache."""
    global _latest_mda
    _latest_mda = dict(mda)
    try:
        _MDA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MDA_CACHE_PATH.write_text(json.dumps(mda, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save MDA cache: %s", e)


@dataclass
class WindowResult:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    sharpe_ratio: float
    total_return_pct: float
    alpha_pct: float
    max_drawdown_pct: float
    hit_rate: float
    num_trades: int
    n_candidates: int = 0
    n_train_samples: int = 0
    n_features: int = 0
    feature_importance_mdi: dict[str, float] = field(default_factory=dict)
    feature_importance_mda: dict[str, float] = field(default_factory=dict)
    predictions: list[dict] = field(default_factory=list)


@dataclass
class BacktestResult:
    windows: list[WindowResult] = field(default_factory=list)
    aggregate_sharpe: float = 0.0
    aggregate_return_pct: float = 0.0
    aggregate_alpha_pct: float = 0.0
    aggregate_max_drawdown_pct: float = 0.0
    aggregate_hit_rate: float = 0.0
    total_trades: int = 0
    feature_importance_mdi: dict[str, float] = field(default_factory=dict)
    feature_importance_mda: dict[str, float] = field(default_factory=dict)
    nav_history: list[dict] = field(default_factory=list)
    strategy_params: dict = field(default_factory=dict)
    all_trades: list[dict] = field(default_factory=list)
    # phase-82.13: a run whose macro features were refused must not be readable as a
    # normal one. Defaulted, so every existing construction site is unaffected.
    # phase-82.21 extends the same record with "fundamentals" -- `historical_fundamentals`
    # has no row before 2024-06-30, so a run over an earlier window trains on a
    # silently narrower feature set. Same defaulting technique, same reason.
    data_availability: dict = field(
        default_factory=lambda: {"macro": True, "fundamentals": True}
    )


class BacktestEngine:
    """
    Main walk-forward backtesting orchestrator.
    Trains GradientBoosting on Triple Barrier labels with:
    - Sample weights (average uniqueness for overlapping labels)
    - Fractional differentiation on non-stationary features
    - Meta-labeling probability for position sizing
    - MDI + MDA feature importance
    """

    def __init__(
        self,
        bq_client,
        project: str,
        dataset: str,
        # Market (Phase 2.9)
        market: str = "US",
        # Walk-forward params
        start_date: str = "2023-01-01",
        end_date: str = "2025-12-31",
        train_window_months: int = 12,
        test_window_months: int = 3,
        embargo_days: int = 5,
        # Triple Barrier params
        holding_days: int = 90,
        tp_pct: float = 10.0,
        sl_pct: float = 10.0,
        # Mean Reversion params
        mr_holding_days: int = 15,
        # Feature params
        frac_diff_d: float = 0.4,
        # Strategy selection
        strategy: str = "triple_barrier",
        # Portfolio params
        starting_capital: float = 100_000.0,
        max_positions: int = 20,
        transaction_cost_pct: float = 0.1,
        target_vol: float = 0.15,
        top_n_candidates: int = 50,
        commission_model: str = "flat_pct",
        commission_per_share: float = 0.005,
        # ML params
        n_estimators: int = 200,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        learning_rate: float = 0.1,
        # Progress callback
        progress_callback=None,
    ):
        # Defensive unwrap: accept BigQueryClient wrapper or raw bigquery.Client
        if hasattr(bq_client, 'client'):
            bq_client = bq_client.client

        # Initialize cache
        cache.init_cache(bq_client, project, dataset)

        self.data_provider = HistoricalDataProvider()
        self.candidate_selector = CandidateSelector()

        self.scheduler = WalkForwardScheduler(
            start_date=start_date,
            end_date=end_date,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            embargo_days=embargo_days,
        )

        self.holding_days = holding_days
        self.mr_holding_days = mr_holding_days
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.frac_diff_d = frac_diff_d
        self.top_n_candidates = top_n_candidates
        # phase-82.16: an unknown strategy silently becomes triple_barrier here. That
        # was harmless while every name was registered; the 82.16 demotion makes it
        # reachable, and a run REPORTED as quality_momentum that actually ran
        # triple_barrier is worse than the tautological label it replaced -- the
        # provenance is now wrong too. Requesting a demoted strategy is still
        # non-fatal (a backtest tool should not crash on a stale config) but it can
        # no longer be silent. NOTE meta_evolution/archetype_library.py still lists
        # both names in its own IMPLEMENTED_STRATEGY_IDS, so a caller really can ask.
        self.requested_strategy = strategy
        self.strategy, self.strategy_was_demoted = resolve_strategy(strategy)
        self.model_trained_at: str = ""  # ISO timestamp of last model training

        # Store BQ refs for auto-ingest check
        self._bq_client = bq_client
        self._project = project
        self._dataset = dataset
        self.market = market  # Phase 2.9: market code (default 'US')

        self.ml_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
        }

        self.trader = BacktestTrader(
            starting_capital=starting_capital,
            max_positions=max_positions,
            transaction_cost_pct=transaction_cost_pct,
            target_vol=target_vol,
            commission_model=commission_model,
            commission_per_share=commission_per_share,
        )

        self.progress_callback = progress_callback
        self.stop_check: Callable[[], bool] | None = None

        # Ablation mask: a set of feature names to zero out in both the
        # training and prediction feature matrices. Empty set (default) is
        # a no-op. Used by scripts/ablation/run_ablation.py to measure the
        # Sharpe delta from removing a feature (leave-one-out). Applied in
        # _build_training_data and _run_window at the point where X is
        # materialized so the model never sees the masked values.
        self._ablation_mask: set[str] = set()
        self._backtest_start_time: float = 0.0
        self._total_windows: int = 0
        self._current_window_id: int = 0
        self._strategy_params = {
            "market": market,  # Phase 2.9
            "start_date": start_date, "end_date": end_date,
            "train_window_months": train_window_months,
            "test_window_months": test_window_months,
            "embargo_days": embargo_days, "holding_days": holding_days,
            "mr_holding_days": mr_holding_days,
            "tp_pct": tp_pct, "sl_pct": sl_pct, "frac_diff_d": frac_diff_d,
            "starting_capital": starting_capital, "max_positions": max_positions,
            "top_n_candidates": top_n_candidates, "strategy": self.strategy,
            **self.ml_params,
        }

    def set_cached_features(self, cache_dict: dict) -> None:
        """Activate feature caching for optimizer runs. Empty dict = populate on first run."""
        self._cached_features = cache_dict

    def get_cached_features(self) -> dict | None:
        """Return cached features if active, else None."""
        return getattr(self, '_cached_features', None)

    def clear_feature_cache(self) -> None:
        """Clear the feature cache between optimizer iterations."""
        self._cached_features = None

    def get_model_trained_at(self) -> str:
        """Return ISO timestamp of last model training. Empty string if never trained."""
        return self.model_trained_at

    def _preload_macro_and_record(self) -> dict:
        """Load macro, act on a REFUSAL, and return the availability record.

        phase-82.13. `preload_macro`'s int return is a DIAGNOSTIC, not a decision:
        0 means BOTH "empty table" and "I refused", and the already-warm path
        returns a positive total having loaded nothing. So this branches on the
        status accessor, never on the int.

        Extracted as its own method so the refusal path is drivable by a test. It
        is the engine's real code path, not a test-only copy -- a guard that
        re-implements the logic it checks proves nothing.
        """
        macro_rows = cache.preload_macro()
        status = cache.macro_load_status()

        if cache.macro_was_refused():
            # Do NOT proceed silently. Before 82.0 armed the staleness gate this
            # branch was unreachable, so the discarded return never mattered. Now a
            # refusal would otherwise leave _macro_full empty and send cached_macro
            # into one BQ round-trip per distinct cutoff date.
            cache.set_macro_unavailable(True)
            logger.warning(
                "backtest: running MACRO-FREE -- preload_macro refused (%s: %s). "
                "The per-cutoff fallback is suppressed, so this run is fast but "
                "macro-blind, and its result is labelled data_availability."
                "macro=False. Do not compare it against a macro-complete run.",
                status.get("outcome"), status.get("detail"),
            )
            try:
                self._report_progress(
                    "preloading", "macro REFUSED -- running macro-free",
                    macro_available=False, macro_outcome=status.get("outcome"),
                )
            except Exception:  # observability must never break a run
                pass
            available = False
        else:
            cache.set_macro_unavailable(False)
            available = cache.macro_is_loaded()

        return {
            "macro": available,
            "macro_outcome": status.get("outcome"),
            "macro_detail": status.get("detail"),
            "macro_rows": macro_rows,
        }

    def _preload_fundamentals_and_record(self) -> dict:
        """phase-82.21. Decide whether this run's window is inside fundamentals
        coverage, REFUSE if the active strategy cannot be labelled without them,
        and return the availability record.

        Deliberately mirrors `_preload_macro_and_record` above, for the same
        reason: extracted as its own method so the refusal path is drivable by a
        test against the engine's REAL code path, not a test-only copy.

        The refusal set is DERIVED (`label_fundamentals_dependent_strategies`
        applies a written-down AST rule), never a hardcoded `{"qarp"}` -- a
        literal would go stale the next time a strategy is added, and a stale
        gate is worse than none because it reads as coverage.

        Two outcomes, and the difference matters:
          * REFUSE -- raise, for a strategy whose LABEL function cannot be
            computed at all without fundamentals. Producing zero labels and
            calling it a backtest is the failure this prevents.
          * RECORD -- for everyone else. Every strategy is FEATURE-dependent via
            `_NUMERIC_FEATURES`, so an uncovered window silently drops 15 of 37
            columns (:852) or imputes a fabricated median company (:881-882).
            That is not fatal, but it must not read as a normal run.
        """
        from backend.backtest.fundamentals_coverage import (
            FUNDAMENTALS_COVERAGE_START,
            label_fundamentals_dependent_strategies,
            window_is_covered,
        )

        window_start = self.scheduler.start_date.isoformat()
        covered = window_is_covered(window_start)
        dependent = label_fundamentals_dependent_strategies()

        if not covered and self.strategy in dependent:
            raise ValueError(
                f"backtest REFUSED: strategy '{self.strategy}' cannot be labelled "
                f"without fundamentals, and the requested window starts "
                f"{window_start} which is before the measured coverage start "
                f"{FUNDAMENTALS_COVERAGE_START}. historical_fundamentals has no row "
                f"before that date (measured; see "
                f"backend/backtest/_fundamentals_coverage.json). Running would "
                f"produce labels from absent inputs, not a backtest. Re-run with a "
                f"window starting on or after {FUNDAMENTALS_COVERAGE_START}, or "
                f"choose a strategy that does not read fundamentals."
            )

        if not covered:
            logger.warning(
                "backtest: running FUNDAMENTALS-INCOMPLETE -- window starts %s, "
                "coverage starts %s. The strategy '%s' does not need fundamentals "
                "to label, but the shared feature matrix does: uncovered rows are "
                "median-imputed. This result is labelled data_availability."
                "fundamentals=False. Do not compare it against a covered run.",
                window_start, FUNDAMENTALS_COVERAGE_START, self.strategy,
            )

        return {
            "fundamentals": covered,
            "fundamentals_coverage_start": FUNDAMENTALS_COVERAGE_START,
            "fundamentals_window_start": window_start,
            "fundamentals_label_dependent": sorted(dependent),
        }

    def run_backtest(
        self,
        universe_tickers: list[str] | None = None,
        skip_cache_clear: bool = False,
    ) -> BacktestResult:
        """
        Run full walk-forward backtest. Main entry point.
        Auto-checks BQ data availability and triggers ingestion if empty.
        Returns BacktestResult with per-window and aggregate metrics.

        Args:
            skip_cache_clear: If True, skip cache.clear_cache() at end.
                Used by QuantStrategyOptimizer to keep warm cache across iterations.
        """
        if universe_tickers is None:
            # phase-50.5: market-aware universe (US -> S&P 500; EU/KR -> curated
            # INTL_UNIVERSE). Was market-blind despite storing self.market.
            universe_tickers = self.candidate_selector.get_universe_tickers(market=self.market)

        # Auto-ingest check: if historical_prices is empty, trigger ingestion
        self._auto_ingest_if_needed(universe_tickers)

        # Reset trader for a clean independent run (critical for optimizer —
        # each trial must be independent per Bailey & López de Prado, 2014)
        self.trader.full_reset()

        windows = self.scheduler.generate_windows()
        self._total_windows = len(windows)
        self._backtest_start_time = time.time()
        logger.info(f"Walk-forward: {len(windows)} windows, {len(universe_tickers)} tickers, strategy={self.strategy}")

        # Bulk-preload all price and fundamental data (2 BQ queries instead of ~50,000)
        self._report_progress("preloading", f"Loading data for {len(universe_tickers)} tickers")
        global_start = (self.scheduler.start_date - timedelta(days=756)).isoformat()
        global_end = (self.scheduler.end_date + timedelta(days=int(self.holding_days * 1.5))).isoformat()
        # phase-50.5: preload the market's benchmark (US -> SPY byte-identical;
        # EU -> ^GDAXI, KR -> ^KS11) so analytics.compute_baseline_strategies
        # can read it from cache.
        from backend.backtest.markets import get_market_config
        _benchmark = get_market_config(self.market).get("benchmark", "SPY")
        cache.preload_prices(universe_tickers + [_benchmark], global_start, global_end)
        cache.preload_fundamentals(universe_tickers)
        _availability = self._preload_macro_and_record()
        _macro_available = _availability["macro"]
        # phase-82.21: same shape, same record. This RAISES for a
        # label-fundamentals-dependent strategy on an uncovered window -- before
        # any window runs, so the refusal is cheap and unambiguous.
        _availability.update(self._preload_fundamentals_and_record())
        # phase-82.43: seeded here so the key always exists; refreshed after the
        # windows run, below, with what the matrix actually received.
        _availability["macro_coverage"] = {}

        result = BacktestResult(
            strategy_params=self._strategy_params,
            # phase-82.13 (criterion 2): label the run at construction, so a
            # macro-free run cannot be mistaken for a normal one when its numbers
            # are read later -- including on an early return.
            data_availability=dict(_availability),
        )
        all_mdi = {}
        all_mda = {}

        for window in windows:
            # Check stop signal between windows
            if self.stop_check and self.stop_check():
                logger.info("BacktestEngine: stopped at window %d/%d", window.window_id, len(windows))
                break

            self._report_progress(
                "screening",
                f"Window {window.window_id}/{len(windows)} — screening candidates",
                window=window.window_id,
            )
            try:
                wr = self._run_window(window, universe_tickers)
                result.windows.append(wr)
                result.total_trades += wr.num_trades

                # Accumulate feature importance
                for feat, imp in wr.feature_importance_mdi.items():
                    all_mdi[feat] = all_mdi.get(feat, 0) + imp
                for feat, imp in wr.feature_importance_mda.items():
                    all_mda[feat] = all_mda.get(feat, 0) + imp

            except Exception as e:
                logger.error(f"Window {window.window_id} failed: {e}")

        # Aggregate metrics from trader
        returns = self.trader.get_returns_series()
        if returns:
            returns_arr = np.array(returns)
            result.aggregate_sharpe = self._sharpe(returns_arr)
            result.aggregate_return_pct = float((np.prod(1 + returns_arr) - 1) * 100)
            result.aggregate_max_drawdown_pct = self._max_drawdown(returns_arr)

        # Aggregate feature importance (average across windows)
        n_windows = len(result.windows) or 1
        result.feature_importance_mdi = {k: v / n_windows for k, v in all_mdi.items()}
        result.feature_importance_mda = {k: v / n_windows for k, v in all_mda.items()}

        # Persist MDA cache for live quant_model tool
        if result.feature_importance_mda:
            _save_mda_cache(result.feature_importance_mda)

        # Aggregate hit rate
        all_preds = [p for w in result.windows for p in w.predictions]
        if all_preds:
            correct = sum(1 for p in all_preds if p.get("correct", False))
            result.aggregate_hit_rate = correct / len(all_preds)

        # NAV history from snapshots
        result.nav_history = [
            {"date": s.date, "nav": s.nav, "cash": s.cash}
            for s in self.trader.snapshots
        ]

        # Extract individual trades (capped at 500 for JSON size)
        result.all_trades = [
            {
                "ticker": t.ticker, "action": t.action, "quantity": round(t.quantity, 4),
                "price": round(t.price, 2), "date": t.date, "label": t.label,
                "probability": round(t.probability, 4), "commission": round(t.commission, 2),
            }
            for t in self.trader.trades[:500]
        ]
        # phase-82.43: carry the LAST window's matrix coverage onto the result,
        # so a run whose model trained on a narrower or median-imputed feature
        # set cannot be read as a normal one. Mirrors how 82.13/82.21 label the
        # run at the same object. Fail-open: a reporting problem must never take
        # down a completed backtest.
        try:
            result.data_availability["macro_coverage"] = dict(
                self._last_matrix_coverage
            )
            _mc = self._last_matrix_coverage
            if _mc.get("macro_rows_degraded"):
                logger.warning(
                    "backtest: %d of %d training rows had incomplete macro "
                    "coverage (min %s of %d series); those cells were "
                    "median-imputed. Recorded as data_availability."
                    "macro_coverage -- do not compare against a macro-complete run.",
                    _mc.get("macro_rows_degraded"), _mc.get("n_samples"),
                    _mc.get("macro_series_min"), len(_MACRO_SERIES),
                )
        except Exception as exc:  # noqa: BLE001 -- observability is never fatal
            logger.warning("backtest: macro_coverage record fail-open: %r", exc)

        logger.info(f"Backtest complete: {len(self.trader.trades)} actual trades recorded, {result.total_trades} signals processed")

        self._report_progress(
            "finalizing",
            f"Aggregating results across {len(result.windows)} windows...",
            window=self._total_windows,
        )
        if not skip_cache_clear:
            cache.clear_cache()
        return result

    def _run_window(self, window: WalkForwardWindow, universe_tickers: list[str]) -> WindowResult:
        """Process a single walk-forward window."""
        wid = window.window_id
        self._current_window_id = wid
        train_end_str = window.train_end.isoformat()
        test_start_str = window.test_start.isoformat()
        test_end_str = window.test_end.isoformat()

        # 1. Screen candidates at train_end
        scoring_weights = {
            k: self._strategy_params[k]
            for k in ("momentum_weight", "rsi_weight", "volatility_weight", "sma_weight")
            if k in self._strategy_params
        }
        candidates = self.candidate_selector.screen_at_date(
            train_end_str, universe_tickers, top_n=self.top_n_candidates,
            scoring_weights=scoring_weights if scoring_weights else None,
        )
        candidate_tickers = [c["ticker"] for c in candidates]

        self._report_progress(
            "screening",
            f"{len(candidate_tickers)} candidates found",
            window=wid, candidates_found=len(candidate_tickers),
        )

        if len(candidate_tickers) < 10:
            logger.warning(f"Window {wid}: only {len(candidate_tickers)} candidates")
            return WindowResult(
                window_id=wid,
                train_start=window.train_start.isoformat(),
                train_end=train_end_str,
                test_start=test_start_str,
                test_end=test_end_str,
                sharpe_ratio=0, total_return_pct=0, alpha_pct=0,
                max_drawdown_pct=0, hit_rate=0, num_trades=0,
            )

        # 2. Build training data
        self._report_progress("building_features", "Building training data", window=wid)
        train_features, train_labels, sample_weights = self._build_training_data(
            candidate_tickers, window.train_start.isoformat(), train_end_str,
            test_start=test_start_str, test_end=test_end_str,  # phase-69.2 purge
        )

        if len(train_features) < 20:
            logger.warning(f"Window {wid}: insufficient training data ({len(train_features)} samples)")
            return self._empty_window_result(window)

        # 3. Train ML model
        self._report_progress(
            "training",
            f"Training GradientBoosting on {len(train_features)} samples",
            window=wid, samples_built=len(train_features),
        )
        model, feature_names = self._train_model(train_features, train_labels, sample_weights)

        # 4. MDI feature importance
        mdi = dict(zip(feature_names, model.feature_importances_))

        # 5. MDA feature importance (permutation importance)
        self._report_progress("computing_mda", "Permutation importance", window=wid)
        mda = self._compute_mda(model, train_features, train_labels, feature_names)

        # 5b. Meta-labeling second stage (AFML Ch. 3.6)
        # If strategy is "meta_label", train a secondary model that predicts
        # whether the primary model's predictions are correct. The secondary
        # model's probability output is used for position sizing (fractional Kelly).
        meta_model = None
        if self.strategy == "meta_label" and len(train_features) >= 50:
            self._report_progress("training", "Training meta-label model (AFML Ch. 3.6)", window=wid)
            meta_model = self._train_meta_label_model(
                model, train_features, train_labels, feature_names, sample_weights,
            )

        # 6. Predict on test period candidates
        self._report_progress("predicting", f"Scoring test candidates at {test_start_str}", window=wid)
        test_candidates = self.candidate_selector.screen_at_date(
            test_start_str, universe_tickers, top_n=self.top_n_candidates,
            scoring_weights=scoring_weights if scoring_weights else None,
        )
        test_tickers = [c["ticker"] for c in test_candidates]

        signals, predictions = self._predict_and_trade(
            model, feature_names, test_tickers, test_start_str, test_end_str,
            meta_model=meta_model,
        )

        # 7. Execute trades in the test window
        # Track actual trades executed (not just signal count)
        trades_before = len(self.trader.trades)

        self._report_progress(
            "trading",
            f"Processing {len(signals) if signals else 0} signals",
            window=wid,
        )
        if signals:
            # Get prices at test_start for trading
            prices = {}
            for ticker in test_tickers:
                px = self._price_asof(ticker, test_start_str)  # phase-69.2 boundary snap
                if px is not None:
                    prices[ticker] = px

            executed = self.trader.execute_trades(signals, test_start_str, prices)
            logger.info(f"Window {wid}: {len(executed)} trades from {len(signals)} signals (BUY={sum(1 for s in signals if s['label']==1)}, SELL={sum(1 for s in signals if s['label']==-1)}, HOLD={sum(1 for s in signals if s['label']==0)})")

        # 8. Daily mark-to-market through the test window
        # Per Lo (2002): Sharpe annualization requires matching return frequency.
        # We need daily NAV snapshots so √252 annualization is correct and DSR
        # has sufficient T (Bailey & López de Prado, 2014).
        active_tickers = list(set(list(self.trader.positions.keys()) + test_tickers))
        test_days = pd.bdate_range(test_start_str, test_end_str)
        for day in test_days:
            day_str = day.strftime("%Y-%m-%d")
            day_prices = {}
            for ticker in active_tickers:
                p = cache.cached_prices(ticker, day_str, day_str)
                if not p.empty:
                    day_prices[ticker] = float(p["close"].iloc[-1])
            if day_prices and self.trader.positions:
                self.trader.mark_to_market(day_str, day_prices)

        # Final mark + close at test end
        end_prices = {}
        for ticker in active_tickers:
            px = self._price_asof(ticker, test_end_str)  # phase-69.2 boundary snap
            if px is not None:
                end_prices[ticker] = px

        self.trader.close_all_positions(test_end_str, end_prices)

        # Actual trades = entries + exits in this window
        actual_trades = len(self.trader.trades) - trades_before

        # 9. Compute window metrics
        window_returns = self.trader.get_returns_series()

        wr = WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            train_end=train_end_str,
            test_start=test_start_str,
            test_end=test_end_str,
            sharpe_ratio=self._sharpe(np.array(window_returns)) if window_returns else 0,
            total_return_pct=0,  # Computed at aggregate level
            alpha_pct=0,  # Computed at aggregate level
            max_drawdown_pct=self._max_drawdown(np.array(window_returns)) if window_returns else 0,
            hit_rate=sum(1 for p in predictions if p.get("correct")) / len(predictions) if predictions else 0,
            num_trades=actual_trades,
            n_candidates=len(candidate_tickers),
            n_train_samples=len(train_features),
            n_features=len(feature_names),
            feature_importance_mdi=mdi,
            feature_importance_mda=mda,
            predictions=predictions,
        )

        self.trader.reset()
        return wr

    # ── Training data construction ───────────────────────────────

    @staticmethod
    def _price_asof(ticker: str, date_str: str, lookback_days: int = 7) -> float | None:
        """phase-69.2 (boundary snap): as-of close for `date_str`, snapping a
        weekend/holiday boundary to the nearest PRIOR trading day. An exact-date
        lookup on a non-trading boundary returns empty -- which silently made a
        window execute zero trades and liquidate every position at its ENTRY
        price. Widen to [date-lookback, date] and take the last available close.
        """
        p = cache.cached_prices(ticker, date_str, date_str)
        if not p.empty:
            return float(p["close"].iloc[-1])
        start = (pd.Timestamp(date_str) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        p = cache.cached_prices(ticker, start, date_str)
        s = p["close"].dropna() if not p.empty else p
        if not p.empty and not s.empty:
            return float(s.iloc[-1])
        return None

    @staticmethod
    def _label_overlaps_test(sample_date: str, horizon_days: int,
                             test_start: str, test_end: str) -> bool:
        """phase-69.2 (purge, AFML Ch.7): True iff a training sample's triple-
        barrier label span [sample_date, sample_date + horizon_days] overlaps the
        test window [test_start, test_end]. Such samples leak test-period outcomes
        into training (label horizon 1.5*holding_days ~= 90-135d >> the 5-day
        embargo) and MUST be purged from the training set.
        """
        s = pd.Timestamp(sample_date)
        label_end = s + timedelta(days=int(horizon_days))
        ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
        return (s <= te) and (label_end >= ts)

    @staticmethod
    def _build_predict_features(fv: dict, feature_names: list[str],
                                train_medians: dict | None,
                                ablation_mask: list | None = None) -> pd.DataFrame:
        """phase-69.2: build the single-row PREDICT feature frame with the SAME
        imputation as training (train-median fill, not the pre-fix fillna(0)) and
        the non-stationary features placed on the train (fracdiff'd) scale via the
        persisted train median -- a single prediction row has no series window to
        reproduce the fixed-width fractional-difference convolution, and feeding
        the model raw price/market-cap levels it never trained on is the defect.
        Testable mirror of the train-time transform (_build_training_data).

        LIMITATION (phase-69.2, tracked follow-on): predict-time non-stationary
        features are median-NEUTRALIZED (a constant train median) and therefore
        carry no cross-sectional signal at predict time -- disclosed harm-reduction,
        NOT exact equivalence (train rows carry varying fracdiff'd values). The
        true fix is a per-ticker time-series FFD with the persisted data-independent
        weight vector applied inside the feature builder (build_feature_vector),
        deferred to a future live-adjacent step. See
        handoff/current/audit_phase69/followons_69.2.md."""
        tm = train_medians or {}
        row = {f: fv.get(f, np.nan) for f in feature_names}
        if ablation_mask:
            for col in ablation_mask:
                if col in row:
                    row[col] = 0
        for col in _NON_STATIONARY:
            if col in row and col in tm:
                row[col] = tm[col]
        X_test = pd.DataFrame([row])[feature_names]
        if tm:
            X_test = X_test.fillna(pd.Series(tm))  # identical to train imputation
        return X_test.fillna(0)

    #: phase-82.43: per-window matrix coverage, refreshed by
    #: `_build_training_data`. Class-level default so a caller that reads it
    #: before any window has run gets an explicit empty record rather than an
    #: AttributeError.
    _last_matrix_coverage: dict = {}

    def _build_training_data(
        self,
        tickers: list[str],
        train_start: str,
        train_end: str,
        test_start: str | None = None,
        test_end: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Build feature matrix + Triple Barrier labels + sample weights
        for the training window.

        phase-69.2: when `test_start`/`test_end` are provided, training samples
        whose label horizon (1.5*holding_days) overlaps the test window are
        PURGED (AFML Ch.7) to remove look-ahead leakage. Omitting them preserves
        the pre-fix behavior (no purge).
        """
        features_list = []
        labels_list = []
        entry_dates = []
        exit_dates = []

        # Sample at biweekly intervals within the training window
        # (~26 samples/ticker/window vs ~12 with monthly — doubles training set)
        current = pd.Timestamp(train_start) + pd.DateOffset(weeks=2)
        end = pd.Timestamp(train_end)

        sample_dates = []
        while current <= end:
            sample_dates.append(current.strftime("%Y-%m-%d"))
            current += pd.DateOffset(weeks=2)

        total_iterations = len(sample_dates) * len(tickers)
        iteration_count = 0

        # phase-69.2: the true triple-barrier label horizon is 1.5*holding_days
        # (see _compute_triple_barrier_label); use it for BOTH the purge overlap
        # test and the recorded exit date (the prior holding_days understated it).
        horizon_days = int(self.holding_days * 1.5)

        for sample_date in sample_dates:
            # phase-69.2 purge (AFML Ch.7): drop samples whose label span
            # [sample_date, sample_date+1.5*holding_days] overlaps the test
            # window -- they leak test-period outcomes into training.
            if test_start and test_end and self._label_overlaps_test(
                sample_date, horizon_days, test_start, test_end
            ):
                continue
            for ticker in tickers:
                iteration_count += 1
                try:
                    fv = self.data_provider.build_feature_vector(ticker, sample_date)
                    if not fv or fv.get("price_at_analysis") is None:
                        continue

                    label = self._compute_label(ticker, sample_date)
                    if label is None:
                        continue

                    features_list.append(fv)
                    labels_list.append(label)

                    # Track entry/exit dates for sample weight computation
                    entry_dates.append(pd.Timestamp(sample_date))
                    exit_dates.append(
                        pd.Timestamp(sample_date) + timedelta(days=horizon_days)
                    )

                    # Throttled progress: emit every 200 samples
                    if len(features_list) % 200 == 0:
                        self._report_progress(
                            "building_features",
                            f"{len(features_list)} samples built ({iteration_count}/{total_iterations} iterations)",
                            window=self._current_window_id,
                            samples_built=len(features_list),
                            samples_total=total_iterations,
                        )
                except Exception:
                    continue

        if not features_list:
            return pd.DataFrame(), np.array([]), np.array([])

        # Convert to DataFrame and extract numeric features
        df = pd.DataFrame(features_list)
        feature_cols = [c for c in _NUMERIC_FEATURES if c in df.columns]
        X = df[feature_cols].copy()

        # phase-82.43: record what the matrix ACTUALLY got, per window.
        #
        # Two different silent degradations meet at this line and neither left a
        # trace before. (a) When NO row carries the macro keys, `feature_cols`
        # never creates those columns -- `pd.DataFrame(features_list)` above
        # cannot invent a column no row has -- so the model trains on a narrower
        # matrix (measured on the 82.43 fixtures: 12 vs 18) and nothing said
        # so. (b) When only SOME
        # rows carry them the columns DO survive, and the missing cells are
        # median-imputed further down, fabricating a value with no trace. Case
        # (b) is the live one: on the real biweekly grid 2018-01-01..2019-12-31
        # (n=53) there were 1 empty, 3 partial and 49 full cutoffs.
        #
        # `macro_series_count` is a per-row 0-6 integer set by
        # `historical_data.build_feature_vector`. Recording its MINIMUM is what
        # distinguishes (b) from a clean window -- a boolean or a mean would not.
        self._last_matrix_coverage = compute_matrix_coverage(df, X)

        # Ablation mask: zero out any feature requested by an ablation
        # run so the model trains as if the feature were uninformative
        # rather than missing. No-op when mask is empty (default).
        if self._ablation_mask:
            for col in self._ablation_mask:
                if col in X.columns:
                    X[col] = 0.0

        # Apply fractional differentiation to non-stationary features
        for col in feature_cols:
            if col in _NON_STATIONARY and col in X.columns:
                series = X[col].dropna()
                if len(series) > 10:
                    diffed = HistoricalDataProvider.fractional_diff(series, d=self.frac_diff_d)
                    X.loc[diffed.index, col] = diffed

        # Fill NaN with median (robust to outliers)
        train_medians = X.median()
        # phase-69.2: persist the (post-fracdiff) per-feature TRAIN medians so the
        # PREDICT path applies the SAME imputation (was fillna(0)) and places the
        # non-stationary features -- which cannot be windowed-fracdiff'd on a
        # single prediction row -- on the train (fracdiff'd) scale rather than
        # feeding the model raw price/market-cap levels it never trained on.
        self._train_feature_medians = {
            k: float(v) for k, v in train_medians.to_dict().items() if pd.notna(v)
        }
        X = X.fillna(train_medians)
        X = X.fillna(0)  # Remaining NaN → 0

        labels = np.array(labels_list)

        # Compute sample weights via average uniqueness
        weights = self._compute_sample_weights(entry_dates, exit_dates)

        return X, labels, weights

    def _compute_triple_barrier_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Triple Barrier Method (López de Prado Ch. 3):
        +1 if price hits TP barrier first, -1 if SL barrier first, 0 if time expires.

        Transaction cost adjustment: barriers are shifted inward by the estimated
        round-trip cost (2 × transaction_cost_pct). This prevents labeling trades
        as winners when the actual profit after costs would be negative.
        Per Almgren & Chriss (2000), realistic cost modeling is essential for
        avoiding strategies that look profitable but lose to friction.
        """
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(self.holding_days * 1.5))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)

        if prices.empty:
            return None

        entry_price = float(prices["close"].iloc[0])

        # Adjust barriers for round-trip transaction costs
        # Round-trip = entry cost + exit cost ≈ 2 × transaction_cost_pct
        round_trip_cost_pct = 2 * self.trader.transaction_cost_pct / 100
        tp_price = entry_price * (1 + self.tp_pct / 100 + round_trip_cost_pct)
        sl_price = entry_price * (1 - self.sl_pct / 100 + round_trip_cost_pct)

        # Walk forward through prices
        trading_days = 0
        for idx in range(1, len(prices)):
            trading_days += 1
            price = float(prices["close"].iloc[idx])

            if price >= tp_price:
                return 1  # Hit take-profit (net of costs)
            if price <= sl_price:
                return -1  # Hit stop-loss
            if trading_days >= self.holding_days:
                return 0  # Time expired

        return 0  # Not enough data, treat as hold

    def _compute_sample_weights(
        self, entry_dates: list, exit_dates: list
    ) -> np.ndarray:
        """
        Sample weights via average uniqueness (López de Prado Ch. 4).
        Prevents GradientBoosting from overfitting to regime-dominant
        overlapping labels.
        """
        n = len(entry_dates)
        if n == 0:
            return np.array([])

        weights = np.ones(n)
        for i in range(n):
            overlap_count = 0
            for j in range(n):
                if i == j:
                    continue
                # Check if labels overlap in time
                if entry_dates[j] < exit_dates[i] and exit_dates[j] > entry_dates[i]:
                    overlap_count += 1

            # Average uniqueness = 1 / (1 + overlap_count)
            weights[i] = 1.0 / (1.0 + overlap_count)

        # Normalize so weights sum to n
        if weights.sum() > 0:
            weights = weights * n / weights.sum()

        return weights

    # ── Model training ───────────────────────────────────────────

    def _train_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[GradientBoostingClassifier, list[str]]:
        """Train GradientBoosting classifier with sample weights."""
        model = GradientBoostingClassifier(
            n_estimators=self.ml_params["n_estimators"],
            max_depth=self.ml_params["max_depth"],
            min_samples_leaf=self.ml_params["min_samples_leaf"],
            learning_rate=self.ml_params["learning_rate"],
            random_state=42,
        )
        model.fit(X, y, sample_weight=sample_weights)
        from datetime import datetime as _dt, timezone as _tz
        self.model_trained_at = _dt.now(_tz.utc).isoformat()
        return model, list(X.columns)

    def _compute_mda(
        self,
        model: GradientBoostingClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 5,
    ) -> dict[str, float]:
        """
        MDA (permutation importance) — primary feature ranking.
        López de Prado Ch. 8: MDI biases toward high-cardinality features;
        MDA is the authoritative ranking.
        """
        try:
            import platform
            _n_jobs = 1 if platform.system() == "Darwin" else -1
            result = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=_n_jobs,
            )
            return dict(zip(feature_names, result["importances_mean"]))
        except Exception as e:
            logger.warning(f"MDA computation failed: {e}")
            return {}

    # ── Prediction + trade signal generation ─────────────────────

    def _predict_and_trade(
        self,
        model: GradientBoostingClassifier,
        feature_names: list[str],
        test_tickers: list[str],
        test_start: str,
        test_end: str,
        meta_model: GradientBoostingClassifier | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Generate predictions on test period candidates.
        Returns (signals for trader, predictions for analytics).

        If meta_model is provided (meta-labeling, AFML Ch. 3.6):
        - Primary model provides direction (label)
        - Meta-model provides probability that primary is correct
        - This probability is used for position sizing (fractional Kelly)
        - Predictions with meta_probability < 0.5 are filtered to HOLD (label=0)
        """
        signals = []
        predictions = []

        for ticker in test_tickers:
            try:
                fv = self.data_provider.build_feature_vector(ticker, test_start)
                if not fv or fv.get("price_at_analysis") is None:
                    continue

                # phase-69.2: build the predict frame via the shared, testable
                # transform (train-median imputation + non-stationary features on
                # the train fracdiff'd scale). Mirrors _build_training_data.
                X_test = self._build_predict_features(
                    fv, feature_names,
                    getattr(self, "_train_feature_medians", None),
                    self._ablation_mask,
                )

                pred_label = int(model.predict(X_test)[0])
                pred_proba = model.predict_proba(X_test)[0]

                # Get probability for the predicted class
                classes = list(model.classes_)
                if pred_label in classes:
                    probability = float(pred_proba[classes.index(pred_label)])
                else:
                    probability = 0.5

                # Meta-labeling: override probability with meta-model's confidence
                # that the primary prediction is correct (AFML Ch. 3.6)
                if meta_model is not None:
                    meta_features = self._build_meta_features(
                        X_test, pred_label, probability, feature_names,
                    )
                    meta_proba = meta_model.predict_proba(meta_features)[0]
                    # Meta-model predicts P(primary is correct)
                    # Class 1 = "primary was correct"
                    meta_classes = list(meta_model.classes_)
                    if 1 in meta_classes:
                        meta_probability = float(meta_proba[meta_classes.index(1)])
                    else:
                        meta_probability = 0.5

                    # Filter: if meta-model says primary is likely wrong, don't trade
                    if meta_probability < 0.5:
                        pred_label = 0  # Override to HOLD
                    probability = meta_probability  # Use meta confidence for sizing

                volatility = fv.get("annualized_volatility", 0.3) or 0.3

                signals.append({
                    "ticker": ticker,
                    "label": pred_label,
                    "probability": probability,
                    "volatility": volatility,
                })

                # Check actual outcome for hit rate
                actual_label = self._compute_label(ticker, test_start)
                predictions.append({
                    "ticker": ticker,
                    "date": test_start,
                    "predicted_label": pred_label,
                    "probability": probability,
                    "actual_label": actual_label,
                    "correct": pred_label == actual_label if actual_label is not None else None,
                })

            except Exception as e:
                logger.debug(f"Prediction failed for {ticker}: {e}")

        return signals, predictions

    # ── Meta-Labeling (AFML Ch. 3.6) ──────────────────────────────

    def _train_meta_label_model(
        self,
        primary_model: GradientBoostingClassifier,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_names: list[str],
        sample_weights: np.ndarray,
    ) -> GradientBoostingClassifier | None:
        """
        Train a secondary (meta-label) model per López de Prado AFML Ch. 3.6.

        The meta-label model learns WHEN the primary model is correct.
        Input: original features + primary model's prediction + primary confidence.
        Output: binary 0/1 (was primary correct?).

        This enables intelligent bet sizing — the meta-model's probability
        output replaces the primary model's raw probability for position sizing,
        implementing a form of fractional Kelly criterion.

        Uses 3-fold cross-validation on the training set to avoid overfitting
        the meta-labels to the primary model's in-sample predictions.
        """
        from sklearn.model_selection import cross_val_predict

        try:
            # Step 1: Generate primary model predictions via cross-validation
            # This avoids leakage — we can't use the model's own training predictions
            primary_preds = cross_val_predict(
                GradientBoostingClassifier(
                    n_estimators=self.ml_params["n_estimators"],
                    max_depth=max(2, self.ml_params["max_depth"] - 1),  # Slightly shallower
                    min_samples_leaf=self.ml_params["min_samples_leaf"],
                    learning_rate=self.ml_params["learning_rate"],
                    random_state=42,
                ),
                X_train, y_train, cv=3, method="predict",
            )

            # Step 2: Create meta-labels — was the primary prediction correct?
            meta_labels = (primary_preds == y_train).astype(int)

            # Need enough positive and negative meta-labels for training
            if meta_labels.sum() < 10 or (len(meta_labels) - meta_labels.sum()) < 10:
                logger.warning("Meta-labeling: insufficient label diversity, skipping")
                return None

            # Step 3: Build meta-features (original features + primary signal info)
            # Get primary model's predictions and probabilities on training data
            primary_proba = primary_model.predict_proba(X_train)
            primary_labels = primary_model.predict(X_train)

            meta_X = self._build_meta_features_batch(
                X_train, primary_labels, primary_proba, feature_names, primary_model,
            )

            # Step 4: Train meta-model (binary classifier: correct vs incorrect)
            meta_model = GradientBoostingClassifier(
                n_estimators=max(50, self.ml_params["n_estimators"] // 2),
                max_depth=max(2, self.ml_params["max_depth"] - 1),
                min_samples_leaf=self.ml_params["min_samples_leaf"],
                learning_rate=self.ml_params["learning_rate"],
                random_state=42,
            )
            meta_model.fit(meta_X, meta_labels, sample_weight=sample_weights)

            accuracy = (meta_model.predict(meta_X) == meta_labels).mean()
            logger.info(
                "Meta-label model trained: %d samples, in-sample accuracy=%.3f, "
                "primary correct rate=%.3f",
                len(meta_labels), accuracy, meta_labels.mean(),
            )
            return meta_model

        except Exception as e:
            logger.warning("Meta-label training failed: %s", e)
            return None

    def _build_meta_features(
        self,
        X_test: pd.DataFrame,
        pred_label: int,
        probability: float,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """Build meta-features for a single test observation."""
        row = X_test.copy()
        row["primary_label"] = pred_label
        row["primary_confidence"] = probability
        row["primary_is_buy"] = int(pred_label == 1)
        row["primary_is_sell"] = int(pred_label == -1)
        return row

    def _build_meta_features_batch(
        self,
        X: pd.DataFrame,
        primary_labels: np.ndarray,
        primary_proba: np.ndarray,
        feature_names: list[str],
        primary_model: GradientBoostingClassifier,
    ) -> pd.DataFrame:
        """Build meta-features for the full training set."""
        meta_X = X.copy()
        meta_X["primary_label"] = primary_labels

        # Get max probability across classes for each sample
        meta_X["primary_confidence"] = primary_proba.max(axis=1)

        # Binary indicators for direction
        meta_X["primary_is_buy"] = (primary_labels == 1).astype(int)
        meta_X["primary_is_sell"] = (primary_labels == -1).astype(int)

        return meta_X

    # ── Metric helpers ───────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free_rate: float = 0.04) -> float:
        # Lazy import to avoid circular dependency (analytics imports BacktestResult)
        from backend.backtest.analytics import compute_sharpe
        return compute_sharpe(returns, risk_free_rate)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        from backend.backtest.analytics import compute_max_drawdown
        return compute_max_drawdown(cumulative)

    def _empty_window_result(self, window: WalkForwardWindow) -> WindowResult:
        return WindowResult(
            window_id=window.window_id,
            train_start=window.train_start.isoformat(),
            train_end=window.train_end.isoformat(),
            test_start=window.test_start.isoformat(),
            test_end=window.test_end.isoformat(),
            sharpe_ratio=0, total_return_pct=0, alpha_pct=0,
            max_drawdown_pct=0, hit_rate=0, num_trades=0,
        )

    def _report_progress(self, step: str, detail: str = "", **kwargs):
        stats = cache.get_cache_stats()
        data = {
            "window": kwargs.get("window", 0),
            "total_windows": self._total_windows,
            "step": step,
            "step_detail": detail,
            "candidates_found": kwargs.get("candidates_found", 0),
            "samples_built": kwargs.get("samples_built", 0),
            "samples_total": kwargs.get("samples_total", 0),
            "elapsed_seconds": round(time.time() - self._backtest_start_time, 1)
            if self._backtest_start_time
            else 0,
            "cache_hits": stats["hits"],
            "cache_misses": stats["misses"],
        }
        if self.progress_callback:
            self.progress_callback(data)
        logger.info("Backtest: [%s] %s", step, detail)

    # ── Strategy Label Dispatcher ────────────────────────────────

    def _compute_label(self, ticker: str, entry_date: str) -> int | None:
        """Dispatch to the active strategy's label method."""
        method_name = STRATEGY_REGISTRY.get(self.strategy, "_compute_triple_barrier_label")
        method = getattr(self, method_name)
        return method(ticker, entry_date)

    # ── Alternative Label Methods ────────────────────────────────

    def _compute_quality_momentum_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Quality Momentum (Asness et al. 2019):
        +1 if 6-month momentum > 0 AND quality_score > median, -1 if both negative, 0 otherwise.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        momentum_6m = fv.get("momentum_6m")
        quality_score = fv.get("quality_score", 0) or 0

        if momentum_6m is None:
            return None

        if momentum_6m > 5 and quality_score > 0.3:
            return 1
        if momentum_6m < -5 and quality_score < 0.1:
            return -1
        return 0

    def _compute_mean_reversion_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Mean Reversion (Lo & MacKinlay 1990, Poterba & Summers 1988):
        Two-stage label:
          1. Signal: is the stock oversold or overbought? (SMA distance + RSI)
          2. Validation: does the price actually revert within mr_holding_days?

        Uses mr_holding_days (default 15, range 5-30) for the short reversion
        horizon that the academic literature recommends. Mean reversion works
        at 1-4 week horizons; longer horizons transition to momentum territory
        (Jegadeesh & Titman 1993).

        +1 if oversold AND price reverts up by ≥ half the SMA gap within mr_holding_days
        -1 if overbought AND price reverts down by ≥ half the SMA gap within mr_holding_days
        0 if no signal or reversion doesn't materialize
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        sma_dist = fv.get("sma_50_distance")
        rsi = fv.get("rsi_14")
        if sma_dist is None or rsi is None:
            return None

        entry_price = fv["price_at_analysis"]

        # Stage 1: Signal detection
        is_oversold = sma_dist < -0.05 and rsi < 35
        is_overbought = sma_dist > 0.10 and rsi > 70

        if not is_oversold and not is_overbought:
            return 0

        # Stage 2: Forward validation — did the price actually revert?
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(self.mr_holding_days * 2))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)

        if prices.empty or len(prices) < 3:
            return None

        # Check if price reverts toward SMA within mr_holding_days trading days
        trading_days = 0
        for idx in range(1, len(prices)):
            trading_days += 1
            if trading_days > self.mr_holding_days:
                break
            price = float(prices["close"].iloc[idx])

            if is_oversold:
                # Reversion target: recover at least half the gap to SMA
                # If SMA dist was -8%, target is entry_price × (1 + 0.04)
                reversion_target = entry_price * (1 + abs(sma_dist) / 2)
                if price >= reversion_target:
                    return 1
            elif is_overbought:
                # Reversion target: fall at least half the gap from SMA
                reversion_target = entry_price * (1 - sma_dist / 2)
                if price <= reversion_target:
                    return -1

        return 0  # Signal present but reversion didn't materialize

    def _compute_factor_label(self, ticker: str, entry_date: str) -> int | None:
        """
        Multi-Factor Composite — Fama & French (2015) "A Five-Factor Model",
        augmented with Carhart (1997) momentum and Novy-Marx (2013) profitability.

        Factors and weights:
          1. Value (25%): P/B ratio — Fama-French use book-to-market (HML).
             P/B is the inverse; lower P/B = higher B/M = deeper value.
          2. Momentum (25%): 12-1 month return — Jegadeesh & Titman (1993).
             Uses 12m return minus 1m return to avoid short-term reversal.
          3. Low Volatility (15%): Frazzini & Pedersen (2014) "Betting Against Beta".
          4. Quality/Profitability (25%): Novy-Marx (2013), using the Asness (2019)
             QMJ composite quality_score from build_feature_vector().
          5. Yield (10%): Dividend yield as payout factor proxy.

        Each factor uses a sigmoid-like normalization centered on typical S&P 500
        values rather than hardcoded min/max caps, making it more robust across
        market regimes. The old hardcoded ranges (P/E 5-40, vol 0.10-0.60) broke
        for growth stocks and extreme market conditions.

        +1 if composite > 0.6, -1 if composite < 0.3, 0 otherwise.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        # Use P/B (Fama-French HML proxy) instead of P/E for value factor
        pb = fv.get("pb_ratio")
        pe = fv.get("pe_ratio")
        # 12-1 momentum: 12m return minus 1m to avoid short-term reversal
        mom_12m = fv.get("momentum_12m")
        mom_1m = fv.get("momentum_1m")
        vol = fv.get("annualized_volatility")
        quality = fv.get("quality_score")  # Now uses full Asness (2019) QMJ
        div_yield = fv.get("dividend_yield", 0) or 0

        # Need at least momentum and one valuation metric
        if vol is None:
            return None
        if pb is None and pe is None:
            return None
        if mom_12m is None:
            return None

        # 1. Value: lower P/B is better. Sigmoid centered at P/B = 3 (S&P median)
        # P/B < 1.5 → high value score, P/B > 6 → low value score
        if pb is not None and pb > 0:
            value_score = max(0, min(1, 1.0 / (1.0 + np.exp((pb - 3.0) / 1.5))))
        elif pe is not None and pe > 0:
            # Fallback to P/E if P/B unavailable. Centered at P/E = 20
            value_score = max(0, min(1, 1.0 / (1.0 + np.exp((pe - 20.0) / 8.0))))
        else:
            value_score = 0.5  # Neutral if no valuation data

        # 2. Momentum: 12-1 month return (Jegadeesh & Titman 1993)
        mom_12_1 = mom_12m - (mom_1m or 0)
        # Sigmoid centered at 0%: positive momentum → higher score
        mom_score = max(0, min(1, 1.0 / (1.0 + np.exp(-mom_12_1 / 15.0))))

        # 3. Low Volatility: lower is better. Sigmoid centered at 25% (S&P median)
        vol_score = max(0, min(1, 1.0 / (1.0 + np.exp((vol - 0.25) / 0.10))))

        # 4. Quality: use the Asness (2019) QMJ composite directly (already 0-1)
        quality_score = quality if quality is not None else 0.5

        # 5. Yield: dividend yield. Sigmoid centered at 2%
        yield_score = max(0, min(1, 1.0 / (1.0 + np.exp(-(div_yield - 0.02) / 0.015)))) if div_yield > 0 else 0.3

        composite = (
            value_score * 0.25 + mom_score * 0.25 + vol_score * 0.15
            + quality_score * 0.25 + yield_score * 0.10
        )

        if composite > 0.6:
            return 1
        if composite < 0.3:
            return -1
        return 0

    # ── phase-82.2: overpriced-market candidate labels ───────────
    #
    # Three properties every method below holds, and the reasons they are not
    # negotiable:
    #
    # FORWARD-LOOKING. Each walks prices AFTER entry_date. `quality_momentum`
    # and `factor_model` above do not -- they classify on features at
    # entry_date, so a model trained on them reproduces a threshold rule over
    # its own inputs and any Sharpe/DSR derived from them is not evidence about
    # the future (step 82.16).
    #
    # SIGMA-SCALED. Barriers are set in units of the name's own volatility, not
    # in raw fractions. A fixed 8% barrier is sub-1-day noise for a high-beta
    # semiconductor and a large move for a utility, so a fixed fraction silently
    # means different things across the cross-section.
    #
    # COST-ADJUSTED. Barriers are shifted by the round-trip cost exactly as
    # `_compute_triple_barrier_label` does. The existing `mean_reversion` label
    # has NO cost adjustment, so it can label a reversion a winner when the
    # round trip loses money after friction.
    #
    # NONE-ON-NO-SIGNAL. A row with no signal returns None (excluded from
    # training) rather than 0. Returning 0 is a principal degeneracy driver in
    # the existing `mean_reversion`, where two of three exit paths yield 0 and
    # the neutral class swamps the label set.

    def _sigma_barriers(self, fv: dict, horizon_days: int) -> tuple[float, float] | None:
        """Per-name barrier half-width in FRACTIONAL terms, scaled to horizon.

        `annualized_volatility` is a percentage (e.g. 35.0 for 35%/yr). Convert
        to a fraction, de-annualise by sqrt(252) and re-scale by sqrt(horizon).
        Returns (sigma_h, round_trip_cost_pct) or None when vol is unusable.
        """
        vol_ann = fv.get("annualized_volatility")
        if vol_ann is None:
            return None
        try:
            vol_ann = float(vol_ann)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(vol_ann) or vol_ann <= 0:
            return None
        sigma_daily = (vol_ann / 100.0) / np.sqrt(252.0)
        sigma_h = float(sigma_daily * np.sqrt(max(1, int(horizon_days))))
        if not np.isfinite(sigma_h) or sigma_h <= 0:
            return None
        round_trip_cost_pct = 2 * self.trader.transaction_cost_pct / 100
        return sigma_h, round_trip_cost_pct

    def _market_stretch(self, entry_date: str) -> float | None:
        """Market turbulence proxy in [0, inf): trailing SPY realised vol over
        its own longer-run average. 1.0 = normal, >1 = turbulent.

        Deliberately uses SPY rather than `compute_turbulence_index`
        (`historical_data.py`), which needs the universe cross-section -- that
        list is not reachable from a label method, and stashing it would mean
        editing `_run_window`. SPY is always preloaded alongside the universe
        (see `.claude/rules/backend-backtest.md`), so this costs nothing.

        Keyed on the research finding that crowded MOMENTUM carries LOWER crash
        probability while crowding predicts crashes rather than means
        (arXiv:2512.11913) -- so the regime must key on turbulence/co-movement,
        NOT on "momentum has run".

        Strictly backward-looking: the window ends at entry_date.
        """
        long_win, short_win = 252, 21
        start = (pd.Timestamp(entry_date) - timedelta(days=int(long_win * 1.7))).strftime("%Y-%m-%d")
        px = cache.cached_prices("SPY", start, entry_date)
        if px is None or px.empty or len(px) < long_win // 2:
            return None
        close = px["close"].astype(float)
        rets = close.pct_change().dropna()
        if len(rets) < short_win + 20:
            return None
        short_vol = float(rets.tail(short_win).std())
        long_vol = float(rets.tail(long_win).std())
        if not np.isfinite(short_vol) or not np.isfinite(long_vol) or long_vol <= 0:
            return None
        return short_vol / long_vol

    def _walk_barriers(self, ticker: str, entry_date: str, horizon_days: int,
                       up: float, down: float) -> int | None:
        """Walk forward from entry_date; +1 if `up` hit first, -1 if `down`
        first, 0 if the horizon expires. `up`/`down` are FRACTIONAL returns."""
        end_date = (pd.Timestamp(entry_date) + timedelta(days=int(horizon_days * 1.6))).strftime("%Y-%m-%d")
        prices = cache.cached_prices(ticker, entry_date, end_date)
        if prices is None or prices.empty or len(prices) < 2:
            return None
        entry_price = float(prices["close"].iloc[0])
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None
        up_price = entry_price * (1.0 + up)
        down_price = entry_price * (1.0 - down)
        for idx in range(1, len(prices)):
            price = float(prices["close"].iloc[idx])
            if price >= up_price:
                return 1
            if price <= down_price:
                return -1
            if idx >= horizon_days:
                return 0
        return 0

    def _compute_stretch_regime_label(self, ticker: str, entry_date: str) -> int | None:
        """Lens (a) valuation/market-stretch REGIME GATE, with lens (d) the
        cash-timing overlay folded in.

        Symmetric sigma barriers, MODULATED by market turbulence. As stretch
        rises the up-barrier widens and the down-barrier tightens, so fewer
        rows earn +1 and the trained model buys less -- the model holds cash by
        being harder to convince, rather than by a bolted-on timing rule. That
        is the (d) overlay: an exposure decision expressed inside the label.

        Returns None when volatility or the market proxy is unavailable, so
        rows we cannot regime-classify are EXCLUDED rather than labelled 0.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None
        bars = self._sigma_barriers(fv, self.holding_days)
        if bars is None:
            return None
        sigma_h, cost = bars

        stretch = self._market_stretch(entry_date)
        if stretch is None:
            return None
        # Clamp so one turbulent week cannot make the label unreachable.
        stretch = float(min(max(stretch, 0.5), 2.5))

        base_k = 1.0
        up = base_k * sigma_h * stretch + cost      # harder to earn +1 when turbulent
        down = base_k * sigma_h / stretch + cost    # easier to earn -1 when turbulent
        return self._walk_barriers(ticker, entry_date, self.holding_days, up, down)

    def _compute_qarp_label(self, ticker: str, entry_date: str) -> int | None:
        """Lens (b) quality-at-a-reasonable-price defensive tilt. LONG-ONLY.

        Gate on fundamentals at entry_date -- cheap, profitable, low leverage --
        then label the forward move with a sigma barrier. A name that fails the
        gate returns None so it never enters training; that is what keeps the
        label set from being swamped by non-candidates.

        Deliberately does NOT reuse `quality_score`: its payout leg is the
        weakest QMJ dimension (Asness/Frazzini/Pedersen) and it rests on an
        `fcf_yield` computed with capex = 0. The individual fundamentals are
        used directly instead.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        pe = fv.get("pe_ratio")
        roe = fv.get("roe")
        de = fv.get("debt_equity")
        margin = fv.get("profit_margin")
        # Require the full QARP triple. A partial view is not a QARP candidate.
        if pe is None or roe is None or de is None:
            return None
        try:
            pe, roe, de = float(pe), float(roe), float(de)
        except (TypeError, ValueError):
            return None

        cheap = 0 < pe <= 25.0            # positive earnings AND not expensive
        quality = roe >= 0.10             # 10% return on equity
        low_debt = 0 <= de <= 1.5
        profitable = margin is None or float(margin) > 0

        if not (cheap and quality and low_debt and profitable):
            return None  # not a QARP candidate -- excluded, not neutral

        bars = self._sigma_barriers(fv, self.holding_days)
        if bars is None:
            return None
        sigma_h, cost = bars
        # Defensive asymmetry: take profit sooner, cut losses later, so the
        # tilt is expressed in the barrier geometry and not only in the screen.
        up = 1.0 * sigma_h + cost
        down = 1.5 * sigma_h + cost
        return self._walk_barriers(ticker, entry_date, self.holding_days, up, down)

    def _compute_reversion_sigma_label(self, ticker: str, entry_date: str) -> int | None:
        """Lens (c) mean-reversion on overextension, in SIGMA units.

        Two changes from `_compute_mean_reversion_label`:
          1. The stretch gate is `sma_50_distance / sigma_daily` -- a z-score --
             rather than the fixed fractions -0.05 / +0.10. Those fractions are
             internally consistent (`sma_50_distance` IS a fraction) but they
             are not volatility-scaled, so they mean different things for a
             high-beta semiconductor and a utility.
          2. No-signal returns **None**, not 0. The existing label returns 0 on
             two of three paths, which is a principal reason its neutral class
             dominates.
        Cost-adjusted, which the existing mean-reversion label is not.
        """
        fv = self.data_provider.build_feature_vector(ticker, entry_date)
        if not fv or fv.get("price_at_analysis") is None:
            return None

        sma_dist = fv.get("sma_50_distance")
        if sma_dist is None:
            return None
        bars = self._sigma_barriers(fv, self.mr_holding_days)
        if bars is None:
            return None
        sigma_h, cost = bars

        try:
            stretch_z = float(sma_dist) / sigma_h
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if not np.isfinite(stretch_z):
            return None

        Z = 1.0  # overextension threshold, in sigma
        if stretch_z <= -Z:
            # Oversold: expect reversion UP by half the gap, net of costs.
            target = abs(float(sma_dist)) / 2.0 + cost
            stop = 1.5 * sigma_h + cost
            return self._walk_barriers(ticker, entry_date, self.mr_holding_days, target, stop)
        if stretch_z >= Z:
            # Overbought: expect reversion DOWN. -1 is the informative outcome.
            target = abs(float(sma_dist)) / 2.0 + cost
            stop = 1.5 * sigma_h + cost
            res = self._walk_barriers(ticker, entry_date, self.mr_holding_days, stop, target)
            return res
        return None  # no overextension -- excluded, NOT neutral

    # ── Auto-Ingest ──────────────────────────────────────────────

    def _auto_ingest_if_needed(self, universe_tickers: list[str]):
        """Check BQ table row counts; auto-ingest if historical_prices is empty."""
        try:
            from backend.backtest.data_ingestion import DataIngestionService
            from backend.config.settings import get_settings

            settings = get_settings()
            ingestion = DataIngestionService(self._bq_client, settings)
            status = ingestion.get_ingestion_status()

            prices_count = status.get("historical_prices", 0)
            if prices_count == 0:
                logger.warning("Auto-ingest: historical_prices is empty, triggering full ingestion...")
                self._report_progress("Auto-ingesting historical data (first run)...")
                ingestion.run_full_ingestion(universe_tickers[:100])
                logger.info("Auto-ingest complete")
            else:
                logger.info(f"Auto-ingest check: {prices_count} price rows found, skipping ingestion")
        except Exception as e:
            logger.warning(f"Auto-ingest check failed (non-fatal): {e}")
