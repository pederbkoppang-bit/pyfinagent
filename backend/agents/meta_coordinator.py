"""
MetaCoordinator — Cross-loop sequencing for the three optimization loops.

Decides which optimizer to run next based on portfolio health signals,
and bridges insights between loops (MDA features → agent targeting).

Architecture:
    QuantOpt (fast, minutes/cycle) → extract MDA → target SkillOpt
    SkillOpt (slow, days/cycle)    → validate via 1-window backtest proxy
    PerfOpt  (fast, minutes/cycle) → independent, latency-driven

Research basis:
    - Karpathy autoresearch: scalar metric, keep/discard, LOOP FOREVER
    - FinRL three-layer: Data→Agent→Analytics formally wired as feedback
    - BlackRock regime-aware: market conditions determine optimizer priority
    - Lopez-Lira 2023: quant-only for historical, LLM for live (contamination guard)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Feature → Agent Mapping ──────────────────────────────────────

# Maps MDA feature names to the agent skill files they're most influenced by.
# When QuantOpt's MDA says a feature matters, MetaCoordinator targets the
# responsible agent's prompt for SkillOpt improvement.
FEATURE_TO_AGENT: dict[str, list[str]] = {
    # Sentiment features → NLP/Social agents
    "nlp_sentiment_score": ["nlp_sentiment_agent"],
    "social_sentiment_score": ["social_sentiment_agent"],
    # Insider/Options → respective agents
    "insider_signal": ["insider_activity_agent"],
    "options_signal": ["options_flow_agent"],
    # Patent/Innovation → patent agent
    "patent_signal": ["patent_innovation_agent"],
    # Earnings → earnings tone agent
    "earnings_confidence": ["earnings_tone_agent"],
    # Sector features → sector agent
    "sector_signal": ["sector_analysis_agent"],
    # Macro features → enhanced macro agent
    "fed_funds_rate": ["enhanced_macro_agent"],
    "cpi_yoy": ["enhanced_macro_agent"],
    "unemployment_rate": ["enhanced_macro_agent"],
    "yield_curve_spread": ["enhanced_macro_agent"],
    "consumer_sentiment": ["enhanced_macro_agent"],
    # Debate features → debate agents
    "bull_confidence": ["bull_agent"],
    "bear_confidence": ["bear_agent"],
    "contradiction_count": ["moderator_agent"],
    # Risk features → risk/scenario agents
    "var_95_6m": ["scenario_analysis_agent"],
    "var_99_6m": ["scenario_analysis_agent"],
    "anomaly_count": ["anomaly_detection_agent"],
    "annualized_volatility": ["scenario_analysis_agent"],
    # Fundamental features → synthesis/RAG (no single enrichment agent)
    "pe_ratio": ["synthesis_agent"],
    "debt_equity": ["synthesis_agent"],
    "revenue_growth_yoy": ["synthesis_agent"],
    # Quant model features → quant model agent
    "quality_score": ["quant_model_agent"],
    "momentum_6m": ["quant_model_agent"],
    "rsi_14": ["quant_model_agent"],
}


# ── Health Signals ───────────────────────────────────────────────

@dataclass
class PortfolioHealth:
    """Snapshot of portfolio health for coordinator decisions."""
    sharpe_ratio: float = 0.0
    agent_accuracy: float = 0.0  # % of directionally correct recommendations
    p95_latency_ms: float = 0.0
    data_quality_score: float = 1.0
    days_since_last_quant_opt: int = 999
    days_since_last_skill_opt: int = 999


@dataclass
class CoordinatorDecision:
    """What the MetaCoordinator decided to do."""
    action: str  # "quant_opt" | "skill_opt" | "perf_opt" | "idle"
    reason: str
    target_agents: list[str] = field(default_factory=list)
    priority: int = 0  # higher = more urgent


# ── Thresholds ───────────────────────────────────────────────────

DEFAULT_SHARPE_TARGET = 0.5
DEFAULT_ACCURACY_TARGET = 0.55  # 55% directional accuracy
DEFAULT_LATENCY_THRESHOLD_MS = 500.0
MIN_QUANT_OPT_INTERVAL_DAYS = 1
MIN_SKILL_OPT_INTERVAL_DAYS = 7


# ── MetaCoordinator ─────────────────────────────────────────────

class MetaCoordinator:
    """
    Sequences the three optimization loops based on portfolio health.

    Decision logic (priority order):
    1. High API latency → PerfOpt (fast, no cost)
    2. Low Sharpe ratio → QuantOpt (fast, no LLM cost)
    3. Low agent accuracy → SkillOpt (slow, uses LLM outcomes)
    4. Otherwise → idle
    """

    def __init__(
        self,
        sharpe_target: float = DEFAULT_SHARPE_TARGET,
        accuracy_target: float = DEFAULT_ACCURACY_TARGET,
        latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
    ):
        self.sharpe_target = sharpe_target
        self.accuracy_target = accuracy_target
        self.latency_threshold_ms = latency_threshold_ms
        self._last_mda_features: list[dict] = []

    def decide(self, health: PortfolioHealth) -> CoordinatorDecision:
        """
        Determine which optimizer loop should run next.

        Returns a CoordinatorDecision with action, reason, and optional
        target agents (for SkillOpt).
        """
        # Priority 1: Latency issues (cheap to fix, user-visible)
        if health.p95_latency_ms > self.latency_threshold_ms:
            return CoordinatorDecision(
                action="perf_opt",
                reason=f"p95 latency {health.p95_latency_ms:.0f}ms > {self.latency_threshold_ms:.0f}ms threshold",
                priority=3,
            )

        # Priority 2: Low Sharpe (quant params need tuning)
        if (
            health.sharpe_ratio < self.sharpe_target
            and health.days_since_last_quant_opt >= MIN_QUANT_OPT_INTERVAL_DAYS
        ):
            return CoordinatorDecision(
                action="quant_opt",
                reason=f"Sharpe {health.sharpe_ratio:.2f} < {self.sharpe_target:.2f} target",
                priority=2,
            )

        # Priority 3: Low agent accuracy (prompts need improvement)
        if (
            health.agent_accuracy < self.accuracy_target
            and health.days_since_last_skill_opt >= MIN_SKILL_OPT_INTERVAL_DAYS
        ):
            target_agents = self._get_mda_target_agents()
            return CoordinatorDecision(
                action="skill_opt",
                reason=f"Agent accuracy {health.agent_accuracy:.1%} < {self.accuracy_target:.1%} target",
                target_agents=target_agents,
                priority=1,
            )

        return CoordinatorDecision(
            action="idle",
            reason="All metrics within targets",
            priority=0,
        )

    # ── MDA → Agent Bridge ───────────────────────────────────────

    def update_mda_features(self, mda_importances: list[dict]) -> None:
        """
        Store MDA feature importances from the latest QuantOpt backtest.

        Args:
            mda_importances: List of {"feature": str, "importance": float}
                sorted by importance descending.
        """
        self._last_mda_features = mda_importances
        if mda_importances:
            top_3 = [f["feature"] for f in mda_importances[:3]]
            logger.info(f"MetaCoordinator: MDA top features updated: {top_3}")

    def _get_mda_target_agents(self) -> list[str]:
        """
        Map top MDA features to responsible agent skill files.

        This is the MDA→Agent bridge — our unique research contribution.
        When backtesting reveals which features predict well, we target
        the agents responsible for producing those features.
        """
        if not self._last_mda_features:
            return []

        target_agents: list[str] = []
        seen = set()

        for feat_info in self._last_mda_features[:5]:
            feature = feat_info.get("feature", "")
            agents = FEATURE_TO_AGENT.get(feature, [])
            for agent in agents:
                if agent not in seen:
                    target_agents.append(agent)
                    seen.add(agent)

        if target_agents:
            logger.info(f"MetaCoordinator: MDA->Agent bridge targets: {target_agents}")
        return target_agents

    # ── Health Gathering ─────────────────────────────────────────

    @staticmethod
    def gather_health(
        bq_client=None,
        perf_tracker=None,
        paper_snapshots: Optional[list] = None,
    ) -> PortfolioHealth:
        """
        Gather portfolio health signals from available sources.

        Pass None for any source that isn't available; defaults will be used.
        """
        health = PortfolioHealth()

        # Sharpe from paper trading snapshots
        if paper_snapshots and len(paper_snapshots) >= 6:
            from backend.services.perf_metrics import compute_sharpe_from_snapshots
            health.sharpe_ratio = compute_sharpe_from_snapshots(paper_snapshots)

        # Agent accuracy from BQ outcome tracking
        if bq_client:
            try:
                stats = bq_client.get_performance_stats()
                health.agent_accuracy = stats.get("benchmark_beat_rate", 0.0) or 0.0
            except Exception:
                pass

        # API latency from perf tracker
        if perf_tracker:
            try:
                summary = perf_tracker.summarize()
                health.p95_latency_ms = summary.get("p95_ms", 0.0)
            except Exception:
                pass

        return health

    # ── Proxy Metric for SkillOpt ───────────────────────────────

    @staticmethod
    def run_proxy_validation(settings) -> Optional[float]:
        """
        Fast proxy metric for SkillOpt: run a single backtest window
        instead of waiting 7+ days for BQ outcome data.

        Returns Sharpe ratio from one test window, or None on failure.
        This does NOT use LLM (quant-only), so it's safe for validation
        without contamination risk (Lopez-Lira 2023).
        """
        try:
            from backend.backtest.backtest_engine import BacktestEngine
            from backend.db.bigquery_client import BigQueryClient

            bq = BigQueryClient(settings)
            engine = BacktestEngine(
                bq_client=bq,
                project=settings.gcp_project_id,
                dataset=settings.bq_dataset_reports,
                start_date=settings.backtest_start_date,
                end_date=settings.backtest_end_date,
                train_window_months=settings.backtest_train_window_months,
                test_window_months=settings.backtest_test_window_months,
                embargo_days=settings.backtest_embargo_days,
                holding_days=settings.backtest_holding_days,
                tp_pct=settings.backtest_tp_pct,
                sl_pct=settings.backtest_sl_pct,
                starting_capital=settings.backtest_starting_capital,
                max_positions=settings.backtest_max_positions,
                top_n_candidates=settings.backtest_top_n_candidates,
            )
            result = engine.run_backtest()
            if result and result.aggregate_sharpe is not None:
                logger.info(
                    f"MetaCoordinator: proxy validation Sharpe={result.aggregate_sharpe:.4f}"
                )
                return result.aggregate_sharpe
        except Exception as e:
            logger.warning(f"MetaCoordinator: proxy validation failed: {e}")
        return None

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        """Current coordinator state for API exposure."""
        return {
            "sharpe_target": self.sharpe_target,
            "accuracy_target": self.accuracy_target,
            "latency_threshold_ms": self.latency_threshold_ms,
            "mda_feature_count": len(self._last_mda_features),
            "top_mda_features": [
                f["feature"] for f in self._last_mda_features[:5]
            ] if self._last_mda_features else [],
        }
