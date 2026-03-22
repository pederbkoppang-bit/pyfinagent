"""
QuantStrategyOptimizer — Karpathy autoresearch-style fast inner optimization loop.

Modifies quant strategy params (Triple Barrier, ML hyperparams, feature selection,
portfolio sizing) and evaluates via walk-forward backtest at zero LLM cost.

Two modes:
- Zero-cost (default): random perturbation of strategy params
- LLM mode ($0.01/proposal): Gemini Flash analyzes experiment history

Guard: Deflated Sharpe Ratio >= 0.95 rejects overfitted improvements.
"""

import copy
import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.backtest.analytics import compute_deflated_sharpe, generate_report

logger = logging.getLogger(__name__)

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
_TSV_PATH = _EXPERIMENTS_DIR / "quant_results.tsv"
_TSV_HEADER = "timestamp\trun_id\tparam_changed\tmetric_before\tmetric_after\tdelta\tstatus\tdsr\ttop5_mda\n"

# All available strategies (categorical param)
AVAILABLE_STRATEGIES = ["triple_barrier", "quality_momentum", "mean_reversion", "factor_model", "meta_label"]

# Strategy param bounds (min, max)
_PARAM_BOUNDS = {
    "tp_pct": (2.0, 30.0),
    "sl_pct": (2.0, 30.0),
    "holding_days": (30, 252),
    "frac_diff_d": (0.1, 0.8),
    "n_estimators": (50, 500),
    "max_depth": (2, 8),
    "min_samples_leaf": (5, 50),
    "learning_rate": (0.01, 0.3),
    "target_vol": (0.05, 0.30),
    "max_positions": (5, 40),
    "top_n_candidates": (20, 100),
    "momentum_weight": (0.0, 1.0),
    "rsi_weight": (0.0, 1.0),
    "volatility_weight": (0.0, 1.0),
    "sma_weight": (0.0, 1.0),
}

# Integer params (must be int after perturbation)
_INT_PARAMS = {"holding_days", "n_estimators", "max_depth", "min_samples_leaf", "max_positions", "top_n_candidates"}

# Categorical params (handled separately from numeric bounds)
_CATEGORICAL_PARAMS = {
    "strategy": AVAILABLE_STRATEGIES,
}


class QuantStrategyOptimizer:
    """
    Fast inner optimization loop for quant strategy parameters.
    Mirrors SkillOptimizer pattern: baseline → modify → measure → keep/discard → log.
    """

    def __init__(
        self,
        backtest_engine,
        status_callback: Optional[Callable] = None,
        dsr_threshold: float = 0.95,
    ):
        from backend.backtest.backtest_engine import BacktestEngine
        self.engine: BacktestEngine = backtest_engine
        self.status_callback = status_callback
        self.dsr_threshold = dsr_threshold

        self.best_params = self._get_current_params()
        self.best_sharpe = None
        self.best_dsr = None
        self.num_trials = 0
        self.kept = 0
        self.discarded = 0
        self._prev_top5_mda: list[str] = []  # Feature drift tracking

        _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def _get_current_params(self) -> dict:
        """Extract current strategy params from engine."""
        return copy.deepcopy(self.engine._strategy_params)

    def run_loop(
        self,
        max_iterations: int = 100,
        use_llm: bool = False,
        stop_check: Optional[Callable] = None,
        on_mda_update: Optional[Callable[[list[dict]], None]] = None,
    ):
        """
        Main autoresearch loop:
        1. Establish baseline
        2. For each iteration: propose → apply → evaluate → keep/discard

        Args:
            on_mda_update: Callback invoked with MDA importances after each kept experiment.
                Used by MetaCoordinator to update MDA→Agent bridge.
        """
        # 1. Baseline
        logger.info("QuantOptimizer: establishing baseline...")
        baseline_result = self.engine.run_backtest()
        baseline_report = generate_report(baseline_result, num_trials=1)
        self.best_sharpe = baseline_report["analytics"]["sharpe"]
        self.best_dsr = baseline_report["analytics"]["deflated_sharpe"]
        self.num_trials = 1

        # Extract baseline MDA top-5
        top5_mda = self._extract_top5_mda(baseline_result)
        self._prev_top5_mda = top5_mda

        self._log_experiment("BASELINE", "—", 0, self.best_sharpe, 0, "BASELINE", self.best_dsr, top5_mda)
        self._report_status()

        # 2. Iteration loop
        consecutive_discards = 0
        for i in range(max_iterations):
            if stop_check and stop_check():
                logger.info(f"QuantOptimizer: stopped after {i} iterations")
                break

            # Model staleness check (every 10 iterations)
            if i > 0 and i % 10 == 0:
                self._check_model_staleness()

            self.num_trials += 1

            # Propose modification (numeric or categorical)
            think_harder = consecutive_discards >= 5
            if use_llm:
                change = self._propose_llm(think_harder)
            else:
                change = self._propose_random(think_harder)

            param_name = change["param"]
            old_value = self.best_params.get(param_name, "?")
            new_value = change["value"]
            change_desc = f"{param_name}: {old_value} → {new_value}"

            # Apply modification
            trial_params = copy.deepcopy(self.best_params)
            trial_params[param_name] = new_value
            self._apply_params_to_engine(trial_params)

            # Evaluate
            try:
                result = self.engine.run_backtest()
                report = generate_report(result, num_trials=self.num_trials)
                trial_sharpe = report["analytics"]["sharpe"]
                trial_dsr = report["analytics"]["deflated_sharpe"]
                trial_top5 = self._extract_top5_mda(result)
            except Exception as e:
                logger.warning(f"QuantOptimizer: experiment crashed: {e}")
                self._apply_params_to_engine(self.best_params)
                self._log_experiment(
                    str(uuid.uuid4())[:8], change_desc,
                    self.best_sharpe, 0, -self.best_sharpe, "crash", 0, [],
                )
                consecutive_discards += 1
                continue

            delta = trial_sharpe - self.best_sharpe

            # Decision: keep / discard / dsr_reject
            if delta > 0 and trial_dsr >= self.dsr_threshold:
                status = "keep"
                self.best_params = trial_params
                self.best_sharpe = trial_sharpe
                self.best_dsr = trial_dsr
                self.kept += 1
                consecutive_discards = 0

                # Feature drift detection on keep
                self._detect_feature_drift(trial_top5)
                self._prev_top5_mda = trial_top5

                # Notify MetaCoordinator with fresh MDA importances
                if on_mda_update and result.feature_importance_mda:
                    mda_list = [
                        {"feature": k, "importance": v}
                        for k, v in sorted(
                            result.feature_importance_mda.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ]
                    on_mda_update(mda_list)

                logger.info(f"QuantOptimizer: KEEP {change_desc} (Sharpe {trial_sharpe:.4f}, DSR {trial_dsr:.4f})")
            elif delta > 0 and trial_dsr < self.dsr_threshold:
                status = "dsr_reject"
                self._apply_params_to_engine(self.best_params)
                self.discarded += 1
                consecutive_discards += 1
                trial_top5 = []
                logger.info(f"QuantOptimizer: DSR_REJECT {change_desc} (Sharpe ↑ but DSR {trial_dsr:.4f} < {self.dsr_threshold})")
            else:
                status = "discard"
                self._apply_params_to_engine(self.best_params)
                self.discarded += 1
                consecutive_discards += 1
                trial_top5 = []

            self._log_experiment(
                str(uuid.uuid4())[:8], change_desc,
                self.best_sharpe, trial_sharpe, delta, status, trial_dsr, trial_top5,
            )
            self._report_status()

        logger.info(
            f"QuantOptimizer: completed. Best Sharpe={self.best_sharpe:.4f}, "
            f"DSR={self.best_dsr:.4f}, kept={self.kept}, discarded={self.discarded}"
        )

    def export_best(self) -> dict:
        """Return best params + metrics + feature importance."""
        return {
            "params": self.best_params,
            "sharpe": self.best_sharpe,
            "dsr": self.best_dsr,
            "num_trials": self.num_trials,
            "kept": self.kept,
            "discarded": self.discarded,
        }

    # ── Proposal Strategies ──────────────────────────────────────

    def _propose_random(self, think_harder: bool = False) -> dict:
        """
        Zero-cost random perturbation.
        think_harder=True widens the perturbation range (±30% instead of ±10%).
        Handles both numeric (bounded) and categorical (strategy) params.
        """
        # 20% chance of proposing a strategy change
        all_params = list(_PARAM_BOUNDS.keys()) + list(_CATEGORICAL_PARAMS.keys())
        param = random.choice(all_params)

        # Categorical param (strategy)
        if param in _CATEGORICAL_PARAMS:
            choices = _CATEGORICAL_PARAMS[param]
            current = self.best_params.get(param, choices[0])
            # Pick a different value
            alternatives = [c for c in choices if c != current]
            new_value = random.choice(alternatives) if alternatives else current
            return {"param": param, "value": new_value}

        # Numeric param
        lo, hi = _PARAM_BOUNDS[param]
        current = self.best_params.get(param, (lo + hi) / 2)

        # Perturbation magnitude
        magnitude = 0.30 if think_harder else 0.15
        delta = current * random.uniform(-magnitude, magnitude)

        new_value = current + delta
        new_value = max(lo, min(hi, new_value))

        if param in _INT_PARAMS:
            new_value = int(round(new_value))

        return {"param": param, "value": new_value}

    def _propose_llm(self, think_harder: bool = False) -> dict:
        """
        LLM-guided proposal via Gemini Flash (~$0.01/call).
        Falls back to random if LLM is unavailable.
        """
        try:
            from backend.config.settings import get_settings
            from backend.agents.llm_client import make_client

            settings = get_settings()
            client = make_client(settings.gemini_model, None, settings)

            # Load recent experiment history
            history = self._load_recent_experiments(20)

            prompt = (
                "You are a quant strategy optimizer. Analyze the experiment history and propose "
                "ONE parameter change to improve the walk-forward backtest Sharpe ratio.\n\n"
                f"Current best params: {json.dumps(self.best_params, indent=2)}\n\n"
                f"Recent experiments:\n{history}\n\n"
                f"Parameter bounds: {json.dumps({k: list(v) for k, v in _PARAM_BOUNDS.items()})}\n\n"
                "Respond with ONLY a JSON object: {\"param\": \"<name>\", \"value\": <number>, \"rationale\": \"<why>\"}"
            )

            config = {"temperature": 0.9 if think_harder else 0.7, "max_output_tokens": 256}
            response = client.generate_content(prompt, config)

            # Parse LLM response
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            parsed = json.loads(text)

            param = parsed["param"]
            value = parsed["value"]

            # Validate and bound
            if param not in _PARAM_BOUNDS:
                raise ValueError(f"Unknown param: {param}")
            lo, hi = _PARAM_BOUNDS[param]
            value = max(lo, min(hi, value))
            if param in _INT_PARAMS:
                value = int(round(value))

            logger.info(f"QuantOptimizer LLM proposal: {param}={value} ({parsed.get('rationale', '')})")
            return {"param": param, "value": value}

        except Exception as e:
            logger.warning(f"LLM proposal failed, falling back to random: {e}")
            return self._propose_random(think_harder)

    # ── Engine Param Application ─────────────────────────────────

    def _apply_params_to_engine(self, params: dict):
        """Apply strategy params back to the engine."""
        engine = self.engine
        for key in ("holding_days", "tp_pct", "sl_pct", "frac_diff_d", "top_n_candidates"):
            if key in params:
                setattr(engine, key, params[key])

        for key in ("n_estimators", "max_depth", "min_samples_leaf", "learning_rate"):
            if key in params:
                engine.ml_params[key] = params[key]

        if "target_vol" in params:
            engine.trader.target_vol = params["target_vol"]
        if "max_positions" in params:
            engine.trader.max_positions = params["max_positions"]
        if "strategy" in params:
            engine.strategy = params["strategy"]

    # ── Logging ──────────────────────────────────────────────────

    def _log_experiment(
        self, run_id: str, change: str,
        metric_before: float, metric_after: float,
        delta: float, status: str, dsr: float,
        top5_mda: list[str] | None = None,
    ):
        """Append experiment to quant_results.tsv."""
        if not _TSV_PATH.exists():
            _TSV_PATH.write_text(_TSV_HEADER)

        mda_str = ",".join(top5_mda) if top5_mda else ""
        row = (
            f"{datetime.now(timezone.utc).isoformat()}\t{run_id}\t{change}\t"
            f"{metric_before:.4f}\t{metric_after:.4f}\t{delta:+.4f}\t{status}\t{dsr:.4f}\t{mda_str}\n"
        )
        with open(_TSV_PATH, "a") as f:
            f.write(row)

    def _load_recent_experiments(self, n: int = 20) -> str:
        """Load last N experiments as text for LLM context."""
        if not _TSV_PATH.exists():
            return "(no experiments yet)"
        lines = _TSV_PATH.read_text().strip().split("\n")
        if len(lines) <= 1:
            return "(no experiments yet)"
        header = lines[0]
        recent = lines[-n:] if len(lines) > n else lines[1:]
        return header + "\n" + "\n".join(recent)

    # ── Feature drift & model staleness ──────────────────────────

    def _extract_top5_mda(self, result) -> list[str]:
        """Extract top 5 features by MDA importance from a BacktestResult."""
        mda = getattr(result, "feature_importance_mda", None)
        if not mda:
            return []
        # mda is a dict[str, float] — sort descending by value
        sorted_features = sorted(mda.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in sorted_features[:5]]

    def _detect_feature_drift(self, new_top5: list[str]):
        """Log a WARNING if the top-5 MDA features changed vs previous."""
        if not self._prev_top5_mda or not new_top5:
            return
        old_set = set(self._prev_top5_mda)
        new_set = set(new_top5)
        if old_set != new_set:
            added = new_set - old_set
            removed = old_set - new_set
            logger.warning(
                "Feature drift detected — top-5 MDA changed: "
                "+%s / -%s", sorted(added), sorted(removed),
            )

    def _check_model_staleness(self):
        """Warn if backtest engine's trained model is >7 days old."""
        trained_at = getattr(self.engine, "model_trained_at", "")
        if not trained_at:
            return
        try:
            ts = datetime.fromisoformat(trained_at)
            age_days = (datetime.now(timezone.utc) - ts).days
            if age_days > 7:
                logger.warning(
                    "Model staleness: trained %d days ago (%s). "
                    "Consider retraining.", age_days, trained_at,
                )
        except (ValueError, TypeError):
            pass

    def _report_status(self):
        """Report progress via callback."""
        if self.status_callback:
            self.status_callback(
                self.num_trials, self.best_sharpe, self.best_dsr,
                self.kept, self.discarded,
            )
