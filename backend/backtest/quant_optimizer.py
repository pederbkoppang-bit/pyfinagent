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
import hashlib
import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.backtest.analytics import compute_deflated_sharpe, generate_report
from backend.backtest import cache as bq_cache

logger = logging.getLogger(__name__)

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
_TSV_PATH = _EXPERIMENTS_DIR / "quant_results.tsv"
_BEST_PARAMS_PATH = _EXPERIMENTS_DIR / "optimizer_best.json"
_TSV_HEADER = "timestamp\trun_id\tparam_changed\tmetric_before\tmetric_after\tdelta\tstatus\tdsr\ttop5_mda\tparams_json\tparent_run_id\n"

# All available strategies (categorical param)
AVAILABLE_STRATEGIES = ["triple_barrier", "quality_momentum", "mean_reversion", "factor_model", "meta_label", "blend"]

# Strategy param bounds (min, max)
_PARAM_BOUNDS = {
    "tp_pct": (2.0, 30.0),
    "sl_pct": (2.0, 30.0),
    "holding_days": (30, 252),
    "mr_holding_days": (5, 30),
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
    # Volatility-adjusted barriers (AFML Ch. 3): 0 = use fixed tp_pct/sl_pct,
    # >0 = barriers = daily_vol × multiplier. Typical range 1.0-5.0.
    "vol_barrier_multiplier": (0.0, 5.0),
    # Strategy blend weights (Dietterich 2000): active when strategy="blend"
    "tb_weight": (0.0, 1.0),
    "qm_weight": (0.0, 1.0),
    "mr_weight": (0.0, 1.0),
    "fm_weight": (0.0, 1.0),
    # Volatility targeting: scale positions to match target annual vol (0 = disabled)
    "target_annual_vol": (0.05, 0.25),  # ENABLED: Phase 1.5 improvement (+0.2 to +0.4 Sharpe)
    # Trailing stop: ENABLED Phase 1.5 improvement (+0.1 to +0.2 Sharpe)
    "trailing_trigger_pct": (2.0, 15.0),
    "trailing_distance_pct": (1.0, 10.0),
}

# Integer params (must be int after perturbation)
_INT_PARAMS = {"holding_days", "mr_holding_days", "n_estimators", "max_depth", "min_samples_leaf", "max_positions", "top_n_candidates"}

# Categorical params (handled separately from numeric bounds)
_CATEGORICAL_PARAMS = {
    "strategy": AVAILABLE_STRATEGIES,
    "trailing_stop_enabled": [True, False],  # ENABLED: Phase 1.5 improvement
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
        self.best_sharpe: float | None = None
        self.best_dsr: float | None = None
        self.num_trials = 0
        self._warm_started = False
        self._load_previous_best()  # Warm-start from disk if available
        self.kept = 0
        self.discarded = 0
        self._prev_top5_mda: list[str] = []  # Feature drift tracking
        self._run_id: str = ""  # Set in run_loop()
        self._current_step: str = ""  # Step-level progress
        self._current_detail: str = ""

        # Feature caching: reuse features when only ML hyperparams change
        self._feature_cache_key: str | None = None

        _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("QuantOptimizer: TSV path = %s", _TSV_PATH.resolve())

    def _get_current_params(self) -> dict:
        """Extract current strategy params from engine."""
        return copy.deepcopy(self.engine._strategy_params)

    def run_loop(
        self,
        max_iterations: int = 100,
        use_llm: bool = False,
        stop_check: Optional[Callable] = None,
        on_mda_update: Optional[Callable[[list[dict]], None]] = None,
        on_result: Optional[Callable[[dict], None]] = None,
    ):
        """
        Main autoresearch loop:
        1. Establish baseline
        2. For each iteration: propose → apply → evaluate → keep/discard

        Args:
            on_mda_update: Callback invoked with MDA importances after each kept experiment.
                Used by MetaCoordinator to update MDA→Agent bridge.
            on_result: Callback invoked with the report dict after baseline and each kept experiment.
                Used to populate backtest Results/Equity/Features tabs.
        """
        # Wire stop_check into engine so mid-backtest stops work
        self._stop_check = stop_check
        self.engine.stop_check = stop_check

        # Generate run_id to tag all experiments in this run
        self._run_id = str(uuid.uuid4())[:8]
        logger.info(f"QuantOptimizer: starting run {self._run_id}")

        # 1. Baseline (skip if warm-started from previous run)
        if self._warm_started:
            logger.info("QuantOptimizer: skipping baseline (warm-started Sharpe=%.4f)", self.best_sharpe)
            self._current_step = "baseline_complete"
            self._current_detail = f"Warm-start Sharpe={self.best_sharpe:.4f}"
            self._report_status()
            self._log_experiment(
                self._run_id, "warm-start", 0, float(self.best_sharpe or 0), 0,
                "BASELINE", float(self.best_dsr or 0), [],
            )
        else:
            # Check stop before starting baseline
            if stop_check and stop_check():
                logger.info("QuantOptimizer: stopped before baseline")
                return

            logger.info("QuantOptimizer: establishing baseline...")
            self._current_step = "establishing_baseline"
            self._current_detail = "Running full walk-forward backtest..."
            self._report_status()
            baseline_result = self.engine.run_backtest(skip_cache_clear=True)
            baseline_report = generate_report(baseline_result, num_trials=1)
            self.best_sharpe = baseline_report["analytics"]["sharpe"]
            self.best_dsr = baseline_report["analytics"]["deflated_sharpe"]
            self.num_trials = 1

            # Extract baseline MDA top-5
            top5_mda = self._extract_top5_mda(baseline_result)
            self._prev_top5_mda = top5_mda

            self._log_experiment(self._run_id, "--", 0, float(self.best_sharpe or 0), 0, "BASELINE", float(self.best_dsr or 0), top5_mda)
            if on_result:
                on_result(baseline_report)
            self._current_step = "baseline_complete"
            self._current_detail = f"Baseline Sharpe={self.best_sharpe:.4f}"
            self._report_status()

        # Initialize feature cache with current best params
        self._feature_cache_key = self._compute_feature_cache_key(self.best_params)
        self.engine.set_cached_features({})  # Activate caching (empty = will populate on first run)

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
            change_desc = f"{param_name}: {old_value} -> {new_value}"

            # Apply modification + Evaluate
            self._current_step = "running_experiment"
            self._current_detail = f"Experiment {i+1}: {change_desc}"
            self._report_status()
            try:
                trial_params = copy.deepcopy(self.best_params)
                trial_params[param_name] = new_value
                self._apply_params_to_engine(trial_params)

                # Feature cache: reuse if only ML params changed
                self._setup_feature_cache(trial_params)

                result = self.engine.run_backtest(
                    skip_cache_clear=True,
                )
                report = generate_report(result, num_trials=self.num_trials)
                trial_sharpe = report["analytics"]["sharpe"]
                trial_dsr = report["analytics"]["deflated_sharpe"]
                trial_top5 = self._extract_top5_mda(result)
            except Exception as e:
                logger.warning(f"QuantOptimizer: experiment {i+1} crashed ({change_desc}): {e}", exc_info=True)
                self._apply_params_to_engine(self.best_params)
                exp_id = f"{self._run_id}-exp{i+1:02d}"
                self._log_experiment(
                    exp_id, change_desc,
                    float(self.best_sharpe or 0), 0, -float(self.best_sharpe or 0), "crash", 0, [],
                    trial_params=trial_params,
                )
                self._current_detail = f"experiment {i+1} CRASHED: {change_desc} -- {e}"
                self._report_status()
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

                # Update feature cache key for the new best params
                self._feature_cache_key = self._compute_feature_cache_key(trial_params)

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
                logger.info(f"QuantOptimizer: DSR_REJECT {change_desc} (Sharpe improved but DSR {trial_dsr:.4f} < {self.dsr_threshold})")
            else:
                status = "discard"
                self._apply_params_to_engine(self.best_params)
                self.discarded += 1
                consecutive_discards += 1
                trial_top5 = []

            exp_id = f"{self._run_id}-exp{i+1:02d}"
            
            # Save JSON for ALL experiments (keep, discard, dsr_reject) so they're viewable
            report["run_id"] = exp_id
            report["parent_run_id"] = self._run_id
            report["experiment_status"] = status
            report["param_changed"] = change_desc
            if on_result:
                on_result(report)
            
            self._log_experiment(
                exp_id, change_desc,
                float(self.best_sharpe or 0), trial_sharpe, delta, status, trial_dsr, trial_top5,
                trial_params=trial_params,
            )
            self._current_step = "evaluated"
            self._current_detail = f"{status}: {change_desc} (Sharpe {trial_sharpe:.4f})"
            self._report_status()

        # Clean up caches after all iterations
        self.engine.clear_feature_cache()
        self._feature_cache_key = None
        bq_cache.clear_cache()

        # Persist best params to disk for next run warm-start
        self._save_best_params()

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
        # Build param list; exclude "strategy" if lock_strategy is set
        all_params = list(_PARAM_BOUNDS.keys()) + list(_CATEGORICAL_PARAMS.keys())
        if getattr(self, "lock_strategy", False):
            all_params = [p for p in all_params if p != "strategy"]
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
        Loads quant_strategy skill for research-backed guidance.
        Falls back to random if LLM is unavailable.
        """
        try:
            from backend.config.settings import get_settings
            from backend.agents.llm_client import make_client

            settings = get_settings()
            client = make_client(settings.gemini_model, None, settings)

            # Load recent experiment history
            history = self._load_recent_experiments(20)

            # Load research-backed strategy guide
            strategy_guide = ""
            try:
                guide_path = Path(__file__).parent.parent / "agents" / "skills" / "quant_strategy.md"
                if guide_path.exists():
                    strategy_guide = guide_path.read_text(encoding="utf-8")
            except Exception:
                pass

            prompt = (
                "You are a quant strategy optimizer. Analyze the experiment history and propose "
                "ONE parameter change to improve the walk-forward backtest Sharpe ratio.\n\n"
                f"Current best params: {json.dumps(self.best_params, indent=2)}\n\n"
                f"Recent experiments:\n{history}\n\n"
                f"Parameter bounds: {json.dumps({k: list(v) for k, v in _PARAM_BOUNDS.items()})}\n\n"
            )
            if strategy_guide:
                prompt += f"## Strategy Research Guide\n{strategy_guide}\n\n"
            prompt += "Respond with ONLY a JSON object: {\"param\": \"<name>\", \"value\": <number>, \"rationale\": \"<why>\"}"

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
        for key in ("holding_days", "mr_holding_days", "tp_pct", "sl_pct", "frac_diff_d", "top_n_candidates"):
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

        # Vol-adjusted barrier multiplier is read from _strategy_params dict
        # (not a direct engine attribute), so update it there
        if "vol_barrier_multiplier" in params:
            engine._strategy_params["vol_barrier_multiplier"] = params["vol_barrier_multiplier"]

        # Volatility targeting (read from _strategy_params by _compute_vol_target_scale)
        if "target_annual_vol" in params:
            engine._strategy_params["target_annual_vol"] = params["target_annual_vol"]

        # Trailing stop params (read from _strategy_params in daily MTM loop)
        for key in ("trailing_stop_enabled", "trailing_trigger_pct", "trailing_distance_pct"):
            if key in params:
                engine._strategy_params[key] = params[key]

        # Blend weights (read from _strategy_params by _compute_blend_label)
        for key in ("tb_weight", "qm_weight", "mr_weight", "fm_weight"):
            if key in params:
                engine._strategy_params[key] = params[key]

    # ── Feature caching ────────────────────────────────────────────

    # Params that affect feature matrix / labels -- changing these invalidates the cache.
    # Everything else (ML hyperparams, blend weights, screening weights) is safe to cache.
    _DATA_AFFECTING_PARAMS = frozenset({
        "tp_pct", "sl_pct", "holding_days", "mr_holding_days",
        "frac_diff_d", "top_n_candidates", "max_positions",
        "strategy", "target_annual_vol", "vol_barrier_multiplier",
    })

    @staticmethod
    def _compute_feature_cache_key(params: dict) -> str:
        """Hash only data-affecting params to determine if features can be reused."""
        key_data = {
            k: params.get(k)
            for k in QuantStrategyOptimizer._DATA_AFFECTING_PARAMS
        }
        raw = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _setup_feature_cache(self, params: dict):
        """Prepare engine feature cache if params allow reuse from previous run."""
        new_key = self._compute_feature_cache_key(params)
        if new_key == self._feature_cache_key:
            # Data params unchanged -- keep existing cache on engine
            logger.info("Feature cache: ML-only change detected, reusing cached features (key=%s)", new_key[:8])
        else:
            # Data params changed -- clear cache, engine will rebuild and populate
            logger.info("Feature cache: data params changed (key %s -> %s), rebuilding features",
                        (self._feature_cache_key or "none")[:8], new_key[:8])
            self.engine.set_cached_features({})  # Empty dict signals "cache active but empty"
            self._feature_cache_key = new_key

    # ── Logging ──────────────────────────────────────────────────

    def _log_experiment(
        self, run_id: str, change: str,
        metric_before: float, metric_after: float,
        delta: float, status: str, dsr: float,
        top5_mda: list[str] | None = None,
        trial_params: dict | None = None,
    ):
        """Append experiment to quant_results.tsv."""
        try:
            if not _TSV_PATH.exists():
                _TSV_PATH.write_text(_TSV_HEADER, encoding="utf-8")

            mda_str = ",".join(top5_mda) if top5_mda else ""
            # Serialize the TRIAL params (not best_params) so each row shows what was actually tested
            params_to_log = trial_params if trial_params is not None else self.best_params
            try:
                params_json = json.dumps(params_to_log, default=str)
            except (TypeError, ValueError):
                params_json = ""
            # parent_run_id: baselines have no parent; experiments link to their baseline's run_id
            parent = "" if status == "BASELINE" else self._run_id
            row = (
                f"{datetime.now(timezone.utc).isoformat()}\t{run_id}\t{change}\t"
                f"{metric_before:.4f}\t{metric_after:.4f}\t{delta:+.4f}\t{status}\t{dsr:.4f}\t{mda_str}\t{params_json}\t{parent}\n"
            )
            with open(_TSV_PATH, "a", encoding="utf-8") as f:
                f.write(row)
                f.flush()
            logger.debug("Logged experiment: run_id=%s status=%s change=%s", run_id, status, change)
        except Exception as e:
            logger.error("Failed to write experiment to TSV: %s (path=%s)", e, _TSV_PATH.resolve())

    def _load_recent_experiments(self, n: int = 20) -> str:
        """Load last N experiments as text for LLM context."""
        if not _TSV_PATH.exists():
            return "(no experiments yet)"
        lines = _TSV_PATH.read_text(encoding="utf-8").strip().split("\n")
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
                self._current_step, self._current_detail, self._run_id,
            )

    def _save_best_params(self):
        """Persist best_params + metrics to JSON for warm-start."""
        try:
            payload = {
                "params": self.best_params,
                "sharpe": self.best_sharpe,
                "dsr": self.best_dsr,
                "run_id": self._run_id,
                "kept": self.kept,
                "discarded": self.discarded,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            _BEST_PARAMS_PATH.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
            logger.info("Saved optimizer best params to %s", _BEST_PARAMS_PATH.name)
        except Exception as e:
            logger.warning("Failed to save best params: %s", e)

    def _load_previous_best(self):
        """Load previous best params from disk if available (warm-start).

        Sources checked in order:
        1. optimizer_best.json  -- written by optimizer at end of run_loop()
        2. result_store.load_latest() -- written by standalone backtests
        """
        # --- Source 1: optimizer's own best params file ---
        if _BEST_PARAMS_PATH.exists():
            try:
                data = json.loads(_BEST_PARAMS_PATH.read_text(encoding="utf-8"))
                prev_params = data.get("params", {})
                if prev_params:
                    for key in list(_PARAM_BOUNDS.keys()) + list(_CATEGORICAL_PARAMS.keys()):
                        if key in prev_params:
                            self.best_params[key] = prev_params[key]
                    self._apply_params_to_engine(self.best_params)
                    prev_sharpe = data.get("best_sharpe", data.get("sharpe"))
                    prev_dsr = data.get("best_dsr", data.get("dsr"))
                    if prev_sharpe is not None:
                        self.best_sharpe = float(prev_sharpe)
                        self.best_dsr = float(prev_dsr) if prev_dsr is not None else 0.0
                        self.num_trials = 1
                        self._warm_started = True
                    logger.info(
                        "Warm-started optimizer from optimizer_best.json (Sharpe=%.4f, run=%s)",
                        data.get("best_sharpe", data.get("sharpe", 0)), data.get("run_id", "?"),
                    )
                    return
            except Exception as e:
                logger.warning("Failed to load optimizer_best.json: %s", e)

        # --- Source 2: latest standalone backtest result ---
        try:
            from backend.backtest import result_store
            latest = result_store.load_latest()
            if latest is None:
                return
            sp = latest.get("strategy_params", {})
            analytics = latest.get("analytics", {})
            if not sp:
                return
            # Merge strategy_params into best_params (only optimizer-known keys)
            for key in list(_PARAM_BOUNDS.keys()) + list(_CATEGORICAL_PARAMS.keys()):
                if key in sp:
                    self.best_params[key] = sp[key]
            self._apply_params_to_engine(self.best_params)
            prev_sharpe = analytics.get("sharpe")
            prev_dsr = analytics.get("deflated_sharpe")
            if prev_sharpe is not None:
                self.best_sharpe = float(prev_sharpe)
                self.best_dsr = float(prev_dsr) if prev_dsr is not None else 0.0
                self.num_trials = 1
                self._warm_started = True
            logger.info(
                "Warm-started optimizer from standalone backtest (Sharpe=%.4f, run=%s)",
                prev_sharpe or 0, latest.get("run_id", "?"),
            )
        except Exception as e:
            logger.warning("Failed to load standalone backtest for warm-start: %s", e)
