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
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from backend.backtest.analytics import generate_report
from backend.backtest import cache as bq_cache

logger = logging.getLogger(__name__)

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"
_TSV_PATH = _EXPERIMENTS_DIR / "quant_results.tsv"
_BEST_PARAMS_PATH = _EXPERIMENTS_DIR / "optimizer_best.json"
_TSV_HEADER = "timestamp\trun_id\tparam_changed\tmetric_before\tmetric_after\tdelta\tstatus\tdsr\ttop5_mda\tparams_json\tparent_run_id\n"

# phase-25.D6: plateau-detection lock-file enforcement. Prevents the
# 62-experiment plateau bug (bucket 24.6 F-5) where the optimizer kept
# iterating despite no productive results. N=10 matches the Keras
# ReduceLROnPlateau default and Optax 5-15 range; second tier above the
# existing `think_harder >= 5` softer signal at this file's line ~205.
PLATEAU_THRESHOLD: int = 10
_PLATEAU_LOCK_PATH = (
    Path(__file__).parent.parent.parent / "handoff" / "locks" / "optimizer_plateau.lock"
)


def write_plateau_lock(run_id: str, consecutive_discards: int) -> None:
    """phase-25.D6: write the plateau lock-file. Subsequent
    `POST /api/backtest/optimize` calls will return 409 until an operator
    calls `DELETE /api/backtest/optimize/lock`. The lock is file-based so
    it survives backend restarts (in-memory counters reset on crash).
    """
    _PLATEAU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trigger": f"plateau_{consecutive_discards}_discards",
        "consecutive_discards": int(consecutive_discards),
        "run_id": str(run_id),
        "cleared_at": None,
    }
    _PLATEAU_LOCK_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.warning(
        "QuantOptimizer: plateau lock written after %d consecutive discards (run_id=%s). "
        "Operator must DELETE /api/backtest/optimize/lock to acknowledge and resume.",
        consecutive_discards, run_id,
    )

# All available strategies (categorical param).
# phase-82.16: DERIVED from STRATEGY_REGISTRY rather than restated. A hardcoded list
# would let the optimizer select a strategy that has been demoted out of the registry
# -- it would log and score a candidate that never actually ran.
#
# phase-82.46 RESOLVED both caveats 82.16 left open, and CORRECTED one of them.
#
# THE CORRECTION, twice over -- and the second correction is mine, forced by the
# 82.46 Q/A. The 82.16 comment here used to say "the trial pool is a direct input
# to DSR deflation". That is wrong in its literal form: `compute_deflated_sharpe`
# has NO pool parameter, and `num_trials` is `self.num_trials += 1` PER ITERATION.
# Both primaries agree that a trial is a configuration ACTUALLY TRIED, so a longer
# menu costs nothing unless sampled.
#
# BUT DO NOT CONCLUDE "the pool cannot affect DSR". I did, and it is an
# over-generalisation from a SIGNATURE to a BEHAVIOUR. At the real call site
# (`analytics.generate_report`) DSR is fed `observed_sr=result.aggregate_sharpe`
# and `variance_of_srs=np.var(window_sharpes)` -- BOTH of which depend on WHICH
# strategy was sampled. So the pool does move DSR, through its inputs rather
# than through a trial-count term.
#
# WHAT IS TRUE, and it is sharper: N is steep (DSR 1.0000 at N<=10, 0.7963 at
# N=26, 0.1164 at N=50, 0.0008 at N=100 for a fixed SR=1.5), and the N increment
# happens BEFORE the try/except that wraps run_backtest -- so an experiment that
# CRASHES still costs a trial. The pool therefore affects DSR through WASTED
# ITERATIONS, not through its size. That is why the two removals below are worth
# making and why enlarging the pool with runnable strategies is not a concern.
#
# THE DECISION (phase-82.46; rationale per member in POOL_DECISION below):
#   * "blend" is REMOVED. It is not a registry key, has no implementation, and
#     resolved to triple_barrier while being SCORED under the requested name --
#     corrupting attribution and burning a trial. Re-implementing it was
#     rejected: the deleted _compute_blend_label voted over quality_momentum and
#     factor_model, which 82.16 demoted for carrying no forward information.
#   * The registry-derived members stay. The membership RULE is what is pinned,
#     not the list, so adding a strategy to the registry adds it here on purpose.
from backend.backtest.backtest_engine import (
    NON_COMPARABLE_STRATEGIES as _NON_COMPARABLE,
    STRATEGY_REGISTRY as _STRATEGY_REGISTRY,
)

#: The membership rule, executable. A strategy is selectable iff it is a
#: registry key AND has not been demoted. Derived so it cannot drift from the
#: registry the way the pre-82.16 hand-written literal did.
def _selectable_strategies() -> list[str]:
    return [s for s in _STRATEGY_REGISTRY if s not in _NON_COMPARABLE]


AVAILABLE_STRATEGIES = _selectable_strategies()

#: phase-82.46 criterion 4: the decision record. Machine-readable on purpose --
#: a guard compares its keys against AVAILABLE_STRATEGIES, so adding or removing
#: a pool member without recording WHY fails the suite. A test that merely
#: restated the list would be a tautology; this one requires a rationale to
#: exist for every member and forbids a rationale for a non-member.
POOL_DECISION: dict[str, str] = {
    "triple_barrier": (
        "incumbent; forward-looking (Lopez de Prado Ch.3), trains and trades on "
        "the configured sample"
    ),
    "mean_reversion": (
        "forward-looking and CAN train; performs badly on this sample. Kept: "
        "poor performance is what the optimizer is for, and excluding a "
        "runnable strategy costs nothing under the corrected DSR premise above"
    ),
    "meta_label": (
        "shares triple_barrier's label fn, so its PBO column is near-collinear "
        "with the incumbent's -- noted as a comparability caveat, not a reason "
        "to exclude a runnable member"
    ),
    "stretch_regime": (
        "phase-82.2 candidate; forward-looking, price/vol only (no fundamentals "
        "dependency per the 82.21 derivation), and the strongest measured "
        "PBO on the full sample"
    ),
    "qarp": (
        "phase-82.2 candidate; LABEL-fundamentals-dependent per 82.21, so it "
        "cannot be labelled on a window starting before the measured "
        "fundamentals coverage start and the engine now REFUSES there. Kept in "
        "the pool because the constraint is about the WINDOW, not the strategy; "
        "see selectable_strategies_for_window()"
    ),
    "reversion_sigma": (
        "phase-82.2 candidate; forward-looking, sigma-scaled, price-only"
    ),
}


def selectable_strategies_for_window(
    window_start: str, dependent_fn=None
) -> list[str]:
    """phase-82.46: the pool MINUS members that cannot run on this window.

    82.21 made the engine RAISE for a label-fundamentals-dependent strategy when
    the window starts before the measured fundamentals coverage start. The
    optimizer catches that (the experiment is logged as a crash) -- but the
    trial counter has ALREADY incremented, so an unrunnable member silently
    costs Deflated Sharpe on every iteration that selects it.

    The exclusion is DERIVED from the same 82.21 predicate the engine uses, not
    a hardcoded {"qarp"}: a name list would go stale the moment another
    fundamentals-dependent strategy is registered, and a stale exclusion is
    worse than none because it reads as coverage.
    """
    from backend.backtest.fundamentals_coverage import (
        label_fundamentals_dependent_strategies,
        window_is_covered,
    )

    pool = _selectable_strategies()
    if window_is_covered(window_start):
        return pool
    # `dependent_fn` is injectable ONLY so a guard can drive this derivation with
    # a synthetic answer. Without that seam a mutant hardcoding {"qarp"} survives
    # -- qarp happens to BE today's answer, so no assertion over live data can
    # tell a derivation from a literal.
    dependent = set((dependent_fn or label_fundamentals_dependent_strategies)())
    return [s for s in pool if s not in dependent]

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
    # phase-82.46: the four blend-weight params (tb_/qm_/mr_/fm_weight) were
    # REMOVED. `_compute_blend_label`, their only consumer, was deleted by
    # 9fbd9cd6 -- a revert whose diff never touched this file -- so the setter
    # below wrote them into engine._strategy_params where NOTHING read them.
    # rotation_runner._DEAD_KEYS already documented them as dead. They were 4 of
    # 24 proposable params, so roughly 1 proposal in 6 spent a full walk-forward
    # run AND a DSR-costing num_trials increment on a parameter with no reader.
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

    def _window_selectable_strategies(self) -> list[str]:
        """phase-82.46: the pool minus members that cannot run on THIS window.

        Reads the engine's configured start date rather than a constant, and
        fails OPEN (returns the full pool) if the window cannot be determined --
        narrowing the search space is an optimisation, and silently emptying it
        would be far worse than an occasional wasted trial.
        """
        try:
            start = self.engine.scheduler.start_date.isoformat()
        except Exception:  # noqa: BLE001 -- never break proposal on introspection
            return list(AVAILABLE_STRATEGIES)
        try:
            return selectable_strategies_for_window(start)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "QuantOptimizer: window-aware strategy filter fail-open: %r", exc)
            return list(AVAILABLE_STRATEGIES)

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
        # phase-82.22: provenance of any warm-started metrics. Set by
        # _load_previous_best; None means "these metrics are this run's own".
        self._warm_started_from_run_id: str | None = None
        self._warm_started_from_artifact: str | None = None
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

        # 2. Iteration loop (max_iterations=0 means run forever until stopped)
        consecutive_discards = 0
        i = 0
        while max_iterations == 0 or i < max_iterations:
            i += 1
            if stop_check and stop_check():
                logger.info(f"QuantOptimizer: stopped after {i-1} iterations")
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
            self._current_detail = f"Experiment {i}: {change_desc}"
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
                logger.warning(f"QuantOptimizer: experiment {i} crashed ({change_desc}): {e}", exc_info=True)
                self._apply_params_to_engine(self.best_params)
                exp_id = f"{self._run_id}-exp{i:02d}"
                self._log_experiment(
                    exp_id, change_desc,
                    float(self.best_sharpe or 0), 0, -float(self.best_sharpe or 0), "crash", 0, [],
                    trial_params=trial_params,
                )
                self._current_detail = f"experiment {i} CRASHED: {change_desc} -- {e}"
                self._report_status()
                consecutive_discards += 1
                # phase-25.D6: a streak of crashes is itself a plateau signal.
                # Check the same threshold here so the crash branch can't
                # bypass the lock-file fence.
                if consecutive_discards >= PLATEAU_THRESHOLD:
                    write_plateau_lock(self._run_id, consecutive_discards)
                    self._current_step = "plateau_locked"
                    self._current_detail = (
                        f"plateau detected after {consecutive_discards} consecutive "
                        f"discards (last was a crash); lock written"
                    )
                    self._report_status()
                    break
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

                # Git commit kept experiment (like Karpathy's autoresearch branch advance)
                self._git_commit_kept(f"{self._run_id}-exp{i:02d}", change_desc, trial_sharpe, trial_dsr)
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

            exp_id = f"{self._run_id}-exp{i:02d}"
            
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

            # phase-25.D6: plateau-detection lock. After PLATEAU_THRESHOLD
            # consecutive discards/dsr_rejects/crashes, halt the loop and
            # write a lock-file. Operators must DELETE /api/backtest/
            # optimize/lock to acknowledge the plateau and resume. Closes
            # phase-24.6 F-5 (62-experiment plateau bypassed planner Rule 1).
            if consecutive_discards >= PLATEAU_THRESHOLD:
                write_plateau_lock(self._run_id, consecutive_discards)
                self._current_step = "plateau_locked"
                self._current_detail = (
                    f"plateau detected after {consecutive_discards} consecutive discards; "
                    f"lock written, operator action required"
                )
                self._report_status()
                break

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
            # phase-82.46: for the STRATEGY dimension, narrow the choices to what
            # can actually run on THIS optimizer's window.
            #
            # This wiring is the whole point and it was missing in cycle 1: the
            # filter existed but nothing called it, so the artifact and the
            # function's own docstring described a mitigation that never ran.
            # Caught by the 82.46 Q/A. Without it, 82.21 makes the engine RAISE
            # for a label-fundamentals-dependent strategy on a pre-coverage
            # window, the optimizer catches the crash -- and `num_trials` has
            # ALREADY incremented, so the wasted trial still deflates DSR.
            if param == "strategy":
                choices = self._window_selectable_strategies() or choices
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
            from backend.agents.llm_client import GeminiModelBundle, make_client

            settings = get_settings()

            # phase-26.3: pair the optimizer's LLM proposal with `code_execution`
            # so the model can verify proposed parameter bounds (`2.0 <= tp_pct
            # <= 30.0`), risk-reward ratios (`tp_pct / sl_pct >= 1.5`), and
            # vol-adjusted barrier arithmetic INSIDE the call rather than
            # producing numerically inconsistent proposals. The verification
            # is enacted via the `## Code Execution Tasks` section in
            # `backend/agents/skills/quant_strategy.md`. Bundle is constructed
            # inline (the optimizer is the only caller; no need to spin a
            # module-level shared bundle).
            _bundle = None
            try:
                from google.genai import types as _genai_types
                from google import genai as _genai_module
                _client_obj = _genai_module.Client(
                    vertexai=True,
                    project=settings.gcp_project_id,
                    location=getattr(settings, "vertex_location", "us-central1"),
                )
                _bundle = GeminiModelBundle(
                    client=_client_obj,
                    model_name=settings.gemini_model,
                    tools=[_genai_types.Tool(code_execution=_genai_types.ToolCodeExecution())],
                    base_config={},
                )
            except Exception as _b_exc:
                logger.debug(f"QuantOptimizer code_execution bundle init skipped: {_b_exc}")
                _bundle = None

            client = make_client(settings.gemini_model, _bundle, settings)

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

        # phase-82.46: the blend-weight setter was removed with the params. The
        # comment here used to claim they were "read from _strategy_params by
        # _compute_blend_label" -- that function has not existed since 9fbd9cd6.

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
                "Feature drift detected -- top-5 MDA changed: "
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

    def _git_commit_kept(self, exp_id: str, change_desc: str, sharpe: float, dsr: float):
        """Git commit after a kept experiment (like Karpathy's branch advance)."""
        try:
            import subprocess
            project_root = Path(__file__).parent.parent.parent
            # Stage optimizer_best.json and quant_results.tsv
            subprocess.run(
                ["git", "add",
                 "backend/backtest/experiments/optimizer_best.json",
                 "backend/backtest/experiments/quant_results.tsv"],
                cwd=project_root, capture_output=True, timeout=10,
            )
            msg = f"Optimizer KEEP {exp_id}: {change_desc} (Sharpe {sharpe:.4f}, DSR {dsr:.4f})"
            subprocess.run(
                ["git", "commit", "-m", msg, "--no-verify"],
                cwd=project_root, capture_output=True, timeout=10,
            )
            logger.info(f"Git committed: {msg[:80]}")
        except Exception as e:
            logger.debug(f"Git commit skipped: {e}")

    def _save_best_params(self):
        """Persist best_params + metrics to JSON for warm-start.

        phase-82.22 PROVENANCE. This used to write `"run_id": self._run_id`
        beside `self.best_sharpe` / `self.best_dsr` unconditionally -- but those
        metrics may have been INHERITED from an earlier run by
        `_load_previous_best`, and nothing recorded where they came from. When a
        run beat nothing (`kept == 0`), the previous run's numbers were
        re-stamped with the current run's identity.

        That is not hypothetical: `optimizer_best.json` carried
        `run_id=60617e0b, sharpe=1.1704633657934074, dsr=0.9525811126193078,
        kept=0` while run 60617e0b's own ten artifacts produced sharpe
        0.5384..0.6506 -- six of them at exactly 0.6455483636. The persisted
        pair belongs to `52eb3ffe-exp10`, four months earlier.

        `run_id` keeps its original meaning (the run that WROTE the file) so no
        consumer changes. `metrics_run_id` says which run actually PRODUCED the
        metrics. Every added key is additive: all 15 readers are `dict.get`
        based.
        """
        try:
            # Which run actually produced the persisted metrics? If this run
            # never improved on the warm-started value, they are not ours.
            metrics_run_id = self._run_id
            metrics_source = None
            if getattr(self, "_warm_started", False) and self.kept == 0:
                metrics_run_id = getattr(self, "_warm_started_from_run_id", None)
                metrics_source = getattr(self, "_warm_started_from_artifact", None)

            payload = {
                "params": self.best_params,
                "sharpe": self.best_sharpe,
                "dsr": self.best_dsr,
                "run_id": self._run_id,
                "kept": self.kept,
                "discarded": self.discarded,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                # ── phase-82.22 provenance (additive) ──────────────────
                "schema_version": 2,
                # The run whose experiment actually produced sharpe/dsr above.
                # None => unknown provenance. ABSENCE MUST NOT READ AS FRESH:
                # a consumer seeing no `metrics_run_id` must treat the metrics
                # as unattributed, never as self-produced by `run_id`.
                "metrics_run_id": metrics_run_id,
                "metrics_source_artifact": metrics_source,
                "warm_started_from": getattr(self, "_warm_started_from_run_id", None),
                # Trials searched, for DSR deflation. Reset-on-warm-start is a
                # separate defect (step 82.25); recorded here so that fix has an
                # input to carry forward.
                "num_trials": getattr(self, "num_trials", None),
                # phase-82.25: whether the count above is a full cumulative depth or
                # only this session's. False means the DSR is an UPPER BOUND.
                "prior_trials_known": getattr(self, "prior_trials_known", None),
            }
            _BEST_PARAMS_PATH.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
            logger.info("Saved optimizer best params to %s", _BEST_PARAMS_PATH.name)
        except Exception as e:
            logger.warning("Failed to save best params: %s", e)

    # phase-82.25: the FLOOR used when a warm-start source records no trial depth.
    # Deliberately NOT 0: see the inline note in _resolve_prior_trials -- 0 makes the
    # DSR gate easier than the defect did. Deliberately not large either: inventing a
    # depth would be fabrication.
    _UNKNOWN_PRIOR_FLOOR = 1

    def _resolve_prior_trials(self, source: dict) -> None:
        """phase-82.25: set the trial counter from a warm-start source, HONESTLY.

        Both warm-start paths used to do `self.num_trials = 1`. DSR is deflated BY the
        number of trials searched, and N is scoped to the RESEARCH PROCESS that produced
        the discovery, not to one session (Bailey & Lopez de Prado 2014; Lopez de Prado
        & Lewis 2018). So resetting to 1 reported a carried-forward DSR as though the
        strategy had been found on the first attempt. The effect is not marginal: in run
        60617e0b, exp01 and exp10 share Sharpe 0.6455483635957818 and differ 72x in DSR
        on trial count alone.

        THE UNKNOWN-PRIOR DECISION (this is a decision, not a default). Today's live
        optimizer_best.json is schema v1 and carries no `num_trials` -- 82.22 changed
        only the writer -- so the unknown branch is the PRODUCTION path. We do not
        fabricate a number:
          * assuming 1 is the single most OPTIMISTIC assumption available, and is
            exactly the defect being fixed;
          * inventing a large number would be fabrication -- the true depth is
            unrecorded;
          * DSR Appendix 3 shows M >= N overstates E[max SR] and LOWERS the DSR, so
            erring HIGH is safe and erring LOW is dangerous.
        Therefore an unrecorded prior is marked UNKNOWN and the resulting DSR is an
        UPPER BOUND (under-deflated). `prior_trials_known` is persisted so the next
        warm start inherits the honesty flag instead of laundering unknown into known.

        NOTE the hard boundary: this changes only FUTURE deflation. A persisted `dsr`
        is never recomputed -- the live file clears the 0.95 go-live gate by 0.0026, and
        re-deflating an inherited number computed at an unrecorded N would be
        fabrication as well as a gate closure.
        """
        # The two warm-start sources nest the count differently: optimizer_best.json
        # has it at the top level (written by _save_best_params), while a result_store
        # report carries it inside its "analytics" block. Reading only the top level
        # would silently miss the result_store path -- i.e. keep half the defect.
        prior = source.get("num_trials")
        if prior is None:
            prior = (source.get("analytics") or {}).get("num_trials")
        # phase-82.25 cycle 2 (F1): READ the honesty flag, do not just write it. The
        # first version persisted `prior_trials_known` and never read it back, so a
        # source whose own depth was unknown warm-started as KNOWN one generation
        # later -- reproducing, one field over, the exact write-without-read root
        # cause this step exists to fix. Unknown is STICKY.
        source_known = source.get("prior_trials_known")
        if isinstance(prior, int) and not isinstance(prior, bool) and prior > 0:
            self.num_trials = prior
            self.prior_trials_known = source_known is not False
            logger.info(
                "warm start: carrying cumulative trial count %d forward for DSR "
                "deflation", prior,
            )
        else:
            # phase-82.25 cycle 2 -- I HAD THIS BACKWARDS. The first version set 0
            # here, and because `self.num_trials += 1` runs BEFORE
            # `generate_report(..., num_trials=self.num_trials)`, session trial k
            # then reported N=k where the DEFECT reported N=k+1. So on the only
            # currently reachable path (the live file is schema v1) the "fix"
            # deflated LESS than the bug it replaced and made the 0.95 KEEP gate
            # EASIER -- measured at Sharpe 0.6455483635957818, k=2 gave DSR 0.999970
            # post-"fix" vs 0.730465 pre-fix, i.e. it KEEPS what the defect
            # DISCARDED. That inverts this step's whole purpose and contradicts the
            # "erring HIGH is safe" principle stated below.
            #
            # _UNKNOWN_PRIOR_FLOOR is a FLOOR, not an estimate. The warm-start source
            # exists only because a prior run produced it, so at least one prior trial
            # is certain; and the floor guarantees this fix can never deflate less than
            # the behaviour it replaces. The true depth stays unknown -- that is what
            # prior_trials_known records, and why the DSR is labelled an upper bound.
            self.num_trials = self._UNKNOWN_PRIOR_FLOOR
            self.prior_trials_known = False
            logger.warning(
                "warm start: the source records NO trial count, so the prior search "
                "depth is UNKNOWN. Using the floor of %d -- the minimum defensible "
                "value, never below what the pre-fix code used. The DSR reported from "
                "this run is an UPPER BOUND: it is under-deflated by however deep the "
                "unrecorded prior search actually was.",
                self._UNKNOWN_PRIOR_FLOOR,
            )

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
                        self._resolve_prior_trials(data)
                        self._warm_started = True
                        # phase-82.22: remember WHOSE metrics these are. Prefer
                        # the file's own metrics_run_id (schema v2) over its
                        # run_id -- if the source was itself warm-started, its
                        # run_id is the writer, not the producer, and copying
                        # that forward would propagate the mis-attribution one
                        # generation further.
                        self._warm_started_from_run_id = (
                            data.get("metrics_run_id") or data.get("run_id")
                        )
                        self._warm_started_from_artifact = data.get(
                            "metrics_source_artifact"
                        ) or "optimizer_best.json"
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
                self._resolve_prior_trials(latest)
                self._warm_started = True
                # phase-82.22: same provenance capture for the standalone path.
                self._warm_started_from_run_id = latest.get("run_id")
                self._warm_started_from_artifact = latest.get("_source_file") or "result_store.load_latest()"
            logger.info(
                "Warm-started optimizer from standalone backtest (Sharpe=%.4f, run=%s)",
                prev_sharpe or 0, latest.get("run_id", "?"),
            )
        except Exception as e:
            logger.warning("Failed to load standalone backtest for warm-start: %s", e)
