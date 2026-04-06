"""
Evaluator Spot Checks — Robustness Validation for Trading Strategies

Phase 3.2.1 GENERATE: Three critical spot checks for evaluator validation
1. Cost Stress Test (2× transaction costs)
2. Regime Shift Detection (walk-forward across regime boundaries)
3. Parameter Sweep (sensitivity analysis for overfitting detection)

Research-backed thresholds from Roncelli (2020), Two Sigma (2021), BuildAlpha
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SpotCheckResult:
    """Container for a single spot check result."""
    check_type: str  # "cost_stress", "regime_shift", "param_sweep"
    passed: bool
    baseline_sharpe: float
    test_sharpe: float
    threshold: float
    reasoning: str
    metrics: Dict  # Additional metrics (dsr, return, maxdd, etc.)
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class SpotChecksAggregated:
    """Aggregated result from all spot checks."""
    overall_pass: bool
    cost_stress_pass: bool
    regime_shift_pass: bool
    param_sweep_pass: bool
    baseline_sharpe: float
    reasoning: str
    cost_stress_result: SpotCheckResult = None
    regime_shift_result: SpotCheckResult = None
    param_sweep_result: SpotCheckResult = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class CostStressTest:
    """
    Test #1: Cost Stress (2× Transaction Costs)
    
    Verifies that the strategy's performance degrades gracefully under doubled costs.
    Roncelli (2020): Robustness coefficient ≥ 0.9
    """
    
    NAME = "cost_stress"
    COST_MULTIPLIER = 2.0
    MIN_ROBUSTNESS = 0.9  # Strategy must retain ≥90% of baseline Sharpe
    
    def __init__(self, run_backtest_fn):
        """
        Args:
            run_backtest_fn: Callable that accepts (params_dict, tx_cost_pct=None) and returns analytics dict
        """
        self.run_backtest_fn = run_backtest_fn
    
    def run(self, proposal_params: Dict) -> SpotCheckResult:
        """
        Run cost stress test on proposal parameters.
        
        Args:
            proposal_params: Original parameters to test
            
        Returns:
            SpotCheckResult with pass/fail and metrics
        """
        logger.info("Starting Cost Stress Test (2× transaction costs)...")
        
        # Step 1: Baseline harness
        baseline_result = self.run_backtest_fn(proposal_params, tx_cost_pct=None)
        baseline_sharpe = baseline_result.get('sharpe', 0.0)
        logger.info(f"Baseline Sharpe: {baseline_sharpe:.4f}")
        
        # Step 2: 2× cost harness
        # Default tx_cost_pct is 0.1 (10 bps), so 2× = 0.2 (20 bps)
        cost_tx_pct = 0.2  # 2× baseline
        
        cost_result = self.run_backtest_fn(proposal_params, tx_cost_pct=cost_tx_pct)
        cost_sharpe = cost_result.get('sharpe', 0.0)
        logger.info(f"2× Cost Sharpe: {cost_sharpe:.4f}")
        
        # Step 3: Check threshold
        threshold_sharpe = self.MIN_ROBUSTNESS * baseline_sharpe
        passed = cost_sharpe >= threshold_sharpe
        
        reasoning = (
            f"Cost Stress: Baseline Sharpe={baseline_sharpe:.4f}, "
            f"2× Cost Sharpe={cost_sharpe:.4f}, "
            f"Threshold={threshold_sharpe:.4f} (90% baseline), "
            f"Result={'PASS' if passed else 'FAIL'}"
        )
        logger.info(reasoning)
        
        return SpotCheckResult(
            check_type=self.NAME,
            passed=passed,
            baseline_sharpe=baseline_sharpe,
            test_sharpe=cost_sharpe,
            threshold=threshold_sharpe,
            reasoning=reasoning,
            metrics={
                'dsr': cost_result.get('dsr', 0.0),
                'return': cost_result.get('return', 0.0),
                'maxdd': cost_result.get('maxdd', 0.0),
                'num_trades': cost_result.get('num_trades', 0),
            }
        )


class RegimeShiftTest:
    """
    Test #2: Regime Shift Detection
    
    Verifies that strategy survives across market regime boundaries.
    Two Sigma (2021): Strategies fail at regime transitions.
    """
    
    NAME = "regime_shift"
    MIN_ROBUSTNESS_REGIMES = 0.8  # Strategy must retain ≥80% in each regime
    
    def __init__(self, run_backtest_fn, regime_detector=None):
        """
        Args:
            run_backtest_fn: Callable that accepts (params_dict, tx_cost_pct=None, start_date=None, end_date=None)
            regime_detector: Instance of RegimeDetector (HMM-based), optional
        """
        self.run_backtest_fn = run_backtest_fn
        self.regime_detector = regime_detector
    
    def run(self, proposal_params: Dict) -> SpotCheckResult:
        """
        Run regime shift test: walk-forward across regime boundaries.
        
        Args:
            proposal_params: Original parameters to test
            
        Returns:
            SpotCheckResult with pass/fail
        """
        logger.info("Starting Regime Shift Test...")
        
        # Step 1: Detect regimes (if detector available)
        if self.regime_detector:
            regimes = self.regime_detector.detect()
            logger.info(f"Detected {len(regimes)} market regimes")
        else:
            # Fallback: 2-regime split (before/after COVID crash March 2020)
            logger.warning("No regime detector available, using fallback 2-regime split")
            regimes = [
                {'name': 'Pre-COVID', 'start_date': '2018-01-01', 'end_date': '2020-03-15'},
                {'name': 'Post-COVID', 'start_date': '2020-03-16', 'end_date': '2025-12-31'},
            ]
        
        # Step 2: Run harness for each regime (walk-forward)
        regime_results = []
        baseline_sharpe = None
        min_regime_sharpe = float('inf')
        
        for i, regime in enumerate(regimes):
            logger.info(f"Testing regime {i+1}: {regime.get('name', f'Regime {i+1}')}")
            
            start_date = regime.get('start_date')
            end_date = regime.get('end_date')
            
            result = self.run_backtest_fn(proposal_params, tx_cost_pct=None, start_date=start_date, end_date=end_date)
            sharpe = result.get('sharpe', 0.0)
            
            if baseline_sharpe is None:
                baseline_sharpe = sharpe
            
            min_regime_sharpe = min(min_regime_sharpe, sharpe)
            regime_results.append({
                'regime': regime.get('name', f'Regime {i+1}'),
                'sharpe': sharpe,
                'dsr': result.get('dsr', 0.0),
            })
            logger.info(f"  Regime Sharpe: {sharpe:.4f}")
        
        # Step 3: Check threshold
        threshold_sharpe = self.MIN_ROBUSTNESS_REGIMES * baseline_sharpe
        passed = min_regime_sharpe >= threshold_sharpe
        
        reasoning = (
            f"Regime Shift: Baseline Sharpe={baseline_sharpe:.4f}, "
            f"Min Regime Sharpe={min_regime_sharpe:.4f}, "
            f"Threshold={threshold_sharpe:.4f} (80% baseline), "
            f"Result={'PASS' if passed else 'FAIL'}"
        )
        logger.info(reasoning)
        
        return SpotCheckResult(
            check_type=self.NAME,
            passed=passed,
            baseline_sharpe=baseline_sharpe,
            test_sharpe=min_regime_sharpe,
            threshold=threshold_sharpe,
            reasoning=reasoning,
            metrics={'regime_results': regime_results}
        )


class ParamSweepTest:
    """
    Test #3: Parameter Sweep (Sensitivity Analysis)
    
    Verifies that strategy parameters are robust, not overfitted.
    BuildAlpha: σ(Sharpe) ≤ 5% indicates robust params; σ > 10% indicates severe overfitting
    """
    
    NAME = "param_sweep"
    MAX_SIGMA_PCT = 5.0  # Success: σ ≤ 5% of mean Sharpe
    WARNING_SIGMA_PCT = 10.0  # Warning: σ > 10% indicates overfitting
    N_COMBOS = 10  # Number of random parameter combinations to test
    
    def __init__(self, run_backtest_fn):
        """
        Args:
            run_backtest_fn: Callable that accepts (params_dict, tx_cost_pct=None)
        """
        self.run_backtest_fn = run_backtest_fn
    
    @staticmethod
    def _perturb_param(value: float, pct_range: float) -> float:
        """Perturb a parameter by a random amount within ±pct_range%."""
        offset = np.random.uniform(-pct_range, pct_range) / 100.0
        return value * (1.0 + offset)
    
    def generate_nearby_params(self, proposal_params: Dict, n: int = 10) -> List[Dict]:
        """
        Generate n random parameter combinations near the baseline.
        
        Args:
            proposal_params: Baseline parameters
            n: Number of combos to generate
            
        Returns:
            List of n perturbed parameter dicts
        """
        combos = []
        
        # Parameters to perturb and their ranges (±%)
        perturb_params = {
            'sl_pct': 10.0,       # Stop loss: ±10%
            'tp_pct': 10.0,       # Take profit: ±10%
            'holding_days': 20.0, # Holding period: ±20%
            'max_depth': 1,       # Tree depth: ±1 (special handling)
            'learning_rate': 20.0, # LR: ±20%
        }
        
        for _ in range(n):
            combo = proposal_params.copy()
            
            for param_name, pct_range in perturb_params.items():
                if param_name not in combo:
                    continue
                
                if param_name == 'max_depth':
                    # Special: integer, ±1
                    combo[param_name] = max(1, combo[param_name] + np.random.randint(-1, 2))
                else:
                    # Float: perturb by ±pct_range%
                    combo[param_name] = self._perturb_param(combo[param_name], pct_range)
            
            combos.append(combo)
        
        return combos
    
    def run(self, proposal_params: Dict) -> SpotCheckResult:
        """
        Run parameter sweep test: test nearby param combos, measure Sharpe variance.
        
        Args:
            proposal_params: Baseline parameters
            
        Returns:
            SpotCheckResult with pass/fail and variance metrics
        """
        logger.info(f"Starting Parameter Sweep Test ({self.N_COMBOS} combos)...")
        
        # Step 1: Run baseline
        baseline_result = self.run_backtest_fn(proposal_params, tx_cost_pct=None)
        baseline_sharpe = baseline_result.get('sharpe', 0.0)
        logger.info(f"Baseline Sharpe: {baseline_sharpe:.4f}")
        
        # Step 2: Generate nearby param combos
        nearby_combos = self.generate_nearby_params(proposal_params, n=self.N_COMBOS)
        logger.info(f"Generated {len(nearby_combos)} nearby parameter combinations")
        
        # Step 3: Run harness for each combo (could be parallelized in Phase 4)
        sharpes = [baseline_sharpe]  # Include baseline in variance calc
        combo_results = []
        
        for i, combo_params in enumerate(nearby_combos):
            logger.info(f"Testing combo {i+1}/{self.N_COMBOS}...")
            result = self.run_backtest_fn(combo_params, tx_cost_pct=None)
            sharpe = result.get('sharpe', 0.0)
            sharpes.append(sharpe)
            combo_results.append({'combo_idx': i, 'sharpe': sharpe})
            logger.info(f"  Combo Sharpe: {sharpe:.4f}")
        
        # Step 4: Compute variance
        sharpe_array = np.array(sharpes)
        mean_sharpe = np.mean(sharpe_array)
        sigma_sharpe = np.std(sharpe_array)
        sigma_pct = (sigma_sharpe / mean_sharpe) * 100.0
        
        # Check threshold
        passed = sigma_pct <= self.MAX_SIGMA_PCT
        
        reasoning = (
            f"Parameter Sweep: Mean Sharpe={mean_sharpe:.4f}, "
            f"σ(Sharpe)={sigma_sharpe:.4f}, "
            f"σ%={sigma_pct:.2f}%, "
            f"Threshold={self.MAX_SIGMA_PCT}%, "
            f"Result={'PASS' if passed else 'FAIL'}"
        )
        logger.info(reasoning)
        
        return SpotCheckResult(
            check_type=self.NAME,
            passed=passed,
            baseline_sharpe=baseline_sharpe,
            test_sharpe=mean_sharpe,
            threshold=self.MAX_SIGMA_PCT,
            reasoning=reasoning,
            metrics={
                'mean_sharpe': mean_sharpe,
                'sigma_sharpe': sigma_sharpe,
                'sigma_pct': sigma_pct,
                'n_combos': len(nearby_combos),
                'combo_results': combo_results,
            }
        )


class SpotCheckRunner:
    """
    Master runner: executes all 3 spot checks and aggregates results.
    """
    
    def __init__(self, run_backtest_fn, regime_detector=None):
        """
        Args:
            run_backtest_fn: Callable that accepts (params_dict, tx_cost_pct=None, start_date=None, end_date=None)
            regime_detector: Optional RegimeDetector instance
        """
        self.run_backtest_fn = run_backtest_fn
        self.cost_test = CostStressTest(run_backtest_fn)
        self.regime_test = RegimeShiftTest(run_backtest_fn, regime_detector)
        self.param_test = ParamSweepTest(run_backtest_fn)
    
    def run_all(self, proposal_params: Dict) -> SpotChecksAggregated:
        """
        Run all 3 spot checks on proposal parameters.
        
        Args:
            proposal_params: Parameters to validate
            
        Returns:
            SpotChecksAggregated with results from all 3 checks
        """
        logger.info("=" * 80)
        logger.info("STARTING SPOT CHECKS SUITE")
        logger.info("=" * 80)
        
        # Run all 3 checks (sequentially for now; can parallelize in Phase 4)
        cost_result = self.cost_test.run(proposal_params)
        regime_result = self.regime_test.run(proposal_params)
        param_result = self.param_test.run(proposal_params)
        
        # Aggregate
        overall_pass = cost_result.passed and regime_result.passed and param_result.passed
        reasoning = (
            f"Cost:{cost_result.passed}, "
            f"Regime:{regime_result.passed}, "
            f"ParamStab:{param_result.passed}"
        )
        
        logger.info("=" * 80)
        logger.info(f"SPOT CHECKS RESULT: {'PASS' if overall_pass else 'FAIL'}")
        logger.info(f"  Cost Stress: {cost_result.passed}")
        logger.info(f"  Regime Shift: {regime_result.passed}")
        logger.info(f"  Param Sweep: {param_result.passed}")
        logger.info("=" * 80)
        
        return SpotChecksAggregated(
            overall_pass=overall_pass,
            cost_stress_pass=cost_result.passed,
            regime_shift_pass=regime_result.passed,
            param_sweep_pass=param_result.passed,
            baseline_sharpe=cost_result.baseline_sharpe,
            reasoning=reasoning,
            cost_stress_result=cost_result,
            regime_shift_result=regime_result,
            param_sweep_result=param_result,
        )
    
    def save_results(self, results: SpotChecksAggregated, output_dir: str = None):
        """
        Save spot check results to JSON files.
        
        Args:
            results: SpotChecksAggregated instance
            output_dir: Directory to save results (default: experiments/results/)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / 'experiments' / 'results'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        summary_file = output_dir / f'{timestamp}_spot_checks_summary.json'
        
        summary_dict = {
            'overall_pass': results.overall_pass,
            'cost_stress_pass': results.cost_stress_pass,
            'regime_shift_pass': results.regime_shift_pass,
            'param_sweep_pass': results.param_sweep_pass,
            'baseline_sharpe': results.baseline_sharpe,
            'reasoning': results.reasoning,
            'timestamp': results.timestamp,
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")
        
        # Save detailed results
        for result in [results.cost_stress_result, results.regime_shift_result, results.param_sweep_result]:
            if result is None:
                continue
            
            detail_file = output_dir / f'{timestamp}_{result.check_type}.json'
            detail_dict = {
                'check_type': result.check_type,
                'passed': result.passed,
                'baseline_sharpe': result.baseline_sharpe,
                'test_sharpe': result.test_sharpe,
                'threshold': result.threshold,
                'reasoning': result.reasoning,
                'metrics': result.metrics,
                'timestamp': result.timestamp,
            }
            
            with open(detail_file, 'w') as f:
                json.dump(detail_dict, f, indent=2)
            logger.info(f"Saved {result.check_type} to {detail_file}")
