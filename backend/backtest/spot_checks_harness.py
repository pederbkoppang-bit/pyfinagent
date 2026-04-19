"""
Harness Integration for Spot Checks — Phase 3.2.1 GENERATE

This module orchestrates spot check execution within the harness.
Call: python -c "from backend.backtest.spot_checks_harness import run_spot_checks_on_proposal; ..."

Usage:
    # From pyfinagent/ root directory:
    python -c "
from backend.backtest.spot_checks_harness import run_spot_checks_on_proposal
import json

proposal = json.load(open('backend/backtest/experiments/optimizer_best.json'))['params']
result = run_spot_checks_on_proposal(proposal)
print(f'Overall Pass: {result.overall_pass}')
print(f'  Cost Stress: {result.cost_stress_pass}')
print(f'  Regime Shift: {result.regime_shift_pass}')
print(f'  Param Sweep: {result.param_sweep_pass}')
"
"""

import logging
import sys
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("spot_checks_harness")

# -- Setup path and env --
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv("backend/.env")

# Suppress noisy GCP logs
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from backend.config.settings import get_settings
from backend.db.bigquery_client import BigQueryClient
import sys
from pathlib import Path
# Add project root to path so scripts/harness imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.harness.run_harness import run_backtest
from backend.backtest.spot_checks import SpotCheckRunner


def run_spot_checks_on_proposal(proposal_params: dict) -> object:
    """
    Execute all 3 spot checks on a proposal using the harness.
    
    Args:
        proposal_params: Dict with strategy parameters (from optimizer_best.json)
        
    Returns:
        SpotChecksAggregated result object with pass/fail for each check
    """
    logger.info("=" * 80)
    logger.info("SPOT CHECKS HARNESS INTEGRATION")
    logger.info("=" * 80)
    logger.info(f"Proposal Sharpe: {proposal_params.get('sharpe', 'N/A')}")
    logger.info(f"Proposal Run ID: {proposal_params.get('run_id', 'N/A')}")
    
    # Initialize settings and BigQuery
    settings = get_settings()
    bq = BigQueryClient(
        gcp_project=settings.gcp_project_id,
        bq_dataset_reports=settings.bq_dataset_reports,
    )
    
    # phase-3.3: opt-in VIX rolling-quantile regime detector (settings-gated).
    # Default False preserves pre-phase-3.3 behavior (static 2-regime split).
    regime_detector = None
    if getattr(settings, "regime_detection_enabled", False):
        from backend.backtest.regime_detector import VIXRollingQuantileRegimeDetector

        regime_detector = VIXRollingQuantileRegimeDetector(
            start_date=getattr(settings, "backtest_start_date", "2018-01-01"),
            end_date=getattr(settings, "backtest_end_date", "2025-12-31"),
        )

    # Create spot check runner with run_backtest as the harness function
    runner = SpotCheckRunner(
        run_backtest_fn=lambda params, tx_cost_pct=None, start_date=None, end_date=None:
            run_backtest(params, settings, bq, start_date=start_date, end_date=end_date, tx_cost_pct=tx_cost_pct),
        regime_detector=regime_detector  # Fallback to 2-regime split when None
    )
    
    # Run all 3 spot checks
    results = runner.run_all(proposal_params)
    
    # Save results
    output_dir = Path(__file__).parent / 'experiments' / 'results'
    runner.save_results(results, output_dir=str(output_dir))
    
    logger.info("=" * 80)
    logger.info("SPOT CHECKS COMPLETE")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    # Load best params and run spot checks
    import json
    
    best_params_path = Path(__file__).parent / 'experiments' / 'optimizer_best.json'
    with open(best_params_path) as f:
        data = json.load(f)
    
    proposal_params = data['params']
    
    # Run spot checks
    results = run_spot_checks_on_proposal(proposal_params)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SPOT CHECKS SUMMARY")
    print("=" * 80)
    print(f"Overall Pass: {results.overall_pass}")
    print(f"  Cost Stress: {results.cost_stress_pass}")
    print(f"  Regime Shift: {results.regime_shift_pass}")
    print(f"  Param Sweep: {results.param_sweep_pass}")
    print(f"Baseline Sharpe: {results.baseline_sharpe:.4f}")
    print(f"Reasoning: {results.reasoning}")
    print("=" * 80)
    
    sys.exit(0 if results.overall_pass else 1)
