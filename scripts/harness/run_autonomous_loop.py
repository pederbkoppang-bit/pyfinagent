#!/usr/bin/env python3
"""
Phase 3.3 Harness Entry Point — Autonomous Planner + Evaluator Loop

Runs the continuous feedback loop:
  Planner → Generator → Evaluator → Learn → Repeat

Usage:
  python run_autonomous_loop.py [--cycles 3] [--max-iterations 10] [--dry-run]

Environment:
  - GOOGLE_APPLICATION_CREDENTIALS: GCP service account JSON
  - GCP_PROJECT_ID: GCP project (default: pyfinagent-prod)
  - ANTHROPIC_API_KEY: Anthropic API key for Planner agent
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.autonomous_loop import AutonomousLoopOrchestrator, LoopStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("handoff/autonomous_loop.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


async def run_autonomous_loop(
    cycles: int = 1,
    max_iterations: int = 10,
    dry_run: bool = False,
) -> None:
    """
    Run the autonomous loop.
    
    Args:
        cycles: Number of independent loop runs (for A/B testing)
        max_iterations: Max iterations per cycle
        dry_run: If True, don't persist results to BQ
    """
    
    logger.info("="*80)
    logger.info("PHASE 3.3: AUTONOMOUS PLANNER + EVALUATOR LOOP")
    logger.info("="*80)
    logger.info(f"Start Time: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Cycles: {cycles}")
    logger.info(f"Max Iterations/Cycle: {max_iterations}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("")
    
    # Get GCP project
    project_id = os.getenv("GCP_PROJECT_ID", "pyfinagent-prod")
    logger.info(f"GCP Project: {project_id}")
    
    try:
        # Initialize orchestrator
        orchestrator = AutonomousLoopOrchestrator(
            project_id=project_id,
            dataset_id="trading",
            planner_model="claude-opus-4-6",
            evaluator_model="gemini-2.0-flash",
        )
        
        orchestrator.max_iterations = max_iterations
        
        # Run autonomous loop
        summary = await orchestrator.run_loop(
            initial_sharpe=1.1705,  # Current best from Phase 3.2.1
        )
        
        logger.info("")
        logger.info("="*80)
        logger.info("LOOP COMPLETE")
        logger.info("="*80)
        logger.info(f"Final Sharpe: {summary['final_sharpe']:.4f}")
        logger.info(f"Sharpe Gain: {summary['sharpe_gain']:.4f} ({summary['sharpe_gain_pct']:.1f}%)")
        logger.info(f"Iterations: {summary['iterations_completed']}/{max_iterations}")
        logger.info(f"Target Reached: {summary['target_reached']}")
        logger.info(f"End Time: {datetime.now(timezone.utc).isoformat()}")
        
        # Write summary to handoff directory
        import json
        handoff_file = "handoff/phase3.3_loop_summary.json"
        with open(handoff_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary written to {handoff_file}")
        
        # Return exit code based on success
        if summary.get("target_reached"):
            logger.info("✅ SUCCESS: Target Sharpe reached!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Target not reached, but loop completed normally")
            sys.exit(0)
        
    except Exception as e:
        logger.error(f"🔥 FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


def main():
    """CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Phase 3.3: Autonomous Planner + Evaluator Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_autonomous_loop.py
    → Run single loop with default max 10 iterations
  
  python run_autonomous_loop.py --max-iterations 5 --dry-run
    → Run in dry-run mode, stop after 5 iterations
  
  python run_autonomous_loop.py --cycles 3 --max-iterations 15
    → Run 3 independent loops, max 15 iterations each
        """
    )
    
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of independent loop runs (default: 1)",
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations per cycle (default: 10)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't persist results to BigQuery",
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(
        run_autonomous_loop(
            cycles=args.cycles,
            max_iterations=args.max_iterations,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
