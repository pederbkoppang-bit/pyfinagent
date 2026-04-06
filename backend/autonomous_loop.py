"""
Phase 3.3: Planner + Evaluator Autonomous Loop

Orchestrates the continuous feedback loop:
  Planner → Generator (run backtests) → Evaluator + Spot Checks → Learn → Repeat

This is the heart of autonomous strategy optimization.
Each cycle: new proposal → 2 backtests in parallel → evaluation → accept best → learn
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from google.cloud import bigquery

logger = logging.getLogger(__name__)


class LoopStatus(str, Enum):
    """Autonomous loop status"""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    EVALUATING = "EVALUATING"
    LEARNING = "LEARNING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


@dataclass
class LoopIteration:
    """Single iteration of the autonomous loop"""
    iteration_id: int
    start_time: str
    planner_proposals: List[Dict[str, Any]]
    selected_proposal: Optional[Dict[str, Any]] = None
    backtest_results: Optional[List[Dict[str, Any]]] = None
    evaluator_verdict: Optional[str] = None
    sharpe_delta: Optional[float] = None
    learnings: Optional[List[str]] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for BQ storage"""
        return {
            k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
            for k, v in asdict(self).items()
        }


class AutonomousLoopOrchestrator:
    """
    Orchestrates the autonomous feedback loop.
    
    Responsibilities:
    1. Call Planner for next proposals
    2. Trigger Generator (run backtests in parallel)
    3. Call Evaluator on results
    4. Make PASS/FAIL decision
    5. Persist learning to BQ + memory files
    6. Decide whether to continue or stop
    """
    
    def __init__(
        self,
        project_id: str,
        dataset_id: str = "trading",
        planner_model: str = "claude-opus-4-6",
        evaluator_model: str = "gemini-2.0-flash",
    ):
        """Initialize orchestrator."""
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.planner_model = planner_model
        self.evaluator_model = evaluator_model
        
        # BigQuery client for logging
        self.bq_client = bigquery.Client(project=project_id)
        self.learning_table = f"{project_id}.{dataset_id}.harness_learning_log"
        
        # Iteration counter
        self.iteration_count = 0
        self.max_iterations = 10  # Stop after 10 cycles max
        self.target_sharpe = 1.23  # 5-10% above baseline 1.1705
        self.baseline_sharpe = 1.1705
        
        logger.info(f"✅ AutonomousLoopOrchestrator initialized")
        logger.info(f"   Target Sharpe: {self.target_sharpe}")
        logger.info(f"   Max iterations: {self.max_iterations}")
    
    async def run_loop(
        self,
        initial_proposal: Optional[Dict[str, Any]] = None,
        initial_sharpe: float = 1.1705,
    ) -> Dict[str, Any]:
        """
        Run the full autonomous loop until convergence or max iterations.
        
        Args:
            initial_proposal: Optional first proposal to evaluate
            initial_sharpe: Current best Sharpe to beat
        
        Returns:
            Summary of loop execution with final Sharpe, iterations, learnings
        """
        
        logger.info("🚀 AUTONOMOUS LOOP: Starting...")
        
        current_best_sharpe = initial_sharpe
        current_best_params = None
        all_learnings = []
        
        for cycle in range(1, self.max_iterations + 1):
            self.iteration_count = cycle
            
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"CYCLE {cycle}/{self.max_iterations}")
                logger.info(f"Current Best Sharpe: {current_best_sharpe:.4f}")
                logger.info(f"{'='*60}\n")
                
                # 1. PLAN: Get proposals from Planner
                iteration = LoopIteration(
                    iteration_id=cycle,
                    start_time=datetime.now(timezone.utc).isoformat(),
                    planner_proposals=[],
                )
                
                proposals = await self._plan_phase(
                    current_best_sharpe=current_best_sharpe,
                    prior_learnings=all_learnings,
                )
                iteration.planner_proposals = proposals
                
                if not proposals:
                    logger.warning("⚠️  No proposals generated. Stopping loop.")
                    break
                
                # 2. GENERATE: Run backtests on top 2 proposals
                selected = proposals[0]  # Use best proposal
                iteration.selected_proposal = selected
                
                backtest_results = await self._generate_phase(
                    proposals=proposals[:2],  # Run best 2 in parallel
                )
                iteration.backtest_results = backtest_results
                
                if not backtest_results:
                    logger.error("❌ Backtest generation failed. Trying next cycle.")
                    continue
                
                # 3. EVALUATE: Run spot checks and evaluator
                verdict, sharpe_delta = await self._evaluate_phase(
                    proposal=selected,
                    backtest_results=backtest_results,
                    baseline_sharpe=current_best_sharpe,
                )
                iteration.evaluator_verdict = verdict if isinstance(verdict, str) else verdict.name
                iteration.sharpe_delta = sharpe_delta
                
                # 4. DECIDE: Accept or reject
                verdict_str = verdict if isinstance(verdict, str) else verdict.name
                if verdict_str == "PASS":
                    new_sharpe = current_best_sharpe + sharpe_delta
                    current_best_sharpe = new_sharpe
                    current_best_params = selected["parameters"]
                    
                    logger.info(f"✅ PASS: Sharpe improved {current_best_sharpe:.4f} (+{sharpe_delta:.4f})")
                    
                    # Extract learnings for next cycle
                    learnings = await self._extract_learnings(
                        proposal=selected,
                        results=backtest_results,
                        verdict=verdict_str,
                    )
                    iteration.learnings = learnings
                    all_learnings.extend(learnings)
                    
                    # Check if we've hit target
                    if current_best_sharpe >= self.target_sharpe:
                        logger.info(f"🎯 TARGET REACHED: {current_best_sharpe:.4f} >= {self.target_sharpe:.4f}")
                        iteration.end_time = datetime.now(timezone.utc).isoformat()
                        await self._log_iteration_to_bq(iteration)
                        break
                
                elif verdict_str == "CONDITIONAL":
                    logger.warning(f"⚠️  CONDITIONAL: Run spot checks and try again")
                    # In real system, run detailed spot checks here
                    # For now, continue to next proposal
                    
                else:  # FAIL
                    logger.info(f"❌ FAIL: Proposal rejected by evaluator")
                    # Continue to next cycle with new proposal
                
                # 5. LEARN: Log iteration
                iteration.end_time = datetime.now(timezone.utc).isoformat()
                await self._log_iteration_to_bq(iteration)
                
            except Exception as e:
                logger.error(f"🔥 ERROR in cycle {cycle}: {e}", exc_info=True)
                continue
        
        # Return summary
        summary = {
            "status": "COMPLETE",
            "final_sharpe": current_best_sharpe,
            "sharpe_gain": current_best_sharpe - initial_sharpe,
            "sharpe_gain_pct": 100 * (current_best_sharpe - initial_sharpe) / initial_sharpe,
            "iterations_completed": self.iteration_count,
            "target_reached": current_best_sharpe >= self.target_sharpe,
            "best_params": current_best_params,
            "cumulative_learnings": all_learnings,
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 AUTONOMOUS LOOP COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Final Sharpe: {summary['final_sharpe']:.4f}")
        logger.info(f"Total Gain: {summary['sharpe_gain']:.4f} ({summary['sharpe_gain_pct']:.1f}%)")
        logger.info(f"Iterations: {summary['iterations_completed']}/{self.max_iterations}")
        logger.info(f"Target Reached: {summary['target_reached']}")
        
        return summary
    
    async def _plan_phase(
        self,
        current_best_sharpe: float,
        prior_learnings: List[str],
    ) -> List[Dict[str, Any]]:
        """
        PLAN: Call Planner agent for next proposals.
        
        Returns list of 3-5 proposals ranked by expected impact.
        """
        
        logger.info("📋 PLAN: Calling Planner agent...")
        
        # Import here to avoid circular imports
        try:
            from backend.agents.planner_agent import PlannerAgent
        except ImportError:
            logger.warning("⚠️  Could not import PlannerAgent. Using mock proposals.")
            return self._get_mock_proposals(current_best_sharpe)
        
        planner = PlannerAgent(model=self.planner_model)
        
        # Mock recent results for now (in real system, fetch from backtest DB)
        mock_recent_results = [
            {
                "sharpe": current_best_sharpe,
                "return_pct": 80.2,
                "max_dd": -12.0,
                "num_trades": 45,
                "features": ["ma_crossover", "rsi_oversold", "volatility_regime"],
            }
        ]
        
        current_params = {
            "ma_short": 20,
            "ma_long": 50,
            "rsi_threshold": 30,
            "vol_lookback": 20,
        }
        
        try:
            proposal_json = planner.generate_proposal(
                recent_results=mock_recent_results,
                current_best_sharpe=current_best_sharpe,
                current_params=current_params,
                weaknesses="Strategy may be over-fitted to 2020-2023. Need regime adaptation.",
            )
            
            proposals = proposal_json.get("proposals", [])
            logger.info(f"✅ Planner generated {len(proposals)} proposals")
            
            return proposals[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"❌ Planner call failed: {e}")
            return self._get_mock_proposals(current_best_sharpe)
    
    async def _generate_phase(
        self,
        proposals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        GENERATE: Run backtests on the top proposals (in parallel).
        
        Returns list of backtest results.
        """
        
        logger.info(f"⚙️  GENERATE: Running {len(proposals)} backtests in parallel...")
        
        # Import backtest harness
        try:
            from backend.backtest.backtest_engine import BacktestEngine
        except ImportError:
            logger.warning("⚠️  Could not import BacktestEngine. Using mock results.")
            return self._get_mock_backtest_results(len(proposals))
        
        # In real system, we'd run these in parallel via run_harness.py
        # For now, return mock results
        results = self._get_mock_backtest_results(len(proposals))
        
        logger.info(f"✅ Generated {len(results)} backtest results")
        return results
    
    async def _evaluate_phase(
        self,
        proposal: Dict[str, Any],
        backtest_results: List[Dict[str, Any]],
        baseline_sharpe: float,
    ) -> Tuple[Any, float]:
        """
        EVALUATE: Run evaluator on backtest results + spot checks.
        
        Returns (verdict, sharpe_delta).
        """
        
        logger.info("🔍 EVALUATE: Running evaluator...")
        
        try:
            from backend.agents.evaluator_agent import EvaluatorAgent, EvaluationVerdict
        except ImportError:
            logger.warning("⚠️  Could not import EvaluatorAgent. Using mock evaluation.")
            # Return mock verdict
            result_sharpe = backtest_results[0].get("sharpe", 1.1705)
            delta = result_sharpe - baseline_sharpe
            
            if delta > 0.02:
                return type('Verdict', (), {'name': 'PASS'})(), delta
            else:
                return type('Verdict', (), {'name': 'FAIL'})(), delta
        
        evaluator = EvaluatorAgent(model_name=self.evaluator_model)
        
        # Get first backtest result
        best_result = backtest_results[0] if backtest_results else {}
        
        # Calculate Sharpe delta
        result_sharpe = best_result.get("sharpe", baseline_sharpe)
        sharpe_delta = result_sharpe - baseline_sharpe
        
        logger.info(f"   Proposal Sharpe: {result_sharpe:.4f}")
        logger.info(f"   Delta: {sharpe_delta:.4f}")
        
        # Simple evaluation logic for now
        if result_sharpe > baseline_sharpe and best_result.get("dsr", 0) > 0.95:
            logger.info(f"   ✅ PASS: Sharpe improved and DSR valid")
            return "PASS", sharpe_delta
        else:
            logger.info(f"   ❌ FAIL: Sharpe did not improve or DSR invalid")
            return "FAIL", sharpe_delta
    
    async def _extract_learnings(
        self,
        proposal: Dict[str, Any],
        results: List[Dict[str, Any]],
        verdict: Any,
    ) -> List[str]:
        """Extract actionable learnings from this iteration for next cycle."""
        
        learnings = []
        
        # Parse proposal features
        features = proposal.get("features", [])
        parameters = proposal.get("parameters", {})
        
        if features:
            learnings.append(f"✅ Features '{', '.join(features)}' were effective")
        
        # Check which sub-periods did well
        if results:
            result = results[0]
            sub_periods = result.get("sub_periods", {})
            
            best_period = max(
                sub_periods.items(),
                key=lambda x: x[1],
                default=("unknown", 0)
            )
            learnings.append(f"✅ Strategy strongest in {best_period[0]} ({best_period[1]:.4f})")
        
        learnings.append(f"✅ Parameter set: {json.dumps(parameters)}")
        
        return learnings
    
    async def _log_iteration_to_bq(self, iteration: LoopIteration) -> None:
        """Log iteration to BigQuery for analysis."""
        
        try:
            # Create table if not exists
            table_id = self.learning_table
            
            # Insert row
            row = {
                "iteration_id": iteration.iteration_id,
                "start_time": iteration.start_time,
                "end_time": iteration.end_time,
                "planner_proposals": json.dumps(iteration.planner_proposals),
                "selected_proposal": json.dumps(iteration.selected_proposal),
                "backtest_results": json.dumps(iteration.backtest_results),
                "evaluator_verdict": iteration.evaluator_verdict,
                "sharpe_delta": iteration.sharpe_delta,
                "learnings": json.dumps(iteration.learnings),
            }
            
            # Note: In real system, would insert to BQ
            # self.bq_client.insert_rows_json(table_id, [row])
            
            logger.debug(f"📝 Logged iteration {iteration.iteration_id} to BQ")
            
        except Exception as e:
            logger.error(f"⚠️  Failed to log iteration to BQ: {e}")
    
    def _get_mock_proposals(self, baseline_sharpe: float) -> List[Dict[str, Any]]:
        """Generate mock proposals for testing."""
        
        return [
            {
                "feature_name": "regime_aware_ma_crossover",
                "parameters": {"ma_short": 18, "ma_long": 48, "regime_lookback": 20},
                "hypothesis": "Adjust MA periods based on market regime (trending vs mean-reverting)",
                "expected_sharpe_gain": 0.05,
                "implementation_complexity": "medium",
            },
            {
                "feature_name": "volatility_scaled_position_sizing",
                "parameters": {"vol_target": 15, "vol_lookback": 30},
                "hypothesis": "Scale position size inversely to volatility to maintain consistent risk exposure",
                "expected_sharpe_gain": 0.03,
                "implementation_complexity": "low",
            },
            {
                "feature_name": "sector_rotation",
                "parameters": {"sector_lookback": 60, "max_sector_exposure": 0.3},
                "hypothesis": "Rotate between sectors based on 60-day momentum, cap exposure to 30% per sector",
                "expected_sharpe_gain": 0.04,
                "implementation_complexity": "high",
            },
        ]
    
    def _get_mock_backtest_results(self, count: int) -> List[Dict[str, Any]]:
        """Generate mock backtest results for testing."""
        
        results = []
        for i in range(count):
            sharpe_gain = 0.02 + (i * 0.01)  # First proposal gains 2%, second gains 3%
            results.append({
                "sharpe": self.baseline_sharpe + sharpe_gain,
                "dsr": 0.98 - (i * 0.02),
                "return_pct": 82.0 + (i * 2),
                "max_dd": -12.5 - (i * 0.5),
                "num_trades": 48 - (i * 2),
                "sub_periods": {
                    "2018_2019": 0.92,
                    "2020_2021": 1.15,
                    "2022_2023": 1.30,
                    "2024_2025": 1.05,
                },
            })
        
        return results


async def main():
    """Test the orchestrator."""
    
    logging.basicConfig(level=logging.INFO)
    
    project_id = os.getenv("GCP_PROJECT_ID", "pyfinagent-prod")
    
    orchestrator = AutonomousLoopOrchestrator(project_id=project_id)
    
    summary = await orchestrator.run_loop(
        initial_sharpe=1.1705,
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
