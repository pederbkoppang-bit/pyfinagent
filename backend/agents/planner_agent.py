"""
LLM-as-Planner Agent — Phase 3.1

Autonomous strategy proposal agent that:
1. Reads recent backtest results + market evidence
2. Identifies weak points in current strategy
3. Proposes next features/parameters to test
4. Ranks proposals by expected impact

Uses Claude Opus for high-quality reasoning.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from anthropic import Anthropic

logger = logging.getLogger(__name__)

META_PLAN = """
STRATEGIC GOAL (Meta Plan):
- Maximize Sharpe Ratio > 1.2
- Maintain annual returns 50-100%
- Keep maximum drawdown < 20%
- Limit trades to <50/month to minimize friction
- Avoid sector concentration > 30% (sector-level risk)
- Survive 2× transaction cost stress test
"""


class PlannerAgent:
    """LLM-as-Planner for autonomous feature generation."""

    def __init__(self, model: str = "claude-opus-4-6"):
        """Initialize planner with Anthropic client."""
        self.client = Anthropic()
        self.model = model
        self.conversation_history = []

    def generate_proposal(
        self,
        recent_results: List[Dict[str, Any]],
        current_best_sharpe: float,
        current_params: Dict[str, Any],
        weaknesses: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate feature/parameter proposal based on recent backtest evidence.

        Args:
            recent_results: Last 5-10 backtest results with Sharpe, Return, MaxDD, trades
            current_best_sharpe: Current best Sharpe ratio
            current_params: Current best parameters
            weaknesses: Optional description of current strategy weaknesses

        Returns:
            {
                'proposals': [
                    {
                        'feature_name': str,
                        'parameters': dict,
                        'hypothesis': str,
                        'expected_sharpe_gain': float,
                        'implementation_complexity': 'low'|'medium'|'high'
                    }
                ],
                'reasoning': str,
                'meta_plan_alignment': str
            }
        """

        # Build evidence summary
        evidence = self._summarize_evidence(recent_results, current_best_sharpe, weaknesses)

        # System prompt with meta-plan
        system_prompt = f"""{META_PLAN}

YOU ARE: LLM-as-Planner for pyfinAgent, an autonomous trading strategy optimizer.

YOUR TASK:
1. Analyze recent backtest evidence
2. Identify gaps in current strategy
3. Propose 3 next features or parameter adjustments
4. Rank by expected impact
5. Align ALL proposals with the meta-plan above

CONSTRAINTS:
- Avoid proposals that violate meta-plan (e.g., >50 trades/month)
- Features must be implementable within 2 hours
- Stress-test the proposal: Does it survive 2× costs? Bear market?
- Be specific: include exact parameter ranges, not vague suggestions

OUTPUT FORMAT: Return valid JSON with 'proposals' array and 'reasoning' field.
"""

        # User prompt
        user_prompt = f"""EVIDENCE SUMMARY:
{evidence}

CURRENT STATUS:
- Best Sharpe: {current_best_sharpe:.4f}
- Current Parameters: {json.dumps(current_params, indent=2)}

GENERATE 3 PROPOSALS to improve on this baseline. Focus on:
1. Features that address identified weaknesses
2. Parameters that have proven effective in similar market conditions
3. Implementations that won't blow up the strategy

Be specific about parameters and expected gains. Include reasoning for each proposal."""

        # Call Claude
        logger.info(f"[think] Planner: Generating proposal (Sharpe={current_best_sharpe:.4f})")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Parse response
        response_text = response.content[0].text

        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                proposal_json = json.loads(response_text[json_start:json_end])
            else:
                # Fallback if no JSON found
                proposal_json = {
                    "proposals": [],
                    "reasoning": response_text,
                    "meta_plan_alignment": "Could not parse JSON response"
                }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planner response: {e}")
            proposal_json = {
                "proposals": [],
                "reasoning": response_text,
                "error": str(e)
            }

        logger.info(f"[OK] Planner: Generated {len(proposal_json.get('proposals', []))} proposals")
        return proposal_json

    def _summarize_evidence(
        self,
        recent_results: List[Dict[str, Any]],
        current_best_sharpe: float,
        weaknesses: Optional[str] = None
    ) -> str:
        """Summarize recent backtest evidence for planner context."""

        summary = f"""RECENT BACKTEST RESULTS (last 5-10 runs):
"""
        for i, result in enumerate(recent_results[-5:], 1):
            summary += f"""
Run {i}:
  - Sharpe: {result.get('sharpe', 0):.4f}
  - Return: {result.get('return_pct', 0):.2f}%
  - MaxDD: {result.get('max_dd', 0):.2f}%
  - Trades: {result.get('num_trades', 0)}
  - Features: {', '.join(result.get('features', [])[:3])}
"""

        if weaknesses:
            summary += f"\nIDENTIFIED WEAKNESSES:\n{weaknesses}\n"

        # Market regime hint (you could enhance this with actual regime detection)
        summary += f"\nCURRENT BEST SHARPE: {current_best_sharpe:.4f}\n"

        return summary

    def reflect_on_feedback(
        self,
        proposal: Dict[str, Any],
        feedback: str
    ) -> Dict[str, Any]:
        """
        Reflect on evaluator feedback and improve proposal.

        Args:
            proposal: Original proposal from planner
            feedback: Evaluator's rejection/revision feedback

        Returns:
            Revised proposal
        """

        logger.info("[think] Planner: Reflecting on evaluator feedback...")

        system_prompt = f"""{META_PLAN}

YOU ARE: LLM-as-Planner, reflecting on feedback from the Evaluator agent.

YOUR TASK:
1. Read evaluator's feedback
2. Understand why proposal was rejected
3. Revise proposal to address concerns
4. Keep same general direction, but fix the issues
"""

        user_prompt = f"""ORIGINAL PROPOSAL:
{json.dumps(proposal, indent=2)}

EVALUATOR FEEDBACK:
{feedback}

REVISE THE PROPOSAL to address these concerns. Keep the same feature focus but:
- Adjust parameters to reduce risk
- Add clearer constraints
- Simplify implementation if too complex
- Address stress test concerns

Return revised proposal in JSON format."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = response.content[0].text

        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            revised = json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError:
            revised = proposal  # Fallback to original

        logger.info("[OK] Planner: Revised proposal based on feedback")
        return revised


def get_planner_agent(model: str = "claude-opus-4-6") -> PlannerAgent:
    """Get or create global planner agent."""
    global _planner
    if '_planner' not in globals():
        _planner = PlannerAgent(model=model)
    return _planner
