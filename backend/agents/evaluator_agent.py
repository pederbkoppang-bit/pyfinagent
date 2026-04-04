"""
Evaluator Agent — Phase 3.1

Skeptical, independent review of planner proposals.
Catches overfitting, validates stress tests, ensures meta-plan alignment.

Uses Claude Sonnet for cost-effective evaluation.
"""

import json
import logging
from typing import Dict, Any

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """Skeptical evaluator of planner proposals."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        """Initialize evaluator with Anthropic client."""
        self.client = Anthropic()
        self.model = model

    def evaluate_proposal(
        self,
        proposal: Dict[str, Any],
        meta_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate proposal for overfitting, feasibility, and stress-test resilience.

        Args:
            proposal: Proposal from Planner agent
            meta_constraints: Meta-plan constraints (max trades, sector limits, etc.)

        Returns:
            {
                'verdict': 'ACCEPT'|'REJECT'|'REVISE',
                'confidence': float (0.0-1.0),
                'reasoning': str,
                'stress_test_concerns': [str],
                'suggested_revisions': [str]
            }
        """

        system_prompt = """YOU ARE: Evaluator Agent, the skeptical critic of trading strategy proposals.

YOUR ROLE: Catch overfitting, validate feasibility, and ensure proposals don't violate constraints.

EVALUATION CRITERIA:
1. **Overfitting Risk:** Does proposal exploit historical patterns that won't repeat?
2. **Stress Test Resilience:** Would it survive 2× costs? Bear market? Regime shift?
3. **Implementation Feasibility:** Can it be coded and backtested in 2 hours?
4. **Meta-Plan Alignment:** Does it violate trade count, sector limits, or Sharpe targets?
5. **Expected Sharpe Gain:** Is the projected gain realistic? (Literature: +2-5% typical)

RED FLAGS:
- Proposals that work ONLY in bull markets
- Features with high look-ahead bias (using future data)
- Parameter ranges that seem extreme
- Proposals that exceed constraint limits

YOUR JUDGMENT STYLE: Constructive skepticism. If rejecting, explain what needs to change.
"""

        # Get constraints from meta-plan
        constraints_text = self._format_constraints(meta_constraints)

        user_prompt = f"""PROPOSAL TO EVALUATE:
{json.dumps(proposal, indent=2)}

META-PLAN CONSTRAINTS:
{constraints_text}

EVALUATE this proposal using the criteria above. Focus on:
1. Will this survive 2× transaction costs? (stress test)
2. Does it work in bear markets or only bull?
3. Is the expected Sharpe gain realistic (+2-5%)?
4. Does it violate any meta-plan constraints?
5. Can it be implemented in 2 hours?

VERDICT: ACCEPT, REJECT, or REVISE. Explain clearly."""

        logger.info("👁️  Evaluator: Reviewing proposal...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        response_text = response.content[0].text

        # Parse response
        verdict_obj = self._parse_evaluation(response_text)
        logger.info(f"✅ Evaluator: {verdict_obj['verdict']} (confidence={verdict_obj['confidence']:.2f})")

        return verdict_obj

    def _parse_evaluation(self, response_text: str) -> Dict[str, Any]:
        """Parse evaluator response to extract verdict and reasoning."""

        # Extract verdict
        verdict = "REJECT"  # Default conservative
        if "ACCEPT" in response_text.upper():
            verdict = "ACCEPT"
        elif "REVISE" in response_text.upper():
            verdict = "REVISE"

        # Confidence: Evaluators should be confident in their verdicts
        confidence = 0.7 if verdict in ["ACCEPT", "REJECT"] else 0.6

        # Extract stress test concerns and revisions
        lines = response_text.split('\n')
        stress_concerns = []
        revisions = []

        for i, line in enumerate(lines):
            if '2×' in line or 'stress' in line.lower() or 'bear' in line.lower():
                stress_concerns.append(line.strip())
            if 'revise' in line.lower() or 'change' in line.lower():
                revisions.append(line.strip())

        return {
            "verdict": verdict,
            "confidence": min(confidence, 1.0),
            "reasoning": response_text,
            "stress_test_concerns": stress_concerns,
            "suggested_revisions": revisions
        }

    def _format_constraints(self, meta_constraints: Dict[str, Any]) -> str:
        """Format meta-plan constraints for evaluator context."""

        constraints = """
- Max Sharpe target: >1.2
- Max trades per month: <50
- Max sector concentration: <30%
- Max drawdown: <20%
- Minimum Sharpe gain: +2-5% (from literature)
- Must survive 2× costs stress test
- Must survive bear market regime shift
"""

        if meta_constraints:
            constraints += "\nADDITIONAL CONSTRAINTS:\n"
            for key, value in meta_constraints.items():
                constraints += f"- {key}: {value}\n"

        return constraints


def get_evaluator_agent(model: str = "claude-sonnet-4-6") -> EvaluatorAgent:
    """Get or create global evaluator agent."""
    global _evaluator
    if '_evaluator' not in globals():
        _evaluator = EvaluatorAgent(model=model)
    return _evaluator
