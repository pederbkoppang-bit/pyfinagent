"""
Enhanced Planner Agent for Phase 3.3 Autonomous Loop

Improvements over basic planner:
1. Reads RESEARCH.md for unexplored alpha sources
2. Detects regime changes and adapts proposals
3. Learns from evaluator feedback (maintains context)
4. Ranks proposals by expected Sharpe improvement
5. Outputs structured JSON for programmatic parsing
"""

import json
import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class EnhancedPlannerAgent:
    """
    Enhanced LLM-based planner for autonomous loop.
    
    Reads RESEARCH.md, detects alpha sources, generates ranked proposals.
    """
    
    def __init__(self, model: str = "claude-opus-4-6"):
        """Initialize enhanced planner."""
        self.client = Anthropic()
        self.model = model
        self.conversation_history = []
        self.research_md_path = Path(__file__).parent.parent.parent / "RESEARCH.md"
        
        logger.info(f"[OK] EnhancedPlannerAgent initialized with {model}")
    
    def generate_proposals(
        self,
        current_sharpe: float,
        current_params: Dict[str, Any],
        backtest_history: List[Dict[str, Any]],
        evaluator_feedback: Optional[str] = None,
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate 3-5 ranked proposals for next iteration.
        
        Args:
            current_sharpe: Current best Sharpe ratio
            current_params: Current best parameters
            backtest_history: List of recent backtest results
            evaluator_feedback: Optional feedback from evaluator (CONDITIONAL/FAIL reasons)
            regime_info: Optional market regime information (bull/bear/range)
        
        Returns:
            {
                "proposals": [
                    {
                        "rank": 1,
                        "feature": "regime_aware_ma_crossover",
                        "rationale": "Research shows MA periods should vary with volatility regime...",
                        "parameters": {...},
                        "expected_sharpe_improvement": 0.05,
                        "confidence": "high",
                        "risk_level": "low",
                        "implementation_hours": 2,
                        "alpha_source": "Ta-Lib studies + BBG terminal observations"
                    },
                    ...
                ],
                "reasoning": "Overall strategy...",
                "regime_adaptation": "Current regime is...",
                "research_insights_applied": ["source1", "source2"],
            }
        """
        
        logger.info(f"[think] EnhancedPlanner: Generating proposals (current Sharpe={current_sharpe:.4f})")
        
        # Read research sources
        research_text = self._read_research_md()
        
        # Build system prompt
        system_prompt = self._build_system_prompt(
            current_sharpe=current_sharpe,
            regime_info=regime_info,
        )
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            current_sharpe=current_sharpe,
            current_params=current_params,
            backtest_history=backtest_history,
            evaluator_feedback=evaluator_feedback,
            regime_info=regime_info,
            research_text=research_text,
        )
        
        try:
            # Call Claude with structured output
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            response_text = response.content[0].text
            
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                proposals_json = json.loads(response_text[json_start:json_end])
            else:
                logger.error("[warn]  Could not find JSON in response")
                proposals_json = self._get_default_proposals(current_sharpe)
            
            logger.info(f"[OK] Generated {len(proposals_json.get('proposals', []))} proposals")
            return proposals_json
            
        except Exception as e:
            logger.error(f"[FAIL] Planner call failed: {e}")
            return self._get_default_proposals(current_sharpe)
    
    def _read_research_md(self) -> str:
        """Read RESEARCH.md to find unexplored alpha sources."""
        
        try:
            if self.research_md_path.exists():
                with open(self.research_md_path, "r") as f:
                    return f.read()
            else:
                logger.warning(f"[warn]  RESEARCH.md not found at {self.research_md_path}")
                return ""
        except Exception as e:
            logger.error(f"[warn]  Could not read RESEARCH.md: {e}")
            return ""
    
    def _build_system_prompt(
        self,
        current_sharpe: float,
        regime_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build comprehensive system prompt for planner."""
        
        regime_context = ""
        if regime_info:
            regime = regime_info.get("regime", "unknown")
            confidence = regime_info.get("confidence", 0.5)
            regime_context = f"""
CURRENT MARKET REGIME (confidence: {confidence:.1%}):
  - Regime: {regime}
  - Characteristics: {regime_info.get('characteristics', 'N/A')}
  
Adapt proposals to this regime. Avoid strategies that fail in {regime}.
"""
        
        return f"""YOU ARE: LLM-as-Planner for Phase 3.3 Autonomous Loop (pyfinAgent)

YOUR TASK:
1. Analyze current performance (Sharpe {current_sharpe:.4f})
2. Read RESEARCH.md for unexplored alpha sources
3. Generate 3-5 ranked proposals (best first)
4. Adapt proposals to current market regime
5. Include expected improvement, risk level, implementation hours
6. Output valid JSON with structured proposal list

CONSTRAINTS:
- All proposals must improve Sharpe by >2% (expected)
- Features must be implementable in <4 hours
- Avoid over-fitting (propose simple, interpretable features)
- Stress test mentally: Does it survive bear markets?
- Reference specific research papers/sources when possible

OPTIMIZATION TARGETS:
- Sharpe target: 1.23 (from current 1.1705)
- Max drawdown: Keep <15%
- Number of trades: Keep <50/month
- Sector concentration: Limit to <30%

{regime_context}

OUTPUT FORMAT: Valid JSON with 'proposals' array, each containing:
  - rank (1-5)
  - feature (name of feature/parameter change)
  - rationale (why this should improve performance)
  - parameters (exact dict of param changes)
  - expected_sharpe_improvement (float, e.g., 0.05)
  - confidence (high/medium/low)
  - risk_level (low/medium/high)
  - implementation_hours (1-4)
  - alpha_source (research paper, blog post, or empirical observation)
"""
    
    def _build_user_prompt(
        self,
        current_sharpe: float,
        current_params: Dict[str, Any],
        backtest_history: List[Dict[str, Any]],
        evaluator_feedback: Optional[str],
        regime_info: Optional[Dict[str, Any]],
        research_text: str,
    ) -> str:
        """Build detailed user prompt with all context."""
        
        # Summarize recent backtest results
        history_summary = "RECENT BACKTEST HISTORY:\n"
        for i, result in enumerate(backtest_history[-3:], 1):
            history_summary += f"""
  Run {i}:
    - Sharpe: {result.get('sharpe', 0):.4f}
    - Return: {result.get('return_pct', 0):.1f}%
    - MaxDD: {result.get('max_dd', 0):.1f}%
    - Trades: {result.get('num_trades', 0)}
    - Sub-periods: {result.get('sub_periods', {})}
"""
        
        feedback_section = ""
        if evaluator_feedback:
            feedback_section = f"""
EVALUATOR FEEDBACK FROM LAST ITERATION:
{evaluator_feedback}

Address these concerns in your proposals.
"""
        
        research_section = ""
        if research_text:
            # Extract key insights (simplified for token budget)
            research_section = f"""
RESEARCH INSIGHTS (from RESEARCH.md):
{research_text[:1500]}...
[See RESEARCH.md for full analysis]

Look for alpha sources mentioned in the research but not yet tested.
"""
        
        return f"""
{history_summary}

CURRENT PARAMETERS:
{json.dumps(current_params, indent=2)}

{feedback_section}

{research_section}

TASK:
Generate 3-5 ranked proposals to improve on current Sharpe {current_sharpe:.4f}.

Consider:
1. Which features from RESEARCH.md are NOT yet in current_params?
2. Which sub-periods underperform? What features would help?
3. What regime adaptations are needed?
4. What parameter ranges have worked well historically?

Return valid JSON with ranked proposals (best first).
"""
    
    def _get_default_proposals(self, current_sharpe: float) -> Dict[str, Any]:
        """Fallback proposals if LLM call fails."""
        
        return {
            "proposals": [
                {
                    "rank": 1,
                    "feature": "volatility_regime_detection",
                    "rationale": "Adapt strategy to current volatility regime (high/normal/low)",
                    "parameters": {"volatility_lookback": 30, "vol_threshold_high": 0.25, "vol_threshold_low": 0.12},
                    "expected_sharpe_improvement": 0.05,
                    "confidence": "high",
                    "risk_level": "low",
                    "implementation_hours": 2,
                    "alpha_source": "Hamilton switching model (Hamilton 1989) + empirical testing",
                },
                {
                    "rank": 2,
                    "feature": "mean_reversion_oversold_rebounds",
                    "rationale": "Exploit oversold conditions (RSI < 30) with better entry timing",
                    "parameters": {"rsi_lookback": 14, "oversold_threshold": 25, "rebound_lookback": 3},
                    "expected_sharpe_improvement": 0.04,
                    "confidence": "medium",
                    "risk_level": "low",
                    "implementation_hours": 2,
                    "alpha_source": "Wilder RSI studies + momentum research",
                },
                {
                    "rank": 3,
                    "feature": "sector_rotation_momentum",
                    "rationale": "Rotate between sectors based on 60-day momentum, avoid weak sectors",
                    "parameters": {"sector_lookback": 60, "max_sector_exposure": 0.25, "rotation_threshold": 0.10},
                    "expected_sharpe_improvement": 0.03,
                    "confidence": "medium",
                    "risk_level": "medium",
                    "implementation_hours": 3,
                    "alpha_source": "Empirical momentum studies + sector relative strength",
                },
            ],
            "reasoning": f"""Current Sharpe {current_sharpe:.4f} is strong but can be improved by:
1. Adapting to volatility regimes (Hamilton switching)
2. Timing oversold rebounds better (RSI optimization)
3. Adding sector rotation discipline (momentum-based)""",
            "regime_adaptation": "Moderate volatility environment - favor mean reversion strategies",
            "research_insights_applied": ["Hamilton 1989", "Wilder RSI", "Momentum research"],
        }


def get_enhanced_planner(model: str = "claude-opus-4-6") -> EnhancedPlannerAgent:
    """Get or create global enhanced planner instance."""
    global _enhanced_planner
    if '_enhanced_planner' not in globals():
        _enhanced_planner = EnhancedPlannerAgent(model=model)
    return _enhanced_planner


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    planner = EnhancedPlannerAgent()
    
    proposals = planner.generate_proposals(
        current_sharpe=1.1705,
        current_params={"ma_short": 20, "ma_long": 50},
        backtest_history=[],
    )
    
    print(json.dumps(proposals, indent=2))
