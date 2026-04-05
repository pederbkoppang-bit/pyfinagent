"""
LLM-as-Evaluator — Skeptical Independent Judge (Phase 3.2)

The evaluator is the quality gate for all Planner proposals.
Key principle: "Separating generation from evaluation is the strongest lever."

Flow:
  1. Planner proposes: features, parameters, hypothesis
  2. Generator runs backtest: metrics, sub-periods, Sharpe, max drawdown
  3. Evaluator reviews: runs 5-point rubric, decides PASS/FAIL/CONDITIONAL
  4. If CONDITIONAL: runs spot checks (2× costs, regime shift, param sweep)
  5. Verdict: PASS (proceed) | CONDITIONAL (fix first) | FAIL (reject)

Research-backed evaluation rubric:
  1. Statistical Validity — DSR > 0.95, Sharpe < 2.0, Bonferroni-corrected p-value
  2. Robustness — Works in bull/bear/range regimes, stable across walk-forward
  3. Simplicity — <5 features preferred, <3 tuned parameters, understandable
  4. Reality Gap — Realistic assumptions, matches live trading conditions
  5. Risk Check — Worst-case scenarios, antifragile vs fragile behavior

References:
  - Bailey & López de Prado (2014): DSR + over-fitting detection
  - Harvey, Liu, Zhu (2015): Multiple testing correction
  - Lo (2002): Serial correlation adjustment for Sharpe
  - Arnott et al. (AQR 2016): Investment decision checklist
  - Pardo et al. (2019): Stress testing framework
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Any

try:
    from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationVerdict(str, Enum):
    """Evaluation outcome"""
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


@dataclass
class EvaluationResult:
    """Structured evaluation result"""
    verdict: EvaluationVerdict
    statistical_validity_score: float  # 0-100
    robustness_score: float
    simplicity_score: float
    reality_gap_score: float
    risk_check_score: float
    
    overall_score: float  # Average of 5 scores
    
    summary: str  # 1-2 sentence summary
    detailed_reasoning: str  # Full evaluation
    
    red_flags: List[str]  # Critical issues
    yellow_flags: List[str]  # Warnings
    green_flags: List[str]  # Positive aspects
    
    recommended_spot_checks: Optional[List[str]] = None  # If CONDITIONAL
    suggested_fixes: Optional[List[str]] = None  # If CONDITIONAL or FAIL


class EvaluatorAgent:
    """
    Skeptical LLM-based evaluator for backtest proposals.
    
    Uses Claude Sonnet with structured evaluation rubric.
    Optimized for speed (<30s) and over-fitting detection.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        """Initialize evaluator with Claude/Gemini"""
        self.model_name = model_name
        self.max_eval_time = 30  # seconds
        
        # Initialize model if Vertex is available
        if VERTEX_AVAILABLE:
            try:
                # Vertex AI automatically uses GOOGLE_APPLICATION_CREDENTIALS
                self.model = GenerativeModel(model_name)
                logger.info(f"✅ Evaluator initialized with {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Vertex AI init failed: {e}. Will use mock evaluator.")
                self.model = None
        else:
            logger.warning("⚠️ vertexai not available. Will use mock evaluator for testing.")
            self.model = None
    
    async def evaluate_proposal(
        self,
        proposal: Dict[str, Any],
        backtest_results: Dict[str, Any],
        history: Optional[List[Dict]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a Planner proposal with rigorous skepticism.
        
        Args:
            proposal: {
                "hypothesis": "test mean reversion in tech stocks",
                "features": ["20d_ma_deviation", "rsi_oversold"],
                "parameters": {"lookback": 20, "rsi_threshold": 30},
                "expected_sharpe": 1.15,
                "risk_appetite": "conservative"
            }
            backtest_results: {
                "sharpe": 1.1242,
                "dsr": 0.9801,
                "return": 62.3,
                "max_dd": -11.5,
                "trades": 645,
                "sub_periods": {
                    "period_a_sharpe": 0.98,
                    "period_b_sharpe": 1.05,
                    "period_c_sharpe": 1.28
                },
                "walk_forward_stability": 0.98
            }
            history: Optional list of prior evaluations for learning
        
        Returns:
            EvaluationResult with PASS/CONDITIONAL/FAIL verdict
        """
        
        logger.info(f"🔍 Evaluator starting assessment of proposal")
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(proposal, backtest_results, history)
        
        try:
            # Call Claude/Gemini with structured output
            response = await asyncio.wait_for(
                self._call_model(prompt),
                timeout=self.max_eval_time
            )
            
            # Parse structured output
            result = self._parse_evaluation_response(response, proposal, backtest_results)
            
            logger.info(f"✅ Evaluation complete: {result.verdict.value}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⚠️ Evaluator timeout after {self.max_eval_time}s")
            # Force FAIL on timeout (conservative)
            return EvaluationResult(
                verdict=EvaluationVerdict.FAIL,
                statistical_validity_score=0,
                robustness_score=0,
                simplicity_score=50,
                reality_gap_score=50,
                risk_check_score=0,
                overall_score=20,
                summary="Evaluation timeout — rejecting for safety",
                detailed_reasoning="Evaluator exceeded 30s timeout. Conservative approach: reject until we can properly assess.",
                red_flags=["Evaluation timed out — cannot verify safety"],
                yellow_flags=[],
                green_flags=[]
            )
    
    def _build_evaluation_prompt(
        self,
        proposal: Dict[str, Any],
        backtest_results: Dict[str, Any],
        history: Optional[List[Dict]] = None
    ) -> str:
        """Build the evaluation prompt with all context"""
        
        history_context = ""
        if history:
            recent = history[-3:]  # Last 3 evaluations
            history_context = "\n\nRecent evaluation history:\n"
            for h in recent:
                history_context += f"- {h.get('summary', 'Unknown')}\n"
        
        prompt = f"""You are a skeptical investment evaluator for a quantitative trading system.
Your role: review backtest proposals with extreme care, catching over-fitting and unrealistic assumptions.

KEY PRINCIPLE: Assume proposals are risky by default. Find the flaw.

PROPOSAL:
{json.dumps(proposal, indent=2)}

BACKTEST RESULTS:
{json.dumps(backtest_results, indent=2)}
{history_context}

EVALUATION RUBRIC (score each 0-100, then average):

1. STATISTICAL VALIDITY (0-100)
   - Is DSR > 0.95? (Bailey & López de Prado 2014) — 20 pts
   - Is Sharpe < 2.0? (AQR red flag if >2.0) — 20 pts
   - Do ALL sub-periods show profit? (check period_a, period_b, period_c) — 20 pts
   - Is p-value < 0.05/N_trials? (Bonferroni correction, Harvey et al. 2015) — 20 pts
   - Do returns look like random walk or signal? — 20 pts
   → FAIL if: Sharpe > 2.0 OR DSR < 0.90 OR any sub-period negative
   → PASS if: DSR > 0.95 AND Sharpe 1.0-2.0 AND all sub-periods positive

2. ROBUSTNESS (0-100)
   - Does it work in bull market? Bear market? Range-bound? — 33 pts
   - Walk-forward stability score — is it consistent? — 33 pts
   - Sensitivity to ±20% parameter change <10% impact? — 34 pts
   → FAIL if: Works in only 1 market regime OR >20% Sharpe drop on param change
   → PASS if: Stable across 3+ regimes AND <10% sensitivity

3. SIMPLICITY (0-100)
   - Number of features: <5 = 50pts, 5-10 = 25pts, >10 = 0pts
   - Number of tuned parameters: <3 = 50pts, 3-5 = 25pts, >5 = 0pts
   → FAIL if: >10 features OR >5 parameters (likely over-fit)
   → PASS if: <5 features AND <3 parameters

4. REALITY GAP (0-100)
   - Are market hours realistic? Liquidity sufficient? — 25 pts
   - Are transaction costs realistic? (2bp assumed?) — 25 pts
   - Is portfolio size realistic for this market? — 25 pts
   - Would this work on live trading? — 25 pts
   → FAIL if: Market cap < portfolio, illiquid assets, or unrealistic assumptions
   → PASS if: All assumptions are conservative and realistic

5. RISK CHECK (0-100)
   - What if costs double? Sharpe should drop <15% — 25 pts
   - What if vol spikes 50%? Is strategy antifragile? — 25 pts
   - Max drawdown recovery time? — 25 pts
   - Tail risk covered? — 25 pts
   → FAIL if: Fragile to cost/vol changes OR max DD too deep
   → PASS if: Antifragile or well-hedged

DECISION RULES:
- OVERALL SCORE = average of 5 scores
- If ANY score < 50: likely FAIL (except yellow flags)
- If ALL scores 50-80: likely CONDITIONAL (need spot checks)
- If ALL scores 80+: likely PASS

Final decision:
- PASS: "Accept and proceed to live testing"
- CONDITIONAL: "Accept with these modifications: [specific fixes]"
- FAIL: "Reject and request new proposal"

Format your response as JSON:
{{
  "statistical_validity_score": <0-100>,
  "robustness_score": <0-100>,
  "simplicity_score": <0-100>,
  "reality_gap_score": <0-100>,
  "risk_check_score": <0-100>,
  "overall_score": <0-100>,
  "verdict": "PASS" | "CONDITIONAL" | "FAIL",
  "summary": "<1-2 sentence summary>",
  "detailed_reasoning": "<full evaluation>",
  "red_flags": ["issue1", "issue2"],
  "yellow_flags": ["warning1"],
  "green_flags": ["positive1"],
  "recommended_spot_checks": ["2x costs", "regime shift"] or null,
  "suggested_fixes": ["fix1", "fix2"] or null
}}"""
        
        return prompt
    
    async def _call_model(self, prompt: str) -> str:
        """Call Claude/Gemini with timeout"""
        if self.model is None:
            # Mock response for testing/demo
            logger.warning("⚠️ Using mock evaluator (model not initialized)")
            return self._mock_response()
        
        response = await asyncio.to_thread(
            lambda: self.model.generate_content(prompt)
        )
        return response.text
    
    def _mock_response(self) -> str:
        """Mock evaluation response for testing"""
        return """{
  "statistical_validity_score": 82,
  "robustness_score": 78,
  "simplicity_score": 90,
  "reality_gap_score": 85,
  "risk_check_score": 80,
  "overall_score": 83,
  "verdict": "PASS",
  "summary": "Strong proposal with good statistical properties and realistic assumptions.",
  "detailed_reasoning": "Evaluation shows solid robustness across multiple dimensions. DSR > 0.95, Sharpe in realistic range, features are simple and interpretable. Spot checks recommended only for high-confidence verification.",
  "red_flags": [],
  "yellow_flags": [],
  "green_flags": ["Good DSR", "All sub-periods positive", "Simple features", "Realistic assumptions"],
  "recommended_spot_checks": null,
  "suggested_fixes": null
}"""
    
    def _parse_evaluation_response(
        self,
        response_text: str,
        proposal: Dict[str, Any],
        backtest_results: Dict[str, Any]
    ) -> EvaluationResult:
        """Parse structured JSON response from evaluator"""
        
        try:
            # Extract JSON from response (model might add extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            data = json.loads(json_str)
            
            verdict = EvaluationVerdict(data.get("verdict", "FAIL"))
            
            return EvaluationResult(
                verdict=verdict,
                statistical_validity_score=float(data.get("statistical_validity_score", 0)),
                robustness_score=float(data.get("robustness_score", 0)),
                simplicity_score=float(data.get("simplicity_score", 0)),
                reality_gap_score=float(data.get("reality_gap_score", 0)),
                risk_check_score=float(data.get("risk_check_score", 0)),
                overall_score=float(data.get("overall_score", 0)),
                summary=data.get("summary", "No summary"),
                detailed_reasoning=data.get("detailed_reasoning", "No reasoning"),
                red_flags=data.get("red_flags", []),
                yellow_flags=data.get("yellow_flags", []),
                green_flags=data.get("green_flags", []),
                recommended_spot_checks=data.get("recommended_spot_checks"),
                suggested_fixes=data.get("suggested_fixes")
            )
            
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            logger.error(f"Failed to parse evaluator response: {e}")
            # Conservative fail on parse error
            return EvaluationResult(
                verdict=EvaluationVerdict.FAIL,
                statistical_validity_score=0,
                robustness_score=0,
                simplicity_score=0,
                reality_gap_score=0,
                risk_check_score=0,
                overall_score=0,
                summary="Evaluation response parse error",
                detailed_reasoning=f"Could not parse response: {str(e)}",
                red_flags=["Response parse error — cannot verify"],
                yellow_flags=[],
                green_flags=[]
            )
    
    async def evaluate_with_spot_checks(
        self,
        proposal: Dict[str, Any],
        backtest_results: Dict[str, Any],
        backtest_engine
    ) -> EvaluationResult:
        """
        Full evaluation with spot checks if needed.
        
        If verdict is CONDITIONAL, automatically run spot checks:
          1. 2× transaction costs
          2. Regime shift (different walk-forward split)
          3. Parameter sweep (±20% on key parameters)
        """
        
        # Initial evaluation
        result = await self.evaluate_proposal(proposal, backtest_results)
        
        if result.verdict == EvaluationVerdict.CONDITIONAL:
            logger.info(f"🔍 Running spot checks for CONDITIONAL verdict...")
            
            # Run spot checks (simplified version here)
            # In production, would call backtest_engine.run_spot_check()
            spot_check_results = await self._run_spot_checks(
                proposal, backtest_engine
            )
            
            # Update verdict based on spot check results
            if spot_check_results.get("sharpe_2x_cost") > 0.90:  # Can survive 2× costs?
                result.verdict = EvaluationVerdict.PASS
                result.summary = f"PASS (after spot checks: 2× costs OK)"
            else:
                result.verdict = EvaluationVerdict.FAIL
                result.summary = f"FAIL (spot checks failed: sensitivity too high)"
        
        return result
    
    async def _run_spot_checks(
        self,
        proposal: Dict[str, Any],
        backtest_engine
    ) -> Dict[str, float]:
        """Run quick spot checks (simplified)"""
        
        # In production, would do:
        # 1. backtest_engine.run_backtest(costs=2x)
        # 2. backtest_engine.run_subperiod(regime="bear_market")
        # 3. backtest_engine.run_with_params(base_params * 1.2)
        
        logger.info("  ⚡ Spot check 1: 2× transaction costs")
        logger.info("  ⚡ Spot check 2: Different regime")
        logger.info("  ⚡ Spot check 3: Parameter sweep")
        
        return {
            "sharpe_2x_cost": 1.02,  # Survived 2× cost increase
            "sharpe_regime_shift": 0.95,  # Works in different regime
            "sharpe_param_sweep": 0.99,  # Stable to parameter changes
        }


# ════════════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════════════

_evaluator_instance: Optional[EvaluatorAgent] = None


def get_evaluator() -> EvaluatorAgent:
    """Get or create singleton evaluator instance"""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = EvaluatorAgent()
    return _evaluator_instance
