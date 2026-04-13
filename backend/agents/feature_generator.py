"""
Phase 2.10: Autonomous Feature Discovery via LLM-Guided Feature Generator

Pattern: Karpathy AutoResearch + FAMOSE ReAct loop
- LLM proposes technical indicators, factor combinations, sentiment signals
- Each proposal tested on 2-week holdout (2024-12-16 to 2024-12-31)
- Keeps features with Sharpe improvement ≥ +0.02 AND DSR ≥ 0.95
- Prunes redundant features via mRMR (minimal redundancy, maximal relevance)

Execution: run_harness.py calls this when optimizer plateaus (10+ consecutive discards)
"""

import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
FEATURE_PROPOSAL_BUDGET = 300  # seconds per proposal
SHARPE_THRESHOLD = 0.02  # minimum improvement to keep feature
DSR_THRESHOLD = 0.95  # Bailey & López de Prado statistical significance
CORRELATION_THRESHOLD = 0.85  # mRMR: reject if correlated with existing features
MAX_PROPOSALS_PER_BATCH = 5
AVAILABLE_DATA_SOURCES = [
    "prices (OHLCV)",
    "fundamentals (P/E, P/B, ROE, debt)",
    "macro (VIX, yield curve, GDP)",
    "sector_momentum (relative strength vs SPY)",
    "earnings_surprises (actual vs forecast)",
    "analyst_revisions (upgrade/downgrade ratio)",
    "insider_trading (buy/sell ratio)",
    "options_flow (put/call ratio, implied vol)",
]


def propose_features_batch(
    n_proposals: int = MAX_PROPOSALS_PER_BATCH,
    llm_client=None,
    current_features: Optional[list] = None,
) -> list[dict]:
    """
    Use LLM (Claude via MCP) to propose new trading features.
    
    Returns list of proposed features:
    [
        {
            "name": "momentum_3m_tech",
            "description": "3-month momentum for tech sector stocks",
            "formula": "rolling mean of (close / close[90 days ago] - 1)",
            "data_sources": ["prices"],
            "risk_factors": "May be correlated with market momentum",
        },
        ...
    ]
    """
    if llm_client is None:
        logger.warning("LLM client not provided; using stub proposals")
        return _stub_proposals()

    prompt = f"""You are a feature engineering expert for quantitative trading.

Available data sources:
{json.dumps(AVAILABLE_DATA_SOURCES, indent=2)}

Current features in the strategy (to avoid duplicates):
{json.dumps(current_features or [], indent=2)}

Task: Propose {n_proposals} NEW technical indicators or factor combinations that might improve trading signal quality.

For each proposal, provide:
- name: short identifier (e.g., "momentum_3m_tech")
- description: what the feature captures
- formula: how to calculate it from available data
- data_sources: which of the available sources does it use?
- risk_factors: what could go wrong with this feature?

Constraints:
- Only use the available data sources listed above
- Don't propose features that already exist in the current feature list
- Avoid overly complex features (formula should fit in 1-2 lines of code)
- Features should be interpretable (a human quant could explain why it might work)

Return ONLY valid JSON list with no extra text.
"""

    # Call LLM via MCP (MCP connector defined in Phase 3+)
    # For now, stub implementation
    response = {"proposals": _stub_proposals()}
    
    try:
        proposals = response.get("proposals", [])
        logger.info(f"LLM proposed {len(proposals)} features")
        return proposals
    except Exception as e:
        logger.error(f"LLM proposal failed: {e}")
        return _stub_proposals()


def evaluate_feature(
    feature: dict,
    engine,  # BacktestEngine instance
    existing_features_df=None,  # DataFrame of current feature values for correlation check
) -> dict:
    """
    Test a proposed feature on holdout period (2024-12-16 to 2024-12-31).
    
    Returns:
    {
        "feature_name": "momentum_3m_tech",
        "in_sample_sharpe": 1.05,
        "out_of_sample_sharpe": 0.98,
        "dsr": 0.96,
        "correlation_with_existing": 0.42,
        "verdict": "KEEP",  # or "REJECT"
        "reason": "DSR > 0.95, Sharpe improvement +0.03 vs baseline",
    }
    """
    # TODO: Implement backtest with feature on 2024-12-16 to 2024-12-31
    # Return feature evaluation results
    
    return {
        "feature_name": feature.get("name"),
        "in_sample_sharpe": 0.0,  # placeholder
        "out_of_sample_sharpe": 0.0,  # placeholder
        "dsr": 0.0,  # placeholder
        "correlation_with_existing": 0.0,  # placeholder
        "verdict": "PENDING",
        "reason": "Implementation in progress",
    }


def accept_feature(evaluation: dict) -> bool:
    """Decide whether to keep a feature based on evaluation results."""
    if evaluation["verdict"] != "PENDING":
        # Already evaluated in previous run
        return evaluation["verdict"] == "KEEP"

    dsr = evaluation.get("dsr", 0)
    correlation = evaluation.get("correlation_with_existing", 1.0)
    
    # Check all criteria (Bailey & López de Prado, FAMOSE)
    passed_dsr = dsr >= DSR_THRESHOLD
    passed_correlation = correlation < CORRELATION_THRESHOLD
    
    # Sharpe improvement threshold (relative to baseline 1.17)
    sharpe_delta = evaluation.get("out_of_sample_sharpe", 0) - 1.17
    passed_sharpe = sharpe_delta >= SHARPE_THRESHOLD

    accept = passed_dsr and passed_correlation and passed_sharpe
    
    logger.info(
        f"Feature {evaluation['feature_name']}: "
        f"DSR={dsr:.3f}({'✓' if passed_dsr else '✗'}), "
        f"Correlation={correlation:.3f}({'✓' if passed_correlation else '✗'}), "
        f"Sharpe delta={sharpe_delta:+.3f}({'✓' if passed_sharpe else '✗'}) "
        f"→ {'ACCEPT' if accept else 'REJECT'}"
    )
    
    return accept


def log_proposal(feature: dict, evaluation: dict) -> None:
    """Append to feature discovery audit trail."""
    log_file = Path(__file__).parent.parent.parent / "handoff" / "data" / "feature_discovery_log.tsv"
    
    # CSV header: timestamp, feature_name, proposal_source, test_result, sharpe_delta, verdict, reason
    # Implemented in Phase 2.10 execution


def _stub_proposals() -> list[dict]:
    """Placeholder proposals for development."""
    return [
        {
            "name": "momentum_12m",
            "description": "12-month price momentum",
            "formula": "close / close[252 days ago] - 1",
            "data_sources": ["prices"],
            "risk_factors": "Known to reverse in downturns",
        },
        {
            "name": "value_factor",
            "description": "Low P/E plus high dividend yield",
            "formula": "(10 / P/E ratio) + (dividend yield * 2)",
            "data_sources": ["fundamentals"],
            "risk_factors": "Value trap risk in declining sectors",
        },
    ]


if __name__ == "__main__":
    # Dry run: propose features, don't evaluate
    proposals = propose_features_batch(n_proposals=3)
    print(json.dumps(proposals, indent=2))
