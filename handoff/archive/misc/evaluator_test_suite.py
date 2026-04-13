"""
Test Suite for LLM-as-Evaluator (Phase 3.2 EVALUATE)

10 test proposals with known characteristics:
- 5 "GOOD" proposals (should be PASS)
- 5 "BAD" proposals (should be FAIL or CONDITIONAL)

Purpose: Measure evaluator accuracy, detection rate, false positives
"""

import asyncio
import sys
sys.path.insert(0, '/Users/ford/.openclaw/workspace/pyfinagent')

from backend.agents.evaluator_agent import get_evaluator, EvaluationVerdict

# ═══════════════════════════════════════════════════════════════════
# TEST PROPOSALS
# ═══════════════════════════════════════════════════════════════════

GOOD_PROPOSALS = [
    {
        "name": "T1_MeanReversion_RSI",
        "proposal": {
            "hypothesis": "Mean reversion in large-cap tech on 20-day oversold RSI",
            "features": ["20d_ma_deviation", "rsi_oversold"],
            "parameters": {"lookback": 20, "rsi_threshold": 30},
            "expected_sharpe": 1.15,
            "risk_appetite": "conservative"
        },
        "backtest_results": {
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
        },
        "expected_verdict": EvaluationVerdict.PASS
    },
    {
        "name": "T2_MomentumFollower",
        "proposal": {
            "hypothesis": "Momentum following in mid-cap growth stocks",
            "features": ["30d_price_trend"],
            "parameters": {"momentum_window": 30},
            "expected_sharpe": 1.0,
            "risk_appetite": "moderate"
        },
        "backtest_results": {
            "sharpe": 1.0534,
            "dsr": 0.9650,
            "return": 51.2,
            "max_dd": -13.2,
            "trades": 423,
            "sub_periods": {
                "period_a_sharpe": 0.91,
                "period_b_sharpe": 1.02,
                "period_c_sharpe": 1.15
            },
            "walk_forward_stability": 0.94
        },
        "expected_verdict": EvaluationVerdict.PASS
    },
    {
        "name": "T3_SimpleMovingAverage",
        "proposal": {
            "hypothesis": "Golden cross / dead cross on large-cap indices",
            "features": ["50d_200d_cross"],
            "parameters": {"fast": 50, "slow": 200},
            "expected_sharpe": 0.95,
            "risk_appetite": "conservative"
        },
        "backtest_results": {
            "sharpe": 0.9612,
            "dsr": 0.9421,
            "return": 48.5,
            "max_dd": -14.1,
            "trades": 156,
            "sub_periods": {
                "period_a_sharpe": 0.85,
                "period_b_sharpe": 0.98,
                "period_c_sharpe": 1.05
            },
            "walk_forward_stability": 0.96
        },
        "expected_verdict": EvaluationVerdict.PASS
    },
    {
        "name": "T4_VolatilityReversion",
        "proposal": {
            "hypothesis": "Buy when VIX spikes, sell when normalizes",
            "features": ["vix_spike", "vix_normalized"],
            "parameters": {"vix_entry": 25, "vix_exit": 18},
            "expected_sharpe": 1.08,
            "risk_appetite": "moderate"
        },
        "backtest_results": {
            "sharpe": 1.0723,
            "dsr": 0.9745,
            "return": 54.1,
            "max_dd": -12.3,
            "trades": 87,
            "sub_periods": {
                "period_a_sharpe": 0.95,
                "period_b_sharpe": 1.08,
                "period_c_sharpe": 1.15
            },
            "walk_forward_stability": 0.97
        },
        "expected_verdict": EvaluationVerdict.PASS
    },
    {
        "name": "T5_DividendGrowth",
        "proposal": {
            "hypothesis": "Buy dividend aristocrats with consistent growth",
            "features": ["dividend_yield", "div_growth_5yr"],
            "parameters": {"min_yield": 2.5, "min_growth": 0.07},
            "expected_sharpe": 1.12,
            "risk_appetite": "conservative"
        },
        "backtest_results": {
            "sharpe": 1.1042,
            "dsr": 0.9834,
            "return": 59.7,
            "max_dd": -10.8,
            "trades": 234,
            "sub_periods": {
                "period_a_sharpe": 1.01,
                "period_b_sharpe": 1.10,
                "period_c_sharpe": 1.20
            },
            "walk_forward_stability": 0.98
        },
        "expected_verdict": EvaluationVerdict.PASS
    }
]

BAD_PROPOSALS = [
    {
        "name": "B1_Overfit_HighSharpe",
        "proposal": {
            "hypothesis": "Complex ensemble of 15 indicators with overfitting risk",
            "features": [
                "ma_5", "ma_10", "ma_20", "rsi", "macd", "atr", "cci",
                "adx", "stoch", "williams_r", "obv", "vpt", "mfi", "ao", "kc"
            ],
            "parameters": {
                "ma5": 5, "ma10": 10, "ma20": 20, "rsi_thresh": 30,
                "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
                "atr_period": 14, "atr_mult": 2.0
            },
            "expected_sharpe": 2.5,  # 🚩 RED FLAG: Too high
            "risk_appetite": "aggressive"
        },
        "backtest_results": {
            "sharpe": 2.4156,  # 🚩 > 2.0 = AQR red flag
            "dsr": 0.7823,  # 🚩 < 0.95 = overfit
            "return": 142.3,  # 🚩 Unrealistically high
            "max_dd": -8.2,
            "trades": 3242,  # 🚩 Way too many trades
            "sub_periods": {
                "period_a_sharpe": 2.35,
                "period_b_sharpe": 2.41,
                "period_c_sharpe": 2.50  # 🚩 "Improving" = overfitting
            },
            "walk_forward_stability": 0.72  # 🚩 Low stability
        },
        "expected_verdict": EvaluationVerdict.FAIL
    },
    {
        "name": "B2_NegativeSubPeriod",
        "proposal": {
            "hypothesis": "Mean reversion in micro-cap stocks",
            "features": ["micro_cap_reversion"],
            "parameters": {"lookback": 10},
            "expected_sharpe": 1.05,
            "risk_appetite": "aggressive"
        },
        "backtest_results": {
            "sharpe": 1.0542,
            "dsr": 0.9512,
            "return": 42.1,
            "max_dd": -15.3,
            "trades": 456,
            "sub_periods": {
                "period_a_sharpe": 0.92,
                "period_b_sharpe": -0.15,  # 🚩 NEGATIVE = FAIL
                "period_c_sharpe": 1.28
            },
            "walk_forward_stability": 0.71  # 🚩 Low stability
        },
        "expected_verdict": EvaluationVerdict.FAIL
    },
    {
        "name": "B3_HighSensitivity",
        "proposal": {
            "hypothesis": "Very tight stop-loss trading",
            "features": ["tight_stops"],
            "parameters": {"stop_pct": 1.0},  # 🚩 1% stop = too tight
            "expected_sharpe": 1.20,
            "risk_appetite": "aggressive"
        },
        "backtest_results": {
            "sharpe": 1.1823,
            "dsr": 0.9401,
            "return": 61.4,
            "max_dd": -11.2,
            "trades": 2845,  # 🚩 Excessive trades from tight stops
            "sub_periods": {
                "period_a_sharpe": 1.18,
                "period_b_sharpe": 1.12,
                "period_c_sharpe": 1.25
            },
            "walk_forward_stability": 0.68  # 🚩 Only 68% stable
        },
        "expected_verdict": EvaluationVerdict.FAIL
    },
    {
        "name": "B4_LowDSR",
        "proposal": {
            "hypothesis": "Data-mining result from parameter search",
            "features": ["parameter_optimized"],
            "parameters": {"param1": 12.7, "param2": 3.14},
            "expected_sharpe": 1.35,
            "risk_appetite": "moderate"
        },
        "backtest_results": {
            "sharpe": 1.3456,
            "dsr": 0.8234,  # 🚩 < 0.95 = likely overfit
            "return": 71.2,
            "max_dd": -12.5,
            "trades": 567,
            "sub_periods": {
                "period_a_sharpe": 1.32,
                "period_b_sharpe": 1.35,
                "period_c_sharpe": 1.38  # 🚩 "Consistent" but low DSR suggests data-mining
            },
            "walk_forward_stability": 0.85
        },
        "expected_verdict": EvaluationVerdict.FAIL
    },
    {
        "name": "B5_UnrealisticCosts",
        "proposal": {
            "hypothesis": "High-frequency micro-trading (unrealistic for retail)",
            "features": ["hft_microtrades"],
            "parameters": {"trade_interval_ms": 100},
            "expected_sharpe": 1.15,
            "risk_appetite": "aggressive"
        },
        "backtest_results": {
            "sharpe": 1.1523,
            "dsr": 0.9634,
            "return": 58.2,
            "max_dd": -11.1,
            "trades": 12543,  # 🚩 HFT-level trading
            "sub_periods": {
                "period_a_sharpe": 1.14,
                "period_b_sharpe": 1.15,
                "period_c_sharpe": 1.17
            },
            "walk_forward_stability": 0.96
        },
        "expected_verdict": EvaluationVerdict.FAIL  # Reality gap: assumes market maker liquidity
    }
]


async def run_test_suite():
    """Run evaluator on all 10 test proposals"""
    
    evaluator = get_evaluator()
    
    results = {
        "good": [],
        "bad": [],
        "overall_accuracy": 0,
        "detection_rate": 0,
        "false_positive_rate": 0
    }
    
    print("\n" + "="*80)
    print("PHASE 3.2 EVALUATE: LLM-as-Evaluator Test Suite")
    print("="*80 + "\n")
    
    # Test GOOD proposals
    print("🟢 TESTING GOOD PROPOSALS (should be PASS)\n")
    good_correct = 0
    for test in GOOD_PROPOSALS:
        print(f"  {test['name']}...")
        result = await evaluator.evaluate_proposal(test["proposal"], test["backtest_results"])
        
        correct = result.verdict == test["expected_verdict"]
        if correct:
            good_correct += 1
            print(f"    ✅ {result.verdict.value} (correct) — Score: {result.overall_score}/100")
        else:
            print(f"    ❌ {result.verdict.value} (expected {test['expected_verdict'].value})")
        
        results["good"].append({
            "name": test["name"],
            "expected": test["expected_verdict"].value,
            "actual": result.verdict.value,
            "correct": correct,
            "score": result.overall_score,
            "flags": {
                "red": result.red_flags,
                "yellow": result.yellow_flags,
                "green": result.green_flags
            }
        })
    
    print(f"\n  Good proposals: {good_correct}/{len(GOOD_PROPOSALS)} correct\n")
    
    # Test BAD proposals
    print("🔴 TESTING BAD PROPOSALS (should be FAIL)\n")
    bad_correct = 0
    for test in BAD_PROPOSALS:
        print(f"  {test['name']}...")
        result = await evaluator.evaluate_proposal(test["proposal"], test["backtest_results"])
        
        correct = result.verdict == test["expected_verdict"]
        if correct:
            bad_correct += 1
            print(f"    ✅ {result.verdict.value} (correct) — Score: {result.overall_score}/100")
        else:
            print(f"    ❌ {result.verdict.value} (expected {test['expected_verdict'].value})")
        
        # Show why it should fail
        if result.red_flags:
            print(f"    Red flags: {', '.join(result.red_flags[:2])}")
        
        results["bad"].append({
            "name": test["name"],
            "expected": test["expected_verdict"].value,
            "actual": result.verdict.value,
            "correct": correct,
            "score": result.overall_score,
            "flags": {
                "red": result.red_flags,
                "yellow": result.yellow_flags,
                "green": result.green_flags
            }
        })
    
    print(f"\n  Bad proposals: {bad_correct}/{len(BAD_PROPOSALS)} correct\n")
    
    # Calculate metrics
    total_correct = good_correct + bad_correct
    total_tests = len(GOOD_PROPOSALS) + len(BAD_PROPOSALS)
    overall_accuracy = total_correct / total_tests
    detection_rate = bad_correct / len(BAD_PROPOSALS)  # How many bad ones we caught
    false_positive_rate = (len(GOOD_PROPOSALS) - good_correct) / len(GOOD_PROPOSALS)  # How many good ones we rejected
    
    results["overall_accuracy"] = overall_accuracy
    results["detection_rate"] = detection_rate
    results["false_positive_rate"] = false_positive_rate
    
    # Print summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    print(f"Overall Accuracy:     {overall_accuracy:.1%} ({total_correct}/{total_tests})")
    print(f"Detection Rate:       {detection_rate:.1%} (bad proposals caught)")
    print(f"False Positive Rate:  {false_positive_rate:.1%} (good proposals rejected)\n")
    
    # Pass/Fail criteria
    print("EVALUATION CRITERIA:")
    print(f"  Overall Accuracy ≥ 80%:        {'✅ PASS' if overall_accuracy >= 0.80 else '❌ FAIL'}")
    print(f"  Detection Rate ≥ 80%:          {'✅ PASS' if detection_rate >= 0.80 else '❌ FAIL'}")
    print(f"  False Positive Rate ≤ 10%:     {'✅ PASS' if false_positive_rate <= 0.10 else '❌ FAIL'}")
    
    print("\n" + "="*80)
    if overall_accuracy >= 0.80 and detection_rate >= 0.80 and false_positive_rate <= 0.10:
        print("VERDICT: ✅ PASS — Evaluator is ready for production")
    else:
        print("VERDICT: ⚠️ CONDITIONAL — Needs tuning or more testing")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    results = asyncio.run(run_test_suite())
