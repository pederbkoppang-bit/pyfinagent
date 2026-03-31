#!/usr/bin/env python3
"""
Test message routing classification for Phase 3.2.1
Tests the detect_question_type() function with sample questions
"""

def detect_question_type(message: str) -> str:
    """
    Classify incoming message as operational, analytical, or research.
    """
    analytical_keywords = [
        "why", "should", "explain", "analyze", "trade-off", "decision",
        "regression", "sharpe", "recommendation", "recommend", "suggest", "improve",
        "compare", "better", "worse", "review", "feedback", "thoughts"
    ]
    
    research_keywords = [
        "research", "paper", "literature", "evidence",
        "experiment", "hypothesis", "theory", "mechanism", "solution",
        "study", "baseline", "benchmark", "findings",
        "investigate", "explore", "discover"
    ]
    
    msg_lower = message.lower()
    
    # Check analytical first (more general)
    if any(kw in msg_lower for kw in analytical_keywords):
        return "analytical"
    
    # Check research (more specific)
    if any(kw in msg_lower for kw in research_keywords):
        return "research"
    
    # Default to operational
    return "operational"


# Test cases
test_cases = [
    # Operational
    ("What's the status?", "operational"),
    ("Start the harness", "operational"),
    ("Check services", "operational"),
    ("Next step?", "operational"),
    ("Commit changes", "operational"),
    
    # Analytical
    ("Why did Sharpe drop?", "analytical"),
    ("Should we refactor this?", "analytical"),
    ("Explain the regression", "analytical"),
    ("Compare these two approaches", "analytical"),
    ("What do you recommend?", "analytical"),
    ("How would you improve this?", "analytical"),
    ("Is this design better?", "analytical"),
    ("Give me feedback on the code", "analytical"),
    
    # Research
    ("Research how to implement regime detection", "research"),
    ("Find papers on HMM in finance", "research"),
    ("What does the literature say about this?", "research"),
    ("Investigate novel approaches for feature engineering", "research"),
    ("Explore evidence-based methods for optimization", "research"),
    ("Study the baseline implementation", "research"),
    ("What are the findings from this study?", "research"),
    ("Discover new approaches to portfolio optimization", "research"),
]

print("=" * 80)
print("Phase 3.2.1: Message Routing Classification Tests")
print("=" * 80)

passed = 0
failed = 0

for message, expected in test_cases:
    result = detect_question_type(message)
    status = "✅ PASS" if result == expected else "❌ FAIL"
    
    if result == expected:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | Expected: {expected:15} | Got: {result:15} | Message: {message}")

print("=" * 80)
print(f"Results: {passed}/{len(test_cases)} passed ({100*passed//len(test_cases)}%)")
print("=" * 80)

if failed == 0:
    print("✅ All tests passed!")
    exit(0)
else:
    print(f"❌ {failed} tests failed")
    exit(1)
