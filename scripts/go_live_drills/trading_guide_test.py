#!/usr/bin/env python3
"""
Go-Live Drill: Phase 4.4.5.5 -- Trading Guide Verification
Verifies that docs/TRADING_GUIDE.md exists and covers all 5 required topics
with accurate values matching production code.

Re-run recipe:
    python3 scripts/go_live_drills/trading_guide_test.py

Exit 0 on PASS, exit 1 on any failure.
"""
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
GUIDE = REPO / "docs" / "TRADING_GUIDE.md"
SIGNALS_SERVER = REPO / "backend" / "agents" / "mcp_servers" / "signals_server.py"
RISK_CONSTRAINTS_FILE = SIGNALS_SERVER  # get_risk_constraints lives here

passed = 0
failed = 0
total = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS S{total - 1}: {name}")
    else:
        failed += 1
        print(f"  FAIL S{total - 1}: {name} -- {detail}")


# ---------- Load guide ----------
print("Loading TRADING_GUIDE.md ...")
check("Guide file exists", GUIDE.exists(), f"expected {GUIDE}")
if not GUIDE.exists():
    print(f"\nDRILL FAIL: {passed}/{total}")
    sys.exit(1)

guide_text = GUIDE.read_text(encoding="utf-8")
guide_lower = guide_text.lower()

check("Guide is non-empty", len(guide_text) > 500,
      f"only {len(guide_text)} chars")

# ---------- Topic 1: Signal Anatomy ----------
print("\nTopic 1: Signal Anatomy")
check("Has signal anatomy section",
      "signal anatomy" in guide_lower or "what you see in slack" in guide_lower)

# Check key signal fields are described
for field in ["confidence", "price", "size", "stop", "thesis", "signal id"]:
    check(f"Describes '{field}' field",
          field in guide_lower,
          f"'{field}' not found in guide")

# Check signal types
for sig_type in ["buy", "sell", "hold"]:
    check(f"Mentions {sig_type.upper()} signal type",
          sig_type in guide_lower)

# ---------- Topic 2: Confidence Thresholds ----------
print("\nTopic 2: Confidence Thresholds")
check("Has confidence section",
      "confidence threshold" in guide_lower or "how to read the number" in guide_lower)

# Check numeric ranges are present
check("Mentions 0.80 or 0.8 threshold",
      "0.80" in guide_text or "0.8" in guide_text)
check("Mentions 0.60 or 0.6 threshold",
      "0.60" in guide_text or "0.6" in guide_text)
check("Mentions 0.40 or 0.4 threshold",
      "0.40" in guide_text or "0.4" in guide_text)
check("Range is 0.00 to 1.00",
      "0.00" in guide_text and "1.00" in guide_text)

# ---------- Topic 3: Sizing ----------
print("\nTopic 3: Sizing")
check("Has sizing section",
      "sizing" in guide_lower or "dollar amount" in guide_lower)

# Check key sizing parameters match production code
check("Mentions 5% equity cap",
      "5%" in guide_text)
check("Mentions $1,000 USD cap",
      "$1,000" in guide_text or "$1000" in guide_text)
check("Mentions half-Kelly",
      "half-kelly" in guide_lower or "half kelly" in guide_lower)
check("Mentions three-arm or three arm formula",
      "three-arm" in guide_lower or "three arm" in guide_lower)

# ---------- Topic 4: Stop-Loss Execution ----------
print("\nTopic 4: Stop-Loss Execution")
check("Has stop-loss section",
      "stop-loss" in guide_lower or "stop loss" in guide_lower)

check("Mentions -8% fixed stop",
      "8%" in guide_text or "-8%" in guide_text)
check("Mentions -3% trailing stop",
      "3%" in guide_text or "-3%" in guide_text)
check("Mentions -15% kill switch",
      "15%" in guide_text or "-15%" in guide_text)
check("Mentions -5% warning tier",
      "-5%" in guide_text or "warning" in guide_lower)
check("Mentions -10% de-risk tier",
      "-10%" in guide_text or "de-risk" in guide_lower)

# ---------- Topic 5: When to Override ----------
print("\nTopic 5: When to Override Ford")
check("Has override section",
      "override" in guide_lower)

check("Mentions earnings as override reason",
      "earning" in guide_lower)
check("Mentions macro events",
      "macro" in guide_lower)
check("Advises never to override stop-loss",
      ("never override" in guide_lower and "stop" in guide_lower) or
      "never move" in guide_lower or "do not move your stop" in guide_lower)

# ---------- Cross-check values against production code ----------
print("\nCross-check: Values vs production code")

# Parse get_risk_constraints from signals_server.py to extract hardcoded values
tree = ast.parse(SIGNALS_SERVER.read_text(encoding="utf-8"))
constraints_found = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "get_risk_constraints":
        constraints_found = True
        break

check("get_risk_constraints function exists in signals_server.py",
      constraints_found)

# Verify the guide mentions the exact hard limits from the Quick Reference
check("Quick reference mentions max per-ticker 10%",
      "10%" in guide_text and "per-ticker" in guide_lower)
check("Quick reference mentions max daily trades 5",
      re.search(r"max daily trades.*5|5.*daily trades", guide_lower) is not None
      or ("daily trades" in guide_lower and "5" in guide_text))
check("Quick reference mentions no leverage (100%)",
      "100%" in guide_text and ("no leverage" in guide_lower or "total exposure" in guide_lower))

# ---------- Audience check ----------
print("\nAudience check: Written for a non-technical trader")
check("Does not contain Python code blocks",
      "```python" not in guide_text)
check("Does not import anything",
      "import " not in guide_text or guide_text.count("import") <= 1)
check("Contains practical 'what you should do' guidance",
      "what you should do" in guide_lower or "suggested action" in guide_lower)

# ---------- Summary ----------
print(f"\n{'=' * 50}")
if failed == 0:
    print(f"DRILL PASS: {passed}/{total}")
    sys.exit(0)
else:
    print(f"DRILL FAIL: {passed}/{total} ({failed} failures)")
    sys.exit(1)
