#!/bin/bash
# Quick check: did Seed 2026 complete? What's the verdict?

RESULT_FILE="handoff/seed_2026_result.json"

if [ ! -f "$RESULT_FILE" ]; then
    echo "❌ Seed 2026 not complete yet (result file missing)"
    exit 1
fi

# Parse result
SHARPE=$(jq '.sharpe' "$RESULT_FILE" 2>/dev/null || echo "0")
RETURN=$(jq '.return_pct' "$RESULT_FILE" 2>/dev/null || echo "0")
MAXDD=$(jq '.max_drawdown_pct' "$RESULT_FILE" 2>/dev/null || echo "0")
TRADES=$(jq '.num_trades' "$RESULT_FILE" 2>/dev/null || echo "0")

BASELINE_SHARPE=1.1705
TOLERANCE=0.02
THRESHOLD=$(echo "$BASELINE_SHARPE - $TOLERANCE * $BASELINE_SHARPE" | bc)

echo "Seed 2026 Result:"
echo "  Sharpe: $SHARPE (baseline: $BASELINE_SHARPE, threshold: $THRESHOLD)"
echo "  Return: $RETURN%"
echo "  MaxDD: $MAXDD%"
echo "  Trades: $TRADES"

# Check if passes (within ±2%)
if (( $(echo "$SHARPE >= $THRESHOLD" | bc -l) )); then
    echo "✅ PASS (Sharpe within ±2% of baseline)"
    exit 0
else
    echo "❌ CONDITIONAL (Sharpe regression > 2%)"
    exit 1
fi
