# Sprint Contract -- Cycle 28
Generated: 2026-04-16T13:00:00Z

## Target
Phase 4.4.5.5: Documentation -- "How to trade pyfinAgent signals" guide for Peder

## Hypothesis
Write `docs/TRADING_GUIDE.md` covering the 5 required topics from the checklist:
1. Signal anatomy (fields, what they mean)
2. Confidence thresholds (how to interpret confidence scores)
3. Sizing (how position sizes are determined)
4. Stop-loss execution (how stops work, when they fire)
5. When to override Ford (override criteria, human judgment)

Create a verification drill that confirms all 5 topics are present and substantive.
Flip the checklist item. Peder's sign-off (Slack acknowledgement) is a separate gate
noted in the evidence line.

## Success Criteria
- SC1: `docs/TRADING_GUIDE.md` exists and is non-empty
- SC2: Guide contains a section on signal anatomy with field descriptions
- SC3: Guide contains a section on confidence thresholds with numeric ranges
- SC4: Guide contains a section on sizing with the formula or rules
- SC5: Guide contains a section on stop-loss execution with trigger conditions
- SC6: Guide contains a section on when to override Ford
- SC7: All field names, thresholds, and limits match production code values
- SC8: Drill `scripts/go_live_drills/trading_guide_test.py` exits 0
- SC9: Checklist item 4.4.5.5 flipped to `[x]` with evidence line
- SC10: Guide is written for a non-technical trader (Peder), not for engineers

## Research Gate
WAIVED per pure-doc rule. All source material comes from production code already
explored via subagent (signals_server.py, portfolio_manager.py, formatters.py,
paper_trader.py). No external research needed.

## Excluded
- Peder's sign-off (separate human gate)
- Changes to any backend/*.py file
- Changes to masterplan.json
