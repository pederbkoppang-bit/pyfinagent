# Research Brief — phase-30.3 P1: Connect stop-loss exits to learn loop

**Tier:** complex | **Effort:** max | **Date:** 2026-05-19

## Scope
One-line append: add `closed_tickers.append(sl_ticker)` as a sibling
to `summary["stop_loss_triggered"].append(sl_ticker)` at
`backend/services/autonomous_loop.py:771`. Initialization-order
side question: `closed_tickers = []` currently lives inside Step 7
at `:838`; for Step 5.6 (line 757-777) to append, the initialization
must be hoisted above Step 5.6.

Audit basis: phase-30.0 `experiment_results.md` Stage 12 (FAIL):
`agent_memories` and `outcome_tracking` BQ tables empty (0 rows
since 2026-04-13 creation) despite 3 closed round trips. Stop-out
exits never reach `_learn_from_closed_trades`.

## Status
WRITE-FIRST skeleton — sections appended below in order.

## TOC
1. Internal code inventory (file:line) — phase-30.3 anchors
2. Q1: Audit diagnosis confirmation
3. Read in full table (>=5 required)
4. Snippet-only table
5. Recency scan (last 2 years, 2024-2026)
6. Search-query composition discipline
7. Q2: External best-practice — learning from stop-outs
8. Q3: Initialization-order subtlety
9. Q4: Test design
10. Q5: Live-check deferral
11. Application to phase-30.3 (one-liner location + init hoist)
12. Research Gate Checklist
13. JSON envelope
