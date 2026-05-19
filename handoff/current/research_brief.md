# Research Brief — phase-30.2 P1: Wire `backfill_missing_stops` into autonomous_loop Step 5.6

**Tier:** complex | **Effort:** max | **Date:** 2026-05-19

## Scope
One-line wiring change: insert `trader.backfill_missing_stops()`
immediately BEFORE `trader.check_stop_losses()` in
`backend/services/autonomous_loop.py` Step 5.6 (line ~756-777).
The helper already exists at `paper_trader.py:465-532` (phase-25.2,
zero production callers per phase-30.0 audit). 7-of-11 open
positions have `stop_loss_price=NULL` (audit basis:
`handoff/archive/phase-30.0/experiment_results.md` Stage 7).

## Status
WRITE-FIRST skeleton — sections will be appended below.

## TOC
1. Read in full table (>=5 required)
2. Snippet-only table
3. Recency scan (last 2 years, 2024-2026)
4. Search-query composition discipline
5. Key findings (external)
6. Internal code inventory (file:line)
7. Q1: External best-practice for retrofitting stop-loss
8. Q2: Idempotency
9. Q3: Ordering (backfill before check)
10. Q4: Test design
11. Q5: Live-check deferral
12. Research Gate Checklist
13. JSON envelope
