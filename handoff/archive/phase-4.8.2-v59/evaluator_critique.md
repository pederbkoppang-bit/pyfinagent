# Evaluator Critique -- Cycle 87 / phase-4.8 step 4.8.10

Step: 4.8.10 2026 regulatory memo + wash-sale filter

## Research-gate upheld

Second consecutive cycle with researcher + Explore spawned in
parallel (Cycle 86 restored; Cycle 87 continues). 22 URLs cited in
the researcher report; memo inherits those citations.

## Dual-evaluator run (parallel, anti-rubber-stamp)

## qa-evaluator: PASS

7-point substantive review:
1. **CALENDAR-day window in code**: `timedelta(days=30)` not
   `bdate_range`; audit explicitly tests Saturday buy 3 days after
   Wednesday sell -> flagged.
2. **Wash-sale fixtures discriminating**: +15 blocked, +30 blocked
   (inclusive), +31 allowed, +61 allowed, different ticker allowed,
   filter partitioning (1 blocked out of 3). Real logic.
3. **T+1 discrimination**: unsettled same-day BUY -> blocked with
   "UNSETTLED_CASH_INSUFFICIENT"; fully-settled -> allowed.
4. **Margin discrimination**: deficit blocked; under allowed;
   threshold_pct=0.05 tightens cap and flips a borderline BUY from
   allowed to blocked.
5. **Memo citations real**: SEC 34-96930, IRC Sec 1091, FINRA 4210
   / SR-2025-017 all present. Audit checks each.
6. **7-section structure**: Scope, Regulatory changes, System
   impact, Operational controls, Monitoring, Review cadence, Open
   items all present.
7. **Sign convention**: `record_loss` raises on non-positive
   disallowed_loss_usd (gain is not a wash candidate).

## harness-verifier: PASS

7/7 mechanical checks green:
- Immutable verification exits 0.
- Audit clean.
- Both artifacts correct.
- Calendar-day proof via direct library call.
- **Mutation A**: WINDOW_DAYS=30 -> 5 -> test rc=1. Caught.
- **Mutation B**: disable T+1 `if buy_notional > settled_cash:`
  -> test rc=1. Caught.
- File restored after each mutation.

## Decision: PASS (evaluator-owned)

Both evaluators substantively green with TWO mutation-resistance
tests on independent parts of the system. Research-gate discipline
upheld second cycle in a row.
