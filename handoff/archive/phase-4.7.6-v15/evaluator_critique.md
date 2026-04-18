# Evaluator Critique -- Cycle 77 / phase-4.8 step 4.8.0

Step: 4.8.0 Transaction Cost Analysis (implementation shortfall)

## Dual-evaluator run (parallel, fresh reads, anti-rubber-stamp)

## qa-evaluator: PASS

Substantive 8-point honesty review:

1. **IS math correct (CFA/Perold canonical)**: `side_sign *
   (fill - arrival)/arrival * 10000`. Traced by hand:
   - Buy with fill>arrival -> positive cost (correct).
   - Sell with fill<arrival -> sign=-1 * negative diff = positive
     cost (correct).
   - Buy with fill<arrival -> negative IS (favorable; correct).

2. **Arrival != fill**: structurally distinct in the seed path.
   Arrival comes from `_deterministic_price(symbol, day)`; fill
   is `arrival * (1 + sign*drift/10000)`. No degenerate-IS=0.

3. **Alert teeth real**: `--force-alert` flipped median to 38.99 bps
   AND flipped `alert_triggered=true` AND emitted WARNING. Not a
   constant-false placeholder.

4. **Seeded transparency**: `data_source: "seeded"`,
   `seeded_rows: 70`, per-row `meta.seeded: true`. Auditor cannot
   mistake synthetic for live.

5. **Drift realism**: 2-9 bps liquid drift sits inside the
   researcher-cited 5-15 bps institutional range. Plausible, not
   fantasy.

6. **Per-fill log shape**: 70 rows (7d x 10 symbols), 11 required
   fields per row (ts, client_order_id, symbol, side, qty,
   fill_price, arrival_price, is_bps, notional_usd, liquid,
   source). 60 fall in the 7-day window matching artifact's
   `rows_in_window`.

7. **Degenerate-zero guard in CODE**: `compute_is_bps` raises
   `ValueError` on arrival<=0. Not a comment, not silent-returns-inf.

8. **Formula matches canonical references**: AnalystPrep CFA III,
   QB, Talos.

**Acknowledged scope limit** (not CONDITIONAL): alert logs WARNING +
writes JSON but does NOT page Slack or kill-switch. Contract
explicitly scoped to "log WARNING + alert JSON"; operator paging is
phase-4.8.x infra.

## harness-verifier: PASS

6/6 mechanical checks green:
- immutable command exits 0, median_bps_liquid = 5.9976
- syntax clean
- math sign convention verified (4 cases + ValueError)
- jsonl: 70 rows, 11 fields each
- weekly JSON: 14 keys, threshold 15.0, alert=false on clean run
- alert teeth: --force-alert produced median=38.99, alert=true,
  WARNING line; clean re-run restored alert=false

## Decision: PASS (evaluator-owned)

Two independent PASS verdicts on substantive checks including a
deliberate alert-teeth mutation test. Seeded data is honest and
documented. No rubber-stamp.
