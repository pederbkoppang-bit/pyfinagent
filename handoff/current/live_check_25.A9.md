# Live-check placeholder — phase-25.A9

**Step:** 25.A9 — Fix cache-write cost premium 1.25x → 2.0x
**Date:** 2026-05-12

## Live-check field (from masterplan 25.A9)

> "Re-process recent BQ cost rows; verify cumulative cost increases by expected ratio"

## Pre-deployment math verification (artifact for live_check_gate)

Direct round-trip test (verifier claim 5):
- 4096 tokens × $5/MTok × **2.0x** ÷ 1M = **$0.04096** (new, correct billing)
- 4096 tokens × $5/MTok × 1.25x ÷ 1M = $0.02560 (old, under-reported)
- Ratio: 2.0 / 1.25 = 1.6 → 60% increase in reported cache-write cost

## Post-deployment verification (to be filled in)

After the next autonomous cycle that triggers cache writes, the operator should observe:
- Per-cycle `cache_write_cost` field in BQ `cost_tracker_events` increases by ~60% vs pre-25.A9 baseline (for equivalent input token volumes)
- Aggregate daily cost per provider reflects the new accurate rate
- Sample BQ query: `SELECT SUM(cost_usd) FROM pyfinagent_data.cost_tracker_events WHERE created_at > '2026-05-12T15:00:00Z' AND model LIKE '%opus%'`

(Output to be appended here once the post-25.A9 cost data has accumulated.)

**Audit anchor for next bucket:** 25.A8 (cost-budget hard-block in llm_client — now unblocked by 25.A9 accurate cost data).
