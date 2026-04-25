---
step: phase-16.19
title: Trading mechanics drills (alpaca shadow + kill switch + zero-orders)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.19

## Research-gate summary

`handoff/current/phase-16.19-research-brief.md`. Envelope: tier=simple, 6 in-full, 11 URLs, recency scan present, gate_passed=true.

Critical findings:

1. **alpaca_shadow_drill.py reuses `uat-17.6-{sym}-{i}` client_order_ids** -- collision risk if those orders are still active. Mitigant: prior 17.6 ran 2026-04-22, DAY orders expired Fri 2026-04-24 at 16:00 ET market close. Today is Sat 2026-04-25 (markets closed), so all prior orders are in terminal states. Drill should succeed; weekend submissions queue at Alpaca for Monday open.

2. **kill_switch_test.py is misnamed** -- it tests `SignalsServer.risk_check` drawdown circuit-breaker (-15% threshold, 4 scenarios), NOT `backend/services/kill_switch.py`'s pause/flatten/resume state machine. The criterion `kill_switch_pause_flatten_resume_pass` is therefore subtly misleading. I'll honor the literal verification command and disclose this in experiment_results so Q/A can audit the gap.

3. **fills_source_alpaca_paper** requires real Alpaca paper API keys in env. Keys are present (verified live earlier today via /api/sovereign and Alpaca account check). Drill should produce `source: alpaca_paper`, not `mock_alpaca`.

4. **zero_orders_drill.py uses in-memory StubBQ** -- no network or Alpaca dependency. Should pass cleanly.

## Hypothesis

All 3 drills pass. Shadow drill produces ≥1 alpaca_paper fill with drift < 2%. Kill switch test (drawdown circuit-breaker) passes 4/4 scenarios. Zero-orders drill passes the StubBQ pipeline checks. The 16.18 TZ fix did not regress any of these (none depend on the scheduler).

## Success Criteria (verbatim from masterplan)

```
python scripts/go_live_drills/alpaca_shadow_drill.py && python scripts/go_live_drills/zero_orders_drill.py && python scripts/go_live_drills/kill_switch_test.py
```

- alpaca_shadow_drill_pass
- kill_switch_pause_flatten_resume_pass
- zero_orders_drill_pass
- fills_source_alpaca_paper

## Plan steps

1. (optional pre-step) Query Alpaca paper for any open `uat-17.6-*` orders. If any, cancel them. If clean, proceed.
2. Run alpaca_shadow_drill.py -- capture stdout including order IDs and fill sources
3. Run zero_orders_drill.py -- capture stdout
4. Run kill_switch_test.py -- capture stdout (4 scenarios)
5. Spawn Q/A

## What Q/A must audit

1. Each drill independently re-run (don't trust Main's stdout)
2. **Naming-mismatch flag**: criterion `kill_switch_pause_flatten_resume_pass` actually tested via SignalsServer.risk_check (drawdown), not pause/flatten/resume state machine. Q/A decide whether this constitutes a real coverage gap or just a naming smell.
3. fills_source verification: was source `alpaca_paper` (real keys) or `mock_alpaca` (no keys)?
4. No code changes claimed by Main this cycle (read-only verification of pre-existing drills)
