---
step: phase-25.G
cycle: 59
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_G.py'
title: Fix Slack digest P&L data source (endpoint + field key) (P0)
---

# Experiment Results — phase-25.G

**Action:** GENERATE. 7 edits across 3 files closing phase-24.5 F-1 + F-2 + F-6.

## Code changes

### scheduler.py — endpoint fix (2 sites L235, L260)
```diff
- portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/portfolio/performance")
+ portfolio_res = await client.get(f"{_LOCAL_BACKEND_URL}/api/paper-trading/portfolio")
```

### commands.py — endpoint fix (L138)
```diff
- res = await client.get(f"{_BACKEND_URL}/api/portfolio/performance")
+ res = await client.get(f"{_BACKEND_URL}/api/paper-trading/portfolio")
```

### formatters.py — field key fallback (3 sites L106, L321, L365)
```diff
- total_return = data.get("total_return_pct", 0)
+ # phase-25.G: paper_trading endpoint returns total_pnl_pct, not total_return_pct
+ total_return = data.get("total_pnl_pct", data.get("total_return_pct", 0))
```

(Fallback chain accommodates both keys — defensive for any consumer still serving the old shape.)

## Verbatim verifier output

```
=== phase-25.G (Slack digest P&L fix) verifier ===
  [PASS] scheduler_uses_paper_trading_portfolio_endpoint_twice_for_morning_and_evening_digest
  [PASS] scheduler_no_longer_references_legacy_in_memory_portfolio_endpoint
  [PASS] portfolio_slash_command_uses_paper_trading_portfolio_endpoint
  [PASS] portfolio_slash_command_no_longer_references_legacy_endpoint
  [PASS] formatters_reads_total_pnl_pct_field_at_least_once
  [PASS] scheduler_py_syntax_clean
  [PASS] formatters_py_syntax_clean
  [PASS] commands_py_syntax_clean
  [PASS] phase_25_G_attribution_comment_present
PASS (9/9) EXIT=0
```

9/9 PASS.

## Hypothesis verdict
CONFIRMED. All 7 edit sites updated; AST clean across all 3 files; legacy endpoint references zeroed in scheduler + commands.

## Live-check
Per masterplan: "Slack screenshot showing non-zero P&L in next morning digest". Operator captures next morning digest at 06:00 ET.

## Next phase
EVALUATE — Q/A pending.
