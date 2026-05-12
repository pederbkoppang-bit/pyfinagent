---
step: phase-25.1
cycle: 57
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_1.py'
title: Wire check_stop_losses() into daily loop with auto-sell (P0)
---

# Experiment Results — phase-25.1

**Action:** GENERATE (real code change). NOT read-only. Added Step 5.6 to `backend/services/autonomous_loop.py` between Step 5.5 (kill-switch) and Step 6 (decide_trades).

## Code changes

### File 1: `backend/services/autonomous_loop.py` — added Step 5.6 block

Insert location: line 332, immediately after the kill-switch early-return block.

```python
# -- Step 5.6: Stop-loss enforcement (phase-25.1) --------------------
logger.info("Paper trading: Step 5.6 -- Stop-loss enforcement")
summary["steps"].append("stop_loss_enforcement")
summary["stop_loss_triggered"] = []
triggered_stops = await asyncio.to_thread(trader.check_stop_losses)
for sl_ticker in triggered_stops:
    try:
        sl_trade = await asyncio.to_thread(
            trader.execute_sell,
            ticker=sl_ticker,
            quantity=None,
            price=None,
            reason="stop_loss_trigger",
            signals=None,
        )
        if sl_trade:
            summary["stop_loss_triggered"].append(sl_ticker)
            logger.warning(
                "Paper trading: stop-loss triggered for %s -- sold at %s",
                sl_ticker, sl_trade.get("price"),
            )
    except Exception as sl_exc:
        logger.exception("Stop-loss execute_sell failed for %s: %s", sl_ticker, sl_exc)
```

Closes phase-24.1 audit finding F-1 (orphan `check_stop_losses` with zero callers).

### File 2: `tests/verify_phase_25_1.py` — new verifier (146 LOC)

8 immutable claims (success_criteria + structural assertions). Stdlib-only. Idempotent.

## Verbatim verifier output

```
=== phase-25.1 (stop-loss wiring) verifier ===
  [PASS] grep_check_stop_losses_in_autonomous_loop_returns_match
  [PASS] stop_loss_trigger_reason_string_present
  [PASS] summary_includes_stop_loss_triggered_field
  [PASS] step_5_6_stop_loss_enforcement_label_present
  [PASS] check_stop_losses_wrapped_in_asyncio_to_thread
  [PASS] execute_sell_called_in_stop_loss_block
  [PASS] autonomous_loop_py_syntax_clean
  [PASS] paper_trader_execute_sell_signature_has_reason_kwarg
PASS (8/8) EXIT=0
```

8/8 PASS. No log-last sentinel — this verifier doesn't depend on harness_log entries (phase-25 verifiers focus on code correctness).

## Hypothesis verdict
CONFIRMED. Step 5.6 wired exactly at line 332 per researcher recommendation. `execute_sell` natural idempotency means no external dedup needed. AST clean.

## Live-check
Per masterplan 25.1 `verification.live_check`: "BQ paper_trades row with reason='stop_loss_trigger' visible after next cycle". Will be confirmed post-deploy when the autonomous cycle next runs against the 6 stop-less positions (TER first — already below any reasonable 8% stop).

## Next phase
EVALUATE — Q/A subagent will:
1. Run 5-item harness-compliance audit
2. Re-run `python3 tests/verify_phase_25_1.py`
3. LLM-judgment: contract alignment, mutation-resistance, anti-rubber-stamp, scope honesty, research-gate compliance
4. Return verdict envelope
