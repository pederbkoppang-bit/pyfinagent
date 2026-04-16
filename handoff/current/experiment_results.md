# Experiment Results -- Zero-Orders Bug Fix (Cycle 27)

**Date:** 2026-04-16
**Cycle:** 27
**Step:** Paper trading zero-orders diagnosis + fix

## Root Cause Analysis

Paper trading generated **zero trades for 27 consecutive days**. Three issues found:

### Issue 1: Recommendation Normalization Bug (CODE FIX)
- **Location:** `backend/services/portfolio_manager.py` lines 89, 90, 129
- **Cause:** Gemini synthesis returns `"Strong Buy"` / `"Strong Sell"` (with spaces). `decide_trades()` used `.upper()` producing `"STRONG BUY"`, but lookup sets `_BUY_RECS` / `_SELL_RECS` use `"STRONG_BUY"` (underscore). Signals with space-delimited recommendations were **silently dropped** -- no error, no log, no trade.
- **Fix:** New `_normalize_rec()` helper: `.strip().upper().replace(" ", "_")`. All 3 comparison sites updated. Lookup sets unchanged.
- **Impact:** Unblocks Phase 4.4.2 (paper trading validation, 5 checklist items).

### Issue 2: Missing ANTHROPIC_API_KEY (PEDER ACTION NEEDED)
- **Location:** `backend/services/autonomous_loop.py` line 364-366
- **Cause:** Claude analysis path fails every cycle because `ANTHROPIC_API_KEY` not configured in `backend/.env`. Confirmed by repeated `"Missing API key for provider anthropic"` bot errors in `#ford-approvals`.
- **Fix:** Peder needs to add `ANTHROPIC_API_KEY=sk-ant-...` to `backend/.env`. Claude analysis produces clean `"BUY"`/`"SELL"`/`"HOLD"` strings at ~$0.01/call. Without it, only the Gemini fallback runs.

### Issue 3: Outdated Claude Model ID (CODE FIX)
- **Location:** `backend/services/autonomous_loop.py` line 387
- **Cause:** `claude-sonnet-4-20250514` is an old model ID. Current is `claude-sonnet-4-6`.
- **Fix:** Updated model string.

## Also Added
- **Diagnostic logging** (`portfolio_manager.py`): When `decide_trades` produces zero orders, emits `logger.warning` with candidate count, recommendation distribution, cash, and NAV. Future zero-trade cycles will be visible in backend logs.

## Verification
- `ast.parse` clean on both modified files
- 18/18 contract SCs + 4/4 adversarial probes PASS
- `_normalize_rec("Strong Buy")` == `"STRONG_BUY"` confirmed
- Stop-loss code path (lines 73-85) untouched
- Lookup sets (`_BUY_RECS`, `_SELL_RECS`, `_DOWNGRADE_RECS`) unchanged
- Zero drill files or docs touched
