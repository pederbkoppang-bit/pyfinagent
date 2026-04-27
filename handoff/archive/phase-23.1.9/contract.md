---
step: phase-23.1.9
title: Paper Trading "Manage" tab — deposit endpoint + 9 paper-settings exposed in API and UI
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_9.py'
research_brief: handoff/current/phase-23.1.9-research-brief.md
---

# Contract — phase-23.1.9

## Hypothesis

A new "Manage" tab on the Paper Trading page lets the operator (a) deposit more capital into the virtual fund without losing P&L meaningfulness, and (b) tune paper-trading-specific settings (max positions, cost cap, stop-loss default, screen/analyze top-N, transaction cost, risk limits) without leaving the page. Deposits correctly increment BOTH `current_cash` AND `starting_capital` so total_pnl_pct stays anchored.

## Plan

1. **Backend `backend/api/settings_api.py`** — extend FullSettings + SettingsUpdate + _FIELD_TO_ENV + _settings_to_full with 10 paper-trading fields per the research brief Part 3 (1 read-only `paper_starting_capital`, 9 writable). Mirror the phase-23.1.6 pattern.

2. **Backend `backend/api/paper_trading.py`** — NEW `POST /deposit` endpoint:
   - `DepositRequest` Pydantic model: `amount: float = Field(..., gt=0, le=1_000_000)`
   - Fetches current portfolio via `bq.get_or_create_portfolio()`
   - Increments `current_cash += amount` AND `starting_capital += amount` AND `total_nav += amount`
   - Recomputes `total_pnl_pct = ((nav − starting) / starting) × 100`
   - Persists via `bq.upsert_portfolio()` or equivalent existing helper
   - Logs to stdout `[paper_trading] deposit accepted: amount=$N portfolio=default before=$X after=$Y`
   - Returns `{status, amount, new_cash, new_starting_capital, new_nav, new_pnl_pct, deposited_at}`
   - Invalidates `paper:*` cache keys

3. **Frontend `frontend/src/lib/types.ts`** — extend `FullSettings` with 10 optional paper fields (mirror phase-23.1.6).

4. **Frontend `frontend/src/lib/api.ts`** — NEW `depositFunds(amount: number)` function calling `POST /api/paper-trading/deposit`. NEW `DepositResponse` interface.

5. **Frontend `frontend/src/app/paper-trading/page.tsx`** — add `"manage"` to the TabId union + tab bar entry. NEW Manage tab content rendered when `tab === "manage"`:
   - **Top-up Fund card**: amount input + Deposit button + success/error banners. On success, `getStatus()` refreshes hero metrics (NAV, Cash, P&L, etc.).
   - **Trading Settings card**: read-only `paper_starting_capital` display + 9 editable inputs with `min`/`max` matching backend Field constraints. Save button calls `updateSettings({...changedFields})`. Inline validation; success/error banners.

6. **Tests** at `tests/api/test_paper_trading_deposit.py`:
   - `DepositRequest(amount=500)` accepts; `amount=0` and `amount=2_000_000` rejected
   - `DepositRequest(amount=-100)` rejected
   - The new 10 settings fields exist in FullSettings + 9 writable in SettingsUpdate
   - Endpoint contract integration test (when feasible without hitting real BQ — use a stubbed BigQueryClient)

7. **Skip for v1** (Phase-2 follow-ups):
   - `paper_deposits` BQ audit table (needs operator --apply migration; deposit log to stdout for now)
   - Deposit history list in the UI (depends on the table)
   - Withdraw / "reset to $10K" button (separate cycle)

## Out of scope

- BQ migration for `paper_deposits` (Phase 2)
- Deposit history list UI (Phase 2)
- Initial balance reset functionality (Phase 2)
- Per-deposit idempotency keys (Phase 2)
- Multi-portfolio support (Phase 2 — sticks with `default` portfolio_id)

## Verification

The front-matter command does five things in one shot:
1. Asserts all 10 new paper-trading fields exist in `FullSettings`
2. Asserts the 9 writable fields (excluding read-only `paper_starting_capital`) exist in `SettingsUpdate`
3. Asserts `DepositRequest(amount=500.0)` validates
4. Asserts `DepositRequest(amount=0)` raises (gt=0 enforced)
5. Asserts `DepositRequest(amount=2_000_000)` raises (le=1_000_000 enforced)

Frontend behavior verified by `cd frontend && npx tsc --noEmit` exit 0.

## Files modified

- `backend/api/settings_api.py` — 10 fields × 4 places (FullSettings, SettingsUpdate, _FIELD_TO_ENV, _settings_to_full)
- `backend/api/paper_trading.py` — NEW DepositRequest + POST /deposit endpoint (~50 LOC)
- `backend/db/bigquery_client.py` — NEW `upsert_portfolio` helper if not already there (verify; the existing pattern is delete-then-insert)
- `frontend/src/lib/types.ts` — 10 optional paper fields + NEW PaperDeposit / DepositResponse types
- `frontend/src/lib/api.ts` — NEW `depositFunds()` client function
- `frontend/src/app/paper-trading/page.tsx` — NEW "manage" tab + 2 BentoCards (top-up + settings)
- `tests/api/test_paper_trading_deposit.py` — NEW (~6 tests)

## References

- `handoff/current/phase-23.1.9-research-brief.md` — full brief (455 lines, 5 sources read in full, gate_passed: true)
- `backend/api/paper_trading.py` — existing endpoints; DepositRequest slots in
- `backend/api/settings_api.py:60-244` — FullSettings + SettingsUpdate + _FIELD_TO_ENV
- `backend/services/paper_trader.py:47-58` — initial portfolio row construction (P&L formula)
- `frontend/src/app/paper-trading/page.tsx:239-245` — tab bar definition
- `frontend/src/lib/types.ts:524-562` — FullSettings interface
