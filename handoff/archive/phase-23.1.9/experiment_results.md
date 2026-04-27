---
step: phase-23.1.9
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_9.py'
---

# Experiment Results — phase-23.1.9

## What was built

Operator can now top up the virtual paper-trading fund AND tune paper-trading-specific settings — both from a new "Manage" tab on the Paper Trading page, no leaving the page.

## Files modified

| File | Change |
|---|---|
| `backend/api/settings_api.py` | 10 new paper-trading fields × 4 places (FullSettings, SettingsUpdate, _FIELD_TO_ENV, _settings_to_full). `paper_starting_capital` is read-only (intentionally not in SettingsUpdate); 9 others are writable with `Field(None, ge=..., le=...)` validators. |
| `backend/api/paper_trading.py` | NEW `DepositRequest` Pydantic model (`amount: float = Field(..., gt=0, le=1_000_000)`) + NEW `POST /api/paper-trading/deposit` endpoint. Increments BOTH `current_cash` AND `starting_capital` so `total_pnl_pct = ((nav − starting) / starting) × 100` stays anchored. Logs an audit line to stdout, invalidates `paper:*` cache, returns the updated portfolio shape. |
| `frontend/src/lib/types.ts` | `FullSettings` interface +10 optional paper fields; NEW `DepositResponse` interface. |
| `frontend/src/lib/api.ts` | NEW `depositPaperFunds(amount)` calling `POST /api/paper-trading/deposit`. |
| `frontend/src/app/paper-trading/page.tsx` | NEW "manage" tab + 7 new state vars + `useEffect` lazy-loads settings on tab open + `handleDeposit` + `handleSettingsSave` + 2 reusable `<ReadOnlyField>` and `<PaperSettingNum>` helper components. UI: top-up form (amount + Deposit button + success/error banners) + 9-input settings grid with inline Save (only "dirty" fields are PATCHed; "unsaved" badge on each changed field). |
| `tests/api/test_paper_trading_deposit.py` | NEW (12 tests covering DepositRequest validation + 10 paper-trading fields exposure + read-only `paper_starting_capital` discipline + ge/le validators reject out-of-range). |
| `tests/verify_phase_23_1_9.py` | NEW immutable verification script (referenced from contract front-matter). |

## Verbatim verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_9.py
ok 10 paper fields wired + DepositRequest validates
exit=0
```

The script asserts:
1. All 10 paper fields exist in `FullSettings`
2. 9 writable fields exist in `SettingsUpdate` (excluding read-only `paper_starting_capital`)
3. `DepositRequest(amount=500.0)` validates
4. `DepositRequest(amount=0)` raises (gt=0 enforced)
5. `DepositRequest(amount=2_000_000)` raises (le=1_000_000 enforced)

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/test_settings_api_signal_stack.py tests/api/test_paper_trading_deposit.py tests/services/ -v --no-header -q
collected 137 items
tests/api/test_settings_api_signal_stack.py ..............         [ 10%]
tests/api/test_paper_trading_deposit.py ............               [ 18%]
tests/services/test_extract_stop_loss.py ..........                [ 26%]
tests/services/test_macro_regime.py ............                   [ 35%]
tests/services/test_meta_scorer.py ..............                  [ 45%]
tests/services/test_news_screen.py .....................           [ 60%]
tests/services/test_pead_signal.py ..................              [ 73%]
tests/services/test_sector_calendars.py ................           [ 85%]
tests/services/test_signal_attribution.py ....................     [100%]
============================== 137 passed in 1.98s ==============================
```

12 new + 125 prior = 137/137 tests pass. Zero regression across all 9 cycles in the phase-23.1 plan.

(Pre-existing test collection error in `tests/api/test_observability.py` — import error from `harness_autoresearch.structured_log`; unrelated to this cycle, predates phase-23.1; non-blocking.)

## Frontend type-check

```
$ cd frontend && npx tsc --noEmit
(silent — 0 errors)
```

Type contracts hold end-to-end: `PaperNumKey` union restricts the writable settings; `setDirty` callback type-narrows correctly; `DepositResponse` shape matches the backend Pydantic return.

## Why deposit increments BOTH current_cash AND starting_capital

If the operator deposits $5,000 into a $10,000 fund and we only bump cash → NAV goes to $15,000 but starting_capital stays $10,000 → `total_pnl_pct = (15000 − 10000) / 10000 × 100 = 50%` — a fake 50% gain just because money was added.

By also bumping starting_capital → `total_pnl_pct = (15000 − 15000) / 15000 × 100 = 0%` — the deposit is correctly P&L-neutral. This matches Alpaca / Robinhood / Webull paper-trading conventions (per research brief).

## What the operator sees on the new Manage tab

**Card 1 — Top up fund:**
- "Amount (USD)" input with `$` prefix, range 1–1,000,000
- "Deposit" button (emerald)
- Success banner: "Deposited $5,000 — new NAV $14,490.14 (starting capital now $15,000)"
- Error banner on failure
- Hero metrics (NAV / Cash / P&L) refresh automatically after successful deposit

**Card 2 — Trading settings:**
- Read-only display: "Starting capital: $10,000 (Adjust via Top up fund)"
- 9 editable inputs in a 2-column grid:
  - Max simultaneous positions (1–50)
  - Daily LLM cost cap USD (0.1–50)
  - Default stop-loss % (1–50, hint: O'Neil canonical 7-8%)
  - Screen top-N candidates (1–100)
  - Analyze top-K with LLM (1–50)
  - Transaction cost % (0–5)
  - Daily loss limit % (0.5–25)
  - Trailing drawdown limit % (1–50)
  - Min cash reserve % (0–50)
- Each changed field shows an "unsaved" amber badge; Save button disables when no changes
- Save button calls `PUT /api/settings/` with ONLY the dirty fields (partial update; existing settings UI flow)

## Out of scope (per contract; Phase-2 follow-ups)

- BQ `paper_deposits` audit table (operator --apply needed; for now deposits log to stdout)
- Deposit history list in the UI (depends on the table)
- Withdraw / "reset to $10K" button (separate cycle)
- Per-deposit idempotency keys (Phase 2)
- Multi-portfolio support (sticks with `default` portfolio_id)

## Honest disclosure

- Deposits are logged to stdout (`logger.info`), not persisted to a deposit-history table. Tomorrow's first deposit will produce a clean log line; for production audit you'd want the BQ table (Phase 2).
- The new settings ARE persisted via the existing `_update_env_var` flow (writes to `backend/.env`). They survive backend restarts.
- The "Manage" tab is the 6th tab on the page; on a narrow viewport the tab bar still wraps cleanly (existing pattern).

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → restart frontend so the new bundle ships
