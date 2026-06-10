# Experiment Results — Step 56.1 (GENERATE)

**Step:** 56.1 — FX/value/fee data-correctness fix. **Date:** 2026-06-10. **Mode:** fix work, finding-ID-driven (F-1, F-2, F-12, F-13); backfill operator-gated; do-no-harm on the US momentum core.

## What was built (8 files)

| File | Change | Finding |
|---|---|---|
| `backend/services/paper_trader.py` | 3-line fix: BUY total_value × `_local_to_usd`; SELL total_value + transaction_cost × `_l2u` (row-build only; cash/round-trip paths deliberately untouched) | F-2 |
| `backend/tests/test_phase_50_2_multicurrency.py` | 4 new row-level tests: KRW BUY row USD, KRW SELL row USD value+fee + cash-credit-no-double-convert, US byte-identity; FAIL-pre/PASS-post captured | F-2 |
| `scripts/migrations/backfill_56_1_kr_trade_values.py` | NEW: operator-gated dry-run-default restatement of the 7 corrupted rows (pinned literals, idempotent, GIPS disclosure docstring) — NOT executed | F-2 |
| `frontend/src/lib/format.ts` | NEW shared helper `positionMarketValueUsd` (the mvUsd pattern extracted) | F-1 |
| `frontend/src/lib/useLiveNav.ts` | NAV root-cause fix: reduce over the FX-safe helper instead of `lp × quantity` | F-1 |
| `frontend/src/components/paper-trading/cockpit-helpers.tsx` | RiskMonitorCard concentration via the helper; per-market benchmark card relabeled "<MKT> holdings" (tooltip retained) | F-1, F-12 |
| `frontend/src/components/paper-trading/trades-columns.tsx` | Header comment corrected: USD post-fix + explicit caveat for the 7 pre-fix KR rows pending backfill | F-2 |
| `frontend/src/app/paper-trading/layout.tsx` | Subtitle amount-neutral ("$10K" removed) | F-13 |

## Verification command output (verbatim)

```
$ cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'fx or paper_trader or krw' -q
26 passed, 723 deselected, 1 warning in 2.97s
$ test -f handoff/current/live_check_56.1.md && echo PASS
PASS
```

## Regression proof (criterion 1)

Pre-fix (tests written FIRST): `2 failed, 8 passed` — `AssertionError: total_value=364175.1 looks like LOCAL currency (KRW), not USD`. Post-fix: `10 passed`. US byte-identity passed on BOTH sides (do-no-harm).

## Live UI evidence (criterion 2)

Playwright captures in `handoff/current/captures_56.1/`: NAV card **23,856.94 USD** (was 345,950.68), Total P&L +19.28% (was +1,629.75%), Max position 3.0% (was 1,527.8%), donut $23,857, currency exposure USD 98.9%/KRW 1.0%, "KR HOLDINGS" card label (was "VS KOSPI"). Full decomposition in live_check_56.1.md §C.

## Backfill state (criterion 3)

Migration present + dry-run verified; NOT executed (operator-gated). The 7 historical rows are flagged via the trades-columns caveat + live_check §D, with the GIPS materiality classification persisted in the script docstring. Operator ask recorded.

## Honest limitations

- The 7 historical KR rows still display local magnitudes until the operator approves the backfill — by design, disclosed in three places.
- `:3000` (operator instance) hot-reloads the frontend fixes via `next dev`; the BACKEND process on :8000 still runs the pre-fix code in memory — the F-2 row fix takes effect for trades written after the next backend restart (the restart is left to the operator/next deploy step per phase-58's "deploy fixes" criterion; no unattended restart of a live trading process without an operator window).
- The four-FX-point consistency statement for mark-to-market/cash relies on 55.1's verification (those paths were measured correct and are untouched here); the new tests assert the cash-credit non-double-conversion directly.
