# Experiment results — step 70.3 (S3 + money-path: atomic cross-sector swap + non-US avg-entry fix)

**Phase/step:** phase-70 → 70.3 | **Date:** 2026-07-17 | **Type:** backend money-path, flag-gated default-OFF
(double-gated behind paper_swap_enabled), $0, paper-only, DARK-until-token, fail-safe. live_check: none (no UI).

## Files changed (5)

1. **`backend/config/settings.py`** — 3 flags, all default-OFF/identity: `paper_atomic_swap_enabled`,
   `paper_cross_sector_rotation_enabled`, `paper_avg_entry_fx_fix_enabled`.
2. **`backend/services/portfolio_manager.py`** — `TradeOrder.swap_group_id` field; `_compute_swap_candidates`
   gains an `available_cash` param (threaded from the :480 call); the swap emit (atomic ON) cash-bounds the BUY
   `min(nav*pct/100, available_cash + freed)`, applies the $50 floor (drops the whole pair if under — never a
   lone SELL), and tags both legs with a shared `swap_group_id`; a flag-gated cross-sector fallback (weakest-
   overall) + a new `_cross_rotation_safe` helper (HHI-strictly-drops + destination count/NAV cap re-validation,
   requires churn-fix). OFF → legacy sizing + same-sector-only + untagged (byte-identical). `import uuid`.
3. **`backend/services/paper_trader.py`** — `execute_buy` gains `reserved_cash` (cash check `total_cost > cash +
   reserved_cash`; 0.0 → byte-identical); the add-on avg_entry is gated: ON →
   `(old_qty*avg_entry + qty*price)/new_qty` (LOCAL-share-weighted), OFF → legacy `new_cost/new_qty`. cost_basis
   stays USD.
4. **`backend/services/autonomous_loop.py`** — `_execute_swap_pair` helper (SELL-feasibility pre-check → BUY-first
   with reserved cash → SELL; BUY drops ⇒ SELL never attempted; SELL-fails-after-BUY ⇒ defensive delete of the
   just-created BUY); wired into Step 7 so atomic-ON swap pairs execute via the helper and are removed from the
   flat loops. OFF → all orders (swap_group_id=None) flow the flat loops (byte-identical).
5. **`backend/tests/test_phase_70_3_atomic_swap.py`** (NEW) — 11 deterministic tests.

## Verification command output (verbatim)

```
$ bash -c 'ls backend/tests/ | grep -Eqi "70_3|swap|atomic" && python -c "import ast; ast.parse(open(\"backend/services/portfolio_manager.py\").read()); ast.parse(open(\"backend/services/paper_trader.py\").read())"'
VERIFICATION: PASS (exit 0)
$ python -m pytest backend/tests/test_phase_70_3_atomic_swap.py -q
11 passed
```
Import-smoke: all 4 changed modules import clean; helpers/flags/params present; flags default False.

## Criterion 1 — atomic swap (both legs or neither) + cash-bound + $50 floor

- `test_atomic_swap_buy_drops_does_not_sell` (RED→GREEN of "SELL-executes-BUY-drops"): with the paired BUY
  dropped (execute_buy→None), the SELL is NEVER attempted → the WEAK position is still held → book unchanged.
  `test_atomic_swap_contrast_old_flat_path_would_lose_position` documents the OLD flat path (SELL then BUY-drops)
  removing the position (net −1).
- `test_atomic_swap_happy_path_both_legs`: BUY-first ordering, both legs execute.
- `test_swap_atomic_cash_bounded_and_grouped`: swap BUY bounded by `min(nav*pct, available + freed)`; both legs
  share one `swap_group_id`. `test_swap_atomic_50_floor_drops_pair`: a sub-$50 swap BUY drops the WHOLE pair
  (no lone SELL). `test_swap_off_no_group_id_and_legacy_sizing`: OFF → untagged + legacy nav*pct sizing.

## Criterion 2 — cross-sector rotation (flag-gated, fail-safe)

`_cross_rotation_safe`: `test_cross_rotation_safe_blocks_count_cap_breach` (a count-cap-blocked candidate entering
its at-cap sector is correctly BLOCKED — fail-safe, count cap never moved); `test_cross_rotation_safe_allows_hhi_
drop_within_caps` (a rotation that strictly lowers HHI and keeps the destination caps is allowed). OFF →
same-sector-only (byte-identical). Honest note: because every `sector_blocked` candidate is count-cap-blocked, the
count-cap re-validation is what makes cross-sector rotation fail-safe — it fires only when the destination caps
permit (HHI-improving, under-cap), which is the correct risk-preserving behavior.

## Criterion 3 — non-US avg_entry FX fix

`test_avg_entry_fx_fix_local_consistent_for_kr`: a KR (KRW) add-on BUY with the fix ON yields avg_entry ≈ 70000
(LOCAL, correct); with the fix OFF the legacy formula yields a tiny (~USD) value (documents the corruption).
Byte-identical for US (quantity*price == amount_usd at fx=1).

## Criterion 4 — fail-safe, default-OFF

`test_flags_present_and_default_off`: all 3 flags default False; `reserved_cash` present on execute_buy. No
risk-limit threshold moved (the cap comparisons only PREPEND default-OFF guards; the cross-sector re-validation
only ADDS a fail-safe block). Every failure path holds/drops rather than corrupting the book.

## Regression (2 PRE-EXISTING failures, NOT caused by 70.3 — proof below)

`pytest test_portfolio_swap.py test_phase_60_2_churn_fix.py test_phase_50_2_multicurrency.py
test_phase_23_2_6_sector_cap_emit.py test_phase_70_2_soft_diversity.py` → 33 passed, 2 failed:
1. `test_phase_23_2_6_backend_log_has_skipping_buy_evidence` — reads `backend.log` for "Skipping BUY" strings; the
   log is freshly rotated/quiet. Environmental; orthogonal to the 70.3 diff (also flagged as pre-existing by the
   70.2 Q/A).
2. `test_swap_framework_fills_zero_buy_gap` — expects 2 swap pairs (its `_make_settings` assumes churn_fix OFF),
   but the operator's live `.env` has `PAPER_SWAP_CHURN_FIX_ENABLED=true` (Settings default is False), so the 2nd
   swap's delta is 24% < the 25% bar → 1 swap. PROOF it is env-driven, not my change: calling
   `_compute_swap_candidates` directly with churn_fix OFF yields **2** swaps and with churn_fix ON yields **1** —
   and 70.3 does NOT touch the churn denom logic (:633, unchanged). So the OFF swap path is byte-identical; the
   test just resolves churn_fix ON from the live env.

## Do-no-harm / scope
Backend only; $0; paper-only; NO risk threshold moved; historical_macro FROZEN; hysteresis untouched. The swap
path is LIVE (paper_swap_enabled=True, max_per_cycle=2), so every fix ships flag-gated default-OFF +
double-gated → the live swap behavior is byte-identical until the operator flips a 70.3 flag. Activation
follow-on (operator): flip paper_atomic_swap_enabled (+ optionally paper_cross_sector_rotation_enabled, which
requires paper_swap_churn_fix_enabled ON) and paper_avg_entry_fx_fix_enabled after review.
