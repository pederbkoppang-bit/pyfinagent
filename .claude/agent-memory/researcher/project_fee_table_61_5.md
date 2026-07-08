---
name: fee-table-61-5
description: Verified 2026 per-market fee rates for step 61.5 — FINRA TAF in the immutable criterion is STALE; plug points + vacuous-window trap
metadata:
  type: project
---

Pre-pay brief for 61.5 written 2026-07-08 (`handoff/current/research_brief_61.5.md`, gate_passed).

**Fee truth (verified 2026-07-08):** the immutable criterion's FINRA TAF
$0.000166/share cap $8.30 EXPIRED 2025-12-31. Current: **$0.000195/share cap
$9.79** (eff 2026-01-01, SR-FINRA-2024-019 fee-adjustment schedule), stepping
ANNUALLY: 2027 $0.000232/$11.61, 2028 $0.000240/$12.05, 2029 $0.000249/$12.50.
The FINRA Section-1 rulebook page still shows the OLD codified text — do not
trust it over the fee-adjustment schedule. SEC Section 31 = $20.60/M (0.00206%)
eff 2026-04-04, was $0.00/M before (appropriation-driven, resets ~annually;
charge date = TRADE date, sells only). KR STT 0.20% eff 2026-01-01 (2025 was
0.15% — 2025-trade replays must use 0.15%). EU IBKR tiered 0.05% min EUR 1.25
cap EUR 29 (fixed plan min EUR 3); per-order min dominates below ~EUR 2,500.

**Why:** fee rates must be a DATED config schedule, never constants; and the
61.5 contract must pre-register charging the lawful rate (not the criterion's
expired figure) for operator adjudication via the FEE TABLE token.

**How to apply:** cost model plugs at paper_trader.py:189 (BUY) / :388 (SELL —
sell_qty + position.market + LOCAL notional in scope); half-spread as a fee
LINE not a fill-price mutation; swap-delta (portfolio_manager.py:495-575) is
NOT a plug site. holding_days persisted per SELL row (:423) makes the churn
query one COUNTIF, but the >=5-day window is VACUOUS until 66.2 redeploys
capital (0 post-flag sells; ~100% cash since 07-03) — invoke the Cycle-66
vacuous-pass doctrine. Slope monitor: NW lag h-1 PLUS non-overlapping check
(Boudoukh et al.); separate "zero slope, adequate n" from "insufficient n".
See [[project_cost_truth_66_3]] for cost-gauge semantics.
