---
name: real-fill-runway-68-0
description: phase-68.0 research — AMD/MU "price defect" is FALSE (real 2026-07 prices; stale-knowledge anchor in live_check_66.2:402); DESC phantom already fixed 66.2; router mode is per-constructor os.getenv; paper_trades has NO source column; alpaca 422 dup client_order_id; PK/AK prefix undocumented; rejected-order books-at-reference-price trap
metadata:
  type: project
---

phase-68.0 research gate (2026-07-10), full brief handoff/current/research_brief_68.0.md:

- **AMD/MU avg_entry $545.42/$1004.70 = REAL market prices** (2026-07-09 closes: AMD $546.72, MU $991.64; MU ATH close $1213.37 on 2026-06-25, Anthropic $22B HBM deal; AMD 52wk high $584.73). The "real ~$150/~$110" claim in live_check_66.2.md:402 was the 66.2 closer's stale world-knowledge anchor (AMD 52wk LOW = $137.59). 68.5 immutable criteria 1-2 unsatisfiable as written; criterion 4 (DESC phantom) ALREADY fixed by 66.2 hotfix commit 9262ed36 (drawdown_alarm.py:65-108 + test_phase_66_2_drawdown_order.py). Lesson: NEVER sanity-check prices against model memory — always live-quote cross-check.
- **Router mode**: per-CONSTRUCTOR `os.getenv("EXECUTION_BACKEND")` (execution_router.py:65-71, :268-269 — module docstring "import time" is STALE); fresh ExecutionRouter() per trade (paper_trader.py:255/:396). pydantic env_file loads model-only (settings.py:584; no execution_backend field exists); plist has only 4 env keys (CLAUDE_CODE_OAUTH_TOKEN/DEV_LOCALHOST_BYPASS/PATH/PYTHONUNBUFFERED).
- **paper_trades has NO source column** (migrate_paper_trading.py:54-66); fill.source dies at the log line (paper_trader.py:359). Dynamic INSERT rejects unknown keys (comment :283-286). 68.3 needs a migration; shadow pairs belong in a NEW paper_shadow_fills table (same-table shadow rows would pollute every consumer).
- **Cutover trap**: router fill-poll (execution_router.py:239-244) only checks filled/partially_filled; a rejected/canceled alpaca order returns fill_price 0.0 and execute_buy substitutes the reference price (:260) — an unfilled order would be BOOKED. Must handle terminal non-fill states before 68.3.
- **Alpaca externals** (all accessed 2026-07-10): 422 "client_order_id must be unique" vs ACTIVE orders (official learn guide); client_order_id <=128 chars, auto-generated; recovery via get_order_by_client_id; NO Stripe-style idempotency header. Paper fills = real-time NBBO quotes, 10% random partial fills, no fees/impact simulated. Paper account RESET invalidates API keys — flatten via close_all_positions(cancel_orders=True), never reset. PDT applies on paper (<$25k). PK(paper)/AK(live) prefix convention is UNDOCUMENTED folklore — repo's PKLIVE guard near-vacuous; real enforcement = paper=True + key-source separation. alpaca-py pinned 0.43.2 = 2025-11-04; 0.43.5 latest (2026-07-02, tolerates deprecated PDT/DTBP fields); no breaking 0.43.x changes.
- Creds: router reads os.environ ALPACA_API_KEY_ID directly; settings SecretStr alpaca fields (:128-129) are the NEWS channel and don't export — same non-propagation class as EXECUTION_BACKEND.

**Why:** 68.1-68.3 (wiring, shadow, cutover) will each re-touch these exact facts; the premise-overturn changes 68.5's scope and must not be re-litigated from the stale live_check.
**How to apply:** cite the brief; verify prices only via live quotes; treat [[funnel-zero-trade-66-2]] as the companion fact set. See also [[project-metric-source-paths]] for the DESC family.
