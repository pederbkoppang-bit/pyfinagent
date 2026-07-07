---
name: funnel-zero-trade-66-2
description: phase-66.2 funnel audit — cash-BUY path needs NO swap; rail death = structural all-HOLD; probe-green != rail-green (local-only check); n_trades excludes stop-loss sells; EXECUTION_BACKEND unset in launchd => bq_sim always; short_market_value is SIGNED; drawdown alarm DESC phantom
metadata:
  type: project
---

phase-66.2 research (2026-07-07), full brief at handoff/current/research_brief_66.2.md:

- **Cash-deployment BUY path exists**: decide_trades §2 (portfolio_manager.py:147-399) is a plain loop over candidate_analyses with rec in _BUY_RECS; swap path (:439) only fires for sector_blocked — impossible on an empty book. paper_swap_churn_fix is a NON-factor for first-BUY-from-cash (all 3 effects need holdings).
- **Rail death = structural all-HOLD**: empty cc response -> no JSON -> {"action":"HOLD","confidence":0,"score":5} (autonomous_loop.py:2092-2096) -> buy_candidates=[] -> 0 orders. Gates never evaluated. 06-11 already degraded (3/5, gz-06-12 line 2648197); "06-15 rail death" date is soft.
- **Probe-green != rail-green**: 15 failing cycles, ZERO "probe FAILED" lines — `claude auth status` is local-only (66.4); 07-06 18:02Z still "claude CLI exited with code 1: <EMPTY msg>" in a believed-post-credential-restore window -> suspect launchd context (keychain/PATH/proxy), not credentials. Discriminator: llm_call_log ok-rate per cycle.
- **n_trades UNDERCOUNTS**: Step-5.6 stop-loss sells never increment trades_executed (:1084-1102 vs :1214). Book emptied to 100% cash via SNDK 06-23 + 000660.KS 07-03 stop-outs while cycle_history showed n_trades=0 throughout. Funnel tools must count paper_trades by reason.
- **Alpaca**: launchd plist (com.pyfinagent.backend.plist) has NO EXECUTION_BACKEND / ALPACA keys; no load_dotenv anywhere -> loop is bq_sim FOREVER, mock alpaca fills even if flipped. short_market_value is SIGNED (equity = cash + long_mv + short_mv, account-plans doc) -> -$13,842.89 = real short positions; origins = alpaca MCP sessions / alpaca_shadow_drill.py (real 1-share BUYs, `uat-shadow-*` order ids, docstring's "+1 SELL" is stale) / mcp_ab_test.py (HAS live sell path :159). Alpaca paper = margin account, naked SELL opens a short. Check = read-only get_account_info + get_all_positions from Main.
- **paper_portfolio single row is BY DESIGN** (portfolio_id='default', USD aggregate; market lives on paper_positions). Real check: KR paper_trades total_value must be USD-magnitude (~$850 not 857843) — execute_sell LOG prints local KRW with $ sign (cosmetic), BQ row converts at paper_trader.py:416.
- **Drawdown alarm phantom**: P1 "drawdown -61.51%" pages on a +20% book (07-06) — drawdown_alarm consumes get_paper_snapshots DESC (phase-47.4 trap family). Fix/ignore; don't let it mask real pages.
- **Sector cap calibration**: paper_max_per_sector=2 + all-Tech momentum slate -> first redeploy caps at ~2 BUYs/cycle; 2-of-5 is correct gating, not a defect. 06-09 REJECT-BUY (066570.KS) went via swap path pre-57.1; binding gate now at candidate-build chokepoint (portfolio_manager.py:194-212) but flag dark.
- **Log archives**: handoff/logs/backend.log.<rotationstamp>.gz retain ~3.5wk; CompactFormatter lines are DATE-LESS local CEST — date cycles by rotation stamp + 20:00-CEST cycle starts. Kill switch NOT latched (14 pause/14 resume, last 06-11 manual).

**Why:** these are the load-bearing facts for 66.2 GENERATE (funnel_report tool + first-BUY verification) and they took a full log-archive + code trace to establish.
**How to apply:** any zero-trade diagnosis starts at llm_call_log ok-rate, THEN analysis_results recommendations, THEN decide_trades skip-lines; never trust n_trades or the probe alone. See [[cc-rail-guard-66-1]], [[swap-churn-engine-60-2]], [[project-multimarket-universe-wiring]].
