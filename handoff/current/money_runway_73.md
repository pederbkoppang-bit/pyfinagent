# Money Runway — phase-73.6 (FINAL — verbatim from gate `wf_9b114107-a7e`, 6 sources read in full; recommend-only, one page)

Ordered stages to MORE ABSOLUTE $. Headline honesty: **go-live is NOT eligible and the clock has not started** — every fill ever made is synthetic (bq_sim close-fills, slippage=0), so `trades_ge_100` counts REAL round trips from the Stage-2 cutover forward (0 today; ~30 synthetic whole-table). The phase-69 register's 'go-live booleans weaker than documented' note is VERIFIED FIXED in 69.2 (sustained-PSR + dd-tolerance now measure what they claim). `real_capital_enabled=False` is the hard gate beyond the 5 booleans (its SR 11-7 citation was superseded by SR 26-2 on 2026-04-17 — principles carry forward; recommend-only doc-drift note for a future docs step). External consensus: live degrades vs sim (slippage 5-10%, partial fills, psychology); even Alpaca paper is optimistically biased; our 100-round-trip + human-token bar is deliberately stricter than the retail base rate.

## STAGE 1 -- PAPER RESTORATION (unblock the credit-dead scoring rail; earn on synthetic fills)

Prerequisites:
- No prerequisite beyond operator keystrokes -- backend/.env is agent-locked, so the session cannot self-apply
- Scoring rail credit-dead since 2026-05-17 03:55:44 CEST (HTTP-400 'credit balance too low'); meta-scorer LLM leg failed every trading day since 05-22; this is the P0 DEFECT, not the P4 regime policy

Evidence anchors:
- money_diagnosis_72.md P0 (root onset + mechanism) and P3/P4 (post-restoration lever + regime policy)
- operator_decision_sheet_72.md ACT-NOW block (#1-#4) -- the verbatim unlock
- 72.0 immutable live_check: a post-restoration cycle log line showing non-degraded scoring (meta-scorer LLM leg active, no degraded-guard fire, >=1 non-HOLD rec OR honest market-driven HOLD)
- Note: every Stage-1 fill remains SYNTHETIC bq_sim (close-fill, slippage=0, unlimited liquidity) -- restores earning capacity but does NOT advance the go-live clock

**Your decisions (verbatim-actionable):**
- Decide Anthropic direct-API credits: top up the metered key at console.anthropic.com, OR reply `ANTHROPIC DIRECT: ABANDON` (reroutes the affected legs via 72.0.1/72.0.2)
- Append to backend/.env: PAPER_SYNTHESIS_INTEGRITY_ENABLED=true
- Append to backend/.env: PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true
- Restart the backend: launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
- Run the reconciliation grep: ! grep -nE 'SYNTHESIS_INTEGRITY|RISK_JUDGE|SWAP|SCALE_OUT|SESSION_BUDGET|PAPER_MARKETS|PAPER_TRADING|MODEL|ANTHROPIC|GEMINI' backend/.env
- Post-restoration, one lever at a time (3-5 cycles between): record KS-PEAK-RESET:APPROVED, then Seq-2 soft-sector-diversity per operator_decision_sheet_72.md §P3

## STAGE 2 -- REAL-FILL RUNWAY (phase-68: swap synthetic bq_sim close-fills for Alpaca-paper real fills)

Prerequisites:
- Stage 1 earning capacity restored (a dead rail produces no fills to execute)
- phase-67 P0s + the 67.4 Sunday revert done (already satisfied)
- Ordered internal chain: 68.1 (execution_router reached, DARK) -> 68.2 (flatten shorts + >=5 shadow days) -> 68.3 (cutover, token-gated) -> 68.4/68.5 -> 68.6; 68.6 is ALSO gated on 68.5 done so the tracker never ingests un-sanitized fills

Evidence anchors:
- masterplan phase-68 header: 'every fill ever made is synthetic'; the 'EXECUTION_BACKEND never reaches execution_router, stuck on bq_sim' clause is from the phase-68 step detail (citation split per Q/A wf_65b25f78-5ec)
- paper_trader.py:268-277 -- bq_sim fills at close with slippage=0; alpaca_paper slippage = fill_price - close (:277)
- design_pack_73/d_cost_promotion.md §1-2 -- the 73.4 slippage haircut is an ESTIMATE precisely because bq_sim fills at close; real fills make slippage_usd measured
- Alpaca docs (read in full): paper fills off real-time quotes, random 10% partial fills, but 'order quantity is not checked against the NBBO quantities' and ignores market impact -- Stage-2 numbers stay optimistically biased; the 68.6 divergence report (GO_LIVE_CHECKLIST 4.4.2.5) is the SR-11-7/26-2 'outcomes analysis / back-testing' leg

**Your decisions (verbatim-actionable):**
- Confirm Alpaca paper API credentials are present in backend/.env before 68.3 (68.1's LOUD missing-creds log surfaces any gap)
- Record the cutover token: EXEC-BACKEND: ALPACA_PAPER (68.3 is DARK until this is recorded)
- Approve the learn-loop reflection LLM cost when 68.4 presents the measured per-close reflection cost (flat-fee Gemini rail; token-throughput, ~$0 marginal)

## STAGE 3 -- GO-LIVE (honest 5-boolean readout; the 58.1 human token; the SR-11-7/26-2 real-capital gate)

Prerequisites:
- 68.6 real-fill clean-window evidence exists (Sharpe vs 0.82, missed days, bq_sim-vs-real divergence) -- 58.1 is fed by 68.6
- ~100 REAL round trips accrued over the immutable clean window (trades_ge_100), which takes months from the first alpaca_paper fill -- the go-live clock only starts at Stage-2 cutover
- All 5 booleans green (promote_eligible) AND a compliance review that wires the real-capital deployment path
- 58.1's own dependency: depends_on_step 56.2

Evidence anchors:
- paper_go_live_gate.py (full): 5 booleans AND-ed -> promote_eligible; TRADES_THRESHOLD=100; two under-spec booleans FIXED in 69.2 (verified)
- register.md:26 + handoffs_69.4.md:21 'FIXED-69.2 (crit5 go-live booleans)' -- the register note asked about is resolved
- settings.py real_capital_enabled=False -- SR 11-7 paper-only hard gate; consumed by monthly_champion_challenger.run_monthly_sortino_gate; MUST stay False until compliance review
- sia-partners SR 26-2 (2026-04-17) supersedes SR 11-7 -- stale code citation, principles survive (recommend-only doc-drift note)
- masterplan 58.1 -- operator verbatim spend decision recorded in live_check_58.1.md BEFORE any live LLM cycle (running one without it = automatic FAIL); go-live gate baseline 1/5 on 2026-06-01
- frontier_map_73.md #10 -- recommend-only, gate-everything, human token as promotion authority (Man-Group identical-gates posture)

**Your decisions (verbatim-actionable):**
- Only after 68.6 shows real-fill evidence: record in live_check_58.1.md `LLM SPEND: APPROVED <budget>` OR `LLM SPEND: DECLINED` (with date)
- If APPROVED: re-enable the cron `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist` and verify the kill switch is ACTIVE before the window starts
- DEFER (not now): keep real_capital_enabled=False until a compliance review wires the real-capital deployment path -- no keystroke in this runway flips real money
- On any eventual go-live, start at reduced size (25-50% of intended / risk 1-2% per trade) and graduate toward half-Kelly only after ~100-200 verified REAL trades -- aligns with 73.3's ~100-150-clean-trade calibrated-sizing defer