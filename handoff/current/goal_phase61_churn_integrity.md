# Goal Prompt -- goal-phase61-churn-integrity

Set by operator 2026-06-11 (4 dashboard screenshots + directive: "audit and fix the issues
shown -- we are currently only buying and selling two stocks a day contributing to high fee;
deep research a better system for higher profit without high trading fee; deep research our
trades for the last 8 days; look at all the reasoning made by the agents; also fix the new
bugs: missing company on Reports, score 0.00, 30D trend empty, missing Learnings, stocks
from other markets shown in wrong currency").

Evidence base: 10-agent fleet audit 2026-06-11 (workflow wf_7a3b2a6c-0da; transcripts under
the session dir). Every data claim re-derived with fresh SQL by an independent verifier;
all 14 code root-cause claims adversarially re-verified (14/14 CONFIRMED, 0 REFUTED).
External research: 11 sources read in full (Garleanu-Pedersen 2013, Guasoni & Muhle-Karbe
2012, Qian/Sorensen/Hua 2007, Dixit 1989 via secondary, Rasekhschaffe arXiv 2602.00196,
Meng & Chen arXiv 2605.23905, Yao & Zheng arXiv 2606.08285, FINRA/SEC/PwC/IBKR fee pages).

INSTALL (Main session): (1) save this file as `handoff/current/goal_phase61_churn_integrity.md`
(drop the DRAFT suffix); (2) append the JSON payload below byte-for-byte to `phases` in
`.claude/masterplan.json` (all statuses `pending` -- never flip anything in the same edit);
(3) replace `handoff/current/active_goal.md` with the refresh payload below; (4) commit
`phase-61.0: install goal-phase61-churn-integrity (operator-approved)`; (5) begin 61.1 with
the researcher spawn per the harness protocol.

## North star linkage (N* = Profit - (Risk Exposure + Compute Burn))

The fee complaint is real but mis-attributed: simulated fees over 2026-06-03..06-10 were only
$17.14 (0.07% of NAV). The actual damage is the churn's realized P&L: 11 of 16 sells closed
positions held <=2 days for a net -$139.83, while the 4 long holds (38-39 days, exited via
trailing stops) made +$1,355.22 in the same window. Churn + fees = ~$157 = 0.66% of NAV per
8 days (~30%/yr pace if sustained) -- pure N* drag. Every major cause already has a fix that
is sitting DARK: the phase-60.2 churn fix, the phase-60.3 data-integrity fix, and the
phase-57.1 binding-REJECT gate are all flag-OFF, and the running backend process predates
ALL phase-60 commits, so none of that code is even loaded. Separately, the decision inputs
the agents reason over are being poisoned right now (synthesis timeouts persisting fake
0.00/HOLD rows -- still firing tonight), and the fee model understates true costs (no KRX
0.20% sell tax, zero slippage), overstating KR paper P&L. This goal converts already-built,
already-PASSed work into live behavior, fixes the integrity bugs, and puts turnover policy
on a measured, cost-aware footing.

## Objective (one sentence)

Stop the churn engine from destroying realized P&L, make every decision input and money
display trustworthy, and adopt a research-grounded per-market cost model with measured,
operator-gated turnover levers -- without touching the trailing-stop + long-hold path that
produces the profits.

## Verified current state

PRIMARY (adversarially re-verified 2026-06-11 against live BQ/API/logs and HEAD 0ea41928):

1. CADENCE: 28 trades 06-03..06-10; 5 of 6 trading days at exactly 2 BUYs + 2 SELLs = the
   hard cap `paper_swap_max_per_cycle=2` (settings.py:305-310). Same ~6 tickers cycle: DELL
   6 trades = 3 complete round-trips; SNDK 4; MU/STX/000660.KS 3 each. MU was stop-sold
   06-05 at +75.1% (38d hold), re-bought 06-08 at a 7.6% HIGHER price, swap-sold 06-09 at
   -6.27%.
2. CHURN MECHANISM (AW-5, live all window): holdings bought 1-2 sessions earlier have no
   same-cycle analysis (`paper_reeval_frequency_days=3`), get sentinel `score = 0.0`
   (portfolio_manager.py:503-507, comment "KNOWN DEFECT (AW-5)"), and the 0.01-epsilon
   denominator (portfolio_manager.py:561) fabricates ~60,000% deltas vs the 25% swap bar.
   9 of 10 swap-out sells displaced unanalyzed 1-2-day-old positions. Fix shipped dark in
   commit 7f0de140 behind `paper_swap_churn_fix_enabled` (settings.py:311, live False);
   replay shows flag-ON suppresses 11/13 recorded swaps (replay_60_2_results.md).
3. DEPLOY GAP: running uvicorn PID 77557 started 2026-06-11 11:43:34, BEFORE the 13:51
   phase-60.2 commit -- phase-60.2/60.3/60.4 code is NOT in the running process. Flags are
   pydantic fields read from backend/.env (env names PAPER_SWAP_CHURN_FIX_ENABLED,
   PAPER_DATA_INTEGRITY_ENABLED), absent from the Settings UI map; `get_settings()` is
   lru_cached -- .env edit + restart is the only deterministic path.
4. GOVERNANCE GAP: two BUYs executed with `risk_judge_decision='REJECT'` (DELL 06-03;
   066570.KS 06-09, stopped out -9.68% next day). Binding gate
   `paper_risk_judge_reject_binding` (settings.py:277) shipped dark in 57.1. RiskJudge also
   sized every trade without portfolio sector context (only built when the binding flag is
   ON, autonomous_loop.py ~783-788) and saw corrupted KRW-as-USD market caps ("$1.63
   quadrillion") -- the 60.3 fix for that is dark too.
5. CONVICTION SATURATION: every BUY all window carried `conviction 10.00; fallback (LLM
   unavailable)` -- meta_scorer's fallback (meta_scorer.py:138-142) clamps 78-163 composite
   scores into [1,10], destroying differentiation exactly when its LLM is down (it was down
   every cycle 06-03..06-10; cause undiagnosed).
6. DEAD EXIT PATH: `signal_downgrade` sells can never fire -- paper_trader.py:305/329
   persists the TRADE REASON ("swap_buy") into the position's `recommendation` field, and
   the downgrade rule requires old_rec in {BUY, STRONG_BUY} (portfolio_manager.py:50,114,127).
   The only live exits are stops and the (defective) swap engine.
7. FEE MODEL: flat 0.1%/side for ALL markets (settings.py:337; paper_trader.py:189,388),
   zero slippage on the default bq_sim backend (execution_router.py:85-126). No KRX sell-side
   securities transaction tax (0.20% since 2026-01-01), no SEC Section 31 ($20.60/M sales
   since 2026-04-04), no FINRA TAF, no EU per-order minimums. KR net paper P&L is overstated
   ~0.2% per round-trip. All 3 KR entries in the window closed at a loss (~-$117 combined).
8. BUG company "—": autonomous `_persist_analysis` reads company_name ONLY from
   `full_report['market_data']['name']` -- a lite-path-only key; the orchestrator's full
   report has zero `market_data` keys (autonomous_loop.py:2408-2412; exposed when 60.1
   restored the deep pipeline). Manual path uses `quant.company_name` and works
   (tasks/analysis.py:211-212).
9. BUG score 0.00 (ACTIVELY FIRING): claude_code rail synthesis/critic calls hit the 120s
   subprocess timeout (claude_code_client.py:85,335; its own :333 comment recommends 150;
   DELL succeeded at 115.15s = 96% of budget); the error result is persisted as a
   legitimate-looking 0.00/HOLD row with no guard (orchestrator.py:1531-1545,
   autonomous_loop.py:1551,1563-1565,2413). A fresh HPE row was poisoned at 19:09Z and ~15
   more timeouts logged 20:42-21:07 CEST tonight. These 0.0 rows win the rk=1 dedup and
   poison every downstream consumer, and silently neutralize the ticker for that cycle's
   trade decisions.
10. BUG 30D TREND: has NEVER rendered. Backend dedups to one row per ticker (phase-25.H,
    bigquery_client.py:264-276) while the sparkline (phase-44.4, reports-columns.tsx:114-152)
    needs >=2 rows per ticker from that same payload. No backend score-history source exists.
11. BUG wrong currency: backend hardcodes `base_currency: "USD"` on every position
    (paper_trader.py:313,334,481 -- it labels the NAV base) while frontend `resolveCurrency`
    gives base_currency top priority (format.ts:161-171) for the per-share columns that are
    LOCAL currency by the phase-50.2 contract (types.ts:653-657) -- so KRW magnitudes render
    under USD symbols. Mixed locales: hardcoded `$${v.toFixed(2)}` templates vs Intl cells
    falling back to the nb-NO browser locale ("1 828,25 USD") -- reproduced byte-exactly.
    P&L +0.00% is the stored mark frozen at the once-per-day mark_to_market (38s after the
    buy, KRX closed). KR stops are checked against marks up to ~24-36h stale
    (paper_trader.py:591-598).
12. LATENT MONEY BUG (not yet triggered): an add-on BUY to an open non-US position writes a
    USD-per-share average into LOCAL `avg_entry_price` (paper_trader.py:288-297) -> garbage
    realized P&L (:443) AND a breakeven stop ~1500x below the KRW price that can never
    trigger (:1137, :591-598). Confirmed unfired so far (000660.KS history is strictly
    full-exit/re-entry).
13. BUG Learnings empty: `get_paper_trades_in_window` compares the STRING `created_at`
    column to a TIMESTAMP (bigquery_client.py:955-964) -> BQ 400, swallowed at
    paper_trading.py:822-823 -> divergences=[] since the endpoint shipped (2026-05-12); 74
    occurrences in backend.log. With SAFE_CAST the same window pairs 13 round-trips that
    would render immediately. regime_buckets=[] is an intentional documented stub.
    Separately, `pyfinagent_data.harness_learning_log` does not exist and its writer
    `log_slot_usage` has zero production callers -- the backtest-page sprint tile has never
    had data.
14. FRONTEND STALENESS: the dev server serves a stale bundle (`/_next/static/chunks/app/
    login/page.js` 404, ChunkLoadError) -- the known launchctl-kickstart pattern.

SECONDARY (true at audit time; re-verify during step research because they move):
- NAV $23,828.73, cash $22,874.93, total_pnl +19.14% (paper_portfolio 06-10/06-11 cycles).
- PID 77557 / process start times; backend.log occurrence counts (log grows).
- Implied KRW fill FX 1532.7 (not persisted anywhere -- recoverable only by division).

## CRITICAL constraints (violating any is an automatic FAIL)

1. Do-no-harm: every BEHAVIOR change is config-gated default OFF with ON-vs-OFF measurement
   (phase-60 discipline). Pure bug fixes (NULL company, SAFE_CAST, locale/display, the
   latent add-on averaging correction) are exempt from flag-gating but require regression
   tests + live evidence.
2. NO score/rank-hysteresis family code (53.1/55.3 rulings + the phase-60 install decision
   that dropped 60.5) unless the operator replies the verbatim token `HYSTERESIS: AUTHORIZE`.
   `paper_swap_min_delta_pct=25.0` is untouchable.
3. The trailing-stop engine and long-hold path (paper_trader.py:1095-1123 + the HWM logic)
   produced +$1,355.22 of the window's realized P&L -- do not modify it except the add-on
   averaging money-safety fix, which must carry a regression test.
4. No flag is flipped without its verbatim operator token (see checkpoint mechanics).
5. Full harness protocol per step: researcher before contract, Q/A after generate, five
   files, log-last, live_check evidence. Playwright MCP capture for every UI claim; BQ MCP
   rows for every data claim. (MCP liveness verified 2026-06-11: Playwright pinned 0.0.76
   works; Figma connector authenticated but View-seat and session-only -- advisory, never
   verification-load-bearing.)
6. No new LLM API spend without an `LLM SPEND:` token. Do not disturb the running phase-58.1
   $25 live window.
7. No emojis anywhere. Restart discipline: backend via
   `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend` (kills parent+children);
   frontend via `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.frontend`.

## Deliberately out of scope

- Real-money go-live and spend decisions (phase-58 owns that runway).
- Strategy rotation work (operator-stopped, low money value).
- New markets; this goal only fixes currency handling for the existing US/KR/EU setup.
- The regime-buckets feature build (documented stub -- only the divergences reader bug is
  in scope).
- Figma-driven redesigns.

## Execution order

61.1 -> 61.2 -> 61.3 -> 61.4 -> 61.5 (sequential). The churn measurement that 61.5 consumes
starts accruing the moment 61.1's flags flip, so calendar time overlaps; 61.5's lever
decision requires >=5 trading days of post-flag data.

## Operator checkpoint mechanics (verbatim reply tokens)

- At 61.1 start, Main asks and the operator replies one token per flag:
  `60.2 FLAG: ON` | `60.2 FLAG: KEEP OFF`  (churn fix -- audit recommendation: ON; replay
  suppresses 11/13 swaps, 10 of them sentinel artifacts)
  `60.3 FLAG: ON` | `60.3 FLAG: KEEP OFF`  (data integrity -- recommendation: ON; RiskJudge
  reasoned over "$1.63 quadrillion" market caps on every KR trade)
  `57.1 FLAG: ON` | `57.1 FLAG: KEEP OFF`  (binding REJECT -- recommendation: ON; two REJECT
  buys executed, one -9.68% next day)
- At 61.4: `SPRINT TILE: WIRE` | `SPRINT TILE: PRUNE` (recommendation: PRUNE -- never-wired
  scaffolding; stress-test doctrine).
- At 61.5: `FEE TABLE: ON` | `FEE TABLE: KEEP OFF` after the ON-vs-OFF replay;
  `TURNOVER LEVERS: APPROVE <subset>` | `TURNOVER LEVERS: DECLINE` for any min-holding /
  re-entry-cooldown / score-smoothing promotion; optionally `HYSTERESIS: AUTHORIZE` to lift
  constraint 2 (new external evidence: Dixit 1989 option-value bands; Garleanu-Pedersen
  partial adjustment; Rasekhschaffe 2026 -- 21-day signal smoothing cut turnover 82% and
  flipped an LLM-signal strategy from net-loss to net-profit at just 3 bps costs, vs this
  system's 30-40 bps KR round-trips).

## Masterplan installation payload (canonical; install byte-for-byte)

Append this one phase object to `phases` in `.claude/masterplan.json`. The
`success_criteria` arrays are the immutable acceptance criteria -- do NOT edit them.

```json
{
  "id": "phase-61",
  "name": "Churn shutdown + decision/display integrity (operator goal 2026-06-11, goal-phase61-churn-integrity). Evidence: 10-agent fleet audit 2026-06-11 (wf_7a3b2a6c-0da), 14/14 root causes adversarially CONFIRMED against live BQ/API/logs; 8-day trade forensics: fees $17.14 (0.07% NAV) vs churn realized -$139.83 on <=2-day holds vs +$1,355.22 on long holds; sentinel churn engine (AW-5) live with fix dark; running backend predates all phase-60 commits. Do-no-harm: behavior changes config-gated default OFF; pure bug fixes exempt but test-covered. Hysteresis family banned absent HYSTERESIS: AUTHORIZE (53.1/55.3 + phase-60 install ruling). Trailing-stop engine untouchable except the add-on averaging money-safety fix.",
  "status": "pending",
  "depends_on": ["phase-60"],
  "gate": null,
  "steps": [
    {
      "id": "61.1",
      "name": "Activate the dark fixes + deploy phase-60 code: operator flag tokens (60.2/60.3/57.1), backend/.env edits, backend restart (running PID predates ALL phase-60 commits so none of that code is loaded), frontend kickstart (stale /login chunk 404), first-cycle live evidence.",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": null,
      "audit_basis": "Fleet audit 2026-06-11: uvicorn PID 77557 started 11:43:34 vs phase-60.2 commit 7f0de140 at 13:51:05 (fixes inert); live settings load: paper_swap_churn_fix_enabled=False (settings.py:311), paper_data_integrity_enabled=False (settings.py:42), paper_risk_judge_reject_binding=False (settings.py:277); flags absent from settings_api.py _FIELD_TO_ENV (manual .env only); REJECT buys executed DELL 06-03 + 066570.KS 06-09 (-9.68% next day); 9/10 swap-outs displaced 1-2-day-old unanalyzed holdings; live_check_60.2.md section D promotion PENDING; Playwright probe: /login ChunkLoadError (stale dev bundle).",
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -c \"from backend.config.settings import get_settings; s = get_settings(); print('churn_fix', s.paper_swap_churn_fix_enabled, 'data_integrity', s.paper_data_integrity_enabled, 'rj_binding', s.paper_risk_judge_reject_binding)\" && test -f handoff/current/live_check_61.1.md",
        "success_criteria": [
          "the operator's verbatim flag tokens (60.2 FLAG / 60.3 FLAG / 57.1 FLAG, each ON or KEEP OFF) are recorded in handoff/current/live_check_61.1.md and backend/.env matches them exactly; no flag changed without its token",
          "post-restart, the running uvicorn process start time is later than the phase-60.4 commit timestamp (ps -o lstart vs git log evidence pasted verbatim), proving phase-60.2/60.3/60.4 code is loaded",
          "frontend kickstarted via launchctl; Playwright capture shows http://localhost:3000/login loads without ChunkLoadError",
          "first post-restart daily-cycle evidence in live_check_61.1.md as verbatim BQ rows: if 60.2 FLAG: ON, zero swap_for_higher_conviction SELLs of holdings lacking a same-cycle analysis_results row; if 57.1 FLAG: ON, zero executed trades with risk_judge_decision='REJECT'",
          "handoff/harness_log.md cycle entry appended before the status flip"
        ],
        "live_check": "live_check_61.1.md containing: verbatim operator flag tokens, ps -o lstart output post-restart vs commit timestamps, Playwright screenshot path for /login, and first post-flag cycle BQ rows from financial_reports.paper_trades"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "61.2",
      "name": "Decision-input integrity: never persist synthetic 0.00/HOLD on synthesis failure (actively firing), claude_code timeout 120 -> >=150s, company_name fallback for autonomous full-path rows, meta-scorer fallback rank-normalization + unavailability alert + root-cause, dead signal_downgrade exit path, RiskJudge portfolio context in advisory mode.",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "61.1",
      "audit_basis": "Verified 2026-06-11: timeout chain claude_code_client.py:85,333,335 -> orchestrator.py:1531-1545 -> autonomous_loop.py:1551,1563-1565,2413 persists 0.00/HOLD with $.final_synthesis.error set (BQ rows MU/009150.KS 18:35Z, new HPE 19:09Z; ~15 more timeouts 20:42-21:07 CEST); company_name NULL on all path='full' autonomous rows (autonomous_loop.py:2408-2412 reads lite-only market_data; manual path tasks/analysis.py:211-212 works); constant 'conviction 10.00; fallback (LLM unavailable)' on all 12 window BUYs (meta_scorer.py:138-142 clamps 78-163 composites); signal_downgrade dead because paper_trader.py:305,329 stores trade reason in recommendation vs _BUY_RECS match at portfolio_manager.py:50,114,127; _rj_portfolio_ctx gated behind the binding flag (autonomous_loop.py ~783-788).",
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q && test -f handoff/current/live_check_61.2.md",
        "success_criteria": [
          "a synthesis result carrying final_synthesis.error (or missing scoring_matrix) is never persisted as a 0.0 final_score with a default HOLD: it is either routed to the existing lite fallback or persisted with NULL score plus an explicit degraded marker; a regression test simulates the timeout and asserts no 0.0/HOLD row is written and the same-cycle trade-decision input is not silently neutralized",
          "claude_code synthesis/critic-class calls run with timeout >= 150s (per the file's own recommended_step_timeout) and the value is configurable",
          "_persist_analysis falls back to the quant company_name when market_data.name is absent; live_check shows BQ rows from a post-fix autonomous full-path cycle with non-null company_name",
          "the meta-scorer fallback no longer emits a constant saturated conviction: composite scores are rank/percentile-normalized into the 1-10 scale, and a WARN-level alert fires after 2 consecutive all-fallback cycles; the root cause of the 06-03..06-10 LLM unavailability is diagnosed and documented in experiment_results.md",
          "positions persist the analysis recommendation (not the trade reason) so the signal_downgrade rule at portfolio_manager.py:127 can match; covered by a unit test",
          "RiskJudge receives portfolio sector-breakdown context regardless of paper_risk_judge_reject_binding"
        ],
        "live_check": "live_check_61.2.md containing BQ rows from at least one post-fix autonomous cycle: non-null company_name on full-path rows, zero new rows with final_score=0.0 AND final_synthesis.error set, and non-constant conviction values in paper_trades.signals"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "61.3",
      "name": "Money-display + currency correctness: pre-emptive fix of the latent add-on-buy USD-into-LOCAL averaging bug (money-corrupting on trigger: garbage realized P&L + untriggerable breakeven stop), market-first currency resolution on positions price columns, single en-US USD locale policy, non-US P&L staleness honesty, per-market mark-to-market decision.",
      "status": "pending",
      "harness_required": true,
      "priority": "P0",
      "depends_on_step": "61.2",
      "audit_basis": "Verified 2026-06-11: paper_trader.py:288-297 averages USD cost into LOCAL avg_entry_price (downstream :443 realized P&L, :1137 breakeven stop, :591-598 stop check -- confirmed unfired so far); base_currency='USD' hardcoded at paper_trader.py:313,334,481 vs resolveCurrency explicit-first at format.ts:161-171 applied to LOCAL columns at positions-columns.tsx:144-148,169-173,254-258 (types.ts:653-657 contract); mixed locale: hardcoded $-templates at positions-columns.tsx:152,261 vs Intl undefined-locale (nb-NO fallback) at positions-columns.tsx:74 + cockpit-helpers.tsx Dollar -- screenshot strings reproduced byte-exactly; stored P&L frozen at once-per-day mark_to_market (callers autonomous_loop.py:988,1028,1255); KR stops checked vs marks up to ~24-36h stale.",
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'addon or avg_entry or currency or 61_3' -q && cd frontend && npm run build && cd .. && test -f handoff/current/live_check_61.3.md",
        "success_criteria": [
          "add-on BUYs average avg_entry_price in LOCAL currency; a regression test performs a KR add-on buy and asserts avg_entry_price and the breakeven-advanced stop both remain KRW-scale",
          "positions Entry/Current/Stop columns resolve display currency market-first (KR rows render KRW, EU rows EUR); an automated test asserts no USD symbol is attached to a KRW-magnitude value (mirroring the 60.3 prompt regex test)",
          "one locale policy for USD cells: hardcoded toFixed templates replaced with the shared formatCurrency, and Intl/NumberFlow USD branches pinned to en-US; a unit test passes under a forced nb-NO default locale",
          "non-US rows no longer mix a live local price with an unlabeled stale P&L: stored P&L carries an as-of indicator (mark timestamp) and/or a clearly-labeled live local return",
          "a researcher-grounded decision on per-market mark_to_market scheduling (e.g. post-KRX-close ~07:00 UTC) is documented, closing or explicitly deferring the stale-KR-stop-check gap with rationale",
          "Playwright capture of the positions table with the live KR position showing corrected currency rendering and consistent number formats"
        ],
        "live_check": "live_check_61.3.md containing the Playwright screenshot path of the positions table (KR row rendering KRW, consistent locales) plus the verbatim BQ paper_positions row it renders from"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "61.4",
      "name": "Learnings + Reports history restoration: SAFE_CAST divergences fix (74 swallowed BQ 400s since 2026-05-12), error-vs-empty distinction in the API, repo-wide STRING-vs-TIMESTAMP audit on paper_trades.created_at, backend per-ticker score history for the 30D TREND sparkline (never rendered since shipping), sprint-tile WIRE|PRUNE operator decision.",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "61.3",
      "audit_basis": "Verified 2026-06-11: bigquery_client.py:955-964 compares STRING created_at to TIMESTAMP (BQ 400, 74 log occurrences), swallowed at paper_trading.py:822-823; SAFE_CAST re-run pairs 13 round-trips for the 30d window; adjacent get_paper_trades_for_ticker_since (:966-985) shows the correct string-comparison pattern; 30D TREND structurally dead: rk=1 dedup (bigquery_client.py:264-276, phase-25.H) vs sparkline needing >=2 rows/ticker (reports-columns.tsx:114-152, phase-44.4) -- live API: 30 rows, 30 unique tickers; pyfinagent_data.harness_learning_log does not exist, writer log_slot_usage (slot_accounting.py:30) has zero production callers, legacy writers point at non-existent 'trading' dataset (learning_logger.py:70, backend/autonomous_loop.py:73,86).",
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'learnings or paper_trades_window or 61_4' -q && curl -s 'http://localhost:8000/api/paper-trading/learnings?window_days=30' | python -c \"import json,sys; d=json.load(sys.stdin); assert len(d['reconciliation_divergences']) >= 10, d; print('divergences', len(d['reconciliation_divergences']))\" && test -f handoff/current/live_check_61.4.md",
        "success_criteria": [
          "get_paper_trades_in_window no longer compares STRING to TIMESTAMP; the live learnings endpoint returns >= 10 reconciliation divergences for window_days=30 (post backend restart, cache expired or busted)",
          "the learnings response distinguishes reader failure from genuinely-empty data (e.g. a divergences_error field) so a silent month-long breakage cannot recur",
          "a repo audit finds and fixes (or records a clean bill for) every other TIMESTAMP-function predicate against paper_trades.created_at",
          "the backend exposes per-ticker score history (ARRAY_AGG of up to the last 30 scores per ticker in get_recent_reports, or a dedicated history endpoint) and the Reports 30D TREND column renders a sparkline for tickers with >= 2 analyses",
          "the operator's verbatim SPRINT TILE: WIRE or SPRINT TILE: PRUNE token is recorded and executed (WIRE = create harness_learning_log via learning_schema.py create_learning_log_table + a production caller for log_slot_usage + fix the 'trading'-dataset writers; PRUNE = remove the tile, endpoint, and dead writers)",
          "CommandPalette links /learnings directly instead of the redirect",
          "Playwright captures of the populated /learnings page and the Reports table with rendered 30D TREND sparklines"
        ],
        "live_check": "live_check_61.4.md containing the verbatim curl output showing >= 10 divergences, Playwright screenshot paths for /learnings and /reports, and the operator's verbatim SPRINT TILE token"
      },
      "retry_count": 0,
      "max_retries": 3
    },
    {
      "id": "61.5",
      "name": "Cost-aware turnover policy (research-grounded, measured, operator-gated): per-market transaction-cost table (config-gated default OFF), >=5-trading-day post-flag churn measurement vs baseline, the 55.3-sanctioned minimum-holding-period lever ONLY if churn persists, per-market score-to-forward-return slope estimation as a standing signal-validity monitor. Hysteresis family stays banned absent HYSTERESIS: AUTHORIZE.",
      "status": "pending",
      "harness_required": true,
      "priority": "P1",
      "depends_on_step": "61.4",
      "audit_basis": "External research 2026-06-11 (11 sources in full): KRX securities transaction tax 0.20% sell-side effective 2026-01-01 (PwC); SEC Section 31 $20.60/M sales effective 2026-04-04 + FINRA TAF $0.000166/share cap $8.30 with 1-cent round-up (FINRA notices); IBKR Xetra tiered 0.05% min EUR 1.25 -- per-order minimums dominate below ~EUR 2,500 tickets; Rasekhschaffe arXiv 2602.00196: unsmoothed daily LLM signals = 249x turnover and net losses at 3 bps, 21-day smoothing cut turnover 82% and flipped net sign, break-even 25.4 bps < KR round-trip 30-40 bps; Garleanu-Pedersen 2013 partial adjustment; Qian 2007 turnover proportional to sqrt(1 - forecast autocorrelation); Meng & Chen 2605.23905 alpha half-lives ~18 months (not days). Internal: current model charges flat 0.1%/side zero-slippage (settings.py:337, execution_router.py:85-126); phase-60 ruling sanctions a minimum-holding-period lever ONLY if post-60.2 churn persists.",
      "verification": {
        "command": "cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'fee or cost_model or holding or 61_5' -q && test -f handoff/current/live_check_61.5.md",
        "success_criteria": [
          "a per-market transaction-cost model exists config-gated default OFF: US sell-side SEC 0.00206% + FINRA TAF $0.000166/share (cap $8.30, 1-cent round-up); KR sell-side +0.20% securities transaction tax; EU 0.05% min EUR 1.25 per order; optional per-market half-spread bps; and a minimum-ticket notional floor -- with an ON-vs-OFF replay over the last 30 days of recorded trades quantifying the paper-P&L restatement, and promotion only on a verbatim FEE TABLE: ON token",
          ">= 5 trading days of post-61.1 churn measurement vs the audited baseline (11/16 sells at <=2-day holds; 2 swap pairs/day) recorded with verbatim BQ evidence",
          "IF more than 30% of sells in the measurement window still close <=2-day holds: a minimum-holding-period exclusion from swap displacement plus a re-entry cooldown is implemented config-gated default OFF with replay evidence (the 55.3-sanctioned time-axis lever), promoted only on a verbatim TURNOVER LEVERS token; ELSE a documented no-lever-needed finding closes this criterion",
          "per-market score-to-forward-return slopes (bps of 5-10-day forward excess return per score point) are estimated from the system's own BQ history with the method documented; if any market's slope is statistically indistinguishable from zero, that finding is escalated to the operator as a signal-validity alarm rather than silently traded through",
          "no hysteresis-family code exists unless the operator replied HYSTERESIS: AUTHORIZE; all operator tokens for this step are recorded verbatim in the live_check"
        ],
        "live_check": "live_check_61.5.md containing the ON-vs-OFF fee-table replay summary, >= 5 trading days of post-flag churn BQ evidence vs baseline, the slope-estimation output per market, and every verbatim operator token for this step"
      },
      "retry_count": 0,
      "max_retries": 3
    }
  ]
}
```

## active_goal.md refresh payload

```markdown
# Active Goal -- goal-phase61-churn-integrity (installed 2026-06-11)

Objective: stop the sentinel churn engine (fees were $17.14/8d -- the real cost is -$139.83
realized on <=2-day holds vs +$1,355.22 on long holds), make decision inputs and money
displays trustworthy, adopt a per-market cost model with measured operator-gated turnover
levers. Trailing-stop + long-hold path is untouchable (it makes the money).

Steps: 61.1 flags+restart (P0) -> 61.2 decision-input integrity (P0) -> 61.3 currency +
latent money bug (P0) -> 61.4 learnings + 30D trend (P1) -> 61.5 cost model + turnover
policy (P1).

Operator tokens pending: 60.2 FLAG / 60.3 FLAG / 57.1 FLAG (at 61.1); SPRINT TILE (61.4);
FEE TABLE, TURNOVER LEVERS, optional HYSTERESIS: AUTHORIZE (61.5).

Cycle ledger:
- (append per cycle)
```

## Review tooling

- BQ evidence: bigquery MCP (`list-tables`/`describe-table`; `execute-query` operator-gated)
  or venv Python ADC client; paper tables in `financial_reports` (us-central1).
- UI evidence: Playwright MCP (verified working 2026-06-11, pinned @playwright/mcp@0.0.76).
- Replays: follow `scripts/replay/replay_60_2_swap_fix.py` as the pattern for ON-vs-OFF
  lever replays.

## Files in scope

backend/config/settings.py; backend/services/portfolio_manager.py; backend/services/
paper_trader.py; backend/services/autonomous_loop.py; backend/services/meta_scorer.py;
backend/agents/claude_code_client.py; backend/agents/orchestrator.py (guard only);
backend/db/bigquery_client.py; backend/api/paper_trading.py; backend/api/reports.py;
backend/tasks/analysis.py; backend/services/execution_router.py (fee model);
frontend/src/components/paper-trading/positions-columns.tsx; frontend/src/components/
reports-columns.tsx; frontend/src/components/cockpit-helpers.tsx; frontend/src/lib/format.ts;
frontend/src/components/CommandPalette.tsx; backend/.env (flag lines only, token-gated).

## References (read before PLAN)

- handoff/current/live_check_60.2.md + replay_60_2_results.md (flag promotion + replay)
- handoff/archive/phase-60.3/contract.md (AW-9 scope: LLM-prompt surface only)
- handoff/archive/phase-55.3/ + 53.1 rulings (hysteresis ban precedent)
- Fleet audit transcripts: session workflows dir, run wf_7a3b2a6c-0da (10 agents)
- External: nbgarleanu.github.io/DynTrad.pdf; ar5iv.labs.arxiv.org/html/1207.7330;
  arxiv.org/html/2602.00196; arxiv.org/html/2605.23905; arxiv.org/abs/2606.08285;
  gyanresearch.wdfiles.com/local--files/alpha/JPM_FA_07_Qian.pdf;
  finra.org/rules-guidance/notices/information-notice-20260317;
  taxsummaries.pwc.com/republic-of-korea/corporate/other-taxes;
  bankeronwheels.com/ibkr-fixed-vs-tiered/; kitces.com opportunistic-rebalancing study
