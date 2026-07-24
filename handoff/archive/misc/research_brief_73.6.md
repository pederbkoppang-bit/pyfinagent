# Research Brief — phase-73.6 "D3 MONEY RUNWAY (recommend-only)"

Tier: **simple** (caller-set). NOT audit-class. Deliverable feeding GENERATE:
an ORDERED runway (paper-restoration -> real-fill -> go-live) with prerequisites,
evidence anchors, and OPERATOR DECISIONS as verbatim-actionable lines.
Consistent-with (never duplicating) existing masterplan entries (phase-68 / 58.1 /
phase-72 ACT-NOW). Recommend-only: no spend, no flags, no code.

Write-first: skeleton created at session start; grown incrementally as each source
was read. Read-only except this brief.

---

## Research: paper-trading -> real-fill -> go-live runway

### Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
|---|---|---|---|---|
| https://docs.alpaca.markets/us/docs/paper-trading | 2026-07-18 | official broker doc | WebFetch | "the system simulates the order filling based on the real-time quotes"; partial fills "for a random size 10% of the time"; **"Your order quantity is not checked against the NBBO quantities"** (fills exceeding real liquidity); "paper trading is only a simulation ... performance may differ" — unaccounted: market impact, slippage-from-latency, queue position, fees, dividends. |
| https://alpaca.markets/learn/paper-trading-vs-live-trading-a-data-backed-guide-on-when-to-start-trading-real-money | 2026-07-18 | broker practitioner (data-backed) | WebFetch | Live adds "partial or delayed fills", "real-world costs incl. spreads and fees", "fear, greed, and stress", "true latency, order book visibility, potential market impact". API-user data (Jun2024-May2025): 34.3% went live <10d, 57.1% <30d, 75.2% <60d. Sizing: risk 1-2% per trade, start with "minimal capital", "limited position sizes to avoid large drawdowns early". |
| https://www.modelop.com/ai-governance/ai-regulations-standards/sr-11-7 | 2026-07-18 | industry governance (SR 11-7) | WebFetch | Two sources of model risk: fundamental errors + misuse ("using models in ways not consistent with their original intent"). **"Effective challenge"** = critical analysis by informed competent parties. Validation = conceptual soundness + **ongoing monitoring** + **outcomes analysis incl. back-testing**. "Strong governance, policies, and controls are essential." |
| https://www.sia-partners.com/en/insights/publications/sr-11-7-vs-sr-26-2-model-risk-management-modernization | 2026-07-18 | industry governance (recency) | WebFetch | **SR 26-2 (issued 2026-04-17) SUPERSEDES SR 11-7.** Retains three-pillar + effective-challenge; adds tailoring by size/materiality, narrower model definition, risk-based (not default-annual) validation. "Generative and agentic AI are explicitly outside the guidance's scope." |
| https://www.blueguardian.com/blogs/paper-trading-vs-live-trading-the-real-differences-traders-must-understand | 2026-07-18 | practitioner | WebFetch | Sim assumes "instant fills at your target price"; live adds "execution delays, partial fills and rejections"; slippage 5-10% (options up to 20%); "your simulated trades don't affect the market". Transition: review 3-6 months / "complete 50-100 trades", risk 1-2% per trade. |
| https://www.tradingsim.com/blog/position-sizing-guide | 2026-07-18 | practitioner | WebFetch | Fixed-fractional 1% "scales automatically with your account"; Kelly example 55%/2:1 -> 32.5% (brutal DD); **half-Kelly** ~75% of growth, survivable; "graduate to half-Kelly after 200+ verified trades"; 10-trade losing streak: 1%->~10% DD, 2%->~18%, 5%->~41%. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|---|---|---|
| https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm | official (Fed SR 11-7) | HTTP 404 on the canonical letter URL; substituted ModelOp + sia-partners full reads for the same principles |
| https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html | official (OCC 2011-12) | HTTP 404; same principles captured via ModelOp full read |
| https://www.quantifiedstrategies.com/paper-trading-vs-live-trading/ | practitioner | Bot-verification wall returned by WebFetch (no article body) |
| https://www.quantifiedstrategies.com/position-sizing-strategies/ | practitioner | Bot-verification wall (substituted tradingsim full read) |
| https://blog.traderspost.io/article/the-reliability-of-paper-trading-insights-and-best-practices | practitioner | Snippet corroborated fill/slippage points already read in full |
| https://funderpro.com/blog/simulated-vs-real-execution-why-fills-change-in-volatility-and-how-slippage-works/ | practitioner | Corroborating (fills change in volatility); not needed for floor |
| https://takeprofittrader.com/blog/trading-in-the-simulated-vs-live-market | practitioner | Corroborating (sim->live shift) |
| https://www.oanda.com/us-en/trade-tap-blog/trading-knowledge/slippage-execution-risk-in-trading/ | broker | Slippage/execution-risk snippet corroboration |
| https://medium.com/@ildiveliu/... Kelly-Lite position sizing | practitioner | Medium; tradingsim covered the frameworks in full |
| https://forum.alpaca.markets/t/paper-trading-market-orders-fill-delay/18681 | community | Fill-delay anecdote; below quality floor |

### Recency scan (2024-2026) — MANDATORY, present
Searched 2026 + 2024-2026 windows on both topics. Findings (do NOT merely complement — one SUPERSEDES a frame the code cites):
1. **SR 26-2 (2026-04-17) supersedes SR 11-7** (sia-partners, read in full). `settings.py` `real_capital_enabled` cites "SR 11-7 paper-only gate" — the cited frame is now retired. Core principles carry forward (effective challenge, validation-before-use, ongoing monitoring/back-testing), so the gate's *intent* is intact; the *citation* is stale. Recommend-only doc-drift note (no charter/code edit in this step).
2. **Alpaca live-transition dataset (Jun 2024-May 2025)**: 34.3%/57.1%/75.2% of API users went live within 10/30/60 days — recent empirical base rate that undercuts the 3-6-month practitioner rule; both are far shorter than our gate's 100-round-trip bar (appropriate: our discipline is stricter than retail norms).
3. No 2024-2026 source overturns the canonical fill/slippage/psychology consensus; newer pieces (Blue Guardian, funderpro) restate it with sharper numbers (5-10% slippage; options to 20%).

### Search queries run (3-variant discipline)
- **Current-year frontier (2026):** "difference between paper trading and live trading psychology 2026"
- **Last-2-year window:** Alpaca data-backed guide (Jun2024-May2025 dataset); "SR 11-7 vs SR 26-2 ... modernization" (2026)
- **Year-less canonical:** "paper trading to live trading transition slippage fills latency"; "SR 11-7 supervisory guidance model risk management summary"; "position sizing small trading account fixed fractional Kelly capital scaling"; "Alpaca paper trading account fill simulation how orders fill"

---

## Key findings (external, per-claim cited)
1. **Broker paper fidelity is imperfect even at its best.** Alpaca paper fills off real-time quotes with random 10% partial fills, but "order quantity is not checked against the NBBO quantities" and it ignores market impact/queue/fees — "not a substitute for real trading" (Alpaca docs, 2026-07-18). So Alpaca-paper numbers carry a *known optimistic bias* the divergence tracker must expect.
2. **Live degrades on four axes vs sim:** worse slippage (5-10%; options to 20%), partial/delayed/rejected fills, real spreads+fees, and psychology (fear/greed) (Blue Guardian; Alpaca learn, 2026-07-18).
3. **Track-record-before-live is the practitioner consensus:** "review 3-6 months / complete 50-100 trades" (Blue Guardian); half-Kelly only "after 200+ verified trades" (tradingsim). Our 100-*round-trip* gate sits inside this band and is stricter than the 75.2%-in-60-days retail base rate.
4. **Transition sizing = start small, scale on evidence:** risk 1-2% per trade, "minimal capital", "limited position sizes to avoid large drawdowns early" (Alpaca learn); fixed-fractional -> half-Kelly graduation (tradingsim).
5. **SR 11-7/26-2 is the governance spine for `real_capital_enabled`:** models are validated BEFORE use, monitored continuously, and subjected to outcomes-analysis/back-testing under "effective challenge" (ModelOp). The go-live gate + 58.1 human token + recommend-only posture ARE this frame in code; the bq_sim-vs-real divergence tracker (68.6) IS the "outcomes analysis / back-testing" leg.

## Internal code inventory
| File | Lines | Role | Status |
|---|---|---|---|
| backend/services/paper_go_live_gate.py | 1-213 (full) | 5-boolean live-capital promotion gate (trades_ge_100=100, psr_ge_95_sustained_30d, dsr_ge_95, sr_gap_le_30pct, max_dd_within_tolerance); AND of all -> promote_eligible | LIVE; two booleans **tightened in 69.2** (see below) |
| backend/services/paper_go_live_gate.py | 71-92 (`_sustained_psr_ge`) | true 30-day min-PSR sustainment (not point-in-time) | 69.2 FIX confirmed |
| backend/services/paper_go_live_gate.py | 95-114,164 (`_load_backtest_max_dd`, dd_tolerance) | realized-DD vs backtest-DD+5pp (falls back to 20% abs cap) | 69.2 FIX confirmed |
| backend/config/settings.py | ~264 (`real_capital_enabled`) | SR 11-7 paper-only hard gate; MUST stay False until compliance review wires real-capital path; consumed by monthly_champion_challenger.run_monthly_sortino_gate | Default False (correct) |
| backend/services/paper_trader.py | 268-277 | bq_sim fills **at close, slippage=0**; alpaca_paper slippage = fill_price - close (:277) | bq_sim is the only path today |
| handoff/current/audit_phase69/register.md | :26, :189; handoffs_69.4.md:21 | phase-69 register flagged the two weak booleans; "FIXED-69.2 (crit5 go-live booleans)" | Fix confirmed in code + register |
| .claude/masterplan.json | phase-68 (8 steps), 58.1, phase-73 (73.6 + siblings) | Real-Fill Runway; go-live runway; this step | 68.0 done, 68.1-68.7 pending; 58.1 pending |
| handoff/current/design_pack_73/d_cost_promotion.md | §1-2 | 73.4 net-of-cost DSR + cost-per-bp: slippage haircut is an ESTIMATE because bq_sim fills at close; real fills make slippage_usd MEASURED | strengthened by Stage 2 |

---

## STAGE 1 — PAPER RESTORATION (unblock the scoring rail; earn on synthetic fills)

**Current state (evidence: money_diagnosis_72.md P0; operator_decision_sheet_72.md ACT-NOW).**
~97-100% cash, scoring rail credit-dead since 2026-05-17 03:55:44 CEST (HTTP-400
"credit balance too low"); meta-scorer LLM leg failed *every trading day since
05-22*; 07-10..07-17 all 0-buy / 100% HOLD / ~100% final_score=0.0. This is the
P0 DEFECT, not the P4 regime policy. **The phase-72 ACT-NOW block is the unlock —
do not re-derive it here; execute it verbatim.**

**"Restored" = the immutable live_check from 72.0:** a post-restoration cycle log
line showing **non-degraded scoring** (meta-scorer LLM leg active, no degraded-guard
fire, >=1 non-HOLD recommendation OR an honest market-driven HOLD). Then the P3
lever sequence STARTS (one flip at a time, 3-5 cycles between — `operator_decision_sheet_72.md` §P3) and the P4 regime policy applies (DEPLOY by default; `MACRO_REGIME_FILTER_ENABLED=true` scales by regime — §P4).

**Prerequisites:** none beyond operator keystrokes (backend/.env is agent-locked).
**Note:** every fill produced in this stage is still SYNTHETIC bq_sim (close-fill,
slippage=0, unlimited liquidity) — LESS realistic than even a broker paper account
(Alpaca paper at least simulates fills off real-time quotes). Stage 1 restores
*earning capacity*; it does NOT advance the go-live clock (Stage 3 counts real fills only).

**Operator decisions (verbatim-actionable — from the 72 ACT-NOW block, referenced not duplicated):**
- Decide Anthropic direct-API credits: top up the metered key at console.anthropic.com, OR reply `ANTHROPIC DIRECT: ABANDON` (reroutes via 72.0.1/72.0.2).
- Append to `backend/.env`: `PAPER_SYNTHESIS_INTEGRITY_ENABLED=true`
- Append to `backend/.env`: `PAPER_RISK_JUDGE_SHAPE_FIX_ENABLED=true`
- Restart backend: `launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`
- Run the reconciliation grep: `! grep -nE 'SYNTHESIS_INTEGRITY|RISK_JUDGE|SWAP|SCALE_OUT|SESSION_BUDGET|PAPER_MARKETS|PAPER_TRADING|MODEL|ANTHROPIC|GEMINI' backend/.env`
- (post-restoration, one at a time) begin the P3 lever queue: record `KS-PEAK-RESET:APPROVED`, then Seq-2 soft-sector-diversity, per `operator_decision_sheet_72.md` §P3.

## STAGE 2 — REAL-FILL RUNWAY (phase-68: swap synthetic bq_sim for Alpaca-paper fills)

**Why (evidence: masterplan phase-68 header; paper_trader.py:268-277).** Every fill
ever recorded is synthetic: `EXECUTION_BACKEND` never reaches `execution_router` in
the launchd process — stuck on `bq_sim`. Until real fills exist, the go-live gate's
trade counter is meaningless and 73.4's slippage term is an estimate.

**Ordered steps + prerequisites (masterplan phase-68 — reference, do not restate criteria):**
- **68.1** — `EXECUTION_BACKEND` demonstrably reaches `execution_router` in the launchd process; byte-identical bq_sim default; LOUD missing-creds log; paper-only triple-enforcement. **DARK.** (prereq: none new)
- **68.2** — flatten stray paper-account shorts; then **>=5 days shadow-mode paired fills** to BQ + drift report; shadow NEVER mutates bq_sim state. (prereq: 68.1)
- **68.3 (P0)** — Cutover: **TOKEN GATE `EXEC-BACKEND: ALPACA_PAPER`**; >=3 SCHEDULED cycles with `source='alpaca_paper'`; reconciliation <2% drift, zero orphans; one rollback drill; risk caps byte-untouched. (prereq: 68.2 + token)
- **68.4** — learn-loop activation: dark write-drill vs live schemas, **token ask with measured reflection cost**, first REAL sell-close writes `outcome_tracking` + `agent_memories`. NO manufactured SELLs. (prereq: 68.3)
- **68.5** — price-integrity pre-persist sanity gate; FX-1 handoff to parked 61.3. (prereq: 68.3)
- **68.6** — Weekly go-live tracker from REAL fills ONLY; **immutable clean-window start**; Sharpe vs 0.82; missed days; bq_sim-vs-real divergence per GO_LIVE_CHECKLIST 4.4.2.5; feeds 58.1; flips nothing. (**prereq: 68.5 done** + 68.3)

**What Alpaca-paper fills ADD (evidence: Alpaca docs; d_cost_promotion.md §1-2):**
real fill prices + real slippage (`fill_price - close`) replacing bq_sim's
slippage=0 close-fills. Caveat from the broker itself: Alpaca paper "is not checked
against NBBO quantities" and ignores market impact — so even Stage-2 real-fill
numbers are optimistically biased; the 68.6 divergence report (4.4.2.5) is exactly
the "outcomes analysis / back-testing" leg SR 11-7/26-2 requires.

**Which phase-73 designs Stage 2 STRENGTHENS:**
- **73.4 (cost-integrated promotion)** — SEAM B `slippage_usd` becomes MEASURED (fill_price-close) instead of the estimated BSIC `c1*spread+c2` haircut; cost-per-bp gains a real slippage term. (d_cost_promotion.md §1 explicitly notes bq_sim fills at close with no modeled slippage.)
- **73.2 (learn-loop v2)** — 68.4 activates the very `outcome_tracker`/`agent_memories` write path 73.2.2 repairs; realized SELL-close (real fill) is 73.2.2's PRIMARY reflection input (demoting the rolling yfinance mark).
- **73.3 (calibrated sizing)** — needs ~100-150 CLEAN closed trades before live calibrated sizing; real fills are the only clean substrate (matches the lit's "200+ verified trades before half-Kelly").

**Operator decisions (verbatim-actionable):**
- Record the cutover token (68.3): `EXEC-BACKEND: ALPACA_PAPER`
- Approve the learn-loop reflection LLM cost when 68.4 presents the measured per-close reflection cost (flat-fee Gemini rail; token-throughput, ~$0 marginal).
- Confirm Alpaca paper API creds are present in `backend/.env` before 68.3 (68.1's LOUD missing-creds log surfaces this).

## STAGE 3 — GO-LIVE (honest gate readout; the human token; the SR-frame)

**The 5 booleans (paper_go_live_gate.py, all must be green -> promote_eligible) — HONEST current answer:**
| Boolean | Threshold | Honest current state | Evidence |
|---|---|---|---|
| trades_ge_100 | >=100 round trips | **FALSE** — ~30 synthetic round trips whole-table (BQ 2026-07-18); and **0 REAL** under 68.6's alpaca_paper clean-window discipline. Authoritative live count = `compute_gate(bq)['details']['n_round_trips']` (not runnable read-only). | money_diagnosis_72.md P2; paper_go_live_gate.py:130-132 |
| psr_ge_95_sustained_30d | min-PSR>=0.95 over last 30 daily NAV pts | **FALSE/indeterminate** — NAV frozen + 5 dropped snapshot days (P2c) means the 30-day sustained series isn't cleanly computable | paper_go_live_gate.py:71-92; money_diagnosis_72.md P2c |
| dsr_ge_95 | DSR>=0.95 | indeterminate from artifacts (metrics-dependent; not asserted) | paper_go_live_gate.py:171 |
| sr_gap_le_30pct | rel gap <=0.30 | **at-risk** — register #31 external-flow bug corrupts live Sharpe on a sibling path (compute_sharpe_from_snapshots); gate uses compute_sharpe_gap 3-tier fallback | register.md:31; paper_go_live_gate.py:149-150 |
| max_dd_within_tolerance | <= backtest-DD+5pp | likely within (frozen book -> low realized DD) but LEAST informative | paper_go_live_gate.py:164,176 |

**Register-note verification (asked): WAS the "booleans weaker than documented" defect fixed?** YES — phase-69.2. register.md:26 flagged it; handoffs_69.4.md:21 records "FIXED-69.2 (crit5 go-live booleans)"; and the live code implements both documented definitions (`_sustained_psr_ge` true 30-day sustainment; dd_tolerance = backtest_max_dd + 5.0). The gate now MEASURES what it claims; it was NOT relaxed.

**The hard gate BEYOND the 5 booleans:** `real_capital_enabled=False` (settings.py). Even 5/5 green + 58.1 APPROVED does NOT deploy real capital — a compliance review must wire the real-capital path AND flip this flag. Governance frame: SR 11-7 (now **SR 26-2**, 2026-04-17) — validation-before-use + effective challenge + ongoing monitoring/back-testing. The code comment cites the retired SR 11-7 letter; principles survive (recommend-only doc-drift note, no edit this step).

**58.1 runway (masterplan 58.1 — reference, do not restate criteria):** operator's
verbatim spend decision recorded in `live_check_58.1.md` BEFORE any live LLM cycle
(running one without it = automatic FAIL). APPROVED -> deploy fixes, verify kill
switch ACTIVE, operator re-enables mas-harness cron, run budgeted window, re-score
DoD-2/5/6/7/9. Go-live gate baseline was 1/5 (2026-06-01). 58.1 is fed by 68.6.

**Honest posture (ratifies frontier_map_73.md #10):** we are NOT go-live-eligible and
the clock has not started — 0 REAL round trips vs a 100 bar, so the gate cannot be
meaningfully evaluated until months of real fills accrue. Recommend-only,
gate-everything, human token as promotion authority — exactly the SR-frame and the
Man-Group identical-gates posture.

**Operator decisions (verbatim-actionable):**
- (only after 68.6 has real-fill evidence) record the spend decision in `live_check_58.1.md`: `LLM SPEND: APPROVED <budget>` OR `LLM SPEND: DECLINED` (with date).
- If APPROVED: re-enable the cron: `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pyfinagent.mas-harness.plist`; verify kill switch ACTIVE before the window.
- **DEFER (not now):** keep `real_capital_enabled=False` until a compliance review wires the real-capital deployment path — no keystroke flips real money in this runway.
- Sizing on any eventual go-live: start at reduced size (lit: 25-50% intended / risk 1-2% per trade), graduate toward half-Kelly only after ~100-200 verified REAL trades — aligns with 73.3's ~100-150-clean-trade calibrated-sizing defer.

---

## Consensus vs debate (external)
Consensus (unanimous across all 6 sources): live diverges from sim on slippage,
fills, fees, latency, psychology; paper fidelity is a floor not a substitute; track
record + small starting size gate the transition. Debate: *how long* to paper trade
— Alpaca's 2024-25 data shows 75% go live <60 days, while practitioners urge 3-6
months / 50-100 trades. Our 100-round-trip + human-token bar is deliberately
stricter than both (correct for an autonomous system whose "paper" is synthetic).

## Pitfalls (from literature)
- Treating bq_sim/paper numbers as live-representative — Alpaca warns "not a
  substitute"; our bq_sim is weaker still (no fill simulation at all).
- Over-sizing early — full Kelly gives "brutal" drawdowns (10-trade streak: 32.5%
  Kelly -> deep DD); start fixed-fractional 1-2%.
- Governance drift — citing a superseded frame (SR 11-7) without re-anchoring to
  SR 26-2; principles hold but the citation is stale.

## Application to pyfinagent (mapping)
The three stages are already encoded: Stage 1 = phase-72 ACT-NOW; Stage 2 = phase-68
(bq_sim -> alpaca_paper); Stage 3 = paper_go_live_gate.py + 58.1 + real_capital_enabled.
The runway's value-add is SEQUENCING them into one money-path and stating the honest
go-live answer (0 real round trips) — not re-deriving any criterion. External lit
supplies the "start small / gate on track record / validate-before-use" discipline
that makes the sequence defensible.

---

## Research Gate Checklist
Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: 2 broker, 2 governance, 2 practitioner)
- [x] 10+ unique URLs total (>30 collected across 8 searches)
- [x] Recency scan (last 2 years) performed + reported (SR 26-2 supersession; Alpaca 2024-25 dataset)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered go-live gate, settings, paper_trader, masterplan phases 68/58, design pack, phase-69 register
- [x] Contradictions/consensus noted (transition-timing debate)
- [x] Claims cited per-claim with URL + access date

## JSON envelope
```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 32,
  "recency_scan_performed": true,
  "internal_files_inspected": 8,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Ordered money runway for phase-73.6, recommend-only. STAGE 1 paper-restoration: scoring rail credit-dead since 2026-05-17; the phase-72 ACT-NOW block (Anthropic credit decision + 2 .env flags + restart + grep) is the unlock; restored = a non-degraded scoring cycle; all fills remain synthetic bq_sim so the go-live clock does not advance. STAGE 2 real-fill (phase-68): EXECUTION_BACKEND never reaches execution_router; 68.1->68.6 swaps bq_sim close-fills (slippage=0) for Alpaca-paper real fills, gated by the EXEC-BACKEND: ALPACA_PAPER token; this makes 73.4 slippage MEASURED and feeds 73.2/73.3. STAGE 3 go-live: 5 booleans (trades>=100 etc.) with the two weak ones already FIXED in 69.2 (verified in code+register); honest state = ~30 synthetic round trips and 0 REAL, so not eligible and the clock hasn't started; real_capital_enabled=False is the hard SR-11-7 gate (now superseded by SR 26-2 2026-04-17 -- stale citation, principles hold); 58.1 records the operator spend token. Lit consensus (6 full reads): live degrades slippage/fills/latency/psychology, gate on track record, start small (1-2%/trade, half-Kelly after 200+ trades). Every operator decision enumerated verbatim per stage.",
  "brief_path": "handoff/current/research_brief_73.6.md",
  "gate_passed": true
}
```
