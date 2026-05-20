# Sprint Contract — phase-31.0 Profit-Protection + Risk-Agent Hardening Audit

**Step ID:** `phase-31.0`
**Date:** 2026-05-20
**Cycle type:** Diagnostic audit (NO CODE EDITS). One full harness pass producing a gap report + ranked remediation proposals. Findings will define the phase-31.x remediation sub-steps (each its own future cycle); this cycle does not implement them.

---

## Research-Gate Summary (per `.claude/rules/research-gate.md`)

- **Tier:** deep.
- **Brief:** `handoff/current/research_brief.md` (378 lines).
- **external_sources_read_in_full:** 22 (3 adversarial, 22-source floor exceeded).
- **snippet_only_sources:** 11.
- **urls_collected:** 33.
- **recency_scan_performed:** true — 8 entries from 2024-2026 window documented (AdaptiveTrend arXiv 2602.11708, QuantAgents arXiv 2510.04643, AQR Q1 2025 paradigm paper, MSCI 2025 quant-wobble post, etc.).
- **internal_files_inspected:** 9 (portfolio_manager.py, paper_trader.py, autonomous_loop.py, risk_judge.md, risk_stance.md, synthesis_agent.md, quant_strategy.md, agent_definitions.py, signals_server.py:1052-1243).
- **Adversarial sources (3):** Kaminski & Lo 2014 (pdfplumber-extracted 42-page paper, Random-Walk Hypothesis Proposition 1 + mean-reversion Proposition 2); Carver qoppac.blogspot.com (anti-breakeven, anti-profit-target); arXiv 1507.01610 (OU mean-reversion + trailing-stop counter).
- **gate_passed:** true.

---

## Hypothesis

The pyfinAgent Layer-2 autonomous loop has **no profit-protection layer**. When a position runs up and then declines, the system rides it back to its **entry-anchored static stop** (or to the next LLM SELL signal, whichever comes first). MFE is tracked monotonically in `paper_trader.mark_to_market` but is never consulted as an exit input. A trailing-stop + tiered-drawdown reference implementation exists at `signals_server.py:1052-1243` but has **zero callers from the live loop** — it is dead code. The dashboard symptoms (MU +34.66%, INTC +27.76%, SNDK +32.03% with give-back patterns) are an accurate signature of this architectural gap, not a perception artefact.

---

## Success Criteria (IMMUTABLE — copied verbatim from the goal)

A passing GENERATE produces `handoff/current/experiment_results.md` that contains, all five, in this order:

1. **Per-practice audit table** with one row per research-justified practice (triple-barrier exit, trailing stops, take-profit ladders, profit-locking ratchets, volatility-adjusted exits, meta-labeling, risk-agent best practices, portfolio-manager best practices). Each row MUST include: practice | present? (Y/N/partial) | file:line | severity (BLOCK/WARN/NOTE) | proposed remediation. **At least one row per the eight research topics in §2 of the brief.**

2. **Specific-question answers** (the goal's "Must check specifically" list):
   - Any trailing-stop logic in the live loop, or only entry-relative static stops?
   - Any take-profit threshold (absolute or R-multiple)?
   - Does `risk_judge` see unrealized P&L and act on it?
   - Does `decide_trades` consider exit signals separately from entry signals?
   - Drawdown-based de-risking (per-position OR portfolio-level)?
   - Scale-out logic at all (or only all-or-nothing closes)?

3. **Give-back-ratio BQ result** quoted verbatim: closed-trade aggregate (sells_60d, avg_mfe_pct, avg_realized_pct, avg_capture_ratio, avg_giveback_ratio_pos_mfe), per-trade detail table, current-positions stop-coverage breakdown (NO_STOP count, STATIC_8PCT_ENTRY count), and sector-concentration table.

4. **Ranked P1/P2/P3 remediation proposals**, each with: (a) hypothesis, (b) ≥1 research citation from §2 or §4 of the brief, (c) code anchor (file:line where the gap lives), (d) implementation site, (e) Kaminski-Lo adversarial guard for any remediation that adds trailing-stop logic.

5. **JSON-ready phase-31 masterplan entry** following the phase-23.8 schema (`id`, `name`, `status`, `description`, `acceptance_criteria` array, `verification` object with `command` and optional `live_check`). Entry must define `phase-31` as the parent and at least three child entries (phase-31.1 through phase-31.3) mapping to the P1 remediation items. **Format:** valid JSON inside a fenced ```json block, parseable by `python -c "import json; json.loads(open('handoff/current/experiment_results.md').read().split('```json')[N].split('```')[0])"` for the N-th JSON block.

A passing EVALUATE requires Q/A's verdict to be PASS (not CONDITIONAL). Q/A must run the 5-item harness-compliance audit first; any compliance violation triggers FAIL not CONDITIONAL.

---

## Hard guardrails (immutable)

- **NO code edits.** Diagnostic only.
- **NO mutating BQ or Alpaca calls.** All reads via `mcp__claude_ai_Google_Cloud_BigQuery__execute_sql_readonly` and `mcp__alpaca__` inspection-only tools.
- **NO `AskUserQuestion`** (overnight mode).
- All findings ranked → phase-31 entries; implementation is OUT OF SCOPE for this cycle.

---

## Plan Steps

1. **RESEARCH** — spawn `researcher` (deep tier, MAX effort) covering the 8 topics. ✅ Done. Brief at `handoff/current/research_brief.md`; gate_passed=true; 22 sources read in full.

2. **PROBE** — query `financial_reports.paper_trades` (last 60d closed trades) for give-back ratio; query `financial_reports.paper_positions` for stop-coverage + sector concentration. ✅ Done. Results inline below.

3. **CODE AUDIT** — verify the brief's file:line claims against current main. Read `portfolio_manager.py` (in full), `paper_trader.py` (mark_to_market, check_stop_losses, kill switch), `autonomous_loop.py` (Step 5.5–5.6), `risk_judge.md`, `risk_stance.md`, `synthesis_agent.md`, `quant_strategy.md`, `agent_definitions.py`. ✅ Done. All audit claims confirmed.

4. **PLAN** — write this contract.md. ✅ This file.

5. **GENERATE** — write `experiment_results.md` with the five success-criteria sections.

6. **EVALUATE** — spawn `qa` ONCE. CONDITIONAL/FAIL → fix + fresh qa. CIRCUIT BREAKER: max 2 retries → mark `blocked` + STOP.

7. **LOG** — append cycle block to `handoff/harness_log.md` BEFORE flipping masterplan.

8. **MASTERPLAN UPDATE** — insert phase-31 (parent) + phase-31.1, phase-31.2, phase-31.3 (children) into `.claude/masterplan.json` with `status: pending`. The audit cycle itself (phase-31.0) flips to `status: done` after the log append. Auto-commit + push.

---

## Live BQ Probe Result Summary (verbatim values, to be re-quoted in experiment_results.md)

**Aggregate (3 closed sells in last 60d):**
- avg_realized_pct = **-0.42%**
- avg_mfe_pct = **+7.35%**
- avg_capture_ratio = **0.408** (keeping only ~41% of best mark)
- avg_giveback_ratio on positive-MFE trades = **0.387** (38.7% of MFE given back)
- 2 of 3 winners had MFE > 5%; 0 of them gave back ≥50% (the third had MFE=0)
- Exit reasons: 3× `sell_signal`, 0× `stop_loss`, 0× `stop_loss_trigger`, 0× kill/flatten

**Per-trade detail:**

| Ticker | Reason | Held d | MFE % | Realized % | Capture | Give-back % |
|---|---|---|---|---|---|---|
| CIEN | sell_signal | 20 | +12.56 | +6.46 | 0.515 | **48.5** |
| FIX  | sell_signal | 15 |  +9.49 | +6.75 | 0.711 | **28.9** |
| TER  | sell_signal | 17 |   0.00 | -14.46 | 0.000 | n/a (MFE=0; MAE was -26.51%) |

**Current positions (n=11):**
- **7 of 11 positions have NO stop_loss_price** at all: SNDK, INTC, WDC, LITE, ON, DELL, GLW (audit claim said 6; we discovered WDC also has none).
- 3 positions have `STATIC_8PCT_ENTRY` stops: MU, KEYS, COHR/GEV (8% below entry, never moved).
- **No position has a trailing stop or take-profit anchored to anything except entry price.**
- High-MFE positions currently giving back:
  - SNDK: MFE +57.64%, now +40.22%, NO_STOP (gave back ~17.4 pp from peak; stop coverage = none)
  - MU: MFE +57.62%, now +42.46%, stop $466.12 = -35.4% below current price (would need to drop 35% to trigger)
  - INTC: MFE +53.85%, now +35.74%, NO_STOP (gave back ~18.1 pp)
  - COHR: MFE +28.36%, now +10.69%, stop = -16.9% below current
  - WDC: MFE +27.75%, now +13.19%, NO_STOP (gave back ~14.6 pp)

**Sector concentration:** 10/11 positions Technology (89.3% of positions' market value). 1 position Industrials (GEV, 10.7%). Mirrors the AQR Q1 2025 concentration-risk paradigm and the QuantAgents `max(SE_j)` saturation scenario.

---

## References (from research_brief.md §7)

- **Foundational:** López de Prado, *Advances in Financial Machine Learning* ch.3 (triple-barrier method, meta-labeling); Van Tharp, *Trade Your Way to Financial Freedom* (R-multiples, expectancy); Carver, qoppac.blogspot.com 2020 (HWM-trailing); Wilder 1978 (ATR origin).
- **Empirical (2024-26):** arXiv 2602.11708 (AdaptiveTrend ablation +0.73 Sharpe, 9.7 pp MaxDD reduction from trailing); arXiv 2510.04643 (QuantAgents `R_score` formula with sector exposure); arXiv 2412.20138 (TradingAgents — pyfinagent's risk-debate parent); arXiv 2504.02249v2 (Triple-Barrier static-threshold optimization).
- **Adversarial:** Kaminski & Lo 2014 J. Financial Markets (Proposition 1: stop-loss negative under RWH; Proposition 2: trailing stops degrade mean-reversion EV); Han, Zhou, Zhu 2014 SSRN 2407199 (counter-evidence on momentum: -49.8% → -11.4% MaxDD, Sharpe doubled); arXiv 1507.01610 (OU mean-reversion analytical limit).
- **Industry:** AQR Q1 2025 (concentration paradigm); MSCI 2025 (summer 2025 quant-wobble crowded-trade unwind); QuantConnect MaximumDrawdownPercentPortfolio model.
- **Internal:** `signals_server.py:1052-1243` (dead-code Chandelier-lite + 5/10/15 tiered DD ladder); `quant_strategy.md:33-36` (gap already acknowledged in optimizer skill).
