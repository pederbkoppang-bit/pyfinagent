# Frontier Baseline — 2026-07-18 (feeds GOAL PHASE-73 "best-in-class LLM quant")

Provenance: ultracode recon `wf_51336138-19c` — external frontier scan (10 dimensions, syllabus-level) + capability inventory + adversarial critic at effort max. Disclosure: the dedicated inventory agent failed (StructuredOutput retry cap); the critic **reconstructed the inventory first-hand and spot-verified every load-bearing grade with file:line evidence** — grades below are verified, not inherited. Companion verified state: `money_diagnosis_72.md` (P0-P4), `operator_decision_sheet_72.md`.

## Graded dimension map (ranked by expected net-P&L headroom AT OUR SCALE — 2-person local paper fund)

| # | Dimension | Our grade (verified) | Frontier gap | Phase-72 overlap |
|---|---|---|---|---|
| 1 | **Confidence calibration of conviction → position sizing** | **F** — zero calibration code anywhere (grep brier/isotonic/platt/hit-rate empty); raw meta-scorer 1-10 + advisory RJ pct feed size directly | LLM/ensemble confidence is systematically overconfident; calibrating conviction to realized hit-rate buckets BEFORE sizing is the direct P&L multiplier | ADJACENT: 72.0.1 restores the raw signal; calibration itself untouched |
| 2 | **Outcome-tracking + reflection outer loop (retrieve-reflect-store)** | **D** — fully coded (memory.py FinancialSituationMemory BM25; outcome_tracker reflections) but dark via multiple independent bugs; flat memory, no decay tiers | FinMem/FinAgent layered-decay memory + written reflections per closed trade = our cheapest compounding edge; prerequisite for #1 | P3 sheet flags learn-loop crash (outcome_tracker.py:50); repair queued nowhere as a BUILD |
| 3 | **Backtest leakage integrity (purge/embargo + LLM lookahead)** | **F** — CONFIRMED: walk-forward has NO purging; triple-barrier labels (90-135d horizons) leak test-window prices into train (`backtest_engine.py:587`); zero LLM-pretraining-cutoff guards | Our DSR/PBO verdicts are only as good as their inputs — the purge leak inflates both. "Profit Mirage": LLM-memorized history makes pre-cutoff backtests evaporate; needs post-cutoff + counterfactual eval | NONE (was routed to phase-68/61.4 owners, never built) |
| 4 | **Net-of-cost / token-cost promotion objective** | **C** — costs tracked (~$51/window) + turnover penalty exists, but token+slippage+fees NOT folded into the promotion objective | Only QuantAgent budgets token cost in the field; it is literally our north star ("NET, cost-adjusted") — cheap to build | NONE (72.2.x fix measurement, not the objective) |
| 5 | Autoresearch tournament → live champion bridge | C — optimizers + DSR/PBO bakeoff built, but best_params never thread into live decide_trades; validation blocked by frozen historical_macro | Self-evolving propose→backtest→promote→deploy loops exist here in pieces; the lever is the WIRING + the un-freeze validation plan | Frozen-macro doctrine reaffirmed; un-freeze token owed since 69.2/3 |
| 6 | News/event RAG evidence weighting | B — rich enrichment live in debate (earnings_tone, sec_insider, sentiment, FRED, RAG) but static relevance, no residual-return-feedback weighting | MarketSenseAI-2.0-style matured-residual-return evidence weighting is the marginal upgrade; reported returns leakage-suspect | NONE; mechanism overlaps #2's feedback plumbing |
| 7 | LLM alpha/factor mining (idea→factor→eval) | C- — LLM tunes PARAMETERS, doesn't mine formulaic factors; no IC/IR gate | AlphaAgent/QuantaAlpha are the 2025-26 frontier BUT self-reported pre-cutoff backtests (HYPE-FLAGGED); low plausibility at our scale until #3 exists | NONE — judge honestly, likely small pilot at most |
| 8 | Multi-agent firm-sim + bull/bear debate + risk veto | **A-** — fully built end-to-end (debate.py:131-362 Bull/Bear/DA/Moderator; risk_debate.py Aggressive/Conservative/Neutral + Judge; 20+ agent orchestrator). One contested MAS retry bug (multi_agent_orchestrator.py:1238) | The field "converged" on what we already run — **NO headroom; do NOT rebuild** | Architecture untouched by 72 |
| 9 | Anti-overfitting gates DSR/PBO/CPCV | **B+** — coded + live-wired (autoresearch/gate.py PromotionGate min_dsr=0.95, max_pbo=**0.20** — stricter than the charter's 0.5; CPCV per AFML Ch.12; sustained-PSR go-live gate) | Canonical already; the only action is CLEAN INPUTS = #3 | 72.2.1/72.2.4 fix inputs |
| 10 | Human-gate production posture | **A** — recommend-only sheets + operator tokens + promotion gates ARE the AQR/Man-Group model | Confirms the moat is data+review+execution; keep | It IS the phase-72 model |

## Read-in-full syllabus for the goal session (per dimension)

- **Architecture (validate, don't rebuild)**: TradingAgents arXiv:2412.20138; FinCon arXiv:2407.06567.
- **Memory/reflection (#2)**: FinMem arXiv:2311.13743; survey arXiv:2408.06361.
- **Self-improvement (#5)**: QuantAgent arXiv:2402.03755 (the only token-cost-budgeted line); Awesome-Self-Evolving-Agents (XMUDeepLIT).
- **Factor mining (#7)**: AlphaAgent arXiv:2502.16789 (KDD'25); QuantaAlpha arXiv:2602.07085; Alpha-GPT (EMNLP 2025).
- **News/RAG (#6)**: MarketSenseAI 2.0 arXiv:2502.00415; ECC Analyzer (ICAIF'24, dl.acm.org/doi/fullHtml/10.1145/3677052.3698689); market-feedback adaptive retrieval arXiv:2605.31201.
- **Calibration→sizing (#1)**: arXiv:2508.06225 (LLM-judge overconfidence); arXiv:2404.09127 (multi-agent deliberation calibration); Amazon Science "Label with Confidence".
- **Gates (canonical, ours)**: Bailey/LdP DSR (SSRN 2460551); PBO (davidhbailey.com backtest-prob); CPCV comparison (S0950705124011110).
- **Leakage (#3, newest + most consequential)**: Profit Mirage arXiv:2510.07920 (FinLake-Bench/FactFin); Look-Ahead-Bench (ResearchGate 399953316).
- **Industry reality**: Kelly (AQR) "Limits to (Machine) Learning" arXiv:2512.12735; Man Group AlphaGPT (ai-street.co + Bloomberg 2025-07-10).
- **Meta-surveys**: "The New Quant" arXiv:2510.05533; arXiv:2408.06361.

## Critic verdicts the goal must respect

1. **Do NOT rebuild** debate/architecture (#8) or the statistical gates (#9) — already at/above field standard; re-scoping them is negative-value churn.
2. **Leakage-skepticism discipline**: most frontier return claims (MarketSenseAI 125.9%, FinAgent ~90%, AlphaAgent IR 1.5) are self-reported, pre-cutoff, leakage-suspect — treat as mechanism evidence only, never as expected-return evidence. Any adopted mechanism must pass OUR gates on post-cutoff data.
3. The compounding chain is **#3 → #2 → #1 → #4** (clean backtests → working learn-loop → calibrated sizing → net-of-cost promotion): each is a prerequisite of trusting the next; #5/#6/#7 are judged pilots behind them.
4. 2026-dated arXiv IDs (2602.*-2607.*) are fresh preprints — verify on full read.
5. Newly surfaced defects for the goal to queue: purge/embargo leak (`backtest_engine.py:587`), MAS retry bug (`multi_agent_orchestrator.py:1238`), PromotionGate PBO 0.20-vs-charter-0.5 discrepancy (document which is intended).
