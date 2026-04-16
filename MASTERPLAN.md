# MASTERPLAN.md - pyfinAgent Intelligence Engine

Top-level narrative synthesis of the pyfinAgent masterplan. Machine-readable
companion lives at `.claude/masterplan.json`; per-phase proposals live under
`handoff/phase-proposals/`.

## Vision

pyfinAgent is not a phased build toward a frozen product. It is an
Intelligence Engine: a self-directed system that ingests global intelligence
(institutional smart money, academic frontier research, AI-frontier lab
releases, and player-driven trader forums), evolves its own prompts, agents,
and algorithms through a recursive Meta-Evolution loop, and continuously
searches for profitable alpha under hard risk and compute caps. Every
subsystem exists to serve one objective: maximize `Net System Alpha` at the
lowest tolerable risk and compute burn, while dynamically reallocating capital
and cognition toward whichever strategy is earning the most today.

## The Red Line

The governance invariant that every phase must defend:

```
Net System Alpha = Profit - (Risk Exposure + Compute Burn)
```

Definitions:

- **Profit (USD)**: Net realized P&L over the evaluation window (weekly,
  rolling), after transaction costs and slippage. Source: paper trading PMS
  and live broker reconciliation in `pyfinagent_pms`.
- **Risk Exposure (USD)**: VaR-style risk charge combining gross position
  exposure, tail-CVaR at 99%, and a drawdown-breach penalty. Governed by
  the Immutable Core risk guard (phase-A).
- **Compute Burn (USD)**: Sum of LLM API cost, BigQuery slot cost, and
  scheduled-run compute cost over the same window. Tracked by the Cron Budget
  Allocator (phase-E).

Governance rule: if `Net System Alpha` falls below zero for two consecutive
weekly windows, the system auto-pauses all discretionary spend (new LLM
experiments, non-essential ingestion, challenger promotions) and surfaces a
Peder-signoff ticket. Only the 5 trading-ops cron slots continue to run.
No phase, no challenger, no meta-evolution cycle may override this rule. The
Red Line is enforced by phase-A's Immutable Core and audited via the
Sovereign Dashboard (phase-C).

## Phases

| Phase | Status | Purpose |
|---|---|---|
| phase-0 | done | Audit and validate baseline quant engine and data sources. |
| phase-1 | done | Quant Engine Optimization - base factor model and backtest scaffolding. |
| phase-2 | in-progress | Three-Agent Harness (Planner, Coder, Evaluator) closing the RESEARCH->LOG loop. |
| phase-3 | pending | LLM-Guided Research and MCP integration for ad-hoc analysis. |
| phase-3.5 | pending | MCP tool audit and adoption across harness agents. |
| phase-3.7 | pending | MAS paper trading and MCP infrastructure hardening. |
| phase-4 | in-progress | Production Readiness - deploy, monitor, and operate the core stack. |
| phase-4.5 | done | Paper Trading Dashboard v2 - evaluation-grade UI. |
| phase-4.6 | pending | Full-stack E2E synthetic probe smoketest. |
| phase-4.7 | pending | UI/UX optimization and frontend audit. |
| phase-4.8 | pending | Pre-go-live risk and compliance hardening (FINRA GenAI, SR 11-7). |
| phase-4.9 (A) | proposed | Immutable Core Risk Guard - defends the Red Line; see `handoff/phase-proposals/phase-A-immutable-core-risk-guard.md`. |
| phase-5 | pending | Multi-market expansion (equities beyond US, FX, futures). |
| phase-5.5 | pending | External data-source audit. |
| phase-6 | pending | News and sentiment cron ingestion. |
| phase-6.5 (D) | proposed | Global Intelligence Directive - four-family ingestion pipeline; see `handoff/phase-proposals/phase-D-global-intelligence.md`. |
| phase-7 | pending | Alt-data and scraping expansion. |
| phase-8 | pending | Transformer / modern LLM signals. |
| phase-8.5 | pending | Autonomous Strategy Research (Karpathy loop). |
| phase-9 | pending | Data refresh and retraining cron. |
| phase-10 (B) | proposed | Recursive Evolution Loop - continuous champion/challenger tournament; see `handoff/phase-proposals/phase-B-recursive-evolution-loop.md`. |
| phase-10.5 (C) | proposed | Sovereign Dashboard - single pane for Red Line, Alpha Velocity, cron budget; see `handoff/phase-proposals/phase-C-sovereign-dashboard.md`. |
| phase-10.7 (E) | proposed | Meta-Evolution Engine - recursive prompt optimization, algorithm discovery, Alpha Velocity allocator; see `handoff/phase-proposals/phase-E-meta-evolution-engine.md`. |

Phase IDs A through E use the human-readable letter in the table header for
narrative clarity; the canonical numeric IDs (phase-4.9, phase-10, phase-10.5,
phase-6.5, phase-10.7) are what `.claude/masterplan.json` tracks.

## Global Intelligence Directive

The Global Intelligence Directive, implemented in phase-6.5, replaces
ad-hoc news ingestion with a disciplined four-family pipeline feeding the
Researcher agent:

1. **Institutional Intelligence** - 13F filings, CFTC COT reports, prime-
  broker flow summaries, sell-side research notes, central-bank speeches.
  Source of smart-money positioning and consensus-breaking outlooks.
2. **Academic Frontier** - arXiv q-fin, SSRN finance working papers, NBER,
  QuantConnect research, Journal of Finance recent issues. Source of new
  factors, risk models, and method critiques.
3. **AI Frontier Labs** - Anthropic, OpenAI, DeepMind, Google Research,
  Meta AI, recent NeurIPS / ICML / ICLR papers, model cards. Source of
  new architectures, evaluation methods, and prompt-engineering primitives
  that feed the Meta-Evolution Engine.
4. **Player-Driven Forums** - r/quant, Wall Street Oasis, Elite Trader,
  QuantStart, Twitter/X quant-finance list, Chinese quant WeChat mirrors
  (via public aggregators). Source of execution craft and emergent
  narratives before they hit mainstream.

All four families run under a strict `cron_slots: 1` discipline: each
family consumes at most one scheduled run per day, and batches all sources
into a single Researcher invocation. This keeps ingestion inside the
15-slot cap (see `## Cron Budget`). The Researcher agent deduplicates,
clusters, and emits a single daily "Intelligence Brief" artifact to
`pyfinagent_data.intelligence_brief` which the Meta-Evolution Engine and
the harness both consume.

See `handoff/phase-proposals/phase-D-global-intelligence.md` for schemas,
source list, dedup strategy, and brief format.

## Meta-Evolution Engine

The Meta-Evolution Engine, implemented in phase-10.7, closes the loop
between intelligence and evolution. Three sub-mechanisms operate as a
single closed loop with the Harness (Layer 3) and the Recursive Evolution
Loop (phase-10):

1. **Recursive Prompt Optimization (RPO)** - the engine reads its own
  Planner/Coder/Evaluator prompts, generates candidate mutations informed
  by AI-frontier-lab ingestion (phase-D), and A/B tests them against
  held-out tasks. Only prompts that beat the incumbent on a composite of
  task success and token efficiency are promoted. Champion prompts live
  in `backend/agents/skills/*.md`.
2. **Algorithm Discovery** - the engine proposes new signal and portfolio-
  construction algorithms by combining academic-frontier ingestion
  (phase-D) with the current factor zoo, drafts each as a backtestable
  module under `backend/backtest/experiments/`, runs the full
  DSR/PBO/Sortino gate, and submits survivors to the Recursive Evolution
  Loop as challengers.
3. **Alpha Velocity Resource Allocator** - the engine measures each
  research branch's Alpha Velocity (see `## Key Metrics`) weekly and
  reallocates the 10 research cron slots (see `## Cron Budget`)
  proportional to demonstrated Alpha Velocity, with a floor of 1 slot per
  active family and a hard cap of 4 slots per branch. This prevents
  starvation while concentrating compute on what pays.

The loop: intelligence (phase-D) -> RPO + Algorithm Discovery -> Harness
challenger -> Recursive Evolution Loop promotion gate (phase-B) ->
Alpha Velocity reallocation -> back to intelligence. See
`handoff/phase-proposals/phase-E-meta-evolution-engine.md` for the full
state machine and promotion gates.

## Cron Budget

The live environment has a **hard cap of 15 Claude scheduled routine runs
per day**. This is a load-bearing architectural invariant: every phase
that introduces a scheduled job must fit inside this cap or reclaim a slot
from elsewhere.

Split:

- **5 trading-ops slots (reserved, non-negotiable)**:
  1. Morning digest (pre-open market-state summary)
  2. Evening digest (post-close P&L and reconciliation)
  3. Watchdog (process-health and data-freshness check)
  4. Daily paper-trading cycle (MAS decision -> ticket queue)
  5. Kill-switch heartbeat (liveness probe for the Immutable Core)
- **10 research/ingestion slots (reallocated by Alpha Velocity)**:
  Distributed across the four Global Intelligence families (phase-D), the
  Recursive Prompt Optimization runs, and Algorithm Discovery sweeps, with
  weekly reallocation driven by the Alpha Velocity Resource Allocator
  (phase-10.7, see `## Meta-Evolution Engine`).

The canonical allocation lives in `.claude/cron_budget.yaml` (to be
created by the masterplan coordinator). The Allocator reads and writes
that file; the Immutable Core (phase-A) verifies the total never exceeds
15 before each scheduled run. See the phase-10.7 Cron Budget Allocator
step for enforcement details.

## Key Metrics

All promotion, reallocation, and pause decisions use these metrics and
thresholds. Thresholds are immutable once set and may only be changed via
explicit Peder signoff recorded in the FINRA-grade audit trail
(phase-4.8.9).

- **Sortino Ratio** - `S = (Rp - T) / sigma_d` where `Rp` is the strategy
  return, `T` is the minimum acceptable return (default: 1-month T-bill
  risk-free rate), and `sigma_d` is the downside deviation (std of returns
  below `T`). Champion-replacement threshold: Challenger `S` > Champion
  `S` + 0.3 over an overlapping 6-month window.
- **Deflated Sharpe Ratio (DSR)** - Bailey and Lopez de Prado (2014).
  Adjusts Sharpe for selection bias, non-normality, and sample length.
  Promotion gate: DSR > 0.95 (single-trial candidate), DSR > 0.99 when
  the candidate came out of a sweep with N > 50 trials.
- **Probability of Backtest Overfitting (PBO)** - Bailey, Borwein, Lopez
  de Prado, Zhu (2016). Combinatorially symmetric cross-validation
  estimate of the probability the backtest is overfit. Promotion gate:
  PBO < 0.2.
- **Alpha Velocity** - `New Realized Alpha (USD) / Compute Cost (USD)`
  per research branch over a rolling weekly window. Input to the Alpha
  Velocity Resource Allocator (phase-10.7) for cron-slot reallocation.
- **Net System Alpha** - `Profit - (Risk Exposure + Compute Burn)`, as
  defined in `## The Red Line`. Triggers the global pause when negative
  for two consecutive weeks.

## Governance

Hardcoded immutable limits (Immutable Core, phase-A):

- Max gross exposure per account: configured in `backend/.env`, enforced
  pre-trade.
- Max single-position weight: 10% of NAV.
- Max daily drawdown before auto-flatten: 3% of NAV.
- Max LLM spend per day: configured, enforced by the cost gate in
  `backend/llm_client.py`.
- Max scheduled cron runs per day: 15 (see `## Cron Budget`).

Human-in-the-loop approvals (Peder signs off in Slack, audit-logged):

- Any promotion of a challenger from paper to live capital.
- Any reallocation that removes a cron slot from the 5 trading-ops
  reserve.
- Any change to the thresholds in `## Key Metrics`.
- Any phase transition from pending to in-progress when the phase spends
  a reserved trading-ops slot.

Audit trail: every agent decision, prompt mutation, algorithm proposal,
cron reallocation, and capital promotion is written to an append-only
BigQuery audit table with a rationale blob, retained for **3 years** in
compliance with the FINRA 2026 GenAI oversight guidance (phase-4.8.9
WORM retention step). The Sovereign Dashboard (phase-C) surfaces this
audit trail to Peder with filters by phase, agent, and decision class.

## References

1. Sortino, F. A., and Price, L. N. (1994). "Performance measurement in a
  downside risk framework." Journal of Investing.
  https://www.actuaries.org/LIBRARY/Colloquia/Pension/Brussels_2003/Campbell.pdf
2. Bailey, D. H., and Lopez de Prado, M. (2014). "The Deflated Sharpe
  Ratio: Correcting for Selection Bias, Backtest Overfitting, and
  Non-Normality." Journal of Portfolio Management.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
3. Bailey, D. H., Borwein, J., Lopez de Prado, M., and Zhu, Q. J. (2016).
  "The Probability of Backtest Overfitting." Journal of Computational
  Finance. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
4. Lopez de Prado, M. (2018). "Advances in Financial Machine Learning."
  Wiley. Chapters on backtest overfitting and combinatorial purged CV.
  https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
5. Bostrom, N. (2014). "Superintelligence: Paths, Dangers, Strategies."
  Oxford University Press. Chapter on recursive self-improvement.
  https://global.oup.com/academic/product/superintelligence-9780199678112
6. Yudkowsky, E. (2013). "Intelligence Explosion Microeconomics." MIRI
  Technical Report. https://intelligence.org/files/IEM.pdf
7. FINRA (2026). "Regulatory Notice on GenAI Oversight, Model Risk, and
  Recordkeeping." https://www.finra.org/rules-guidance/notices
8. Federal Reserve SR 11-7 (2011). "Guidance on Model Risk Management."
  https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm
9. BCBS (2019). "Minimum capital requirements for market risk (FRTB)."
  Basel Committee. https://www.bis.org/bcbs/publ/d457.htm
10. Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio
  Management. https://web.stanford.edu/~wfsharpe/art/sr/sr.htm
11. Harvey, C. R., Liu, Y., and Zhu, H. (2016). "... and the Cross-
  Section of Expected Returns." Review of Financial Studies.
  https://academic.oup.com/rfs/article/29/1/5/1843824
12. Karpathy, A. (2023). "Intro to Large Language Models" and the
  autonomous-agent loop talks. https://karpathy.ai/
13. Anthropic (2024). "Building effective agents."
  https://www.anthropic.com/research/building-effective-agents
14. Kahneman, D., Sibony, O., and Sunstein, C. R. (2021). "Noise: A Flaw
  in Human Judgment." Little, Brown Spark. Cited for decision-audit
  rationale. https://www.hachettebookgroup.com/titles/daniel-kahneman/noise/
15. SEC (2023). "Predictive Data Analytics Rule Proposal." Cited for
  conflicts-of-interest governance under model-driven decisions.
  https://www.sec.gov/rules/proposed/2023/34-97990.pdf
16. NIST AI RMF 1.0 (2023). "AI Risk Management Framework."
  https://www.nist.gov/itl/ai-risk-management-framework
17. CFTC (2024). "Request for Comment on the Use of AI in CFTC-Regulated
  Markets." https://www.cftc.gov/PressRoom/PressReleases/8853-24
