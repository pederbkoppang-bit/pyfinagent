# Masterplan Draft - Three New Phases

Status: draft (pending Peder approval before integration into `.claude/masterplan.json`)
Author: harness / lead architect review, 2026-04-17
Scope: three new phases proposed in response to the lead-architect brief.

## The Red Line

Every phase in this proposal is gated on a single objective:

> **Maximize expected profit per dollar of compute + data + execution cost, while
> capping realized drawdown via hard risk gates.**

This is the durable system goal (saved in memory `project_system_goal.md`
and codified as the top-level masterplan `goal`). Every phase below has an
explicit "Red Line contract": what cost it introduces, what profit edge it
defends, and what risk cap it enforces.

---

## Step 1 - Research findings (summary; evidence in Appendix)

### 1a. Codebase state (audited 2026-04-17)

- **MAS today**: 3 agents (Planner / Strategy Generator / Evaluator) with an
  async event bus (`backend/agents/mas_events.py`) but synchronous
  orchestration; communication is via handoff files, not a shared
  blackboard.
- **Paper trading**: internal BigQuery simulation, not broker-backed. Daily
  cron cadence. Metrics v2 done (PSR / DSR / Calmar / Sortino + bootstrap
  CI). Kill-switch hardened.
- **MCP surface**: only Slack MCP live in `.mcp.json`. Three in-house FastMCP
  stubs exist (`backend/mcp/data_server.py`, `backtest_server.py`,
  `signals_server.py`) but are dormant. Phase-3.5 (just added) plans
  external MCP adoption (Alpaca / EDGAR / FMP / FRED).
- **Frontend**: 11 routes; no orphans; homepage is a thin shell; `/agents`
  page exposes MAS live but is lightly used; `/paper-trading` is
  evaluation-grade (phase-4.5 complete).
- **Karpathy hook**: phase-2 step 2.10 is a STUB - a verification command
  only, no contract, no budget, no candidate space.

### 1b. Karpathy autoresearch (github.com/karpathy/autoresearch)

The loop is: LLM reads `train.py` + `results.tsv` + git log -> proposes a
diff to `train.py` -> runs training for **exactly 5 wall-clock minutes** ->
measures scalar `val_bpb` -> commits if better, reverts if not -> repeat.

Adaptation surface to trading is small (3-4 files / functions):

1. Replace the validation-loss evaluator with a backtest call returning the
   chosen scalar (Sharpe / DSR / Calmar).
2. Extend `results.tsv` schema with `sharpe, dsr, max_dd, profit_factor`.
3. Rewrite `program.md` objective from "minimize val_bpb" to
   "maximize DSR subject to max_dd <= X and cost <= Y".
4. Flip comparator and add multi-metric gate (reject on drawdown breach
   even when scalar improves).

**Critical replication concern**: running 100+ backtests/night is the exact
overfitting trap Bailey & Lopez de Prado's DSR and PBO papers were written
about. A harness-in-a-loop that does not apply DSR deflation after each
cycle will reliably select overfit strategies. PBO literature puts the
false-positive rate at >50% once trials exceed 20 without purged CV.
Therefore DSR + Combinatorial Purged CV (CPCV) are **mandatory** gates
before any autoresearch candidate can be promoted to paper-live.

### 1c. MCP as MAS backbone (April 2026 consensus)

MCP spec 2025-11-25 is **tool-to-agent, not agent-to-agent**. The 2026
roadmap introduces a `Tasks` primitive (SEP-1686) as an experimental step
toward agent-to-agent lifecycle, but it is not shipped. The idiomatic 2026
stack that has emerged:

- **MCP** (vertical): tool / data access.
- **A2A** (horizontal, Google, April 2025, 150+ adopters): agent-to-agent
  coordination with task lifecycle semantics (retry, expiry, cancellation).
- **Orchestrator LLM**: one per system, owns routing.

For a 3-node trading MAS (Data / Strategy / Risk), the recommended shape
is: ONE orchestrator LLM + THREE MCP tool servers (each agent exposes its
capabilities as an MCP server to the orchestrator). For agent-to-agent
handoffs (e.g. Strategy -> Risk without round-tripping through the
orchestrator) use A2A or a lightweight task-delegation layer. MCP alone
cannot express task delegation with lifecycle semantics.

Published failure modes to mitigate:
- Tool-call storms (observed: 100+ redundant calls); fix with explicit
  terminal states + debounce on duplicate calls within a sliding window.
- Context leakage of secrets; fix with capability tokens scoped per
  session + PII filter at input.
- Context window overflow from large MCP responses; fix with hard output
  size limits and async `handleId` returns instead of full payloads.

---

## Step 2 - Three new phases

### Phase X - phase-3.7: MAS Paper Trading & MCP Infrastructure

**Placement**: after phase-3.5 (MCP audit decides adoptees) and before
phase-4.6 (smoketest validates the new surface end-to-end). It must land
before phase-5 multi-market expansion because multi-market inherits the
MAS wiring chosen here.

**Why this order**: phase-3.5 tells us *which* external MCPs we adopt;
phase-3.7 wires them into the MAS; phase-4.6 smoketests the wired system
before go-live. Doing the wiring before the audit would force rework; doing
it after the smoketest would invalidate the smoketest.

**Red Line contract**:
- **Profit**: switching paper-trading execution from BQ simulation to
  Alpaca paper account (from phase-3.5) removes the sim-to-live reality
  gap, which is the #1 reason paper-trading alpha vanishes in production.
- **Cost**: one orchestrator LLM + thin MCP servers is cheaper than the
  current multi-LLM MAS because only the orchestrator pays the big-model
  tax; tool servers are deterministic code.
- **Risk**: A2A task-delegation layer gives retry / expiry / cancellation
  semantics the harness lacks today, which caps the blast radius of a
  runaway agent.

**Steps (proposed)**:

| # | Name | Verification scalar |
|---|------|---------------------|
| 3.7.0 | MAS communication contract: tool-to-agent (MCP) vs agent-to-agent (A2A) decision doc + ADR | ADR landed; orchestrator shape chosen |
| 3.7.1 | Promote `backend/mcp/data_server.py` to first-class MCP server exposing signals + fundamentals + prices | Data agent reachable via MCP handshake; parity >= 0.95 vs direct Python client |
| 3.7.2 | Promote `backend/mcp/signals_server.py` to MCP Strategy Agent (proposes candidates) | Strategy agent emits >= 5 well-formed candidates/call; DSR annotated |
| 3.7.3 | New `backend/mcp/risk_server.py` Risk Agent MCP (wraps `kill_switch` + Lopez de Prado PBO check) | Risk agent vetoes any candidate with PBO > 0.5 or projected DD > cap |
| 3.7.4 | A2A lightweight task-delegation layer (or AutoGen-style message bus if A2A SDK not ready) with retry/expiry | Task handoff Data -> Strategy -> Risk round-trip p95 <= 2s |
| 3.7.5 | Swap paper-trading execution from BQ sim to Alpaca paper account (from phase-3.5 adoption) behind a feature flag | Paper orders placed via Alpaca MCP; reconciliation drift <= 1% vs BQ shadow for 1 wall-clock week |
| 3.7.6 | Context-budget guardrails: per-MCP output size cap + debounce on duplicate tool calls within 60s sliding window | Tool-call storm regression test passes (no tool called > 5x in 60s) |
| 3.7.7 | Capability tokens scoped per session; PII filter on MCP input layer | Secret-leak regression test passes (no env-var name ever reaches an LLM) |

Depends on: phase-3.5 (for the adoption shortlist of external MCPs).

---

### Phase Y - phase-4.7: UI/UX Optimization & Frontend Audit

**Placement**: after phase-4.6 (smoketest) and before phase-5 (multi-market
expansion adds more pages). It cannot land before 4.6 because we need the
smoketest to tell us which pages are actually exercised end-to-end; it
cannot wait until after phase-5 because multi-market adds views that we do
NOT want to design on top of a stale IA.

**Why this order**: the smoketest gives us an honest usage profile (which
pages the harness / MAS actually hits), so the audit's "remove redundant
pages" decisions are evidence-backed rather than taste-driven. Running the
audit after multi-market lands would let bad IA propagate into new
markets' views.

**Red Line contract**:
- **Profit**: every redundant page the operator has to navigate to
  override a bad signal costs P&L when volatility spikes; consolidating
  operator-critical state on the homepage is an alpha-preserving
  intervention, not cosmetic.
- **Cost**: the `/agents` page is already wired to the MAS event bus and
  is under-used; surfacing it in the homepage avoids building a second
  observability stack.
- **Risk**: the homepage is the first thing a human opens during an
  incident. A clear homepage shortens mean-time-to-flatten when the
  kill-switch needs manual invocation.

**Steps (proposed)**:

| # | Name | Verification scalar |
|---|------|---------------------|
| 4.7.0 | Frontend route inventory + usage telemetry (which pages did the harness / human actually open over last 30 days) | `frontend_usage.json` emitted; every route has a usage count |
| 4.7.1 | Remove or merge pages with 0 opens and no MAS consumer | At most 8 top-level routes after audit |
| 4.7.2 | Redesign `/` (homepage) as the MAS operator cockpit: OpsStatusBar + live signal ticker + kill-switch shortcut + next-scheduled-cycle + current cost-spend | Lighthouse perf >= 90; first meaningful paint <= 1.5s |
| 4.7.3 | Add MAS Monitoring view elements missing today: per-agent latency, per-agent cost-per-cycle, agent health heartbeat | Matches `mas_events.py` emitted events 1:1 |
| 4.7.4 | Add Autoresearch Run view (consumes phase-8.5 output): candidate leaderboard with DSR, PBO, realized P&L if promoted | Leaderboard refresh <= 10s; DSR column present |
| 4.7.5 | Cross-page consistency pass against `.claude/rules/frontend.md` + `frontend-layout.md` (OpsStatusBar pattern, Phosphor icons, no raw emoji, scrollbar-thin, etc.) | Lint passes; visual diff review landed |
| 4.7.6 | Accessibility + keyboard-nav audit (operator must be able to flatten positions via keyboard alone during an incident) | WCAG 2.1 AA; keyboard-only kill-switch workflow green |

Depends on: phase-4.6 (smoketest usage telemetry), phase-3.7 (MAS
cockpit needs the final agent shape).

---

### Phase Z - phase-8.5: Autonomous Strategy Research (the Karpathy loop)

**Placement**: after phase-8 (transformer signals land so the candidate
space includes TimesFM / Chronos / Moirai / FinGPT) and before phase-9
(nightly data refresh cron supports the overnight loop). It supersedes the
phase-2 step 2.10 stub, which should be retired with a link here.

**Why this order**: the loop is only as good as its candidate space. Running
it before phase-6 news-sentiment, phase-7 alt-data, and phase-8
transformers means the LLM proposer is searching a tiny, stale space; doing
it after all signal-expansion phases land gives maximum surface. Phase-9
data-refresh cron is the mechanical prerequisite because the loop assumes
fresh data at each iteration.

**Red Line contract**:
- **Profit**: autonomous overnight search expands strategy diversity
  (100+ experiments per night vs ~5 manual per week), which is the mechanism
  by which the system realizes the durable goal of *dynamically shifting
  strategy to whichever one is currently making the most money*.
- **Cost**: wall-clock budget cap per experiment (default 5 min, matching
  Karpathy) + absolute USD budget cap per night + kill-switch on
  consecutive-losing-experiments count. No experiment runs if the
  remaining night-budget is below threshold.
- **Risk**: **DSR and PBO gates are blocking**. A candidate cannot be
  promoted to paper-live unless DSR > 0.95 AND PBO < 0.2 AND realized
  drawdown over the backtest window <= risk cap. No exceptions. This is
  the single most important line item in this phase - the Karpathy loop
  without DSR/PBO gating is a strategy-overfitting factory.

**Steps (proposed)**:

| # | Name | Verification scalar |
|---|------|---------------------|
| 8.5.0 | Retire phase-2 step 2.10 stub with a decision log linking here | `handoff/phase-2.10-supersede.md` landed |
| 8.5.1 | Define the candidate space: quant param ranges, agent prompt edits, feature-selection flags, model-arch toggles (TimesFM / Chronos / FinGPT from phase-8) | `candidate_space.yaml` committed; size >= 1e4 combinations |
| 8.5.2 | Wall-clock + USD budget enforcer (Karpathy 5-min default; USD cap per night per candidate) | Budget-exhausted termination fires deterministically in regression test |
| 8.5.3 | LLM proposer (Claude Opus) with read access to `results.tsv` + git log + `program.md` objective spec; write access restricted to the candidate file (same narrow-surface pattern Karpathy uses) | Proposer emits valid diff per cycle; diff touches only whitelisted files |
| 8.5.4 | Evaluator: run backtest against frozen OOS period, emit Sharpe / DSR / PBO / max_dd / profit_factor / cost / realized_pnl | `results.tsv` schema stable; each cycle appends one row |
| 8.5.5 | DSR + PBO blocking gate (Bailey & Lopez de Prado 2014; Bailey et al. 2016) with Combinatorial Purged CV | Any candidate failing DSR > 0.95 OR PBO < 0.2 is rejected and reverted; regression test covers both rejection paths |
| 8.5.6 | Promotion path: candidates clearing the gate enter paper-live via phase-3.7 Alpaca MCP with position size tied to realized DSR | Promotion shadow-runs for 5 trading days minimum before full capital; kill-switch auto-triggers on DD breach |
| 8.5.7 | Overnight orchestration cron (fires after phase-9 data refresh completes); throughput target ~100 experiments per night | Cron runs nightly; >= 80 experiments completed within budget; all results visible in phase-4.7 Autoresearch Run view |
| 8.5.8 | Weekly human-in-loop review packet (top-10 promotable candidates + DSR/PBO + cost-per-experiment + P&L if promoted) posted to Slack; no auto-promotion to real capital ever | Slack post rendered weekly; Peder approval required for capital promotion |

Depends on: phase-3.7 (MCP execution path), phase-6, 7, 8 (candidate
space), phase-9 (overnight data refresh), phase-4.7 (Autoresearch Run view).

---

## Step 2.5 - Virtual Fund Learnings Integration (cross-cutting)

Feedback from Peder (2026-04-17): the virtual-fund period (paper trading
since phase-4.5) is the richest source of truth we have about what will
break under real capital. Every learning we can harvest from it now is a
dollar we do not lose on day 1 of go-live. This is folded into all three
proposed phases as explicit steps rather than a separate phase, because
the learnings are substrate for each phase's decisions, not a parallel
track.

**What the virtual fund can teach us (concrete signals already in
BigQuery)**:

- `paper_portfolio` table: realized vs modeled returns per position -> the
  sim-to-live reality gap (slippage, fill latency, partial fills the BQ
  sim does not emulate).
- `paper_trades` table: every placed / executed / rejected trade with
  reason -> which risk gates actually fired, which candidates the Risk
  Agent killed pre-execution, decision latency.
- `paper_snapshots` daily NAV series: drawdown episodes, vol regimes the
  current strategy underperformed in, stability of the chosen strategy
  across market regimes.
- `reconciliation` overlay (phase-4.5.3): where paper NAV diverged from
  the parallel OOS backtest -> divergence episodes are where the live
  model is wrong.
- Kill-switch audit (`handoff/kill_switch_audit.jsonl`): every time we
  hit a hard gate, the input conditions, and whether the trigger was a
  true positive.
- MFE/MAE scatter (phase-4.5.9): exit-quality distribution -> how much
  realized P&L we leave on the table per trade.

**How each new phase consumes the learnings**:

### In phase-3.7 (MAS + MCP Infrastructure)

Add step **3.7.8 - Virtual Fund Reality-Gap Calibration**: run the Alpaca
MCP paper broker and the internal BQ sim in parallel for a full trading
week (shadow mode). Any divergence > 1% on price-fill, > 200ms on fill
latency, or any partial-fill the sim did not model becomes a calibration
patch on the BQ sim. This is the prerequisite to retiring the BQ sim as
the paper-trading execution path. The Red Line: we close the sim-to-live
gap *before* real capital, not after.

### In phase-4.7 (UI/UX Audit)

Add step **4.7.7 - Virtual Fund Learnings Dashboard**: a new page
(`/learnings` or a homepage tab) that surfaces:
- Top 10 reality-gap divergences from reconciliation.
- Kill-switch trigger distribution (true positives vs false alarms).
- Regime-keyed performance (which market conditions the current strategy
  underperformed in).
- Exit-quality degradation trend (MFE/MAE over time).

This is the operator cockpit the human uses to sign off go-live with
eyes open. Red Line: the human must be able to name the top 3 known
failure modes of the system before real capital flips on.

### In phase-8.5 (Autoresearch)

Add step **8.5.9 - Seed the candidate space from virtual-fund failure
cases**: the LLM proposer's first-cycle candidates are not random; they
are explicitly targeted at the regime-underperformance buckets and the
reality-gap buckets identified in 4.7.7. In other words, the Karpathy
loop starts by trying to fix what the virtual fund already told us is
broken, before it explores novel candidates. This tightens the feedback
loop from "100 random experiments per night" to "100 experiments per
night aimed at known-failure-modes first, novel search second."

**Additional cross-cutting artifact**: `handoff/virtual_fund_postmortem.md`
- a living document updated at the end of every virtual-fund week, owned
by the Evaluator agent, that feeds steps 3.7.8 / 4.7.7 / 8.5.9. Schema:
(regime, divergence episode, kill-switch trigger, MFE/MAE bucket) ->
(hypothesis, proposed fix, candidate pointer in `candidate_space.yaml`).

**Go-live gate update**: phase-4 step 4.4 ("Go-Live Checklist") picks up
a new deterministic boolean: `virtual_fund_postmortem_has_zero_unresolved_P0`.
If the postmortem contains any P0 (profit-destroying) unresolved item, the
gate is red and real capital cannot be deployed.

---

## Step 3 - Updated masterplan order (draft)

After integration, the phase order would be:

```
[x]  phase-0     Audit & Validate
[x]  phase-1    Quant Engine Optimization
[~]  phase-2    Three-Agent Harness (retire 2.10 when phase-8.5 lands)
[ ]  phase-3    LLM-Guided Research + MCP Integration
[ ]  phase-3.5  MCP Tool Audit & Adoption              *** just added ***
[ ]  phase-3.7  MAS Paper Trading & MCP Infrastructure *** NEW (X) ***
[x]  phase-4.5  Paper Trading Dashboard v2
[ ]  phase-4.6  Smoketest (step-by-step E2E)
[ ]  phase-4.7  UI/UX Optimization & Frontend Audit    *** NEW (Y) ***
[ ]  phase-5    Multi-Market Expansion
[ ]  phase-5.5  External Data-Source Audit
[ ]  phase-6    News & Sentiment Cron
[ ]  phase-7    Alt-Data & Scraping Expansion
[ ]  phase-8    Transformer / Modern LLM Signals
[ ]  phase-8.5  Autonomous Strategy Research (Karpathy) *** NEW (Z) ***
[ ]  phase-9    Data Refresh & Retraining Cron
[~]  phase-4    Production Readiness (depends on everything above; step 4.9 aggregate smoketest gates go-live)
```

## Summary of placement logic

- **phase-3.7 sits between 3.5 and 4.6** because MCP infrastructure must be
  wired immediately after adoption is decided and must be smoketested
  before any downstream phase uses it.
- **phase-4.7 sits after 4.6 and before 5** because the UI audit needs
  honest usage data from the smoketest and must not propagate into
  multi-market views.
- **phase-8.5 sits after phase-8 and before phase-9** because the
  autoresearch loop is only as good as its candidate space (needs all
  prior signal phases) and needs the data-refresh cron (phase-9) to
  guarantee fresh data per experiment.
- **phase-4 (Production Readiness) already depends on every other phase**
  after our prior reorder; adding 3.7, 4.7, and 8.5 to
  `phase-4.depends_on` keeps the go-live gate honest.

## Red Line thread (cross-phase summary)

| Phase | Profit lever | Cost cap | Risk cap |
|-------|--------------|----------|----------|
| 3.7 | Real-broker execution closes sim-to-live gap | One orchestrator LLM + deterministic MCP tool servers | A2A retry/expiry/cancel + context-size + capability tokens |
| 4.7 | Operator override is fast during vol spikes | Consolidate, don't duplicate observability | Keyboard-only kill-switch + clear MAS cockpit during incidents |
| 8.5 | Autonomous search over expanded candidate space | Wall-clock + USD budget per experiment + consecutive-loss kill-switch | DSR + PBO + CPCV gates are blocking; no capital promotion without Peder approval |
| 2.5 (cross-cut) | Virtual fund teaches us what breaks before real capital does | One postmortem doc fed by existing BQ tables - no new ingest cost | Go-live gate picks up `virtual_fund_postmortem_has_zero_unresolved_P0` |

---

## Appendix - Research references (cited per claim above)

**Karpathy autoresearch**:
- https://github.com/karpathy/autoresearch (repo, full read)
- https://github.com/karpathy/autoresearch/blob/master/README.md (README, full read)
- https://www.verdent.ai/guides/what-is-autoresearch-karpathy (loop walkthrough)
- https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai (5-min budget, ~100 experiments/night)
- https://jangwook.net/en/blog/en/karpathy-autoresearch-overnight-ml-experiments/ (wall-clock budget details)
- https://www.philschmid.de/autoresearch (adaptation surfaces)
- https://arxiv.org/html/2603.24647 (Schwanke et al. 2026, bandit extension)
- https://www.datacamp.com/tutorial/guide-to-autoresearch (candidate selection)

**Overfitting / DSR / PBO**:
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 (Bailey & Lopez de Prado 2014, DSR)
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 (Bailey et al., PBO)
- https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf (PBO paper PDF)
- https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110 (CPCV 2024 survey)

**MCP / A2A / MAS protocols**:
- https://modelcontextprotocol.io/specification/2025-11-25 (MCP spec)
- https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/ (Tasks SEP-1686)
- https://onereach.ai/blog/guide-choosing-mcp-vs-a2a-protocols/ (MCP vs A2A)
- https://www.gravitee.io/blog/googles-agent-to-agent-a2a-and-anthropics-model-context-protocol-mcp (MCP/A2A stack)
- https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/ (A2A launch)
- https://stellagent.ai/insights/a2a-protocol-google-agent-to-agent (A2A timeline)
- https://medium.com/google-cloud/building-a-financial-ai-agent-with-google-adk-a2a-and-mcp-084f5937f1e8 (financial agent using MCP+A2A+ADK)
- https://arxiv.org/html/2504.21030v1 (MAS survey April 2025)
- https://www.digitalapplied.com/blog/ai-agent-protocol-ecosystem-map-2026-mcp-a2a-acp-ucp (ecosystem map)
- https://www.ruh.ai/blogs/ai-agent-protocols-2026-complete-guide (2026 protocols guide)
- https://github.com/rinadelph/Agent-MCP (3-node MCP MAS implementation)

**MCP failure modes**:
- https://dev.to/aws/why-ai-agents-fail-3-failure-modes-that-cost-you-tokens-and-time-1flb (tool-call storms, 14->2 fix)
- https://arxiv.org/html/2511.20920v1 (MCP security 2025-11)
- https://simonwillison.net/2025/Apr/9/mcp-prompt-injection/ (prompt injection)
- https://authzed.com/blog/timeline-mcp-breaches (breach timeline)
- https://foojay.io/today/best-practices-for-working-with-ai-agents-subagents-skills-and-mcp/ (best practices)
