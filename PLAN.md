# PyFinAgent — Master Plan v2

> **Goal**: Make money. Ship a validated, evidence-based trading signal system by May 2026.
> **Budget**: TIGHT — negative cash flow (-$10K, $200/month costs). Every dollar must have ROI. **⚠️ LLM API costs require Peder's explicit approval.**
> **Timeline**: March-April = validate + optimize + research. May = go live (Slack signals → manual trading).
> **Architecture**: Three-agent harness (Planner → Generator → Evaluator) inspired by [Anthropic's harness design for long-running apps](https://www.anthropic.com/engineering/harness-design-long-running-apps).

---

## Evidence-Based Development

**Every design decision traces back to a paper, industry practice, or empirical evidence.**

Research log: [`RESEARCH.md`](RESEARCH.md) — maintained alongside the plan.

Before implementing any major feature:
1. **Deep Research** — actively search the web for latest papers, blog posts, documentation. Don't rely on existing knowledge alone. Use `web_search` and `web_fetch` to find:
   - Google Scholar for peer-reviewed papers and citation networks
   - arXiv/SSRN preprints on the specific technique
   - University research groups (MIT Sloan, Stanford GSB, Oxford MFE, Princeton ORF, Chicago Booth, NYU Stern)
   - Consulting & industry reports (McKinsey, BCG, Bain, Deloitte, Oliver Wyman — especially on AI in finance)
   - Quant firm research (Two Sigma, AQR, Man Group/AHL, Citadel, Renaissance, DE Shaw, Bridgewater)
   - AI lab engineering blogs (Anthropic, OpenAI, Google DeepMind, Meta FAIR, Microsoft Research)
   - GitHub repos and open-source implementations
   - Practitioner discussions and post-mortems
2. **Read & Extract** — fetch and read the most relevant 3-5 sources. Extract concrete methods, thresholds, pitfalls.
3. **Document** — add to RESEARCH.md with URL, citation, key insight, and how it applies to our work
4. **Reference** — cite in PLAN.md and `handoff/contract.md` when justifying design
5. **Validate** — evaluator criteria should use published thresholds (e.g., t-stat ≥ 3.0 from Harvey et al.)

**Deep Research Sources (priority order):**
1. Academic (peer-reviewed): Google Scholar, arXiv, SSRN, NBER, Journal of Finance, Journal of Financial Economics, Review of Financial Studies
2. University research: MIT Sloan, Stanford GSB, Oxford MFE, Princeton ORF, Chicago Booth, NYU Stern, Imperial College, ETH Zurich
3. AI labs: Anthropic engineering blog, OpenAI research, Google DeepMind, Meta FAIR, Microsoft Research
4. Quant firms: Two Sigma, AQR Capital, Man Group/AHL, Citadel, Renaissance Technologies, DE Shaw, Bridgewater, WorldQuant
5. Consulting & industry: McKinsey (QuantumBlack), BCG (GAMMA), Deloitte AI Institute, Oliver Wyman, Accenture Applied Intelligence
6. Practitioner: López de Prado, Ernie Chan, Karpathy, Cliff Asness, QuantConnect forums
7. Open-source: FinRL, TradingAgents, autoresearch, QuantLib, zipline-reloaded
8. Regulatory: SEC, FINRA, MiFID II, Norwegian FSA (Finanstilsynet)

**Deep Research Checklist (run for each plan step):**
```
□ Google Scholar search: "[topic]" — sorted by relevance and recency (last 2 years)
□ arXiv/SSRN search: "[topic] financial machine learning"
□ University research groups: check relevant lab pages (MIT, Stanford, Oxford, etc.)
□ AI lab blogs: Anthropic, OpenAI, DeepMind — searched for related techniques
□ Quant firm publications: AQR, Two Sigma, Man Group whitepapers
□ Consulting reports: McKinsey/BCG/Deloitte on AI in finance (if applicable)
□ GitHub: recent implementations/repos
□ Read 3-5 most relevant sources in full (not just abstracts)
□ Documented findings in RESEARCH.md with URLs
□ Identified concrete thresholds/methods to adopt
□ Noted warnings/pitfalls from literature
```

---

## Harness-Driven Execution

**Every phase of this plan follows the harness pattern — not just the optimizer.**

The same Planner → Generator → Evaluator cycle that runs backtests also governs how we execute the master plan itself:

```
For each plan step:
  0. RESEARCH — Deep research: web search for latest papers, industry docs, peer implementations.
               Search arXiv, SSRN, Anthropic/OpenAI/DeepMind blogs, practitioner posts.
               Fetch and read relevant papers. Update RESEARCH.md with findings.
               This is NOT optional — every step must be grounded in current best practices.
  1. PLAN     — Define what "done" looks like (contract in handoff/contract.md, cite research)
  2. GENERATE — Do the work (code, config, research)
  3. EVALUATE — Independently verify it worked (tests, validation, review)
  4. DECIDE   — PASS → move to next step | FAIL → revert and retry | CONDITIONAL → fix and re-evaluate
  5. LOG      — Update PLAN.md progress log, HEARTBEAT.md, memory, Slack
```

**Why this matters:** Without evaluation, we ship broken things. Phase 1 proved this — the optimizer said Sharpe improved, but independent sub-period validation showed it actually destroyed the strategy (Sharpe -4.57). The harness caught it.

**Concrete enforcement — NO EXCEPTIONS:**
- Before starting any phase step → deep research → write `handoff/contract.md` with hypothesis + success criteria
- After completing the work → run independent validation (not just "it compiles")
- After validation → write `handoff/evaluator_critique.md` with pass/fail + evidence
- `run_harness.py` automates this for backtest work; for non-backtest work, Ford follows the same protocol manually
- **Every step in this plan has a `> Harness:` tag.** If it doesn't, it's a bug in the plan.
- **Ford cannot skip the harness.** If time pressure tempts skipping evaluation, that's exactly when evaluation matters most.
- **Each step produces 3 handoff files minimum:** `contract.md`, `experiment_results.md`, `evaluator_critique.md`

---

## Architectural Philosophy

### Why a multi-agent harness, not a monolith

Our previous approach was a single optimizer loop: propose param → run backtest → keep/discard. This is the "solo agent" equivalent. It works but hits two ceilings described by Anthropic's research:

1. **Self-evaluation failure.** The optimizer can't judge whether its own improvements are real alpha or overfitting. It "praises its own work" — a Sharpe improvement passes DSR and gets kept, even when a human quant would flag it as fragile.

2. **Context degradation.** Long optimization runs lose coherence. The optimizer doesn't remember why earlier experiments failed, can't spot patterns across 40+ experiments, and can't propose structural changes (new features, strategy rewrites) — only parameter perturbations.

### The Anthropic harness pattern applied to quant research

From the article's core insight: **separate the agent doing the work from the agent judging it.** Tuning a standalone evaluator to be skeptical is far more tractable than making a generator critical of its own work.

Applied to our domain:

| Anthropic Pattern | PyFinAgent Equivalent |
|---|---|
| **Planner** — expands 1-sentence prompt into full spec | **Research Planner** — reads experiment log + academic literature, proposes next research direction |
| **Generator** — implements features sprint by sprint | **Strategy Generator** — implements code changes, runs backtests, produces experiment results |
| **Evaluator** — tests the running app via Playwright, grades against criteria | **Quant Evaluator** — runs independent validation: overfitting tests, robustness checks, out-of-sample verification, academic cross-checks |
| **Sprint contracts** — generator and evaluator negotiate "done" criteria before work | **Experiment contracts** — evaluator defines what constitutes a valid improvement BEFORE the generator runs it |
| **Context resets** — fresh agent with structured handoff artifact | **Run boundaries** — each optimization cycle starts fresh with a handoff file containing: best params, experiment history, evaluator critique, next research directions |
| **Grading criteria** — concrete, gradable terms for subjective quality | **Validation criteria** — concrete, gradable terms for strategy quality (see below) |

### Grading criteria for strategy quality

Inspired by the article's four design criteria, adapted for quantitative finance:

1. **Statistical validity** (most important): Is the improvement statistically significant? DSR ≥ 0.95. Sharpe stable across 5+ random seeds. No look-ahead bias.

2. **Robustness**: Does performance hold across regimes? Test on 2008 crisis, 2020 COVID crash, 2022 bear market separately. Feature importance stable across windows.

3. **Simplicity**: Is the complexity justified? Every new parameter must improve Sharpe by at least +0.05 to be kept. Reject complexity that doesn't justify its delta. (Harvey, Liu & Zhu 2016: t-stat ≥ 3.0 for new factors.)

4. **Reality gap**: How close is backtest to real trading? Transaction costs modeled realistically. No partial fills assumed. Survivorship bias addressed. Slippage estimated.

---

## Phase 0: Audit & Validate ✅ COMPLETE
*Completed March 25-26. Foundation is solid.*

- [x] Walk-forward leakage audit — no future data leaks found
- [x] Formula validation — Sharpe, DSR, sample weights verified against papers
- [x] Quality Score — full Asness (2019) QMJ implementation (4 dimensions)
- [x] Mean reversion — mr_holding_days correctly wired
- [x] Cross-sectional percentile ranking — replaces hardcoded normalization

---

## Phase 1: Quant Engine Optimization ✅ COMPLETE
*Completed March 27-28. Sharpe 0.9848 → 1.1705 (+19%).*

### Implemented improvements (all zero LLM cost):
- [x] **1.2** Cross-sectional percentile features (13 new regime-independent features)
- [x] **1.3** Turbulence index integration (market stress → position sizing)
- [x] **1.4** Fractional Kelly Criterion (half-Kelly, Thorp recommendation)
- [x] **1.5** Volatility targeting + trailing stops + dynamic risk-free rate (FRED)
- [x] **1.6** Regime-aware strategy selection (normal/stressed/crisis)
- [x] **1.7** Performance-based position scaling (20-day adaptive risk)
- [x] **1.8** Sector-based diversification (correlation penalties)
- [x] **1.9** Market microstructure transaction costs (Almgren-Chriss model)

### Current best: Sharpe 1.1705 | DSR 0.9984 | Return 80.2% | MaxDD -12.0%

---

## Phase 2: Three-Agent Harness (Weeks 3-4) — ✅ CORE IMPLEMENTED
*Automated harness loop operational. `run_harness.py` runs autonomous Planner → Generator → Evaluator cycles.*

### 2.0 Harness Architecture ✅ OPERATIONAL

**Implementation:** `run_harness.py` (718 lines) — autonomous cycle orchestrator.

```
Usage: python run_harness.py [--cycles N] [--iterations-per-cycle N] [--dry-run]
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PLANNER ✅                            │
│  Reads: experiment_log.tsv, evaluator_critique.md               │
│  Writes: research_plan.md, contract.md (sprint contract)        │
│  Heuristic: plateau detection, param saturation, weak periods   │
│  Runs: Once per cycle (automatic)                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ research_plan.md + contract.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STRATEGY GENERATOR ✅                           │
│  Wraps: QuantStrategyOptimizer.run_loop() (Karpathy autoresearch)│
│  Reads: optimizer_best.json, research_plan.md                    │
│  Writes: experiment_results.md, updated optimizer_best.json      │
│  Runs: N iterations per cycle (configurable, default 10)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ experiment_results.md
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUANT EVALUATOR ✅                             │
│  Runs: 5 INDEPENDENT backtests (separate engine instances)       │
│  Tests: 3 sub-periods + 2× costs + full-period DSR              │
│  Grades: 4 criteria independently (anti-leniency protocol)       │
│  Writes: evaluator_critique.md (PASS/FAIL/CONDITIONAL + scores)  │
│  Decision: PASS → save | FAIL → auto-revert | CONDITIONAL → warn │
└──────────────────────────┬──────────────────────────────────────┘
                           │ evaluator_critique.md
                           ▼
                    (feeds back to Planner — next cycle)
```

**Cycle time:** ~25-40 minutes. Three cycles = ~2 hours autonomous work.

### 2.1 Handoff Artifacts ✅ IMPLEMENTED

Each agent communicates via files, not conversation. This is the key unlock from the article — files survive context resets and carry precise state.

**`handoff/research_plan.md`** — Planner output:
```markdown
## Cycle N Research Direction
### Hypothesis: [what we think will improve Sharpe]
### Evidence: [patterns from experiment log that support this]
### Proposed changes: [specific code modifications or new params]
### Success criteria: [what the evaluator should check]
### Risk: [what could go wrong / overfitting concerns]
```

**`handoff/experiment_results.md`** — Generator output:
```markdown
## Cycle N Results
### Changes made: [code diffs, new params]
### Experiments run: [N iterations, best/worst Sharpe, kept/discarded]
### Best result: [params, Sharpe, DSR, return, drawdown]
### MDA feature importance: [top 10 features and stability]
### Observations: [anything unexpected]
```

**`handoff/evaluator_critique.md`** — Evaluator output:
```markdown
## Cycle N Evaluation
### Statistical Validity: [score 1-10, detailed assessment]
### Robustness: [score 1-10, regime-specific tests]
### Simplicity: [score 1-10, complexity vs improvement tradeoff]
### Reality Gap: [score 1-10, backtest-to-live concerns]
### Verdict: PASS / FAIL / CONDITIONAL
### Required fixes: [if FAIL, what must change]
### Suggestions for next cycle: [what the planner should consider]
```

### 2.2 Research Planner Agent ✅ IMPLEMENTED (Heuristic)

**Role:** The "quant researcher" who reads all experiment data and decides what to investigate next. This replaces random parameter perturbation with directed research.

**Implementation:** Heuristic rule-based planner in `run_harness.py`. Reads experiment TSV, detects plateaus (10+ discards), saturated params (5+ consecutive discards on same param), diminishing returns (<0.02 delta), and evaluator-flagged weak sub-periods. Writes sprint contracts. LLM planner upgrade deferred to Phase 3 (requires budget approval).

**Planner reads:**
- Full experiment TSV (40+ experiments with params, Sharpe, DSR, kept/discarded)
- Evaluator's previous critique
- Academic papers relevant to current research direction
- MDA feature importance trends across experiments

**Planner decides:**
- What type of change to try next (param tuning vs feature engineering vs strategy change)
- Which parameters to explore (based on sensitivity analysis from experiment log)
- Whether to continue current direction or pivot (analogous to the article's "refine or pivot" decision)
- Whether we've hit diminishing returns on current approach

### 2.3 Strategy Generator Agent ✅ INTEGRATED

**Role:** The "quant developer" who implements changes and runs experiments. This is our current optimizer, enhanced with the ability to make structural changes (not just param sweeps).

**Implementation:** `run_generator()` in `run_harness.py` wraps `QuantStrategyOptimizer.run_loop()`. Creates fresh engine each cycle. Exports results to `handoff/experiment_results.md`. The optimizer itself is Karpathy autoresearch-style (zero LLM cost, param perturbation + walk-forward backtest).

**Generator capabilities (expanding):**
- [x] Single-parameter perturbation (current, Karpathy autoresearch-style)
- [x] Warm-start from previous best (skip baseline if optimizer_best.json exists)
- [x] Feature caching (reuse features when only ML hyperparams change)
- [x] Early stopping (abort experiments below 85% of best Sharpe)
- [ ] Multi-parameter coordinated proposals (param groups that interact) — Phase 2.7
- [ ] Feature addition/removal experiments (add a feature, measure impact) — Phase 2.7
- [ ] Strategy switching experiments (try blend vs triple_barrier vs quality_momentum) — Phase 2.7
- [ ] Ablation studies (remove one improvement at a time, measure drop) — Phase 2.7

### 2.4 Quant Evaluator Agent ✅ IMPLEMENTED

**Role:** The "skeptical reviewer" who independently validates results. This is the key missing piece from our current system. The article's core insight: self-evaluation is unreliable; external evaluation drives real quality.

**Implementation:** `run_evaluator()` in `run_harness.py`. Runs 5 independent backtests with SEPARATE engine instances. Grades 4 criteria independently with anti-leniency protocol (score before verdict, never upgrade after). Writes `handoff/evaluator_critique.md` with pass/fail verdict and detailed scoring.

**Evaluator checks (grading criteria):**

#### Statistical Validity (weight: 40%)
- [x] DSR ≥ 0.95 on the kept result — **automated in evaluator**
- [x] Full-period Sharpe check — **automated in evaluator**
- [ ] Sharpe stable across 5 random seeds (std < 0.1) — Phase 2.7
- [ ] No single window drives >30% of total return (concentration check) — Phase 2.7
- [ ] Ljung-Box test on returns — Phase 2.7
- [ ] Compare raw vs Lo (2002) adjusted Sharpe — Phase 2.7

#### Robustness (weight: 30%)
- [x] Run backtest on 3 sub-periods: 2018-2020, 2020-2022, 2023-2025 — **automated in evaluator**
- [x] Sharpe positive in all 3 sub-periods (not just aggregate) — **automated in evaluator**
- [ ] Feature importance top-5 stable across sub-periods — Phase 2.7
- [x] Performance under 2x transaction costs (reality buffer) — **automated in evaluator**

#### Simplicity (weight: 15%)
- [x] Count active parameters vs baseline (penalize complexity) — **automated in evaluator**
- [ ] Each new parameter must contribute ≥ +0.05 Sharpe (ablation test) — Phase 2.7
- [ ] Reject if improvement < t-stat 3.0 (Harvey, Liu & Zhu threshold) — Phase 2.7

#### Reality Gap (weight: 15%)
- [x] Transaction costs ≥ 10 bps round-trip — **automated in evaluator (tests at 20 bps)**
- [ ] No execution at exact close price (add 5 bps slippage) — Phase 2.7
- [ ] Max position < 10% of portfolio — Phase 2.7
- [ ] Universe includes at least some mid-cap stocks (not just mega-cap) — Phase 2.7

### 2.5 Sprint Contracts (Experiment Contracts) ✅ IMPLEMENTED

Before each optimization cycle, the planner writes `handoff/contract.md` with hypothesis, current baseline, success criteria, and excluded params. This prevents the generator from optimizing the wrong thing.

**Example contract:**
```markdown
## Cycle 5 Contract
Hypothesis: Adding correlation-based position sizing will improve risk-adjusted returns.
Generator will: Implement pairwise correlation computation, modify size_position(), run 15 iterations.
Evaluator will verify:
  1. Sharpe improves by ≥ +0.05 vs baseline
  2. Max drawdown improves (less negative)
  3. No single sector > 40% of portfolio at any point
  4. Feature importance remains stable
  5. Performance holds under 2x transaction costs
Fail conditions: DSR < 0.95, Sharpe improvement < +0.03, drawdown worsens.
```

### 2.6 Context Management Strategy ✅ IMPLEMENTED

From the article: "Context resets — clearing the context window entirely and starting a fresh agent, combined with a structured handoff — addresses both [coherence loss and context anxiety]."

**Our approach:**
- Each optimization cycle is a **clean context** (optimizer restarts from handoff file)
- The handoff file contains: best params (JSON), experiment log (TSV), evaluator critique (markdown)
- No reliance on conversation history — everything is in files
- The `optimizer_best.json` already implements this for params; extend to include evaluator state

**Warm-start protocol:**
```
1. Read handoff/evaluator_critique.md → understand what failed last cycle
2. Read handoff/research_plan.md → understand what to try next
3. Read optimizer_best.json → current best params
4. Read quant_results.tsv → full experiment history
5. Execute research plan → run optimizer with new direction
6. Write handoff/experiment_results.md → results for evaluator
```

---

### 2.6.0 Operational Resilience — Zero Downtime (FIRST PRIORITY) ✅ CONDITIONAL PASS
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG
> **Completed:** 2026-03-28 23:10 | **Verdict:** CONDITIONAL PASS (8.4/10) — Section C (rate limits) deferred to pre-Phase 2.7

*Ford must always be available. No excuses. If something breaks, fix it automatically and notify Peder on Slack.*

**Principle:** The system self-heals. Ford stays online. Peder gets notified of every issue and every recovery.

**A. Gateway & OpenClaw Self-Healing**
- [x] Watchdog cron: check `openclaw gateway status` every 5 minutes
  - If gateway down → `openclaw gateway restart` → notify Slack "⚠️ Gateway crashed, auto-restarted"
  - If restart fails → notify Slack "🔴 Gateway restart FAILED — manual intervention needed"
  - Log all crashes with timestamp to `memory/incidents.md`
- [x] macOS LaunchAgent for OpenClaw gateway (auto-start on boot, auto-restart on crash)
  - Official `ai.openclaw.gateway.plist` already running (removed redundant custom plist)
  - `KeepAlive.SuccessfulExit: false`, `RunAtLoad: true`
- [ ] Heartbeat self-check: if heartbeat hasn't run in >60 min, something is broken
  - Cron job (system-level, not OpenClaw) checks heartbeat state file timestamp
  - If stale → restart gateway → notify Slack

**B. Slack Availability — Non-Negotiable**
- [ ] Slack bot health check in every heartbeat (first thing checked)
  - Post a silent health ping to a test channel or check Slack API connectivity
  - If Slack unreachable → log to `memory/incidents.md`, retry with exponential backoff
- [ ] Fallback notification path: if Slack is down, send critical alerts via iMessage to Peder
- [ ] Morning cron (7am) MUST post to Slack — if it doesn't, the system-level watchdog catches it

**C. API Rate Limit Handling**
- [ ] Global rate limit handler in backend:
  - Detect 429/Retry-After from all external APIs (BQ, Alpha Vantage, FRED, Yahoo Finance, Vertex AI)
  - Exponential backoff with jitter (1s → 2s → 4s → 8s → max 60s)
  - After 3 consecutive 429s on same API → notify Slack: "⚠️ [API name] rate limited, backing off"
  - After 10 consecutive failures → pause that data source, notify Slack: "🔴 [API name] suspended"
  - Track rate limit incidents in `memory/incidents.md`
- [ ] LLM cost/rate protection:
  - If Claude API returns 429 → back off, notify Slack with estimated wait time
  - If BQ billing exceeds daily threshold → pause non-critical queries, notify Slack
  - Never silently fail — always notify

**D. System Health Monitoring**
- [ ] Disk space check in heartbeat (warn at <10GB free, alert at <5GB)
- [ ] Memory usage check (warn if Python processes >4GB RSS)
- [ ] Port check: backend (8000) and frontend (3000) — restart if down during active work
- [ ] Git status check: uncommitted changes >24h old → notify Slack to review
- [ ] Process zombie detection: check for orphaned Python workers, kill if found

**E. Incident Log**
- All incidents logged to `memory/incidents.md` with:
  ```
  ## YYYY-MM-DD HH:MM — [SEVERITY] [Component]
  **Issue:** What broke
  **Detection:** How it was caught (heartbeat/watchdog/cron)
  **Action:** What was done automatically
  **Recovery:** When it came back online
  **Root cause:** Why it happened (if known)
  **Prevention:** What we changed to prevent recurrence
  ```

**G. OpenClaw Infrastructure Settings (Control UI → Infrastructure)**

Configure all infrastructure panels:

- [ ] **ACP (Agent Communication Protocol):**
  - Set ACP Default Agent → `codex` (for coding task delegation)
  - Add ACP Allowed Agents: `codex`, `claude-code` (so Ford can spawn coding sub-agents)
- [ ] **MCP (Model Context Protocol):**
  - Configure pyfinAgent MCP servers when Phase 3 starts:
    - `pyfinagent-data` → localhost:8000/mcp/data
    - `pyfinagent-backtest` → localhost:8000/mcp/backtest
    - `pyfinagent-signals` → localhost:8000/mcp/signals (Phase 4)
  - For now: document planned MCP config, implement when servers are built
- [ ] **Gateway:**
  - Verify `bind: loopback` (secure — local only)
  - Verify auth token rotation schedule
- [ ] **Browser:**
  - Enable if needed for deep research (web_fetch may be sufficient)
- [ ] **NodeHost:**
  - Verify node configuration for Mac Mini
- [ ] **Media:**
  - Review media handling settings

**H. OpenClaw Automation Settings (Control UI → Automation)**

Configure all automation panels for operational resilience:

- [ ] **Approvals:**
  - Enable Forward Exec Approvals → ON
  - Approval Agent Filter: `["main"]`
  - Approval Forwarding Mode: `both` (session + targets for redundancy)
  - Approval Forwarding Targets: Add Slack #ford-approvals so Peder can approve commands remotely
  - Approval Session Filter: `["webchat", "slack"]`
- [ ] **Commands:**
  - `commands.bash`: enable (for quick host checks via Slack)
  - `commands.config`: enable (for remote config inspection)
  - `commands.debug`: enable (for runtime troubleshooting)
- [ ] **Hooks:**
  - All 4 bundled hooks already enabled ✅ (boot-md, bootstrap-extra-files, command-logger, session-memory)
- [ ] **Bindings:**
  - Review channel bindings for Slack ↔ main agent routing
- [ ] **Cron:**
  - Morning report (7am) ✅
  - Evening summary (6pm) ✅
  - Gateway watchdog (5min) ✅
  - Add: Service health check (15min) — lighter than watchdog, just port checks
- [ ] **Plugins:**
  - Review available plugins for any useful additions

**I. OpenClaw Config & Communications (Control UI → Settings + Communications)**

Review and optimize all OpenClaw configuration sections for production readiness:

- [ ] **Settings (Raw Config — openclaw.json):**
  - [ ] `agents.defaults.model` — verify Sonnet as default, Opus for webchat, Haiku for heartbeats/cron
  - [ ] `agents.defaults.heartbeat` — verify 30m interval, 07:00-23:00 Oslo active hours
  - [ ] `agents.defaults.maxConcurrent` — verify 6 (agents) + 12 (subagents) sufficient
  - [ ] `agents.defaults.compaction.mode` — verify `safeguard` is correct for our use case
  - [ ] `session.dmScope` — verify `per-channel-peer` (isolate webchat vs Slack sessions)
  - [ ] `session.reset.mode` — currently `idle` with 525600 min (1 year) — review if appropriate
  - [ ] `session.maintenance` — verify pruneAfter, maxEntries, rotateBytes settings
  - [ ] `messages.ackReactionScope` — currently `group-mentions`, review if we want `all` for Slack
  - [ ] `messages.queue.mode` — review if `collect` or `steer` is better for our workflow
  - [ ] `messages.inbound.debounceMs` — add Slack-specific debounce (1500ms recommended)
  - [ ] `commands.native` — `auto` is fine, verify all needed commands available
  - [ ] `commands.ownerDisplay` — currently `raw`, review if appropriate
  - [ ] `tools.profile` — currently `coding`, verify this gives us all needed tools
  - [ ] `tools.web.search` — verify Gemini provider + API key working

- [ ] **Environment:**
  - [ ] Verify all environment variables set correctly
  - [ ] Check PATH includes Python venv, node, git

- [ ] **Authentication:**
  - [ ] Verify Anthropic API key valid and not expiring soon
  - [ ] Verify Gemini/Google API key for web search
  - [ ] Review auth profile configuration

- [ ] **Meta:**
  - [ ] Review version tracking, update check settings

- [ ] **Logging:**
  - [ ] Set appropriate log levels (info for production, debug for troubleshooting)
  - [ ] Verify log rotation configured

- [ ] **Diagnostics:**
  - [ ] Review diagnostic flags
  - [ ] Ensure `openclaw doctor` passes

- [ ] **Secrets:**
  - [ ] Verify all 4 redacted secrets are properly stored
  - [ ] Ensure no secrets in plaintext logs

- [ ] **Communications — Channels:**
  - [ ] **Slack channel config:**
    - [ ] Verify `mode: socket` with valid botToken + appToken
    - [ ] `allowBots: true` — needed for our Slack bot integration
    - [ ] `groupPolicy: allowlist` with `C0ANTGNNK8D` (#ford-approvals) enabled
    - [ ] `requireMention: false` for #ford-approvals (Ford responds to all messages)
    - [ ] Review `streaming: partial` + `nativeStreaming: true` settings
    - [ ] Consider adding `typingReaction: "hourglass_flowing_sand"` for UX
    - [ ] Consider adding `ackReaction` for message acknowledgment
    - [ ] Review `thread.historyScope` and `thread.inheritParent` settings
    - [ ] Consider enabling `actions.reactions` for lightweight acknowledgment
    - [ ] Consider enabling `actions.pins` for important messages
  - [ ] **Webchat config:**
    - [ ] Verify webchat enabled and accessible
    - [ ] Review heartbeat/reconnect settings

- [ ] **Communications — Messages:**
  - [ ] Review `ackReactionScope` — should Ford react to messages it's processing?
  - [ ] Review `responsePrefix` — currently none, consider adding for Slack clarity
  - [ ] Configure `queue.mode` for optimal message handling (collect vs steer)
  - [ ] Set `inbound.debounceMs` per channel (Slack: 1500ms, webchat: 2000ms)

- [ ] **Communications — Broadcast:**
  - [ ] Review if broadcast groups needed (e.g., broadcast to Slack + iMessage simultaneously)

- [ ] **Communications — Talk/Audio:**
  - [ ] Review if voice/audio features needed (probably not for Phase 2.6.0)

**Harness deliverables for this step:**
- `handoff/contract.md` — success criteria: gateway uptime >99.5%, Slack always reachable, all incidents auto-logged, all config settings reviewed and optimized
- `handoff/experiment_results.md` — what was built, configs deployed, settings changed
- `handoff/evaluator_critique.md` — test: crash gateway, verify auto-restart + Slack notification; verify all config changes took effect
- RESEARCH.md updated with: system reliability patterns, launchd best practices

**F. Recovery Playbook (automated)**
| Component | Detection | Auto-Fix | Notify |
|-----------|-----------|----------|--------|
| Gateway down | 5min watchdog cron | `openclaw gateway restart` | Slack ⚠️ |
| Gateway restart fails | watchdog retry (3x) | Log + escalate | Slack 🔴 + iMessage |
| Backend (8000) down | Heartbeat check | `cd pyfinagent && source .venv/bin/activate && uvicorn backend.main:app --port 8000 &` | Slack ⚠️ |
| Frontend (3000) down | Heartbeat check | `cd pyfinagent/frontend && npm run dev &` | Slack ⚠️ |
| Slack unreachable | Heartbeat check | Retry with backoff, fallback to iMessage | iMessage 🔴 |
| API rate limited | Request interceptor | Exponential backoff | Slack ⚠️ (after 3x) |
| Disk space low | Heartbeat check | Clean caches, notify | Slack ⚠️ |
| Harness stuck | Heartbeat check (>2h) | Kill process, log state | Slack ⚠️ |

### 2.6.1 Backtest Page — Harness Dashboard ✅ PASS
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG
> **Completed:** 2026-03-28 23:20 | **Verdict:** PASS (9/10)

*Expose all autoresearch experiments and harness cycle results on the backtest page so Peder can see everything.*

**Current state:** The backtest page already shows:
- ✅ Run leaderboard (top 10 by Sharpe)
- ✅ Sharpe comparison bar chart
- ✅ Optimizer progress chart (Karpathy-style)
- ✅ Experiment log table (param changed, before/after, DSR, keep/discard)
- ✅ Run selector with parent/child grouping
- ✅ Equity curve, feature importance (MDA), trade list

**Missing — needs to be built:**
- [ ] **Harness Cycle View:** Show RESEARCH→PLAN→GENERATE→EVALUATE→DECIDE→LOG cycles from `handoff/harness_log.md`
  - Each cycle: hypothesis, generator results, evaluator verdict, scores (4 criteria), decision
  - Timeline/accordion view of all cycles with pass/fail status
- [ ] **Evaluator Critique Panel:** Render `handoff/evaluator_critique.md` as formatted HTML
  - Sub-period results table (A/B/C + 2× costs)
  - Grading cards (Statistical Validity, Robustness, Simplicity, Reality Gap) with scores
  - Root cause analysis sections
  - Key lessons highlighted
- [ ] **Experiment Comparison:** Side-by-side parameter diff between any two experiments
  - Show exactly which params changed and their impact
  - Highlight the "kept" vs "discarded" experiments visually
- [ ] **Harness API endpoints:**
  - `GET /api/backtest/harness/log` — parsed harness_log.md as JSON
  - `GET /api/backtest/harness/critique` — latest evaluator_critique.md as JSON
  - `GET /api/backtest/harness/contract` — current contract.md as JSON
  - `GET /api/backtest/harness/research-plan` — current research_plan.md as JSON
  - `GET /api/backtest/harness/validation` — validation_results.json + subperiod_validation_results.json
- [ ] **New "Harness" tab** on backtest page (alongside Overview, Results, Equity, Features, Optimizer)

### 2.6.2 Budget Intelligence Dashboard & Cost Autoresearch ✅ PASS (Phase A)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG
> **Completed:** 2026-03-28 23:30 | **Verdict:** PASS (Phase A: Visibility). Phase B (Cost Autoresearch) deferred.
> **Cost cleanup:** 2026-03-28 23:48 — Deleted Redis ($76) + VPC connectors ($19). GCP $176→$5/mo.
> **Note:** Redis/VPC may be re-added when scaling to cloud deployment if economically justified.

*Give Peder full visibility into cash flow and an automated system to optimize costs down.*

**Current state:**
- ✅ `backend/agents/cost_tracker.py` — tracks per-agent LLM token costs per analysis run
- ✅ `CostDashboard` component — shows per-analysis LLM costs on analyze page
- ✅ BQ `INFORMATION_SCHEMA` available for GCP cost queries
- ✅ Budget overview card on backtest page (static numbers)

**Phase A: Budget Visibility (frontend + API)**

- [ ] **Cash Flow Graph** — dual-axis line chart showing:
  - **Cash Out (actual):** Monthly costs from BQ billing export + Claude Max subscription ($200/mo) + any other fixed costs
  - **Cash In (actual):** Any income/funding (manual input or BQ table)
  - **Cash Flow (forecast):** Projected burn rate extrapolated from trailing 3-month average, with confidence band
  - **Break-even line:** When cash hits zero at current burn rate
  - Time range selector (3mo, 6mo, 1yr, custom)
- [ ] **Cost Breakdown** — pie/bar chart by category:
  - Google Cloud (BQ queries, storage, Cloud Functions) — from BQ billing export
  - Claude Max subscription ($200/mo fixed)
  - LLM API costs (per-model breakdown from `cost_tracker.py`)
  - Data APIs (Alpha Vantage, FRED — currently $0)
  - Infrastructure (Mac Mini amortized)
- [ ] **Monthly Trend** — stacked bar chart showing cost categories over time
- [ ] **Budget Alerts** — configurable thresholds:
  - Warn at 80% of monthly budget
  - Alert at 100% (Slack notification)
  - Auto-pause non-critical services at 120%
- [ ] **API endpoints:**
  - `GET /api/budget/summary` — current month costs, forecast, budget status
  - `GET /api/budget/history` — monthly cost breakdown for cash flow graph
  - `GET /api/budget/forecast` — projected costs based on usage trends
  - `POST /api/budget/target` — set monthly budget target (persisted to settings)

**Phase B: Cost Autoresearch (Karpathy-style cost optimization)**

*Apply the same autoresearch pattern we use for Sharpe optimization — but for monthly cost.*

- [ ] **Cost Optimizer loop** — same keep/discard pattern as `quant_optimizer.py`:
  ```
  1. Measure current monthly cost (baseline)
  2. Propose a cost reduction (e.g., reduce BQ query frequency, cache more aggressively, batch API calls)
  3. Implement the change
  4. Measure new monthly cost (or projected cost)
  5. Verify quality didn't degrade (Sharpe still ≥ threshold, analysis quality maintained)
  6. KEEP if cost reduced AND quality maintained, DISCARD otherwise
  ```
- [ ] **Budget target adjustment** — user sets desired monthly budget on the dashboard
  - If budget reduced → triggers cost autoresearch to find savings
  - If budget increased → unlocks previously disabled features (more GCF calls, higher-tier models, etc.)
- [ ] **Cost reduction strategies (autoresearch explores these):**
  - [ ] BQ query batching (consolidate multiple small queries into fewer large ones)
  - [ ] Aggressive caching TTLs (trade freshness for cost — prices: 15min→1hr, fundamentals: 1day→1week)
  - [ ] Model downgrade for non-critical agents (use Flash instead of Pro for sentiment, use Haiku for formatting)
  - [ ] Cloud Function call frequency (daily→weekly for non-time-sensitive data)
  - [ ] Feature pruning (drop MDA-bottom features → fewer computations)
  - [ ] Batch analysis runs (analyze 5 tickers at once instead of 1-by-1)
  - [ ] Local model fallback for simple tasks (phi-4 for formatting, summarization)
- [ ] **Quality gates** — cost optimizer cannot:
  - Reduce Sharpe below 90% of current best
  - Remove any evaluator-passing feature
  - Disable risk management components
  - Skip data freshness below configurable minimum
- [ ] **Experiment log** — `experiments/cost_results.tsv` tracking:
  - Change made, cost before, cost after, quality impact, keep/discard
- [ ] **Dashboard integration:**
  - "Optimize" button on budget dashboard → triggers cost autoresearch cycle
  - Shows proposed savings with projected impact
  - User approves/rejects each cost reduction before it's applied

**Fixed costs to track:**
| Item | Monthly | Source |
|------|---------|--------|
| Claude Max | $200 | Fixed subscription |
| Google Cloud | ~$10-25 | BQ billing export |
| Mac Mini (amortized) | ~$28 | $1,000 / 36 months |
| FRED / Alpha Vantage | $0 | Free tiers |
| GitHub Copilot Pro | $0 | Included for students? |

### 2.7 Paper Trading — Real-World Validation (Start Early April)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Trade with fake money in the live market BEFORE going live in May. This is the ultimate test — backtest says Sharpe 1.17, but does it hold in real time?*

**⚠️ START THIS AS EARLY AS POSSIBLE.** Every day of paper trading before May is data we can't get any other way. Backtests can be fooled. Live markets can't.

**Why this matters:**
- Backtests are always better than reality (hindsight bias, execution assumptions, data quality)
- Paper trading catches: slippage, partial fills, data delays, stale quotes, market impact
- 2+ weeks minimum required before go-live checklist can pass
- If paper Sharpe < 0.7× backtest Sharpe → we have a problem to fix before May

**Infrastructure (already built — 60% done):**
- [x] `backend/services/paper_trader.py` (361 lines) — full paper trading engine, BQ persistence, NAV tracking
- [x] `backend/services/autonomous_loop.py` (328 lines) — daily cycle: Screen → Analyze → Decide → Trade → Snapshot → Learn
- [x] `backend/services/portfolio_manager.py` (230 lines) — sell-first-then-buy, Risk Judge sizing
- [x] `backend/services/outcome_tracker.py` (184 lines) — evaluates past recs vs actual prices, learns
- [x] `backend/api/paper_trading.py` (284 lines) — API endpoints + APScheduler integration
- [x] `backend/tools/screener.py` (213 lines) — quant-only universe screening (zero LLM cost)

**Remaining to activate:**
- [ ] **Validate paper trading engine** — run a test day, verify BQ writes, NAV tracking, trade logging
- [ ] **Load current best params** (Sharpe 1.17) into paper trader config
- [ ] **Activate autonomous daily cycle** via APScheduler cron (run at market open 9:30 ET)
- [ ] **Paper Trading Dashboard** — new page or tab showing:
  - Current paper portfolio (positions, P&L, NAV)
  - Trade history (every simulated buy/sell with timestamp, price, reason)
  - Performance vs backtest expectations (daily Sharpe comparison)
  - Performance vs SPY (live benchmark)
  - Drawdown tracking (kill switch at -15%)
- [ ] **Daily Slack report** — morning: "Paper portfolio: $X NAV, Y positions, Sharpe Z (vs backtest 1.17)"
- [ ] **Divergence alerts:**
  - Paper Sharpe < 0.7× backtest → Slack 🔴 "Paper trading underperforming — investigate"
  - Single-day loss > 3% → Slack ⚠️ "Large paper loss: -X%"
  - Drawdown > 10% → Slack 🔴 "Paper drawdown approaching kill switch"
  - Position concentration > 20% single stock → Slack ⚠️
- [ ] **Weekly evaluation report** — compare paper results to backtest:
  | Metric | Backtest | Paper | Delta | Status |
  |--------|----------|-------|-------|--------|
  | Sharpe | 1.17 | ? | ? | ✅/❌ |
  | Return (annualized) | ~11% | ? | ? | ✅/❌ |
  | Max DD | -12% | ? | ? | ✅/❌ |
  | Hit Rate | ~55% | ? | ? | ✅/❌ |
  | Avg Holding Days | ~45 | ? | ? | ✅/❌ |
- [ ] **Go-live gate:** Paper trading must show:
  - Sharpe ≥ 0.7× backtest (≥ 0.82) over 2+ weeks
  - No kill switch triggered (drawdown < 15%)
  - Signal generation reliable (no missed days)
  - Execution assumptions validated (fills, timing, costs)

**Timeline:**
- Early April: Activate paper trading with current params
- Mid April: First weekly evaluation
- Late April: Go/no-go decision for May launch
- May: Go live (if paper trading passes)

### 2.8 Harness Hardening & Advanced Evaluator (After Paper Trading Starts)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Enhance the automated harness with deeper statistical tests and generator capabilities.*

- [ ] **Seed stability test:** Run best params with 5 random seeds, check Sharpe std < 0.1
- [ ] **Concentration check:** No single window drives >30% of total return
- [ ] **Ljung-Box autocorrelation test** on returns
- [ ] **Lo (2002) adjusted Sharpe** comparison
- [ ] **Feature importance stability** across sub-periods
- [ ] **Ablation studies:** Remove one Phase 1 improvement at a time, measure drop
- [ ] **Multi-param proposals:** Coordinate param groups (e.g., tp_pct + sl_pct together)
- [ ] **Strategy switching:** Generator can propose trying different strategies
- [ ] **Slippage modeling:** 5 bps execution slippage on top of transaction costs
- [ ] **Position concentration limits:** Max position < 10% in evaluator checks

### 2.9 Multi-Market Abstractions (Lightweight)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Prepare data layer for future market expansion (CA, EU, NO, KR) without building actual multi-market support. ~4 hours of work, zero risk to May launch.*

- [ ] Add `market` STRING column to BQ tables (prices, fundamentals, macro) — default `'US'` for existing data
- [ ] Add `currency` STRING column to prices table — default `'USD'`
- [ ] Namespace tickers: `{market}:{ticker}` format in universe table
- [ ] Abstract trading calendar: `exchange_calendars` library (supports OSE, KRX, TSX, XETRA)
- [ ] Add `market` filter to MCP server tools (default `'US'`)
- [ ] Add `base_currency` param to portfolio/returns calculations (passthrough for now, always `'USD'`)
- [ ] Document market expansion checklist in Phase 5

**Target markets (future):**

| Market | Exchange | Currency | Calendar | Notes |
|--------|----------|----------|----------|-------|
| US | NYSE/NASDAQ | USD | XNYS | Current (default) |
| NO | Oslo Børs (OSE) | NOK | XOSL | Home market, Peder knows it well |
| CA | TSX | CAD | XTSE | Similar market structure to US |
| EU | XETRA (DE), Euronext | EUR | XETR/XPAR | Large cap European equities |
| KR | KRX (KOSPI/KOSDAQ) | KRW | XKRX | High retail activity, different dynamics |

**Ticker namespacing convention:**
```
US:AAPL    NO:EQNR    CA:RY    DE:SAP    KR:005930
```

### 2.10 Karpathy Autoresearch Integration
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*The Generator (QuantStrategyOptimizer) already follows Karpathy's autoresearch pattern for zero-cost param optimization. The harness adds the missing evaluation and planning layers. If the harness proves beneficial on pyfinAgent, apply the same three-agent pattern to the upstream [autoresearch](https://github.com/karpathy/autoresearch) optimizer — wrapping its research loop with independent evaluation and heuristic planning.*

---

## Phase 3: LLM-Guided Research + MCP Integration (Weeks 4-5)
### ⚠️ GATE: REQUIRES PEDER'S EXPLICIT APPROVAL BEFORE STARTING

*With the harness in place, we can now add LLM reasoning to the Planner and Evaluator agents — and give them direct tool access to pyfinAgent via MCP.*

### 3.0 MCP Server Architecture
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

**Key insight:** Instead of dumping experiment data as text into prompts, we expose pyfinAgent's capabilities as [MCP servers](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector). Claude's MCP connector (beta: `mcp-client-2025-11-20`) lets the LLM Planner and Evaluator **directly call tools** — query experiments, trigger backtests, read results — through the Messages API. No separate MCP client needed.

**Why this matters for us:**
- The heuristic planner (Phase 2) reads files. The LLM planner (Phase 3) can **interact** with the system.
- The evaluator can run additional spot-checks on demand, not just the 5 predefined backtests.
- Signal generation (Phase 4) gets a clean tool interface instead of fragile script wiring.

#### MCP Servers to Build

We expose pyfinAgent as **three remote MCP servers** (FastAPI + Streamable HTTP transport):

```
┌──────────────────────────────────────────────────────────────────────┐
│  Claude Messages API (with MCP connector beta)                        │
│                                                                        │
│  mcp_servers: [                                                        │
│    { name: "pyfinagent-data",     url: "https://..." },               │
│    { name: "pyfinagent-backtest", url: "https://..." },               │
│    { name: "pyfinagent-signals",  url: "https://..." }                │
│  ]                                                                     │
│                                                                        │
│  tools: [                                                              │
│    { type: "mcp_toolset", mcp_server_name: "pyfinagent-data" },       │
│    { type: "mcp_toolset", mcp_server_name: "pyfinagent-backtest",     │
│      default_config: { enabled: false },                               │
│      configs: { run_backtest: { enabled: true },                       │
│                 get_experiments: { enabled: true } } },                 │
│    { type: "mcp_toolset", mcp_server_name: "pyfinagent-signals",      │
│      default_config: { enabled: false },                               │
│      configs: { generate_signals: { enabled: true },                   │
│                 validate_signal: { enabled: true } } }                  │
│  ]                                                                     │
└───────────────┬──────────────────┬──────────────────┬────────────────┘
                │                  │                  │
                ▼                  ▼                  ▼
┌───────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ pyfinagent-data   │ │ pyfinagent-backtest│ │ pyfinagent-signals │
│                   │ │                    │ │                    │
│ Tools:            │ │ Tools:             │ │ Tools:             │
│ • query_prices    │ │ • run_backtest     │ │ • generate_signals │
│ • query_fundmntls │ │ • get_experiments  │ │ • validate_signal  │
│ • query_macro     │ │ • get_best_params  │ │ • publish_signal   │
│ • get_universe    │ │ • run_subperiod    │ │ • get_portfolio    │
│ • get_features    │ │ • compare_params   │ │ • risk_check       │
│                   │ │ • run_ablation     │ │                    │
│ Reads: BigQuery   │ │ Reads: backtest    │ │ Reads: latest      │
│        + cache    │ │        engine      │ │        model +     │
│                   │ │        + optimizer  │ │        market data │
└───────────────────┘ └────────────────────┘ └────────────────────┘
```

**Implementation:** Each MCP server is a lightweight FastAPI app using `mcp` Python SDK with Streamable HTTP transport, deployed alongside our existing backend. Authentication via OAuth token or shared secret.

**Security:** Allowlist pattern — only explicitly enabled tools per request. The LLM Planner gets read-only data + experiments. The Evaluator gets backtest tools. Signal generation is Phase 4 only.

#### MCP Server Implementation Plan

| Server | Tools | Phase | Cost |
|--------|-------|-------|------|
| **pyfinagent-data** | `query_prices`, `query_fundamentals`, `query_macro`, `get_universe`, `get_features` | 3.0 | $0 (just wraps existing BQ cache) |
| **pyfinagent-backtest** | `run_backtest`, `get_experiments`, `get_best_params`, `run_subperiod`, `compare_params`, `run_ablation` | 3.0 | $0 (wraps existing engine) |
| **pyfinagent-signals** | `generate_signals`, `validate_signal`, `publish_signal`, `get_portfolio`, `risk_check` | 4.1 | $0 (wraps existing pipeline) |

### 3.1 LLM-as-Planner (with MCP tool access)

**The article's lesson applied:** "Taking inspiration from GANs, I designed a multi-agent structure with a generator and evaluator agent."

**Upgrade from heuristic planner:** The Phase 2 planner reads TSV files and applies rules. The Phase 3 planner is Claude with MCP tools — it can query experiment history, compare parameter sets, and propose research directions based on actual data analysis.

Our adaptation for budget constraints:
- Run 15-20 experiments (free CPU) → call Claude with MCP tools once
- Claude acts as Research Planner: queries experiments via `pyfinagent-backtest` tools, analyzes patterns, proposes next research direction
- Max 3-5 LLM reasoning calls per optimization cycle (~$2-5/cycle)
- LLM can propose code changes but Ford reviews before running

**Planner API call structure:**
```python
response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",  # Budget: Sonnet, not Opus
    max_tokens=2000,
    mcp_servers=[
        {"type": "url", "url": MCP_BACKTEST_URL, "name": "pyfinagent-backtest",
         "authorization_token": AUTH_TOKEN},
        {"type": "url", "url": MCP_DATA_URL, "name": "pyfinagent-data",
         "authorization_token": AUTH_TOKEN},
    ],
    tools=[
        {"type": "mcp_toolset", "mcp_server_name": "pyfinagent-backtest",
         "default_config": {"enabled": False},
         "configs": {"get_experiments": {"enabled": True},
                     "get_best_params": {"enabled": True},
                     "compare_params": {"enabled": True}}},
        {"type": "mcp_toolset", "mcp_server_name": "pyfinagent-data",
         "default_config": {"enabled": False},
         "configs": {"get_features": {"enabled": True}}},
    ],
    system=PLANNER_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": planner_context}],
    betas=["mcp-client-2025-11-20"],
)
```

**Planner system prompt:**
```
You are a quantitative researcher with direct access to experiment data.
Use your tools to query the experiment log, compare parameter sets, and analyze patterns.

Your job:
1. Call get_experiments to review recent results
2. Call compare_params to identify which parameters have the most impact
3. Hypothesize what structural change would improve Sharpe
4. Propose specific, testable changes (params, features, or strategy logic)
5. Estimate expected impact and risk of overfitting
6. Define success criteria for the evaluator

Do NOT propose changes that:
- Add complexity without justification (t-stat < 3.0)
- Ignore the evaluator's previous critique
- Violate the budget constraint (no external API calls)
```

### 3.2 LLM-as-Evaluator (with MCP backtest access)

**The article's lesson:** "Out of the box, Claude is a poor QA agent. I watched it identify legitimate issues, then talk itself into deciding they weren't a big deal."

**Upgrade from automated evaluator:** The Phase 2 evaluator runs 5 predefined backtests. The Phase 3 evaluator is Claude with `pyfinagent-backtest` tools — it can run **additional spot-checks** it decides are needed based on what it sees. "This Sharpe jump looks suspicious — let me run a sub-period test on just 2022 to check."

Our calibration approach:
- Explicitly prompt for skepticism: "Your job is to find problems, not praise."
- Give it `run_subperiod` and `run_ablation` tools so it can investigate suspicions
- Provide few-shot examples of overfitting patterns (high Sharpe but fragile)
- Grade against concrete criteria (not "is this good?" but "does this pass each specific test?")
- Hard thresholds that can't be argued away

**Evaluator API call structure:**
```python
response = client.beta.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=3000,
    mcp_servers=[
        {"type": "url", "url": MCP_BACKTEST_URL, "name": "pyfinagent-backtest",
         "authorization_token": AUTH_TOKEN},
    ],
    tools=[
        {"type": "mcp_toolset", "mcp_server_name": "pyfinagent-backtest",
         "default_config": {"enabled": False},
         "configs": {"run_subperiod": {"enabled": True},
                     "run_ablation": {"enabled": True},
                     "get_experiments": {"enabled": True},
                     "compare_params": {"enabled": True}}},
    ],
    system=EVALUATOR_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": evaluator_context}],
    betas=["mcp-client-2025-11-20"],
)
```

**Evaluator system prompt:**
```
You are a skeptical quant reviewer. Your reputation depends on catching overfitting.
You have direct access to the backtest engine — use it to verify claims.

If something looks suspicious:
- Call run_subperiod to test specific date ranges
- Call run_ablation to check if a parameter actually contributes
- Call compare_params to see what changed

Grade each criterion 1-10 with specific evidence:
1. Statistical Validity: [DSR, seed stability, window concentration]
2. Robustness: [sub-period performance, regime sensitivity]
3. Simplicity: [parameter count, marginal contribution per param]
4. Reality Gap: [transaction costs, execution assumptions]

A score below 6 on ANY criterion is a FAIL.
If the improvement is real, say so. If it smells like overfitting, say that.
Do not be generous. The cost of approving a bad strategy is losing real money.
```

### 3.3 Regime Detection (zero LLM cost)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

- [ ] HMM-based regime detector (2-3 states from returns + volatility)
- [ ] Per-regime parameter optimization
- [ ] Rolling re-optimization via cron

### 3.4 Agent Skill Optimization — ⚠️ PARTIALLY BUILT
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Codebase audit finding: `backend/agents/skill_optimizer.py` (864 lines) already implements Karpathy-style autoresearch for prompt optimization. `backend/agents/meta_coordinator.py` (306 lines) already sequences QuantOpt → SkillOpt → PerfOpt with MDA→Agent bridge.*

**Already implemented:**
- [x] `SkillOptimizer` class — modifies `## Prompt Template` in skills.md, measures composite score, keep/discard
- [x] `MetaCoordinator` — cross-loop sequencing, bridges MDA features → agent targeting
- [x] `PerfOptimizer` (`backend/services/perf_optimizer.py`) — autoresearch loop for API cache TTL tuning
- [x] Skills stored in `backend/agents/skills/` as .md files loaded by `backend/config/prompts.py`

**Remaining work:**
- [ ] Run SkillOpt on highest-impact agents (Synthesis, Moderator, Risk Judge) — needs LLM budget approval
- [ ] Validate MDA → Agent bridge works end-to-end with harness output
- [ ] Wire SkillOpt results into harness evaluator criteria

### 3.5 Enrichment MCP Server
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Wraps all 16 external data APIs behind a single MCP interface for LLM access.*

| Tool | Source | Current Implementation |
|------|--------|----------------------|
| `search_news` | Alpha Vantage | `backend/tools/social_sentiment.py` |
| `get_sec_filings` | SEC EDGAR | `backend/tools/sec_insider.py` |
| `get_analyst_estimates` | Alpha Vantage | `backend/tools/alphavantage.py` |
| `get_insider_trades` | SEC EDGAR | `backend/tools/sec_insider.py` |
| `get_earnings_calendar` | Yahoo Finance | `backend/tools/earnings_tone.py` |
| `get_sentiment` | NLP + social | `backend/tools/nlp_sentiment.py` + `social_sentiment.py` |
| `get_macro_indicators` | FRED | `backend/tools/fred_data.py` |
| `get_options_flow` | Yahoo Finance | `backend/tools/options_flow.py` |
| `get_anomaly_scan` | Statistical | `backend/tools/anomaly_detector.py` |
| `get_sector_analysis` | Yahoo Finance | `backend/tools/sector_analysis.py` |
| `get_google_trends` | pytrends | `backend/tools/alt_data.py` |
| `get_monte_carlo` | GBM simulation | `backend/tools/monte_carlo.py` |
| `get_patent_data` | PatentsView | `backend/tools/patent_tracker.py` |
| `get_quant_signal` | ML model | `backend/tools/quant_model.py` |
| `get_financials` | Yahoo Finance | `backend/tools/yfinance_tool.py` |
| `get_price_history` | Yahoo Finance | `backend/tools/yfinance_tool.py` |

- [ ] Wrap all 16 tools as MCP server endpoints
- [ ] Central rate limiting (one place for 429/retry/backoff)
- [ ] Central caching (news 15min, SEC 24h, prices 5min)
- [ ] Central auth (API keys in one place)
- [ ] Cost tracking per API call through gateway

---

## Phase 4: Production Readiness (Week 6 — Late April)
*Get ready for real money. The evaluator agent becomes the live QA system. MCP signals server goes live.*

*⚠️ CODEBASE AUDIT FINDING: More of Phase 4 is already built than previously thought.*

### Existing Infrastructure (from codebase audit)

**Already implemented and working:**
- [x] `backend/services/paper_trader.py` (361 lines) — full paper trading engine with BQ persistence, NAV tracking, virtual trades
- [x] `backend/services/autonomous_loop.py` (328 lines) — daily cycle: Screen → Analyze → Decide → Trade → Snapshot → Learn
- [x] `backend/services/portfolio_manager.py` (230 lines) — sell-first-then-buy logic with Risk Judge position sizing
- [x] `backend/services/outcome_tracker.py` (184 lines) — evaluates past recs vs actual prices, generates LLM reflections, persists to BQ
- [x] `backend/api/paper_trading.py` (284 lines) — API endpoints + APScheduler integration
- [x] `backend/slack_bot/` — Socket mode bot with `/analyze`, `/portfolio`, `/report` commands
- [x] `backend/slack_bot/scheduler.py` (107 lines) — morning digest + proactive anomaly alerts
- [x] `backend/slack_bot/formatters.py` (227 lines) — Block Kit message formatting
- [x] `backend/tools/screener.py` (213 lines) — quant-only universe screening (zero LLM cost)
- [x] `backend/agents/orchestrator.py` (1477 lines) — full 15-step analysis pipeline with 20+ agents
- [x] `backend/api/signals.py` (202 lines) — on-demand enrichment data endpoints
- [x] `backend/services/perf_metrics.py` (224 lines) — canonical P&L and portfolio metrics

**What this means:** Phase 4 is ~60% built. The main gaps are MCP server wrappers, evaluator-gated signal publishing, and hardened risk limits.

### 4.1 Slack Signal Delivery (via MCP Signals Server)
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Daily trading signals delivered to Slack, validated by LLM evaluator before publishing.*

**4.1.1 Deploy pyfinagent-signals MCP Server**
- [ ] Create `backend/mcp/signals_server.py` wrapping existing `signals.py` + `autonomous_loop.py`
- [ ] Implement tools: `generate_signals`, `validate_signal`, `publish_signal`, `get_portfolio`, `risk_check`
- [ ] Add Streamable HTTP transport on port 8001 (separate from main backend 8000)
- [ ] Authentication: shared secret token (same as backtest MCP server)
- [ ] Register in OpenClaw MCP config

**4.1.2 Signal Generation Pipeline**
- [ ] Morning cron (9:00 AM ET / 3:00 PM Oslo): trigger `generate_signals`
  - Screener filters universe (zero LLM cost)
  - ML model predicts probabilities for top candidates
  - Generate signal list: ticker, direction (BUY/SELL/HOLD), confidence, position size
- [ ] LLM evaluator gate: Claude calls `validate_signal` for each signal before publishing
  - Checks: signal makes sense given recent news, no earnings within 2 days, not in excluded sectors
  - Rejects signals that fail validation (logged, not published)
- [ ] Wire existing `slack_bot/scheduler.py` morning digest → validated signals → Slack Block Kit message

**4.1.3 Signal Format (Slack Block Kit)**
- [x] Block Kit formatter already exists in `slack_bot/formatters.py` (227 lines)
- [ ] Enhance signal card:
  ```
  📊 DAILY SIGNALS — March 28, 2026
  ─────────────────────────────
  🟢 BUY  NVDA  | Conf: 78% | Size: 5.2% | Reason: Strong momentum + positive earnings tone
  🟢 BUY  MSFT  | Conf: 72% | Size: 4.8% | Reason: Quality score improving, sector rotation
  🔴 SELL TSLA  | Conf: 65% | Size: 3.1% | Reason: Turbulence rising, insider selling
  ─────────────────────────────
  Portfolio: 12 positions | NAV: $104,230 | Day: +0.3% | Sharpe (30d): 1.05
  ⚠️ Signals require manual review before trading
  ```
- [ ] Add "Approve" / "Reject" buttons on each signal (Slack interactive messages)
- [ ] Peder approves → signal logged as "approved" in BQ
- [ ] Track: approved vs rejected vs auto-filtered signals for evaluation

**4.1.4 Signal Quality Tracking**
- [ ] BQ table `signals_log`: timestamp, ticker, direction, confidence, approved/rejected, outcome (after N days)
- [ ] Weekly accuracy report: what % of approved signals were profitable?
- [ ] Per-enrichment-tool accuracy: which data sources actually add alpha to signals?
- [ ] Drop enrichment tools that don't improve signal accuracy after 30 days

### 4.2 Paper Trading Evaluation (LLM as Live QA) — ⚠️ Core Promoted to Phase 2.7
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Paper trading starts in Phase 2.7. By Phase 4, we have 2-4 weeks of data. This phase adds LLM-powered evaluation.*

**4.2.1 Daily LLM Portfolio Review**
- [ ] Evening cron (4:30 PM ET): Claude calls `get_portfolio` + `risk_check` via MCP
- [ ] LLM reviews: "Are we overweight in any sector? Any position violating risk limits? Any unexpected correlations?"
- [ ] Posts daily evaluation to Slack: portfolio summary, risk flags, recommendations
- [ ] If risk_check fails → auto-reduce positions to within limits next trading day

**4.2.2 Signal Accuracy Tracking**
- [ ] For each past signal: compare predicted direction vs actual price movement after holding period
- [ ] Track per-enrichment-tool accuracy: `{tool: accuracy_pct}` — drop tools below 50% accuracy
- [ ] Track per-sector accuracy: are we better at tech than healthcare?
- [ ] Weekly Slack report: "Signal accuracy this week: 62% (vs 55% backtest hit rate)"

**4.2.3 Paper vs Backtest Divergence Analysis**
- [ ] Daily: compare paper portfolio metrics to expected backtest metrics
- [ ] Track divergence over time: is paper consistently under/over-performing?
- [ ] Root cause analysis when divergence > 20%:
  - Execution timing (we assumed close price, actual execution at different time?)
  - Data staleness (live data different quality than historical?)
  - Universe drift (new stocks in universe not in backtest period?)
- [ ] If paper Sharpe < 0.7 × backtest → auto-trigger STOP → investigate → Slack 🔴

### 4.3 Risk Management
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Hardened risk limits that the LLM cannot override. The kill switch is code, not a suggestion.*

**4.3.1 Position Limits (hardcoded in risk_check MCP tool)**
- [x] Position sizing exists in `portfolio_manager.py` (Risk Judge + inverse-volatility)
- [ ] Max single position: 10% of portfolio (hardcoded, not configurable)
- [ ] Max sector exposure: 30% of portfolio
- [ ] Max correlated positions: no more than 3 stocks with correlation > 0.8
- [ ] Min cash reserve: always hold 5% cash

**4.3.2 Drawdown Protection**
- [ ] Trailing drawdown tracker: continuously monitors peak-to-trough
- [ ] Warning at -10% drawdown → Slack ⚠️ + reduce position sizes by 50%
- [ ] Kill switch at -15% drawdown → liquidate all positions to cash → Slack 🔴 → iMessage
- [ ] Kill switch is in `risk_check` code, NOT in LLM prompt — cannot be talked out of it
- [ ] Manual override requires Peder to change code and redeploy

**4.3.3 Event Calendar Integration**
- [ ] Pull earnings calendar from Yahoo Finance (already in `earnings_tone.py`)
- [ ] Pre-earnings rule: reduce position by 50% if earnings within 2 trading days
- [ ] FOMC meeting dates: reduce overall exposure by 25% day before
- [ ] Ex-dividend dates: factor into signal generation (avoid buying before ex-date for short-term trades)

**4.3.4 Stop-Loss Monitoring**
- [ ] Per-position stop-loss: exit if loss > 8% from entry
- [ ] Portfolio stop-loss: if 3+ positions hit stop-loss in one day → pause new entries for 24h
- [ ] Trailing stop: once position is +5%, trail stop at 3% below peak

### 4.4 Go-Live Checklist
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

*Every item must be checked and verified before May launch. No exceptions.*

**4.4.1 Statistical Validation**
- [ ] All evaluator criteria passing (statistical validity ≥ 6, robustness ≥ 6, simplicity ≥ 6, reality gap ≥ 6)
- [ ] DSR ≥ 0.95 on out-of-sample data
- [ ] Sharpe stable across 5 random seeds (std < 0.1)
- [ ] No single walk-forward window drives > 30% of total return

**4.4.2 Paper Trading Validation**
- [ ] Paper trading ran for ≥ 2 weeks (ideally 4 weeks)
- [ ] Paper Sharpe ≥ 0.82 (70% of backtest 1.17)
- [ ] Paper max drawdown < 15% (kill switch never triggered)
- [ ] No missed trading days (signal generation reliable)
- [ ] Paper vs backtest divergence < 20% on key metrics

**4.4.3 Infrastructure Validation**
- [ ] MCP servers deployed and authenticated (data + backtest + signals)
- [ ] Slack signals tested end-to-end (generate → validate → publish → Block Kit message)
- [ ] Gateway uptime > 99.5% over trailing 2 weeks
- [ ] All monitoring crons operational (watchdog, morning, evening)
- [ ] Incident log shows no unresolved P0 incidents

**4.4.4 Risk Management Validation**
- [ ] Kill switch tested: simulate -15% drawdown → verify auto-liquidation
- [ ] Position limits tested: submit oversized position → verify rejection
- [ ] Stop-loss tested: simulate loss > 8% → verify auto-exit
- [ ] Risk limits hardcoded in `risk_check` (not configurable without code change)

**4.4.5 Human Process Validation**
- [ ] Peder's daily review process defined: check Slack signals, approve/reject, review portfolio
- [ ] Escalation path defined: Ford alerts on Slack → iMessage → manual intervention
- [ ] Weekly review meeting: paper trading results, signal accuracy, plan progress
- [ ] Manual trading process: Peder executes approved signals through his broker
- [ ] Documentation: "How to trade pyfinAgent signals" guide for Peder

**4.4.6 Final Sign-Off**
- [ ] Peder explicitly approves go-live (not Ford's decision)
- [ ] Budget approved for Phase 4 operational costs (~$268-348/month)
- [ ] First week: extra monitoring — daily review call, immediate Slack alerts
- [ ] Rollback plan: if live Sharpe < 0.5 in first 2 weeks → stop signals, investigate

---

## Phase 5: Multi-Market Expansion (Post-Launch)
*After US is profitable and validated, expand to additional markets. Each market follows the same pattern: data pipeline → backtest validation → paper trade → go live.*

### 5.1 Market Expansion Framework
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG (per market)

*Each new market follows the same validated pipeline. No shortcuts — the harness proved its value on US.*

**5.1.1 Market Selection Criteria (choose next market based on):**
- [ ] Data availability and quality (free vs paid sources)
- [ ] Liquidity (can we realistically trade the signals?)
- [ ] Regulatory complexity (short-selling rules, settlement, reporting)
- [ ] Strategic value (diversification benefit, Peder's expertise)
- [ ] Cost to add (data feeds, FX conversion, timezone handling)

**Recommended order:** Norway → Canada → Europe → South Korea (risk-ordered)

**5.1.2 Per-Market Expansion Checklist**
Each market must complete ALL of these before going live:

**Data Pipeline (Week 1-2)**
- [ ] Data source identified and tested (Yahoo Finance, local exchange API, Bloomberg?)
- [ ] BQ tables populated with historical prices (≥5 years), fundamentals, macro for target market
- [ ] Data quality audit: missing data, corporate actions, survivorship bias check
- [ ] Trading calendar configured via `exchange_calendars` (handles holidays, early closes)
- [ ] FX rate feed operational (daily close rates, from ECB or Yahoo Finance)
- [ ] Universe defined: which tickers, min market cap, min liquidity thresholds

**Backtest Validation (Week 2-3)**
- [ ] Run full walk-forward backtest on target market data
- [ ] Sharpe > 0.5 (lower bar than US — new market, different dynamics)
- [ ] DSR ≥ 0.90 (slightly relaxed for new market with less data)
- [ ] All sub-period tests pass (positive Sharpe in each)
- [ ] Transaction costs calibrated for local market (differ significantly by exchange)
- [ ] Feature importance analysis: which features transfer from US, which are market-specific?

**Paper Trading (Week 3-5)**
- [ ] Paper trading activated for target market (same engine, market-specific config)
- [ ] Run ≥ 2 weeks paper trading
- [ ] Paper Sharpe ≥ 0.7× backtest
- [ ] No kill switch triggered
- [ ] Signal delivery adapted (timezone-aware scheduling)

**Go-Live (Week 5-6)**
- [ ] Regulatory constraints documented (short-sell rules, position limits, T+N settlement)
- [ ] Peder approves go-live for this market
- [ ] Separate Slack channel or signal prefix for market identification
- [ ] First week: extra monitoring

### 5.2 Market-Specific Research & Considerations
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

**5.2.1 Norway (OSE) — FIRST EXPANSION TARGET**
- ~200 liquid stocks, dominated by energy (Equinor, Aker BP), seafood (Mowi, SalMar), finance (DNB)
- Advantages: Peder knows the market, less algo competition, NOK/USD correlation with oil
- Risks: thin liquidity (wide bid-ask spreads = higher slippage), small universe limits diversification
- Data: Yahoo Finance covers OSE tickers, Oslo Børs has free delayed data
- Regulatory: Norwegian FSA (Finanstilsynet) — no special algo trading requirements for manual execution
- Transaction costs: typically 5-15 bps for liquid names, 20-50 bps for illiquid
- Trading hours: 09:00-16:20 CET (overlap with US pre-market)
- [ ] Deep research: OSE microstructure papers, Nordic factor models, liquidity studies

**5.2.2 Canada (TSX)**
- Similar market structure to US, ~1,500 liquid stocks
- Resource-heavy: mining (Barrick, Nutrien), energy (CNQ, Suncor), banks (RY, TD, BNS)
- Advantages: CAD/USD correlation means less FX risk, market hours overlap with NYSE
- Data: Yahoo Finance covers TSX fully, FRED has Canadian macro data
- Transaction costs: similar to US (5-10 bps for large caps)
- Trading hours: 09:30-16:00 ET (same as NYSE)
- [ ] Deep research: TSX factor models, CAD/USD hedging costs, Canadian sector dynamics

**5.2.3 Europe (XETRA/Euronext)**
- Large cap universe comparable to US: ~500 liquid stocks across Germany, France, Netherlands
- Sector mix: luxury (LVMH, Hermès), industrials (Siemens, Airbus), pharma (Novo Nordisk, ASML)
- MiFID II: extensive data availability requirements, best execution obligations
- FX: EUR/USD is liquid and cheap to hedge
- Transaction costs: 5-10 bps for major indices, higher for smaller exchanges
- Trading hours: 09:00-17:30 CET
- [ ] Deep research: European factor models, MiFID II compliance, cross-exchange arbitrage effects

**5.2.4 South Korea (KRX)**
- High retail participation creates momentum-driven dynamics
- KOSPI (~800 stocks) + KOSDAQ (~1,500 stocks) — different dynamics
- KRW volatility adds significant FX risk (unhedged)
- Short-selling restrictions: retail can't short, institutional limits
- T+2 settlement, different corporate governance culture
- Trading hours: 09:00-15:30 KST (US night → rolling 24h signal generation)
- [ ] Deep research: KRX microstructure, retail flow dynamics, chaebols governance, KRW hedging

### 5.3 Cross-Market Intelligence
> **Harness:** RESEARCH → PLAN → GENERATE → EVALUATE → DECIDE → LOG

**5.3.1 Correlation & Diversification**
- [ ] Cross-market correlation matrix: US ↔ NO ↔ CA ↔ EU ↔ KR
- [ ] Identify diversification benefits: which markets are most uncorrelated?
- [ ] Portfolio optimization across markets (mean-variance with FX-adjusted returns)
- [ ] Regime-dependent correlations: do markets become more correlated in crises?

**5.3.2 FX Management**
- [ ] FX-hedged vs unhedged portfolio comparison for each market
- [ ] FX hedging cost estimation (forward premium/discount)
- [ ] Decision framework: when to hedge, when to leave unhedged
- [ ] FX rate data pipeline: daily close rates from ECB + Yahoo Finance

**5.3.3 Global Signal Aggregation**
- [ ] Same-sector signals across markets (e.g., energy in US, NO, CA — correlated?)
- [ ] Global macro regime detection: VIX + currency volatility + yield curves → global risk-off/risk-on
- [ ] Lead-lag relationships: do US signals predict European opens? Does KRX predict US?

**5.3.4 Timezone-Aware Operations**
- [ ] Rolling signal generation: KRX closes → generate Asian signals → EU opens → generate EU signals → US opens
- [ ] 24h monitoring: cron jobs for each market's trading hours
- [ ] Slack signal delivery adapted per timezone
- [ ] Weekend processing: prepare weekly analysis for all markets before Monday opens

---

## Dead/Stale Code to Clean Up (Low Priority)

*Identified during full codebase audit (132 files, 27,453 lines). None of these block progress — clean up when convenient.*

### Stale Applications
| Path | Lines | What | Action |
|------|-------|------|--------|
| `pyfinagent-app/` | ~2,500 | Old Streamlit frontend (replaced by Next.js 15) | Archive to `archive/` or delete |
| `pyfinagent-app/risk-management-agent/` | 1 file | Orphaned Cloud Function stub inside old app | Delete with parent |

### Stale Virtual Environments
| Path | What | Action |
|------|------|--------|
| `.venv312/` | Old Python 3.12 venv (current is `.venv/` on 3.14) | Delete (`rm -rf .venv312/`) — saves ~500MB disk |

### Superseded Planning Documents
| File | Lines | Replaced By | Action |
|------|-------|-------------|--------|
| `PHASE0_FINDINGS.md` | 290 | `PLAN.md` Phase 0 section | Move to `docs/archive/` |
| `PHASE0_LEAKAGE_AUDIT.md` | 120 | Phase 0 audit results in PLAN.md | Move to `docs/archive/` |
| `PHASE0_PROGRESS.md` | 64 | Phase 0 ✅ COMPLETE in PLAN.md | Move to `docs/archive/` |
| `PHASE0_REALITY_GAP.md` | 164 | Phase 0 audit + evaluator criteria | Move to `docs/archive/` |
| `OPTIMIZATION_PLAN.md` | 268 | `PLAN.md` Phase 1-2 sections | Move to `docs/archive/` |
| `OPTIMIZER_SPEEDUP_PLAN.md` | 154 | Implemented (cache, early stopping) | Move to `docs/archive/` |
| `DEEP_RESEARCH_OPTIMIZATION.md` | 568 | `PLAN.md` Phase 3 section | Move to `docs/archive/` |
| `RESEARCH_OPTIMIZATION.md` | 239 | `PLAN.md` Phase 3 section | Move to `docs/archive/` |

**Total stale docs:** ~1,867 lines across 8 files. Historical value only.

### Cloud Function Stubs (deployed separately, not part of backend)
| Path | Lines | What | Action |
|------|-------|------|--------|
| `ingestion_agent/` | ~180 | Cloud Function for data ingestion | Keep (deployed to GCF) |
| `quant-agent/` | ~260 | Cloud Function for quant data | Keep (deployed to GCF) |
| `earnings-ingestion-agent/` | ~170 | Cloud Function for earnings | Keep (deployed to GCF) |

*These are separate deployments, not stale — just noting they exist outside the backend.*

### Other Cleanup
- [ ] Add `.venv312/` and `pyfinagent-app/` to `.gitignore` if not already excluded
- [ ] Add `.DS_Store` to `.gitignore`
- [ ] Archive 8 planning docs: `mkdir -p docs/archive && mv PHASE0_*.md OPTIMIZATION_PLAN.md OPTIMIZER_SPEEDUP_PLAN.md DEEP_RESEARCH_OPTIMIZATION.md RESEARCH_OPTIMIZATION.md docs/archive/`

---

## Harness Simplification Principles

From the article: "Every component in a harness encodes an assumption about what the model can't do on its own, and those assumptions are worth stress testing."

### What to strip as models improve:
- If Opus 4.6 can maintain coherence for 2+ hours → drop sprint decomposition ✅ (already simplified)
- If evaluator catches <5% of issues → reduce evaluation frequency
- If planner's suggestions match what optimizer finds naturally → simplify planner
- **Always test:** remove one component, measure impact, keep only what's load-bearing

### What remains load-bearing regardless of model quality:
- **Separation of generation and evaluation** — self-evaluation stays unreliable
- **Structured handoff files** — context resets need precise state
- **Grading criteria** — concrete terms beat vague quality judgments
- **DSR gate** — statistical validation can't be model-intuited away

---

## Budget Tracking

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Claude Max subscription | $200 | Updated from $125. Ford + OpenClaw |
| BigQuery | ~$10-25 | Storage + queries (tracked in BQ billing export) |
| GitHub Models (Copilot Pro) | $0 | gpt-4.1 included |
| FRED / Alpha Vantage | $0 | Free tiers |
| MCP servers hosting | $0 | Run alongside existing backend (same Mac Mini) |
| Mac Mini (amortized) | ~$28 | $1,000 / 36 months |
| LLM-guided research (Phase 3) | $2-5/cycle | ⚠️ Needs approval. Sonnet via API for Planner+Evaluator |
| Signal generation (Phase 4) | ~$1-3/day | ⚠️ Needs approval. Daily Claude calls via MCP |
| **Total** | **~$228-253/month** (Phase 2) → **~$268-348/month** (Phase 4) | |

**Budget constraint:** Negative cash flow (-$10K). Phase 2.6.2 cost autoresearch aims to reduce this.
**Cost visibility:** Phase 2.6.2 adds real-time dashboard with cash flow graph + forecast.

---

## Success Criteria

1. **Backtest Sharpe > 1.0** with DSR ≥ 0.95 ✅ ACHIEVED (1.1705)
2. **Evaluator passes** all four criteria (statistical validity, robustness, simplicity, reality gap)
3. **Paper trading** matches backtest within 20% tolerance over 2+ weeks
4. **Beat SPY** over backtest period after realistic transaction costs
5. **No known overfitting** — stable features, robust to seeds, holds across regimes

---

## Progress Log

| Date | Milestone | Sharpe | Notes |
|------|-----------|--------|-------|
| 2026-03-25 | Phase 0 complete | 0.905 | Audit, validation, bug fixes |
| 2026-03-26 | First optimizer run | 0.9848 | 12 experiments, baseline established |
| 2026-03-27 | Target achieved | 1.0391 | min_samples_leaf 20→18 |
| 2026-03-28 | Phase 1 complete | 1.1705 | 9 improvements, +19% over baseline |
| 2026-03-28 | v2 Plan (harness) | — | Three-agent architecture adopted |
| 2026-03-28 | Sub-period validation | 1.01 | ALL pass: A=0.89, B=0.92, C=1.88, 2×costs=0.91 |
| 2026-03-28 | Hang bug fixed | — | Macro preload + BQ timeouts, 5× speedup |
| 2026-03-28 | **Phase 2 core** | 1.1705 | `run_harness.py` operational (Planner→Generator→Evaluator) |
| 2026-03-28 | MCP integration | — | MCP connector added to Phase 3+4, enrichment server planned |
| 2026-03-28 | Multi-market (2.9) | — | Market abstractions, ticker namespacing, exchange_calendars |
| 2026-03-28 | **Codebase audit** | — | 132 files (27K lines) read. Phase 4 ~60% built. Stale code identified. |
| 2026-03-28 | Plan v3 finalized | — | Added: resilience (2.6.0), harness dashboard (2.6.1), budget intelligence (2.6.2), paper trading promoted to 2.7, deep research protocol, evidence-based development |

---

*This plan follows the Anthropic harness design pattern: Planner → Generator → Evaluator.*
*"The space of interesting harness combinations doesn't shrink as models improve. Instead, it moves."*
*Last updated: 2026-03-28 21:16 by Ford — Plan v3: resilience, harness dashboard, budget intelligence, paper trading promoted, deep research, evidence-based development*
