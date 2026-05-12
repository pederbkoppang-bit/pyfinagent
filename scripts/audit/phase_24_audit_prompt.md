# Phase-24 Application-Wide Audit — Master Prompt

**Purpose:** drive a comprehensive read-only audit of the pyfinagent
codebase across 15 buckets, producing 15 findings docs at
`docs/audits/phase-24-2026-05-12/`. Each findings doc proposes ranked
phase-25.x code-change candidates with file paths, draft verification
commands, and grounded research-gate citations.

**How to use this file:** open a fresh Claude Code session, paste this
entire file as your first message. The agent (Main, this session)
will execute the audit bucket-by-bucket through the standard
harness protocol (research → contract → no GENERATE → Q/A → log →
flip status). Auto-commit-and-push fires per bucket.

**Approved plan:** `/Users/ford/.claude/plans/sunny-jingling-deer.md`
(2026-05-12). **Masterplan steps:** `phase-24.0` through `phase-24.14`
(already added to `.claude/masterplan.json`).

**Phase-24 is READ-ONLY.** No code changes. All proposed fixes are
phase-25.x candidates in the findings docs.

---

## Charter (read before starting any bucket)

### Red-line invariants (from `project_system_goal.md` memory)

The system's primary goal: **"maximize profit at lowest cost live;
dynamically shift strategy to whichever is making the most money."**

Every audit bucket must measure its findings against this red line.
A finding that surfaces a feature working "as designed" but
mis-aligned with the red line is still a valid phase-25 candidate.

### Concrete bugs reported by operator (audit must close all)

1. **Stop-loss orphan**: `backend/services/paper_trader.py:414-423`
   defines `check_stop_losses()` but zero callers exist in the repo.
   Stocks in the red are not being sold. TER held at -12.30% with
   no stop set (only 6 of 11 positions have stops).
2. **Full 28-agent pipeline unused**: `backend/services/autonomous_loop.py:564-615`
   branches on `settings.lite_mode`. Default is lite (4-field analyzer).
   The full `AnalysisOrchestrator` (15-step / 28-skill) is opt-in
   and currently unreached in paper trading.
3. **No new reports**: `_persist_lite_analysis` at `autonomous_loop.py:749-777`
   only persists for lite path. `/reports` page is empty.
4. **Autoresearch isolated**: `backend/meta_evolution/cron.py:87-125`
   runs Sunday-only. Zero cross-module imports between
   `backend/autoresearch/` and `backend/services/autonomous_loop.py`.
5. **Agent rationale aliasing**: Trader and RiskJudge rationale text
   is byte-identical in the `/paper-trading` drawer (FIX BUY screenshot
   2026-05-01). RiskJudge weight 0.00. Sparse drawer (3 of ~20 agents).
6. **Slack notifications wrong**: Morning Digest (2:00 PM) and Evening
   Digest (11:00 PM) both show `Portfolio: +$0.00 (+0.0%)` when
   portfolio has 11 active positions with mostly positive P&L.
   Morning Digest "Recent Analyses" shows 5× SNDK instead of current
   holdings. No trade/kill-switch/drawdown/error notifications.

### Research-gate enforcement (MANDATORY per bucket)

Per `.claude/rules/research-gate.md`, every bucket's researcher spawn
MUST:
- Fetch ≥5 sources IN FULL via WebFetch (snippets do NOT count).
- Collect ≥10 unique URLs.
- Run a recency scan (2024-2026 window) — empty section is OK if no
  new findings supersede canonical sources.
- Use three-variant search-query discipline (current-year frontier +
  last-2-year window + year-less canonical).
- Cite per claim with `[file:line]` for internal anchors and
  `[URL accessed YYYY-MM-DD]` for external sources.
- End with the standard JSON envelope:
  ```json
  {
    "tier": "simple|moderate|complex",
    "external_sources_read_in_full": 0,
    "snippet_only_sources": 0,
    "urls_collected": 0,
    "recency_scan_performed": false,
    "internal_files_inspected": 0,
    "gate_passed": false
  }
  ```
  `gate_passed: true` iff `external_sources_read_in_full >= 5` AND
  `recency_scan_performed == true` AND every hard-blocker satisfied.

If any researcher returns `gate_passed: false`, that bucket's contract
is REJECTED. Re-spawn researcher with the missing inputs; do not
proceed to Q/A.

### Findings-doc structure (every bucket)

Every findings doc at `docs/audits/phase-24-2026-05-12/<bucket-id>-<slug>-findings.md`
MUST have these sections in order:

```markdown
---
bucket: 24.<N>
slug: <kebab-case-slug>
cycle: <N>
cycle_date: 2026-MM-DD
researcher_gate: {gate_passed: true, ...}
---

# Findings — phase-24.<N> — <Title>

## Executive summary

(1 paragraph TL;DR; what's broken and what's not)

## Code-grounded findings

(Every claim cited with file:line anchor + grep evidence.
Include every concrete bug touching this bucket.)

## External-research summary

(Citations to canonical Anthropic / Google / MCP / academic docs
fetched in full by the researcher. Verbatim URLs.)

## Recency scan (2024-2026)

(New findings from last-two-year window OR "No relevant new
findings in the window".)

## Proposed phase-25.x candidate steps

### Candidate 1: <name>

- **Priority**: P0|P1|P2
- **Files to edit/create**: (absolute paths)
- **Draft verification command**: `python3 tests/verify_phase_25_X.py`
- **Rationale**: (1 paragraph)
- **Estimated effort**: small|medium|large

(Repeat for at least 3 candidates per bucket.)

## Open questions

(Anything the bucket surfaced but couldn't answer without code
changes or live data.)

## References

(Full URL list — canonical sources first, supplementary after.)
```

### Per-bucket harness loop (each of 24.0 → 24.14)

1. **Read** the bucket's section below.
2. **Spawn researcher** with the prompt template at §"Researcher
   spawn template" — include the bucket's canonical-URL whitelist.
3. **Verify gate_passed: true** in the brief returned. If not,
   re-spawn researcher with what's missing.
4. **Write contract** at `handoff/current/contract.md` with the
   bucket's success_criteria copied verbatim from `.claude/masterplan.json`
   step `24.<N>` verification.success_criteria.
5. **GENERATE phase = AUDIT phase**: write the findings doc at
   `docs/audits/phase-24-2026-05-12/<bucket-id>-<slug>-findings.md`
   per the template above.
6. **Write experiment_results.md** at `handoff/current/experiment_results.md`
   with the verbatim verifier output.
7. **Spawn Q/A subagent** to certify the findings. Q/A reads the
   findings doc, the contract, and runs `python3 tests/verify_phase_24_<N>.py`.
8. **Q/A returns PASS** (or CONDITIONAL/FAIL → re-do, no
   verdict-shopping).
9. **Append harness_log.md** with the cycle entry (`## Cycle M --
   2026-MM-DD -- phase=24.<N> result=PASS`).
10. **Flip masterplan status to done** — auto-commit-and-push fires.

### Researcher spawn template

```
**Step under research:** phase-24.<N> — <bucket title>

**Effort tier:** moderate (use complex if the bucket touches >5 backend modules)

**Hypothesis to falsify/confirm:** <bucket-specific hypothesis>

**External research focus (≥5 sources read in full via WebFetch):**

<bucket-specific canonical URL whitelist — copy from this prompt>

**Three-variant search-query discipline (required):**
- Current-year frontier: `"<topic> 2026"`
- Last-2-year window: `"<topic> 2025"`
- Year-less canonical: `"<topic>"`

**Internal codebase audit:**

<bucket-specific files to inspect — copy from this prompt>

**Hard-blocker checklist:**

- [ ] ≥5 external sources fetched in full via WebFetch
- [ ] Recency scan (2024-2026) section present
- [ ] Three-variant search-query discipline visible
- [ ] file:line anchors for every internal claim
- [ ] JSON envelope at end with gate_passed flag

**Output:** write the brief to `handoff/current/research_brief.md`
ending with the standard JSON envelope. Return a ≤200 word summary +
envelope contents.
```

### Q/A spawn template

```
**Step to evaluate:** phase-24.<N> — <bucket title>

**Cycle:** <N>
**Date:** 2026-MM-DD

## 5-item harness-compliance audit (MANDATORY FIRST)

1. researcher gate cleared (gate_passed: true in research_brief.md)
2. contract pre-commit (contract.md written before findings doc)
3. results: experiment_results.md complete with verbatim verifier output
4. log-last: harness_log NOT yet appended for this cycle (will be last)
5. no verdict-shopping: first Q/A spawn for this bucket

## Deterministic checks

- Run `python3 tests/verify_phase_24_<N>.py` — must report all PASS
  except claim `harness_log_has_phase_24_<N>_cycle_entry` (log-last)
- Grep findings doc for canonical URL substring (must be present)
- Confirm research_brief.md has gate_passed: true envelope

## LLM-judgment leg

1. Contract alignment: does the findings doc address every bullet
   in contract.md's "Plan steps" section?
2. Mutation-resistance: do the verifier claims actually catch
   regression (vs being trivially-true assertions)?
3. Anti-rubber-stamp: is there anything in this bucket that should
   have been investigated but wasn't?
4. Scope honesty: are the deferrals + open questions explicit?
5. Research-gate compliance: are all 5 sources cited verbatim?

## Verdict envelope

Return `{ok, verdict, violated_criteria, violation_details,
certified_fallback, checks_run, reason}`. Write the full critique to
`handoff/current/evaluator_critique.md`.
```

---

## Bucket 24.0 — Charter + red-line invariants

**Purpose:** establish the audit charter, embed the coverage matrix
(every backend/frontend/infra path mapped to a bucket), and document
the canonical-URL whitelist per bucket. This is the gate that unlocks
24.1-24.12.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.0-charter-findings.md`

**Hypothesis:** the 15-bucket structure is sufficient + non-overlapping
to cover the entire codebase against the profit-maximization red line.

**Researcher canonical-URL whitelist (must WebFetch all in full):**
- `https://www.anthropic.com/engineering/harness-design-long-running-apps`
- `https://www.anthropic.com/engineering/building-effective-agents`
- `https://www.anthropic.com/engineering/built-multi-agent-research-system`
- `https://code.claude.com/docs/en/hooks`
- (Plus 1+ academic/industry source on AI-trading-system audit methodology)

**Internal files to inspect:**
- `.claude/masterplan.json` — confirm phase-24 entry has 15 children
- `.claude/rules/research-gate.md`
- `.claude/rules/backend-agents.md`
- `.claude/rules/frontend.md`
- `CLAUDE.md`, `ARCHITECTURE.md`
- `docs/audits/dev-mas-2026-05-11/` — prior-art audit; reuse its shape

**Bucket-specific guidance:**
- Embed the entire coverage matrix from `/Users/ford/.claude/plans/sunny-jingling-deer.md` §B.1 verbatim in the findings doc.
- For each bucket, list its canonical-URL whitelist (so subsequent
  buckets' researchers have a single source of truth).
- Explicitly cite `project_system_goal.md` memory as the red-line spec.

**Verifier:** `python3 tests/verify_phase_24_0.py` (15 claims)

---

## Bucket 24.1 — Trading-execution + governance (P0)

**Purpose:** close the stop-loss-orphan bug. Address why TER fell to
-12.30% without a sell. Audit entry-path stop attachment for the 5
stop-less positions. Audit governance limits (sector caps, position
limits, kill-switch wiring).

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.1-execution-trading-findings.md`

**Hypothesis:** `check_stop_losses()` at `paper_trader.py:414-423`
is orphan code. The daily-loop scheduler at `autonomous_loop.py` does
not call it. The entry path (`execute_trade` or equivalent) does
not write `stop_loss` on new positions for some code branch
(causing 5 of 11 positions to have no stop).

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (read full)
- Academic: AQR / Two Sigma on stop-loss design (trailing stops, ATR-based, fixed-percentage)
- Industry: any 2024-2026 paper or blog on systematic exit rules
- Academic: behavioral-finance on disposition effect (why retail holds losers)
- A reference text on order-state machines (broker API patterns)

**Internal files to inspect:**
- `backend/services/paper_trader.py` (full file — 500+ LOC)
- `backend/services/autonomous_loop.py` (specifically lines around
  cycle_run / execute_decisions / position_management)
- `backend/governance/` (limits-loader watcher — does it actually
  block entries? grep for callers)
- `backend/services/kill_switch*.py` (kill-switch state machine)
- `backend/services/risk_manager*.py` (if exists)
- Recent paper-trades from BQ — sample 50 to find any sell
  transactions (or confirm zero sells)

**Bucket-specific guidance:**
- Document every position in the current portfolio (FIX, MU, KEYS,
  GEV, COHR, ON, INTC, TER, DELL, GLW, CIEN) and tag which have stops.
- Trace the entry code path — find why ON, INTC, TER, DELL, GLW, CIEN
  do NOT have stops set despite being 15 days old.
- Confirm `check_stop_losses()` has zero callers via `grep -rn "check_stop_losses" backend/ scripts/`.
- Propose phase-25 candidates (at least 5):
  - (a) Wire `check_stop_losses()` into the daily loop.
  - (b) Ensure entry path writes a stop on every new position.
  - (c) Backfill migration to attach stops to the 5 stop-less positions.
  - (d) Add a watchdog test for "no-sells-in-N-days" anomaly detection.
  - (e) Verify governance sector caps + position limits actually fire.

**Verifier:** `python3 tests/verify_phase_24_1.py` (14 claims)

---

## Bucket 24.2 — Pipeline routing + report persistence (P1)

**Purpose:** explain why the 28-agent / 15-step full pipeline is
unused in paper trading. Explain why `/reports` page is empty. Decide
whether to route paper trading through the full pipeline (and the
cost implication).

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md`

**Hypothesis:** `autonomous_loop.py:564-615` branches on
`settings.lite_mode` (default `True`). Lite mode skips
`AnalysisOrchestrator` entirely. `_persist_lite_analysis` at
`autonomous_loop.py:749-777` is the only persistence path. Full
pipeline runs would persist via a different mechanism (or not at all
into the same BQ table the `/reports` page reads).

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (full)
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full)
- Google Generative AI: structured output + grounding docs
- Anthropic: agent-loop best practices
- One 2024-2026 source on cost-vs-quality tradeoffs for LLM pipelines

**Internal files to inspect:**
- `backend/services/autonomous_loop.py` (full file)
- `backend/agents/orchestrator.py` (15-step pipeline, head + skill-list)
- `backend/agents/skills/*.md` (32 skill prompts — count + categorize)
- `backend/api/reports.py` (where does it source data?)
- `backend/db/bigquery_client.py` (`save_report` callers)
- BQ schema for `analysis_results` or `reports` table

**Bucket-specific guidance:**
- Cost analysis: estimate per-ticker cost difference between lite and
  full path (token counts × model pricing).
- Confirm `/reports` page query points at `analysis_results` (or
  similar) and that only lite-path writes hit that table.
- Propose phase-25 candidates:
  - (a) Add settings toggle to flip paper trading to full pipeline
    on a per-ticker or per-cycle basis (cost-bounded).
  - (b) Wire full-pipeline outputs to BQ persistence so `/reports`
    populates.
  - (c) Add a per-ticker A/B (lite vs full) to measure quality lift.

**Verifier:** `python3 tests/verify_phase_24_2.py` (13 claims)

---

## Bucket 24.3 — Autoresearch ↔ daily-loop wiring (P1)

**Purpose:** explain why no autoresearch has evolved the application.
Champion/challenger runs but doesn't feed back into daily strategy.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.3-autoresearch-wiring-findings.md`

**Hypothesis:** `backend/meta_evolution/cron.py:87-125` runs only on
Sunday 02:00 ET. `backend/autoresearch/` produces proposals →
gates → promotions on a weekly cadence. Zero cross-module imports
between `backend/autoresearch/` and `backend/services/autonomous_loop.py`
means the daily trading cycle never sees evolution outputs.

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full)
- One 2024-2026 source on continuous-deployment patterns for ML systems
- One source on champion/challenger A/B testing in trading systems
- Reinforcement-learning literature on online policy updates
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (full)

**Internal files to inspect:**
- `backend/meta_evolution/` (all 12 files — read headers + main entry points)
- `backend/autoresearch/` (all 14 files — same)
- `backend/services/autonomous_loop.py` (grep for autoresearch / meta_evolution imports — confirm zero)
- `backend/agents/skill_optimizer*.py` (if exists)

**Bucket-specific guidance:**
- Diagram the current data flow: daily loop → BQ logs → weekly cron
  reads BQ → proposes change → ??? → no feedback to daily loop.
- Propose phase-25 candidates:
  - (a) Daily loop reads the latest "promoted" strategy from BQ on
    each cycle start.
  - (b) Champion/challenger A/B test runs in shadow mode during the
    daily cycle, not weekly batch.
  - (c) Add a "promoted strategy" registry table that daily loop polls.

**Verifier:** `python3 tests/verify_phase_24_3.py` (12 claims)

---

## Bucket 24.4 — Agent topology + per-agent rationale flow (P0)

**Purpose:** close the Trader=RiskJudge aliasing bug + sparse drawer
(3 of ~20 agents). Surface Layer-1 28-skill outputs in per-trade
rationale.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md`

**Hypothesis:** the rationale-producing code path in
`backend/agents/multi_agent_orchestrator.py` (or equivalent) writes
RiskJudge rationale as `risk_rationale = trader_rationale` (or
similar field-aliasing). Layer-1 28-skill outputs are not surfaced
because the lite-mode path (bucket 24.2 upstream) skips them.

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full — independent-evaluator pattern)
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (full — "agents praise own work" warning)
- Anthropic: building-effective-agents (full)
- One 2024-2026 source on multi-agent voting / consensus
- Academic: independent-evaluator design (NIST evaluation methodology, etc.)

**Internal files to inspect:**
- `backend/agents/agent_definitions.py` (full — 5 agent types)
- `backend/agents/multi_agent_orchestrator.py` (full)
- `backend/agents/planner_agent.py` + `evaluator_agent.py`
- `backend/api/paper_trading.py` (specifically the
  `/trades/{trade_id}/rationale` endpoint)
- `frontend/src/app/paper-trading/components/AgentRationale*.tsx` (or similar)
- `backend/agents/_inventory.json`

**Bucket-specific guidance:**
- Grep for where `RiskJudge` rationale field is populated. Compare
  with `Trader` rationale population.
- Run a single trade through the system (or examine a sampled trade
  from BQ) and trace every agent that should contribute to the
  rationale.
- Cross-link with bucket 24.2: lite mode skips 28-skill pipeline →
  no per-ticker rationale to surface.
- Propose phase-25 candidates:
  - (a) Decouple RiskJudge rationale from Trader's text — RiskJudge
    must call its own LLM with risk-specific prompt.
  - (b) Surface Layer-1 28-skill outputs in the rationale drawer
    (only when full pipeline runs).
  - (c) Add a per-agent contribution-weight column.
  - (d) Add a frontend toggle to expand from "3 main agents" view to
    "all contributing agents" view.

**Verifier:** `python3 tests/verify_phase_24_4.py` (13 claims)

---

## Bucket 24.5 — Slack notifications + operator alerting (P0)

**Purpose:** close the wrong-P&L digests bug, the 5×SNDK
recent-analyses bug, the mis-scheduled morning digest, the absence
of trade/kill-switch/drawdown/error notifications.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md`

**Hypothesis:** the digest builder in `backend/slack_bot/app.py`
(or `backend/slack_bot/handlers/*.py`) queries the wrong BQ table or
applies an incorrect filter for the portfolio P&L computation. The
"Recent Analyses" section uses a `LIMIT 5 ORDER BY analysis_date DESC`
query that doesn't filter by ticker, so a single ticker that recently
got 5 analyses dominates. The morning digest scheduler uses UTC
instead of ET, causing 2:00 PM ET dispatch instead of 06:00 ET.
Missing notification types are simply not implemented — only the two
digests exist.

**Researcher canonical-URL whitelist:**
- `https://api.slack.com/docs` (or equivalent Slack Bolt SDK docs)
- One 2024-2026 source on financial alerting / pager design (PagerDuty
  blog, Sentry blog, or similar)
- One source on alert fatigue / deduplication best practices
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (alerting-as-observability)
- One source on Slack Block Kit + thread management for financial alerts

**Internal files to inspect:**
- `backend/slack_bot/app.py` (full)
- `backend/slack_bot/handlers/*.py` (every file)
- `backend/slack_bot/scheduler*.py` (if exists)
- The actual BQ queries the digest builder runs (sample one)
- `backend/api/paper_trading.py::portfolio` endpoint (compare — that
  one returns CORRECT P&L)
- Settings for Slack webhook + Socket Mode (`backend/.env` references)

**Bucket-specific guidance:**
- Reproduce the bugs:
  - Run `/portfolio` slash command — does it return correct P&L?
  - Trigger the morning-digest function manually — does the bug repro?
- Find the divergence between the WORKING `/api/paper-trading/portfolio`
  query and the BROKEN digest query.
- Find the morning-digest scheduler entry — verify cron expression
  + TZ.
- Enumerate every notification type that SHOULD exist (real-time
  trade confirmation, stop-loss trigger, kill-switch state change,
  daily P&L threshold alarm, drawdown alarm, cost-budget alarm,
  watchdog alert, cycle-completion notification, error escalation,
  weekly summary).
- Propose phase-25 candidates:
  - (a) Fix digest P&L data source.
  - (b) Fix recent-analyses query (filter or rotate tickers).
  - (c) Fix morning digest schedule (cron + TZ).
  - (d) Add real-time trade confirmation notifications.
  - (e) Wire kill-switch state change to Slack.
  - (f) Add drawdown alarm.
  - (g) Add cost-budget breach alert.
  - (h) Add cycle-completion summary.
  - (i) Add error-escalation routing.
  - (j) Add weekly autoresearch summary.

**Verifier:** `python3 tests/verify_phase_24_5.py` (15 claims)

---

## Bucket 24.6 — Backtest engine + walk-forward + live-vs-backtest (P2)

**Purpose:** audit the 1167-LOC backtest engine + quant optimizer.
Walk-forward correctness, seed stability, champion/challenger A/B
discipline. Are backtest results predictive of live P&L?

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.6-backtest-engine-findings.md`

**Hypothesis:** the backtest engine is correct but its outputs don't
flow back into the live trading system. Live-vs-backtest drift is
not measured. Seed stability may be undocumented.

**Researcher canonical-URL whitelist:**
- AQR / Two Sigma / Robeco on walk-forward validation
- Academic: López de Prado "Advances in Financial Machine Learning" (or successor)
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full)
- One 2024-2026 source on backtesting AI-driven strategies
- A source on champion/challenger A/B testing in finance

**Internal files to inspect:**
- `backend/backtest/backtest_engine.py` (head, key methods)
- `backend/backtest/quant_optimizer.py`
- `backend/api/backtest.py` (25 endpoints — categorize)
- `backend/backtest/experiments/optimizer_best.json`
- `scripts/harness/run_harness.py` (the quant harness driver)
- `tests/quant_results.tsv` (recent results)

**Bucket-specific guidance:**
- Sample a recent walk-forward run from BQ — confirm it ran end-to-end.
- Test seed stability — run the same parameter set twice, expect
  identical results.
- Compare a recent backtest's predicted Sharpe vs the live paper-trading
  realized Sharpe over the same window. Quantify drift.
- Propose phase-25 candidates:
  - (a) Add a live-vs-backtest reconciliation report.
  - (b) Add seed-stability test to the harness.
  - (c) Add champion/challenger output → BQ → daily loop wire.

**Verifier:** `python3 tests/verify_phase_24_6.py` (13 claims)

---

## Bucket 24.7 — Data quality + BQ freshness + yfinance fallback (P1)

**Purpose:** are BQ tables actually fresh? Is yfinance fallback firing
silently? Is signal freshness measured?

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.7-data-quality-findings.md`

**Hypothesis:** BQ tables (`pyfinagent_data`, `_staging`, `_hdw`, `_pms`)
have inconsistent freshness windows. The `/freshness` endpoint reports
a single age, but per-table ages may differ. yfinance fallback fires
silently when BQ is stale, masking the underlying problem.

**Researcher canonical-URL whitelist:**
- `https://cloud.google.com/bigquery/docs/best-practices-performance-input`
- `https://yfinance-python.org/` (or successor) — yfinance behavior
- One 2024-2026 source on data-quality monitoring for trading systems
- Source on circuit-breaker / fallback patterns
- BQ scheduled-query documentation

**Internal files to inspect:**
- `backend/db/bigquery_client.py`
- `backend/tools/yfinance_*.py` (or equivalent)
- `backend/api/paper_trading.py::freshness`
- `backend/services/ticker_meta*.py` (if exists)
- Recent BQ logs / schedule (use MCP `mcp__bigquery__describe-table`
  on `pyfinagent_data.signals` etc.)

**Bucket-specific guidance:**
- For each major BQ table (signals, prices, fundamentals, macro),
  report last-write timestamp.
- Find the yfinance fallback path. Confirm it logs when firing.
- Audit cache preload (`cache.preload_macro()` per CLAUDE.md).
- Propose phase-25 candidates:
  - (a) Per-table freshness endpoint + Slack alarm when stale.
  - (b) yfinance-fallback-fired counter + alarm.
  - (c) Signal freshness SLA + monitoring.

**Verifier:** `python3 tests/verify_phase_24_7.py` (13 claims)

---

## Bucket 24.8 — Observability + monitoring + safety rails (P1)

**Purpose:** audit watchdog, kill-switch, SLA monitor, governance
limits-loader, cost-budget enforcement, monthly-approval gate.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.8-observability-findings.md`

**Hypothesis:** safety rails exist but may not be wired end-to-end.
Kill-switch may exist but not be reachable from the UI. Watchdog may
not page operator on backend down. Cost budget may be honor-system.

**Researcher canonical-URL whitelist:**
- Google SRE Book chapters on observability + alerting
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (full)
- One 2024-2026 source on AI safety rails for autonomous systems
- One source on cost-budget enforcement in LLM apps
- One source on watchdog timer patterns

**Internal files to inspect:**
- `backend/services/watchdog*.py`
- `backend/services/kill_switch*.py`
- `backend/services/sla_monitor*.py`
- `backend/governance/` (limits-loader, full)
- `backend/api/cost_budget_api.py`
- `backend/api/monthly_approval_api.py`
- `backend/api/observability_api.py`
- `handoff/logs/backend-watchdog.log`

**Bucket-specific guidance:**
- For each safety rail, trace: trigger → action → operator notification.
- Confirm kill-switch is operator-reachable from `/paper-trading` UI.
- Confirm watchdog actually restarts backend or alerts on down.
- Confirm cost budget HARD-blocks LLM calls when breached (vs warn).
- Propose phase-25 candidates:
  - (a) Add kill-switch hot-key (per `scripts/audit/keyboard_flatten.py` if relevant).
  - (b) Cost budget hard-block + Slack alert.
  - (c) SLA monitor → Slack escalation on breach.

**Verifier:** `python3 tests/verify_phase_24_8.py` (15 claims)

---

## Bucket 24.9 — LLM provider conformance (P2)

**Purpose:** audit Claude + Gemini usage against current docs. Are
we using the right features? Missing batch / files API / citations?
Is prompt caching at the right depth? Are thinking budgets tuned?

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.9-llm-conformance-findings.md`

**Hypothesis:** infrastructure is largely correct but features like
batch API + files + citations are unused. Thinking budgets may be
over- or under-tuned. Prompt cache hit rate may be measurable but
not optimized.

**Researcher canonical-URL whitelist:**
- `https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching` (full)
- `https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking` (full)
- `https://docs.anthropic.com/en/docs/build-with-claude/tool-use` (full)
- `https://docs.anthropic.com/en/docs/build-with-claude/batch-processing` (full)
- `https://docs.anthropic.com/en/docs/build-with-claude/citations` (full)
- `https://ai.google.dev/gemini-api/docs/structured-output` (full)
- `https://ai.google.dev/gemini-api/docs/grounding` (full)
- `https://ai.google.dev/gemini-api/docs/thinking` (full)

**Internal files to inspect:**
- `backend/services/llm_client.py` (full — 1100+ LOC)
- `backend/services/cost_tracker.py` (full)
- `backend/agents/orchestrator.py` (thinking budgets, schemas)
- `backend/agents/multi_agent_orchestrator.py` (tool-use loop)

**Bucket-specific guidance:**
- Audit prompt caching hit rate (use `cost_tracker.cache_hit_rate`
  property). Identify low-hit-rate call sites.
- Audit thinking budgets per agent: Critic 8192, Moderator 8192,
  Risk Judge 4096, Synthesis 4096. Are these justified by quality?
- Identify unused Anthropic features (batch, files, citations) and
  estimate cost savings from adopting them.
- Audit Google Search grounding usage — confirm it's only on Gemini
  call sites.
- Propose phase-25 candidates:
  - (a) Adopt batch API for non-interactive analyses (cheaper).
  - (b) Adopt files API for repeated large-document analyses.
  - (c) Adopt citations for source-grounded outputs (audit trail).
  - (d) Tune thinking budgets per realized quality lift.

**Verifier:** `python3 tests/verify_phase_24_9.py` (16 claims)

---

## Bucket 24.10 — MCP infrastructure + security (P1)

**Purpose:** audit `.mcp.json` (alpaca, bigquery), deny rules, auth
(NextAuth + WebAuthn), secrets handling, BQ access patterns. Identify
new MCP gaps (signals MCP? news MCP? slack-claude MCP?).

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.10-mcp-security-findings.md`

**Hypothesis:** current MCP setup is minimal (2 servers, both deny-listed
for writes). Security posture is good but not comprehensive. New
MCPs could improve operator workflow (e.g., a signals MCP for
ad-hoc analysis).

**Researcher canonical-URL whitelist:**
- `https://modelcontextprotocol.io/specification` (full)
- `https://docs.claude.com/en/docs/claude-code/mcp` (full)
- `https://next-auth.js.org/v5/getting-started` (NextAuth v5 docs, full)
- WebAuthn / passkey spec (FIDO Alliance or W3C)
- One 2024-2026 source on secret management in Python apps

**Internal files to inspect:**
- `.mcp.json` (full)
- `.claude/settings.json` (permissions section)
- `backend/auth/` (full)
- `backend/.env.example` (or template if exists)
- `scripts/mcp_servers/` (smoke tests)

**Bucket-specific guidance:**
- For each MCP tool, confirm deny rules are tight (no write ops
  without explicit prompt).
- Audit `backend/.env` for any committed secrets (use git history grep).
- Confirm WebAuthn passkey flow works end-to-end.
- Propose phase-25 candidates:
  - (a) Add a signals MCP server (ad-hoc signal queries from Claude Code).
  - (b) Add a news MCP server (or claude-news / similar).
  - (c) Audit Alpaca API key rotation policy.

**Verifier:** `python3 tests/verify_phase_24_10.py` (15 claims)

---

## Bucket 24.11 — Frontend ↔ Backend data-layer wiring (P2)

**Purpose:** type drift, API contract, learnings-page hookup, every
page → endpoint mapping, error-response handling.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.11-frontend-data-wiring-findings.md`

**Hypothesis:** wiring is mostly clean. Orphan: `/paper-trading/learnings`
has UI but no backend. Type drift between Pydantic models and TS types
is minimal but non-zero.

**Researcher canonical-URL whitelist:**
- `https://nextjs.org/docs/14/app-router` (or current Next.js version)
- `https://react.dev/learn/you-might-not-need-an-effect` (React 19 patterns)
- TanStack Query or SWR docs (data-fetching)
- TypeScript handbook on type drift / codegen
- One 2024-2026 source on Pydantic ↔ TypeScript type generation

**Internal files to inspect:**
- `frontend/src/lib/api.ts` (full)
- `frontend/src/lib/types.ts` (full — 104 interfaces)
- `backend/api/models.py` (Pydantic models)
- `frontend/src/app/paper-trading/learnings/page.tsx`
- `frontend/src/app/**/page.tsx` (sample 3-4 for endpoint patterns)

**Bucket-specific guidance:**
- For each of 14 frontend pages, list its called endpoints.
- For each of ~132 endpoints, list the frontend caller (or note
  orphan).
- Sample 5-10 Pydantic models and confirm matching TypeScript types.
- Propose phase-25 candidates:
  - (a) Wire `/paper-trading/learnings` backend.
  - (b) Add Pydantic-to-TS type codegen.
  - (c) Standardize error-response shape.

**Verifier:** `python3 tests/verify_phase_24_11.py` (14 claims)

---

## Bucket 24.12 — Frontend UI/UX presentation layer (P2)

**Purpose:** design-system conformance, a11y, responsive design,
per-page UX flows, cross-tab KPI consistency, visual regression
baseline.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.12-ui-ux-presentation-findings.md`

**Screenshots dir:** `docs/audits/phase-24-2026-05-12/screenshots/`
(committed in this bucket; one image per page).

**Hypothesis:** the codebase has strict rules (`.claude/rules/frontend.md`)
but enforcement is inconsistent. Phosphor icon imports may bypass
`@/lib/icons.ts`. Loading/error/empty states are missing on some
pages. Cross-tab KPI mismatches likely exist.

**Researcher canonical-URL whitelist:**
- WCAG 2.2 (W3C) — accessibility
- `https://playwright.dev/docs/screenshots` (Playwright headless)
- Tailwind CSS docs (current major)
- `https://www.w3.org/WAI/ARIA/apg/` (ARIA Authoring Practices)
- One 2024-2026 source on design-system enforcement in React apps

**Internal files to inspect:**
- `.claude/rules/frontend.md` + `frontend-layout.md`
- All 14 `frontend/src/app/**/page.tsx`
- `frontend/src/lib/icons.ts`
- `frontend/src/components/**` (representative sample)
- Phosphor icon import pattern across all components

**Bucket-specific guidance:**
- Take Playwright headless screenshots of all 14 pages (mobile +
  desktop breakpoints). Commit to `screenshots/`.
- Grep for `@phosphor-icons/react` direct imports (should be zero —
  all imports via `@/lib/icons.ts`).
- Audit each page for error/loading/empty states.
- Cross-tab KPI: compare same metric on `/` vs `/paper-trading` vs
  `/sovereign` — do numbers match?
- Propose phase-25 candidates:
  - (a) Add visual regression CI gate (Playwright + screenshot diff).
  - (b) Enforce icon-import lint rule (block direct `@phosphor-icons/react`).
  - (c) Add missing loading/empty/error states.
  - (d) Reconcile cross-tab KPI discrepancies.

**Verifier:** `python3 tests/verify_phase_24_12.py` (16 claims)

---

## Bucket 24.13 — Profit-maximization red-line alignment (P1)

**Purpose:** synthesis of buckets 24.1-24.9. Does current behavior
match "maximize profit at lowest cost; shift to winning strategy"?

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.13-redline-synthesis-findings.md`

**Hypothesis:** current behavior is mis-aligned in multiple ways:
- Stops don't fire → not minimizing loss (anti-profit).
- Full pipeline unused → potentially missing alpha (anti-profit).
- Autoresearch isolated → no strategy switching (anti-shift-to-winning).
- Cost budget not hard-enforced → not minimizing cost.

**Depends on:** 24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9
findings must be complete before this bucket runs.

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full)
- One academic source on multi-objective optimization (profit + cost)
- Markowitz / mean-variance portfolio theory
- One 2024-2026 source on AI-trading-system performance attribution
- A source on cost-of-cognition tradeoffs (LLM cost vs alpha)

**Internal files to inspect:**
- Findings from buckets 24.1-24.9 (read all 9 docs)
- `project_system_goal.md` memory
- BQ data: 30-day P&L vs 30-day LLM cost
- `backend/api/sovereign_api.py` (red-line monitor)
- `backend/api/performance_api.py`

**Bucket-specific guidance:**
- Quantify the gap: estimate $ lost from stop-loss-orphan vs $ saved
  by lite-mode default.
- Estimate alpha left on the table by skipping full pipeline.
- Audit the strategy-switching mechanism (if any).
- Propose phase-25 candidates:
  - (a) Define a profit-per-LLM-dollar metric, track in real time.
  - (b) Strategy auto-switching policy (mandatory after 24.3 wired).
  - (c) Daily P&L attribution report.

**Verifier:** `python3 tests/verify_phase_24_13.py` (14 claims)

---

## Bucket 24.14 — Final synthesis + ranked phase-25.x candidate list

**Purpose:** aggregate all 14 prior buckets' phase-25.x candidates
into a single ranked list. Emit proposed masterplan JSON entries.

**Findings doc:** `docs/audits/phase-24-2026-05-12/24.14-final-synthesis-findings.md`

**Depends on:** 24.0-24.13 findings complete.

**Researcher canonical-URL whitelist:**
- `https://www.anthropic.com/engineering/harness-design-long-running-apps` (full — multi-step planning)
- One 2024-2026 source on dependency-graph prioritization
- One source on technical-debt ranking
- `https://www.anthropic.com/engineering/built-multi-agent-research-system` (full)
- One source on backlog prioritization (RICE / MoSCoW / WSJF)

**Internal files to inspect:**
- All 14 prior findings docs in `docs/audits/phase-24-2026-05-12/`
- `.claude/masterplan.json` (current state — for the proposed phase-25 entry format)

**Bucket-specific guidance:**
- Collect every "Proposed phase-25 candidate" from 24.1-24.13.
- Deduplicate (e.g., kill-switch fixes from 24.1 and 24.8 may overlap).
- Rank by:
  - **P0**: blocks live trading correctness (stops, aliasing, Slack
    P&L). Likely 8-12 candidates.
  - **P1**: high-value but not blocking (pipeline routing, autoresearch,
    backtest reconciliation, safety rails). Likely 10-15 candidates.
  - **P2**: nice-to-have (LLM conformance, frontend polish, MCP
    expansion). Likely 8-12 candidates.
- For each candidate, emit a proposed `.claude/masterplan.json` step
  entry (JSON literal) with: id, name, status=pending,
  harness_required, priority, depends_on_step, audit_basis (back-ref
  to bucket), verification (command, success_criteria, live_check).
- Write the proposed entries as a single JSON array at the bottom of
  the findings doc so the operator can paste them into masterplan.json.

**Verifier:** `python3 tests/verify_phase_24_14.py` (14 claims)

---

## Out of scope reminders

- **No code changes in phase-24.** All fixes live in the phase-25
  candidates emitted by 24.14.
- **Phase-23.x leftovers** (FLIPPED_STEP race from cycle 41,
  `if`-predicate over/under-fire) — separate phase-23.8.5+ cycles.
- **R-5 (qa.md fail-mode)** + **qa.md follow-on for live_check** —
  deferred per separation-of-duties.
- **Crypto re-introduction** — explicitly out (already removed
  phase-5).
- **Multi-market expansion** (phase-5 — FX, futures, international
  equities) — pre-existing pending phase; out of phase-24 audit
  scope (may surface as phase-25 candidate if 24.13 finds the
  single-market path too fragile to expand).
- **Penetration testing** — bucket 24.10 covers security posture
  (auth, secrets, deny rules) but NOT active pen-test.

---

**End of master audit prompt. Start with bucket 24.0.**
