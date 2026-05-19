# Sprint Contract -- phase-30.0

**Step:** phase-30.0 -- E2E paper-trading pipeline audit (diagnostic-only).
**Date:** 2026-05-19.
**Mode:** OVERNIGHT. NO code edits. NO mutating BQ/Alpaca calls. NO
`AskUserQuestion`.
**Cycle owner:** Main (this Claude Code session) + Researcher
(`.claude/agents/researcher.md`, complex tier) + Q/A
(`.claude/agents/qa.md`, single spawn on UPDATED evidence).

## Research-gate summary

Researcher spawned with `complex` tier. Brief at
`handoff/current/research_brief.md`. JSON envelope reports:

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 10,
  "snippet_only_sources": 10,
  "urls_collected": 20,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "gate_passed": true
}
```

Floor of 5 sources read in full cleared 2x. Three-variant search-
query composition (2026 / 2025 / year-less canonical) is visible
in the brief's "1. Three-variant search queries run" table.
Recency-scan section present with 5 last-2-year findings. Internal
codebase audit covers all 12 named stages with file:line anchors.

Backup researcher (`researcher-backup`) spawned at 20:55 as a
hedge against the primary stalling at the skeleton for ~10
minutes. Primary unstuck shortly after the backup launched and
delivered the comprehensive brief; backup brief (moderate tier)
is on disk at `handoff/current/research_brief_phase30_backup.md`
but the primary brief is the authoritative source for this
cycle's gate-pass.

## Hypothesis

The pyfinagent paper-trading dashboard surfaces five concrete
anomalies (Sharpe -6.26 vs P&L +9.35%, GATE 0/5 NOT ELIGIBLE with
11 open positions, 10/11 Technology sector concentration, 6-of-11
positions without stop-loss, cycle 3 days stale). The HYPOTHESIS
is that each anomaly is explainable by **a wiring or coverage gap
in the 12-stage pipeline rather than an algorithmic mistake** --
and that the gaps are addressable with bounded, P1/P2/P3-ranked
remediation steps to be planned as `phase-30: E2E Paper-Trading
Pipeline Remediation`.

This cycle is diagnostic-only. The outcome is a gap report at
`handoff/current/experiment_results.md` + a phase-30 masterplan
entry that lists the remediation steps. No code is touched.

## Immutable success criteria

All criteria below are immutable per `.claude/rules/research-gate.md`
and `CLAUDE.md` "Never edit verification criteria in masterplan.json".

### SC-1: 12-stage trace with per-stage verdict + file:line anchors

The gap report at `handoff/current/experiment_results.md` MUST
contain a per-stage section for each of these 12 stages, each
emitting a verdict of `PASS` / `PARTIAL` / `FAIL`:

1. Universe + candidates (`backend/tools/screener.py`)
2. Analysis (28 Gemini agents OR lite path)
3. MAS debate (Layer 2 strategy router)
4. Decision (`portfolio_manager.py::decide_trades`)
5. Risk-gate ordering (eligibility vs execution)
6. Sector concentration enforcement
7. Stop-loss assignment + coverage
8. Order routing (`execution_router.py` -> bq_sim / alpaca_paper)
9. Mark-to-market (`autonomous_loop.py` Step 5)
10. Stop-loss enforcement (Step 5.6 vs Step 6 ordering)
11. Exit path (closed trade through `paper_round_trips`)
12. Learning loop (`_learn_from_closed_trades` + outcome_tracker)

Each stage verdict MUST cite:
- At least one `file:line` anchor from the relevant backend code.
- At least one BQ-query evidence row (table name + WHERE clause +
  result count) OR a documented "table empty" / "no rows" finding.

### SC-2: Live-anomaly cross-validation

The gap report MUST reconcile each of these five dashboard
observations against the BQ ground truth, citing query results:

A. **Sharpe -6.26 vs P&L +9.35%** -- explain the mathematical
   source of the disagreement (different return series, deposit-
   contamination, etc.) with reference to
   `backend/services/paper_metrics_v2.py::_nav_to_returns` AND at
   least one BQ query against `paper_portfolio_snapshots`.
B. **vs SPY -4.62%** -- confirm whether the SPY-benchmark window
   aligns with the portfolio inception_date.
C. **Cycle 3 days ago** -- identify the silent-failure date from
   `handoff/cycle_history.jsonl` AND `handoff/kill_switch_audit.jsonl`
   AND the LLM call log (`pyfinagent_data.llm_call_log`).
D. **GATE 0/5 NOT ELIGIBLE** -- enumerate the 5 booleans in
   `backend/services/paper_go_live_gate.py::compute_gate`, classify
   which is red/green for the current state, AND explain why
   paper trading runs despite the gate (gate scope: PROMOTE-to-
   LIVE, NOT a trading-block).
E. **Sector cap not blocking 10/11 Tech** -- identify whether
   the cap is wired (`portfolio_manager.py:194,219-229`), whether
   the default is 2 (per `settings.py:159`), and why the
   historical bootstrap positions slipped through (phase-23.1.13
   `sector` enrichment gap pre-fix).

### SC-3: P1/P2/P3 ranked remediation themes

The gap report MUST emit a single section "Phase-30 remediation
plan" with at least 6 themes split across P1/P2/P3. Each theme
MUST:
- name a concrete file or table to touch (no abstract themes
  like "improve observability"),
- cite either an external reference from the research brief OR an
  internal phase number (e.g., "extends phase-25.2 backfill").

### SC-4: Phase-30 masterplan entry (JSON-ready)

The gap report MUST end with a copy-pasteable JSON block for
`phase-30: E2E Paper-Trading Pipeline Remediation` with at least
6 steps (one per P1/P2 theme + at least one P3), using the
phase-23.8 schema:
- `id` (e.g. `30.0`, `30.1`, ...)
- `name`
- `status` = `"pending"` (or `"done"` only for phase-30.0 itself)
- `harness_required`
- `priority` (P1 / P2 / P3)
- `depends_on_step`
- `audit_basis` (verbatim reference to this contract + research
  brief)
- `verification.command` (a deterministic shell command)
- `verification.success_criteria` (>=2 named criteria)
- `verification.live_check` (description string -- the
  `live_check_<step_id>.md` artifact requirement)
- `retry_count = 0`, `max_retries = 3`

### SC-5: Hard guardrails

The gap report MUST be diagnostic-only:
- No code changes in `backend/`, `frontend/`, `scripts/`.
- No mutating BigQuery calls (no INSERT / UPDATE / DELETE / DROP).
- No mutating Alpaca calls (no order create / cancel / replace).
- All BQ queries used `LIMIT` + a partition / date filter.
- All `mcp__alpaca__*` calls are inspection-only (no
  `submit_order`, `cancel_order`, `close_position`,
  `close_all_positions`).

### SC-6: Q/A on UPDATED evidence

Q/A is spawned exactly ONCE on the final gap report. On
CONDITIONAL or FAIL, the cycle-2 flow is:
1. Main reads the critique's `violated_criteria`.
2. Main fixes the blockers AND updates the handoff files
   (`experiment_results.md`, `evaluator_critique.md` Follow-up,
   plus any cited support docs).
3. Main spawns a FRESH Q/A on the updated evidence.
4. Circuit breaker: max 2 retries. If 2 retries fail, mark
   `phase-30.0` status `blocked` in masterplan.json with a
   `blocker` field describing the failed Q/A audit.

NO second-opinion-shopping. NO fresh Q/A on unchanged evidence.

## Plan steps

1. RESEARCH (done) -- spawn `researcher` complex tier, write
   `research_brief.md`. **DONE**.
2. PLAN (this file) -- write `contract.md`. **DONE**.
3. GENERATE -- read-only forensic diagnostic. Main reads 12
   stages of code + runs ~15 BQ read-only queries + inspects
   `cycle_history.jsonl` and `kill_switch_audit.jsonl`. Writes
   findings to `experiment_results.md` per SC-1 + SC-2 + SC-3 +
   SC-4. Strict diagnostic-only per SC-5.
4. EVALUATE -- spawn `qa` once. Q/A runs harness-compliance
   audit first (5 items), then verdict on the gap report.
   Verdict is `PASS` / `CONDITIONAL` / `FAIL`.
5. LOG -- append cycle entry to `handoff/harness_log.md` AFTER
   PASS and BEFORE the masterplan status flip.
6. FLIP -- insert phase-30 + phase-30.0 entry into
   `.claude/masterplan.json`; set phase-30.0 status = `done` if
   Q/A PASS or `blocked` if Q/A FAIL after 2 retries. Auto-
   commit + auto-push triggered by the hooks.

## References

- `handoff/current/research_brief.md` (primary, complex tier, 10
  sources read in full, gate_passed=true).
- `handoff/current/research_brief_phase30_backup.md` (backup,
  partial).
- `CLAUDE.md` "Harness Protocol" section.
- `.claude/rules/research-gate.md` (three-variant query
  composition + JSON envelope).
- `docs/runbooks/per-step-protocol.md` (Plan -> Generate ->
  Evaluate -> Log cycle).
- `backend/services/autonomous_loop.py` (12 stages of cycle).
- `backend/services/paper_trader.py` (assignment + enforcement).
- `backend/services/portfolio_manager.py::decide_trades` (sector
  cap, sell-first-then-buy).
- `backend/services/paper_go_live_gate.py::compute_gate` (the 5
  booleans).
- `backend/services/perf_metrics.py` +
  `backend/services/paper_metrics_v2.py` (Sharpe sources).
- `handoff/cycle_history.jsonl` + `handoff/kill_switch_audit.jsonl`
  (process-plane cycle health).
- `pyfinagent_data.llm_call_log` (cycle-level LLM activity).
- `financial_reports.{paper_positions,paper_trades,paper_round_trips,paper_portfolio_snapshots,signals_log,outcome_tracking,agent_memories}`
  (data-plane state).

## Anti-patterns to avoid (from auto-memory)

- Do not skip the research gate. (`feedback_research_gate.md`)
- Do not log before Q/A PASS. (`feedback_log_last.md`)
- Do not second-opinion-shop on unchanged evidence. (`feedback_harness_rigor.md`)
- Do not bundle status-flip ahead of log. (`feedback_log_last.md`)
- Do not edit verification criteria mid-cycle. (CLAUDE.md
  "Never edit verification criteria").
- Do not git-stash with active hooks. (`feedback_no_git_stash_with_active_hooks.md`)
