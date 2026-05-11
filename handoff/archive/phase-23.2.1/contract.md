---
step: phase-23.2.1
title: Verify autonomous loop ran daily for 7+ days
cycle_date: 2026-05-07
harness_required: true
verification: "bq SELECT DATE(snapshot_date), COUNT(*) FROM paper_portfolio_snapshots WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 9 DAY) GROUP BY 1 ORDER BY 1; expect ~9 rows, no gaps"
research_brief: handoff/current/phase-23.2.1-research-brief.md
---

# Contract — phase-23.2.1

## Hypothesis

The autonomous paper-trading loop (`paper_trading_daily` APScheduler job
firing `run_daily_cycle`) wrote a `paper_portfolio_snapshots` row on each
of the last ~9 calendar days. Because Step 8 of `run_daily_cycle`
(`backend/services/autonomous_loop.py:443-452`) is the canonical write
site and the snapshot insert uses MERGE on `snapshot_date`
(`backend/db/bigquery_client.py:695-712`, idempotent since phase-23.1.18),
a 7+ day daily run produces 7+ distinct DATE rows in the verification
window with no gaps.

This is a **verification-only** step. No code changes. The hypothesis
is falsifiable by the immutable BQ query in `verification` above. If
the query returns fewer than ~9 rows or shows gaps, the **verification
step itself succeeded** (we ran the check) but the **system did not
behave as hypothesized** — that is a real finding to surface, not a
test to rewrite.

## Research-gate summary

`researcher` agent `a2b8525d67d0d837e` ran tier=simple and returned
`gate_passed: true` with:
- 7 external sources fetched in full via WebFetch (≥5 floor cleared)
- 10 snippet-only sources (17 URLs total — clears the ≥10 floor)
- Recency scan 2024-2026 performed (3 hits — Feb-Mar 2026 BigQuery MERGE
  / cron-monitoring sources)
- Three-query discipline followed (current-year frontier + last-2-year
  + year-less canonical)
- 10 internal files inspected with file:line anchors

Brief: `handoff/current/phase-23.2.1-research-brief.md`.

**Critical finding from internal half** (load-bearing for this contract):
`handoff/cycle_history.jsonl` shows completed cycles only on
2026-04-20, -21, -24, -26 (multiple), -27 (multiple), -28, -29, and
-05-06. The 9-day verification window from `DATE_SUB(CURRENT_DATE(),
INTERVAL 9 DAY)` (= 2026-04-28) contains likely **3-4 distinct dates
with completed cycles** — well below the "expect ~9 rows" target. The
verification will probably FAIL on the count/gap criterion, which is
a legitimate finding (the loop did not run daily) rather than a
verifier defect. Main and Q/A must treat this as a real-world finding,
not a contract bug.

External research confirms the BQ query is fit for purpose (a
GROUP-BY-DATE COUNT is equivalent to the date-spine left-join gap
detection pattern in the practitioner literature; cron-job dead-man's
switch is the post-deployment verification idiom).

## Immutable success criteria (verbatim — DO NOT EDIT)

Copied verbatim from `.claude/masterplan.json::23.2.1.verification`:

```
bq SELECT DATE(snapshot_date), COUNT(*) FROM paper_portfolio_snapshots
WHERE PARSE_DATE('%Y-%m-%d', snapshot_date) >= DATE_SUB(CURRENT_DATE(),
INTERVAL 9 DAY) GROUP BY 1 ORDER BY 1; expect ~9 rows, no gaps
```

Decoded into deterministic checks the GENERATE step + Q/A must perform:

1. The query executes against BQ project `sunny-might-477607-p8`,
   dataset `pyfinagent_data` (or whichever dataset hosts
   `paper_portfolio_snapshots` — Q/A confirms via `list-tables`),
   without runtime error and within the 30s BQ timeout.
2. Result rows have `snapshot_date` (STRING `%Y-%m-%d`) parseable as
   DATE within the 9-day window.
3. Row count and gap analysis are reported verbatim. The "expect ~9
   rows, no gaps" target is the success line; any shortfall must be
   reported with the missing dates listed.
4. PASS = exactly 9 rows OR the report explains why fewer rows is the
   correct outcome (e.g., system was paused with operator approval).
   The default interpretation per Anthropic immutable-criteria
   doctrine: shortfall = system finding to log, not a contract to
   rewrite.

## Plan steps

1. (DONE — RESEARCH phase) Researcher returned brief +
   `gate_passed: true`. See above.
2. (DONE — PLAN phase) This contract.
3. **GENERATE phase:**
   a. Use BigQuery MCP `mcp__bigquery__describe-table` on
      `pyfinagent_data.paper_portfolio_snapshots` to confirm the
      column type matches the migration spec (STRING).
   b. Use `mcp__bigquery__execute-query` to run the immutable
      verification query verbatim. Capture full result.
   c. Compute gap analysis in-line: enumerate the 9 expected dates,
      list which are present and which are missing, count rows.
   d. Write `handoff/current/experiment_results.md` with: table
      results verbatim + gap analysis + finding statement.
   e. Optionally write `tests/verify_phase_23_2_1.py` — a short
      reproducer that re-runs the BQ query via the Python client (not
      MCP, since MCP is a session-scoped tool) so the verification
      can be replayed in CI / by Q/A. Use a thin wrapper that prints
      results and exits non-zero only on **query failure**, not on
      data shortfall (the data shortfall is the finding to capture).
4. **EVALUATE phase:** spawn fresh `qa` agent with full
   harness-compliance audit (5-item checklist must come BEFORE any
   code check per `feedback_qa_harness_compliance_first.md`).
5. **LOG phase:** append `handoff/harness_log.md` AFTER Q/A returns
   PASS / CONDITIONAL / FAIL. Flip `.claude/masterplan.json` 23.2.1
   status only after the log append (log-last per
   `feedback_log_last.md`).

## Anti-patterns guarded (≥2)

1. **Verdict-shopping on FAIL**: per Anthropic harness-design and the
   project's own `feedback_harness_rigor.md`, if Q/A returns FAIL on
   the data shortfall, Main does NOT spawn a fresh Q/A on the same
   evidence to overturn it. Q/A's verdict reflects ground truth (the
   loop has gaps); the cycle-2 fix is at the *system* level (operator
   investigates why the loop missed days), not at the verification
   level.
2. **Rewriting immutable criteria** (Anthropic: "unacceptable to
   remove or edit tests"). The criterion "expect ~9 rows, no gaps" is
   verbatim from masterplan; if data shows fewer rows, that is a
   finding, not a contract bug. The contract must not be amended to
   "expect 3-4 rows" because that is the data we have.
3. **Self-evaluation by Main** — Q/A is mandatory per CLAUDE.md.
   Same-session pragmatism (skipping Q/A) is forbidden when the
   verdict will inform a system-level finding.

## Out of scope

- Investigating *why* the loop missed days. That's a follow-up phase
  (likely a new step in 23.2 or 23.3 depending on root cause). This
  step only verifies the run-frequency hypothesis; the diagnosis is
  separate.
- Restarting the autonomous loop or backfilling snapshots. That's an
  operator action gated on the finding from this step.
- Any code changes to `bigquery_client.py`, `paper_trader.py`, or
  `autonomous_loop.py`. Those are owned by phase-23.1.x.

## Backwards compatibility

Pure additive: new `experiment_results.md` + optional
`tests/verify_phase_23_2_1.py`. No code changes.

## Risk

- BQ MCP tool may not be attached in this session — fallback is the
  Python client per CLAUDE.md `BigQuery Access (MCP)::point 6`. The
  fallback path is documented in CLAUDE.md and uses `GCP_PROJECT_ID`
  from `backend/.env`; ADC covers both.
- The verifier script's exit code semantics matter: it must exit 0 on
  query success regardless of row count, so Q/A's deterministic check
  doesn't conflate "verification succeeded" with "data matched
  hypothesis." The data verdict is the LLM-judgment leg's job.

## References

- Research brief: `handoff/current/phase-23.2.1-research-brief.md`
  (researcher `a2b8525d67d0d837e`, 7 sources read in full).
- Masterplan: `.claude/masterplan.json::23.2.1` — verification
  copied verbatim above.
- Phase-23.1.18 (MERGE idempotency): `handoff/archive/phase-23.1.18/`.
- Phase-23.2.0 (post-deployment audit framework):
  `handoff/archive/phase-23.2.0/` (if present).
- Anthropic harness-design: https://www.anthropic.com/engineering/harness-design-long-running-apps
  ("verification criteria are immutable").
- BQ MERGE idempotency: https://oneuptime.com/blog/post/2026-02-17-how-to-use-merge-statements-in-bigquery-for-upsert-operations/view
- Cron dead-man's-switch pattern: https://dev.to/cronmonitor/how-to-monitor-cron-jobs-in-2026-a-complete-guide-28g9
