# Sprint Contract — phase-11 (Frontend surface audit + planning; no code)

**Step id:** phase-11 (planning meta-step) **Date:** 2026-04-21 **Tier:** complex

## Why

Phases 7-10 shipped a lot of backend capability (alt-data ingestion, transformer signals, autoresearch candidate pipeline, Slack-bot scheduled jobs, BQ cost watcher, sprint machinery). User requested a coverage audit: what's user-visible vs what's invisible, then a planned phase-11 roadmap — NO code this step, only a plan.

## Research-gate summary

Fresh researcher (complex tier): `handoff/current/phase-11-audit-brief.md` — 5 sources in full, 16 URLs, 34 internal files inspected, gate_passed=true.

Two anti-rubber-stamp findings from the audit:
1. `BudgetDashboard` LOOKS like coverage for the cost watcher but shows **NOK monthly totals from a static cost config**, not the live BQ-bytes-billed signal from `cost_budget_watcher`. Different data sources; real trip state is invisible.
2. `SignalDashboard/SignalCards` has a generic `alt_data` field populated from `tools/alt_data.py` (Google Trends only). Phase-7 `alt_congress_trades` and `alt_13f_holdings` have **no API endpoint** and never reach the UI.

## Immutable success criteria (self-imposed — this is a planning step, not coded work)

1. `brief_exists_with_coverage_matrix` — `handoff/current/phase-11-audit-brief.md` contains a full matrix (feature → exists? → surfaced? → location)
2. `gap_list_prioritized` — ordered by user-visibility/operational impact
3. `phase_11_block_drafts_present` — 10 sub-steps in masterplan-JSON shape, each with verification.command + 3-4 success_criteria
4. `no_code_shipped` — `git status --porcelain` shows ONLY the contract + brief + results in handoff/current/ (no `.py`/`.tsx` changes)

## Plan

1. Research brief already written (10 sub-steps across 4 priority tiers).
2. Q/A verifies: (a) all confirmed-surfaced features are correctly identified (no false negatives), (b) the two "misleading partial matches" (BudgetDashboard, SignalDashboard alt_data) are accurate, (c) sub-step JSON blocks are well-formed and verification commands are plausible.
3. If Q/A PASS: paste the 10 sub-steps into `.claude/masterplan.json` under a new `phase-11` entry. Each sub-step starts `status: pending`.
4. Log, close task. The 10 sub-steps then become individual execution tickets handled one at a time later.

## Priority-ordered 10 sub-steps (cycle-2 post qa_11_v1)

**Cycle-2 changes from qa_11_v1 CONDITIONAL:**
- Added new **11.10 Observability wiring** — structured logs, latency metrics, cost-per-call for the 7 new endpoints (was missing; Q/A found)
- **Moved original 11.10 (log_slot_usage wiring) out of phase-11** — it's backend-only, no UI. Renumbered to phase-10.8.1 and appended to phase-10 masterplan entry.
- Strengthened verification commands on 11.1 (add UI grep), 11.3 (POST round-trip), 11.6 (grep selectedWeekIso — not just TSC)

| Step | Name | Rationale |
|---|---|---|
| 11.1 | BQ cost-budget watcher tile | Operational safety — spend visibility against live $5/day $50/month caps. Replaces NOK static-config BudgetDashboard with live BQ bytes-billed. |
| 11.2 | Slack job heartbeat tile | Operational awareness — 7 scheduled jobs × last-run status/error |
| 11.3 | Monthly HITL approval UI | Governance — 48h approval windows expire with no current action path |
| 11.4 | Rollback events log viewer | Audit transparency over `demotion_audit.jsonl` |
| 11.5 | Weekly ledger history viewer | Research accountability |
| 11.6 | Sprint tile week selector | Low-effort; API already supports `?week_iso=` |
| 11.7 | Alt-data signal viewer | Congress/13F surfaced on analysis page (separate from existing google_trends `alt_data`) |
| 11.8 | Transformer signal viewer | Shadow-mode panel (gated — phase-8.4 REJECT still stands; shows forecasts only) |
| 11.9 | Candidate-space viewer | 15,000-combo sampling DSR/PBO distribution |
| 11.10 | Observability wiring for new endpoints | Structured logs + p50/p95/p99 latency + cost-per-call for 11.1-11.9's 7 new endpoints |

**Moved to phase-10 (backend-only):**
- `phase-10.8.1` Wire `log_slot_usage` calls into 10.3 (thursday_batch), 10.4 (friday_promotion), 10.6 (monthly_champion_challenger), 10.7 (rollback). Pure backend; required for 11.6/11.5 to show real data.

## References

- `handoff/current/phase-11-audit-brief.md` (321 lines)
- `backend/alt_data/`, `backend/autoresearch/`, `backend/slack_bot/jobs/`, `backend/api/harness_autoresearch.py` (backend inventory)
- `frontend/src/app/`, `frontend/src/components/`, `frontend/src/lib/api.ts` (frontend inventory)

## Carry-forwards (out of scope)

- Actually implementing any of 11.1-11.10 — that's each step's own harness cycle
- Fixing `BudgetDashboard`'s data source misalignment — specifically 11.1's first success criterion
- Re-architecting `SignalDashboard`'s alt_data field to read from BQ — 11.7's problem
