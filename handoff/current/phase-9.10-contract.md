# Sprint Contract — phase-9 / 9.10 (cron runbook) — REMEDIATION v1

**Step id:** 9.10 **Remediation cycle:** 1 **Date:** 2026-04-20 **Tier:** simple

## Why remediation

Previous cycle inline-authored. Fresh MAS re-run.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.10-research-brief.md` — 6 sources in full, 16 URLs, three-variant, recency, gate_passed=true.

Validated: runbook structure (inventory, wiring, manual invocation, observability, failure-modes, test suite, re-enabling, schedule-change, references) aligns with 2026 SRE runbook templates (OneUptime, Squadcast). 70-line length is appropriate for a 7-job system.

Carry-forwards (NOT in 9.10 scope — documented for follow-up):
1. **Silent-no-op failure class missing from §5 failure-modes table.** Runbook should add rows for:
   - `cost_budget_watcher` TypeError (CRITICAL — every tick) — covered by phase-9.9.1 code fix
   - `weekly_data_integrity` empty-dict inertness (MEDIUM silent no-op) — covered by phase-9.9.1 code fix
   Per Robust Perception + deadmanping: silent no-ops are worse than loud failures because alerts never fire.
2. No escalation severity / SLA per job (2026 SRE standard: sev-1/2/3/4 with MTTR targets)
3. No rollback procedure documented (revert-and-restart implicit, not explicit)
4. Schedule-change governance informal ("edit + restart launchd") — should require PR + review + changelog per 2026 SRE doc best practices
5. Observability §4 notes "logger.info sink; wire to BQ in a later phase" — acceptable deferral but must close before go-live
6. No MTTR targets per failure class

## Immutable criterion

`test -f docs/runbooks/phase9-cron-runbook.md && grep -c "^| " docs/runbooks/phase9-cron-runbook.md | python -c "import sys; n=int(sys.stdin.read()); sys.exit(0 if n >= 14 else 1)"`

Expected: exit 0 (≥14 table rows across all tables).

## Plan

1. Re-verify.
2. Capture output.
3. Spawn fresh Q/A.
4. Log and confirm.

## References

- `handoff/current/phase-9.10-research-brief.md`
- `docs/runbooks/phase9-cron-runbook.md` (70 lines, unchanged)
- `tests/slack_bot/test_scheduler_phase9.py::test_runbook_exists`
- Cross-phase: phase-9.9.1 will update the runbook's §5 after CRITICAL bugs are fixed
- `.claude/masterplan.json` → phase-9 / 9.10
