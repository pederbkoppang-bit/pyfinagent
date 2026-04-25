---
step: phase-17.1
title: Research gate + contract for Alpaca MCP integration
cycle_date: 2026-04-24
harness_required: false
retrospective: true
---

# Sprint Contract -- phase-17.1

## Honest framing (read first)

This contract is being written AFTER steps 17.2-17.8 have shipped and
the phase-17 parent has been marked `done` in `.claude/masterplan.json`.
That is a protocol-ordering breach under
`feedback_contract_before_generate.md`
("Contract MUST be written before GENERATE").

The research brief itself (`handoff/current/alpaca-mcp-research-brief.md`,
dated 2026-04-24) was produced at the right point in the timeline -- it
pre-dates the 17.2-17.8 work -- so the *research* phase of 17.1 was
honored in substance. What was skipped is the formal contract artifact
and the masterplan-bookkeeping step flip.

This cycle closes that gap. Q/A MUST audit the retrospective nature
explicitly and call it out in the critique (verdict may still be PASS
if the artifacts meet verification criteria, but the timing breach
belongs in `violated_criteria` as a soft-violation note so it stays
visible in the audit trail).

## Research-gate summary

Source: `handoff/current/alpaca-mcp-research-brief.md`

JSON envelope (verbatim from the brief):
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/alpaca-mcp-research-brief.md",
  "gate_passed": true
}
```

Floor checks:
- >=5 sources read in full: PASS (5 WebFetch'd pages -- Alpaca README,
  vendor landing, official docs, V2 blog, practitioner blog)
- >=10 URLs collected: PASS (15 total, 10 snippet-only)
- Recency scan performed: PASS (last-2-year window covered; CVE-2026-32211,
  OWASP MCP Top 10 2025, OX Security 2026-04-16 advisory, Alpaca V2 launch)
- 3-variant search discipline: PASS (year-less canonical + 2025 window +
  2026 current enumerated in the brief)
- Source quality hierarchy: PASS (official docs primary; vendor + practitioner
  blogs secondary; OWASP + CVEs in snippet tier)

## Hypothesis

Phase-17 (Alpaca MCP integration) as planned in the masterplan is
executable on a paper-only (scope-1/2) basis without live-flip risk,
provided:

1. Credentials stay in `backend/.env` (gitignored)
2. `ALPACA_PAPER_TRADE=true` is the default
3. Every execution path is gated behind `ExecutionRouter.submit_order`
   with a bq_sim fallback
4. A `max_notional_usd` clamp fires before any real-fill path
5. Live-flip requires explicit scope-3 gate (BLOCKER-4 / task #46)

## Success Criteria (verbatim from .claude/masterplan.json step 17.1)

1. alpaca-mcp-research-brief.md exists with gate_passed=true
2. contract.md present for this cycle
3. no_regressions

Verification command (verbatim, immutable):
```
test -f handoff/current/alpaca-mcp-research-brief.md && test -f handoff/current/contract.md && grep -c gate_passed handoff/current/alpaca-mcp-research-brief.md
```

## Plan steps (retrospective -- reconstructed)

The following is reconstructed from git history + masterplan notes, not
a forward plan (since 17.2-17.8 are already closed):

| # | Deliverable | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Research brief with gate envelope | DONE | `handoff/current/alpaca-mcp-research-brief.md` line 200-210 |
| 2 | .mcp.json entry for alpaca stdio server | DONE | commit 70cbf355; `.mcp.json` top-level |
| 3 | backend/.env paper PK keys + ALPACA_PAPER_TRADE=true | DONE | 17.2 masterplan note (`ALPACA_API_KEY_ID starts PK`) |
| 4 | Smoke-test MCP tools reachable | DONE | 17.3 masterplan note (account=PA3VQZZLAKE2, buying_power=$200k) |
| 5 | ExecutionRouter wired through paper_trader | DONE | 17.5 masterplan note; commit 70cbf355 |
| 6 | Shadow mode 5-order drill | DONE | 17.6 masterplan note (AAPL/MSFT/NVDA/GOOGL/AMZN, uat-17.6-*) |
| 7 | max_notional_usd clamp + rollback runbook | DONE | 17.7 masterplan note + `alpaca-mcp-runbook.md` |
| 8 | Scope-3 prereqs checklist feeding BLOCKER-4 | DONE | 17.8 note + `alpaca-scope3-prereqs.md` |
| 9 | Researcher subagent MCP dry-run | BLOCKED | 17.4 in-progress; `.mcp.json` alpaca entry not attached in current session; requires fresh session restart with env vars exported to shell |

## References

- `handoff/current/alpaca-mcp-research-brief.md` -- full research brief
- `handoff/current/alpaca-mcp-smoketest.md` -- 17.3 smoke evidence
- `handoff/current/alpaca-mcp-runbook.md` -- 17.7 rollback runbook
- `handoff/current/alpaca-scope3-prereqs.md` -- 17.8 scope-3 gate
- `handoff/current/alpaca-researcher-dryrun.log` -- 17.4 dry-run log (0 mcp__alpaca* hits, expected for --dry-run)
- commit 70cbf355 "phase-17: wire Alpaca MCP + paper-trader router + max-notional clamp"
- commit 89dd4400 "plan: add phase-17 Alpaca MCP server integration to masterplan"
- `.claude/rules/research-gate.md` -- research-gate discipline
- `CLAUDE.md` -- harness protocol (5-file, cycle-2 flow)

## What Q/A must audit

1. Research brief gate JSON is valid and truthful (not padded)
2. Verification command exits 0 when run
3. Retrospective-closure timing is acknowledged (not silently hidden)
4. No regressions in downstream 17.x work
5. 17.4 remains `in-progress` (NOT flipped by this cycle)
6. Parent `phase-17` already-done status is noted (data-integrity flag
   for the audit trail: parent closed before all children)
