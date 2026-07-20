# Live check -- Step 75.0

## Verification command (verbatim, exit 0)

```
$ bash -c 'test -f handoff/current/research_brief_75.0.md && test -f handoff/current/audit_phase75/register.md && jq -e ".confirmed | length >= 1" handoff/current/audit_phase75/confirmed_findings.json >/dev/null && jq -e "[.phases[] | select(.id==\"phase-75\") | .steps[] | select(.status==\"pending\")] | length >= 8" .claude/masterplan.json >/dev/null'
exit=0
```

## Workflow evidence

- Research gate: run `wf_646a6e15-a94` (Fable 5/max) -> `handoff/current/research_brief_75.0.md`, envelope `gate_passed:true`, `external_sources_read_in_full:24`, `recency_scan_performed:true`, `coverage.dry:true` (10 rounds, 2 dry).
- Audit GENERATE: run `wf_03d6e7c4-fda` (245 agents, all `model:claude-fable-5`, ~6.76M subagent tokens, 1858 tool calls; one resume through a session-limit reset on the same run id).
- Pipeline: 14 finders -> raw 145 -> dedup 136 -> +6 gap probes (+58 confirmed) -> **184 confirmed / 16 refuted-or-duplicate** (P1:20 P2:100 P3:64, P0:0) -> synthesis 16 steps -> independent step-review (13 approved, 3 revised, 0 missing).

## Change-surface proof (audit-only)

`git status --short` for this step is confined to:
- `.claude/masterplan.json` (phase-75 install + 16 pending steps + 74.2 name re-anchor)
- `handoff/current/**` (research_brief_75.0, contract(_75.0), experiment_results(_75.0), audit_phase75/{register.md, confirmed_findings.json}, goal draft, live_check)
- hook-managed append-only audit streams under `handoff/audit/` + `handoff/**` JSONLs (PreToolUse/InstructionsLoaded hooks -- not authored by this step)

Risk-gate files byte-untouched (empty `git status` for `backend/governance/limits.yaml`, `backend/services/paper_trader.py`, `backend/risk`). No product code edited; the 16 remediation steps defer all code changes to their executor sessions.
