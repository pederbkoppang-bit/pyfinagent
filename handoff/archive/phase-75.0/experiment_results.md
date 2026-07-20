# Experiment results -- Step 75.0: Ultracode full-stack code-quality audit

**Date:** 2026-07-19  **Session:** Fable 5 + ultracode (all agents on `claude-fable-5`, effort max, operator override).

## What was built/changed

This step is AUDIT + QUEUE ONLY. No product code, no `backend/.env`, no flag flips, no optimizer runs. The change surface is confined to `handoff/**` and `.claude/masterplan.json` (+ hook-managed CHANGELOG).

### Artifacts produced

| File | Purpose |
|---|---|
| `handoff/current/research_brief_75.0.md` | Research gate (24 full-read sources, audit-class coverage.dry) -- gate_passed:true |
| `handoff/current/contract_75.0.md` (+ rolling `contract.md`) | PLAN with 5 immutable criteria |
| `handoff/current/audit_phase75/confirmed_findings.json` | 184 confirmed + 16 refuted/dup + 78 dropped + step-review verdict (machine-readable) |
| `handoff/current/audit_phase75/register.md` | Human-readable register: stats, 20 P1s, step map, revisions, refuted table |
| `.claude/masterplan.json` phase-75 | step 75.0 (this audit) + 16 pending remediation steps 75.1..75.16 |

### Pipeline (Workflow `wf_03d6e7c4-fda`)

14 read-only role auditors (`agentType: Explore`, `model: fable`, effort max) -> JS key-dedupe (145->136) + fuzzy-cluster agent -> adversarial verify (every finding >=1 verifier; each P1 a 2nd maximally-skeptical refuter; duplicates vs phase-69..74 registers killed) -> completeness-critic + 6 targeted gap finders (+58 confirmed) -> synthesis into 16 step candidates -> independent adversarial step-review. 245 agents, ~6.76M subagent tokens, 1858 tool calls. One mid-run resume through a session-limit reset (cached agents replayed; 40 lost verifiers + gap round + synthesis ran live on the same run id).

## Verbatim verification-command output

```
$ bash -c 'test -f handoff/current/research_brief_75.0.md && test -f handoff/current/audit_phase75/register.md && jq -e ".confirmed | length >= 1" handoff/current/audit_phase75/confirmed_findings.json >/dev/null && jq -e "[.phases[] | select(.id==\"phase-75\") | .steps[] | select(.status==\"pending\")] | length >= 8" .claude/masterplan.json >/dev/null'; echo "exit=$?"
```

(Exit code recorded in `handoff/current/live_check_75.0.md`.)

## Artifact shape

- **Confirmed findings:** 184 (P1:20, P2:100, P3:64; P0:0). By role: api-design 15, docs-drift 13, gaps 58 across 6 probes, ts-contract 12, sre-ops/qa-tests 10 each, architecture/react/llm-eng/perf/deps 9 each, py-core/data-bq 6, py-services 5, security 4.
- **Headline P1 cluster:** unauthenticated `white_check_mark`-reaction git-push (commands.py:338), dead deploy-from-Slack control plane with zero caller authz (self_update.py + ~2,900 lines dead modules), middleware auth keyed only on Google provider (Passkey config bypass), Slack assistant streaming un-awaited coroutine (whole feature broken), P0 kill-switch pager logs "sent" on failed subprocess, MCP publish path risk-checks against wrong portfolio, CC-rail silently drops Pydantic schemas, promotion-gauntlet writes seeded-noise reports that pass all gates, governance limits.yaml (2%) vs live setting (4%) divergence.
- **16 remediation steps** 75.1..75.16, executor-tagged (sonnet-4.6/high mechanical, opus-4.8/xhigh judgment), each with an immutable testable `success_criteria` set + offline-exit-0 `verification.command` + a `live_check` gate. All ship security/observability/infra fixes with explicit BOUNDARY notes (kill-switch/stops/sector-caps/DSR/PBO byte-untouched; live-loop changes flag-gated DARK; historical_macro frozen).

## Boundary compliance

`git diff --name-only` for this step touches only `handoff/**`, `.claude/masterplan.json`, and the goal/contract drafts. Risk-gate files (`backend/governance/limits.yaml`, kill-switch, stops, sector caps, DSR/PBO gate code) byte-untouched. No product code edited; the 16 steps defer all code changes to their executor sessions.
