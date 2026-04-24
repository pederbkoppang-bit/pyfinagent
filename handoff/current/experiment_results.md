# Full-App End-to-End UAT masterplan phase -- Experiment results

## What was built

Added `phase-16` (Full-application end-to-end UAT, status=pending) to
`.claude/masterplan.json`, plus a 1-page operator runbook at
`handoff/current/uat-runbook.md`.

The phase has 15 sub-steps (`16.1` -> `16.15`), each with a
copy-pasteable `verification.command` and an immutable
`success_criteria` list. The phase exercises every shipped subsystem
integration-level, not hermetically:

| Sub-step | Subsystem |
|---|---|
| 16.1 | Infrastructure readiness (launchctl + /api/health + BQ + disk) |
| 16.2 | Analysis pipeline (Layer 1 -- 28 Gemini agents, 15-step orchestrator) |
| 16.3 | MAS Orchestrator live round-trip (Layer 2 -- planner/evaluator reflection) |
| 16.4 | Autonomous paper-trading cycle (with paper-lockout assertion) |
| 16.5 | Self-improving loops (MetaCoordinator + skill_optimizer + perf_optimizer) |
| 16.6 | Kill switch + risk guards drill |
| 16.7 | HITL C/C gate e2e (hitl_gate_drill + real BQ audit row assert) |
| 16.8 | Slack bot + APScheduler next-fire assertion |
| 16.9 | Backtest + quant optimizer (cache.preload_macro first) |
| 16.10 | Frontend full-page sweep (all 10 pages 200 + non-blank) |
| 16.11 | Auth + OWASP headers |
| 16.12 | Observability freshness + perf_tracker |
| 16.13 | Drills aggregate gate |
| 16.14 | Harness MAS full cycle dry-run |
| 16.15 | Go/No-Go verdict (Q/A spawn required -- self-evaluation forbidden) |

All three gotchas from the research brief are baked in as immutable
criteria:
- 16.9 criterion #1 requires `cache.preload_macro()` before backtest.
- 16.4 criterion #1+#2 requires ALPACA_PAPER_TRADE lockout assertion.
- 16.15 criterion #5 marks Q/A PASS as immutable -- no self-evaluation.

## Files changed

1. `.claude/masterplan.json` (added phase-16 after phase-14).
2. `handoff/current/uat-runbook.md` (new operator runbook).

## Verification command output (verbatim)

```
$ python3 -c "<contract-verification script>"
ALL_ASSERTS_OK
$ test -f handoff/current/uat-runbook.md && echo RUNBOOK_OK
RUNBOOK_OK
$ python3 -c "import json; json.loads(open('.claude/masterplan.json').read()); print('JSON_OK')"
JSON_OK
```

The contract-verification script (reproduces all 10 immutable criteria):
- Walks the masterplan tree for phase-16; asserts status==pending.
- Asserts sub-step ids are exactly 16.1..16.15 in order.
- Asserts every sub-step has a non-null verification.command and a
  success_criteria list of length >= 2.
- Asserts 16.9.verification.command contains "preload_macro" (gotcha 1).
- Asserts 16.4 command+criteria contains "paper" AND ("live keys" OR
  "lockout") (gotcha 2).
- Asserts 16.15 criteria contains "qa"/"Q/A" AND "pass" (gotcha 3).
- Asserts phase-16 has every top-level key sibling phase-12 has
  (id, status, name, description, created_at, completed_at, steps).

All assertions pass.

## Success-criteria coverage

| # | Criterion | Evidence |
|---|---|---|
| 1 | phase-16 entry with status=pending | PASS |
| 2 | 15 sub-steps 16.1..16.15 in order | PASS |
| 3 | every sub-step has verification.command | PASS |
| 4 | every sub-step has criteria >= 2 | PASS |
| 5 | 16.9 contains literal "preload_macro" | PASS |
| 6 | 16.4 contains "paper" AND ("live keys"/"lockout") | PASS |
| 7 | 16.15 contains "qa"/"Q/A" AND "pass" | PASS |
| 8 | masterplan.json is valid JSON | PASS |
| 9 | sibling-shape compatibility with phase-12 | PASS |
| 10 | handoff/current/uat-runbook.md exists | PASS |

## Scope discipline

- Did NOT execute the UAT. This cycle PLANS it; the UAT itself is a
  future cycle against the added masterplan entries.
- Did NOT change any application code.
- Did NOT flip any existing masterplan statuses.
- Did NOT renumber or consolidate existing phases (phase-12 is already
  Rainbow Deploys).

## Notes / follow-ups

- When phase-16 is executed, each sub-step's verification.command is
  the starting point. If a step fails, the fix lands in a normal
  masterplan cycle (research -> contract -> generate -> qa -> log).
- 16.15 requires a fresh Q/A spawn with the evidence bundle from
  16.1-16.14. This is the anti-rubber-stamp gate per CLAUDE.md.
- After phase-16 PASS, BLOCKER-4 (Paper->Live transition, task #46)
  is the final pre-go-live gate -- human-only by design.
