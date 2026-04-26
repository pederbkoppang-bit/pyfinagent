---
step: phase-16.51
title: API dead-route audit -- doc-only deliverable
cycle_date: 2026-04-26
harness_required: true
forward_cycle: true
parent_phase: phase-16
deliverables:
  - docs/architecture/api-route-audit-2026-04-26.md (NEW)
---

# Sprint Contract -- phase-16.51

## Research-gate summary

`handoff/current/phase-16.51-research-brief.md`. tier=simple,
internal-only, gate_passed=true. 211-line brief. 18+ files inspected
across backend/api/, frontend, slack_bot, scripts.

## Findings

- **116 total backend routes** (researcher correction; prior estimate was 114)
- **13 high-confidence DEAD-CANDIDATE routes** across 5 router files
- **Conservative-keep cluster** (jobs, observability, harness, mas-events, signals sub-routes) deferred pending future use

## Dead-candidate list (13 routes)

| File | Line | Method | Path |
|------|------|--------|------|
| `backend/api/backtest.py` | 638 | GET | `/api/backtest/runs/{run_id}` |
| `backend/api/cost_budget_api.py` | 98 | GET | `/api/cost-budget/status` |
| `backend/api/performance_api.py` | 37 | GET | `/api/perf/slow` |
| `backend/api/performance_api.py` | 49 | GET | `/api/perf/llm/p95` |
| `backend/api/signals.py` | 119 | GET | `/api/signals/{ticker}/insider` |
| `backend/api/signals.py` | 124 | GET | `/api/signals/{ticker}/options` |
| `backend/api/signals.py` | 141 | GET | `/api/signals/{ticker}/patents` |
| `backend/api/signals.py` | 149 | GET | `/api/signals/{ticker}/earnings-tone` |
| `backend/api/skills.py` | 52 | POST | `/api/skills/optimize` |
| `backend/api/skills.py` | 63 | POST | `/api/skills/stop` |
| `backend/api/skills.py` | 74 | GET | `/api/skills/status` |
| `backend/api/skills.py` | 81 | GET | `/api/skills/experiments` |
| `backend/api/skills.py` | 88 | GET | `/api/skills/analysis` |

## Concrete plan

1. Write `docs/architecture/api-route-audit-2026-04-26.md` (~150 lines).
   Sections:
   - Methodology (cross-reference frontend api.ts + Slack bot + scripts)
   - Total inventory: 116 routes across 18 router files
   - Per-router-file breakdown (HOT/HARNESS/DEAD-CANDIDATE/CONSERVATIVE-KEEP counts)
   - 13 dead-candidate detail (caller-search evidence per route)
   - Conservative-keep cluster + rationale
   - Recommendation: NO route deletions this cycle (defer to dedicated cleanup or migration cycle); doc serves as decision record + scaffolding for future cleanup

2. **NO route deletions.** Conservative — some flagged routes may be
   revived when skill-level optimization returns or when signal
   sub-routes are re-plumbed.

## Success Criteria (verbatim, immutable)

```
test -f docs/architecture/api-route-audit-2026-04-26.md && \
grep -q "DEAD-CANDIDATE" docs/architecture/api-route-audit-2026-04-26.md && \
grep -q "/api/skills/optimize" docs/architecture/api-route-audit-2026-04-26.md && \
grep -q "/api/cost-budget/status" docs/architecture/api-route-audit-2026-04-26.md && \
grep -q "Methodology\|methodology" docs/architecture/api-route-audit-2026-04-26.md && \
grep -q "116 " docs/architecture/api-route-audit-2026-04-26.md && \
[ "$(wc -l < docs/architecture/api-route-audit-2026-04-26.md)" -ge "100" ]
```

Plus:
- `no_route_deletions`: zero changes to `backend/api/*.py` or `backend/main.py`
- `no_backend_edits`: only the new doc + handoff/* rolling

## What Q/A must audit

1. Compound `&&` immutable verification command exits 0.
2. Doc exists at the expected path with all 13 dead-candidate routes listed.
3. Doc explains methodology + has per-file breakdown.
4. Recommendation explicitly says "no deletions this cycle" with rationale.
5. NO backend code edited (git status confined to docs + handoff).
6. Conservative-keep cluster is documented (skills agents, signals sub-routes, jobs, observability).
