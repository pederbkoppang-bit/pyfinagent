---
step: phase-16.51
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - docs/architecture/api-route-audit-2026-04-26.md (144 lines, NEW)
---

# Experiment Results -- phase-16.51

## What was done

Doc-only API route audit. Cross-referenced 116 backend routes
against frontend api.ts callers, frontend components, Slack bot
internal calls, and harness/smoke scripts. Identified 13
high-confidence DEAD-CANDIDATE routes + 18 CONSERVATIVE-KEEP +
78 HOT + 6 HARNESS. **No route deletions in this cycle.**

## Deliverable

`docs/architecture/api-route-audit-2026-04-26.md` (144 lines):

- Methodology section (4-step cross-reference protocol)
- Total inventory table (per-router-file breakdown across 18 files +
  main.py = 116 routes)
- 13 dead-candidate detail with caller-search evidence per route
- 18 conservative-keep cluster with rationale per group
- Recommendation: defer all deletions to a future cleanup cycle
- Methodology caveats (generic getSignalDetail overlap, Slack bot
  httpx, smoketest stub registry)
- Future cleanup priority order

## Files touched

| Path | Action | Note |
|------|--------|------|
| `docs/architecture/api-route-audit-2026-04-26.md` | CREATED | 144 lines, doc-only |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-16.51-research-brief.md` | (created earlier by researcher) | research evidence |

NO backend code changes. NO route registrations modified.
NO frontend changes.

## Verification

```
$ test -f docs/architecture/api-route-audit-2026-04-26.md && \
  grep -q "DEAD-CANDIDATE" docs/architecture/api-route-audit-2026-04-26.md && \
  grep -q "/api/skills/optimize" docs/architecture/api-route-audit-2026-04-26.md && \
  grep -q "/api/cost-budget/status" docs/architecture/api-route-audit-2026-04-26.md && \
  grep -q "Methodology" docs/architecture/api-route-audit-2026-04-26.md && \
  grep -q "116 " docs/architecture/api-route-audit-2026-04-26.md && \
  [ "$(wc -l < docs/architecture/api-route-audit-2026-04-26.md)" -ge "100" ] && \
  echo "ALL VERIFICATION PASS"
ALL VERIFICATION PASS
```

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | doc exists at canonical path | PASS |
| 2 | DEAD-CANDIDATE section present | PASS |
| 3 | skills + cost-budget routes both listed | PASS |
| 4 | methodology section present | PASS |
| 5 | 116 total route count documented | PASS |
| 6 | doc >= 100 lines | PASS (144) |
| 7 | no route deletions / backend edits | PASS |

## Honest disclosures

1. **Conservative path: zero deletions.** This audit produces a decision
   record + scaffolding for future cleanup, not the cleanup itself.
   Rationale documented in the recommendation section: 5 of the 13
   dead candidates may be revived if specific roadmap items return
   (skill-level optimization, signal sub-routes when the analysis
   pipeline re-enables them).

2. **Generic getSignalDetail overlap.** The signal sub-routes
   (`/insider`, `/options`, `/patents`, `/earnings-tone`) are flagged
   dead based on hardcoded-call grep, but a generic
   `getSignalDetail(ticker, signal)` function exists in `api.ts` and
   could theoretically call any sub-path. Dead classification assumes
   call-site grep accurately reflects current usage.

3. **Slack bot httpx callers** were inventoried. Endpoints reached
   internally (`/api/health`, `/api/paper-trading/status`, etc.) are
   classified HOT. The audit acknowledges that some endpoint hits
   may be one-off curl-style calls not in the static grep.

4. **`/api/backtest/runs/{run_id}` (1 route)** is the only HOT-or-DEAD
   ambiguity worth re-checking after a UI deep-dive — the list
   endpoint `/runs` is HOT, but the parameterised single-fetch could
   be wired in by a future Run Detail view.

5. **Researcher correction:** initial route count of 114 was off by 2
   (two DELETE routes in `backtest.py` were missed). Corrected to 116.

6. **Future cleanup cycle proposed** with priority order in the doc:
   `/api/skills/*` first (biggest cluster + clear ownership boundary),
   then `/api/cost-budget/status`, then performance pair, etc.

## Closes

Task list item #73. Phase-16.51.

## Next

Spawn Q/A.
