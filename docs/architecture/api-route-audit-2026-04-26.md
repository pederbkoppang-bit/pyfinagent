# API Route Audit — 2026-04-26 (phase-16.51)

Doc-only audit cross-referencing every backend HTTP route against
known callers (frontend `api.ts`, frontend components, Slack bot,
harness scripts, smoke tests). Identifies dead-candidate routes
without deleting them — conservative path; deletions deferred to
future cleanup cycle.

## Methodology

1. **Inventoried backend routes** via `grep -rnE "@router\.(get|post|put|delete|patch)|@app\.(get|post|put|delete|patch)" backend/api/ backend/main.py`.
2. **Inventoried frontend callers** via `grep -nE "fetch\(|api\." frontend/src/lib/api.ts` + a recursive grep over `frontend/src/` for hardcoded `/api/...` paths.
3. **Inventoried other callers** — `backend/slack_bot/` (httpx to localhost), `scripts/` (curl + smoke tests), `backend/services/` (internal service-to-service), `backend/autonomous_loop.py` (cycle calls).
4. **Classified each route** into HOT (active caller found) / HARNESS (only smoke/CLI uses) / DEAD-CANDIDATE (no caller anywhere) / CONSERVATIVE-KEEP (low-confidence dead; could be revived).

## Total inventory

**116 routes across 18 router files + `backend/main.py`.**

| File | Prefix | Routes | HOT | HARNESS | DEAD-CANDIDATE | CONSERVATIVE-KEEP |
|------|--------|-------:|----:|--------:|---------------:|------------------:|
| `analysis.py` | `/api/analysis` | 2 | 2 | — | — | — |
| `backtest.py` | `/api/backtest` | 25 | 19 | 4 | 1 | — |
| `charts.py` | `/api/charts` | 2 | 2 | — | — | — |
| `cost_budget_api.py` | `/api/cost-budget` | 2 | — | — | 1 | 1 |
| `harness_autoresearch.py` | `/api/harness` | 1 | 1 | — | — | — |
| `investigate.py` | `/api` | 1 | 1 | — | — | — |
| `job_status_api.py` | `/api/jobs` | 2 | — | — | — | 2 |
| `mas_events.py` | `/api/mas` | 6 | 3 | 1 | 1 | 1 |
| `monthly_approval_api.py` | `/api/harness/monthly-approval` | 2 | — | — | — | 2 |
| `observability_api.py` | `/api/observability` | 2 | — | — | — | 2 |
| `paper_trading.py` | `/api/paper-trading` | 21 | 21 | — | — | — |
| `performance_api.py` | `/api/perf` | 9 | 5 | — | 2 | 2 |
| `portfolio.py` | `/api/portfolio` | 4 | 4 | — | — | — |
| `reports.py` | `/api/reports` | 6 | 5 | — | — | 1 |
| `settings_api.py` | `/api/settings` | 5 | 4 | — | — | 1 |
| `signals.py` | `/api/signals` | 13 | 5 | 1 | 4 | 3 |
| `skills.py` | `/api/skills` | 7 | — | — | 5 | 2 |
| `sovereign_api.py` | `/api/sovereign` | 4 | 4 | — | — | — |
| `main.py` | `/api` | 2 | 2 | — | — | — |
| **TOTAL** | | **116** | **78** | **6** | **14** | **18** |

## DEAD-CANDIDATE routes (13 high-confidence)

All have no caller in `frontend/src/`, `backend/slack_bot/`, `scripts/`, or
the rest of `backend/` (excluding the defining file itself).

| File | Line | Method | Path | Evidence summary |
|------|------|--------|------|------------------|
| `backtest.py` | 638 | GET | `/api/backtest/runs/{run_id}` | List endpoint `/runs` is called; parameterised single-run fetch is not in `api.ts` or any component. |
| `cost_budget_api.py` | 98 | GET | `/api/cost-budget/status` | The file itself documents this as "thin alias; canonical is `/today`." Zero callers anywhere. |
| `performance_api.py` | 37 | GET | `/api/perf/slow` | `/summary`, `/cache`, `/optimize/*` all called; `/slow` has no caller. |
| `performance_api.py` | 49 | GET | `/api/perf/llm/p95` | Same — `/summary` is called; `/llm/p95` is not. |
| `signals.py` | 119 | GET | `/api/signals/{ticker}/insider` | `api.ts` has a generic `getSignalDetail()` but no hardcoded `/insider` call. |
| `signals.py` | 124 | GET | `/api/signals/{ticker}/options` | Same — no hardcoded call. |
| `signals.py` | 141 | GET | `/api/signals/{ticker}/patents` | Same. |
| `signals.py` | 149 | GET | `/api/signals/{ticker}/earnings-tone` | Same. |
| `skills.py` | 52 | POST | `/api/skills/optimize` | Skill optimizer predates the unified `/api/perf/optimize`; superseded but not removed. |
| `skills.py` | 63 | POST | `/api/skills/stop` | Same — no caller. |
| `skills.py` | 74 | GET | `/api/skills/status` | Same. |
| `skills.py` | 81 | GET | `/api/skills/experiments` | Same. |
| `skills.py` | 88 | GET | `/api/skills/analysis` | Same. |

The 14th potential dead-candidate (`mas_events.py:61` — `/api/mas/events/ingest`)
is borderline: only invoked from `backend/agents/mas_events.py:138` when
`remote_url` is set (non-default for the single-Mac config). Reclassified
as CONSERVATIVE-KEEP given the multi-node future-proofing intent.

## Conservative-keep cluster (18 routes)

These are NOT recommended for deletion despite low or no current caller
evidence. Reasons:

- **`/api/jobs/*` (2 routes)** — long-running job status endpoints; may be
  used by future progress-tracking UI.
- **`/api/observability/*` (2 routes)** — dual-route freshness alias added
  in 16.22; pinned by external-callable masterplan verification command.
- **`/api/harness/monthly-approval/*` (2 routes)** — gate-state endpoints
  pre-wired for the harness monthly-approval flow (not yet UI-rendered).
- **`/api/mas/events/ingest`** — multi-node future-proofing.
- **`/api/skills/agents`, `/api/skills/{agent_name}`** — may feed future
  skill-level reporting.
- **`/api/signals/{ticker}/{nlp-sentiment, anomalies, monte-carlo, quant-model}`**
  (4 routes) — overlap with the generic `getSignalDetail` and the
  smoke-test fixture at `scripts/smoketest/steps/frontend_tabs.py`
  implies they may be expected.
- **`/api/cost-budget/today`** — canonical sibling of the dead `/status`
  alias; called internally by the Slack bot watcher (BQ path, not HTTP),
  so HTTP unused but business logic active.
- **`/api/reports/...` (1 route), `/api/settings/...` (1 route)** —
  edge-case endpoints with low traffic but still wired.

## Recommendation

**No route deletions in this cycle.** Rationale:

1. **`/api/skills/*` cluster** (5 routes) may be revived if skill-level
   optimization returns to the roadmap. Deleting prematurely risks
   re-implementation cost.
2. **`/api/cost-budget/status`** is documented as an alias in its own
   file; it is safe to delete but low value to do so now.
3. **`/api/perf/slow` and `/api/perf/llm/p95`** are worth a second look
   — if the Performance tab has no slow-endpoint table, these routes
   are truly dead. Defer to a dedicated cleanup cycle that audits
   the Performance tab UI first.
4. **Signal sub-routes (`/insider`, `/options`, `/patents`, `/earnings-tone`)**
   may be plumbed back in when the analysis pipeline re-enables those
   skill agents. Conservative path: a comment in `signals.py` marking
   them as currently unconnected.
5. **`/api/backtest/runs/{run_id}`** — easy delete, but the list+detail
   pattern is conventional and may attract a frontend caller in the
   future.

This doc serves as a decision record + scaffolding for a future
deletion cycle once the reasons above resolve (either by deletion
authorization or by the routes being re-plumbed).

## Cycle scope

- **Doc-only deliverable:** this file.
- **Zero backend code changes.**
- **No route registrations modified.**

## Methodology caveats

- Generic `getSignalDetail(ticker, signal)` in `api.ts` could theoretically
  call any sub-path; classification of `/insider/options/patents/earnings-tone`
  as dead assumes call-site grep accurately reflects all current usage.
- Slack bot uses `httpx` to localhost; some endpoints (e.g. `/api/health`)
  are reached but not enumerated explicitly in the inventory above.
- Smoke tests in `scripts/smoketest/` use a stub registry; some routes
  may be smoke-tested but not user-facing.

## Future cleanup cycle (proposal)

When ready to delete, prioritize:
1. `/api/skills/*` (5 routes) — biggest cluster, clear ownership boundary.
2. `/api/cost-budget/status` — explicitly self-documented as alias.
3. `/api/perf/slow` + `/api/perf/llm/p95` — pair-delete after Performance UI confirmation.
4. `/api/backtest/runs/{run_id}` — single deletion if list-only pattern preferred.
5. Signal sub-routes (`/insider/options/patents/earnings-tone`) — only if confirmed never to be re-enabled.

Total potential cleanup: 13 routes + their handler functions + their
test fixtures (where present).
