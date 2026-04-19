# Sprint Contract -- phase-12.0 Rainbow Deploys Audit + Scope Boundary

**Written:** 2026-04-19 PRE-commit.
**Step id:** `12.0` in phase-12.
**Immutable verification:** `test -f docs/RAINBOW_DEPLOY_PLAN.md && python -c "import pathlib; p=pathlib.Path('docs/RAINBOW_DEPLOY_PLAN.md'); assert p.stat().st_size > 2000, 'plan doc too thin'; print('ok')"`.

## Research-gate summary

Researcher envelope `{tier: moderate, external_sources_read_in_full: 7, snippet_only_sources: 10, urls_collected: 17, recency_scan_performed: true, internal_files_inspected: 6, gate_passed: true}`. Three-query-variant discipline confirmed (year-less + 2024-2026 recency scan + current-year; year-less surfaced Martin Fowler's blue/green post which a year-locked query misses).

Key findings locked from research:
- **Dimcheff pattern** uses git-SHA-as-color (first 6 chars hex); `color` label on Deployment; Service selector flip; old Deployments survive for TCP connection drain.
- **N>2 colors** buy unlimited deploy frequency + proper drain vs blue/green's 24h wait.
- **No 2024-2026 tool implements Rainbow natively** — Argo Rollouts, Flagger, Knative all stop at canary/blue-green. Always hand-rolled.
- **pyfinagent is greenfield for K8s**: `backend/Dockerfile` exists; zero K8s YAML, no `deploy/` dir, no CI deploy pipeline, no service mesh.
- **Lowest blast-radius primitive**: plain K8s Service selector `kubectl patch` — 2-10s propagation, no mesh / Gateway API needed for MVP.

Staked rec adopted: MVP = 2 colors (`blue`/`green`), named not SHA-derived for operator clarity, backend + slack_bot in scope, frontend + harness + MCP out of scope for phase-12 MVP.

## Deliverable: single doc

`docs/RAINBOW_DEPLOY_PLAN.md` pre-produced by the researcher during the research gate (efficient: the plan doc IS the audit output). 7,135 bytes — 3.5x the 2000-byte immutable floor.

Required sections (per masterplan phase-12 step 12.0 success_criteria):
1. Current deploy surface ✓ (section 1)
2. Scope (which components get Rainbow; which out of scope) ✓ (section 2)
3. Color palette decision ✓ (section 3 — 2 colors MVP, expandable to 7)
4. Rollback SLO ✓ (section 4 — Service-selector flip under 30s, no pod restart)

## Hypothesis

The plan doc's scope (backend + slack_bot IN; frontend + harness + MCP OUT) is right-sized for an MVP and leaves room for phase-12.1-12.4 to execute without architecture re-litigation.

## Success criteria

**Functional:**
1. `docs/RAINBOW_DEPLOY_PLAN.md` exists + >2000 bytes (immutable).
2. Doc has all 4 required sections (current surface, scope, palette, rollback SLO).
3. No code changes in this cycle (audit + plan only; phase-12.1 will start writing yaml).
4. No backend regressions.

**Correctness verification:**
- Immutable: shown above.
- Regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` → 79p/1s unchanged.
- Scope grep: `git diff --name-only` shows only `docs/RAINBOW_DEPLOY_PLAN.md`, `handoff/current/phase-12.0-*`, `handoff/current/{contract,experiment_results,evaluator_critique,research_brief}.md`, `.claude/masterplan.json`, `handoff/harness_log.md`.

## Non-goals

- NOT writing any K8s yaml (phase-12.1).
- NOT writing promote.py / rollback.py (phase-12.2).
- NOT implementing canary split (phase-12.3).
- NOT executing a real Rainbow migration (phase-12.4; coordinated with phase-11.4 already closed — so phase-12.4 becomes a rehearsal against the next migration candidate, not a Vertex cutover, since Vertex is already migrated outside Rainbow).
- NOT touching backend/frontend/harness code.

## Unusual flow note

This cycle the researcher co-authored the deliverable doc. Per CLAUDE.md, Main is still responsible for the contract + pre-Q/A validation + Q/A spawn. Q/A must verify the doc's content independently (not just existence + byte count).

## References

- `handoff/current/phase-12.0-research-brief.md` (envelope + research trail)
- Dimcheff 2018 post + demo repo
- Martin Fowler Blue-Green post
- Gateway API HTTPRoute docs
- Argo Rollouts / Flagger / Knative (all snippet-only; confirmed rainbow-not-supported)

## Researcher agent id

`a8a639ff7f747c385`
