# Experiment Results -- phase-12.0 Rainbow Deploys Audit

**Step:** 12.0 (first of phase-12). Planning doc only; zero code.
**Date:** 2026-04-19.

## What was built

One doc, zero code changes: `docs/RAINBOW_DEPLOY_PLAN.md` (7,135 bytes, 3.5x the 2,000-byte immutable floor).

**Unusual flow**: the researcher co-authored the deliverable doc during the research gate — efficient because the plan doc IS the audit output. Main's role this cycle: write the contract, run deterministic checks, flag the co-authoring in the contract, spawn Q/A for independent content review.

## Doc sections (all 4 required + 3 bonus)

| # | Section | Coverage |
|---|---------|----------|
| 1 | Current Deploy Surface | Mac Mini runtime; `backend/Dockerfile` only container; zero K8s YAML; 5 CI workflows with no deploy steps |
| 2 | Scope — In/Out | backend + slack_bot IN; frontend + harness + MCP OUT |
| 3 | Color Palette Decision | 2 colors MVP (blue/green); expandable to 7; named over SHA for operator clarity |
| 4 | Rollback SLO | Service-selector `kubectl patch` under 30s; no pod restart; 2-10s propagation |
| 5 | Traffic Splitting Primitive | Plain Service selector (no mesh/Gateway API for MVP); HTTPRoute weights as future layer |
| 6 | Phase-12 Implementation Steps | High-level sketch of 12.1-12.4 |
| 7 | References | 7 read-in-full + 10 snippet-only sources |

## File list

Created:
- `docs/RAINBOW_DEPLOY_PLAN.md`

Modified: none. No backend/scripts/config/test changes.

## Verification command output

### Immutable (from contract + masterplan)

```
$ test -f docs/RAINBOW_DEPLOY_PLAN.md && python -c "import pathlib; p=pathlib.Path('docs/RAINBOW_DEPLOY_PLAN.md'); assert p.stat().st_size > 2000, 'too thin'; print('ok')"
ok
```

Exit 0. Size 7,135 bytes.

### Section headers present

```
$ grep -E "^#+ " docs/RAINBOW_DEPLOY_PLAN.md
# Rainbow Deploy Plan — pyfinagent
## 1. Current Deploy Surface
## 2. Scope — What Gets Rainbow; What Does Not
### In scope for phase-12 MVP
### Out of scope (phase-12)
## 3. Color Palette Decision
### Palette: 2 colors for MVP, expandable to 7
# Deployment
# Service (selector flipped per deploy)
## 4. Rollback SLO
## 5. Traffic Splitting Primitive
## 6. Phase-12 Implementation Steps (high-level)
## 7. References
```

All 4 required sections present + bonus sections. The `# Deployment` and `# Service` headers appear to be markdown-in-YAML-snippet rendering — informational, not structural.

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
79 passed, 1 skipped, 1 warning in 5.33s
```

Zero regressions. Same 79p/1s baseline as every phase-11 cycle.

### Scope (no code change)

```
$ git diff --name-only HEAD -- backend/ scripts/ frontend/ | wc -l
0
```

Zero code-file changes. Non-goal honored: this is a planning cycle.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `docs/RAINBOW_DEPLOY_PLAN.md` >2000 bytes | PASS (7,135) |
| 2 | All 4 required sections present | PASS (+ 3 bonus) |
| 3 | No code changes | PASS |
| 4 | No backend regressions | PASS (79p/1s) |

## Known caveats

1. **Researcher co-authored the deliverable doc** during the research gate. Efficient given the step is planning-only, but unusual flow vs other cycles where Main wrote the deliverable. Q/A must review the doc's content independently (not just existence + size). Flagged here + in the contract.
2. **Traffic primitive choice (plain Service selector vs Gateway API HTTPRoute)** is a MVP decision; if canary ramp is needed later, HTTPRoute weights can be layered on without a mesh. The doc defers that decision to phase-12.3.
3. **No K8s cluster exists yet** — the plan doc assumes a k3s / kind / minikube dev cluster will stand up in phase-12.1. The runbook concept is cluster-agnostic; real execution requires operator to provision.
4. **Pre-Q/A self-check ran**: immutable verify + regression + section grep before submission. No issues found.
5. **Phase-12.4 scope shift**: plan doc notes phase-12.4's "first real migration" is no longer Vertex (phase-11 already shipped without Rainbow). Next natural candidate: whatever the next risky change is (a SDK bump, a model-version flip, etc.). Doc notes this but doesn't name a specific candidate.
