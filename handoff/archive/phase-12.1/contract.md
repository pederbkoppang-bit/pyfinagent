# Sprint Contract -- phase-12.1 Colored Deployment + Service Manifests

**Written:** 2026-04-19 PRE-commit.
**Step id:** `12.1` in phase-12.
**Immutable verification:** `ls deploy/rainbow/*.yaml 2>/dev/null | wc -l | awk '{exit ($1<2)}'` — ≥2 yaml files under `deploy/rainbow/`.

## Research-gate summary

Researcher envelope `{tier: simple, external_sources_read_in_full: 5, snippet_only_sources: 7, urls_collected: 12, recency_scan_performed: true, internal_files_inspected: 3, gate_passed: true}`. Three-query compliance confirmed.

Staked recs adopted:
- Backend probe: `httpGet path: /api/health port: 8000` (confirmed endpoint at `backend/main.py:294`, in `_PUBLIC_PATHS`, suppressed from access logs). `initialDelaySeconds: 15` liveness, `5` readiness.
- Slack bot probe: no readiness (outbound-only), liveness via exec probe OR omitted for MVP; omit for v1 simplicity.
- Image tag: `pyfinagent-backend:v0.1.0` explicit semver, `imagePullPolicy: Never` for initial local-load, upgrade to `IfNotPresent` when registry wired.
- Raw manifests (no kustomize) per plan doc MVP decision.

## Hypothesis

Four yaml files under `deploy/rainbow/` (2 backend Deployments + 1 Service + 1 slack_bot Deployment) + a README documenting the palette/labels/rollout recipe, all passing `kubectl apply --dry-run=client`, closes the immutable verify and gives phase-12.2 the manifests it needs to write promote.py / rollback.py against.

## Success criteria

**Functional:**
1. `deploy/rainbow/backend-blue.yaml` (Deployment, color=blue, pyfinagent-backend:v0.1.0).
2. `deploy/rainbow/backend-green.yaml` (Deployment, color=green, same shape).
3. `deploy/rainbow/backend-service.yaml` (Service with selector `color: blue` as the initial live color; this is the file operators patch during a color flip).
4. `deploy/rainbow/slack-bot-blue.yaml` (Deployment for the slack_bot Socket Mode worker; no Service because it's outbound-only; liveness via `exec` probe on `/tmp/healthy` — kept simple).
5. `deploy/rainbow/README.md` (≥1,500 bytes): palette convention, label schema, rollout recipe, rollback recipe, validation recipe (`kubectl apply --dry-run=client -f deploy/rainbow/`).
6. All 4 yaml files pass a basic YAML schema parse (no live kubectl needed — use `python -c "import yaml; yaml.safe_load(open('...'))"` in the verification path).
7. `terminationGracePeriodSeconds: 60` set on all Deployments.
8. `readinessProbe` + `livenessProbe` on backend Deployments (httpGet /api/health). Liveness-only on slack_bot (exec probe).
9. `resources` requests + limits set (CPU 500m/1; memory 512Mi/1Gi for backend; half those for slack_bot).

**Correctness verification commands:**
- Immutable: `ls deploy/rainbow/*.yaml 2>/dev/null | wc -l | awk '{exit ($1<2)}'` exit 0 (contract target: 4 yaml).
- YAML parse: `python -c "import yaml; [yaml.safe_load_all(open(f)) for f in ['deploy/rainbow/backend-blue.yaml', 'deploy/rainbow/backend-green.yaml', 'deploy/rainbow/backend-service.yaml', 'deploy/rainbow/slack-bot-blue.yaml']]; print('ok')"` exit 0.
- README size: `python -c "import pathlib; assert pathlib.Path('deploy/rainbow/README.md').stat().st_size > 1500; print('ok')"` exit 0.
- Backend regression: `pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py` → 79p/1s.
- Grep invariants: `grep -l "color: blue" deploy/rainbow/*.yaml | wc -l` ≥ 2; same for `color: green` ≥ 1.

**Non-goals:**
- NOT running against a real K8s cluster (phase-12.3+).
- NOT writing promote.py (phase-12.2).
- NOT wiring CI/CD deploy workflow (phase-12.4-adjacent).
- NOT pushing images to a registry (operator action).
- NOT frontend manifests (out of phase-12 scope).

## Plan

1. Create `deploy/rainbow/` directory.
2. Write the 4 yaml files.
3. Write `deploy/rainbow/README.md`.
4. Run YAML parse + size + grep invariants.
5. Run backend regression.

## References

- `docs/RAINBOW_DEPLOY_PLAN.md` (phase-12.0 plan)
- `handoff/current/phase-12.1-research-brief.md`
- `backend/main.py:294` (/api/health endpoint)
- `backend/Dockerfile` (base image + port)

## Researcher agent id

`ab151d816307ce1f4`
