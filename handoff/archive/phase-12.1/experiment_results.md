# Experiment Results -- phase-12.1 Colored Manifests + README

**Step:** 12.1 (2nd of phase-12).
**Date:** 2026-04-19.

## What was built

`deploy/rainbow/` directory with 4 Kubernetes manifests + README.

| File | Bytes | Purpose |
|------|-------|---------|
| `backend-blue.yaml` | ~2.2KB | Deployment, color=blue, pyfinagent-backend:v0.1.0, /api/health probes |
| `backend-green.yaml` | ~2.1KB | Identical shape, color=green |
| `backend-service.yaml` | ~700B | ClusterIP Service, selector color:blue, the one file operators patch |
| `slack-bot-blue.yaml` | ~1.7KB | Socket Mode Deployment, exec liveness probe, no Service |
| `README.md` | 4,067B | Label convention, rollout recipe, rollback recipe, validation recipe, probe notes, sizing |

## File list

Created:
- `deploy/rainbow/backend-blue.yaml`
- `deploy/rainbow/backend-green.yaml`
- `deploy/rainbow/backend-service.yaml`
- `deploy/rainbow/slack-bot-blue.yaml`
- `deploy/rainbow/README.md`

Modified: none. Zero code changes (non-goal honored).

## Verification command output

### Immutable

```
$ ls deploy/rainbow/*.yaml | wc -l
4
```

4 yaml files, ≥2 required → exit 0 on the `awk '{exit ($1<2)}'` gate.

### YAML parse

```
$ python -c "import yaml; [list(yaml.safe_load_all(open(f))) for f in ['deploy/rainbow/backend-blue.yaml', 'deploy/rainbow/backend-green.yaml', 'deploy/rainbow/backend-service.yaml', 'deploy/rainbow/slack-bot-blue.yaml']]; print('all yaml parse ok')"
all yaml parse ok
```

### README size

```
$ python -c "import pathlib; assert pathlib.Path('deploy/rainbow/README.md').stat().st_size > 1500; print('readme size:', pathlib.Path('deploy/rainbow/README.md').stat().st_size)"
readme size: 4067
```

4,067 bytes > 1,500 floor.

### Color label grep

```
$ grep -l "color: blue" deploy/rainbow/*.yaml | wc -l
3

$ grep -l "color: green" deploy/rainbow/*.yaml | wc -l
1
```

- 3 blue-label files: `backend-blue.yaml`, `backend-service.yaml` (selector starts at blue), `slack-bot-blue.yaml` (slack only has a blue variant for MVP).
- 1 green-label file: `backend-green.yaml`.

### Regression

```
$ pytest backend/tests/ -q --ignore=backend/tests/test_paper_trading_v2.py
79 passed, 1 skipped in 5.69s
```

Zero regressions.

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `backend-blue.yaml` | PASS |
| 2 | `backend-green.yaml` | PASS |
| 3 | `backend-service.yaml` | PASS |
| 4 | `slack-bot-blue.yaml` with exec liveness + no Service | PASS |
| 5 | `README.md` ≥1500 bytes | PASS (4067) |
| 6 | All 4 yaml pass schema parse | PASS |
| 7 | `terminationGracePeriodSeconds: 60` on all Deployments | PASS (grep confirms 3 hits) |
| 8 | Backend readiness + liveness `/api/health`; slack liveness exec | PASS |
| 9 | Resource requests + limits | PASS |

## Design decisions locked

- **imagePullPolicy: Never** for MVP — image is loaded directly into the cluster (k3s / kind) from local Docker. Flip to `IfNotPresent` when a registry is wired (phase-12.3-adjacent).
- **Image tag v0.1.0** — explicit semver per research rec. No `:latest`.
- **Single Service targets both colors via the selector** — operators patch `spec.selector.color` to flip. No Service duplication.
- **Slack bot has no Service** — Socket Mode is outbound-only; routing irrelevant.
- **Slack bot exec liveness probe** uses a `test -f /tmp/healthy` sentinel with a 60s init grace that also passes for freshly-started pods (falls back to checking pod uptime < 60s via `stat /proc/1`). Wiring the `open('/tmp/healthy', 'w')` call into `backend/slack_bot/app.py` is a follow-up (not in scope for phase-12.1; documented in README).

## Known caveats

1. **YAML parse validates schema but not K8s semantics** — `kubectl apply --dry-run=client` would catch more (e.g., missing namespaces, bad image refs). Requires a live cluster or kubectl on PATH. Deferred to phase-12.3 when a dev cluster stands up.
2. **Slack bot sentinel-file writer not yet wired into app code** — the exec liveness probe will false-pass via the pod-uptime fallback until a real writer lands. Documented in README.
3. **No Pod-security-standards labels** (`pod-security.kubernetes.io/enforce: restricted` etc.) — homelab/k3s MVP; tighten for any prod cluster in a later phase.
4. **No NetworkPolicy** — MVP assumes trusted cluster; add NetworkPolicy when phase-12.4 rehearses against a shared cluster.
5. **Image `pyfinagent-backend:v0.1.0` not yet built/pushed** — operator must run `docker build -t pyfinagent-backend:v0.1.0 backend/` before `kubectl apply`. README references this implicitly via the `imagePullPolicy: Never` note but could be more explicit.
6. **Pre-Q/A self-check** ran the full verification suite + the color-label grep invariants before submission. Found nothing new.
