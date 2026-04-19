# Rainbow Deploy Manifests -- pyfinagent

**Scope:** phase-12.1 deliverable. Two-color (blue/green) Kubernetes manifests for the backend + Slack bot, per `docs/RAINBOW_DEPLOY_PLAN.md`.

## File map

| File | Purpose |
|------|---------|
| `backend-blue.yaml` | Deployment with `color: blue`. 1 replica, /api/health probes. |
| `backend-green.yaml` | Identical Deployment shape, `color: green`. |
| `backend-service.yaml` | Single Service with selector `color: blue` as the live color. **The one file operators patch to flip a release.** |
| `slack-bot-blue.yaml` | Slack bot Deployment (Socket Mode, no Service, exec liveness). Blue-only for MVP. |

All Deployments use `image: pyfinagent-backend:v0.1.0` + `imagePullPolicy: Never`. Upgrade `imagePullPolicy` to `IfNotPresent` once a registry is wired.

## Label convention

```
app:   pyfinagent-backend    # shared across both colors
color: blue | green          # flip target for Service selector
```

Selector in `backend-service.yaml` matches **both** labels, so the Service routes to whichever color is currently the `color:` value. Flipping the color is the full "promote" action.

## Rollout recipe (blue -> green)

```
# 1. Pre-deploy the new (green) Deployment. Old blue stays up.
kubectl apply -f deploy/rainbow/backend-green.yaml

# 2. Wait for green pods to be Ready.
kubectl rollout status deployment/pyfinagent-backend-green --timeout=180s

# 3. Flip the Service selector from blue -> green.
kubectl patch service pyfinagent-backend \
  -p '{"spec":{"selector":{"color":"green"}}}'

# 4. Wait for selector propagation (2-10s) and verify.
curl http://<service-cluster-ip>:8000/api/health

# 5. Keep blue up for ~1 hour to allow TCP/SSE drain. After that,
#    scale replicas to 0 or delete backend-blue.yaml.
```

## Rollback recipe (under 30 seconds)

```
# Flip the selector back -- no pod restart required.
kubectl patch service pyfinagent-backend \
  -p '{"spec":{"selector":{"color":"blue"}}}'

# Verify.
curl http://<service-cluster-ip>:8000/api/health
```

If blue pods were already deleted, re-apply:

```
kubectl apply -f deploy/rainbow/backend-blue.yaml
kubectl rollout status deployment/pyfinagent-backend-blue --timeout=180s
kubectl patch service pyfinagent-backend \
  -p '{"spec":{"selector":{"color":"blue"}}}'
```

## Validation recipe

Before applying any yaml change:

```
# Offline schema validation (no cluster required):
python -c "import yaml; [yaml.safe_load_all(open(f)) for f in ['deploy/rainbow/backend-blue.yaml', 'deploy/rainbow/backend-green.yaml', 'deploy/rainbow/backend-service.yaml', 'deploy/rainbow/slack-bot-blue.yaml']]; print('ok')"

# With a live cluster (k3s / kind / minikube):
kubectl apply --dry-run=client -f deploy/rainbow/

# Semantic validation (needs cluster RBAC):
kubectl apply --dry-run=server -f deploy/rainbow/
```

## Probe notes

**Backend:** `/api/health` is the canonical live-check endpoint. See `backend/main.py:294`. It's in `_PUBLIC_PATHS` (auth-skipped) and silenced from access logs.

**Slack bot:** Socket Mode is outbound-only; no HTTP port. Liveness uses an `exec` probe looking for a sentinel file `/tmp/healthy`. The app must write this file when the WebSocket connection is established; wiring the `open('/tmp/healthy', 'w')` call into `backend/slack_bot/app.py` is a follow-up (phase-12.2 or 12.3). Until then the probe's 60s grace window lets a healthy pod survive during the first minute while it connects to Slack.

## Resource sizing

- Backend: 500m CPU request / 1 CPU limit; 512Mi memory request / 1Gi limit.
- Slack bot: 250m / 500m CPU; 256Mi / 512Mi memory.

Sized for a homelab k3s cluster. Adjust for production load in phase-12.3 canary sizing.

## Phase-12 step handoff

- **phase-12.2** (next): writes `promote.py` + `rollback.py` CLI scripts that wrap the two `kubectl patch` commands above with `--dry-run` support.
- **phase-12.3**: adds canary split + SLO diff tooling (weighted Service or Gateway API HTTPRoute) on top of this MVP.
- **phase-12.4**: first real migration rehearsal; candidate TBD after 12.3 lands.
