# Rainbow Deploy Plan — pyfinagent

_Phase 12.0 — Research gate output. Authored: 2026-04-19._

---

## 1. Current Deploy Surface

pyfinagent runs today on a Mac Mini (OpenClaw runtime host, `Europe/Oslo`) with no Kubernetes infrastructure. All processes start directly via CLI.

| Component | Current runtime | Container? | K8s today? |
|-----------|----------------|-----------|-----------|
| Backend API | `uvicorn backend.main:app --port 8000` | YES — `backend/Dockerfile` exists | NO |
| Frontend | `npm run dev` (port 3000, Next.js 15) | YES — `frontend/Dockerfile` exists (`FROM node:20-alpine`, 27 lines), but not yet wired into any prod runtime | NO |
| Harness | `python scripts/harness/run_harness.py` (CLI, cron-driven) | NO | NO |
| Slack bot | `python -m backend.slack_bot.app` (Socket Mode, persistent WebSocket) | NO | NO |
| MCP servers | In-process with uvicorn (same pod as backend) | Covered by backend | NO |
| BigQuery / GCS | Managed GCP services — no deploy needed | N/A | N/A |

CI/CD (`.github/workflows/`) contains 5 workflows: Claude Code review trigger, pip-audit, governance-lint, limits-tag-enforcement, and claude-code-review. Zero deploy/push steps exist in any workflow today.

---

## 2. Scope — What Gets Rainbow; What Does Not

### In scope for phase-12 MVP

**Backend API** (`backend/`) — primary target. Stateful long-running connections include any SSE streams from the analysis pipeline. `backend/Dockerfile` already exists. This is the highest-value Rainbow candidate and the logical first step.

**Slack bot** (`backend/slack_bot/app.py`) — secondary target. Runs Socket Mode (persistent WebSocket to Slack). This is the textbook Rainbow use case: old pods must drain naturally so in-flight message handlers complete; forced restarts lose messages. Once backend gets a Deployment, Slack bot can be separated into its own Deployment or co-deployed in the same pod at operator discretion.

### Out of scope (phase-12)

| Component | Reason |
|-----------|--------|
| Frontend | `frontend/Dockerfile` exists but Next.js is currently served via `npm run dev` (development server, not `next start` behind a prod runtime). Productionizing it is a prerequisite for Rainbow that's out of scope here; revisit when the backend Rainbow pattern has baked in. |
| Harness (`run_harness.py`) | CLI batch process; cron-invoked; not a server; Rainbow adds no value (no persistent connections) |
| MCP servers | In-process with uvicorn; covered by backend Rainbow by default |
| BigQuery / GCS | Managed services; no deploy artifact to Rainbow-ize |

---

## 3. Color Palette Decision

### Palette: 2 colors for MVP, expandable to 7

The Dimcheff reference uses git-SHA-hex-prefixes as "colors" (infinite palette, no exhaustion). For pyfinagent, named human-readable colors are preferred for operator clarity — matching the Dimcheff spirit while keeping runbooks legible.

**MVP (phase-12):** `blue` and `green`.
- Blue = currently live.
- Green = new deployment candidate.
- Service selector flips from `blue` to `green` on promotion.
- Rollback = flip selector back to `blue`.

**Full Dimcheff (future phase):** extend to 7 colors — red, orange, yellow, green, blue, indigo, violet — once deploy frequency justifies N>2 simultaneous active slots. Each color slot maps to a distinct Deployment object. CI assigns the next unused color on each deploy cycle.

**Label convention:**

```yaml
# Deployment
metadata:
  name: pyfinagent-backend-blue
  labels:
    app: pyfinagent-backend
    color: blue

# Service (selector flipped per deploy)
spec:
  selector:
    app: pyfinagent-backend
    color: blue   # patched to "green" on promotion
```

Promotion command (no pod restart, no downtime):

```bash
kubectl patch service pyfinagent-backend \
  -p '{"spec":{"selector":{"color":"green"}}}'
```

Rollback command:

```bash
kubectl patch service pyfinagent-backend \
  -p '{"spec":{"selector":{"color":"blue"}}}'
```

---

## 4. Rollback SLO

**Service-selector flip completes in under 30 seconds. No pod restart required.**

Kubernetes propagates Service selector changes to kube-proxy on all nodes within seconds (typically 2-10s in a single-node or small cluster). Existing TCP connections to the old Deployment's pods continue to be served until they close naturally — no forced termination.

Rollback procedure:

1. Run the rollback `kubectl patch` command above.
2. Verify traffic is routing to the prior color: `kubectl get endpoints pyfinagent-backend`.
3. Confirm backend healthcheck: `curl http://<service>/health`.
4. If the old (rollback target) Deployment's pods are still Running, rollback is complete — total elapsed time should be under 30 seconds.

Pod drain policy: set `terminationGracePeriodSeconds: 60` on all Deployments to allow in-flight SSE streams and Slack Socket Mode messages to complete before pod shutdown. This is independent of the selector flip and does not affect rollback speed.

---

## 5. Traffic Splitting Primitive

For phase-12 MVP: **plain Kubernetes Service selector flip** (no service mesh, no Gateway API).

Rationale: lowest blast radius for a greenfield cluster. No additional CRDs, no ingress controller dependency, no Istio/Linkerd. A single `kubectl patch` is the entire traffic-switch mechanism. This matches the Dimcheff reference implementation exactly.

Future option (if canary ramp is required): layer **Gateway API HTTPRoute weights** (`backendRefs[].weight`) on top. Supported by Nginx Gateway Fabric, Traefik, and Envoy Gateway without a service mesh. Allows progressive traffic shift (e.g., 5% green, 95% blue) before full flip. Not required for phase-12 MVP.

---

## 6. Phase-12 Implementation Steps (high-level)

1. Create `deploy/` directory with K8s manifests:
   - `deploy/backend-blue.yaml` — Deployment + Service template
   - `deploy/backend-green.yaml` — Deployment template (Service shared)
   - `deploy/service.yaml` — single Service with selector `color: blue`
2. Add `terminationGracePeriodSeconds: 60` to both Deployment specs.
3. Add a CI workflow step (`.github/workflows/deploy.yml`) that:
   - Builds and pushes `backend/Dockerfile` to a container registry (GCR or Docker Hub).
   - Applies the new-color Deployment via `kubectl apply`.
   - Waits for `kubectl rollout status`.
   - Patches the Service selector to the new color.
   - Tags the prior Deployment for deferred deletion (cleanup after N hours).
4. Document runbook in `docs/runbooks/rainbow-deploy-ops.md`.
5. Test rollback SLO: measure selector-flip time on target cluster; confirm <30s.

### Phase-12.4 scope reassignment (2026-04-19)

Phase-12.4 was originally scoped as "First real migration using Rainbow: Vertex -> google-genai cutover". Phase-11 (Vertex migration) shipped in full on 2026-04-19 WITHOUT Rainbow (direct in-place migration; 79p/1s regression; zero incidents). The Vertex migration is therefore no longer available as phase-12.4's first rehearsal candidate.

Replacement candidate for phase-12.4 (to be picked after phase-12.1-12.3 land):

- **Best option:** next risky SDK bump or model-version flip (e.g., `anthropic==0.87.0 -> 0.96.0` minor, or a Claude Opus 4.7 -> 4.8 rollout). Short-duration bumps that typically have silent-breakage risk (like the ThinkingConfig case in phase-11.3) are the textbook Rainbow rehearsal.
- **Fallback:** a contrived "dummy" migration (identical manifests, color-flip-only) purely to prove the machinery works end-to-end on the target cluster.

Masterplan step 12.4 description will be updated to reflect this reassignment when phase-12.3 closes and a concrete candidate is identified.

---

## 7. References

- Brandon Dimcheff, "Rainbow Deploys with Kubernetes" (2018): https://brandon.dimcheff.com/2018/02/rainbow-deploys-with-kubernetes/
- Dimcheff demo repo: https://github.com/bdimcheff/rainbow-deploys
- Martin Fowler, "BlueGreenDeployment" (canonical): https://martinfowler.com/bliki/BlueGreenDeployment.html
- Kubernetes Gateway API — HTTP Traffic Splitting: https://gateway-api.sigs.k8s.io/guides/traffic-splitting/
- Release.com, "Rainbow Deployment: Why and How": https://release.com/blog/rainbow-deployment-why-and-how-to-do-it
- DEV.to (McClung), "Rainbow Deployment Why and How": https://dev.to/tmcclung/rainbow-deployment-why-and-how-to-do-it-32d3
- Codefresh, "Blue-Green vs Canary": https://codefresh.io/learn/software-deployment/blue-green-deployment-vs-canary-5-key-differences-and-how-to-choose/
