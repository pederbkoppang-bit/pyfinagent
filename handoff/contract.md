# Cost Reduction Analysis — Harness Method

## Hypothesis
GCP costs can be reduced by 60-80% by eliminating unused resources and switching to cheaper alternatives.

## Current State (March 2026)
Total GCP: $176.18/month
- Redis Memorystore M1: $76.07 (43%) — STOPPED March 7
- Vertex AI (Gemini 2.5 Pro): $69.21 (39%) — ALL usage on March 7 only
- Compute Engine (7× VPC connectors): $18.87 (11%) — STOPPED March 7
- Networking (Network Intelligence): $33.07 — credits covered most
- Other (Dataplex, Artifact Registry, Storage): ~$1

## Key Findings

### 1. Redis Memorystore ($76/mo) — ELIMINATE
- **What:** Basic M1 instance in Iowa, used for Celery task queue
- **Status:** Last usage March 7 — already stopped/deleted
- **But:** Still accrued $76 for 7 days of March
- **Action:** Verify it's deleted. If still running → DELETE IMMEDIATELY
- **Codebase:** `backend/tasks/analysis.py` and `settings.py` reference `redis://localhost:6379` — this is for Cloud Run deployment, not our Mac Mini local setup
- **Savings:** $76/mo → $0 (we run locally now, no Redis needed)

### 2. VPC Connectors / Compute Engine ($19/mo) — ELIMINATE
- **What:** 7× `aet-uscentral1-pyfinagent--connector-*` E2 instances
- **Purpose:** Serverless VPC Access connectors for Cloud Run → Redis connectivity
- **Status:** Last usage March 7 — should already be stopped
- **Action:** Verify all connectors deleted. They exist only to connect Cloud Run to Redis.
- **Savings:** $19/mo → $0

### 3. Networking ($33/mo) — ELIMINATE
- **What:** Network Intelligence Center (Analyzer, Topology, Performance)
- **Status:** Credits covered most, but still generating usage
- **Action:** Disable Network Intelligence Center in GCP Console
- **Savings:** ~$33/mo → $0 (currently covered by credits, but credits will run out)

### 4. Vertex AI ($69/mo) — OPTIMIZE
- **What:** Gemini 2.5 Pro — 1.1M input tokens + 54K thinking output + 25K output
- **When:** ALL on March 7 only (single day of heavy usage)
- **Why:** This was likely analysis runs during development
- **Optimization options:**
  a. Use Gemini 2.0 Flash instead of 2.5 Pro (10x cheaper per token)
  b. Use Gemini Flash for screening, Pro only for final synthesis
  c. Cache common prompts to reduce input tokens
  d. Set daily cost caps in code
- **Current run rate:** $0/day when not running analyses. $69 was a one-day spike.

## Projected Monthly Costs After Cleanup

| Service | Before | After | Savings |
|---------|--------|-------|---------|
| Redis Memorystore | $76 | $0 | -$76 |
| Compute (VPC connectors) | $19 | $0 | -$19 |
| Networking | $33 | $0 | -$33 |
| Vertex AI | $69 | ~$5-20 | -$49 to -$64 |
| Other (storage, etc) | $1 | $1 | $0 |
| **GCP Total** | **$176** | **$6-21** | **-$155 to -$170** |
| Claude Max | $200 | $200 | $0 |
| Other fixed | $35 | $35 | $0 |
| **Grand Total** | **$411** | **$241-256** | **-$155 to -$170** |

## Success Criteria
1. Redis Memorystore confirmed deleted (verify in GCP Console)
2. VPC connectors confirmed deleted
3. Network Intelligence Center disabled
4. Vertex AI usage switched to Flash where possible
5. Monthly GCP cost drops below $25

## Actions Needed from Peder
1. **Verify/delete Redis instance** in GCP Console → Memorystore
2. **Verify/delete VPC connectors** in GCP Console → VPC Network → Serverless VPC Access
3. **Disable Network Intelligence Center** in GCP Console → Network Intelligence
4. These are GCP Console actions — Ford can't do them from CLI without gcloud auth
