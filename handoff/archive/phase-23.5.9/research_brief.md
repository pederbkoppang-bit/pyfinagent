## Research: phase-23.5.9 — Cron job verification: nightly_mda_retrain (slack_bot, phase-9.4)

Tier assumed: **simple** (stated by caller). Search-query three-variant discipline applied.

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://scikit-learn.org/stable/modules/permutation_importance.html | 2026-05-09 | Official doc | WebFetch full | "Permutation feature importance... also known as MDA. It can be computed on unseen data. Only compute for models with good validation scores." |
| https://www.robustperception.io/idempotent-cron-jobs-are-operable-cron-jobs/ | 2026-05-09 | Authoritative blog | WebFetch full | "multiple runs of a job result in the same state as a single run"; checkpoint-based recovery + frequency doubling as resilience pattern |
| https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning | 2026-05-09 | Official doc (Google Cloud) | WebFetch full | "ensure the new model produces better performance than the current model before promoting it to production" (MLOps level 1 CT gates) |
| https://www.comet.com/site/blog/importance-of-machine-learning-model-retraining-in-production/ | 2026-05-09 | Authoritative blog | WebFetch full | Four retraining triggers: performance-based, data-driven, interval-based, manual. Threshold-gated automation is the recommended pattern. |
| https://temporal.io/blog/idempotency-and-durable-execution | 2026-05-09 | Authoritative blog (Temporal) | WebFetch full | Idempotency keys tied to operation identity + execution window; deterministic workflow IDs for cron jobs; "Reject Duplicate" policies prevent overlapping runs |
| https://lakefs.io/mlops/mlops-pipeline/ | 2026-05-09 | Authoritative blog | WebFetch full | Model versioning + revert capabilities for rollback; promotion gates via CI/CD with pre-merge hooks; idempotent pipelines as a foundational requirement |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.neovasolutions.com/2026/03/12/ml-model-drift-monitoring-a-continuous-evaluation-framework/ | Blog | WebFetch returned truncated/empty content |
| https://aerospike.com/blog/model-drift-machine-learning/ | Blog | Snippet only — sufficient context from other drift sources |
| https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/ | Blog | Snippet only |
| https://21devs.com/model-monitoring/ | Blog | Snippet only |
| https://www.clarifai.com/blog/mlops-best-practices | Blog | Snippet only |
| https://medium.com/@surajs78/why-is-my-job-running-twice-understanding-idempotency-and-deduplication-in-distributed-systems-d56edbcad051 | Blog | Snippet only — idempotency covered in full via temporal.io |
| https://traveling-coderman.net/code/node-architecture/idempotent-cron-job/ | Community | Snippet only |
| https://medium.com/@annxsa/when-your-php-cron-job-runs-twice-what-really-happens-and-how-to-stop-it-b377db2e84a1 | Community (2026) | Snippet only — PHP-specific, lower relevance |
| https://www.conduktor.io/glossary/model-drift-in-streaming | Vendor glossary | Snippet only |

---

### Recency scan (2024-2026)

Searched: "nightly ML model retrain cron job 2025", "idempotency ML retrain cron guard 2026", "feature importance retrain production ML pipeline 2026".

Findings: The neovasolutions.com article (2026-03-12) specifically covers a continuous ML drift-monitoring evaluation framework with a 2026 publication date but its content was unreadable via WebFetch (truncated). The Medium article about PHP cron jobs firing twice (2026-03) and the Temporal idempotency blog are the freshest usable sources. No 2025-2026 work supersedes the canonical idempotency-key + promotion-gate pattern; the industry has converged on this pattern and phase-9 already implements it correctly.

---

### Queries run (three-variant discipline)

1. Current-year frontier: "scikit-learn permutation_importance MDA feature importance retrain nightly cron 2026"
2. Last-2-year window: "nightly ML model retrain cron job operational patterns drift detection rollback 2025"
3. Year-less canonical: "feature importance retrain production ML pipeline operational best practices"
4. Additional: "idempotency ML retrain job cron guard duplicate execution 2025 2026"

---

### Key findings

1. **MDA = permutation feature importance**: MDA (Mean Decrease Accuracy) is the canonical alias for scikit-learn's `permutation_importance`. The job name is accurate. The literature recommends computing it on held-out data, not training data, to avoid overfitting artifacts. (Source: scikit-learn docs, 2026, https://scikit-learn.org/stable/modules/permutation_importance.html)

2. **Promotion-gate pattern is MLOps standard**: The Google Cloud MLOps guide explicitly documents that automated retraining must gate promotion by comparing the new model to the production baseline before committing. The phase-9.4 `PromotionGate` (DSR >= 0.95, PBO <= 0.20) is precisely this pattern. (Source: Google Cloud Architecture, 2026, https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

3. **Idempotency keys are the canonical duplicate-fire guard**: An idempotency key scoped to `{job_name}:{iso_date}` means a second fire on the same calendar day returns the cached result without re-executing. The `IdempotencyStore.seen()` / `.mark()` pattern at `job_runtime.py:32-36` matches the documented pattern exactly. (Source: Temporal, https://temporal.io/blog/idempotency-and-durable-execution; Robust Perception, https://www.robustperception.io/idempotent-cron-jobs-are-operable-cron-jobs/)

4. **coalesce=True + misfire_grace_time prevents restart-era duplicate fires**: APScheduler's `coalesce=True` collapses multiple missed ticks into one. Combined with `misfire_grace_time=3600`, a slack-bot restart does not immediately fire a missed tick if it was more than 1 hour ago. (Source: scheduler.py:518-525)

5. **Production-stub pattern defers real side-effects**: `_default_train()` returns a stub dict (`dsr=0.80, pbo=0.30, sharpe=1.0`). With DSR 0.80 < gate threshold 0.95, the default stub will NEVER promote. The job runs, heartbeats, evaluates, rejects, and exits cleanly — no data is committed. This is intentional per the phase-9 deferred-production pattern. (Source: `nightly_mda_retrain.py:46-48`, `autoresearch/gate.py:35-36`)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/nightly_mda_retrain.py` | 52 | Phase-9.4 job: train, evaluate via PromotionGate, commit if promoted | Active, production-stub default |
| `backend/slack_bot/job_runtime.py` | 117 | Heartbeat context manager + IdempotencyStore + IdempotencyKey | Active, used by all phase-9 jobs |
| `backend/autoresearch/gate.py` | 62 | PromotionGate (DSR >= 0.95, PBO <= 0.20) + cpcv_folds | Active, pure, no side-effects |
| `backend/slack_bot/scheduler.py` (lines 491-548) | ~60 | `_PHASE9_JOB_IDS` tuple + `register_phase9_jobs()` | Active; `nightly_mda_retrain` at hour=3, misfire_grace_time=3600, coalesce=True |
| `backend/api/job_status_api.py` (lines 50-100) | 50+ | `_JOB_NAMES` registry, bridge merge, `JobStatus` model, `record_heartbeat()` | Active; `nightly_mda_retrain` at line 58 |
| `tests/slack_bot/test_nightly_mda_retrain.py` | 53 | 3 tests: promote-and-commit, reject-no-commit, idempotent-same-day | Active, full coverage of all three branches |

---

### Consensus vs debate (external)

Consensus: Promotion gates before committing retrained models are the documented standard (Google MLOps, Comet, lakeFS). Idempotency keys scoped to the execution window are the canonical duplicate-fire guard (Temporal, Robust Perception). No debate in the literature.

---

### Pitfalls (from literature)

1. **Computing MDA on training data instead of held-out data**: scikit-learn warns that impurity-based importances (MDI) overfit; the same risk applies if permutation importance is computed on training data. The job's `_default_train()` stub does not specify which split it uses — the production `train_fn` must use a held-out set.
2. **Correlated features produce misleadingly low MDA**: If two features are collinear, permuting one leaves the model access via the other. The production `train_fn` should cluster correlated features before selecting.
3. **In-memory IdempotencyStore resets on process restart**: The global `_GLOBAL_STORE` in `job_runtime.py:39` is in-memory. If the slack-bot restarts between the first fire and a second fire of the same day, the idempotency guard is not effective. The production stub for the store defers BQ/Redis backing — this is a known gap under the deferred-production pattern.
4. **Stub result DSR=0.80 always blocked by gate**: This is intentional, but it means the live status will always show `promoted=False` until the production `train_fn` is wired. The criterion tests `status != "manifest"` and `next_run is not None` — these are agnostic to whether the model actually promotes.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| External finding | File:line in pyfinagent |
|-----------------|------------------------|
| Promotion-gate before commit (Google MLOps) | `nightly_mda_retrain.py:37-42` — PromotionGate.evaluate() → commit_fn only if promoted |
| Idempotency key scoped to daily window (Temporal/RobustPerception) | `job_runtime.py:45-48`, `nightly_mda_retrain.py:28` — `IdempotencyKey.daily(JOB_NAME, day=day)` |
| Idempotency check before work (Temporal "check pre-existing results") | `job_runtime.py:92-98` — `if idempotency_key is not None and s.seen(idempotency_key): yield state; return` |
| coalesce + misfire_grace_time (Robust Perception: restart resilience) | `scheduler.py:524-525` — `"misfire_grace_time": 3600, "coalesce": True` for nightly_mda_retrain |
| Production-stub pattern with deferred real commit (Comet: manual retraining trigger) | `nightly_mda_retrain.py:46-48` — `_default_train()` returns stub; production `commit_fn=None` means no commit even if promoted |

---

### Function body (verbatim from nightly_mda_retrain.py:19-51)

```python
def run(
    *,
    train_fn: Callable[[], dict[str, Any]] | None = None,
    gate: PromotionGate | None = None,
    commit_fn: Callable[[dict], None] | None = None,
    store: IdempotencyStore | None = None,
    day: str | None = None,
) -> dict[str, Any]:
    """Retrain; evaluate via gate; commit baseline only if promoted."""
    key = IdempotencyKey.daily(JOB_NAME, day=day)
    g = gate or PromotionGate()
    result: dict[str, Any] = {"promoted": False, "key": key, "skipped": False, "reason": None}

    with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:
        if state.get("skipped"):
            result["skipped"] = True
            return result
        new_model = (train_fn or _default_train)()
        verdict = g.evaluate(new_model)
        result["promoted"] = verdict["promoted"]
        result["reason"] = verdict.get("reason")
        result["trial_id"] = new_model.get("trial_id")
        if verdict["promoted"] and commit_fn is not None:
            commit_fn(new_model)
    return result


def _default_train() -> dict[str, Any]:
    """Injected in tests; production invokes backend/backtest/quant_optimizer.py."""
    return {"trial_id": "stub_nightly", "dsr": 0.80, "pbo": 0.30, "sharpe": 1.0}
```

---

### Three answers for the decision block

**1. Docker-alias bug?**
NO. The job imports only from `backend.autoresearch.gate` and `backend.slack_bot.job_runtime` — both are local Python modules, no Docker aliases, no HTTP calls to `http://backend:8000` or any external API. The pattern matches the phase-9 canonical shape exactly.

**2. heartbeat() wired correctly?**
YES. `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:` at `nightly_mda_retrain.py:32` passes the job name, a daily idempotency key, and the injectable store. The `heartbeat()` context manager (`job_runtime.py:66-114`) emits started/ok/failed events, enforces the idempotency skip, and marks the key on success. The state dict skipped-check at line 33-35 is also correctly wired: if the key was already seen, `heartbeat` yields `{skipped: True}` and the function returns early without calling `train_fn`. All three test cases (`test_good_model_promotes_and_commits`, `test_rejected_model_does_not_commit`, `test_idempotent_same_day`) pass with the current implementation.

**3. Criterion is TRUE liveness, or affected by production-stub pattern?**
The verification criterion (`status != "manifest"` AND `next_run is not None`) tests whether the job exists in the `/api/jobs/all` registry with a non-manifest status and a scheduled next_run. This is a REGISTRATION liveness check, not a model-promotion check. The production-stub pattern does NOT affect the criterion: the job registers, fires at 03:00, heartbeats (status transitions to `ok` or `skipped_idempotent`), and the next_run is pushed by the APScheduler listener. The criterion will be TRUE as long as the job is registered and has fired at least once (or has a next_run projected). Historical live evidence: `/api/jobs/all` showed `status="ok"`, `next_run="2026-05-10T03:00:00+02:00"` — the criterion was already met.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (15 total: 6 read + 9 snippet)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (job file, job_runtime, gate, scheduler, job_status_api, tests)
- [x] Consensus vs debate noted (consensus: promotion gate + idempotency key pattern is standard)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 9,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 6,
  "gate_passed": true
}
```
