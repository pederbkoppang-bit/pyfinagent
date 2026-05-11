---
step_id: phase-23.5.12
step_name: "Cron job verification: weekly_data_integrity (slack_bot, phase-9.7)"
tier: simple
researcher_session: 2026-05-10
---

## Research: phase-23.5.12 -- weekly_data_integrity cron job verification

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://atlan.com/data-integrity-best-practices/ | 2026-05-10 | doc/blog | WebFetch | "After every batch load, compare row counts between source and target. Any discrepancy triggers an alert." Automated validation reduces engineering workload 53%. |
| https://docs.cloud.google.com/bigquery/docs/scheduling-queries | 2026-05-10 | official doc | WebFetch | Weekly schedule pattern "every monday 23:30"; idempotency warning: "Scheduled queries running exactly on the hour might trigger multiple times." Use WRITE_TRUNCATE for idempotent overwrites. |
| https://www.sparvi.io/blog/dbt-data-quality-testing | 2026-05-10 | industry blog | WebFetch | "Tests run when you run `dbt test`, not continuously. Data can go bad between builds." Custom row_count_threshold pattern for volume checks; confirms need for scheduled SQL audit supplementation. |
| https://www.anomalo.com/blog/beyond-dbt-tests-taking-a-comprehensive-approach-to-data-quality/ | 2026-05-10 | industry blog | WebFetch | Three-pillar strategy: validation rules + ML-based metric anomaly detection + unsupervised data monitoring. ML adapts thresholds automatically; reserved rules for high-stakes well-defined scenarios. |
| https://betterstack.com/community/guides/scaling-python/apscheduler-scheduled-tasks/ | 2026-05-10 | tech blog | WebFetch | APScheduler cron trigger patterns for weekday scheduling, `replace_existing=True` prevents ConflictingIdError on reload. SQLAlchemy job store for persistence across restarts. |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-05-10 | official doc | WebFetch | `misfire_grace_time` controls whether misfired jobs still run; `coalesce=True` collapses multiple queued executions into one. `add_listener(fn, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)` is the canonical event hook pattern. |
| https://www.integrate.io/blog/what-is-data-integrity-and-why-is-it-important/ | 2026-05-10 | industry blog | WebFetch | Row count reconciliation after every batch load; weekly schedule appropriate for batch processes. Remediation: automate repair workflows + Slack alerting on drift detection. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.datafold.com/blog/dbt-expectations/ | blog | Redundant with sparvi dbt coverage; snippets sufficient |
| https://www.datadoghq.com/blog/dbt-data-quality-testing/ | blog | Covered by anomalo + sparvi reads |
| https://www.paradime.io/guides/best-dbt-test-packages | blog | Test package catalogue; not directly applicable to APScheduler jobs |
| https://www.datasunrise.com/knowledge-center/database-audit-sql-server/ | doc | SQL Server specific; not BigQuery-relevant |
| https://dev.to/hexshift/how-to-run-cron-jobs-in-python-the-right-way-using-apscheduler-4pkn | blog | APScheduler user-guide covers same ground |
| https://cronradar.com/blog/python-scheduler-monitoring | blog | Heartbeat pattern covered in APScheduler official docs |
| https://oneuptime.com/blog/post/2026-02-17-how-to-set-up-bigquery-scheduled-queries-for-automated-reporting/view | blog | GCP-native scheduling; project uses APScheduler not BQ Scheduled Queries |
| https://hevodata.com/learn/data-integrity-tools/ | blog | Tool catalogue; not pattern-relevant |
| https://us.pycon.org/2026/schedule/presentation/87/ | conference | Schedule page only; no content |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on: (1) "data integrity monitoring SQL audit patterns 2026", (2) "weekly cron job idempotency SQL audit BigQuery 2025", (3) "APScheduler cron job idempotency heartbeat pattern Python 2025", (4) "data quality monitoring great expectations dbt tests row count drift detection".

Result: Found several relevant 2025-2026 findings:
- BigQuery Pipelines (previewed Sep 2024, renamed Mar 2025) is a new scheduling/orchestration option but project uses APScheduler -- no action needed.
- Atlan (2026) reports 67% of organizations distrust their data (up from 55% in 2023), validating weekly row-count drift audits.
- dbt-expectations and Elementary 2025 add moving-stddev anomaly detection and volume_anomalies tests -- more sophisticated than pyfinagent's fixed-20%-threshold approach, but the fixed threshold is simpler to verify and appropriate for the project's scale.
- APScheduler 3.11.2 (current) matches the version in use; misfire_grace_time + coalesce pattern documented in phase-23.3.3 is current best practice per official docs.
- No findings that supersede the existing heartbeat + idempotency design.

### Key findings

1. `weekly_data_integrity` uses `IdempotencyKey.weekly()` + `heartbeat()` context manager correctly -- the same pattern as the 2 non-stub-affected siblings (nightly_mda_retrain, hourly_signal_warmup). (Source: backend/slack_bot/jobs/weekly_data_integrity.py:39, 48)

2. `alert_fn` parameter is optional and NOT wired by the scheduler -- APScheduler calls `run()` with zero args (per docstring line 6-8: "APScheduler fires run() with zero args"). When `alert_fn is None`, drift detection still runs but no Slack alert fires. This is the "alert_fn not wired" gap noted in phase-23.3.3. (Source: weekly_data_integrity.py:54-58)

3. The production stub gap pattern (fake HTTP call instead of real work) does NOT apply here. `weekly_data_integrity.run()` performs real work: calls `_default_fetch_counts()` which executes a BigQuery SQL query against `__TABLES__`, then computes drift against a JSON snapshot, then saves updated snapshot. No HTTP stub. No production-stub gap. (Source: weekly_data_integrity.py:78-112)

4. APScheduler registers the job as `"weekly_data_integrity"` with `day_of_week="mon", hour=5, misfire_grace_time=7200, coalesce=True`. (Source: scheduler.py:531)

5. `"weekly_data_integrity"` is present in `_JOB_NAMES` at `job_status_api.py:62`, pre-seeded in `_registry`, and exposed via `/api/jobs/all`. (Source: job_status_api.py:62)

6. Slack bot log confirms successful registration on 2026-05-09 23:24: `"phase-9 jobs registered: ['daily_price_refresh', 'weekly_fred_refresh', 'nightly_mda_retrain', 'hourly_signal_warmup', 'nightly_outcome_rebuild', 'weekly_data_integrity', 'cost_budget_watcher']"` (Source: handoff/logs/slack_bot.log)

7. The `_seed_next_run_registry()` call in `start_scheduler()` pushes `status="scheduled"` + `next_run_time` for every registered job (including `weekly_data_integrity`) at startup, which is why `/api/jobs/all` shows `next_run="2026-05-11T05:00:00+02:00"`. (Source: scheduler.py:96-123, 213)

8. The Docker-alias bug (`http://backend:8000` vs `http://127.0.0.1:8000`) does NOT affect `weekly_data_integrity`. The job makes no HTTP calls -- it uses direct Python import (`BigQueryClient`) and filesystem snapshot I/O. All fixed URLs (`_HEARTBEAT_URL`, `_HEALTH_PROBE_URL`, `_LOCAL_BACKEND_URL`) are in `scheduler.py` for the APScheduler event listener and digest handlers, not in the job itself. (Source: weekly_data_integrity.py:78-112, scheduler.py:36-46)

9. Test coverage: 3 tests in `tests/slack_bot/test_weekly_data_integrity.py` covering drift-above-threshold, drift-below-threshold, and missing-prior-baseline. Tests pass an explicit `store=IdempotencyStore()` and `iso_year_week` so they are fully isolated -- no BQ calls, no filesystem I/O. (Source: tests/slack_bot/test_weekly_data_integrity.py:15-57)

10. Row-count drift detection with a fixed percentage threshold (20%) is an established pattern. Industry trend is toward ML-based adaptive thresholds (Anomalo, Elementary), but a fixed threshold is sufficient for the project's scale and verifiability. (Source: anomalo.com blog, sparvi.io 2025 guide)

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| backend/slack_bot/jobs/weekly_data_integrity.py | 116 | Phase-9.7 job: BQ row-count drift audit | Active, real work (no stub) |
| backend/slack_bot/scheduler.py | 549 | APScheduler + phase-9 job registration | Active; weekly_data_integrity at L531 |
| backend/slack_bot/job_runtime.py | ~120 | heartbeat() + IdempotencyStore + IdempotencyKey | Active primitives |
| backend/api/job_status_api.py | 189 | In-memory registry + /api/jobs/status + /api/jobs/heartbeat | Active; weekly_data_integrity in _JOB_NAMES L62 |
| backend/api/cron_dashboard_api.py | 60+ | /api/jobs/all bridge merging registry + manifest | Active; imports job_status_api |
| tests/slack_bot/test_weekly_data_integrity.py | 57 | Unit tests: 3 test cases | Exists, isolated, complete |
| handoff/logs/slack_bot.log | N/A | Runtime log | Confirms registration 2026-05-09 23:24 |

### Consensus vs debate (external)

Consensus: weekly row-count drift audits are a well-established data quality pattern (Integrate.io, Atlan, Anomalo all confirm). Fixed percentage thresholds are appropriate for smaller-scale systems; ML-based adaptive thresholds are the 2025-2026 frontier for large data estates. For pyfinagent's scale, the 20% fixed threshold is correct and simpler to verify.

No significant debate: APScheduler's `misfire_grace_time + coalesce` pattern is unambiguously correct for weekly jobs.

### Pitfalls (from literature)

- Alert not firing silently (alert_fn=None when called by APScheduler with zero args): drift is detected but no Slack message posted. This is the gap noted in phase-23.3.3 -- Slack alerting path is dormant unless wired explicitly in the scheduler add_job call. Currently not wired; this is a known limitation, not a bug blocking verification.
- BQ `__TABLES__` query fail-open: if BigQuery is unreachable, `_default_fetch_counts()` returns `{}`. This means no drift is computed (both cur and prev empty) -- safe fail-open, but the audit is silently skipped. Acceptable for a weekly scan on a local-only system.
- First-run baseline: on the very first run, `_load_snapshot()` returns `{}` (no prior file). `_compute_drifts()` skips all tables where `prev_n is None`. No false alert on first run -- correct behavior.

### Application to pyfinagent (mapping findings to code)

- heartbeat() wired at: `weekly_data_integrity.py:48` -- `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:`
- IdempotencyKey.weekly() at: `weekly_data_integrity.py:39`
- APScheduler trigger config at: `scheduler.py:531` -- `day_of_week="mon", hour=5, misfire_grace_time=7200, coalesce=True`
- Registry pre-seed at: `job_status_api.py:62` (in `_JOB_NAMES`)
- Startup next_run push at: `scheduler.py:96-123` (`_seed_next_run_registry`)
- No Docker-alias exposure in job module: confirmed by reading `weekly_data_integrity.py` end-to-end -- no `httpx`, no `requests`, no `http://` URL strings.

---

## Three answers (plain text)

**Docker-alias bug?** No. `weekly_data_integrity.py` makes no HTTP calls. It uses a direct Python import of `BigQueryClient` and filesystem JSON snapshot I/O. The Docker-alias bug (`http://backend:8000` vs `http://127.0.0.1:8000`) is confined to `scheduler.py` for the event-listener heartbeat POST and digest handlers -- none of which are inside the job module itself.

**heartbeat() correctly wired?** Yes. `run()` wraps its core logic in `with heartbeat(JOB_NAME, idempotency_key=key, store=store) as state:` at line 48. IdempotencyKey.weekly() generates the key at line 39. The context manager emits started/ok/failed events and enforces the idempotency skip. This matches the same pattern used by nightly_mda_retrain and hourly_signal_warmup (the two non-stub-affected siblings from the prior phase-9 sibling matrix).

**Production-stub affected?** No. `weekly_data_integrity.run()` performs real work: queries BQ `__TABLES__` for row counts, computes drift against a JSON snapshot, optionally calls `alert_fn` (not wired by APScheduler, known gap), and saves updated snapshot. There is no HTTP stub, no fake return, no placeholder. This job is NOT in the stub-affected category. The prior sibling matrix (daily_price_refresh, weekly_fred_refresh, nightly_outcome_rebuild = stub-affected; nightly_mda_retrain, hourly_signal_warmup = not affected) is confirmed extended: weekly_data_integrity also not affected.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 fetched)
- [x] 10+ unique URLs total incl. snippet-only (16 collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 files)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 9,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "report_md": "handoff/current/phase-23.5.12-research-brief.md",
  "gate_passed": true
}
```
