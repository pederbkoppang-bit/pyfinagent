# Research Brief: Phase-9.10 Cron Runbook Documentation

**Tier:** simple
**Step id:** 9.10
**Date:** 2026-04-20

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://dev.to/cronmonitor/how-to-monitor-cron-jobs-in-2026-a-complete-guide-28g9 | 2026-04-20 | Blog/guide | WebFetch | "when they fail, they fail silently" -- exit-0 no-op is the most insidious class |
| https://deadmanping.com/blog/cron-job-silent-failure-detection | 2026-04-20 | Blog/guide | WebFetch | "Script exits with code 0 but doesn't perform work" -- defines silent no-op class explicitly |
| https://www.robustperception.io/idempotent-cron-jobs-are-operable-cron-jobs/ | 2026-04-20 | Authoritative blog (Prometheus experts) | WebFetch | Checkpoint-based idempotency; alert on last-success timestamp rather than individual runs |
| https://oneuptime.com/blog/post/2026-02-02-effective-runbooks/view | 2026-04-20 | Blog/guide | WebFetch | 9 required runbook sections including escalation paths, rollback, edge cases, and expected outputs |
| https://sreschool.com/blog/mttr/ | 2026-04-20 | SRE reference | WebFetch | MTTR targets: MTTD <10 min, acknowledge <2 min, mitigate <30 min; "Runbook absent" is a documented failure mode |
| https://grafana.com/whats-new/2025-10-22-dedicated-job-observability-in-grafana-cloud-kubernetes-monitoring/ | 2026-04-20 | Official vendor docs | WebFetch | "Last scheduled vs Last succeeded" comparison detects silent non-execution; dashboard required for "guaranteed execution integrity" |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.techtarget.com/searchitoperations/tip/An-introduction-to-SRE-documentation-best-practices | SRE blog | Fetched; content thin on cron specifics -- covered by other sources |
| https://www.squadcast.com/sre-best-practices/runbook-template | SRE template | Fetched; covered -- template structure captured |
| https://devtoolspro.org/articles/how-to-monitor-cron-jobs-in-kubernetes-production/ | K8s guide | Snippet; K8s-specific, less relevant to APScheduler context |
| https://moss.sh/devops-monitoring/how-to-monitor-cron-jobs/ | Blog | Snippet |
| https://www.justaftermidnight247.com/insights/site-reliability-engineering-sre-best-practices-2026-tips-tools-and-kpis/ | SRE blog | Snippet |
| https://rootly.com/sre/top-observability-tools-for-sre-teams-2025-rootly-guide | Blog | Snippet |
| https://dev.to/jasper_brookers_145826ebd/10-cron-jobs-that-silently-fail-and-how-to-detect-them-5d8i | Blog | Snippet; taxonomy aligned with deadmanping source |
| https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/ | Official docs | Snippet; K8s-specific concurrencyPolicy not applicable |
| https://sreschool.com/blog/production-readiness-review-prr/ | SRE reference | Snippet |
| https://www.solarwinds.com/sre-best-practices/runbook-template | SRE template | Snippet |

---

## Recency scan (2024-2026)

Searched: "cron job runbook SRE 2026", "cron job failure modes observability 2025", "silent failure cron job no-op detection 2025".

Findings: The 2025-2026 window produced substantive new material. The DEV.to 2026 cron monitoring guide and the Grafana October 2025 dedicated job observability release both post-date the canonical literature. The 2026 guide explicitly names exit-0 silent no-ops as the primary undetected failure class -- an evolution from older guidance that focused on exit-code monitoring alone. The Grafana 2025 release adds "Last scheduled vs Last succeeded" as a first-class dashboard primitive. No findings supersede the Robust Perception idempotency piece, which remains the canonical reference.

---

## Key findings

1. **Silent no-op is the primary undetected failure class in 2026.** A job that completes with exit code 0 but performs no useful work (e.g., due to empty inputs, skipped logic path, or missing injected dependency) is indistinguishable from a successful run without output-side validation. (Source: deadmanping.com, dev.to/cronmonitor 2026)

2. **Heartbeat alone is insufficient for no-op detection.** The heartbeat/dead-man-switch pattern detects job non-execution and explicit exceptions. It does NOT detect a job that runs to completion but writes zero rows, fires no alerts, and returns an empty result dict. (Source: deadmanping.com 2026; confirmed by internal code inspection -- see Internal Inventory below)

3. **Idempotency restart semantics are safe IFF downstream writes are themselves idempotent.** The in-memory IdempotencyStore reset on restart means daily jobs re-run after outage. If BQ ingesters use date-partitioned dedup keys this is harmless; if any write path lacks a dedup key, double-fire causes duplicate rows. (Source: robustperception.io; confirmed by job_runtime.py lines 112-113)

4. **MTTR targets for batch job systems.** Industry standard: MTTD <10 min, acknowledge <2 min, mitigate <30 min. Runbooks directly reduce MTTR; "Runbook absent" is listed as a standalone failure mode in SRE School's MTTR guide. (Source: sreschool.com/blog/mttr)

5. **Required runbook sections per 2026 standards.** OneUptime's 2026 guide identifies 9 mandatory sections: metadata header, trigger conditions, prerequisites, step-by-step procedure, expected outputs, edge case handling, verification steps, escalation paths, rollback procedure. The phase-9 runbook covers ~5 of 9. Missing: escalation paths (beyond "owner"), rollback procedure, expected output examples, edge cases. (Source: oneuptime.com 2026)

6. **Schedule-change governance.** The 2026 SRE standard requires change history tracking and reviewer-based governance for schedule changes. "Edit + restart launchd" without a PR/changelog does not meet the standard. (Source: oneuptime.com, techtarget.com SRE docs)

7. **Observability: logger.info heartbeat sink is insufficient pre-go-live.** Grafana's 2025 dedicated job observability release defines "guaranteed execution integrity" as requiring a persistent store comparing last-scheduled vs last-succeeded. A logger.info sink provides no such persistence. (Source: grafana.com 2025)

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| docs/runbooks/phase9-cron-runbook.md | 70 | The artifact under remediation | Active |
| backend/slack_bot/jobs/cost_budget_watcher.py | 64 | cost_budget_watcher job | BUG: run() has mandatory keyword args daily_spend_usd + monthly_spend_usd; APScheduler invokes run() with no args --> TypeError on every scheduled fire |
| backend/slack_bot/jobs/weekly_data_integrity.py | 57 | weekly_data_integrity job | BUG: run() defaults current_counts=None, prior_counts=None --> _compute_drifts({},{}) always returns []; silent no-op unless BQ counts injected by caller |
| backend/slack_bot/scheduler.py | 379 | APScheduler wiring; register_phase9_jobs() | register_phase9_jobs adds func=run directly with no args/kwargs injection for the two buggy jobs (lines 366-377); both bugs originate here |
| backend/slack_bot/job_runtime.py | 117 | heartbeat + IdempotencyStore + IdempotencyKey | Healthy; heartbeat marks idempotency key only on status=ok (line 112); skipped keys yield skipped=True to caller |

### Bug anatomy

**cost_budget_watcher TypeError (loud failure):**
- `register_phase9_jobs` calls `scheduler.add_job(func, trigger="cron", id="cost_budget_watcher", hour=6)` at scheduler.py line 374.
- `func` is `cost_budget_watcher.run`, which has signature `run(*, daily_spend_usd: float, monthly_spend_usd: float, ...)`.
- APScheduler fires `run()` with no arguments. Python raises `TypeError: run() missing 2 required keyword-only arguments`.
- Heartbeat context manager never entered; no idempotency key marked; job retries the TypeError on every scheduled tick.
- Runbook failure-modes table (§5) does NOT document this. It documents "scheduler.py import fail" but not "job callable raises TypeError on invocation".

**weekly_data_integrity silent no-op (silent failure):**
- `register_phase9_jobs` registers `weekly_data_integrity.run` with no args injection (scheduler.py line 374).
- APScheduler fires `run()` with no arguments. All parameters default: `current_counts=None`, `prior_counts=None` --> `cur={}`, `prev={}`.
- `_compute_drifts({}, {})` iterates over an empty dict; returns `[]`.
- Heartbeat completes with `status=ok`; idempotency key is marked; Slack alert never fires.
- The runbook's §4 heartbeat sink shows `status=ok`; §5 failure row "data_integrity detects drift" is never triggered. No operator signal.
- This is a textbook exit-0 silent no-op: the job runs, succeeds by all observable metrics, and does nothing useful.

**Idempotency restart double-fire analysis:**
- In-memory IdempotencyStore resets on process restart (job_runtime.py line 39 creates a fresh `_GLOBAL_STORE`).
- Daily jobs will re-run once after restart even if they already ran that day.
- If downstream write paths use BQ date-partitioned MERGE/INSERT IGNORE semantics (ingesters' dedup keys per §7), double-fire is harmless.
- Risk: if any job triggers a non-idempotent side effect (e.g., a Slack alert, an email, a payment-like downstream action) the double-fire will duplicate it. `cost_budget_watcher.alert_fn` is one such path -- if the circuit breaker trips, the alert fires again after restart.
- §7 of the runbook correctly states the restart semantics but does not call out the Slack alert double-fire risk for cost_budget_watcher.

---

## Consensus vs debate (external)

**Consensus:** Heartbeat/dead-man-switch is necessary but not sufficient. Output validation (row count, alert count, meaningful return value) is required to catch silent no-ops. All 2025-2026 sources agree.

**Debate:** Whether logger.info is acceptable pre-go-live. Grafana (2025) argues a persistent store is required; the Robust Perception piece argues last-success timestamp monitoring via Pushgateway is sufficient. For a single-node launchd deployment, a BQ `job_heartbeat` table (already planned per §4) bridges both views.

---

## Pitfalls (from literature)

- Alert on last-success age, not on individual run failure (robustperception.io): avoids false positives when automatic retry is in place.
- Never assume exit-0 = useful work: validate outputs, not just completion (deadmanping.com 2026).
- Duplicate runbook steps cause confusion under incident pressure: use imperative single-step format, include expected outputs (oneuptime.com 2026).
- MTTR inflates without escalation paths: the runbook's "owner re-auths BQ" is not an escalation path with contact + SLA -- it is an instruction without a fallback. (sreschool.com)

---

## Application to pyfinagent (mapping external findings to file:line anchors)

| External finding | Runbook gap | File:line anchor |
|-----------------|-------------|-----------------|
| Exit-0 silent no-op is primary failure class | §5 failure-modes table omits "silent no-op" row entirely. weekly_data_integrity always succeeds silently. | weekly_data_integrity.py:30-53; job_runtime.py:104 |
| Loud failure (TypeError) not documented | §5 documents scheduler import fail but not callable TypeError on invocation | cost_budget_watcher.py:19-28; scheduler.py:374 |
| Idempotency restart double-fire of alert_fn | §7 mentions harmless re-run but doesn't flag alert_fn duplication | cost_budget_watcher.py:56-59; job_runtime.py:39 |
| 9 required runbook sections (2026 standard) | §8 schedule-change is "edit + restart" -- no PR/changelog, no rollback | phase9-cron-runbook.md:62-63 |
| MTTD <10 min requires persistent heartbeat sink | §4 says "wire to BQ in a later phase" -- pre-go-live gap | phase9-cron-runbook.md:39; job_runtime.py:83 |
| Escalation paths per failure severity | §5 rows say "owner" without severity tier, contact method, or SLA | phase9-cron-runbook.md:43-50 |

---

## Design critique: gaps vs 2026 SRE standards

1. **Missing failure class: silent no-op.** The §5 table covers only loud failures (exceptions, import errors, circuit-breaker trips). It does not document the class where a job runs successfully but produces no useful output. weekly_data_integrity is currently a confirmed member of this class. The runbook should add a row: "job completes status=ok but output is empty / zero rows written / no drifts computed" with symptom "heartbeat shows ok; downstream table row count unchanged" and runbook "verify BQ row counts directly; check job received non-empty current_counts input."

2. **Missing failure class: callable TypeError on invocation.** cost_budget_watcher.run() raises TypeError on every scheduled fire. This is distinct from "scheduler import fail" (which the runbook documents). A row is needed: "job callable raises TypeError or missing-args error" with symptom "APScheduler logs TypeError; heartbeat never emitted; idempotency key never marked; job re-fires on next tick."

3. **Escalation paths are incomplete.** §5 says "Owner reviews" or "Owner checks" without specifying a contact method, severity tier, or time-to-respond SLA. Per MTTR standards, a runbook without escalation paths is a documented MTTR risk. Add severity (P0/P1/P2), contact (Slack @peder, iMessage), and target acknowledge time.

4. **No rollback procedure.** OneUptime's 2026 standard: "Every runbook that makes changes needs a rollback plan." The phase-9 runbook has no rollback for schedule changes, no rollback for a failed nightly_mda_retrain promotion, and no rollback if a BQ write corrupts a table.

5. **Schedule-change governance is informal.** §8 says "edit register_phase9_jobs mapping + restart Slack bot via launchd." No requirement for PR, changelog entry, or reviewer sign-off. Industry standard in 2026 requires at minimum a changelog entry + one-reviewer approval for schedule changes that could affect go-live timing.

6. **Observability: logger.info sink is pre-go-live only.** §4 defers BQ wiring to "a later phase." Go-live should not proceed with logger.info as the only heartbeat sink: there is no persistent record of which jobs ran, when, and whether they produced useful output. The runbook should state explicitly: "BQ job_heartbeat table wiring is a go-live prerequisite; logger.info is acceptable in dev/staging only."

7. **MTTR targets absent.** The runbook documents no SLO for detection or recovery. Adding a single line "Target MTTD <10 min, MTTR <30 min for P1 failures; <2 hr for P2" would bring it into 2026 standards alignment.

8. **Table row count.** The immutable criterion requires >=14 pipe-delimited rows across all tables. Current count: job inventory (7 rows) + failure modes (6 rows) = 13. One row short. Adding the two missing failure classes (silent no-op + callable TypeError) brings the count to 15, satisfying the criterion with margin.

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (10 snippet-only + 6 read = 16 total)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (5 files inspected)
- [x] Contradictions / consensus noted (logger.info debate)
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/phase-9.10-research-brief.md",
  "gate_passed": true
}
```
