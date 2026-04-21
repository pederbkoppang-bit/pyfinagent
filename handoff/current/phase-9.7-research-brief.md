---
step_id: 9.7
tier: simple
date: 2026-04-20
topic: Weekly data-integrity row-count-drift scan
---

## Research: Phase-9.7 — Weekly Data-Integrity Row-Count-Drift Scan

### Queries run (three-variant discipline)

1. Current-year frontier: `data integrity monitoring row count drift detection 2026`
2. Last-2-year window: `data quality monitoring row count drift threshold financial data 2025`
3. Year-less canonical: `data observability row count anomaly detection Great Expectations dbt Soda Monte Carlo`
4. Supplemental: `direction-aware row count drift monitoring drops vs growth data pipeline alerts`
5. Supplemental: `rolling median baseline vs prior week snapshot data quality monitoring robustness`
6. Supplemental: `Great Expectations dbt Soda row count check implementation 2025 comparison bespoke`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://oneuptime.com/blog/post/2026-01-30-data-drift-detection/view | 2026-04-20 | blog/tutorial | WebFetch | Drift 30% of features triggers overall alert; consecutive-drift-count pattern avoids transient noise; reference-distribution baseline pre-computed at init |
| https://www.synq.io/blog/data-observability-guide | 2026-04-20 | vendor doc | WebFetch | "Layered testing: volume monitors at sources, logic tests at transformations, business-logic validation at data products"; ML-driven adaptive thresholds preferred over static |
| https://medium.com/@lasyachowdary1703/day-22-monitoring-alerting-for-data-pipelines-make-failures-visible-fast-b7863ae09e36 | 2026-04-20 | practitioner blog | WebFetch | Row count < 10% of baseline example trigger; Great Expectations, dbt, Soda cited for declarative checks; zero-row loads as critical failure signal |
| https://www.montecarlodata.com/blog-data-quality-monitoring/ | 2026-04-20 | vendor doc | WebFetch | "Monitor triggers when 2 consecutive days show 40% increase or decrease"; dynamic learning periods recommended over static snapshots; metadata-first strategy reduces compute |
| https://www.acceldata.io/blog/adaptive-data-quality-thresholds-moving-beyond-static-rules | 2026-04-20 | vendor doc | WebFetch | Sliding 7-day statistical windows (median/quantile) more robust than prior-week snapshot; financial tables warrant 99% confidence intervals; ensemble short-term + long-term prevents outlier contamination |
| https://www.thedataletter.com/p/tool-review-soda-core-vs-great-expectations | 2026-04-20 | practitioner review | WebFetch | Soda: YAML checks deployable in minutes (SQL-where fluency sufficient); Great Expectations: Python-fluent, deeper programmatic control; neither replaces bespoke for highly constrained tables |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.integrate.io/blog/what-is-data-integrity-and-why-is-it-important/ | blog | General intro; superseded by richer sources fetched |
| https://labelyourdata.com/articles/machine-learning/data-drift | blog | ML-centric drift, not pipeline row-count focused |
| https://www.dqlabs.ai/blog/understanding-data-drift-and-why-it-happens/ | vendor blog | Snippet sufficient; directional advice not present |
| https://www.metaplane.dev/state-of-data-quality-monitoring-2024 | vendor report | Fetched but returned only tool-landscape catalog, no threshold specifics |
| https://www.metaplane.dev/blog/data-quality-metrics-for-data-warehouses | vendor blog | Fetched; only "no one-size-fits-all" guidance, no specific row-count thresholds |
| https://www.alation.com/blog/mastering-data-quality-monitoring/ | vendor blog | Snippet: "monitor row counts, null rates, key uniqueness against dynamic baselines" |
| https://www.montecarlodata.com/blog-data-quality-metrics/ | vendor blog | Snippet: ML thresholds for volume anomalies noted |
| https://github.com/calogica/dbt-expectations | code | Implementation reference for expect_table_row_count_to_equal |
| https://sixthsense.rakuten.com/blog/Anomaly-Detection-in-Data-Observability-Techniques-and-Tools | vendor blog | Snippet: row-count drops as primary observability signal |
| https://www.sparvi.io/blog/best-data-observability-tools | vendor comparison | Snippet: 2025 tool landscape overview |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on data integrity row-count drift and data observability thresholds. Results:

- The 2026-dated OneUptime post confirms that `consecutive_drift_count` (requiring multiple consecutive scan cycles before alerting) is the current best practice for suppressing transient noise — this supersedes naive single-pass 20% threshold triggers.
- Acceldata (2025) formalises the shift from static to adaptive/rolling-window thresholds as the production standard.
- Monte Carlo (2025) confirms bidirectional threshold logic (trigger on either 40% increase OR decrease vs baseline) is standard in commercially deployed platforms.
- No peer-reviewed paper specifically on financial table row-count thresholds was found in the 2024-2026 window; the closest is Acceldata's financial-critical-table guidance (99% CI, tighter freshness SLA).
- The direction-aware alert pattern (treating drops and growth differently in severity) is discussed in practitioner literature but has not yet received canonical peer-reviewed treatment as of April 2026.

---

### Key findings

1. **20% absolute threshold is reasonable but not direction-aware.** Industry examples (OneUptime 30%, Monte Carlo 40%, practitioner blog 10%) scatter around 20% as a mid-range default. The critical gap is directionality: the current implementation uses `abs(delta)/prev`, treating a 25% drop identically to a 25% growth. Drops may signal data loss or pipeline failure; growth is often normal for append-only financial tables (harness_learning_log, price history). (Sources: OneUptime 2026-01-30, Monte Carlo data-quality-monitoring)

2. **Rolling median baseline is more robust than prior-week snapshot.** A prior-week snapshot is vulnerable to outlier weeks (holiday low-volume, backfill spikes). Acceldata recommends sliding 7-14 day windows computing the median to smooth noise while adapting to recent trends. (Source: Acceldata adaptive-thresholds)

3. **Great Expectations / Soda / dbt are viable but bring friction.** Soda YAML checks deploy faster (minutes vs days) and suit continuous monitoring of warehouse tables. Great Expectations is more expressive but Python-heavy. Neither eliminates the need for bespoke logic when tables have directional invariants (e.g., harness_learning_log must not shrink). The current bespoke implementation is appropriate for a tightly constrained job list; the main gap is not the framework choice but the missing directionality. (Source: The Data Letter tool review, practitioner blog)

4. **Direction-aware drift should alert at asymmetric thresholds.** Append-only tables (price tables, harness_learning_log) should use a separate lower bound (e.g., drop > 5% = high severity) and an upper bound for unexpected bulk-ingestion detection (e.g., growth > 50% = medium severity). Monotonic-growth tables should never alert on growth below a spike level. (Sources: SYNQ observability guide, Monte Carlo monitoring)

5. **First-scan tolerance (missing prior baseline = skip) is correct.** All reviewed sources confirm this pattern. The alternative — alerting on the first scan with no baseline — generates guaranteed false positives. The current implementation correctly skips tables without prior_counts entry. (Sources: Monte Carlo, practitioner blog)

6. **Consecutive-run confirmation reduces false positives.** The current implementation alerts on a single-week breach. The OneUptime 2026 pattern suggests requiring the threshold to be exceeded on two consecutive scan cycles before sending a Slack alert, particularly for growth events. This is optional hardening; a single-cycle alert remains acceptable for drops. (Source: OneUptime 2026-01-30)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/weekly_data_integrity.py` | 57 | Row-count drift scanner: `run()` + `_compute_drifts()` | Active, produced by phase-9.7 |
| `tests/slack_bot/test_weekly_data_integrity.py` | 57 | 3 tests: above-threshold, below-threshold, missing-baseline | Active, produced by phase-9.7 |
| `backend/slack_bot/job_runtime.py` | 117 | `heartbeat` context manager + `IdempotencyStore` + `IdempotencyKey` | Active, produced by phase-9.1 |
| `backend/slack_bot/scheduler.py` | 379 | APScheduler wiring; `weekly_data_integrity` registered at Mon 05:00 | Active; phase-9.9 block wires all 9.x jobs |

**Key observations from code audit:**

- `_compute_drifts` (line 45-53): uses `abs(cur_n - prev_n) / prev_n` — direction-blind. Drop and growth treated identically.
- `DRIFT_THRESHOLD = 0.20` (line 15): constant; not per-table, not directional.
- `prev_n is None or prev_n == 0` skip (line 49): correctly handles first-scan and zero-row edge case.
- `alert_fn` fail-open (lines 38-41): exception in Slack send does not raise; logged at WARNING. Correct.
- Idempotency via `IdempotencyKey.weekly` ensures one run per ISO week (line 27). Correct.
- No rolling baseline: `prior_counts` is a single point-in-time snapshot passed as argument. The caller (scheduler.py `run`) would need to supply a rolling median rather than a single prior row.
- No per-table threshold tuning: all tables use the same 20% threshold regardless of whether the table is append-only vs. update-in-place.
- `scheduler.py` line 362: `weekly_data_integrity` job runs `run()` directly without passing `current_counts` or `prior_counts` — the production wiring must supply these from a BQ query or snapshot store; that wiring is not visible in the scheduler.py registration block (it passes no args). This means the production path likely calls `run()` with empty dicts, triggering no drifts. **This is a latent gap**: the job registers `run` but the scheduler `add_job` at line 374 passes no `args=` for data-integrity. All other jobs that need external data presumably resolve it inside their own `run()`.

---

### Consensus vs debate (external)

**Consensus:**
- Row-count monitoring is the entry-level baseline for data integrity; all sources agree it should be present.
- Rolling/adaptive baselines are superior to single-week snapshots for production use.
- First-scan skip (no prior baseline = no alert) is correct.
- Direction-aware alerting is considered best practice by practitioners.
- Bespoke solutions are acceptable for small, constrained job sets; frameworks (Soda, dbt) pay off at scale (>50 tables).

**Debate:**
- Threshold value: industry examples range from 10% to 40% (with 20% in the middle); no authoritative peer-reviewed number for financial tables specifically.
- When to require consecutive-cycle confirmation: OneUptime recommends it; Monte Carlo uses a 2-day window; single-cycle alert is still common for drops.

---

### Pitfalls (from literature)

- **Alert fatigue from symmetric thresholds on append-only tables.** Append-only tables (price history) grow continuously; a symmetric 20% threshold will alert every week after a bulk-load. Use directional or growth-aware bounds.
- **Snapshot brittleness.** A prior-week snapshot that happened to include a backfill will inflate the baseline, causing the next normal week to appear as a 30% drop.
- **Zero-row edge case.** `prev_n == 0` skip is required; otherwise division by zero. Current code handles this correctly (line 49).
- **Framework lock-in without directionality.** Soda and Great Expectations handle absolute row-count bounds well but require custom checks to express "this table must never shrink" — that logic is simpler in the bespoke implementation.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

| Finding | Recommendation | File:line |
|---------|---------------|-----------|
| Direction-blind `abs()` threshold | Split into `drop_threshold` and `growth_threshold`; append-only tables (harness_learning_log, price tables) should use a tighter drop threshold (e.g., 5%) and a looser or no growth threshold | `weekly_data_integrity.py:45-53` |
| Single prior-week snapshot | Caller should pass a 4-week rolling median as `prior_counts` values; this requires the production scheduler wiring to compute the median from BQ before calling `run()` | `weekly_data_integrity.py:26`, `scheduler.py:362` |
| No `args=` in `add_job` for data_integrity | The scheduler registration block does not pass `current_counts` or `prior_counts`; `run()` will execute with empty dicts and no drifts will ever be detected in production | `scheduler.py:374` |
| 20% threshold is reasonable as a starting default | Keep 20% for general tables; add per-table overrides dict as an optional parameter for known monotonic tables | `weekly_data_integrity.py:15`, `weekly_data_integrity.py:25` |
| Consecutive-cycle confirmation for growth | Optional hardening: track prior week's drift set; only alert on growth if two consecutive weeks exceed the threshold | `weekly_data_integrity.py:35-42` |

---

### Design critique

**Bespoke vs Great Expectations/Soda:**
The bespoke approach (57 lines) is appropriate here. The job monitors a small, known set of BQ tables with specific invariants (append-only vs. update-in-place). Soda or Great Expectations would add framework overhead (YAML config, connectivity setup) for marginal gain. The main value of a framework would be UI-driven threshold management and automatic anomaly detection, neither of which is needed for a weekly Slack alert job. Stick with bespoke; add directionality as a code change.

**Threshold tuning:**
20% is a reasonable general-purpose default (sits in the middle of the 10%-40% industry range). The gap is direction-awareness. `harness_learning_log` is append-only: any shrink is a P1 event regardless of percentage; growth is expected and should only alert above 100% to catch runaway writes. Price tables grow ~1 row per ticker per day: a weekly drop > 5% deserves attention; weekly growth > 30% could signal a duplicate ingest.

**Direction-aware drift:**
The `_compute_drifts` function should be extended to return a `direction` field (`"drop"` or `"growth"`) and allow per-direction thresholds. The alert payload would then include direction, enabling different Slack severities (P1 for drops, P2 for unexpected growth).

**Rolling baseline:**
Replace the single `prior_counts` snapshot with a `rolling_prior_counts` argument that the caller computes as the median of the last 4 weeks. The `run()` signature does not change (backward compatible); only the caller's preparation logic changes. This is a scheduler.py concern, not a `weekly_data_integrity.py` concern.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only): 16 URLs
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (4 files read in full)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim (not just listed in a footer)

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 10,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 4,
  "report_md": "handoff/current/phase-9.7-research-brief.md",
  "gate_passed": true
}
```
