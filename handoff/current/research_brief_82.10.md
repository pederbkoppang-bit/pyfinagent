# Research Brief -- masterplan step 82.10

**Topic:** Give the data-freshness alarm an ACTIVE scheduled evaluator that
emits through the operator notification path, independent of any browser
session.

**Tier:** moderate (caller-stated). **Audit-class:** false.
**Researcher:** Layer-3 merged researcher + explorer.
**Started:** 2026-08-05. Write-first: this file is created before any
source is read and appended incrementally.

**Status: COMPLETE** -- gate PASSED (7 sources read in full, 30 URLs, recency
scan done). Envelope in section 11 is authoritative and is the LAST content in
this file.

---

## HEADLINE (read this first)

**The step description is materially wrong in one high-value way.** It says the
freshness alarm "cannot page". In fact **the emitter already exists and already
pages**:

- `backend/services/cycle_health.py:100-135` -- `_fire_freshness_alarm(sources)`
  dispatches a **P1** via `raise_cron_alert_sync(source="cycle_health",
  error_type=f"freshness_critical_{table}", severity="P1", ...)` for every
  source whose `band == "red"`.
- `backend/services/cycle_health.py:564-565` -- `compute_freshness` calls it
  itself: `if overall_band == "red": _fire_freshness_alarm(sources)`.
- `backend/services/observability/alerting.py:46-53` records a **real page
  storm from this exact alarm** (phase-66 hotfix: "~120 pages/hour the moment a
  dashboard tab was open against a red table") -- direct evidence the emitter
  was live all along.

The missing piece is **only a trigger**. But there is a second, unstated
requirement without which the fix makes things worse: **the `AlertDeduper` does
NOT suppress steady state** (measured -- a P1 re-fires every `repeat_hours`
forever), so a bare timer would convert a silent alarm into a
4-pages-a-day-for-128-days alarm. A **state-transition gate** is mandatory.
See sections 7-9.

---

---

## Section index

1. Sources read in full (>=5 required)
2. Snippet-only sources
3. Search queries run (three-variant discipline)
4. Recency scan (last 2 years)
5. Key external findings
6. Internal code inventory (re-derived file:line)
7. Recommendation for the contract
8. Traps
9. Stale/wrong claims in the step description
10. JSON gate envelope

---

## 1. Sources read in full

_(populated incrementally below)_

All accessed 2026-08-05. All fetched via `WebFetch` (full page, not snippet).

| # | URL | Tier | Read in full | What it establishes |
|---|-----|------|--------------|---------------------|
| 1 | https://sre.google/sre-book/monitoring-distributed-systems/ | official-docs (canonical, year-less) | yes | **The direct indictment of the current design:** *"SRE teams carefully avoid any situation that requires someone to 'stare at a screen to watch for problems.'"* Also the symptom-vs-cause rule (*"'what' versus 'why' is one of the most important distinctions in writing good monitoring with maximum signal and minimum noise"*) and the page test: *"Every page should be actionable... If a page merely merits a robotic response, it shouldn't be a page."* |
| 2 | https://prometheus.io/docs/practices/alerting/ | official-docs (canonical, year-less) | yes | **The exact threshold rule for this job.** For batch/cron work: *"page if the batch job has not succeeded recently enough, and this will cause user-visible problems"*, with thresholds set *"at least enough time for 2 full runs of the batch job."* That is literally `CRITICAL_RATIO = 2.0` -- the repo's existing constant is already the textbook value, independently corroborated. Also metamonitoring: *"It is important to have confidence that monitoring is working."* |
| 3 | https://sre.google/workbook/alerting-on-slos/ | official-docs (canonical) | yes | The four axes to grade an alerting design: **precision, recall, detection time, reset time** -- *"Reset Time: How long alerts fire after an issue is resolved. Long reset times can lead to confusion or to issues being ignored."* Directly names the 128-day-red failure mode: a level-triggered alarm has an unbounded reset time and gets ignored. |
| 4 | https://sre.google/sre-book/practical-alerting/ | official-docs (canonical) | yes | Alerts *"can 'flap' (toggle their state quickly); therefore, the rules allow a minimum duration for which the alerting rule must be true before the alert is sent"* (`for 2m`). And the dedup responsibility sits in the alert manager: *"Deduplicate alerts from multiple Borgmon that have the same labelsets."* Severity-routing: *"page-worthy alerts to their on-call rotation and their important but subcritical alerts to their ticket queues."* |
| 5 | https://apscheduler.readthedocs.io/en/3.x/userguide.html | official-docs (vendor, year-less) | yes | `replace_existing`: *"you **MUST** define an explicit ID for the job and use `replace_existing=True` or you will get a new copy of the job every time your application restarts!"* `coalesce`: *"if coalescing is enabled ... it will only trigger it once. No misfire events will be sent for the 'bypassed' runs"* -- **default is `False`**. `misfire_grace_time` gates whether a missed run still fires. Together these are the anti-128-catch-up-spam controls the step description worries about. |
| 6 | https://docs.getdbt.com/docs/deploy/source-freshness | official-docs (industry-standard tool) | yes | The cadence rule: *"It's important that your freshness jobs run frequently enough to measure data latency in accordance with your SLAs"* -- rule of thumb *"you should run your source freshness jobs with at least double the frequency of your lowest SLA."* Gives a defensible number for the new job's interval (see Recommendation). Also the two-mode design (non-blocking check vs failing run step). |
| 7 | https://montecarlo.ai/blog-data-freshness-explained | practitioner (data-observability vendor) | yes | Freshness is the #1 of the "five pillars"; the canonical failure is the *silent* one -- a pipeline that stops without erroring. Corroborates that freshness must be measured as **time since last update against a per-table expectation**, which is exactly `_TABLE_MAX_AGE_SEC`. |

**Count read in full: 7** (floor is 5).

---

## 2. Snippet-only sources (evaluated, NOT read in full -- do not count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://landing.google.com/sre/sre-book/chapters/monitoring-distributed-systems/ | official-docs | Duplicate of source 1 (legacy URL) |
| https://sre.google/resources/book-update/monitoring-distributed-systems/ | official-docs | Duplicate/summary of source 1 |
| https://sre.google/workbook/monitoring/ | official-docs | Adjacent chapter; source 3 was the on-point one |
| https://www.usenix.org/sites/default/files/conference/protected-files/srecon16europe_slides_rabenstein.pdf | conference slides | Binary PDF; SRE book covers the same symptom/cause thesis |
| https://training.promlabs.com/training/monitoring-and-debugging-prometheus/metrics-based-meta-monitoring/end-to-end-watchdog-alerts/ | authoritative-blog (PromLabs) | Paywalled training; the Watchdog pattern is covered by source 2 |
| https://engineering.hellofresh.com/who-monitors-the-monitoring-system-is-my-prometheus-alive-at-all-2789fd3647b3 | practitioner | Meta-monitoring; out of scope (we monitor data, not the monitor) |
| https://seifrajhi.github.io/blog/securing-monitoring-stack-dead-man-switch/ | community | Lower tier; same pattern as source 2 |
| https://crontap.com/blog/dead-man-switch-explained-for-developers | community/vendor | Vendor explainer |
| https://updog.watch/learn/what-is-dead-mans-switch | community/vendor | Vendor explainer |
| https://oneuptime.com/blog/post/2026-02-06-heartbeat-dead-man-switch-opentelemetry-pipeline/view | vendor blog (2026) | Counted in recency scan; no new mechanism |
| https://www.paulsprogrammingnotes.com/2026/07/dead-mans-switch-single-host-monitoring.html | community (2026) | Counted in recency scan; single-host relevance noted |
| https://nurbak.com/en/blog/dead-mans-switch/ | vendor (2026) | Vendor explainer |
| https://drumbeats.io/heartbeat-monitoring | vendor | Vendor explainer |
| https://apscheduler.readthedocs.io/en/latest/modules/schedulers/base.html | official-docs | API reference for source 5 |
| https://apscheduler.readthedocs.io/en/master/userguide.html | official-docs | 4.x branch; repo is on 3.x |
| https://www.getdbt.com/blog/data-slas-best-practices | vendor blog | Complements source 6 |
| https://www.conduktor.io/glossary/data-freshness-monitoring-sla-management | vendor glossary | Low information density |
| https://www.databricks.com/blog/what-is-data-observability | vendor blog | General overview |
| https://datahub.com/products/data-observability/ | vendor product page | Marketing |
| https://www.pantomath.com/guide-data-observability/data-pipeline-monitoring | vendor guide | Marketing |
| https://docs.datadoghq.com/monitors/guide/reduce-alert-flapping/ | official-docs (Datadog) | Recency-scan hit; recovery-delay mechanism noted below |
| https://web-alert.io/blog/alert-flapping-detection-taming-unstable-alerts | vendor blog (2026) | Recency-scan hit; hysteresis/recovery-delay |
| https://www.datadoghq.com/blog/best-practices-to-prevent-alert-fatigue/ | vendor blog | Recency-scan hit |

**Snippet-only count: 23. Total unique URLs collected: 30.**

---

## 3. Search queries run (three-variant discipline)

| # | Query | Variant |
|---|-------|---------|
| 1 | `Google SRE book monitoring distributed systems symptom versus cause alerting` | **YEAR-LESS canonical** |
| 2 | `dead man's switch alerting pattern watchdog heartbeat monitoring` | **YEAR-LESS canonical** |
| 3 | `APScheduler misfire_grace_time coalesce documentation scheduling jobs` | **YEAR-LESS canonical** (vendor docs) |
| 4 | `data observability freshness SLA monitoring 2025 2026 detection incident data downtime` | last-2-year + current-year |
| 5 | `alert fatigue state transition alerting flapping level-triggered vs edge-triggered 2026` | **current-year frontier** |

All three mandated variants are represented.

---

## 4. Recency scan (last 2 years, 2024-2026) -- MANDATORY SECTION

**Searched:** queries 4 and 5 above, explicitly scoped to 2024-2026.

**Result: 2 findings that COMPLEMENT (do not supersede) the canonical sources.**

1. **Alert flapping / recovery-delay is now standard vendor practice** (Datadog
   "Reduce alert flapping"; web-alert.io 2026). The 2024-2026 framing is that
   *"each transition is treated as a new event, resulting in a storm of alerts
   for what is really one ambiguous, borderline condition"*, and the mitigation
   is a **recovery delay** -- once an alert fires it stays active for a minimum
   period rather than re-firing. This is the modern packaging of the SRE book's
   `for 2m` minimum-duration rule (source 4) and it VALIDATES the repo's
   existing `_watchdog_last_was_healthy` state-transition idiom. Naming caution:
   the vendor literature calls this "hysteresis". That word is a **banned term
   in this repo's TRADING context** (phase-61 churn integrity) -- it is
   unrelated here. Do not let the vocabulary collision cause a false objection.
2. **ML-based freshness anomaly detection** is the 2025-2026 commercial
   frontier (Monte Carlo, DataHub Cloud): learn each table's arrival pattern
   instead of hand-setting thresholds. **Explicitly NOT recommended for 82.10** --
   it is a platform-scale feature, the repo has 6 tables with known cadences,
   and Monte Carlo itself notes hand-rolled checks are fine below ~50 tables.
   No new finding supersedes the 2x-interval threshold rule.

**Nothing found in the window invalidates `CRITICAL_RATIO = 2.0`, the
state-transition gate, or the `register_*_cron` shim pattern.**

---

## 5. Key external findings

1. **The current design is a named SRE anti-pattern.** *"SRE teams carefully
   avoid any situation that requires someone to 'stare at a screen to watch for
   problems.'"* (Google SRE Book Ch.6, accessed 2026-08-05). A browser-driven
   alarm is definitionally that situation. This is the one-line justification
   for the whole step.
2. **The 2x threshold is textbook, not arbitrary.** For batch/cron work,
   Prometheus prescribes *"page if the batch job has not succeeded recently
   enough"* with thresholds *"at least enough time for 2 full runs of the batch
   job."* The repo's `CRITICAL_RATIO = 2.0`
   (`backend/services/cycle_health.py:42`) already equals this. **No threshold
   change is needed or justified by 82.10.**
3. **Reset time is the axis the current alarm fails on.** *"Reset Time: How long
   alerts fire after an issue is resolved. Long reset times can lead to
   confusion or to issues being ignored."* (SRE Workbook). A level-triggered
   alarm on a 128-day-red table has effectively infinite reset time -- exactly
   why an edge-triggered (state-transition) gate is required, not optional.
4. **Cadence rule of thumb:** *"run your source freshness jobs with at least
   double the frequency of your lowest SLA"* (dbt). The tightest SLA in
   `_TABLE_MAX_AGE_SEC` is 26h (`historical_prices`,
   `paper_portfolio_snapshots`, `cycle_health.py:49,52`), so >= every 13h. A
   1h-or-shorter interval is far more often than necessary; see recommendation.
5. **`replace_existing=True` is mandatory** -- *"you **MUST** define an explicit
   ID for the job and use `replace_existing=True` or you will get a new copy of
   the job every time your application restarts!"* (APScheduler userguide).
6. **`coalesce` defaults to `False`.** The caller's worry about "a job that
   catches up 128 missed runs would spam" is real for APScheduler generally --
   but see Trap 6: it does **not** apply to this repo's scheduler config.

---

## 6. Internal code inventory (every file:line re-derived this session)

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `backend/services/cycle_health.py` | 585 lines total | Freshness + heartbeat | Correct; unreachable by any scheduler |
| " | `:41-42` | `WARN_RATIO = 1.5`, `CRITICAL_RATIO = 2.0` | Matches Prometheus 2x rule |
| " | `:48-53` | `_TABLE_MAX_AGE_SEC` -- 4 keys only | See Trap 4 |
| " | `:78-86` | `_band()` -> `red`/`amber`/`green`/`unknown` | |
| " | `:89-97` | `_worst_band()` helper **already exists** | Evaluator need NOT walk sources |
| " | `:100-135` | `_fire_freshness_alarm()` -- P1 per red table | **Emitter already built** |
| " | `:489` | `compute_freshness(bq, cycle_interval_sec)` | |
| " | `:561` | `overall_band = _worst_band(...)` | Top-level key `overall_band` |
| " | `:564-565` | `if overall_band == "red": _fire_freshness_alarm(sources)` | **Alert is emitted INSIDE compute_freshness** |
| " | `:567-585` | Return shape: `sources`/`overall_band`/`heartbeat`/`bq_ingest_lag_sec`/`thresholds`/`computed_at` | Per source: `last_tick_age_sec`, `interval_sec`, `ratio`, `band` |
| `backend/api/paper_trading.py` | `:497` | HTTP call site #1 | Browser-only |
| `backend/api/observability_api.py` | `:36`, `:55` | HTTP call sites #2 and #3 | Browser-only |
| `backend/backtest/macro_cron.py` | `:110-148` | **`register_macro_ingest_cron`** | **THE template** (phase-82.0, sibling step) |
| `backend/meta_evolution/cron.py` | `:43-84` | `register_meta_evolution_cron` | Same idiom, older |
| `backend/harness_self_audit_report.py` | `:84-106` | `register_harness_self_audit_cron` | Third instance of idiom |
| `backend/main.py` | `:307` | `if settings.paper_trading_enabled:` | **Gates the whole scheduler block** |
| " | `:309-312` | `AsyncIOScheduler()` created + `.start()` in FastAPI lifespan | The canonical scheduler |
| " | `:336-342` | `register_macro_ingest_cron(scheduler)` wired, fail-open | **Precedent call site for the new job** |
| `backend/slack_bot/scheduler.py` | `:113` | `_cycle_heartbeat_last_was_stale` | State-transition gate |
| " | `:117` | `_ingestion_silence_last_was_stale` | Second instance |
| " | `:761-795` | Watchdog body: `verdict -> is_stale_now -> prior -> fire only on transition` | **THE alert-fatigue template** |
| " | `:249-256` | `watchdog_health_check` interval job | Runs every `watchdog_interval_minutes` |
| `backend/services/observability/alerting.py` | `:54` | `_CRITICAL_SEVERITIES` incl. `P1` | |
| " | `:83-93` | P1 bypasses consecutive threshold, **repeat window still applies** | |
| " | `:136-176` | `_bot_token_fallback` | **The path actually alive** |
| " | `:217-218` | Empty webhook + P1 -> bot-token fallback | Confirms phase-62 finding |
| " | `:253-287` | `raise_cron_alert_sync` | The ONE function to reach the operator |
| `backend/config/settings.py` | `:123` | `slack_webhook_url` default `""` | |
| " | `:151,153` | `alert_consecutive_failure_threshold=3`, `alert_repeat_hours=1` | |
| " | `:620` | `watchdog_interval_minutes = 15` | |
| `backend/db/bigquery_client.py` | `:1117` | `def get_bq_client()` | Module-level factory |
| `tests/verify_phase_25_A7.py` | `:210-320` | Claims 8/9/10 | **Criteria 2+3 already tested at the compute_freshness level** |
| `backend/tests/test_phase_82_0_macro_ingestion.py` | `:21-29`, `:133-159` | `_StubScheduler` + registration test | **Exact criterion-1 test shape** |
| `backend/tests/test_cycle_heartbeat_alarm.py` | whole file | Sibling alarm test | Verdict-dict style |
| `tests/services/test_phase9_registration.py` | `:50-80` | misfire/coalesce invariant | **Scoped to slack_bot only** -- see Trap 6 |

**Measurement methods:** `wc -l` + targeted `grep -n` on `cycle_health.py`;
`grep -rn "compute_freshness" --include="*.py"` (3 production call sites, all
HTTP); `grep -rn "add_job" --include="*.py"`; `grep -rn "freshness" ... | grep
-i "cron\|add_job\|scheduler\|register"` returned **ZERO rows** -- there is no
existing or disabled freshness job to duplicate; `ls ~/Library/LaunchAgents |
grep pyfin` confirmed `com.pyfinagent.backend.plist` exists, so the main
scheduler really does run 24/7.

**Not measured (stated honestly):** whether `SLACK_WEBHOOK_URL` is populated in
`backend/.env` -- the researcher sandbox denies reading that file. The code
path is unambiguous either way (`alerting.py:217-218` routes P1 to the
bot-token fallback when the webhook is empty), so the design does not depend on
the answer.

---

## 7. Recommendation for the contract

**Build a ~120-line `backend/services/freshness_cron.py` in the exact
`macro_cron.py` shape. Do not touch `compute_freshness`'s maths, thresholds, or
return shape.**

```
JOB_ID = "freshness_evaluator"
_last_red_sources: set[str] | None = None      # module-level, None = unknown baseline

def run_freshness_check(*, bq=None, settings=None, notify=None) -> dict
def register_freshness_cron(scheduler, *, replace_existing=True, hours=6) -> str | None
```

1. **Trigger.** `trigger="interval", hours=6` (or `cron` 4x/day). Justified by
   dbt's ">= 2x the tightest SLA" rule: tightest SLA is 26h, so >= q13h is
   sufficient; 6h gives margin without approaching the alert-fatigue zone. **Do
   NOT use minutes-scale intervals** -- there is no SLA here shorter than 26h,
   so a fast interval buys nothing and multiplies page risk.
2. **Registration site.** `backend/main.py` immediately after the
   `register_macro_ingest_cron` block (`:336-342`), same try/except fail-open
   shape. Use the **backend `AsyncIOScheduler`**, not the slack-bot one -- the
   backend process shares the `AlertDeduper` singleton with the HTTP handlers,
   so a browser poll and the cron job dedup against each other. Putting it in
   the slack-bot process would create a SECOND deduper and double the pages.
3. **Suppress the inner emitter; own the gating.** `compute_freshness` fires
   `_fire_freshness_alarm` internally at `:564-565`. Add an opt-out parameter
   (e.g. `compute_freshness(bq, interval, emit_alarm=True)`) defaulting to the
   current behaviour so the three HTTP call sites are byte-identical, and have
   the cron pass `emit_alarm=False`. Then the cron applies the
   **state-transition gate** (`slack_bot/scheduler.py:761-795` pattern) over the
   set of red source names and calls `raise_cron_alert_sync` itself. Fire only
   on `newly_red = red_now - red_prior`. Log-only on steady-state; log-only on
   recovery.
4. **Notification.** `raise_cron_alert_sync(source="cycle_health",
   error_type=f"freshness_critical_{table}", severity="P1", ...)` -- unchanged
   from `cycle_health.py:119-131`. Keep `severity="P1"` so the empty-webhook
   bot-token fallback (`alerting.py:217-218`) engages; a P2/P3 would be
   **silently dropped** on this machine.
5. **`cycle_interval_sec`.** Pass `86400.0` via the identical
   `float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))`
   expression used at all three HTTP sites, so bands cannot drift between the
   dashboard and the pager.
6. **BQ client.** `from backend.db.bigquery_client import get_bq_client`
   (`:1117`) inside the function body, or accept an injected `bq=` for tests --
   `macro_cron.py:62` uses `bq_client or BigQueryClient(settings)`; the
   injectable-parameter shape is what makes the criterion-2/3 fixtures clean.
7. **Fail-open at the top level**, `_write_failure_receipt`-style logging
   (`macro_cron.py:77-85`), ASCII-only logger messages.

**Tests (map 1:1 to the three criteria):**
- **C1:** `_StubScheduler` (copy `test_phase_82_0_macro_ingestion.py:21-29`);
  assert `register_freshness_cron(stub)` returns `JOB_ID`, `len(stub.jobs)==1`,
  `replace_existing is True`. Plus the source-scan that `backend/main.py`
  contains `register_freshness_cron` (`:151-159` precedent) -- **and** a
  behavioural test that `run_freshness_check` calls `compute_freshness` with a
  fake BQ and **no FastAPI TestClient / no HTTP request** in the call path.
- **C2:** fake BQ with `historical_macro` age > 2x its 3_024_000s interval;
  patch `backend.services.observability.alerting.raise_cron_alert_sync`
  (**patch the alerting module, not cycle_health** -- `_fire_freshness_alarm`
  does a function-local import at `:109`, so the name resolves from the
  alerting module at call time); assert >= 1 call with `severity == "P1"` and
  `details["table"] == "historical_macro"`.
- **C3:** all-green fake BQ -> assert `call_count == 0`. Copy the
  `_build_fake_bq` helper from `tests/verify_phase_25_A7.py`.
- **Reset module state between C2 and C3** (`_last_red_sources = None`) or the
  transition gate will make one of them pass for the wrong reason.

---

## 8. Traps

1. **THE BIG ONE -- the deduper does NOT solve steady-state spam. MEASURED.**
   I ran `AlertDeduper(window_minutes=5, repeat_hours=1,
   consecutive_threshold=3)` and called `should_fire(..., severity="P1")`:
   5 back-to-back calls gave `[True, False, False, False, False]`, but after
   rewinding `last_fired_at` by 1h1m the next call returned `True`. So a P1
   re-fires **every `repeat_hours`, forever**. A 6h-interval job on a
   permanently-red table would page 4x/day indefinitely (~512 pages over a
   128-day outage). `_fire_freshness_alarm`'s docstring at `:103-105` claims
   "Dedup is handled by `AlertDeduper` ... so a polling-loop caller doesn't spam
   Slack" -- that is **true only relative to a 60s browser poll** (the phase-66
   hotfix at `alerting.py:46-53` cut 120/hr to 1/hr). It is **not** a
   steady-state suppressor. **A state-transition gate is mandatory, not
   optional.** Anything that just calls `compute_freshness` on a timer inherits
   this bug.
2. **A naive implementation double-fires.** If the cron calls
   `compute_freshness` without suppressing the internal emitter AND also does
   its own gating, red tables page twice. Suppression (recommendation 3) is
   load-bearing.
3. **`paper_cycle_interval_sec` DOES NOT EXIST in settings.** Measured:
   `grep -rn "paper_cycle_interval_sec" --include="*.py"` returns only the three
   `getattr(...)` fallbacks (`paper_trading.py:496`, `observability_api.py:40`,
   `:59`) plus one test mock -- **no field in `backend/config/settings.py`**.
   The comment at `paper_trading.py:495` even says "if future phases add one".
   So the effective value is always `86400.0`. Do not "fix" this in 82.10 and do
   not invent a different constant.
4. **`_TABLE_MAX_AGE_SEC` has only 4 keys** (`cycle_health.py:48-53`);
   `paper_trades` and `signals_log` use the caller-supplied
   `cycle_interval_sec`. An evaluator that assumes a per-table SLA for all six
   will KeyError. Use `_worst_band` / the returned `sources` dict instead of
   re-deriving.
5. **Patch target matters.** Patching
   `backend.services.cycle_health.raise_cron_alert_sync` will silently do
   nothing -- that name does not exist at module scope; the import is
   function-local at `:109`. Patch
   `backend.services.observability.alerting.raise_cron_alert_sync`. A test that
   gets this wrong shows `call_count == 0` and would make the criterion-3 guard
   pass vacuously while the criterion-2 guard fails confusingly.
6. **Do NOT bolt `misfire_grace_time`/`coalesce` on reflexively "because the
   step description warns about 128 catch-up runs".** Measured: the invariant
   test that demands them (`tests/services/test_phase9_registration.py:50-80`)
   is scoped to `slack_bot.register_phase9_jobs` **only**, and neither
   `macro_cron.py:128-137` nor `meta_evolution/cron.py:62-71` passes them. More
   importantly the catch-up risk is **not real here**: `main.py:310` constructs
   a bare `AsyncIOScheduler()` with the **default in-memory jobstore**, which
   does not persist jobs across restarts, so there is nothing to catch up. (This
   matches the standing memory that a restart can never double-fire
   `paper_trading_daily`.) Adding `coalesce=True` is harmless belt-and-braces
   and arguably good hygiene, but the contract should say **why**, not cite a
   risk that does not exist in this configuration.
7. **`settings.paper_trading_enabled` gates the whole block** (`main.py:307`).
   If paper trading is ever disabled, the freshness evaluator dies with it --
   the monitor would be disabled by the same switch that disables the thing it
   monitors. Flag it; do not silently accept it. (Fixing it is arguably out of
   scope for 82.10, but the contract should name it.)
8. **Criteria 2 and 3 are already satisfied at the `compute_freshness` level**
   by `tests/verify_phase_25_A7.py` claims 8 and 9. A new test that merely
   re-asserts them against `compute_freshness` would be a **guard that cannot
   fail** on the pre-fix tree -- the exact anti-pattern in
   `feedback_mutation_test_guards_and_fixtures`. The new tests must drive the
   **scheduled entry point** (`run_freshness_check`), which does not exist
   pre-fix, so they fail at import against the current tree.
9. **`raise_cron_alert_sync` returns `True` optimistically** when a running
   event loop exists (`alerting.py:274-277` -- fire-and-forget
   `loop.create_task`). Inside an `AsyncIOScheduler` job there IS a running
   loop, so the return value is **not** evidence of delivery. Do not assert on
   it; assert on the captured call.
10. **`_HANDOFF.mkdir` runs at import** (`cycle_health.py:34-35`) -- importing
    the module has a filesystem side effect. Harmless, but it means the module
    cannot be imported in a read-only context.

---

## 9. Where the step description is WRONG or STALE (high value)

1. **"the data-freshness alarm is BROWSER-DRIVEN and therefore cannot page"** --
   **half wrong.** The *trigger* is browser-driven (correct), but the *alarm and
   the paging path already exist and work*: `_fire_freshness_alarm`
   (`cycle_health.py:100-135`) emits a **P1** through `raise_cron_alert_sync`,
   and `compute_freshness` already calls it (`:564-565`). It genuinely paged --
   every time a browser hit the endpoint with a red table. The phase-66 hotfix
   note at `alerting.py:46-53` documents a **real page storm from this exact
   alarm** ("~120 pages/hour the moment a dashboard tab was open against a red
   table"), which is direct evidence the emitter was live. **The step is
   therefore much smaller than described: add a trigger, add a transition gate.
   Do not build a new alerting channel.**
2. **"give freshness an ACTIVE evaluator ... and emits through a channel the
   operator actually receives (the existing Slack / away-ops path)"** -- the
   channel work is already done; only the trigger is missing.
3. **"a job that catches up 128 missed runs would spam"** -- **not applicable to
   this repo's configuration.** `main.py:310` uses a default in-memory jobstore;
   missed runs across a restart do not exist to be caught up. See Trap 6.
4. **Implied claim that the operator had no alert at all for 128 days** --
   more precisely, the operator had an alert that fired *only while a dashboard
   tab was open*, and (per the phase-66 hotfix) when one was open it fired so
   hard it had to be throttled. The failure is **coupling to a human's browser**,
   not absence of an emitter. Worth stating accurately in the contract because
   it changes the fix.
5. **Unstated but load-bearing:** the step description does not mention alert
   *reset time* / steady-state suppression, yet without it the fix converts a
   silent alarm into a 4-pages-a-day-forever alarm -- an arguably worse outcome
   (SRE Workbook: long reset times "lead to ... issues being ignored"). The
   contract MUST add this even though it is not in the three immutable criteria.

---

## 10. Research Gate Checklist

Hard blockers:
- [x] >= 5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total (**30**)
- [x] Recency scan (last 2 years) performed + reported (section 4)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (section 6), all re-derived

Soft checks:
- [x] Internal exploration covered every module named in the caller's 6-point scope
- [x] Contradictions noted (step description vs measured code, section 9)
- [x] Claims cited per-claim with URLs + file:line
- [ ] Not measured: `SLACK_WEBHOOK_URL` value (sandbox denies `backend/.env`) -- stated in section 6

---

## 11. JSON gate envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 23,
  "urls_collected": 30,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "The step description is half wrong: the freshness alarm's EMITTER already exists and already pages P1 (cycle_health.py:100-135, called at :564-565). Only the TRIGGER is missing. Build backend/services/freshness_cron.py in the macro_cron.py shape (phase-82.0 sibling, backend/backtest/macro_cron.py:110-148), register it in main.py next to :336-342 on the backend AsyncIOScheduler. CRITICAL, MEASURED: the AlertDeduper does NOT suppress steady state -- a P1 re-fires every repeat_hours forever, so a timer-driven caller would page ~4x/day for 128 days. A state-transition gate (slack_bot/scheduler.py:761-795 idiom) is mandatory. Suppress the inner emitter so the cron owns gating. Interval 6h per dbt's 2x-tightest-SLA rule (tightest SLA 26h). paper_cycle_interval_sec does not exist in settings; always 86400.0. Patch alerting.raise_cron_alert_sync, not cycle_health. Criteria 2+3 are already tested at the compute_freshness level (verify_phase_25_A7.py claims 8/9), so new guards must drive the scheduled entry point or they cannot fail.",
  "brief_path": "handoff/current/research_brief_82.10.md",
  "gate_passed": true
}
```

**Status: COMPLETE. The JSON envelope above is the last content in this file.**
