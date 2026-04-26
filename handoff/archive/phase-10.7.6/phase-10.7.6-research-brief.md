# Research Brief: phase-10.7.6 -- Weekly APScheduler Wiring for the Meta-Evolution Loop

Tier assumption: **moderate** (stated in caller prompt).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html | 2026-04-26 | Official doc | WebFetch | Full CronTrigger signature: `CronTrigger(year, month, day, week, day_of_week, hour, minute, second, start_date, end_date, timezone, jitter)`; `day_of_week` accepts `"sun"`; `timezone` accepts `datetime.tzinfo`; `get_next_fire_time(previous_fire_time, now)` returns next dt; DST note: "This is not a bug." |
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | 2026-04-26 | Official doc | WebFetch | "If you schedule jobs in a persistent job store during your application's initialization, you MUST define an explicit ID for the job and use `replace_existing=True` or you will get a new copy of the job every time your application restarts!" |
| https://github.com/agronholm/apscheduler/blob/master/tests/triggers/test_cron.py | 2026-04-26 | Official test code | WebFetch | Upstream test pattern: instantiate `CronTrigger(...)` directly, call `trigger.next()` (APScheduler 4.x) or `trigger.get_next_fire_time(None, now)` (3.x) and assert on the returned datetime; DST fold-testing uses `replace(tzinfo=..., fold=1)` |
| https://sre.google/sre-book/distributed-periodic-scheduling/ | 2026-04-26 | Authoritative book (Google SRE) | WebFetch | "we prefer to 'fail closed' to avoid systemically creating bad state"; pre-computed job names include scheduled launch time to prevent duplicate launches across failovers; "Cron job owners can (and should!) monitor their cron jobs" |
| https://martinheinz.dev/blog/39 | 2026-04-26 | Authoritative blog | WebFetch | BackgroundScheduler pattern: `scheduler = BackgroundScheduler(..., timezone=utc)`; crontab form: `scheduler.add_job(task, CronTrigger.from_crontab('0 17 * * sat,sun'))`; `replace_existing` not shown but `scheduler.print_jobs()` confirms job IDs are tracked |
| https://apscheduler.readthedocs.io/en/latest/modules/triggers/cron.html | 2026-04-26 | Official doc (3.11.2 latest) | WebFetch | Same API as 3.x; confirms `day_of_week='sun'` is valid; `from_crontab('0 2 * * 0')` maps to Sunday 02:00 |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://coderslegacy.com/python/apscheduler-cron-trigger/ | Blog tutorial | Fetched but only had partial content; day_of_week='mon-fri' example confirmed; no new patterns beyond official docs |
| https://pypi.org/project/APScheduler/ | PyPI | Snippet shows APScheduler 3.11.2 (3.x line) is current stable; 4.0a6 exists but is alpha -- project pin `>=3.10.0` confirmed correct |
| https://github.com/agronholm/apscheduler/blob/master/tests/test_schedulers.py | Test code | Snippet confirms `MagicMock` used for trigger mocking; `next_run_time` asserted on job objects |
| https://github.com/agronholm/apscheduler/issues/346 | GitHub issue | CronTrigger.from_crontab() timezone default to scheduler timezone confirmed; explicit tz kwarg preferred |
| https://medium.com/@ThinkingLoop/7-scheduler-strategies-for-python-jobs-celery-rq-arq-48b1eb5f8f79 | Blog | Covers Celery/RQ patterns; APScheduler section confirms idempotency-key pattern for deduplication |
| https://github.com/agronholm/apscheduler/discussions/637 | GitHub discussion | AndTrigger combining pattern; not relevant for single weekly cron |
| https://cronbase.dev/cron/weekly-sunday-8pm | Tool | Confirms cron expression `0 20 * * 0` for Sunday 8pm; `0 2 * * 0` for Sunday 2am |

---

## Recency scan (2024-2026)

Searched "APScheduler weekly cron sunday timezone python 2026" and "APScheduler 3 CronTrigger day_of_week test pytest 2024 2025". Result:

- APScheduler 4.0.0a6 released April 27, 2025 and 4.0.0a5 May 15, 2024. The 4.x API changed from `get_next_fire_time()` to `trigger.next()`. Since the project pin is `>=3.10.0` and the installed version is 3.11.2, the 3.x API (`get_next_fire_time(prev, now)`) is authoritative. Do NOT use the 4.x `trigger.next()` API in tests.
- No new canonical papers on APScheduler job testing patterns in 2024-2026; patterns remain: StubScheduler + direct trigger instantiation + `get_next_fire_time` assertions.
- Google SRE pattern (fail-closed for non-idempotent, fail-open for monitoring) still the canonical reference for meta-evolution cron discipline.
- No superseding patterns found; 3.x `CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=ZoneInfo("America/New_York"))` + `replace_existing=True` remains the correct approach.

---

## Key findings

1. **CronTrigger for Sunday 2am ET is straightforward.** `CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=ZoneInfo("America/New_York"))` is the exact constructor call. Alternatively `CronTrigger.from_crontab("0 2 * * 0", timezone=ZoneInfo("America/New_York"))` is equivalent. (Source: APScheduler 3.x docs, https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html)

2. **`replace_existing=True` is mandatory.** Without it, every process restart creates a duplicate job in any persistent jobstore. The slack_bot scheduler already applies this pattern for every job (file:line `backend/slack_bot/scheduler.py:43`, `56`, `65`, `79`). (Source: APScheduler userguide, https://apscheduler.readthedocs.io/en/3.x/userguide.html)

3. **BackgroundScheduler vs AsyncIOScheduler.** The meta-evolution cron module should use `BackgroundScheduler` since `run_meta_evolution_cycle()` is synchronous (calls pure-function modules: `cron_allocator.allocate()`, `archetype_library.get_archetype()`, `alpha_velocity.compute_alpha_velocity()`, `provider_rebalancer.allocate()`). The Slack bot uses `AsyncIOScheduler` because its job functions are `async`. The new module's jobs are sync, so `BackgroundScheduler` is the correct choice. (Source: APScheduler userguide, https://apscheduler.readthedocs.io/en/3.x/userguide.html)

4. **Testing without a live scheduler.** Two valid patterns from the codebase:
   - **StubScheduler** (preferred; used in `tests/slack_bot/test_scheduler_phase9.py:14-21`): a minimal class with `add_job()` that records calls -- no real scheduler started, no threads, no DST-sensitive timers.
   - **Direct `CronTrigger.get_next_fire_time()` assertion**: instantiate the trigger, call `get_next_fire_time(None, reference_now)`, assert the returned datetime is the expected Sunday 02:00 ET. This is how the upstream APScheduler test suite tests triggers (file fetched from GitHub tests/triggers/test_cron.py).

5. **Fail-open discipline for the meta-evolution cycle.** The Google SRE book distinguishes fail-closed (payroll, never skip) from fail-open (monitoring, safe to skip). Meta-evolution reallocation is monitoring-tier: a skipped Sunday is acceptable; a crashed cycle that partially applies cron_allocator output is risky. All sub-calls should be individually try/except wrapped, with the exception logged and the cycle continuing. This mirrors the existing pattern in `register_phase9_jobs()` (`backend/slack_bot/scheduler.py:375-382`).

6. **APScheduler 3.x `add_job` with trigger string vs CronTrigger object.** Both are valid. The project uses the string form (`trigger="cron", day_of_week="sun", hour=2`) in `register_phase9_jobs`. For the new module either form works; string form is simpler and consistent with the existing codebase idiom.

7. **Idempotent weekly job key.** Per SRE book: pre-computed job names that include the scheduled window prevent duplicate launches. Using a fixed `id="meta_evolution_weekly"` combined with `replace_existing=True` satisfies this for single-node (no distributed coordination needed for a local-only Mac deployment).

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 383 | APScheduler cron hub; `AsyncIOScheduler`; all existing jobs use `replace_existing=True` + `ZoneInfo("America/New_York")` | Active; pattern source |
| `backend/autoresearch/cron.py` | 79 | In-memory shim for autoresearch; `register()` accepts a scheduler and calls `add_job()`; fail-open pattern | Active; structural precedent for a separate cron module |
| `backend/meta_evolution/alpha_velocity.py` | 161 | `AlphaVelocitySample` dataclass + `compute_alpha_velocity()` + `persist_sample(bq_client, sample)` (fail-open) | Active; entry point for the cron cycle |
| `backend/meta_evolution/directive_rewriter.py` | 411 | `rewrite_directive(current_text, briefs, ...)` + `persist_version(bq_client, version)` (fail-open); returns `None` if < 5 briefs or score < 0.6 | Active; optional sub-call in cron cycle |
| `backend/meta_evolution/archetype_library.py` | 253 | `ARCHETYPES` tuple; `get_archetype(strategy_id)` pure lookup | Active; read-only in cron cycle |
| `backend/meta_evolution/cron_allocator.py` | 158 | `allocate(yaml_path)` -> `{job_name: tokens}` pure; reads `.claude/cron_budget.yaml` | Active; called in cron cycle |
| `backend/meta_evolution/provider_rebalancer.py` | 230 | `allocate(yaml_path)` -> `{provider: usd}` pure; reads `.claude/provider_budget.yaml` | Active; called in cron cycle |
| `backend/meta_evolution/__init__.py` | 21 | Package init; exports archetype_library symbols only | Active; needs cron.py added to exports |
| `.claude/cron_budget.yaml` | 190 | Slot definitions; slot 14 = `meta_evolution_weekly_reallocation`, cadence `weekly`, priority `medium`, `alpha_velocity_eligible: false` | Active; slot 14 is the canonical Sunday slot |
| `tests/slack_bot/test_scheduler_phase9.py` | 54 | `StubScheduler` class pattern + job-ID assertions | Active; template for `tests/scheduler/test_meta_cron.py` |
| `tests/meta_evolution/test_cron_allocator.py` | 256 | `tmp_path` fixture + YAML write helper; subprocess validator; `pytest.raises` pattern | Active; template for allocator sub-call test |

---

## Consensus vs debate (external)

**Consensus:**
- `replace_existing=True` + explicit `id=` are non-negotiable for any job registered at startup (all sources agree).
- `ZoneInfo("America/New_York")` is the correct DST-aware TZ for US Eastern; never pass a bare string `"US/Eastern"` (deprecated alias; `zoneinfo` is the stdlib approach since Python 3.9).
- StubScheduler pattern (or direct `get_next_fire_time` on the trigger object) is the correct way to test APScheduler without a live scheduler thread.

**No debate:**
- 3.x vs 4.x API: 4.x alpha exists but the project is pinned to 3.x. No migration in scope.

---

## Pitfalls (from literature + code audit)

1. **DST ambiguity at 2am ET.** US Eastern falls back in November; 2am occurs twice. APScheduler's documented behavior: "This is not a bug" -- it may fire once or twice depending on fold. Mitigation: slot 14 is monitoring-only (meta-evolution reallocation is fail-open); double-fire is acceptable. If it becomes a problem, shift to 03:00 ET (post-transition).

2. **`provider_budget.yaml` may not exist yet.** `provider_rebalancer.py` reads `.claude/provider_budget.yaml`. The cron cycle should `try/except` this call individually and log a warning if the file is missing, not raise.

3. **`directive_rewriter.rewrite_directive()` may call an LLM.** If the weekly cron fires during a low-budget period, the LLM call inside `directive_rewriter` could fail or cost tokens. The function is already fail-open (returns `None` on LLM failure); the cron wrapper just needs to log the `None` result and continue.

4. **`persist_sample` requires a real BQ client.** Tests must not pass a real `google.cloud.bigquery.Client`. Use a `FakeBQ` stub (`insert_rows_json` returns `[]`). This pattern is implied by `alpha_velocity.py:138` docstring: "tests pass a FakeBQ stub".

5. **Job module location.** Adding the weekly job to `backend/slack_bot/scheduler.py` would couple meta-evolution to the Slack bot process. The cleaner path mirrors `backend/autoresearch/cron.py`: a standalone `backend/meta_evolution/cron.py` that can be imported by any scheduler (Slack bot, standalone process, test). The Slack bot's `start_scheduler()` can call `register_meta_evolution_cron(scheduler)` as a one-liner -- no file bloat.

6. **`tests/scheduler/` does not exist.** The masterplan verification command is `python -m pytest tests/scheduler/test_meta_cron.py`. The directory and `__init__.py` must be created.

---

## Application to pyfinagent (mapping to file:line anchors)

| Design decision | Anchor in existing code |
|----------------|------------------------|
| Use `CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=ZoneInfo("America/New_York"))` | `scheduler.py:35-43` (pattern: hour + timezone + replace_existing) |
| `id="meta_evolution_weekly"`, `replace_existing=True` | `scheduler.py:43`, `56`, `65`, `79` -- every existing job uses both |
| Separate `backend/meta_evolution/cron.py` | `backend/autoresearch/cron.py:17-41` -- same register(scheduler) shim pattern |
| `register_meta_evolution_cron(scheduler)` function signature | `scheduler.py:351` -- `register_phase9_jobs(scheduler, replace_existing=True)` |
| Fail-open per sub-call: `try: ... except Exception as e: logger.warning(...)` | `scheduler.py:375-381` -- each phase-9 job import is individually try/except |
| `StubScheduler` in tests | `tests/slack_bot/test_scheduler_phase9.py:14-21` |
| `FakeBQ.insert_rows_json` stub | `alpha_velocity.py:138` docstring |
| `tmp_path` fixture for YAML reads | `tests/meta_evolution/test_cron_allocator.py:49-58` |

---

## Proposed module structure for `backend/meta_evolution/cron.py`

```python
# backend/meta_evolution/cron.py  (~120 LOC)
from __future__ import annotations
import logging
from zoneinfo import ZoneInfo
from typing import Any

logger = logging.getLogger(__name__)

_JOB_ID = "meta_evolution_weekly"
_CRON_BUDGET_YAML = Path(__file__).resolve().parents[2] / ".claude" / "cron_budget.yaml"
_PROVIDER_BUDGET_YAML = Path(__file__).resolve().parents[2] / ".claude" / "provider_budget.yaml"


def register_meta_evolution_cron(scheduler: Any, *, replace_existing: bool = True) -> bool:
    """Add the weekly meta-evolution job to a scheduler. Returns True if successful."""
    ...

def run_meta_evolution_cycle(
    *,
    cron_budget_yaml: Path | None = None,
    provider_budget_yaml: Path | None = None,
) -> dict[str, Any]:
    """Execute one weekly meta-evolution cycle. Fail-open: individual sub-call
    failures are caught and logged; the dict always returns."""
    results: dict[str, Any] = {}
    # 1. archetype_library -- pure lookup, no I/O
    # 2. cron_allocator.allocate(cron_budget_yaml)
    # 3. provider_rebalancer.allocate(provider_budget_yaml)
    # 4. optional: directive_rewriter.rewrite_directive (LLM, fail-open)
    # 5. telemetry: log results summary
    return results
```

---

## Proposed test plan for `tests/scheduler/test_meta_cron.py`

8 tests (aligns with the ~6-8 stated in the prompt):

1. `test_register_adds_job_to_scheduler` -- StubScheduler receives one job with id=`meta_evolution_weekly`
2. `test_job_id_is_meta_evolution_weekly` -- job id matches constant
3. `test_register_twice_uses_replace_existing` -- second call passes `replace_existing=True` (no duplicate)
4. `test_trigger_fires_sunday_2am_et` -- instantiate `CronTrigger(day_of_week="sun", hour=2, minute=0, timezone=ZoneInfo("America/New_York"))`, call `get_next_fire_time(None, reference_monday)`, assert result is a Sunday and hour==2
5. `test_timezone_is_explicitly_new_york` -- assert the registered job's trigger has `ZoneInfo("America/New_York")` (check trigger kwargs or trigger.timezone attr)
6. `test_run_cycle_calls_cron_allocator` -- monkeypatch `cron_allocator.allocate` to a spy; assert it was called
7. `test_run_cycle_calls_provider_rebalancer` -- same for `provider_rebalancer.allocate`
8. `test_run_cycle_handles_bq_failure_gracefully` -- pass a FakeBQ that raises; assert no exception propagates from `run_meta_evolution_cycle()`

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total (incl. snippet-only) (13 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (all 7 meta_evolution/*.py files read, scheduler.py read, autoresearch/cron.py read, test patterns read)
- [x] Contradictions / consensus noted (APScheduler 3.x vs 4.x; fail-open vs fail-closed)
- [x] All claims cited per-claim (not just listed in a footer)

---

## Search queries run (three-variant discipline)

1. **Current-year frontier (2026):** "APScheduler weekly cron sunday timezone python 2026"
2. **Last-2-year (2024-2025):** "APScheduler 3 CronTrigger day_of_week test pytest 2024 2025"
3. **Year-less canonical:** "APScheduler CronTrigger weekly day_of_week configuration", "idempotent cron job design patterns Python scheduler 2025", "APScheduler BackgroundScheduler unit test CronTrigger next_run_time assert"

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 7,
  "urls_collected": 13,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/phase-10.7.6-research-brief.md",
  "gate_passed": true
}
```
