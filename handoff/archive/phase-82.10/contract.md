# Contract -- masterplan step 82.10

**Step id:** 82.10 (phase-82, priority P1, harness_required: true)
**Date:** 2026-08-05
**Cycle:** 1

---

## 1. Research gate summary

**Brief:** `handoff/current/research_brief_82.10.md`
**Envelope:** `gate_passed: true` -- `external_sources_read_in_full: 7`,
`snippet_only_sources: 23`, `urls_collected: 30`, `recency_scan_performed: true`,
`internal_files_inspected: 14`, `tier: moderate`, `audit_class: false`.

**The gate changed the step.** Three findings, each independently
**re-measured by Main** before being adopted (commands + outputs recorded in
`experiment_results.md` section 0):

1. **The step description is half wrong.** It says the alarm "cannot page". The
   *emitter already exists and already pages P1*:
   `backend/services/cycle_health.py:100-135` (`_fire_freshness_alarm`) calls
   `raise_cron_alert_sync(severity="P1", ...)`, and `compute_freshness` already
   invokes it at `:564-565` (`if overall_band == "red"`). The phase-66 hotfix
   note at `backend/services/observability/alerting.py:46-53` documents a real
   page storm *from this exact alarm* -- direct evidence the emitter was live.
   **Only the TRIGGER is missing.** This step therefore adds a trigger and a
   gate; it does NOT build an alerting channel.

2. **THE LOAD-BEARING TRAP: `AlertDeduper` does NOT suppress steady state.**
   A P1 bypasses the consecutive threshold but the *repeat window still
   applies per (source, error_type)* -- meaning it re-fires **every
   `repeat_hours`, forever**. MAIN RE-MEASURED THIS:

   ```
   P1 back-to-back:                        [True, False, False, False, False]
   after rewinding last_fired_at by 1h1m:  True
   ```

   So a naive timer-driven caller on a permanently-red table would page
   ~4x/day for 128 days (~512 pages). `_fire_freshness_alarm`'s own docstring
   at `:103-105` claims the deduper stops a "polling-loop caller" from
   spamming -- true only relative to a 60s browser poll, **not** a
   steady-state suppressor. **A state-transition gate is mandatory, not
   optional.** Any implementation that merely calls `compute_freshness` on a
   timer ships a page storm.

3. **Two test-correctness traps.** (a) `raise_cron_alert_sync` is imported
   **function-locally** at `cycle_health.py:109` and `:234` -- MAIN VERIFIED
   there is no module-scope binding -- so a test patching
   `backend.services.cycle_health.raise_cron_alert_sync` silently patches
   nothing, which would make criterion 3 pass *vacuously*. Patch
   `backend.services.observability.alerting.raise_cron_alert_sync`.
   (b) Criteria 2 and 3 are **already** exercised at the `compute_freshness`
   level by `tests/verify_phase_25_A7.py` claims 8-9, so a new test that
   re-asserts them there is a guard that cannot fail on the pre-fix tree. The
   new guards MUST drive the scheduled entry point.

**Also verified by Main:** `paper_cycle_interval_sec` has **no field in
`backend/config/settings.py`** -- only three `getattr(...)` fallbacks
(`paper_trading.py:496`, `observability_api.py:40`, `:59`). The effective value
is always `86400.0`. Not in scope to "fix"; the cron reuses the identical
expression so dashboard and pager bands cannot drift.

---

## 2. Hypothesis

`compute_freshness` is correct and its notification path is live; the defect is
purely that nothing **invokes** it without a browser. Adding a scheduled
evaluator on the existing backend `AsyncIOScheduler` -- with a
**state-transition gate** so only a *newly*-red source pages -- converts a
correct-but-unreachable alarm into a monitor, without touching the freshness
mathematics, thresholds, or return shape, and without changing the behaviour of
any of the three existing HTTP call sites.

---

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. `a scheduled evaluator invokes compute_freshness without any HTTP request, asserted by a test against a stub scheduler`
2. `a fixture in which a source breaches its critical threshold produces an outbound alert through the operator notification path, asserted by a test capturing the emitted alert`
3. `a fixture in which all sources are healthy produces NO alert, so the guard cannot pass by always firing`

**Verification command (immutable):**

```
source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q
```

---

## 4. Plan

### 4.1 New module `backend/services/freshness_cron.py`

Built in the **exact `backend/backtest/macro_cron.py:110-148` idiom** (the
phase-82.0 sibling), which is itself the third instance of the repo's
`register_*_cron` pattern (`backend/meta_evolution/cron.py:43-84`,
`backend/harness_self_audit_report.py:84-106`).

```
JOB_ID = "freshness_evaluator"
DEFAULT_INTERVAL_HOURS = 6

_last_red_sources: set[str] | None = None    # module state; None = no baseline yet

def run_freshness_check(*, bq=None, settings=None, notify=None) -> dict
def register_freshness_cron(scheduler, *, replace_existing=True,
                            hours=DEFAULT_INTERVAL_HOURS) -> str | None
def reset_transition_state() -> None          # test seam
```

- **Interval 6h.** Justified by the dbt/data-observability ">= 2x the tightest
  SLA" rule: the tightest SLA in `_TABLE_MAX_AGE_SEC` is 26h, so a >=13h cadence
  suffices; 6h adds margin while staying far from the alert-fatigue zone. A
  minutes-scale interval buys nothing (no SLA here is shorter than 26h) and
  multiplies page risk.
- **BQ client injectable** (`bq or get_bq_client()`), mirroring
  `macro_cron.py:62`. This is what makes the criterion-2/3 fixtures clean.
- **`cycle_interval_sec`** computed with the byte-identical expression used at
  the three HTTP sites: `float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))`.
- **Fail-open at the top level** (`macro_cron.py:77-85` shape); ASCII-only
  logger messages per `.claude/rules/security.md`.

### 4.2 State-transition gate (the anti-page-storm mechanism)

```
red_now   = {name for name, info in payload["sources"].items() if info["band"] == "red"}
newly_red = red_now - (_last_red_sources or set())
fire for each source in newly_red;  then _last_red_sources = red_now
```

- **Steady-state red -> log only, NO page.** This is the whole point.
- **Recovery (red -> green) -> log only, NO page.** Deliberate: criterion 3
  requires an all-healthy fixture to emit no alert, and a resolution page would
  make that criterion ambiguous. Recorded as a conscious trade-off, not an
  oversight.
- **First run after process start (`_last_red_sources is None`) -> a red source
  DOES page.** Deliberate: on restart the operator should be told the table is
  red. The backend runs under launchd 24/7 so restarts are rare and operator-
  initiated. Explicitly tested so the choice is visible rather than incidental.

### 4.3 Suppress the inner emitter (`backend/services/cycle_health.py`)

Add an opt-out parameter to `compute_freshness`:

```
def compute_freshness(bq, cycle_interval_sec, *, emit_alarm: bool = True) -> dict
    ...
    if emit_alarm and overall_band == "red":
        _fire_freshness_alarm(sources)
```

`emit_alarm` is **keyword-only with a `True` default**, so all three existing
HTTP call sites are behaviourally byte-identical and need no edit. The cron
passes `emit_alarm=False` and owns the gating. Without this, a red table pages
twice per cron tick (once from inside `compute_freshness`, once from the gate).

### 4.4 Registration in `backend/main.py`

Immediately after the `register_macro_ingest_cron` block (currently `:336-342`),
same `try/except` fail-open shape, on the **backend `AsyncIOScheduler`** created
at `:310`. The backend process is the correct host because it shares the
`AlertDeduper` singleton with the HTTP handlers -- registering in the separate
slack_bot process would create a second deduper and double the pages.

### 4.5 Tests -- `backend/tests/test_phase_82_10_freshness_paging.py`

Each guard must be able to OBSERVE its defect, drive the subject's real channel,
and assert its own preconditions took effect.

- **C1** -- `_StubScheduler` (shape copied from
  `backend/tests/test_phase_82_0_macro_ingestion.py:21-29`): assert
  `register_freshness_cron(stub)` returns `JOB_ID`, registers exactly one job,
  and passes `replace_existing=True`. PLUS a **behavioural** test that
  `run_freshness_check(bq=fake)` actually calls `compute_freshness` with **no
  HTTP layer in the call path** (no TestClient, no ASGI app imported), asserted
  by a spy on `compute_freshness`. Plus the `main.py` wiring check.
- **C2** -- fake BQ whose `historical_macro` age exceeds `2 x 3_024_000s`;
  patch `backend.services.observability.alerting.raise_cron_alert_sync` (the
  **correct** target per section 1.3a); assert >= 1 call with
  `severity == "P1"` and `details["table"] == "historical_macro"`. The test
  **asserts its own precondition**: that the computed band for that source
  really is `"red"`, so a fixture that failed to breach cannot pass silently.
- **C3** -- all-green fake BQ; assert `call_count == 0`, and assert the
  precondition that `overall_band != "red"`, so the zero is a real negative
  rather than a broken fixture.
- **Anti-vacuity guard** -- a test asserting that the *wrong* patch target
  (`backend.services.cycle_health.raise_cron_alert_sync`) does not exist,
  pinning trap section 1.3a so a future refactor that moves the import to
  module scope forces this file to be revisited.
- **State reset** via `reset_transition_state()` between C2 and C3, else the
  transition gate makes one of them pass for the wrong reason.
- **Mutation matrix** in `experiment_results.md`: for each guard, a concrete
  production mutation that kills it, run with `pytest -rf` so the report names
  the tests **actually** killed rather than the ones intended.

### 4.6 Explicitly OUT of scope (named, not silently dropped)

- **`settings.paper_trading_enabled` gates the whole scheduler block**
  (`main.py:307`), so the freshness monitor would be disabled by the same
  switch that disables one of the things it monitors. Real coupling defect;
  **queued as its own masterplan step**, not fixed here (fixing it means
  restructuring scheduler construction, well outside this step's surface).
- `paper_cycle_interval_sec` not existing as a settings field.
- `misfire_grace_time` / `coalesce`: the catch-up risk implied by the step
  description does not exist in this configuration -- `main.py:310` constructs a
  bare `AsyncIOScheduler()` with the default **in-memory** jobstore, so there
  are no persisted missed runs to replay. Not added; rationale recorded rather
  than the risk cited.
- No change to freshness mathematics, thresholds, `_TABLE_MAX_AGE_SEC`, or the
  `compute_freshness` return shape.

---

## 5. Files expected to change

| File | Change |
|------|--------|
| `backend/services/freshness_cron.py` | NEW -- scheduled evaluator + transition gate |
| `backend/services/cycle_health.py` | `compute_freshness` gains keyword-only `emit_alarm=True` |
| `backend/main.py` | register the cron beside the macro cron |
| `backend/tests/test_phase_82_10_freshness_paging.py` | NEW -- the immutable verification target |
| `.claude/masterplan.json` | new queued step for the `paper_trading_enabled` coupling; 82.10 status flip LAST |

---

## 6. References

- `handoff/current/research_brief_82.10.md` (7 sources read in full; envelope `gate_passed: true`)
- Google SRE Book -- *Monitoring Distributed Systems* / *Practical Alerting* (symptom-based alerting; alert fatigue)
- Prometheus alerting docs (`for:` duration, state transitions vs level-triggered spam)
- APScheduler userguide (`replace_existing`, interval triggers, jobstore semantics)
- dbt / data-observability freshness-SLA prior art (cadence >= 2x tightest SLA)
- Internal precedent: `backend/backtest/macro_cron.py:110-148`,
  `backend/meta_evolution/cron.py:43-84`,
  `backend/harness_self_audit_report.py:84-106`,
  state-transition idiom at `backend/slack_bot/scheduler.py:761-795`
