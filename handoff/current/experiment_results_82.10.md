# Experiment Results -- masterplan step 82.10

**Step:** 82.10 (P1) -- the data-freshness alarm is browser-driven and cannot page
**Date:** 2026-08-05 | **Cycle:** 1
**Verification command (immutable):**
`source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q`

---

## 0. Research findings RE-MEASURED by Main before being adopted

The brief made three load-bearing claims. Main did not take them on trust.

**(a) `AlertDeduper` does not suppress steady state** -- verbatim output:

```
$ python -c "
from datetime import timedelta
from backend.services.observability.alerting import AlertDeduper
d = AlertDeduper(window_minutes=5, repeat_hours=1, consecutive_threshold=3)
print('P1 back-to-back:', [d.should_fire('cycle_health','freshness_critical_historical_macro',severity='P1') for _ in range(5)])
st = d._state[('cycle_health','freshness_critical_historical_macro')]
st.last_fired_at = st.last_fired_at - timedelta(hours=1, minutes=1)
print('after rewinding last_fired_at by 1h1m:', d.should_fire('cycle_health','freshness_critical_historical_macro',severity='P1'))
"
P1 back-to-back: [True, False, False, False, False]
after rewinding last_fired_at by 1h1m: True
```

CONFIRMED. A P1 re-fires every `alert_repeat_hours` forever. A bare timer on a
permanently-red table pages ~4x/day -- roughly 512 pages over the 128-day
outage that motivated this step. **The state-transition gate is therefore not
a nicety; it is the difference between a monitor and a page storm.**

**(b) `paper_cycle_interval_sec` has no settings field** -- verbatim:

```
$ grep -rn "paper_cycle_interval_sec" backend --include="*.py"
backend/api/paper_trading.py:495:    # settings.paper_cycle_interval_sec if future phases add one.
backend/api/paper_trading.py:496:    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
backend/api/observability_api.py:40:    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
backend/api/observability_api.py:59:    cycle_interval_sec = float(getattr(settings, "paper_cycle_interval_sec", 24 * 3600.0))
```

CONFIRMED -- three `getattr` fallbacks, no field. Effective value always
`86400.0`. Not fixed here; the cron reuses the identical expression so the
pager and the dashboard cannot disagree about a band.

**(c) `raise_cron_alert_sync` is bound function-locally in `cycle_health`** --
verbatim:

```
$ grep -n "raise_cron_alert_sync" backend/services/cycle_health.py
103:    Dedup is handled by `AlertDeduper` inside `raise_cron_alert_sync` so a
109:        from backend.services.observability.alerting import raise_cron_alert_sync
119:            raise_cron_alert_sync(
234:        from backend.services.observability.alerting import raise_cron_alert_sync
241:        raise_cron_alert_sync(
```

CONFIRMED -- `:109` and `:234` are inside function bodies; there is no
module-scope binding. Patching `backend.services.cycle_health.raise_cron_alert_sync`
would silently patch nothing. All test patches target the alerting module, and
`test_wrong_patch_target_does_not_exist` pins the fact.

---

## 1. What was built

| File | Change |
|------|--------|
| `backend/services/freshness_cron.py` | **NEW** (256 lines). `JOB_ID` `:53`, `DEFAULT_INTERVAL_HOURS=6` `:62`, `_last_red_sources` `:69`, `reset_transition_state` `:72`, `_red_sources` `:83`, `run_freshness_check` `:99`, `register_freshness_cron` `:229`. |
| `backend/services/cycle_health.py` | `compute_freshness` gains keyword-only `emit_alarm: bool = True` (`:490`); the alarm dispatch becomes `if emit_alarm and overall_band == "red":` (`:581`). |
| `backend/main.py` | Registers the cron beside the macro cron -- import `:353`, call `:355`. |
| `backend/tests/test_phase_82_10_freshness_paging.py` | **NEW** -- 13 tests, the immutable verification target. |

**Design, and why each piece is load-bearing:**

1. **Trigger.** `trigger="interval", hours=6` on the backend `AsyncIOScheduler`.
   Cadence from the dbt source-freshness rule (check at >= 2x the tightest
   SLA); the tightest SLA in `_TABLE_MAX_AGE_SEC` is 26h, so anything <= 13h
   suffices and 6h leaves margin. A minutes-scale interval buys nothing here.
2. **The backend process, not slack_bot.** The backend shares the
   `AlertDeduper` singleton with the HTTP handlers, so a browser poll and a
   cron tick dedup against each other. Registering in the separate slack_bot
   process would create a second deduper and double the pages.
3. **State-transition gate.** Fires only on `newly_red = red_now - red_prior`,
   keyed on the SET of red source names so a second table going dark while the
   first is still red is not swallowed. Steady-state red -> log only. Recovery
   -> log only.
4. **`emit_alarm=False`.** Suppresses the level-triggered path inside
   `compute_freshness` so the cron owns gating; without it every tick pages
   twice, and the inner path re-fires forever regardless of the gate.
5. **`emit_alarm` defaults True and is keyword-only**, so the three HTTP call
   sites are behaviourally unchanged and needed no edit. Two tests pin that.

**Deliberate choices recorded so they are visible, not incidental:**

- **First run after a restart pages a red source** (`_last_red_sources is None`
  means "no baseline", not "nothing is new"). On restart the operator *should*
  be told a table is red. Pinned by `test_first_run_after_restart_pages_a_red_source`.
- **Recovery emits no alert.** Criterion 3 requires an all-healthy fixture to
  emit nothing; a resolution page would make that criterion ambiguous. Recorded
  as a trade-off, not an oversight.

---

## 2. Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q
.............                                                            [100%]
13 passed in 0.04s
```

Exit code 0.

---

## 3. Mutation matrix -- CONTROL re-derived in the same run

Harness: `scratchpad/mutate_82_10.py`. It applies one mutation at a time,
asserts the anchor is unique, runs `pytest -rf`, restores the file, and finally
re-runs to prove the tree is byte-identical to control. **CONTROL measured in
this same run: `rc=0 passed=13`. POST-RESTORE: `rc=0 passed=13`.**

Two mutant FORMS reported separately (auto-memory
`feedback_two_mutant_forms_separate_artifact_from_kill`).

| Mutant | Form | Mutation | Result | Tests ACTUALLY killed (from `-rf`) |
|---|---|---|---|---|
| M1 | code | Drop the state-transition gate (fire on every red, every tick) | **KILLED** | `test_a_newly_red_source_pages_even_when_another_is_already_red`, `test_steady_state_red_does_not_re_page_and_the_inner_emitter_is_suppressed` |
| M2 | code | Cron stops suppressing the inner level-triggered emitter | **KILLED** | `test_a_newly_red_source_pages_even_when_another_is_already_red`, `test_first_run_after_restart_pages_a_red_source`, `test_steady_state_red_does_not_re_page_and_the_inner_emitter_is_suppressed` |
| M3 | code | `compute_freshness` ignores `emit_alarm` (always fires inner alarm) | **KILLED** | same three as M2 |
| M4 | code | `main.py` never registers the cron (module exists but is dead) | **KILLED** (see §3.1) | `test_main_wires_the_cron_at_startup` |
| M5 | code | `replace_existing` defaults False (restart duplicates the job) | **KILLED** | `test_registers_exactly_one_job_on_a_stub_scheduler` |
| M6 | code | Severity downgraded P1 -> P2 (silently dropped on empty webhook) | **KILLED** | `test_breaching_source_pages_through_the_real_notification_path` |
| M7 | code | `_red_sources` never sees a red band (the alarm goes blind) | **KILLED** | 4 tests incl. `test_breaching_source_pages_through_the_real_notification_path` |
| M8 | code | Fail-open path reports `ok=True` on error | **KILLED** | `test_run_freshness_check_is_fail_open` |
| M9 | code | Registered callable is a no-op lambda, not `run_freshness_check` | **KILLED** | `test_registers_exactly_one_job_on_a_stub_scheduler`, `test_the_job_the_scheduler_received_actually_reaches_compute_freshness` |
| F1 | fixture | Red fixture no longer breaches (age just UNDER 2x SLA) | **KILLED** | 3 tests -- proves the precondition assertions are load-bearing |

### 3.1 M4 SURVIVED the first version of this guard -- and that is the finding

The original `test_main_wires_the_cron_at_startup` was a substring scan:
`assert "register_freshness_cron" in src`. M4 deleted only the **call**, leaving
the `from backend.services.freshness_cron import register_freshness_cron`
line -- so the substring was still present and **the guard passed against a
main.py that would never run the job**. That is precisely the guard-that-cannot-
fail anti-pattern (auto-memory `feedback_mutation_test_guards_and_fixtures`),
and I only caught it because the mutation ran.

**Fixed, not disclosed:** the guard now parses `main.py` with `ast` and requires
an `ast.Call` node whose func is `register_freshness_cron`. Re-running the
matrix, M4 is KILLED by exactly that test. The other source-level guard in this
file (`test_evaluator_has_no_web_dependency`) is a deliberate *absence* check
where a substring scan is the correct instrument, and it is supplementary --
criterion 1's real guard executes the scheduled callable.

---

## 4. LIVE CHECK -- measured against the running service, not asserted

The backend does not auto-reload, so the code above was inert until restart.

```
$ launchctl list | grep " com.pyfinagent.backend$"     # BEFORE
62664   -15  com.pyfinagent.backend

$ launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend
$ until curl -sf -o /dev/null http://localhost:8000/api/health; do sleep 2; done
backend up

$ launchctl list | grep " com.pyfinagent.backend$"     # AFTER (new pid)
60478   -15  com.pyfinagent.backend
```

Live scheduler introspection (`GET /api/jobs/all`), APScheduler jobs only:

```
main_apscheduler | paper_trading_daily        | next=2026-08-05T14:00:00-04:00
main_apscheduler | freshness_evaluator        | next=2026-08-05T23:38:38.142553+02:00
main_apscheduler | macro_ingest_daily         | next=2026-08-06T08:10:00-04:00
main_apscheduler | harness_self_audit_weekly  | next=2026-08-09T03:00:00-04:00
main_apscheduler | ticket_queue_process_batch | next=2026-08-05T17:39:03.154236+02:00
```

`freshness_evaluator` is **registered on the live main scheduler** with a
concrete `next_run` 6h after the 17:38 restart, matching
`DEFAULT_INTERVAL_HOURS = 6`. This is the evidence for "the alarm now has a
trigger": before this step there was no such row.

**What this live check does NOT prove, stated plainly:** it does not prove a
page was delivered to Slack. No source is currently red (`historical_macro` was
repaired by 82.0), so there is nothing to page about, and I did not manufacture
a red table on the live system to force one. Delivery through
`raise_cron_alert_sync` is covered by the fixture tests and by the pre-existing
phase-62.7 bot-token live proof, not by this capture.

---

## 5. Regression check

```
$ python -c "import ast; ast.parse(...)  # main.py, cycle_health.py, freshness_cron.py"
syntax OK

$ python tests/verify_phase_25_A7.py
11/11 claims PASS, 0 FAIL          # incl. claims 8 + 9, the pre-existing
                                    # compute_freshness alarm behaviour

$ python -m pytest backend/tests/test_dod4_tier1_coverage_investment.py \
      backend/tests/test_cycle_heartbeat_alarm.py \
      backend/tests/test_phase_82_0_macro_ingestion.py -q
1 failed, 94 passed
FAILED backend/tests/test_dod4_tier1_coverage_investment.py::test_paper_trader_execute_buy_average_up_recomputes_avg_entry
```

**The one failure is PRE-EXISTING and unrelated -- proven, not assumed.** I
built a detached worktree at HEAD (`9b722a97`), copied `backend/.env` in so the
run was valid, and ran the single test there:

```
$ git worktree add <scratch>/head-wt HEAD --detach
HEAD is now at 9b722a97 chore: auto-changelog hook entry for c47be8e9
$ cd <scratch>/head-wt && python -m pytest ...::test_paper_trader_execute_buy_average_up_recomputes_avg_entry -q
backend/tests/test_dod4_tier1_coverage_investment.py:627: AssertionError
ERROR backend.services.paper_trader: kill_switch: REFUSING BUY AAPL ($600.00) --
      the kill switch is PAUSED (pause_reason='manual')
1 failed
```

Identical assertion, identical line, identical cause at HEAD without any of my
changes. Worktree removed afterwards. **Root cause: that unit test reads the
LIVE kill-switch state, so it fails whenever the operator has the book paused
-- a test coupled to live operator state. Queued as its own masterplan step
(82.36), not fixed here.**

---

## 6. Discovered defects -- QUEUED as their own steps, not prose-only

| New step | What |
|---|---|
| **82.35** (P2) | `settings.paper_trading_enabled` (`backend/main.py:307`) gates the entire scheduler block, so the freshness monitor is disabled by the same switch that disables one of the things it monitors. Named in contract section 4.6 as out of scope. |
| **82.36** (P2) | `test_paper_trader_execute_buy_average_up_recomputes_avg_entry` reads the live kill-switch state and fails whenever the book is paused. A unit test must not depend on live operator state. |

---

## 7. Scope honesty

**Changed:** one new service module, one keyword-only parameter with a
behaviour-preserving default, one registration block, one new test file, plus
two new pending masterplan steps.

**NOT changed, deliberately:** freshness mathematics, `WARN_RATIO`,
`CRITICAL_RATIO`, `_TABLE_MAX_AGE_SEC`, the `compute_freshness` return shape,
any HTTP handler, `AlertDeduper`, `_fire_freshness_alarm`, any live position,
any credential, any operator-gated flag. Paper trading left running.

**`misfire_grace_time` / `coalesce` deliberately NOT added.** The step
description warns about "a job that catches up 128 missed runs". Measured: that
risk does not exist in this configuration -- `backend/main.py:310` constructs a
bare `AsyncIOScheduler()` with the default **in-memory** jobstore, so no jobs
persist across a restart and there is nothing to replay. Recording why rather
than citing a risk that is not real.

---

# CYCLE 2 -- response to the cycle-1 Q/A CONDITIONAL

**Cycle-1 verdict:** CONDITIONAL (verbatim in
`handoff/current/evaluator_critique_82.10.md`, raw return archived at
`handoff/current/qa_returns/82.10_cycle1.output.json`).

**The blocker, in the Q/A's words:** `test_registers_exactly_one_job_on_a_stub_scheduler`
asserted `job['id']`, `job['replace_existing']` and `job['func']` "but never
`job['trigger']` or `job['hours']`, so NOTHING in the suite pins the job's
RECURRENCE." Its own mutants `trigger='interval' -> 'date'` (fires once at
startup, then gone) and `DEFAULT_INTERVAL_HOURS 6 -> 99999` (~11 years) each
passed all 13 tests. Either one ships an evaluator that is scheduled in name
only -- re-creating the exact browser-blind blind spot this step exists to
close -- with a fully green suite. **The finding is correct and I reproduced
both survivals.**

## 8. Sweeping the CLASS, not the named instance

The class is: *a registration test that pins a job's IDENTITY but not the
kwargs that decide whether it ever runs again.*

**First I measured whether the sibling crons share the gap** -- if they did,
this would be a repo-wide defect needing its own step:

```
$ grep -n "trigger\|hours\|day_of_week" backend/tests/test_phase_82_0_macro_ingestion.py
144:    assert job["trigger"] == "cron"
$ grep -n "trigger\|hours\|day_of_week" backend/tests/test_phase_71_6_self_audit_cron.py
70:    assert call["trigger"] == "cron"
73:    assert call["day_of_week"] == "sun"
```

**They do not.** Both sibling cron tests already assert recurrence. So there is
no repo-wide gap to queue -- the honest finding is narrower and worse for me:
**my test had fallen BELOW an established repo convention**, and the Q/A caught
it. Recorded as such rather than inflated into a systemic issue.

**Four guards added** (not the one-line fix the Q/A offered -- that would patch
the instance and leave the class open):

| Guard | What it pins |
|---|---|
| `test_registers_exactly_one_job_on_a_stub_scheduler` (extended) | `trigger == "interval"` and `hours == DEFAULT_INTERVAL_HOURS` |
| `test_registered_cadence_is_tight_enough_to_catch_the_tightest_sla` | **Semantic**, not a magic-number pin: derives the bound live from `min(cycle_health._TABLE_MAX_AGE_SEC.values()) / 2` (the dbt >=2x-tightest-SLA rule) and asserts `0 < hours <= that`. Kills an absurd interval for the RIGHT reason and stays correct if 6 is later tuned to 8. |
| `test_the_hours_parameter_is_actually_forwarded` | The `hours=` knob is not decorative |
| `test_registration_kwargs_are_pinned_exactly` | The **full** kwarg surface. Asserting a hand-picked subset is what let two behaviour-changing kwargs go unobserved; this fails on any added/dropped/renamed kwarg. |

## 9. Verification command re-run (verbatim, UNPIPED per the Q/A's note)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q; echo "BARE_EXIT=$?"
................                                                         [100%]
16 passed in 0.03s
BARE_EXIT=0
```

## 10. Mutation matrix RE-RUN -- now including the Q/A's two survivors

CONTROL re-derived in the same run: `rc=0 passed=16`. POST-RESTORE: `rc=0
passed=16`. **13 of 13 mutants killed; zero survivors.**

| Mutant | Form | Mutation | Result | Tests ACTUALLY killed |
|---|---|---|---|---|
| M1-M9, F1 | code / fixture | (unchanged from section 3) | **all KILLED** | as tabulated above |
| **Q1** | code | *Q/A survivor:* `trigger` `"interval" -> "date"` (fires once, never recurs) | **KILLED** | `test_registers_exactly_one_job_on_a_stub_scheduler` |
| **Q2** | code | *Q/A survivor:* `DEFAULT_INTERVAL_HOURS` `6 -> 99999` (~11 years) | **KILLED** | `test_registered_cadence_is_tight_enough_to_catch_the_tightest_sla` |
| **Q3** | code | `hours` accepted but ignored (decorative knob) | **KILLED** | `test_the_hours_parameter_is_actually_forwarded` |

Q2 dies by the **semantic** cadence guard rather than by a restated constant --
that is the difference between pinning the number and pinning the requirement.

## 11. What did NOT change in cycle 2

No production code was touched. The fix is entirely additive test coverage
(`backend/tests/test_phase_82_10_freshness_paging.py` only, plus one import
line). `backend/services/freshness_cron.py`, `backend/services/cycle_health.py`
and `backend/main.py` are byte-identical to cycle 1, so the live capture in
section 4 remains valid evidence for the shipped code and did not need
re-taking. No masterplan step was added or altered in cycle 2.

---

# CYCLE 3 -- response to the cycle-2 Q/A CONDITIONAL

Cycle-2 verdict transcribed verbatim in `handoff/current/evaluator_critique_82.10.md`;
raw return at `handoff/current/qa_returns/82.10_cycle2.output.json`. TWO violated
criteria, both accepted.

## 12. RETRACTION -- my cycle-2 section-8 claim was FALSE

Section 8 above concluded: *"They do not [share the gap] ... there is no repo-wide gap
to queue."* **That conclusion is wrong and I am retracting it here rather than editing
section 8, so the error stays visible in the record.**

The cycle-2 Q/A caught it as `scope_honesty`: I compared against an **author-chosen
population of 2 files**. The membership rule was never written down -- the exact failure
class in auto-memory `feedback_measure_dont_assert_claims` ("a claim about a SET whose
membership rule was never written down").

**The derived population, measured this cycle -- and it is WORSE than the Q/A said.**
The Q/A grepped only `backend/tests/` and found 4. The repo has a second test tree:

```
$ grep -rln "add_job" backend/tests/ tests/ | sort
backend/tests/test_phase_70_5_reschedule.py
backend/tests/test_phase_71_6_self_audit_cron.py
backend/tests/test_phase_82_0_macro_ingestion.py
backend/tests/test_phase_82_10_freshness_paging.py
tests/scheduler/test_meta_cron.py
tests/services/test_phase9_registration.py
tests/slack_bot/test_phase9_production_wiring.py
tests/slack_bot/test_scheduler_phase9.py
tests/verify_phase_23_3_1.py

$ for f in $(...); do echo "$f -> trigger-assertions=$(grep -c ...)"; done
backend/tests/test_phase_70_5_reschedule.py      -> 0
backend/tests/test_phase_71_6_self_audit_cron.py -> 1
backend/tests/test_phase_82_0_macro_ingestion.py -> 1
backend/tests/test_phase_82_10_freshness_paging.py -> 2
tests/scheduler/test_meta_cron.py                -> 2
tests/services/test_phase9_registration.py       -> 0
tests/slack_bot/test_phase9_production_wiring.py -> 1
tests/slack_bot/test_scheduler_phase9.py         -> 1
tests/verify_phase_23_3_1.py                     -> 0
```

**Population 9, not 2. THREE files assert no trigger at all**, each verified by reading
the assertion body rather than trusting the count:

- `backend/tests/test_phase_70_5_reschedule.py:43-45` -- asserts `kw["hour"]`,
  `kw["replace_existing"]`, `kw["id"]`. This guards the **paper-trading daily job**, so a
  trigger regression there stops the book trading.
- `tests/services/test_phase9_registration.py:50` -- asserts `misfire_grace_time` +
  `coalesce` only.
- `tests/verify_phase_23_3_1.py:29-33` -- a **source scan** for id/name/replace_existing
  strings, structurally unable to observe an argument change.

**So the repo-wide gap is real and I dismissed it on a hand-picked scope. Queued as
step 82.37**, which records the derivation mistake explicitly so its executor does not
repeat it.

## 13. The call-site seam -- the same class, one seam over

Cycle-2 blocker: `test_main_wires_the_cron_at_startup` asserted only that an `ast.Call`
node EXISTS. It never inspected the call's arguments, and every other test in the file
calls `register_freshness_cron(stub)` with **defaults** -- so nothing constrained what
`main.py` actually passes. The Q/A proved `hours=99999`, `hours=0` and
`replace_existing=False` all ship green.

Fixed by extending that guard to read the call's keywords from the AST and validate
them: `hours` must be a literal in the same semantically-derived band
`(0, min(_TABLE_MAX_AGE_SEC)/2]`, `replace_existing` must be literal `True`, and **any
other keyword fails** as an unreviewed override of an audited default.

## 14. Mutation matrix RE-RUN -- 16 mutants, ZERO survivors

CONTROL re-derived in the same run: `rc=0 passed=16`. POST-RESTORE: `rc=0 passed=16`.

| Mutant | Form | Mutation | Result | Test ACTUALLY killed |
|---|---|---|---|---|
| M1-M9, Q1-Q3, F1 | code / fixture | (sections 3 + 10) | **all KILLED** | as tabulated |
| **C1** | code | *Q/A cycle-2 survivor:* `main.py` passes `hours=99999` | **KILLED** | `test_main_wires_the_cron_at_startup` |
| **C2** | code | *Q/A cycle-2 survivor:* `main.py` passes `hours=0` | **KILLED** | `test_main_wires_the_cron_at_startup` |
| **C3** | code | *Q/A cycle-2 survivor:* `main.py` passes `replace_existing=False` | **KILLED** | `test_main_wires_the_cron_at_startup` |

## 15. Verification command (verbatim, unpiped)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q; echo "BARE_EXIT=$?"
................                                                         [100%]
16 passed in 0.04s
BARE_EXIT=0
```

## 16. What changed in cycle 3

`backend/tests/test_phase_82_10_freshness_paging.py` (the call-site guard) and
`.claude/masterplan.json` (new pending step 82.37). **No production code touched in
cycle 2 or cycle 3** -- `backend/services/freshness_cron.py`,
`backend/services/cycle_health.py` and `backend/main.py` are byte-identical to cycle 1,
so the section-4 live capture remains valid evidence for the shipped code.

## 17. Honest statement of what is still not covered

The suite constrains the registration call and the evaluator's behaviour. It does **not**
prove a Slack message was delivered to the operator -- no source is currently red, and I
did not manufacture a red table on the live system to force a page. Delivery is covered
by the fixture tests and the pre-existing phase-62.7 bot-token live proof, not by any
capture in this step.

---

# CYCLE 4 -- response to the cycle-3 FAIL

Cycle-3 verdict (FAIL) transcribed verbatim in `evaluator_critique_82.10.md`; raw return
at `handoff/current/qa_returns/82.10_cycle3.output.json`. The FAIL was produced by the
**3rd-CONDITIONAL auto-FAIL rule** -- the Q/A stated its two findings were WARN-level and
would otherwise have been CONDITIONAL. Both findings are accepted; neither is argued
with. All three immutable criteria were judged MET in every cycle by three independent
Q/A instances.

## 18. SECOND RETRACTION -- my section-12 census was also wrong

Section 12 retracted section 8's false claim. **Section 12's own replacement number was
also wrong**, and I am retracting it here rather than editing it, for the same reason.

The Q/A recovered my elided command: the column I labelled `trigger-assertions=` came
from counting the **literal string** `"trigger"`. That matches:

- a stub scheduler's own recording dict -- `tests/slack_bot/test_scheduler_phase9.py:21`
  and `tests/slack_bot/test_phase9_production_wiring.py:37` both build
  `{... "trigger": trigger ...}` inside their fake `add_job`;
- prose in a failure message --
  `test_phase9_production_wiring.py:214 assert posts == [], "empty drifts should not trigger a Slack post"`.

**A grep over source text cannot distinguish an ASSERTION from FIXTURE PLUMBING.** I used
a text count to define a set and then "verified by reading" only the files the count had
already flagged -- so the reading discipline never touched the membership rule. That is
the same defect as section 8 (`feedback_measure_dont_assert_claims`), committed *inside
the retraction of section 8*.

### The properly derived answer -- AST census, and BOTH numbers reported

Instrument: parse each file, walk `ast.Assert`, examine **`.test` only, never `.msg`**,
strip string literals from the expression, then re-admit subscript keys so
`kw["trigger"]` still counts. Population derived, then asserted non-empty before
reporting.

```
$ grep -rln --include=*.py "add_job" backend/tests/ tests/   # 9 files, none in __pycache__

Files NOT asserting the TRIGGER: 5 of 9
  - backend/tests/test_phase_70_5_reschedule.py
  - tests/services/test_phase9_registration.py
  - tests/slack_bot/test_phase9_production_wiring.py
  - tests/slack_bot/test_scheduler_phase9.py
  - tests/verify_phase_23_3_1.py

Files asserting NOTHING about the schedule at all: 4 of 9   (the same five minus 70_5)
```

**FIVE, not three.** The Q/A's figure reproduces exactly. My AST run also yields FOUR on
a *different* question -- `test_phase_70_5_reschedule.py:44` asserts `kw["hour"] == 18`
and `kw["replace_existing"]` but never the trigger, so `trigger="date"` with `hour=18`
would still pass there. Both numbers are now reported, because quoting either alone
misstates the class. That file guards the **paper-trading daily job**, so a trigger
regression there stops the book trading.

**Step 82.37's description has been corrected in `.claude/masterplan.json`** -- the false
"THREE" is removed, both derived numbers are recorded, and the description now explains
*why* the grep-count instrument was wrong so its executor does not repeat it.

## 19. The call-site guard -- three more holes, all closed

Cycle-3 defeated the keywords-only check three ways:

| Mutant | Mechanism | Measured effect |
|---|---|---|
| `**{"hours": 99999}` | `kw.arg is None` for a `**` expansion, so the band check never ran | job registered at 11.4 years |
| positional `99999` | `hours` is keyword-only -> `TypeError` -> `main.py`'s fail-open `except` | **ZERO jobs registered**, silently |
| `register_freshness_cron(None)` | the helper's own fail-open | **ZERO jobs**, no raise |

The last two are the sharper ones: a fail-open registration that registers *nothing* is
precisely the browser-blind blind spot this step exists to close. So the call **shape** is
now pinned, not merely its keyword values: no `**` expansion, exactly one positional
argument, and that argument must be the `scheduler` Name.

## 20. Mutation matrix -- 19 mutants, ZERO survivors

CONTROL re-derived in the same run: `rc=0 passed=16`. POST-RESTORE: `rc=0 passed=16`.

| Mutant | Form | Mutation | Result | Test ACTUALLY killed |
|---|---|---|---|---|
| M1-M9, Q1-Q3, C1-C3, F1 | code / fixture | (sections 3, 10, 14) | **all KILLED** | as tabulated |
| **D1** | code | *Q/A cycle-3 survivor:* `**{"hours": 99999}` | **KILLED** | `test_main_wires_the_cron_at_startup` |
| **D2** | code | *Q/A cycle-3 survivor:* positional `99999` -> fail-open -> zero jobs | **KILLED** | `test_main_wires_the_cron_at_startup` |
| **D3** | code | *Q/A cycle-3 survivor:* `register_freshness_cron(None)` -> zero jobs | **KILLED** | `test_main_wires_the_cron_at_startup` |

## 21. Verification command (verbatim, unpiped)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q; echo "BARE_EXIT=$?"
................                                                         [100%]
16 passed in 0.04s
BARE_EXIT=0
```

## 22. What changed in cycle 4, and the standing honest caveats

Changed: `backend/tests/test_phase_82_10_freshness_paging.py` (call-shape guard) and
`.claude/masterplan.json` (82.37's description corrected). **No production code has been
touched since cycle 1** -- `freshness_cron.py`, `cycle_health.py` and `main.py` are
byte-identical to the tree that produced the section-4 live capture, so that capture
remains valid evidence.

Still not covered, stated plainly and unchanged from section 17: nothing here proves a
Slack message reached the operator. No source is currently red, and I did not manufacture
a red table on the live system to force a page.

**Track record on this step, recorded rather than glossed:** three Q/A cycles, three
findings against my guards and TWO false claims of my own about derived sets -- the
second one made inside the retraction of the first. The production code was judged
correct every cycle; every failure was in how I measured and what I claimed.
