# Experiment results — phase-85.4 (P0 ENGINE HEALTH)

**Step id:** 85.4 · **Cycle:** 183 — 2026-08-09
**Contract:** `handoff/current/contract_85.4.md` · **Research gate:** `handoff/current/research_brief_85.4.md` (`gate_passed: true`, 9 sources read in full, 25 URLs)

Built against the CONTRACT, not the step's audit basis — the gate refuted two of
the step's own premises (terminal rows *are* written; the failure *was* alerted,
just buried). Nothing below re-litigates that; it re-derives from scratch.

---

## 1. What was built

| File | Change |
|---|---|
| `scripts/diagnostics/measure_analysis_phase.py` | **NEW.** Read-only measurement harness: per-cycle analysis-phase duration, per-ticker wall-clock, effective parallelism, and the projected cycle cost had every dispatched ticker finished. This is the C1 evidence generator and it is re-runnable. |
| `backend/services/autonomous_loop.py` | (a) `dispatch_analyses()` extracted as a module-level seam and wired at the production call site; (b) the kill-switch halt path now sets `status="halted_kill_switch"` instead of leaking the `:362` placeholder `"running"`; (c) the non-completion P1 now names the phase it died in, in the **title** and in a new `died_in_phase` detail. |
| `backend/services/cycle_health.py` | `cycle_heartbeat_alarm()` gains a SECOND, independent clock — `last_success_at` / `success_age_sec` / `success_stale` / `should_alarm_success` / `last_terminal_status`. New `fire_cycle_completed_stale_alarm()` dispatcher. Existing keys and callers untouched. |
| `backend/slack_bot/scheduler.py` | Cycle-health leg extracted to a **callable** `check_cycle_health_alarms()`; the new completed-age alarm wired in behind its own state-transition gate. |
| `backend/config/settings.py` | `paper_merged_analysis_dispatch_enabled: bool = False` — **DARK**, not promoted (ask #25). |
| `backend/tests/test_phase_85_4_cycle_loudness.py` | **NEW**, 4 tests. Fault-injected cycles (timeout / crash / kill-switch halt). |
| `backend/tests/test_phase_85_4_completed_age_alarm.py` | **NEW**, 12 tests. The C4 clock + the watchdog that pages on it. |
| `backend/tests/test_phase_85_4_dispatch_barrier.py` | **NEW**, 16 tests. The C2 root-cause reproduction + (b)/(c) elimination + equivalence. |
| `scripts/qa/mutation_matrix_85_4.py` | **NEW.** 9-mutation harness proving every guard above can actually fail. |

---

## 2. Criterion 1 — the analysis phase, MEASURED

Command (re-runnable, read-only):

```
$ source .venv/bin/activate && python scripts/diagnostics/measure_analysis_phase.py \
      --log backend.log --json handoff/current/analysis_phase_measurement_85.4.json
```

Verbatim output (162,395 log lines parsed, 3 cycles reached the analysis phase):

```
==============================================================================
CYCLE  started=2026-08-05 20:00:01.868000  terminal=None  wall=Nones
  screening        : 570.939s
  analysis phase   : 5098.931s  (end reason: reached_mark_to_market)
  tickers          : planned=6 dispatched=6 finished=6 unfinished=[]
  concurrency cap  : 3
  per-ticker wall  : {'CRWD': 175.8, 'DDOG': 196.7, 'DELL': 2361.5, 'HPE': 2141.8, 'NTAP': 2595.6, 'PANW': 2325.4}
  per-ticker mean  : 1632.8s  median=2233.6s
  serial ticker-s  : 9796.6s   effective parallelism=1.92
  PROJECTED cycle   (screening + analysis)              : 5670s  vs budget 7200s
  VERDICT          : within budget (delta -1530s)
  cc_rail calls    : started=124 timed_out=29 rate=0.2339 subprocess_timeout_s=150
==============================================================================
CYCLE  started=2026-08-06 20:00:01.472000  terminal=timeout  wall=7200.117s
  screening        : 428.778s
  analysis phase   : 6771.339s  (end reason: cycle_timeout)
  tickers          : planned=6 dispatched=6 finished=5 unfinished=['NTAP']
  concurrency cap  : 3
  per-ticker wall  : {'CRWD': 2197.7, 'DELL': 2260.3, 'FTNT': 2509.9, 'HPE': 2226.4, 'PANW': 2358.7}
  per-ticker mean  : 2310.6s  median=2260.3s
  serial ticker-s  : 11553.1s   effective parallelism=1.71
  PROJECTED cycle   (screening + analysis)              : 8554s  vs budget 7200s
  VERDICT          : OVER BUDGET (delta +1354s)
  cc_rail calls    : started=175 timed_out=26 rate=0.1486 subprocess_timeout_s=150
==============================================================================
CYCLE  started=2026-08-07 20:00:01.752000  terminal=timeout  wall=7200.077s
  screening        : 554.894s
  analysis phase   : 6645.183s  (end reason: cycle_timeout)
  tickers          : planned=6 dispatched=6 finished=5 unfinished=['NTAP']
  concurrency cap  : 3
  per-ticker wall  : {'CRWD': 2176.4, 'DELL': 2468.0, 'HPE': 2313.3, 'HUM': 2048.8, 'PANW': 2593.0}
  per-ticker mean  : 2319.9s  median=2313.3s
  serial ticker-s  : 11599.4s   effective parallelism=1.75
  PROJECTED cycle   (screening + analysis)              : 8529s  vs budget 7200s
  VERDICT          : OVER BUDGET (delta +1329s)
  cc_rail calls    : started=177 timed_out=32 rate=0.1808 subprocess_timeout_s=150
```

### The criterion's question, answered: **7200s is too short for the current ticker count.**

At 6 tickers, semaphore 3, and a measured mean per-ticker wall-clock of
**2310–2320s**, a cycle in which every dispatched ticker finishes projects to
**8554s (08-06)** and **8529s (08-07)** against a **7200s** budget — over by
**1354s and 1329s**. Both timed out with exactly one ticker (NTAP) never
analysed. The phase does not hang; it is arithmetically too expensive.

### The one completing cycle completed for a reason that is not good news

08-05 is the only cycle in the window that reached mark-to-market, and its
projection (5670s) sits comfortably inside budget — but only because **two of
its six analyses failed fast**: CRWD at 175.8s and DDOG at 196.7s, against a
~2300s cost for the four healthy ones. Both are recorded in `backend.log` as
`Claude analysis for <T>: PARSE-FAILED -> degraded HOLD placeholder`.

Counterfactual at that cycle's own healthy mean (2356s over its four healthy
tickers) and its own measured parallelism (1.92):
`6 x 2356 / 1.92 = 7363s` analysis + `571s` screening = **7934s — also over
budget.** So the completion rate in this window is not 1-in-3; it is 0-in-3
on a healthy book, and the single success was bought by two failures.

---

## 3. Criterion 2 — root cause to file:line, with a reproduction

**Verdict: (a) legitimate slowness, amplified by a structural barrier.**
(b) and (c) are ruled out, each by a test, not by an opinion.

### The barrier — `backend/services/autonomous_loop.py:1157` and `:1164` (pre-fix)

Two `asyncio.gather` calls, awaited in sequence, sharing ONE
`asyncio.Semaphore(3)`. The second gather cannot start until the first has
fully drained, so a slot freed by an early finisher **idles** rather than
picking up a re-eval.

Observed verbatim in `backend.log` for the 2026-08-07 cycle:

```
20:09:16  Orchestrator pre-dispatch ticker=DELL     (3/3 slots busy)
20:09:17  Orchestrator pre-dispatch ticker=PANW
20:09:19  Orchestrator pre-dispatch ticker=CRWD
20:45:35  Lite analysis persisted for CRWD -> pre-dispatch HPE
20:50:24  Lite analysis persisted for DELL -> pre-dispatch HUM
20:52:30  Lite analysis persisted for PANW -> *** no dispatch: NTAP is in batch 2 ***
21:24:33  Lite analysis persisted for HUM  -> pre-dispatch NTAP
22:00:01  Paper trading cycle TIMED OUT after 7200s
```

The freed slot idled **1923s (32 min)**, and NTAP started **4517s** into the
analysis phase instead of ~2593s. The cycle overran by **1329s** — i.e. the
wasted window alone (1923s) is **larger than the overrun**.

### Reproduction

`backend/tests/test_phase_85_4_dispatch_barrier.py` drives the **production**
`dispatch_analyses` seam on a real event loop with the same shape (5 new + 1
re-eval, semaphore 3) and a quantised clock:

- legacy makespan **3 units**, merged makespan **2 units** (asserted exactly)
- `test_a_legacy_two_gather_path_idles_a_slot_and_starts_the_reeval_late`
- `test_a_merged_path_never_exceeds_the_concurrency_cap` — the saving is NOT
  extra concurrency (which would re-open the 429 incident the cap exists for);
  peak concurrency is asserted `== 3`.

### (c) — unhandled per-ticker failure stalling the gather: **RULED OUT**

`test_c_a_failing_ticker_does_not_stall_the_gather` (both paths): a raising
ticker is captured in place by `return_exceptions=True`, every sibling still
retires, and positional alignment survives so the caller's
`[r for r in results if isinstance(r, dict)]` filter still works.

### (b) — hang/deadlock: **RULED OUT for the dispatch path**

`test_b_no_deadlock_all_dispatched_tickers_retire`: dispatched set == finished
set on both paths. Corroborated in production: in all three measured cycles
every ticker that did not finish (NTAP, twice) was still issuing `cc_rail`
calls when the budget expired — 175 and 177 rail calls started in-window.

**Residual risk, stated honestly:** `_run_single_analysis` still has no inner
timeout, so a genuine unbounded hang inside one ticker remains *possible*. It
is not what happened on 08-06 or 08-07. Fixing it is out of 85.4's criteria and
is filed as a queued step rather than smuggled in here.

---

## 4. Criterion 3 — a non-completing cycle is LOUD (fault-injected, not inspected)

### The defect fixed

The kill-switch halt returned at `autonomous_loop.py:~1327` **without setting a
status**, so the `:362` initializer's placeholder `"running"` became the
terminal value. Consequences, all three real:

1. `cycle_history.jsonl` got a *terminal* row claiming the cycle was RUNNING.
2. The P1 was titled `Autonomous trading cycle running`.
3. The completed-age clock (§5) could not distinguish a halt from a success.

### Proof — `backend/tests/test_phase_85_4_cycle_loudness.py`

These drive the **real `run_daily_cycle`** with `cycle_history.jsonl`, the
heartbeat, the cycle lockfile and `raise_cron_alert_sync` all redirected into
`tmp_path`/recorders. Three fault modes:

| Test | Injected fault | Asserts |
|---|---|---|
| `test_c3_timeout_midanalysis_...` | `_run_single_analysis` sleeps 300s against a 10s budget | status `timeout`; exactly ONE terminal row with `completed_at`; P1 title contains `analyzing`; `details.died_in_phase == "analyzing"` |
| `test_c3_crash_midanalysis_...` | `dispatch_analyses` raises | status `error`; one terminal row; phase named in title and details |
| `test_c3_killswitch_halt_...` | injected paused kill-switch state | status `halted_kill_switch`, **explicitly asserted `!= "running"`**; `died_in_phase == "kill_switch_halted"`; `"running"` absent from the title |
| `test_c3_halted_status_is_not_counted_as_a_completion` | same | the C4 clock reads `last_success_at is None`, `success_stale is True` |

The timeout test asserts its **own precondition** (`entered_analysis` non-empty)
so it cannot silently prove nothing by dying in screening instead.

---

## 5. Criterion 4 — last-completed-cycle age, pageable by the existing watchdog

### The gap, precisely

`cycle_heartbeat_alarm` measured the age of the last terminal row of **any**
status, and `record_cycle_end` stamps `completed_at` on `timeout` rows too. A
cycle that timed out every weekday therefore reset the alarm's clock every
weekday. Between **2026-07-31 and 2026-08-08** the heartbeat read ~24h while the
book had not completed a cycle in **7 days**, and the only way to find out was
to hand-read the JSONL — which is verbatim what the criterion forbids.

### The fix

Second clock in the **same verdict dict, same single file read** — purely
additive, existing keys and callers unchanged:

```
age_sec         -- "when did a cycle last END?"   (any terminal status)
success_age_sec -- "when did a cycle last WORK?"  (status == completed)
```

Threshold `_CYCLE_COMPLETED_STALE_SEC = 345_600.0` (96h), weekday-gated. Chosen
so the largest legitimate gap (Fri→Mon = 72h) stays quiet and one extra failed
weekday pages. Both boundaries are asserted:
`test_friday_to_monday_weekend_gap_does_not_page` and
`test_one_extra_failed_weekday_past_the_weekend_does_page`.

`fire_cycle_completed_stale_alarm` raises a **P1** titled *"Autonomous cycle has
not COMPLETED -- book is not trading"*, carrying `last_completed_cycle_at`,
`age_since_last_completion`, and `last_terminal_status`.

### Wired, and proven wired behaviourally

The watchdog's cycle-health block was extracted to a **callable**
`scheduler.check_cycle_health_alarms()`. The tests drive that function, not a
grep of it — deliberately, because phase-36.12 already learned that a source
scan cannot tell a live branch from `if False and ...`. Mutation **M8** neuters
exactly that branch, and the behavioural tests catch it (an earlier source-scan
version of this test did **not** — see §7).

Verbatim, the production regression replayed as a test
(`test_daily_timeouts_keep_the_old_clock_green_and_the_new_clock_red`):
old clock `stale=False`, new clock `success_stale=True`,
`success_age_sec ≈ 7 days`.

---

## 6. Criterion 5 — no order/sizing/risk change; behavioural changes are DARK

- **No order, sizing or risk logic was touched.** The diff is confined to
  status strings, alert payloads, a health-signal computation, a dispatch-order
  seam, one default-`False` flag, tests, and two scripts.
- `paper_merged_analysis_dispatch_enabled` defaults **False** and is **not
  promoted**. Asserted by `test_production_default_is_the_legacy_path`.
- The two config recommendations from the research gate — `timeout_s` 150→210
  (`claude_code_client.py:593`) and `paper_cycle_max_seconds` 7200→10800 — were
  **NOT applied**. They are behavioural config changes; both are filed as
  operator asks **#23** and **#24** in `handoff/current/operator_ask_2026-08-07.md`
  (ask **#25** covers promoting the dark dispatch flag).
- No `backend/.env` writes. `historical_macro` untouched. No service restarted,
  no cycle triggered, no kill-switch state changed.

### One behaviour change that is NOT flag-gated, disclosed explicitly

The kill-switch halt now writes `status="halted_kill_switch"` instead of
`"running"`, and the failure P1's title gained the phase name. These are
**observability fidelity**, which is the step's subject matter, so gating them
dark would gate the fix itself. They change: (1) the status string in
`cycle_history.jsonl`, (2) one alert title, (3) the alert `error_type` /
dedup key for the halt path (`cycle_running` → `cycle_halted_kill_switch`).
Consumers checked — `paper_trading.py:472` and `scheduler.py:402` read rows
without switching on the status string, and no frontend type pins it.

---

## 7. Mutation matrix — every guard proven able to fail

```
$ source .venv/bin/activate && python scripts/qa/mutation_matrix_85_4.py
precondition OK -- baseline suite green

[KILLED] M1 kill-switch halt leaks the ':362' placeholder status again      2 failed
[KILLED] M2 alert title stops naming the phase                             2 failed
[KILLED] M3 alert details drop died_in_phase                               3 failed
[KILLED] M4 a timeout is treated as a completion                           2 failed
[KILLED] M5 the completed-age clock never goes stale                       3 failed
[KILLED] M6 a ledger with zero completions falls back to the sentinel      2 failed
[KILLED] M7 merged dispatch silently falls back to the two-gather barrier  2 failed
[KILLED] M8 the watchdog stops paging on the completed-age verdict         2 failed
[KILLED] M9 STUB MUTATION: the injected kill-switch state stops being paused 2 failed

MUTATION MATRIX PASSED -- 9/9 mutations killed, tree restored byte-for-byte, suite green.
```

### The matrix found two real defects in my own first-draft tests. Both are fixed; both are disclosed.

1. **M8 was LIVE on the first run.** `test_the_real_watchdog_block_wires_the_alarm`
   asserted the scheduler *source text* contained `fire_cycle_completed_stale_alarm`
   and `should_alarm_success`. Neutering the branch to `if False and ...` keeps
   both strings, so the test passed against a dead alarm. **Fix:** extracted
   `check_cycle_health_alarms()` and replaced the source scan with three tests
   that call it (fire-once-per-transition, stay-silent-when-healthy,
   payload-contents).

2. **M9 was LIVE on the first run — and the reason is the phase-36.28 defect.**
   Mutating the fake trader's kill-switch verdict did not turn the halt tests
   red, because `cycle_halt_reason` also consults
   `kill_switch.get_state().is_paused()`, which **replays the operator's real
   on-disk audit journal**. My tests were passing because the operator's book
   happens to be paused, and would have flipped red the moment it was resumed.
   **Fix:** an autouse fixture now injects a *healthy* stub by default, and a
   `paused_kill_switch` fixture injects the pause explicitly. The new tests read
   no live operator state.

A third latent bug was caught while writing the tests: a concurrency probe
raised inside a `gather(return_exceptions=True)`, which converted the crash
into a *result* and left the probe's counter pinned — the test "failed for the
wrong reason". It now asserts the probe itself survived before trusting its own
measurement.

---

## 8. Verification command (immutable) — verbatim output

```
$ bash -c 'python3 -c "import json,collections;rows=[json.loads(l) for l in open(\"handoff/cycle_history.jsonl\") if l.strip()];term=[r for r in rows if r.get(\"status\") in (\"completed\",\"timeout\",\"error\")];print(collections.Counter(r[\"status\"] for r in term[-10:]))"'
Counter({'completed': 7, 'timeout': 3})
exit=0
```

**Stated plainly: this command cannot fail.** It prints a histogram and exits 0
regardless of content. It is a reporter, not a gate. The gate is the six
success criteria, and the evidence for each is §2–§7 above. Recording this
rather than letting a green exit code imply a green step
(auto-memory `feedback_immutable_criteria_must_be_green_able`).

Note the counter is unchanged from the contract's baseline because **no cycle
has run since** — the cron is weekday-only and this work was done overnight
into a Sunday. No new cycle was triggered.

## 8b. Python lint gate (qa.md §1a) — added in cycle-2 after the Q/A CONDITIONAL

**This gate was NOT run in EVALUATE pass 1. That omission was the blocking
finding, and it was a real one:** it let a dead `import types` into a file this
step created. Recorded here so it is measured, not assumed.

Scope is DERIVED from git — never hand-listed — with a non-empty guard, and
built into a bash **array** (zsh does not word-split an unquoted `$VAR`, which
has previously produced runs that executed nothing and printed success):

```
$ FILES=()
$ while IFS= read -r f; do [ -n "$f" ] && [[ "$f" == *.py ]] && [ -f "$f" ] && FILES+=("$f"); done < <(
    { git show --name-only --pretty=format: e0287161; git diff --name-only HEAD; git ls-files --others --exclude-standard; } | sort -u )
DERIVED SCOPE: 9 python file(s)
  backend/config/settings.py
  backend/services/autonomous_loop.py
  backend/services/cycle_health.py
  backend/slack_bot/scheduler.py
  backend/tests/test_phase_85_4_completed_age_alarm.py
  backend/tests/test_phase_85_4_cycle_loudness.py
  backend/tests/test_phase_85_4_dispatch_barrier.py
  scripts/diagnostics/measure_analysis_phase.py
  scripts/qa/mutation_matrix_85_4.py

$ uvx ruff check --select F821,F401,F811 "${FILES[@]}"
All checks passed!
ruff_exit=0
```

### Proof the gate is LIVE, not vacuously green

A green gate is worth nothing unless it can go red on this exact scope. The
dead import is re-injected through `--stdin-filename`, so the tree is never
touched:

```
$ python3 -c "<re-insert 'import types'>" | uvx ruff check --select F821,F401,F811       --stdin-filename backend/tests/test_phase_85_4_cycle_loudness.py -
F401 [*] `types` imported but unused
  --> backend/tests/test_phase_85_4_cycle_loudness.py:36:8
help: Remove unused import: `types`
Found 1 error.
ruff_mutated_exit=1

$ uvx ruff check --select F821,F401,F811 backend/tests/test_phase_85_4_cycle_loudness.py
All checks passed!
ruff_tree_exit=0
```

### The broader default rule set — measured, and why it is not the gate

```
$ uvx ruff check "${FILES[@]}"
Found 225 errors.
ruff_default_exit=1
```

**Number correction (EVALUATE pass 2).** The Q/A re-derived **223**, not 225,
and **21**, not 23, for the step-created files. Both of my figures were captured
BEFORE the `chmod 0755` described at the end of this section, which cleared the
two `EXE001` findings. The Q/A confirmed the deltas reconcile exactly against
that fix. Current true values: **223** total, **21** on step-created files.

```
```

**The repository has no ruff configuration** — no `ruff.toml`, no `.ruff.toml`,
no `[tool.ruff]` in `pyproject.toml` (there is no `pyproject.toml`). So a bare
`ruff check` runs ruff's entire default rule set, which this codebase has never
been held to. Per-file counts make that plain:

| file | default-ruff findings | authored by this step? |
|---|---|---|
| `backend/services/autonomous_loop.py` | 109 | no — pre-existing file |
| `backend/slack_bot/scheduler.py` | 57 | no |
| `backend/services/cycle_health.py` | 34 | no |
| `backend/config/settings.py` | 2 | no |
| `scripts/diagnostics/measure_analysis_phase.py` | 13 | **yes** |
| `scripts/qa/mutation_matrix_85_4.py` | 5 | **yes** |
| the 3 new test files | 5 total | **yes** |

The 5 files this step created carry **21** of the 223 (see the correction above). Their rule codes:
`UP045` x7 (`Optional[X]` -> `X | None`, a style preference the surrounding
modules do not follow), `PLR0402` x3, `ISC004` x3, `I001` x2 (import sorting),
`BLE001` x2 (deliberate fail-open `except Exception`), `S112` (deliberate
skip-unparseable-log-line), `PLW1510` (`subprocess.run` without `check=` —
deliberate; the mutation matrix WANTS a non-zero exit), `DTZ007` (naive
`strptime` — deliberate; `backend.log` timestamps are naive local time),
`RUF046`. **None is a correctness-class finding**, and none is in the
`F821/F401/F811` family the qa.md gate names.

Two `EXE001` findings (shebang present, file not executable) WERE a real
inconsistency introduced by this step, and are fixed: both new scripts are now
`0755`. Re-checked — `EXE001` cleared.

Bringing the repository as a whole onto the default rule set is a separate
piece of work, not something to smuggle into a P0 engine-health cycle-2 fix.

## 9. Test totals

```
backend/tests/test_phase_85_4_cycle_loudness.py        4 passed
backend/tests/test_phase_85_4_completed_age_alarm.py  14 passed   (12 + 2 from cycle-2)
backend/tests/test_phase_85_4_dispatch_barrier.py     16 passed
                                                      34 passed
```

Mutation matrix: **10/10 killed** (9 + M10 from cycle-2).

Full-suite regression vs the pre-existing 26 failures: see
`handoff/current/live_check_85.4.md`.

## 10. What this step does NOT fix

- **The book still cannot trade**, for two reasons this step deliberately does
  not touch: the kill switch is latched paused and resume returns 409 (that is
  **85.6**), and the timeout itself is only *diagnosed* here — the remedy is a
  dark flag plus two operator asks, per criterion 5.
- `_run_single_analysis` still has no inner per-ticker timeout.
- `orphan_rows()` still has zero production callers (the SIGKILL case). The
  completed-age clock covers the "nothing completed in N days" symptom, which
  is what criterion 4 asked for; wiring `orphan_rows` is a separate concern.

Each is queued rather than mentioned in prose only.


---

## 11. Cycle-2 — what changed after the Q/A CONDITIONAL (EVALUATE pass 1)

Verdict transcribed verbatim in `handoff/current/evaluator_critique_85.4.md`
(machine copy: `evaluator_critique_85.4.json`). Q/A found criteria 1-5 MET and
independently re-derived them — it re-ran the C1 measurement script to
byte-identical numbers and constructed **four mutations of its own**, all
killed. It blocked PASS on one deterministic gate and flagged one code gap.

### Blocker 1 — qa.md §1a lint gate RED, and never run by the author

> `F401 [*] `types` imported but unused --> backend/tests/test_phase_85_4_cycle_loudness.py:36:8 ... Found 1 error.`

**Fixed.** The import was genuinely dead — `grep -nE '\btypes\b'` on that file
now returns zero lines. `types` is used in the *age-alarm* file, not this one;
it was copied into the wrong header.

**Root cause of the miss, stated plainly:** I ran pytest and a mutation matrix
and treated those as "the gates". qa.md §1a makes lint REQUIRED on any
`*.py`-touching diff and I never ran it. An author-side gate that was never run
is an unmeasured claim, not a passed one. §8b now records the gate, its derived
scope, its exit code, **and a proof it can still go red**.

### Blocker 2 — no lint evidence anywhere in the handoff

**Fixed** by §8b, which additionally quantifies the broader default rule set and
says why it is not the project's standard rather than quietly ignoring it.

### WARN — fail-open in the completed-age alarm

Q/A found: if the newest `status == "completed"` row exists but its
`completed_at` fails `_parse_iso`, `success_dt` is None **and** the
no-completion fallback does not fire, so `success_stale` stayed False. One
corrupt timestamp would silence the "book is not trading" P1 indefinitely — a
fail-OPEN hole in an alarm whose entire purpose is loudness.

**Fixed** at `cycle_health.py` with an explicit `elif last_success_row is not
None: success_stale = True`. An unevaluable clock is not a healthy clock: the
age stays `None` (honest — it is unknown) and the P1 renders it as `never`
rather than printing a number it cannot support.

Two new tests
(`test_unparseable_success_timestamp_pages_instead_of_going_silent`,
`test_unparseable_success_timestamp_still_renders_a_sendable_p1`) plus mutation
**M10**, which re-opens the hole and is killed.

### NOTE accepted, not actioned

Q/A observed that the `_died_in` comment asserts every phase appends its name on
entry, which it verified for all five phases it checked but which is not
asserted in general. Recorded as a known limit rather than fixed — asserting it
would need a static check over every `summary["steps"].append` site, which is
its own piece of work.

### Not done during EVALUATE, deliberately

36.28 was **not** widened despite this step surfacing a second instance of its
defect class (the live `.cycle_heartbeat.json` overwrite). A gap noticed
mid-evaluation goes in the NEXT cycle, not into the tree being graded.