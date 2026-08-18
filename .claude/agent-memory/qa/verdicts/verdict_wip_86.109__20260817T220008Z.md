STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.109
WRITTEN: 2026-08-17T22:00:08Z

## Write-first record (crash-survival). NOT a verdict.

Q/A spawn for masterplan step 86.109 (freshness weekend false-positive).
Launch: Workflow structured-output rail.

### Plan
A. Harness compliance audit (5 items)
B. Deterministic: immutable cmd, git scope, ruff lint, scoped pytest, runtime smoke
C. Mutation / guard-vacuity on the new guards
D. Criteria 1-6 MET/NOT MET

### Findings log (appended as established)

#### Attempt / sequence evidence
- `qa_wip.py 86.109 --spawned-at 2026-08-17T22:00:08Z`: source_present=true,
  attempt_number=1, attempt_number_status="ok", prior_attempts=0,
  records_retained=1 (gauge, includes my own), prior_records=[].
- `verdict_history_86_21.py --step 86.109 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). Cross-check prior_attempts(0) vs ledger rows(0) -> not stale by
  the rule; but ledger has NO rows for this step at all and nothing writes it
  automatically. This is attempt 1 => no verdict-shopping possible.

#### A. Harness compliance (mtimes, `stat -f %Sm` LOCAL time)
- research_brief_86.109.md  2026-08-17T23:21:35  (112,228 bytes)
- contract_86.109.md        2026-08-17T23:26:14
- code files                2026-08-17T23:53:44 .. 23:55:20
- experiment_results        2026-08-17T23:59:34
- live_check                2026-08-17T23:59:04
=> ORDER CORRECT: research < contract < generated artifacts < results.

#### B. Immutable verification command
```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/cycle_health.py\").read())" && echo parses'
parses
IMMUTABLE_CMD_EXIT=0
```
NOTE: cycle_health.py mtime is 2026-08-11 -- UNCHANGED by this step. The
immutable command is therefore a weak (but green) gate; it does not exercise
the fix. Not the step's fault (criteria are immutable).

#### Production call-site census (derived, not typed)
`grep -rn --include=*.py "compute_freshness" backend/ scripts/` -> production
call sites are exactly:
- backend/api/paper_trading.py:509      emit_alarm=False
- backend/api/observability_api.py:41   emit_alarm=False
- backend/api/observability_api.py:63   emit_alarm=False
- backend/services/freshness_cron.py:146 emit_alarm=False (pre-existing, 82.10)
=> ALL four production call sites now pass emit_alarm=False. The only remaining
notifier is freshness_cron's own `notify(...)` loop. Criterion 4's option (a)
"stop emitting" is implemented at all three read paths.

#### Deferral logic re-derived by hand (freshness_cron.py:148-212)
baseline=B, red_now=R, newly_red=N = R (if B is None) else R-B.
- trading: `_last_red_sources = R`  (unchanged semantics)
- non-trading: `_last_red_sources = R - N` = R∩B (or ∅ when B is None)
  -> withheld sources held OUT of baseline; still-red-from-before retained;
     recovered sources dropped. Page loop is `sorted(N) if trading else []`.
Traced Fri-dead-writer: Sat/Sun withheld+logged, Mon N={A} -> PAGES. Correct.

#### Scoped test reproduction (my run, unpiped exits)
- `.venv/bin/python -m pytest backend/tests/test_phase_86_109_freshness_calendar.py -q` -> 17 passed
- `.venv/bin/python -m pytest backend/tests/test_phase_82_10_freshness_paging.py -q`   -> 16 passed
  (17 + 16 = 33 -- reproduces the artifact's "33 passed"; dot-count matches both.)

#### MY OWN mutation probes (sys.modules injection; TREE NEVER TOUCHED)
Harness: scratchpad/qa_probe_86109.py -- injects a mutated
`backend.services.cycle_health` before collection, runs the step's own suite.
POSITIVE CONTROL RUN FIRST so a survivor cannot be an injection artifact.

| cell | mutation | collected | result |
|---|---|---|---|
| Q0 control | none | 17 | 17 passed (rc=0) -- control GREEN |
| Q3 pos-control | `_band` always "green" | 17 | **3 failed** (rc=1) -> injection DOES reach the tests' module |
| **Q1** | calendar moved INTO DETECTION: `_band` returns "green" when ET weekday>=5 | 17 | **SURVIVED, 17 passed** |
| **Q2** | `emit_alarm` made INERT: `if False and emit_alarm ...` so the inner alarm can never fire | 17 | **SURVIVED, 17 passed** |

Q1 non-equivalence PROVEN (not an equivalent mutant):
```
Mon 2026-08-17 (today ET)  _band(64h,86400)='red'    _band(50h,86400)='red'
Sat 2026-08-15             _band(64h,86400)='green'  _band(50h,86400)='green'
real ET now: 2026-08-17T18:06:22-04:00  weekday=0 (Monday)
```
=> the guard against "the calendar reaches _band()" is DATE-CONDITIONAL: it can
only fire on Sat/Sun. Vacuity shapes #4 (tautology: `first == second` is two
identical calls in the same instant) + #9 (executor-environment
non-reproducibility).

FALSIFIED CLAIMS (both are the artifact's own words):
- test file module docstring: "The calendar gates NOTIFICATION only; **if it
  ever reaches `_band()` these tests break**." -> Q1 SURVIVED.
- `test_band_has_no_day_of_week_term_after_the_fix` docstring: "If a future
  change moves the calendar into detection, this fails." -> it did not.
- live_check s4: "That last one [`test_compute_freshness_still_pages_when_asked`]
  is the anti-vacuity control: the first three would also pass if `emit_alarm`
  had simply stopped working." -> Q2 SURVIVED. Reading the test confirms why:
  it never calls `compute_freshness`, never asserts its own `fired` list, and
  its `_BQ` class + `_fire_freshness_alarm` swap are dead code. Its only live
  assertion is `__kwdefaults__["emit_alarm"] is True` -- a signature check that
  phase-82.10 already had at test_phase_82_10:462.

#### 82.10 inverted guard is COMMENT-SATISFIABLE (executed, not reasoned)
`assert "emit_alarm=False" in src` -- the same diff ADDS a comment containing
that exact literal to both files. Measured:
```
api/paper_trading.py:       code-line occurrences=1  comment-line occurrences=1
api/observability_api.py:   code-line occurrences=2  comment-line occurrences=2
after mutating EVERY real call site to emit_alarm=True -> assertion still True (both files)
```
=> vacuity shape #8 (comment-token trap). MITIGATION: the test's own docstring
says "This is a source scan and is not the real guard", and the real guards
(the three DRIVING tests) exist and are killed by matrix cells N4/N5.

#### AUTHOR MATRIX REPRODUCED BY ME (`.venv/bin/python scripts/qa/mutation_86_109.py`)
```
CONTROL rc=0  collected=17          <- control GREEN observed FIRST
N1..N8 all KILLED, named test failing, collected 17 == control each
KILLED=8/8  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.
```
Post-matrix `git diff --stat HEAD` on the 5 prod files is byte-for-byte the
same as pre-matrix (8/14/30/61/9 lines) -- tree restored, nothing leaked.
N1 IS criterion 5 literally: `trading_day = True  # MUTANT: gate removed`.

#### MY OWN end-to-end drive (REAL compute_freshness, all-red fixture, no network)
```
paper_trading /freshness           overall_band=red, alarm invocations = 0
get_observability_freshness        overall_band=red, alarm invocations = 0
get_observability_data_freshness   overall_band=red, alarm invocations = 0
CONTROL compute_freshness(emit_alarm=True)  overall=red, alarm invocations = 1
```
=> suppression is TARGETED, not a blanket break. This is the guard the author's
suite lacks (their three tests stub compute_freshness itself).

#### Live-system state (side-effect-free; I did NOT curl the freshness routes,
#### because on the pre-restart process that would itself fire the defect)
- `/api/health` 200, version 6.93.235.
- `ps -o pid,lstart -p 41635` -> started man. 17 aug. 15.57.16 2026 CEST =
  2026-08-17T13:57:16Z. Disclosure ACCURATE (local/Z conversion correct).
- backend.log: `registered freshness_evaluator: every 6h` at 15:57:19; 11
  evaluator ticks; latest `overall=red red=2 newly_red=2 alerts=2` 21:57:30.
  => the SOLE remaining notifier is genuinely alive on the running process.

#### Claim audit (all re-derived by me)
REPRODUCE EXACTLY:
- 1149 "Data freshness critical" lines across 7 .gz + live log.
- 867 / 204 / 78 split: per-archive 158+148+416+91+54 = 867; 107+97 = 204; 78.
  Sums to 1149.
- `detected_by` = 0 occurrences in the whole corpus (disclosed by Main as a
  probe that could not have failed -- correct and creditable).
- ruff over the DERIVED scope: exactly 1 finding,
  `backend/backtest/markets.py:9:20 F401 typing.Optional`. PRE-EXISTING --
  reproduces on `git show HEAD:...` copy; Optional count 1 in HEAD and worktree.
- 17 passed / 16 passed / 33 combined; scoped selection AFTER the disposition:
  `218 passed, 7 skipped, 3421 deselected` (= 217+1, consistent with the
  stated BEFORE state).
- Contract carries all 6 criteria VERBATIM from masterplan.json.
- Research gate: brief_status COMPLETE, gate_passed true, 40 read-in-full,
  134 urls (135 distinct in file), recency scan section present + non-empty,
  audit-class dry after 23 rounds.
- Zero non-ASCII characters ADDED by the diff; zero secret-shaped literals.
- All 6 changed backend modules import cleanly in the venv.

DO NOT FULLY REPRODUCE:
- live_check s8 "(9 files)": my derivation of the SAME command yields **10**.
  Every file involved predates the live_check write (23:59:04) -- peer files
  15:54 and 21:42, step files 23:53-23:57 -- so the scope was already 10 when
  the count was written. Shape #10 (hand-derived-scope staleness). NO hidden
  defect: ruff over all 10 finds the identical single pre-existing F401.
- live_check s10 "38.4% of the pages landed on a MONDAY": the 38.4% is EXACT
  (Mon 165) but over **430** datable lines, not 1149 -- 719 corpus lines carry
  a time only (`17:27:19 W [alerting]`), no date, so they cannot be weekdayed.
  Denominator undisclosed. Direction unaffected (Sat+Sun = 30.0%).
- live_check s5 "It lands at the notifier -- **one of the three named**":
  FALSE. `git diff HEAD -- backend/services/cycle_health.py` is EMPTY, so
  `_band`, `compute_freshness` AND `_fire_freshness_alarm` are ALL unmodified.
  The calendar lands in `backend/services/freshness_cron.py::run_freshness_check`,
  which is none of the three. The DEVIATION ITSELF I judge SOUND (see below);
  only its description is wrong.

#### Criterion 6 evidence
- `.claude/masterplan.json` byte-identical to HEAD (`git diff --stat` blank);
  step 86.109 status still `pending`.
- Last masterplan commit 8200283c: 4 `+ "status": "pending"` lines, ZERO `-`
  status lines -> filings only, no flip.
- No `evaluator_critique*` / `verdict_ledger.jsonl` file modified.
- `grep -F "86.109" handoff/harness_log.md` -> 0 hits (LOG-LAST correct).

#### JUDGMENT ON THE TWO DEVIATIONS I WAS ASKED TO RULE ON
1. Calendar at the NOTIFIER, not in `_band()`: **SOUND**. Grafana/PagerDuty/
   Alertmanager all put mute timings on the routing leg; a calendar-aware
   `_band()` would make a Friday-dead writer indistinguishable from an idle
   weekend and would put criterion 3 at risk; cycle_health already carries
   three calendar notions, one holiday-blind. Disclosed in contract, live_check
   s5, experiment_results and the spawn prompt -- routed, not silently edited.
   Accepted.
2. 82.10 `test_http_call_sites_were_not_edited_to_pass_emit_alarm` INVERTED:
   **CORRECT DISPOSITION, not a loosened gate**. Its subject changed by an
   AUTHORISED immutable criterion (86.109 C4); the docstring proves it was
   82.10 pinning its OWN scope, not a standing policy; it was inverted in
   place with the supersession at the site of the original claim, not deleted
   or no-op'd; and it points at the driving guards. Defect is only that the
   replacement literal is comment-satisfiable (F3).

#### VERDICT REASONING (worst-of-N lenses)
- correctness lens: PASS -- product logic verified independently end-to-end;
  deferral algebra re-derived; Fri-death, first-run-on-Saturday, restart,
  recover-over-weekend and steady-state edges all traced correct.
- does-it-reproduce lens: CONDITIONAL -- every load-bearing number reproduces;
  the "9 files" annotation does not.
- scope-honesty lens: CONDITIONAL -- deviation disclosed but MIS-DESCRIBED,
  and THREE guard-capability claims are falsified by execution (Q1, Q2, F3).
min(lenses) = CONDITIONAL.

All findings are EVIDENCE-side. Zero product defects found. Per qa.md 4c the
vacuous guards coexist with genuine, mutation-proven behavioural guards
(N1-N8), so they are WARN-level, not blocking.

#### HEAD-move recheck before returning
Session-start gitStatus said HEAD=b35c5606; live HEAD is 81c5e7fa. NOT a
mid-eval move: 8200283c/81c5e7fa committed 2026-08-17T21:49:23Z, BEFORE my
first tool call at 22:00:08Z. The snapshot was stale. Every diff I ran was
against a stable HEAD=81c5e7fa.
- 8200283c (phase-86.108) also touches `backend/api/observability_api.py`, but
  HEAD does NOT contain 86.109's `emit_alarm=False` (they appear as `+` lines
  in `git diff HEAD`), so 86.108's `git add -A` did not sweep in 86.109's code.
  It DID sweep `contract_86.109.md` + `research_brief_86.109.md` (both written
  before 21:49Z) -- normal auto-commit behaviour, artifacts unmodified since.
- `handoff/verdict_ledger.jsonl` and `handoff/harness_log.md` in that commit:
  **0 removed lines each** -> append-only, no prior verdict altered. C6 holds.

#### Attempt/sequence, re-read after the HEAD check
- `qa_wip.py 86.109 --spawned-at 2026-08-17T22:00:08Z`: source_present=true,
  attempt_number=1 (status "ok"), prior_attempts=0.
- ledger `--evidence-only`: status=`no_rows_for_step`, verdicts=(none).
- prior_attempts(0) vs ledger rows(0): equal -> NOT stale by the rule; but the
  ledger holds no rows for this step at all and nothing writes it
  automatically, so it is weak evidence either way.

COMPLETED: 2026-08-17T22:14:30Z

