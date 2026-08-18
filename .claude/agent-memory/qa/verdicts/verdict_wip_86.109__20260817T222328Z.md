STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.109
WRITTEN: 2026-08-17T22:23:28Z
COMPLETED: 2026-08-17T22:37:43Z

# Q/A write-first record -- step 86.109 (freshness weekend false positive)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git scope, lint, tests, mutation matrix re-run
C. LLM judgment vs 6 immutable criteria

## Log
- [start] Read .claude/agents/qa.md in full. Creating WIP record.

### Attempt / sequence evidence
- `qa_wip.py 86.109 --spawned-at 2026-08-17T22:23:28Z`: source_present=true,
  attempt_number=2, attempt_number_status=ok, attempt_number_is_lower_bound=false,
  prior_attempts=1, records_retained=2 (GAUGE), prior_records=[verdict_wip_86.109__20260817T220008Z.md].
- `verdict_history_86_21.py --step 86.109 --evidence-only`: status=`ok`,
  detail="1 verdict(s) from the ledger", verdicts=`CONDITIONAL`.
- CROSS-CHECK: prior_attempts(1) == ledger rows(1) -> ledger NOT stale for this step.
  Sequence: [CONDITIONAL]. This spawn is attempt 2.
- No verdict-shopping: experiment_results/live_check mtime 2026-08-18T00:22:53,
  test file 00:18:30, mutation script 00:20:51 -- ALL later than the prior WIP's
  COMPLETE at 00:14. Evidence CHANGED.

### A. Harness compliance (mtimes via `stat -f %Sm`, LOCAL time)
- research_brief_86.109.md  2026-08-17T23:21:35  (112,228 bytes)
- contract_86.109.md        2026-08-17T23:26:14
- evaluator_critique        2026-08-18T00:17:43
- experiment_results        2026-08-18T00:22:53
- live_check                2026-08-18T00:22:53
=> ORDER CORRECT: research < contract < code < results.
- Research gate: brief present, contract cites `wf_8a25910d-384`, 40 sources
  read in full, 134 URLs, audit-class dry after 23 rounds. Contract records the
  FIRST run was enforced-FAILED on a one-URL over-claim and reconciled.
- log-last: step not yet in harness_log with a result; masterplan not flipped (checked below).

### B. Deterministic
```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/cycle_health.py\").read())" && echo parses'
parses
IMMUTABLE_CMD_EXIT=0
```
```
$ .venv/bin/python -m pytest backend/tests/test_phase_86_109_freshness_calendar.py \
      backend/tests/test_phase_82_10_freshness_paging.py -q -p no:cacheprovider
33 passed, 1 warning in 2.59s          <- REPRODUCES the artifact's "33 passed"
```
```
$ FILES=$({ git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)
$ echo "$FILES" | wc -l   -> 10        <- REPRODUCES "10 files", not 9
$ echo "$FILES" | xargs uvx ruff check --select F821,F401,F811 --no-cache --output-format=concise
backend/backtest/markets.py:9:20: F401 [*] `typing.Optional` imported but unused
Found 1 error.
RUFF_EXIT=1
```
- PRE-EXISTENCE INDEPENDENTLY VERIFIED: `git show HEAD:backend/backtest/markets.py`
  copied to scratchpad -> same F401 at 9:20, HEADCOPY_RUFF_EXIT=1; `grep -c Optional`
  = 1 on BOTH HEAD copy and worktree. Not introduced by this step. Filed as 86.113.
```
$ .venv/bin/python -m pytest backend/tests/ -q -p no:cacheprovider -k "freshness or cycle_health or observab or paper_trading or scheduler or markets or calendar or 82_10 or 51_3"
218 passed, 7 skipped, 3421 deselected  <- REPRODUCES verbatim
```
- `git diff --stat HEAD -- backend/services/cycle_health.py` -> EMPTY. Confirms the
  live_check S5 disclosure: `_band`/`compute_freshness`/`_fire_freshness_alarm` are
  ALL THREE untouched; the gate lands at a FOURTH site (freshness_cron.run_freshness_check).
- Production diff reviewed: markets.py (+30, new helper), scheduler.py (delegation),
  freshness_cron.py (gate + deferral), paper_trading.py + observability_api.py
  (emit_alarm=False x3), test_phase_82_10 (guard inverted in place).

### B2. Claim reproduction (4b -- point the instrument at the PROSE)
Every reproducible number in the artifacts REPRODUCED, several to the decimal:
- Criterion-1 capture REGENERATED from the live module (not read): 65h ratio=2.71 red,
  48h ratio=2.00 red, 20h ratio=0.83 green, 50h ratio=2.08 red; CRITICAL=2.0 WARN=1.5.
- Log corpus: 7 rotated .gz per-file = 158/148/416/91/54/107/97 = 1071; live `backend.log`
  (repo ROOT, not handoff/logs/) = 78. TOTAL 1149 EXACT.
  <=2026-08-04 rotations = 158+148+416+91+54 = **867** EXACT; 204 EXACT; 78 EXACT.
- `b7c69bb9` = "feat(82.10): give the freshness alarm a trigger", 2026-08-05 18:29:49 +0200.
  So the >=867 pre-cron bound is sound.
- Weekday census re-derived independently: total 1149, DATABLE 430, undatable 719,
  Mon 165 38.4%, Tue 38 8.8%, Wed 27 6.3%, Thu 35 8.1%, Fri 36 8.4%, Sat 46 10.7%,
  Sun 83 19.3%, Sat+Sun 129 30.0%. EVERY CELL EXACT.
- `CycleHealthStrip.tsx`: `window.setInterval(tick, 30_000)` -- the 30s poll claim holds.
- NOT-YET-IN-FORCE: `ps -p 41635` -> started "man. 17 aug. 15.57.16 2026" local
  (= 13:57:16Z), uvicorn backend.main:app :8000. Predates the 00:21 edits. Claim TRUE.
  I deliberately did NOT curl the live /freshness route: on the pre-fix process that
  call would fire a real P1 Slack page at the operator.
- `/api/health` -> {"status":"ok", version 6.93.235}. Runtime smoke: all 5 changed
  backend modules import clean; is_us_trading_day_now('US')=True; scheduler wrapper
  returns the same and its last source line is `return is_us_trading_day_now("US")`.
- Research gate re-checked at the BRIEF, not the contract: brief_status COMPLETE,
  external_sources_read_in_full 40, urls_collected 134, recency_scan_performed true,
  coverage.dry true, gate_passed true. Recency-scan sections present.

### B3. Call-site completeness (criterion 4) -- ENUMERATED, not assumed
`grep -rn compute_freshness backend scripts --include='*.py' | grep -v /tests/`:
exactly FOUR production invocations --
  paper_trading.py:509 (emit_alarm=False), observability_api.py:36 + :57 (both False),
  freshness_cron.py:146 (False).
So NO production caller emits any more. `_fire_freshness_alarm` is called only at
cycle_health.py:733 inside the now-always-suppressed branch. No missed entry point.

### C. INDEPENDENT MUTATION (mine, not the author's)
Author's matrix RE-RUN: CONTROL rc=0 collected=33; **KILLED=11/11, SURVIVORS=none,
UNSCORABLE=none**; restore SHA-verified. Reproduces.

My own cells. Two techniques: in-process `pytest.main(plugins=[...])` swapping
`ch._band` (linecache-registered so `inspect.getsource` resolves) -- tree untouched;
and file mutation with SHA-256 restore.

| cell | result | note |
|---|---|---|
| QA0 null mutant (identical `_band` re-defined elsewhere) | SURVIVED | harness produces no spurious kills |
| QAP always-green `_band` | KILLED (7 failed) | POSITIVE CONTROL: the swap IS live inside the band tests |
| QA2 calendar via module-level `datetime` | KILLED | |
| QA2b calendar via `getattr(ch,"dat"+"etime")` + `tm_wday`, regex MISSES it | **KILLED** | proves the BEHAVIOURAL freeze half fires on its own -- not dead weight |
| **QA1 calendar via `time.localtime().tm_wday >= 5`** | **SURVIVED 33/33** | evades the 7-token regex AND the `ch.datetime` freeze |
| QB1 revert ONLY the 2nd observability alias | KILLED | `...[get_observability_data_freshness]` -- 2nd site individually covered |
| QB2 cron grows its OWN calendar definition | KILLED | criterion-2 drift is durably guarded |
| QB3 `_last_red_sources = set()` on a non-trading day | **SURVIVED 33/33** | drops already-known-red state |
| QB5 transition gate -> level trigger (`sorted(red_now)`) | KILLED | edge-trigger preserved |
| QB7 `ZoneInfo("America/New_York")` -> `ZoneInfo("UTC")` | **SURVIVED 33/33** | the helper's explicit TZ claim is unguarded |

### FINDINGS (all EVIDENCE-class; ZERO product defects found)
F1 [Overgeneralization, WARN-with-named-fix]. live_check S12 + the test docstring say
the replacement scans `_band`'s source for "any calendar reference". It scans a
SEVEN-TOKEN regex. QA1 above is a genuinely calendar-aware `_band` that survives all
33 tests. NOT sole coverage and NOT vacuous (QA2b/QAP/N9 all fire), and no immutable
criterion fails under QA1 -- criterion 3's literal property (stale WEEKDAY still red)
still holds. COMPLETE named fix, measured: `ch._band.__code__.co_names` IS
`('CRITICAL_RATIO','WARN_RATIO')` and `co_freevars` is `()`; asserting that pins the
CLASS -- any calendar read requires a global name lookup and cannot be renamed around.
F2 [NOTE]. QB3: nothing pins "hold already-known-red out of the baseline". The mutant
re-pages on Monday a source red since before the weekend -- i.e. it re-creates the
38.4% Monday bucket this step is about. Named fix: a third deferral test with a
PRE-EXISTING red source across the weekend asserting NO Monday re-page.
F3 [NOTE]. QB7: `is_us_trading_day_now`'s docstring argues ET explicitly ("A UTC
'today' would be the wrong day for five hours of every evening") and nothing guards
it; the drift would hit BOTH consumers. Named fix: freeze the clock to 20:00 ET and
assert the ET date, not the UTC date, is what reaches `is_trading_day`.

### Code-review heuristics (5 dimensions)
No secret literal in the diff; no kill_switch / paper_trader / risk_engine /
perf_metrics / backtest_engine file touched; no subprocess/eval/exec added; the one new
`except Exception` is a documented fail-open notification gate that LOGS a warning
(N6 proves the polarity); no non-ASCII introduced in any added line (security.md
logger rule); pre-existing non-ASCII is in comments/data labels only.
Peer-session claim CHECKED: `sovereign_api.py` / `autonomous_loop.py` diffs contain
zero "86.109"/"freshness" hits -- unrelated, and ruff finds nothing in them.

### CRITERIA
1 MET  -- regenerated the capture from the live module; synthetic ages, not the filing's ratios.
2 MET (declared deviation judged SOUND, re-derived not deferred). ONE definition:
       `markets.is_us_trading_day_now`, digest wrapper delegates (N8 + QB2 killed).
       Deviation: `cycle_health.py` is byte-unmodified (`git diff --stat` EMPTY, verified
       by me), so ALL THREE named functions are untouched and the gate lands at a FOURTH
       site. Sound because (a) the criterion's own rationale -- no second drifting
       definition -- is over-satisfied (two definitions became one); (b) the literal site
       conflicts with criterion 3, which has behavioural coverage; (c) `_fire_freshness_alarm`
       is now dead in production (all 4 callers pass emit_alarm=False), so gating it would
       gate a dead path. Disclosed accurately, incl. that the earlier wording understated it.
3 MET  -- weekday control driven through the REAL run_freshness_check alongside the
       weekend cell; N2 + QB2 + QB5 kill. Holds under every mutant I built.
4 MET  -- choice STATED (emit entirely removed, not a transition gate) with RFC 9110 +
       Azure basis; all FOUR production call sites enumerated by me and all pass
       emit_alarm=False; N4/N5/QB1 cover the three read paths individually.
5 MET  -- control observed GREEN FIRST (rc=0, collected=33) then N1 removes the gate
       (`trading_day = True`) and the named test goes RED. Reproduced by me.
6 MET  -- masterplan diff 0 lines, 86.109 still `pending`; no evaluator_critique file
       modified; `verdict_ledger.jsonl` diff is a pure +1 APPEND. Nothing flipped.

### Harness compliance: CLEAN (5/5)
research<contract<code<results order correct; gate_passed true at the brief; results
present; step absent from harness_log and masterplan unflipped (log-last intact);
evidence CHANGED since the prior verdict (test 00:18, matrix 00:20, results/live_check
00:22 vs prior WIP COMPLETE 00:14) -> fresh-respawn, not verdict-shopping.

