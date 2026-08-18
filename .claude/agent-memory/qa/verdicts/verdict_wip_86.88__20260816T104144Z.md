STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.88
WRITTEN: 2026-08-16T10:41:44Z

# Q/A write-first record -- step 86.88, cycle 2

## Plan
1. Harness-compliance audit (5 items)
2. Deterministic: immutable command exit code; git diff scope; ruff; scoped pytest
3. Mutation matrix: attack the additive key `judge_verdict_absent`, the three new tests,
   criterion 2's "asserts no order results", criterion 4 dead branch, criterion 5 seam
4. Criterion-by-criterion MET/NOT MET

## Findings log

### Prior-attempt / sequence evidence
- `qa_wip.py 86.88 --spawned-at 2026-08-16T10:41:44Z`: source_present=true,
  identity_checked=true, attempt_number=2 (status "ok", is_lower_bound false),
  prior_attempts=1, records_retained=2 (gauge).
- `verdict_history_86_21.py --step 86.88 --evidence-only`: status
  `no_rows_for_step`, verdicts `(none)`.
- CROSS-CHECK: attempt_number 2 > ledger count 0 -> THE LEDGER IS STALE for this
  step. Sequence from the ledger is unreliable. (Cycle-1 verdict CONDITIONAL is
  recorded in handoff/current/evaluator_critique_86.88.md, transcribed verbatim.)

### Deterministic
- IMMUTABLE: `python scripts/qa/verify_lite_risk_seam_86_86.py` -> exit 0,
  "checks emitted: 9 (PASS 9 / FAIL 0) RESULT: OK". 4 whole-dict routes SEEN at
  [3243, 3248, 3477, 3482].
- pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q -> **75 passed**.
- ruff --select F821,F401,F811 over commit-derived scope (3 .py files from
  4e01f3b6+03386529+786b5a55, xargs -0) -> "All checks passed!" exit 0.
- git: HEAD=33f5cf7d. Working tree dirty ONLY in backend/api/sovereign_api.py
  (+ 5 frontend files) -- the peer session's "1y" red-line window; in NEITHER
  86.88 commit. Not this step's doing (same as cycle 1 noted).

### FINDING A (MEASURED) -- the additive key does NOT reach any persisted record
Drove the REAL `_run_claude_analysis` twice (judge-failure prose vs real 3% JSON),
then drove the REAL `_persist_analysis` with a save_report-capturing stub:
- in-memory: FAIL judge_verdict_absent=true / REAL=false  -> the key IS set correctly
- `full_report` keys on the lite path = ['analysis','market_data','source'];
  `risk_assessment` is NOT inside full_report
- `'judge_verdict_absent' in persisted full_report_json` -> **False** for BOTH
- persisted full_report blob sha256 = 03051590ade45d6b for BOTH (identical)
- save_report named cols: risk_judge_decision='APPROVE_REDUCED',
  risk_level='MODERATE', recommended_position_pct=3.0 -- IDENTICAL for both states
- only differing kwarg is `summary` (from risk_assessment["reason"]), which is the
  PRE-EXISTING fabricated-reasoning path filed as 86.87 -- unchanged by 86.88
Repo census: `judge_verdict_absent` appears in exactly 1 production line
(autonomous_loop.py:2469) + 3 test assertions. NOTHING consumes it.
=> the artifact/production-comment claim "IN THE RECORD, where a downstream reader
or an auditor can see it" does not reproduce for any persisted artifact.
The docstring at autonomous_loop.py:2325 explicitly contrasts with "no auditor
reading the persisted row can see it" -- implying the key IS row-visible. It is not.
Remediation is one line: `_persist_analysis` ALREADY stamps `_path`/`_degraded`
into `full_report` (:3543-:3556); the same stamp would carry this flag.

### FINDING B (MEASURED) -- live_check_86.88.md was NOT regenerated in cycle 2
`git log -- handoff/current/live_check_86.88.md` -> last touched by 786b5a55
(cycle 1). `git show --stat 4e01f3b6` does NOT list it. Yet the cycle-2 spawn
prompt + experiment_results claim the bound is "corrected in BOTH artifacts".
Stale content still shipped in the operator-facing gate artifact:
 - Sec4 "POST-FIX mutation matrix -- 7/7 KILLED": CONTROL 69 passed; M1 "1 failed,
   68 passed". MEASURED on the shipped tree: control 75, M1(N1@Claude) 2 failed /
   73 passed. Numbers do not reproduce. This is cycle-1 finding 3 recurring in the
   OTHER file.
 - Sec7 gate block: "72 passed" -- shipped suite is 75.
 - Sec8 stated bound: "`{**_LITE_RISK_DEFAULT}` would NOT be seen". MEASURED after
   the cycle-2 widening: seen=True. The bound is now wrong in the OPPOSITE
   direction.
 - Sec2/Sec3 line numbers (3214/3219/3448/3453) are pre-cycle-2; immutable command
   now reports 3243/3248/3477/3482.

### Independent mutation matrix (in-memory sys.modules injection, tree never written)
pre/post sha256 autonomous_loop.py 16fd1fbd... IDENTICAL;
portfolio_manager.py 042cd8e5... IDENTICAL.
```
C0 CONTROL                                     75 passed            GREEN
Q1 drop the additive key            (=M8)       3 failed, 72 passed  KILLED
Q2 key always False                 (=M9)       1 failed, 74 passed  KILLED
Q3 key always True                  (=M10)      2 failed, 73 passed  KILLED
Q4 IDENTITY instead of value equality           1 failed, 74 passed  KILLED
Q5 subset match ignoring 'reasoning'           75 passed             SURVIVED
Q6 revert the seam early-return     (=M4)       4 failed, 71 passed  KILLED
Q7 N1 pre-mangle @ Claude route     (=M1)       2 failed, 73 passed  KILLED
Q8 PM: _coerce_pct collapses 0.0               22 failed, 53 passed  KILLED (not isolating)
Q9 decide_trades: SIZE(0.0)->None              75 passed             EQUIVALENT (see below)
Q10 _sizing_pct: SIZE(0.0)->10% default         1 failed (crit-2 test ONLY, at :944)
```
- Main's M1/M4/M8/M9/M10 counts REPRODUCE EXACTLY on the shipped 75-test tree.
- Q10 is the discriminating cell for criterion 2: it kills at test line 944, the
  NEW `assert _buy(orders) is None`, with "a judge verdict of 0% produced a BUY"
  and TradeOrder amount_usd=2399.77 -- so the decide_trades drive is load-bearing
  beyond the upstream pct assert (which still passes under Q10). Criterion 2 MET.
- Q9 SURVIVED but is EQUIVALENT, checked not assumed: `position_pct_state` carries
  SIZE independently, and `_sizing_pct` returns 0.0 for SIZE-with-no-number
  (fail-closed, portfolio_manager.py:1044-1048). No behavioural differential.
- Q5 SURVIVED with a REAL differential: a judge returning every default value with
  its own `reasoning` would be labelled absent. Inert today (nothing reads the
  key); becomes material if Finding A is fixed by threading the key to persistence.

### Criterion 7 -- re-derived independently, STRONGER than the artifact
Loaded the parent blob (22dd1fc3) and the shipped file side by side; for all 7
disclosure-table inputs computed pct, PositionVerdict, and the REAL `decide_trades`
order under binding=False AND True. "INPUTS WHOSE NUMBER / VERDICT / ORDER MOVED:
NONE". Key-set delta PRE->POST = ['judge_verdict_absent'], removed: []. Purely
additive. WHOLE DEFAULT: 3.0 -> 3.0, SIZE(3.0) -> SIZE(3.0), 1 BUY $719.93 both.

### Criterion 4 -- known-member recall on the widened checker (all 7 shapes)
dict(X) / deepcopy(X) / copy.deepcopy(X) / copy.copy(X) / dict(**X) / X.copy() /
{**X} -> seen=True for ALL 7. Negative controls: prose+comment quoting the idiom
-> seen=False; unrelated dict -> seen=False. Residual alias (d=X; dict(d)) ->
seen=False, correctly stated as a residual in both the code comment and
experiment_results. Immutable command shows the branch firing on 4 REAL matches.

### Criterion 8 / scope
Commits 03386529 + 786b5a55 + 4e01f3b6 touch exactly: autonomous_loop.py, the test
file, verify_lite_risk_seam_86_86.py, and 5 handoff artifacts. NO settings.py,
NO .env, NO masterplan.json. Immutable criteria unamended.

### Consumer-break audit (Main's question C) -- CLEAN, measured
No `extra="forbid"` model covers risk_assessment (the 10 hits are unrelated signal
schemas). TS `RiskAssessment` (types.ts:439) is the FULL-path shape and TS does not
validate at runtime. The key never reaches BQ, so no schema risk. Repo census:
`judge_verdict_absent` = 1 production line + 3 test assertions, zero consumers.

### Harness compliance (5 items)
1. research_brief_86.88.md: brief_status COMPLETE, sources 15, urls 30,
   recency_scan true, coverage.dry true, gate_passed true. mtime 12:08:43.
2. contract mtime 12:11:20 > brief; code edits 12:39:38 / 12:40:33 > contract.
3. experiment_results_86.88.md present (15,398 B), cycle-2 Follow-up appended.
4. log-last: `grep -cF "phase=86.88" handoff/harness_log.md` = 0; masterplan
   86.88 status=pending. Clean.
5. no-verdict-shopping: evidence CHANGED (4e01f3b6 = 2 production files + checker
   + 2 artifacts). Legitimate cycle-2 respawn.

### Runtime smoke / gates not binding
- `python -c "import backend.services.autonomous_loop"` OK; helper returns
  False/True correctly. `/api/health` -> status ok (version 6.93.222). NOTE: the
  RUNNING backend still holds the pre-fix module; committed is not in force until
  the session-end restart (CLAUDE.md batched-restart rule) -- operator business,
  not a Q/A blocker.
- 1b frontend gate does NOT bind (no frontend/** in the 86.88 commits; the dirty
  frontend files are a peer session's "1y" red-line work).
- 1c live-UI gate does NOT bind (no UI claims in contract, criteria or diff);
  no Playwright capture taken.
- Code-review heuristics: only hits are `anthropic_api_key="test-key"` inside test
  fixtures -- negation-list exempt. No broad-except, no subprocess/eval/exec, no
  kill-switch/stop-loss/perf-metrics path touched.

### CRITERION ROLL-UP
1 MET · 2 MET (Q10-proven load-bearing) · 3 MET · 4 MET (7/7 recall + 2 neg
controls) · 5 PARTIAL -- letter met (enumerated/classified/stated/seam-located)
but the STATED REACH does not reproduce · 6 MET · 7 MET (re-derived with order
outcomes) · 8 MET.

### VERDICT REACHED: CONDITIONAL
Two blockers: (A) auditor-visible-provenance claim does not reproduce -- the key
reaches no persisted artifact; (B) live_check_86.88.md not regenerated, carries
a superseded 69-test matrix, "72 passed", and a now-inverted stated bound.
Plus WARN: Q5 subset-match mutant survived (unpinned exactness).

COMPLETED: 2026-08-16T10:53:44Z


