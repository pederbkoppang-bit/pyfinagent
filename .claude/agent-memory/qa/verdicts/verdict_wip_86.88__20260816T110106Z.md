STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.88
WRITTEN: 2026-08-16T11:01:06Z

# Q/A write-first record -- step 86.88, cycle 3

Spawned via Workflow rail. Read `.claude/agents/qa.md` in full (STEP 0 done).

## Prior-attempt EVIDENCE (not a trigger)
- `qa_wip.py 86.88 --spawned-at 2026-08-16T11:01:06Z`:
  `attempt_number: 3`, `prior_attempts: 2`, `attempt_number_status: "ok"`,
  `attempt_number_is_lower_bound: true`, `source_present: true`,
  `records_retained: 3` (GAUGE, not used), `records_pruned_known: null`.
- `verdict_history_86_21.py --step 86.88 --evidence-only`:
  `status: no_rows_for_step`, `verdicts: (none)`.
- CROSS-CHECK: attempt_number (3) > ledger verdict count (0) => **the ledger is
  STALE** for this step; its sequence is UNRELIABLE. Main's advisory disclosure
  says [CONDITIONAL, CONDITIONAL]; the cycle-2 verdict is on disk in
  `evaluator_critique_86.88.md` (verbatim, CONDITIONAL) but never reached the
  ledger. Recorded as evidence; every rollup over it is the caller's.

## A. Harness compliance -- CLEAN
- mtimes: research_brief 12:08:43 < contract 12:11:20 < code 12:57/12:58 <
  experiment_results 13:00:35. ORDER OK.
- Research gate envelope re-read from the brief itself (not the contract's
  summary): `brief_status "COMPLETE"`, `external_sources_read_in_full 15`,
  `urls_collected 30`, `recency_scan_performed true`, `gate_passed true`,
  `coverage` present. 38,584 chars.
- Evidence CHANGED since cycle 2 (a2ac7cca: autonomous_loop +38, test +60,
  live_check 110/-97). NOT verdict-shopping.
- log-last: `grep -cF "phase=86.88" handoff/harness_log.md` = 0; masterplan
  86.88 status = `pending`. Clean.
- Scope: `git diff --name-only 22dd1fc3..HEAD` = autonomous_loop.py, the test
  file, scripts/qa/verify_lite_risk_seam_86_86.py, 5 handoff artifacts,
  CHANGELOG. NO unintended production change inside the step's commits.

## B. Deterministic
- IMMUTABLE CMD -> **exit 0**, "checks emitted: 9 (PASS 9 / FAIL 0) RESULT: OK".
  Reproduces live_check §1 line-for-line (sites 2452/2454/2462/2463 +
  3243/3248/3496/3501).
- `pytest test_phase_66_2_risk_judge_shape.py -q` -> **77 passed**. Matches.
- ruff F821,F401,F811 over a GIT-DERIVED scope (committed 22dd1fc3..HEAD union
  uncommitted, via xargs) -> "All checks passed!", exit 0.
- `import backend.services.autonomous_loop` OK; `/api/health` 200
  (version 6.93.222 -- the RUNNING process still holds the PRE-fix module;
  restart is batched to session end per CLAUDE.md, not a Q/A blocker).
- Matrix row sums: all 12 rows sum failed+passed = 77 = the SHIPPED tree.
- 1b/1c NOT BINDING: no `frontend/**` in any 86.88 commit; no UI claims in the
  contract, criteria or diff. No Playwright capture taken, correctly.

## judge_these A -- PROVENANCE IS REAL THIS TIME (re-derived independently)
Drove the real `_run_claude_analysis` -> real `_persist_analysis`, with a stub
reproducing the REAL writer's serialization verbatim (`bigquery_client.py:269`
`"full_report_json": json.dumps(full_report)`; no key filter between :62 and
:269):
```
key in PERSISTED-COLUMN json: True (failed) / True (real3)
full_report_json sha256: 1273c06b63bcd61c (failed) vs b259205ffa25cea1 (real3)
DIFFER: True
provenance: {'judge_verdict_absent': True} vs {'judge_verdict_absent': False}
recommended_position_pct column: 3.0 / 3.0   (criterion 7 holds)
```
Same re-derivation on the GEMINI route: key present, blobs DIFFER
(189f53d05ef0 vs 123dedd51b4a), pct 3.0/3.0. **CONFIRMED on both paths.**
Residual bound: the BQ network insert itself is not exercised; the writer's
transformation was read from source and reproduced.

## judge_these B -- consumer risk: LOW, verified
- Column type (`backend/db/_schema_snapshot.json`):
  `financial_reports.analysis_results/full_report_json {"mode":"NULLABLE",
  "type":"JSON"}`. A JSON column has no sub-schema; an added key changes no column.
- Consumer census (tests excluded): bigquery_client 288 `JSON_VALUE($._path)`,
  320/321/376/377 `json.loads`; api/reports.py:130 `.get("cost_summary")`;
  outcome_tracker.py:126-129; autonomous_loop.py:3849; frontend
  reports/page.tsx:180 + performance/page.tsx:44 (`?.final_synthesis ??` then
  NAMED-key reads only). No consumer enumerates keys; no pydantic/TS runtime
  validation. ADDITIVE IS SAFE.

## judge_these C -- the regeneration script is NOT IN THE TREE
`find . -name "*86_88*" -o -name "*86.88*"` returns only handoff artifacts and
memory files; a2ac7cca touches 5 files, none a script. The claim "a script that
asserts the bytes changed AND that every placeholder was substituted" is
UNAUDITABLE, so the question as asked (would it catch an EMPTY substitution?)
cannot be answered. I graded the ARTIFACT instead: live_check WAS genuinely
regenerated (110 insertions / 97 deletions, 5,897 -> 7,374 bytes), §1 and §2
reproduce against my own runs, and no empty placeholder remains.

## judge_these D + INDEPENDENT MUTATION -- *** TWO SURVIVORS ***
Harness: mutated source compiled and exec'd into a module injected via
`sys.modules`, then `pytest.main` in-process. ZERO repo writes; tree sha256
c68ebad5c45f281a unchanged throughout.
CONTROL (unmutated, injected the SAME way): rc=0, **77 passed** -- harness live.
PC (revert whole-default detection): rc=1, 4 failed -- harness DISCRIMINATES.

Author cells reproduced (12/12 KILLED confirmed):
```
M1 2f/75p EXACT   M2 1f/76p EXACT   M3 3f/74p EXACT   M4 4f/73p EXACT
M8 4f/73p EXACT   M11 1f/76p EXACT  M12 1f/76p EXACT
M5,M6,M7 KILLED (my construction differs -> different counts; both forms kill)
M9 KILLED 2f (author 3f), M10 KILLED 3f (author 4f) -- off-by-one fully
   explained: the author mutated the HELPER, which additionally fails
   test_the_equality_is_EXACT_not_a_subset_match; I mutated the producer's call.
M11 killed BY test_the_equality_is_EXACT_not_a_subset_match -- right reason.
M9/M10 killed by the record + persisted-payload tests -- right reason.
```

MY OWN cells (the author did not choose these):
```
IND1 drop provenance from CLAUDE persisted blob ONLY   KILLED  1 failed
IND2 drop provenance from GEMINI persisted blob ONLY   *** SURVIVED *** 77 passed
IND3 persisted provenance pinned False on BOTH         KILLED  1 failed
IND4 persisted provenance pinned False, GEMINI ONLY    *** SURVIVED *** 77 passed
IND5 subset match ignoring 'decision'                  *** SURVIVED *** 77 passed
IND6 subset match ignoring 'risk_level'                *** SURVIVED *** 77 passed
IND7 subset match ignoring 'recommended_position_pct'  *** SURVIVED *** 77 passed
IND8 subset match ignoring 'risk_limits'               *** SURVIVED *** 77 passed
IND9 superset-tolerant (extra keys ignored)            *** SURVIVED *** 77 passed
```
Every persistence kill is by the SAME single test
(`test_judge_failure_is_distinguishable_IN_THE_PERSISTED_PAYLOAD`), which drives
`self._drive` = the CLAUDE route only. **The Gemini persisted provenance has
ZERO coverage.**

BEHAVIOURAL DIFFERENTIAL (not equivalent mutants): at baseline the Gemini
judge-failed and judge-said-3% blobs DIFFER (189f53d05ef0 vs 123dedd51b4a).
Under IND2/IND4 they become byte-identical again -- the exact defect the step
closes, on one of the two production paths. This is the same shape the author
found and fixed for the in-memory key in cycle 2 ("with only the Claude route
driven, the identical N1 pre-mangle at the GEMINI producer call still SURVIVED").
Cell M12 drops the key from BOTH paths, which is the weaker cell and cannot
detect a single-path regression.

The mechanism the artifact credits: "Threaded into the lite `full_report` on
BOTH paths, with the literal count ASSERTED == 2 rather than assumed."
`grep -rn "risk_assessment_provenance" backend/ scripts/` returns EXACTLY 2 hits,
both PRODUCTION lines (autonomous_loop.py:3302, :3544). No test, no checker
assertion. The checker's `len(n_calls) == 2` (line 298) pins
`_build_lite_risk_assessment` CALL SITES -- a pre-existing 86.86 check, and the
checker was not modified by a2ac7cca. Read as "I measured it once" the sentence
is TRUE; read as a shipped guard it is not, and it is what left the Gemini half
unprotected.

IND5-IND9 reachability, stated rather than implied: all require a judge whose
other four fields equal the defaults EXACTLY, including the verbatim string
"risk-judge parse failed; falling back to conservative default sizing". Low
reachability -> WARN, not blocking. But M11 pins the CLASS from a single
INSTANCE ('reasoning'); the other four keys and the superset direction are
uncovered.

## Criteria 4/5/6/7/8 -- independently re-derived
- C4: immutable command shows the branch firing on 4 REAL matches (3243/3248/
  3496/3501); "made LIVE by widening", reason stated. MET.
- C5: enumeration reproduces; the fix is inside `_lite_position_pct` (the seam),
  not at the four call sites; "judge FAILURE as SIZE 3.0 is NOT acceptable"
  stated. MET.
- C6: four route tests, each asserting `calls["n"] == 2`. Pre-fix claim checked:
  the pre-step copy of this test file contains 0 references to
  `_run_*_analysis`. Other files reference them only via identity asserts,
  `monkeypatch.setattr` REPLACEMENT, and `inspect.getsource` -- none drive the
  bodies. Verified by running N1 against those 5 files: no real new failure.
  MET.
- C7: PRE (22dd1fc3) vs POST resolved numbers over all 7 disclosure-table
  inputs -> IDENTICAL, "moved: NONE"; key delta added ['judge_verdict_absent'],
  removed [], shared keys with changed values NONE. Real `decide_trades` under
  BOTH `paper_risk_judge_reject_binding` states: 1 order each, PRE==POST
  identical (BUY TST 300.0). `portfolio_manager` never reads the new key. MET.
- C8: no .env/settings/config/flag file in scope; no gate constant changed;
  the immutable command went RED mid-work and was answered by classifying. MET.
- Checker recall (live_check §6 completeness claim): 7/7 of the claimed shapes
  SEEN; prose/comment + unrelated-dict negative controls invisible; the stated
  alias residual reproduces as UNSEEN. Extra unnamed blind spots I found:
  `X | {}`, `{} | X`, `copy.deepcopy(x=X)`, dict-comprehension. All are caught
  by the RUNTIME value-equality guard, so only the tripwire is weaker -- exactly
  as the artifact states. NOTE, not a violation.

## PROBE HONESTY
One "new failure" under my injection
(`test_60_4_no_naked_yfinance_in_async_analyzers`) is a HARNESS ARTIFACT: it
uses `inspect.getsource`, which resolves line numbers against the on-disk file
while my module is compiled from a different string. It passes in an
uninjected run and fails identically for the unmutated PRE-STEP module. NOT
reported as a finding.

## Out-of-scope observation
`git status` carries UNCOMMITTED, UNRELATED production changes:
`backend/api/sovereign_api.py` (+ a `1y` red-line window) and 5 frontend
components. They are in none of 86.88's commits. `auto-commit-and-push.sh` runs
`git add -A`, so flipping 86.88 to `done` will sweep them into 86.88's commit,
and those 5 frontend files have not passed a frontend lint/typecheck gate
(1b does not bind for 86.88's own diff).

COMPLETED: 2026-08-16T11:13:52Z
