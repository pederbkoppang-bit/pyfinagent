STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.81
WRITTEN: 2026-08-14T11:37:45Z

# Q/A write-first record -- step 86.81 (rail retry proof)

## Prior-attempt / sequence evidence
- `qa_wip.py 86.81 --spawned-at 2026-08-14T11:37:45Z`: `source_present: true`,
  `attempt_number: 1`, `attempt_number_status: "ok"`, `prior_attempts: 0`,
  `prior_records: []`, `records_retained: 1` (gauge, includes this record).
- `verdict_history_86_21.py --step 86.81 --evidence-only`: `status: no_rows_for_step`,
  `verdicts: (none)`. Cross-check: attempt_number (1) > ledger count (0 rows) --
  but with prior_attempts=0 there is no missing verdict to explain. First spawn.

## A. Harness compliance (5/5 clean)
1. Research gate: research_brief_86.81.md 11:21:22Z; envelope `brief_status COMPLETE`,
   `external_sources_read_in_full: 6` (>=5), `urls_collected: 30` (>=10),
   `recency_scan_performed: true`, audit_class false. Contract cites it 3x.
2. Contract-before-generate: contract 11:25:47Z < rail_drop_rate.py 11:26:57Z <
   verify_rail_retry.mjs 11:29:18Z < mutation_matrix 11:30:14Z. OK.
3. experiment_results_86.81.md 11:37:04Z + live_check_86.81.md 11:36:05Z present.
4. Log-last: `grep -cF "phase=86.81" handoff/harness_log.md` = 0; masterplan
   status = "pending". Not yet logged, not yet flipped. OK.
5. No verdict-shopping: first spawn (prior_attempts 0).

## B. Deterministic
- `node scripts/qa/verify_rail_retry.mjs` -> **exit 0, ALL GREEN: 38 passed, 0 failed**.
- Lint 1a, scope DERIVED (`git diff --name-only HEAD -- '*.py'` + untracked .py):
  backend/api/sovereign_api.py, scripts/qa/rail_drop_rate.py -- non-empty; ruff
  F821,F401,F811 -> `All checks passed!` exit 0. (xargs -0, not an unquoted var.)
- ast.parse(rail_drop_rate.py) OK. node --check green on all 3 new .mjs.
- `verify_research_gate_workflow.mjs` -> exit 0 (independently run).
- frontend `npx tsc --noEmit` -> exit 0.

## C. Vacuity attack on section A -- NOT vacuous (mutations I ran myself)
Seam: `QA_WF = process.env.PYFIN_QA_VERDICT_OVERRIDE || .claude/workflows/qa-verdict.js`;
the checker readFileSync's the shipped file, brace-matches `agentRetryingDrops`, and
writes a temp module that re-exports the REAL body. No hand-copy.
- CONTROL via override (scratchpad byte-copy, md5 equal): ALL GREEN 38/0 -> seam live.
- M1 delete `if (!msg.includes('without calling StructuredOutput')) throw e`:
  RED, exit 1 -- A3b (`calls=2`), A3c (logs show 2 DROP lines), A4 (every non-drop
  shape retried). 35 passed, 3 failed.
- M2 `maxAttempts = 2` -> `1`: RED, exit 1 -- "maxAttempts default is READ OFF the
  shipped source" (`parsed maxAttempts=1`), A1 (threw), A1b (`calls=1`). 35/3.
- Repo file md5 `c7d1953d44e16becc6baa22b40a594cd` identical before and after.
- NOTE: that first check pins the value to 2 as well as parsing it; the label says
  only "read off". Defensible (a retry budget should not silently widen) but the
  label under-describes it. NOTE-level, not a defect.

## D. C3 live drive -- verified from the run record, not from the quote
`~/.claude/projects/.../workflows/wf_9f387ad8-b5c.json`:
- `logs: ['qa-verdict: StructuredOutput DROP on attempt 1/2 -- retrying']`
- `agentCount: 2`, `error: None`, `result.recovered: true`,
  `result.result.attempt_seen: 'SECOND'`, `marker_value_read: 'SECOND'`
  (independent on-disk marker channel, not the log line).
- `retry_span_sha256: 1366d49acf843666f8cac718d289c6b4303f55700c54219ce312b5de75bde974`
  -- I recomputed sha256 over the CURRENT shipped `agentRetryingDrops` span
  (705 chars): **byte-identical**. The live drive exercised the shipped code.
- Invalid first attempt `wf_ba771f51-1f7`: `agentCount: 1`, `logs: []` -- real, and
  honestly disclosed in the artifacts (not buried).

## E. C5 -- contamination reproduces on the REAL population, not only the fixture
Extracted the pre-fix reader (`git show HEAD:scripts/qa/rail_drop_rate.py`) and ran it:
- PRE-FIX : `2026-08-14 ... retried=5`; `on/after runs=22 exhausted=2 retried=5`
  (DATE split)
- CORRECTED: `2026-08-14 ... retried=1`; `on/after runs=5 exhausted=0 retried=1`
  (launch-instant split, 2026-08-14T10:15:17Z, commit 6b4df8f9)
5 phantom retries vs 1 real, on real records. E7 (logs-only) / E0,E2 (all three log
literals) / E8 (error-field-only) green and non-vacuous per section C.

## F. C6 -- disclosed deviation, judged adequate
Criterion's frozen parenthetical ("0 post-fix runs where it currently shows 18") is
time-dependent and structurally unsatisfiable -- this step's own research-gate runs
launched after the fix. Main disclosed it first-person at experiment_results:107-113,
owned it, and did NOT edit the criterion. Substance IS demonstrated (error-field-only
classification; launch-instant split). NOTE: the disclosed "3 post-fix runs" is already
stale -- I measure 5; "18" is now 22. Monotone drift, not a false claim at write time.
The meaningful reading of "0" -- zero post-fix EXHAUSTED runs -- is true (exhausted=0).

## G. C7 -- sweep re-run
Tracked tree: single surviving hit `.claude/workflows/qa-verdict.js:394`, which is the
retraction notice itself (names the figures to FORBID them, points at the re-runnable
reader). It REPLACES rather than accompanies: the CLAUDE.md/research-gate.js diffs show
the old text deleted. Keeping it is correct.
My wider sweep (incl. ignored files) found `.claude/workflows/qa-verdict.js.export.mjs:357`
still carrying `21.8%` ON DISK -- Main found this too, disclosed it as a real defect of
commit f237bb8d ("committed the very file it set out to exclude"), ran `git rm --cached`
(now gitignored via .gitignore:106, untracked), and queued the on-disk deletion as an
operator call because it is a concurrent session's working file. Residual risk stated,
not hidden: the file still sits in the dispatch dir carrying `name: 'qa-verdict'`.
Remaining grep hits are substring false positives (`84.8%`, `-14.8%`, `21.81s`) or this
step's own artifacts.

## H. C2 -- both stage loops genuinely driven
- Stage 2: driven directly (verify_rail_retry section C, C1-C4).
- Stage 1: delegated, and the delegation is EXECUTED (D1 runs
  verify_research_gate_workflow.mjs; I ran it independently -> exit 0). I checked that
  checker's stage-1 cell is BEHAVIOURAL, not a scan:
  `:524-536` calls `drive({...}, ..., dropsOnceThenSucceeds)` and asserts
  `researcher_calls === 2`, `!recovered.rail_dropped`, `gate_passed === true`.
  Section header states the delegation plainly ("[D] stage 1 -- coverage ASSERTED
  from the checker that already owns it"). D2's literal-match is redundancy beside a
  real driven guard, so NOT sole-coverage vacuity. I did not file this as a finding.

## I. C8 / C9
- C8: CLAUDE.md diff REMOVES the old contrary sentence and adds the measured
  scriptPath-vs-name finding (three named dispatches byte-identical 18,321 chars from
  a commit 8h36m earlier; scriptPath A/B took 22,961). It also correctly separates
  SCRIPT snapshot from AGENT-DEFINITION snapshot (qa.md additions live, deletions need
  restart). Correction replaces, does not accompany.
- C9: F1-F4 green in a section proven live by M1/M2. Exhaustion rethrows the ORIGINAL
  error and yields no value; research-gate recomputes gate_passed via enforceGate after
  the loop; the retry loop assigns no verdict/gate field.

## J. THE ONE OPERATIONAL FINDING -- out-of-scope tree contamination (NOT a criterion miss)
`git diff HEAD` carries 6 files unrelated to 86.81 and claimed by NO 86.81 artifact:
backend/api/sovereign_api.py, frontend/src/app/page.tsx, HomeQuickActionsPanel.tsx,
LatestTransactionsBox.tsx, RecentReportsTable.tsx, RedLineMonitor.tsx (adds a `"1y"`
RedLineWindow). A concurrent session is ACTIVE: `before-1y-click.md` existed at repo
root in my first `git status` and was gone by a later one.
At the flip, `auto-commit-and-push.sh` runs `git add -A` and will ship all of it under
86.81's subject. Main must commit 86.81 with an explicit PATHSPEC. tsc exit 0, so the
peer's work at least typechecks. Not attributable to 86.81's author; no criterion covers
it; recorded as a flagged non-blocking condition rather than a verdict degradation.

## Anti-rubber-stamp check on the step's headline claim
"The retry was never exercised before and is now proven." Both halves hold: E-section
and the reader show `retried=0` across 564 pre-fix runs on the corrected (logs-only)
reader, and the sha-matched live run is the first observed recovery. I could not
falsify either half.

COMPLETED: 2026-08-14T11:44:34Z
