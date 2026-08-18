STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-16T21:52:51Z

# Q/A write-first record -- step 86.94 (sliding-window / bare-date measurement drift)

Spawn: Workflow rail, attempt reported by Main as 1 (advisory only; to be
cross-checked with qa_wip.py + verdict_history_86_21.py --evidence-only).

## Plan
A. Harness compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, ruff, syntax, tests
C. Criterion-by-criterion judgment + independent mutation testing of the new guard

## Findings log (appended as established)

### Prior-attempt / verdict evidence
- qa_wip.py 86.94 --spawned-at 2026-08-16T21:52:51Z: attempt_number=1,
  prior_attempts=0, source_present=true, attempt_number_status=ok,
  identity_checked=true, prior_records=[].
- verdict_history_86_21.py --step 86.94 --evidence-only: status=no_rows_for_step,
  verdicts=(none). No staleness signal (attempt_number 1 vs ledger 0 prior).
- sequence: KNOWN (empty) -- first graded attempt on this step-id.

### B. DETERMINISTIC
- Immutable command `verify_changelog_flip_86_91.py > /dev/null && echo green`
  -> "green", exit=0. REPRODUCED.
- verify_no_sliding_windows_86_94.py -> "ALL GREEN: 30 passed, 0 failed", exit 0.
- ruff F821,F401,F811 on the 3 touched .py -> All checks passed! exit=0.
- git diff --name-only HEAD -- '*.py' -> backend/api/sovereign_api.py ONLY.
  That file (+5 frontend files) mtime 2026-08-14 13:2x, two days BEFORE this
  step's first commit (f1b02a36 2026-08-16 23:33). PRE-EXISTING, unrelated
  (adds a "1y" window option to the sovereign UI). NOT an unintended change
  from this step.
- 86.94 commits: f1b02a36, 757c58ad, 4f2bba7f, b3df71f6 (+ auto-changelog).
  Production files touched: replay_changelog_rule_86_68.py (1 char),
  verify_changelog_flip_86_91.py (comment only), NEW
  verify_no_sliding_windows_86_94.py. Scope matches the contract.

### INDEPENDENT RE-DERIVATION -- every headline number REPRODUCES
- TZ measurement (mine, fresh):
    Europe/Oslo naive=707 Z=707 | Asia/Seoul naive=787 Z=707
    America/New_York naive=707 Z=707 | UTC naive=707 Z=707
  EXACT match to live_check section B. The TZ finding is REAL.
- Band explanation reproduces: [08-10T22:00Z,08-11T04:00Z)=0 commits (Oslo/NY),
  [08-10T15:00Z,08-11T00:00Z)=80 commits (Seoul). The 80-spread is explained.
- Drift arithmetic reproduces: `git rev-list --count a5cbfd67..27f8c6f6` = 4;
  08-13 commits in [22:50:20,23:51:09) reachable from a5cbfd67 = 20.
  376+4-20=360 and 424+4=428 both land, as claimed.
- git resolver reproduces on MY run: `git rev-parse --since=today` =
  --max-age=1786917320 == `date -u +%s` = 1786917320 (NOW, not midnight);
  `--since=2026-08-13` = 1786658120 = now-259200 = exactly now minus 3 days,
  i.e. today's clock time carried onto 08-13.
  `--since=today` -> 0 commits vs `--since=<today>T00:00:00` -> 72.
- THIRD independent measurement (mine, 23:55:07 CEST, HEAD 05b516d7):
  bare 08-13 = 362, pinned 08-13 = 432. vs M2 (23:51:09) 360/428, with 4
  commits added: pinned +4 (correct), bare +2 (2 of the 4 offset by slide).
  The drift reproduces a third time under an independent observer.

### A. HARNESS COMPLIANCE -- all 5 CLEAN
1. research gate: research_brief_86.94.md brief_status=COMPLETE, gate_passed=true,
   sources=17 (floor 5), urls=45 (floor 10), recency_scan=true, coverage.dry=true
   (audit_class). Contract cites run wf_2c05296c-5d4 and uses the findings.
2. order (local mtime): brief 23:07:21 < contract 23:10:18 < guard 23:34:54 <
   experiment_results 23:52:16. Criterion-1 Measurement 1 is quoted IN the contract.
3. experiment_results_86.94.md present (9,523 B) + live_check_86.94.md (15,842 B).
4. log-last respected: `grep -c 'phase=86.94' handoff/harness_log.md` = 0;
   masterplan 86.94 status = "pending"; masterplan.json byte-unchanged since
   a5cbfd67 (`git diff --stat a5cbfd67..HEAD -- .claude/masterplan.json` empty).
5. not verdict-shopping: attempt 1, no prior verdict.
RAIL R5 respected: `git log a5cbfd67..HEAD -- .claude/agents/qa.md
.claude/workflows/qa-verdict.js .claude/workflows/research-gate.js` -> EMPTY.

### MY OWN MUTATION MATRIX (in-memory: pathlib.Path.read_text patched, repo
### untouched, nothing written, nothing committed; CONTROL OBSERVED GREEN FIRST)
CONTROL (no injection)                                   exit=0  ALL GREEN 30/0
M1 --since=<bare date> into scripts/harness/run_harness.py   exit=1  KILLED
M2 --since=<TZ-naive pin> into run_harness.py                exit=1  KILLED
M7 revert the Z: CORPUS_SINCE -> 2026-08-11T00:00:00         exit=1  KILLED (3 fails)
M8 revert to bare date: CORPUS_SINCE -> 2026-08-11           exit=1  KILLED
M9 same, indirection still resolvable                        exit=1  KILLED
M10 literal made unresolvable (indented out of module scope)  exit=1  KILLED (fails closed)
--> the guard is NOT vacuous: it is behavioural end-to-end and it protects its
    own one-character fix. Credit where due.

SURVIVORS (exit 0, guard stayed ALL GREEN 30/0):
S1 --after=<bare date>            SURVIVED. EXACT git synonym: verified
   `git rev-parse --after=2026-08-13` -> --max-age=1786658210, identical shape to
   --since; `git log --after=2026-08-13` = 362 = `--since=2026-08-13`.
   NAMED VERBATIM in the step's own audit_basis.
S2 --before=<bare date>           SURVIVED (`--before` -> --min-age). Also named
   verbatim in the audit_basis.
S3 `--since <space> 2026-08-11`   SURVIVED. WINDOW_RE has `[=\s]` so the LINE
   matches, but VALUE_RE requires `=`, so raw=="" and scan_text `continue`s --
   a fail-OPEN inside the module whose central claim is that it fails closed.
S4 `(datetime.utcnow() - timedelta(days=30))`  SURVIVED. Criterion 6's text is
   "a new bare-date OR NOW-RELATIVE window"; audit_basis names timedelta(days=N)
   and BigQuery CURRENT_DATE() explicitly. 23 tracked scripts carry such windows
   (tca_report.py `now - timedelta(days=7)`, funnel_report.py `today - 7d`,
   metered_spend.py `CURRENT_DATE()`); all three are named in masterplan.json.
S5 bare date inside an EXECUTED triple-quoted command string
   (`CMD = """git log --since=2026-08-11"""; subprocess.run(CMD, shell=True)`)
   SURVIVED -- strip_docstrings blanks it. A false negative OPENED by the H2
   docstring-stripper fix. (Control: the same window as plain code is FLAGGED.)

### CRITERION 5 -- DERIVED SWEEP, not Main's list
`git grep -ln "707 / 251 / 9 / 11"` -> 7 files. Corrected: 2
(experiment_results_86.91.md, live_check_86.91.md). UNCORRECTED, verified
per-occurrence (0 TZ words within 4 lines of the figure):
  - handoff/current/day_report_2026-08-16.md:67
  - handoff/current/escalation_86.90_86.91.md:46
  - handoff/harness_log.md:35557 ("707 / 251 / 9 / 11 over [2026-08-11T00:00:00 .. 8dc70502]")
  - handoff/current/evaluator_critique_86.91.md (x4)
experiment_results_86.94.md says "Both 86.91 artifacts asserting that claim now
carry the bound" and that the enumeration "was driven by the CLAIM ... not by my
own phrasing". That does not reproduce: the claim-driven grep returns 7 carriers.

### CRITERION 4 -- the stated judgement is FALSE
ALLOWLIST entry for frontend_route_inventory.py asserts: "outside this step's own
artifacts the name appears nowhere in the masterplan, CHANGELOG or handoff tree".
The executing code scans QUOTE_DIRS = ["handoff/current"] only (line 371) --
handoff/archive/** and handoff/harness_log.md are OUT of corpus.
MEASURED: `grep -rl frontend_route_inventory handoff/` = 55 files, incl.
handoff/harness_log.md:3855. And handoff/archive/_quarantine_2026-04-21/
phase-3.7.5-v22/experiment_results.md QUOTES figures derived from that exact
`--since=30.days` window AS EVIDENCE: "usage_source: git_activity_30d",
"/portfolio 2 /login 1", "No route has opens_30d=0 in this window",
"| every_route_has_usage_count | PASS (12/12 integer opens_30d) |", and
"/backtest 47 vs /login 1 ... enough to unblock step 4.7.1 decision-making".
So the "may be left / no count is load-bearing" exemption rests on a false premise.

### EVIDENCE INTEGRITY -- live_check_86.94.md "verbatim" blocks do not reproduce
The file opens "Every block is verbatim tool output from this session."
  - Sec.G: "verify_no_sliding_windows_86_94.py ALL GREEN: 24 passed" vs measured 30;
    Sec.H in the SAME file says "24 -> 30 assertions". Internal contradiction.
    Commit b3df71f6 ("correct the stale assertion counts") touched
    experiment_results_86.94.md ONLY -- live_check was left stale.
  - Sec.C enumeration lists frontend_route_inventory.py:70 AND :73; measured
    output lists :73 only (line 70 is a docstring, correctly stripped since
    757c58ad). Sec.E table repeats ":70,73".
  - Sec.C lists replay:114 '{CORPUS_SINCE}' -> ALLOWED; measured -> REPRODUCIBLE
    (and the replay has no allowlist entry any more).

### NOTE (non-blocking)
contract_86.94.md H2 labels the 766/846 pair "both ends pinned". Measured, that
pair is the OPEN-ENDED whole-history count (now 774 Oslo / 854 Seoul); the
both-ends-pinned pair is 707/787. The guard's own docstring (lines 23-25) states
both correctly and labels them correctly; the contract does not.

### VERDICT REASONING
C1 MET (excellent), C2 MET for the rule as written, C3 MET (hard gate verified),
C7 MET. C4 NOT MET (stated judgement measurably false). C5 NOT MET (2 of 6
carriers). C6 NOT MET (5 executed survivors; 3 of the shapes named verbatim in
the step's own audit_basis). -> FAIL.

COMPLETED: 2026-08-16T22:02:03Z

