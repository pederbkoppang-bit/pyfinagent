STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-17T08:10:29Z

# Q/A write-first record -- step 86.94 (cycle 6, HEAD dc8d64d9 claimed)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, scoped tests
C. Independent mutation matrix M-A..M-J (control GREEN first)
D. Main's 5 attack questions + criterion-by-criterion MET/NOT MET

## Log
- 08:10:29Z started. qa.md read in full.
- IMMUTABLE CMD: `source .venv/bin/activate && python scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green` -> printed `green`, EXIT=0.
  DISCLOSED BY MAIN and CONFIRMED: this runs the 86.91 checker and cannot fail on 86.94's class.
- qa_wip.py 86.94 --spawned-at 2026-08-17T08:10:29Z -> attempt_number=6, prior_attempts=5,
  source_present=true, attempt_number_status=ok, attempt_number_is_lower_bound=true,
  records_retained=6 (GAUGE, not a counter), records_pruned_known=null.
- verdict_history_86_21.py --step 86.94 --evidence-only -> status=no_rows_for_step, verdicts=(none).
  CROSS-CHECK: attempt_number(6) > ledger count(0) => LEDGER IS STALE. sequence: UNKNOWN from the
  authoritative source. Main's ADVISORY disclosure (FAIL,FAIL,CONDITIONAL,CONDITIONAL,FAIL) is
  consistent in CARDINALITY with prior_attempts=5 but is not independently corroborated.
- HEAD = fac590c9 (auto-changelog for dc8d64d9). Work commit dc8d64d9 = cycle 6. Evidence CHANGED
  vs cycle 5 (fca21bc6): 381 insertions / 95 deletions across 6 files => NOT verdict-shopping.
- GUARD CONTROL: `python scripts/qa/verify_no_sliding_windows_86_94.py` -> "ALL GREEN: 77 passed, 0 failed".
  Reproduced independently. Matches Main's claim.

## CRITERION 1 -- INDEPENDENTLY RE-DERIVED, ALL 10 FIGURES REPRODUCE
Replayed both measurements at their pinned capture instants against the two recorded HEADs
(a5cbfd67 / 27f8c6f6, both present in the object store):
  bare@M1 2026-08-13T22:50:20+02:00 = 376 (says 376) | bare@M2 23:51:09 = 360 (says 360)
  pin@M1  2026-08-13T00:00:00+02:00 = 424 (says 424) | pin@M2                = 428 (says 428)
  bare0811@M1=434 (434) M2=438 (438) | pin0811@M1=766 (766) M2=770 (770)
  H1..H2 commits = 4 (says 4) | slid out of the 08-13 window 22:50:20->23:51:09 = 20 (says 20)
  Gap = 1h00m49s >= 1h. Bare DIFFERS (376 vs 360); pinned (424/428) DIFFERS FROM BOTH.
  Arithmetic 376+4-20=360 closes with no residual. => CRITERION 1 MET.

## SECTION E -- Main's attack question (2): ANSWERED YES
Exactly ONE fenced block in section E. Diffed against a live run's [3b] section: reproduces
line-for-line, including 'NAMED in 281 tracked file(s); a FIGURE it produced is QUOTED in 1'.

## FINDING A (REAL, REPRODUCIBLE) -- live_check_86.94.md:354 carries an UNCORRECTED stale figure
Section G "NO REGRESSION (criterion 7)" fenced block reads verbatim:
    verify_no_sliding_windows_86_94.py   ALL GREEN: 45 passed, 0 failed
MEASURED NOW: 77 passed. The other two lines in that same block DO reproduce
(verify_changelog_flip_86_91.py 42/0 confirmed; verify_workflow_args_boundary.mjs 96/0 confirmed),
so the block is not a dated snapshot -- it is 2 live lines + 1 stale line.
PROVENANCE: `git log -S "ALL GREEN: 45 passed, 0 failed" -- handoff/current/live_check_86.94.md`
returns exactly ONE commit, d6c732b7, subject "phase-86.94: cycle-3 -- my correction accompanied
instead of replacing". Never revised across cycles 4, 5, 6 while the count went 45 -> 68 -> 74 -> 77.
AND THE SWEEP CLAIMS OTHERWISE: J5 Class B disposition row asserts
  "`handoff/current/live_check_86.94.md`, `experiment_results_86.94.md`, `contract_86.94.md`
   | current-cycle artifacts; each quotes the stale figure only to identify it."
Enumerated all 10 hits of `45/0|45 assertions|45 passed` in live_check_86.94.md: 9 are
descriptions/identifications (:533 :691 :694 :704 :739 :761 :762 :889 + :755 coincidental DOI);
:354 is an AFFIRMATIVE CLAIM. So the disposition does not reproduce.
This is the identical shape as section I4's own recorded defect ("the cycle-1 commit ... touched
experiment_results_86.94.md only and left the operator-facing gate artifact stale").

## FINDING B (REAL) -- section H1's tracked-file census invalidated BY THIS CYCLE'S OWN COMMIT
live_check §H1 fenced block: "tracked py/sh: 851   scanned: 850   excluded: 1".
MEASURED NOW: `git ls-files scripts .claude/hooks backend | grep -E '\.(py|sh)$' | wc -l` = 852,
so the live triple is 852/851/1. The +1 is scripts/qa/gen_shipped_today_fixture_86_94.py, added
by dc8d64d9 -- THIS cycle's own commit (`git log --diff-filter=A` confirms). Entered at 4f2bba7f
(cycle 2) and never regenerated. File header line 5-8 promises "Every block is verbatim tool
output from this session" and "No count in this file is quoted without the clock time and HEAD
it was taken at" -- neither holds for this block.

## SECTION F -- incomplete but NOT false (NOTE only)
All 11 lines of §F's mutation block are present verbatim in a live run; it shows 10 of the
live [4] section's 42 result lines. A truthful subset. The full matrix is in §J4.

## J5 ENUMERATION COMMANDS -- BOTH REPRODUCE
`git grep -l "mentions_reviewed" -- .` -> 8 files, member-for-member identical to the table.
`git grep -l -E "45/0|45 assertions" -- .` -> 11 files, member-for-member identical.

## LINT GATE (diff touches *.py)
Scope DERIVED from git (`git show --name-only --pretty=format: 88d7d84c fca21bc6 dc8d64d9 | sort -u`),
not hand-typed. 9 files touched; 2 are *.py:
  scripts/qa/gen_shipped_today_fixture_86_94.py, scripts/qa/verify_no_sliding_windows_86_94.py
Non-empty guard satisfied. `xargs -0 uvx ruff check --select F821,F401,F811` -> "All checks passed!" exit=0.
NO production code outside scripts/qa/ + handoff/ + masterplan was touched by the three commits.

## INDEPENDENT MUTATION MATRIX (in-memory exec; ZERO repo writes; control run FIRST every batch)
Method: read the guard source, string-mutate, `exec(compile(...))` with __file__ pinned to the real
path so REPO resolves, catch SystemExit. CONTROL rc=0 "ALL GREEN: 77 passed, 0 failed" in EVERY batch.

  M-A restore fail-OPEN continue      KILLED rc=1  69 passed,  8 failed   (Main claimed 69/8)  MATCH
  M-B VALUE_ARGV_RE never-matching    KILLED rc=1  75 passed,  2 failed   (claimed 75/2)       MATCH
  M-C drop WINDOW_RE argv alternative KILLED rc=1  67 passed, 10 failed   (claimed 67/10)      MATCH
  M-D frontend True->False            KILLED rc=1  76 passed,  1 failed   (claimed 76/1)       MATCH
  M-E scheduler True->False           KILLED rc=1  76 passed,  1 failed   (claimed 76/1)       MATCH
  M-F `if _r in _TRACKED:`->`if True:`KILLED rc=1  76 passed,  1 failed   (claimed 76/1)       MATCH
  M-G frontend probes never-matching  KILLED rc=1  75 passed,  2 failed   (claimed 75/2)       MATCH
  M-H scheduler probes never-matching KILLED rc=1  75 passed,  2 failed   (claimed 75/2)       MATCH
  M-I delete scheduler probe_fixtures KILLED rc=1  75 passed,  2 failed   (claimed 75/2)       MATCH
  M-J co-written probe+fixture, False KILLED rc=1  76 passed,  1 failed   (claimed 76/1)       MATCH
    -> killed=10 survived=0 unscorable=0. Main's matrix REPRODUCES CELL-FOR-CELL AND NUMBER-FOR-NUMBER.
    -> each kill is attributable: M-A/M-B/M-C name the value/visibility leg; M-J dies specifically on
       "every positive control is PROVENANCED", which is the cycle-6 addition.

## FINDING C (REAL, MY OWN CELLS) -- the provenance check IS circular; Main's residual #1 CONFIRMED
Two cells I built, both SURVIVED at a clean 77/0 with a manufactured criterion-4 claim:
  K1 fixture source = "scripts/qa/verify_no_sliding_windows_86_94.py" (THE GUARD ITSELF),
     probe r'UTC_QUALIFIED', fixture text "UTC_QUALIFIED", quoted_as_evidence False
     -> SURVIVED rc=0 ALL GREEN: 77 passed, 0 failed
  K2 fixture source = "scripts/qa/gen_shipped_today_fixture_86_94.py" (added by dc8d64d9, the
     SAME COMMIT as the provenance check), probe r'REFUSING TO WRITE', text "REFUSING TO WRITE",
     quoted_as_evidence False
     -> SURVIVED rc=0 ALL GREEN: 77 passed, 0 failed
Both tokens were ALREADY on disk, so the attack required writing no new text anywhere.
=> The guard's own comment (:776-778) "Each fixture now names a TRACKED FILE and the text must
   actually be in it, so a control cannot be invented" and live_check §J8 "A control can no longer
   be invented" are OVERCLAIMS, falsified by execution.
SEVERITY: WARN, not BLOCK. The provenance check is not sole coverage for criterion 4 -- the primary
binding `quoted_as_evidence == bool(_figs)` over the tracked corpus is, and today's three judgements
are all True and all backed by corpus hits I reproduced. But the ABSOLUTE prose claim is the same
overclaim shape cycle 5 was FAILED for, recurring one level up.
NAMED FIX: reject a fixture whose source is SELF_REL or the fixture generator; and assert the
generated render regenerates byte-identically (the generator already exists and does).

## J8 MISQUOTE WITHDRAWAL -- REPRODUCES
handoff/archive/misc/live_check_62.8.md:31 contains '"*Shipped today*" with 12 real commit lines'.
Its only "Steps closed" text is :36 = "Steps closed: 61.1, 62.0," -- confirming the cycle-5
"Steps closed: 6" was a regex truncation, not a quotation. Every surviving occurrence of the
string in the tree is inside a CORRECTION describing the withdrawal (guard :181/:185/:315,
experiment_results :348, live_check :800). No surviving affirmative use. Main's claim (2) HOLDS.

## MAIN'S ATTACK QUESTION (4) -- ANSWERED YES, the judgement rests on genuine window evidence
Traced the producer: scheduler.py:501-507 `_git_today()` runs
  ["git","log","--since-as-filter=midnight","--pretty=%h %s"] -> d["commits_today"]
rendered at formatters.py:101-109 through add() at :71-76 as "*Shipped today*\n- <sha> <subj>".
d["steps_flipped_today"] is a SEPARATE producer, _steps_closed_from_log() over harness_log.md at
:511-513 -- Main's finding (3) is correct.
K5 (my cell): delete the r'\d+\s+real commit lines' probe, keep only the render-shape probe ->
  KILLED rc=1 76/1, "claim says True but 0 quoted figure(s) were measured in 0 tracked file(s)".
So the whole scheduler True judgement rests on that ONE hit -- and that hit IS this window's output.
NOTE (precision, not capping): formatters.py:105 renders `commits[:12]`, so "12 real commit lines"
is min(N,12) -- the saturation point, uninformative about N. The allowlist calls it "a count of
exactly what _git_today() emitted"; "exactly" is imprecise. The True judgement is unaffected.

## MAIN'S ATTACK QUESTION (5) -- independent search for other quoted figures
Searched the tracked corpus for alternative phrasings of all three windows' figures
("Shipped today", "commits_today", "commits today", opens_30d/git_activity_30d/route usage,
"decision lines=", "recursion guard=", "gap=N"). Notable non-probe hits:
  .claude/masterplan.json:1085 -- verification command asserting `'opens_30d' in r`. A FIELD NAME
    in a criterion, not a figure; frontend_route_inventory is already True, so no judgement moves.
  handoff/current/research_brief_86.97.md:257 -- "28 of 56 commits today", a 86.97 decision-log
    measurement, not the Slack digest window.
No member judged False anywhere, so no additional hit can flip a judgement. No finding.

## CRITERION 2 -- INDEPENDENT SYMMETRIC-DIFFERENCE CHECK (qa.md 4b)
My own naive scan of all 852 tracked py/sh under scripts/ .claude/hooks/ backend/ for
`--(since|until|after|before)(-as-filter)?` -> 73 lines in 10 files. After removing docstrings/
comments and the 3 non-git argparse flags -- rail_drop_rate.py:184 (args.since filters rows, never
reaches git), census_qa_write_guard_log_86_31.py:64 (ns.before compares a ts field), congress.py:312
("kept for CLI shape parity; not yet used") -- the REAL git-window call sites are exactly:
  scheduler.py:503, frontend_route_inventory.py:73, replay_changelog_rule_86_68.py:114,
  verify_decision_log_86_97.py:360
SYMMETRIC DIFFERENCE against the guard's [2] output = EMPTY. The enumeration is complete.
Sections C ([2]) and D ([1]) reproduce BYTE-FOR-BYTE against a live run.

## CRITERION 3 -- MET, with the disclosed residual CONFIRMED by execution
[1] finds 06c3265f:72 '2026-08-11' -> SLIDING "bare date -- git applies the CURRENT TIME OF DAY".
K4 (my cell): KNOWN_MEMBER_PATH -> a file with no window  => KILLED rc=1 74/1. The gate has force.
K3 (my cell): KNOWN_MEMBER_REF -> HEAD (window ALREADY FIXED) => SURVIVED 77/0. The gate does NOT
  discriminate. MECHANISM, which I isolated: [1]'s inline loop calls classify(val) WITHOUT
  resolve(val, text), so at HEAD it reports '{CORPUS_SINCE}' as SLIDING ("indirection could not be
  resolved") while [2] resolves it to the Z-qualified literal and reports REPRODUCIBLE. [1] and [2]
  genuinely disagree. On the 06c3265f blob they agree (symmetric difference empty) because the
  value there is a literal. Main DISCLOSED this and queued it as 86.104 -- accurate and honest.

## CRITERION 7 -- MET
`git diff 88d7d84c~1..dc8d64d9 -- .claude/masterplan.json | grep '"status"'` -> ONE added line,
`+ "status": "pending"` (the new 86.104). Nothing flipped to done, nothing removed.
No evaluator_critique / verdict_ledger / harness_log file touched by the three commits.
86.94 masterplan status = pending, retry_count = 3.

## FIXTURE REGENERATION -- Main's claim HOLDS
md5 before = md5 after = 79bbdffe677f0151cf9b3aa107592413; `git diff` on the fixture empty.
DISCLOSURE: running the generator is a WRITE. I ran it to verify the byte-identical claim. Content
is unchanged (md5 identical, git diff clean); only the mtime moved. Stated rather than hidden.

## HARNESS COMPLIANCE -- CLEAN (5/5)
1 research-gate-before-contract: research_brief_86.94_cycle4.md envelope gate_passed=true,
  external_sources_read_in_full=7 (>=5), recency_scan_performed=true, urls_collected=23 (>=10).
  (research_brief_86.94.md, 46KB, is the cycle-1 brief.)
2 contract-before-generate (mtime): brief 09:06:38 < contract 09:09:15 < guard 10:03:05 <
  live_check 10:08:04 < experiment_results 10:09:27. ORDER CORRECT.
3 experiment_results_86.94.md present (23,332 bytes) and CURRENT (:287 quotes 77/0).
4 log-last: the only harness_log row for 86.94 is Cycle 230 (the overnight PARK, pre-cycle-4).
  No row for cycles 4/5/6. Masterplan not flipped. COMPLIANT.
5 no-verdict-shopping: evidence CHANGED between spawns -- dc8d64d9 is +381/-95 over 6 files
  vs cycle 5's fca21bc6. Documented cycle-2 flow, not a re-spawn on unchanged evidence.

## UNINTENDED PRODUCTION CHANGE -- NONE FROM THIS STEP
The three 86.94 commits touch only scripts/qa/**, handoff/current/**, .claude/masterplan.json.
The uncommitted working-tree edits to backend/api/sovereign_api.py and 5 frontend components are
PRE-EXISTING (present at session start; phase-10.5.0 "1y" red-line window + phase-16.45 Latest
Transactions follow-up dated 2026-08-14). Out of scope, not introduced here. No UI claim is made
by 86.94, so the live-UI capture gate (qa.md 1c) does not bind.

## VERDICT MAPPING
crit1 MET | crit2 MET | crit3 MET (residual disclosed + filed 86.104) | crit4 MET with a WARN
crit5 NOT MET (Findings A + B) | crit6 MET | crit7 MET
=> Cannot be PASS. FAIL on the criterion-5 miss: two figures inside blocks labelled verbatim do
not reproduce (qa.md 4b: "Prefer FAIL when a number in a 'verbatim' artifact does not reproduce"),
and the sweep's own disposition row makes a claim about one of them that is measurably false.
Not CONDITIONAL: criterion 5 is the criterion this step was FAILED on last cycle, the miss is in
the same artifact, and it is the third occurrence of the class inside this step (I4, E, now G/H1).

COMPLETED: 2026-08-17T08:22:38Z
