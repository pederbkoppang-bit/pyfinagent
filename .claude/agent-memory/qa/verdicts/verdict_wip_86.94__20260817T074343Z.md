STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-17T07:43:43Z

# Q/A write-first record -- step 86.94 (cycle 5 evidence at HEAD fca21bc6)

Role file read in full: .claude/agents/qa.md (runtime read, Workflow path).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command exit code; git status/diff scope; lint; scoped tests
C. Independent re-run of the 9-cell mutation matrix
D. Attack the 4 named hard points: scheduler probe recall; probe_fixtures tautology risk;
   J5 reproduction; the excluded `\*Shipped today\*` probe
E. Criterion-by-criterion MET / NOT MET

## Findings log (append-only)

### Prior-attempt / sequence evidence (gathered, NOT applied as a trigger)
- `qa_wip.py 86.94 --spawned-at 2026-08-17T07:43:43Z`: source_present=true,
  attempt_number=5 (status ok, is_lower_bound true), prior_attempts=4,
  records_retained=5 (gauge), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.94 --evidence-only`: status=`no_rows_for_step`,
  verdicts=(none). CROSS-CHECK: attempt_number(5) > ledger count(0) => LEDGER IS STALE.
  sequence: UNKNOWN from the authoritative source. Main's advisory (not authoritative):
  FAIL, FAIL, CONDITIONAL, CONDITIONAL.

### B. Deterministic
- HEAD is `7ae001f0` (auto-changelog for `fca21bc6`); the graded work commit is
  `fca21bc6` (2026-08-17T09:43:01+02:00). Prompt said HEAD fca21bc6 -- benign, the
  changelog companion is the only later commit. Re-checked at grade time.
- IMMUTABLE COMMAND: `source .venv/bin/activate && python
  scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green` -> prints
  `green`, EXIT=0. Main's disclosure that it cannot fail on this step's class is
  accurate and honest.
- `python scripts/qa/verify_no_sliding_windows_86_94.py` -> EXIT=0,
  `ALL GREEN: 74 passed, 0 failed`. Reproduces Main's claim exactly.
- Working tree: no uncommitted change to any 86.94 artifact, to scripts/qa/, or to
  .claude/masterplan.json. Dirty paths are unrelated (frontend components, audit
  jsonl, goal_next, agent-memory).

### D3. J5 REPRODUCES EXACTLY (attack 3)
- Class A `git grep -l "mentions_reviewed" -- .` -> 8 files, list byte-identical to J5.
- Class B `git grep -l -E "45/0|45 assertions" -- .` -> 11 files, byte-identical to J5.
- Coincidental=2 CONFIRMED: research_brief_81.0.md:81 + research_brief_85.3.md:185,
  both the DOI `10.2345/0899-8205-46.4.268` ("2345/0" contains "45/0").
- day_report_2026-08-17.md:49 correction is IN PLACE (strikethrough + 44/1 / 42/3),
  verified by `git diff 88d7d84c fca21bc6`.

### D4. The excluded bare `\*Shipped today\*` probe is LEGITIMATE (attack 4)
- `git grep -l 'Shipped today'` over tracked files = {.claude/masterplan.json,
  handoff/archive/misc/live_check_62.8.md, 86.94's own two artifacts (excluded)}.
  Adding the header probe changes NO file set and NO bool. Main's stated reason
  (header != figure) holds and is not number-narrowing. NOT a finding.

### D1. Scheduler probe recall -- no additional carrier found (attack 1)
- Independent hunt over the tracked away-digest corpus
  (`git grep -ln -E 'Away-mode report|away_digest|away digest' -- 'handoff/**.md'`,
  20 files) for count-shaped figures: only live_check_62.8.md:31 carries one.
- `commits_today` in handoff/archive/misc/research_brief_75.7.md:102 is a SOURCE
  citation, not a quoted figure -- correctly unmatched.
- Main's "exactly one file" survives my independent attempt to break it.

### F1 (NEW, mine). `.claude/masterplan.json` is now a SELF-CONTAMINATED corpus member
- Current run prints `scheduler.py: ... QUOTED in 2: .claude/masterplan.json,
  handoff/archive/misc/live_check_62.8.md`. The masterplan hit was introduced BY
  THIS CYCLE (`git log -S'12 real commit lines' -- .claude/masterplan.json` ->
  fca21bc6 only). SELF="86.94" excludes by PATH, and the masterplan path carries no
  step id, so 86.94's own note is inside its own corpus.
- Not judgement-changing here (62.8 supplies the hit independently), but for a
  member whose true judgement is False, writing a figure-shaped sentence into the
  masterplan note would flip bool(hits) and force the entry to state True.

### F2 (NEW, mine). `Steps closed: 6` is NOT a quotation, and the figure is NOT from this window
- Allowlist reason (scheduler entry) asserts live_check_62.8.md:31 "quotes ...
  'Steps closed: 6' -- counts of exactly what _git_today() emitted through this
  window". MEASURED:
  * `grep -n 'Steps closed' handoff/archive/misc/live_check_62.8.md` -> ONE hit, at
    line **36** (not :31), reading `"Steps closed: 61.1, 62.0, 17.4, 62.3"` -- a LIST
    OF STEP IDS, not the count "6". `'Steps closed: 6'` is the regex's truncated
    match presented inside quote marks as a quotation. This is the identical defect
    live_check §E lines 331-335 claims to have already corrected for the other
    member ("Paraphrase inside quote marks is not a quote").
  * PROVENANCE: scheduler.py:513 `d["steps_flipped_today"] =
    _steps_closed_from_log(f, today)` reads `handoff/harness_log.md`. The
    `--since-as-filter=midnight` git window is scheduler.py:501-507 and feeds only
    `d["commits_today"]`. So the `Steps closed:` figure is NOT produced by the
    allowlisted window at all.
  * Carried in FOUR places + the masterplan note: guard ALLOWLIST_REASONS
    (:178-181), live_check §E table (:326), live_check §J8 fenced block (:800-804
    + the sentence at :806), experiment_results_86.94.md:101.

### C. MUTATION MATRIX -- independently re-run IN MEMORY (no repo writes)
Method: source mutated in memory, exec'd with __file__ = the real path so
`REPO = parents[2]` still resolves. CONTROL FIRST: rc=0, 74 passed, 0 failed.
All nine of Main's declared cells reproduce as KILLED, counts identical to his:
  M-A 66/8 | M-B 72/2 | M-C 64/10 | M-D 73/1 | M-E 73/1 | M-F 73/1
  M-G 72/2 | M-H 72/2 | M-I 72/2      => killed=9 survived=0 unscorable=0

MY OWN CELLS:
  QA-E strip_docstrings -> identity                       KILLED (73/1)
  QA-A co-written probe+fixture, blind to corpus, bool=False   **SURVIVED 74/0**
  QA-B drop `Steps closed:` probe only                    SURVIVED 74/0
  QA-C drop `real commit lines` probe only                SURVIVED 74/0
  QA-D keep ONLY `Steps closed:` probe                    SURVIVED 74/0
  QA-F neutralise the dead-probe detector (_dead=[])      SURVIVED 74/0 (inert on a clean tree)
  K1 KNOWN_MEMBER_REF -> HEAD (the FIXED blob)            **SURVIVED 74/0**
  K2 BARE_DATE never matches                              SURVIVED 74/0
  K4 BARE_DATE+NAIVE_TS+RELATIVE_WORD all blind           SURVIVED 74/0
  K3 KNOWN_MEMBER_PATH -> a file with no window           KILLED (72/2)

### F3 (NEW, mine). The fixture mechanism is a TAUTOLOGY in the direction that matters
- QA-A: scheduler figure_probes -> [r'QQ_SELF_WRITTEN_TOKEN_\d+'], probe_fixtures ->
  ["QQ_SELF_WRITTEN_TOKEN_7"], quoted_as_evidence -> False. Guard: **rc=0, 74/0**.
  Fixture PROVENANCE is asserted in a comment and enforced by nothing, so a
  probe+fixture pair co-written from the same misreading that caused the cycle-4
  defect still ships green. J8:848 ("it protects any future member claimed
  `False`, not just this one") is therefore an OVERCLAIM, measured by execution.
- Not hypothetical: probe #4 `commits_today["\']?\s*[:=]\s*[\[\d]` matches NOTHING
  real. I EXECUTED `format_away_digest_sections` with the live `_git_today()`
  output: probes 2 and 3 match the render, probe 4 does not, and probe 4 has no
  tracked-corpus hit. Its only match is fixture #4 `'"commits_today": ['`, which
  the source comment does not account for (it describes only three of four).
- Verified renderer behaviour (mine): `*Shipped today*\n- 7ae001f0 chore: ...`;
  `Steps closed: 86.94, 86.92`. Fixtures 1-3 DO have real provenance.

### F4 (NEW, mine). Criterion-3 gate is weaker than its framing
Section [1] RE-IMPLEMENTS the scan inline WITHOUT `resolve()`, so:
- K1: repointing KNOWN_MEMBER_REF at HEAD -- where the window is ALREADY FIXED to
  `{CORPUS_SINCE}` = `...Z` -- still prints "ok [1] and classifies it SLIDING"
  (via the fail-closed indirection branch). Section [2], which DOES use the
  shipped path, calls the same line REPRODUCIBLE. The two sections disagree.
- K2/K4: blinding BARE_DATE / NAIVE_TS / RELATIVE_WORD leaves [1] green, because
  classify() falls through to "SLIDING, unrecognised form".
- K3 does kill, so the gate has SOME force. WARN, not BLOCK. Named fix: route [1]
  through `scan_text()` (what §[4] already does and says it does) and assert the
  bare-date branch, not merely "SLIDING".

### F5 (NEW, mine). §E LABELS ARE INVERTED -- the SUPERSEDED FALSE judgement is
### presented as "The current run prints:"  [criterion 5]
- live_check_86.94.md:282 -- "**THE BLOCK ABOVE IS THE CYCLE-3 STATE AND IS
  SUPERSEDED. The design it shows -- `mentions_reviewed` ...**". Block #1
  (:247-280) contains NO `mentions_reviewed` line; it is the CYCLE-5 output
  (POSITIVE CONTROLS assertions, quoted_as_evidence=True, QUOTED in 1).
- :288 "The current run prints:" -> block #2 (:290-316) shows
  `QUOTED in 0` and `quoted_as_evidence=False` -- the CYCLE-4 output, i.e. exactly
  the FALSE judgement this cycle exists to correct.
- :318-322 then asserts the answer is 1. Three mutually inconsistent statements in
  40 lines, with the false one labelled current.
- ROOT CAUSE, from `git diff 88d7d84c fca21bc6 -- handoff/current/live_check_86.94.md`:
  the cycle-5 edit rewrote block #1 in place and touched neither block #2 nor
  either label. In cycle 4 the labels were correct; the fix inverted them.
- ACTUAL current output (my run): `QUOTED in 2: .claude/masterplan.json,
  handoff/archive/misc/live_check_62.8.md`, quoted_as_evidence=True, 3
  positive-control assertions per member. NEITHER block reproduces.

### CRITERIA
1 MET   - two runs 1h00m49s apart; bare 08-13 376 -> 360 (DOWN while the repo grew),
          pinned 424 -> 428; `git rev-parse` resolver output; no count pinned into the
          criterion. I RE-DERIVED the arithmetic myself: 20 commits slid out of the
          08-13 band in (22:50:20, 23:51:09], 4 commits between the two measurement
          HEADs (both shas exist) => 376+4-20 = 360 exactly.
2 MET   - rule written down at script :70-93; command = the guard; output quoted;
          4 members each classified with a per-member reason (re-run by me).
3 MET   - 06c3265f:72 '2026-08-11' -> SLIDING "bare date -- git applies the CURRENT
          TIME OF DAY". WARN per F4 (the gate would also pass on the FIXED blob).
4 MET   - a judgement is stated for every member and the corrected scheduler `True`
          is CORRECT ("12 real commit lines" IS a count of _git_today() output quoted
          as verification evidence in a tracked file). Evidence defects F2/F3.
5 NOT MET - F5. A figure this cycle found wrong ("QUOTED in 0" /
          "quoted_as_evidence=False") is still carried, mislabelled as the CURRENT
          run, in the primary handoff artifact -- the correction ACCOMPANIES rather
          than REPLACES, which is the criterion's literal wording. F2's
          'Steps closed: 6' is likewise carried unchanged in 4 files + the masterplan.
6 MET   - control GREEN at 74/0 observed FIRST; 9/9 declared cells independently
          KILLED with matching counts; QA-E kills the stripper neutering; kills are
          attributable (each cell asserts value != "<unparsed>").
7 MET   - diff = guard script + handoff artifacts + a masterplan NOTE only. The
          86.94 `verification` block is byte-identical to f1b02a36. Nothing touches
          qa.md, qa-verdict.js or any verdict path. Guard is allowlist/report-only.

### HARNESS COMPLIANCE (5 items) -- CLEAN
1 research gate: research_brief_86.94.md gate_passed=true, 17 sources, 45 URLs,
  recency scan true; research_brief_86.94_cycle4.md gate_passed=true, 7 sources,
  23 URLs, recency true. Both brief_status COMPLETE.
2 order: brief(09:06) < contract(09:09) < guard(09:36) < experiment_results(09:42).
3 experiment_results_86.94.md present (19,836 bytes).
4 log-last: masterplan status still `pending`; the single `phase=86.94` row in
  harness_log is Cycle 230's PARK of the prior cycle, not this one.
5 no verdict-shopping: evidence CHANGED materially between 88d7d84c and fca21bc6
  (+108 lines of guard, +257 of live_check, +45 of experiment_results).

### DETERMINISTIC
- immutable command: exit 0, prints `green` (and cannot fail on this class -- Main
  disclosed this accurately).
- guard: exit 0, ALL GREEN 74 passed 0 failed.
- ruff --select F821,F401,F811 on the derived scope (scripts/qa/verify_no_sliding_
  windows_86_94.py, non-empty set asserted): "All checks passed!", exit 0.
- ast.parse OK. No frontend/backend code in the diff => 1b/1d N/A.
- working tree carries no uncommitted change to any 86.94 artifact, scripts/qa/, or
  .claude/masterplan.json.

### VERDICT REASONING
worst-of-lenses: correctness PASS (product is right); does-it-reproduce FAIL (four
artifact claims do not reproduce: the "current run" block, the 'Steps closed: 6'
quotation, J8's artifact=False table cell, and J8:848's protection claim);
scope-honesty CONDITIONAL (J6/W2 disclosure is exemplary, J8:848 overclaims).
min(lenses) = FAIL, and criterion 5 is missed on its literal wording.
=> FAIL

### J8 probe-table contradiction, confirmed directly
`Steps closed:\s*\S` DOES match handoff/archive/misc/live_check_62.8.md
(-> 'Steps closed: 6'), but live_check_86.94.md:829 records that row as
artifact=**False**, contradicting the guard output quoted in §E of the same file.

COMPLETED: 2026-08-17T07:58:30Z

