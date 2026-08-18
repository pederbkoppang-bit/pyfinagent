STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-17T07:18:54Z

# Q/A write-first record -- step 86.94 (cycle 4 evidence at HEAD d572a556)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, syntax/lint, re-run checker
C. Independent mutation matrix re-run (M-A..M-G), control first
D. Three attack targets: (1) loosening?, (2) figure_probes provenance, (3) criterion-5 correction-vs-annotation
E. Criterion-by-criterion MET/NOT MET

## Findings log (appended as established)

### Prior-attempt evidence
- qa_wip.py 86.94 --spawned-at 2026-08-17T07:18:54Z -> attempt_number=4,
  prior_attempts=3, source_present=true, attempt_number_status=ok,
  attempt_number_is_lower_bound=true, records_retained=4 (GAUGE, not counter).
- verdict_history_86_21.py --step 86.94 --evidence-only -> status=no_rows_for_step,
  verdicts=(none). LEDGER IS STALE: attempt_number(4) > ledger verdict count(0).
  sequence: UNRELIABLE from the ledger; Main's advisory disclosure says FAIL, FAIL,
  CONDITIONAL for cycles 1-3 (ADVISORY ONLY, Main is the constrained party).

### B. Deterministic
- IMMUTABLE CMD: `bash -c 'source .venv/bin/activate && python
  scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'` -> printed
  `green`, exit=0. DISCLOSED (by author, confirmed by me): this runs the 86.91
  checker and cannot fail on this step's defect class.
- `python scripts/qa/verify_no_sliding_windows_86_94.py` -> ALL GREEN: 68 passed,
  0 failed. Reproduced.
- HEAD = d572a556 (auto-changelog for 88d7d84c, the cycle-4 work commit).

### C. INDEPENDENT MUTATION MATRIX (in-memory exec, synthetic __file__, zero writes)
CONTROL FIRST: pass=68 failures=0 (identical to the shipped run).
Author's 7 claimed cells, all re-run by me, ALL KILLED:
  M-A restore fail-OPEN continue            KILLED  +8 new failures (pass 68->60)
  M-B VALUE_ARGV_RE never matches           KILLED  +2  (pass 66)
  M-C WINDOW_RE drops argv alternative      KILLED  +10 (pass 58)
  M-D frontend quoted True->False           KILLED  +1  (pass 67)
  M-E scheduler quoted False->True          KILLED  +1  (pass 67)
  M-F corpus widened to untracked           KILLED  +1  (pass 67)
  M-G frontend probes match nothing         KILLED  +1  (pass 67)
=> killed=7 survived=0 unscorable=0. The claimed matrix REPRODUCES.

My own additional cells:
  QA-3  self-exclusion widened to 2 files   KILLED  +2
  QA-6  classify bare-date -> REPRODUCIBLE  KILLED  +15
  QA-7  strip_docstrings -> identity        KILLED  +1
  QA-8  KNOWN_MEMBER_REF -> bogus sha       KILLED  +1
  QA-10 tautology on the equality + M-D     SURVIVED  => the equality
        `quoted_as_evidence == bool(_figs)` is the SOLE and load-bearing
        mechanism for M-D/M-E/M-G (positive confirmation, not a finding)
  QA-2  drop SELF="86.94" exclusion         SURVIVED (exclusion currently inert;
        NOTE only, it is stated in-source as a safety measure)

### ATTACK 1 -- is this a LOOSENING? NO, verified on the pre-cycle-4 blob
NB the prompt cited `d572a556^` as pre-cycle-4; that resolves to 88d7d84c, the
cycle-4 commit ITSELF. Correct blob is `88d7d84c^`. Its control: 42 passed, 3
failed (matches the contract). Under fail-set delta on THAT blob:
  M-A SURVIVED  M-B SURVIVED  M-C KILLED  M-D SURVIVED  M-E SURVIVED
=> four cells moved SURVIVED->KILLED; none moved KILLED->SURVIVED. Strictly
stronger on the tested surface. Capability genuinely lost: a figure quoted ONLY
in a gitignored file is now invisible -- DISCLOSED in source at :584-589 and the
True claim re-grounded on tracked evidence (I reproduce 5 hits / 3 files).

### ATTACK 2 -- probe provenance. MOSTLY sound, ONE measured failure.
- 86.97 probes reproduce the emitter verbatim (verify_decision_log_86_97.py:368-369). OK
- frontend probe 1 reproduces frontend_route_inventory.py:109/115/135. OK
  probes 2-3 (`\d+/\d+ integer opens_30d`, `opens_30d=\d+`) are shaped from the
  ARTIFACT's phrasing; the script emits JSON `"opens_30d": N`, never `opens_30d=N`.
  Benign for a True claim but contradicts "each derived from the emitting expression".
- scheduler probe 3 `Shipped today\s*\n(?:\s*[-*]\s+[0-9a-f]{7,}\s)` claims to derive
  from formatters.py:102-109 and PROVABLY CANNOT MATCH what that code renders:
  add() at formatters.py:71-76 emits `*{title}*\n{body}`, so a `*` sits between
  "today" and the newline. DRIVEN FOR REAL via format_away_digest_sections -> no match.

### FINDING 1 (capping) -- scheduler quoted_as_evidence:False, recall hole + counterexample
  QA-12 inject the REAL rendered digest into the corpus  -> scheduler probes 0, GREEN
  QA-13 inject the literal sentence from a TRACKED file  -> scheduler probes 0, GREEN
  QA-9  scheduler probes -> "zzzz_never_zzzz"            -> whole guard 68/0, GREEN
  QA-1  scheduler probes -> "the"                        -> KILLED (bool IS bound)
  COUNTEREXAMPLE ON DISK, TRACKED: handoff/archive/misc/live_check_62.8.md:31
    '"*Shipped today*" with 12 real commit lines' -- a count produced by that
    window, quoted as read-back verification evidence for phase-62.8.
  The allowlist asserts "no COUNT produced by this window is quoted as evidence".
  Named fix: add a probe for the rendered header (\*?Shipped today\*? /
  \d+ real commit lines) and restate as "quoted, unreproducible, inert"
  (quoted_as_evidence: True) -- the same disposition already used for frontend.

### FINDING 2 (capping) -- criterion-5 sweep scope does not reproduce
  Class A stated "6 carriers"; `git grep -l mentions_reviewed` returns 8.
    UNDISPOSITIONED: handoff/current/goal_next_2026-08-17.md (FORWARD-LOOKING day
    goal, still calls mentions_reviewed "the live tripwire") and
    handoff/current/research_brief_86.94_cycle4.md (carries 283/9/50 and 42/3).
  Class B stated "12 files, 9 coincidental" under pattern "45/0, 45 assertions";
    `git grep -l -E "45/0|45 assertions"` returns 11 with only 2 coincidental-only
    (two briefs w/ DOI 46.4.268). The cited coincidental examples ("all 45 trades",
    "all 457 test files") CANNOT be produced by the stated pattern -> the pattern
    quoted and the count reported come from different derivations.
  handoff/current/day_report_2026-08-17.md:49 still reads "Guard ships green at
    45/0", which day_halt.md:82 itself calls false; in no disposition list.
  Neither class quotes its enumeration command.

### ATTACK 3 -- day_halt.md verbatim: ANSWERED IN THE AUTHOR'S FAVOUR
A block labelled verbatim must be regenerated, never edited (qa.md 4b); editing a
dated preflight capture would falsify the record. It IS dispositioned explicitly
with a stated reason. This is NOT "accompanying". The criterion-5 gap is Finding 2.

### Criteria roll-up
1 MET (A0-A3: 376->360 bare vs 424->428 pinned, 1h00m49s apart, arithmetic closes
  with no residual; criterion text carries no pinned count)
2 MET (rule at :70-93; [2] prints command+output, 4 sites each classified w/ reason)
3 MET (hard gate; my QA-8 bogus-ref mutant KILLS it -- it cannot skip)
4 MET ON LITERAL WORDING, but see FINDING 1 -- the judgement for one of three
  members is contradicted by a tracked artifact and its instrument cannot detect it
5 PARTIAL -- main carriers genuinely REPLACED (verified), but see FINDING 2
6 MET (7/7 author cells + 4 of mine KILLED, control GREEN 68/0 FIRST, and it is
  strictly stronger than the pre-cycle-4 blob)
7 MET (88d7d84c touches only the checker + 4 artifacts + masterplan NOTES; zero
  criteria/verification/status lines; qa.md and .claude/workflows/ untouched)

### Deterministic tail
- ruff F821,F401,F811 over derived scope (git diff --name-only 88d7d84c^ 88d7d84c
  -- '*.py' = scripts/qa/verify_no_sliding_windows_86_94.py): All checks passed, exit 0
- worktree py scope (backend/api/sovereign_api.py, PRE-EXISTING at spawn, NOT this
  step's): All checks passed, exit 0
- ast.parse OK
- 89.5% gitignore claim reproduces: 49,094 total handoff .md / 5,168 tracked
- harness compliance 5/5: brief 09:06:38 < contract 09:09:15 < checker 09:11:38 <
  experiment_results 09:16:36 < live_check 09:17:11; gate_passed true (7 srcs, 23
  urls, recency true, brief_status COMPLETE); harness_log holds only the PARK row
  (Cycle 230), masterplan 86.94 status=pending; evidence CHANGED (45 -> 68 asserts)

VERDICT: CONDITIONAL

COMPLETED: 2026-08-17T07:32:14Z
