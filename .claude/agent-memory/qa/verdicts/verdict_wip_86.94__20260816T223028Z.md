STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-16T22:30:28Z

# Q/A write-first record -- step 86.94, cycle 3

Spawn context supplied by Main (ADVISORY ONLY): attempt_number=3,
verdict_sequence=[FAIL, FAIL], prior run_ids wf_eb4c97d0-c34, wf_b5066952-bf4.

## Prior-attempt evidence (gathered, not used as a trigger)
- `qa_wip.py 86.94 --spawned-at 2026-08-16T22:30:28Z`: source_present=true,
  attempt_number=3 (status ok, is_lower_bound true), prior_attempts=2,
  records_retained=3 (GAUGE), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.94 --evidence-only`: status
  **no_rows_for_step**, verdicts (none).
- CROSS-CHECK: attempt_number 3 > ledger count 0 -> **THE LEDGER IS STALE**.
  Sequence per ledger: UNRELIABLE. Main's advisory [FAIL, FAIL] is recorded as
  advisory only.

## A. Harness-compliance audit (5/5 clean)
1. research-gate-before-contract: research_brief_86.94.md present, envelope
   brief_status COMPLETE, gate_passed true, 17 sources read in full, 45 URLs,
   recency scan, audit-class coverage.dry true (12 rounds, 3 dry). Enforced
   via research-gate.js rail run wf_2c05296c-5d4 quoted in contract:18-30. OK
2. contract-before-generate: all five artifacts landed in ONE commit f1b02a36
   so git cannot order them; internal evidence does -- contract carries
   Measurement 1 (22:50:20 CEST) and says "Measurement 2 is taken in GENERATE",
   and M2 is stamped 23:51:09, after the 23:33:03 first commit. OK
3. experiment_results present (12,679 b). OK
4. log-last: `grep -F 'phase=86.94' handoff/harness_log.md` -> 0 rows;
   masterplan 86.94 status still "pending". OK
5. no-verdict-shopping: evidence CHANGED 379be687 -> d6c732b7 (guard 37->45
   assertions, WINDOW_RE widened, git-proximity filter added, ALLOWLIST_CLAIMS
   added, 4 artifacts edited). OK

## B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python
  scripts/qa/verify_changelog_flip_86_91.py > /dev/null && echo green'`
  -> `green`, **exit=0**.
- New guard bare: `python scripts/qa/verify_no_sliding_windows_86_94.py`
  -> `ALL GREEN: 45 passed, 0 failed`, **exit=0**.
- ruff F821/F401/F811 on the git-DERIVED scope
  (`git diff --name-only a5cbfd67 HEAD -- '*.py'`, non-empty asserted, xargs -0;
  3 files) -> "All checks passed!" **exit=0**.
- Scope: `git diff --name-only a5cbfd67 HEAD` = 13 files, all handoff/*,
  scripts/qa/* (3 py), CHANGELOG.md. **No verdict machinery touched** (grep for
  qa-verdict|qa.md|run_harness|masterplan.json -> NONE).
- NOTE (not attributable to 86.94): the working tree carries UNCOMMITTED edits
  to backend/api/sovereign_api.py + 5 frontend components (a `1y` red-line
  window). Not in any 86.94 commit, not in scope; flagged only because a future
  `git add -A` would sweep them under this step's name.

## C. Criterion-by-criterion (independently re-derived, not read)

### C1 -- drift REPRODUCED by execution, no pinned figures  -> MET
Artifact: M1 22:50:20 CEST HEAD a5cbfd67 bare 376 / pinned 424; M2 23:51:09
(1h00m49s later) bare 360 / pinned 428. Bare DIFFERS; pinned differs from both.
MY INDEPENDENT CORROBORATION (all reproduce exactly):
- `git log a5cbfd67 --since=2026-08-13T00:00:00 | wc -l` = **424** (matches M1).
- commits in [2026-08-13T22:50:20+02, 2026-08-13T23:51:09+02) = **20**
  (the "20 slid out").
- commits in [2026-08-16T22:50:20+02, 2026-08-16T23:51:09+02) = **4**
  (the "4 landed"). 376 + 4 - 20 = 360 and 424 + 4 = 428 both close.
- MECHANISM re-derived at 2026-08-17 00:38:58 CEST:
  `git rev-parse --since=2026-08-13` -> --max-age=1786574338 = 2026-08-13
  00:38:58 local (today's clock carried onto the target date);
  `--since=today` -> --max-age=1786919938 = NOW, 0 commits (vs git-log(1)'s
  "midnight"); pin 1786572000 vs pinZ 1786579200 = exactly 7200s = the CEST
  offset (the TZ-naive claim).
- My own bare/pinned counts tonight are 436/436 -- IDENTICAL, because at 00:38
  the slid cutoff is only 39 min past midnight and that band is empty. This is
  the "intermittently invisible" property the author disclosed in A1b, and it
  corroborates rather than refutes.
- The criterion text in masterplan.json pins no figures and matches
  contract:127 verbatim.

### C2 -- class enumerated FROM SOURCE, rule written down  -> MET
Rule in source lines 70-94 with an explicit REPRODUCIBLE (a)/(b)/(c)
definition; command + output quoted; 4 members each classified with a reason.
MY INDEPENDENT ENUMERATION (851 tracked *.py/*.sh under scripts/,
.claude/hooks/, backend/; raw token grep for
`--(since|until|after|before)(-as-filter)?`, 59 raw occurrences) yields exactly
the SAME 4 live git-window CODE sites. **Symmetric difference = 0.**
Excluded correctly: `congress.py:312 --since-days`, `census_..._86_31.py:64
--before`, `rail_drop_rate.py:184 --since` -- all argparse flags for non-git
tools (I verified rail_drop_rate.py never invokes git). No live
`--max-age`/`--min-age` and no `$(date ...)` feeding a git window in scope.

### C3 -- known-member recall is a HARD gate  -> MET
`git show 06c3265f:scripts/qa/replay_changelog_rule_86_68.py` line 72 =
`sh("git","log","--since=2026-08-11",...)` -- the bare-date pre-86.91 form,
confirmed by me. Guard [1] finds it and classifies SLIDING. Hardness PROVEN by
mutation M16: KNOWN_MEMBER_REF -> "deadbeef" turns the guard **RED** (exit 1),
it does not skip.

### C4 -- per-member quoted-figure judgement, stated not silent  -> MET
All four members carry a stated judgement in source, experiment_results and
live_check. I VERIFIED the corrected frontend_route judgement:
`handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md`
really does carry `usage_source = "git_activity_30d"`, `opens_30d`, and
`every_route_has_usage_count | PASS (12/12 integer opens_30d)` -- the cycle-2
correction is TRUE. scheduler.py's `midnight` window produces `commits_today`
(a `[:20]` list of `%h %s` lines for a Slack digest, scheduler.py:501-505) --
consistent with `quoted_as_evidence: False`.
BUT see W1/W3: the supporting COUNTS quoted for this criterion are stale.

### C5 -- corrected in EVERY file, replace not accompany  -> MET (literal)
Verified every carrier of the naive window string myself:
- `git grep -n "2026-08-11T00:00:00[^Z]" -- handoff/` -> the only survivors are
  past-tense descriptions of the defect (experiment_results_86.91.md:149/:157,
  live_check_86.91.md:100) plus prior evaluator critiques. No present-tense
  assertion of the naive constant survives.
- experiment_results_86.91.md:141/:146 region, live_check_86.91.md:90-91 and
  harness_log.md:35558 all now carry `2026-08-11T00:00:00Z`. REPLACED.
- The one verbatim capture (experiment_results_86.91.md:77) REGENERATED: I ran
  `replay_changelog_rule_86_68.py` and its output matches the capture
  line-for-line (`corpus: 707 commits in [2026-08-11T00:00:00Z .. 8dc70502 =
  8dc70502]` ... `251` ... exit 0).
The criterion's literal subject (figures found UNREPRODUCIBLE, i.e. clock/TZ
dependent) is discharged in every carrier. Its stated PRINCIPLE is nonetheless
violated by W1 below, on a superseded-by-scope figure.

### C6 -- regression guard, mutation-tested, control GREEN first  -> MET
Control observed GREEN first: in-suite ([4] CONTROL: the replay has no unlisted
sliding window) and in my own driver (C0 = 45/0, exit 0).
**END-TO-END mutation the author did NOT run** -- I patched
`pathlib.Path.read_text` in-process so a REAL tracked file
(`scripts/harness/run_harness.py`) appears to gain a window, with ZERO disk
writes, then exec'd the shipped guard:
| mutant injected into a real tracked file | guard exit | result |
|---|---|---|
| `subprocess.run(["git","log","--since","2026-08-11"])` | 1 | KILLED (flags :1232 '2026-08-11') |
| `--since=2026-08-11T00:00:00` (TZ-naive) | 1 | KILLED |
| `--since=30.days` | 1 | KILLED |
| `--since=2026-08-11T00:00:00Z` (NEG CONTROL) | 0 | correctly NOT flagged |
16 further mutants against the guard's own source; **12 KILLED**:
strip_docstrings->identity (1 fail), is_prose->False (5), resolve->fail-OPEN
(4), mentions_reviewed 282->281 (1), WINDOW_OPTS drop --after/--before (3),
git-proximity filter removed (1, flags census_qa_write_guard), classify->always
REPRODUCIBLE (14), KNOWN_MEMBER_REF->deadbeef (1), WINDOW_RE narrowed to
pre-cycle-3 (2), argv-off + fail-open both (2).
**4 SURVIVORS**, adjudicated:
- M12 SELF_REL -> "_86_94.py": EQUIVALENT MUTANT, discarded -- `git ls-files |
  grep -c '_86_94\.py$'` = 1, so the exclusion is still exactly one file and
  the `== 1` assertion still binds against a real widening.
- M9/M9b `quoted_as_evidence` flipped to the WRONG bool: SURVIVED -> W4.
- M14 restore fail-OPEN on the `<unparsed>` branch: SURVIVED -> W5.
- M15 disable VALUE_ARGV_RE: SURVIVED -> W6.
Criterion 6's literal demand is met and PROVEN BY EXECUTION, not claimed.

### C7 -- verdict semantics unchanged  -> MET
`git diff --name-only a5cbfd67 HEAD` touches no verdict machinery (no
qa-verdict.js, no qa.md, no run_harness.py, no masterplan.json). Nothing in the
diff can turn a non-PASS into a PASS.

## D. Findings (all WARN/NOTE; no BLOCK)

**W1 [WARN, Contradiction] -- a superseded figure survives in TWO artifacts,
inside the paragraph that announces its supersession.**
`live_check_86.94.md:265` states: "this block previously showed `37 / 5 / 0`
mention counts -- the cycle-1 numbers ... It is regenerated here from the
shipped run, which prints `282 / 6 / 49`." Nine lines below, `:274` still reads
"Name appears in **37** files", and `experiment_results_86.94.md:99` reads the
same. The shipped guard prints 282 and `ALLOWLIST_CLAIMS[...scheduler.py]
["mentions_reviewed"] = 282` asserts it. The CAPTURE was regenerated; the TABLE
beside it was not -- the identical accompany-not-replace shape one element down,
in the artifact the masterplan's live_check field names.

**W2 [WARN, Overgeneralization] -- an effectiveness claim about cycle-3's own
work that measures ZERO.** `experiment_results_86.94.md:205` and commit
d6c732b7's message both state "the widened rule immediately found a live site
the old one missed". MEASURED two ways and it does not reproduce:
(a) reverting ONLY the WINDOW_RE widening in the shipped guard (M18) leaves the
live-site enumeration byte-identical -- same 4 sites, same values, same
verdicts; (b) exec'ing the actual cycle-2 blob `379be687` enumerates the SAME 4
sites (37/0 green). All four live git windows in this repo use the `=` spelling,
so the argv widening could not have found one. Its real effect is confined to
the mutation cells.

**W3 [WARN, Contradiction] -- two counts for one measurement inside one file.**
`verify_no_sliding_windows_86_94.py:198` (allowlist REASON) says "Measured over
the whole handoff tree: **55** files mention it", while `:228`
(ALLOWLIST_CLAIMS) and the shipped run both say **49**. The prose figure is
unasserted and stale; only the structured one is checked.

**W4 [WARN, illusory-guard #17b] -- `quoted_as_evidence` is TYPE-checked only.**
`:539` asserts `isinstance(_claim.get("quoted_as_evidence"), bool)`. I flipped
scheduler.py's False->True and frontend_route's True->False; the guard stayed
**45/0 green** both times. The source's "A bool cannot be satisfied by
vocabulary" is true but is not a claim of verification. Mitigation is real:
`mentions_reviewed` IS compared to the measurement (M8 282->281 KILLED), so
corpus drift re-opens the judgement. Not sole coverage -- the judgement is also
stated in prose, which is what criterion 4 literally requires.

**W5 [WARN, guard vacuity] -- the fail-closed `<unparsed>` branch has no cell.**
Source :369-378 calls it load-bearing ("a fail-OPEN inside the module whose
central claim is that it fails closed"), but restoring the old `continue`
leaves all 45 assertions green (M14 SURVIVED). Nothing in the suite reaches
that branch, because the space form is now caught by VALUE_SP_RE.

**W6 [WARN, mis-attributed kill mechanism, qa.md 4c shape #11] -- the two argv
cells are killed by the fail-closed path, not by the argv extractor.**
Disabling VALUE_ARGV_RE alone leaves 45/0 green (M15 SURVIVED); disabling
VALUE_ARGV_RE **and** the fail-closed branch together kills exactly those two
cells (M17, 43/2). So `argv-list-form` / `argv-list-after` credit the value
extractor while the assertion that actually fires is the `<unparsed>` catch-all.
The gate outcome is unaffected (both still classify SLIDING).

**NOTE** -- uncommitted, out-of-scope tree state (sovereign_api.py `1y` window +
5 frontend components). Not 86.94's; flagged for `git add -A` risk only.

## E. Lens split (worst-of-N)
- correctness: PASS -- guard is behavioural, enumeration complete, known-member
  gate hard, 86.91 corrections real and verified.
- does-it-reproduce: CONDITIONAL -- W1 and W2 are claims in the handoff that my
  own execution contradicts.
- scope-honesty: CONDITIONAL -- disclosure is unusually strong (self-exclusion
  residual, git-proximity residual, one-level indirection, space-form residual,
  [5] non-git census, and the immutable command's own uselessness are all
  stated), but W1 sits under a paragraph announcing the opposite.
min = **CONDITIONAL**.

## F. Verdict reasoning
No immutable criterion is MISSED on its literal wording: 7/7 MET, six of them
re-derived by me rather than read. Harness compliance 5/5. No unintended
production change in the graded commits. PASS is nonetheless unavailable
because two figures/claims in the handoff are contradicted by the instruments
shipped in the same commit (W1, W2), both one-line fixes. Per the runbook,
fixable gaps -> CONDITIONAL.

COMPLETED: 2026-08-16T22:45:57Z
