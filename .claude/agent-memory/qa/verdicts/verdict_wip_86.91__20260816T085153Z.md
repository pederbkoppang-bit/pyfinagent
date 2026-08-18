STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.91
WRITTEN: 2026-08-16T08:51:53Z

# Q/A cycle-2 evaluation of masterplan step 86.91

Role: Layer-3 Q/A (merged qa-evaluator + harness-verifier). Read-only.
Launch: Workflow rail. qa.md read in full at 08:51:53Z.

## Plan
A. harness-compliance audit (5 items)
B. deterministic: immutable command, git scope, lint, re-runnable checkers
C. LLM judgment vs the 8 immutable criteria + the 4 "judge these specifically" asks

## Log (appended as established)

### Prior-attempt / sequence evidence
- `qa_wip.py 86.91 --spawned-at 2026-08-16T08:51:53Z`: attempt_number=2,
  prior_attempts=1, attempt_number_status=ok, source_present=true,
  records_retained=2 (gauge), identity_checked=true.
  prior record: verdict_wip_86.91__20260816T082544Z.md
- `verdict_history_86_21.py --step 86.91 --evidence-only`:
  status=`no_rows_for_step`, verdicts=(none).
- CROSS-CHECK: attempt_number (2) > ledger count (0) => LEDGER IS STALE.
  sequence: UNRELIABLE from the ledger. Main's advisory disclosure says cycle 1
  was CONDITIONAL (run wf_96cff705-af0); that is ADVISORY ONLY.

### A. Harness compliance (5 items)
1. research-gate-before-contract: research_brief_86.91.md exists, 21,062 chars,
   mtime 09:58:08 < contract 10:14:17. Contract cites run wf_6f758470-f84,
   8 sources read in full (floor 5), 28 URLs (floor 10), recency scan performed,
   gate_passed=true RECOMPUTED. OK
2. contract-before-generate: research 09:58 < contract 10:14:17 < hook edit
   10:14:54 < checker/replay 10:42:57 < artifacts 10:49/10:50. OK
3. experiment_results present: yes, 15,547 bytes, cycle-2 Follow-up section. OK
4. log-last: `grep -F 86.91 handoff/harness_log.md` = ZERO hits; masterplan
   86.91 status=pending, 86.90 status=pending. Correctly NOT yet logged/flipped. OK
5. no-verdict-shopping: evidence CHANGED between spawns -- 98c5b6ab modified
   experiment_results_86.91.md (+79), live_check_86.91.md (+24),
   replay_changelog_rule_86_68.py (+25), verify_changelog_flip_86_91.py (+127).
   This is the documented cycle-2 flow, not a re-spawn on unchanged evidence. OK

### B. Deterministic
- IMMUTABLE COMMAND `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh &&
  echo parses'` -> `parses`, **exit=0**.
- heredoc python parses: OK, 327 lines.
- ast.parse both changed .py: OK.
- LINT GATE, scope DERIVED from git (`git diff --name-only a21a5889 HEAD --
  '*.py'`; a21a5889 is 8dc70502's parent, so the step's own commits are INCLUDED):
  scripts/qa/replay_changelog_rule_86_68.py, scripts/qa/verify_changelog_flip_86_91.py
  -> `uvx ruff check --select F821,F401,F811` (xargs -0, non-empty set asserted)
  -> "All checks passed!" exit=0.
- `python scripts/qa/verify_changelog_flip_86_91.py` -> ALL GREEN: 31 passed,
  0 failed, EXIT=0. (109 lines extracted from the hook.)
- `python scripts/qa/replay_changelog_rule_86_68.py` run TWICE by me, outputs
  byte-identical (diff empty), exit 0 both times.
  707 commits in [2026-08-11T00:00:00 .. 8dc70502]; OLD 251 / SHIPPED 9 / FIXED 11.
  W1's fix REPRODUCES in my environment.
- no frontend/**, no backend/** in the derived scope -> 1b/1d not applicable.
  NOTE: working tree carries UNCOMMITTED, UNRELATED prod edits (sovereign `1y`
  window: backend/api/sovereign_api.py + 5 frontend components). Not in either
  86.91 commit; not attributable to this step. Flag for Main: `git add -A` in
  auto-commit-and-push.sh would sweep them into the flip commit.

### C. Criterion-by-criterion (my own re-derivations)
- C1 MET. I RE-DERIVED it myself against real history:
  e4f2e844 -> `86.86 before: None -> after: 'done'`; SHIPPED newly_done `[]`;
  FIXED `['86.86']`. Commit shipped backend/services/autonomous_loop.py (+133)
  and a 199-line test. Ordering: reproduction is quoted in the contract (10:14:17)
  which PRECEDES the hook edit (10:14:54).
- C2 MET. Predicate is a key-space membership test with an `_ABSENT` identity
  sentinel; three states. No step id in the fix -- checker [1] drives 9.99 and
  12.7 in unrelated phases and requires the same bump; I confirmed by reading
  the shipped source that no literal step id appears.
- C3 MET, re-derived by MY execution: 251 / 9 / 11, +2, both commits accounted:
  e4f2e844 closed 86.86 (done today), 8b520f6c closed 86.81 (done today).
  Both genuinely shipped code. PARKED 86.9/86.44 still 0 under the fixed rule
  vs 13 each under OLD -- 86.68 not reintroduced.
- C4 MET. Closed reason set; `_log_decision(bump_type)` at hook line 262 runs
  BEFORE every `sys.exit(0)` (346/352). Live decision log on disk has 3 rows,
  all `reason=no_flip` -- correct, those commits closed nothing.
- C6 MET with a residual (see QA-C2-1/2/3 below). All 6 cells re-run by me;
  all 6 mutants LOAD and their probes DISCRIMINATE (no phantom kills).
- C7 MET. Fault injection: OSError from the git call -> no propagation,
  bump="none", `flip-detect FAILED` on stderr. Re-run by me.
- C8 MET. masterplan 86.90/86.91 still `pending`; no verdict machinery in the
  86.91-attributable files.

### FINDINGS
**QA-C2-1 [WARN] -- SURVIVING MUTANT on the W1 fix's own guard.**
Checker [5] "the replay corpus is PINNED AT BOTH ENDS" is a pure SUBSTRING SCAN:
`"CORPUS_SINCE" in SRC and "2026-08-11T00:00:00" in SRC and "CORPUS_UNTIL" in SRC
and "CORPUS_UNTIL = None" not in SRC`.
MUTANT: replace the single line `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)`
with `pass`. Every literal survives; the guard predicate returns True (SURVIVED,
31/31 stays green); the [5]/[6] behavioural drive is unaffected (it only touches
`newly_done_ids`). MEASURED REAL EFFECT: pinned corpus 707 commits vs unpinned
712 (+5 today, growing), AND the header would still PRINT `= 8dc70502` because
`_end` is rev-parse'd from the CONSTANT, not from the selected commits -- a more
confident false endpoint than before the fix. This is vacuity shape #3
(literal kept, behaviour stripped) -- the exact shape cycle-1's W2 found, left in
place on the W1 remediation. Also note the printed endpoint is a claim about
INTENT, never a measurement: `commits[0]` is never compared to CORPUS_UNTIL.

**QA-C2-2 [NOTE] -- exception-as-KILLED is latent, not live.**
`[4]` and `[6]` both score `except Exception: killed = True`. NEGATIVE CONTROL I
ran: a syntactically broken mutant raises SyntaxError in extraction and would be
scored KILLED without any probe running. On the SHIPPED cells this is NOT firing:
all 6 mutants load cleanly and every probe returns True on a real behaviour
change (verified individually). So the concern Main flagged is real in design and
inert in fact today.

**QA-C2-3 [NOTE] -- the source-derived denominator fails OPEN.**
The rule counts `ast.Return` nodes whose value is the literal Constant "none"
inside `_flip_magnitude`. Shipped count = 4, matching the 4 branches; the check
is `len(_observed) >= _none_sites`.
(a) I added a FIFTH literal `return "none"` branch: `_none_sites` -> 5, check
    goes RED. The property Main claims DOES hold for the likely shape.
(b) I converted ONE branch to `_v = "none"; return _v` (behaviour identical --
    the branch still returns 'none' with reason 'first_commit'): `_none_sites`
    -> 3, check stays GREEN, MEMBER MISSED. An UNDERCOUNT makes the check
    EASIER to pass, so the rule fails OPEN, not closed.
(c) The walk also covers the NESTED `_statuses` helper (its `return out` /
    `return None` are inside the walked FunctionDef), so the scope is the whole
    nested tree rather than the detector's own branches.
Verdict on Main's question: the enumeration RULE is SOUND in the detecting
direction for literal returns and INCOMPLETE for non-literal ones. The artifact
claim "A future 5th branch fails the check instead of slipping past it" is stated
WITHOUT its bound -- true only for a literal-constant return. Same
overgeneralisation pattern the author names in his own 86.90 commit body
("a guard whose stated rule exceeds its measured behaviour is this series'
failure mode").

**QA-C2-6 [WARN] -- SECOND SURVIVING MUTANT: the replay guard is SINGLE-ID.**
`[5]`/`[6]` drive `newly_done_ids` with ONE fixture whose only created id is
`86.86`. MUTANT: narrow the shipped predicate to
`... is _ABSENT and s=="86.86"` (anchor unique). MEASURED: all four `[5]`
assertions stay GREEN (extractable/runnable; True-arm `['86.86']`; False-arm
`[]`; the two arms disagree) -- the mutant SURVIVES the whole checker. On an
unrelated id the shipped predicate returns `['9.99']` while the mutant returns
`[]`. This is EXACTLY the shape criterion 2 forbids ("a fix that special-cases
86.86 or any single step id rather than the CLASS fails"), left unguarded on the
replay half -- and `newly_done_ids` is the instrument that PRODUCES criterion 3's
three numbers, so a narrowed predicate would silently under-report FIXED (it
would find 86.86 and miss 86.81) with the guard green.
CONTRAST, and this is why it is WARN not BLOCK: the HOOK guard `[1]` DOES drive
unrelated ids 9.99 and 12.7, so the PRODUCTION predicate is genuinely
class-tested. Only the sibling replay predicate is not.

**QA-C2-5 [WARN] -- live_check §4's "verbatim" capture does NOT reproduce.**
`live_check_86.91.md` §4 quotes `$ python scripts/qa/verify_changelog_flip_86_91.py`
with `ALL GREEN: 24 passed`, "74 lines extracted", 3 `[4]` cells and NO `[6]`
section. FRESH run of the identical command TODAY: `ALL GREEN: 31 passed`,
"109 lines extracted", 4 `[4]` cells + 2 `[6]` cells, and a `[2]` recall line.
The cycle-2 commit 98c5b6ab DID update §2 of this file (706/250 -> 707/251, the
BOTH-ENDS-PINNED header) and left §4 at the cycle-1 state. So the operator-facing
live_check gate artifact still presents the SUPERSEDED guard as current -- its
`[5]` list is literally the three substring scans cycle-1 killed, and a reader of
live_check alone would conclude W2 and W3 were never fixed. Not fabricated and
not overclaiming (it UNDER-claims), but a capture labelled verbatim that does not
reproduce. Same class as W1: the correction was applied to one end and claimed
for both.

**QA-C2-4 [NOTE] -- criterion 5's end-to-end half is PENDING, honestly.**
Version header still frozen at `### v6.93.222 ... (2026-08-14)`. All 4
CHANGELOG.md commits in scope are `chore: auto-changelog hook entry for X` --
hook-produced, no hand-edit anywhere in the diff. The 4 live decisions are all
`no_flip`, which is CORRECT (none closed a step). The render half of the hook
(the `### vX.Y.Z` insert, gated on `bump_type != "none"`) is NOT driven by any
guard -- the guard stops at `bump_type`. live_check §6 discloses this and
correctly predicts the flip will read `flip_transitioned`, NOT `flip_created`.
I CLOSED THIS MYSELF rather than leave it asserted: I exec'd the hook's whole
python heredoc in memory with `open` and `subprocess.run` stubbed (repo
untouched, nothing written, nothing committed) and drove four cases --
  [A] created-and-closed 86.99  -> `bump=patch reason=flip_created
      created_done=86.99` AND a new header `### v6.93.223 ... (2026-08-16)` is
      actually WRITTEN. The render half works.
  [B] chore, closes nothing     -> `reason=no_flip`, NO new header.
  [C] the ACTUAL flip shape for this step (86.90+86.91 pending->done)
      -> `reason=flip_transitioned transitioned_done=86.90,86.91` + v6.93.223.
      live_check §6's prediction is CONFIRMED by execution.
  [D] MUTANT, None exclusion restored (anchor unique), re-run [A]
      -> `reason=no_flip`, NO header. The mutation is killed END-TO-END, not
      only at `bump_type`.
So criterion 5 is not at risk; the residual is that the AUTHOR's evidence stops
at `bump_type` and the operator-facing proof is deferred to the flip.

### VERDICT (returned via StructuredOutput)
CONDITIONAL. All 8 immutable criteria MET in substance -- criteria 1, 3, 6 and 7
independently RE-DERIVED by my own execution, and the render half additionally
driven end-to-end. Harness compliance clean on all 5 items. No unintended
production change in the step's own commits.
Capped at CONDITIONAL by three WARN findings, two of them mutants I EXECUTED and
watched SURVIVE (QA-C2-1, QA-C2-6), all three located in the cycle-2 remediation
surface itself: the W1 fix's guard is a substring scan, the replay predicate's
drive is single-id, and the live_check §4 capture was not regenerated.
Lens split (worst-of-N): correctness=PASS, does-it-reproduce=CONDITIONAL
(live_check §4), scope-honesty=CONDITIONAL (the unbounded "future 5th branch"
claim + stale §4). min = CONDITIONAL.

COMPLETED: 2026-08-16T09:04:11Z

