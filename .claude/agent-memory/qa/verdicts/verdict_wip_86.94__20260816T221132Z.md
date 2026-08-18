STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.94
WRITTEN: 2026-08-16T22:11:32Z

# Q/A write-first record -- step 86.94 (CYCLE 2)

Prior cycle: wf_eb4c97d0-c34 => FAIL on criteria 4, 5, 6 (Main's disclosure; verified
against the cycle-1 WIP record verdict_wip_86.94__20260816T215251Z.md, which is COMPLETE
and records exactly that reasoning).

## Prior-attempt / verdict evidence
- qa_wip.py 86.94 --spawned-at 2026-08-16T22:11:32Z: attempt_number=2, prior_attempts=1,
  source_present=true, attempt_number_status=ok, identity_checked=true,
  records_retained=2 (GAUGE, not a counter).
- verdict_history_86_21.py --step 86.94 --evidence-only: status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: attempt_number (2) > ledger rows for this step (none). Per qa.md the
  ledger is therefore STALE/incomplete for 86.94. sequence: UNKNOWN from the ledger.
  Main's advisory disclosure says ["FAIL"], consistent with attempt_number=2, but Main
  is the constrained party so it is advisory only.

## A. HARNESS COMPLIANCE (5 items)
1. Research gate: research_brief_86.94.md exists (46,123 B, mtime 2026-08-16 23:07:21
   local). PRECEDES contract. [verify envelope below]
2. Contract before generate: contract_86.94.md mtime 23:10:18 local; first 86.94 commit
   f1b02a36 at 23:33:03 local. ORDER OK (research 23:07 < contract 23:10 < code 23:33).
3. experiment_results_86.94.md present (9,677 B) + live_check_86.94.md (22,196 B).
4. Log-last: masterplan 86.94 status=pending (NOT yet done). harness_log has no
   `phase=86.94 result=` row. OK.
5. No verdict-shopping: EVIDENCE CHANGED. Commit 379be687 (2026-08-16 22:10:57Z) landed
   35 s before my WRITTEN stamp and touched 7 files incl. +179/-34 on the guard.

## B. DETERMINISTIC
- IMMUTABLE COMMAND: `verify_changelog_flip_86_91.py > /dev/null && echo green`
  -> "green", exit=0. REPRODUCED.
- verify_no_sliding_windows_86_94.py -> "ALL GREEN: 37 passed, 0 failed", exit 0.
  Matches Main's claim of 37.
- RUFF F821,F401,F811 over a DERIVED, non-empty 4-file scope
  (git diff HEAD + git ls-files --others + git diff f1b02a36~1..HEAD, sorted -u,
  NUL-delimited via xargs -0 to defeat the zsh word-split trap):
  backend/api/sovereign_api.py, scripts/qa/replay_changelog_rule_86_68.py,
  scripts/qa/verify_changelog_flip_86_91.py, scripts/qa/verify_no_sliding_windows_86_94.py
  -> "All checks passed!" exit=0.
- UNINTENDED PRODUCTION CHANGE: none from this step. The dirty tree
  (backend/api/sovereign_api.py + 5 frontend files) has mtime 2026-08-14 13:2x, TWO DAYS
  before this step's first commit. Pre-existing, unrelated (a "1y" sovereign window).

## C1 -- REPRODUCED INDEPENDENTLY, EXACTLY. **MET**
Mechanism re-derived live in my own shell at 2026-08-17 00:17:18 CEST:
    git rev-parse --since=2026-08-13  ->  --max-age=1786573038  ->  2026-08-13 00:17:18
i.e. the bare date carried MY current clock time onto the target date.
Every recorded figure that CAN be regenerated, regenerated exactly:
    git log a5cbfd67 --since=2026-08-13T00:00:00 | wc -l  -> 424  (claimed 424)
    git log 27f8c6f6 --since=2026-08-13T00:00:00 | wc -l  -> 428  (claimed 428)
    git log a5cbfd67 --since=2026-08-11T00:00:00 | wc -l  -> 766  (claimed 766)
    git log 27f8c6f6 --since=2026-08-11T00:00:00 | wc -l  -> 770  (claimed 770)
    git rev-list --count a5cbfd67..27f8c6f6                -> 4    (claimed 4)
    commits in the 08-13 band [22:50:20, 23:51:09) local   -> 20   (claimed 20)
    predicted bare 376 + 4 - 20 = 360, measured 360. Arithmetic closes, no residual.
The two runs are 1h00m49s apart, the bare counts DIFFER (376 -> 360, DOWN while the repo
GREW by 4), and the midnight-pinned form differs from both (424, 428). No figure is
pinned into the criterion. A1b's near-refutation control (08-11's drift band was
exhausted, so the obvious date would have shown NO change) is a real finding and is
disclosed rather than smoothed.

## C2 / C3 -- enumeration rule + known-member recall
Rule is written down in source (verify_no_sliding_windows_86_94.py:70-94) with the
REPRODUCIBLE/SLIDING criteria and the widening rationale. Enumeration output quoted:
  backend/slack_bot/scheduler.py:503 'midnight' -> ALLOWED
  scripts/harness/frontend_route_inventory.py:73 '30.days' -> ALLOWED
  scripts/qa/replay_changelog_rule_86_68.py:114 '{CORPUS_SINCE}' -> REPRODUCIBLE
  scripts/qa/verify_decision_log_86_97.py:360 '{first_stamp}' -> ALLOWED
[1] known-member recall from git blob 06c3265f finds :72 '2026-08-11' -> SLIDING, and
the gate FAILS (not skips) if the blob is unrecoverable. HARD GATE, verified live.

## C4 -- verified by re-derivation. **MET**
The archived quote claim REPRODUCES:
  handoff/archive/_quarantine_2026-04-21/phase-3.7.5-v22/experiment_results.md
  :53 '"usage_source": "git_activity_30d"'  :55 'every_route_has_usage_count | PASS
  (12/12 integer opens_30d)'  :62-63 '/portfolio 2 ... /login 1' -- all under a
  "## Success criteria alignment" heading. So the figures WERE quoted as evidence.
"Nothing live depends on them" also holds: handoff/frontend_usage.json was last written
2026-04-18 and has NO code consumer (git grep over backend/scripts/frontend/.claude
returns only masterplan .bak blobs).
NOTE (non-blocking): the denominator is stated two ways without a normalisation rule --
the allowlist entry says "55 files mention it", the executed [3b] prints 49. I measured
55 over all file types incl. this step's own artifacts and 51 excluding them; 49 is
.md-only excluding them. All three are true of different populations; none says which.

## C7 -- verdict semantics UNCHANGED. **MET**
Diff touches only scripts/qa/{replay_changelog_rule_86_68.py (1 char, "Z"),
verify_changelog_flip_86_91.py (comment only), verify_no_sliding_windows_86_94.py (new)}
plus handoff prose. Nothing in .claude/workflows/, .claude/agents/, or
scripts/harness/attempt_budget.py. No path by which a non-PASS becomes a PASS.

## C6 -- INDEPENDENT MUTATION MATRIX (sliced module, real scan_text driven)
Slice: exec'd chars 0..17777 of the guard (up to the "[1] THE RULE FINDS ITS OWN KNOWN
MEMBER" marker) so scan_text is the SHIPPED function, then PROVED the slice is live by
reproducing a shipped cell: baseline SLIDING=[] and the bare-date cell -> SLIDING at
:192. Control clean first.

  KILLED   M3  single-quoted `=` form, sliding value
  KILLED   M5  space form + `yesterday`
  KILLED   M6  space form + `30.days`
  KILLED   M7  `--since-as-filter=<bare date>`
  KILLED   M8  `--since='2026-08-11'` -> <unparsed>, fails CLOSED
  KILLED   M11 `git rev-list --since=<bare date>`
  SURVIVED M1  subprocess.run(["git","log","--since","2026-08-11"])   <-- ARGV-LIST FORM
  SURVIVED M1b subprocess.run(["git","log","--after","2026-08-11"])
  SURVIVED M9  ["git","log","--since",(datetime.now()-timedelta(days=30)).isoformat()]
  SURVIVED M2  sh('git log --since "$(date -d \'30 days ago\')"')
  SURVIVED M10 `--max-age=<epoch>` passed directly

M1/M1b/M9 are the finding. WINDOW_RE requires `[=\s]` IMMEDIATELY after the option name,
so `"--since", "2026-08-11"` (the char after `since` is a quote) never matches at all --
the site is not merely unclassified, it is INVISIBLE, so the module's fail-closed
`<unparsed>` path never fires. This is the repo's DOMINANT git idiom: the subject script
itself builds `_log_args = ["git","log",f"--since={CORPUS_SINCE}",...]` at
replay_changelog_rule_86_68.py:114 and the guard itself calls
`subprocess.run(["git","ls-files",...])`. One token's difference in spelling from the
code under test.
NOT covered by the stated residual, which reads "a space-separated window whose value
matches none of the known shapes is NOT detected" -- M1's value IS a known shape (a bare
date) and it is still missed, because the failure is in WINDOW_RE, not PLAUSIBLE_VALUE.
M2 IS covered by that stated residual. M10 is a scope choice (rule declares
since/until/after/before), undisclosed but minor.
NO LIVE SITE is currently missed: `git grep -nE '"--(since|until|after|before)"'` over
scripts/backend/.claude/hooks returns only two argparse DEFINITIONS
(census_qa_write_guard_log_86_31.py:64, rail_drop_rate.py:184), neither a git call.
So this is a FUTURE-introduction gap -- which is exactly what criterion 6 governs
("would go RED if a NEW ... window is INTRODUCED").
Not a recall failure against the filing: the audit_basis names --since/--until/--after/
--before but not the argv-list spelling. Independent finding, not a repeat.

## C5 -- THE CORRECTION STILL ACCOMPANIES INSTEAD OF REPLACING. **NOT MET**
Criterion 5, verbatim: "a correction must replace, not accompany".
Measured on the shipped tree. Every 86.94 "correction" is a parenthetical APPENDED at
the end of a paragraph while the defective present-tense text a few lines ABOVE survives
verbatim:
  handoff/harness_log.md:35557 note inserted; :35558 STILL reads
      "[2026-08-11T00:00:00 .. 8dc70502]"  -- the naive window, one line below the note.
  handoff/current/experiment_results_86.91.md:141 STILL reads
      "The replay now pins **both** ends -- `CORPUS_SINCE = "2026-08-11T00:00:00"`"
      -- PRESENT TENSE and NOW FALSE: the shipped constant is
      replay_changelog_rule_86_68.py:99 `CORPUS_SINCE = "2026-08-11T00:00:00Z"`.
      The correction sits at :152-155, eleven lines later.
  handoff/current/experiment_results_86.91.md:146 STILL reads
      "The reproducible figures are 707/251/9/11 over `[2026-08-11T00:00:00 .. 8dc70502]`"
  handoff/current/live_check_86.91.md:90-91 STILL reads
      "answered on a corpus pinned at BOTH ends -- `CORPUS_SINCE = "2026-08-11T00:00:00"`"
      -- present tense, now false; its correction is at :100, in a later subsection.
Main's own W1 row (experiment_results_86.91.md:319) asserts "Every superseded figure is
listed and REPLACED in both artifacts". Measured, the naive window token was replaced in
exactly ONE occurrence (:155, inside the added note itself) out of five present-tense or
window-spec occurrences across the three files Main lists as corrected.
The evaluator_critique_86.91.md exemption (:161,:263) is JUDGED SOUND and is NOT the
finding -- editing a returned verdict would falsify the record. The finding is the four
occurrences above, in files Main DID edit, where the wrong text survives beside the note.
This is the identical shape the criterion names and the shape cycle 1 already flagged;
the remedy widened the FILE SET (2 -> 5) without changing the correction SHAPE.

## Evidence-integrity spot checks
- 37 assertions / exit 0 / ruff clean: all three REPRODUCE.
- live_check §G "30 passed" vs my measured 37: the file was regenerated at 379be687 and
  now states 37 -- consistent. [to re-verify]
- contract_86.94.md H2 "766/846 both ends pinned" -- CONFIRMED MISLABELLED and now also
  STALE. The block is headed "both ends pinned" but the command shown has NO upper bound.
  Measured by me now: open-ended Oslo 776 / Seoul 856 (they have MOVED from 766/846
  because the window is open-ended); genuinely both-ends-pinned is a stable Oslo 707 /
  Seoul 787. The guard's own docstring :23-25 labels both pairs correctly; the contract
  does not. An unreproducible figure under a label implying reproducibility, inside the
  contract of the step whose thesis is exactly that. Flagged by cycle 1, still unfixed.

## §E OF THE live_check IS NOT REGENERATED -- and it re-asserts the killed C4 claim
Main's disclosure: "live_check §C/§E/§G REGENERATED from a fresh run". Measured:
379be687's hunks on that file are @@1,6 @@193,215(§C) @@252,259(§E, ONE line)
@@294,301(§G) @@356+(appendix). §C and §G WERE regenerated. §E was NOT.
  live_check_86.94.md:248-257, a block presented as `[3b]` output, says
      scheduler.py 37 | verify_decision_log_86_97.py 5 | frontend_route_inventory.py 0
  the SHIPPED guard, run by me at 2026-08-17T00:1x, prints
      scheduler.py 282 | verify_decision_log_86_97.py 6 | frontend_route_inventory.py 49
  and :262 STILL reads "Mentioned in **0** files outside this step's own artifacts, so no
  count from it is load-bearing. It *does* print per-route figures -- they are simply
  never quoted as evidence."
That sentence IS the cycle-1 falsified claim. The cycle-2 commit EDITED THAT EXACT LINE
(`:70,73` -> `:73`) and left the false half untouched. So the C4 correction landed in the
source allowlist and NOT in the artifact the masterplan's live_check field requires
("the per-member classification").

## WHY NOTHING CAUGHT IT: the [3b] predicate is vacuous (shape #8)
Predicate: `("Criterion 4" in _entry) or ("quoted" in _entry)`. Executed by me over five
wordings: the TRUE entry PASSES, the cycle-1 FALSE entry PASSES, the live_check's "never
quoted as evidence" wording PASSES, and the bare token "quoted" PASSES. Only an entry
with no judgement at all fails. The check proves a sentence was WRITTEN, never that it is
TRUE. Main DOES disclose this in the source comment ("The check enforces disclosure, not
absence"), so it is honest -- but it provides zero protection against precisely the
regression that occurred. WARN, not BLOCK, because the mention-site printout beside it is
a genuine audit surface.

## VERDICT REASONING
C1 MET (reproduced to the digit). C2 MET. C3 MET (hard gate). C7 MET.
C6 substantially met -- 37 assertions, 14 mutation cells, control observed clean FIRST,
paired negative controls, shell + docstring + resolver legs -- but 3 of my 11 independent
mutants survive in the repo's DOMINANT argv-list idiom and are NOT covered by the stated
residual. WARN.
C4 NOT MET -- correct in source, INVERTED in the required artifact, on a line edited this
cycle; and the "§E regenerated" disclosure does not reproduce.
C5 NOT MET -- the correction still ACCOMPANIES surviving false text in 4 measured
occurrences across 3 of the files listed as corrected; :141 is now false about the code.
Worst-of-N lenses: correctness OK, reproduce OK, scope-honesty FAILS (two disclosures to
the evaluator -- "§E regenerated", "the entry is rewritten to what is true" -- do not
reproduce against the shipped tree). min => FAIL.

COMPLETED: 2026-08-16T22:22:03Z
(NB: I first wrote a LOCAL time with a Z suffix here -- CEST 00:26 is UTC 22:26, not
 00:26Z. Corrected by re-reading `date -u`. Same class as the finding above: a stamp
 labelled with a zone it was not taken in.)
