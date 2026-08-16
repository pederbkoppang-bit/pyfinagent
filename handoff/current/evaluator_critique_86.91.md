# evaluator_critique -- phase-86.91

**Cycle 1 verdict: CONDITIONAL** · run `wf_96cff705-af0` · 1 agent · 37 tool uses
· 205,001 tokens · 679 s · rail `.claude/workflows/qa-verdict.js` launched by
**scriptPath**.

Main records the verdict; Main never authors it. The Q/A return value is
transcribed VERBATIM below, with no editorial edit and no paraphrase.

## Adjudications the Q/A was asked for, and gave

- **Criterion 3 ("the same 348-commit corpus") is SATISFIED, not missed.**
  Answering on a pinned deterministic replacement, with the drift disclosed and
  the criterion unamended, is correct: the named corpus is provably
  non-regenerable, a criterion cannot demand an impossible act, and every
  operative demand (three executed numbers, an accounted increase, each
  newly-bumping commit shown to have closed a step) is met. The Q/A confirmed the
  drift arithmetic independently.
- **Criterion 5 does NOT demand retro-bumping** the two swallowed versions -- it
  is a prohibition on hand-editing, not a mandate. Treating a rewrite of released
  version history as an operator call is defensible and disclosed.
- **`live_check` section 6's non-claim is CORRECT**: both 86.90 and 86.91 exist
  at `HEAD~1`, so the flip commit reads `flip_transitioned`. Withholding the
  created-and-closed claim is right, not a gap.

## The three WARN findings (all accepted; see the Follow-up section)

| # | Finding | Why it lands |
|---|---|---|
| W1 | `706 / 250 / 9 / 11` does not reproduce -- the Q/A measured `710 / 252 / 9 / 11` | `CORPUS_UNTIL = None` pins only the LOWER bound; the upper still floats with HEAD. In a step whose thesis is "that is a number about a clock", the reproducibility claim was itself unreproducible |
| W2 | Section `[5]`'s replay guards are pure substring scans -- **both** of the Q/A's replay mutants SURVIVED at 24/24 green | The instrument that PRODUCES criterion 3's three numbers had no behavioural guard |
| W3 | `QA-1` SURVIVED: deleting the `masterplan_unreadable_at_HEAD` reason leaves the guard green | The assertion is *named* "EVERY branch that returns 'none' sets a reason" but drives 3 of 4 |

The shipped hook is correct in all three cases -- W2 and W3 are **guard-coverage**
gaps, not behaviour defects. That distinction does not soften them: the criterion
they sit on is the one demanding the CLASS be closed rather than the instance.

---

## Verdict, VERBATIM

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 8 immutable criteria are substantively MET and independently re-derived: immutable command exit=0 (\"parses\"); criterion 1 reproduced with from-scratch code on e4f2e844 (86.86 before=None -> after='done', OLD newly_done=[]) and on 8b520f6c (86.81); the three-arm replay re-run gives SHIPPED 9 / FIXED 11 identical to Main's, with corpus 710 / OLD 252 vs Main's 706 / 250 explained exactly by the 4 commits that landed between the runs; PARKED 86.9/86.44 still 0 vs 13; guard 24/24 green with CONTROL first; masterplan byte-identical since c627a810 and no verdict altered. My own 13-cell mutation matrix (driving the SHIPPED checker against mutated hook/replay sources in memory) killed 10 cells including extraction-drift (QA-9 fails LOUD, so the ast extraction genuinely drives production) and the fixture mutation (QA-10 -> RED, the fixture discriminates); I also fault-injected the NEW _log_decision myself (mkdir->PermissionError, open->OSError) and never-raises holds. CONDITIONAL on three WARN findings, each an executed measurement rather than a reading: (W1) experiment_results:130-132 claims \"Anyone re-running it gets 706 / 250 / 9 / 11, today and next month\" -- I re-ran and got 710 / 252 / 9 / 11, because CORPUS_UNTIL = None pins only the LOWER bound while the upper bound still floats with HEAD, so in a step whose thesis is \"a number about a clock\" the reproducibility claim is itself unreproducible; (W2) section [5]'s replay guards are pure substring scans and BOTH of my replay mutants SURVIVED at 24/24 green -- QA-11 made newly_done_ids ignore count_created while keeping the literal, QA-12 restored the None exclusion reworded as not in (\"done\", None) -- leaving the instrument that produces criterion 3's numbers guarded only by text matching (vacuity shapes 3 and 2); (W3) QA-1 SURVIVED -- deleting the masterplan_unreadable_at_HEAD reason leaves the guard fully green while the assertion is named \"[2] EVERY branch that returns 'none' sets a reason\", i.e. 3 of 4 branches covered on the very criterion that says \"closed, not just this instance\". ADJUDICATION Main asked for on criterion 3: answering on a pinned deterministic replacement, with the drift disclosed and the criterion unamended, SATISFIES it and is NOT a miss -- the named 348-commit corpus is provably non-regenerable (git applies a bare --since at the current time of day; I confirmed the drift arithmetic independently), a criterion cannot demand an impossible act, and the operative demands (three executed numbers, an accounted increase, each newly-bumping commit shown to have closed a step) are all met; FIXED is a superset of SHIPPED by construction so no unreported \"lost\" set can exist. Criterion 5 does NOT demand retro-bumping: it is a prohibition on hand-editing, not a mandate, and treating rewriting released version history as an operator call is defensible and disclosed. live_check §6's non-claim is CORRECT and verified -- 86.90 and 86.91 both exist at HEAD~1, so the flip commit will read flip_transitioned; withholding that claim is right, not a gap. The +2/710 sample is small but the stronger evidence Main points to does hold: PARKED 86.9/86.44 bump 0 against 13 each, reproduced.",
  "violated_criteria": [
    "Overgeneralization: reproducibility claim 706/250/9/11 does not reproduce (measured 710/252/9/11; CORPUS_UNTIL=None)",
    "illusory-guard: replay-side guards are source scans -- QA-11 and QA-12 mutants SURVIVED",
    "Threshold_Not_Met: criterion-4 guard covers 3 of 4 none-branches -- QA-1 mutant SURVIVED"
  ],
  "violation_details": [
    {
      "violation_type": "Overgeneralization",
      "action": "python scripts/qa/replay_changelog_rule_86_68.py, re-run independently ~2h after Main's run",
      "state": "WARN. Measured corpus 710 / OLD 252 / SHIPPED 9 / FIXED 11 against the artifact's 'Anyone re-running it gets 706 / 250 / 9 / 11, today and next month' (experiment_results_86.91.md:130-132; same claim in live_check section 2). Delta = exactly the 4 commits that landed after Main's run, 2 of them phase-86.91 -> +2 OLD. replay_changelog_rule_86_68.py:100 carries CORPUS_UNTIL = None  # None = HEAD",
      "constraint": "qa.md 4b -- a numeric claim in a verbatim artifact must reproduce; the pin fixes the LOWER bound only, so the corpus still slides with HEAD. Named fix: pin CORPUS_UNTIL to a sha (or state the count as HEAD-dependent) and re-quote the numbers."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "in-memory mutation of scripts/qa/replay_changelog_rule_86_68.py, then run the shipped checker against it",
      "state": "WARN. QA-11 (return transitioned  # count_created ignored) -> checker exit 0, ALL GREEN 24 passed. QA-12 (created predicate restored as before.get(s) not in (\"done\", None)) -> checker exit 0, ALL GREEN 24 passed. verify_changelog_flip_86_91.py:256-264 guards the replay only with substring scans ('count_created' in replay; the literal None-exclusion string absent; 'CORPUS_SINCE' present)",
      "constraint": "qa.md 4c vacuity shapes 2 (scan defeated by rewording) and 3 (literal kept, behaviour stripped). The replay is the instrument producing criterion 3's three numbers and has no behavioural guard. Named fix: assert newly_done_ids() returns DIFFERENT results for count_created True vs False on a fixture, i.e. drive the predicate instead of scanning its text."
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "in-memory mutation QA-1: delete _FLIP_DECISION[\"reason\"] = \"masterplan_unreadable_at_HEAD\" from the hook, then run the shipped checker",
      "state": "WARN. Checker exit 0, ALL GREEN: 24 passed, 0 failed -- the mutant SURVIVED. verify_changelog_flip_86_91.py:191-192 asserts '[2] EVERY branch that returns none sets a reason -- none is left unrecorded' but iterates CASES = [no_flip, first_commit] plus a separately-asserted detector_error; the after-is-None branch is never driven. Shipped code is CORRECT (all four branches do set a reason) -- the gap is guard coverage, on the criterion that demands the class be closed, not the instance.",
      "constraint": "qa.md 4c -- a guard that cannot fail when its subject is broken does not count; a completeness claim requires a known-member recall test over all members. Named fix: add ('masterplan_unreadable_at_HEAD', before=any, after=None) to CASES and a 4th mutation cell deleting that assignment."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_ast_parse",
    "python_lint_gate_ruff_F821_F401_F811",
    "git_commit_scope_audit",
    "independent_reproduction_criterion_1",
    "replay_rerun_three_arm",
    "regression_guard_rerun",
    "independent_mutation_matrix_13_cells",
    "fixture_mutation",
    "extraction_drift_mutation",
    "fault_injection_log_decision",
    "decision_log_inspection",
    "changelog_provenance",
    "masterplan_state_diff",
    "prior_attempt_evidence_qa_wip",
    "verdict_ledger_evidence",
    "code_review_heuristics",
    "claim_auditing_4b",
    "guard_vacuity_4c"
  ],
  "harness_compliance_ok": true,
  "notes": "Prior-attempt EVIDENCE (reported, not aggregated): qa_wip.py 86.91 --spawned-at 2026-08-16T08:25:44Z -> source_present=true, attempt_number=1, attempt_number_status=\"ok\", attempt_number_is_lower_bound=false, prior_attempts=0, prior_records=[], records_retained=1 (gauge, includes my own write-first record). verdict_history_86_21.py --step 86.91 --evidence-only -> status=no_rows_for_step, verdicts=(none). Both sources agree at zero priors, so the sequence is EMPTY, not UNKNOWN, and the ledger is not stale for this step (it is stale in general -- its last row is 2026-08-11). Write-first record at .claude/agent-memory/qa/verdicts/verdict_wip_86.91__20260816T082544Z.md, marked COMPLETE; it is evidence, not a verdict. No write was blocked.\n\nHarness compliance detail: research_brief_86.91.md 21,062 B, brief_status COMPLETE, gate_passed true, 8 sources read in full (floor 5), 28 URLs (floor 10), recency scan performed, audit_class false; contract cites the researcher's I3/I5/ISSTA findings. mtime order research 09:58:08 < contract 10:14:17 < experiment_results 10:23:14 < live_check 10:24:31, with the fix commit at 10:23:32 local. 86.91 appears in no harness_log row and is still status=pending, so LOG-last and the flip are correctly pending.\n\nAdditional NOTE-level findings, none verdict-degrading: (N1) live_check_86.91.md:136 quotes \"(driving the SHIPPED detector, 74 lines extracted...)\" inside a block presented as verbatim; measured 109 at 8dc70502, at 952ed521, at HEAD and in the worktree, and 76 at 8dc70502~1 -- 74 reproduces at no committed state, so that capture is stale or edited. Everything else in the block reproduces byte-for-byte. (N2) Criterion 4's residual: three bash `exit 0` paths run BEFORE the python heredoc and write nothing -- subject matching ^chore: (auto-changelog|changelog drift), CHANGELOG.md absent, and no \"### Recent Activity\" section. The first fires on roughly every second commit here (the decision log holds 2 lines against 4 commits since the fix), so the log is not a census of hook invocations; the other two are the silent-swallow shape exactly (rename that header and every commit silently stops bumping with zero output). None of this is disclosed, and experiment_results section 5's \"An unexplained none is no longer expressible\" reads broader than what it proves, which is scoped to branches that RETURN \"none\". (N3) The shipped checker does not cover _log_decision, the new raise surface this step adds, so criterion 7's \"by injection, not by reading\" is met for the detector but was met for the new logging code only by MY injection. (N4) experiment_results section 1's Files-changed table omits three files riding in 8dc70502: evaluator_critique_86.86.md (named in the commit message; 0 deletions, a pure append of a fresh re-grade, so criterion 8 is intact), experiment_results_86.90.md and live_check_86.90.md (named nowhere; their 9 deletions are a filename-pointer correction, no verdict changed). (N5) My QA-5 cell (sentinel -> `is None`) was killed only by the anchor-uniqueness check, not behaviourally -- it is a semantically equivalent mutant on string-valued statuses, and I record the kill mechanism accurately rather than crediting a guard that did not act.\n\nBounded residual risk, inherent to the criterion rather than a defect: the fix's new bump surface is \"any step whose id first appears at HEAD already done\", so a step FILED as done for bookkeeping would now bump. Measured exposure on the pinned corpus is 2 commits in 710, both verified to have shipped real work, and the M2 over-crediting cell plus the PARKED 0-vs-13 result bound the 86.68 direction.\n\nNo UI claims in this step, so section 1c does not apply; no backend/** or frontend/** in the diff, so 1b and 1d do not apply. The peer session's uncommitted edits to backend/api/sovereign_api.py and frontend/src/* are confirmed NOT present in 8dc70502 / 952ed521 / c627a810 (git show --name-only).",
  "escalation": {
    "sequence_supplied": [],
    "sequence_status": "ok",
    "consecutive_conditionals": 0,
    "would_auto_fail": false,
    "attempt_number": 1,
    "budget_exhausted": false,
    "max_attempts": 5,
    "burden_on": "the party departing from the computed escalation",
    "override": null,
    "override_reason": null,
    "judge_was_told_consequence": false
  },
  "verdict_unmodified": true
}
```


---

# Cycle 2 verdict: CONDITIONAL

Run `wf_fa56f83d-814` · 48 tool uses · 228,382 tokens · 926 s · rail launched by
**scriptPath** · `verdict_sequence: ["CONDITIONAL"]` passed as DATA,
`attempt_number: 2`.

All 8 criteria MET in substance; harness compliance clean 5/5. Capped by three
WARN findings, **two of which are mutants the Q/A executed and watched SURVIVE**
-- all three inside the cycle-2 remediation surface itself.

| # | Finding | Why it lands |
|---|---|---|
| QA-C2-1 | `[5] the replay corpus is PINNED AT BOTH ENDS` is a **pure substring scan**. Replacing `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)` with `pass` keeps every scanned literal, leaves the guard at 31/31 green, and measurably unpins the corpus (707 -> 712). The header still prints `= 8dc70502` because `_end` is `rev-parse`d from the constant and never compared to `commits[0]` | **I fixed exactly this vacuity shape in W2 and left it on the W1 remediation, one guard over.** The guard for the cycle-1 finding was itself vacuous |
| QA-C2-6 | Every `[5]`/`[6]` fixture uses a SINGLE id (`86.86`). Narrowing the shipped predicate to `... and s == "86.86"` leaves all four `[5]` assertions green, including *"the two arms genuinely DISAGREE"* | That is precisely the shape **criterion 2 forbids**, on the instrument that produces criterion 3's numbers. WARN not BLOCK only because the production guard `[1]` does drive unrelated ids (`9.99`, `12.7`) |
| QA-C2-5 | `live_check_86.91.md` section 4's "verbatim" capture does not reproduce: it quotes `ALL GREEN: 24 passed`, "74 lines extracted", 3 cells and no `[6]` section, against today's 31 / 109 / 4+2 | Commit `98c5b6ab` updated section 2 of that file and left section 4 at the cycle-1 state. **A reader of the live_check alone would conclude W2 and W3 were never applied** |

## What the Q/A did beyond grading, and it is worth recording

It **closed criterion 5's un-driven half itself**, by exec'ing the hook's entire
heredoc in memory with `open` and `subprocess.run` stubbed (repo untouched):

- a created-and-closed step produces `reason=flip_created` **and writes
  `### v6.93.223`**;
- a chore commit writes no header;
- the actual 86.90+86.91 flip shape produces `reason=flip_transitioned`, which
  **confirms `live_check` section 6's prediction by execution** rather than by my
  reasoning;
- the None-exclusion mutant produces **NO header** -- so the mutation is killed
  end-to-end, not merely at `bump_type`.

## Answers to the four things I asked it to judge

1. **The source-derived denominator is sound in the DETECTING direction** -- adding
   a 5th literal `return "none"` branch turns the recall check RED -- **but it
   fails OPEN**: converting a branch to `_v = "none"; return _v` drops
   `_none_sites` from 4 to 3 and the check stays GREEN while the branch behaves
   identically. My artifact claim *"a future 5th branch fails the check instead of
   slipping past it"* is therefore stated **without its bound**.
2. **Pinning `CORPUS_UNTIL` is a FIX, not an evasion** -- the replay/live-gate
   division holds. But the pin's own guard is the vacuous one (QA-C2-1).
3. **The `[6]` cells DO discriminate today** -- all six mutants load cleanly and
   every probe returns True on a real behaviour change. But its negative control
   confirms a syntactically broken mutant would be scored KILLED with no probe
   running: `except: killed = True` is a latent design weakness, inert today.
4. **The +2 accounting still holds** at 707 / 251 / 9 / 11, verified commit by
   commit, both ids `done` today, both commits shipping real code.

Named fixes, all five accepted: regenerate live_check section 4 from a fresh run;
drive the corpus pin behaviourally (assert `commits[0]` equals the resolved
`CORPUS_UNTIL`) and mutate the append line; add a second unrelated created id to
the `[5]`/`[6]` fixtures; record THREW vs RETURNED instead of scoring exceptions
as kills; bound the "future 5th branch" claim to literal-constant returns.

---

## Verdict, VERBATIM

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 8 immutable criteria MET in substance and harness compliance is clean on all 5 items, but three WARN findings cap the verdict — two are mutants I EXECUTED and watched SURVIVE, all three inside the cycle-2 remediation surface itself. Deterministic: immutable command `bash -n .claude/hooks/post-commit-changelog.sh` -> `parses` exit=0; heredoc ast.parse OK (327 lines); ruff F821/F401/F811 on the git-DERIVED scope (`git diff --name-only a21a5889 HEAD -- '*.py'`, non-empty asserted, xargs -0) -> \"All checks passed!\" exit=0; `verify_changelog_flip_86_91.py` -> ALL GREEN 31/0 exit=0; `replay_changelog_rule_86_68.py` run TWICE by me, byte-identical, exit 0 -> 707 commits in [2026-08-11T00:00:00 .. 8dc70502], OLD 251 / SHIPPED 9 / FIXED 11 — W1's fix reproduces in MY environment. I independently RE-DERIVED criteria 1, 3, 6, 7 rather than reading them: e4f2e844 gives `86.86 before: None -> after: 'done'`, SHIPPED `[]`, FIXED `['86.86']`; 8b520f6c likewise for 86.81; both commits shipped real code (autonomous_loop.py +133 / qa-verdict.js +49) and both ids are `done` today, so the +2 is fully accounted; PARKED 86.9/86.44 still bump 0 vs 13 each under OLD, so 86.68 is not reintroduced. I also closed criterion 5's un-driven half myself by exec'ing the hook's whole heredoc in memory with `open` and `subprocess.run` stubbed (repo untouched): a created-and-closed step produces `reason=flip_created` AND writes `### v6.93.223`, a chore writes nothing, the actual 86.90+86.91 flip shape produces `reason=flip_transitioned` (confirming live_check §6's prediction by execution), and the None-exclusion mutant produces NO header — the mutation is killed end-to-end, not only at `bump_type`. WARN 1 (QA-C2-1): checker [5] \"the replay corpus is PINNED AT BOTH ENDS\" is a pure substring scan; replacing the single line `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)` with `pass` keeps every scanned literal, leaves the guard at 31/31 green, and measurably unpins the corpus (707 -> 712 today) while the header still PRINTS `= 8dc70502` because `_end` is rev-parse'd from the constant, never compared to `commits[0]` — vacuity shape #3 left on the W1 remediation, the same shape cycle-1's W2 killed one guard over. WARN 2 (QA-C2-6): [5]/[6] drive `newly_done_ids` with a single-id fixture (`86.86` only); narrowing the shipped predicate to `... is _ABSENT and s==\"86.86\"` (anchor unique) leaves all four [5] assertions green — including \"the two arms genuinely DISAGREE\" — so the mutant survives, and that is exactly the shape criterion 2 forbids, on the instrument that produces criterion 3's numbers. The production predicate IS class-tested ([1] drives 9.99 and 12.7), which is why this is WARN not BLOCK. WARN 3 (QA-C2-5): live_check §4's \"verbatim\" capture does not reproduce — it quotes `ALL GREEN: 24 passed`, \"74 lines extracted\", 3 [4] cells and no [6] section, against today's 31 / 109 / 4+2; 98c5b6ab updated §2 of that file and left §4 at the cycle-1 state, so the operator-facing gate artifact still lists the three substring scans cycle-1 killed. Answering Main's four asks directly: (1) the source-derived denominator is SOUND in the detecting direction — I added a 5th literal `return \"none\"` branch and the recall check goes RED — but it fails OPEN, not closed: converting one branch to `_v = \"none\"; return _v` drops `_none_sites` 4->3 and the check stays GREEN with the branch behaving identically, and the walk also covers the nested `_statuses` helper; the artifact claim \"a future 5th branch fails the check instead of slipping past it\" is therefore stated without its bound. (2) Pinning CORPUS_UNTIL is a FIX, not an evasion — the replay/live-gate division holds — but the pin's own guard is the vacuous one (WARN 1). (3) The [6] cells DO discriminate: all six mutants across [4] and [6] load cleanly and every probe returns True on a real behaviour change; my negative control confirms a syntactically broken mutant would be scored KILLED without a probe running, so `except: killed = True` is a latent design weakness that is inert today. (4) The +2 accounting still holds at 707/251/9/11, verified commit by commit above. Lens split (worst-of-N): correctness=PASS, does-it-reproduce=CONDITIONAL, scope-honesty=CONDITIONAL -> min=CONDITIONAL. Fixes: regenerate live_check §4 from a fresh run; drive the corpus pin behaviourally (assert `commits[0]` equals the resolved CORPUS_UNTIL, or mutate the append line); add a second, unrelated created id to the [5]/[6] fixture; record THREW vs RETURNED instead of scoring exceptions as kills; bound the \"future 5th branch\" claim to literal-constant returns.",
  "violated_criteria": [
    "illusory-guard [WARN]: corpus-pin check is a substring scan (QA-C2-1)",
    "illusory-guard [WARN]: replay predicate guard is single-id, the shape criterion 2 forbids (QA-C2-6)",
    "stale-verbatim-capture [WARN]: live_check section 4 does not reproduce (QA-C2-5)"
  ],
  "violation_details": [
    {
      "violation_type": "Circular_Reasoning",
      "action": "mutate scripts/qa/replay_changelog_rule_86_68.py:115 `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)` -> `pass`, then evaluate the guard at scripts/qa/verify_changelog_flip_86_91.py:335-338",
      "state": "guard predicate returns True on the mutant (checker stays ALL GREEN 31/0) because it only scans for the literals CORPUS_SINCE / 2026-08-11T00:00:00 / CORPUS_UNTIL and the absence of 'CORPUS_UNTIL = None'. Measured real effect: pinned corpus 707 commits vs unpinned 712, and the header still prints '= 8dc70502' because _end is rev-parse'd from the constant and never compared to commits[0]",
      "constraint": "SEVERITY WARN. qa.md 4c vacuity shape #3 (literal kept, behaviour stripped) -- a guard that cannot fail when its subject is broken does not count. This is the guard for the W1 remediation, i.e. the cycle-1 finding itself."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "mutate the shipped replay predicate to `if st==\"done\" and before.get(s, _ABSENT) is _ABSENT and s==\"86.86\"` (anchor uniqueness asserted), then re-run every [5]/[6] assertion in scripts/qa/verify_changelog_flip_86_91.py",
      "state": "all four [5] assertions stay GREEN -- extractable/runnable, True-arm ['86.86'], False-arm [], and 'the two arms genuinely DISAGREE' -- so the mutant SURVIVES the 31-green checker, while on an unrelated id the shipped predicate returns ['9.99'] and the mutant returns []. Every [5]/[6] fixture uses the single id 86.86.",
      "constraint": "SEVERITY WARN. Immutable criterion 2: 'a fix that special-cases 86.86 or any single step id rather than the CLASS fails this criterion'. newly_done_ids is the instrument that produces criterion 3's three numbers. WARN not BLOCK because the PRODUCTION guard [1] does drive unrelated ids 9.99 and 12.7."
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "re-run the command quoted verbatim in handoff/current/live_check_86.91.md section 4: `python scripts/qa/verify_changelog_flip_86_91.py`",
      "state": "artifact quotes 'ALL GREEN: 24 passed', '74 lines extracted', 3 [4] mutation cells and no [6] section; the fresh run gives 'ALL GREEN: 31 passed', '109 lines extracted', 4 [4] cells + 2 [6] cells and a [2] recall line. Commit 98c5b6ab updated section 2 of the same file (706/250 -> 707/251, BOTH-ENDS-PINNED header) and left section 4 at the cycle-1 state, so the live_check gate artifact still lists the three substring scans cycle-1 killed.",
      "constraint": "SEVERITY WARN. qa.md 4b -- a capture labelled verbatim must be regenerated, never carried forward; a reader of live_check alone would conclude the W2 and W3 remediations were never applied."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command",
    "syntax_ast_parse",
    "heredoc_python_parse",
    "python_lint_gate_ruff_F821_F401_F811_derived_scope",
    "git_scope_derivation_and_unintended_change_audit",
    "rerunnable_checker_verify_changelog_flip_86_91",
    "replay_reproducibility_two_runs_diffed",
    "criterion_1_independently_rederived_on_real_commits",
    "criterion_3_plus2_accounting_rederived_commit_by_commit",
    "mutation_matrix_independent_reexecution_6_cells",
    "mutation_new_cells_by_qa_corpus_pin_and_single_id_fixture",
    "negative_control_exception_scored_as_kill",
    "source_derived_denominator_recall_probe_literal_and_nonliteral",
    "hook_end_to_end_render_drive_in_memory_4_cases",
    "fault_injection_never_raises",
    "decision_log_on_disk",
    "changelog_hand_edit_audit",
    "masterplan_state_unchanged",
    "artifact_claim_reproduction_audit",
    "prior_attempt_and_verdict_ledger_evidence",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "SEQUENCE / ATTEMPT EVIDENCE (reported, not aggregated): `python scripts/qa/verdict_history_86_21.py --step 86.91 --evidence-only` -> status=`no_rows_for_step`, verdicts=(none), with the tool's own caveat that absence there is weak evidence because nothing writes the ledger automatically. `python scripts/qa/qa_wip.py 86.91 --spawned-at 2026-08-16T08:51:53Z` -> source_present=true, attempt_number=2, prior_attempts=1, attempt_number_status=ok, attempt_number_is_lower_bound=false, records_retained=2 (gauge, not a counter), identity_checked=true, one prior record verdict_wip_86.91__20260816T082544Z.md. CROSS-CHECK: attempt_number (2) > the ledger's verdict count (0), so THE LEDGER IS STALE and the sequence from it is unreliable. Main's advisory disclosure (cycle 1 = CONDITIONAL, run wf_96cff705-af0) is consistent with attempt_number=2 but is advisory only; I did not scan prior_records bodies for verdict words. `grep -F 86.91 handoff/harness_log.md` returns zero hits, which is correct at EVALUATE time since LOG runs after.\n\nHARNESS COMPLIANCE, all 5 clean: (1) research_brief_86.91.md exists, 21,062 chars, gate_passed=true RECOMPUTED by research-gate.js, 8 sources read in full vs floor 5, 28 URLs vs floor 10, recency scan performed, run wf_6f758470-f84 cited in the contract's references. (2) mtime chain research 09:58:08 < contract 10:14:17 < hook 10:14:54 < checker/replay 10:42:57 < artifacts 10:49/10:50 -- and criterion 1's reproduction is quoted IN the contract, i.e. before the hook edit. (3) experiment_results_86.91.md present with a cycle-2 Follow-up section. (4) log-last respected: not in harness_log, masterplan 86.90 and 86.91 both still `pending`. (5) not verdict-shopping: 98c5b6ab changed experiment_results (+79), live_check (+24), the replay (+25) and the checker (+127) -- the documented cycle-2 flow on CHANGED evidence.\n\nWHAT I VERIFIED RATHER THAN ACCEPTED, per Main's ask: the claim that the hook itself is UNCHANGED this cycle is TRUE -- `.claude/hooks/post-commit-changelog.sh` appears in 8dc70502 only, not in 98c5b6ab. All three cycle-1 findings were indeed in the artifacts and guards, not in shipped behaviour.\n\nEVIDENCE I BUILT MYSELF (disclosure): the end-to-end render proof for criterion 5 is MY execution, not the author's -- I exec'd the hook's python heredoc in memory with `open` and `subprocess.run` stubbed, so the repo was not touched, nothing was written and nothing committed. The author's evidence chain stops at `bump_type`; the operator-facing proof is deferred to the flip and live_check section 6 is explicitly PENDING. Its stated prediction that the flip will read `flip_transitioned` and NOT `flip_created` is correct -- I confirmed it by driving the actual 86.90+86.91 pending->done shape.\n\nNO UI CLAIMS in this step, so gate 1c is not applicable and no Playwright capture was taken. No backend/** or frontend/** in the derived scope, so 1b and 1d are not applicable.\n\nFLAG FOR MAIN, not a 86.91 violation: the working tree carries UNCOMMITTED, UNRELATED production edits (the sovereign `1y` window: backend/api/sovereign_api.py plus five frontend components, 155 insertions). They are in neither 86.91 commit and are not attributable to this step, but `git add -A` in auto-commit-and-push.sh would sweep them into the flip commit under 86.91's subject.\n\nWRITE-FIRST RECORD: .claude/agent-memory/qa/verdicts/verdict_wip_86.91__20260816T085153Z.md, marked COMPLETE at 2026-08-16T09:04:11Z. It is a crash-survival record and is NOT a verdict. One blocked write to report, handled per qa.md rather than worked around: qa-write-guard.sh correctly denied a probe script I tried to place in the session scratchpad, so I ran every mutation and probe through `python - <<heredoc` instead; no repo file was modified by me at any point.",
  "escalation": {
    "sequence_supplied": [
      "CONDITIONAL"
    ],
    "sequence_status": "ok",
    "consecutive_conditionals": 1,
    "would_auto_fail": false,
    "attempt_number": 2,
    "budget_exhausted": false,
    "max_attempts": 5,
    "burden_on": "the party departing from the computed escalation",
    "override": null,
    "override_reason": null,
    "judge_was_told_consequence": false
  },
  "verdict_unmodified": true
}
```
