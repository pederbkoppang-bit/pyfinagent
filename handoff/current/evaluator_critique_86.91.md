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
