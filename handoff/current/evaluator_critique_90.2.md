# Evaluator critique -- step 90.2

## Cycle 1 -- Q/A verdict, TRANSCRIBED VERBATIM

Launched `Workflow({scriptPath: '.claude/workflows/qa-verdict.js'})`, run
`wf_0e5b781a-bf9`, 210,030 tokens, 758s, 44 tool calls. `harness_compliance_ok: true`.

Reproduced below unedited. Main records the verdict; Main never authors it.

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "Criterion 4 is missed: the replay must prove \"the 41 ... route to queue_residual and the remaining 247 route to remediate\", but it prints 41/244 and asserts \"247 DOES NOT reproduce\". I re-derived it independently — over the population the masterplan audit_basis names (\"441 qa-verdict Workflow run records\", i.e. workflowName.startsWith('qa-verdict')) the census reproduces the filing byte-for-byte (441 records / 397 verdicts / PASS 109 / COND 221 / FAIL 67 / nonPASS 288) and driving the shipped enforceSeverityRouting over it yields queue_residual=41, remediate=247 EXACTLY. The shipped replay narrows the corpus with `if (wn !== 'qa-verdict') continue`, dropping 5 records (3 non-PASS: 247-3=244), and the stated cause (\"the gap lives in parseable: 43 of 436 records carry result:null\") explains a different gap — the replay's own header prints startsWith=441 exact=436. Second blocker: criterion 1's THROW clause has sole-coverage source-scan vacuity — Section I is four regexes over the workflow file and the guard is never executed; applied in-memory, a mutant making the throw unreachable and a mutant deleting the if/throw (message left in a comment) BOTH leave all four checks GREEN, and no matrix cell targets it. Criteria 2, 3, 5, 6 MET; harness compliance clean; no unintended production change; the product code itself is correct (I executed the leak guard: no-throw on the sibling shape, throws on flattening and on a colliding judge key).",
  "violated_criteria": [
    "criterion_4_replay_247_claimed_unreproducible_but_reproduces_exactly",
    "criterion_1_throw_clause_illusory_guard_source_scan_sole_coverage"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "node scripts/qa/verify_severity_routing_90_2.mjs --replay, then independent re-derivation over workflowName.startsWith('qa-verdict') at pin 1787056437731",
      "state": "Shipped replay filters `if (wn !== 'qa-verdict') continue` -> non-PASS=285, remediate=244, and experiment_results_90.2.md S3-C4 + live_check_90.2.md:92-93 state '\"247\" does not reproduce ... The number is not edited to match'. My re-derivation over the filing's own population: records=441 verdicts=397 PASS=109 COND=221 FAIL=67 nonPASS=288 -> queue_residual=41 remediate=247. Every filing figure matches byte-for-byte. The 5 dropped records are workflowName 'qa-verdict-writefirst-82-5' (x3) and '-82-7' (x2), verdicts PASS/COND/FAIL/COND/PASS; the 3 non-PASS all route to remediate. The stated cause (43 of 436 records with result:null) accounts for the 436->393 parseable gap, not the 397->392 verdict gap.",
      "constraint": "Immutable criterion 4: 'replay proof on the measured population: the 41 all-WARN/NOTE non-PASS runs route to queue_residual and the remaining 247 route to remediate, printed as a confusion table carrying the run ids'. The measured population is the one masterplan 90.2 audit_basis names: '441 qa-verdict Workflow run records'. qa.md 4b: scopes must be DERIVED, not chosen by the author; a claim that does not reproduce is a Contradiction finding."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "Applied two neutering mutants to an in-memory copy of qa-verdict.js and evaluated verify_severity_routing_90_2.mjs Section I's four predicates verbatim (no file written)",
      "state": "CONTROL: 4/4 PASS. L1 `if (false && leakedS.length > 0)`: 4/4 PASS. L2 if/throw deleted with the message left in a comment: 4/4 PASS. All four checks are regexes over wfSrc read from disk (`/const merged = ...severity_routing \\}/`, a negative flatten scan, `/phase-90\\.2 invariant violated: severity_routing fields leaked/`, and a literal pin of the `const leakedS = ...` line); the guard is never executed and no cell in the 11-cell matrix (N0/M1-M9/QX, all inside enforceSeverityRouting) targets it. A behavioural test IS feasible: extracting the guard span the same way severityTags is extracted, I drove it and it discriminates (sibling shape -> NO THROW; flattened -> THROWS; judge-emitted `route` key -> THROWS), so the no-observation-possible exemption does not apply.",
      "constraint": "Immutable criterion 1: 'the existing sibling-leak invariants are extended to THROW if any of its keys appear inside the judge's own object'. qa.md 4c: name the concrete mutation that makes the guard fail — two exist and both survive; sole-coverage vacuity on a behavioural criterion is BLOCKING. Skill heuristic #17 illusory-guard, shapes #3 (literal kept, behaviour stripped) and #8 (comment-token trap)."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "immutable_verification_command",
    "git_scope_derived_incl_untracked",
    "python_lint_gate_NA_zero_py_files",
    "frontend_lint_typecheck_NA",
    "backend_smoke_NA",
    "live_ui_capture_NA",
    "replay_mode_reproduction",
    "independent_corpus_re_derivation_both_populations",
    "fixture_verbatim_verification_24_of_24",
    "independent_mutation_battery_Q1_Q6",
    "survivor_behavioural_differential_906_entries",
    "leak_guard_source_scan_vacuity_probe",
    "leak_guard_behavioural_execution",
    "supporting_numeric_claim_reproduction",
    "code_review_heuristics",
    "prior_attempt_and_verdict_evidence"
  ],
  "harness_compliance_ok": true,
  "notes": "SEQUENCE: qa_wip.py 90.2 --spawned-at 2026-08-21T07:14:22Z -> source_present=true, attempt_number=1 (status ok, is_lower_bound=false), prior_attempts=0, prior_records=[], records_retained=1 (my own record). verdict_history_86_21.py --step 90.2 --evidence-only -> status=no_rows_for_step, verdicts=(none). prior_attempts (0) is not > the ledger count (0), so no staleness signal; the ledger simply has no rows for this step and says so itself that absence there is weak evidence. Consistent with Main's cycle-1 disclosure. harness_log grep for phase=90.2 = 0 rows and masterplan status is still \"pending\" (log-last honoured).\n\nHARNESS COMPLIANCE (all 5 clean): research_brief_90.2.md envelope brief_status=COMPLETE, external_sources_read_in_full=7, urls_collected=17, recency_scan_performed=true, gate_passed=true; mtime order brief 08-20 21:39 < contract 21:52 < code 08-21 09:08-09:12 < experiment_results 09:13; experiment_results + live_check present; step not logged and not flipped; cycle 1 so no verdict-shopping surface. git: only append-only handoff/audit/*.jsonl modified plus my own WIP file; commit c09bd96b touches exactly the 7 files the contract scopes. HEAD re-checked at the end: 4c449680, unchanged during evaluation.\n\nWHAT I ATTACKED AND FOUND SOUND (so Main does not re-litigate these): (a) the delimited-tag matcher shipped instead of the contract's negation-aware derivation is well argued and independently corroborated — the shipped matcher and the filing's own token-anywhere matcher produce identical run sets, 41/41, zero disagreement, at both censuses and under both populations; (b) \"32 does not reproduce\" STANDS and is population-independent — I ran six strict definitions over both populations and got 41/26/11/4/0 under each, never 32; (c) criterion 3 is genuinely behavioural — all 24 fixtures VERBATIM-match the real run records field-by-field (24/24), and my mutant Q1 (mutating verdict.verdict in place) differs on 21 of the checker's 47 inputs, so it would be killed; (d) criterion 6's matrix is honest — control green first, N0 SURVIVED, QX ERROR, no NO-OPs, and my independent Q4/Q5/Q6 are all killed; (e) the residual_close_gate is fail-closed and correct, including against the real masterplan.\n\nTWO OF MY OWN CANDIDATE FINDINGS I RETIRED after a behavioural differential (a survivor is not automatically a finding): Q2 (delimiter `after` window widened 45 -> 400) and Q3 (non-string entry filter removed) both SURVIVE every one of the checker's 47 inputs, but on the real corpus each leaves queue_residual at 41 with an EMPTY symmetric difference and changes 0 of 906 real entries. Equivalent mutants on the observed population. The 45-char window remains an unmeasured magic number; that is a NOTE, not a violation.\n\nNOTE-LEVEL, non-blocking: (N1) residual_close_gate.mjs has no caller outside its own checker (grepped .claude/, scripts/, docs/, CLAUDE.md) — criterion 5 only asks for \"a checker\", so it is MET, but the artifact calls it \"the consumer half ... refuses a parent step's close\" without disclosing that nothing in a close path invokes it. (N2) the routing classifies violated_criteria only; violation_details content is never scored — 3 of the 41 queue_residual runs carry unmatched detail rows, all three explicitly \"SEVERITY NOTE\", so there is no live counterexample, but the bound is undisclosed. (N3) wf_555a4380-3e8 routes to queue_residual carrying a kill-switch-behaviour finding; all three of its entries are judge-tagged [WARN], so the routing is faithful to 86.98's \"severity comes from the judge\", but an operator should see that a money-path finding sits inside the 41 before this routing obliges anything. (N4) the tag-form table (initial 41 / bracket 88 / paren 29 / colon 20 / dash 7, bare 12) publishes no reproducing command; my independent tally gives 41/91/37/1/5/2 — \"initial 41\" matches exactly, the rest depends on an unpublished precedence rule. (N5) supporting counts \"0 of 978 violation_details rows\" and \"0 of 66 FAILs\" measure 969 and 67 on the filing's population; the substance (zero in both cases) holds — the deltas are the same population narrowing as the headline finding, which is why it is worth fixing once at the source.\n\nNAMED FIXES: (F1) replay over the filing's population so criterion 4's 41/247 table is actually printed — or print both with the criterion's population named — and retract the \"247 DOES NOT reproduce\" claim together with its parseable/result:null attribution. (F2) add a behavioural cell that EXECUTES the leak guard (extract the `leakedS` span exactly the way severityTags is extracted; assert NO-THROW on the sibling shape, THROW on the flattened shape and on a judge-emitted colliding key) plus a matrix mutant that neuters the throw while leaving the scanned literals intact. (F3) disclose N1 and N2.\n\nWrite-first record at .claude/agent-memory/qa/verdicts/verdict_wip_90.2__20260821T071422Z.md, flipped to COMPLETE. No writes attempted outside that path; nothing was blocked.",
  "escalation": {
    "sequence_supplied": null,
    "sequence_status": "not_supplied",
    "consecutive_conditionals": null,
    "would_auto_fail": null,
    "attempt_number": null,
    "budget_exhausted": null,
    "max_attempts": 5,
    "burden_on": "the party departing from the computed escalation",
    "override": null,
    "override_reason": null,
    "judge_was_told_consequence": false,
    "judge_was_told_consequence_evidence": null
  },
  "research_routing": {
    "research_needed": null,
    "research_brief_spec": null,
    "next_action_on_research_needed": null
  },
  "severity_routing": {
    "route": "remediate",
    "severity_source": "derived_from_prose",
    "derived_severities": [
      {
        "index": 0,
        "severity": "UNTAGGED"
      },
      {
        "index": 1,
        "severity": "UNTAGGED"
      }
    ],
    "governing_severities": [
      "UNTAGGED",
      "UNTAGGED"
    ],
    "disagreed": null,
    "disagreement_status": "nothing_emitted_to_compare",
    "reliability": {
      "derivation_is_authoritative": false,
      "what_is_read": "a DELIMITED severity tag in violated_criteria -- a syntactic property, decidable -- never sentiment inferred from the finding's prose",
      "measured_by_this_step": {
        "corpus": "qa-verdict Workflow run records pinned at startTime <= 2026-08-18T12:33:57.731Z: 441 startsWith / 436 exact / 393 parseable / 392 carrying a verdict / 285 non-PASS",
        "all_warn_note_conditional_runs": 41,
        "remainder_routed_remediate": 244,
        "agreement_with_the_filings_token_anywhere_matcher": "EXACT -- identical run sets, zero disagreement in either direction",
        "discarded_alternative": "a proximity negation filter moved 6 runs and all 6 were false positives on inspection"
      },
      "measured_by_the_research_brief": {
        "agreement_pct": 40,
        "cohens_kappa": 0.129,
        "majority_class_baseline_pct": 56.7,
        "scope": "whole-record prose, which is a different and harder problem than reading a delimited tag out of violated_criteria; carried as the brief's measurement, not re-attributed to this step"
      }
    },
    "next_action_on_queue_residual": null
  },
  "verdict_unmodified": true
}
```

## Cycle 1 -- Main's follow-up (fix record, NOT a re-grading)

**Both blockers are correct. I reproduced each before changing anything.**

### Blocker 1 -- criterion 4's 247 reproduces; my claim that it did not was wrong

Verified independently:

```
workflowName variants under startsWith at the pin: {'qa-verdict': 436, 'qa-verdict-writefirst-82-5': 3, 'qa-verdict-writefirst-82-7': 2}
records= 441  with_verdict= 397
verdict mix: Counter({'CONDITIONAL': 221, 'PASS': 109, 'FAIL': 67})
the records `exact` DROPS: 5
    ('wf_2ec8140b-1a8', 'qa-verdict-writefirst-82-5', 'CONDITIONAL')
    ('wf_3545b96e-47d', 'qa-verdict-writefirst-82-7', 'FAIL')
    ('wf_7bc53984-97d', 'qa-verdict-writefirst-82-5', 'PASS')
    ('wf_5fd6e20a-12f', 'qa-verdict-writefirst-82-5', 'CONDITIONAL')
    ('wf_188cd9bc-c04', 'qa-verdict-writefirst-82-7', 'PASS')
non-PASS among the dropped: 3
```

247 - 3 = 244, exactly the number I published. The masterplan `audit_basis` names *"441
`qa-verdict` Workflow run records"* and **441 is the `startsWith` count**. I chose the
exact-match scope, then wrote a paragraph explaining the discrepancy instead of
questioning the scope -- and the explanation I offered (43 of 436 records carrying
`result: null`) describes the **parseable** gap, not the **verdict** gap. It was checkable
and it did not check out. **An explanation that survives because nobody tested it is not
evidence.**

Fixed at source: the replay derives the population from the filing, prints the variant
names and the exact-match count beside it, and now reports **41 / 247**. Every supporting
count that inherited the same narrowing was corrected too -- `violation_details` rows
0 of **969** (was 978), all-WARN/NOTE FAILs 0 of **67** (was 66).

### Blocker 2 -- the leak guard was source-scanned, never executed

Confirmed: section I was four regexes over the workflow file, and the Q/A's two neutering
mutants -- `if (false && leakedS.length > 0)` and deleting the if/throw with the message
left in a comment -- left all four GREEN. The guard span is now lifted into a callable the
same way `severityTags` is extracted, and driven: NO-THROW on the sibling shape, THROW on
the flattened shape, THROW on a colliding judge key, NO-THROW on an empty routing object.
The Q/A's own two mutants are matrix cells **L1** and **L2** and both are **KILLED**.

### The five NOTEs

- **N1** (close gate not wired) -- DISCLOSED at source in `residual_close_gate.mjs`.
- **N2** (`violation_details` content never scored) -- DISCLOSED at source in
  `qa-verdict.js`, with the measured 3-of-41 and the absence of a live counterexample.
- **N5** (supporting counts inherited the narrowing) -- FIXED at source.
- **N3** (a kill-switch finding sits inside the 41, `wf_555a4380-3e8`) -- recorded for the
  operator in `experiment_results_90.2.md`.
- **N4** (the tag-form table publishes no reproducing command, and the Q/A's independent
  tally differs on four of six rows) -- ACKNOWLEDGED, not silently reprinted. The table is
  labelled illustrative and rule-dependent; nothing shipped depends on it.

### What the Q/A found sound, recorded so it is not re-litigated

The delimited-tag matcher shipping in place of the contract's negation-aware derivation
(independently corroborated: identical run sets under both matchers and both populations);
"32 does not reproduce" (six strict definitions, both populations, never 32); criterion 3
genuinely behavioural (24/24 fixtures verbatim against the run records); criterion 6's
matrix honest (control green first, N0 SURVIVED, QX ERROR, no NO-OPs); and the
`residual_close_gate` fail-closed and correct against the real masterplan. The Q/A also
**retired two of its own candidate findings** after a behavioural differential showed them
to be equivalent mutants on the observed population -- a survivor is not automatically a
finding.
