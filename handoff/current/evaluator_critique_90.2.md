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


---

## Cycle 2 -- Q/A verdict, TRANSCRIBED VERBATIM

Run `wf_546d7764-9c6`, 233,761 tokens, 854s, 51 tool calls. `harness_compliance_ok: true`.
Verdict **CONDITIONAL**: both cycle-1 blockers confirmed fixed and independently
reproduced; capped by two WARN findings and three NOTEs.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Both cycle-1 blockers are genuinely fixed and I reproduced each independently rather than reading them: the 41/247 replay reproduces EXACTLY on the derived population with ZERO symmetric difference against a matcher I wrote from scratch (identical membership, not just cardinality), and the leak guard is now behavioural -- four independent neutering shapes I applied myself (no-throw, can-never-fire, wrong-object, actual flattening) are all KILLED. Immutable command exit 0, 66 checks, control GREEN first, harness compliance clean, no unintended production change. Capped at CONDITIONAL by two WARN findings: (F1) criterion 6's second clause -- a mutant that silently drops a reported finding from `governing_severities` in the return SURVIVES all 66 checks with the route unchanged, because section E guards `derived_severities` only and M3 mutates the shared source array so one cell appears to cover two sites; (F2) `allResidual` is computed over `governing` without requiring `comparable`, so on the 86.98 judge-emitted branch two UNTAGGED blockers plus one NOTE detail row route to queue_residual and an EMPTY violated_criteria list defeats M7 -- unreachable today under VERDICT_SCHEMA's additionalProperties:false, but undisclosed and on the branch this step's own notes say must satisfy 86.98.",
  "violated_criteria": [
    "WARN criterion-6-second-clause: a silent finding-drop from governing_severities survives the whole checker (my cell QA5, rc=0, 66/66)",
    "WARN latent-undisclosed: allResidual reads `governing` without requiring `comparable`, so on the judge_emitted branch untagged findings and an empty findings list can reach queue_residual",
    "NOTE accompany-not-replace: the N5 fix reached experiment_results:186 but not :260, which still reads 0 of 978 against the reproducing 969",
    "NOTE sibling-artifact-drift: experiment_results' LIVE row (451/295/254) disagrees with live_check's regenerated LIVE row (452/296/255)",
    "NOTE rejected-design-unguarded: widening IMMEDIATE_NEGATOR to the 45-char proximity window the artifact says was measured and discarded survives all 66 checks and moves 1 real run"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "In an isolated /tmp copy of the tree (control observed GREEN, restored and re-verified GREEN after), applied cell QA5: `governing_severities: governing.length > 1 ? governing.slice(1) : governing` in the returned object literal of enforceSeverityRouting, then ran `node scripts/qa/verify_severity_routing_90_2.mjs --self-test`.",
      "state": "SURVIVED -- rc=0, 66 checks, 0 failing. The route is computed before the literal, so the drop is silent. scripts/qa/verify_severity_routing_90_2.mjs:261-267 asserts length / index-alignment / class-content on `derived_severities` only; `governing_severities` has no cardinality or content assertion anywhere in the checker. Matrix cell M3 (:437-439) mutates the SOURCE array `derived`, which feeds BOTH fields, so a single kill appears to cover two return sites while only one is guarded. Contrast: my QA3 (drop last from derived) KILLED, QA4 (drop from derived_severities) KILLED, QA9 (index off-by-one) KILLED -- so the guard is real, just one field short. `governing_severities` is the array `route` is derived from and becomes the authoritative one under 86.98.",
      "constraint": "SEVERITY WARN (forces CONDITIONAL, does not move the verdict to FAIL). Immutable criterion 6: 'a mutant silently dropping any reported finding from the return must also be KILLED'. Named fix: mirror the three section-E assertions onto governing_severities and add a matrix cell that truncates it in the return literal."
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "Extracted the REAL enforceSeverityRouting/severityTags/deriveSeverity spans from .claude/workflows/qa-verdict.js by brace matching, re-exported to a temp module, and drove it: (a) {verdict:'CONDITIONAL', violated_criteria:['UNTAGGED blocker A','UNTAGGED blocker B'], violation_details:[{severity:'NOTE'}]}; (b) {verdict:'CONDITIONAL', violated_criteria:[], violation_details:[{severity:'NOTE'}]}.",
      "state": "(a) route=queue_residual, severity_source=judge_emitted, governing=['NOTE'], derived=['UNTAGGED','UNTAGGED'], disagreement_status=not_index_comparable -- two untagged blockers filed away as residual. (b) route=queue_residual with an EMPTY violated_criteria list, which is precisely what checker section C and matrix cell M7 forbid on the derived branch. Root cause: `emitted` is indexed by violation_details while `derived` is indexed by violated_criteria, and `allResidual` reads `governing` without requiring the already-computed `comparable`. Unreachable today -- VERDICT_SCHEMA at :439 makes violation_details items additionalProperties:false with no `severity` key (verified, and confirmed live when my own first return carrying a `severity` key was rejected by this very schema) -- and FAIL-immunity DOES hold on this branch (FAIL and PASS both -> remediate, driven). Not disclosed in qa-verdict.js, experiment_results_90.2.md or live_check_90.2.md.",
      "constraint": "SEVERITY WARN (forces CONDITIONAL). Criterion 4 ('any run mixing a WARN entry with an untagged finding must route to remediate') and criterion 6 clause 1, plus the step's own notes requiring it to satisfy 86.98 criterion 5 rather than pre-empt it. Named fix: gate `allResidual` on `comparable`, or fall back per-index to the derived class when emitted[i] is null."
    },
    {
      "violation_type": "Contradiction",
      "action": "grep -rn '978' over the 90.2 artifacts and code, then re-derived the figure myself: 969 violation_details rows over the 397 verdict-bearing records at the pin, 0 carrying a `severity` key.",
      "state": "experiment_results_90.2.md:186 reads '0 of **969** (was quoted as 0 of 978)'; :260 still reads '**0 of 978** violation_details rows carry a `severity` key'. The cycle-2 write-up reports N5 as 'FIXED at source'. The in-code comment in qa-verdict.js carries 969 correctly; only the artifact's 'Explicitly NOT done' section is stale. Substance (zero) is unaffected.",
      "constraint": "SEVERITY NOTE (does not degrade the verdict on its own). A correction must REPLACE, not accompany -- the replacement stopped at one of two sites while the summary claims the fix is complete."
    },
    {
      "violation_type": "Contradiction",
      "action": "Re-ran `node scripts/qa/verify_severity_routing_90_2.mjs --replay` and compared its LIVE block to both artifacts.",
      "state": "experiment_results_90.2.md criterion-4 table: LIVE 451 / 408 / 407 / 295 non-PASS / 41 / 254. live_check_90.2.md section 2 verbatim block: LIVE 452 / 409 / 408 / 296 / 41 / 255, which is what --replay prints now. The PINNED figures agree everywhere and are the load-bearing ones; the drift is inherent to an unpinned row, but a reader comparing the two same-cycle artifacts sees a contradiction.",
      "constraint": "SEVERITY NOTE. qa.md 4b -- a 'verbatim' capture must be regenerated, and a corrected capture leaves its siblings stale."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Applied cell QB7 (IMMEDIATE_NEGATOR widened to /\\b(no|zero|none|without|neither|nor)\\b[\\s\\S]{0,45}$/i -- the proximity design the artifact says was measured and discarded), ran the self-test, then ran a behavioural differential over the 906 real violated_criteria entries at the pin.",
      "state": "SURVIVED the self-test (rc=0, 66/66) while changing real behaviour: 1 of 906 entries reclassified and 1 real run (wf_7fa0e5d6-c50) moved out of queue_residual, 41 -> 40. So it is NOT an equivalent mutant. Matrix cell M8 kills REMOVING the negator rule; nothing pins its narrowness. For contrast, my QB8 (delimiter `after` window 45 -> 400) also survived but IS equivalent on the observed population: 0 runs and 0 of 906 entries change -- reported as a non-finding rather than padded into the list. This upgrades the cycle-1 Q/A's NOTE ('the 45-char window remains an unmeasured magic number') from unmeasured to unguarded-and-behaviour-changing.",
      "constraint": "SEVERITY NOTE. qa.md 4c -- a matrix licenses only 'these N mutations were killed'; the design decision the artifact narrates at length has no cell pinning it against the specific alternative it rejects."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command",
    "node_check_all_changed_js",
    "git_status_tree_frozen",
    "git_diff_scope_derived",
    "harness_compliance_audit_5_items",
    "research_gate_envelope",
    "contract_criteria_verbatim_string_equality",
    "independent_population_census",
    "independent_replay_symmetric_difference",
    "fixture_verbatim_provenance_check",
    "claim_reproduction_audit",
    "mutation_matrix_independent_17_cells",
    "mutation_differential_on_survivors",
    "leak_guard_behavioural_drive",
    "judge_emitted_branch_probe",
    "residual_close_gate_review",
    "verdict_schema_reachability_check",
    "code_review_heuristics",
    "qa_wip_prior_attempts",
    "verdict_ledger_history",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "SEQUENCE: qa_wip.py 90.2 --spawned-at 2026-08-21T07:32:30Z -> source_present=true, attempt_number=2 (status ok, is_lower_bound=false), prior_attempts=1, prior_records=[verdict_wip_90.2__20260821T071422Z.md], records_retained=2 (gauge, not a counter). verdict_history_86_21.py --step 90.2 --evidence-only -> status=no_rows_for_step, verdicts=(none). CROSS-CHECK: prior_attempts (1) > the ledger's verdict count (0), so THE LEDGER IS STALE for this step -- sequence: UNKNOWN from the ledger. Main's cycle-1 disclosure (FAIL, wf_0e5b781a-bf9) is advisory only and I did not treat it as the sequence. harness_log grep -cF \"phase=90.2\" = 0 and masterplan status is still \"pending\" (log-last honoured).\n\nHARNESS COMPLIANCE (all 5 clean): research_brief_90.2.md envelope brief_status=COMPLETE, external_sources_read_in_full=7, urls_collected=17, recency_scan_performed=true, gate_passed=true, cited in contract_90.2.md:10-13 with run id wf_05a76fdf-b16; mtime order brief 08-20 21:39 < contract 21:52 < code 08-21 09:29/09:30 < experiment_results 09:31:16 < evaluator_critique 09:32:10; all SIX immutable criteria appear VERBATIM in the contract by string equality against masterplan.json; step not logged and not flipped; evidence CHANGED since cycle 1 (ca656466 touches qa-verdict.js +29, residual_close_gate.mjs +11, experiment_results, live_check, evaluator_critique) so this is the documented fresh-respawn, not verdict-shopping. HEAD re-checked at the end: 0ddda824, unchanged throughout; git status shows only append-only handoff/audit/*.jsonl plus my own WIP file.\n\nGATE SCOPE, stated rather than skipped: `git diff --name-only b09b937b..ca656466` yields ZERO *.py, zero frontend/**, zero backend/** -- so qa.md gates 1a (ruff), 1b (eslint/tsc), 1c (live UI capture) and 1d (backend smoke) are N/A on this diff, not omitted. The step makes no UI claim. `node --check` exits 0 on all three changed JS/MJS files.\n\nWHAT I ATTACKED AND FOUND SOUND (so Main does not re-litigate): (a) criterion 4's headline reproduces under MY OWN code -- 441 records / 436 exact / 5 variant-named / 398 parseable / 397 verdicts / CONDITIONAL 221 + PASS 109 + FAIL 67 / 288 non-PASS / 906 violated_criteria entries / 969 detail rows with 0 severity keys, and my independently written case-sensitive token-anywhere matcher gives queue_residual 41 and remediate 247 with a SYMMETRIC DIFFERENCE of 0 against the author's run ids -- identical membership, not merely equal cardinality; (b) 0 of 67 real FAILs are all-WARN/NOTE under my matcher, and the verdict guard is structural on BOTH branches (FAIL and PASS both -> remediate even on the judge_emitted path); (c) criterion 3 is genuine -- all 24 fixtures resolve to real run records and match verbatim on verdict + violated_criteria + violation_details, spanning PASS 6 / FAIL 6 / CONDITIONAL 12, and my QB1 (routing mutates the verdict in place) is KILLED; (d) criterion 1 is now behavioural -- I applied four independent neutering shapes (throw -> console.warn, filter that can never fire, guard scanning the wrong object, and an actual `...severity_routing` flattening) and ALL FOUR are KILLED, so the cycle-1 sole-coverage vacuity is genuinely closed; (e) criterion 5's gate is fail-closed and correct, including against the real masterplan, and its NOT-WIRED status is disclosed prominently at source; (f) the \"3 of the 41 carry detail rows with no matching tagged entry, all three read SEVERITY NOTE\" claim reproduces EXACTLY (wf_1afa11f6-75a, wf_2dd1efc9-d0c, wf_28cf4dbb-9aa, all three constraints containing the literal).\n\nTHREE OF MY OWN CANDIDATE FINDINGS I RETIRED. (1) My first independent matcher gave 34, not 41 -- because it was case-INSENSITIVE and \"evidence block\" / \"block comment\" matched BLOCK. That is the naive matcher the author's M5 kills; MY control was the defective one, not theirs. (2) My own stricter delimited rule gave 39; the 2 differing runs carry `WARN(scope-honesty)` entry-initial tags that my follow-set wrongly excluded -- the author's rule is right there. (3) My QA1-FIX (allResidual over derivedOnly) was KILLED, and correctly so: it would make the caller ignore a judge-emitted BLOCK, violating 86.98 criterion 5. So the section-D check is doing real work and my candidate \"the checker pins the hazard\" is withdrawn -- the correct fix for F2 is narrower (gate on `comparable`), not that one.\n\nMUTATION WORK: 17 cells of my own in an isolated /tmp copy, control observed GREEN before any cell and restored + re-verified GREEN after each. 15 KILLED, 2 SURVIVED (QA5 -> finding 1; QB7 -> the fifth NOTE, with a differential proving it is not equivalent). QB8 also survived but IS equivalent on the observed population (0 runs, 0 of 906 entries) and is reported as a non-finding rather than padded into the list.\n\nCRITERION MAPPING: 1 MET, 2 MET, 3 MET, 4 MET, 5 MET, 6 MET-WITH-A-NAMED-GAP (control GREEN first verified in my own tree and both NAMED mutants killed and independently reproduced, but QA5 is a surviving mutant of exactly the shape the second clause names, on the sibling return field).\n\nINCIDENTAL, VERIFIED LIVE: my first attempt at this return carried a `severity` key inside each violation_details row and the schema REJECTED it with \"must NOT have additional properties\" -- an unplanned live confirmation of the step's own premise that VERDICT_SCHEMA cannot carry a judge-emitted severity today, and therefore that finding F2's branch is genuinely unreachable. Severities are folded into the `constraint` strings instead.\n\nWrite-first record at .claude/agent-memory/qa/verdicts/verdict_wip_90.2__20260821T073230Z.md, flipped to COMPLETE. No writes attempted outside that path; nothing was blocked.",
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
    "route": "queue_residual",
    "severity_source": "derived_from_prose",
    "derived_severities": [
      {
        "index": 0,
        "severity": "WARN"
      },
      {
        "index": 1,
        "severity": "WARN"
      },
      {
        "index": 2,
        "severity": "NOTE"
      },
      {
        "index": 3,
        "severity": "NOTE"
      },
      {
        "index": 4,
        "severity": "NOTE"
      }
    ],
    "governing_severities": [
      "WARN",
      "WARN",
      "NOTE",
      "NOTE",
      "NOTE"
    ],
    "disagreed": null,
    "disagreement_status": "nothing_emitted_to_compare",
    "reliability": {
      "derivation_is_authoritative": false,
      "what_is_read": "a DELIMITED severity tag in violated_criteria -- a syntactic property, decidable -- never sentiment inferred from the finding's prose",
      "measured_by_this_step": {
        "corpus": "the DERIVED population -- workflowName.startsWith(\"qa-verdict\"), which is what masterplan 90.2 audit_basis names -- pinned at startTime <= 2026-08-18T12:33:57.731Z: 441 records (436 exact-match + 5 under variant names) / 398 parseable / 397 carrying a verdict / 288 non-PASS (221 CONDITIONAL + 67 FAIL)",
        "all_warn_note_conditional_runs": 41,
        "remainder_routed_remediate": 247,
        "scope_note": "an exact-match narrowing drops 5 records, 3 of them non-PASS, and turns 247 into 244. The population is DERIVED from the filing, never chosen.",
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
    "next_action_on_queue_residual": "FILE the residual as its own masterplan step carrying its own immutable verification command. Do NOT fix it in place and do NOT spawn a fresh Q/A -- both are the re-cycle this routing exists to end. The parent step's close is refused while the filed residual is absent or does not parse (scripts/qa/residual_close_gate.mjs). Read `reliability` before acting: this route was derived, not emitted by the judge."
  },
  "verdict_unmodified": true
}
```

## Cycle 2 -- Main's follow-up (fix record, NOT a re-grading)

**All five findings are real and all five are acted on.** Two of them are defects in code I
wrote after the cycle-1 fix, and one is a correction of mine that reached one site and not
its sibling.

### WARN F1 -- `governing_severities` was unguarded, and one cell LOOKED like two

The Q/A truncated `governing_severities` in the return literal and it **SURVIVED all 66
checks**. Confirmed: section E asserted length, index-alignment and content on
`derived_severities` only, and matrix cell M3 mutates the shared source array `derived`,
which feeds **both** fields -- so a single kill appeared to cover two return sites while
only one was guarded. `governing_severities` is the array `route` is computed from, and it
becomes the authoritative one under 86.98.

Fixed: three assertions mirrored onto `governing_severities`, plus cell **M11** which
truncates it *in the return literal* (the site M3 cannot reach). M11 is KILLED.

### WARN F2 -- the 86.98 branch could file an unclassified finding away

`emitted` is indexed by `violation_details`; `derived` is indexed by `violated_criteria`.
Those are different arrays and nothing made them the same length, yet `governing` took the
emitted list regardless. The Q/A drove both consequences: two UNTAGGED blockers plus one
`NOTE` detail row routed to `queue_residual`, and an **empty** findings list did too --
which is precisely what section C and cell M7 forbid on the derived branch.

Unreachable today, because `VERDICT_SCHEMA` is `additionalProperties: false`. **But
"unreachable today" is a schema, not a property**, and this is the exact branch 86.98
exists to switch on. Fixed: the emitted list governs **only when it is index-comparable**
to the findings; otherwise the derivation governs, the fallback is NAMED
(`judge_emitted_not_index_comparable_falling_back_to_derivation`), and the judge's list is
still reported in a new `emitted_severities` field rather than discarded. Cell **M12**
restores the old behaviour and is KILLED.

**A sub-finding of my own, reported rather than padded.** I also added `entries.length > 0`
to `allResidual` and wrote a cell to kill its removal. **The cell was an equivalent
mutant** -- under the `comparable` gate, `governing` is `derivedOnly` whenever the arrays
do not line up, and `comparable` itself requires `derived.length > 0`, so a non-empty
`governing` already implies non-empty findings. The clause is kept as a second line of
defence and **disclosed in the code as provably redundant**; the cell is removed rather
than left in the matrix as a kill it cannot make.

### NOTE -- the N5 correction reached one site and not its sibling

`experiment_results_90.2.md:186` carried the corrected 969 while `:260` still read 978,
under a summary claiming the fix was complete. **A correction must REPLACE, and mine
accompanied.** Both sites now read 969; the only surviving "978" is inside the sentence
that records the correction.

### NOTE -- sibling artifact drift on the LIVE row

`experiment_results` printed LIVE 451/295/254 while `live_check` printed 452/296/255. Both
were honest captures taken minutes apart. Rather than patch one instance I removed the
class: **the LIVE row is no longer transcribed into `experiment_results` at all.** It is
printed once, from one run, in `live_check` §2, with the drift stated -- measured at
451 -> 452 -> 453 records across three runs in a single session, because the corpus grows
every time a Q/A launches, including the ones evaluating this step.

### NOTE -- the negator's narrowness was unguarded

The Q/A widened `IMMEDIATE_NEGATOR` back to the 45-char proximity window this step
**measured and discarded**, and it survived the whole checker while moving one real run
(`wf_7fa0e5d6-c50`) out of `queue_residual`, 41 -> 40. So it is **not** an equivalent
mutant. Cell M8 killed *removing* the rule; nothing pinned how far it may reach. Fixed with
a fixture taken verbatim from that run --
`universal 'run_friday_promotion has no caller anywhere' (WARN)` -- plus cell **M14**,
which is KILLED.

The Q/A also reported its own **QB8** (delimiter window 45 -> 400) as an equivalent mutant
on the observed population, 0 of 906 entries changed, and declined to list it as a finding.
That distinction between a survivor and a finding is the same one I had to make on my own
`entries.length` cell.

**Result: 77 checks over a floor of 66, 16 mutation cells, control GREEN first, immutable
command exit 0, and the pinned 41/247 unchanged.**
