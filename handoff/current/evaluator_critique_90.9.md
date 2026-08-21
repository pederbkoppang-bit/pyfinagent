# Evaluator critique -- step 90.9

## Cycle 1 -- Q/A verdict, TRANSCRIBED VERBATIM

`Workflow({scriptPath: '.claude/workflows/qa-verdict.js'})`, run `wf_fae0a8c6-4c1`,
212,538 tokens, 937s, 48 tool calls. `harness_compliance_ok: true`. Verdict **FAIL**.

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "Criteria 1/3/4/5/6 MET with independently re-derived evidence; criteria 2 and 7 NOT MET. (2) The immutable command EXITS 1 under the documented invocation: `source .venv/bin/activate && bash -c '<cmd>'` -> EXIT=1, \"2 problem(s): 2 unexpected cell score(s)\", QX and QXI both \"KILLED expected ERROR\" -- the exact clause criterion 2 forbids (\"a mutant that fails to run scores ERROR, never a kill\"). Deterministic and bifurcates on interpreter: no-venv /usr/bin/python3 3.9.6 -> EXIT 0 / ERROR 2 (reproduces Main's claim); venv Python 3.14.4 -> EXIT 1 / ERROR 0. Mechanism: Python 3.13+ colorizes tracebacks and FORCE_COLOR=3 is injected by the agent runtime (absent from ~/.zshrc/.zprofile/.zshenv/.bash_profile), so stderr carries '\\x1b[1;35mNameError\\x1b[0m: ' and score_error()'s literal `f\"{t}:\" in observed` (mutation_matrix_90_9.py:71-78) never matches; line 229 then falls through to KILLED. Fail-DANGEROUS -- it INFLATES kills 10->12. The in-run probe at lines 212-218 is built from hand-typed UNCOLOURED literals so it cannot represent the failure, and printed \"ok: the discriminator reads the TYPE, not the shape\" in the very run where both cells it certifies misscored. The docstring records the 90.1 cycle-5 lesson; shape-dependence was replaced by FORMAT-dependence one seam over -- a fresh instance of the defect class of queued step 90.12, which this step itself files. Neither artifact discloses the interpreter (grep for FORCE_COLOR|NO_COLOR|Python 3|interpreter|venv|3.9|3.14|environment -> 0 hits), so an environment-conditional green is presented as unconditional. (7) SURVIVING MUTANT, my own cell: QA-MUT-1c moved a live read of handoff/verdict_ledger.jsonl into an unlisted helper `_shape_hint()` that classify() calls on every classification; run exit 0, 0 FAIL lines, guard printed \"[PASS] no classification function references a verdict history ... neither handed in nor SELF-read\" and the I/O surface still listed only load_plan. Differential: the helper aborts if the read returns empty and \"MUTANT PROOF\" never appeared, so the read LANDED -- the mutant is live, not inert. Root cause: classifier_consequence_refs()/classifier_io_calls() (:501/:533) walk only the 8 bodies in CLASSIFIER_FNS (:493); the scope is NON-TRANSITIVE, so the scan is defeated by moving the read one function over. It is SOLE coverage for the binding \"never READS\" half (purity and signature checks both pass). M9 proves the scan runs, nothing proves its scope covers the path. Harness compliance CLEAN 5/5 (gate_passed=true, 8 sources, 53 URLs, recency scan; all 7 criteria verbatim in contract; research<contract<code<results; no harness_log row; status pending; cycle 1). No unintended production change: tree clean, subject md5s identical start and end.",
  "violated_criteria": [
    "criterion_2_error_control_scores_kill_not_error",
    "criterion_2_immutable_command_exit_nonzero",
    "criterion_7_consequence_scan_defeated_by_unlisted_helper",
    "illusory-guard"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "source .venv/bin/activate && bash -c 'python3 scripts/qa/criteria_shape_90_9.py --verify && python3 scripts/qa/mutation_matrix_90_9.py --verify'",
      "state": "EXIT=1. mutation_matrix_90_9.py --verify prints 'BAD QX KILLED expected ERROR', 'BAD QXI KILLED expected ERROR', 'KILLED 12 | SURVIVED 0 (excl. N0) | ERROR 0 | error controls: [KILLED, KILLED]', '2 problem(s): 2 unexpected cell score(s)'. Same command with no venv (/usr/bin/python3 3.9.6) exits 0 with ERROR 2. Subject md5 a7a501978673ec1f17c4684872b21b01 unchanged throughout.",
      "constraint": "Criterion 2: '...and a mutant that fails to run scores ERROR, never a kill'. The immutable verification command must exit 0."
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "score_error(observed) at scripts/qa/mutation_matrix_90_9.py:71-78 -- 'for t in UNRESOLVABLE: if f\"{t}:\" in observed'",
      "state": "Measured subprocess stderr under Python 3.14.4 with FORCE_COLOR=3 (injected by the agent runtime, absent from the user's shell profiles): '\\x1b[1;35mNameError\\x1b[0m: \\x1b[35mname \\'verify_v2\\' is not defined...'. The literal 'NameError:' never occurs, err is None, and line 229 (got = 'ERROR' if err else ('KILLED' if code != 0 else 'SURVIVED')) misscores the cell as KILLED. Direction is fail-dangerous: kill count inflates 10 -> 12.",
      "constraint": "The ERROR discriminator must key on the exception TYPE regardless of stream formatting. Decolorize (re.sub(r'\\x1b\\[[0-9;]*m','',stream)) or drive() with env NO_COLOR=1 / PYTHON_COLORS=0, and add a cell feeding score_error a COLORIZED stream."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "the in-run ERROR-discriminator probe at scripts/qa/mutation_matrix_90_9.py:212-218 (hand-typed 'swallowed' and 'domain' literals)",
      "state": "In the failing run the probe printed 'ok: the discriminator reads the TYPE, not the shape' while both cells it certifies (QX, QXI) misscored KILLED. The fixture is built from uncoloured literals and therefore CANNOT represent the state in which its subject fails.",
      "constraint": "qa.md 4c vacuity shape 5: a fixture that cannot represent the failure does not count as a guard. The probe must be exercised against the same byte stream the real drive produces."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "QA-MUT-1c (independent evaluator cell, run in the author's own build_sandbox/drive with FORCE_COLOR unset): insert `def _shape_hint(criterion): t = Path('handoff/verdict_ledger.jsonl').read_text(...); if not t.strip(): raise SystemExit('MUTANT PROOF: the ledger read did NOT land'); return len(t)` and call it unconditionally as the first statement of classify()",
      "state": "SURVIVED. exit=0, 0 FAIL lines. Section F printed '[PASS] no classification function references a verdict history, a WIP record, a round index or a remaining attempt budget -- neither handed in nor SELF-read (AST, scoped to the classification path)' and 'classification-path I/O calls, in full: load_plan: run(); load_plan: read_text()'. The 'MUTANT PROOF' abort never fired, proving the verdict-ledger read landed on every classify() call. Root cause: classifier_consequence_refs()/classifier_io_calls() (:501/:533) walk only the 8 function bodies named in CLASSIFIER_FNS (:493) and do not follow calls.",
      "constraint": "Criterion 7: 'the classifier is never given, and never reads, a step's verdict history, round index or remaining attempt budget -- asserted by a test over its inputs'. The scan is sole coverage for the 'never READS' half (the purity and signature checks both pass under this mutant), so per qa.md 4c sole-coverage vacuity is BLOCKING. Fix: walk the call graph transitively from the CLASSIFIER_FNS roots, or assert at RUNTIME that no path other than the plan of record is opened during a classification run, and add QA-MUT-1 as a permanent cell."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "qa_md_read_in_full_at_runtime",
    "write_first_wip_record",
    "prior_attempt_evidence_qa_wip",
    "verdict_ledger_evidence_only",
    "harness_compliance_5_item_audit",
    "research_gate_envelope_and_url_count",
    "contract_criteria_verbatim_7_of_7",
    "mtime_ordering_research_contract_code_results",
    "log_last_and_status_not_flipped",
    "immutable_verification_command",
    "interpreter_bifurcation_venv_vs_system",
    "force_color_root_cause_isolation",
    "syntax_ast_parse",
    "ruff_F821_F401_F811_derived_commit_scope",
    "ruff_default_note_only",
    "runtime_smoke_verify_census_stepfile",
    "independent_rederivation_155_980_at_pin",
    "independent_cli_exit_codes_step_file",
    "mutation_matrix_reproduction",
    "independent_mutation_QA_MUT_1_and_1c",
    "guard_vacuity_scope_analysis",
    "sha256_md5_containment_crosscheck",
    "masterplan_node_byte_identity_vs_HEAD",
    "queued_step_90_12_verified_in_plan",
    "head_recheck_mid_evaluation",
    "code_review_heuristics",
    "worst_of_n_lenses"
  ],
  "harness_compliance_ok": true,
  "research_needed": false,
  "notes": "sequence: no prior verdicts recorded for 90.9. verdict_history_86_21.py --step 90.9 --evidence-only returned status=no_rows_for_step, verdicts=(none), with its own caveat that nothing writes the ledger automatically so absence is weak evidence. qa_wip.py 90.9 --spawned-at 2026-08-21T09:59:22Z returned source_present=true, attempt_number=1 (status ok, is_lower_bound=false), prior_attempts=0, prior_records=[], records_retained=1 (the gauge, which is my own write-first record). Cross-check: prior_attempts (0) == ledger rows (0), so no staleness signal. Consistent with Main's cycle-1 disclosure.\n\nWIP record (crash-survival evidence, NOT a verdict): /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_90.9__20260821T095922Z.md, marked COMPLETE, COMPLETED: 2026-08-21T10:13:35Z.\n\nNo writes attempted outside the permitted verdicts/ path; nothing was blocked.\n\nTREE MOVED DURING EVALUATE (recorded, not held against the step): HEAD advanced 632a8fc3 -> a3a2d368 mid-evaluation via two masterplan-queue/changelog commits (0d59de57, a3a2d368), and .claude/masterplan.json md5 changed e50b33b9... -> 4bc0aad7... . I verified the 90.9 node is BYTE-IDENTICAL to HEAD (criteria not amended, status still pending) and both subject scripts' md5s were unchanged at the start and end of my evaluation (c2099832..., a7a501978...), so every finding rests on the bytes I measured. One side effect worth naming: the transient masterplan write made an unrelated matrix run report \"CONTAINMENT BREACHED\" -- that was the concurrent edit, not a containment failure of the tool.\n\nFAIRNESS / SCOPE OF THE FAIL. This is strong work and the FAIL is narrow. Five of seven criteria are MET on evidence I re-derived rather than read: my own independent walk of the masterplan at rev 252090a3 returns 156/987 nodes with id + dict verification + non-empty success_criteria in phases 86..90, and 155/980 excluding 90.9 -- exactly the filed pair, so criterion 1's step-inclusion rule is genuinely recovered and printed. I drove the real CLI as a separate process (--step-file /dev/stdin) on named steps: 86.47 -> returncode 2, 87.1 -> returncode 0, matching the property detector. Step 90.12 is really filed (status pending, P1, own verification command), verified by walking the plan rather than grepping. The author disclosed two of their own surviving mutants (M5/M6) and a real regex bug -- cm.start() < m.start() compared two matches anchored on the SAME quantifier so 0 < 0 was never true and the corpus-precedence branch had never discriminated anything -- fixed to compare noun positions, moving the pinned count 62 -> 56. Criterion 6 returns an honest NEGATIVE (the bound is recommended against). Criterion 1 prints 438/44.7% beside the filed 403/41.1% without editing the filed number and discloses that the NARROW variant's 2.21x falls outside the filed range. Criterion 4's write scan is whole-file and AST-level and catches Path.write_text, which is stricter than the criterion's two literal patterns -- it is NOT vulnerable to the move that defeats criterion 7's scan.\n\nON THE ENVIRONMENT QUESTION, stated precisely so it is not overclaimed: the trigger is FORCE_COLOR (or PYTHON_COLORS), not a TTY -- inside subprocess.run(capture_output=True) stderr is a pipe, so an operator running the command in a plain terminal without FORCE_COLOR would NOT see the failure. What makes this a defect rather than a quirk is that criterion 2 states an unconditional property, the implementation satisfies it only conditionally, the condition is undisclosed and unasserted, there is no guard that fails loudly when the discriminator can be defeated, and the failure direction converts ERROR into KILLED (it inflates the kill count rather than deflating it). qa.md vacuity shape 9 governs: a claim green in one shell and red in another is a finding to investigate, not to average away.\n\nREMEDIATION, both small and mechanical. F1: decolorize before typing the failure -- obs = re.sub(r\"\\x1b\\[[0-9;]*m\", \"\", out.stdout + \"\\n\" + out.stderr) -- and/or drive() with env={**os.environ, \"NO_COLOR\": \"1\", \"PYTHON_COLORS\": \"0\"}; then add a cell that feeds score_error a COLORIZED NameError stream so the probe at :212-218 can represent the failure it currently cannot; and disclose the interpreter in live_check_90.9.md. F2: make criterion 7's scan reach the whole classification path -- walk the call graph transitively from the CLASSIFIER_FNS roots, or assert at runtime (an open() shim / audit hook around a classification run) that no path other than the plan of record is opened -- and add QA-MUT-1 as a permanent cell.\n\nWorst-of-N lenses: correctness FAIL (two real defects), does-it-reproduce FAIL (the artifact's \"IMMUTABLE COMMAND EXIT: 0\" does not reproduce under the documented invocation), scope-honesty CONDITIONAL (points a-e are disclosed generously and the criterion-6 negative is genuinely honest, but the interpreter under which the green was captured is disclosed nowhere). verdict = min = FAIL.\n\nLint note (not part of the gate, NOTE severity only): ruff with the qa.md selection --select F821,F401,F811 over the derived commit scope (git show --name-only 4165e67d -- '*.py', xargs -0, non-empty set asserted: 2 files) printed \"All checks passed!\" exit 0. Ruff's default rule set reports 27 style findings (FURB167 re.I, PLW1510 missing check=, F541, ISC004, I001) which are outside the gate and did not affect this verdict. No frontend/** or backend/** files in the commit, so gates 1b and 1d are N/A; the step makes no UI claims, so gate 1c is N/A.",
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
    "research_needed": false,
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
      },
      {
        "index": 2,
        "severity": "UNTAGGED"
      },
      {
        "index": 3,
        "severity": "UNTAGGED"
      }
    ],
    "governing_severities": [
      "UNTAGGED",
      "UNTAGGED",
      "UNTAGGED",
      "UNTAGGED"
    ],
    "disagreed": null,
    "disagreement_status": "nothing_emitted_to_compare",
    "emitted_severities": null,
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
    "next_action_on_queue_residual": null
  },
  "verdict_unmodified": true
}
```

## Cycle 1 -- Main's follow-up (fix record, NOT a re-grading)

**Both findings are correct, and the first one invalidates green results I reported
earlier today across FOUR steps. That is the most important thing on this page.**

### Finding 1 -- my "EXIT 0" was environment-conditional and I never disclosed it

CLAUDE.md's first Critical Rule is *"Always `source .venv/bin/activate` before running
Python."* **I did not, on any of the runs I reported.** Measured now:

```
system python3            3.9.6
project venv python3      3.14.4
FORCE_COLOR=3             (injected by the agent runtime)
```

Python 3.13+ **colorizes tracebacks**, so a subprocess stderr carries
`\x1b[1;35mNameError\x1b[0m: ` and a literal `"NameError:" in stderr` **never matches**.
The direction is **fail-dangerous**: an ERROR becomes a KILL, and the kill count *inflates*.

**Measured across every immutable command I claimed green today, under the venv:**

```
BEFORE the fix          AFTER
90.1  : 0               90.1  : 0
90.12 : 1   <-- RED     90.12 : 0
90.9  : 1   <-- RED     90.9  : 0
90.3  : 0               90.3  : 0
```

**Two of the four were red under the documented invocation while I reported them as
green.** The Q/A found it on 90.9; 90.12 was mine to find and I had not. Fixed in all
four by stripping ANSI before reading the type *and* driving with colour disabled --
belt and braces, because either alone is a bet on an environment. Both invocations now
exit 0, so the bifurcation is closed rather than moved.

**This is the same defect class one seam over, for the third time.** 90.1 cycle 5: the
type was invisible because a fail-open handler swallowed the traceback. 90.3: invisible
because a nested subprocess boundary swallowed it. Here: invisible because the *format*
changed. **Shape-dependence became format-dependence.** And my in-run probe certified the
discriminator with hand-typed uncoloured literals — *a fixture that cannot represent the
failure is not a guard* — printing "ok: the discriminator reads the TYPE, not the shape"
in the very run where both cells it certifies misscored.

**A second correction it forced.** My check asserting QX2 scored ERROR in *both* columns
was asserting a property of my shell: under the venv the pre-fix baseline is blind to QX2
too. The BEFORE value is now reported, not asserted.

### Finding 2 -- criterion 7's scan is non-transitive, and it is sole coverage

The Q/A moved a live `handoff/verdict_ledger.jsonl` read into an unlisted helper that
`classify()` calls on every classification. **It SURVIVED** — exit 0, and the guard
printed *"no classification function references a verdict history ... neither handed in
nor SELF-read"* while the read was landing on every call. `classifier_consequence_refs`
walks only the eight bodies named in `CLASSIFIER_FNS` and **does not follow calls**.

It is sole coverage for the binding *"never READS"* half — the purity and signature
checks both pass under that mutant. **NOT FIXED HERE.** The remedy is a transitive
call-graph walk from the roots, or a runtime assertion that no path other than the plan
of record is opened during a classification run. That is a real build, not a patch, and
90.9's budget stands at 3 of 5.

### Disposition

**90.9 stays `pending`.** Criteria 1, 3, 4, 5 and 6 MET with independently re-derived
evidence; criteria 2 and 7 NOT MET. Criterion 2 is now fixed and re-verified under both
interpreters. Criterion 7 is open.
