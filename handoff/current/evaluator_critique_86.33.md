# Evaluator critique -- step 86.33

**WRITTEN LATE, AND THAT IS ITSELF A PROTOCOL MISS.** I went from each verdict
straight to remediation without transcribing it here, so this file did not exist
until after cycle 3. The five-file protocol requires the verdict be recorded; all
three are transcribed below from their run journals, verbatim and in order. Same
class as 86.7's missing harness_log row, caught the same way -- by a later step
tripping over the absence.

Graded history: **c1 CONDITIONAL, c2 CONDITIONAL, c3 PASS**. The 3rd-CONDITIONAL
auto-FAIL rule was armed going into cycle 3 and did not fire, because the verdict
was not CONDITIONAL.

---

# CYCLE 1 -- verdict: CONDITIONAL

**`ok: False` | violated: ['criterion_2_answer_absent_from_experiment_results', 'criterion_1_partition_arithmetic_and_unstated_normalization'] |
harness_compliance_ok: True** | run `wf_87bc566d-64d`

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Harness compliance 5/5 clean, immutable command exit=0 (\"guard-parses\"), ruff clean, no unintended production change, and criteria 1/3/4/5/6 independently REPRODUCED by me (researcher prover 34 ALLOW + control BLOCK rc=2; mutation matrix control-green-first with M1/M2/M3 all KILLED and restore byte-identical; guard md5 f90a01405e3f21577695ee4fedb800a2 unchanged before/after every run). I settled criterion 2 myself: my own Write drove the real hook and the INSTALLED platform sent 12 keys [agent_id, agent_type, cwd, effort, hook_event_name, permission_mode, prompt_id, session_id, tool_input, tool_name, tool_use_id, transcript_path] with agent_type='qa', agent_id='afd21026f4056c9e0' -- versus 10 keys with agent_type AND agent_id ABSENT on Main-shaped rows -- so the ANSWER IS NO: there is exactly one caller-chosen role field plus an opaque instance id, and no field separates TYPE from NAME. CONDITIONAL (not PASS) because that answer exists only in the log and in this verdict: experiment_results_86.33.md:60-64/:165 and live_check_86.33.md:55 all still say the real answer is PENDING, so criterion 2 has NO covering evidence in the handoff record (qa.md section 4 contract-completeness CAPS the verdict). Not FAIL: the instrument was built correctly and the measurement now reproduces (two real rows, 12:59:29.967328Z and 13:06:06.080792Z). Second finding: criterion 1's sentence \"72 distinct ... covers 34 ... misses 37\" is arithmetically impossible (34+37=71); derive_agent_type_population_86_33.py:154 silently drops the EMPTY agent_type (2072 rows, the largest bucket) from the denominator via `if t` while the headline 72 includes it, and \"quality-auditor (11)\" measures 21 now. Criterion 5 is correct RESTRAINT, not evasion, and I proved it rather than accepting it: a 612-cell differential (17 agent_types x 9 paths x 4 tools) between the pre-step guard d23a981e and HEAD found 0 decision differences, with the probe first shown DISCRIMINATING (15 ALLOW / 6 BLOCK) so the zero is not vacuous. Main's correction of the research gate also stands: SyntaxError=5, NameError=1, Traceback=1, all beside /T/qa_guard_probe_z02c0bak/g.sh and pythonNOPE3 -- probe artifacts, not silent production breakage; the brief's \"32\" was its own count of unparseable LINES re-quoted as TRACES.",
  "violated_criteria": [
    "criterion_2_answer_absent_from_experiment_results",
    "criterion_1_partition_arithmetic_and_unstated_normalization"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "Map immutable criterion 2 to its covering evidence in handoff/current/experiment_results_86.33.md",
      "state": "experiment_results_86.33.md:60-64 says 'THIS IS NOT YET THE ANSWER, AND I WILL NOT PRESENT IT AS ONE', :165 says 'Not that criterion 2 is fully answered', and live_check_86.33.md:55 says 'the REAL answer is pending'. Only the 8-key synthetic echo is recorded. The real 12-key platform payload (agent_type='qa', agent_id='afd21026f4056c9e0', rows 2026-08-11T12:59:29.967328Z and 13:06:06.080792Z) exists in handoff/logs/qa_write_guard.log and in this verdict, but nowhere in the handoff artifacts. SEVERITY: WARN.",
      "constraint": "qa.md section 4 'Contract completeness (phase-71.3)': EVERY immutable criterion must map to covering evidence in experiment_results.md; a criterion with no covering evidence is a Missing_Assumption that CAPS the verdict at CONDITIONAL."
    },
    {
      "violation_type": "Contradiction",
      "action": "Read the criterion-1 population claim at experiment_results_86.33.md:16-18 and re-derive every number in it",
      "state": "Prose asserts '72 distinct agent_type values. A qa-role prefix match covers 34 of them and misses 37' -- 34+37=71, not 72. Mechanism reproduced at scripts/qa/derive_agent_type_population_86_33.py:154, `evade = sorted(t for t in types if t and t not in qa_like)`: the `if t` clause drops the EMPTY agent_type (2072 rows, the single largest bucket, = Main-shaped writes) from the 'NOT matched' denominator while the headline 72 counts it. Same sentence's 'quality-auditor (11)' re-measures as 21 (last 10 rows landed 12:58:34Z, 26s AFTER experiment_results was written, from Main's own live_check run; the log is gitignored so no committed baseline exists). Separately the script REIMPLEMENTS the guard predicate as startswith(('qa-','qa_','QA-','QA_')) instead of the guard's `.lower()` form at qa-write-guard.sh:121 -- divergent on input 'Qa-Mixed', which the guard MATCHES but the script would report as EVADING. SEVERITY: WARN.",
      "constraint": "qa.md section 4b claim-auditing: every numeric/set-membership claim must reproduce, and a ratio must be stated with its normalization rule; a partition that omits its largest member without disclosing the rule is a Contradiction."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "immutable_verification_command_exit_0",
    "git_status_no_unintended_production_change",
    "python_lint_gate_ruff_F821_F401_F811_derived_scope_37_files",
    "criterion_1_rerun_derivation_script",
    "criterion_1_independent_population_rederivation",
    "criterion_2_real_platform_payload_measurement_this_spawn",
    "criterion_3_rerun_researcher_rail_prover",
    "criterion_4_rerun_mutation_matrix_control_green_first",
    "criterion_5_independent_612_cell_failclosed_differential",
    "criterion_5_probe_discrimination_control",
    "criterion_6_apostrophe_audit_of_single_quoted_body",
    "guard_md5_before_after_every_run",
    "research_gate_envelope_verification",
    "gate_syntaxerror_claim_recount",
    "3rd_conditional_harness_log_grep",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "CYCLE 1 -- zero `phase=86.33 result=` lines in harness_log.md, no prior evaluator_critique_86.33, masterplan status=pending, retry_count=0/max 3. The 3rd-CONDITIONAL auto-FAIL rule does not bind; certified_fallback=false. REMEDIATION (mechanical, no code change needed): R1 transcribe into experiment_results §2 and live_check §4 the real 12-key subagent payload, the 10-key Main-shaped payload with agent_type/agent_id ABSENT, and the answer NO (one caller-chosen agent_type plus an opaque agent_id; nothing separates TYPE from NAME; the four keys the synthetic probe lacked -- effort, prompt_id, tool_use_id, transcript_path -- are none of them role attributes), citing both log rows. R2 restate criterion 1's partition with its rule: 72 = 34 qa-role + 37 other named + 1 EMPTY (2072 rows), and date-stamp or drop the frozen quality-auditor figure. R3 optional: have the derivation script reuse the guard's lowercasing predicate instead of reimplementing it. CREDIT WHERE DUE, and it is unusual: Main refused to present its own synthetic echo as criterion 2's answer, disclosed three of its own wrong measurements (24/68, 23/33, 85%), reported that its first keyset_recorded() check was RED against a HEALTHY guard and aborted the matrix rather than scoring three cells on a broken instrument, reported M3's actual red set (4 legs) exceeding its expected (1) instead of trimming it, and declined to choose on ASK #6. Criterion 5 is genuine restraint: the criterion literally instructs that a fail-closed change NOT be shipped, and my differential confirms none was. NO UI claims in this step so qa.md 1c does not apply; the hook was exercised live (by the provers and by my own real Writes) which satisfies the runtime-smoke intent of 1d. NO WRITE WAS BLOCKED -- the only file I wrote is the permitted WIP record at /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.33__20260811T125920Z.md, which is also, incidentally, the instrument that produced criterion 2's measurement."
}
```

---

# CYCLE 2 -- verdict: CONDITIONAL

**`ok: False` | violated: ['criterion_1_census_not_in_handoff_record', 'live_check_missing_required_census_element'] |
harness_compliance_ok: True** | run `wf_b8f1381f-fca`

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Both cycle-1 remediations VERIFY and criteria 2/3/4/5/6 are MET, but criterion 1's record is still incomplete: the masterplan live_check field requires live_check_86.33.md to carry \"the re-derived agent_type census (full distribution + outside-memory-dir counts)\" and it carries none, while immutable criterion 1 also requires per-value outside-.claude/agent-memory/qa/ counts, the derived-class recall (20 events / 10 identities) and a --before cutoff with its excluded-row count -- none of which derive_agent_type_population_86_33.py produces and none of which experiment_results_86.33.md cites, even though scripts/qa/census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z produces all of it and reproduced for me today (rows counted 3012, excluded 6866, 113 events / 69 outside, \"20 events across 10 identities\", exit 0). R1 VERIFIED: the two cited log rows reproduce byte-exact (12 keys, agent_type='qa', agent_id='afd21026f4056c9e0') and I reproduced the answer INDEPENDENTLY -- my own Write logged the identical 12-key set with agent_type='qa' and a DIFFERENT agent_id 'a88e054bb964e863b', confirming one caller-chosen role field plus one opaque instance id; the 3 Main-shaped rows carry 10 keys with both genuinely absent; the 'pending' sweep over contract/experiment_results/live_check returns ZERO hits and the old \"answer is pending\" block was removed, not left beside. R2 VERIFIED: 78 = 36 matched + 42 not-matched with EMPTY printed (2152), and the predicate now reuses the guard's own lowercasing form -- the divergence case is real, 'Qa-Mixed' exists in the log and is the ONLY value of 78 where the guard and the old reimplementation disagree. R3 disclosure ADEQUATE: perishability reproduces (quality-auditor 97, EMPTY 2152 vs the artifact's 2151 one hour earlier) and every figure is date- or rule-stamped. Deterministic: immutable command exit=0 \"guard-parses\"; ruff F821/F401/F811 over 15 git-derived .py files \"All checks passed!\" exit=0; both scripts exit 0; git status shows no unintended production change; and the guard is proven ALIVE, not merely parsing, because my own write produced a payload_keys row. Two NOTEs: commit 335257a8's message says the script \"asserts\" the partition sums when it only prints it (the sum is now guaranteed by construction, so an assert would be tautological -- the wording, not the fix, is wrong); and section 2's \"78 logged values\" quotes the contaminated unfiltered count where 65 is the real-spawn figure. One finding I RETIRED after indicting my own probe: filtering on platform-minted agent_id showed zero name-shaped agent_types, which looked like a falsification until I found the pre-contamination slice (3012 rows, 65 distinct types including qa-80-2-c2) carries no agent_id at all because this step's own P0 added the field.",
  "violated_criteria": [
    "criterion_1_census_not_in_handoff_record",
    "live_check_missing_required_census_element"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "read handoff/current/live_check_86.33.md (88 lines, sections 1-5) and grep it for census/distribution/outside-memory content",
      "state": "ZERO census content: no full agent_type distribution table, no per-identity counts of Write/Edit events targeting paths outside .claude/agent-memory/qa/. Verified absent in BOTH committed revisions (c88484be and 335257a8), so the remediation did not remove it -- it was never there. The four required elements are census / TYPE-vs-NAME probe / researcher-rail decisions / mutation cells; only the last three are present.",
      "constraint": "masterplan.json 86.33 verification.live_check: 'live_check_86.33.md with: the re-derived agent_type census (full distribution + outside-memory-dir counts), the probe output ..., the researcher-rail regression decisions verbatim, and the mutation cell output'"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "run scripts/qa/derive_agent_type_population_86_33.py and grep it for argparse/--before/outside/evaluator_critique; grep experiment_results_86.33.md for census_qa_write_guard_log_86_31 / '20 events' / 'outside'",
      "state": "The 86.33 derivation script has no argparse, no --before cutoff, no excluded-row count, no per-value outside-.claude/agent-memory/qa/ counts and no breach-class recall test; experiment_results_86.33.md cites only that script (grep for the 86.31 census returns rc=1). The covering derivation EXISTS and reproduces -- census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z prints rows counted 3012, rows excluded 6866, 27 named qa-role identities, 113 events of which 69 outside the memory dir, and 'NO-SELF-EVAL BREACHES ... 20 events across 10 identities', exit 0 -- but the 86.33 handoff record never points at it. This is the identical shape cycle 1 raised and Main accepted for criterion 2 (evidence lived only in the log and the verdict), surviving on criterion 1's other legs.",
      "constraint": "masterplan.json 86.33 success_criteria[0]: the derivation must report 'the FULL distribution plus, for every value, how many of its Write/Edit events targeted paths outside .claude/agent-memory/qa/', with recall 'VALIDATED AGAINST THE DERIVED CLASS' (20 events across 10 identities; 'A method that misses ANY of them is rejected'), and 'any census must take a --before cutoff and REPORT the excluded row count'"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope",
    "mtime_and_commit_ordering",
    "verification_command_immutable",
    "python_lint_gate_ruff_F821_F401_F811",
    "git_status_unintended_change",
    "guard_liveness_via_own_write",
    "log_row_reproduction_criterion_2",
    "independent_payload_measurement",
    "pending_claim_sweep",
    "partition_arithmetic_reproduction",
    "guard_predicate_reuse_and_mixed_case_divergence",
    "perishable_count_reproduction",
    "committed_script_exit_codes",
    "census_86_31_recall_reproduction",
    "live_check_element_coverage",
    "contract_completeness_map",
    "code_review_heuristics",
    "claim_audit_prose",
    "self_probe_indictment"
  ],
  "harness_compliance_ok": true,
  "notes": "Blocker in one sentence: the handoff record does not carry criterion 1's census -- live_check_86.33.md has no full distribution and no outside-.claude/agent-memory/qa/ counts, and neither artifact cites scripts/qa/census_qa_write_guard_log_86_31.py, whose --before 2026-08-10T09:30:00Z run supplies all of it (3012 counted / 6866 excluded / 113 events / 69 outside / 20 breaches across 10 identities) and reproduced for me at exit 0. Fix is transcription into live_check + a citation in experiment_results, not new work. Two NOTE-level items to fold in while doing that: (a) commit 335257a8's message claims the script \"asserts\" the partition sums -- it only prints \"(must equal 78)\"; the sum is guaranteed by construction now (exact set complement), so an assert would be tautological, but the word should be corrected so no later reader relies on a guard that is not there; (b) experiment_results section 2's \"2 definitions exist against 78 logged values\" quotes the contaminated unfiltered count -- 76 of the 78 include prover-fabricated identities, and the real-spawn figure is 65 distinct in the pre-contamination slice; section 1 already discloses the population problem so the conclusion stands, but the corroborating sentence should use the clean-slice number. Nothing the remediation touched is broken: the live_check diff was a pure replacement of the \"pending\" block with the measured answer, and no census content was lost. No UI claims in this step, so gate 1c does not apply; no backend/** runtime surface changed (the only guard edit is log-only), so 1d is satisfied by the guard-liveness observation. I did not re-run the mutation matrix or the researcher-rail prover per the spawn instruction; criteria 3 and 4 are graded on cycle 1's reproduction plus the verbatim live_check sections. This verdict rests on evidence I captured myself, not on Main-supplied captures."
}
```

---

# CYCLE 3 -- verdict: PASS

**`ok: True` | violated: NONE |
harness_compliance_ok: True** | run `wf_85185088-7d3`

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Cycle 3, deciding. The cycle-2 blocker is measurably CLOSED and I verified it independently rather than reading it. (1) CENSUS TRANSCRIPTION FAITHFUL: I re-ran `scripts/qa/census_qa_write_guard_log_86_31.py --before 2026-08-10T09:30:00Z` myself and mechanically difflib-diffed my stdout against the fenced block in handoff/current/live_check_86.33.md section 0 -- 11 diff lines, ONE substantive hunk, the live-log line `rows excluded` (mine 6868, recorded 6867). Every other line byte-identical: 3012 counted, 0 unparsed, all 27 identity names in order, 113 Write/Edit events, 69 outside .claude/agent-memory/qa/, 20 breaches across 10 identities (all 20 rows identical), the stated class rule, and the disclosed residual workflow-subagent 80 / general-purpose 22. The single drifting line is the one the artifact itself predicts and discloses (6866 -> 6867 -> my 6868, monotonic on a live gitignored log). (2) BOTH SCRIPTS CITED: experiment_results_86.33.md section 1 now names census_qa_write_guard_log_86_31.py (cutoff + excluded rows + outside-dir counts + breach recall) and derive_agent_type_population_86_33.py (full distribution + guard-predicate partition); I re-ran both (exit=0, exit=0) and confirmed the split-of-labour claim is true -- derive has no argparse/--before, no outside-dir counts, no breach class. (3a) The \"asserts\" error was in commit 335257a8's MESSAGE; 556389ac supersedes it in the same medium, naming 335257a8; I verified the substance -- grep -nE \"assert|raise|sys.exit\" over the derive script yields only a comment at :73 and sys.exit(main()) at :185, and no handoff artifact claims otherwise. (3b) Section 2 now uses 65; I DERIVED 65 independently (distinct agent_type over rows before the cutoff = 65 exactly; all-time = 78). (4) NOTHING BROKEN: zero backslashes anywhere in live_check_86.33.md, sections 1-5 intact, +114/-0 on live_check and +24/-4 on experiment_results with the -4 accounted for. Immutable command RUN BY ME: `bash -c 'bash -n .claude/hooks/qa-write-guard.sh && echo guard-parses'` -> \"guard-parses\", exit=0. Lint gate over a DERIVED 15-file scope (git diff --name-only 8935be78^ HEAD -- '*.py', count asserted >0, passed via xargs) -> ruff F821,F401,F811 \"All checks passed!\" exit=0; the remediation commit contains zero .py files. git status --short shows NO unintended production change (only another agent's researcher memory, rotating audit/heartbeat/health logs, and my WIP). All six immutable criteria MET; harness compliance 5/5 (research brief gate_passed=true / 12 sources / 22 URLs / recency scan; research 14:43 < contract 14:49 < results 15:22; step still status=pending with grep -cE 'phase=86\\.33' handoff/harness_log.md = 0; evidence CHANGED at 556389ac so this is the documented fresh-respawn, not verdict-shopping). Two NOTE-level findings recorded below; neither is verdict-capping, so the 3rd-CONDITIONAL auto-FAIL rule is not engaged.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "verification_command_exit0",
    "census_script_rerun_and_difflib_diff_vs_live_check",
    "derive_script_rerun",
    "independent_derivation_of_65_and_78_and_13",
    "independent_per_value_outside_dir_table",
    "assert_grep_over_derive_script",
    "stray_backslash_scan",
    "git_diff_of_remediation_commit",
    "ruff_F821_F401_F811_derived_scope",
    "git_status_unintended_change",
    "harness_compliance_5_item",
    "research_gate_envelope",
    "harness_log_conditional_count",
    "masterplan_verification_block",
    "prior_cycle_wip_records",
    "code_review_heuristics"
  ],
  "harness_compliance_ok": true,
  "notes": "TWO NOTES (PASS-with-flag, fix at leisure, neither blocking). NOTE-1 (experiment_results_86.33.md section 2): the new parenthetical \"the unfiltered figure is 78, but 76 of those include prover-fabricated identities\" does not reproduce as a contamination count. I measured it: 13 distinct agent_type values appear ONLY at/after the 2026-08-10T09:30:00Z cutoff (QA-80-2, QA-Upper, Qa-Mixed, main, qa-86-31-c2, qa-86-33, qa-86-34-c2, qa2, qa_85_5_c3, qa_86_31, qax, quality-auditor, subagent) -- not 76. 76 = 78 minus the 2 agent definitions, so the sentence is defensible read as \"the 76 non-definition labels include prover-fabricated ones\" (the verb is \"include\", not \"are\"), but the natural misreading \"76 of 78 are fabricated\" is FALSE, and the author's own script states the synthetic/real split cannot be derived from this log at all. The headline figure it modifies (65) reproduces exactly and criterion 2's conclusion rests on the measured 12-key/10-key payload, not on this number. NOTE-2 (same file, section 1): the bullet enumerates THREE bolded elements, then says derive produces \"none of the four elements above\" -- off-by-one prose; the substantive claim is correct. RESIDUAL I CHECKED AND DELIBERATELY DID NOT ESCALATE: criterion 1's phrase \"for every value, how many of its Write/Edit events targeted paths outside .claude/agent-memory/qa/\" is served by an AGGREGATE (113 W/E, 69 outside) plus per-identity breach rows, not a per-value column. I derived the per-value table myself to test whether the aggregate hides anything -- it totals exactly 113 and 69 over the same 27 identities (23 have >=1 outside write), so nothing is concealed; the recall test the criterion actually specifies (\"20 events across 10 identities ... a method that misses ANY of them is rejected\") IS reported per-identity and reproduces exactly. Cycle 2's named remedy was \"transcribe THIS output\" and it was executed verbatim; raising granularity as a NEW cycle-3 blocker would be goalpost-moving, not verification. SCOPE DISCLOSURE: per the spawn tasking I did NOT re-run criteria 2/3/4/5/6 or the mutation matrix -- cycles 1 and 2 reproduced them (their WIP records are on disk at verdict_wip_86.33__20260811T125920Z.md and __20260811T131131Z.md) and I re-ran only the immutable command among them. No UI claims in this step, so section 1c does not apply and I took no Playwright capture. No backend/** runtime smoke was needed: the remediation commit touches only .md artifacts. My write-first record: .claude/agent-memory/qa/verdicts/verdict_wip_86.33__20260811T132312Z.md. Reminder for Main: criterion 5's ASK #6 (fail-closed vs platform disallowedTools vs document-as-convention) is still an OPEN operator decision -- PASS here means the step correctly did NOT ship a fail-closed change, not that the guard is now a boundary."
}
```

---

## Main's response to the cycle-3 PASS

**IT VERIFIED THE TRANSCRIPTION MECHANICALLY.** It re-ran the census and
difflib-diffed its own stdout against my fenced block: 11 diff lines, **one**
substantive hunk -- `rows excluded`, mine 6867 vs its 6868. All 27 identity names in
order and all 20 breach rows byte-identical. The drift is now monotonic across three
runs (6866 -> 6867 -> 6868), which is the perishability claim reproducing rather
than an error.

**AND IT REFUSED TO MOVE THE GOALPOSTS.** Criterion 1 asks for per-value outside-dir
counts; my transcription gives an aggregate plus per-identity breach rows. It derived
the per-value table itself to test whether the aggregate concealed anything -- totals
exactly 113 and 69 over the same 27 identities -- then declined to raise it, because
cycle 2's named remedy was "transcribe THIS output" and that was executed verbatim.

### Its two NOTEs are mine and both are fixed

**NOTE-1 -- a parenthetical that is false on the natural reading.** I wrote *"the
unfiltered figure is 78, but 76 of those include prover-fabricated identities"*.
76 = 78 minus the 2 agent definitions, so it survives a charitable reading -- but
"76 of 78 are fabricated" is **FALSE**, and that is how it reads. Measured
contamination: **13** distinct values appear only at/after the cutoff. My own script
says the split cannot be derived from this log at all, so quoting any contamination
count invites the error.

**NOTE-2 -- off-by-one prose**: three bolded elements, described as "the four
elements above".

### What PASS does NOT mean

Keeping its closing line verbatim: *"PASS here means the step correctly did NOT ship
a fail-closed change, not that the guard is now a boundary."* **ASK #6 is still open.**
The guard remains keyed on a caller-chosen field, and I demonstrated the bypass on
myself twice while measuring it.
