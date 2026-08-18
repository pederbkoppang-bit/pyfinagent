# Evaluator critique -- step 86.28 (CURRENT)

> **Full verbatim history of all five verdicts:**
> `handoff/current/evaluator_critique_86.28_history.md` (609 lines, byte-identical
> copy of this file before compaction, md5 c0480e4123b0aa05ae3ee4542a278c17).
> Nothing was deleted; the history was moved out of the mandatory read path
> because two Q/A spawns dropped reading it (197K and 185K tokens).

## Verdict ledger

| # | Cycle | Run | Verdict | Headline finding |
|---|---|---|---|---|
| 1 | 1 | `wf_10c6cbd2-cad` | CONDITIONAL | W1 illusory ordering guard; W2 three stale line citations |
| 2 | 2 | `wf_d0934c91-70b` | CONDITIONAL | W3 block-comment defeat of the hardened guard; W4 class audit excluded the production file |
| - | 3 | `wf_01c83c86-09d` | **DROPPED** (no verdict) | rail failure at 197,091 tokens |
| 3 | 3 | `wf_e262facc-cdc` | **FAIL** | typed-not-captured transcript, arithmetically impossible (68+3=71) |
| 4 | 4 | `wf_5a217e41-9b9` | CONDITIONAL | Q1: ABSENT-tier half unguarded; fixture could not represent production |
| 5 | 5 | `wf_344395f1-4ac` | CONDITIONAL | criterion 5: 3 of 5 new checks had no demonstrated mutant |
| - | 6 | `wf_9c55b720-ef3` | **DROPPED** (no verdict) | rail failure at 184,753 tokens |

**F1 counters:** one FAIL (cycle 3) -> `retry_count: 1` of 3. The FAIL reset
the consecutive-CONDITIONAL counter, so the two CONDITIONALs after it are
#1 and #2 of that run; the 3rd-CONDITIONAL auto-FAIL threshold is NOT
reached. Dropped runs are NOT verdicts and count toward neither.

All findings from cycle 3 onward concern the CHECKER or the EVIDENCE.
`.claude/workflows/research-gate.js` has been frozen since cycle 3 and was
comment-only before that; three consecutive Q/As found the shipped code
correct under every probe.

---

# CYCLE 5 VERDICT -- Q/A run `wf_344395f1-4ac`

> Transcribed VERBATIM.

## Verdict: **CONDITIONAL** (`ok: False`)

### reason

CYCLE 5 -- Q/A verdict CONDITIONAL. 8 of 9 immutable criteria MET and measured by me; criterion 5 is NOT MET on its literal terms. Deterministic: `node scripts/qa/verify_research_gate_workflow.mjs` exit=0, ALL GREEN 78 passed / 0 failed (run bare, not piped). Criterion 1 MET beyond the count: I audited by SYMMETRIC DIFFERENCE of check() names, not totals -- ZERO checks removed since the pre-86.28 baseline (cad38647~1), 37 added overall, exactly 5 added since cycle 3 (73->78), and the 5 are precisely the ones claimed. Criterion 2 MET, measured by driving the real driver: UNSUPPORTED 'deep' -> spawns=0, tier_requested='deep', tier_applied='moderate', tier_supported=false, violation tier_unsupported, gate_passed=false; ABSENT -> spawns=2, tier_requested=null, tier_applied='moderate', no violation, check line tier_absent_defaulted_ok. Distinguishable in the RETURN VALUE. Criterion 3 MET (VALID_TIERS=['simple','moderate','complex'], every 'deep' occurrence a comment, no producer fan-out, divergence filed as an operator decision in live_check S7). Criterion 4 MET (mutants 'recency corroboration removed' and 'urls corroboration removed' both KILLED in my own run; absent-verification still fails closed via the fail-closed path ONLY, corroboration correctly does not double-fire). Criterion 6 MET (0 +/- lines touching dry|floors since cycle 3; reasons recorded at experiment_results:225-226 and live_check:71). Criterion 7 MET (code pins agentType:'researcher' at the stage-1 spawn research-gate.js:598; researcher.md agrees; CLAUDE.md now reads 'researcher', explicitly names and retracts its own prior 'general-purpose' self-contradiction, and switches to grep-for-symbol over line numbers). Criterion 8 MET (every rider assertion GREEN in my run: no `minimum:`, no `minItems:`, gate_passed not const:true, additionalProperties:false, 0 static imports of any form, model 'opus', no Monitor/watchdog, exactly one export). Criterion 9 MET with the author's disclosed gap intact: I PROVED cycle-3's research-gate.js commit was COMMENT-ONLY (0 executable-line changes after stripping // lines) and HEAD is byte-identical to it (md5 054f4e98d8f56260de2fb40294aabfe4 both ways), so the live spawn wf_23d9ed4b-22c covers the shipped EXECUTABLE behaviour; the full stage-1+stage-2 path remains not re-run live post-change, which the author discloses and which fails closed.

PRIORITY ITEMS FROM THE SPAWN, all executed independently in an out-of-tree scratchpad mirror (production tree never written; control run on the mirror reproduced 78/0 before any mutation).
(1) Q1 and Q5 REPRODUCED AND KILLED BY ME. Q1 (`tierUnsupported = !tierAbsent && !tierSupported` -> `= !tierSupported`) -> FAILED: 76 passed, 2 failed, killing 'ABSENT tier still SPAWNS' (recorded 0 agent() calls) and 'ABSENT tier raises NO tier_unsupported violation'. Q5 (`tierInfo.unsupported === true` -> `tierInfo.supported !== true`) -> FAILED: 75 passed, 3 failed. Both numbers and both failure-line texts match live_check S26 EXACTLY, which independently corroborates that S26 was CAPTURED, not typed -- the cycle-4 defect has NOT reappeared. Confirmed separately: all 9 occurrences of the arithmetically impossible '68 passed, 3 failed' (=71) in the handoff sit inside prose DISCUSSING the defect, none is presented as a live verbatim capture.
(2) THE OTHER DIRECTION -- 3 measured survivors, all ALL GREEN 78/0, each with a behavioural differential I measured rather than argued. See violation_details 2-4. A fourth mutant (drifting the tierInfo `supported` field passed to enforceGate at :660) also survived but is an EQUIVALENT MUTANT -- enforceGate provably never reads tierInfo.supported -- so I do NOT report it as a defect.
(3) research-gate.js is byte-identical to cycle 3: empty diff, identical md5, and 0 executable-line differences. NO production change this cycle. This step's commit 49793961 contains exactly 4 files (checker + 3 handoff artifacts) and zero .py / frontend/ / backend/ files, so the ruff, ESLint, tsc and backend-runtime-smoke gates are N/A on a DERIVED scope. (The uncommitted files in `git status` -- test_phase_86_24_clock_dependence.py, contract_86.24.md et al -- belong to a concurrent session's step 86.24, not to 86.28.)
(4) ARITHMETIC AUDIT re-derived by me over all four handoff artifacts: totals 40, 61, 64, 73, 78 all valid; the single invalid total (71) is the quoted cycle-4 defect under discussion, not a live claim.

WHY CONDITIONAL AND NOT PASS -- criterion 5. It states: "each new check has its own MUTANT ... the mutant must be shown KILLED, and the mutation output is recorded verbatim. A check whose mutant is not demonstrated is not counted as delivered." Q1 and Q5 demonstrate only 2 of the 5 new checks. Grep returns ZERO hits, in every handoff artifact, for the names of the other three: 'ABSENT tier reports tier_requested null and applied moderate', 'TIER_ABSENT fixture matches the driver', 'TIER_UNSUPPORTED fixture matches the driver'. The checker's own internal mutation matrix (7 members at :480-:495) covers none of them either, so the gap holds under both readings of the criterion. I built the three missing mutants myself and ALL THREE ARE KILLED -- so the checks are substantively SOUND and this is an evidence-recording gap, not a vacuity. To close: record these three captures verbatim -- (a) `tier_requested: tierRequested` -> `tier_requested: tier` in the main return => FAILED: 77 passed, 1 failed, killing 'ABSENT tier reports tier_requested null and applied moderate'; (b) revert TIER_ABSENT to the cycle-4 shape `tierOpts({ requested: null, absent: true })` => FAILED: 77 passed, 1 failed, killing 'TIER_ABSENT fixture matches the driver -- fixture supported=true, driver tier_supported=false' (this also proves the fidelity check would have caught the very defect it was written for); (c) refusal-path `tier_supported: false` -> `true` => FAILED: 76 passed, 2 failed, killing 'TIER_UNSUPPORTED fixture matches the driver' and the cycle-3 check 'UNSUPPORTED tier returns gate_passed:false with the tier reported'.

CONVERGING, NOT THRASHING -- stated plainly as asked. The executable driver has been frozen since 08:51 and every cycle since has been checker/evidence work; each cycle's finding is strictly narrower than the last (ordering guard -> line citations -> source-scan vacuity -> typed transcripts -> one-directional guard -> three missing mutant captures). The cycle-5 fix is real and correct: I verified the ABSENT half is now genuinely guarded and that the fix kills the exact mutant that defeated cycle 4. The remaining blocker is recording three captures I have already produced.

LIMITS OF THIS VERDICT. Live UI gate N/A (no UI claims). I could not execute the Workflow runtime from my surface, so the three live run records (wf_4da39b31-695, wf_23d9ed4b-22c, wf_60de95f7-5dc) are the AUTHOR's evidence, read not reproduced -- but I closed the load-bearing half of criterion 9 deterministically by proving the shipped bytes are executably identical to the version those runs exercised. Contract-ordering: research_brief_86.28.md (08:38, gate_passed true, 7 sources >=5, 34 URLs, recency true) precedes contract_86.28.md, and the contract was committed no later than the first production change -- but both landed inside the peer-attributed sweep commit cad38647, so their relative order WITHIN that commit is not establishable from git. I name that rather than let the mtime check pass silently; the cross-attribution is already queued as 86.15.

### violated_criteria

- criterion_5: mutant not demonstrated for 3 of the 5 new checks (not counted as delivered)
- illusory-guard [WARN]: the two new fixture-fidelity checks pin the one tier field enforceGate never reads, leaving the branch-steering field unpinned
- coverage-gap [WARN]: a SUPPORTED tier being silently downgraded survives the suite -- the known-positive is driven at the default value
- coverage-gap [WARN]: the enforceGate absent-branch label that makes the cases distinguishable is itself unguarded

### violation_details

#### 1. Threshold_Not_Met

**action**

grep each of the 5 new check() names across handoff/current/live_check_86.28.md + experiment_results_86.28.md + evaluator_critique_86.28.md; enumerate the checker's internal mutation matrix at scripts/qa/verify_research_gate_workflow.mjs:480-495; then build and run the 3 missing mutants myself in an out-of-tree mirror

**state**

5 new checks delivered this cycle (73->78; symmetric-difference audit: 0 removed, 5 added). Mutant recorded for only 2 of them -- Q1 (FAILED: 76 passed, 2 failed) and Q5 (FAILED: 75 passed, 3 failed), both of which I reproduced EXACTLY, confirming live_check S26 is a genuine capture. ZERO recorded mutant output anywhere in the handoff for the remaining three: 'ABSENT tier reports tier_requested null and applied moderate', 'TIER_ABSENT fixture matches the driver (supported:false for an absent tier)', 'TIER_UNSUPPORTED fixture matches the driver (supported:false)' -- grep returns 0 hits for all three names, and the checker's 7-member internal matrix covers none of them. I demonstrated all three KILLED myself: (a) main-return 'tier_requested: tierRequested' -> 'tier_requested: tier' => 77 passed, 1 failed; (b) TIER_ABSENT reverted to the cycle-4 shape => 77 passed, 1 failed with detail 'fixture supported=true, driver tier_supported=false'; (c) refusal-path 'tier_supported: false' -> 'true' => 76 passed, 2 failed. The checks are therefore SOUND and non-vacuous; what is missing is the demonstration the criterion requires as the delivery.

**constraint**

criterion 5 verbatim: "each new check has its own MUTANT in scripts/qa/verify_research_gate_workflow.mjs proving it can FAIL: the mutant must be shown KILLED, and the mutation output is recorded verbatim. A check whose mutant is not demonstrated is not counted as delivered."

#### 2. Missing_Assumption

**action**

mutate the checker's TIER_ABSENT fixture from 'absent: true' to 'absent: false' and run the suite; separately grep every tierInfo field read inside enforceGate

**state**

SURVIVED -- ALL GREEN: 78 passed, 0 failed. enforceGate reads tierInfo.unsupported, .requested, .valid, .applied and .absent (research-gate.js:368, :370-372, :374, :377) and provably NEVER reads .supported. The two new fidelity checks pin ONLY .supported -- the single tier field enforceGate ignores -- and pin it against the RESULT field tier_supported (:706) rather than the tierInfo object actually passed to enforceGate (:660). Corroborating equivalent mutant: drifting :660 'supported: tierSupported' -> 'supported: true' also survives 78/0 (EQUIVALENT, reported only to locate the blind spot, NOT as a defect). Consequence: with absent:false the TIER_ABSENT fixture silently describes the SUPPORTED branch instead of the absent one, and the sole enforceGate probe that uses it (checker:350, which asserts gate_passed===true only) cannot tell the two branches apart -- so the fixture can regress to a non-representative state exactly as it did in cycle 4. Named fix: pin .absent and .unsupported against the driver too, and assert the ABSENT run emits the checks entry 'tier_absent_defaulted_ok'.

**constraint**

qa.md 4c shape #5 (a fixture that cannot represent the failure) and criterion 5's purpose -- the fidelity checks were added precisely to stop a fixture drifting away from the production state, so they must pin the field that determines which production state is represented. WARN severity: a genuine behavioural guard (the three driven ABSENT-tier checks) coexists.

#### 3. Overgeneralization

**action**

mutate research-gate.js "const tier = tierSupported ? tierRequested : 'moderate'" -> "const tier = 'moderate'", run the suite, then drive the real driver at tier:'complex' and tier:'simple' and diff against the unmutated baseline

**state**

SURVIVED -- ALL GREEN: 78 passed, 0 failed. Measured behavioural differential (not argued): baseline tier:'complex' -> tier_applied='complex', checks ['tier_supported_ok: "complex"']; mutated -> tier_applied='moderate', checks ['tier_supported_ok: "moderate"'], with tier_supported still true and ZERO violations raised. A caller's SUPPORTED tier is silently downgraded and the gate certifies at the substituted standard -- the same over-claim shape criterion 2 forbids for UNSUPPORTED, in the third direction. Root cause is the SAME CLASS the step already failed on in cycle 4: [6d]'s only SUPPORTED run is driven at tier:'moderate', the identical value an ABSENT tier defaults to, so the known-positive is value-degenerate and cannot represent the failure. One-line fix: drive the known-positive at 'complex' (the value the TIER_OK fixture already carries) and assert supported.result.tier_applied === 'complex'.

**constraint**

criterion 2's principle "the gate does not certify as though the requested standard had been met". WARN, not BLOCK: criterion 2's literal text scopes the requirement to UNSUPPORTED and ABSENT tiers, so this is a coverage gap adjacent to the criterion rather than a miss of it. Per feedback_queue_discovered_defects_in_masterplan and feedback_freeze_the_tree_during_evaluate, close it in the next cycle or as its own queued step -- do not patch the tree being graded.

#### 4. Missing_Assumption

**action**

mutate research-gate.js "} else if (tierInfo && tierInfo.absent === true) {" -> "} else if (false) {", run the suite, then drive the real driver on all four tier cases and diff the emitted checks array against baseline

**state**

SURVIVED -- ALL GREEN: 78 passed, 0 failed. Measured differential: the ABSENT run's tier check-line flips from 'tier_absent_defaulted_ok: no tier passed, ran at "moderate"' to 'tier_supported_ok: "moderate"' -- the output now reports an absent-tier caller as though a supported tier had been named. Criterion 2 is NOT breached: ABSENT vs UNSUPPORTED remain distinguishable via the violations array, and ABSENT vs SUPPORTED-moderate remain distinguishable via tier_requested (null vs 'moderate') in the return value. But the labelling branch the driver deliberately emits to make the cases legible is itself covered by no check, so it can be deleted without the suite noticing.

**constraint**

criterion 2 "An ABSENT tier still defaults to moderate as today -- the two cases must be distinguishable in the output" -- the primary channel holds, the secondary (the enforceGate checks label) is unguarded. WARN: assert the ABSENT run's checks array contains 'tier_absent_defaulted_ok' and the SUPPORTED run's contains 'tier_supported_ok'.

### checks_run (26)

- `harness_compliance_audit_5_item`
- `verification_command_immutable_exit_0_78_passed_0_failed`
- `research_gate_envelope_gate_passed_7_sources_34_urls_recency_true`
- `contract_before_generate_mtime_and_git_chain_with_disclosed_limit`
- `log_last_harness_log_zero_entries_masterplan_status_pending`
- `no_verdict_shopping_evidence_changed_commit_49793961`
- `production_change_scope_derived_from_git_show_name_only`
- `research_gate_js_byte_identity_md5_and_empty_diff_vs_cycle3`
- `executable_only_diff_cycle3_comment_only_zero_code_lines`
- `symmetric_difference_audit_of_check_names_zero_removed`
- `mutation_reproduction_Q1_killed_76_2`
- `mutation_reproduction_Q5_killed_75_3`
- `adversarial_mutants_new_M1_M2_M3_M5_M9_C3probe_C5probe`
- `equivalent_mutant_ruled_out_tierInfo_supported_never_read`
- `behavioural_differential_probe_four_tier_cases_vs_baseline`
- `control_run_on_unmutated_mirror_before_any_mutation`
- `arithmetic_audit_all_suite_totals_re_derived`
- `typed_not_captured_transcript_recurrence_check`
- `criterion_6_dry_and_floors_untouched_plus_reason_recorded`
- `criterion_7_doc_drift_agentType_code_researcher_md_claude_md`
- `criterion_8_rider_traps_all_green`
- `criterion_9_live_spawn_coverage_via_byte_identity`
- `contract_completeness_all_9_criteria_mapped_to_evidence`
- `code_review_heuristics`
- `lint_gates_ruff_eslint_tsc_runtime_smoke_N_A_on_derived_scope`
- `3rd_conditional_counter_check`

### notes

3rd-CONDITIONAL RULE, checked before issuing: `grep -F "86.28" handoff/harness_log.md | grep result=` returns ZERO entries -- correct and expected mid-cycle, since log-last appends only at close (auto-memory feedback_derived_scope_misses_untracked_files). I therefore counted from handoff/current/evaluator_critique_86.28.md: CONDITIONAL (c1, :13), CONDITIONAL (c2, :167), FAIL (c3, :308), then the cycle-4 CONDITIONAL that drove "Follow-up by Main -- cycle 5" (:428). The FAIL RESET the consecutive-CONDITIONAL counter, so this is the 2nd consecutive CONDITIONAL, not the 3rd -- the auto-FAIL rule does not fire and CONDITIONAL was legitimately available. masterplan retry_count=1, max_retries=3, so certified_fallback=false.

MUTATION METHOD, disclosed. I mutated nothing in the repository. I copied research-gate.js and the checker into the session scratchpad (the qa-write-guard hook correctly blocked my Write tool, so all edits were made to the scratchpad copies via `python3 -c`), ran a CONTROL first to confirm the mirror reproduced 78/0, and restored from a pristine copy after every mutant. Final state re-verified: `git status` clean for .claude/workflows/ and scripts/qa/, driver md5 unchanged, immutable command still exit=0.

NOTE (out of scope, do not patch mid-EVALUATE -- queue it). The stage-2 brief-verifier spawns `agentType: 'Explore'` at research-gate.js:649, but `.claude/agents/` contains only qa.md and researcher.md -- there is no Explore agent, and CLAUDE.md's Layer-3 doctrine states Explore was merged into Researcher and that reintroducing it "is the old pattern". This is PRE-EXISTING (present at d0a98817~1, before 86.28 began) and research-gate.js is untouched by this cycle, so it is not a criterion-7 miss -- criterion 7 scopes to the stage-1 agentType:'researcher' pin, which is correct in all three places. The failure direction is safe (the try/catch sets verification = null and enforceGate fails closed), and the author's live Run 3 shows stage 2 did execute. Per feedback_queue_discovered_defects_in_masterplan this deserves its own research-gated step rather than a prose disclosure.

NOTE (cosmetic). live_check S27's arithmetic self-check now scans its own prior output, producing doubled rows ("71 <-   71 <- ..."). Self-referential and harmless -- my independent sweep over all four artifacts reproduced the same conclusion (only the quoted 71 is invalid, and it is discussion of the cycle-4 defect, not a live claim). No verdict effect.

WHAT I CONFIRMED IS SOUND, so it is not re-litigated next cycle: the three driven ABSENT-tier checks are genuinely behavioural and kill the exact mutant that defeated cycle 4; the fixture-fidelity checks are non-vacuous (M1 proves they fail when the fixture regresses); the S26 transcripts are real captures, reproduced by me to the digit; the recency and urls corroboration mutants are killed; the fail-closed path does not double-fire; every rider trap holds; and the shipped driver is executably identical to the version exercised live.

CODE-REVIEW HEURISTICS: evaluated, no security / trading-domain findings -- the diff touches no backend, no frontend, no execution path, no risk guard, no secret, no dependency pin. The findings above are Dimension-4 anti-rubber-stamp (illusory-guard #17, shapes (c) and (e)) and Dimension-5 (no sycophancy: I did not flip a prior verdict on unchanged evidence -- commit 49793961 changed the checker and three artifacts).

---

# Follow-up by Main -- cycle 6

> Appended. All five verbatim verdicts above untouched.

Criterion-5 finding accepted. The three checks now have standing mutants in
a new `[7b]` driver-level matrix -- not the Q/A's captures pasted in, since
the criterion requires the mutant to live in the checker. The `[7]` matrix
structurally could not cover them (it probes enforceGate; the checks are
end-to-end driver behaviour).

Worth recording: my first version of one mutant SURVIVED, and the fault was
the mutant. `tier_requested: tierRequested,` occurs twice and a first-match
replace hit the refusal branch the probe never executes. Every driver-mutant
now asserts anchor UNIQUENESS, so a mis-aimed mutant fails loudly instead of
masquerading as a weak check.

Both WARNs closed: fidelity now pins the branch-steering fields
(`unsupported`/`absent`) rather than the one `enforceGate` ignores; the
known-positive drives at `complex` so a silently-downgraded SUPPORTED tier
can no longer pass; the absent-branch label is asserted directly.

40 -> 61 -> 64 -> 73 -> 78 -> 92, 0 failed. Production code unchanged since
cycle 3.


---

## Cycle 7 -- verdict TRANSCRIBED VERBATIM (2026-08-17, Workflow rail `wf_f2d3e348-a93`)

**VERDICT: PASS** (ok=true). All NINE criteria re-derived by execution: the 40/0 baseline independently reproduced by running checker+workflow at the pre-step commit in a mirrored tree; symmetric difference 0-removed/84-added baseline->HEAD; the tier machinery driven with field-identity to the live run record on all 10 keys; the [7b] permanent matrix all KILLED today; 22 of the evaluator's own mutants (open-direction, fail-closed-direction, fixture-drift, label-drift, harness) with ZERO survivors and a disclosed near-miss (a crash first read as '0 checks red' -- the probe indicted itself). Five queue-class residuals: R1 stale artifact headlines (92/'six cycles'), R2 the growth attribution off by this step's own 92->97 slice, R3 a byte-identity sentence outlived by 11 other-step commits, R4 the backfill omitted the cycle-7 drop (FIXED at closure with the named run id), R5 no standing mutant cells for the positive controls. None a criterion miss; none hides a defect.

> Transcribed verbatim. Main records the verdict; Main never authors it.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 9 immutable criteria MET, each re-derived by execution rather than read. C1: immutable command exit=0, ALL GREEN 124 passed/0 failed (run twice); the 40/0 baseline INDEPENDENTLY REPRODUCED by running the checker+workflow both at 089726f9 in a mirrored tree; symmetric difference of check NAMES baseline->HEAD = 0 removed/84 added, and at 86.28's own close (d2e987f1) = 97/0 with 0 removed; 29/29 literal-named baseline assertions byte-identical at HEAD and all 6 baseline [7] mutant cells retained (+3 added), so nothing was deleted or weakened. C2: driven, not scanned -- today's driver at tier:'deep' spawns 0 agents and returns tier_requested/tier_applied/tier_supported + a tier_unsupported violation, FIELD-IDENTICAL on all 10 compared keys to the live run record wf_23d9ed4b-22c; ABSENT still spawns, defaults to moderate, reports tier_requested null, raises no violation. C3: VALID_TIERS=['simple','moderate','complex'], every 'deep' occurrence is a comment, only two await agent() sites (stage1 researcher + stage2 verifier) so no fan-out; divergence disclosed in live_check S7 as an operator decision. C4: both corroborations run inside the EXISTING stage-2 verifier and derive from the file (verifyBrief reads recency_section_present and distinct_urls_in_brief from disk); fail-closed preserved across 7 verification shapes driven directly through enforceGate (undefined/null/string/array/number/true/{}). C5: the [7b] DRIVER-level matrix is permanent (4 driver mutants + 1 fixture mutant, anchor-uniqueness asserted per cell), all KILLED in today's run and recorded verbatim -- the cycle-5 NOT-MET is closed. C6: opts.floors and coverage.dry byte-identical to baseline, reasons recorded. C7: researcher.md, CLAUDE.md and the code all say agentType:'researcher', CLAUDE.md's self-contradiction retracted in place, checker asserts the pin. C8: all rider assertions green in [8]. C9: post-change live spawn verified against the run record itself (status completed), and I proved it still describes today's shipped behaviour by driving today's file. ANTI-RUBBER-STAMP: I ran 22 of my own mutants -- 7 open-direction production, 7 fail-closed-direction, 3 fixture-tier-drift, 1 label-drift, 4 fixture/harness -- with the control re-verified GREEN (124/0) before every batch: ZERO survivors, each kill attributed to a named guard. The fixture and harness shapes qa.md says the author's own matrix historically misses (H1/H2/H3/H4) all make the suite go red, so the fixtures and the spawn recorder are load-bearing. One near-miss worth recording: my C7 mutant first read as \"0 checks red\" -- it was a CRASH, not a survivor; the probe indicted itself. Harness compliance clean 5/5 (brief 08:38:58 < contract 09:09:12 < first code commit 09:26:39; gate recomputed live by wf_60de95f7-5dc at 7>=5 sources / 34>=10 URLs / recency_scan_ok; experiment_results present; 0 rows for phase=86.28 in harness_log and masterplan still pending; evidence changed in commit 3315546c and the prior spawn was a NO_VERDICT drop, so this is not verdict-shopping). No unintended production change: 86.28's six commits touch only research-gate.js, the checker, researcher.md, CLAUDE.md, its own handoff artifacts and verdict_ledger.jsonl; secret scan 0 hits; node --check green on both files. FIVE RESIDUALS, all evidence-quality only, named for QUEUEING rather than iteration per the operator directive in force: (R1) experiment_results_86.28.md still states \"Current: 92 passed, 0 failed\" and \"Six evaluate cycles\" -- stale against this step's own delivery (97) and today's tree (124); the criterion's designated evidence carrier, live_check, is current and dated. (R2) live_check's closing sentence attributes the whole 92->124 growth to other steps; the true other-step span is 97->124 (27 checks, 0 removed, independently re-derived by me), because 92->97 was this step's own cycle-7 fix documented directly above it. (R3) live_check S5's \"HEAD is byte-identical to it\" no longer reproduces -- research-gate.js went 40,582 -> 65,098 bytes across 11 commits by OTHER steps since 86.28's freeze; I re-measured the substance and it holds. (R4) the ledger backfill carries 7 rows and omits the cycle-7 drop wf_e03ec2d0-c07 that live_check S9 documents in detail (its source table omits it too, as CLAUDE.md itself notes). (R5) the positive-control checks have no STANDING mutant cell; I demonstrated them killable ad hoc (C1-C5, F1a/F1b/F1c, F2, F3) but the checker does not carry those cells permanently. None of the five hides a defect, none is a criterion miss, and none changed a number the criteria depend on.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command_exit_0_124_passed",
    "immutable_command_rerun_stability",
    "baseline_40_0_independently_reproduced_at_089726f9",
    "symmetric_difference_check_names_baseline_vs_head_0_removed",
    "symmetric_difference_check_names_86_28_close_vs_head_0_removed",
    "baseline_assertion_bodies_byte_identical_29_of_29",
    "baseline_mutant_table_retained_6_of_6",
    "independent_mutation_matrix_22_mutants_zero_survivors",
    "fixture_and_harness_mutation_H1_H4",
    "fail_closed_direction_mutants_C1_C9",
    "fixture_tier_drift_mutants_F1a_F1b_F1c",
    "b1_mutant_builds_and_spawns_not_a_build_failure_kill",
    "fail_closed_on_7_verification_shapes",
    "live_workflow_run_record_verified_wf_23d9ed4b_22c",
    "today_driver_field_identical_to_live_return_10_keys",
    "criterion_3_valid_tiers_and_no_fanout",
    "criterion_6_floors_and_coverage_dry_untouched",
    "criterion_7_doc_drift_grep_researcher_claude_qa",
    "criterion_8_rider_assertions_green",
    "harness_compliance_5_item_audit",
    "research_gate_envelope_and_live_recompute",
    "log_last_and_masterplan_status",
    "git_scope_86_28_committed_file_set",
    "python_lint_gate_ruff_F821_F401_F811",
    "node_check_both_files",
    "secret_scan_over_step_diff",
    "qa_wip_prior_attempt_evidence",
    "verdict_ledger_sequence_evidence_only",
    "code_review_heuristics",
    "evaluator_critique",
    "experiment_results",
    "live_check"
  ],
  "harness_compliance_ok": true,
  "research_needed": false,
  "notes": "SEQUENCE EVIDENCE (gathered, not applied -- any threshold is the caller's). `python scripts/qa/verdict_history_86_21.py --step 86.28 --evidence-only`: status=ok, \"7 verdict(s) from the ledger\", sequence CONDITIONAL -> CONDITIONAL -> NO_VERDICT -> FAIL -> CONDITIONAL -> CONDITIONAL -> NO_VERDICT. Carried through as-is, NO_VERDICT rows included. `python scripts/qa/qa_wip.py 86.28 --spawned-at 2026-08-17T15:39:38Z`: source_present=true, attempt_number=1, attempt_number_status=ok, attempt_number_is_lower_bound=false, prior_attempts=0, records_retained=1 (gauge, includes my own record), records_pruned_known=null, prior_records=[]. CROSS-CHECK per qa.md: prior_attempts (0) is NOT greater than the ledger's 7 rows, so the ledger is NOT stale by that rule. The divergence runs the OTHER way and is worth flagging: the WIP sink holds ZERO prior records for a step the ledger shows 7 verdicts for, and records_pruned_known is null, so the WIP attempt number for this step is an undercount of unknown size -- the ledger is the richer source here and I let it govern. Separately, the ledger's 7 rows were backfilled TODAY (commit 3315546c, recorded_at 15:39:07Z) as labelled reconstructions from the critique's own table, and they omit the cycle-7 drop wf_e03ec2d0-c07 that live_check S9 documents; CLAUDE.md's F1b fixture describes this series as 8 attempts. I did NOT infer verdicts from prior_records bodies. harness_log holds 0 rows for phase=86.28 (secondary cross-check only, and LOG runs after EVALUATE).\n\nWRITE-FIRST: `.claude/agent-memory/qa/verdicts/verdict_wip_86.28__20260817T153938Z.md`, born inert, appended through the run, flipped to COMPLETE with COMPLETED: 2026-08-17T15:55:57Z as my final act. It is a crash-survival record, not a verdict.\n\nWRITE BLOCK DISCLOSED: qa-write-guard.sh correctly denied a Write into the session scratchpad, so the mutation harness scripts were created with bash heredocs into the scratchpad only. NOTHING in the repository was written, moved or deleted at any point; every mutation ran against COPIES in /private/tmp/.../scratchpad/m3 with the control re-verified GREEN (124/0) before each batch, and HEAD was 36e42227 at the start and at the end.\n\nFOR MAIN, not a criterion finding: the auto-commit hook runs `git add -A`, and the working tree currently carries unrelated in-flight work (backend/api/sovereign_api.py, 5 frontend components, several audit jsonl streams, ~40 untracked agent-memory files). Flipping 86.28 to done right now would sweep all of it into a commit titled 86.28. Ruff F821/F401/F811 on the one changed .py is exit 0 (\"All checks passed!\"), so nothing is broken -- it is an attribution hazard, not a correctness one.\n\nNO UI CLAIMS in this step, so gate 1c does not apply and no Playwright capture was needed. No backend/** in the step's diff, so gate 1d does not apply. The step's own diff is .js/.mjs/.md only; the ruff gate was run on the DERIVED working-tree scope for transparency.\n\nThe five residuals in `reason` (R1-R5) are evidence-quality only. Stated explicitly per the operator directive of 2026-08-17: every criterion is substantively met and independently verified by execution, so these belong in a queued follow-up, not another remediation cycle on this step."
}
```
