# Evaluator critique -- phase-86.17

## Cycle 1 verdict: **CONDITIONAL** (`wf_2c6036e2-5b7`)

Workflow rail, `model: opus`, 17 tool calls, 164,118 tokens, 520s. Main RECORDS
this verdict; Main did not author it. Transcribed VERBATIM.

**Three findings, all correct, and the first is a hole in MY OWN checker.**

1. **Criterion 6 NOT MET.** `qa-verdict.js`'s blind early-return was never
   mutated. The Q/A mutated it and the mutant **SURVIVED** with the checker
   still ALL GREEN -- because my `drive()` slices the source at the FIRST of
   `["if (inputHealth.blind)", "phase('Research')", "phase('QA')"]`, so deleting
   that block merely moves the cut and every assertion stays green. The
   behavioural differential is real and measured: 0 spawns becomes 1 max-effort
   spawn labelled `qa-verdict:UNSPECIFIED` returning a PASS-shaped object. Two
   further new guards also survived.
2. **Criterion 3, second sentence.** A class-A `research-gate` run STILL spawns
   a max-effort researcher whose prompt names
   `handoff/current/research_brief_UNSPECIFIED.md` -- the exact artifact this
   step's own defect narrative calls the harm. `qa-verdict` closes this with an
   early return; the identical reasoning was not applied to `research-gate`, and
   I disclosed neither the asymmetry nor the residual.
3. **Scope honesty.** I justified the un-launched `research-gate` class-A path
   with "it shares the identical `classifyArgs` body". The bodies are not even
   literally identical (brace style), but the material problem is that the
   inference does not transfer: the two scripts diverge IMMEDIATELY after
   `classifyArgs`, and that divergence is precisely the property I was inferring.

Criteria 1, 2, 4, 5 and 7 were independently reproduced, including rebuilding the
40/0 pre-fix baseline in an isolated tree and proving the reproduce table really
reads git by repointing `PREFIX_REF`.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Criteria 1, 2, 4, 5, 7 MET and independently reproduced (I re-measured the 40/0 pre-fix baseline in an isolated tree built from the 178a6a59 blobs, proved the reproduce table is genuinely read from git by repointing PREFIX_REF, and drove BOTH full drivers with args UNBOUND against a perfect self-reported envelope -- the class-A cannot-pass property held on every path I could construct). Criterion 3 MET on its operative clause. Criterion 6 NOT MET: it explicitly names \"remove the blind-run marking ... Mutate BOTH scripts\", but qa-verdict.js's blind early-return was never mutated -- I mutated it and the mutant SURVIVED (checker still ALL GREEN 70/0) with a real behavioural differential (0 spawns -> 1 max-effort spawn labelled qa-verdict:UNSPECIFIED returning a PASS-shaped object); two further new guards also survived. Plus one undisclosed residual: a class-A research-gate run still spawns a max-effort researcher instructed to write handoff/current/research_brief_UNSPECIFIED.md, the exact artifact the step's own defect narrative names as the harm. Harness compliance clean 5/5; no unintended production change; cycle 1 with zero prior CONDITIONALs for 86.17, so the 3rd-CONDITIONAL auto-FAIL rule is not in play.",
  "violated_criteria": [
    "criterion_6_mutation_test_every_new_guard_BOTH_scripts",
    "illusory-guard: qa-verdict blind early-return has zero re-runnable coverage",
    "criterion_3_second_sentence: brief still writable under UNSPECIFIED identity via research-gate class A",
    "scope-honesty: the 'identical classifyArgs body' inference in experiment_results section 8 / live_check section 5"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Built three isolated mutant trees (copied checker + both workflow scripts into scratchpad, GIT_DIR pointed at the real object DB so section [1] still runs) and re-ran scripts/qa/verify_workflow_args_boundary.mjs. CONTROL: same tree, unmutated = exit 0, ALL GREEN 70 passed/0 failed.",
      "state": "M1 (qa-verdict.js: delete the entire `if (inputHealth.blind) { ... }` early-return block) -> checker exit=0, ALL GREEN 70 passed/0 failed: MUTANT SURVIVED. Non-equivalence measured through the real driver with args UNBOUND: baseline = 0 agents spawned, returns {\"dry_run\":true,\"verdict\":null,\"ok\":false}; M1 mutant = 1 agent spawned (\"qa-verdict:UNSPECIFIED\", agentType qa, effort max) and the agent's PASS-shaped object is returned verbatim -- i.e. removing the guard re-opens exactly the defect this step exists to close, with zero test signal. Root cause: the checker's drive() cuts the source at the FIRST of [\"if (inputHealth.blind)\", \"phase('Research')\", \"phase('QA')\"], so deleting that block merely moves the cut to phase('QA') and the [2] absent-args assertions (which only read classifyArgs' inputHealth.blind) stay green. M2 (qa-verdict.js: post-parse plain-object check -> if(false); this guard was mutated on research-gate ONLY) -> SURVIVED, 70/0. M3 (research-gate.js: empty/blank-string guard -> if(false); never mutated in either script) -> SURVIVED, 70/0. M2/M3 are diagnosis-level only -- the layering still throws, so the safety property holds -- but they are exactly the diagnosis-ownership standard the author defined in section [4] and then applied to only some cells.",
      "constraint": "Criterion 6 (immutable): \"MUTATION-TEST every new guard: reverting each one individually to the old behaviour (restore `catch (_e) { a = {} }`; restore `|| 'UNSPECIFIED'`; remove the blind-run marking) must make the specific new check FAIL. Mutate BOTH scripts. A guard whose mutant survives does not count and must be rewritten.\" Also qa.md 4c: a guard that cannot fail when its subject is broken does not count."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Drove the FULL .claude/workflows/research-gate.js driver (export stripped, wrapped in an async IIFE, runtime primitives agent/phase/log stubbed) with `args` UNBOUND -- class A -- and a stubbed agent returning a perfect envelope plus a perfect stage-2 verification.",
      "state": "Criterion 4 HOLDS: gate_passed=false, violations=[\"dry_run_no_step_id: ...\"], input_health={status:dry_run,blind:true}, self_report_disagreed=true, WARNING logged. BUT the same run spawns 2 agents, the first labelled \"research-gate:UNSPECIFIED\" (agentType researcher, effort max), whose prompt carries \"WRITE-FIRST IS NON-NEGOTIABLE. Create this file within your first few tool calls\" at exactly handoff/current/research_brief_UNSPECIFIED.md (measured: briefPathInPrompt = handoff/current/research_brief_UNSPECIFIED.md). qa-verdict.js closes this with an early return (0 spawns) and its in-code comment gives the reason -- \"no max-effort Q/A session is spent evaluating a step that was never named\" -- but the identical reasoning was not applied to research-gate.js, and neither experiment_results section 8 nor live_check section 5 discloses the asymmetry.",
      "constraint": "Criterion 3, second sentence (immutable): \"No brief and no verdict may ever be written under an UNSPECIFIED identity\" -- and the step's own defect narrative (contract section 1, experiment_results section 1) names research_brief_UNSPECIFIED.md, \"a name that collides across every step that ever hits that path\", as the harm being fixed."
    },
    {
      "violation_type": "Overgeneralization",
      "action": "Compared the two classifyArgs bodies token-by-token after stripping comments and unifying the script-name token, then compared the post-classifyArgs control flow of the two scripts.",
      "state": "experiment_results section 8 and live_check section 5 justify the un-launched research-gate class-A path with \"it shares the identical classifyArgs body\" with the qa-verdict launch that WAS proven live (wf_9e15e7ae-456). Measured: the bodies are NOT identical even after normalisation (first divergence at token 109: `if (!bound || raw === null) { return {...} }` vs `if (!bound || raw === null) return {...}`) -- that part is cosmetic. The material problem is that the inference does not transfer: the two scripts DIVERGE immediately after classifyArgs (qa-verdict early-returns with 0 spawns; research-gate proceeds to phase('Research') and spawns), which is precisely the behaviour being inferred. The author does disclose that the duplication is unguarded, but uses the sameness claim to cover a property the sameness does not cover.",
      "constraint": "Scope honesty / qa.md 4b claim auditing: an inference from a proven case must hold over the specific property being inferred; a shared helper does not license a claim about divergent downstream control flow."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_items",
    "immutable_verification_command_exit_code",
    "independent_prefix_baseline_reconstruction_at_178a6a59",
    "reproduce_table_git_liveness_probe_repointed_and_bogus_ref",
    "full_driver_class_A_simulation_both_scripts",
    "independent_mutation_testing_3_new_mutants_with_control",
    "mutant_behavioural_differential",
    "verbatim_throw_message_reproduction",
    "classifyArgs_cross_script_diff",
    "git_scope_and_unintended_change_check",
    "node_syntax_check_both_workflows",
    "live_check_artifact_cross_check",
    "code_review_heuristics",
    "contract_completeness_criterion_mapping"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE 5/5 CLEAN. (1) Research gate: handoff/current/research_brief_86.17.md on disk with envelope gate_passed=true, external_sources_read_in_full=10 (>=5), urls_collected=54, recency_scan_performed=true; contract cites it. (2) Order by mtime: brief 2026-08-09 17:24:23 < contract 22:13:45 < research-gate.js 22:18:52 / qa-verdict.js 22:19:24 < new checker 22:21:35 < experiment_results + live_check 22:23:39 -- contract genuinely precedes code. (3) experiment_results_86.17.md + live_check_86.17.md present. (4) LOG-LAST respected: `grep -cF \"86.17\" handoff/harness_log.md` = 0 and masterplan status=pending, retry_count=0/max 3. (5) Cycle 1: no evaluator_critique_86.17.* exists, so no verdict-shopping and the 3rd-CONDITIONAL rule is not engaged.\n\nDETERMINISTIC. Immutable command `bash -c 'node scripts/qa/verify_research_gate_workflow.mjs && node scripts/qa/verify_workflow_args_boundary.mjs'` -> exit 0; components exit 0 individually; \"ALL GREEN: 40 passed, 0 failed\" then \"ALL GREEN: 70 passed, 0 failed\". BASELINE INDEPENDENTLY MEASURED, not taken on trust: verify_research_gate_workflow.mjs is unchanged by this step (`git diff --name-only 178a6a59 HEAD` omits it), so I rebuilt an isolated tree from `git show 178a6a59:` for both that checker and the PRE-FIX research-gate.js and ran it -> \"ALL GREEN: 40 passed, 0 failed\". Delta 0 confirmed at BOTH ends; combined total 110. Scope: diff vs 178a6a59 = exactly 6 files (the 2 workflow scripts, the new checker, 2 handoff artifacts, CHANGELOG via the auto hook); working tree clean apart from a hook-appended handoff/audit/pre_tool_use_audit.jsonl. No *.py, no frontend/**, no backend/** in the diff, so the ruff F821/F401/F811 gate, the eslint+tsc gate, the backend runtime smoke and the live-UI capture gate are all N/A for this step. `node --check` OK on both workflow scripts; the pre-existing checker independently asserts the four Workflow-runtime constraints node --check cannot see.\n\nATTACKS I WAS ASKED TO RUN, AND THEIR OUTCOMES. (a) Baseline delta -- verified at both ends, see above. (b) Reproduce-table liveness -- it is genuinely read from git, not hardcoded: with ARGS_BOUNDARY_PREFIX_REF=HEAD the table changed to THROWs and check [1] went RED (69 passed, 1 failed, \"blind resolutions counted: 2\"); with a bogus ref both scripts reported \"pre-fix blob unavailable ... section [1] SKIPPED\" AND check [1] FAILED -- degrades loudly, never silently. (c) \"ABSENT cannot pass\" -- I could NOT defeat it. Driving the real research-gate driver with args UNBOUND and a stubbed agent returning a perfect envelope (gate_passed:true, 9 sources, 40 URLs, recency true, coverage.dry true) plus a perfect stage-2 verification still yielded gate_passed:false with the named dry_run_no_step_id violation, and self_report_disagreed:true was logged. The only enforceGate call site always threads inputHealth, the blind branch pushes unconditionally, and gate_passed is `violations.length === 0` -- there is no bypass of both layers. qa-verdict returned {dry_run:true, verdict:null, ok:false} with 0 spawns. (d) The corrected mutation method is a REAL test, not a weakened one: the CONTROL-then-mutate-then-require-a-different-diagnosis shape is sound and I confirmed each control phrase is genuinely produced -- my finding is not that the method is weak but that it was not applied to every new guard. (e) The two classifyArgs bodies are NOT identical (brace-style drift measured); more importantly the sameness does not cover the inferred behaviour -- see violation 3.\n\nVERBATIM CROSS-CHECK. I independently regenerated the throw messages and got byte-identical text to live_check section 2, e.g. `research-gate: args are PRESENT but not parseable as JSON (typeof=string isArray=false len=19 preview=\"{\\\"step_id\\\": \\\"86.17\\\"\") -- ...`. No splicing or editing detected in any block labelled verbatim; the checker output in live_check section 1 matches what I ran.\n\nNOTE-LEVEL, NON-BLOCKING. (i) The brief's envelope claims urls_collected=54 but only 23 unique http(s) URLs appear literally in the artifact; both binding floors (>=5 read in full, >=10 URLs) are still satisfied by what IS in the file, and this is a prior-session researcher self-report rather than a claim by this step's author -- recorded for transparency per the claim-auditing rule, not counted against the step. (ii) The disclosed gaps in section 8 are otherwise honest: I confirmed empty `criteria` on a present-args qa-verdict launch is still accepted (spawns qa-verdict:99.9 at effort max with no criteria) and that input_health has no consumer -- both are correctly declared out-of-criteria deferrals. (iii) The verdict-shaped `envelope` object is still returned alongside the enforced result, so a consumer reading `.envelope.gate_passed` would see the raw self-report `true`; that is the designed disagreement-reporting behaviour and is logged, not a defect.\n\nSELF-DISCLOSURE. My own launch carried present args with a step_id, so it exercises only the class-ok path; I treated it as a starting observation and verified everything independently as instructed. Like the author, I made no live class-B or class-C launch -- my class-B/C evidence is in-module plus full-driver simulation with stubbed runtime primitives. All mutation work was done on copies in the session scratchpad; the repository tree was never modified (git status is unchanged apart from the hook-appended audit stream).\n\nTO CLEAR THIS TO PASS (cycle 2, cheap and precise): (1) add a mutation cell for qa-verdict.js's blind early-return -- it must be driven through something that observes the RETURN/spawn, not the sliced classifyArgs, since drive() cannot see that block; and add cells for the post-parse plain-object guard on qa-verdict and the empty/blank-string guard on both. (2) Either mirror qa-verdict's early return in research-gate.js's class-A path (refuse to spawn when blind), or explicitly disclose the residual and queue it as its own masterplan step, and correct the section-8 wording so the class-A claim is not carried by the \"identical classifyArgs body\" inference."
}
```


---

## Cycle 2 verdict: **PASS** (`wf_4dc64bbb-039`)

Workflow rail, `model: opus`, 19 tool calls, 145,426 tokens, 446s. Main RECORDS
this verdict; Main did not author it. Transcribed VERBATIM.

It did not take the fix on trust. It **re-ran its own three cycle-1 survivors
against isolated mutated trees and all three now DIE**, then added **three NEW
mutants of its own** (delete research-gate's blind block; neuter enforceGate's
blind violation; flip `dry_run: true` to `false`) -- all killed. And it wrote
its **own independent full-driver harness** rather than trusting mine, to test
whether §[5] was simply the next illusory guard. It is not: the stub
demonstrably distinguishes spawn from no-spawn (blind = 0 on both scripts;
usable = 1 for qa-verdict and 2 for research-gate).

**TWO NON-BLOCKING NOTES it recorded, which I am NOT fixing in this tree** --
the verdict is in, and a gap noticed after EVALUATE belongs to the next cycle,
not to the tree being graded. Both are filed as **86.23**:

- **NOTE-1:** §[5]'s CONTROL cell reads `qa-verdict.js` only, so research-gate's
  zero-spawn cell has no control-side discrimination inside the checker -- and
  my §9a wording says "it carries a CONTROL" without disclosing it covers one of
  the two scripts.
- **NOTE-2:** the §[5] kill cells assert `spawns > 0 || threw !== null`. Both
  mutants kill on the strong left disjunct today, so it is not vacuous now, but
  a future mutant that merely throws would score as a kill without demonstrating
  a spawn.

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 7 immutable criteria MET and independently reproduced; the cycle-1 blocker is closed. Immutable command exit=0 (verify_research_gate_workflow.mjs ALL GREEN 40 passed/0 failed -- delta 0 vs the recorded baseline; verify_workflow_args_boundary.mjs ALL GREEN 87 passed/0 failed; combined 127 ok lines, matching the stated total exactly). CRITERION 6, the cycle-1 FAIL, is now genuinely satisfied: I re-ran my own three survivors against isolated mutated trees (control = ALL GREEN 87/0) and ALL THREE NOW DIE -- M1 (brace-matched deletion of qa-verdict's whole `if (inputHealth.blind)` block) exit=1 FAILED 82/5 with \"[5] qa-verdict.js: a blind run spawns ZERO agents -- spawned 1: qa-verdict:UNSPECIFIED\"; M2 (qa-verdict post-parse plain-object check -> if(false)) exit=1 85/1; M3 (research-gate empty/blank-string guard -> if(false)) exit=1 85/1. I added three NEW mutants of my own, all killed: M4 (delete research-gate's blind block) exit=1 82/5, \"spawned 2: research-gate:UNSPECIFIED, brief-verify:UNSPECIFIED\"; M5 (neuter enforceGate's inputBlind violation push) exit=1 85/2; M6 (flip `dry_run: true`->`false` in qa-verdict's blind return) exit=1 86/1. Section [5] is HONEST, not the new illusory guard: through my own independently-written full-driver harness (not the author's runDriver) the same stub demonstrably distinguishes spawn from no-spawn -- blind = 0 spawns on both scripts, usable = 1 spawn (qa-verdict:86.17) and 2 spawns (research-gate:86.17 + brief-verify:86.17) -- and M6 proves the \"cannot pass and is marked dry_run\" cell is not a tautology. The normal usable-args path is unbroken on both scripts (plain object AND valid-JSON-string forms; research-gate returns gate_passed:true with check \"input_health_ok: ok\"), and the blind early-return cannot fire on a usable launch: blind is reachable only from `!bound || raw === null`, and I exercised the adjacent shapes -- {}, {step_id:\"\"}, \"null\", array, scalar and double-encoded all THROW rather than going blind. Moving both returns below phase() changed nothing observable for a usable launch; enforceGate still sits above the slice boundary and everything below it is now covered by [5]. Criterion 3's second sentence is corroborated on the LIVE runtime, read by me directly from the run records: wf_a5de9d05-c78 agentCount=0 totalTokens=0 result {\"step_id\": null, \"gate_passed\": false, \"dry_run\": true, \"brief_path\": null}, and wf_9e15e7ae-456 agentCount=0 totalTokens=0 result {\"dry_run\": true, \"verdict\": null, \"ok\": false}. Harness compliance clean 5/5; scope clean (step diff touches only the 2 workflow scripts, the new checker, 3 handoff artifacts and the hook-generated CHANGELOG); Python/frontend/backend lint gates N/A by DERIVATION (git diff --name-only 178a6a59 HEAD -- '*.py' returns zero files, as does the frontend/backend/qa.md pathspec). Two NON-BLOCKING notes recorded (see notes): the [5] CONTROL covers qa-verdict.js only, and the kill cells carry an `|| r.threw !== null` escape hatch that currently passes on the strong disjunct.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command_exit_0",
    "regression_delta_vs_40_0_baseline_measured",
    "criterion_1_reproduce_table_regenerated_from_git_at_178a6a59",
    "criterion_2_verbatim_throw_messages_typeof_isArray_len_preview",
    "criterion_3_step_id_throw_both_scripts",
    "criterion_3b_zero_spawns_no_UNSPECIFIED_prompt",
    "criterion_4_absent_args_no_throw_and_cannot_pass",
    "criterion_5_measured_delta_40_plus_87_equals_127",
    "criterion_6_independent_mutation_matrix_6_mutants_isolated_trees_with_green_control",
    "criterion_7_stale_comment_read_at_source",
    "independent_full_driver_harness_written_by_evaluator",
    "control_discrimination_spawn_vs_no_spawn",
    "usable_path_regression_both_scripts",
    "blind_false_positive_probe_empty_object_empty_stepid_null_string",
    "live_workflow_run_records_wf_a5de9d05_c78_and_wf_9e15e7ae_456",
    "harness_compliance_audit_5_of_5",
    "git_scope_derived_178a6a59_to_HEAD",
    "python_lint_gate_NA_by_derivation_zero_py_files",
    "frontend_backend_gates_NA_by_derivation",
    "node_check_syntax_checker",
    "code_review_heuristics",
    "third_conditional_rule_checked_not_armed"
  ],
  "harness_compliance_ok": true,
  "notes": "HARNESS COMPLIANCE 5/5. (1) Research gate: handoff/current/research_brief_86.17.md envelope gate_passed:true, external_sources_read_in_full:10 (>=5), urls_collected:54 (>=10), recency_scan_performed:true; committed ea716698 08-09 19:42, before the contract. (2) Contract-before-generate: contract committed at 178a6a59 08-09 22:16, subject \"phase-86.17: contract (written BEFORE any code)\" -- and 178a6a59 is exactly the PREFIX_REF the checker uses as the PRE-FIX blob source, so the contract provably predates the code rather than merely claiming to; mtimes agree (brief 17:24:23 < contract 22:13:45 < code 22:34:20/22:35:34 < results 22:36:26). (3) experiment_results_86.17.md present (9 sections); live_check_86.17.md present, 15,227 bytes, 5 fenced blocks. (4) Log-last honoured: 0 rows matching \"phase=86.17\" in handoff/harness_log.md; masterplan status still \"pending\". (5) No verdict-shopping: evidence CHANGED between cycles -- a212dfe9 modified qa-verdict.js (+14), research-gate.js (+47), verify_workflow_args_boundary.mjs (+122) and all three artifacts. 3rd-CONDITIONAL rule NOT armed (cycle 1 was the only prior verdict).\n\nTWO NON-BLOCKING NOTES (recommend fixing in a follow-up; neither degrades the verdict):\nNOTE-1 (asymmetric control). Section [5]'s CONTROL cell reads SCRIPTS['qa-verdict.js'] only -- there is NO positive control that a usable research-gate launch spawns, so research-gate's \"a blind run spawns ZERO agents\" cell has no control-side proof of discrimination inside the checker. experiment_results \u00a79a says \"It carries a CONTROL (a usable launch must still spawn 1)\" without disclosing that it covers one of the two scripts. I closed the gap by measurement (research-gate usable = 2 spawns, gate_passed:true, checks include \"input_health_ok: ok\") and the M4 mutant supplies the discrimination independently (0 -> 2 spawns). Fix: add a research-gate CONTROL cell.\nNOTE-2 (OR-escape-hatch, qa.md 4c shape #8). The [5] kill cells assert `r.spawns.length > 0 || r.threw !== null`. Measured today both mutants kill on the STRONG left disjunct (qa-verdict spawns 1, research-gate spawns 2, threw=null in both), so the cell is not vacuous now -- but a future mutant that merely throws would be scored as a kill without demonstrating a spawn. Fix: require spawns > 0 and report a throw separately.\n\nKILL-MECHANISM ATTRIBUTION (qa.md 4c shape #11, naming WHICH assertion killed): M1/M4/M5/M6 die on genuine behavioural assertions (spawn count, returned object shape, violations list). M2/M3 die on the checker's ANCHOR-UNIQUENESS cell (\"found 0 occurrences\"), i.e. a text-moved signal rather than a behaviour-regressed signal -- that is inherent to a text-anchored matrix and is NOT a vacuity here, because the [4] cells perform their mutation IN-checker behind a CONTROL that requires the unmutated source to own its diagnosis, and all 8 control+kill pairs pass on the real code. Reporting \"found N\" instead of silently mutating nothing is the phase-75.5 no-match-replace defect class being explicitly defended against.\n\nMINOR OBSERVATIONS, no action required: (a) criterion 2 says the throw must name the received type AND SIZE -- `len=` IS emitted in every message (verified verbatim, e.g. `research-gate: args did not reduce to a plain object (typeof=number isArray=false len=2 preview=\"42\")`), but the [2] assertion only checks typeof/isArray/preview, so the size half is satisfied by the artifact rather than asserted by the checker. (b) \u00a79c's claim that the anchor-uniqueness assertion reported \"found 2\" for the dead block cannot be re-derived now that the block is gone; the mechanism is real and I watched it fire (\"found 0\") under M1/M2/M3. (c) research-gate's USABLE-path driver has no [5] cell (blind path and mutant only) -- I verified it by hand.\n\nDISCLOSED GAPS I CONFIRM ARE HONESTLY SCOPED in experiment_results \u00a78: no live class-B/class-C launch (in-module coverage across 10 shapes on both scripts is the substitute; the class-A path IS proven live); input_health has no consumer; empty `criteria` on a present-args qa-verdict launch still accepted (out of criteria, deliberately deferred); the duplicated classifyArgs is unguarded against cosmetic drift. The cycle-1 \"identical classifyArgs body\" inference is correctly WITHDRAWN and replaced by a live launch, and \u00a79's closing paragraph names the root cause accurately.\n\nCODE-REVIEW HEURISTICS (5 dimensions evaluated, zero BLOCK/WARN findings): the diff touches no execution path, no kill switch, no stop-loss, no perf-metrics, no secret literal, no subprocess/eval sink, no dependency pin. Dimension 4 anti-rubber-stamp: the mutation matrix is genuine and I re-ran it myself against mutated trees rather than reading it. Dimension 5: the CONDITIONAL -> PASS reversal is on CHANGED code (a212dfe9), which is the documented cycle-2 flow, not sycophancy-under-rebuttal.\n\nWHAT I DID NOT REACH within budget: I did not spend a real Workflow launch to observe a class-B/class-C throw on the live runtime (same bound the author disclosed); I did not exhaustively enumerate every args shape beyond the 10 in SHAPES plus the 4 adjacent edge shapes I probed by hand. Working tree at verdict time carries only hook-appended handoff/audit/pre_tool_use_audit.jsonl and handoff/away_ops/health.jsonl, both side effects of my own Bash calls -- no production file was modified by this evaluation."
}
```
