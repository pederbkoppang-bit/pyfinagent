# Evaluator critique -- step 86.28
**Phase**: EVALUATE  
**Date**: 2026-08-10  
**Cycle**: 1  
**Launch**: Workflow rail `.claude/workflows/qa-verdict.js`, run `wf_10c6cbd2-cad`  
**Agent**: `qa` / `claude-opus-5[1m]` / 176,900 tokens / 35 tool calls / 654s

> Transcribed VERBATIM from the Q/A return value. Main records the verdict;
> Main does not author it. No editorial edits, no paraphrase.

---

## Verdict: **CONDITIONAL**  (`ok: False`, `certified_fallback: False`, `harness_compliance_ok: True`)

### reason

All 9 immutable criteria are met on their literal wording and every numeric claim reproduced independently (baseline 40/0 re-run at the pre-change commit; post 61/0; all 9 mutants KILLED; research brief 7 sources / 48 distinct URLs >= 34 claimed / recency section present) — but two WARN-level findings block a clean PASS: (W1) the criterion-8 ordering assertion is a first-match `indexOf` source scan that I DEFEATED with an executed mutant (comment token before the spawn + real refusal moved after it => checker still printed "ALL GREEN: 61 passed, 0 failed"), and (W2) the criterion-7 doc fix introduced three NEW false line-number citations, two of them into CLAUDE.md and researcher.md.

### violated_criteria

- criterion_7_doc_accuracy: the doc-drift fix introduced three stale line-number citations (research-gate.js:419 -> actual :584 in BOTH CLAUDE.md and researcher.md; verify_research_gate_workflow.mjs:271 -> actual :399 in live_check §8)
- illusory-guard [WARN]: the 'refusal is placed BEFORE the researcher spawn' assertion is defeatable by a comment token (qa.md §4c vacuity shapes #2 + #8), demonstrated by an executed surviving mutant

### violation_details

#### 1. Contradiction

**action**

```
git diff cad38647^..HEAD -- CLAUDE.md .claude/agents/researcher.md; sed -n '417,421p' .claude/workflows/research-gate.js; grep -n "agentType: 'researcher'" .claude/workflows/research-gate.js
```

**state**

CLAUDE.md:275 and researcher.md:77 (both lines ADDED this cycle) assert "The shipped code pins it at .claude/workflows/research-gate.js:419". MEASURED: the pin is at :584. Line :419 now reads `checks.push('not_audit_class: coverage.dry informational only')`. Root cause: :419 WAS correct at BASE 089726f9 (verified: `git show 089726f9:...js | grep -n` returns 419), and the author's own +208-line edit moved it — the citation was written pre-edit and never re-derived. live_check_86.28.md:190 adds a third: it credits the checker's agentType assertion to verify_research_gate_workflow.mjs:271, but :271 is a comment about the refusal-path regression; the real assertion is :399.

**constraint**

SEVERITY WARN. Criterion 7 — 'the doc drift is fixed so .claude/agents/researcher.md, CLAUDE.md and the shipped code agree'. The agentType VALUE does agree in all three (criterion met literally), but a doc-accuracy fix must not ship new false claims into governance files loaded into every session. qa.md §4c shape #10 (labels derived before the last edit); CLAUDE.md carries an explicit standing warning of this exact class ('Re-derive the line number before citing it again -- it has moved twice').

#### 2. Circular_Reasoning

**action**

```
Executed mutant M5 (repo reconstructed in mkdtemp, zero repo writes): inserted `// harmless note: if (tierUnsupported) { we would refuse here }` before the spawn and relocated the REAL refusal block to AFTER `const envelope = await agent(PROMPT`, then ran the author's own checker unmodified.
```

**state**

Checker output: `ALL GREEN: 61 passed, 0 failed`. The mutant SURVIVED. Both `src.indexOf('if (tierUnsupported) {')` and the spawn indexOf are first-match over raw source including comments, so a comment token satisfies the ordering guard while the production refusal sits after the spawn. Contrast: my mutants M1 (suppression `if (!tierUnsupportedHere)` -> `if (true)`) and M3' (exact pre-86.28 empty-guard shape restored) were both KILLED (60/1 and 59/2), so the enforceGate-side regression checks the author added are genuinely load-bearing and NOT vacuous.

**constraint**

SEVERITY WARN, not BLOCK. qa.md §4c — 'a guard that cannot fail when its subject is broken does not count'; shapes #2 (source-scan defeated by moving the scanned text) and #8 (comment-token trap). WARN rather than BLOCK because a genuine behavioural guard coexists: live run wf_23d9ed4b-22c measured agentCount 0 / totalTokens 0 / durationMs 5, and enforceGate's complete mediation (mutation-tested) means a defeated ordering guard costs wasted tokens, not a false certification. Named fix: strip `//` lines before indexing, or match the block body comment-insensitively, e.g. /if \(tierUnsupported\) \{[\s\S]*?return \{/.

### checks_run (23)

- `harness_compliance_audit_5_item`
- `research_gate_corroboration_independent`
- `mtime_ordering_research_contract_code_results`
- `immutable_verification_command_exit_0`
- `baseline_40_0_reproduced_at_pre_change_commit`
- `no_preexisting_check_name_deleted_or_renamed`
- `all_6_preexisting_mutants_still_killed`
- `3_new_mutants_reproduced_killed`
- `independent_mutant_M1_suppression_KILLED`
- `independent_mutant_M3prime_original_defect_KILLED`
- `independent_mutant_M5_ordering_decoy_SURVIVED`
- `fixture_mutation_F1_F2_both_red`
- `fail_closed_probes_on_omitted_stage2_fields`
- `guard_vacuity_check_4c`
- `claim_auditing_4b`
- `derived_changed_file_scope`
- `python_lint_gate_NA_zero_py`
- `frontend_lint_typecheck_NA_zero_frontend`
- `backend_runtime_smoke_NA_zero_backend`
- `live_ui_capture_NA_no_ui_claims`
- `node_check_both_scripts_exit_0`
- `code_review_heuristics`
- `3rd_conditional_counter_grep`

### notes

HARNESS COMPLIANCE (5/5 clean). Research gate PASSED and independently corroborated by me, not taken on trust: brief 41,731 bytes, recency section at line 92 ("## Recency scan (2024-2026) -- PERFORMED"), 48 distinct URLs on disk >= the 34 claimed, envelope gate_passed true / 7 sources (floor 5). mtime chain strictly ordered: research 08:38:58 < contract 08:41:55 < code 08:46:12-08:49:06 < experiment_results 08:50:42 < live_check 08:51:20. Criteria were committed at 85acb789 (08:27:34) BEFORE research and code — so criterion 3 is NOT post-hoc criteria-shopping. Step still status=pending, zero harness_log entries for 86.28 (log-last OK). Cycle 1: no evaluator_critique_86.28 exists, so no verdict-shopping is possible and the 3rd-CONDITIONAL rule does not apply (0 prior CONDITIONALs).

CRITERION-BY-CRITERION. (1) MET, verified three independent ways rather than by reading the write-up: I re-ran the checker at BASE 089726f9 in a reconstructed tmp repo and got exactly "ALL GREEN: 40 passed, 0 failed"; post is 61/0; ZERO of the 29 pre-existing named checks are missing from the new file; and all 6 pre-existing mutants are still KILLED — which is itself the proof of no masking, since a mutant is only KILLED if removing THAT guard lets the probe through, i.e. no new guard and no friendlier fixture is covering for it. (2) MET — tier_requested/tier_applied/tier_supported are in the RETURN object (:685-690) and in the live JSON; ABSENT vs UNSUPPORTED are distinguishable (absent => tier_requested null, no violation, gate can still pass; unsupported => violation + refusal). (3) MET — VALID_TIERS is still ['simple','moderate','complex'] (:187); every 'deep' occurrence is a comment; only 2 agent() calls exist (:580 stage-1, :602 stage-2), i.e. the pre-existing two-stage design, no producer fan-out; divergence disclosed with two operator options. The underlying justification is verbatim-corroborated: researcher.md:253 is literally "Multi-subagent fork option" with "2-3 parallel deep-tier researcher subagents", "Each subagent must meet the >=20-source floor INDEPENDENTLY" and "~1 Claude Max 5-hour rolling window per subagent". (4) MET. (5) MET — I reproduced all 9 mutants KILLED and added 2 of my own. (6) MET — the diff touches coverage.dry/opts.floors in a COMMENT only. (7) MET on the literal wording, with the W2 defect above. (8) MET — model 'opus', 0 static imports, exactly 1 export, 0 minimum/minItems, 0 Monitor(, gate_passed not const:true, node --check exit 0. (9) MET on its literal verb ("a LIVE spawn ... is exercised at least once after the change") — wf_23d9ed4b-22c qualifies; the criterion does not require the full stage-1+stage-2 path.

DIRECT ANSWERS TO THE SIX AUDIT QUESTIONS ASKED. (1) Fixture masking: DISPROVEN by execution, and the fixture change was necessary rather than lenient. Reverting makeBrief to its pre-change shape (F1) and dropping the snippet URLs (F2) each turned 8 checks RED including "control passes". So the compliant fixture is REQUIRED for the truthful control envelope to pass — it was made honest, not permissive. (2) Refuse-to-spawn "one seam short": YES, partially — this is W1 above, and it is a real executed finding, not a theoretical one. (3) Criterion 9 gap: ACCEPTABLE, and I upgraded the author's argument to a measurement. They claimed the failure direction is safe; I verified it rather than accepting it — with distinct_urls_in_brief omitted, recency_section_present omitted, BOTH omitted (the exact legacy stage-2 shape), and both null, the gate returned gate_passed:false in all four cases. So the residual risk is a spurious gate FAILURE (loud, self-announcing, recoverable), never a false certification. Not a criterion miss. (4) Criterion 3 re-scope: NOT a silent scope reduction — the immutable criterion itself mandates not adding 'deep', and it was committed 11 minutes before the research brief and ~19 before the code. (5) The live-caught defect's regression checks are NOT vacuous — proven by two of my own mutants: restoring the exact pre-86.28 empty-guard shape (M3') fails 2 checks, and removing only the suppression (M1) fails 1. (6) Content intact regardless of attribution: working tree is clean apart from a hook-appended audit JSONL; the four production files match the declared list; the checker runs 61/0 against the on-disk files. Attribution to the peer's cad38647 is out of scope (queued as 86.15).

SCOPE HONESTY (strong). The author disclosed the criterion-9 gap unprompted, disclosed a defect they introduced and fixed mid-cycle, disclosed the deep-tier divergence with two operator options, flagged their own separation-of-duties concern on editing researcher.md, and listed four deliberate omissions with reasons. Nothing I checked was overstated in the author's favour.

TWO ADDITIONAL NOTES (no verdict effect). N1: the -1 sentinel from `n()` leaks into an operator-facing string — an omitted field renders as "only -1 distinct URLs appear in the brief". Fails closed correctly, but the message is confusing; consider "no count reported". N2: experiment_results and live_check call the fork "deep's FOURTH REQUIREMENT"; researcher.md:253 titles it an "option" conditioned on "If the caller requests it, OR if the topic has >=3 clearly separable sub-questions". Slightly overstated as unconditional — this does not change criterion 3, which independently mandates not adding deep.

GATES CORRECTLY N/A, scope DERIVED via `git diff --name-only` (not hand-typed): 0 *.py (§1a Python lint), 0 frontend/** (§1b), 0 backend/** (§1d), qa.md untouched, and the step makes no UI claims (§1c). I took no Playwright capture because none is required; no verdict here rests on evidence produced by the author except the two workflow run IDs, which I could not re-execute (the Workflow runtime is not reachable from my surface) — I state that plainly rather than implying I re-ran them.

TO CLEAR TO PASS (both small, ~10 minutes): (a) correct the three line-number citations — research-gate.js:419 -> :584 in CLAUDE.md and .claude/agents/researcher.md, and verify_research_gate_workflow.mjs:271 -> :399 in live_check §8 (or drop the line numbers and cite the symbol, which cannot go stale); (b) harden the ordering assertion against the comment-token defeat, e.g. strip `//` lines before indexOf or match /if \(tierUnsupported\) \{[\s\S]*?return \{/, then re-run the checker and add my M5 as a standing mutant so the ordering has a re-runnable regression guard rather than a one-time live observation. Neither finding impugns the shipped behaviour, which I found correct under every probe I ran.

---

# Follow-up by Main -- cycle 2

> Appended, not edited. Everything above is the Q/A's verbatim return and
> stays untouched; this section is Main's response to it.

Both WARN findings were real and both are fixed. Evidence is in
`live_check_86.28.md` §9-§13 and `experiment_results_86.28.md`
"Follow-up -- cycle 2".

## W1 -- illusory ordering guard: FIXED and now watched failing

- Predicate extracted as `refusalPrecedesSpawn(src)`; strips `//` lines
  before indexing and matches the refusal as a block reaching `return {`.
- Three standing vacuity tests added, including one asserting that M5
  still defeats the ORIGINAL naive predicate -- so the probe cannot go
  quietly stale.
- **Independently reproduced the Q/A's method**: a repo copy with M5
  applied, run against the unmodified checker, now gives
  `FAILED: 62 passed, 2 failed` where cycle 1 gave `ALL GREEN: 61 passed,
  0 failed`. Mutant KILLED.

## W2 -- false line citations: FIXED, and the CLASS was audited

The Q/A named three. I grepped every `file:line` citation written this
cycle and found **six**, including one in `experiment_results` the Q/A did
not flag. All present-tense claims now cite the SYMBOL instead of a line,
because every `researcher.md` citation was stale by ~7 lines from this
cycle's own edit to that file -- including the Q/A's own `researcher.md:253`
for "Multi-subagent fork option", which measures at `:255`.

Not rewritten, with reasons: the contract (a PLAN records what was
planned -- annotated instead), the research brief (another agent's evidence
artifact), and the Q/A verdict above (verbatim; editing it would break the
no-self-eval guarantee).

## Q/A notes

- **N2 ACCEPTED** -- I called the fork deep's "fourth requirement"; it is
  deep's fourth listed element and it is *conditional*. The Q/A is right
  and I was overstated. Corrected in both artifacts. Criterion 3 is
  unaffected (it independently mandates not adding `deep`), and so is the
  reason for refusing: a conditional fork on an N=1 artifact rail is still
  a fork the rail cannot support.
- **N1 QUEUED, not patched** -- the `-1` sentinel renders as "only -1
  distinct URLs". Fails closed correctly; message is confusing. Outside
  the frozen criteria and the tree was under evaluation.

## Checker

40 (baseline) -> 61 (cycle 1) -> **64 (cycle 2)**, 0 failed. No check
removed or weakened.

## What did NOT change

No production behaviour. The Q/A found all 9 criteria met on their literal
wording and every shipped probe correct; cycle 2 touched only the checker's
ordering guard and documentation accuracy.
