# Evaluator Critique — Step 71.1 (Workflow structured-output as the first-class Q/A + Researcher launch)

**Evaluator:** fresh, independent Q/A via the **newly-delivered** `.claude/workflows/qa-verdict.js` (dogfood) on
the Workflow structured-output path (Opus 4.8, `effort: max`, `model: opus`, $0 Max rail). Verdict = the captured
`agent()` return value; transcribed VERBATIM by Main (no-self-eval guardrail).

---

## Cycle 1 — VERDICT: PASS (run wf_c0f9d225-ab5; transcribed verbatim)

**ok:** true | **verdict:** PASS | **harness_compliance_ok:** true | **certified_fallback:** false
**violated_criteria:** [] | **violation_details:** []

**reason (verbatim):**
> All 4 immutable success criteria for step 71.1 are MET, harness compliance is 5/5 clean, the immutable
> verification command exits 0, and no unintended production code changed. C1 is empirically PROVEN by dogfood:
> this Q/A is running inside .claude/workflows/qa-verdict.js -- its PROMPT is byte-identical to my received prompt
> and my verdict is the captured agent() return value via VERDICT_SCHEMA, with no dependence on a file-write flush.
> C2: qa.md, docs/runbooks/per-step-protocol.md (+ Subagent-runtime-semantics), and CLAUDE.md each carry the
> 'Launch -- Workflow structured-output is FIRST-CLASS; Agent-tool is the fallback' section with the 'Main
> transcribes the returned verdict VERBATIM into evaluator_critique.md' guardrail (researcher.md too). C3:
> single-Q/A-per-step + exactly-3-agents + no-second-opinion-shopping / file-based fresh-respawn preserved
> unchanged; no separate harness-verifier or parallel pair re-introduced. C4: the roster-snapshot caveat is
> explicitly noted (Workflow reads qa.md from disk live; only the Agent-tool qa type snapshots at session start).
> Deterministic gates all green: verification cmd exit=0, node --check exit=0, tsc --noEmit exit=0, eslint 0 errors.

**checks_run (verbatim):** harness_compliance_audit_5_of_5, research_gate_gate_passed_true_7_sources_recency_scan,
contract_before_generate_mtime_ordering, immutable_verification_command_exit_0, node_check_qa_verdict_js_exit_0,
dogfood_prompt_identity_match, structuredoutput_schema_matches_VERDICT_SCHEMA, frontend_tsc_noEmit_exit_0,
frontend_eslint_0_errors, git_scope_no_production_code_changed, archive_baseline_hook_sideeffect_benign,
docs_C2_launch_section_grep, docs_C3_single_qa_exactly3_no_shopping_grep,
re_split_guard_no_harness_verifier_reintroduced, log_last_no_71.1_harness_log_entry,
no_verdict_shopping_first_qa_on_71.1, researcher_md_launch_note, R11_monitor_watchdog_exclusion_documented

**notes (verbatim excerpt):** INDEPENDENT dogfood evaluation (FIRST Q/A on 71.1). HARNESS COMPLIANCE 5/5
(research-gate gate_passed=true 7 sources; contract-before-generate mtime-proven; results present; log-last —
no phase=71.1 in harness_log, masterplan pending; no-verdict-shopping — critique on disk was still the 71.0 file).
DETERMINISTIC all green. C1 MET (empirical — I AM the live return-value proof, launched by this very script). C2
MET (all three named docs + researcher.md). C3 MET (single-Q/A + exactly-3 + no-shopping preserved; re-split guard
clean). C4 MET on the docs/caveat portion.
> NON-BLOCKING NOTES: (a) experiment_results.md claimed the JS header bakes in R1/R4/R11 but the header named only
> R1+R4+no-auto-PASS; R11 was documented in per-step-protocol.md, not the JS header — a documentation-location
> wording imprecision, not a substantive gap (R11 IS genuinely excluded + documented). (b) C4 REMINDER: the LOG
> step MUST include the separation-of-duties Peder-review request AND the note that
> scripts/qa/verify_qa_roster_live.sh must confirm the new qa.md/researcher.md roster next session.
> Also observed: stepId defaulted to 'UNSPECIFIED' / empty criteria → the args reached the script as a JSON string
> (empty-args launch), and the script self-recovered context from the masterplan + handoff.

---

## Cycle-1 → post-PASS hardening (Main; deterministic re-verify + fresh Q/A on changed code)

The verdict was PASS, but the Q/A's transparency surfaced two real items I acted on rather than deferring — the
harness's own core evaluator-launcher should not ship half-working:

1. **Robust args parsing** (the "empty-args launch" observation). `qa-verdict.js` now parses `args` whether it
   arrives as a parsed object, a JSON string (the Workflow tool stringifies scriptPath args on some paths), or
   absent — proven by a node unit check across all three forms (`string→71.1/2`, `object→71.1/1`, `undef→{}`).
   The parameterized launch now genuinely threads its parameters; empty/unparseable args still self-recover.
2. **R11 wording** (note a). R11 (no Monitor/transcript-mtime watchdog) is now named in the JS header alongside
   R1/R4/no-auto-PASS, so the experiment_results claim is accurate at its source; experiment_results wording synced.

Both are supersets (they cannot make any criterion FAIL). `node --check` OK; immutable verification still exit 0.
Because these edited the graded script, a **fresh Cycle-2 Q/A runs on the changed code, WITH args threaded** —
double duty: re-confirm the criteria hold on the new script AND prove the parameterized path end-to-end. This is
fresh-respawn-on-CHANGED-evidence, the documented pattern (not verdict-shopping — the cycle-1 verdict was already
PASS; the respawn re-verifies an improvement, it does not shop for a better verdict).

## Cycle 2 — VERDICT: PASS (fresh Q/A via qa-verdict.js WITH args; run wf_b1ed5841-167; transcribed verbatim)

**ok:** true | **verdict:** PASS | **harness_compliance_ok:** true | **certified_fallback:** false
**violated_criteria:** [] | **violation_details:** []

**reason (verbatim excerpt):**
> CYCLE-2 fresh Q/A on CHANGED code (post-PASS robustness hardening of qa-verdict.js; cycle-1 was PASS, so this is
> fresh-respawn-on-changed-evidence, NOT verdict-shopping). All 4 immutable criteria for 71.1 remain MET, harness
> compliance is 5/5 clean, and no backend/frontend production code changed. SPECIAL CHECK (A) PARAMETERIZED-PATH
> PROOF — CONFIRMED: my received prompt was shaped by .claude/workflows/qa-verdict.js with stepId='71.1'
> substituted AND the 4 immutable criteria enumerated INLINE (the criteria.map branch, not the fallback). I
> received step_id and criteria THREADED FROM ARGS via my prompt; I did NOT have to self-recover them. The
> args-threading fix WORKS. SPECIAL CHECK (B): node --check exit=0 and the immutable verification command exit=0;
> the hardening is strictly ADDITIVE and did NOT weaken any of the 4 criteria. C1 MET (empirical — I AM running
> inside it; verdict is the constrained-decoding return value, not a file-write flush). C2 MET (qa.md +42,
> per-step-protocol +37, CLAUDE.md +26, researcher.md +23). C3 MET (docs diff purely ADDITIVE, 128 insertions /
> 0 deletions; single-Q/A + exactly-3 + no-shopping preserved; no parallel pair / harness-verifier reintroduced).
> C4 MET (roster-snapshot caveat documented; the harness_log Peder-review + verify_qa_roster_live.sh note is a
> LOG-step commitment, correctly deferred per log-last). Deterministic gates all green.

**notes (verbatim excerpt):** PARAMETERIZED-PATH PROOF CONFIRMED WORKING. ARGS-PARSER independently replicated
(node -e on the actual lines): JSON-string→71.1/2, object→71.1/1, absent→UNSPECIFIED/0, unparseable→graceful (no
throw), camelCase→handled — matches the experiment_results claim exactly (real anti-rubber-stamp verification).
NON-BLOCKING carried forward: (a) C4 LOG-STEP REMINDER — the LOG entry MUST include the separation-of-duties
Peder-review request + the verify_qa_roster_live.sh next-session note; (b) frontend/tsconfig.tsbuildinfo untracked
build-cache artifact (tsc side-effect, arguably gitignore); (c) research_brief header stale "IN PROGRESS/2026-07-13"
(cosmetic; envelope authoritative). None affects any immutable criterion.

## Main's disposition (recorded; not a verdict edit)
- Both Cycle-2 non-blocking hygiene items were **fixed before the flip**: (b) `*.tsbuildinfo` + the specific path
  added to `.gitignore` (verified `git check-ignore` matches → the build cache can never be committed); (c) the
  research-brief header updated to `COMPLETE — gate_passed=true`, date 2026-07-17.
- (a) is actioned in the LOG step below (the separation-of-duties Peder-review request + `verify_qa_roster_live.sh`
  next-session note), satisfying the C4 harness_log clause.
- **Separation of duties honored:** Main authored the qa.md/researcher.md edits; the evaluator was an INDEPENDENT
  fresh general-purpose instance that read qa.md **fresh from disk** and evaluated the artifacts — it did not run
  under a Main-authored verdict. The Peder-review request below covers the standing agent-file-edit rule before a
  LATER step depends on the new wording.
