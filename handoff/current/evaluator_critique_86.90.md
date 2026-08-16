# evaluator_critique -- phase-86.90

**Cycle 1 verdict: CONDITIONAL** · run `wf_70a3e2c4-a6e` · 1 agent · 62 tool uses
· 235,499 tokens · 980 s · rail `.claude/workflows/qa-verdict.js` launched by
**scriptPath**.

Main records the verdict; Main never authors it. The Q/A return value is
transcribed VERBATIM below, with no editorial edit and no paraphrase.

## What the Q/A confirmed independently

All 7 immutable criteria MET on re-derivation, not on my prose:

- the reproduction and layer localisation taken from the run record
  `wf_4588d8a7-e70` itself;
- the 22-spawn census re-derived by its own strict receipt scan over 587 run
  records, with **symmetric difference EMPTY in both directions**, verdicts and
  step ids derived from `args.step_id` rather than typed
  (7 CONDITIONAL / 4 FAIL / 4 PASS / 7 drops = 22);
- criterion 7 proven by byte-identity of `VERDICT_SCHEMA` (1205==1205) and
  `enforceEscalation` (11451==11451) across the commit;
- my "the args-boundary checker was already red" claim **independently
  re-verified** by loading `enforceGate` from `a21a5889~1`, `a21a5889` and the
  working tree and getting byte-identical violation arrays;
- the guard survived its own 12-mutant matrix, including a fixture-side mutation
  of `runDriver`.

## The three WARN findings -- all accepted

| # | Finding | My response |
|---|---|---|
| D1 | `experiment_results_86.90.md:250` says "13 non-PASS verdicts and 6 rail drops"; the same table gives **11 and 7**. 13+6=19 > the 18 rows | Accepted. I asserted a split instead of counting it. Origin: 6 rows read `*(rail drop)*` and one reads `*(rail drop -- no verdict)*`, so a literal count misses the 7th. **Corrected by REPLACEMENT**, and now counted by a command |
| D2 | Four follow-ups asserted "queued" while **no masterplan step exists** for any of them | Accepted, and it is the sharper of the three. `.claude/masterplan.json` had not been committed since `c627a810`, which PREDATES the work commit. "Queued" prose without a step loses the follow-up and reads as done to the next reader -- the standing project rule I cite elsewhere. **Fixed by filing them** |
| E | The in-code absolute "THE RULE IS LOSSLESS-OR-THROW" over-states the measured guarantee: five constructions render lossily WITHOUT throwing | Accepted. The walk used `Object.keys` -- own **enumerable** string keys only. **Fixed by widening the walk AND narrowing the claim** |

E in detail, because it is a real hole and not only a wording problem: a
non-enumerable own data property was silently dropped; a **non-enumerable
`toJSON`** replaced the whole object with the single string `"REPLACED"` -- a
placeholder substitution reached *through* the guard written to forbid
substitutions; a non-deterministic getter was read once by the walk and **again**
by `JSON.stringify` (a TOCTOU); a nested non-enumerable was dropped; an array
with a non-index own property dropped it. Controls behaved correctly (an
enumerable `toJSON` THROWS; `Object.create(null)` renders losslessly).

**Reachability, stated so the severity is not inflated or deflated:** the Q/A
established that none of the five is reachable from a real caller --
`classifyArgs` either `JSON.parse`s a string or passes the runtime object
through, and a JSON-derived object has no non-enumerables, no getters and no
`toJSON`. So this was a CLAIM defect over a live one. It is fixed anyway: a guard
whose stated rule is broader than its measured behaviour is the failure mode this
whole series is about.

---

## Verdict, VERBATIM

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 7 immutable criteria are MET on independent re-derivation -- the reproduction and layer localisation come from the run record wf_4588d8a7-e70 itself (runtime_typeof_evidence=\"object\", script_concat_result=\"EVIDENCE / FILES TO READ: [object Object]\", agent_received.received_line=\"[object Object]\"), the 22-spawn blast-radius census reproduces with an EMPTY symmetric difference in both directions against my own strict receipt scan (verdicts and step ids re-derived from args.step_id, not typed: 7 CONDITIONAL / 4 FAIL / 4 PASS / 7 drops = 22), criterion 7 is proven by byte-identity of VERDICT_SCHEMA (1205==1205) and enforceEscalation (11451==11451) across a21a5889~1..a21a5889, and the guard survived my own independent 12-mutant matrix including a fixture-side mutation of runDriver. CONDITIONAL, not PASS, on three claim defects, all WARN-level and none of which touches the fix or any verdict: (D1) experiment_results_86.90.md:250 states \"13 non-PASS verdicts and 6 rail drops\" when its OWN table gives 11 and 7 (13+6=19 > the 18 rows), (D2) four follow-ups are asserted \"queued\" while .claude/masterplan.json has not been committed since c627a810, which PREDATES the work commit, and (E) the in-code absolute \"THE RULE IS LOSSLESS-OR-THROW\" over-states what was measured: five constructions render lossily without throwing.",
  "violated_criteria": [
    "scope_honesty: numeric rollup in experiment_results does not reproduce (13/6 vs 11/7)",
    "scope_honesty: four follow-ups claimed 'queued' with zero masterplan steps",
    "criterion_5_residual: 'lossless-or-throw' absolute over-states the measured guarantee"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "re-derive the disposition split from Main's own blast-radius table (experiment_results_86.90.md:214-236) and from the workflow run records",
      "state": "prose at experiment_results_86.90.md:250 says '13 non-PASS verdicts (CONDITIONAL/FAIL) and 6 rail drops'; the table and the run records both give 7 CONDITIONAL + 4 FAIL = 11 non-PASS and 7 drops. 13+6=19 exceeds the 18 non-PASS-or-dropped rows in the same table. Origin: 6 rows are labelled '*(rail drop)*' and 1 is labelled '*(rail drop -- no verdict)*', so a literal count misses the 7th. The TABLE is exactly right and reproduces; only the rollup sentence is wrong, and the disposition is unaffected because both sub-buckets are the 'no re-grade needed' class.",
      "constraint": "SEVERITY WARN. qa.md 4b -- every numeric claim in experiment_results must reproduce from the command that produces it"
    },
    {
      "violation_type": "Contradiction",
      "action": "walk .claude/masterplan.json for the deferred items, then run `git log --oneline -5 -- .claude/masterplan.json`",
      "state": "contract_86.90.md P2 and section 8 assert 'queued' for (a) the 85.5 / 86.25 / 86.34 re-grades, (b) harness-self-audit.js:68 (same concat shape), (c) stronger unknown-key treatment, (d) the stale .claude/workflows/qa-verdict.js.export.mjs. NONE exist as steps: the masterplan's newest commit is c627a810, which PREDATES a21a5889. Keyword greps that appear to hit ('harness-self-audit', 'unknown key', '86.92') all resolve inside unrelated steps' prose. Main DID file 86.91 correctly in its own commit, so this is an omission rather than disregard. The live-RED verify_workflow_args_boundary.mjs is likewise unqueued.",
      "constraint": "SEVERITY WARN. Standing project rule -- every discovered out-of-scope defect gets its OWN masterplan step; 'queued' prose without a step loses the follow-up and reads as done to the next reader"
    },
    {
      "violation_type": "Overgeneralization",
      "action": "drive the SHIPPED renderArgField (module sliced at the phase('QA') driver boundary) with values whose loss an Object.keys walk cannot see",
      "state": "Five constructions render LOSSILY and do NOT throw: a non-enumerable own data property (silently dropped); a NON-ENUMERABLE toJSON (the whole {real_evidence, more} object rendered as the single string \"REPLACED\" -- a placeholder substitution reached through the guard written to forbid them); a non-deterministic getter (the walk reads it once, JSON.stringify reads it AGAIN); a nested non-enumerable; an array with a non-index own property. Controls behaved: enumerable toJSON THROWS, Object.create(null) renders losslessly. REACHABILITY: classifyArgs either JSON.parses a string or passes the runtime object through, and a JSON-derived object has no non-enumerables, no getters and no toJSON -- so none of the five is reachable from a real caller, which is why this is a claim defect rather than a live one.",
      "constraint": "SEVERITY WARN. Criterion 5 -- fail loudly on a value it cannot render rather than substituting a placeholder; the in-code claim 'THE RULE IS LOSSLESS-OR-THROW. Never coerce, never substitute.' is broader than the measured guarantee"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command (exit=0, stdout 'parses')",
    "node --check on BOTH workflow scripts",
    "verify_prompt_render_86_90.mjs (ALL GREEN 53/0)",
    "verify_research_gate_workflow.mjs (ALL GREEN 124/0)",
    "verify_escalation_86_78.mjs (51 checks, 0 failed)",
    "verify_rail_retry.mjs (ALL GREEN 38/0)",
    "verify_workflow_args_boundary.mjs (84 passed, 3 failed -- pre-existing, proven)",
    "pre_existing_failure_A/B (enforceGate loaded from a21a5889~1 vs a21a5889 vs working tree: byte-identical violations; enforceGate source identical 11843==11843)",
    "blast_radius_census_re-derived (strict receipt scan of every agent transcript; symmetric difference vs Main's 22 EMPTY both ways; positive control wf_b1747d75-eec found)",
    "verdict_and_step_id_re-derivation_from_run_records (587 records; args.step_id derived, not typed)",
    "86.86_receipt_verbatim (prompt lines 61 and 63 read directly from wf_b1747d75-eec)",
    "run_record_verification (wf_4588d8a7-e70 pre-fix probe, wf_9bd7e233-f38 gate, wf_a09930e2-3d7 re-grade)",
    "criterion_5_adversarial_lossy_render_probe (8 constructions + 2 controls)",
    "independent_mutation_matrix (12 mutants the author did not use, scored by failing-cell NAME diff)",
    "fixture_side_mutation (runDriver blanked -> section [0] goes RED)",
    "anchor_uniqueness_recount (all 4 author cells = 1 occurrence)",
    "criterion_7_byte_identity (VERDICT_SCHEMA + enforceEscalation across the commit)",
    "P2_unknown_key_warning_behavioural_drive (fires + clean negative control)",
    "git_scope_audit (git show --stat a21a5889; a21a5889..HEAD drift; working-tree cleanliness of the changed files)",
    "secret_scan_on_diff",
    "code_review_heuristics",
    "research_gate_compliance (envelope + enforced gate record)",
    "harness_compliance_5_item_audit",
    "qa_wip.py + verdict_history_86_21.py --evidence-only"
  ],
  "harness_compliance_ok": true,
  "notes": "SEQUENCE: qa_wip.py 86.90 --spawned-at 2026-08-16T08:25:15Z -> source_present=true, attempt_number=1 (status \"ok\"), prior_attempts=0, records_retained=1 (my own record). verdict_history_86_21.py --step 86.90 --evidence-only -> status=no_rows_for_step, verdicts=(none). prior_attempts (0) == ledger count (0), so the two sources AGREE and the ledger is not detectably stale for this step id; note the ledger is hand-written in general and 86.85 is open on exactly that. Cycle 1, no prior verdict -> no verdict-shopping possible.\n\nHARNESS COMPLIANCE, all 5 clean: research_brief_86.90.md 09:59:05 < contract 10:01:10 < qa-verdict.js 10:02:56 < guard 10:06:10 < experiment_results 10:22:06; gate ENFORCED not self-reported (wf_9bd7e233-f38: gate_passed=true, self_report_disagreed=false; brief envelope brief_status COMPLETE, 12 sources, 45 URLs, recency true, coverage.dry); experiment_results + live_check present; masterplan 86.90 status is still \"pending\" and `grep -cF 'phase=86.90' handoff/harness_log.md` = 0, so log-last is respected.\n\nNO UNINTENDED PRODUCTION CHANGE: a21a5889 is 8 files -- the 2 workflow scripts, the guard, 4 x 86.90 artifacts, and the 86.91 brief (disclosed by Main). Nothing in backend/ or frontend/. The uncommitted edits to backend/api/sovereign_api.py and frontend/src/* appear in NONE of a21a5889 / c627a810 / 8dc70502 / 952ed521 -- peer-session work, Main's claim VERIFIED. HEAD has moved 4 commits past a21a5889 (86.91 work); the only 86.90 artifacts touched since were experiment_results/live_check, updated to record the 86.86 re-grade RESULT -- I graded the current files.\n\nMAIN'S THREE SELF-FLAGGED WEAK POINTS, ruled on:\n(1) verify_workflow_args_boundary.mjs \"3 failed is pre-existing\" -- VERIFIED, by a DIFFERENT method than Main used. I loaded enforceGate from a21a5889~1, a21a5889 and the working tree and drove the checker's exact case-[3] fixture: byte-identical violation arrays in all three, and the enforceGate source is byte-identical pre/post (11843 chars). Root cause is fixture drift, not 86.90: handoff/current/research_brief_86.17.md (written 2026-08-09) has `grep -c brief_status` = 0 and phase-86.37 later made the marker mandatory. That checker is nonetheless live-RED and unqueued -- name it.\n(2) Criterion 4 completeness -- the 22 is defensible as a FLOOR and I found no laundering into a total: the array-coercion blind spot and the pruned-session blind spot are both disclosed, and the census is receipt-keyed. My independent strict scan returned 23 receipt-confirmed runs; the 23rd is Main's own declared pre-fix probe wf_4588d8a7-e70. A loose grep returns 4 extra (the probe, the 86.90 gate run, the 86.86 re-grade, and one same-session agent that had READ the contract) -- all prose contamination, correctly excluded. On \"does stating it suffice for 85.5/86.25/86.34\": YES on the criterion's literal terms -- it says \"state for each\", and singles out only 86.86 as requiring explicit resolution, which was done and whose run record I verified (verdict PASS, ok true, violated_criteria []). The gap is that the three re-grades are not actually queued (D2).\n(3) Criterion 6 anchor-uniqueness -- I recounted all four author anchors independently: 1, 1, 1, 1. `topic` IS routed through renderArgField (research-gate.js:282), so cell 2 mutates the real fix. On \"is each kill about the guard, not the probe\": my first name-stripping mutant SURVIVED and I retired it as a CONSTRUCTION ARTIFACT rather than reporting it -- the violation string is seeded with `where`, so the field name survived my edit. Rebuilt correctly and applied to BOTH copies (so [6] byte-identity could not be the killer), it was KILLED by 12 novel cells, all of them the [3] \"...THROWS and names the field\" assertions. That assertion is genuinely load-bearing.\n(4) Log-only unknown-key handling -- NOT a scope dodge. Main's rationale reproduces: qa-verdict.js:683 carries the phase-86.78 invariant that throws if caller-authored fields become siblings of the judge's output, so returning them is the thing that invariant forbids. Throwing on unknown keys would also fail-closed on any additive key for every existing caller. I drove it behaviourally: it fires on {questions, bogus} and on research-gate's {questions}, with a clean negative control on all-known keys. It has no check() coverage in the committed guard -- NOTE (unguarded addition), not a criterion miss, since criterion 6 is about stringification.\n(5) Not editing .claude/agents/qa.md -- correct call, and it does NOT leave criterion 5 short: a machine-enforced throw in the script is strictly stronger than a prompt instruction, and CLAUDE.md's separation-of-duties rule discourages a session authoring an agent .md that its own evaluator then depends on.\n\nMY OWN MUTATION MATRIX (12 mutants the author did not use; staged in a scratch copy, the repo was never written; scored by DIFFERENCE IN FAILING CELL NAMES against a staged control, because the staged control loses exactly the 2 git-dependent [1]/[4] cells and that offset is identical on every row): drop the lossless walk KILLED(9); arrays bypass the renderer KILLED(3); non-plain-instance check removed KILLED(3); finite-number guard removed KILLED(3); undefined-member check removed KILLED(3); renderer truncates every key KILLED(3); throw stops naming the field KILLED(12); identity arg renders objects instead of throwing KILLED(4); FIXTURE-SIDE runDriver reports no spawns KILLED(16, section [0] goes red -- the fixture is non-vacuous). One survivor: neutering the guard's own mutation SCORER produces zero new failures. That is the inherent watcher regress, it terminates at this independent matrix by design, and the anchor-uniqueness pre-check covers the shape that has actually bitten this repo -- NOTE, not a finding against criterion 6.\n\nFIRST-PERSON RECEIPT: this spawn's own prompt carried EVIDENCE / FILES TO READ and ADDITIONAL CONTEXT as fenced ```json blocks, not the literal -- the fix is working on the evaluator grading it.\n\nWORST-OF-N LENSES: correctness=PASS, does-it-reproduce=CONDITIONAL (D1, D2), scope-honesty=CONDITIONAL (D2). min = CONDITIONAL.\n\nTO CLOSE: (a) correct the 13/6 rollup to 11/7 in experiment_results_86.90.md:250; (b) file real masterplan steps for the four \"queued\" items plus the live-RED verify_workflow_args_boundary.mjs; (c) soften the in-code absolute to name its measured scope (JSON-derived values), or extend the walk to own non-enumerables and reject a toJSON. None of these touches the fix, the guard, or any verdict.\n\nWIP record (evidence for any next spawn, NOT a verdict): .claude/agent-memory/qa/verdicts/verdict_wip_86.90__20260816T082515Z.md. No write was blocked. UI gate N/A (no UI claims); ruff gate N/A (no *.py in the diff); backend smoke N/A (no backend/** in the diff).",
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
