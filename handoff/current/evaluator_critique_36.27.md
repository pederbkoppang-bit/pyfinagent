# Evaluator critique — phase-36.27

**Cycle 187, EVALUATE pass 1.** Workflow rail run `wf_d5022922-89f`, agent `qa`,
model `opus`, effort `max`, 35 tool uses, 178,786 tokens, 659s.

Main did NOT author this verdict. Transcribed **VERBATIM** below.

## Verdict: **CONDITIONAL**  (`ok: false`) — all 6 criteria MET; two WARN findings

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "All 6 immutable criteria are MET and I reproduced every one independently rather than reading the author's claims: immutable command exit=0; `node scripts/qa/verify_research_gate_workflow.mjs` -> ALL GREEN 40 passed 0 failed; I replayed the SHIPPED `enforceGate` against the live journal (wf_9880694c-d30) and it reproduces experiment_results 4.3 field-for-field and string-for-string (all 7 checks) - the \"verbatim\" capture is genuine, not spliced; I re-verified all 8 claimed URLs myself (8/8 present, char_count 36790 = `wc -m` exactly, 36998 bytes = `wc -c`, both figures correct); I corroborated write-first by a route the author did not cite (stage-1 transcript shows 3 SEPARATE Write calls to research_brief_86.1.md across 38 tool_use blocks, and 38+2=40 reproduces the \"40 tool uses\" claim); I built an INDEPENDENT 7-mutant matrix on guards the author's matrix does NOT cover (brief_exists, brief_non_empty, urls_missing, recompute-defeated, empty-return, n() coercion, selfReported-hardcoded) -> 6/7 killed, the 1 survivor being a layered-defense equivalent killed by the over-claim guard, not a vacuity; and I proved HARNESS non-vacuity by rebuilding the checker in an isolated mimic tree (os.tmpdir, repo untouched) and running it against 6 mutated workflows - control exit 0, all 6 mutants drove it RED exit 1. Harness compliance clean on all 5 items (research 12:07:36 < contract 12:08:31 < code 12:16:15 < results 12:29:42; masterplan still `pending`; ZERO `phase=36.27` entries in harness_log, so this is cycle 1 and the 3rd-CONDITIONAL rule does not apply). Scope DERIVED from git: 10 files, 0 *.py / 0 frontend/** / 0 backend/**, so the Python-lint and frontend gates are N/A by measurement not assertion; masterplan diff is exactly 1 insertion + 1 deletion on the `name` line with the `verification` block untouched as context (disclosure 7 verified). CONDITIONAL on two named, minutes-to-fix WARN findings, not on any criterion miss: (F1) the `NO static node: imports` structural guard at verify_research_gate_workflow.mjs:256 has 1-of-6 recall against the known-member set - it catches only the single-quoted default-import form the author happened to measure and MISSES the double-quoted form of that identical construct, plus named, namespace, side-effect-only and bare-specifier imports; it is the ONLY automated repeatable defense against the exact class this step discovered (green `node --check` on an unlaunchable script), since criterion 2's live spawn is manual. Named fix: assert ZERO static imports, `(src.match(/^\\s*import\\b/gm)||[]).length === 0`, which the shipped file already satisfies. (F2) `handoff/current/live_check_36.27.md` does not exist while masterplan 36.27.verification.live_check is set; live_check_gate.py:72 is a bare `artifact.exists()` and the hold exit-0s BEFORE `git add -A`, so the status flip will skip commit + changelog + push. All three required content parts exist and I verified them - they need collecting into that path before the flip.",
  "violated_criteria": [
    "WARN: structural guard `NO static node: imports` has 1-of-6 known-member recall (scripts/qa/verify_research_gate_workflow.mjs:256)",
    "WARN: handoff/current/live_check_36.27.md absent while masterplan verification.live_check is set -- auto-commit hook will hold commit+changelog+push at the flip"
  ],
  "violation_details": [
    {
      "violation_type": "Threshold_Not_Met",
      "action": "Recall test of the structural guard: injected 6 known-member import variants into a copy of research-gate.js and ran the shipped regex /^\\s*import\\s+\\w+\\s+from\\s+'node:/m against each",
      "state": "SEVERITY=WARN. CAUGHT 1/6: `import fs from 'node:fs'` (the one form the author measured). MISSED 5/6: `import fs from \"node:fs\"` (double quote -- IDENTICAL construct, same runtime failure is certain, not inferred), `import { readFileSync } from 'node:fs'`, `import * as fs from 'node:fs'`, `import 'node:fs'`, `import fs from 'fs'`. Independently confirmed the SHIPPED file is clean: all 5 lines containing `import` are comments (lines 52,54,55,291,294) and exactly ONE line-anchored `export` exists (line 1), so no criterion is affected today -- this is a durability gap on future edits.",
      "constraint": "scripts/qa/verify_research_gate_workflow.mjs:256 check name asserts `NO static node: imports (the Workflow runtime has no fs/path)`. qa.md 4c: name the concrete mutation that makes the guard fail -- `import { readFileSync } from 'node:fs'` breaks the subject (unlaunchable, same class the step exists to prevent) and the guard does NOT catch it. qa.md 4b: a completeness scan must find all members of a known-member set the author did not choose. Named fix: `(src.match(/^\\s*import\\b/gm) || []).length === 0`. SCOPING: for the double-quote member the runtime failure is certain; for the other four it is INFERRED from the runtime's own error text (`import call expects one or two arguments` => only the dynamic-import expression form parses) -- I could not execute the Workflow runtime from the Q/A rail."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "ls handoff/current/live_check_36.27.md ; read .claude/hooks/lib/live_check_gate.py:72 ; ls handoff/current | grep -c '^live_check_'",
      "state": "SEVERITY=WARN. File absent. Gate helper returns `\"passed\" if artifact.exists() else \"skip\"` -- existence only. Convention is well established: 43 live_check_*.md in handoff/current, 302 in handoff/archive. The three content parts the live_check names all EXIST and were verified by me: (a) envelope verbatim -- journal.jsonl + experiment_results 4.3, which I replayed and reproduced exactly; (b) the brief -- handoff/current/research_brief_86.1.md, 36998 bytes / 452 lines, 8/8 URLs re-checked; (c) a deliberate short-of-floor rejection -- checker section [2], 6 cases green in my run.",
      "constraint": "masterplan 36.27 verification.live_check = 'A real researcher spawn through the new rail: the returned envelope verbatim, plus the brief file it wrote, plus a deliberate short-of-floor case showing rejection.' CLAUDE.md: the hold exit-0s at auto-commit-and-push.sh:155/181/206, BEFORE `git add -A` at :239, so it skips the commit AND changelog AND push -- not just the push. Fix: create handoff/current/live_check_36.27.md with those three parts before the status flip."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "immutable_verification_command (exit=0)",
    "rerunnable_checker_verify_research_gate_workflow_mjs (ALL GREEN 40/0, exit=0)",
    "independent_replay_of_shipped_enforceGate_on_live_journal (reproduces experiment_results 4.3 field-for-field)",
    "independent_url_recheck_8of8_plus_char_and_byte_counts",
    "independent_writefirst_corroboration_from_stage1_transcript (3 Write calls / 38 tool_uses)",
    "independent_mutation_matrix_7_mutants_on_uncovered_guards (6/7 killed, survivor differential named)",
    "harness_non_vacuity_test_mimic_tree_6_mutants (control exit 0, all 6 RED exit 1)",
    "known_member_recall_test_of_structural_import_guard (1/6)",
    "derived_scope_git_diff_name_only (0 py / 0 frontend / 0 backend -> lint+tsc gates N/A by measurement)",
    "masterplan_diff_numstat_verification_block_untouched",
    "docs_diff_criterion5_CLAUDE_md_and_rules_research_gate_md",
    "spawn_timeline_vs_file_mtime (shipped version ran)",
    "stage2_method_audit_from_transcript (deterministic wc -m + grep -c -F)",
    "explore_agenttype_doctrine_check (precedent harness-self-audit.js:84,90; no .claude/agents/Explore.md)",
    "secret_scan_and_exec_eval_scan_on_diff",
    "live_check_gate_artifact_existence",
    "harness_log_conditional_counter (0 prior)",
    "code_review_heuristics",
    "existing_results_check"
  ],
  "harness_compliance_ok": true,
  "notes": "NO UI CLAIMS in this step (0 frontend/** files in the derived diff), so qa.md 1c does not apply and no Playwright capture was required or taken. NO backend/** files, so 1d runtime smoke is N/A. I did NOT run the backend test suite and did NOT POST to localhost:8000 (hard constraint honoured). All writes I made went to os.tmpdir() via mkdtemp from inside `node --input-type=module` heredocs; the repo tree was never modified (`git diff --quiet HEAD -- .claude/workflows/research-gate.js` = IDENTICAL to HEAD after all mutation work), and I ran no rm/mv/sed -i/git commit/git push and no shell redirects.\n\nAUTHOR DISCLOSURES ADJUDICATED (all seven scrutinised, not accepted at face value):\n(1) Immutable-command weakness -- handling is CORRECT. Not amending an immutable command and carrying criteria 3/4/6 in a separate re-runnable checker is the right call. I independently confirmed the shipped file has ZERO import statements and EXACTLY ONE line-anchored export, so the two runtime-rejection classes the author measured are absent from what shipped.\n(2) Floors-in-JS-not-schema -- SOUND, not a dodge. The author's framing (\"I do not need this claim to be true in order to be safe -- I assert in JS unconditionally; schema enforcement would be redundancy, not reliance\") is the correct direction of dependence, and I proved each floor load-bearing by mutation, so the safety property does not rest on the strip-behaviour claim at all.\n(3) gate_passed recompute -- VERIFIED it governs. Checker [6] plus my own M10 (`gate_passed := selfReported`) and M13 (`selfReported` hardcoded true) mutants, all killed. An envelope with gate_passed:true and 2 sources returns gate_passed:false with self_report_disagreed:true.\n(4) Stage-2 LLM cross-check -- the disclosure is CONSERVATIVE, i.e. honest in the safe direction. I read stage 2's actual transcript: it ran ONE deterministic shell command (`ls -la` + `wc -m` + a `grep -c -F` loop per URL) and reported its output, so in practice it was deterministic-in-execution, not eyeballed. Judged ADEQUATE rather than blocking: the runtime denies the script filesystem access so this is the strongest available in-rail check; it fails CLOSED on any null/non-object/array return (checker [5], 4 shapes, plus my mutant); and it converts self-attestation into an independent read, which is the EviBound property the step was aiming at. The residual risk the author names -- a future stage-2 agent could Read-and-eyeball instead -- is real and correctly disclosed.\n(5) Live spawn -- I re-verified the 8-of-8 claim MYSELF and it reproduces exactly, plus an extra corroboration the author did not claim: `urls_collected: 44` independently reproduces (my regex finds exactly 44 distinct URLs in the brief).\n(6) Name-not-dispatchable -- should NOT block. It is a session-scoped registry-snapshot caveat of the same class as the existing `.claude/agents/*.md` roster rule, honestly disclosed in section 6, and the docs never instruct the `{name:...}` form. But CLAUDE.md does NOT currently carry the caveat and should: one sentence next to the new two-row table.\n(7) masterplan name-only edit -- VERIFIED. `git diff --numstat 22582714~1..3d1f7a3f -- .claude/masterplan.json` = 1 insertion, 1 deletion, a single hunk at line 12444; the `verification` block (command, all 6 success_criteria, live_check) appears as unchanged context. No verification field touched anywhere in the range.\n\nA FINDING I BUILT AND THEN RETIRED (recording it so it is not re-raised): stage 2 uses `agentType: 'Explore'`, which looked like a re-split of the Explore subagent that CLAUDE.md lists under \"Never do\". It does not survive scrutiny -- there is no `.claude/agents/Explore.md` (the roster on disk is qa.md + researcher.md only), the prohibition is specifically about reintroducing it \"as separate files\", and `.claude/workflows/harness-self-audit.js:84,90` already uses `agentType:'Explore'` for exactly this tool-restriction reason (read-only: no Edit/Write/Agent). It is the precedented idiom and the right choice for a read-only verifier. Confirmed live: the meta.json records {\"agentType\":\"Explore\",\"spawnDepth\":1,\"model\":\"opus\"}.\n\nSURVIVING MUTANT, DIFFERENTIAL NAMED (qa.md shape 11): my M12 replaced the numeric coercion helper `n()` with a constant 999, defeating BOTH floor comparisons -- yet the composite gate still returned false. The killing assertion is the over-claim guard `listed.length < sources` at research-gate.js:243, which fires at 1 listed vs 999 claimed. This is genuine layered defense, not vacuity; the floor checks themselves are individually load-bearing (author mutants M1/M2, reproduced).\n\nOTHER NOTES, none degrading the verdict:\n- Stage 1's `await agent(...)` is not wrapped in try/catch, so a terminal stage-1 rejection aborts the workflow before enforceGate runs and surfaces as a workflow error rather than a structured gate_passed:false. This never yields gate_passed, so criterion 4 holds, and it matches the qa-verdict.js precedent (also unwrapped) -- optional hardening only.\n- `urls_collected` is self-reported and NOT artifact-cross-checked (only `sources_read_in_full` is). Not listed in the section-6 honest limits. It happens to reproduce on this run.\n- \"the brief measured 54 lines mid-run\" is transient and not reproducible post-hoc; I corroborated write-first by a different route instead (3 Write calls in the stage-1 transcript).\n- The checker leaves roughly 7 mkdtemp directories per run under os.tmpdir(), never cleaned. Harmless.\n- Research-gate compliance for THIS step: brief_36.27 envelope gate_passed:true, 8 read-in-full (>=5), 25 urls_collected (I derive 26 distinct URLs in the brief -- above the claim, so no over-claim), recency-scan section present, and the contract cites the brief and its four design-deciding findings by name.\n\nTO CLEAR THIS CONDITIONAL: (1) broaden the import guard to assert zero static imports and re-run the checker; (2) create handoff/current/live_check_36.27.md carrying the section-4.3 envelope, the brief path, and the checker [2] short-of-floor rejection output; (3) optionally add the one-sentence workflow-registry-snapshot caveat to CLAUDE.md. Then update experiment_results and spawn a FRESH Q/A on the changed evidence -- that is the documented cycle-2 flow, not verdict-shopping."
}
```

---

## Follow-up — Main's response to EVALUATE pass 1 (cycle-2 flow)

Both WARN findings were right and both are fixed. A fresh Q/A follows on
changed evidence.

### F1 — the `NO static node: imports` guard had 1-of-6 recall. FIXED.

**Reproduced before fixing**, against the Q/A's known-member set:

```
  default single-quote   old=CAUGHT  new=CAUGHT
  default DOUBLE-quote   old=MISSED  new=CAUGHT
  named                  old=MISSED  new=CAUGHT
  namespace              old=MISSED  new=CAUGHT
  side-effect only       old=MISSED  new=CAUGHT
  bare specifier         old=MISSED  new=CAUGHT

  OLD recall: 1/6   NEW recall: 6/6
```

The old guard was `/^\s*import\s+\w+\s+from\s+'node:/` — it caught **only the
single-quoted default form, which is the one instance I had actually hit**, and
missed the double-quoted spelling of that *identical* construct. **I built a
guard from the instance instead of from the class**, which is the same failure
mode as the census that produced a false negative in 86.3 earlier today.

Fixed to the Q/A's named fix: assert **zero static imports of any form**,
`(src.match(/^\s*import\b/gm) || []).length === 0`. Justified by the runtime's
own error text — *"import call expects one or two arguments"* — which says only
the dynamic `import()` expression parses. Proven to bite on all six members
against a mutated copy (repo untouched):

```
  KILLED  import fs from 'node:fs'
  KILLED  import fs from "node:fs"
  KILLED  import { readFileSync } from 'node:fs'
  KILLED  import * as fs from 'node:fs'
  KILLED  import 'node:fs'
  KILLED  import fs from 'fs'
```

Checker still **ALL GREEN: 40 passed, 0 failed**.

### F2 — `live_check_36.27.md` absent. FIXED.

Created, with all three parts the masterplan's `live_check` names: the verbatim
envelope from the real spawn, the brief it wrote (plus Main's independent
re-check of stage 2), and a deliberate short-of-floor case showing rejection
with the agent's `gate_passed: true` overridden.

The consequence the Q/A named is worth restating because it is easy to get
wrong: the hold `exit 0`s at `auto-commit-and-push.sh:155/181/206`, which are
**before** `git add -A` at `:239` — so a missing artifact skips the **commit and
changelog and push**, not just the push.

### Not changed

No production code. `research-gate.js` is byte-identical to the graded version —
the only code change is the checker's guard regex. No `verification` block
touched anywhere.

---

## EVALUATE pass 2 — verdict (transcribed VERBATIM)

Workflow rail run `wf_2bdb75d1-347`, agent `qa`, model `opus`, effort `max`,
30 tool uses, 147,733 tokens, 578s. Spawned on changed evidence (`e0cd5a1d`).

Both pass-1 WARNs verified fixed. One NEW WARN. All 6 criteria remain MET, and
the Q/A states plainly that **none** of the five disclosed limits should block.

### Verdict: **CONDITIONAL**  (`ok: false`)

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Pass 2 on genuinely CHANGED evidence (commit e0cd5a1d: 3 files -- new 169-line live_check, 122-line Follow-up, 20-line checker fix), so the no-verdict-shopping test passes decisively. All 6 immutable criteria remain MET and BOTH pass-1 WARN findings are genuinely fixed -- I reproduced each rather than reading the claims. F1 VERIFIED AND EXCEEDED: I extracted the guard regex FROM the checker source (not retyped), built a mimic tree (os.tmpdir, repo untouched) and drove the REAL checker end-to-end against 15 injected import variants, control first (unmutated copy -> ALL GREEN 40/0). 14/15 were CAUGHT BY THE GUARD ITSELF -- all 6 members Main tested PLUS 8 he did not: multi-line braces, default+named, default+namespace, tab-indent+semicolon, 4-space indent, leading BOM, newline-after-`import`-keyword, node:path. M15 (relative './helper.js') turned the checker RED but by ERR_MODULE_NOT_FOUND in loadModule, NOT by the guard -- honest attribution (qa.md shape 11); the regex itself does match it. FALSE-POSITIVE half, which is the half that matters here: the guard is SILENT on all 5 shipped comment lines fed verbatim, on `// import fs from 'node:fs'`, on an indented line comment, on a jsdoc `*  import call expects` line, and on mid-line prose -- ZERO false positives, and the checker's own run reports \"found 0\". Checker re-run bare: ALL GREEN 40 passed, 0 failed, exit 0; immutable command exit 0; research-gate.js byte-identical (md5 3dfa44d596a347abceb129ad754bc1a3 in HEAD and worktree, `git diff --stat 3d1f7a3f..HEAD` empty) so I skipped re-grading it as instructed. F2 VERIFIED: live_check Part 1 metadata reproduces EXACTLY from the journal (agentCount=2, totalToolCalls=40, totalTokens=191253, durationMs=686871, status=completed); Part 2 every number re-derived independently (wc -c 36998, decoded char length 36790, 8/8 claimed URLs present, kill_switch live census {'pause':44,'resume':10,'sod_snapshot':8}=62 with ZERO peak rows); Part 3 I drove the SHIPPED enforceGate myself and my output is field-for-field EQUAL to the published block (4 named violations, self-reported gate_passed:true OVERRIDDEN, self_report_disagreed:true). Bonus: my first enforceGate call used the wrong arg order and the gate fail-closed with `empty_or_errored_return` -- an unplanned live demonstration of criterion 4. Scope DERIVED from git (22582714~1..HEAD): 12 files, 1 js / 1 json / 9 md / 1 mjs, ZERO *.py, ZERO frontend/**, ZERO backend/** -- so the ruff, eslint/tsc, UI-capture and backend-smoke gates are N/A BY MEASUREMENT, not assertion; no UI claims so no Playwright capture was required or taken. No criteria erosion (check() count 28 -> 28, diff is exactly 1 removed + 1 added); secret scan 0; no eval/exec/os.system/shell=True/child_process added; working tree clean on every production path. Harness compliance clean on all 5 items (research 12:07:36 < contract 12:08:31 < code 12:16:15 < results 12:29:42 < fix 12:41:55 < live_check 12:42:45 < critique 12:43:07; masterplan still `pending`; `grep -cF 'phase=36.27' handoff/harness_log.md` = 0, so the 3rd-CONDITIONAL rule does NOT bite -- this is at most CONDITIONAL #2). ON THE FIVE DISCLOSED LIMITS, ASKED PLAINLY: NONE of them should block, and I say so having tested rather than deferred. (a) No live audit-class run -- criterion 3's own wording permits \"a test or the script itself\", and the audit-class dry mutant is killed. (b) Stage 2 is an LLM, not a deterministic read -- the runtime genuinely denies filesystem access (the measured SyntaxError is the proof), so this is the strongest in-rail check that can exist; I read enforceGate:254-262 and it FAILS CLOSED on null/string/array verification. (c) Substring URL test -- \"read in full\" is not artifact-falsifiable by ANY mechanism; the check still converts a bare self-report into an artifact-cross-checked number, which is a strict improvement. (d) Floors enforced post-return -- unavoidable; an envelope cannot be validated before it exists. (e) Name not yet dispatchable -- I compared `export const meta` against qa-verdict.js (which IS name-dispatchable) and the shape is identical (name/description/whenToUse/phases), so the session-registry-snapshot explanation is grounded rather than guessed, `{scriptPath:}` is proven working, and it is the same class as the documented `.claude/agents/*.md` roster rule, disclosed with a named next action. CONDITIONAL rests on ONE new WARN of my own, minutes to fix and not a criterion miss: live_check Part 1 is headed \"The returned envelope, verbatim\" over a block that silently drops `envelope.summary` (1193 chars) with no elision marker. I did the full leaf-path symmetric difference against the journal -- ZERO value mismatches and ZERO invented fields, so nothing gate-bearing is hidden -- but qa.md 4b is explicit that an edited capture in a block labelled verbatim is an Invalid_Precondition finding regardless of whether the underlying command passed, and the masterplan's live_check uses the word \"verbatim\" as its stated contract. The rule exists precisely because materiality cannot be assessed without re-deriving, which is what an operator reading the artifact will not do. Fix: paste the full `result` object, or mark the elision explicitly. Then update experiment_results and spawn a fresh Q/A.",
  "violated_criteria": [
    "WARN: handoff/current/live_check_36.27.md Part 1 is labelled \"The returned envelope, verbatim\" but omits `envelope.summary` (1193 chars) with no elision marker -- qa.md 4b Invalid_Precondition"
  ],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "Flattened both the journal `result` object (/Users/ford/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/263e0f5c-a712-4a84-989f-e965a0149b11/workflows/wf_9880694c-d30.json) and the first fenced json block of handoff/current/live_check_36.27.md to leaf paths, then took the SYMMETRIC DIFFERENCE rather than comparing counts",
      "state": "SEVERITY=WARN. Present in JOURNAL, MISSING from the block labelled verbatim: `envelope.summary` (1193 chars of 86.1 research prose). Present in live_check but absent from the journal: (none). VALUE MISMATCHES on shared leaf paths: (none). Run metadata reproduces exactly: agentCount=2, totalToolCalls=40, totalTokens=191253, durationMs=686871, status=completed, scriptPath=/Users/ford/.openclaw/workspace/pyfinagent/.claude/workflows/research-gate.js vs the file's \"2 agents, 40 tool uses, 191,253 tokens, 686,871 ms\". So NOTHING gate-bearing is hidden -- the defect is the undisclosed elision itself, not a masked discrepancy. Parts 2 and 3 are clean: every Part-2 number re-derived (36998 bytes, 36790 decoded chars, 8/8 URLs present, kill_switch census {'pause':44,'resume':10,'sod_snapshot':8}=62 with zero peak rows) and Part 3 reproduces field-for-field when I drive the SHIPPED enforceGate myself.",
      "constraint": "masterplan 36.27 verification.live_check = 'A real researcher spawn through the new rail: the returned envelope VERBATIM, plus the brief file it wrote, plus a deliberate short-of-floor case showing rejection.' qa.md 4b: 'A verbatim capture must be regenerated, never edited... An edited capture in a block labelled verbatim is an Invalid_Precondition finding regardless of whether the underlying command passed.' The rule is designed so a reader need not re-derive in order to trust the word verbatim. FIX (either is sufficient): paste the full `result` object including `envelope.summary`, or keep the trim and mark it, e.g. \"summary\": \"... [1193 chars of 86.1 findings, elided -- not gate-bearing]\"."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item (research<contract<code<results<fix<live_check<critique; masterplan pending; 0 harness_log rows)",
    "immutable_verification_command (bare exit=0)",
    "rerunnable_checker_verify_research_gate_workflow_mjs (ALL GREEN 40/0, bare exit=0)",
    "research_gate_js_byte_identity (md5 3dfa44d596a347abceb129ad754bc1a3 HEAD==worktree; git diff 3d1f7a3f..HEAD empty)",
    "F1_independent_recall_test_15_members_via_mimic_tree_driving_the_REAL_checker (control ALL GREEN; 14/15 caught BY THE GUARD; 8 members Main did not test)",
    "F1_false_positive_test_on_the_5_shipped_comment_lines_plus_4_comment_shapes (0 false positives)",
    "F1_evasion_attempts_export_from_and_export_star (killed by the sibling exactly-ONE-export check -- attributed, not credited to the guard)",
    "F1_residual_gap_probe_non_line_start_imports (2 survivors found, NOTE-level, reproducers recorded)",
    "F2_live_check_part1_leaf_path_symmetric_difference_vs_journal (1 dropped key, 0 value mismatches, 0 invented fields)",
    "F2_live_check_part1_run_metadata_vs_journal (agentCount/toolCalls/tokens/durationMs exact)",
    "F2_live_check_part2_independent_rederivation (wc -c 36998, 36790 chars, 8/8 URLs, kill_switch census 44/10/8=62, 0 peak rows)",
    "F2_live_check_part3_replay_against_the_SHIPPED_enforceGate (field-for-field equal)",
    "criterion_4_accidental_live_demo (wrong-arity call fail-closed with empty_or_errored_return)",
    "enforceGate_source_read_201_265 (all four floors + over-claim + fail-closed branches)",
    "derived_scope_git_diff_name_only_22582714~1..HEAD (12 files; 0 py / 0 frontend / 0 backend -> ruff+eslint+tsc+backend-smoke N/A by measurement)",
    "criteria_erosion_check (check() count 28->28; diff exactly 1 removed + 1 added)",
    "criterion_5_docs_recheck (CLAUDE.md:249 table row; rules/research-gate.md:197-208 and :199 Agent-tool fallback)",
    "meta_export_shape_vs_qa_verdict_js_precedent (grounds the name-dispatch caveat)",
    "secret_scan_and_exec_eval_shell_scan_on_step_diff (0 / 0)",
    "working_tree_production_path_cleanliness",
    "harness_log_conditional_counter (grep -cF 'phase=36.27' = 0)",
    "code_review_heuristics",
    "existing_results_check"
  ],
  "harness_compliance_ok": true,
  "notes": "NO UI CLAIMS in this step (0 frontend/** files in the DERIVED diff), so qa.md 1c does not apply and no Playwright capture was required or taken. NO backend/** files, so 1d is N/A. Hard constraints honoured: I did NOT run the backend test suite and did NOT POST to localhost:8000. Every mutation I ran wrote only into os.tmpdir() mimic trees via `node --input-type=module` heredocs; the repo was never modified (research-gate.js md5 identical to HEAD after all mutation work), and I ran no rm/mv/sed -i/git commit/git push and no output redirects.\n\nCRITERIA, EACH RE-ESTABLISHED THIS PASS: (1) MET -- exists, node --check exit 0, ENVELOPE_SCHEMA present with the structural properties intact (no stripped `minimum:`/`minItems:`, gate_passed NOT const:true, additionalProperties:false). (2) MET -- wf_9880694c-d30 returned a schema-valid envelope AND left research_brief_86.1.md (36998 bytes) on disk; I verified both against the journal and the file itself, so \"both, not either\" holds. (3) MET -- I read enforceGate:201-265 and all four floors are enforced there; criterion 3's own wording permits \"a test\" for audit-class coverage.dry and that mutant is killed. (4) MET -- proved twice: the checker's undefined/string/array cases, plus my own accidental non-object call. (5) MET -- CLAUDE.md:249 + rules/research-gate.md:197-208, with :199 keeping the Agent-tool researcher as the documented fallback. (6) MET -- all 6 floor mutants killed in my run.\n\nNOTES, none degrading the verdict:\n- RESIDUAL GAP IN THE NEW GUARD (mine, beyond pass 1). `^\\s*import\\b` requires line-start, so two valid ESM static imports SURVIVE with the checker still reporting ALL GREEN 40/0: `const __qaProbe = 1;import fs from 'node:fs'` and `/* c */ import fs from 'node:fs'`. Both would make the script unlaunchable -- the exact class the guard exists to prevent. I am grading this NOTE, not WARN, and the reasoning matters: the realistic authoring shape is line-start (it is how the original defect was actually introduced), the new guard is 14/14 on realistic forms versus the old 1/6, the shipped file today has ZERO static imports of any form, and a regex can never be complete against a grammar. If it is ever hardened, the honest fix is a parser, not a longer regex.\n- The guard is deliberately OVER-STRICT (fails SAFE) on three shapes absent from the file: a block comment with `import` at line start, a template literal with `import` at line start, and a dynamic `import()` written at line start. The shipped file uses no dynamic `import(` at all, so none of these bite today.\n- experiment_results_36.27.md was NOT updated in cycle 2 (last touched by 3d1f7a3f). It is NOT contradicted -- its \"40 checks\" and \"ALL GREEN: 40 passed, 0 failed\" claims both still reproduce, because the fix replaced one check with one check -- but the F1 evidence lives ONLY in the evaluator_critique Follow-up, so a reader of experiment_results alone would not learn the guard was broadened from 1/6 to 6/6. Worth one line in section 4.2 when the verbatim fix is made. CLAUDE.md's cycle-2 flow names experiment_results.md explicitly.\n- KILL-MECHANISM ATTRIBUTIONS (qa.md shape 11), recorded so they are not miscredited later: M15 (relative specifier) was killed by ERR_MODULE_NOT_FOUND in loadModule, not by the guard; E1/E2 (`export {x} from 'node:fs'`, `export * from 'node:fs'`) were killed by the sibling \"exactly ONE export\" check, not by the guard. That is genuine layered defense, and the guard deserves credit for exactly the 14 it killed itself.\n- Main's F1 fix adopted my predecessor's named regex verbatim AND reproduced the 1/6 baseline before fixing. That is the right order -- measure the defect, then fix, then re-measure -- and it is worth noting because the commit message names the root cause correctly (\"I built a guard from the instance instead of from the class\").\n- The pass-1 verdict was CONDITIONAL and this is CONDITIONAL again, on a DIFFERENT finding I derived myself from evidence that did not exist at pass 1. That is not sycophancy-under-rebuttal nor verdict-stacking; harness_log carries zero rows for this step-id so the 3rd-CONDITIONAL auto-FAIL is not triggered. If the verbatim block is corrected, I see nothing else standing between this step and PASS."
}
```

---

## Follow-up 2 — Main's response to EVALUATE pass 2

The finding was right and is fixed. I hand-transcribed that block and trimmed
`envelope.summary` without marking the elision. The Q/A's own leaf-path
symmetric difference showed nothing gate-bearing was hidden — but a block
labelled *verbatim* that was edited is a defect on its own terms, and the
masterplan's `live_check` uses the word *verbatim* as its contract.

**Fixed by REGENERATING the block programmatically** from the run's stored
return value via `json.dumps(..., indent=2)` — not by pasting the missing field
back in by hand. Hand-editing is what produced the defect; re-emitting removes
the class rather than the instance.

**Round-trip proven:** `EXACT MATCH against workflows/wf_9880694c-d30.json: True`.

**One correction to my own first attempt at that proof.** I initially compared
the regenerated block against `subagents/workflows/<run>/journal.jsonl` and got
`EXACT MATCH: False`, which looked alarming. It was my comparison that was
wrong: `journal.jsonl` holds the **per-agent** returns (stage 1 and stage 2
separately), while the **script's** return value lives in
`workflows/<run>.json`. Different objects, not disagreeing ones — the leaf-count
diff was 40 vs 0. The Q/A had cited the correct file in its own finding; I
reached for the wrong one. Recorded in the live_check so the next reader does
not repeat it.

Minor: `envelope.summary` measures 1,191 chars by `len()`; the verdict said
1,193. A counting-method difference, not a content one — the block is now
byte-equal to the source either way.

### Not changed

`.claude/workflows/research-gate.js` remains byte-identical to the version
graded in pass 1 (`md5 3dfa44d596a347abceb129ad754bc1a3`). No production code.
No `verification` block touched.

---

## EVALUATE pass 3 — verdict (transcribed VERBATIM)

Workflow rail run `wf_0b689d48-1b6`, agent `qa`, model `opus`, effort `max`,
27 tool uses, 156,420 tokens, 516s. Changed evidence: `3a7789d2`.

**The Q/A independently verified the counter disclosure** (2 prior CONDITIONALs
in this file, lines 13 and 139, from two distinct rail runs) and states that
CONDITIONAL was therefore NOT available to it — only PASS or FAIL.

### Verdict: **PASS**  (`ok: true`)

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "Verdict #3, and I applied the REAL CONDITIONAL count rather than the grep's. I verified the counter disclosure independently instead of taking Main's word: `grep -cF 'phase=36.27' handoff/harness_log.md` = 0 AND `grep -cE 'phase=36\\.27'` = 0, but that zero is an artifact of log-last discipline, not an absence of verdicts \u2014 handoff/current/evaluator_critique_36.27.md carries TWO verbatim CONDITIONAL blocks at lines 13 and 139, from two DISTINCT rail runs (wf_d5022922-89f, wf_2bdb75d1-347), with no intervening PASS or FAIL. So CONDITIONAL was NOT available to me this pass: the only honest outcomes were PASS or FAIL, and I reached PASS on reproduction, not on deference. THE ONE CHANGED ARTIFACT SECTION \u2014 VERIFIED AND EXCEEDED. I did not read the round-trip claim, I re-did it: extracted the first fenced block from live_check_36.27.md, loaded `result` from workflows/wf_9880694c-d30.json, re-emitted `json.dumps(indent=2)` and byte-compared -> EXACT MATCH True (identical under ensure_ascii True and False; the payload has zero non-ASCII). Leaf-path symmetric difference: 40 leaves both sides, 0 missing, 0 invented, 0 value mismatches \u2014 the pass-2 finding (`envelope.summary` silently dropped) is genuinely gone, not papered over. I then proved MY OWN check is non-vacuous by execution rather than by reasoning (qa.md 4c): control True, then 4/4 mutants caught \u2014 drop envelope.summary -> False, flip a floor 8->4 -> False, truncate one char -> False, insert one space -> False. ANTI-CIRCULARITY, WHICH MAIN DID NOT CLAIM AND WHICH IS THE REAL RISK HERE: a regenerated block is only as good as its source, so I checked the source could not have been fitted to it. workflows/wf_9880694c-d30.json mtime 12:28:11 PREDATES live_check 12:54:35 by 26 minutes; and the script return is corroborated by a SECOND independent file \u2014 `result.envelope` == the stage-1 per-agent return in journal.jsonl EXACTLY, and `result.brief_verification` == the stage-2 return EXACTLY. The block is now double-sourced with an mtime ordering, which is materially stronger evidence than the single EXACT MATCH that was asked for. THE 1191/1193 QUESTION IS SETTLED, WITH THE RULE STATED SO IT IS NOT RE-LITIGATED: len(summary)=1191, utf-8 bytes=1191, zero non-ASCII, json-escaped body=1191, and len(json.dumps(summary))=1193 \u2014 i.e. 1191 + 2 delimiter quotes. 1191 is the CHARACTER CONTENT; 1193 is the JSON STRING LITERAL INCLUDING ITS QUOTES. Same string, no content discrepancy; the journal's independent copy is also 1191 and byte-equal. Main's 1191 is the right figure when the unit is \"chars\"; the pass-2 verdict was measuring the quoted literal. MAIN'S FILE-DISTINCTION CLAIM REPRODUCES: journal.jsonl holds 4 records (2 started / 2 result) = per-agent returns of 23 and 5 leaves, while the script's return has 40 \u2014 different objects, not disagreeing ones, exactly as stated. DETERMINISTIC, ALL RE-RUN BARE BY ME: immutable command `node --check .claude/workflows/research-gate.js && ls ...` exit=0; research-gate.js md5 3dfa44d596a347abceb129ad754bc1a3 with `git diff HEAD` and `git diff 3d1f7a3f..HEAD` both empty, so byte-identity to what was graded in passes 1 and 2 is derived, not asserted; `node scripts/qa/verify_research_gate_workflow.mjs` -> ALL GREEN 40 passed, 0 failed, exit 0; checker unchanged since e0cd5a1d (`git diff` empty, md5 c219c8e35353fc097e13f448c9006d63); no criteria erosion \u2014 check() count 32 at 3d1f7a3f == 32 now (my pattern yields 32 where pass 2's yielded 28; both agree on EQUALITY, which is the property that matters). Scope DERIVED from git (22582714~1..HEAD): 12 files, 1 js / 1 json / 9 md / 1 mjs, ZERO *.py, ZERO frontend/**, ZERO backend/** \u2014 so ruff, eslint, tsc, the 1c live-UI capture and the 1d backend smoke are N/A BY MEASUREMENT, not by assertion, and no Playwright capture was required or taken. No unintended production change: worktree clean on every production path (only hook-appended audit streams and agent-memory dirs are dirty, from concurrent agents). Secret scan 0; the single exec/eval/shell grep hit is a FALSE POSITIVE \u2014 it is the pass-2 Q/A's own prose \"no eval/exec/os.system/shell=True/child_process added\" inside the transcribed verdict, not code. HARNESS COMPLIANCE CLEAN on all 5: research 12:07:36 < contract 12:08:31 < code 12:16:15, results present 12:29:42, masterplan 36.27 still status=pending with retry_count 0, zero harness_log rows, and evidence genuinely CHANGED (3a7789d2: live_check +36/-4, critique +96) so the no-verdict-shopping test passes decisively \u2014 this is the documented cycle-2 flow, not a reversal on unchanged evidence. Research gate for 36.27 itself: brief envelope gate_passed true, 8 read-in-full (>=5), urls_collected 25 against 26 distinct URLs I derive from the brief (above the claim, so no over-claim), recency scan present, contract cites the brief at line 14. CRITERIA: all 6 MET, and I strengthened criterion 2 rather than re-reading it \u2014 research_brief_86.1.md is on disk at 36998 bytes / 36790 decoded chars EXACTLY matching the envelope's char_count, 8/8 claimed sources present in the text, and 44 distinct URLs derived == the 44 claimed, so \"both, not either\" holds on measurement.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item (research<contract<code; results present; masterplan pending, retry_count 0; evidence changed 3a7789d2)",
    "true_conditional_count_derived_from_critique_file_not_the_grep (2 verbatim CONDITIONAL blocks, lines 13+139, runs wf_d5022922-89f and wf_2bdb75d1-347)",
    "harness_log_counter_both_patterns (grep -cF and grep -cE 'phase=36\\.27' = 0; zero is a log-last artifact, not an absence)",
    "immutable_verification_command_bare (exit=0)",
    "rerunnable_checker_verify_research_gate_workflow_mjs_bare (ALL GREEN 40/0, exit=0)",
    "research_gate_js_byte_identity (md5 3dfa44d596a347abceb129ad754bc1a3; git diff HEAD and 3d1f7a3f..HEAD both empty)",
    "checker_unchanged_since_pass2 (git diff e0cd5a1d..HEAD empty; md5 c219c8e35353fc097e13f448c9006d63)",
    "DELTA_independent_regeneration_and_byte_compare (json.dumps(indent=2) == fenced block, EXACT MATCH True, ensure_ascii both ways)",
    "DELTA_leaf_path_symmetric_difference (40 vs 40; 0 missing, 0 invented, 0 value mismatches)",
    "DELTA_non_vacuity_mutation_of_my_own_roundtrip_check (control True; 4/4 mutants caught: drop-leaf, flip-floor, truncate, insert-space)",
    "ANTI_CIRCULARITY_source_mtime_predates_artifact (12:28:11 < 12:54:35)",
    "ANTI_CIRCULARITY_second_independent_file (result.envelope == stage-1 journal return; result.brief_verification == stage-2 journal return, both exact)",
    "char_count_method_reconciliation_1191_vs_1193 (len=1191, utf8=1191, 0 non-ASCII, len(json.dumps)=1193 = +2 delimiter quotes)",
    "journal_vs_workflow_file_distinction_reproduced (4 records, per-agent 23+5 leaves vs script 40)",
    "criterion_2_both_not_either_rederived (36998 bytes, 36790 decoded chars == char_count, 8/8 sources present, 44 derived URLs == 44 claimed)",
    "criterion_5_docs_read_in_full (CLAUDE.md:242-266 two-row table + rationale; rules/research-gate.md:193-215 Launch section, :199 Agent-tool fallback retained)",
    "criteria_erosion_check (check() count 32 -> 32 across the fix)",
    "derived_scope_git_diff_name_only_22582714~1..HEAD (12 files; 0 py / 0 frontend / 0 backend -> ruff+eslint+tsc+UI-capture+backend-smoke N/A by measurement)",
    "unintended_production_change_check (worktree clean on all production paths)",
    "secret_scan_and_exec_eval_shell_scan_on_step_delta (0 secrets; 1 exec hit adjudicated as prose false-positive)",
    "experiment_results_4_3_reproduction_vs_source (16 of 40 leaves, 0 invented, 0 value mismatches)",
    "research_gate_compliance_for_36_27 (gate_passed true, 8>=5, 25 claimed vs 26 derived URLs, recency scan, contract cites brief)",
    "code_review_heuristics",
    "existing_results_check"
  ],
  "harness_compliance_ok": true,
  "notes": "NO UI CLAIMS (0 frontend/** in the DERIVED diff) so qa.md 1c does not apply and no Playwright capture was required or taken; NO backend/** so 1d is N/A. Hard constraints honoured: I did NOT run the backend test suite and did NOT POST to localhost:8000. I made no writes to the repo tree other than my own .claude/agent-memory/qa/ entry; no rm/mv/sed -i/git commit/git push and no shell redirects. All comparisons ran in-memory via `python - <<EOF` heredocs.\n\nON THE PRESSURE THIS VERDICT WAS UNDER, STATED PLAINLY SO IT IS AUDITABLE: Main disclosed the true count and invited a FAIL. The doctrine removed CONDITIONAL from my options, which creates an obvious hazard \u2014 reaching for PASS to avoid an unpleasant FAIL. I checked the counterfactual explicitly: at verdict #2, with CONDITIONAL available, I would still have reached PASS. The only candidate I considered raising is NOTE (b) below, and I reason it down on grounds I can defend rather than on convenience.\n\nNOTES, none degrading the verdict:\n(a) experiment_results_36.27.md was NOT updated in cycle 2 or 3 (mtime 12:29:42, still the 3d1f7a3f version). Nothing in it is contradicted \u2014 I re-derived 4.1 (exit 0), 4.2 (\"ALL GREEN: 40 passed, 0 failed\" plus the 6/6 matrix), and 4.3 (16 leaves, 0 invented, 0 value mismatches vs source). The F1 guard broadening (1/6 -> 6/6) and the cycle-3 regeneration are recorded ONLY in the evaluator_critique Follow-ups. CLAUDE.md's cycle-2 flow names evaluator_critique's \"appended Follow-up section\" as a designated home for exactly this and Main used it, so the information is present in the five-file handoff, not missing from it. Completeness NOTE, not a compliance breach; pass 2 graded it identically.\n(b) experiment_results section 4.3's json block sits under the heading \"## 4. Verification \u2014 verbatim\" while presenting 16 of 40 leaves with no elision marker \u2014 structurally the same shape pass 2 flagged in live_check Part 1, so I took it seriously rather than waving it through. I grade it NOTE, not WARN, for four measured reasons: (i) the block is visibly hand-compacted (multiple keys per line, non-json.dumps indentation) so it does not masquerade as machine output, which is the specific deception qa.md 4b targets \u2014 contrast live_check Part 1, which was json.dumps-SHAPED and therefore looked like a byte copy while being edited; (ii) I measured it: 0 invented fields, 0 value mismatches on all 16 shared paths; (iii) the artifact the masterplan's `live_check` field contractually binds to the word \"verbatim\" is live_check_36.27.md, and that one is now programmatically regenerated and byte-exact against a double-sourced original; (iv) both prior passes examined 4.3 explicitly and cleared it (pass 1: \"reproduces experiment_results 4.3 field-for-field and string-for-string\"), and 4.3 has not changed since \u2014 reversing two clearances on unchanged evidence would itself be the goalpost-move anti-pattern the skill's Dimension-5 negation list warns against. Optional one-line fix if the file is ever touched: mark the trim.\n(c) Main's \"the leaf-count diff was 40 vs 0\" describes its own RETRACTED first comparison. The 40 reproduces exactly (script result = 40 leaves) and the substantive file distinction reproduces exactly, but I cannot reproduce the specific 0 without the ad-hoc script Main ran and did not preserve \u2014 the journal's stage-2 record is a 5-leaf dict, not 0. Recording this rather than letting an unreproduced number sit unexamined; it points at a discarded error, not at evidence a reader would rely on, so it carries no assurance weight.\n(d) The residual line-start regex gap pass 2 recorded (`const x=1;import fs from 'node:fs'` survives `^\\s*import\\b`) still stands by design. I agree with pass 2's NOTE-level grading and its reasoning: a regex cannot be complete against a grammar, the shipped file has zero static imports of any form, and the honest hardening is a parser rather than a longer regex.\n(e) The name-not-dispatchable caveat is still absent from CLAUDE.md (pass 1 suggested one sentence). Non-blocking and now doubly grounded: I read CLAUDE.md:242-266 and rules/research-gate.md:193-215 in full, and neither instructs the broken `{name:...}` form \u2014 both present the script by path \u2014 while live_check section 4 discloses the gap plainly with a named next action. The next session should confirm the name resolves.\n\nONE THING I WOULD FLAG TO THE OPERATOR, OUTSIDE THIS STEP'S CRITERIA: five other agents are live in this session (main, qa-85-5-c3, research-85-4, research-85-5, researcher-36-27, researcher-86-3) and the worktree carries unrelated dirty paths (agent-memory for two roles, several handoff audit streams, four untracked handoff dirs). At the status flip, `git add -A` will sweep all of it under this step's commit message \u2014 the known cross-attribution class. Use an explicit pathspec for the 36.27 flip."
}
```
