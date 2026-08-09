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
