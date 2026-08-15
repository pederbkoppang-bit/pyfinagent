# Evaluator critique -- step 86.85

## Cycle 1 -- Workflow rail `wf_5f5ce4b6-266` -- verdict **FAIL**

Launched by `scriptPath`. 55 tool calls, 199,802 tokens, 659s. HEAD `b3e0047b`.
**This spawn passed `args.verdict_sequence` as DATA** (`[]`), unlike 86.74 cycle 7
where it arrived as prose and the machinery reported `not_supplied`.

**A correct and well-earned FAIL.** 6 of 8 criteria met and independently
re-derived; C8 and C2 not met, and the mandatory lint gate RED. Main reproduced
all three blockers itself before fixing any of them.

### The finding that matters most: I built a guard named for a property it could not test

My self-test carried a check literally named `"sequence is oldest->newest"`. Its
fixture was `["CONDITIONAL","CONDITIONAL","CONDITIONAL"]` -- **a palindrome**. The
Q/A's independent mutant QA-M1 (`emit_sequence` -> `return out[::-1]`) therefore
**SURVIVED with all 11 checks green, including that one.**

Main reproduced it exactly: reversing `emit_sequence` and re-running `--self-test`
gives `SELF-TEST PASSED`, exit 0.

The materiality is not theoretical. Driven on the byte-exact shipped
`enforceEscalation`:

```
oldest->newest  [PASS,C,C] + CONDITIONAL -> n=2  would_auto_fail=TRUE
reversed        [C,C,PASS] + CONDITIONAL -> n=0  would_auto_fail=FALSE
```

An ordering regression in **the one function that feeds `args.verdict_sequence`**
silently **DISARMS the escalation**, and nothing in my suite or my 5-cell matrix
could see it -- none of M1-M5 touches ordering. The Q/A's own QA-M4/M5/M6 were
killed, so the suite is not globally weak; **this one guard was.**

This is the `feedback_a_control_built_from_your_own_pattern_tests_nothing` /
`feedback_mutation_probe_must_discriminate` class, and I walked into it while
writing a step whose entire subject is "a rule with no input cannot fire".

### Remediation applied (cycle 2)

| # | finding | fix |
|---|---|---|
| C8 | palindromic fixture; QA-M1 survived | fixture is now `[PASS, CONDITIONAL, FAIL]` on a dedicated step id, so reversal is observable. Added a **guard-on-the-guard**: `check("order fixture is NOT palindromic (anti-vacuity)", ordered != list(reversed(ordered)))`, so a future edit cannot silently re-palindrome it. Added **M6** to the matrix -- the exact QA-M1 mutant -- and it is now **KILLED**. |
| C2 | "33 of 35" run_id, unreproducible | Main re-measured: **35/35 on every predicate** (key present, non-empty, `wf_`-prefixed; non-`wf_` values `[]`). Corrected at all three sites with the population rule and command beside the number: `verdict_ledger_write.py`, `contract_86.85.md` §1 F4, and `research_brief_86.85.md` (the ORIGIN -- marked in place, since the propagation path is the lesson). |
| lint | `F401 shutil imported but unused` | import removed. `ruff --select F821,F401,F811` now **exit 0, "All checks passed!"** |
| WARN | `emit_sequence` silently dropped out-of-vocabulary tokens | now **raises** `LedgerError` (exit 4), consistent with `read_rows`'s existing loudness. A filtered sequence is SHORTER than the truth, so it can only ever UNDER-count a consecutive run -- it fails OPEN and bypasses the consumer's `unparseable` fail-closed branch. New self-test check + matrix cell **M7**, killed. |

**Matrix after remediation: 7 cells, 7 KILLED, 0 survived, 0 unscorable**, control
observed GREEN first, target sha256 identical before/after (temp-copy mutants, zero
repo writes). Self-test 13/13. Immutable command `parses`, exit 0.

**One honest note on the matrix itself.** On the first re-run M2 scored
**UNSCORABLE (anchor matched 2x)** rather than KILLED, because its anchor
`if verdict not in VALID_VERDICTS:` became a *substring* of the more-indented copy
M7 introduced -- `str.count` matches text, not lines. The matrix caught this and
refused to score it rather than mis-reporting a kill; the anchor was made unique by
including its preceding line. Recorded because a matrix that silently mis-scores is
exactly the failure this step's C8 finding is about.

### VERBATIM RETURN -- transcribed unedited from the rail

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "6 of 8 immutable criteria MET and independently re-derived (C1 localisation, C3 cross-process, C4 driven auto-FAIL, C5, C6, C7 exhaustive) — but C8 and C2 are NOT met and the mandatory lint gate is RED. C8: the NEW guard `check(\"sequence is oldest->newest\", ...)` at scripts/qa/verdict_ledger_write.py:317-318 asserts against `[\"CONDITIONAL\",\"CONDITIONAL\",\"CONDITIONAL\"]`, a PALINDROMIC fixture, so my independent mutant QA-M1 (`emit_sequence` → `return out[::-1]`) SURVIVED with all 11 checks green including that one; the differential is material, driven on the byte-exact shipped enforceEscalation: `[PASS,C,C]+CONDITIONAL → n=2 would_auto_fail=TRUE` vs reversed `[C,C,PASS]+CONDITIONAL → n=0 would_auto_fail=FALSE` — an order regression in the one function feeding args.verdict_sequence silently DISARMS the escalation and no guard can see it. C2: verdict_ledger_write.py:132 and contract_86.85.md §1 F4 state run_id \"is present on 33 of the 35 pre-existing rows\" with no population rule and no quoted command; measured on d1c4a79d~1 it is 35/35 non-empty and 35/35 wf_-prefixed — no predicate yields 33. Lint: `uvx ruff check --select F821,F401,F811` over the commit range → `F401 shutil imported but unused --> scripts/qa/mutation_matrix_86_85.py:22:8`, exit=1 (writer alone is clean). Everything else reproduced exactly: immutable cmd `parses` exit=0; before-state from git 10814 bytes/35 rows/{main:35}/86.74=0/max 2026-08-11; author matrix 5/5 KILLED with control GREEN first and sha256 146cf84e…1904 + ledger md5 1d2150f3… unchanged; settings.local.json sha256 8f03f194…d966 matches the claimed before AND after; 10 WIP files, 3 critique cycle sections at :321/:459/:573, and the duplicate 193/194/195 cycle numbers all verified. Harness compliance 5/5 clean; attempt 1 (qa_wip attempt_number=1, prior_attempts=0, source_present=true); ledger --step 86.85 = no_rows_for_step; sequence for 86.85: NONE (correctly not read as zero). No unintended production change from this step.",
  "violated_criteria": [
    "C8: mutation-test every new guard (illusory-guard, sole coverage of a behavioral property)",
    "C2: population rule + enumeration command stated beside every count of ledger rows",
    "python-lint-gate (qa.md 1a): ruff exit=1 on a file this step introduced"
  ],
  "violation_details": [
    {
      "violation_type": "Circular_Reasoning",
      "action": "QA-M1: mutate scripts/qa/verdict_ledger_write.py::emit_sequence to `return out[::-1]`, then run `python mutant_QA_M1.py --self-test`",
      "state": "exit=0, ALL 11 checks green INCLUDING the one literally named \"sequence is oldest->newest\". Root cause: the fixture at verdict_ledger_write.py:315-318 appends three rows all verdict=CONDITIONAL, so the asserted list [\"CONDITIONAL\",\"CONDITIONAL\",\"CONDITIONAL\"] is palindromic and reversal is unobservable. Materiality proven by driving the byte-exact shipped enforceEscalation (2225 bytes extracted from .claude/workflows/qa-verdict.js, asserted `fn in src`): oldest->newest [PASS,C,C]+CONDITIONAL -> n=2 would_auto_fail=true; newest->oldest [C,C,PASS]+CONDITIONAL -> n=0 would_auto_fail=false. Author matrix M1-M5 all KILLED but none touches ordering. My QA-M4/M5/M6 were KILLED, so the suite is not globally weak - this one guard is.",
      "constraint": "Immutable criterion 8: 'mutation-test every new guard with the control observed GREEN first and a byte-identical restore'; qa.md 4c: 'a guard that cannot fail when its subject is broken does not count'; skill heuristic #17 illusory-guard [BLOCK when sole coverage for a behavioral criterion] - nothing else in the suite or matrix exercises emit_sequence ordering, and ordering is the load-bearing contract of the single function that feeds args.verdict_sequence"
    },
    {
      "violation_type": "Contradiction",
      "action": "git show d1c4a79d~1:handoff/verdict_ledger.jsonl | python3 -c \"...\" counting run_id by two predicates",
      "state": "non-empty run_id = 35/35; wf_-prefixed = 35/35; non-wf values = []. Both scripts/qa/verdict_ledger_write.py:132 and handoff/current/contract_86.85.md section 1 (F4) state run_id 'is present on 33 of the 35 pre-existing rows'. No predicate I could construct yields 33, no population rule accompanies the count, and no enumeration command is quoted beside it. (The headline ledger counts DO carry the rule and command and reproduced exactly: 43 total / 8 added / 11 step_ids / 8 for 86.74 / 29 with recorded_at.)",
      "constraint": "Immutable criterion 2: 'the population rule is stated beside every count of ledger rows, and the enumeration command is quoted'; qa.md 4b: a number whose reproducing command is absent or which does not reproduce is a Contradiction finding"
    },
    {
      "violation_type": "Threshold_Not_Met",
      "action": "FILES=$(git diff --name-only d1c4a79d~1 d1c4a79d -- '*.py'); test -n \"$FILES\" || exit 1; echo \"$FILES\" | xargs uvx ruff check --select F821,F401,F811",
      "state": "FILES = {scripts/qa/mutation_matrix_86_85.py, scripts/qa/verdict_ledger_write.py} (non-empty, guard satisfied; untracked .py = 0; the only uncommitted tracked .py is the PRE-EXISTING backend/api/sovereign_api.py, mtime 2026-08-14T13:28, which lints CLEAN). Output verbatim: 'F401 [*] `shutil` imported but unused --> scripts/qa/mutation_matrix_86_85.py:22:8 ... Found 1 error.' exit=1. verdict_ledger_write.py alone: 'All checks passed!' exit=0.",
      "constraint": "qa.md section 1a: 'Non-zero exit = FAIL (quote the finding verbatim)' - REQUIRED gate because the diff touches *.py"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "Write a temp ledger with 4 rows for step 77.7 = [CONDITIONAL, \"COND\", CONDITIONAL, \"\"] and run `verdict_ledger_write.py --ledger <tmp> --emit-sequence --step 77.7`; separately feed the raw tokens to the shipped enforceEscalation",
      "state": "emit_sequence prints [\"CONDITIONAL\",\"CONDITIONAL\"], exit=0, EMPTY stderr - the two unrecognisable rows vanish silently (verdict_ledger_write.py:261). Contrast: the real consumer given those tokens raw returns sequence_status='unparseable', consecutive_conditionals=null (FAILS CLOSED). So the silent filter BYPASSES the consumer's fail-closed branch and manufactures a confident number. Internally inconsistent with read_rows(), which is deliberately LOUD (LedgerError exit 4) about a corrupt LINE for exactly the 'would under-count' reason. Reachable because the writer is NOT WIRED (disclosed in experiment_results section 4.1) so hand-appended rows remain the normal path. No out-of-vocabulary token exists in the live ledger today, so this is latent, not manifest - WARN severity, recorded so the C8 fix can cover it.",
      "constraint": "Immutable criterion 6: 'a missing row must not be readable as no prior verdict - an absent field supports not recorded, never did not happen'; and the module's own stated fail-loud doctrine (docstring: 'every failure path here is loud and non-zero')"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command (exit=0, 'parses')",
    "python_lint_gate_ruff_F821_F401_F811 (exit=1 - RED)",
    "git_scope_derivation (commit range + uncommitted + untracked)",
    "unintended_production_change_check (mtime attribution)",
    "ledger_enumeration_reproduced (43 rows, population rule stated)",
    "ledger_before_state_reproduced_from_git (d1c4a79d~1: 10814 bytes / 35 rows)",
    "writer_self_test (11/11 ok, exit 0)",
    "author_mutation_matrix_rerun (5/5 KILLED, control GREEN first, sha256 + ledger md5 unchanged)",
    "independent_mutation_cells_QA_M1..QA_M6 (1 SURVIVED with proven differential)",
    "byte_exact_extraction_and_drive_of_shipped_enforceEscalation",
    "cross_process_readback_independent_non_palindromic",
    "exhaustive_flag_combination_verdict_immutability (128 combos, 0 findings)",
    "out_of_vocabulary_laundering_probe",
    "backfill_reconstruction_audit_vs_harness_log_and_critique",
    "settings_local_json_restore_verification (sha256 match)",
    "claim_audit_every_numeric_claim",
    "harness_compliance_5_items",
    "research_gate_envelope (gate_passed true, 8 sources, 23 URLs)",
    "contract_criteria_verbatim_check (8/8)",
    "qa_wip_prior_attempt_evidence",
    "verdict_ledger_history_evidence_only",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "sequence: NONE for 86.85 - `verdict_history_86_21.py --step 86.85 --evidence-only` returns status=no_rows_for_step (correctly NOT readable as zero verdicts). qa_wip.py 86.85 --spawned-at 2026-08-15T13:45:10Z: source_present=true, attempt_number=1, attempt_number_status=ok, prior_attempts=0, records_retained=1 (gauge, includes my own WIP). Auto count of priors (0) does not exceed the ledger count (0), so no staleness signal on this step. First attempt - no verdict-shopping possible. Write-first record at /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.85__20260815T134510Z.md (COMPLETE; evidence for the next spawn, not a verdict).\n\nWHAT IS GENUINELY STRONG AND SHOULD NOT BE REDONE: the localisation is rigorous and every number reproduced from git - before-state 10814 bytes / 35 rows / recorded_by {main:35} / 86.74 rows 0 / max date 2026-08-11; the positive control (--step 86.21 -> status=ok, 5 verdicts through the same reader and key) genuinely licenses reading 86.74's zero as measured; the re-scope test was actually run and honestly answered (10 WIP files = 10 verified; 3 critique cycle sections verified at evaluator_critique_86.74.md:321/459/573; harness_log cycle numbers verified non-unique - 193/194/195 exist for phase=36.17 on 2026-08-09 AND 193/195 for phase=86.74 on 2026-08-14, with Cycle 194 genuinely absent for 86.74, corroborating \"cycle 5 was never logged\"). Every backfilled 86.74 row maps 1:1 to a named source. The real reader on the REAL ledger now returns 8 verdicts (NV->NV->C->C->PASS->C->C->C) where it returned no_rows_for_step. The settings.local.json restore is verifiable a day later: sha256 8f03f1949599866fe3875266557ff23818d1d1dc5e1cf7a4eef337e68124d966 equals both claimed values, 0 probe hooks remain, file is gitignored. C7 I verified EXHAUSTIVELY myself (128 flag combinations x 4 verdicts; 96 writes round-trip the exact verdict through a separate read process; 32 unkeyed combos refused exit=3; 0 verdict-altering findings). Honest-limits sections are accurate rather than defensive.\n\nREMEDIATION, all cheap: (1) make the ordering guard discriminating - assert emit_sequence on a NON-palindromic fixture (e.g. write PASS then CONDITIONAL then FAIL and assert [\"PASS\",\"CONDITIONAL\",\"FAIL\"]), then re-run a reversal mutant and show it goes RED; (2) fix or delete the \"33 of the 35\" claim in verdict_ledger_write.py:132 and contract_86.85.md section 1 F4 - the measured value is 35/35 - and attach the population rule + command as criterion 2 requires; (3) remove the unused `import shutil` at mutation_matrix_86_85.py:22 and re-run the gate bare; (4) optional but recommended, make emit_sequence LOUD on an out-of-vocabulary verdict (raise LedgerError, matching read_rows) so the consumer's fail-closed branch is not bypassed, and add that as a 6th matrix cell.\n\nNOTE TO MAIN, not a criterion violation: 6 production files are uncommitted in the tree (backend/api/sovereign_api.py + 5 frontend components), mtimes 2026-08-14 13:12-13:35 local, i.e. PREDATING this step's 15:37+ window; sovereign_api.py lints clean so the F401 is unambiguously this step's new file. But auto-commit-and-push.sh does `git add -A` on the masterplan status flip and WILL sweep all six into the 86.85 commit under this step's name.\n\nMETHOD DISCLOSURES: (a) no UI claims in this step, so section 1c does not apply and no Playwright capture was taken; no backend/** or frontend/** file is in the step's diff, so 1b and 1d do not apply. (b) Lint scope: qa.md 1a names `git diff --name-only HEAD` as the authority, but this step's work is already COMMITTED as d1c4a79d, so HEAD-diff returns only the unrelated pre-existing sovereign_api.py; I used the commit range as the authority for a committed step and separately confirmed untracked .py = 0, disclosing both derivations rather than picking the convenient one. (c) I appended to my own write-first record with `cat >>` heredocs; that is a redirect, which qa.md's Constraints discourage - it was confined to the single WIP path the guard permits and touched nothing else, and I used the Edit tool for the final status flip. Flagging it rather than leaving it implicit. (d) I drove the REAL enforceEscalation, never a retyped copy: the shipped file has a top-level `return` so it will not import as ESM; I brace-matched the function out of .claude/workflows/qa-verdict.js, asserted the 2225-byte slice is a byte-exact substring (`fn in src`), and imported that. All mutants ran on copies in the scratchpad - zero repo writes, and the live ledger md5 (1d2150f36d187c8f0a69bdfd3a44b62d) is identical before and after everything I ran.",
  "escalation": {
    "sequence_supplied": [],
    "sequence_status": "ok",
    "consecutive_conditionals": 0,
    "would_auto_fail": false,
    "attempt_number": null,
    "budget_exhausted": null,
    "max_attempts": 5,
    "burden_on": "the party departing from the computed escalation",
    "override": null,
    "override_reason": null,
    "judge_was_told_consequence": false
  },
  "verdict_unmodified": true
}
```
