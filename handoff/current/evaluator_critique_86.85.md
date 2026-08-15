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


---

## Cycle 2 -- Workflow rail `wf_879d28f2-9fc` -- verdict **FAIL**

55 tool calls, 230,548 tokens, 787s. HEAD `faae3efb`. `verdict_sequence` supplied as
DATA (`["FAIL"]`) sourced from the ledger this step built -- the loop feeding itself.

**FAIL again, and right again.** 6 of 8 met; C8 not met, C2 partial, plus a
scope-honesty finding. Main reproduced every blocker before fixing any of it.

### The lesson: I fixed the INSTANCE cycle 1 named, not the CLASS

Cycle 1 found the ordering guard untested; I added M6. Cycle 2 then found **two more
brand-new guards with zero coverage**:

- **QA-M6 -- the fail-loud I/O guard.** Reverted, it survives all 13 self-test
  checks and all 7 matrix cells. Probe: append into a `0o500` dir. Baseline
  `exit=4`, no file, loud stderr. Mutant `exit=0`, **row printed to stdout, nothing
  on disk, empty stderr** -- a silent writer, the exact state the module's docstring
  forbids and criterion 6 calls unfalsifiable.
- **QA-M4 -- `step_id` in the dedup key.** Dropped, the same `run_id` collides
  across steps and a legitimate row is refused and lost. Fails OPEN.

**Notably the Q/A disclosed that its own first probes for BOTH cells failed to
discriminate and were discarded rather than reported**, and that it ruled QA-M5
(lowercased verdict) an EQUIVALENT mutant and did not report it. That is the
discrimination discipline being applied to itself.

**Class-level fix.** Enumerated all **9 `raise LedgerError` sites** plus every
`_dedup_key` branch from source. The uncovered set was **larger than the two
reported** -- `build_row`'s empty-`step_id` guard and both CLI argument guards were
also untested. Self-test 13 -> 19 checks; matrix 7 -> 10 cells, **10 KILLED**.

**M9 first scored UNSCORABLE** because my new check called `append_row` unguarded,
so the mutant crashed the suite rather than failing a check -- and a crash is not
evidence a guard discriminated. Wrapped in `try/except`, it became a genuine KILL.

### C2 -- two faults, both mine

1. **Unanchored counts.** 43/11/FAIL-5/29-of-43 stated; 44/12/6/30-of-44 measured
   later. The drift is **self-referential** -- this step records its own verdicts
   into the file it counts. Every figure is now anchored to a commit.
2. **`33/35` was ANNOTATED, not REPLACED** -- correction block added at `:115`, wrong
   number left standing at `:29`, `:126`, `:182`. **The exact failure I spent this
   morning fixing in 86.74**, committed the same day in the opposite direction. All
   three sites now replaced.

### Scope honesty

`contract_86.85.md` §6.3 promised a pytest file; I shipped `--self-test` instead and
did not disclose the substitution. The file now exists (25 tests, one per matrix
cell). The two are not duplicates -- `--self-test` is the matrix target, the pytest
file is the regression suite.

### VERBATIM RETURN -- transcribed unedited from the rail

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "6 of 8 immutable criteria MET and independently re-derived (C1, C3, C4, C5, C6, C7). C8 is NOT MET and C2 is only partially met. C8: my independent 8-cell matrix (control observed GREEN first, temp-copy mutants, target sha256 identical before/after) found TWO uncovered new guards. QA-M6 reverts the fail-loud I/O guard at verdict_ledger_write.py:258-259 (`except OSError: raise LedgerError(..., EXIT_IO)`) and SURVIVES all 13 self-test checks and all 7 shipped matrix cells; the differential is material and was driven with a discriminating probe (write into a 0o500 parent dir): BASELINE exit=4, no row, loud stderr, versus MUTANT exit=0, row PRINTED to stdout, nothing on disk, EMPTY stderr — a silent writer, which is exactly the state the module's own docstring forbids (\"a silent writer would manufacture exactly the LEDGER_EMPTY / absent-row state the reader is built to refuse\") and exactly the state criterion 6 names (\"prove the recorder ran before treating a zero as evidence\"). This guard has its own exit code and its own docstring section and has NO self-test check and NO matrix cell. QA-M4 (drop step_id from _dedup_key:156-157) also survives with a demonstrated differential (same run_id + DIFFERENT step: BASELINE 2 rows exit 0/0, MUTANT exit 2 and the legitimate second row is LOST) — lower trigger probability, still an uncovered new guard. My first probes for BOTH cells failed to discriminate and were discarded rather than reported, and QA-M5 (verdict stored lowercased) was ruled an EQUIVALENT mutant and NOT reported, because emit_sequence and the real reader verdict_history_86_21.py:169 both do .strip().upper(). C2: the headline block in experiment_results §2 and live_check §8 does not reproduce today under its own stated population and quoted command — stated 43 rows / FAIL 5 / 29-of-43 / 11 step_ids, measured now 44 / FAIL 6 / 30-of-44 / 12 — reproducing exactly at the stating commit d1c4a79d, drift +1 being this step's own cycle-1 FAIL row (self-referential), with no \"as of <sha>\" anchor even though the same document anchors its C2 remediation figure to d1c4a79d~1; and research_brief_86.85.md still carries \"33/35\" at lines 29 (envelope summary), 126 and 182 as counts of ledger rows with no population rule and no command, annotated at :115 rather than replaced. Everything else reproduced exactly and should NOT be redone: immutable command `parses` exit=0; ruff F821,F401,F811 over the git-derived commit scope \"All checks passed!\" exit=0; self-test 13/13; shipped matrix 7/7 KILLED with sha256 2f0d1000…c168 identical before/after; the mutation HARNESS itself proven non-vacuous by an inert-cell probe (reports SURVIVED, exits 1); pre-state re-derived from d1c4a79d~1 (35 rows, 10 step_ids, 86.74=0, recorded_by {main:35}, max date 2026-08-11, run_id 35/35 on every predicate, 12 rows sharing one microsecond); positive control 86.21 = [C,C,FAIL,C,C]; the re-scope test genuinely performed (10 WIP files, harness_log Cycle 194 absent for 86.74, \"## Cycle 193 \" and \"## Cycle 195 \" each appearing twice); C3 reproduced across three separate python invocations; the shipped enforceEscalation brace-extracted and driven with an anti-vacuity control (1 prior -> false, 2 priors -> n=2 true, PASS/FAIL -> false, [C,C,NO_VERDICT]+C -> n=2 true, absent -> null not 0, reversal -> n=0), 86.74's real ledger priors giving n=2/would_auto_fail=true, and args.verdict_sequence confirmed as the consumed input at qa-verdict.js:514; C7 swept exhaustively (128 combos, 96 round-trips through a separate read process, 32 refusals exit=3, 0 findings). Harness compliance 5/5 clean. No unintended production change from this step; ZERO repo writes made by this evaluation (writer/matrix md5 unchanged, git status on scripts/qa and the ledger empty).",
  "violated_criteria": [
    "C8: mutation-test EVERY new guard -- the fail-loud I/O guard and the dedup key's step_id component have no coverage; QA-M6 survives with a material differential",
    "C2: population rule + enumeration command beside every count of ledger rows -- headline counts no longer reproduce and are unanchored; three brief counts still carry the retired 33/35 with no rule or command",
    "contract-plan-item-dropped (scope honesty): backend/tests/test_phase_86_85_verdict_ledger_write.py promised in contract §6.3 does not exist and the substitution is undisclosed"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "QA-M6: copy scripts/qa/verdict_ledger_write.py to tmp, replace `    except OSError as exc:\\n        raise LedgerError(f\"failed to append to {path}: {exc}\", EXIT_IO) from exc` with `    except OSError as exc:\\n        return row`, run `--self-test`, then run BOTH baseline and mutant with `--ledger <ro_dir>/ledger.jsonl` where ro_dir is chmod 0o500",
      "state": "self-test on the mutant: 13/13 ok, exit 0 -- the entire suite and all 7 shipped matrix cells are blind to it. Discriminating differential: BASELINE exit=4, row_file_created=False, stderr='verdict_ledger_write: failed to append to ...'; MUTANT exit=0, row_file_created=False, stdout='{\"step_id\": \"77.7\", \"cycle\": \"1\", \"verdict\": \"CONDITIONAL\", ...}', stderr=''. So the caller is told the verdict was recorded when no row exists, and the next spawn's zero is indistinguishable from 'no prior verdict'. A first probe (ledger path = a directory) did NOT discriminate -- both sides exit 1 with a traceback because read_rows dies on path.read_text() first -- and was discarded rather than reported.",
      "constraint": "Immutable criterion 8: 'mutation-test every new guard with the control observed GREEN first and a byte-identical restore'; qa.md §4c: 'a guard that cannot fail when its subject is broken does not count'; skill heuristic #17 illusory-guard; and the module's own docstring, 'EXIT CODES -- the writer FAILS LOUD, never silently ... A silent writer would manufacture exactly the LEDGER_EMPTY / absent-row state the reader is built to refuse, so every failure path here is loud and non-zero'"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "QA-M4: replace `    if run:\\n        return (step, f\"run:{run}\")` with `    if run:\\n        return (\"\", f\"run:{run}\")`, run --self-test, then append step 88.1 and step 99.9 with the SAME --run-id wf_same",
      "state": "self-test on the mutant: 13/13 ok, exit 0 -- SURVIVED. Differential: BASELINE step 88.1 exit=0, DIFFERENT step 99.9 same run_id exit=0, rows=2; MUTANT step 88.1 exit=0, step 99.9 exit=2 ('duplicate key (\\'\\', \\'run:wf_same\\')'), rows=1 -- a legitimate verdict row for a second step is silently refused and lost, which under-counts a consecutive run in the fail-OPEN direction. Trigger probability is low because run ids are uuids, so this is the lesser of the two, but the step_id component of the dedup key has no self-test check and no matrix cell. A first probe using cycle keys never reached the mutated branch and was discarded.",
      "constraint": "Immutable criterion 8: 'mutation-test every new guard'; qa.md §4c requires naming the concrete mutation that makes each guard fail"
    },
    {
      "violation_type": "Contradiction",
      "action": "Re-run the artifact's own quoted enumeration over its own stated population: `python3 -c \"rows=[json.loads(l) for l in open('handoff/verdict_ledger.jsonl') if l.strip()]\"`, and compare against `git show d1c4a79d:handoff/verdict_ledger.jsonl | grep -c .`",
      "state": "experiment_results_86.85.md §2 and live_check_86.85.md §8 state 'total rows 43', 'step_ids present 11', 'verdict distribution {CONDITIONAL 23, FAIL 5, PASS 8, NO_VERDICT 7}', 'rows with recorded_at 29 / 43'. Measured now: 44 rows, 12 step_ids, FAIL 6, recorded_at 30/44. The stated figures reproduce EXACTLY at d1c4a79d (43) and the drift is exactly +1, this step's own cycle-1 FAIL row, i.e. self-referential -- but neither block carries an 'as of <sha>' anchor, while the same document anchors its C2 remediation figure to d1c4a79d~1. Separately, research_brief_86.85.md lines 29 (JSON envelope `summary`), 126 (F4) and 182 (C) still state run_id 'present on 33/35 rows' with no population rule and no command; a correction note exists at :115 but the three occurrences were annotated, not replaced, and line 29 sits above the note.",
      "constraint": "Immutable criterion 2: 'the population rule is stated beside every count of ledger rows, and the enumeration command is quoted'; qa.md §4b: a number that does not reproduce under its own stated command is a Contradiction finding"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "find . -name '*86_85*'; grep -rn 'verdict_ledger_write' backend/tests/ tests/; grep -rn 'verdict_ledger_write' --exclude-dir=node_modules --exclude-dir=.git .",
      "state": "contract_86.85.md §6 plan item 3 promises 'Tests backend/tests/test_phase_86_85_verdict_ledger_write.py: dedup refusal, NO_VERDICT does not clear an escalation, absent row reads as unknown not zero, cross-process read-back, and the driven 3rd-CONDITIONAL auto-FAIL'. The file does not exist; no pytest file references the writer; and outside the two new scripts and the handoff artifacts there is no caller of verdict_ledger_write at all. Neither experiment_results §1 'What was changed' nor §4 'What I could NOT verify' discloses that the planned test module was replaced by an in-script --self-test. Consequence: the writer's only regression coverage is a self-test that nothing invokes automatically -- it is not collected by pytest and has no CI caller. Severity WARN (the functional ground is largely covered by the 13 self-test checks), recorded because a silently-dropped contract plan item is the scope-honesty class.",
      "constraint": "Harness protocol: experiment_results must disclose what was built versus what the contract planned; skill Dimension 3 test-coverage-delta and Dimension 4 scope honesty"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "immutable_verification_command (exit=0, 'parses')",
    "python_lint_gate_ruff_F821_F401_F811 over git-derived commit scope (exit=0, 'All checks passed!')",
    "git_scope_derivation (commit range d1c4a79d + 5a3b0766, uncommitted tracked, untracked)",
    "unintended_production_change_check",
    "writer_self_test (13/13 ok, exit 0)",
    "author_mutation_matrix_rerun (7/7 KILLED, control GREEN first, sha256 2f0d1000...c168 unchanged)",
    "independent_mutation_matrix_QA_M1..QA_M8 (2 SURVIVED with proven differentials, 1 ruled EQUIVALENT and not reported)",
    "mutation_harness_vacuity_probe (inert cell -> SURVIVED + exit 1; harness is not vacuous)",
    "byte_exact_brace_extraction_and_drive_of_shipped_enforceEscalation",
    "verify_escalation_86_78.mjs (51 checks, 0 failed)",
    "cross_process_readback (three separate python invocations)",
    "idempotency_replay_on_a_copy_of_the_real_ledger (exit=2, rows 44->44)",
    "C7_exhaustive_flag_combination_sweep (128 combos, 96 round-trips, 32 refusals, 0 findings)",
    "ledger_enumeration_reproduced_current_and_at_d1c4a79d_and_pre_state_d1c4a79d~1",
    "claim_audit_every_numeric_and_set_membership_claim",
    "harness_log_and_WIP_artifact_corroboration (cycle 194 absent, duplicate cycle numbers, 10 WIP files)",
    "research_gate_envelope (brief_status COMPLETE, gate_passed true, 8 sources, 23 URLs, recency scan)",
    "contract_criteria_verbatim_check (8/8)",
    "research_before_contract_ordering_from_git",
    "log_last_check (no phase=86.85 row in harness_log; masterplan status pending)",
    "no_verdict_shopping_check (evidence CHANGED between cycle 1 and cycle 2)",
    "qa_wip_prior_attempt_evidence",
    "verdict_ledger_history_evidence_only",
    "code_review_heuristics",
    "evaluator_critique"
  ],
  "harness_compliance_ok": true,
  "notes": "SEQUENCE EVIDENCE (gathered, not applied): `python scripts/qa/verdict_history_86_21.py --step 86.85 --evidence-only` -> source handoff/verdict_ledger.jsonl, status \"ok\", detail \"1 verdict(s) from the ledger\", verdicts: FAIL. `python scripts/qa/qa_wip.py 86.85 --spawned-at 2026-08-15T14:01:37Z` -> source_present true, attempt_number 2, attempt_number_status \"ok\", attempt_number_is_lower_bound false, prior_attempts 1, records_pruned_known null, records_retained 2 (gauge, includes my own record). Cross-check: attempt_number (2) exceeds the ledger verdict count (1) by exactly one, which is the current spawn, so the ledger is NOT stale for this step. Prior record: verdict_wip_86.85__20260815T134510Z.md. Write-first record for this spawn: /Users/ford/.openclaw/workspace/pyfinagent/.claude/agent-memory/qa/verdicts/verdict_wip_86.85__20260815T140137Z.md (COMPLETE; evidence for the next spawn, never a verdict).\n\nWHAT IS GENUINELY STRONG AND SHOULD NOT BE REDONE. The three cycle-1 blockers are really fixed and I verified each: the ordering fixture is now [PASS, CONDITIONAL, FAIL] with a guard-on-the-guard asserting it is not its own reverse, and M6 (the exact cycle-1 survivor) is KILLED by \"sequence is oldest->newest\"; the 33/35 figure is corrected to 35/35 in verdict_ledger_write.py and contract_86.85.md with the population rule and command attached; ruff is exit 0. The mutation procedure is exemplary and better than most in this repo -- control observed GREEN before any cell, temp-copy mutants so there is no restore to get wrong, sha256 printed and compared in-run, UNSCORABLE scoring so a broken mutant cannot score as a kill, and the M2 anchor-collision caveat recorded rather than smoothed. I confirmed the harness is not itself vacuous. The localisation is rigorous and every number reproduced from git. The honest-limits sections (§4 / §9) are accurate rather than defensive, including the two that matter most: the writer is NOT wired to the seam, and no live spawn has yet consumed the ledger for args.verdict_sequence.\n\nREMEDIATION, all cheap. (1) Add two self-test checks and two matrix cells: M8 = swallow the OSError in append_row (assert a write into an unwritable parent exits 4 and creates no file); M9 = drop step_id from _dedup_key (assert two DIFFERENT steps sharing a run_id both append). (2) Anchor the headline count blocks -- \"as of d1c4a79d\" -- in experiment_results §2 and live_check §8, or re-derive them at HEAD; a self-referential count needs the sha, not a bigger number. (3) Either add the promised backend/tests/test_phase_86_85_verdict_ledger_write.py, or state in experiment_results §1/§4 that --self-test replaced it and that nothing invokes it automatically. (4) Optional: extend the brief's :115 correction note inline at :29, :126 and :182 so the retired 33/35 cannot be read without it -- annotate, do not rewrite, since the brief is a dated gate artifact.\n\nMETHOD DISCLOSURES. (a) No UI claims in this step and no frontend/** or backend/** file in its diff, so qa.md §1b, §1c and §1d do not apply and no Playwright capture was taken. (b) Lint scope: qa.md §1a names `git diff --name-only HEAD` as the authority, but this step's work is already COMMITTED, so HEAD-diff returns only the unrelated, pre-existing backend/api/sovereign_api.py (a +\"1y\" window addition, mtime predating this step, not attributable to 86.85). I used `git show --name-only --format=\"\" d1c4a79d 5a3b0766 | grep '\\.py$' | sort -u` as the authority for a committed step, asserted the set non-empty before reading the exit code, piped through xargs (never an unquoted variable), and separately confirmed untracked .py = 0. Both derivations are disclosed rather than the convenient one picked. (c) I drove the REAL enforceEscalation, never a retyped copy: the shipped file carries top-level `return` statements and `export const meta`, so it will not load as ESM or via new Function; I brace-matched the 2225-byte function span out of .claude/workflows/qa-verdict.js and evaluated that exact slice. (d) Every mutant ran on a copy in the OS tmpdir or the scratchpad; I made ZERO writes to the repo -- verdict_ledger_write.py md5 d6b07cfba13900c94349e3bacdc6f66b and mutation_matrix_86_85.py md5 46ed4604a3123a4aebdba59f40486540 are unchanged, and `git status --short scripts/qa/ handoff/verdict_ledger.jsonl` is empty. The idempotency replay was run against a tmp COPY of the real ledger, not the real file. (e) I re-checked HEAD at the end of the evaluation: still faae3efb / 5a3b0766, no commits landed mid-eval.\n\nNOTE TO MAIN, not a criterion violation and already raised by cycle 1: six unrelated production files are uncommitted in the tree (backend/api/sovereign_api.py plus five frontend components) and auto-commit-and-push.sh does `git add -A` on the status flip, so they will be swept into the 86.85 commit under this step's name.",
  "escalation": {
    "sequence_supplied": [
      "FAIL"
    ],
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
