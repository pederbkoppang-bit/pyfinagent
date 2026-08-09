# Evaluator critique -- phase-86.21

## Cycle 1 verdict: **CONDITIONAL** (`wf_cb85c901-472`)

Workflow rail, `model: opus`, 16 tool calls, 124,152 tokens, 325s. Main RECORDS
this verdict; Main did not author it. Transcribed VERBATIM.

**Criteria 1-5 MET, all re-derived independently.** The Q/A rebuilt the git
reproduction itself at both commits, reproduced the zsh `:h` trap under zsh 5.9
to confirm my disclosure, verified 36.17's six-verdict history from
`harness_log.md` -- a source I did not supply it -- and checked every numeric
claim (1189 headers, 574 without `phase=`, 18 result tokens, the 3/2/0 depth
parser, 18 filename shapes). All reproduced exactly.

**It ruled FOR me on the six-vs-five question** and said it would have flagged
the opposite choice: trimming the closing PASS row to match the criterion's
wording would have been a fabricated measurement.

**It also agreed the immutable command is weak by construction** and put that on
the record: it counts lines in an unrelated file and would exit 0 against no
implementation at all. It graded the step on the six criteria instead.

**THREE FIXES NAMED, and the first is a real behavioural gap:**

1. **`ledger_empty` is missing.** A present-but-ZERO-BYTE ledger falls through
   to `no_rows_for_step`, returning consecutive=0 at exit 0 with the detail
   "it has genuinely not been graded yet" -- a confident and FALSE claim.
   Criterion 6 names "corrupt **or empty**" by word, and empty was untested.
2. **The arming threshold is one-sided.** `c >= 2` -> `c >= 1` SURVIVED: the
   self-test asserts `would_auto_fail` is True at c=2 but nothing asserts it is
   False at c=1 or c=0. Q1 and Q2 mutate the same line in opposite directions
   and only one dies -- the signature of a one-sided guard.
3. **A blank/non-string `verdict` field** is silently skipped instead of counted
   bad -- the malformed-FIELD analogue of the malformed-LINE case that IS
   guarded.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Cycle 1, 86.21. Harness compliance CLEAN (research dc621419 -> contract -> code 7897cb8c; gate_passed=true, 7 sources >= floor 5, 26 URLs >= floor 10, recency scan present; harness_log has 0 rows for 86.21 and masterplan status=pending, so LOG-is-last holds; no prior critique, so no verdict-shopping and the 3rd-CONDITIONAL rule is not in play). Immutable command exit=0 (1189 + three filenames). Commit ships exactly 5 in-scope files, zero unintended production change; ruff F821/F401/F811 over the git-DERIVED 2-file set (non-empty guard satisfied) = All checks passed, exit=0. C1 MET: I re-derived the reproduction myself at both commits -- 688ac349 (86.20 pending, 1 recorded verdict, log-grep=0) and 7145f566 (86.20 pending, 2 recorded verdicts, log-grep=0); the N>1 case is exact. C2 MET: ledger is a separate file, harness_log untouched, and both rejection measurements reproduce (depth-2 critique parser returns 3/2/0 for 86.20/86.17/36.17; 18 distinct evaluator_critique filename shapes). C3 MET. C4 MET. C5 MET with one gap. C6 PARTIALLY MET -- this is the only thing holding PASS. Criterion 6 says \"corrupt OR EMPTY the source\"; \"empty\" is untested, and when I test it a ZERO-BYTE ledger returns status=no_rows_for_step, consecutive=0, exit=0, with detail \"it has genuinely not been graded yet\" -- a confident and FALSE claim, i.e. a silent zero on one of the two failure modes the criterion names by word. Separately, I built 6 mutants the author did not include and 3 SURVIVED, the load-bearing one being would_auto_fail `c >= 2` -> `c >= 1`: the arming threshold is pinned only from ABOVE (case (i) asserts True at c=2) and nothing asserts False at c=1 or c=0, so a mutant that arms the auto-FAIL after ONE CONDITIONAL goes undetected. Both are small, named fixes on a step whose shipped logic is otherwise correct -- CONDITIONAL, not FAIL.",
  "violated_criteria": [
    "criterion_6_mutation_test: 'empty the source' untested and behaviourally unnoticed (zero-byte ledger reports a silent 0 at exit 0 with a false detail string)",
    "criterion_6_mutation_test: illusory-guard (WARN) -- would_auto_fail arming threshold pinned only from above; 3 of 6 Q/A-authored mutants survived",
    "criterion_5_fail_direction: no ledger_empty status distinguishes a truncated ledger from a genuinely ungraded step"
  ],
  "violation_details": [
    {
      "violation_type": "Missing_Assumption",
      "action": "read_ledger('A', <zero-byte handoff/verdict_ledger.jsonl>) via the real entry point, then _report()",
      "state": "status=no_rows_for_step, consecutive=0, would_auto_fail=False, CLI exit=0, detail='no rows for step A -- it has genuinely not been graded yet'. No ledger_empty status and no file-size check exist; verdict_history_86_21.py:109-150 branches only on path.exists(), bad>0, and seen_step. Self-test cases (i)-(v) cover corrupt (iii) and missing (iv) but never a 0-byte file.",
      "constraint": "Criterion 6 verbatim: 'MUTATION-TEST the counter: corrupt or EMPTY the source and assert it NOTICES rather than silently reporting zero -- silently reporting zero is the defect being fixed.' A truncated ledger is indistinguishable from a never-graded step, and the detail string asserts a fact that is false. Fix: a ledger_empty status (stat size 0 with the file present) returning None or a caution + non-zero exit, plus a self-test case."
    },
    {
      "violation_type": "Circular_Reasoning",
      "action": "Q/A-authored mutant Q1: verdict_history_86_21.py:106 'return c >= 2' -> 'return c >= 1', loaded in-memory (repo never written, target md5 96f91ffb50a0a5a3fb68ab6df69c105a unchanged), then self_test()",
      "state": "SURVIVED, rc=0. Control rc=0. The mutant makes would_auto_fail return True at consecutive=1, i.e. the counter reports the auto-FAIL ARMED after ONE CONDITIONAL -- an observable, wrong output on the exact semantic the counter exists to compute. Not an equivalent mutant. Root cause: self-test (i) asserts would_auto_fail is True at c=2 and nothing anywhere asserts it is False at c=1 or c=0, so the threshold is one-sided. Also SURVIVED: Q6 (blank/non-string 'verdict' field silently skipped instead of counted bad -- the malformed-FIELD analogue of the malformed-LINE case that IS guarded) and Q4 (exact step match -> prefix match; shipped code is correctly '!=', but 86.2/86.20/86.21 and 36.1/36.17 genuinely collide under a prefix rule) and Q3 (dropping .upper() normalisation). KILLED: Q2 ('>= 3') and Q5 (NO_ROWS_FOR_STEP collapsing into OK), confirming the matrix has real reach.",
      "constraint": "qa.md 4c: 'a guard that cannot fail when its subject is broken does not count' -- name the concrete mutation that makes each guard fail. WARN-level per the 4c verdict wiring (a one-sided guard ALONGSIDE genuine behavioural guards, not sole coverage). Fix: assert would_auto_fail is False at c=1 and c=0, and add a blank-verdict-field row to the corrupt case."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (gate_passed=true, 7 sources, 26 URLs, recency scan present, cited in contract as wf_f916b683-d59)",
    "contract_before_generate_mtime_and_commit_order (dc621419 research+contract BEFORE 7897cb8c code+results)",
    "log_last (grep -F 'phase=86.21' handoff/harness_log.md = 0; masterplan status=pending)",
    "no_verdict_shopping (no evaluator_critique_86.21.* exists; cycle 1)",
    "immutable_verification_command (exit=0)",
    "criterion_1_reproduction_re_derived_independently_at_688ac349_and_7145f566",
    "zsh_colon_h_quoting_trap_confirmed_and_no_leftover_in_executed_code",
    "ruff_F821_F401_F811_on_git_derived_scope (2 files, non-empty guard, exit=0)",
    "commit_scope_diff (5 files, no unintended production change, harness_log untouched)",
    "counter_run_against_36.17",
    "counter_self_test (5 cases)",
    "author_mutation_matrix_rerun (3/3 KILLED, control green, md5 unchanged)",
    "qa_authored_mutation_matrix (6 new mutants: 3 SURVIVED, 2 KILLED, 1 anchor OK)",
    "four_status_reachability_and_CLI_exit_codes_via_real_entry_point",
    "zero_byte_ledger_probe",
    "ledger_cross_checked_against_independent_source (harness_log cycles 190-195 + evaluator_critique_36.17.md run_id)",
    "numeric_claim_reproduction (1189/574/48.3%/18 result= tokens/3-2-0 depth parser/18 filename shapes)",
    "code_review_heuristics",
    "guard_vacuity_check_4c",
    "claim_auditing_4b"
  ],
  "harness_compliance_ok": true,
  "notes": "ATTACKS ANSWERED, in order.\n\n(a) SIX vs FIVE verdicts -- HONEST HANDLING, criterion 3 MET, ruling FOR the author. I verified 36.17's real history from a source the author did not supply me: handoff/harness_log.md lines 32056/32075/32081/32090/32098/32135 = Cycle 190 CONDITIONAL, 191 FAIL, 192 FAIL, 193 CONDITIONAL, 194 CONDITIONAL, 195 PASS. The ledger matches row-for-row INCLUDING cycle numbers, and the cycle-190 run_id wf_6bc4c0a4-d9c matches evaluator_critique_36.17.md:4. Six is the true history; the criterion's \"five\" was frozen mid-flight. Trimming the PASS row to match the criterion's wording would have been a fabricated measurement -- exactly the class this project keeps paying for. Criterion 3's literal ask is still discharged: self-test case (i) loads the exact five-verdict sequence and returns consecutive=2, armed=True, with the FAIL pair resetting the opening CONDITIONAL. Both states are shown. I would have flagged the opposite choice.\n\n(b) THE ZSH :h TRAP -- CONFIRMED REAL, NO LEFTOVER. Under zsh 5.9 I reproduced it: `git show $C:handoff/harness_log.md` with C=688ac349 resolves to '.andoff/harness_log.md' (\":h\" on a slash-less value yields \".\"), git errors, and grep -c faithfully prints 0. The quoted form is correct at both commits. `grep -rn 'git show \\$' handoff/current/*86.21* scripts/qa/*86_21*` returns exactly ONE hit -- experiment_results_86.21.md:50, inside the prose describing the bug. No executed command carries the unquoted form. Disclosing a self-inflicted near-fabrication rather than quietly fixing it is the right call and I verified it independently rather than taking it on trust.\n\n(c) FAIL DIRECTION IS GENUINELY TESTED, NOT MERELY ASSERTED -- with one gap. I exercised all four statuses through the real read_ledger()/_report() entry point on real files in a tempdir, not through the author's harness: ok -> count + exit 0; no_rows_for_step -> 0, exit 0 (fail-open, and correct -- genuinely ungraded); unparseable -> consecutive=None, armed=None, exit 1 (\"refusing to print 0\", \"treat as ARMED\") -- confirmed it returns None and NOT 0; ledger_missing -> exit 1 with the bootstrap caution. All four reachable. The gap is the FIFTH state nobody enumerated: a zero-byte file present on disk, which falls through to no_rows_for_step at exit 0. See violation_details.\n\n(d) I WROTE MY OWN MUTANTS -- 3 OF 6 SURVIVED. The author's matrix reproduces exactly (control rc=0, 3/3 KILLED, target md5 unchanged) and is a REAL guard, not a pass-for-the-wrong-reason: M1 directly restores the silent zero and dies. Its claim is also correctly bounded (\"every guard IN THIS MATRIX can fail\") -- no global overclaim, which is the right discipline. But the matrix mutates only the three shapes the author was already thinking about. Mine, all in-memory: Q1 arming threshold >=2 -> >=1 SURVIVED (load-bearing); Q6 blank verdict field skipped instead of counted bad SURVIVED; Q4 exact -> prefix step match SURVIVED; Q3 drop .upper() SURVIVED; Q2 >=2 -> >=3 KILLED; Q5 NO_ROWS_FOR_STEP -> OK KILLED. Note Q1 and Q2 mutate the SAME line in opposite directions and only one dies -- that asymmetry is the signature of a one-sided guard.\n\n(e) THE INDEPENDENCE ANSWER IS THE RIGHT CALL, NOT AN EVASION. \"A Main-supplied count is ADVISORY, not authoritative; git-committed append-only buys AUDITABILITY rather than INDEPENDENCE\" is the strictly weaker and therefore honest claim. I checked the constraint is real, not rhetorical: the Q/A on the Workflow rail has no Write tool and no filesystem access, so Main or a hook is the only possible writer -- the audited party writes the auditor's input, and no phrasing changes that. Claiming independence would have been the overclaim; the fix (a hook writer) is named in section 8 as NOT DONE. Correct.\n\nON THE IMMUTABLE COMMAND -- I AGREE IT IS WEAK, AND SAY SO ON THE RECORD. `grep -c \"^## Cycle\" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3` counts lines in an unrelated file and lists three filenames. It has no causal connection to any of the six criteria and would exit 0 against an empty implementation, or against no implementation at all. It cannot go red. I ran it (exit=0) for the record and graded the step on the six criteria and on my own re-derivation. This is the mirror image of the phase-81.0 lesson: there the immutable command was already red for 128 unrelated reasons and the step became uncloseable; here it is unconditionally green and the step becomes ungradeable by it. Both failures are the same root cause -- a verification command not bound to what the change touches.\n\nCLAIM AUDITING (qa.md 4b) -- EVERY NUMBER I SAMPLED REPRODUCES EXACTLY. 1189 \"## Cycle\" headers -> measured 1189. \"574 of 1189 carry no phase=\" -> measured 574 (48.3%, claim said ~48%). \"12+ distinct result= tokens\" -> measured 18, and PASS_WITH_FINDINGS and PASS_AFTER_RETRY are both really present, which is what makes the M2 mutant's point true rather than rhetorical. \"a depth-2 parser returns 3 for 86.20 and 2 for 86.17 but 0 for 36.17\" -> measured exactly 3, 2, 0, with 36.17 carrying five depth-1 headers. \"17+ filename shapes\" -> measured 18. This is unusually good claim hygiene for this repo and I want it recorded as such.\n\nCODE-REVIEW HEURISTICS: no security or trading-domain findings. The diff is two new read-only analysis scripts under scripts/qa/ plus one new JSONL data file -- no kill-switch, stop-loss, perf-metrics, execution or signal path is touched, no secrets, no backend/** or frontend/**, so 1b/1c/1d do not trigger. mutation_matrix_86_21.py's importlib/exec_module use is a test harness operating on the repo's own file in a tempdir, not a production path with external input -- not command-injection per the negation list. The one heuristic that fires is #17 illusory-guard at WARN (one-sided arming-threshold guard coexisting with genuine behavioural guards), which is what caps this at CONDITIONAL rather than BLOCK.\n\nWHAT I DID NOT REACH: I did not verify that the 11 seeded run_ids resolve to real Workflow run records under .claude/../workflows/ -- I corroborated the 36.17 rows (6 of 11) against harness_log and the critique file instead, which is the stronger check; the 86.20 and 86.17 rows (5 of 11) I did not independently corroborate. I did not run backend/tests (no backend code touched). I took no Playwright capture (no UI claims in this step).\n\nTO CLEAR TO PASS -- three small, named fixes: (1) add a ledger_empty status for a present-but-zero-byte ledger returning None or a caution with non-zero exit, and a self-test case for it; (2) assert would_auto_fail is False at consecutive=1 and 0, closing the one-sided threshold; (3) add a row with a blank/missing \"verdict\" field to the corrupt case so the malformed-FIELD path is pinned alongside the malformed-LINE path. None of these changes the shipped counter's logic except (1), which is a genuine behavioural gap against a criterion that names \"empty\" by word."
}
```
