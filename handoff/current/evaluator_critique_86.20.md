# Evaluator critique -- phase-86.20

## Cycle 1 verdict: **CONDITIONAL** (`wf_080ec07a-af2`)

Launched via the Workflow structured-output rail (`.claude/workflows/qa-verdict.js`,
`model: opus`, 22 tool calls, 161,549 tokens, 551s). The verdict IS the captured
return value. Main RECORDS this verdict; Main did not author it. Transcribed
VERBATIM below -- no edits, no paraphrase.

**The blocker is a real defect I introduced, not an evidence complaint.**
`_resolve_rec` applies `.strip()` and re-bases the `or default` fallback on the
STRIPPED value BEFORE reading the flag, so the normalisation leaks past the dark
flag onto the money path. The Q/A proved it unguarded by constructing a
legacy-parity mutant that left all 56 tests green -- the suite had no legacy
oracle, so the 12-cell matrix structurally could not catch it.

```json
{
  "ok": false,
  "verdict": "CONDITIONAL",
  "reason": "Cycle 1. Harness compliance clean and every reproducible claim reproduced exactly: immutable command `-k \"portfolio_manager or decide_trades\"` = 70 passed / 3106 deselected, exit 0 (14 with the new module ignored, so the rename weakened nothing); ruff F821/F401/F811 over a git-DERIVED 6-file scope = All checks passed, exit 0; wider regression 294 passed 1 skipped exit 0; mutation matrix re-run by me = 12/12 KILLED with both digests unchanged and baseline+restored green; the section-4 population table reproduced BYTE-FOR-BYTE against my own independent BigQuery query. Both step-text corrections are TRUE (independently re-derived): 'Buy' does match after .upper() and reaches 8.80 genuine > Strong Buy's 8.36, so the step's \"higher than any row that did match\" is false as written; and 'Strong Sell' has ZERO rows, so the fail-dangerous half is LATENT. Criteria 1-6 are MET with genuine, mutation-killed behavioural guards. BLOCKER TO PASS: the load-bearing \"OFF = byte-identical legacy resolution\" claim is FALSE, and no guard covers it. `_resolve_rec` applies `.strip()` and re-bases the `or default` fallback on the STRIPPED value BEFORE reading the flag, so with the flag OFF 18 input shapes resolve differently from the pre-86.20 expression and THREE change an actual ORDER: (1) `' BUY '`/`'BUY \\n'`/`'\\tBUY'` produce a real `new_buy_signal` BUY order where legacy produced none -- a new BUY outside the flag; (2) `'   '` resolves to `'HOLD'`, which is in _DOWNGRADE_RECS, producing a `signal_downgrade` SELL of a held position where legacy produced none; (3) `0`/`False`/`[]`/`{}` resolve to `'0'`/`'FALSE'`/... instead of legacy `'HOLD'`, SUPPRESSING a signal_downgrade SELL legacy would have made. I proved it unguarded: a mutant restoring true legacy parity in the OFF branch only (all logging preserved, control output shows it genuinely differs on three shapes, patch confirmed applied) left ALL 56 tests GREEN -- the suite contains no legacy oracle, so the 12-cell matrix structurally cannot catch this. That makes criterion 7 (\"MUTATION-TEST every new guard\") incompletely met on new money-path code, and the false claim is not confined to the handoff: it ships verbatim in operator-facing production text at backend/config/settings.py (flag description) and portfolio_manager._resolve_rec's docstring, which is what an operator reads when deciding to arm. Mitigating and why this is CONDITIONAL not FAIL: I measured LENGTH<>LENGTH(TRIM) = 0 across every row of analysis_results, so no padded value exists in the corpus today -- the defect is LATENT, the same class the author correctly discloses for the sell half -- and the fix is small and contained. Attack points (b), (c), (d), (e) all held up under independent testing and are NOT findings.",
  "violated_criteria": [
    "criterion_7_mutation_test_every_new_guard__OFF_branch_resolution_has_zero_coverage",
    "dark_flag_byte_identity_claim_false__unconditional_normalisation_leaks_past_the_flag_onto_the_money_path"
  ],
  "violation_details": [
    {
      "violation_type": "Contradiction",
      "action": "decide_trades(candidate_analyses=[candidate('AAA', ' BUY ')], settings=paper_recommendation_vocab_fix_enabled=False)",
      "state": "ACTUAL = [('AAA','BUY','new_buy_signal')]. Legacy expression `(' BUY ' or 'HOLD').upper()` = ' BUY ' which is NOT in _BUY_RECS, so pre-86.20 code produced NO order. Same for 'BUY \\n' and '\\tBUY'. Cause: _resolve_rec (backend/services/portfolio_manager.py) computes `raw_str = str(raw).strip()` and `source = raw_str or default` BEFORE `flag_on` is read, so the strip and the re-based default apply unconditionally. 18 differing shapes measured across the two default sites.",
      "constraint": "[WARN] backend/config/settings.py::paper_recommendation_vocab_fix_enabled description and portfolio_manager._resolve_rec docstring both state 'OFF = byte-identical legacy resolution' / 'OFF returns exactly what the legacy expression returned, so the path is byte-identical'; experiment_results_86.20.md section 1 repeats it. A DARK flag must not arm an order the pre-change code did not place."
    },
    {
      "violation_type": "Contradiction",
      "action": "decide_trades(current_positions=[held BBB with recommendation='BUY'], holding_analyses=[analysis rec='   '], flag OFF)",
      "state": "ACTUAL = [('BBB','SELL','signal_downgrade')] -- a held position is SOLD with the fix DARK. Legacy token for '   ' is '   ' (whitespace is truthy), which is in none of the three sets, so pre-86.20 code produced NO sell. Converse leak measured too: analysis rec of 0 / False / [] yields ACTUAL = [] where legacy token 'HOLD' IS in _DOWNGRADE_RECS and WOULD have sold. So the OFF path both adds and removes sells.",
      "constraint": "[WARN] Same byte-identity guarantee as above. This is the risk-bearing direction: an unconditional normalisation that reaches _DOWNGRADE_RECS liquidates a position while the operator believes the change is dark."
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "pytest.main on the 56-test module with pm._resolve_rec replaced by a legacy-parity mutant (OFF branch returns `(raw or default).upper()`; ON branch and ALL logging untouched; control printed shipped='BUY' vs mutant=' BUY ', shipped='HOLD' vs mutant='   ', shipped='0' vs mutant='HOLD'; `pm._resolve_rec is mutant` = True)",
      "state": "56 passed, pytest rc=0 -- MUTANT SURVIVED. No assertion anywhere in the suite compares the flag-OFF output against the pre-86.20 expression, so the byte-identity property has no oracle. The author's 12-cell matrix cannot detect this class: M6 ('always canonical') and M7 ('always legacy') only vary the ARMED side.",
      "constraint": "[WARN] Immutable criterion 7: 'MUTATION-TEST every new guard, including reverting the normalisation, and confirm each mutant is killed by the assertion that names it.' The OFF-branch resolution is new code on the money path with zero mutation coverage. qa.md section 4c: a guard that cannot fail when its subject is broken does not count."
    },
    {
      "violation_type": "Unjustified_Inference",
      "action": "Reading experiment_results_86.20.md section 3 and live_check_86.20.md section 1a",
      "state": "Both state: 'The reproduce tests still pass with the fix ON DISK, because the flag is OFF -- which is the evidence that OFF is byte-identical legacy behaviour rather than a claim about it.' Two passing test cases ('Strong Buy', 'Strong Sell') cannot establish byte-identity over the input space, and the property is in fact false (findings 1-2). The test docstring at line 329-330 makes the same claim ('The OFF path must stay byte-identical').",
      "constraint": "[WARN] qa.md section 4b -- every scope/completeness claim must be re-derivable by the command that produces it; a universal claim supported by two instances is an overgeneralisation, not evidence."
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "harness_compliance_audit_5_item",
    "research_gate_envelope (gate_passed=true, 8 read in full, 42 URLs, recency scan performed, contract cites brief)",
    "mtime_ordering (research 21:15:20 < contract 21:18:19 < code 21:21:10-21:23:49 < experiment_results 21:29:41 < live_check 21:30:26)",
    "log_last (0 phase=86.20 entries in harness_log; masterplan status=pending; no prior evaluator_critique_86.20.*)",
    "immutable_verification_command (70 passed, 3106 deselected, exit 0)",
    "pre_existing_selector_baseline (14 passed with new module ignored -- rename weakened nothing)",
    "wider_regression (294 passed, 1 skipped, exit 0)",
    "ruff_F821_F401_F811_on_git_derived_scope (6 files, All checks passed, exit 0)",
    "runtime_smoke (import portfolio_manager + recommendation_vocab OK; /api/health=200)",
    "unintended_change_check (commit ad9a7d02 = exactly 8 in-scope files; working tree only hook-generated audit JSONL + agent-memory)",
    "independent_BQ_re_derivation_of_the_population (table reproduces byte-for-byte; LENGTH<>LENGTH(TRIM) = 0 rows)",
    "step_text_correction_1_re_derived (Buy matches, maxgen 8.80 > 8.36 -- correction TRUE)",
    "step_text_correction_2_re_derived (zero Strong-Sell rows -- LATENT, correction TRUE)",
    "author_mutation_matrix_re_run (12/12 KILLED, md5 unchanged, baseline+restored green)",
    "independent_mutant_legacy_parity_OFF_branch (SURVIVED, with control + patch-applied assertions)",
    "flag_OFF_token_level_differential_probe (18 differing shapes across both default sites)",
    "flag_OFF_decision_level_differential_probe (3 shapes change an actual TradeOrder)",
    "source_verification_of_the_UNKNOWN_sentinel_rationale (_DOWNGRADE_RECS contains HOLD; signal_downgrade branch reproduced)",
    "code_review_heuristics",
    "claim_auditing_sec_4b",
    "guard_vacuity_check_sec_4c",
    "adversarial_worst_of_N_lenses"
  ],
  "harness_compliance_ok": true,
  "notes": "NAMED FIX (small, contained): in backend/services/portfolio_manager.py::_resolve_rec, compute the legacy token from the RAW value -- `legacy = (raw or default).upper()` guarded for non-str -- and use the stripped/normalised value ONLY for canonicalisation and for the log lines; i.e. move the `.strip()` and the re-based `or default` inside `if flag_on`. Then add a legacy-parity oracle test that asserts, for the flag OFF, `_resolve_rec(v) == (v or default).upper()` over a value table including ' BUY ', 'BUY \\n', '\\tBUY', '   ', '', None, 0, False, [] and {} at BOTH default sites. That test kills my surviving mutant and would have caught this. Also correct the four places that assert byte-identity: settings.py flag description, _resolve_rec docstring, experiment_results sections 1 and 3, live_check section 1a, and the test docstring at line 329-330.\n\nATTACK POINTS THAT HELD UP (checked independently, NOT findings):\n(b) The UNKNOWN-to-sentinel rationale is CORRECT at source. `_DOWNGRADE_RECS = {\"HOLD\",\"SELL\",\"STRONG_SELL\"}` and the branch is `if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS -> SELL signal_downgrade`; my own probe reproduced exactly that SELL for rec='HOLD' on a BUY-row position. Resolving UNKNOWN to HOLD really would liquidate on a parse failure, and M8's kill is genuine and correctly attributed to test_armed_an_unrecognised_holding_is_not_sold_either.\n(c) 'Strong Buy!' staying UNKNOWN is right, not a rationalised gap: narrow-over-permissive is correct on a money path, and the value is LOUD (UNRECOGNISED warning), so the drift is caught rather than guessed. NOTE only.\n(d) The 12 cells re-ran identically for me. They do cover the criterion-named guards; the gap is not inside the matrix's scope but adjacent to it (see finding 3). The author's self-disclosed gap (reproduce-test control assertions not separately covered) is real but minor; mine is a second, undisclosed one on the money path.\n(e) The rename weakened nothing: I measured 14 collected with the new module ignored versus 70 with it, and commit ad9a7d02 contains no deletion or rename of any other test module.\nSCOPE: I agree the seven other consumers belong in 86.22, not here -- every immutable criterion names decide_trades / the trade gate specifically. Correctly deferred with a measured enumeration.\n\nQUALITY WORTH RECORDING: fixtures use the PRODUCER spelling rather than the gate dialect (the reason the existing suite was blind); both reproduce tests carry an explicit control assertion so neither can pass vacuously; the matrix mutates in-memory with digest assertions so the repo is never written; isolation is enumerated per channel with a byte-comparison guard on the live risk_overrides trail; and section 10 discloses five things that were NOT verified, including the NOT-IN-FORCE status of pid 6644. That disclosure discipline is why this is CONDITIONAL rather than FAIL.\n\nNOTE (not a violation): experiment_results section 3 says the reproduce output was \"Run against the un-fixed tree\", but the test file's mtime (21:23:49) postdates both production edits (21:21:10, 21:22:17), so the chronology is not corroborable from artifacts. M6's kill of BOTH reproduce tests is strong substitute evidence that they genuinely detect the defect state, so I treat this as a disclosure NOTE only.\n\nWORST-OF-N LENSES (P1 money path): correctness = CONDITIONAL (a real order differential exists outside the flag); does-it-reproduce = PASS (every number and command output reproduced exactly); scope-honesty = CONDITIONAL (disclosure is unusually good, but the byte-identity claim is an overclaim repeated in shipped operator-facing text). min = CONDITIONAL.\n\nThis is cycle 1 for 86.20 (0 entries matching phase=86.20 in harness_log, no prior evaluator_critique_86.20.*), so the 3rd-CONDITIONAL auto-FAIL rule is not in play. I did not restart the backend and did not trigger a paper-trading cycle; my probes ran in-process with risk_overrides._AUDIT_PATH redirected to the scratchpad, and I verified the live handoff/risk_overrides_audit.jsonl md5 was unchanged afterwards. No UI claims in this step, so gate 1c does not apply."
}
```
