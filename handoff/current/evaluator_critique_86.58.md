# Evaluator critique — step 86.58

**Verdict: FAIL**  |  `ok: False`
**Run:** `wf_b127735e-55b` (Workflow qa-verdict rail, attempt 1 — `qa_wip.py 86.58` returned `records_retained: 0`)
**Date:** 2026-08-13

> Transcribed VERBATIM from the Q/A's returned structured output. Main records
> the verdict and never authors it. Nothing below is edited, paraphrased, or
> summarised — Main's response to it is in the "Main's follow-up" section at the
> end, kept strictly separate.

---

## Verdict payload (verbatim)

```json
{
  "ok": false,
  "verdict": "FAIL",
  "reason": "Criterion 3 is NOT MET: the \"flag-ON behaviour\" was never executed with the flags ON — drive_86_58_dead_downgrade.py:60 asserts both flags are False and aborts otherwise, and the blast radius was inferred from a hand-set pos.recommendation='BUY' (a proxy for the assumed post-fix value). I drove decide_trades with both flags actually ON (control cell GREEN: pos='BUY'+fresh HOLD fires signal_downgrade; discrimination control: 'Strong Buy'+HOLD is dead flag-OFF and fires flag-ON, so the probe reads flag state) and the published number inverts: a held row carrying 'new_buy_signal' still produces NO signal_downgrade with the flags ON, because flag-ON _resolve_rec returns _UNRECOGNISED_REC ('__UNRECOGNISED__'), a member of none of _BUY_RECS/_SELL_RECS/_DOWNGRADE_RECS, and _pos_rec is written ONLY by execute_buy (paper_trader.py:488,:512) while the partial-sell path at :676 preserves the stored value — flipping a flag does not rewrite existing rows. Measured blast radius at promotion time is 0 of 2 currently-held rows, not \"1 of 1 (100%)\", and \"Promoting the 06-8 flags today would sell the book's only position on an empty analysis\" (experiment_results_86.58.md:167-168) is false in both halves; exposure begins at the next execute_buy that rewrites the field (production does pass analysis_recommendation via portfolio_manager.py:578,:918 -> autonomous_loop.py:251,:1768, so the fix is not inert, just not retroactive). Separately, criterion 2's counts are stale: Main's own SQL re-run verbatim returns n=2 ('new_buy_signal' NTAP + DELL, DELL entered 2026-08-13T19:31:19Z = 8 minutes BEFORE experiment_results was written at 21:39 local) against the published \"TOTAL 1 rows\"/\"the book holds 1 position\" — the method and query are correct and the qualitative finding is strengthened (2 of 2 = 100% off-vocab, 0 in closed set), but the cardinality does not reproduce. What DOES hold: criterion 1 reproduces byte-equivalently on my own re-run (exit 0, control cell B green, cells A and C dead); the immutable command from masterplan.json exits 0 (\"parses\"); paper_round_trips.exit_reason reproduces exactly (stop_loss_trigger 16 / swap_for_higher_conviction 13 / sell_signal 3 / signal_downgrade 0, positive-controlled); criterion 6 is MET (zero backend/ diff, and the UNRECOGNISED line fired twice in my own run); no unintended production change (git show --stat 9740c64f = masterplan + contract + scripts/qa only); ruff F821/F401/F811 on the commit-derived scope passed; backend/tests/test_phase_61_2_decision_integrity.py 33 passed. Criterion 5's not-applicable reading is ACCEPTED — the criterion is conditional (\"any guard added\"), no guard was added, and I verified the write seam at paper_trader.py:452 is genuinely unguarded and reached only from execute_buy, so deferring it to the operator-gated flag decision and 86.63 is defensible rather than an evasion.",
  "violated_criteria": [
    "criterion_3_flag_on_blast_radius_not_measured",
    "criterion_2_population_counts_stale",
    "criterion_4_recommendation_rests_on_refuted_blast_radius",
    "verbatim_capture_command_unrunnable",
    "live_check_86.58_absent"
  ],
  "violation_details": [
    {
      "violation_type": "Invalid_Precondition",
      "action": "criterion 3: 'flag-ON behaviour is measured in a NON-LIVE environment' — evidenced by scripts/qa/drive_86_58_dead_downgrade.py cell B with pos.recommendation hand-set to 'BUY'",
      "state": "drive_86_58_dead_downgrade.py:60 asserts paper_position_recommendation_fix_enabled is False and paper_recommendation_vocab_fix_enabled is False and aborts otherwise; the production flag-read is executed in ZERO cells. Re-driven by me with Settings().model_copy(update={both flags True}): control F (pos='BUY', fresh='HOLD') -> [('NTAP','SELL','signal_downgrade')] GREEN; cell E (pos='new_buy_signal', fresh='HOLD') -> []; cell G (pos='swap_buy') -> []; discrimination control 'Strong Buy'+HOLD is [] flag-OFF and fires flag-ON. flag-ON _resolve_rec returns _UNRECOGNISED_REC='__UNRECOGNISED__' for 'new_buy_signal'.",
      "constraint": "criterion 3 requires the FLAG-ON behaviour to be measured, not a substituted field value; qa.md 4c (a guard/measurement that cannot observe its subject) and the assert-the-property-not-a-proxy rule"
    },
    {
      "violation_type": "Contradiction",
      "action": "experiment_results_86.58.md:162-168 states 'Blast radius: 1 of 1 currently-held positions (100%) would become a signal_downgrade SELL candidate' and 'Promoting the 06-8 flags today would sell the book's only position on an empty analysis'",
      "state": "Measured 0 of 2 at promotion time. Mechanism verified in source: _pos_rec is written only at paper_trader.py:488 and :512 (both inside execute_buy); the partial-sell path at :676 writes position.get('recommendation','') unchanged; both currently-held rows (NTAP, DELL) carry 'new_buy_signal', which flag-ON resolves to '__UNRECOGNISED__' and is in none of _BUY_RECS/_SELL_RECS/_DOWNGRADE_RECS. Exposure begins only at the next execute_buy that rewrites the field.",
      "constraint": "a quantified claim in experiment_results.md must reproduce when its named command is re-run (qa.md 4b claim auditing); FAIL is preferred when a number in a verbatim artifact does not reproduce"
    },
    {
      "violation_type": "Contradiction",
      "action": "criterion 2: re-ran the published SQL against sunny-might-477607-p8.financial_reports.paper_positions verbatim",
      "state": "Returns {'rec':'new_buy_signal','n':2,'in_closed_set':0,'tot_qty':10.15308}. Rows: NTAP qty 5.346643 entry 2026-07-31T18:47:37Z; DELL qty 4.806437 entry 2026-08-13T19:31:19Z. experiment_results_86.58.md:111-118 publishes 'TOTAL 1 rows', 'tickers=1', and :140 'The book holds 1 position'. DELL entered 8 minutes before the artifact's 21:39 local mtime. Proportion claim survives and strengthens: 2 of 2 (100%) off-vocab, 0 in closed set.",
      "constraint": "criterion 2 requires the count of CURRENTLY held rows; a population measured before the contract (disclosed protocol-order breach) must be re-derived at publication time"
    },
    {
      "violation_type": "Unjustified_Inference",
      "action": "criterion 4: recorded operator recommendation for ask 06-8 ('do not promote until 86.69 closes') justified by 'promoting the fix converts a dead SELL rule into one that fires on fabricated verdicts' on the current book",
      "state": "No flag was promoted — verified (both read False from Settings(), script asserts it). But the justification rests on the refuted 1-of-1 immediate blast radius. Corrected framing: 0 immediate candidates; the hazard is real but deferred to the next execute_buy that rewrites paper_positions.recommendation.",
      "constraint": "an operator-facing recommendation on a live-money flag must rest on a measurement that reproduces"
    },
    {
      "violation_type": "Invalid_Precondition",
      "action": "experiment_results_86.58.md:213-218 presents the immutable verification command and reports 'Result: parses (exit 0)'",
      "state": "The pasted string drops the closing double quote; run as printed it returns `bash: -c: line 0: unexpected EOF while looking for matching '\"'` and never reaches python. The command stored in .claude/masterplan.json is correct and I ran it: printed 'parses', exit=0.",
      "constraint": "a block labelled as the verbatim verification command must be the command that was run (qa.md 4b: a verbatim capture must be regenerated, never edited); severity NOTE — substance unaffected"
    },
    {
      "violation_type": "Missing_Assumption",
      "action": "check for handoff/current/live_check_86.58.md required by masterplan verification.live_check",
      "state": "File does not exist. Its required content includes 'the measured flag values read from the RUNNING process', which experiment_results_86.58.md:230-234 honestly discloses it cannot obtain (all three flags absent from GET /api/settings/; 45 keys, positive control 15 paper_* keys ARE exposed). The driven test asserts the defaults path from a fresh Settings(), not the running process.",
      "constraint": "verification.live_check gate — the artifact must exist with the three named contents before the status flip; severity NOTE at EVALUATE time, blocking at closure"
    }
  ],
  "certified_fallback": false,
  "checks_run": [
    "qa_md_read_in_full_at_runtime",
    "write_first_wip_record",
    "attempt_count_via_qa_wip_py",
    "harness_compliance_5_item_audit",
    "research_gate_envelope_verification",
    "criteria_verbatim_match_masterplan_vs_contract",
    "immutable_verification_command",
    "driven_proof_reproduction",
    "flag_on_driven_measurement_with_green_control",
    "discrimination_control_flag_off_vs_flag_on",
    "bigquery_population_rederivation",
    "bigquery_round_trip_exit_reason_rederivation",
    "bigquery_ntap_reeval_rederivation",
    "git_status_and_commit_scope_audit",
    "ruff_lint_F821_F401_F811_derived_scope",
    "pytest_phase_61_2_decision_integrity",
    "source_trace_of_pos_rec_write_seam",
    "live_check_artifact_existence",
    "code_review_heuristics",
    "claim_auditing_4b",
    "guard_vacuity_4c"
  ],
  "harness_compliance_ok": true,
  "notes": "ATTEMPT COUNT (qa.md 3rd-CONDITIONAL rule, stated as required): `python scripts/qa/qa_wip.py 86.58` returns records_retained=1 — that single record is MY OWN write-first file created this run — with prior_records=[]. Derived prior attempts = 0, so this is ATTEMPT 1 and the prior-verdict sequence is EMPTY. Secondary cross-check: `grep -F \"86.58\" handoff/harness_log.md` finds only the original filing mention at line 34336 and no `result=` row; ledger and log agree, so no disagreement to adjudicate. CONDITIONAL was therefore available to me and I did not take it — this is a criterion miss, not a fixable-presentation gap: the criterion-3 measurement never touched the code path it names, and when I ran that path the answer inverted.\n\nHARNESS COMPLIANCE — CLEAN on all five items. Research gate: brief present (31,129 chars), envelope brief_status COMPLETE, gate_passed true, external_sources_read_in_full=6 (floor 5), urls_collected=38 (floor 10), recency_scan_performed true with the section at :255. Criteria verbatim: all 6 masterplan strings are byte-present in contract_86.58.md (programmatic membership test, 6/6 True) — the \"copied programmatically\" claim reproduces. Log-last: masterplan 86.58 is status=pending with no harness_log result row, so the log/flip has correctly not happened. No verdict-shopping: attempt 1. Note retry_count/max_retries are both null on this step, so certified_fallback is not triggered.\n\nON DISCLOSURE 1 (protocol order). The disclosure is adequate AS A DISCLOSURE — it is specific, it names the time window, and it correctly warns that a file-mtime check would pass (I confirmed: brief 21:22 < contract 21:24 < script 21:25 < results 21:39, all \"clean\"). I am not capping the verdict for the disclosure. But the breach was NOT harmless, and that should be recorded: because criteria 1-2 were executed ~25-40 minutes before publication, the live book moved underneath them — DELL was bought at 19:31Z, eight minutes before experiment_results.md was written — and the artifact shipped a population count that was already wrong when it was saved. That is Finding 2, and it is a direct consequence of doing the measurement before the contract and not re-deriving at write time.\n\nON DISCLOSURE 2 (criterion 5 not-applicable). I tested the reading and I ACCEPT it. The criterion is conditional on its own antecedent (\"any guard added is mutation-tested\"); no guard was added, so there is no subject and nothing to mutate. I independently verified the substantive half of Main's reasoning rather than taking it: `_pos_rec = reason` at paper_trader.py:452 is genuinely unparsed, the only two writes of paper_positions.recommendation are at :488 and :512 (both inside execute_buy), and the partial-sell path at :676 passes the stored value through unchanged — so a guard at that seam would indeed change SELL behaviour on the live book and does belong with the operator-gated flag decision and 86.63's single boundary guard, not as a sixth site patch. Criterion 5 is vacuously satisfied, not evaded. This is the one place where I am NOT the adversary Main expected.\n\nON DISCLOSURE 3 (separation of duties). I read the qa.md that this same Main session edited today under phase-86.75, and I checked whether any edited text is load-bearing for this verdict. It is not: (a) the attempt counter reads 1 under BOTH the new ledger source and the old harness_log grep, so the source swap changes nothing here; (b) the deleted \"a prior FAIL/CONDITIONAL is ground truth\" clause has no subject — there is no prior verdict on 86.58; (c) the deleted weighted quant rubric (DSR/Sharpe/PBO) is irrelevant to a measurement-and-disclosure step with no strategy math. My independence is not compromised on this specific step, and my FAIL rests entirely on commands I ran myself. The operator review already requested in handoff/harness_log.md for that qa.md edit should still stand on its own merits — I am not clearing it.\n\nWHAT MAIN SHOULD DO NEXT (cycle-2, on CHANGED evidence). Small and mechanical; no production code needs to change. (1) Re-run the driven proof with a flag-ON arm — construct Settings().model_copy(update={\"paper_position_recommendation_fix_enabled\": True, \"paper_recommendation_vocab_fix_enabled\": True}), keep a green control (pos='BUY'+fresh HOLD fires) and add the discrimination control ('Strong Buy'+HOLD: dead OFF, fires ON) so the harness is proven to see the flag state. (2) Replace the blast-radius number with the measured one: 0 of 2 currently-held rows at promotion time, with the mechanism stated (flag-ON _resolve_rec -> '__UNRECOGNISED__'; _pos_rec written only by execute_buy at :488/:512; :676 preserves) and the forward-looking exposure stated separately (begins at the next execute_buy). (3) Re-derive the paper_positions population AT WRITE TIME and state the timestamp next to it — the qualitative finding gets stronger, 2 of 2 = 100% off-vocab, and DELL entering today at 19:31Z with 'new_buy_signal' is fresh live corroboration that the unguarded write seam is still producing off-vocabulary values. (4) Rewrite the criterion-4 operator recommendation so its justification matches: the direction (\"do not promote until 86.69 closes\") survives, but the reason is that promotion arms a SELL on every FUTURE buy while 81.2% of analyses are empty HOLD placeholders — not that it would sell the current book today. (5) Fix the unrunnable pasted command (restore the closing double quote; the masterplan's stored version is correct) and produce live_check_86.58.md, disclosing plainly that the flag values cannot be read from the running process rather than leaving that requirement silently unmet."
}
```

---

## Main's follow-up (cycle 2) — NOT part of the verdict

The Q/A was right on every blocker. Each is fixed below; evidence changed, so a
FRESH Q/A is warranted on the updated files (canonical cycle-2 flow), not a
re-grade of the same evidence.

| Violated criterion | What I did |
|---|---|
| `criterion_3_flag_on_blast_radius_not_measured` | Rewrote `scripts/qa/drive_86_58_dead_downgrade.py` to ENTER the condition via `Settings().model_copy(update={both flags: True})`. Added a **discrimination control** (`'Strong Buy'`: dead OFF, fires ON) proving the probe reads flag state. Measured **0 of 2**, reproducing the Q/A's finding. |
| `criterion_2_population_counts_stale` | Re-derived at publication time (2026-08-13T20:24:53Z): **2 of 2** off-vocabulary, 0 in the closed set. |
| `criterion_4_recommendation_rests_on_refuted_blast_radius` | Withdrew the justification. Corrected: promotion has **zero immediate effect**; the hazard is **deferred** to the next `execute_buy`. The 86.69-first sequencing stands on the narrower reason. |
| `verbatim_capture_command_unrunnable` | Restored the dropped closing quote and **executed the block as printed**: exit 0, `parses`. |
| `live_check_86.58_absent` | Created `handoff/current/live_check_86.58.md`. |

**On the one thing I could not fix:** the live_check requires flag values read from
the RUNNING process. They are **not obtainable** — `/api/settings/` returns a curated
`FullSettings` (45 keys, 0 hits for either flag), and `…/all`, `…/flags`, `…/debug`
and `/api/paper-trading/config` all 404. Positive control: 15 `paper_*` keys **are**
exposed, so the probe is live and the absence is real. This is recorded in the
live_check as **UNVERIFIED**, not papered over — the driven test asserts the
**defaults path**, not the running process.

**Root cause of my error, recorded so it is not repeated:** my harness contained
`assert flag is False` and aborted otherwise. A test that asserts its subject cannot
vary can never measure that subject — the assertion *was* the proxy, written by me,
in plain sight. Third instance of this class; auto-memory
`feedback_assert_the_property_not_a_proxy` now carries the greppable signature
(`assert.*is False`, a hard-coded stand-in "representing" the real value).

**Second miss:** I published "the book holds 1 position" eight minutes after DELL
opened — a trade I had recorded myself, that same hour, in another file. Knowing a
fact is not re-deriving the count that depends on it.
