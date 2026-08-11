# live_check -- step 86.5

Required by `verification.live_check`: the full accounting table, the measured
signatures, the kill-switch non-touch pair, and the filed-step list.

**Written after the cycle-1 Q/A CONDITIONAL**, whose blocking finding was that
this file did not exist and no line-by-line accounting existed anywhere. All
tables below are GENERATED from the audit_basis and the captured run, not typed.

---

## A. FILE-LEVEL DISPOSITION -- all 26 baseline failures accounted for

**A literal 26-node-id table is NOT derivable**: the 2026-08-08 baseline was
recorded at FILE granularity (`GROUPING BY FILE` in the step's `audit_basis`),
so the individual node ids of the 26 were never written down. This table
accounts for all 26 at the granularity they exist at; section B gives node-level
signatures for today's 17.

| # | file | baseline | now | disposition | filed step |
|---|---|---|---|---|---|
| 1 | `test_64_3_currency_path.py` | 3 | 0 | **already fixed** -- absent from today's run | -- |
| 2 | `test_64_4_multi_market_e2e.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 3 | `test_book_safety_69.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 4 | `test_dod4_tier1_coverage_investment.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 5 | `test_phase_23_2_15_verify_23_1_smoke.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 6 | `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 7 | `test_phase_23_2_6_sector_cap_emit.py` | 1 | 1 | unchanged | 86.50 |
| 8 | `test_phase_40_2_claude_code_v2_1_140_features.py` | 1 | 1 | unchanged | 86.50 |
| 9 | `test_phase_57_1_reject_binding.py` | 3 | 3 | unchanged | 86.48 |
| 10 | `test_phase_60_3_data_integrity.py` | 1 | 1 | unchanged | 86.48 |
| 11 | `test_phase_70_3_atomic_swap.py` | 1 | 0 | **already fixed** -- absent from today's run | -- |
| 12 | `test_phase_70_4_gate_observability.py` | 2 | 0 | **already fixed** -- absent from today's run | -- |
| 13 | `test_phase_75_17_verification_paths.py` | 2 | 3 | GREW +1 | 86.50 |
| 14 | `test_phase_75_prompt_contracts.py` | 1 | 1 | unchanged | 86.50 |
| 15 | `test_phase_75_sre_ops.py` | 1 | 2 | GREW +1 | 86.49/86.50 |
| 16 | `test_phase_82_39_outcome_rebuild_query.py` | 1 | 1 | unchanged | 86.50 |
| 17 | `test_portfolio_swap.py` | 1 | 1 | unchanged | 86.51 |
| 18 | `test_price_tolerance_gate.py` | 3 | 0 | **already fixed** -- absent from today's run | -- |
| 19 | `test_phase_75_19_preflight_calibration.py` | 0 | 1 | **NEW since baseline** | 86.50 |
| 20 | `test_phase_82_48_outcome_write_schema.py` | 0 | 2 | **NEW since baseline** | 86.52 |
| | **TOTAL** | **26** | **17** | | |

`26 - 14 + 2 + 3 = 17`  -- every baseline failure has a disposition; none unclassified.

## B. NODE-LEVEL MEASURED SIGNATURES -- today's 17

| # | node id | measured signature | filed step |
|---|---|---|---|
| 1 | `test_60_3_flag_defaults_off` | `AssertionError: assert True is False` | 86.48 |
| 2 | `test_c1_runbook_and_operator_token_drafted` | `FileNotFoundError: [Errno 2] No such file or directory: '/Users/ford/.openclaw` | 86.49/86.50 |
| 3 | `test_c6_no_launchctl_bootstrap_executed_in_ops_scripts` | `AssertionError: reissue_cc_oauth_token.sh: "RELOAD_HINT_2='launchctl bootstrap` | 86.49/86.50 |
| 4 | `test_live_masterplan_is_currently_clean` | `AssertionError: unexpected genuine residue: {` | 86.50 |
| 5 | `test_masterplan_diff_touches_only_the_ten_sibling_insert` | `AssertionError: non-comma-artifact removal found: '  "updated_at": "2026-07-23` | 86.50 |
| 6 | `test_off_identity_prompts_are_verbatim_constants` | `AssertionError: assert 'You are an independent Risk Judge for a paper trading ` | 86.48 |
| 7 | `test_operator_decision_note_exists_with_token` | `FileNotFoundError: [Errno 2] No such file or directory: 'handoff/current/opera` | 86.50 |
| 8 | `test_phase_23_2_6_backend_log_has_skipping_buy_evidence` | `AssertionError: no 'Skipping BUY' line in backend.log OR its newest archive (r` | 86.50 |
| 9 | `test_phase_40_2_settings_json_still_valid_json_after_edi` | `AssertionError: phase-29.2 effortLevel=xhigh invariant must survive phase-40.2` | 86.50 |
| 10 | `test_reject_binding_main_path_off_emits_on_blocks` | `AssertionError: assert True is False` | 86.48 |
| 11 | `test_reject_binding_swap_path_off_emits_on_blocks` | `AssertionError: flag-OFF must preserve the (vulnerable) swap BUY; swap_buys={'` | 86.48 |
| 12 | `test_swap_framework_fills_zero_buy_gap` | `AssertionError: Expected 2 swap SELLs, got 1; orders=[TradeOrder(ticker='TECH0` | 86.51 |
| 13 | `test_sweep_over_live_masterplan_is_clean` | `AssertionError: unexpected genuine defects remain: {'86.31': [{'path': '.claud` | 86.50 |
| 14 | `test_sweep_shape_census_matches_the_corrected_figures` | `AssertionError: assert {'dict': 1057...4, 'str': 126} == {'dict': 720,...4, 's` | 86.50 |
| 15 | `test_the_fetch_supplies_every_field_the_write_REQUIRES` | `AssertionError: with no recommendation source the outcome must be skipped` | 86.52 |
| 16 | `test_the_sweeps_recall_limit_is_recorded_not_assumed` | `AssertionError: no OPEN step OWNS the live phantom-column defect the sweep can` | 86.50 |
| 17 | `test_write_really_persists_into_bigquery` | `AssertionError: assert 'UNKNOWN' == 'BUY'` | 86.52 |

17 rows.

## C. CRITERION 4 -- REDONE, and my original claim was reached by luck

The cycle-1 Q/A found I had **hand-narrowed the scope and asserted a proxy**.
Corrected: the six files are DERIVED from 86.5's own `audit_basis`, and the
test is the COUPLING PROPERTY (does the test reach the operator's live
kill-switch state?) rather than a grep count.

```
file                                            refs  property
test_64_3_currency_path                            0  no live reach
test_64_4_multi_market_e2e                         0  no live reach
test_dod4_tier1_coverage_investment               68  tmp-isolated
test_phase_70_3_atomic_swap                        0  no live reach
test_price_tolerance_gate                          0  no live reach
test_phase_70_4_gate_observability                 0  no live reach
```

**ZERO of the six are LIVE-COUPLED** -- five have no live reach at all, and
`dod4` monkeypatches `kill_switch._AUDIT_PATH` to `tmp_path`, so it is
tmp-isolated. That is a **stronger** result than my original "ONE, not six".

**MY ORIGINAL CLAIM WAS RIGHT BY LUCK.** I measured four files, only THREE of
which are among the six, and certified "the one" using
`test_phase_23_2_4` -- which **is not one of the six**. The member I never
measured, `dod4`, carries **68 references, more than the 43** I used to certify
coupling. Under my own stated proxy it would have read as COUPLED and my
conclusion would have been wrong. The proxy was wrong in a way that happened
not to matter.

Ref-count note: I measure 68 with a case-insensitive pattern; the Q/A measured
63. Different regex, same conclusion -- recorded rather than reconciled away.

`test_phase_23_2_4_pause_resume_no_deadlock_live` (43 refs) IS genuinely
live-coupled but is **not among the six**, and was already fixed by 86.3
(`4f10b024`). So no duplicate step is owed for the 36.28 class.

## D. CRITERION 5 -- kill-switch audit non-touch, before/after pair

```
BEFORE  2026-08-11T11:45:34Z
  lines  66
  sha256 ab7324ebf501e3d3886e62a5d8fd2ed4f01f675849702b6553a4df691aab455f
  bytes  6618
AFTER   2026-08-11T11:52:34Z
  lines  66
  sha256 ab7324ebf501e3d3886e62a5d8fd2ed4f01f675849702b6553a4df691aab455f
  bytes  6618
```

UNCHANGED across a full suite run. 86.3's egress guard holds. The cycle-1 Q/A
reproduced this independently with its own bracket and got the same sha256.

## E. CRITERION 6 -- no test edited

```
$ git status --porcelain backend/tests/
(empty)
```
