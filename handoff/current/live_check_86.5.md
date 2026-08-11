# live_check -- step 86.5

Required by `verification.live_check`: the full accounting table, the measured
signatures, the kill-switch non-touch pair, and the filed-step list.

**Written after the cycle-1 Q/A CONDITIONAL**, whose blocking finding was that
this file did not exist and no line-by-line accounting existed anywhere. All
tables below are GENERATED from the audit_basis and the captured run, not typed.

---

## A. FILE-LEVEL DISPOSITION -- all 26 accounted for (CORRECTED after the cycle-2 FAIL)

**A literal 26-node-id table is NOT derivable**: the 2026-08-08 baseline was recorded
at FILE granularity (`GROUPING BY FILE` in the step's `audit_basis`), so those node
ids were never written down. The cycle-2 Q/A confirmed this argument SOUND.

| # | file | baseline | now | disposition | owner |
|---|---|---|---|---|---|
| 1 | `test_64_3_currency_path.py` | 3 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 2 | `test_64_4_multi_market_e2e.py` | 1 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 3 | `test_book_safety_69.py` | 1 | 0 | already fixed -- absent, and NOT pause-coupled | -- |
| 4 | `test_dod4_tier1_coverage_investment.py` | 1 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 5 | `test_phase_23_2_15_verify_23_1_smoke.py` | 1 | 0 | already fixed -- absent, and NOT pause-coupled | -- |
| 6 | `test_phase_23_2_4_pause_resume_no_deadlock_live.py` | 1 | 0 | already fixed -- absent, and NOT pause-coupled | -- |
| 7 | `test_phase_23_2_6_sector_cap_emit.py` | 1 | 1 | unchanged | 86.50 |
| 8 | `test_phase_40_2_claude_code_v2_1_140_features.py` | 1 | 1 | unchanged | 86.50 |
| 9 | `test_phase_57_1_reject_binding.py` | 3 | 3 | unchanged | 86.48 |
| 10 | `test_phase_60_3_data_integrity.py` | 1 | 1 | unchanged | 86.48 |
| 11 | `test_phase_70_3_atomic_swap.py` | 1 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 12 | `test_phase_70_4_gate_observability.py` | 2 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 13 | `test_phase_75_17_verification_paths.py` | 2 | 3 | GREW +1 | 86.50 |
| 14 | `test_phase_75_prompt_contracts.py` | 1 | 1 | unchanged | 86.50 |
| 15 | `test_phase_75_sre_ops.py` | 1 | 2 | GREW +1 | 86.49/86.50 |
| 16 | `test_phase_82_39_outcome_rebuild_query.py` | 1 | 1 | unchanged | 86.50 |
| 17 | `test_portfolio_swap.py` | 1 | 1 | unchanged | 86.51 |
| 18 | `test_price_tolerance_gate.py` | 3 | 0 | **ENVIRONMENT ARTIFACT** -- green only because the book is UNPAUSED; RED under paused | **36.28** (still pending) |
| 19 | `test_phase_75_19_preflight_calibration.py` | 0 | 1 | **NEW since baseline** | 86.50 |
| 20 | `test_phase_82_48_outcome_write_schema.py` | 0 | 2 | **NEW since baseline** | 86.52 |
| | **TOTAL** | **26** | **17** | | |

**11 of the 26 are ENVIRONMENT ARTIFACTS owned by 36.28** -- corrected from
'already fixed'. The remaining 15 are dispositioned above.

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

## C. CRITERION 4 -- ANSWER REVERSED. ALL SIX ARE COUPLED (11 of the 26)

**The cycle-2 Q/A FAILED this step because my answer here was INVERTED, and it was
right to.** Two prior revisions of this section said "ONE, not six" and then "ZERO
of the six". **The measured answer is ALL SIX.**

### The measurement I should have made: flip the state

Control = plain per-file pytest, no patching. Mutant = the kill_switch singleton
forced `paused`, with the real audit COPIED so baselines replay identically:

```
control (today, book UNPAUSED):  all six GREEN
mutant  (paused=True):           ALL SIX RED
  test_64_3_currency_path              3
  test_64_4_multi_market_e2e           1
  test_dod4_tier1_coverage_investment  1
  test_phase_70_3_atomic_swap          1
  test_price_tolerance_gate            3
  test_phase_70_4_gate_observability   2
                                    = 11, matching the 2026-08-08 baseline EXACTLY
```

### Why every grep-shaped probe was blind to it

`backend/services/paper_trader.py:202` (and `:1273`):

```python
state = self._injected_ks_state or get_state()
```

It falls back to the module singleton, which replays the on-disk audit. **Any test
constructing `PaperTrader` without injecting `kill_switch_state` is coupled while
containing ZERO textual "kill_switch" references.** Verified by me -- all five
construct it uninjected:

```
test_64_3_currency_path:59             trader = pt.PaperTrader(s, bq)              injected=0
test_64_4_multi_market_e2e:144         trader = pt.PaperTrader(s, bq)              injected=0
test_phase_70_3_atomic_swap:207        trader = pt.PaperTrader(s, bq)              injected=0
test_price_tolerance_gate:63           return PaperTrader(settings=..., bq_client=bq)  injected=0
test_phase_70_4_gate_observability:68  trader = pt.PaperTrader(get_settings(), bq)  injected=0
test_dod4_tier1_coverage_investment:40 trader = PaperTrader(settings=s, bq_client=bq) injected=0
```

**SIX sites, not five** -- corrected after the cycle-3 Q/A found this artifact
UNDERSTATING its own case. Cycle 1 had called `dod4` "tmp-isolated" because the file
monkeypatches `kill_switch._AUDIT_PATH`; verified here, **those monkeypatches belong
to different tests** (`test_kill_switch_pause_*` / `_resume_*`, each constructing its
own `KillSwitchState()`), while `dod4`'s PaperTrader helper `_make_trader` at `:40`
is uninjected and unpatched. So `dod4` couples through the same `:202` fallback as
the other five.

### The failure was mine twice, and the second time was worse

Cycle 1 rejected my ref-count proxy. My remediation **read the same refs column,
relabelled it "the coupling PROPERTY", and re-derived the same wrong answer with
more confidence.** Renaming a proxy does not make it a property.

**And my very first hypothesis was correct.** The pre-gate census raised "H1: the
36.28 kill-switch-coupled cluster", and I refuted it with the bad instrument and
moved on. I had the right answer and argued myself out of it.

The rule that would have saved all three passes: **measure the thing that CHANGES
if the hypothesis is true.** Flip the state and see what breaks. A grep cannot do
that; a mutation can.

### What follows

**No duplicate step is owed for these 11 -- but NOT because they are uncoupled.**
They are owned by **36.28, which is still `status: pending`**. Nothing fixed the
coupling; the book is simply unpaused today. They return the moment it pauses.

`test_phase_23_2_4_pause_resume_no_deadlock_live` is separately coupled, is **not**
among the six, and was fixed by 86.3 (`4f10b024`).

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
