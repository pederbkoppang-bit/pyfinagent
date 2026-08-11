# 86.5 -- the failing-test census, MEASURED

**Not a contract.** The research gate has not returned an envelope yet, so PLAN has
not started. This is the measurement the contract will rest on, recorded now so it
is not re-derived.

## The count in the step's title does not reproduce

```
$ source .venv/bin/activate && python -m pytest backend/tests/ -q -p no:randomly
17 failed, 3417 passed, 12 skipped, 5 xfailed, 1 xpassed, 48 warnings in 441.51s (0:07:21)
```

**The step says 26. The measurement is 17**, across **11 distinct files**. A peer
measured 17 twice on different trees; this is a third independent run and agrees.
Tree: `cff92516`-era working copy, 2026-08-11 ~13:1x CEST.

By exception type: **15 `AssertionError`, 2 `FileNotFoundError`**.

## The distinction the step exists to make, with two worked examples

The failure list is nearly useless for triage: almost everything is
`AssertionError` on live state. Two failures that look identical in that list need
**opposite** remedies.

### A. OBSOLETE EXPECTATION -- the test is wrong

`test_phase_40_2_claude_code_v2_1_140_features.py::test_phase_40_2_settings_json_still_valid_json_after_edit`

```
E  AssertionError: phase-29.2 effortLevel=xhigh invariant must survive phase-40.2 edit
E  assert 'max' == 'xhigh'
```

`.claude/settings.json` really does carry `effortLevel = 'max'`. It was **raised
xhigh -> max on 2026-08-04 by direct operator instruction**, and CLAUDE.md:59
records it. The production config changed deliberately; the test was not updated.

**The remedy is to update the TEST.** "Fixing" this by reverting `settings.json` to
`xhigh` would silently undo an operator decision -- the precise hazard in
*"the risk of 'fixing' a test that was correctly failing"*, inverted.

### B. A REAL DEFECT THE TEST CAUGHT -- and it is mine, from today

`test_phase_75_17_verification_paths.py::test_sweep_over_live_masterplan_is_clean`

```
E  AssertionError: unexpected genuine defects remain:
E  {'86.31': [{'path': '.claude/hooks/lib/qa_write_guard.py', 'class': 'never-existed', ...}]}
```

Verified: **`.claude/hooks/lib/qa_write_guard.py` does not exist.** The real guard
is `.claude/hooks/qa-write-guard.sh`. Step **86.31 -- which I closed today** --
references a path that never existed.

**The test is not fragile. It is working**, and it caught a bad reference in a step
that closed hours earlier.

> **I nearly mis-filed this.** My first reading grouped it with "tests that take
> live project state as their fixture are brittle." That would have been wrong, and
> worse than wrong: it would have taught the codebase to ignore a detector that was
> doing its job. A and B are indistinguishable from the failure list alone --
> both `AssertionError`, both about live state -- and they invert each other. **The
> triage must therefore rest on per-failure evidence, never on the summary line.**

## Preliminary grouping -- to be confirmed in the contract

| # | provisional class | tests |
|---|---|---|
| 1 | obsolete expectation (production changed deliberately) | `test_phase_40_2_settings_json_still_valid_json_after_edit` |
| 2 | live-masterplan sweeps -- genuine finds, need per-case adjudication | `75_17` x3, `75_19` x1 |
| 3 | missing operator artifacts (archived when their step closed) | `75_prompt_contracts`, `75_sre_ops::c1` -- both `FileNotFoundError` |
| 4 | dark-flag defaults (flags OFF by design, awaiting activation) | `57_1` x3, `60_3` x1 |
| 5 | ops-script drift | `75_sre_ops::c6` (a `launchctl bootstrap` hint in `reissue_cc_oauth_token.sh`) |
| 6 | ownership orphaned by a closed step | `82_39::test_the_sweeps_recall_limit_is_recorded_not_assumed` |
| 7 | log/archive-rotation dependent | `23_2_6::test_..._backend_log_has_skipping_buy_evidence` |
| 8 | apparently genuine product assertions | `82_48` x2 (`assert 'UNKNOWN' == 'BUY'`), `test_portfolio_swap` (expected 2 swap SELLs, got 1) |

**Class 8 is the one that matters for money** and must not be triaged as
housekeeping. `'UNKNOWN' == 'BUY'` and a missing swap SELL are behaviour claims on
the trading path.

## Carried defect, for queueing

**86.31's step text cites `.claude/hooks/lib/qa_write_guard.py`, which never
existed.** The step is `done`; the text is wrong. Small, but it is exactly the
"never-existed path" class the 75.17 sweep is built to catch, and leaving it makes
that sweep permanently red for a reason unrelated to whatever it is next asked to
find.

## Gate status -- PLAN HAS NOT STARTED

The first gate run (`wf_078f4125-57a`) **DROPPED**: `gate_passed: false`,
`empty_or_errored_return`, after 266,235 tokens / 86 tool uses / 23 minutes.
Per `.claude/rules/research-gate.md` an errored return **is a failed gate, never a
pass**, so no contract may be written on it.

phase-86.37's write-first held: the brief survived at **76,648 chars** with
`brief_status: COMPLETE`, self-reporting 37 sources read in full (floor 5), 58 URLs
(floor 10), a recency scan, and audit-class coverage of 18 rounds with rounds 17
and 18 both dry (K=2 satisfied). A verify-and-return re-run is in flight for the
envelope alone.
