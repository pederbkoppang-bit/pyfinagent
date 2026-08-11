# Contract -- step 86.5

**Step**: `86.5` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-11 (~14:0x CEST, read from `date`) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** No production file is modified at this moment.

**The deliverable of this step is THE STEPS, not the fixes.**

---

## 1. Research gate

**PASSED** -- `wf_df74423d-2f9`, tier `moderate`, audit-class. Script-enforced and
recomputed: **37 sources read in full** (floor 5), **58 URLs** (floor 10, 66
distinct present on disk), recency scan present, **all 37 claimed URLs verified in
the brief**, `brief_status: COMPLETE`, `coverage.dry = true` (18 rounds, trailing
rounds 17-18 dry, K=2), 19 internal files inspected.

**The first gate run DROPPED** (`wf_078f4125-57a`: `gate_passed: false`,
`empty_or_errored_return`, 266,235 tokens / 86 tool uses / 23 min). Per
`.claude/rules/research-gate.md` an errored return **is a failed gate, never a
pass**, so no contract was written on it. phase-86.37's write-first preserved the
brief; a **verify-and-return-only** re-run supplied the envelope in **142,129
tokens / 20 tool uses / 254s** -- the lean-prompt discipline measured on 86.32.

## 2. The count, and why it is wrong in a specific way

```
$ python -m pytest backend/tests/ -q -p no:randomly
17 failed, 3417 passed, 12 skipped, 5 xfailed, 1 xpassed in 441.51s (0:07:21)
```

**The step's title says 26; the measurement is 17.** Three independent runs agree
(a peer twice, me once). **The gate adds the nuance that matters: 26 is STALE, not
WRONG** -- the suite has gained ~400 tests since the 2026-08-08 baseline, so the
population moved rather than the earlier count being a mistake.

Also measured by the gate, and load-bearing for the filing shape: **all 17 are
deterministic -- 0 order-dependent, 0 clock-dependent.** So none of the flaky-test
machinery in the literature applies. This is not a flakiness problem; it is a
**test-correctness** problem, and that inverts the default remedy.

## 3. THE HEADLINE: in 9 of 17, THE TEST IS WRONG AND THE CODE IS RIGHT

Five of eight root-cause groups (A, C, D, E, F) are cases where production is
correct. **Two of them carry fixes that would cause real harm if applied naively.**

### Group A (4 tests) -- would DISARM TWO ARMED MONEY-PATH FLAGS

The tests assert a **code default**, but `Settings()` resolves from `backend/.env`,
so they actually measure an **operator-set override**. Verified by me:

```
backend/.env:83  PAPER_DATA_INTEGRITY_ENABLED=true
backend/.env:84  PAPER_RISK_JUDGE_REJECT_BINDING=true

Settings().paper_data_integrity_enabled     = True
Settings().paper_risk_judge_reject_binding  = True
```

This is a **test-isolation defect** (Fowler, "lack of isolation") -- neither an
obsolete expectation nor a code defect. The invariant the tests mean to protect
(flag-OFF is byte-identical) is still live and still valuable.

**"Fixing" these by flipping the defaults or the `.env` would silently disarm two
deliberately-armed money-path features.** That is the single most dangerous action
available anywhere in this step.

> **I VERIFIED THE BRIEF'S REMEDY AND ITS FIRST OPTION DOES NOT WORK.** The brief
> proposes `Settings(_env_file=None)`. Executed, that raises
> `ValidationError: 4 validation errors` -- `gcp_project_id`, `rag_data_store_id`,
> `ingestion_agent_url`, `quant_agent_url` are all required and also sourced from
> `.env`. **Its second option -- override the two fields explicitly -- is the one
> that runs.** Recorded because a filed step must not carry a remedy that fails on
> first contact.

> **AND THIS CORRECTS MY OWN EARLIER CLASSIFICATION.** My pre-gate census filed
> these as *"dark-flag defaults (flags OFF by design, awaiting operator
> activation)"*. **They are the opposite: armed.** I inferred the flag state from
> the test's own wording instead of reading `.env`. Same defect class as everything
> else today -- a claim about state I had not measured.

### Group D (1 test) -- would DELETE THE OPERATOR'S RESTART INSTRUCTIONS

`test_c6_no_launchctl_bootstrap_executed_in_ops_scripts` scans `*.sh` line by line,
skipping only `#` comments, and **is blind to heredocs**. The hit in
`reissue_cc_oauth_token.sh` is a *variable assignment* whose only expansion sits
inside a `cat <<EOF` -- **printed to the operator, never executed**. The script
states why at `:107-109`: bootout/bootstrap is deliberately not automated because
away-ops rail 9 reserves it for the operator.

**The production code is correct and the test's INTENT is correct; the
implementation of the check is wrong.** Editing the script to satisfy it would
delete operator restart instructions. The remedy is a heredoc-aware scanner. This
is `feedback_a_red_check_may_indict_the_probe` in the wild.

### The rest

| group | n | shape |
|---|---|---|
| B | 5 | frozen snapshot of a MOVING artifact (`.claude/masterplan.json`) -- includes one that caught a **real** never-existed path in 86.31, a step I closed today |
| C | 2 | artifact lifecycle: the file moved (archived on step close), the test did not |
| E | 1 | obsolete expectation, provably deliberate: asserts `effortLevel == 'xhigh'`; actual is `'max'`, raised by operator instruction 2026-08-04 |
| F | 1 | live-system-dependent, and **its declared quarantine was never wired** -- the skip never fires because the 14MB `backend.log` is present |
| G | 2 | live-BigQuery behavioural assertions; **likely duplicates 86.25** |
| H | 1 | **genuine behavioural regression** (`test_portfolio_swap`: expected 2 swap SELLs, got 1) |

## 4. Immutable success criteria

Copied verbatim from `verification.success_criteria`. **Carried here in the
contract itself** -- the 86.32 cycle-1 Q/A raised their absence from a contract
as a harness-compliance finding, and I am not repeating that.

> 1. every one of the 26 recorded node ids is accounted for: assigned to a named root-cause group, or shown to be already fixed, or shown to be an environment artifact -- with NONE left unclassified, and the accounting is a table an auditor can check line by line

> 2. each root cause is filed as its OWN masterplan step with harness_required true, an audit_basis written for an executor with no memory of this triage, and green-able immutable verification criteria (run the proposed verification command BEFORE freezing it -- a criterion that is already red for unrelated reasons is structurally uncloseable)

> 3. the grouping is justified by a MEASURED signature (the assertion/exception each test actually produces), not by filename similarity -- record the signature per test

> 4. the overlap with 36.28 and 86.3 is resolved explicitly: state which of the 26 are instances of the live-kill-switch-coupling class and therefore must NOT get duplicate steps

> 5. the re-measured live-tree baseline is recorded, and the measurement did NOT write to handoff/kill_switch_audit.jsonl -- proven by line count and sha256 before/after

> 6. no test is edited to obtain green in this step; fresh Q/A PASS


## 5. Plan

**P1 -- FILE 4-5 STEPS, NOT 17 AND NOT 8.** The gate's recommendation, and I agree:
groups collapse by *remedy*, not by root cause. One step per remedy shape.

**P2 -- EVERY FILED STEP CARRIES ITS OWN "DO NOT DO THIS" LINE.** Groups A and D
have naive fixes that are actively harmful; a step that names the failure without
naming the trap is worse than not filing it, because the next executor has less
context than I do now.

**P3 -- GROUP H IS THE MONEY ONE AND GETS ITS OWN STEP.** A missing swap SELL is a
behaviour claim on the trading path. It must not be filed alongside housekeeping.

**P4 -- G IS CHECKED AGAINST 86.25 BEFORE FILING.** The gate says likely duplicate.
Filing a duplicate of a closed step is waste; I will confirm rather than assume.

**P5 -- THE HEADLINE NEGATIVE IS REPORTED.** 18 research rounds found **no
mechanical procedure** for deciding test-wrong vs code-wrong. That dryness is a
finding: it means the per-case adjudication in this step cannot be automated, and
any future step promising to automate it should be treated sceptically.

### Explicitly NOT doing

- **Not** fixing any of the 17. The deliverable is the steps.
- **Not** touching `backend/.env`, any flag default, or `reissue_cc_oauth_token.sh`.
- **Not** quarantining anything silently -- Group F is already an instance of a
  quarantine that was declared and never wired.

### Risk

Two of the eight groups have harmful naive remedies, both involving live money-path
configuration. This step writes no code and changes no flag, so the risk is
entirely in what the FILED STEPS say. A step filed without its trap is the hazard.

## 6. References

- `handoff/current/research_brief_86.5.md` (gate `wf_df74423d-2f9`, 80,671 chars)
- `handoff/current/measurement_86.5_failure_census.md` (my pre-gate census)
- Fowler *NonDeterminism* + *SelfTestingCode*; Google Testing Blog; Meta
  probabilistic flakiness; GitLab quarantine process; iDFlakies; abseil SWE-book ch11
- `backend/.env:83-84`; `backend/config/settings.py:46,342`;
  `scripts/ops/reissue_cc_oauth_token.sh:17,107-109,110,117`;
  `backend/tests/test_phase_75_sre_ops.py:360-368`
