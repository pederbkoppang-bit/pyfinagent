# experiment_results -- phase-86.86 (D6)

**Step:** 86.86 (P1, LIVE MONEY). **Date:** 2026-08-15. **Cycle:** 1.
**Verbatim command output lives in `handoff/current/live_check_86.86.md`** --
this file is the summary of what was built and why; that file is the evidence.

---

## What was built

A defect where the lite risk-judge paths rewrote an explicit **0% position
verdict** into the **3.0 default** at the construction seam, upstream of every
guard phase-86.74 built. Pre-fix, judge-said-zero and judge-said-nothing were
indistinguishable after the line, and a 0.0 verdict produced a **BUY of $719.93
on NAV 23,997.71** where the true verdict produces no order -- in **all four**
flag combinations, because the decision was `APPROVE_REDUCED` and
`paper_risk_judge_reject_binding` only blocks an exact `REJECT`.

### Files changed

| file | change |
|---|---|
| `backend/services/autonomous_loop.py` | `+_lite_position_pct` (the one seam), `+_build_lite_risk_assessment` (the one producer), both lite paths routed through it, import extended with `SIZE`/`UNPARSEABLE`/`_resolve_position_pct` |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | **+21 tests (41 -> 62)**, driving the REAL producer and the REAL `decide_trades` |
| `scripts/qa/verify_lite_risk_seam_86_86.py` | **NEW** -- re-runnable AST class enumerator + seam guard, with positive AND negative controls |
| `scripts/qa/mutation_matrix_86_86.py` | **NEW** -- 6 producer mutation cells |
| `.claude/masterplan.json` | `+86.86` (this step), `+86.87` (queued sweep finding) |
| `handoff/current/` | `contract_86.86.md`, `research_brief_86.86.md`, `live_check_86.86.md`, this file |

`scripts/qa/mutation_matrix_86_74.py` was **not** touched -- different step,
different subject (the consumer).

### The fix, in one line

```python
"recommended_position_pct": _lite_position_pct(risk_dict, ticker),
```

`_lite_position_pct` routes through `_resolve_position_pct` -- the **same**
three-state resolver the full path uses, not a second parallel idiom:
`SIZE` -> the judge's number (0.0 included); `UNPARSEABLE` -> 0.0, fail closed
**and loud**; `ABSENT` -> the 3.0 default, and only ABSENT reaches it.

**Why the producer was extracted rather than edited in place:** the two lite
paths held byte-identical copies of the dict literal, so a fix to one would
silently miss the other; and a literal buried in a 300-line async LLM function
cannot be driven by a test, which makes any mutation cell aimed at it
UNSCORABLE. The research gate named this explicitly.

---

## Verification performed (all output verbatim in `live_check_86.86.md`)

| check | result |
|---|---|
| Pre-fix reproduction, seam + real `decide_trades`, 4 flag combos | 0.0 and absent both -> 3.0; `BUY $719.93` in all 4 |
| AST class enumeration, pre-fix | 10 sites / 5 keys; positive control found all known members |
| Seam checker `verify_lite_risk_seam_86_86.py` | **8 checks emitted, 8 PASS, 0 FAIL**, exit 0 |
| Positive control (criterion 4) | `0.0 -> 0.0` **while** `absent -> 3.0` |
| Disclosure table (criterion 6) | 9 inputs, matches the contract's pre-written prediction row for row |
| End-to-end, real `decide_trades`, 4 flag combos | `0.0 = no order`; `3.0` and `absent` unchanged at `BUY $719.93` |
| Mutation matrix | **6 cells, 6 KILLED**, both controls GREEN first, restore sha256-verified |
| Immutable command | **62 passed** (was 41) |
| Full backend suite | 21 failed / 3493 passed -> **zero attributable to this change** (see below) |

### Regression causality, measured not assumed

`autonomous_loop.py` was reverted to HEAD and the identical 21 failing node ids
re-run, then restored byte-identically (sha256 checked): **20 failures at HEAD,
20 with the fix, set difference EMPTY.** One extra failure in the first
full-suite run (`test_phase_86_6_subprocess_channel.py::test_the_optin_IS_honoured...`)
passed in isolation and did not reproduce; it shells out against the live
backend on :8000. Reported, not attributed. The pre-existing failures are
already tracked as masterplan steps **86.51** (the swap test) and **86.5** (the
suite triage).

---

## Findings that changed the plan

1. **The research gate corrected the fix location.** 86.74 hardened the
   *consumer*; the collapse is at the *producer*. `PositionVerdict(SIZE, 3.0)`
   downstream was a correct reading of an already-falsified value -- the
   resolver was blind, not wrong.

2. **`paper_risk_judge_shape_fix_enabled` has ZERO production readers**
   (verified by Main, not taken on trust). Criterion 5 is satisfied, but the
   live_check states plainly that parametrising over that flag proves
   *insensitivity*, not branch coverage -- there is no gated branch. A green
   matrix must not be over-read here.

3. **I corrected the research brief on two of its three "harmful" calls, by
   measurement.** The brief called `decision` a fail-open and `risk_level`
   harmful-mild. Driven through the real `decide_trades`, neither changes any
   order: `""` and `APPROVE_REDUCED` both BUY $719.93 (only exact `REJECT`
   blocks), and `risk_level` is read zero times by `portfolio_manager`. The
   brief's insight is kept rather than discarded: `decision` is a **latent**
   fail-open that would invert under an allow-list gate.

4. **The harm bound in the D6 brief was too generous.** It bounded exposure by
   `paper_risk_judge_reject_binding=True`. With a non-REJECT decision paired
   with 0.0 -- the brief's own case (a) -- the BUY fires **with binding ON**.

5. **A serialisation asymmetry nobody had recorded:** the string `"0"` survived
   the old expression while the float `0.0` died, because the falsy test
   preceded `float()`.

---

## Scope calls, stated rather than left to be discovered

- **Fixed:** the one decision-inverting member.
- **Queued as 86.87:** the three audit-fabricating members (`reasoning`,
  `decision`, `risk_level`). The substituted `reasoning` reads *"risk-judge
  parse failed; falling back to conservative default sizing"* when the parse
  **succeeded** -- a false statement in a persisted audit column. Real, but a
  design question (what should be written instead?), not a P1 money fix.
- **Deliberately untouched:** `risk_limits`. Its substitution **installs** a
  stop where none existed (90.0 vs 92.0 on a 100.0 entry) -- protective. A test
  and a comment both say so, so a later reader does not "finish the sweep" by
  removing it.

---

## Status

- **NOT YET IN FORCE.** Backend restarts are batched to session end; the running
  process still holds the pre-fix module. Every measurement above is from a
  fresh process. Pending-restart list: `backend/services/autonomous_loop.py`.
- **Unverified and stated as such** (full list in `live_check_86.86.md` §9): no
  BigQuery query was run for a historical lite row carrying a 0.0 verdict, so
  the *frequency* of live exposure is unmeasured -- only its mechanism and
  magnitude; and the flaky subprocess test was not root caused, only shown not
  to be attributable here.
