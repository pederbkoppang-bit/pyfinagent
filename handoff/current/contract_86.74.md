# Contract -- step 86.74

**Step:** the risk judge REJECTED DELL at 0% and the book bought it at 10% of NAV
-- a falsy-zero check inverts the strongest risk signal into the largest default
position.
**Priority:** P0 (live money, paper book).
**Written:** 2026-08-14, BEFORE any code change (per
`feedback_contract_before_generate`).

---

## 1. Research gate -- SATISFIED, and NOT re-run

`handoff/current/research_brief_86.74.md` (32,922 chars) cleared the gate. The
verdict cited here is the **script-enforced** one from the `research-gate`
Workflow run record, **not** the brief's self-report -- `research-gate.js`
recomputes `gate_passed` and never trusts the agent's value:

```
run       wf_b20b0f86-52f   status=completed   completion 2026-08-14T10:28:31Z
gate_passed                       : true    (ENFORCED, recomputed)
agent_self_reported_gate_passed   : true
self_report_disagreed             : false
sources_floor_ok  : 7 >= 5        urls_floor_ok : 27 >= 10
brief_on_disk_ok  : 32922 chars, independently read
brief_status      : COMPLETE
all_7_claimed_sources_present_in_brief
```

**This gate was deliberately NOT re-run.** Re-researching a step whose brief is
on disk and whose gate is enforced-green is precisely the waste step **86.76**
exists to stop (86.75 spent 1.45M tokens rebuilding shipped work). Per
`feedback_read_the_steps_prior_artifacts_first`, the prior artifacts were read
first and are the input to this contract.

### 1a. What the research adds beyond the incident memo

Four findings the incident write-up does **not** contain, all re-verified by me
against source before being restated here:

| # | Finding | Anchor |
|---|---|---|
| R1 | The **second** source `analysis["risk_judge_position_pct"]` keeps the falsy-zero check under **every** flag setting -- the approved fix never covered it | `portfolio_manager.py:948-953` |
| R2 | `except (ValueError, TypeError): pass` makes a **malformed** pct fall through to the 10% default -- fail-OPEN on parse error | `portfolio_manager.py:944-946, 951-953` |
| R3 | The **swap path** sizes via an unguarded `or 10.0` | `portfolio_manager.py:878` |
| R4 | Three default-OFF flags whose **conjunction** is the safety property; promoting one alone can escalate a parse failure from 3% to 10% NAV | `settings.py:342/346/350` |

---

## 2. Measurements I made before planning (not inherited)

### 2a. The 10%-default class is FOUR sites, not one (criterion 3 input)

Derived from source by enumeration, not asserted. Rule: every occurrence of a
literal `10.0` sizing fallback in `backend/services/portfolio_manager.py`,
found with `grep -n "or 10\.0\|10\.0  #"` and then read in context.

| line | path | flag-guarded? |
|---|---|---|
| `:507` | main buy loop | **YES** -- `shape_fix` picks `is not None` vs `or 10.0` |
| `:800` | cross-sector rotation safety probe (`_cross_rotation_safe` buy_amount) | **NO** |
| `:853` | swap sector-cap projection | **NO** |
| `:878` | swap execution sizing | **NO** |

**Three of four are unguarded under every flag state.** The brief found `:878`;
`:800` and `:853` are additional and are mine. Criterion 3 requires the set be
*derived*, so the derivation rule is stated above and the count is 4.

### 2b. Live flag state -- resolves the brief's open question 2

The brief recorded this as **UNMEASURED** (`GET /api/settings/` omits the keys,
and it warned: do not read absence as OFF). Measured in-process instead:

```
paper_risk_judge_reject_binding      True     <- from backend/.env:84
paper_risk_judge_parse_fail_reject   False    (code default; absent from .env)
paper_risk_judge_shape_fix_enabled   False    (code default; absent from .env)  <- THE DEFECT
paper_atomic_swap_enabled            False
```

Running backend `pid 27945`, started `2026-08-14 13:30:35 CEST`, i.e. **after**
`.env:84` -- so the running process holds these values, not merely the file
(`feedback_committed_is_not_in_force`).

**`reject_binding` is ON, and both the incident memo and the brief proceeded as
though all three flags were off.** It does not rescue the full path: it binds a
REJECT on the **lite** path only, while DELL went through the **full**
orchestrator, whose verdict is nested under `risk_assessment["judge"]` and is
never reached with `shape_fix` OFF.

### 2c. Test baseline -- the "9" is TESTS, not assertions (criterion 8 input)

```
test functions in test_phase_66_2_risk_judge_shape.py : 9    <- pytest -q prints "9 passed"
`assert` statements                                   : 17
```

Criterion 8 says "the total assertion count is reported against today's baseline
of 9". **9 is the test-function count, not the assertion count.** Both are
recorded with the population rule so a net removal is visible in either
denominator and the two are never conflated.

---

## 3. Hypothesis

A verdict of "0%" is not the absence of a size -- it is the **strongest possible
size instruction**. The defect is that `Optional[float]` cannot represent three
distinct states (REJECT-at-zero / explicit-size / no-verdict), so `None` is
overloaded and every downstream `or 10.0` resolves the ambiguity in the **most
dangerous** direction.

Fixing truthiness alone (`if pct:` -> `if pct is not None:`) is necessary but
insufficient: it leaves `None` still meaning both "judge said nothing" and
"parse failed". The fix must make the three states **representable**, and must
hold with `shape_fix` **OFF**, because OFF is the shipped production state.

---

## 4. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. the falsy-zero is fixed AT THE HELPER, not worked around: _extract_position_pct distinguishes an explicit 0.0 from an absent value, and the fix holds with paper_risk_judge_shape_fix_enabled OFF as well as ON -- prove both flag states by executed test, since the shipped production state is OFF
2. a REJECT binds: demonstrate by driving the real buy path that a risk verdict of 0% produces NO order, and that the assertion fails if the guard is reverted -- source inspection alone does not satisfy this criterion
3. the sizing default can no longer be reached from a verdict that specified a size: enumerate every path that yields the 10%-NAV default, derive that set from source rather than asserting it, and show each remaining one is a genuinely absent verdict
4. the verdict is PERSISTED per ticker -- risk_judge_decision, risk_level and recommended_position_pct are non-empty for a completed debate -- and the post-fix populated share is reported against the measured baseline of 0 of 129 rows over 2026-07-20..2026-08-13 with the query that produced each
5. the risk-debate completion log line carries its ticker, so concurrent debates are attributable without inference; state explicitly that this removes the elimination-based attribution this step's own evidence had to rely on
6. the RiskJudge contribution appears in signals_log.factors_json for a gated buy regardless of the pct value, including a 0% REJECT -- compare against the two measured records (DELL 3 agents/517 chars, NTAP 4 agents/1232 chars)
7. paper_trades history is SWEPT for prior buys opened under this inversion: report how many positions were sized at the 10%-NAV default while a completed risk verdict existed, state the enumeration rule, and report zero as a measured zero with a positive control rather than as an absence of evidence
8. the flag-ON-only test blindness is closed: every new and existing assertion in test_phase_66_2_risk_judge_shape.py is exercised in BOTH flag states, and the total assertion count is reported against today's baseline of 9 so a net removal is visible
9. mutation-test with the control observed GREEN first and a byte-identical restore: (M1) restore `if pct:` in the helper -- a 0% verdict must go red; (M2) restore `or 10.0` at the sizing seam -- the same; (M3) delete the persistence write -- criterion 4's check must go red; each cell scored UNSCORABLE if its control was not green
10. NO risk threshold is loosened and NO gate is weakened to make anything pass, and the DELL position opened under this defect is NOT liquidated or resized by this step -- any position remedy is a numbered operator ask, not executor work

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q'`

**live_check:** `live_check_86.74.md` carrying a driven 0%-verdict producing no
order with the flag OFF, the post-fix persisted-verdict share against the
0-of-129 baseline, and the paper_trades sweep result with its enumeration rule.

---

## 5. Plan

**P1 -- make the three states representable (criterion 1).** Introduce an
explicit sentinel distinguishing *absent* from *zero*, and rewrite
`_extract_position_pct` to use `is not None` on **both** sources (R1) and to
**stop swallowing** parse failures into a silent 10% (R2). The helper fix is
**unconditional** -- not behind `shape_fix` -- because criterion 1 requires it
to hold with the flag OFF.

**P2 -- close all four sizing seams (criterion 3).** `:507`, `:800`, `:853`,
`:878`. After the change, the 10% default must be reachable only from a
genuinely absent verdict, and each remaining reachable path is shown to be one.

**P3 -- persist the verdict per ticker (criterion 4).** `risk_judge_decision`,
`risk_level`, `recommended_position_pct` written for a completed debate; report
the post-fix share against the measured 0-of-129 baseline with the query.

**P4 -- put the ticker in the completion log line (criterion 5).**
`risk_debate.py:351` currently logs decision/risk_level/position/rounds and **no
ticker**, which is why this step's own evidence needed elimination-based
attribution.

**P5 -- keep RiskJudge in `factors_json` when pct is 0 (criterion 6).**

**P6 -- sweep `paper_trades` history (criterion 7)**, with the enumeration rule
stated and a positive control so a zero is a *measured* zero.

**P7 -- tests in BOTH flag states (criterion 8)** + **mutation matrix M1/M2/M3
(criterion 9)**, each cell control-green-first and byte-identically restored.

### 5a. DELIBERATE BEHAVIOUR CHANGE UNDER FLAG-OFF -- declared, not smuggled

The existing flags are documented "OFF -> byte-identical". **P1 and P2
deliberately break that for the falsy-zero path**, because criterion 1 demands
the fix hold with the flag OFF and OFF is the shipped production state. A
0%-REJECT will stop producing a 10%-NAV buy **regardless of flag**. That is the
entire point of the step, and it is recorded here rather than discovered in
review.

This is **not** a flag promotion and touches no `.env` (goal §4): it changes
code so the *default* path is safe. Flag 79.1 remains the operator's.

### 5b. Non-scope (criterion 10)

- **No threshold loosened, no gate weakened.** The change only ever makes a buy
  *less* likely.
- **The DELL position is NOT liquidated or resized.** Any position remedy is a
  numbered operator ask.
- No `.env` write, no flag promotion, no manual cycle, no backend restart before
  session end.
- Defects discovered mid-cycle are **queued as their own steps**, never fixed
  inline (`feedback_queue_discovered_defects_in_masterplan`).

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| The helper's other caller breaks on the new return type | Enumerate every caller before changing the signature; prefer a change that keeps `Optional[float]` at the boundary |
| A test passes because it re-encodes the buggy policy | Mutation matrix M1/M2 must go **red**; a green mutant is a failed cell (`feedback_green_assertion_may_encode_the_policy_you_invert`) |
| Criterion 2's "driving the real buy path" is faked with a stub | Drive `decide_trades` itself; assert the mutant fails (`feedback_drive_the_real_thing_for_behavioural_claims`) |
| Reporting 0 swept rows as "no exposure" | Criterion 7 requires a positive control -- a *measured* zero, not an absence of evidence |

## 7. References

- `handoff/current/research_brief_86.74.md` -- gate-enforced, run `wf_b20b0f86-52f`
- `handoff/current/incident_86_70_risk_gate_bypass_DELL.md` -- the diagnosis
- `.claude/masterplan.json` step 86.74; step 79.1 (operator flag promotion)
- Memory: `feedback_measure_dont_assert_claims`, `feedback_count_the_class_not_your_list`,
  `feedback_a_green_suite_can_be_blind`, `feedback_guards_stop_one_seam_short`
