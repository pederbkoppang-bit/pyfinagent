# Experiment results -- phase-86.88

**Cycle:** 1 · **Written:** 2026-08-16 · **Contract:** `handoff/current/contract_86.88.md`
**Commit:** `03386529`

---

## 1. Files changed

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | +37 -- the whole-default detection at the SEAM in `_lite_position_pct` |
| `backend/tests/test_phase_66_2_risk_judge_shape.py` | +322 -- `TestLiteRouteEndToEnd`: the first tests that drive `_run_claude_analysis` / `_run_gemini_analysis` at all. **62 -> 72** |
| `scripts/qa/verify_lite_risk_seam_86_86.py` | +55/-2 -- widened node shapes so the `<whole-dict>` branch fires on real matches, plus the classification the checker then demanded |
| `handoff/current/{contract,research_brief}_86.88.md` | handoff artifacts |

No `.env`, no flag promoted, no gate loosened.

---

## 2. Criterion 1 -- REPRODUCED first, control GREEN, controls discriminating

`backend/services/autonomous_loop.py` sha256 identical before and after every
cell: `5b714a9e5f43753c1eb1f59ade87e51c9f082511abc79f9afad19d32846ec586`.

```
control (pre-fix):  62 passed
```

N1 injected immediately before the producer call in `_run_claude_analysis`
(anchor `    risk_assessment = _build_lite_risk_assessment(risk_dict, ticker)`,
2 occurrences, first only -- as the 86.86 Q/A did):

```python
risk_dict['recommended_position_pct'] = (risk_dict.get('recommended_position_pct') or 3.0)
```

| mutation | 62-test suite | AST checker | verdict |
|---|---|---|---|
| **N1** caller-side pre-mangle | **62 passed -- BLIND** | **RESULT: OK -- BLIND** | **SURVIVED** |
| PC1 restore the D6 falsy-or at the seam | 11 failed | RESULT: FAILED (exit 1) | KILLED |
| PC2 neuter `_lite_position_pct` | 11 failed | RESULT: OK | KILLED |

Both controls KILL, so N1's survival is a fact about the guards and not a dead
probe -- which is what criterion 1 demands.

**PC2 adds something the step's `audit_basis` does not state:** the checker is
blind to a neutered *resolver* as well. The two guards have **different** blind
spots and N1 sits in the intersection, so closing only one would have left it
alive. That is why the fix has two halves.

**Root cause, measured rather than inferred:** the suite, the checker and all six
86.86 mutation cells anchor **at or below** `_build_lite_risk_assessment`. **No
test executed `_run_claude_analysis` or `_run_gemini_analysis`**, so not one
route into the seam was ever driven.

---

## 3. THE STEP'S PREMISE WAS WRONG, and saying so is criterion 4's answer

The step title and `audit_basis` call the checker's `<whole-dict>` branch
**DEAD**. Measured -- by me, and independently by the research gate:

```
=== the SHIPPED file ===                    <whole-dict> fired: False
=== CONTROL `x or _LITE_RISK_DEFAULT` ===   sites: [(1,'<whole-dict>'), (2,'recommended_position_pct')]
                                            <whole-dict> fired on the control: True
=== dict(_LITE_RISK_DEFAULT) ===            sites: []
```

The branch **fires**. It was *unreachable on this file's idioms* because the four
routes are `dict(...)` **Call** nodes and a Call is not a `BoolOp` operand.

**Criterion 4 offers "made LIVE ... or DELETED", and the answer is MADE LIVE by
widening the accepted node shapes** -- deleting a branch that works, because the
codebase happens not to use its idiom, would remove real coverage. Shown firing
on real matches:

```
  line  3214  <whole-dict-copy>
  line  3219  <whole-dict-copy>
  line  3448  <whole-dict-copy>
  line  3453  <whole-dict-copy>
```

**The widening announced itself, which is the behaviour worth keeping.** On the
first run after it, the immutable command went RED:

```
FAIL  unexpected member(s) in the class: ['<whole-dict-copy>'] -- classify them before shipping (criterion 3)
```

The checker refused to accept a newly-enumerated member until it was classified.
That is not an obstacle worked around -- it is criterion 5 being enforced by the
tool, so the classification below is a requirement the checker imposed rather
than a courtesy.

---

## 4. Criterion 5 -- the four routes, enumerated FROM SOURCE and classified

```
total ast.Name refs to _LITE_RISK_DEFAULT: 12
  1  Assign (the definition)
  7  subscript-read      2359 2362 2392 2394 2402 2403 2461
  4  whole-dict copy     3177 3182 3411 3416   (pre-fix line numbers)
```

**Is a judge FAILURE persisting as SIZE 3.0 rather than ABSENT acceptable? NO.**
It is the phase-86.74/86.86 collapse one seam over: one value carrying three
domain states, with *judge-crashed* and *judge-chose-3%* made indistinguishable
in the persisted audit trail.

**Fixed at the SEAM, as criterion 5 requires -- not at the four call sites.**
`_lite_position_pct` detects the whole-default by **value equality** (every route
arrives via `dict()`, which copies, so identity would fail) and resolves ABSENT.
CERT OBJ06-J's copy-then-validate, applied to a dict we do not own. A call-site
fix would have to be repeated at every future handler that falls back.

**The number is deliberately unchanged.** ABSENT resolves to the same 3.0
default, so no order outcome moves (criterion 7). Only the recorded provenance
changes.

---

## 5. Criterion 6 -- all four routes reached BY DRIVING

Four routes = {Claude, Gemini} x {no-JSON, exception}. Counted, not assumed --
three of the four were covered at one point and the fourth was added because the
enumeration said so, not because a test failed.

| Route | Driven by |
|---|---|
| Claude, no-JSON | `test_unparseable_judge_output_takes_the_whole_dict_route` |
| Claude, exception | `test_claude_route_risk_judge_EXCEPTION_takes_the_whole_dict_route` |
| Gemini, no-JSON | `test_gemini_route_no_JSON_takes_the_whole_dict_route` |
| Gemini, exception | `test_gemini_route_risk_judge_EXCEPTION_takes_the_whole_dict_route` |

Each asserts the risk-judge leg actually ran (`calls["n"] == 2`), so a stub that
silently failed to drive the path cannot pass vacuously.

---

## 6. Criterion 2 -- the end-to-end tests

Per Fowler (*Mocks Aren't Stubs*) and the Google SWE Book ch13: **stub the
transport, assert the state.** Only two external boundaries are stubbed --
market data (`_fetch_yf_market_data`) and the LLM transport. Everything between
is production code, including the JSON parse, the four fallbacks and the
producer. **No copy of the dict construction exists in the test.**

The Gemini route uses the shipped `client_override` parameter, and pins its model
from the shipped `GEMINI_WORKHORSE` constant rather than a hardcoded id, so a
workhorse change cannot silently skip that route.

---

## 7. Criteria 1+3 -- the mutation matrix, 7/7 KILLED

```
CONTROL: 69 passed | checker exit 0 -> GREEN
```

| Cell | Mutation | Result |
|---|---|---|
| M1 | N1 pre-mangle @ Claude route | **KILLED** (1 failed) |
| M2 | N1 pre-mangle @ Gemini route | **KILLED** (1 failed) |
| M3 | N1 pre-mangle @ BOTH routes | **KILLED** (2 failed) |
| M4 | revert the whole-default seam detection | **KILLED** (1 failed) |
| M5 | over-fire: treat EVERY dict as the whole default | **KILLED** (15 failed) |
| M6 | restore the D6 falsy-or at the seam | **KILLED** (14 failed, checker exit 1) |
| M7 | neuter `_lite_position_pct` entirely | **KILLED** (14 failed) |

Every restore sha256-verified byte-identical. **This matrix licenses exactly one
claim: these seven mutations were killed.**

**Two cells exist because I measured them SURVIVING first, and both are worth
recording:**

1. **N1 at the Gemini route survived** the first fix. Criterion 2 asks for "at
   least one" real path, so stopping there would have satisfied the letter --
   and would have shipped a guard I had already measured covering one of two
   routes. M5 exists for the same reason in the other direction.
2. **Reverting the N2 seam fix SURVIVED** with the whole suite green. The fix
   deliberately does not move the number, so **no number-asserting test can
   catch it** -- the deliverable is the recorded provenance, so the provenance is
   what is now asserted, with a discriminating negative (`test_a_real_judge_
   verdict_is_NOT_recorded_as_absent`) so the assertion cannot pass on an
   implementation that logs the line unconditionally.

---

## 8. Verification commands

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_lite_risk_seam_86_86.py'
checks emitted: 9  (PASS 9 / FAIL 0)
RESULT: OK                                          # exit 0  (immutable command; was 8 checks)

$ python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q
72 passed, 1 warning in 2.06s                       # was 62

$ ruff check --select F821,F401,F811 <the commit's .py files>
All checks passed!
```

---

## 9. Criteria 7 and 8

**7 -- MEASURED, under both flag states.** Every input in the 86.86 disclosure
table driven through the real `_lite_position_pct`, pre-fix vs post-fix, with
`paper_risk_judge_reject_binding` both `true` and `false`:

```
input                    PRE binding=true  POST binding=true  PRE binding=false  POST binding=false
judge 0.0                             0.0                0.0                0.0                 0.0
judge 3.0                             3.0                3.0                3.0                 3.0
judge ABSENT                          3.0                3.0                3.0                 3.0
judge string '0'                      0.0                0.0                0.0                 0.0
judge garbage 'high'                  0.0                0.0                0.0                 0.0
judge None                            3.0                3.0                3.0                 3.0
WHOLE DEFAULT dict                    3.0                3.0                3.0                 3.0

inputs whose RESOLVED NUMBER moved: NONE
restore byte-identical: True
```

The last row is the one this step changes, and it changes **provenance only**:
3.0 before, 3.0 after, now recorded as ABSENT rather than as an explicit SIZE.
Cell M5 is the guard against an over-fire that *would* move outcomes; it kills
with 15 failures.

*(This paragraph previously read "not yet demonstrated under both
`paper_risk_judge_reject_binding` states -- that is the gap named in §10", and
§10 carried it as the cycle's weakest point. It was measured rather than left
for the evaluator to find; the gap entry is REPLACED, not annotated.)*

**8 --** no gate loosened, no flag promoted, no `.env` written. The immutable
command went RED mid-work and was answered by **classifying** the new member, not
by relaxing the check.

---

## 10. Stated gaps, not discovered later

- The `<whole-dict-copy>` enumeration covers `dict()`, `copy()` and `deepcopy()`
  call shapes. A route using `{**_LITE_RISK_DEFAULT}` (a `Dict` node with
  `**` unpacking) would **not** be seen. No such route exists today; the bound is
  stated rather than implied.
- `86.87` (the retained `or _LITE_RISK_DEFAULT[...]` keys fabricating the
  persisted audit trail) remains separate and unfixed, as filed.
