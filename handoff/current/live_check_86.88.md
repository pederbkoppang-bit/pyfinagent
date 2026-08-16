# live_check -- phase-86.88

Evidence artifact for the `verification.live_check` gate. Verbatim command output
only, captured 2026-08-16. Every mutation cell restores byte-identically.

---

## 1. N1 REPRODUCED (pre-fix) with its discriminating controls

```
control (pre-fix):  62 passed, 1 warning in 2.59s

=== N1: caller-side pre-mangle before the producer call (_run_claude_analysis) ===
  suite  : 62 passed, 1 warning in 1.96s
  checker: RESULT: OK
  -> N1 SURVIVED
  restore byte-identical: True (5b714a9e5f43753c...)

  PC1 restore the D6 falsy-or at the seam (known-killable)
      suite  : 11 failed, 51 passed        checker: RESULT: FAILED (exit 1)   -> KILLED
  PC2 make _lite_position_pct collapse 0.0 to the default (known-killable)
      suite  : 11 failed, 51 passed        checker: RESULT: OK (exit 0)       -> KILLED
```

Both controls KILL, so the SURVIVED result is about the guards, not a dead probe.
PC2 additionally shows the checker is blind to a neutered resolver, so the two
guards have DIFFERENT blind spots and N1 sat in the intersection.

## 2. The `<whole-dict>` branch -- NOT dead, and now firing on real matches

```
=== the SHIPPED file, through the shipped scanner (pre-widening) ===
  1 'decision'  1 'reasoning'  1 'risk_level'  1 'risk_limits'
  <whole-dict> fired: False
=== POSITIVE CONTROL: a module that DOES contain `x or _LITE_RISK_DEFAULT` ===
  sites: [(1, '<whole-dict>'), (2, 'recommended_position_pct')]
  <whole-dict> fired on the control: True
=== does dict(_LITE_RISK_DEFAULT) reach the branch? ===
  sites for a dict() copy: []

=== AFTER widening: fires on REAL matches in the shipped file ===
  line 2429 reasoning   line 2431 decision   line 2439 risk_level   line 2440 risk_limits
  line 3214 <whole-dict-copy>   line 3219 <whole-dict-copy>
  line 3448 <whole-dict-copy>   line 3453 <whole-dict-copy>
```

The widening made the immutable command go RED until the new member was
classified -- the checker enforcing criterion 5 on its own author:

```
FAIL  unexpected member(s) in the class: ['<whole-dict-copy>'] -- classify them before shipping (criterion 3)
```

## 3. The class, enumerated FROM SOURCE

```
total ast.Name refs to _LITE_RISK_DEFAULT: 12
  1  Assign (the definition itself)
  7  subscript-read      2359 2362 2392 2394 2402 2403 2461
  4  whole-dict copy     3177 3182 3411 3416   (pre-fix line numbers)
```

## 4. POST-FIX mutation matrix -- 7/7 KILLED, control GREEN first

```
CONTROL: 69 passed, 1 warning | checker exit 0 -> GREEN

  KILLED    M1 N1 pre-mangle @ Claude route          1 failed, 68 passed | checker exit 0
  KILLED    M2 N1 pre-mangle @ Gemini route          1 failed, 68 passed | checker exit 0
  KILLED    M3 N1 pre-mangle @ BOTH routes           2 failed, 67 passed | checker exit 0
  KILLED    M4 revert the whole-default detection    1 failed, 68 passed | checker exit 0
  KILLED    M5 over-fire: EVERY dict is the default 15 failed, 54 passed | checker exit 0
  KILLED    M6 restore the D6 falsy-or at the seam  14 failed, 55 passed | checker exit 1
  KILLED    M7 neuter _lite_position_pct entirely   14 failed, 55 passed | checker exit 0

7/7 killed | restore byte-identical: True
```

Two of these were measured SURVIVING first and are recorded as such: N1 at the
Gemini route survived until its E2E test existed (criterion 2 asks for "at least
one" path -- one would have satisfied the letter while shipping a guard already
measured to cover one of two routes), and reverting the N2 seam fix survived with
the whole suite green, because a fix that deliberately does not move a number
cannot be caught by a number-asserting test.

## 5. Criterion 7 -- MEASURED under BOTH flag states

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

The last row is what this step changes, and it changes PROVENANCE only.

## 6. Criterion 6 -- all four routes reached BY DRIVING

| Route | Test |
|---|---|
| Claude, no-JSON | `test_unparseable_judge_output_takes_the_whole_dict_route` |
| Claude, exception | `test_claude_route_risk_judge_EXCEPTION_takes_the_whole_dict_route` |
| Gemini, no-JSON | `test_gemini_route_no_JSON_takes_the_whole_dict_route` |
| Gemini, exception | `test_gemini_route_risk_judge_EXCEPTION_takes_the_whole_dict_route` |

Each asserts `calls["n"] == 2`, so a stub that failed to drive the real path
cannot pass vacuously.

## 7. Gates

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_lite_risk_seam_86_86.py'
checks emitted: 9  (PASS 9 / FAIL 0)
RESULT: OK                                          # exit 0 -- the IMMUTABLE command

$ python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q
72 passed, 1 warning in 2.06s                       # was 62

$ ruff check --select F821,F401,F811 <commit .py files>
All checks passed!
```

## 8. Stated bound

The `<whole-dict-copy>` enumeration covers `dict()`, `copy()` and `deepcopy()`
call shapes. A route written as `{**_LITE_RISK_DEFAULT}` (a `Dict` node with
`**` unpacking) would NOT be seen. No such route exists today; the bound is
stated rather than implied.
