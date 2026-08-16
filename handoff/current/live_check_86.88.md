# live_check -- phase-86.88

Evidence artifact for the `verification.live_check` gate.
**REGENERATED WHOLESALE at cycle 3 from live runs -- never patched.**

*(Cycle 2 attempted to patch this file with a string replace whose anchor did not
match. The call silently no-op'd, the file was never committed, and the cycle-2
remediation nonetheless claimed the bound was "corrected in both artifacts". A
no-match replace looks identical to success. Found by the cycle-2 Q/A via
`git log -- handoff/current/live_check_86.88.md`, whose newest commit was still
cycle 1's. This file is now produced by a script that regenerates it and ASSERTS
the bytes changed.)*

---

## 1. The immutable command

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_lite_risk_seam_86_86.py'
phase-86.86 -- lite risk-judge INGRESS seam checker
======================================================================
  PASS  control(+): scanner FOUND the idiom it hunts (['recommended_position_pct', 'risk_level'])
  PASS  control(-): prose/comments quoting the idiom did NOT register (AST, not grep)

  Enumerated `or _LITE_RISK_DEFAULT[...]` sites in autonomous_loop.py: 8
    line  2452  key='reasoning'
    line  2454  key='decision'
    line  2462  key='risk_level'
    line  2463  key='risk_limits'
    line  3243  key='<whole-dict-copy>'
    line  3248  key='<whole-dict-copy>'
    line  3496  key='<whole-dict-copy>'
    line  3501  key='<whole-dict-copy>'

  PASS  'recommended_position_pct' appears in ZERO `or _LITE_RISK_DEFAULT[...]` nodes (the decision-inverting member is gone from the class)
  PASS  the 4 judge-failure whole-dict routes are SEEN by the scanner at [3243, 3248, 3496, 3501] (phase-86.88: the branch now fires on a real match, not only on a control)
  PASS  remaining members are exactly the retained set ['decision', 'reasoning', 'risk_level', 'risk_limits'] plus the classified <whole-dict-copy> routes
  PASS  exactly ONE function can reach _LITE_RISK_DEFAULT['recommended_position_pct']: _lite_position_pct (at line(s) [2401, 2402, 2404, 2419, 2422])
  PASS  _build_lite_risk_assessment defined exactly once (line 2425)
  PASS  BOTH lite paths route through the one producer (call sites [3252, 3507])
  PASS  _lite_position_pct: 1 definition (line 2335), 1 call site (line 2461) -- no second parallel idiom
======================================================================
checks emitted: 9  (PASS 9 / FAIL 0)

RESULT: OK
```

## 2. The scoped suite

```
$ python -m pytest backend/tests/test_phase_66_2_risk_judge_shape.py -q
77 passed, 1 warning in 2.07s
```

## 3. The `<whole-dict>` branch, firing on REAL matches

Driving the SHIPPED `or_default_sites` over the shipped module:

```
line  2452  reasoning
  line  2454  decision
  line  2462  risk_level
  line  2463  risk_limits
  line  3243  <whole-dict-copy>
  line  3248  <whole-dict-copy>
  line  3496  <whole-dict-copy>
  line  3501  <whole-dict-copy>
```

The branch was never dead -- it fires on `x or _LITE_RISK_DEFAULT` and was blind
only to the `dict(...)` **Call** shape, which is the shape this codebase uses.
Criterion 4's "made LIVE" is therefore satisfied by WIDENING the accepted node
shapes; the cycle-1 Q/A confirmed that reading is legitimate, not a
reinterpretation.

## 4. Mutation matrix -- 12 cells, all KILLED, on the SHIPPED tree

```
CONTROL: 77 passed | checker exit 0 -> GREEN

  KILLED  M1  N1 pre-mangle @ Claude route                  2 failed, 75 passed
  KILLED  M2  N1 pre-mangle @ Gemini route                  1 failed, 76 passed
  KILLED  M3  N1 pre-mangle @ BOTH routes                   3 failed, 74 passed
  KILLED  M4  revert the whole-default detection            4 failed, 73 passed
  KILLED  M5  over-fire: EVERY dict is the default         16 failed, 61 passed
  KILLED  M6  restore the D6 falsy-or at the seam          18 failed, 59 passed | checker exit 1
  KILLED  M7  neuter _lite_position_pct entirely           18 failed, 59 passed
  KILLED  M8  drop the additive key                         4 failed, 73 passed
  KILLED  M9  provenance key always False                   3 failed, 74 passed
  KILLED  M10 provenance key always True                    4 failed, 73 passed
  KILLED  M11 subset match ignoring 'reasoning'             1 failed, 76 passed
  KILLED  M12 drop the provenance from the PERSISTED blob   1 failed, 76 passed

12/12 killed | restore byte-identical: True
```

**M11 and M12 exist because they SURVIVED in cycle 2.** M11 is the exactness the
Q/A found unpinned (a subset match ignoring `reasoning` passed all 75 tests); M12
is the persistence the cycle-2 claim asserted and the code did not do.

**This matrix licenses exactly one claim: these twelve mutations were killed.**

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
```

The cycle-2 Q/A re-derived this independently, parent blob vs HEAD, including the
REAL `decide_trades` order outcome under both flag states: nothing moved, and the
key-set delta is purely additive.

## 6. The stated bound -- CORRECTED, in the right direction this time

All SEVEN shapes the Q/A probed are now SEEN by the checker: `dict(X)`,
`deepcopy(X)`, `copy.deepcopy(X)`, `copy.copy(X)`, `dict(**X)`, `X.copy()` and
the dict-unpacking form. Prose/comment and unrelated-dict negative controls
remain invisible.

*(Cycle 1 stated coverage of "dict(), copy() and deepcopy() call shapes" -- true
only of the bare-Name forms, i.e. NARROWER than reality. The un-regenerated
cycle-2 copy of this file then asserted the dict-unpacking form would NOT be
seen, which after the widening was WRONG IN THE PERMISSIVE DIRECTION. Both are
superseded by the measurement above.)*

**Residual, stated:** a route reaching the default through an intermediate alias
(`d = _LITE_RISK_DEFAULT; dict(d)`) is still unseen by this syntactic rule. The
runtime value-equality guard fires for it regardless, so the money path is
protected and only the "a fifth route announces itself" tripwire is weaker.

## 7. The provenance now reaches a PERSISTED artifact

Cycle 1 claimed a `logger.warning` was provenance. Cycle 2 claimed an in-memory
dict key was, and the Q/A measured the persisted blob sha256 IDENTICAL for a
judge that failed and a judge that chose 3%. Cycle 3 threads it into the lite
`full_report` on BOTH paths (count asserted == 2, not assumed), and
`test_judge_failure_is_distinguishable_IN_THE_PERSISTED_PAYLOAD` drives the real
`_persist_analysis` and asserts the two payloads DIFFER while the
`recommended_position_pct` column does not move.
