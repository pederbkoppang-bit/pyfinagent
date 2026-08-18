---
name: debrittling-an-or-clause-makes-a-guard-vacuous
description: A clause added to stop a guard's false positives can be unconditionally true, killing the working half; mutate BOTH sides of an OR and check whether the escape clause is already True on the control
metadata:
  type: feedback
---

When an author records "my first version of this guard was too BRITTLE, so I
replaced/relaxed it", that is the guard to mutate first. The remedy for a false
positive is very often an `or <fallback>` clause, and if that fallback is
**already True on the unmutated tree** it overrides the working half forever.

**Why:** step 86.116 cycle 2, `scripts/qa/verify_86_116.py:319-325`. Main had
just been capped for crediting a dead key, added two tripwires so "the correction
cannot silently rot", and recorded that the FIRST tripwire attempt was a
grep-with-filters that fired on the setter's own guard -- too brittle. The
shipped replacement for the sibling tripwire was:

```python
"vol" not in label_src.split("tp_price")[0].split("def ")[-1].lower()
or "self.tp_pct" in label_src
```

Measured in-memory against `backtest_engine.py`:

| mutant | A | B | guard |
|---|---|---|---|
| control | True | True | passes |
| vol term ADDED, `self.tp_pct` kept | **False** | True | **passes -- SURVIVED** |
| vol term ADDED + `self.tp_pct` renamed | False | False | fires |

Clause A -- the working half -- DOES fire. Clause B is unconditionally true on
the current code AND on any realistic re-wiring (a vol-scaled barrier is
`vol_mult * self.tp_pct`, which KEEPS the token), so the guard detects a RENAME,
not the volatility term its own failure message names. Main's code comment called
this one "the claim that actually matters".

**How to apply:** (1) evaluate EACH clause of an OR separately **on the control**
-- an escape clause that is already True there is dead weight by construction,
and that check costs one print; (2) build the mutant in the shape a real change
would take (keep the tokens a real refactor would keep) rather than the shape
that is easiest to detect; (3) treat "I made it less brittle" in a spawn prompt
as a mutation target, not as reassurance; (4) also check fixed-width source
slices (`src[i:i+2600]` over a 1,788-char function overshoots into the next
function, so a neighbour can satisfy the clause). Related: shape #8 in qa.md 4c,
[[mutate-each-half-of-an-ANDed-guard]], [[static-form-guard-rejects-one-syntax-not-the-class]],
[[anti-vacuity-check-that-is-itself-a-tautology]].
