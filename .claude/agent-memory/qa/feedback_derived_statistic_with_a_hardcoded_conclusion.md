---
name: derived-statistic-with-a-hardcoded-conclusion
description: A script can compute a statistic correctly and then print its INTERPRETATION unconditionally -- mutate the input and read the conclusion sentence, not the number
metadata:
  type: feedback
---

When a checker prints a computed table, mutate the table's INPUT and read the
**prose conclusion**, not the recomputed number. The number will move; the
sentence may not.

**Why:** 86.47 cycle 2. `drought_census_86_47.py` computed a four-null
sensitivity table genuinely (p and P(0 in n) both derived from `FUNNEL`), then
printed the interpretation as unconditional `print()` calls. Gutting the
healthy-null cell (`A_pre ok (unmarked)` BUYs 100 -> 1) produced, at exit 0 with
"OK: all 13 invariants hold":

```
Under the HEALTHY null (p=0.0177)
the silence IS surprising -- P(0 in 13) = 0.7928, and only 168 analyses
are needed to reach that bar, so 13 is already MORE than enough:
```

P=0.79 is the opposite of surprising and 13 < 168. Both halves self-contradict
on one page. Same shape at criterion 3: `19/34 -> 34/34` printed
"paper_trades BUY 34/34 = 100.0%" followed by "=> too sparse to key a funnel on".
The comparison `n >= need` was **asserted in prose**, never evaluated.

**How to apply:**
- The derivation being real does NOT make the guard real. Ask which `if` decides
  the sentence; if there is none, the conclusion is a literal.
- **Mutate the DENOMINATOR that feeds every statistic.** Here `n_an` came from
  `DAILY_TAIL` while the window table had 13 rows -- two representations of the
  same 13, never cross-checked. Corrupting one gave "P(0 in 76)" beside a row
  labelled "matches all 13 window rows" and "REACHED THE GATE 13/13".
- **A self-reported guard count is a literal too.** `N_INVARIANTS = 13` against
  14 `_check` calls: `--verify` printed a number it never measured, and neutering
  one guard (`True or ...`) stayed green. Always `grep -c` the guard calls and
  compare to the count the tool prints.
- Sibling shapes: [[assert-the-property-not-a-proxy]],
  [[anti-vacuity-check-that-is-itself-a-tautology]],
  [[survivor-needs-behavioural-differential]].
