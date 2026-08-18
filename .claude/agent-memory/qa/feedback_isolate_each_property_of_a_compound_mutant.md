---
name: isolate-each-property-of-a-compound-mutant
description: A cell that mutates two properties at once is killed by whichever check fires first, so the other check is never proven to own anything -- build the single-property mutant; and a stderr-substring guard asserts PRESENCE, not PROVENANCE
metadata:
  type: feedback
---

When a fix adds TWO properties in one block (86.71 c5: `verdict_outcomes`'
except branch had to be **loud** AND **fail-closed**), the author's matrix
cells mutate the whole block. Cell G9 replaced `print(...) + return []` with
`return [Outcome.PASS]` -- it removes the loudness AND flips the direction, so
it dies to the loudness check and the **direction check is never shown to kill
anything**. `failed[0]` is all the matrix prints, so the report reads as if one
check did all the work.

**Build the isolated mutant: change ONE property, keep the other intact.**
Measured on 86.71 c5, matrix + self-test each run three ways:

| mutant | loudness check | direction check |
|---|---|---|
| G8 = revert to silent `return []` | RED | green |
| **V2_ISOLATED = keep the print, only `return []` -> `return [Outcome.PASS]`** | green | **RED** |
| G9 = the shipped compound cell | RED | RED |

Both checks own exactly one property; V2_ISOLATED flipped at_vlerr rc 2 -> 0,
a real fail-OPEN budget bypass. Without that row the direction check could have
been vacuous and the matrix would still have printed 9/9 KILLED.

**Second half: a substring assertion on stderr asserts PRESENCE, not
PROVENANCE.** I reverted the swallow to silent AND added an unconditional
`print("verdict-ledger read failed (decoy)")` elsewhere in the file -- the
matrix went 0 failing checks, SURVIVED. Not a capping finding (the *real*
revert does go red, which is what "revert it and show the check goes red"
asks, and the paired rc check cannot be decoyed by a print), but the named
hardening is to assert something only the branch can produce: the exception
type name, or the step id interpolated at the site.

**Why:** qa.md 4c shape #11 (mis-attributed kill mechanism) usually shows up as
"the kill was made by a different artifact". This is its quieter form -- the
kill was made by the right suite but by a *different check than the cell's
description credits*, so a second check rides along untested forever.

**How to apply:** for any cell whose `find` block spans more than one
behaviour, decompose it and run one mutant per property; then diff WHICH check
each one turns red. If two mutants fail the same single check, one property has
no guard. See [[matrix-oracle-inherits-selftest-blindspots]],
[[mutate-each-half-of-an-ANDed-guard]],
[[unreachable-except-branch-survives-everything]].
