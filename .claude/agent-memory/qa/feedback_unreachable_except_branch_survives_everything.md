---
name: unreachable-except-branch-survives-everything
description: An error-handling branch whose collaborator never raises under the test fixtures is unreachable, so EVERY mutation of it survives -- including flipping fail-closed to fail-open; probe the collaborator before crediting coverage
metadata:
  type: feedback
---

Before crediting any coverage to an `except` / error-handling branch, **ask whether
the collaborator can raise at all under the harness's fixtures.** If it cannot, the
branch is unreachable, every mutation of it survives, and the suite is green for every
possible state of that code.

**Why:** 86.71 cycle 4 rewrote `verdict_outcomes`' broad except (added a loud stderr
disclosure, restated the fail-closed direction). Measured: the fix's revert (V1,
silent `except Exception: return []`) survived the 9-check mutation matrix AND the
16-check self-test -- **nothing** went red. So did V2, `return [Outcome.PASS]`, a
straight fail-OPEN budget bypass. Root cause was one line: every drive and every
self-test points `ATTEMPT_GATE_VERDICT_LEDGER` at an *absent* path, and
`emit_sequence` on an absent ledger returns `[]` **quietly** (`rc=0, absent -> []`).
The except never executes. `grep "verdict-ledger read failed"` confirmed it: two hits,
the source line and a narrative capture -- no test, no check, no cell.

**How to apply:**
- The tell is a fixture that *names* the error condition ("absent -> no PASS
  exception") while the collaborator handles it **without raising**. Absent is not
  the same as unreadable.
- Drive the collaborator directly first: `emit_sequence(x, absent_path)` -- does it
  raise or return? One `python -c` settles it.
- Make the branch reachable with a fixture that genuinely raises. Pointing the path at
  a **directory** gives `IsADirectoryError` for free -- that is what the author's own
  hand-run demonstration used, and it is exactly the fixture the automated check was
  missing.
- A live one-off demonstration is not a revert-test. Criterion wording is usually
  "revert it and show the check goes red"; a demo shows the code works today and says
  nothing about whether anything would notice if it stopped.
- Always mutate the *direction* too, not just the disclosure: fail-closed -> fail-open
  is the mutation that matters, and it is the one an unreachable branch hides.

Related: [[a-provenance-fix-that-only-logs]],
[[oracle-with-silent-fallback-survives-absent-subject]],
[[mutate-each-half-of-an-ANDed-guard]].
