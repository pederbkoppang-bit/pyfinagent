---
name: regenerated-label-is-a-claim-check-the-diff
description: A block labelled "(REGENERATED cycle N)" is a claim -- check `git show <sha> -- <file>`; a one-line diff means only the TOTAL was edited, and a count-fixing audit leaves the enumeration under the count stale
metadata:
  type: feedback
---

When an artifact carries a provenance marker on a captured command block --
`(REGENERATED cycle 4)`, "verbatim output", "captured from a live run" -- treat
the marker as a CLAIM and settle it with `git show <sha> -- <file>`. A one-line
diff to a 15-line capture means only the summary total was retyped.

**Why:** phase-86.90 cycle 4 (2026-08-16). The cycle-4 remediation W2 was
explicitly "every count is now DERIVED, not typed... with a post-audit that fails
if any stale count survives", and it printed `STALE COUNTS REMAINING: none`. It
still shipped three instances of its own class, because **the audit was pointed at
the NUMBERS and not at the LISTS the numbers head, nor at the captures**:
- `### Mutation matrix (6 cells, all KILLED)` over a table with **5** rows -- the
  cell added that very cycle was never added to the matrix it introduced.
- `ALL GREEN: 95 passed ... (REGENERATED cycle 3)` -- a cycle-4 figure under a
  cycle-3 label; marker and number updated inconsistently in one edit.
- `live_check ... (REGENERATED cycle 4)` where `git show` gave a **one-line** diff;
  the body carried 4 `KILLED` / 0 `CONTROL is clean` against a real run's 6 / 6,
  and named the new mutation cell **zero** times (`grep -c` = 0) -- so the
  operator-facing gate artifact never mentioned the evidence for that cycle's fix.

**How to apply:** three cheap mechanical checks, all seconds long.
1. `git show <sha> -- <artifact>` -- diff size vs the size of the block it claims
   to have regenerated.
2. Re-run the command and diff the SECTION, not the total: compare
   `grep -c ': KILLED'` and `grep -c 'CONTROL is clean'` (or the equivalent
   per-line markers) between the live run and the pasted block.
3. For every "N items" heading, count the rows under it programmatically. A
   correct derived count over a stale enumeration is the shape to look for --
   both cannot be checked by the same instrument.

Direction of harm is usually UNDERSTATEMENT, so this is WARN, not BLOCK: the
guard is real and re-runnable, only its transcript is wrong. But when the defect
is CREATED BY the remediation edit itself (it did not exist the cycle before),
it is a new finding, not an anchoring artifact -- check `git show` to tell those
apart before deciding.

**THE ZERO-LINE CASE -- run `git log -- <artifact>`, not just `git show <sha>`.**
86.88 cycle 2 (2026-08-16): the remediation said the bound was "corrected in BOTH
artifacts". `git log --oneline -- handoff/current/live_check_86.88.md` showed its
last commit was the CYCLE-1 one, and `git show --stat <cycle-2 sha>` did not list
it at all. `experiment_results` was fully rewritten; `live_check` -- the artifact
the masterplan's `verification.live_check` gate names -- was untouched, so it still
shipped a superseded 69-test mutation matrix whose M1 row said `1 failed, 68 passed`
against a measured `2 failed, 73 passed`, a `72 passed` gate line against a shipped
75, and a stated bound (`{**X}` "would NOT be seen") that the same cycle's widening
had made FALSE IN THE OPPOSITE DIRECTION. Nothing mechanical catches this:
`live_check_gate.py` checks only that the FILE EXISTS, never its content.
**So: enumerate the artifacts a claim says were updated, and `git log` each one
separately. "Both artifacts" is a set-membership claim; derive the set.** A bound
that becomes wrong in the *permissive* direction is worse than the original narrow
one -- direction of harm flips, so this variant is not automatically WARN.

Related: [[a-correction-must-replace-not-accompany]],
[[verbatim-paste-drift-arithmetic]], [[fixing-the-code-does-not-fix-the-prose]],
[[matrix-row-sums-pin-the-tree]].
