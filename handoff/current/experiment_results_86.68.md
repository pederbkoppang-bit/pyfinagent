# Experiment results — step 86.68


> **TIMESTAMP CORRECTION (2026-08-14 04:35 CEST).** Wall-clock times in this file were
> **narrated, not measured** — I read the clock once at session start and invented a
> progression from it. The real session spans **08-13 23:10 → 08-14 04:26** (~5h), not the
> 16+ hours the original times implied. Times below are now the **git commit timestamps**
> of this artifact, which are ground truth. Durations and orderings derived from the old
> figures should be disregarded; the measurements themselves are unaffected.
**Step:** 86.68 — the version number counts ATTEMPTS, not shipped work
**Date:** 2026-08-14 ~03:27 CEST (git)
**Contract:** `handoff/current/contract_86.68.md` | **Gate:** PASSED (`91ab018e`)
**Immutable command:** `test -f … && bash -n .claude/hooks/post-commit-changelog.sh && echo classifier-parses` → **`classifier-parses`, exit 0**

> **The code was already shipped and governing every commit before this artifact existed.**
> This step's work is therefore *verification*, not construction — which is why criterion 1
> insists the distribution be **re-derived at execution time**: both the history and the
> classifier have moved since the change landed.

Replay harness (new, read-only): `scripts/qa/replay_changelog_rule_86_68.py`.

---

## C1 — the distribution, re-derived, with the rule stated beside it

**Rule as replayed:**
- **OLD** = subject-only. `phase-X.Y:` → patch, `phase-X.0:` → minor, `feat:` → minor,
  `chore|docs|refactor|test|style|ci|build:` → none, anything else → patch.
- **NEW** = the subject may force **major only** (`feat!:`/`fix!:`/`BREAKING CHANGE:`);
  otherwise `_flip_magnitude()` decides from the **parsed masterplan `id→status` map** at
  `HEAD~1` vs `HEAD`.

```
corpus: 482 commits since 2026-08-11 (re-derived at execution time)

  version bumps under OLD rule : 186
  version bumps under NEW rule : 8
```

**Divergence from this step's own `audit_basis`, reported not adopted.** The basis says
*348 commits → 136 old / 7 new*. I measure **482 → 186 / 8**. The corpus has grown by 134
commits since the basis was written; the ratio is stable (**73.1%** vs 73.4% of commits
losing their bump). Criterion 1 anticipates exactly this — *"since both the commit history
and the classifier may have moved."*

## C2 — the trigger, and the alternative it beat

**Chosen:** bump on a **masterplan status flip to `done`**, magnitude from the closure —
`major` if the flip emptied a whole top-level phase, `minor` if the step is a phase kickoff
(`X.0`), `patch` otherwise.

**Alternative considered and REJECTED, on evidence rather than taste:** grep the unified
diff for an added `"status": "done"` line. The docstring records why it was abandoned — a
scratch-repo test caught it **silently returning `none` whenever `masterplan.json` is
written compact**, since the whole file becomes one line and no line-anchored pattern
matches. Parsing both revisions is formatting-independent and detects a genuine state
transition rather than a textual coincidence.

**The second alternative — keep the subject prefix — is refuted by C1:** it produces 186
bumps over the same corpus, one per *attempt*.

## C3 — PARKED steps do not bump

```
  86.9   commits= 13  OLD bumps= 13  NEW bumps=  0   masterplan status=pending
  86.44  commits= 13  OLD bumps= 13  NEW bumps=  0   masterplan status=pending
```

**26 bumps become 0**, and both steps are still `pending` — neither ever reached a PASS.
(The basis said these two "moved the version 19 times"; re-derived at this tree it is
**26**. The larger figure strengthens the finding; it is reported rather than silently
adopted.)

## C4 — rows are independent of the bump — CORRECTED after the cycle-1 Q/A

> **My cycle-1 demonstration was CONFOUNDED, and the Q/A caught the exact thing I had
> flagged as a worry.** I wrote *"20 commits this session … 20 rows ← all still written"*
> and read that as coverage. **The 20 is `MAX_ROWS=20`** (`post-commit-changelog.sh:17`) —
> the trim cap — which I never mentioned. A count numerically identical to the cap cannot
> distinguish "every commit got a row" from "the table is simply full". My companion check
> ("10 of 10 substantive commits present") was a hand-assembled scope whose members all
> lay inside the surviving window, so it *structurally could not* observe the trimmed ones.

**Population rule:** a commit is ROW-ELIGIBLE unless its subject matches
`^chore: (auto-changelog|changelog drift)` — the skip at `post-commit-changelog.sh:27`,
which fires **before** any row is written.

**Derived (not hand-assembled), 2026-08-14:**

| quantity | value |
|---|---:|
| commits dated 2026-08-14 | **86** |
| skipped as chore | 43 |
| **row-eligible** | **43** |
| rows surviving in the table | **20** ← exactly `MAX_ROWS=20` |
| eligible commits **trimmed** | **23** |

**So a row count is NOT a census.** 23 eligible commits did get a row and then aged out
(e.g. `d5736cce phase-86.62`, `c5ad55d8 phase-86.62 cycle-3`).

### The evidence that actually carries the separation

```
row-eligible commits on 2026-08-14        : 43
of those that BUMPED the version          :  0
surviving rows in the table               : 20
surviving rows whose commit did NOT bump  : 20 of 20
```

**Rows exist exactly where bumps do not**, and this does not depend on the trimmed 23. The
mechanism confirms it structurally: the row-insert at `:252-270` is **unconditional**, while
the version header (`:212`) and the bullet (`:228`) are gated on `bump_type`.

*(Independently reproduced by the cycle-1 Q/A, which measured 84/42/22 against my 86/43/23 —
the delta is two commits I made between its run and this re-derivation.)*

## C5 — the documentation matches the code

`CLAUDE.md`'s classifier paragraph carries the new rule: *"Every commit gets a
Recent-Activity row; only SHIPPED WORK gets a version bump (phase-86.68)"*, the
flip-detection basis, the three magnitudes, the `feat!:`/`BREAKING CHANGE:` override, and
the never-raises contract with its `[changelog] flip-detect FAILED` stderr marker.

**One precision gap, flagged rather than fixed** (it is doc wording, and the step's
criteria do not own it): CLAUDE.md says detection is *"from the masterplan diff"*, which
reads naturally as a **text** diff — the exact approach the implementation tested and
rejected. The docstring is precise; the doc sentence is loose. Queue-worthy, not a defect.

## C6 — mutation test, control observed GREEN first

Mutant: remove the flip gate, so the subject verdict governs again.

```
  86.9   CONTROL=0 (GREEN)  MUTANT=13  -> KILLED
  86.44  CONTROL=0 (GREEN)  MUTANT=13  -> KILLED
```

**The control is asserted green before the mutant is scored** — a cell whose control is red
is UNSCORABLE, not a pass.

**Hardened after the cycle-1 Q/A.** It found a real residual: the exit code gated **only**
on control-greenness, so its MUTANT B (mutant arm neutered) exited **0** while both cells
reported SURVIVED — meaning the quoted `REAL exit=0` did not by itself evidence a kill. The
gate now requires **control-green AND all-cells-killed AND cells_scored > 0**, and prints
its own reasoning:

```
exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
REAL exit=0     (captured bare, never through a pipe)
```

The Q/A also verified the gate cannot be bypassed from inside: its MUTANT A (flip gate dead
in *both* arms) produced `CONTROL=13 (NOT GREEN -- cell UNSCORABLE)` and exit 1.

---

## Scope honesty

- **No production or trade-path file touched.** One new read-only replay script.
- **The 482/186/8 figures are re-derived at this tree** and will move again; the *rule* is
  the durable claim, not the counts.
- **C5 is met but with a stated wording gap** in CLAUDE.md, disclosed above.
- **A Q/A HAS graded this: CONDITIONAL on attempt 1** (`wf_aebf89bf-bfd`), transcribed
  verbatim in `evaluator_critique_86.68.md`. Criteria 1, 2, 3, 5, 6 MET and independently
  reproduced; **criterion 4 NOT MET as evidence** and the `qa.md` §1a lint gate red. Both
  are fixed above and in `scripts/qa/replay_changelog_rule_86_68.py`; a fresh Q/A must grade
  the changed evidence. **The step is NOT flipped.**
- **The Q/A retired the obvious objection to the replay harness by execution**: it extracted
  `classify_commit` and `_flip_magnitude` verbatim from the hook and drove them per-sha —
  **0 mismatches** against my re-implementation across 496 commits, production counts
  OLD=191 / NEW=8.
- **C1 narrowness, noted by the Q/A and worth carrying**: the criterion asks for the
  bump-per-**step** distribution; I gave corpus totals plus two named steps. Its derivation:
  43 steps bumped under the old rule, 177 of 191 bumps attributable to a step, and the top
  offender is **86.38 at 22 bumps** — above both 86.9 (13) and 86.44 (13).
