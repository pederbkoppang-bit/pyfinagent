# Experiment results — step 86.68

**Step:** 86.68 — the version number counts ATTEMPTS, not shipped work
**Date:** 2026-08-14 ~06:20 CEST
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

## C4 — Recent-Activity rows are UNCHANGED, demonstrated by a natural experiment

**This session is the demonstration**: 20 commits, of which 8 are substantive
`phase-86.x:` commits.

```
  version at session start (34e5d0c6) : v6.93.221
  version now                          : v6.93.221      <- UNCHANGED
  Recent-Activity rows dated 2026-08-14: 20             <- all still written
```

Every substantive commit was checked individually against `CHANGELOG.md`: **10 of 10
present**, including all 8 that produced no bump.

> **A correction I owe, because I asserted the opposite mid-analysis.** Reading
> `is_chore = bump_type == "none"` and its comment (*"no version row AND no bullet"*), I
> announced a confirmed defect: that the new rule would strip Recent-Activity rows. **That
> was wrong.** `is_chore` gates the **bullet under a version header** (`:228`); the
> **Recent Activity table** is written by a separate, unconditional block. I inferred which
> artifact "bullet" meant instead of checking. The measurement above is what settled it.

The only commits without their own row are `chore: auto-changelog hook entry for <sha>` —
the hook's own bookkeeping commits, `none` under **both** rules. That predates 86.68 and is
unchanged by it.

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

**The control is asserted green before the mutant is scored**, and the harness exits
non-zero if it is not — a cell whose control is red is UNSCORABLE, not a pass. `REAL exit=0`
(captured directly, not through a pipe).

---

## Scope honesty

- **No production or trade-path file touched.** One new read-only replay script.
- **The 482/186/8 figures are re-derived at this tree** and will move again; the *rule* is
  the durable claim, not the counts.
- **C5 is met but with a stated wording gap** in CLAUDE.md, disclosed above.
- **No Q/A has graded this**, and the step is not flipped.
