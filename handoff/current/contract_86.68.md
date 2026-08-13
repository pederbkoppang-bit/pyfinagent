# Contract — step 86.68

**Step:** 86.68 — the version number counts ATTEMPTS, not shipped work -- two steps that shipped nothing moved it 19 times
**Priority:** P3  |  **Status at contract time:** pending
**Date:** 2026-08-14

---

## Unusual posture, stated first: THE CODE ALREADY SHIPPED

The change is **live and governing every commit in this repo** (`fbac40d7`, 2026-08-13),
while the step is `pending` with **no handoff artifacts at all**. This contract exists so
the shipped change is **verified rather than trusted** — the same posture as 86.75.

**A change that governs every commit went in with no contract, no verdict and no
live_check. That is the gap this step closes, independent of whether the code is right.**

---

## Research gate — PASSED

Run `wf_79862bd1-cde`, tier `simple`. `gate_passed: true` recomputed, **no rail drop**.
7 sources read in full (floor 5), 36 URLs (floor 10), corroborated 36 <= 36, all 7 claimed
sources present, recency section present, `brief_status: COMPLETE`.
Brief: `handoff/current/research_brief_86.68.md` (38,128 chars).

---

## The load-bearing question is already answered: detection is on PARSED STATE

CLAUDE.md claims the flip is read from the **masterplan diff**, not the commit subject.
**If that were false the fix would reintroduce the exact defect it exists to remove** — a
subject is a claim, a diff is what happened. **Verified in source:** `_flip_magnitude`
(`:98`) takes no args and calls `_statuses(ref)`, which runs
`git show {ref}:.claude/masterplan.json`, JSON-parses it, and walks it into an
`id -> status` map (`:129-157`). It diffs **parsed state**. Not the subject, not text.

---

## Evidence already gathered (RE-DERIVE in GENERATE; do not inherit)

**Gate's controlled replay** over `fbac40d7..HEAD` (107 commits, 53 after the `:27`
skip-list): **retired rule = 40 bumps (39 patch); shipped rule = 1.** The discriminator is
the part that matters — the single survivor `2b50904a` **shares its subject shape with 34
non-bumping `phase-8X.Y:` commits**, a differential the subject cannot explain.

**Live natural experiment (mine):** ~107 commits → **exactly 1 bump** (6.93.220 →
6.93.221) on the **1** step that flipped, while **86.62 FAILED 4× and 86.9 FAILED 2×**,
all committed with `phase-86.x` subjects, producing **zero** bumps. That is criterion 3
demonstrated **in production against real failing steps**, stronger than a replay.

**Never-raise, both halves:** the `except -> "none"` path (`:170-173`) can never reach the
`:212` bump test, and PostToolUse runs **after** the commit exists — so a detector failure
can neither break a commit nor silently bump.

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. the bump-per-step distribution is RE-DERIVED at execution time with the classifying rule stated beside the counts, since both the commit history and the classifier may have moved
2. a bump trigger is chosen and justified against at least one alternative; the leading candidate is 'bump on the masterplan status flip to done' (the auto-commit hook already fires exactly there), which would make a patch mean one shipped step
3. whichever trigger is chosen, PARKED and FAILED steps are shown NOT to bump -- proven by replaying the 86.9 and 86.44 commit sequences through the new classifier and showing zero bumps where there were 9 and 10
4. Recent-Activity rows are UNCHANGED -- every commit still appears; only the version bump becomes rarer, and that separation is demonstrated rather than asserted
5. CLAUDE.md's classifier documentation is updated in the same change, since the current behaviour matches the current doc and leaving them out of sync would make the doc wrong instead of the code
6. mutation-test the new rule: revert it and show a replay of a parked step's commits produces bumps again, with the control observed GREEN first

Immutable verification command:
```
bash -c 'test -f .claude/hooks/post-commit-changelog.sh && bash -n .claude/hooks/post-commit-changelog.sh && echo classifier-parses'
```
Required live_check: `live_check_86.68.md with the before/after bump counts for the replayed 86.9 and 86.44 sequences`

---

## Plan

1. **Criterion 1 — re-derive the distribution at execution time**, stating the
   classifying rule beside the counts. Both history and classifier may have moved; the
   figures above are from 2026-08-14 and will be stale.
2. **Criterion 2 — justify the trigger against an alternative.** The gate measured both:
   **tag-triggered and phase-completion-only each yield 0 bumps in-window.** Use that,
   and say why status-flip beats them rather than asserting it.
3. **Criterion 3 — PARKED/FAILED must not bump.** The live 86.62/86.9 evidence above is
   the strongest available; replay their sequences as the criterion asks and report both.
4. **Criterion 4 — Recent-Activity rows unchanged. WATCH THE TRAP:** the row count reads
   flat (116 → 116) because **`MAX_ROWS=20` is BINDING at exactly 20 data rows** — it is a
   rolling window, so an unchanged count is **uninformative alone**. Demonstrate that
   every commit still gets a row by spot-checking commits *inside* the window, not by
   comparing totals.
5. **Criterion 5 — CLAUDE.md doc updated.** Already true, **but the gate found it now
   partly stale:** the `:176` gate makes `classify_commit` effectively **binary
   major/not-major**, so its minor/patch/none branches are **unreachable as bump
   decisions** and the docstring at `:60-70` no longer describes behaviour. Fix that in
   the same change, which is exactly what this criterion demands.
6. **Criterion 6 — mutation-test.** Revert the rule, replay a **parked** step's commits,
   show bumps return. **Observe the control GREEN first**, and restore byte-identically.

---

## Also found by the gate — record, do not silently fix

- **`is_chore` (`:180`) silently suppresses What's-New bullets too**, undocumented
  (Keep-a-Changelog endorses the behaviour, but it is undeclared).
- **The stderr marker's operator VISIBILITY is unverified** — PostToolUse stderr has
  historically been invisible here, so *"prints `[changelog] flip-detect FAILED`"* may not
  mean anyone sees it. A silent failure mode in a never-raise detector is worth a criterion
  of its own if it proves true; **file it, do not absorb it here** (only a criterion owns a
  fix).
- **Phase objects are singleton groups**, so flipping a phase object would be
  unconditionally major. Latent edge case; 86.58's flip correctly produced a patch because
  it did not empty phase-85.

---

## Constraints

Paper trading only. **Do not hand-edit a version number** — the whole defect class is
versions moving for reasons other than shipped work. No flag promotions, no `.env` writes.
**Do not weaken the never-raise property**: this hook must never break a commit.

---

## References

- `handoff/current/research_brief_86.68.md` — gate PASSED
- `.claude/hooks/post-commit-changelog.sh` — `:17` MAX_ROWS, `:59` classify_commit,
  `:98`/`:129-157` _flip_magnitude and _statuses, `:170-173` never-raise, `:176` the
  binary gate, `:180` is_chore, `:212` the bump test
- CLAUDE.md's changelog bullet — the rule, and the now-stale docstring it mirrors
- SemVer; Conventional Commits; semantic-release; changesets; Keep a Changelog
