# Experiment results -- step 86.29

**Step**: `86.29` (phase-86, P2) | **Phase**: GENERATE
**Driver**: Main (`pyfinagent-51`), Opus 5 / effort max, 2026-08-11
**Contract**: `handoff/current/contract_86.29.md` (commit `c806cad6`, written by
`pyfinagent-06` before any code; `git diff -- .claude/hooks/archive-handoff.sh`
was empty at that moment). This session executed GENERATE against that contract
after the peer session ceded the step.

---

## 0. What was wrong, in one paragraph

`.claude/hooks/archive-handoff.sh` has two branches. The step-specific one
iterates `"$CURRENT_DIR/${sid}-"*.md` and `"$CURRENT_DIR/phase-${sid}-"*.md` --
patterns that expect the step id at the FRONT, before a hyphen. The project has
named its per-step files in the SUFFIX form (`contract_86.29.md`) since around
phase-4.9. The two never meet, so that branch matched **zero** files for every
step id and the rolling branch was the only one that ever fired -- copying
whatever was last written to the unsuffixed `handoff/current/contract.md` and its
three siblings. Those four rolling files are stale, and from **three different
steps**, so an archive directory did not merely get "the wrong step", it got a
mixture: 82.54's contract, 82.6's results and critique, 80.2's brief.

---

## 1. Files changed

| file | change |
|---|---|
| `.claude/hooks/archive-handoff.sh` | the fix: derived names, guarded rolling fallback, loud empty-archive failure, `PROVENANCE.md` |
| `scripts/qa/derive_archive_misattribution_86_29.py` | **modified** -- added synthetic controls and a precision measurement with its own controls |
| `scripts/qa/prove_archive_provenance_86_29.py` | **new** -- scratch-tree before/after driver + 4-cell mutation matrix |
| `handoff/current/live_check_86.29.md` | **new** -- verbatim evidence |
| `handoff/current/experiment_results_86.29.md` | this file |

Nothing else. The scope was derived from `git status --porcelain` over those
explicit paths, not from a directory glob -- a peer session is live in this
repository and a directory-scoped claim would be false the moment they commit.

---

## 2. The change, and the two design decisions inside it

**Branch 1 -- DERIVE the names (criterion 4, first branch).** The hook now
builds `contract_<sid>.md` from the step id and copies it to
`<archive>/contract.md`, for the five artifact bases (`contract`,
`experiment_results`, `evaluator_critique`, `research_brief`, `live_check`), plus
a variant glob so `research_brief_86.29_rerun.md` -- the artifact that actually
passed this step's gate -- is not dropped. This removes the hook/convention
agreement dependency rather than re-tuning it, which the step text asked for
explicitly: *"DO NOT let the fix depend on the archive hook and the naming
convention agreeing by convention -- that agreement is exactly what silently
broke."*

**Branch 2 -- the rolling files become a GUARDED fallback.** They are copied
only if the derived branch did not already supply that artifact AND the rolling
file **declares this step**. Blind substitution IS the defect, so "unsure" now
means "do not copy". The declaration grammar is implemented in `python3` (not
sed/grep) for identical behaviour under BSD and GNU userland, and is
deliberately the SAME pattern set the census script uses -- one grammar, two
consumers, so the hook and the audit tool cannot drift into disagreeing about
what "declares a step" means.

**DECISION 1 -- COPY, not move, and why that is not laziness.** The original
step-specific branch used `git mv`. Making a never-firing `git mv` branch
actually fire would activate a dormant race: `archive-handoff.sh` and
`auto-commit-and-push.sh` are both PostToolUse hooks on the same
`Write(.claude/masterplan.json)` matcher, and hooks under one matcher run in
**parallel**. A `git mv` racing a `git add -A && git commit` is a live-infra
failure mode this repository has already been bitten by in an adjacent form. The
archive is a *snapshot*; nothing in criteria 1-6 requires the source file to be
removed. `handoff/current/` hygiene stays with
`scripts/housekeeping/verify_handoff_layout.py`, where it already lives.
**Stated as a decision, not left as an implementation detail** -- it means
per-step files still accumulate in `handoff/current/` after a step closes, and
that is unchanged from today's behaviour, not a regression introduced here.

**DECISION 2 -- `PROVENANCE.md` in every archive dir.** The audit trail's
failure mode was not losing data, it was answering *confidently with another
step's work*. Every dir now records which source produced each archived file and
which rolling files were skipped and why. This is the OAIS provenance
requirement the research brief cited (650.0-M-3: Provenance must record "any
changes ... and who has had custody"): undocumented alteration is the
prohibition, not alteration.

**Loud failure (criterion 4, second branch).** An archive that captured nothing
for its own step now writes a FAILURE line to stderr, appends to
`handoff/logs/archive-handoff.log`, records `## RESULT: FAILED` inside
`PROVENANCE.md`, and emits a `systemMessage` on stdout. The `systemMessage` is
load-bearing: a warning written only to stderr or to a gitignored log is
invisible in practice, which is how this defect ran unnoticed for five days. The
JSON is serialised by `python3` so a message containing quotes or newlines
cannot emit malformed JSON. It does **not** block the masterplan Write -- the
fail-open `trap 'exit 0' EXIT` discipline is preserved unchanged.

---

## 3. Criterion by criterion

| # | criterion | evidence | status |
|---|---|---|---|
| 1 | population re-derived, recall validated against the two known positives before use; the 610 unparsed re-classified or explicitly still-unclassified | live_check D. Recall 2/2, controls 4/4, precision 1.0000 with its own SUSPECT/CONFIRMED controls. **153 mismatch / 386 agree / 255 unclassified / 24 no-contract over 818 dirs** at tree `f2eff942`. Of the former 610: 206 are harness per-cycle contracts (declare no step by design), **49 remain genuinely opaque and are reported as unclassified, not clean** | MET |
| 2 | mechanism demonstrated, not asserted: globs match ZERO files, rolling branch is the only one that fires | live_check B. Zero for five independent sids, **with a positive control returning 1** so the zeros are not a broken counter | MET |
| 3 | after the fix, archiving a step yields a dir whose contract.md declares THAT step -- driven against a synthetic step in a scratch tree, never against handoff/archive, before and after | live_check C. BEFORE = pre-fix hook recovered by `git show` and executed -> declares `82.54`. AFTER -> declares `99.1`. Isolation asserted: real hook digest and 818-dir archive list unchanged | MET |
| 4 | the fix does not rely on hook/convention agreement: hook DERIVES the names, **or** fails loudly on nothing-to-archive. A silent copy-nothing or copy-another-step's-files must be a visible failure | **Both branches implemented, not one.** Derivation (section 2) and the loud-failure path, covered by check `loud_on_empty` and mutation cell M4 | MET |
| 5 | state explicitly whether the 89 wrong dirs are backfilled; if backfilled, show the mapping came from git history not guesswork | live_check E. **NOT backfilled, stated plainly**, with reasons. Note the number is 153 at this tree, not 89 -- it is a moving target and the tree is named | MET |
| 6 | mutation-test: revert the fix and show the archiving test goes red | live_check C section D. 4 cells, **4 KILLED**. Each asserts its anchor exists before applying and refuses to score a no-op replace | MET |

---

## 4. What I did NOT do, and what remains open

- **Not backfilled** the 153 dirs (criterion 5, decision stated above).
- **Not touched** `phase-phase-*` directory naming. That is 86.19's Class A root
  cause -- the same file interpolating a raw `$sid` that already carries the
  `phase-` prefix -- and 4 such dirs exist. Same file, different defect; the
  contract names it out of scope and I did not absorb it silently.
- **Not changed** the legacy front-hyphen glob branch. It matches zero files
  today and will continue to; changing it adds risk with no benefit, and its
  `${sid}` vs `${short_sid}` inconsistency is part of the 86.19 defect above.
- **49 archive dirs remain genuinely unclassified.** Not clean -- unclassified.
- **Population grows if steps close before this lands.** Any step flipped to
  `done` with the pre-fix hook active mints another poisoned dir. If 86.25 /
  86.34 close before this commit, the count moves again; the census must be
  re-run and the tree named.

---

## 4b. DISCLOSURE -- this change ran on the live system before it was graded

**Stated because it is a real process defect of mine, not because anything broke.**

While I was still building the evidence, the peer session `pyfinagent-43` flipped
step 86.31 to `done`. That masterplan Write fired `archive-handoff.sh`, which
executed **my uncommitted, ungraded edit** on the live repository, and created
`handoff/archive/phase-86.31/`.

I had actually identified the mechanism -- live_check section F says this hook
"is invoked fresh by `bash` on every PostToolUse event ... it takes effect on the
next masterplan Write with no restart". I wrote that sentence and did not act on
its obvious implication: in a repository with a concurrent session, editing a
PostToolUse hook puts the edit in force for *the other session* immediately.
There is no staging period and no restart to batch it to. Had the edit been
broken, the casualty would have been the peer's step, not mine.

**What the live run produced, verified by me rather than taken from the peer's
report:**

```
handoff/archive/phase-86.31/PROVENANCE.md
## RESULT: ok -- derived=5 rolling_copied=0 legacy_moved=0 rolling_skipped=0

contract.md            0a108c9010247b11 == contract_86.31.md          IDENTICAL
experiment_results.md  15b07899d3e6b6c9 == experiment_results_86.31.md IDENTICAL
evaluator_critique.md  ce749cd746272bc1 == evaluator_critique_86.31.md IDENTICAL
research_brief.md      3e382df4df4ea65e == research_brief_86.31.md     IDENTICAL
live_check.md          28c38e75fa1f68dd == live_check_86.31.md         IDENTICAL

rolling  handoff/current/contract.md   9a819a708324f0ba  "# Contract -- phase-82.54"
archived phase-86.31/contract.md       0a108c9010247b11  "# Contract -- step 86.31"
```

`phase-86.31/` is the first archive directory since 2026-08-06 to contain its own
step's files.

**What this witness does NOT prove, stated precisely.** `rolling_skipped=0`. The
rolling loop never evaluated a single file, because the derived branch had
already supplied all four artifacts and the guard sits behind an early
`continue`. So the live run exercises **the derivation branch only** -- it is
*not* evidence that `rolling_declares_step` refuses poisoned input. That guard is
covered by the scratch-tree check `no_poison_substitution` and by mutation cells
M2 and M3, and by nothing else. A summary that reported this run as proving "the
poison was not copied" would be overclaiming: the poison was never offered.

**Consequence for criterion 1's population**: the archive grew from 818 to 819
directories during this step. The census in live_check D is pinned to tree
`f2eff942` and is a moving target by construction; section 6 below re-derives it.

---

## 5. Verification commands, verbatim output in the live_check

```
bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'   # EXIT=0
python scripts/qa/prove_archive_provenance_86_29.py                                              # RESULT: PASS (0 problems)
python scripts/qa/derive_archive_misattribution_86_29.py                                         # recall 2/2, controls 4/4, precision 1.0000
```

The immutable command is a **syntax check only**. It proves criterion 2 and
nothing else. The contract says so and it is repeated here so no future reader
mistakes a green command for a green step.

---

## 6. The population moved DURING the step -- both measurements, with their trees

Criterion 1's number is not a constant. It grows by one every time a step closes
under the old hook and stays flat when one closes under the new hook. Both
measurements are recorded rather than the flattering one:

| when | dirs | mismatch | agree | unclassified | no_contract |
|---|---|---|---|---|---|
| before the peer's 86.31 flip (tree `f2eff942`) | 818 | 153 | 386 | 255 | 24 |
| after it (same tree, archive grew) | 819 | **153** | **387** | 255 | 24 |

The delta is exactly one directory, and it landed in `agree`:

```
phase-86.31 -> ('agree', '86.31')
```

**That is the whole step in one line.** Under the old hook, closing a step added
one to `mismatch`. Under the new hook, closing a step adds one to `agree`. The
153 historical mismatches are unchanged because they are not backfilled
(criterion 5), and the population has stopped growing in the wrong direction.

Any future reader re-running the census will get a different total. That is
correct and expected; the tree must be named next to the number, which is why
both rows above carry theirs.
