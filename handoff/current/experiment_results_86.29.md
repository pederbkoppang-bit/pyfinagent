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
| `scripts/qa/derive_archive_misattribution_86_29.py` | **modified** -- synthetic controls, a precision measurement with its own controls, and (cycle 2) en/em-dash separators plus the corrected mention-vs-declare reporting |
| `scripts/qa/prove_archive_provenance_86_29.py` | **new** -- scratch-tree before/after driver; **6 behavioural checks and a 7-cell mutation matrix** as of cycle 3, with control-gating so a cell cannot score against an already-red check |
| `handoff/current/live_check_86.29.md` | **new**; **REGENERATED IN FULL at cycle 2** rather than edited in place |
| `handoff/current/evaluator_critique_86.29.md` | **new** at cycle 2 -- records the cycle-1 rail drop and the rescued write-first record |
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
sed/grep) for identical behaviour under BSD and GNU userland.

**This paragraph used to claim the hook and the census "cannot drift into
disagreeing" because they carry the SAME pattern set. That claim was false and
it was falsified within one cycle.** Cycle 2 widened the census separator to
accept en/em-dashes and did not touch the hook, so an identical declaration
written `# Contract - step 99.6` (em-dash) was ACCEPTED by the census and
REFUSED by the hook -- proven behaviourally by the cycle-2 Q/A, not inferred.
Two copies of a grammar kept in step by convention is precisely the failure
mode this whole step exists to remove, reproduced inside the fix for it.
Cycle 3 shares the separator as a single named alternation in both files,
**states the drift as a residual RISK rather than denying it is possible**, and
adds the `dash_grammar_parity` check plus mutation cell M7 so a future
divergence is caught by a test instead of by a reader. The drift direction is
fail-closed -- an unrecognised header means "do not copy", which surfaces as
the loud empty-archive failure rather than a silently misattributed archive.

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

**Numbers below are CYCLE-2 values at tree `eceb3a3b`.** The cycle-1 values they
supersede are kept only in section 7's movement table, so this table cannot be
read against a stale figure. Section references are to the REGENERATED live_check.

| # | criterion | evidence | status |
|---|---|---|---|
| 1 | population re-derived, recall validated against the two known positives before use; the 610 unparsed re-classified or explicitly still-unclassified | live_check 4. Recall 2/2, controls 4/4, precision **0.9936 with one live suspect** (`phase-69`) plus its own SUSPECT/CONFIRMED oracle controls. **156 mismatch / 419 agree / 222 unclassified / 24 no-contract over 821 dirs**. Of the former 610: 206 are harness per-cycle contracts (declare no step by design), **16 remain genuinely opaque and are reported as unclassified, not clean**. The figure is a **FLOOR**, and the en/em-dash gap that made cycle-1's 153 a floor is documented in 4a | MET |
| 2 | mechanism demonstrated, not asserted: globs match ZERO files, rolling branch is the only one that fires | live_check 2. Zero for eight independent sids, **with a positive control returning 1** so the zeros are not a broken counter; run under `bash` (the hook's own shell) because `zsh` aborts the loop on `nomatch` | MET |
| 3 | after the fix, archiving a step yields a dir whose contract.md declares THAT step -- driven against a synthetic step in a scratch tree, never against handoff/archive, before and after | live_check 3. BEFORE = pre-fix hook recovered by `git show` and EXECUTED -> declares `82.54`. AFTER -> declares `99.1`. Isolation asserted by the script itself: real hook digest and 821-dir archive list unchanged | MET |
| 4 | the fix does not rely on hook/convention agreement: hook DERIVES the names, **or** fails loudly on nothing-to-archive. A silent copy-nothing or copy-another-step's-files must be a visible failure | **Both branches implemented, not one.** Derivation (section 2); the loud-failure path via check `loud_on_empty` + cell M4; **and the "copies another step's files" half via `no_alien_files` + cell M5**, which the cycle-1 fixture could not express at all (section 7 F1) | MET |
| 5 | state explicitly whether the 89 wrong dirs are backfilled; if backfilled, show the mapping came from git history not guesswork | live_check 5. **NOT backfilled, stated plainly**, with reasons. The number is **156 at this tree**, not 89 and not cycle-1's 153 -- it moves with every closure and with the grammar, so it is always quoted with both | MET |
| 6 | mutation-test: revert the fix and show the archiving test goes red | live_check 3. **6 cells, 6 KILLED** (M5/M6 added at cycle 2). Each asserts its anchor exists before applying and refuses to score a no-op replace | MET |


### 4d. The 16 "genuinely opaque" dirs, ADJUDICATED -- and the floor is 158, not 156

**The cycle-3 Q/A found that I had shipped a bucket labelled "needs a human read"
while a human read of two of its members was already sitting in my own critique
file.** The cycle-2 verdict named `phase-3.2` and `phase-60` as real uncounted
mismatches; neither dir, nor the fact that any of the 16 were adjudicated,
appeared anywhere in these artifacts. That is a disclosure failure, and the
correct repair is not to add the two names -- it is to read all sixteen. Done:

| dir | first heading | verdict |
|---|---|---|
| `phase-3.2` | `# Phase 3.2.1 Contract: Agentic Coordination Loop ...` | **MISMATCH** -- holds 3.2.1 |
| `phase-60` | `# Contract -- 60.4 Observability + ops residuals ...` | **MISMATCH** -- holds 60.4 |
| `phase-2.9`, `phase-2.12`, `phase-3.0`, `phase-3.1`, `phase-3.2.1`, `phase-3.3`, `phase-3.3.1`, `phase-4.0` | `# Phase <sid> Contract:` / `# Phase <sid> PLAN:` / `# Phase <sid>:` | AGREE -- the dir is correct; only the header shape is unparsed |
| `phase-60.1`, `phase-60.2`, `phase-60.3`, `phase-60.4` | `# Contract -- <sid> ...` | AGREE -- same |
| `phase-29.8`, `phase-29.9` | `# phase-housekeeping -- Batch residual closures per closure_roadmap.md` | **GENUINELY UNDETERMINABLE** -- declares no step id at all |

**So the corrected picture is: 2 mismatches + 12 agree + 2 undeterminable.** The
published mismatch total of **156 is a FLOOR understated by exactly 2 named
members; the true figure at this tree is 158**, and only two directories in the
entire archive remain genuinely unclassifiable.

**Two header shapes defeat the grammar** and both are visible above:
`# Phase <sid> Contract:` (the word "Phase", capitalised, no separator before the
sid) and `# Contract -- <sid> ...` (a separator, but no `step`/`phase-` token
before the sid).

**I am NOT widening the grammar again in this cycle, and that is a decision, not
an omission.** Each of the two previous widenings fixed a real gap and introduced
a new defect in the same motion -- cycle 2's produced a dead-code branch, and
cycle 3's produced a probe that reported the hook broken when it was not. A third
widening on the same evidence, in the same cycle as the disclosure fix, is how
that pattern continues. The adjudication above gives a reader the true number
today; widening the grammar so the census derives it automatically is filed as
follow-up work, and until then the artifact states the floor and names what it
misses rather than implying completeness.

---

## 4. What I did NOT do, and what remains open

- **Not backfilled** the mismatched dirs -- **156 at tree `eceb3a3b`**, not the
  cycle-1 figure of 153 (criterion 5, decision stated above).
- **Not touched** `phase-phase-*` directory naming. That is 86.19's Class A root
  cause -- the same file interpolating a raw `$sid` that already carries the
  `phase-` prefix -- and 4 such dirs exist. Same file, different defect; the
  contract names it out of scope and I did not absorb it silently.
- **Not changed** the legacy front-hyphen glob branch. It matches zero files
  today and will continue to; changing it adds risk with no benefit, and its
  `${sid}` vs `${short_sid}` inconsistency is part of the 86.19 defect above.
- **16 archive dirs remain genuinely unclassified** (was 49 before the cycle-2
  grammar fix). Not clean -- unclassified.
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
python scripts/qa/derive_archive_misattribution_86_29.py                                         # recall 2/2, controls 4/4, precision 0.9936 (1 suspect)
```

The immutable command is a **syntax check only**. It proves criterion 2 and
nothing else. The contract says so and it is repeated here so no future reader
mistakes a green command for a green step.

---

## 6. The population moved DURING the step -- both measurements, with their trees

Criterion 1's number is not a constant. It grows by one every time a step closes
under the old hook and stays flat when one closes under the new hook. Both
measurements are recorded rather than the flattering one:

| when | grammar | dirs | mismatch | agree | unclassified | no_contract |
|---|---|---|---|---|---|---|
| before the peer's 86.31 flip (tree `f2eff942`) | ASCII `--` only | 818 | 153 | 386 | 255 | 24 |
| after it (same tree, archive grew) | ASCII `--` only | 819 | **153** | **387** | 255 | 24 |
| cycle 2, after 86.25 + 86.34 closed and the grammar was fixed (tree `eceb3a3b`) | `--` or en/em-dash | 821 | **156** | **419** | 222 | 24 |

**Two different things moved and they must not be conflated.** Rows 1->2 are the
archive GROWING by a real closure. Rows 2->3 mix a further two closures with a
GRAMMAR FIX that reclassified 33 dirs -- so the jump from 153 to 156 is not three
new bad dirs, it is the census finally seeing members it had been blind to
(section 7 F3). Reading that delta as "three more got broken" would be wrong.

The per-closure delta is exactly one directory, and under the new hook it lands
in `agree` every time:

```
phase-86.31 -> ('agree', '86.31')
phase-86.25 -> ('agree', '86.25')
phase-86.34 -> ('agree', '86.34')
```

**That is the whole step in one line.** Under the old hook, closing a step added
one to `mismatch`. Under the new hook, closing a step adds one to `agree` --
three consecutive times now, on the live system. The historical mismatches are
unchanged because they are not backfilled (criterion 5), and the population has
stopped growing in the wrong direction.

Any future reader re-running the census will get a different total. That is
correct and expected; the tree must be named next to the number, which is why
both rows above carry theirs.

---

## 7. CYCLE 2 -- five findings from a dropped Q/A, all confirmed and remediated

The cycle-1 Q/A dropped without returning a verdict. Its write-first record
survived and named five findings. **A recovered record is evidence, never a
verdict** -- so none of its conclusions were adopted, and each finding was
re-measured by Main before any code changed. All five reproduced.

### F1 -- the fixture could not express criterion 4's own failure class

`make_scratch` put ONLY the step-under-test's files in `handoff/current/`. The
real directory holds 400-500 files belonging to ~200 steps. A mutant widening the
variant glob from `${base}_${short_sid}_*.md` to `${base}_*.md` therefore had
nothing to sweep up and **SURVIVED** the author's suite.

**Fixed**: `make_scratch(..., alien=True)` now seeds three other steps' artifacts,
a new check `no_alien_files` asserts the archive contains nothing belonging to
another step, and cell **M5** performs exactly that widening. It now copies **15**
alien files into `phase-99.1/` and dies. This is verbatim the criterion-4 failure
-- *"copies another step's files ... must be a visible failure"*.

**The general lesson, recorded because it is the reusable part**: a fixture that
cannot CONTAIN the defect cannot TEST for it. The suite was green because the
world it built was too small to be wrong in the relevant way.

### F2 -- the no-declaration fall-through had zero coverage

Every fixture rolling file declared *some* step, which exercises the `!=`
comparison but never the **no-pattern-matched** path -- while the hook's own
comment calls that "unsure means do not copy" asymmetry *the whole fix*.

**Fixed**: new check `undeclared_rolling_refused` builds rolling files with no
declaration at all, and cell **M6** flips the fall-through to success. It copies
all four undeclared files and dies.

### F3 -- the census grammar was ASCII-only, and it was hiding real members

`_DECLARE` hard-coded a `--` separator, so `# Contract — Step 76.9.2` (em-dash)
matched nothing and fell into "unclassified". **Measured: 38 of 255 unclassified
dirs carry an en/em-dash heading and 7 of them are genuine mismatches the census
was not counting.** The cycle-1 figure of 153 was a FLOOR, not a count.

**Fixed**: `_DASH = r"(?:--|—|–)"`. After the fix `phase-76.9.2` appears in the
top-8 with 6 dirs -- exactly six of the seven above.

Also conceded: the precision oracle **shares the classifier's grammar**,
differing only in aggregation. It detects "right pattern, wrong order" and is
blind to "the grammar does not recognise this header", which is this exact class.
That is a real independence limit and it is now stated in the live_check rather
than defended. Precision consequently reads **0.9936 with one live suspect**
instead of a suspiciously perfect 1.0000.

### F4 -- a printed sentence overstated its own result

The census printed, and the cycle-1 live_check reproduced: *"no mismatched dir
mentions its own step id anywhere in its contract head."* **False.** Measured
**47 of 153 under the cycle-1 ASCII-only grammar, and 43 of 156 under the
current one** -- the figure moves with the grammar, so it is never quoted
without it. Example: `phase-10.5.0/contract.md` heading `step: phase-10.5-batch (covers
10.5.0, 10.5.1, ...)`. The tabular line above it stated the correct narrower
property; the summary claimed the broader one.

**Fixed at cycle 2, AND THE FIX WAS DEAD CODE -- corrected at cycle 3.** The
cycle-2 remediation printed the mention-vs-declare numbers from inside an
`if not suspect:` branch, and the cycle-2 grammar fix had itself produced a
suspect (`phase-69`), so the corrected output **never printed once** while this
document asserted "the code now prints both numbers". A remediation gated
behind a condition its own tree falsifies is not a remediation. It is
unconditional now, and the figure re-derives to **43 of 156** under the cycle-2
grammar -- the "47 of 153" quoted below was measured under the old one.

### F5 -- a figure with no command next to it

Section B printed "456 suffix-convention files" with no `$ command` line, and it
did not reproduce under four rules a reader tried. **Re-derived**: my original
command yields 462 today (the tree grew by six during this step); 456 was every
`*.md` in `handoff/current`, which is a different set from the one the sentence
named.

**Fixed**: the live_check now prints three explicitly-labelled rules with their
counts and states that the rule is what is stable, not the number.

### Numbers that MOVED as a result

| figure | cycle 1 | cycle 2 | why |
|---|---|---|---|
| archive dirs | 819 | 821 | 86.25 and 86.34 closed |
| mismatch | 153 | **156** | grammar fix found more; still a FLOOR |
| agree | 387 | 419 | 33 dashy dirs reclassified |
| unclassified | 255 | 222 | same |
| genuinely opaque | 49 | **16** | same |
| precision | 1.0000 | **0.9936** | oracle now disagrees on live data |
| behavioural checks | 3 | **5** | F1, F2 |
| mutation cells | 4 | **6** | M5, M6 |

**No number in this table is a constant.** Each is quoted with its tree, because
the population grows every time a step closes.
