# Live check -- step 86.29

**Step**: `86.29` -- the per-step handoff archive has been snapshotting the wrong step's files since 2026-08-06.
**Captured**: 2026-08-11T06:35:05Z by Main (`pyfinagent-51`), Opus 5 / effort max.
**Tree at capture**: `bc7c4f01`. **Hook sha256[:16]**: `6dc68f781edb4fd0`.

Every block below is the verbatim stdout/stderr of the command named above it.
Nothing is transcribed by hand.

---

## A. Immutable verification command (criterion 2 only -- a SYNTAX CHECK)

```
$ bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'
EXIT=0
```

**Stated so no reader mistakes a green command for a green step**: this proves
the script parses. It proves nothing about archive contents. The real evidence
is sections B-E.

---

## B. The mechanism -- the step-specific globs match ZERO files (criterion 2)

```
sid=86.29  ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.6  ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=82.54  ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.31  ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.26  ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
POSITIVE CONTROL (dir containing 86.29-contract.md) -> 1   [must be 1]
suffix-convention files in handoff/current: 456
```

The zero counts are meaningless without the positive control on the second-last
line: the same counting loop returns **1** when a file of the matched shape
exists, so the zeros are a property of the tree, not of a broken counter. The
globs expect the sid at the FRONT before a hyphen (`4.5.9-contract.md`); the
convention since ~phase-4.9 puts it at the END after an underscore
(`contract_86.29.md`), and 456 files in `handoff/current/` use that form.
So `moved=0` always, and before this change the rolling branch was the only
branch that ever fired.

---

## C. Before / after, driven in a SCRATCH TREE (criteria 3, 4, 6)

Criterion 3 requires the demonstration to run against a synthetic step in a
scratch tree, **never against `handoff/archive`**. Every fixture below lives
under `tempfile.mkdtemp()`; the script asserts at exit that the real hook's
digest and the real archive's directory list are unchanged.

```
$ python scripts/qa/prove_archive_provenance_86_29.py
============================================================================
phase-86.29 -- archive provenance, driven in a SCRATCH TREE
============================================================================
hook under test : .claude/hooks/archive-handoff.sh
hook sha256     : 6dc68f781edb4fd0
real archive    : 818 phase-* dirs (must not change)

----------------------------------------------------------------------------
A. BEFORE -- the PRE-FIX hook, recovered from git and executed
----------------------------------------------------------------------------
  pre-fix hook, step 99.1 -> archive contract declares: '82.54'
  CONFIRMED: the archive dir named 99.1 holds phase-82.54's contract.

----------------------------------------------------------------------------
B/C. AFTER -- the current hook (control run; all three must be GREEN)
----------------------------------------------------------------------------
  right_step                 GREEN
  no_poison_substitution     GREEN
  loud_on_empty              GREEN

----------------------------------------------------------------------------
D. MUTATION -- a cell is KILLED only if its target check goes RED
----------------------------------------------------------------------------
  M1 KILLED   [right_step] -- revert the derived branch (never build names from the step id)
      check went RED: archive contract declares None, expected '99.1'
  M2 KILLED   [no_poison_substitution] -- revert the rolling guard to an unconditional copy
      check went RED: poisoned rolling contract WAS substituted (declares '82.54')
  M3 KILLED   [no_poison_substitution] -- make the declaration check answer TRUE for every file
      check went RED: poisoned rolling contract WAS substituted (declares '82.54')
  M4 KILLED   [loud_on_empty] -- delete the empty-archive failure branch
      check went RED: no FAILURE line on stderr; no systemMessage on stdout; PROVENANCE.md does not record the failure

----------------------------------------------------------------------------
ISOLATION -- the real repository must be untouched
----------------------------------------------------------------------------
  hook sha256 unchanged            : True
  handoff/archive dir list unchanged: True (818 dirs)

============================================================================
RESULT: PASS (0 problem(s))
============================================================================
prove EXIT=0
```

Reading of that output:

- **A (BEFORE)** is not an argument from source. The pre-fix hook is recovered
  with `git show c806cad6:.claude/hooks/archive-handoff.sh` and EXECUTED on the
  same fixture; it produces `phase-99.1/contract.md` declaring **82.54**. The
  script refuses to score this half if the recovered text already contains the
  fix, which would make the BEFORE vacuous.
- **B/C (AFTER)** three behavioural checks, all GREEN.
- **D (MUTATION)** 4 cells, 4 KILLED. Each mutation asserts its anchor text
  exists **before** applying, and refuses to score if the replace changed
  nothing -- a no-match `str.replace` looks exactly like a successful mutation
  and would otherwise score a survivor as a kill.
- **ISOLATION** the real hook digest and the real 818-dir archive list are
  unchanged by the run.

---

## D. Population re-derived, recall- AND precision-gated (criterion 1)

```
$ python scripts/qa/derive_archive_misattribution_86_29.py
==========================================================================
RECALL VALIDATION -- the census is not printed unless this passes
==========================================================================
  phase-86.6       -> mismatch      declares='82.54'   FLAGGED (correct)
  phase-86.26      -> mismatch      declares='82.54'   FLAGGED (correct)
  recall 2/2 -- proceeding

==========================================================================
CONTROLS -- the method must be able to answer BOTH ways
==========================================================================
  positive         dir=phase-99.7     -> mismatch     (declares '82.54', expected mismatch)   ok
  negative         dir=phase-99.8     -> agree        (declares '99.8', expected agree)   ok
  alnum_sid        dir=phase-25.A     -> agree        (declares '25.A', expected agree)   ok
  alnum_sid_wrong  dir=phase-25.A     -> mismatch     (declares '25', expected mismatch)   ok
  controls 4/4 -- proceeding

==========================================================================
CENSUS over 818 `handoff/archive/phase-*` directories
==========================================================================
  mismatch        153
  agree           386
  unclassified    255
  no_contract      24

  255 dirs matched none of the 7 declaration patterns.
  They are NOT evidence of cleanliness. Broken down rather than left opaque:
       206  harness per-cycle contract (declares NO step, by design)
        49  genuinely opaque -- needs a human read
  Only the 'genuinely opaque' row is an open question; the harness
  per-cycle contracts are not per-step artifacts at all.

==========================================================================
PRECISION -- every mismatch re-checked by an independent second pass
==========================================================================
  can report SUSPECT   (self-declaring dir) : True
  can report CONFIRMED (clean mismatch)     : True
  mismatches reported           153
  CONFIRMED (dir sid appears in no declaration in the head)   153
  SUSPECT   (dir sid DOES appear -- possible parser error)      0
  precision                    1.0000
  no suspects: no mismatched dir mentions its own step id anywhere
  in its contract head, so none of them is the 86.19 truncation shape.

  what the mismatched dirs actually declare (top 8):
      declares phase-82.54        31 dir(s)
      declares phase-62.6         14 dir(s)
      declares phase-80.2         12 dir(s)
      declares phase-10.5          8 dir(s)
      declares phase-45.0          7 dir(s)
      declares phase-69            6 dir(s)
      declares phase-62.2          5 dir(s)
      declares phase-40.8          5 dir(s)
census EXIT=0
```

**The numbers moved and the step text is superseded.** The step text recorded
89 mismatch / 93 agree / 24 no-contract / 610 unparsed over 816 dirs. Measured
at tree `bc7c4f01`: **153 mismatch / 386 agree / 255 unclassified / 24
no-contract over 818 dirs**. The 610 unclassified are re-classified as required:
206 are the harness runner's own per-cycle contracts, which declare no step *by
design* and are not per-step artifacts at all; **49 remain genuinely opaque and
are reported as still-unclassified, not as clean.**

Three gates run before the census is believed, and each can refuse to print it:

1. **Recall 2/2** against the two known positives. A method that reports either
   clean is rejected, not adjusted.
2. **Controls 4/4** -- recall alone is satisfied by a classifier that answers
   "mismatch" always, so the method is forced to produce BOTH answers. Two cells
   exercise the exact 86.19 shape: an alphanumeric segment (`25.A`) that an
   earlier `[0-9]+` pattern truncated to `25`, turning 46 correct dirs into
   reported mismatches.
3. **Precision oracle controls** -- precision came back **1.0000 with zero
   suspects**, which is the shape worth distrusting. The oracle is therefore
   driven against a fixture engineered to be SUSPECT (declares another step
   first, its own step later in the head) and one engineered to be CONFIRMED.
   Both answers are produced, so the 1.0000 is a measurement rather than a
   constant. If the suspect fixture ever confirms, the figure is WITHHELD.

This answers the research gate's standing critique of this script, carried
forward in the contract rather than smoothed over: *"precision is unmeasured and
its 2 known positives are one instance."* Precision is now measured, controlled,
and reported next to recall.

---

## E. Criterion 5 -- the backfill decision, stated plainly

**The 153 already-wrong archive directories are NOT backfilled. They are left
as they are.**

Reasons, in the order they matter:

1. The correct content for each dir would have to be reconstructed from git
   history (the `handoff/current/*_<sid>.md` blob at the commit that flipped
   `<sid>` to done). A reconstruction that is wrong is strictly worse than a
   dir that is known-wrong, because it is wrong *and plausible*.
2. Nothing is lost. The real per-step artifacts are on disk in
   `handoff/current/` and in git history. This is an audit-trail defect, not a
   data-loss defect.
3. The step text blesses this outcome explicitly, provided it is stated rather
   than implied: *"Leaving them wrong is an acceptable, stated outcome; leaving
   them wrong while implying they are fixed is not."*

**What this step does instead** is stop the population growing and make every
future dir self-describing: each archive dir now carries `PROVENANCE.md`
recording which source file produced each archived file, and which rolling files
were skipped and why.

---

## F. What I did NOT verify, and what is not in force

- **The change is committed but NOT in force in any already-running process.**
  This hook is invoked fresh by `bash` on every PostToolUse event, so unlike a
  Python module held in `sys.modules` it takes effect on the next masterplan
  Write with no restart. That is a property of hooks, not a claim I measured
  against a running daemon.
- **I have not driven the new hook against the real `handoff/current/`.**
  Criterion 3 forbids it and a live run would create real archive dirs. The
  first real exercise will be the next genuine step closure.
- **The 49 genuinely-opaque dirs are unclassified, not clean.** I did not read
  them individually.
- **A concurrent peer session (`pyfinagent-43`) is live in this repository**
  and owns step 86.31. Every command above is read-only on repository content
  except the three files this step changes; the demonstration ran entirely in
  temp dirs.
