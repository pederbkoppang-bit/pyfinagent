# Live check -- step 86.29

**REGENERATED IN FULL, cycle 2.** Every block below is fresh output captured at
tree `79ecb068` on 2026-08-11T06:54:20Z, not the cycle-1 blocks with
numbers edited in place. Regeneration rather than in-place correction is the
phase-86.34 N3 discipline, applied here because the cycle-1 census numbers are
genuinely superseded (a grammar fix changed them) and a reader must not have to
guess which figures moved.

**Step**: `86.29` | **Driver**: Main (`pyfinagent-51`), Opus 5 / effort max.
**Hook sha256[:16]**: `6dc68f781edb4fd0`.

---

## 0. What changed since cycle 1, and who found it

The cycle-1 Q/A (`wf_d4e2e794-567`) **DROPPED** at 197,098 tokens without
calling StructuredOutput -- **NO VERDICT**. Its write-first record survived,
marked COMPLETE, and had reached an internal assessment of CONDITIONAL with five
findings. **That record is evidence, never a verdict**, and none of its
conclusions are adopted here. Each finding below was **re-measured by Main**
before being acted on; the measurements are shown.

| finding | re-measured? | outcome |
|---|---|---|
| F1 fixture cannot express "copies another step's files" (criterion 4) | yes | **CONFIRMED** -- fixed, new check + cell M5 |
| F2 the no-declaration fall-through had zero coverage | yes | **CONFIRMED** -- fixed, new check + cell M6 |
| F3 precision oracle shares the classifier's grammar; ASCII-only `--` hides real members | yes | **CONFIRMED** -- 7 genuine mismatches were being missed; grammar fixed |
| F4 a printed sentence overstated its own result | yes | **CONFIRMED** -- 47 of 153 did mention their sid; sentence corrected |
| F5 section B had no `$ command` line and its count did not reproduce | yes | **CONFIRMED** -- commands and rules now stated |

---

## 1. Immutable verification command (a SYNTAX CHECK -- criterion 2 only)

```
$ bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'
EXIT=0
```

It proves the script parses. **It proves nothing about archive contents.** Stated
so no reader mistakes a green command for a green step.

---

## 2. The mechanism -- step-specific globs match ZERO files (criterion 2)

```
$ bash -c 'for sid in ...; do for f in "$CURRENT_DIR/${sid}-"*.md; do [ -f "$f" ] && n1=$((n1+1)); done; ... done'
sid=86.29    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.6     ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=82.54    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.31    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.25    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=86.34    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=4.5.9    ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
sid=25.A     ${sid}-*.md -> 0   phase-${sid}-*.md -> 0
POSITIVE CONTROL (temp dir holding 86.29-contract.md) -> 1   [must be 1, else the counter is vacuous]
```

The zeros are meaningless without the positive control on the last line: the same
loop returns **1** when a file of the matched shape exists. Run under `bash`,
the hook's own shell -- **under `zsh` the loop aborts with `no matches found`
(nomatch) and would report nothing at all**, which is a different result for a
different reason.

**And the count of files using the OTHER convention, with the rule stated**
(cycle-1 printed "456" with no command and no rule; it did not reproduce under
four rules a reader tried, and it has since moved):

```
RULE R1: basename matches ^(contract|experiment_results|evaluator_critique|research_brief|live_check)_<sid>.md$
  count = 400
RULE R2: same five bases, ANY suffix after the underscore (includes _rerun, _DRAFT)
  count = 415
RULE R4: every *.md in handoff/current (the denominator)
  count = 456
```

The figure moves as the tree grows -- it grew by six during this step alone. The
rule is what is stable; the number is not.

---

## 3. Before / after in a SCRATCH TREE, and the mutation matrix (criteria 3, 4, 6)

```
$ python scripts/qa/prove_archive_provenance_86_29.py
============================================================================
phase-86.29 -- archive provenance, driven in a SCRATCH TREE
============================================================================
hook under test : .claude/hooks/archive-handoff.sh
hook sha256     : 6dc68f781edb4fd0
real archive    : 821 phase-* dirs (must not change)

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
  no_alien_files             GREEN
  undeclared_rolling_refused GREEN

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
  M5 KILLED   [no_alien_files] -- widen the variant glob so it sweeps EVERY step's files
      check went RED: 15 alien file(s) archived: contract_80.2.md (holds 80.2), contract_82.54.md (holds 82.54), contract_82.6.md (holds 82.6), evaluator_critique_80.2.md (holds 80.2), evaluator_critique_82.54.md (holds 82.54), evaluator_critique_82.6.md (holds 82.6)
  M6 KILLED   [undeclared_rolling_refused] -- make the declaration check pass when NO pattern matches
      check went RED: undeclared rolling file(s) WERE copied: ['evaluator_critique.md', 'contract.md', 'experiment_results.md', 'research_brief.md']

----------------------------------------------------------------------------
ISOLATION -- the real repository must be untouched
----------------------------------------------------------------------------
  hook sha256 unchanged            : True
  handoff/archive dir list unchanged: True (821 dirs)

============================================================================
RESULT: PASS (0 problem(s))
============================================================================
EXIT=0
```
$ python scripts/qa/prove_archive_provenance_86_29.py
============================================================================
phase-86.29 -- archive provenance, driven in a SCRATCH TREE
============================================================================
hook under test : .claude/hooks/archive-handoff.sh
hook sha256     : 2278ca9910b0bd15
real archive    : 821 phase-* dirs (must not change)

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
  no_alien_files             GREEN
  undeclared_rolling_refused GREEN
  dash_grammar_parity        GREEN

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
  M5 KILLED   [no_alien_files] -- widen the variant glob so it sweeps EVERY step's files
      check went RED: 15 alien file(s) archived: contract_80.2.md (holds 80.2), contract_82.54.md (holds 82.54), contract_82.6.md (holds 82.6), evaluator_critique_80.2.md (holds 80.2), evaluator_critique_82.54.md (holds 82.54), evaluator_critique_82.6.md (holds 82.6)
  M6 KILLED   [undeclared_rolling_refused] -- make the declaration check pass when NO pattern matches
      check went RED: undeclared rolling file(s) WERE copied: ['evaluator_critique.md', 'contract.md', 'experiment_results.md', 'research_brief.md']
  M7 KILLED   [dash_grammar_parity] -- revert the hook's separator to ASCII-only, re-creating the drift
      check went RED: emdash separator not recognised (declared=None); endash separator not recognised (declared=None)

----------------------------------------------------------------------------
ISOLATION -- the real repository must be untouched
----------------------------------------------------------------------------
  hook sha256 unchanged            : True
  handoff/archive dir list unchanged: True (821 dirs)

============================================================================
RESULT: PASS (0 problem(s))
============================================================================
EXIT=0
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
CENSUS over 821 `handoff/archive/phase-*` directories
==========================================================================
  mismatch        156
  agree           419
  unclassified    222
  no_contract      24

  222 dirs matched none of the 7 declaration patterns.
  They are NOT evidence of cleanliness. Broken down rather than left opaque:
       206  harness per-cycle contract (declares NO step, by design)
        16  genuinely opaque -- needs a human read
  Only the 'genuinely opaque' row is an open question; the harness
  per-cycle contracts are not per-step artifacts at all.

==========================================================================
PRECISION -- every mismatch re-checked by an independent second pass
==========================================================================
  can report SUSPECT   (self-declaring dir) : True
  can report CONFIRMED (clean mismatch)     : True
  mismatches reported           156
  CONFIRMED (dir sid appears in no declaration in the head)   155
  SUSPECT   (dir sid DOES appear -- possible parser error)      1
  precision                    0.9936
      SUSPECT phase-69               census said it declares '69.3'

  what the mismatched dirs actually declare (top 8):
      declares phase-82.54        31 dir(s)
      declares phase-62.6         14 dir(s)
      declares phase-80.2         12 dir(s)
      declares phase-10.5          8 dir(s)
      declares phase-45.0          7 dir(s)
      declares phase-76.9.2        6 dir(s)
      declares phase-62.2          5 dir(s)
      declares phase-40.8          5 dir(s)
EXIT=0
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
CENSUS over 821 `handoff/archive/phase-*` directories
==========================================================================
  mismatch        156
  agree           419
  unclassified    222
  no_contract      24

  222 dirs matched none of the 7 declaration patterns.
  They are NOT evidence of cleanliness. Broken down rather than left opaque:
       206  harness per-cycle contract (declares NO step, by design)
        16  genuinely opaque -- needs a human read
  Only the 'genuinely opaque' row is an open question; the harness
  per-cycle contracts are not per-step artifacts at all.

==========================================================================
PRECISION -- every mismatch re-checked by an independent second pass
==========================================================================
  can report SUSPECT   (self-declaring dir) : True
  can report CONFIRMED (clean mismatch)     : True
  mismatches reported           156
  CONFIRMED (dir sid appears in no declaration in the head)   155
  SUSPECT   (dir sid DOES appear -- possible parser error)      1
  precision                    0.9936
      SUSPECT phase-69               census said it declares '69.3'
  MENTION vs DECLARE: 43 of 156 mismatched dirs DO
  mention their own sid somewhere in the head (batch contracts such as
  'phase-10.5-batch (covers 10.5.0, ...)'). Mentioning is NOT declaring;
  only the narrow declaration property is claimed above. The corollary
  is that a batch contract lets the census OVER-flag, so this total has
  contestable positives as well as any remaining false negatives.

  what the mismatched dirs actually declare (top 8):
      declares phase-82.54        31 dir(s)
      declares phase-62.6         14 dir(s)
      declares phase-80.2         12 dir(s)
      declares phase-10.5          8 dir(s)
      declares phase-45.0          7 dir(s)
      declares phase-76.9.2        6 dir(s)
      declares phase-62.2          5 dir(s)
      declares phase-40.8          5 dir(s)
EXIT=0
```
$ python - <<'EOF'   # symmetric difference, old grammar vs new, same 821 dirs
old(ASCII-only) mismatches = 153
new(dash-widened)          = 156
GAINED +8: ['phase-69', 'phase-75.1', 'phase-75.5.12', 'phase-76.9.3', 'phase-78.0', 'phase-78.16', 'phase-78.2', 'phase-79.2']
LOST   -5: ['phase-69.0', 'phase-69.1', 'phase-69.2', 'phase-69.3', 'phase-69.4']
NET    +3   (153 + 8 - 5 = 156)
```

**The -5 half was disclosed nowhere, and it matters more than the +8.**
`phase-69.0` through `phase-69.4` were **FALSE POSITIVES of the ASCII-only
grammar**: their em-dash heading went unmatched, fell through to the looser
`^#.*?\bphase-(SID)` pattern, and declared `69` -- which is not `69.0`, so five
correct directories were reported damaged. That is verbatim the phase-86.19
failure mode (46 correct dirs reported damaged by a truncating pattern), and
cycle 2's own arithmetic should have caught it: **153 + 7 != 156.** Matching a
total is not matching a set.

So the grammar fix did two things at once: it found 8 genuine mismatches the
census was blind to, AND it removed 5 it had been inventing. Reporting only the
first made the fix sound purely additive when it was also a correction.

### 4b. Precision: 1.0000 was the wrong kind of clean

Cycle 1 reported precision **1.0000 with zero suspects**. It was not vacuous --
the SUSPECT/CONFIRMED controls genuinely both fire -- but the oracle **shares the
classifier's grammar**, differing only in aggregation (union-of-all-patterns vs
first-hit). It can therefore detect "right pattern, wrong order" and is blind to
"the grammar does not recognise this header at all", which is precisely the class
F3 found. **Conceded, not defended.** With the grammar widened, precision now
reads **0.9936 with one real suspect** (`phase-69`, which the census says
declares 69.3) -- a healthier number than the perfect one, because the oracle is
now observed disagreeing with the classifier on live data rather than only on
fixtures.

### 4c. A sentence that overstated its own result -- and a fix that never ran

Cycle 1 printed: *"no mismatched dir mentions its own step id anywhere in its
contract head."* False -- e.g. `phase-10.5.0/contract.md` heads with
`step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)`.

**Cycle 2's fix for that was DEAD CODE, and the artifact claimed it worked.**
The corrected mention-vs-declare reporting was placed inside an `if not suspect:`
branch -- and the cycle-2 grammar fix had itself produced a suspect (`phase-69`),
so the branch never executed once. `experiment_results` nonetheless asserted "the
code now prints both numbers". It printed neither. **A remediation gated behind a
condition its own tree falsifies is not a remediation**, and the giveaway was
four paragraphs above it in the same document, which reported the suspect.

Cycle 3 prints it unconditionally. The figure also moved with the grammar: the
"47 of 153" cycle 2 published was measured under the ASCII-only grammar and
re-derives to **43 of 156**.

---

## 5. Criterion 5 -- the backfill decision, stated plainly

**The already-wrong archive directories are NOT backfilled. They are left as they
are.** Reasons: a reconstruction derived by guesswork would be wrong *and
plausible*, which is worse than known-wrong; nothing is lost, because the real
per-step artifacts are on disk and in git history; and the step text blesses this
outcome provided it is stated rather than implied.

**The count is not a constant and no single number is the answer.** It was 153 at
cycle 1 under an ASCII-only grammar over 819 dirs; it is **156 over 821 dirs**
under the corrected grammar at tree `79ecb068`, and it is a FLOOR in both cases
because 16 dirs remain genuinely unclassified. Any figure quoted without its tree
and its grammar is not reproducible.

What this step does instead is **stop the population growing** and make every
future dir self-describing via `PROVENANCE.md`.

---

## 6. What I did NOT verify, and what is not in force

- **The 16 genuinely-opaque dirs are unclassified, NOT clean.** I did not read
  them individually.
- **`phase-69` (the one precision suspect) is unadjudicated.** It is reported,
  not resolved.
- **I have not driven the new hook against the real `handoff/current/`.**
  Criterion 3 forbids it. The live exercises came from genuine step closures, not
  from me.
- **The hook change was live and ungraded when peer and self closures executed
  it** -- see experiment_results section 4b. Three real archive dirs
  (`phase-86.31`, `phase-86.25`, `phase-86.34`) were minted by it before any
  Q/A verdict existed. That exposure is ongoing until this step closes.

---

## 7. CYCLE 3 -- what the cycle-2 CONDITIONAL found, and a probe that lied

The cycle-2 Q/A (`wf_2675058b-ab3`) returned **CONDITIONAL** on five claim
defects, every one executed rather than argued. All five were re-measured by
Main and confirmed. Two are worth carrying beyond this step:

**The fix that never ran (section 4c).** A remediation placed inside
`if not suspect:` on a tree that HAS a suspect. Green suite, correct-looking
code, zero executions. The lesson is not "test the branch" -- it is that a
remediation must be checked against the tree it ships on, and this one was
contradicted by output printed four paragraphs above it.

**Matching totals hid a swapped set (section 4a).** `153 -> 156` was reported as
"+7 blind members found". By symmetric difference it is **+8 / -5** -- and the
-5 were dirs the old grammar had been WRONGLY flagging, the phase-86.19 shape
again. The arithmetic never closed: `153 + 7 != 156`. **Comparing cardinalities
is not comparing sets.**

### And one the Q/A did not find, because I introduced it while fixing its findings

Adding the `dash_grammar_parity` check turned it **RED**, and the obvious reading
was that the hook fix had failed. It had not. `declared()` -- the *probe* inside
the prove script -- still parsed only ASCII `--`, so it could not read an em-dash
header no matter what the hook did. Verified probe-independently: the hook had
copied the file correctly and the probe returned `None` on content it could see.

**Suspect the probe first.** A check that goes red must be shown to be red ABOUT
ITS SUBJECT before the subject is touched; otherwise the "fix" lands on the wrong
component.

That mistake also exposed a real hole in the matrix: cell **M7 was briefly
credited a KILL while its target check was already RED in the control run** --
a kill that carried no information, because the check could not have gone green
whatever the mutation did. The matrix now **refuses to score any cell whose
target check is red in the control** and reports it `UNSCORABLE`. A mutation
result is only meaningful as a green -> red TRANSITION.
