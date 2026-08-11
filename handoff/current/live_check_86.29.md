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

**Five behavioural checks, six mutation cells, all killed.** Two of each are new
in cycle 2 and they close the gaps F1/F2 named:

- **`no_alien_files` + M5.** `handoff/current/` really holds 400-500 files from
  ~200 steps. The cycle-1 fixture held one step's files, so a mutant widening the
  variant glob to `${base}_*.md` had nothing to sweep up and **SURVIVED**. With
  alien files present the same mutant copies **15** of them into `phase-99.1/`
  and dies. That is verbatim the failure criterion 4 names -- *"copies another
  step's files ... must be a visible failure"*. **A fixture that cannot contain
  the defect cannot test for it.**
- **`undeclared_rolling_refused` + M6.** Every cycle-1 fixture gave the rolling
  files a declaration for some *other* step, which exercises the `!=`
  comparison but never the **no-pattern-matched** path -- while the hook's own
  comment calls that asymmetry "the whole fix". M6 flips that fall-through to
  success and the new check catches all four undeclared files being copied.

**A (BEFORE) is executed, not argued.** The pre-fix hook is recovered with
`git show c806cad6:.claude/hooks/archive-handoff.sh` and run on the same
fixture; it archives `phase-99.1/contract.md` declaring **82.54**. The script
refuses to score this half if the recovered text already contains the fix.

**Every mutation asserts its anchor exists before applying** and refuses to score
a no-op replace -- a `str.replace` that matches nothing looks exactly like a
successful mutation and would otherwise score a survivor as a kill.

**ISOLATION** -- the real hook digest and the real archive directory list are
unchanged by the run, asserted by the script itself at exit.

---

## 4. Population re-derived, recall- AND precision-gated (criterion 1)

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

### 4a. The grammar gap F3 found, and what it was hiding

The declaration patterns hard-coded an ASCII `--` separator. A header written
`# Contract — Step 76.9.2` (EM-DASH) matched nothing and fell into
"unclassified". **That was not cosmetic.** Measured before the fix: 38 of the 255
unclassified dirs carry an en/em-dash heading, and **7 of them are genuine
mismatches the census was not counting**:

```
phase-75.1      actually declares 75.2
phase-75.5.12   actually declares 76.9.2
phase-76.9.3    actually declares 76.9.2
phase-78.0      actually declares 76.9.2
phase-78.16     actually declares 76.9.2
phase-78.2      actually declares 76.9.2
phase-79.2      actually declares 76.9.2
```

So the cycle-1 figure of 153 was a **FLOOR, not a count** -- a one-character
grammar gap concealing real members of the very population the criterion asks to
be re-derived. After the fix `phase-76.9.2` appears in the top-8 with 6 dirs,
which is exactly those six.

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

### 4c. A sentence that overstated its own result (F4)

Cycle 1 printed, and the cycle-1 live_check reproduced verbatim: *"no mismatched
dir mentions its own step id anywhere in its contract head."* **That is false.**
Measured: **47 of 153 did** -- for example
`handoff/archive/phase-10.5.0/contract.md` heads with
`step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)`. The tabular line
immediately above it stated the correct, narrower property ("appears in no
DECLARATION in the head"), and the narrow property is the one that supports the
conclusion -- but the summary sentence claimed the broad one. The code now prints
both numbers so the distinction cannot be glossed again, and notes the corollary:
a batch contract means the census can also **over-flag**, so the total has
contestable positives as well as the known false negatives.

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
