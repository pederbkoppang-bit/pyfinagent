# live_check — phase-86.97

**STATUS: COMPLETE.** *(Sections are CHRONOLOGICAL — §H cycle 2, §I cycle 3, §J cycle 4. Blocks in earlier sections are verbatim output OF THEIR OWN CYCLE; the three that a later cycle changed are marked SUPERSEDED in place. The "verbatim tool output" promise below is per-section, not per-file.)* session, complete and unelided.

---

## A. DEFECT 2 REPRODUCED — the delete-the-call mutant SURVIVES (criterion 1)

Control observed GREEN **first**, inside the same worktree, so the kill/survive
score below is against a working baseline rather than a broken harness:

```
=== control INSIDE the worktree (must be GREEN before any mutation) ===
ALL GREEN: 42 passed, 0 failed

=== MUTANT: delete the production call _log_decision(bump_type) ===
  call DELETED (25 bytes removed); remaining occurrences of '_log_decision(bump_type)': 0
ALL GREEN: 42 passed, 0 failed
  ^ if ALL GREEN, the mutant SURVIVED -- the guard cannot see a deleted call

=== does the hook still even run? (bash -n on the mutant) ===
  mutant still parses -- it is a BUILDABLE mutant, so the kill/survive score is valid
```

The buildability check is not ceremony: a mutant that does not build is
**UNSCORABLE**, and scoring it as a kill is how a guard flatters itself.

*Cycle-2 note on that captured line:* `bash -n` alone turned out to be the wrong
oracle — it does not parse inside a quoted heredoc, which is where this mutation
lands. It happens to give the right answer **here** (a clean deletion does
compile), but the oracle has since been strengthened to `bash -n` **plus** a
`compile()` of the heredoc body. See §H.

### It is worse than "surviving" — it is INVISIBLE

`detector_source()` walks `tree.body` collecting only `FunctionDef` / `Assign` /
`AnnAssign` — **definition** classes that bind a name. A bare
`_log_decision(bump_type)` is an `ast.Expr(Call)` and binds nothing, so it can
never match. **Enlarging `NEEDED` cannot help.** Measured:

```
# measured at 52358053 -- this step's PARENT commit. See the re-derivation below.
extracted SHIPPED, unmutated   : 7,597 B  sha1 f7458a6ab1f5fe96
extracted SHIPPED, call DELETED: 7,597 B  sha1 f7458a6ab1f5fe96
BYTE-IDENTICAL: True   -> the mutant is INVISIBLE to the extraction, not merely surviving
CONTROL (edit inside the def)  : 7,621 B  differs from base: True  (+24 B)
```

The control is what makes this a finding rather than a dead extractor: the
extraction *is* sensitive to changes inside the definition. It is specifically
the **call** it cannot see — so no assertion added to
`verify_changelog_flip_86_91.py`, however clever, could ever kill this mutant.
That is why criterion 4 requires driving the whole heredoc instead.

Literature name for this class: **pseudo-tested** (Vera-Perez et al.;
Niedermayr et al., 291/2041, 14 of 25 inspected being side-effect methods —
`_log_decision`'s exact shape).

---

## B. DEFECT 1 REPRODUCED — the gap, RE-DERIVED, never copied (criterion 1)

The step was filed as **"10 commits vs 5 decision lines"**. That figure is a
stale snapshot and criterion 1 forbids copying it. The checker recomputes at run
time, with the window pinned to the decision log's **own first timestamp** — a
bare date would slide with the clock:

```
       RE-DERIVED at execution time (window pinned to 2026-08-16T08:23:33Z):
         commits=51  decision lines=26  gap=25
         commits matching the recursion guard=26
  ok   [3] the gap is explained by the recursion guard (criterion 3: a BOUND, not an unexplained loss)
```

Measured three times across the session, and it moves — which is the point:
21:03 → `47 / 24 / 23 / 24`; 21:52 → `51 / 26 / 25 / 26`; 22:12 → `53 / 27 / 26 / 27`
(commits / decision lines / gap / recursion-guard commits). A pinned figure would
already be wrong three times over. The gap tracks the recursion-guard count, not
a mystery — which is why the checker asserts the *relationship*, never a number.

---

## C. ENUMERATION FROM SOURCE, BY A WRITTEN-DOWN RULE (criterion 2)

**The rule**, stated in the checker's own source: *an EXIT PATH is any line,
outside the detector heredoc, on which `exit` appears as a command — matching
`(^|;|&&|\|\|)\s*exit\b`.*

**The self-test**, because a scan that quietly matches nothing looks identical to
a clean bill of health: the rule is cross-checked against a deliberately dumber
scan (every line containing the token `exit`). Every line the dumb scan finds and
the rule does not must be explainable as a comment; anything else fails the gate.

**The soundness preconditions are asserted, not assumed.** The classification is
lexical ("an exit before the detector cannot reach it"), which is only valid
while bash's execution order matches source order:

```
[1] PRECONDITIONS for the lexical rule (criterion 2)

  ok   [1] the hook defines NO bash functions (so lexical order == execution order)
  ok   [1] no trap / source / . / eval (nothing reorders execution)
  ok   [1] exactly ONE heredoc -- the detector
  ok   [1] the heredoc terminator is found exactly once
  ok   [1] the detector region is non-empty and ordered
       detector heredoc: lines 43..387 (terminator 'PYEOF')
```

Result:

```
[2] ENUMERATION of exit paths from source (criterion 2)

  ok   [2] the rule finds a non-zero number of exit paths (a scan that matches nothing is not a clean bill of health)
  ok   [2] every line the dumber scan finds but the rule does not is a comment (so the rule is not under-matching)
  ok   [2] at least one PRE-detector exit path exists (else this step's premise is void)
  ok   [2] at least one POST-detector exit path exists
       3 pre-detector exit path(s), 3 post-detector

       :28  LEGITIMATELY-SILENT    if echo "$MSG" | grep -qiE "^chore: (auto-changelog|changelog dr
       :33  MUST-LOG               if [ ! -f "$CHANGELOG" ]; then
       :37  MUST-LOG               if ! grep -q "### Recent Activity" "$CHANGELOG"; then
  ok   [2] every pre-detector exit path is classified MUST-LOG or LEGITIMATELY-SILENT
       :394  POST-DETECTOR          (decision already written)
       :396  POST-DETECTOR          (decision already written)
       :397  POST-DETECTOR          (decision already written)
```

**The classification is keyed on the guard's CONDITION TEXT, never on a line
number** — line numbers are exactly the fixture that rotted in phase-86.92. This
was demonstrated live rather than argued: my own docstring correction (§E) added
16 lines to the hook, moving the heredoc from `43..371` to `43..387` and the
post-detector exits from `378/380/381` to `394/396/397`. **The enumeration
tracked the move with no edit**, and the classification still matched.

**An unclassified pre-detector exit is a FAILURE, not a default.** If someone adds
a fourth early exit, the gate goes red saying *"UNCLASSIFIED guard ... Classify
it; do not widen the rule."*

---

## D. THE RECURSION GUARD IS A BOUND, JUDGED NOT ASSUMED (criterion 3)

Driven for real, not argued from source:

```
  ok   [3] recursion guard: an auto-changelog commit exits 0
  ok   [3] recursion guard: and writes NO decision line (the BOUND, measured)
```

**Verdict: LEGITIMATELY-SILENT.** A commit this hook created is by construction
not a bump candidate, so there is no decision to explain; logging it would fill
the log with entries about the logger. It accounts for essentially the entire
gap (§B). This is recorded as a **bound to state**, exactly as criterion 3 frames
it — not a defect to fix.

`:33` and `:37` are classified **MUST-LOG**: a missing CHANGELOG or a lost
`### Recent Activity` anchor is silent structural breakage that, from outside,
looks identical to "nothing to do".

---

## E. THE END-TO-END GUARD, AND THE MUTANT IT KILLS (criteria 4, 6)

```
[3] END-TO-END -- drive the REAL hook in a temp git repo (criterion 4)

  ok   [3] the real hook runs to completion in a temp repo
  ok   [3] a decision line is WRITTEN TO THE FILE (the observable effect, not an extracted namespace)
  ok   [3] the decision line carries a reason      <-- SUPERSEDED BY §J1: this assertion was VACUOUS and is DELETED
  ok   [3] ISOLATION after the baseline drive: the real repo's decision log is untouched
  ok   [3] recursion guard: an auto-changelog commit exits 0
  ok   [3] recursion guard: and writes NO decision line (the BOUND, measured)
  ok   [3] ISOLATION after the recursion-guard drive: the real repo's decision log is untouched
  ok   [3] the real decision log exists, so the gap CAN be re-derived

       RE-DERIVED at execution time (window pinned to 2026-08-16T08:23:33Z):
         commits=53  decision lines=27  gap=26
         commits matching the recursion guard=27
  ok   [3] the gap is explained by the recursion guard (criterion 3: a BOUND, not an unexplained loss)

[4] MUTATION -- the guard must SEE a deleted call (criteria 4, 6)

  ok   [4] ORACLE: buildable() says YES to the unmutated hook
  ok   [4] ORACLE: buildable() says NO to a Python SyntaxError INSIDE the heredoc
  ok   [4] CONTROL: the UNMUTATED hook writes a decision line
  ok   [4] delete-the-production-call: the mutant still runs cleanly (rc=0), so an empty log means the GUARD caught it and not that the hook crashed
  ok   [4] delete-the-production-call: KILLED -- removing the call the 86.91 extraction is structurally blind to makes the guard RED
  ok   [4] retarget-the-log-write: the mutant still runs cleanly (rc=0), so an empty log means the GUARD caught it and not that the hook crashed
  ok   [4] retarget-the-log-write: KILLED -- removing the write's destination, with no name left undefined and no exception raised makes the guard RED
  ok   [3] ISOLATION after all mutant drives: the real repo's decision log is untouched

ALL GREEN: 27 passed, 0 failed
```

Immutable command:

```
$ bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'
parses
```

**Disclosed:** a parse check cannot fail on either defect. The evidence is above.

---

## F. THE 86.91 CLAIM, CORRECTED BY REPLACEMENT (criterion 5)

The claim was corrected in **three** places. The enumeration was driven by the
claim's own semantics, not by my wordings — which mattered: my first sweep found
two sites, and searching for the *unbounded assertion form* surfaced a third
(`contract_86.91.md:145`, inside the plan section rather than the criteria copy).

| artifact | before | after |
|---|---|---|
| `.claude/hooks/post-commit-changelog.sh:222` | "Every invocation records WHY" | "Every invocation **THAT REACHES THIS DETECTOR** records WHY", plus the three unreached paths named |
| `handoff/current/experiment_results_86.91.md:161` | "Criterion 4 -- the silent-swallow CLASS, closed" / "Every invocation now appends..." | "...closed **FOR INVOCATIONS THAT REACH THE DETECTOR**" |
| `handoff/current/contract_86.91.md:145` | "Every invocation appends... An unexplained `none` becomes impossible" | "Every invocation **that reaches the detector**... impossible **within the detector**" |

**Why replacement and not a footnote.** `experiment_results_86.91.md` already
*contained* the correct bound — at line 428, 265 lines below the claim it
qualifies. A reader of §5 would never meet it. That is precisely the failure mode
criterion 5 names: *"a correction must replace, not accompany."*

**Deliberately NOT edited:** `contract_86.91.md:105` and
`.claude/masterplan.json`, which carry the **immutable criterion text verbatim**.
Amending a criterion is forbidden; the claim about what was achieved is what
needed fixing. Also not edited: `evaluator_critique_86.91.md` (an evaluator's own
verdict is not mine to rewrite) and the verbatim checker output in
`live_check_86.91.md:167`, whose assertion labels are accurate about the
detector's internal branches.

Sweep for any surviving unbounded form:

```
handoff/current/contract_86.91.md:145:**P2. The decision log** (criterion 4). Every invocation **that reaches the
handoff/current/contract_86.91.md:152:said "Every invocation ... An unexplained `none` becomes impossible", which
handoff/current/experiment_results_86.91.md:164:silent-swallow CLASS, closed" and "Every invocation now appends one structured
handoff/current/experiment_results_86.91.md:170:Every invocation **that reaches this detector** appends one structured line to
.claude/hooks/post-commit-changelog.sh:222:    SCOPE, CORRECTED IN phase-86.97. Every invocation THAT REACHES THIS
.claude/hooks/post-commit-changelog.sh:227:    The earlier wording here was "Every invocation records WHY", and that was an
```

Every hit is either the corrected form naming its bound, or the correction
quoting the wording it replaced. All other repo hits for that phrase are
unrelated subjects (pytest-subprocess, the harness cycle index, Bolt listeners).

---

## G. NO REGRESSION (criterion 7)

```
bash -n .claude/hooks/post-commit-changelog.sh   parses
verify_decision_log_86_97.py                     ALL GREEN: 35 passed, 0 failed   <-- SUPERSEDED BY §J: now 52
verify_changelog_flip_86_91.py                   ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs                ALL GREEN: 96 passed, 0 failed
verify_research_gate_workflow.mjs                ALL GREEN: 124 passed, 0 failed
```

No masterplan step was flipped and no verdict altered by this work. The only
behavioural change to production code is **none**: the hook edit is a docstring.

---

## H. CYCLE-2 REMEDIATION — my buildability oracle had the step's own defect

Cycle-1 verdict `wf_3be25861-bde`: **CONDITIONAL**. Criteria 1, 2, 3, 4, 5 and 7
MET and independently re-executed by the evaluator. Capped on **criterion 6**,
and the finding is the sharpest kind — the guard reproduced, in its own scoring,
the exact blindness it was built to close.

### H1. `buildable()` could not fail for the mutants it gated (WARN)

`buildable()` was `bash -n` alone. **`bash -n` does not parse inside a quoted
heredoc** (`<< 'PYEOF'`), and *both* mutants are Python-side edits inside exactly
that heredoc. I re-measured it myself rather than taking the verdict's word:

```
bash -n on a mutant with a Python SyntaxError inside the heredoc -> rc = 0 (0 means bash -n CANNOT see it)
unmutated heredoc compiles: True
MUTANT heredoc compile -> SyntaxError: '(' was never closed (<heredoc>, line 235)
```

So a crashing mutant produced an empty log, and the scoring rule
`m_log.strip() == ""` recorded it as **KILLED**. Criterion 6's UNSCORABLE arm
existed but its oracle was structurally blind to the only build failures these
cells could have.

**Fixed two ways.** `buildable()` now checks the bash half **and** compiles the
heredoc body; and every cell additionally asserts the mutant's **`rc == 0`**, so
an empty log can only mean the guard caught it, never that the hook crashed. The
oracle is now self-tested **in both directions** — a one-sided control proves
nothing:

```
  ok   [4] ORACLE: buildable() says YES to the unmutated hook
  ok   [4] ORACLE: buildable() says NO to a Python SyntaxError INSIDE the heredoc
```

### H2. The second mutant killed by the WRONG mechanism (WARN)

`neuter-the-log-write` redirected the write to `os.devnull` — but `os` is
imported **zero** times inside the heredoc. Confirmed by driving both:

```
=== OLD mutant (os.devnull) -- the mis-attributed one ===
  rc=0 log_empty=True
  stderr: ["[changelog] decision-log FAILED (NameError: name 'os' is not defined)"]

=== NEW mutant (retarget the filename) -- the intended mechanism ===
  rc=0 log_empty=True
  stderr mentions any error?: False
  -> kills by the STATED mechanism (bytes land elsewhere), no exception path
```

The cell passed for a mechanism nobody intended, and two artifacts repeated the
wrong explanation. It is now `retarget-the-log-write`: every name stays defined,
the write succeeds, nothing raises — the *only* change is that the bytes land
where the guard does not read. That isolates the property under test (the guard
is bound to the FILE, not to call text). The claim text in
`experiment_results_86.97.md` was **replaced**, not annotated.

### H3. Isolation covered only the first drive (NOTE)

The snapshot was taken once and checked once, so the recursion-guard drive and
both mutant drives ran with no isolation assertion — while the artifact claimed
the property broadly. Now asserted after **every** drive:

```
  ok   [3] ISOLATION after the baseline drive: the real repo's decision log is untouched
  ok   [3] ISOLATION after the recursion-guard drive: the real repo's decision log is untouched
  ok   [3] ISOLATION after all mutant drives: the real repo's decision log is untouched
```

### H4. The gap re-derivation could vanish silently (NOTE)

It sat inside `if real_before:`, so a missing decision log would have made the
whole block — including its `check()` — disappear rather than go red. A skipped
check is indistinguishable from a passing one in the summary line. There is now
an explicit assertion that the input exists.

### Net

Assertions **20 → 27**. No criterion reinterpreted; nothing weakened. The two
WARN findings were both real defects in my work, and both were of the class this
step exists to attack.

---

## I. CYCLE-3 REMEDIATION — a figure that expired, and guards with no cells

Cycle-2 verdict `wf_2dd1efc9-d0c`: **CONDITIONAL**. All 7 criteria MET and every
one independently re-executed by the evaluator, capped on two WARNs. Both were
real; both were instances of defect classes this step is about.

### I1. My measured figure was invalidated by my own edit, in the commit that stated it

I wrote `7,597 B / sha1 f7458a6ab1f5fe96` as MEASURED. It was exact — **at the
parent commit**. The criterion-5 docstring correction I made to `_log_decision`
(one of the four names the extractor lifts) rode in **the same commit** that
stated the number, and moved it. Re-derived with the shipped extractor:

```
  52358053: 7,597 B  sha1 f7458a6ab1f5fe96
  WORKTREE: 8,617 B  sha1 072056e58af2befa

AT HEAD, call DELETED: 8,617 B  sha1 072056e58af2befa
BYTE-IDENTICAL at HEAD: True  <- the PROPERTY still holds; only the cycle-1 FIGURE was stale
CONTROL (edit inside the def): 8,641 B  differs: True  (+24 B)
```

**The property is invariant; the byte count is not.** Every one of the six sites
carrying it now names the commit it was measured at — including
`.claude/agent-memory/researcher/project_uncalled_function_86_97.md`, which is
auto-loaded into every future researcher session and was therefore the most
forward-looking consumer of the wrong number.

The lesson generalises past this step: a figure labelled *measured* with no
commit attached is a claim that quietly expires, and it expires most easily when
the thing that invalidates it is your own edit in the same commit.

### I2. Criterion 6 says "every new guard" — sections [1] and [2] had no cells

Cycles 1–2 mutation-tested only the `[3]`/`[4]` guards. The preconditions, the
enumeration recall and the classification keying had **no cell at all**, and
unlike the disclosed `:214` gap this subsetting was not in the scope-honesty list.
The cycle-2 Q/A ran four such mutations itself and found zero survivors — but
zero survivors is not the same as tested.

To make those guards reachable, the `[1]`/`[2]` logic was refactored into a single
`analyse(src)` function that **both** the shipped assertions and the new cells
consume. That detail is load-bearing: a mutation section driving a
re-implementation would be testing a copy, not the guard.

```
[5] MUTATION of the [1]/[2] guards (criterion 6, 'every new guard')

  ok   [5] CONTROL: the real hook satisfies all four [1]/[2] properties
  ok   [5] bash-function-defined: KILLED -- the lexical rule stops being sound and [1] must say so
  ok   [5] trap-reorders-execution: KILLED -- execution order no longer matches source order
  ok   [5] exit-the-RULE-under-matches: KILLED -- the dumber scan sees an exit the rule missed
  ok   [5] unclassified-pre-detector-exit: KILLED -- a fourth early exit nobody has classified
  ok   [5] classification-keys-on-condition-text: KILLED -- rewording the recursion guard's condition makes it UNCLASSIFIED
  ok   [5] isolation-check: KILLED -- a corrupted snapshot makes it report FALSE
  ok   [5] isolation-check CONTROL: the true snapshot still reports TRUE

ALL GREEN: 35 passed, 0 failed   <-- SUPERSEDED BY §J: now 52
```

Note the two paired assertions on the isolation check. Cycle 2's version of this
probe corrupted the snapshot, let the real assertion FAIL, then popped the entry
off `_failures` and decremented the counter by hand — a cell that edits the
scoreboard to run. It is now a pure predicate (`isolation_holds`) probed in both
directions, so nothing is patched and no spurious `FAIL` line is printed.

### I3. Two diagnostics named the wrong leg (NOTE)

The UNSCORABLE message said *"bash -n rejected the mutant"* unconditionally — but
after cycle 2 the rejection usually comes from the `compile()` leg, which is
exactly the leg cycle 2 added. And the KILLED message said *"the mutant STILL
produced a decision line"* even when the real cause was a non-zero rc and the log
was in fact empty. A maintainer debugging a red run was being sent to the wrong
place. Both now name the leg that actually fired, verified in both directions:

```
python-syntax-error-in-heredoc     bash_parses=True  buildable=False -> message would name: the heredoc compile() leg
unclosed-if (REAL bash error):     bash_parses=False buildable=False -> names: bash -n
```

*(My first attempt at the bash-side control was invalid — a missing `]` is still
syntactically valid bash, so `bash -n` accepted it. That is the probe being
wrong, not the code, and it is recorded rather than quietly replaced.)*

### I4. The accompany-form residual in `experiment_results_86.91.md` (NOTE)

Line 186 still read *"An unexplained `none` is no longer expressible *(bounded —
see below)*"* — a pointer, 15 lines inside the very section rewritten for
accompanying rather than replacing. Now stated in place: *"no longer expressible
**BY THE DETECTOR**"*. Pre-existing from 86.91 cycle-3 (`468c7908`), not
introduced here, but my artifact claimed the correction was complete.

### Net

Assertions **27 → 35**. Nothing weakened; no criterion reinterpreted. The
refactor moved section-[1]/[2] logic into a shared function so it could be
mutated, and every new cell kills.


---

## §J. CYCLE-4 — both parked blockers closed (2026-08-17)

Regenerated from live runs in the same pass that wrote this section.

### J1. Criterion 4 — the guard now asserts the DECISION, not the existence of a line

The park note called the `:305` assertion weak. It was **vacuous**: `reason=` is a
literal in the writer's format string (`post-commit-changelog.sh:271`), so
`"reason=" in log_text` holds for every non-empty line the writer can emit and is
strictly subsumed by the "a line was written" check above it.

Four scenarios now drive four of the nine reason states, spanning all four bump
magnitudes, each asserted by exact equality against a table derived from branch
structure **before** anything was driven:

```
       no_flip -- masterplan unchanged
         -> {'bump': 'none', 'reason': 'no_flip', 'created': '-', 'transitioned': '-'}
       flip_transitioned -- 99.1 pending -> done
         -> {'bump': 'patch', 'reason': 'flip_transitioned', 'created': '-', 'transitioned': '99.1'}
       flip_created -- 98.0 appears already done
         -> {'bump': 'minor', 'reason': 'flip_created', 'created': '98.0', 'transitioned': '-'}
       subject_forced_major -- a `!` subject
         -> {'bump': 'major', 'reason': 'subject_forced_major', 'created': '-', 'transitioned': '-'}
  ok   [3a] the scenarios DISCRIMINATE -- they do not all produce one reason
       RE-DERIVED at execution time (window pinned to 2026-08-16T08:23:33Z):
         commits=87  decision lines=44  gap=43
         commits matching the recursion guard=44
```

`unrecorded` is a standing negative control: no branch assigns it, so observing it
means the detector never ran.

### J2. Criterion 5 — the last unbounded carrier is bounded IN PLACE

`live_check_86.91.md:104` read *"every decision now explains itself"*; that file
contained **zero** bounding language (`grep -c` = 0). The heading now reads
*"every decision THAT REACHES THE DETECTOR explains itself"* with the reason
stated in place, not appended.

**FIGURE CORRECTED (cycle-4 Q/A).** I wrote "the same grep now returns 5"; the
command I quoted returns **4**. `grep -c` counts matching LINES, and line 112
carries two matches, so the honest statement is **5 matches across 4 lines** --
and the 5 came from a *different* pattern I actually ran (it carried an extra
`REACHES THE DETECTOR` alternative that the quoted pattern does not). Quoting one
pattern and reporting a count from another is the same defect that capped 86.94
three times. Both forms, verbatim:

```
$ grep -cE "reach(es|ed)? the detector|pre-detector|bash exit|recursion guard|86\.97" handoff/current/live_check_86.91.md
4
$ grep -oE "reach(es|ed)? the detector|pre-detector|bash exit|recursion guard|86\.97" handoff/current/live_check_86.91.md | wc -l
5
```

The substantive property is unchanged and is the one that matters: that file had
**0** such references before this cycle and has them now.

Swept by claim class, **seeded from `night_diagnostics.md:51`** — an artifact I
did not write for this purpose — rather than from my own phrasing. Cycle 3
searched `"every invocation"`, my wording, while the survivor said
`"every decision"`. Recall test: both members named in the masterplan note are
found (`live_check_86.91.md:104`; `experiment_results_86.91.md:444`, the note's
`:441` being off by three).

### J3. Mutation matrix — control GREEN first, hook never touched on disk

```
  rc=0   ALL GREEN: 48 passed, 0 failed
  CONTROL GREEN.
--- N-1 delete the _flip_magnitude() call (hook :214)
    KILLED   rc=1  FAILED: 41 passed, 7 failed
--- N-2 never record a reason (force the :267 .get default)
    KILLED   rc=1  FAILED: 42 passed, 6 failed
--- N-3 swap flip_created / flip_transitioned
    KILLED   rc=1  FAILED: 46 passed, 2 failed
--- N-4 subject-major branch stops recording its reason
    KILLED   rc=1  FAILED: 46 passed, 2 failed
  rc=0  ALL GREEN: 34 passed, 0 failed
  SURVIVED, as required -- without [3a] the same mutant is invisible.
  => the N-1 kill is attributable to the new decision-content assertions.
killed=4  survived=0  unscorable=0  of 4
N-5 attribution control: OK
production hook byte-identical after the run: True
PASS
```

**N-5 is the attribution control**: keep the N-1 mutant, delete the new `[3a]`
section, and the guard returns to green at 34/0 — so the kill belongs to the new
assertions and not to something else that moved. Every mutant is applied to the
hook **source string** the checker writes into a throwaway temp repo; the
production hook is verified byte-identical after each run.

### J4. What is NOT claimed

- The immutable command (`bash -n`) is a parse check and **cannot fail on this
  class**; it proves only that the hook still parses.
- Four of nine reason states are driven, not all nine. `masterplan_unreadable_at_HEAD`,
  `first_commit`, `detector_error:<Type>` and `flip_created_and_transitioned`
  remain undriven — stated as a bound rather than implied to be covered.


### J5. CYCLE-4 Q/A REMEDIATION — a field I parsed and never asserted

The cycle-4 Q/A returned **CONDITIONAL with all 7 immutable criteria MET and
independently re-executed**, and capped on three findings. All three reproduce.

**1. `bump` was parsed and asserted by nothing.** The evaluator executed
`return "minor"` → `return "patch"` in `_flip_magnitude()`'s kickoff branch and it
**SURVIVED at 48/0**: `DECISION_RE` captured `bump`, `parse_decision` returned it,
and no assertion read it — while my own artifacts said the 4-tuple was "compared
by exact equality". `bump` is now pinned in all four scenarios, and that mutant is
cell **N-6**, now KILLED.

**2. A figure that did not reproduce under the command it names.** Corrected in
§J2 above: `grep -c` returns 4 (lines), the `-o | wc -l` form returns 5
(matches). My 5 came from a pattern carrying an alternative I never quoted — the
same *pattern-quoted-≠-derivation-used* defect that capped 86.94 three times.

**3. Superseded blocks left un-annotated.** §E:171 and the two `35 passed` lines
are now marked **SUPERSEDED in place**, and the header's "verbatim tool output"
promise is scoped per-section rather than per-file.

```
  rc=0   ALL GREEN: 52 passed, 0 failed
  CONTROL GREEN.
--- N-1 delete the _flip_magnitude() call (hook :214)
    KILLED   rc=1  FAILED: 44 passed, 8 failed
--- N-2 never record a reason (force the :267 .get default)
    KILLED   rc=1  FAILED: 46 passed, 6 failed
--- N-3 swap flip_created / flip_transitioned
    KILLED   rc=1  FAILED: 50 passed, 2 failed
--- N-4 subject-major branch stops recording its reason
    KILLED   rc=1  FAILED: 50 passed, 2 failed
--- N-6 kickoff magnitude minor -> patch (the Q/A's surviving Q1 mutant)
    KILLED   rc=1  FAILED: 51 passed, 1 failed
--- N-7 subject classifier phase-X.0 magnitude minor -> patch
    *** SURVIVED ***  rc=0  ALL GREEN: 52 passed, 0 failed
  rc=0  ALL GREEN: 34 passed, 0 failed
  SURVIVED, as required -- without [3a] the same mutant is invisible.
killed=5  survived=1  unscorable=0  of 6
N-5 attribution control: OK
production hook byte-identical after the run: True
```

**N-7 is an EQUIVALENT mutant, and that is proven rather than asserted.** It
mutates the *subject* classifier's `phase-X.0 → minor` rule at `:81`. That value
is unobservable: `bump_type = classify_commit(...)` at `:95` is **unconditionally
overwritten** at `:213-214` (`if bump_type != "major": bump_type =
_flip_magnitude()`) before its first read at `:278`, so only `major` survives
that assignment. It is reported as a survivor rather than hidden, and scored as
equivalent rather than silently re-aimed — it surfaced because my first N-6 anchor
matched `:81` instead of the kickoff branch.

**Guard: 35 → 48 → 52 assertions.**


### J6. CYCLE-6 — POST-VERDICT, UNGRADED. The cycle-5 CONDITIONAL's three WARNs

**No Q/A has graded these.** The day's token ceiling (R3) was exceeded —
**4,585,189 of 4,500,000** — at the step boundary immediately after the cycle-5
verdict, so step work stopped. One attempt of three remained; the budget, not the
attempt cap, is why this step parks. Recorded here rather than folded in silently.

All three findings reproduce. All three are mine.

**W1 — a FALSE claim still standing, and it was the original PARK blocker.**
`experiment_results_86.97.md:185-187` still read *"It is covered incidentally by
the end-to-end driver (if it were deleted the hook would fail)"*. Measured: rc=**0**,
and the hook writes `bump=minor reason=unrecorded`. The hook does not fail.
Cycles 4 and 5 **added** section J1 and cell N-1 and left the false sentence
standing — accompany-not-replace, one file from the criterion that forbids it.
**Replaced, not annotated.**

**W2 — a LIVE, non-equivalent surviving mutant.** `_flip_magnitude()`'s
phase-emptied branch (`:201`, `return "major"`) had **no scenario at all**: no
seed emptied a phase, and scenario 4's `major` comes from the *subject* path
(`:216`) where `_flip_magnitude()` is never called. So my "spanning all four bump
magnitudes" was true of the observed **values** and false as **branch coverage** —
the producing branch was 3 of 4. A fifth scenario closes both steps of a
two-step phase; the mutant is now cell **N-8**, KILLED.

**W3 — the "END-TO-END" drive was silently truncated in every cycle.**
`CHANGELOG_SEED` used `|---|---|---|` while the hook requires
`startswith("|------")` (`:357`), so `insert_idx` stayed `None` and the heredoc
`sys.exit(0)`d at `:362`. The dedup guard, the row insert, the MAX_ROWS trim, the
file write and the bash tail executed in **zero** drives, across every cycle of
this step, while the artifacts called it end-to-end. Criterion 4's load-bearing
clause still held (`_log_decision(bump_type)` at `:278` is *before* the cut, and
delete-the-call was genuinely killed), but the fixture could not represent
production — inside the guard built to close a fixture-blindness defect.

Fixed with the one-line separator, **and the inference is now an assertion**: the
drive records whether the CHANGELOG actually changed, so a truncated heredoc can
no longer look identical to a successful run. That assertion is mutation-tested —
restoring the old seed turns it RED (`52 passed, 1 failed`) against a green
control.

```
  rc=0   ALL GREEN: 57 passed, 0 failed
  CONTROL GREEN.
--- N-1 delete the _flip_magnitude() call (hook :214)
    KILLED   rc=1  FAILED: 46 passed, 11 failed
--- N-2 never record a reason (force the :267 .get default)
    KILLED   rc=1  FAILED: 50 passed, 7 failed
--- N-3 swap flip_created / flip_transitioned
    KILLED   rc=1  FAILED: 54 passed, 3 failed
--- N-4 subject-major branch stops recording its reason
    KILLED   rc=1  FAILED: 55 passed, 2 failed
--- N-6 kickoff magnitude minor -> patch (the Q/A's surviving Q1 mutant)
    KILLED   rc=1  FAILED: 56 passed, 1 failed
--- N-8 phase-emptied magnitude major -> patch (the Q/A's surviving W2 mutant)
    KILLED   rc=1  FAILED: 56 passed, 1 failed
--- N-9 the end-to-end seed cannot represent production (W3)
    KILLED   rc=1  FAILED: 56 passed, 1 failed
--- N-7 subject classifier phase-X.0 magnitude minor -> patch
    *** SURVIVED ***  rc=0  ALL GREEN: 57 passed, 0 failed
  rc=0  ALL GREEN: 35 passed, 0 failed
  SURVIVED, as required -- without [3a] the same mutant is invisible.
killed=7  survived=1  unscorable=0  of 8
N-5 attribution control: OK
production hook byte-identical after the run: True
```

**N-7 remains the only survivor and is EQUIVALENT — now independently confirmed.**
The cycle-5 Q/A checked my proof two ways rather than taking it: `classify_commit`'s
result is unconditionally overwritten unless it is `major`, so the subject
classifier's `minor`/`patch` never reaches the log.

**Guard: 35 → 48 → 52 → 57 assertions.**
