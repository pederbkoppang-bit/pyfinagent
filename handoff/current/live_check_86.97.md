# live_check — phase-86.97

**STATUS: COMPLETE.** Every block below is verbatim tool output from this
session, complete and unelided.

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
  ok   [3] the decision line carries a reason
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
verify_decision_log_86_97.py                     ALL GREEN: 35 passed, 0 failed
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

ALL GREEN: 35 passed, 0 failed
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
