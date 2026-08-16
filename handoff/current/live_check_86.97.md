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

### It is worse than "surviving" — it is INVISIBLE

`detector_source()` walks `tree.body` collecting only `FunctionDef` / `Assign` /
`AnnAssign` — **definition** classes that bind a name. A bare
`_log_decision(bump_type)` is an `ast.Expr(Call)` and binds nothing, so it can
never match. **Enlarging `NEEDED` cannot help.** Measured:

```
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

Measured twice, hours apart, and it moves — which is the point: at 21:03 it was
`commits=47 lines=24 gap=23 recursion=24`; at 21:52, `51 / 26 / 25 / 26`. A
pinned figure would already be wrong. The gap tracks the recursion-guard count,
not a mystery.

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
  ok   [3] ISOLATION: the real repo's decision log is untouched by this driver

[4] MUTATION -- the guard must SEE a deleted call (criteria 4, 6)

  ok   [4] CONTROL: the UNMUTATED hook writes a decision line
  ok   [4] delete-the-production-call: KILLED -- removing the call the 86.91 extraction is structurally blind to makes the guard RED
  ok   [4] neuter-the-log-write: KILLED -- removing the write itself, so the effect disappears without the call moving makes the guard RED
```

`delete-the-production-call` is **the same mutant that SURVIVED in §A**. That
before/after pair is the whole step in two lines.

The second mutant exists because the first alone is not enough: it removes the
*write* while leaving the call in place, so a guard that merely checked "the call
text is present" would pass. Both are checked for **buildability** before being
scored.

**Isolation is asserted, not hoped for.** The driver runs the real hook, which
ends in `git add` + `git commit` — so the real repo's decision log is snapshotted
and required byte-identical afterwards. A driver that contaminated the log would
be corrupting the very evidence §B reasons about.

```
ALL GREEN: 20 passed, 0 failed        (exit 0)
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
verify_decision_log_86_97.py                     ALL GREEN: 20 passed, 0 failed
verify_changelog_flip_86_91.py                   ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs                ALL GREEN: 96 passed, 0 failed
verify_research_gate_workflow.mjs                ALL GREEN: 124 passed, 0 failed
```

No masterplan step was flipped and no verdict altered by this work. The only
behavioural change to production code is **none**: the hook edit is a docstring.
