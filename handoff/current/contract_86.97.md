# Contract — phase-86.97

**Written BEFORE any GENERATE work.** Research gate cleared first.

---

## Step

`86.97` (P2) — *"the changelog decision log is blind to three bash exit-0 paths
that run BEFORE the detector, and its only production invocation is unguarded —
measured 10 commits against 5 decision lines"*

---

## Research gate — PASSED (enforced, not self-reported)

`.claude/workflows/research-gate.js` by `scriptPath` (rail R7), run
`wf_71bc038d-45a`, 2 agents, 191,440 tokens, 530s.

```
gate_passed: true          agent_self_reported_gate_passed: true
self_report_disagreed: false               violations: []
sources_floor_ok: 8 >= 5                   urls_floor_ok: 23 >= 10
recency_scan_ok                            listed_sources_consistent: 8 >= 8
brief_on_disk_ok: handoff/current/research_brief_86.97.md (29386 chars, independently read)
brief_status_in_brief: COMPLETE            all_8_claimed_sources_present_in_brief
urls_collected_corroborated: 23 <= 23 distinct URLs in the brief
```

Read in full: Python `ast` docs; ShellCheck SC2317; ar5iv 2103.08480; ar5iv
1807.05030; bats-core writing-tests; Fowler *Harness Engineering*; `githooks(5)`;
arXiv 2602.10133v1.

---

## Hypothesis (reproduced by execution BEFORE this contract)

### H1 — the guard is not weak, it is **structurally blind**

`detector_source()` (`verify_changelog_flip_86_91.py:81-94`) walks `tree.body`
and collects only `ast.FunctionDef`, `ast.Assign` and `ast.AnnAssign` — all
**definition** classes that bind a name. A bare `_log_decision(bump_type)` is an
`ast.Expr(Call)`; it binds nothing, so it can never match. **Enlarging the
`NEEDED` tuple cannot fix this** — the call has no name to enlarge toward.

Measured, and this is the load-bearing distinction: deleting the production call
leaves the extracted source **byte-identical**.

```
extracted SHIPPED, unmutated   : 7,597 B  sha1 f7458a6ab1f5fe96
extracted SHIPPED, call DELETED: 7,597 B  sha1 f7458a6ab1f5fe96
BYTE-IDENTICAL: True
CONTROL (edit inside the def)  : 7,621 B  differs from base: True  (+24 B)
```

The control matters: the extraction *is* sensitive to changes inside the
definition, so the blindness is specific to the call, not a dead extractor. The
mutant is therefore **INVISIBLE, not surviving** — no assertion added to that
file, however clever, can ever kill it. That is why criterion 4 demands the
whole heredoc be driven end-to-end.

Confirmed end-to-end, with the control observed GREEN first (per criterion 6 and
`feedback_a_mutant_that_cannot_build_scores_as_a_kill`):

```
control INSIDE the worktree   : ALL GREEN: 42 passed, 0 failed
MUTANT (call deleted, 25 B)   : ALL GREEN: 42 passed, 0 failed   <- SURVIVED
bash -n on the mutant          : parses  -> BUILDABLE, so the score is valid
```

The research adds a second exposed call site I had not found:
`bump_type = _flip_magnitude()` at hook `:214` is inside an `ast.If`, likewise
absent from the extraction — and **the checker manufactures both calls itself**
(`:132`, `:534`), which is precisely how the production call went unnoticed.

Literature name for this defect class: **pseudo-tested** methods (Vera-Perez;
Niedermayr — 291/2041, and 14 of 25 manually-inspected were side-effect methods,
which is `_log_decision`'s exact shape).

### H2 — most of the commits-vs-lines gap is the recursion guard, and that may be correct

The classification rule can be **lexical**, and its soundness preconditions were
checked rather than assumed:

- the detector is a single `python3 - ... << 'PYEOF'` invocation spanning lines
  **43–371**;
- the hook defines **no bash functions** (so lexical order == execution order);
- it contains **no `trap`, `source`, `.` or `eval`** (so nothing reorders).

Therefore: an `exit` lexically before line 43 provably cannot reach
`_log_decision`. Members found from source: `:28`, `:33`, `:37` (before) and
`:378`, `:380`, `:381` (after — the detector already ran, so a decision line
already exists).

Criterion 3 asks specifically whether the recursion guard is a defect or a
bound. Evidence that it is a **bound**: the auto-changelog commit is by
construction not a bump candidate. Re-derived at execution time (window pinned
to the decision log's own first stamp, not a bare date, which slides):

```
decision lines: 24   first stamp: 2026-08-16T08:23:33Z
commits since that stamp: 47      gap: 23
commits matching the recursion guard's own pattern: 24
```

The gap is the recursion guard almost exactly. **The step's "10 commits vs 5
lines" is a stale snapshot** — criterion 1 requires re-derivation, and the figure
must be recomputed at run time, never pinned.

---

## Immutable success criteria — copied VERBATIM from `.claude/masterplan.json`

1. "both defects are REPRODUCED by execution first: the commits-vs-lines gap re-derived at execution time (not the 10-vs-5 figure copied), and the delete-the-call mutant shown SURVIVING the current guard with the control observed GREEN first"
2. "the early-exit paths are enumerated FROM SOURCE by a written-down rule, not hand-listed, and each is classified as MUST-LOG or LEGITIMATELY-SILENT with the reason per member; a scan that cannot find its own known members is a FAILED gate"
3. "the recursion guard specifically is judged rather than assumed: a hook re-entering itself may be correct to stay silent, and if so that is a BOUND to state, not a defect to fix"
4. "the production invocation is guarded by driving the WHOLE heredoc end-to-end against a temp repo, so that deleting the call turns the guard RED -- an extraction-based guard structurally cannot see a bare call and must not be patched to pretend otherwise"
5. "phase-86.91's criterion-4 claim is CORRECTED in every artifact that carries it, to state that decisions are explained for invocations that reach the detector; a correction must replace, not accompany"
6. "every new guard is mutation-tested with the control observed GREEN first, and a mutant that does not BUILD is scored UNSCORABLE and FAILS rather than counting as a kill"
7. "verdict semantics and masterplan state are UNCHANGED: nothing here may flip a step or alter a verdict"

Immutable command:
`bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`

**Disclosed weakness:** as with 86.92, this is a *parse* check and cannot fail on
either defect. The real evidence goes in `live_check_86.97.md`.

---

## Plan

**P1 — end-to-end driver (criterion 4).** Add a guard that runs the **real hook**
(`bash .claude/hooks/post-commit-changelog.sh`) inside a **temp git repo**, then
asserts on the **decision-log FILE** — not on any extracted namespace. This is the
only instrument that can observe a deleted call, and it is also the only one that
reaches the three pre-detector `exit 0` paths, which live *outside* the heredoc
and are therefore invisible to every Python-side test. `bats-core` is the
documented idiom; a plain bash harness in `scripts/qa/` matches this repo's
existing shape, so I will not add a dependency.

**P2 — source-derived enumeration with a written-down rule (criterion 2).**
Enumerate exits mechanically, classify each, and **self-test the scan**: it must
find its own known members, else FAIL. Explicitly record the soundness
preconditions (no functions / no trap / no eval) and assert them, since the
lexical rule is only valid while they hold.

**P3 — classify (criteria 2, 3).** `:28` recursion guard →
LEGITIMATELY-SILENT, stated as a BOUND, with the measured share of the gap it
accounts for. `:33` and `:37` → MUST-LOG (a missing CHANGELOG or a missing
`### Recent Activity` section is machinery breakage an operator needs to see).
`:378/:380/:381` → post-detector, decision already written.

**P4 — correct 86.91's criterion-4 claim everywhere (criterion 5).** A
correction must **replace**, not accompany. Enumerate every artifact carrying the
claim and diff each — per `feedback_verification_probe_built_from_edited_strings`,
the enumeration must not be built from my own wordings.

**P5 — mutation-test every new guard (criterion 6).** Control GREEN first; every
mutant checked that it **BUILDS** (`bash -n` / `python -c compile`) before its
kill/survive is scored. A mutant that does not build is **UNSCORABLE and FAILS**.

---

## Non-goals

- Patching `detector_source` to "see" the call. Research is explicit that this is
  a category error; criterion 4 forbids pretending otherwise.
- Changing any bump semantics or the classifier's behaviour.
- Adding `bats-core` as a dependency.
- Flipping any step or altering any verdict (criterion 7).

---

## References

- `handoff/current/research_brief_86.97.md` (gate PASSED, 8 sources / 23 URLs)
- `.claude/hooks/post-commit-changelog.sh` — detector heredoc `:43–:371`
- `scripts/qa/verify_changelog_flip_86_91.py:81-94` — `detector_source`
