# Experiment Results -- step 90.3

> **STATUS: BUILT AND VERIFIED, NOT EVALUATED. NOT CLOSEABLE.** No Q/A spawned; the step
> stays `pending`, ungraded. **Research gate PASSED (enforced)** -- `wf_8f0a6091-2d0`,
> 10 sources read in full, 34 URLs, `self_report_disagreed: false`, `violations: []`.
> Contract: `handoff/current/contract_90.3.md`.
>
> **The mechanism ships DEFAULT-OFF** behind `ATTEMPT_GATE_PROGRESS_DIGEST`. It adds a new
> DENY path to a PreToolUse hook that runs on every Workflow launch, and it has not been
> graded. An ungraded deny-capable gate that misfires blocks the entire harness, so it stays
> dark on the live rail until an operator enables it deliberately. The self-test and the
> matrix both set the flag, so the mechanism is fully exercised while the rail is untouched.

**Step:** 90.3 -- progress-gated retry, corrected: the digest must exempt rail drops and
must never be read as proof of convergence. **Date:** 2026-08-21.

---

## 1. The gate found the central defect BEFORE any code existed

Criterion 1 derives the file set from *declared masterplan paths ∪ `git diff --name-only
HEAD` ∪ untracked*. I ran that union myself:

```
handoff/audit/attempt_budget_audit.jsonl     ← TRACKED, written BY THE GATE on every launch
handoff/audit/pre_tool_use_audit.jsonl       ← written on every tool call
.claude/agent-memory/researcher/MEMORY.md    ← written by the researcher DURING that gate
```

**Three self-references in one union.** Without exclusions the digest advances on every
launch regardless of the work — **89.1's defect through a different door, in the step built
to correct 89.1.** And *"declared masterplan paths"* has no source at all: **0 of 1222 steps
carry a `files` or `paths` key.**

So `DIGEST_EXCLUDED_ROOTS` is the load-bearing part of this build, not a detail — and cell
**D1**, which removes `handoff/audit/` from it, is the cell this step exists for.

## 2. What this instrument is, stated so nothing overreads it

A byte-digest is the **weakest** of the three published stagnation signals: CUDABeaver
measures SHA-256 `duplicate_code` at 0–50.8% and `code_cycle` at 0.7–3.8%, against *semantic*
`no_progress` at **44.6–84.6%**. It is built because it is deterministic and cheap. **It
detects an exact repeat and an A→B→A oscillation, and nothing else.** Optimal-stopping work
triggers on an **absolute score**, explicitly not on inter-iteration change — which is
criterion 5's whole point, and criterion 5 is the load-bearing half of this step, not
criterion 1.

The drop exemption is not a carve-out either: Temporal defines a permanent failure as one
that *"requires a change to your input"*, and a dropped rail supplies no such change. On the
real ledger, 14 of 16 `NO_VERDICT` rows were followed by a retry and 8 of the 11 affected
steps later reached PASS. **89.1 would have denied all 14.**

## 3. Verification

```
$ bash -c 'python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_3.py --verify'
EXIT 0

CONTROL GREEN
  ok   N0   SURVIVED  expected SURVIVED
  ok   D1   KILLED    expected KILLED    handoff/audit/ removed from the exclusions -- the digest advances by construction
  ok   D2   KILLED    expected KILLED    mtime mixed into the digest, so os.utime moves it
  ok   D3   KILLED    expected KILLED    the digest becomes a constant
  ok   D4   KILLED    expected KILLED    a missing input becomes a silent skip over a subset
  ok   D5   KILLED    expected KILLED    the NO_VERDICT exemption removed -- the post-drop retry denied
  ok   D6   KILLED    expected KILLED    comparison narrows to the previous digest, so A->B->A oscillates forever
  ok   D7   KILLED    expected KILLED    another step's digests leak into this comparison
  ok   QX   ERROR     expected ERROR     NameError
real tree untouched: True
```

Red-first baseline before either half existed: **EXIT 2**, captured unpiped.

## 4. Four defects this work surfaced in my own harness

**The self-test was testing the functions and not the wiring.** Cell QX renames a name used
only inside `handle_hook`'s digest branch — and it **SURVIVED**, because every cell called
`decide()` / `compute_digest()` directly and nothing executed the hook. *A mechanism whose
functions are tested and whose wiring is not can be disconnected without a single check going
red.* Fixed with six cells that drive the real hook as a subprocess, end to end.

**The sandbox was missing an import, so the control was red for a reason unrelated to any
mutation.** `verdict_outcomes` does `from verdict_ledger_write import emit_sequence`, that
module lives under `scripts/qa/`, and the gate puts *both* `scripts/harness` and `scripts/qa`
on `sys.path`. My sandbox copied two hand-picked harness modules. The import failed, the
function failed **closed exactly as designed**, and the control went red. *A sandbox missing
an import does not test the subject — it tests the sandbox.* Now it mirrors the real import
surface rather than a guess at it.

**The sandbox seeded the real 146-row verdict ledger**, which collided with the self-test's
own fixtures. It now gets an empty one — better containment, and no real verdict history in a
temp dir for no reason.

**The ERROR discriminator went blind at one more process boundary.** With the wiring driven,
QX's `NameError` is caught by the production fail-open handler and printed *inside the nested
drive* — invisible to the outer matrix, so the mutant scored KILLED where the phase-90.12
discipline requires ERROR. **The same swallowed-signal defect 90.12 fixed, one boundary
further out: a discriminator can only read what it is given.** The nested stderr is now
surfaced, and QX scores ERROR.

## 5. A containment slip of mine, disclosed

While driving the mechanism by hand I set the environment with `env $E` in zsh, where `$E`
is **not word-split** — so the overrides never applied and the drive hit the real paths,
writing `handoff/current/escalation_attempt_budget_90.1.md`. It was untracked and had never
been committed, so nothing was overwritten; I checked before removing it. This is the exact
containment failure the matrix's guard exists to prevent, committed by hand outside the
matrix. The zsh behaviour is already in my memory notes, which is what makes it worth
recording rather than quietly deleting the file.

## 6. What is NOT done

- **No Q/A verdict.** Not closeable, not flipped.
- **Criterion 5's grep test is not built.** As written it returns 1086 lines over 111 files,
  overwhelmingly `.bak` copies of the criterion's own text — *a grep that matches its own
  specification cannot go red for the right reason.* The contract states the scoping rule it
  needs; the test itself is outstanding, and criterion 5 is the load-bearing half of this
  step, so this gap is the largest one here.
- **Criterion 6's disabled-control leg** (control GREEN with the check disabled, RED with it
  enabled) is covered for the *flag* (the last hook-drive cell) but not as a separate
  matrix control.
- **No flip of 89.1 to `superseded`.** That is an operator decision, queued in 90.3's notes.
