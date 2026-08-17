# Contract — phase-86.97, cycle 4

**Step:** `86.97` — the changelog decision log is blind to three bash exit-0 paths
that run BEFORE the detector, and its only production invocation is unguarded.

**Cycle:** 4 (prior: CONDITIONAL, CONDITIONAL, FAIL — parked overnight at the
3-attempt rail). **Attempt budget today: 3.**

---

## 1. Research gate — PASSED (enforced, not self-reported)

`.claude/workflows/research-gate.js`, run `wf_aeceef87-d82`.

```
gate_passed: true      violations: []      self_report_disagreed: false
sources_floor_ok: 6 >= 5        urls_floor_ok: 26 >= 10
recency_scan_ok                 all_6_claimed_sources_present_in_brief
brief_on_disk_ok: handoff/current/research_brief_86.97_cycle4.md
                  (25593 chars, independently read)
brief_status_in_brief: COMPLETE
```

Brief: `handoff/current/research_brief_86.97_cycle4.md`. **It sharpened the
diagnosis beyond the park note**, and two findings drive the design:

- **The assertion is not weak, it is VACUOUS.** `"reason=" in log_text` tests for
  a token the hook emits from a **literal format string**
  (`post-commit-changelog.sh:271`), so it is true for *every non-empty line the
  writer can produce* and is **strictly subsumed** by the existence check at
  `verify_decision_log_86_97.py:301`. It cannot distinguish any decision from any
  other.
- **The E2E driver exercises 1 of 9 reason states.** The `{"phases": []}` seed at
  `:260` only ever produces `no_flip`, and pins **zero** values.
- Sources 2/6 (arXiv 2410.21136, 2402.11041) warn that oracles drift toward
  *observed* rather than *expected* output; the expected table must be derived
  from **branch structure before driving**. Bats docs give exact-equality as the
  shell idiom. The QGS literature names my cycle-3 sweep failure verbatim: seeding
  a search with the same terms you search for makes recall ~100% and meaningless.

---

## 2. Both blockers REPRODUCED by execution, before any fix

### Blocker A — the `:214` mutant survives and writes a spurious decision

Driven through section [3]'s own `drive()` helper, which takes the hook **source
as a string** and writes it into a throwaway temp repo, so the real hook is never
touched (verified: `diff` against a pre-run backup is identical).

```
CONTROL
    rc = 0
    decision line = '...Z 342ffc8 bump=none reason=no_flip created_done=- transitioned_done=-'
    bump   = none      reason = no_flip

M1 delete the _flip_magnitude() call  (`bump_type = _flip_magnitude()` -> `pass`)
    rc = 0
    decision line = '...Z 342ffc8 bump=minor reason=unrecorded created_done=- transitioned_done=-'
    bump   = minor     reason = unrecorded

DIFFERENT DECISION: True
BOTH satisfy the shipped assertion "'reason=' in log_text": True
```

Non-equivalent (a **spurious `minor` bump** — exactly what 86.68 exists to
prevent — with an **unexplained reason** — exactly what 86.91 criterion 4 exists
to close) and **invisible** to the shipped guard.

### The 9 reason states, enumerated FROM SOURCE before driving

| # | reason | site | set by |
|---|---|---|---|
| 1 | `masterplan_unreadable_at_HEAD` | `:160` | detector |
| 2 | `first_commit` | `:163` | detector |
| 3 | `no_flip` | `:190` | detector |
| 4 | `flip_created` | `:193` | detector |
| 5 | `flip_transitioned` | `:194` | detector |
| 6 | `flip_created_and_transitioned` | `:195` | detector |
| 7 | `detector_error:<Type>` | `:207` | detector |
| 8 | `subject_forced_major` | `:216` | **outside** `_flip_magnitude()` |
| 9 | `unrecorded` | `:267` `.get` default | **nobody** — the signature of a detector that never ran |

State 9 is the mutant's fingerprint and must never appear in a healthy run.

### Blocker B — one carrier is still unbounded

Sweep seeded from an **independent** artifact (`night_diagnostics.md:51`, which
names the survivor), not from my own phrasing — the cycle-3 failure was seeding
with `"every invocation"`, my own wording, while the survivor says
`"every decision"`.

Recall test: both members named in the masterplan note are found
(`live_check_86.91.md:104`; `experiment_results_86.91.md:444` — the note said
`:441`, off by three).

Measured dispositions:

| carrier | state |
|---|---|
| `handoff/current/live_check_86.91.md:104` | **UNBOUNDED — the survivor.** `grep -cE "reach(es\|ed)? the detector\|pre-detector\|bash exit\|recursion guard\|86\.97"` over that file returns **0**. |
| `handoff/current/experiment_results_86.91.md:444` | already bounded — "holds only for invocations that REACH the detector" |
| `.claude/hooks/post-commit-changelog.sh:227` | already corrected in place |
| `handoff/current/live_check_86.97.md:217` | the correction table itself |

---

## 3. Immutable success criteria — VERBATIM from `.claude/masterplan.json`

1. "both defects are REPRODUCED by execution first: the commits-vs-lines gap re-derived at execution time (not the 10-vs-5 figure copied), and the delete-the-call mutant shown SURVIVING the current guard with the control observed GREEN first"
2. "the early-exit paths are enumerated FROM SOURCE by a written-down rule, not hand-listed, and each is classified as MUST-LOG or LEGITIMATELY-SILENT with the reason per member; a scan that cannot find its own known members is a FAILED gate"
3. "the recursion guard specifically is judged rather than assumed: a hook re-entering itself may be correct to stay silent, and if so that is a BOUND to state, not a defect to fix"
4. "the production invocation is guarded by driving the WHOLE heredoc end-to-end against a temp repo, so that deleting the call turns the guard RED -- an extraction-based guard structurally cannot see a bare call and must not be patched to pretend otherwise"
5. "phase-86.91's criterion-4 claim is CORRECTED in every artifact that carries it, to state that decisions are explained for invocations that reach the detector; a correction must replace, not accompany"
6. "every new guard is mutation-tested with the control observed GREEN first, and a mutant that does not BUILD is scored UNSCORABLE and FAILS rather than counting as a kill"
7. "verdict semantics and masterplan state are UNCHANGED: nothing here may flip a step or alter a verdict"

**Immutable command:** `bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'`
**Disclosed:** it is a parse check and **cannot fail on this class**. The real
evidence is `live_check_86.97.md`.

---

## 4. Plan

**P1 — assert the DECISION, not the existence of a line.** Replace the vacuous
`"reason=" in log_text` with a parse of the tuple
`(bump, reason, created_done, transitioned_done)` and an **exact-equality**
assertion per scenario, against a table derived from the branch structure above
*before* any run.

**P2 — drive ≥4 of the 9 states**, not 1: `no_flip`, `flip_created`,
`flip_transitioned`, and `subject_forced_major` (the only reason set outside the
detector, so it is the one scenario the `:214` mutant must NOT change).

**P3 — `reason == "unrecorded"` is a hard failure anywhere**, as a standing
negative control: it is reachable only when the detector never ran.

**P4 — bound `live_check_86.91.md:104` in place** (replace, not accompany).

## 5. Mutations — named BEFORE the work, each to be RUN with the control green

| id | mutation | required |
|---|---|---|
| N-1 | `bump_type = _flip_magnitude()` → `pass` | RED (today: **SURVIVES**) |
| N-2 | force `_FLIP_DECISION["reason"]` unset before the writer | RED (`unrecorded` control) |
| N-3 | swap `flip_created` / `flip_transitioned` in the hook | RED (per-scenario table discriminates) |
| N-4 | make the subject-major branch skip `subject_forced_major` | RED |
| N-5 | revert the new assertions, keep the mutant | control: guard must go green again, proving the kill is the assertion's |

A mutant that does not build is **UNSCORABLE and fails**, never a kill
(criterion 6).

## 6. Rails

- **R5:** no edits to `qa.md`, `qa-verdict.js`, `research-gate.js`.
- Criterion 7: no step flipped, no verdict altered.
- **The real hook is never mutated on disk** — all mutation goes through
  `drive(hook_src)` into a temp repo.
- **Lesson carried from 86.94:** every quoted figure in `live_check_86.97.md`
  must be regenerated from a live run in the same pass that writes it. Three
  consecutive 86.94 cycles capped on hand-maintained numbers.
