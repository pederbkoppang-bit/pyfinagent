# Experiment results — step 86.21

**Step:** 86.21 — the 3rd-CONDITIONAL counter is structurally blind to any step still in flight, and fails open silently
**Date:** 2026-08-14 ~04:00 CEST
**Immutable command:** `grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/evaluator_critique_*.md | head -3` → **exit 0**, `1229`

> **Context an executor needs:** phase-86.75 (2026-08-13) independently rediscovered this
> defect and repointed the counter at `scripts/qa/qa_wip.py`. It did **not** reference
> 86.21, which is why step **86.76** now exists. This step's remaining work was to prove
> the repoint actually satisfies 86.21's six criteria — and **three of them did not hold**
> until the work below.

---

## C1 — the blindness, REPRODUCED

The criterion asks for a step with N>1 recorded verdicts, still pending, where the grep the
rule prescribed returns zero rows.

**I did not pick the subject; I measured every candidate and let the data pick it.** My
first attempt used 86.21 itself and I had pre-written the sentence "returns 0 for the
pending one" — the measurement came back **2**. The claim was written before the number
was read, so the subject was wrong, not the finding.

| step | status | ledger (`records_retained`) | anchored log rows |
|---|---|---:|---:|
| **86.62** | pending | **4** | **0** ← **C1 subject** |
| 86.44 | pending | 4 | 1 |
| 86.9 | pending | 4 | 3 |
| 86.38 | pending | 4 | 1 |
| 86.5 | pending | 3 | 1 |
| 86.29 | pending | 3 | 3 |
| 86.21 | pending | 2 | 2 |
| 86.32 | done | 5 | 1 |

**86.62 is the reproduction in its most consequential form:** four Q/A spawns — the step
that *escalated* after four attempts — and the prescribed counter would have read **zero**
every single time.

Generalised: the log **undercounts in 6 of 8** steps and agrees in 2. Never overcounts,
once the grep is header-anchored.

**Positive control:** the same anchored grep returns **6** rows for `phase=36.17`, a
CLOSED step. The probe is live; the zeros are real.

---

## C2 — the counting source, and why it does not touch harness_log

**Chosen:** `scripts/qa/qa_wip.py` over `.claude/agent-memory/qa/verdicts/` — run-stamped
WIP records written write-first by each Q/A spawn.

**LOG-is-last is preserved untouched:** nothing here writes a `harness_log.md` row
mid-step. That ordering is deliberate (the file feeds the Harness tab and the next cycle's
resume detection), and 86.21's design constraint explicitly forbids fixing the counter by
breaking it.

The ledger also survives the **8.2% of spawns that drop and return no verdict at all**,
because the record is written *before* the analysis rather than flushed after it.

---

## C3 — the counter against 36.17's real history, and a DEFECT THIS FOUND

36.17's sequence, verified by anchored grep (cycles 190–195, all 2026-08-09):

```
190 CONDITIONAL   191 FAIL   192 FAIL   193 CONDITIONAL   194 CONDITIONAL   195 PASS
```

**36.17 has ZERO ledger records** — it ran 2026-08-09 and the ledger began 2026-08-10.
The criterion cannot be met by pointing the tool at 36.17; it is met by replaying the
sequence through the rule. Stated plainly rather than papered over.

**The replay exposed that phase-86.75 silently changed the rule.** 86.75's repoint wrote:
*"If this would be the third **attempt** or later, return FAIL"* — but CLAUDE.md:371-376
defines the trigger as **3 consecutive CONDITIONALs**. Those are different rules:

| attempt | actual | consecutive-CONDITIONAL rule | attempt-count rule (86.75, live) |
|---:|---|---|---|
| 1 | CONDITIONAL | allows | allows |
| 2 | FAIL | allows | allows |
| 3 | FAIL | allows | allows |
| 4 | CONDITIONAL | allows | **forces FAIL** ← diverge |
| 5 | CONDITIONAL | allows | **forces FAIL** ← diverge |
| 6 | PASS | allows | allows |

**Longest consecutive run on 36.17 is 2, so the correct rule never fires. The
attempt-count rule would have failed 36.17 twice and denied the PASS it earned at attempt
6.** It was also stricter than CLAUDE.md's F1b cumulative budget (**5** attempts, which
**escalates to the operator** rather than auto-failing). Three bounds existed; they
disagreed; the tightest was live by accident.

**Fixed in both rails** — `.claude/agents/qa.md` and `.claude/workflows/qa-verdict.js`
(the first-class rail carried the same text). The reset-on-FAIL path is exercised at
attempts 2 and 3 (run 1 → 0). Zero survivors of the superseded trigger, negative-controlled.

---

## C4 — the independence question, answered

**Main-supplied counts are ADVISORY. The ledger is AUTHORITATIVE. `harness_log` is a
secondary cross-check that loses on disagreement.**

86.21's objection was precise: a spawn-prompt count is *authored by Main, and Main is the
party the rule constrains*. The ledger removes exactly that — records are written by each
**Q/A** spawn, not by Main, so the audited party no longer authors the counter's input.

**The residual, stated rather than hidden:** the rule constrains the Q/A role, and the Q/A
role writes the records. A given spawn cannot alter its predecessors (they are on disk and
run-stamped) but *could* skip its own write and undercount its successors. That write is
mandated by `qa.md` STEP 0b and is the only path `qa-write-guard.sh` permits the Q/A to
write. **So: more independent than a Main-supplied count, not perfectly independent.**

---

## C5 — fail-safe direction: it failed OPEN, and now it fails LOUD

**Before this step the ledger inherited the exact defect it replaced.**
`list_wip_records` returns `[]` when the sink directory is missing, so
`records_retained: 0` meant both "no prior attempts" and "the counter has no input" —
indistinguishable, and the escalation rule silently disabled.

Added `source_present()` and a `source_present` key. A count of zero is a fact about
attempts **only** when it is `true`; otherwise the guidance now reads *"SOURCE MISSING …
records_retained=0 is NOT a statement about prior attempts"*, and both rails instruct the
Q/A to report the attempt number as **UNKNOWN**.

**Direction: it now fails LOUD (closed).** That is right here because the failure mode
being removed is precisely a silent zero that reads as a clean slate.

---

## C6 — mutation test

`scripts/qa/mutate_counter_source_86_21.py`, all mutations inside a `TemporaryDirectory`:

```
[PASS] CONTROL  2 real records -> counted          records_retained=2
[PASS] M1 sink dir DELETED    -> notices           "SOURCE MISSING: the WIP sink ..."
[SKIP] M2 records DELETED     -> UNSCORED (see below)
[PASS] BASELINE genuine 1st attempt

genuine-first-attempt output == sink-DELETED output : False   (was True before the fix)
mutants surviving (undetected): 0
```

**M2 is unscored, with its reason, not quietly passed.** `prune_wip_records` deletes old
records *by design* (`DEFAULT_KEEP=3`), so "sink present, no record" is a state the module
produces deliberately and is genuinely identical to a first attempt. Detecting it needs a
second monotonic counter outside the sink — more machinery than the defect warrants.
**Stated limit: record loss inside an existing sink is not self-detectable.**

### What the mutation work incidentally found: FOUR DEAD CELLS in the 86.31 matrix

Running `mutation_matrix_86_31.py` as a regression surfaced that **4 of its 24 cells had
silently stopped testing anything** — `ANCHOR-BAD`, meaning the text they pin no longer
exists:

| cell | anchor drifted because | broken since |
|---|---|---|
| P2, Q2 | 86.36 inserted `, path revised by phase-86.36` into the `STEP 0b` line | `6e8f3169`, 2026-08-11 |
| M3 | later phases added dict keys *after* `"guidance": ""` | before 86.75 |
| M5 | 86.36 refactored the one-line `return` into a `sink` local | before 86.75 |

**None was caused by tonight's edits** — verified against `HEAD` and `9a59a4fa~1`, where
all four already counted 0. All four anchors repointed to text that occurs **exactly once**
(measured before editing; a 2-occurrence candidate was rejected).

**Measured state, staged so the claim matches the evidence:**

| run | result |
|---|---|
| before any repoint | P2, Q2 `ANCHOR-BAD`. M3, M5 **also** bad but hidden — I had piped through `tail -6` |
| after P2/Q2 repoint | **22/24 KILLED**; P2 and Q2 now genuinely kill (Q2: *"8 assertion(s) red"*) |
| after M3/M5 repoint | **RUNNING at the time of writing — not yet verified.** Do not read 24/24 into this table. |

**I found M3/M5 only because I stopped truncating the output.** The first two runs looked
like a 2-cell problem because I could not see the rest of the report.

**A correction to my own reporting:** I first said the suite "still exits 0" with dead
cells. **That was wrong** — `:390` returns `1` unless every cell is killed. My `exit=$?`
read the exit status of `tail` through a pipe. Demonstrated: `(exit 7) | tail -1` → `$?`
is `0`; with `pipefail` it is `7`. The script was right; my measurement was not.

---

## Files changed

| File | Change |
|---|---|
| `scripts/qa/qa_wip.py` | `source_present()` + `source_present` key + a SOURCE-MISSING guidance branch |
| `.claude/agents/qa.md` | trigger corrected to 3 **consecutive** CONDITIONALs; `source_present` check; F1b note |
| `.claude/workflows/qa-verdict.js` | same two corrections on the first-class rail |
| `scripts/qa/mutate_counter_source_86_21.py` | **new** — C6 mutation matrix |
| `scripts/qa/mutation_matrix_86_31.py` | 4 dead anchors repointed |

**Regression:** `verify_wip_retention_86_36.py` 23/23, `mutation_matrix_86_36.py` 5/5
cells killed, `verify_qa_write_first_86_31.py` 238/238, `prove_qa_write_separation_86_31.py`
OK. `verify_research_gate_workflow.mjs` **121 passed, 0 failed**.

## Scope honesty

- **No production/trade-path file touched.** Everything above is harness tooling and agent
  prompts.
- **86.21's criteria are addressed but NOT self-certified** — a Q/A has not yet graded this.
- The `qa.md` and `qa-verdict.js` edits are **agent-file changes I authored**, so the
  separation-of-duties rule applies: they need operator review, and this step now adds a
  second such edit on top of 86.75's.
