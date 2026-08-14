# PROPOSED — NOT APPLIED — one-line `qa.md` correction for step 86.79

**Status: WRITTEN OUT FOR THE OPERATOR, DELIBERATELY NOT APPLIED.**

`.claude/agents/qa.md` already carries **four Main-authored edits awaiting
operator review** under CLAUDE.md's separation-of-duties rule
(*"the same Claude Code session should not both author an agent `.md` change AND
self-evaluate work that depends on it"*). Applying this would make it **five**
and deepen a hold the operator owns. The operator instruction for this session is
explicit: *"If a fix genuinely needs `qa.md`, stop and ask."* The masterplan step's
own notes say the same: *"prefer changing `qa_wip.py`, or hand it to a fresh
executor."*

So this file is the ask. **Nothing in `.claude/agents/qa.md` was modified by step
86.79** — verify with `git diff --stat .claude/agents/qa.md`.

---

---

## ⚠ CYCLE-2 CORRECTION — the divergence had TWO members, and I had enumerated ONE

The cycle-1 Q/A (`wf_61338c26-b90`) independently found that the same false
statement was **duplicated in the launch rail's own prompt** —
`.claude/workflows/qa-verdict.js` — which this file, `experiment_results_86.79.md`
and `live_check_86.79.md` mentioned **0 / 0 / 0 times**. Its point is decisive:

> *"qa-verdict.js is NOT under `.claude/agents/`, so CLAUDE.md's separation-of-duties
> scope does not block editing it — routes A/B as written would fix only one of two
> consumers."*

**It was right, and the second member is now FIXED** — no operator gate was ever
needed for it. Enumerating the whole class rather than the two lines the Q/A named
(`:147`, `:152`) found **four** lines, all corrected in cycle 2:

| line | was | now |
|---|---|---|
| `:147` | *"reading records_retained / prior_records"* | reads `attempt_number` / `prior_attempts`, and **requires `--spawned-at`** |
| `:152` | *"records_retained gives the ATTEMPT number (authoritative)"* | `attempt_number` is the attempt number, INCLUSIVE; `records_retained` named a **gauge** |
| `:159` | *"If records_retained > the ledger verdict count…"* | compares `attempt_number` |
| `:172` | *"records_retained==0 is a fact about ATTEMPTS only when…"* | fail-closed via `attempt_number_status`, plus `attempt_number_is_lower_bound` |

**So the class is now 1 of 2 fixed, and the remaining member is the operator-gated
one below.** Scope note: the consequence-framing text in the same `qa-verdict.js`
block (*"return FAIL instead of a third"*, *"at 5+, recommend operator escalation"*)
was **deliberately left alone** — that is sibling step **86.78**'s subject, not this
step's.

---

## The divergence (the REMAINING member)

`.claude/agents/qa.md`, in the **3rd-CONDITIONAL auto-FAIL** section (anchor:
grep for `` `records_retained` is the count of prior ``; it was at **line 622**
when this was written — re-derive rather than trusting the number):

```
  `records_retained` is the count of prior Q/A spawns on this step — the
  **attempt number**, and it is authoritative. The JSON deliberately carries no
  `verdict` key (`is_verdict: false`) and never will.
```

**Two descriptions of one integer, differing by exactly one.** "count of prior
Q/A spawns" and "the attempt number" cannot both be true. Measured:
`records_retained` counts every retained record file **including the current
spawn's own write-first record**, so the *second* half is the true one and the
first half is wrong. Reproduced in `handoff/current/live_check_86.79.md` §1.

It is also a **gauge described as a counter** — Prometheus, *Metric types*:
*"Do not use a counter to expose a value that can decrease."* Pruning can lower
it, so it must not be compared to an escalation threshold at all.

## The proposed replacement

```
  `attempt_number` is this spawn's attempt, INCLUSIVE of itself — a first attempt
  is `1`. Use it, together with `prior_attempts`. Both are `null` (never `0`)
  when they cannot be computed, and `attempt_number_guidance` says why.

  Do NOT use `records_retained` as the attempt number. It counts retained record
  FILES, it includes your own write-first record, and pruning can lower it — it
  is a gauge, not a counter (`records_retained_unit` states this in the payload).

  The JSON deliberately carries no `verdict` key (`is_verdict: false`) and never
  will.
```

## Why this is safe to apply, and in which direction it errs

- It **cannot loosen** anything. It replaces a number that can be silently *too
  low* (which suppresses escalation) with one that refuses rather than guesses.
- It changes **no verdict semantics**. `report()` still has no `verdict` key.
- The fields it names already exist and are live as of this step — verify with
  `python scripts/qa/qa_wip.py 86.32 --spawned-at <iso>`.

## Three routes — the operator's choice

| route | what happens | who authors |
|---|---|---|
| **A** | Main applies the patch above | Main — becomes the **5th** Main-authored `qa.md` edit |
| **B** | A fresh executor applies it | not Main — separation of duties preserved |
| **C** | Leave it; the code-side mitigations stand alone | nobody |

**Recommended: B**, matching the step's own notes and §5 of
`handoff/current/goal_next_2026-08-14.md`.

## What already protects the reader if NOTHING is applied (route C)

Criterion 4 forbids leaving the divergence **silent**. It is not silent:

1. `records_retained_unit` — a string in the payload the Q/A actually reads,
   stating that the field is a gauge and pointing at `attempt_number` instead.
   This is the remedy the research itself prescribes (E1: *a name is not a unit*;
   Temporal's `MaximumAttempts` is inclusive, Step Functions' `MaxAttempts` is
   exclusive — same word, opposite unit, both official docs).
2. `attempt_number_guidance` — on the `no_record_for_this_spawn` path it says in
   words: *"Do NOT fall back to records_retained here … a low number SUPPRESSES
   escalation."*
3. This file.

That is a mitigation, **not** the fix. The sentence in `qa.md` is still wrong
until route A or B is taken, and step 86.79 should be judged on that basis.
