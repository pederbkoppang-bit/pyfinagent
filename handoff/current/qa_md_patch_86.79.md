# APPLIED at cycle 4 (commit 9b4d5281) — the `qa.md` corrections owed by step 86.79, retained as the HISTORICAL RECORD

**Status: APPLIED by a fresh executor at cycle 4 (commit 9b4d5281; `git diff --stat 9b4d5281^ 9b4d5281 -- .claude/agents/qa.md` => 1 file changed, 116 insertions(+), 45 deletions(-)).** *(cycle-5 correction: the original 'deliberately not applied' framing was true through cycle 3 and falsified by that commit -- caught by the cycle-4 Q/A running this file's own stated verification command. My first fix attempt targeted a hyphen variant of this title, missed, and shipped a GENERATE claiming the fix had landed -- caught by my own post-commit check this time, fixed in the same hour.)*
**Scope: TWO sites (`:622` FALSE, `:645` STALE), one optional (`:692`), one
deliberately left alone (`:713`).** This file said "one-line" through cycles 1–2;
that was wrong twice over and the enumeration below replaces it.

`.claude/agents/qa.md` already carries **four Main-authored edits awaiting
operator review** under CLAUDE.md's separation-of-duties rule
(*"the same Claude Code session should not both author an agent `.md` change AND
self-evaluate work that depends on it"*). Applying this would make it **five**
and deepen a hold the operator owns. The operator instruction for this session is
explicit: *"If a fix genuinely needs `qa.md`, stop and ask."* The masterplan step's
own notes say the same: *"prefer changing `qa_wip.py`, or hand it to a fresh
executor."*

*(cycle-6 REPLACEMENT of the sentence that stood here -- 'Nothing in `.claude/agents/qa.md` was modified by step 86.79 -- verify with `git diff --stat .claude/agents/qa.md`'. That was FALSE twice over (9b4d5281 applied the correction at cycle 4, +116/-45; 2dbe09d4 landed the prior_attempts operand at cycle 5), and the offered command was VACUOUS -- a working-tree diff on a committed tree can never dissent. The verifying command that CAN fail: `git log --oneline -- .claude/agents/qa.md | head -5` shows both commits. This is the file's third correction; each prior one fixed the headline and left this sentence.)*
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

---

## ⚠ CYCLE-3 CORRECTION — inside `qa.md` the class is FOUR sites, not one

The cycle-2 Q/A found a **second** stale `qa.md` site (`:645`) that this file did
not enumerate — after cycle 2 had just been corrected for enumerating one member of
a two-member class. So the enumeration was run properly this time, over the whole
file rather than over the sites someone had named:

```
$ grep -n 'records_retained' .claude/agents/qa.md
622:  `records_retained` is the count of prior Q/A spawns on this step — the
645:  > if `records_retained` (auto) **>** the ledger's verdict count, **the ledger
692:  the WIP sink does not exist, so `records_retained: 0` means **the counter
713:  Measured: `qa_wip.py 86.33` returns `records_retained: 3` and lists
```

**Four sites, and they are not all the same kind of problem.** Classifying rather
than pooling them, since three of the four are not false:

| site | text | classification | needs changing? |
|---|---|---|---|
| **:622** | *"the count of prior Q/A spawns … the **attempt number**"* | **FALSE, and false on both halves.** It is `len(records)`, INCLUSIVE of the current spawn | **YES — this is the defect** |
| **:645** | *"if `records_retained` (auto) **>** the ledger's verdict count, the ledger is STALE"* | **STALE, not false.** The comparison still works today, but it is the one whose `qa-verdict.js` counterpart (`:165`) was corrected to `attempt_number`, so the two rails now disagree | **YES — for consistency** |
| **:692** | *"`records_retained: 0` means the counter has no input, not 'this is attempt 1'"* | **ACCURATE.** `records_retained` really is 0 there. Superseded only in the sense that `attempt_number` now fails closed automatically | optional |
| **:713** | *"Measured: `qa_wip.py 86.33` returns `records_retained: 3`"* | **A HISTORICAL MEASUREMENT**, true when taken | **NO — rewriting a dated measurement would be falsifying a record** |

**So the operator-gated work is 2 sites (`:622`, `:645`), with `:692` optional and
`:713` deliberately left alone.**

---

## The divergence (site `:622` — the FALSE one)

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

---

## Site `:645` — the second gated change (cycle-3 addition)

Current:

```
  > if `records_retained` (auto) **>** the ledger's verdict count, **the ledger
  > is STALE** — say so in `notes` and treat the sequence as unreliable.
```

Proposed:

```
  > if `attempt_number` (auto) **>** the ledger's verdict count, **the ledger
  > is STALE** — say so in `notes` and treat the sequence as unreliable. If
  > `attempt_number` is `null`, the comparison CANNOT be made: say
  > `sequence: UNKNOWN` rather than substituting `records_retained`, which
  > counts files and can be lowered by pruning.
```

**Why it matters even though the current text is not false.** Its counterpart in the
launch rail — `.claude/workflows/qa-verdict.js` — **was** corrected to
`attempt_number` in cycle 2. Leaving `qa.md` on `records_retained` means the two
files the Q/A reads now prescribe **different comparisons for the same decision**,
and under pruning they give different answers. That is the same
two-sources-one-fact hazard this whole step is about.

## Site `:692` — OPTIONAL, listed for completeness

Accurate as written. `attempt_number` now fails closed on that path by itself
(`attempt_number_status = source_missing`), so the paragraph is belt-and-braces
rather than wrong. Adding a pointer to `attempt_number_status` would be an
improvement, not a correction.

## Site `:713` — DELIBERATELY NOT CHANGED

A dated measurement (`qa_wip.py 86.33` returned `records_retained: 3`) that was true
when taken. Rewriting it would falsify a record. It is listed here so that the
enumeration is complete and nobody has to re-derive the class a fourth time.
