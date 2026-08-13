# Experiment results — step 86.63 (PARTIAL)

**Step:** 86.63 — put ONE guard at the recommendation-vocabulary boundary
**Date:** 2026-08-14
**Contract:** `handoff/current/contract_86.63.md`  |  **Gate:** PASSED (`wf_667c754e-635`)

> **PARTIAL AND SAYING SO UP FRONT.** Criteria **1, 2 and 5** are addressed here by
> measurement and analysis. Criteria **3, 4 and 6 are NOT done** — each requires a guard
> to exist in production code, on the live trade path. **No file under `backend/` was
> modified.** This is not a claim the step is ready to close; it is the half that can be
> established without writing to the money path.

---

## Criterion 1 — class membership RE-DERIVED, and the criterion's own phrase is the blocker

**Population rules, stated because the counts are meaningless without them:**

- **WRITE seam** = an assignment into a field named `recommendation` / `*_rec` /
  `consensus` (regex `(recommendation|_rec|consensus)[a-z_]*\s*=\s*[^=]`), excluding
  tests and comparisons.
- **READ seam** = a membership test against a recommendation vocabulary
  (`in _BUY_RECS` / `_SELL_RECS` / `_DOWNGRADE_RECS`, `canonical_recommendation`,
  `is_buy_intent`), excluding tests.
- Scope: `backend/**/*.py`, `/usr/bin/grep` pinned.

**Measured:**

| Quantity | Value |
|---|---:|
| write-seam hits | **50** |
| read-seam hits | **30** |
| **distinct files containing a write seam** | **19** |
| **distinct files containing a read seam** | **1** |
| files doing both (intra-module, NOT a crossing) | 1 — `portfolio_manager.py` |

**Positive control:** the probes independently rediscover the two seams 86.58 already
established — the write at `paper_trader.py` (`_pos_rec = reason`) and the read at
`portfolio_manager.py` (`old_rec in _BUY_RECS`).

### The disagreement, and why I am not resolving it with a number

The research gate reported **~25 write seams vs ~11 read seams**. I measure **50 / 30**.
**I am not adopting either figure**, because the criterion asks for fields carrying a
recommendation-ish string **"across a module boundary"** — and **that phrase is not
operationalised anywhere.**

My rule counts *every* assignment, including same-file write-then-read, which is **not**
a boundary crossing. So **50 is the wrong population for this criterion**, by my own
rule's admission. The gate's narrower figure may be right, but its rule was not stated,
so the two cannot be reconciled.

**This is the finding, not a failure to count:** criterion 1 cannot be satisfied by any
number until "across a module boundary" has an operational definition. Whoever closes
this step must state one — *different file? different package? crosses a persistence
boundary?* — and derive the count from it. **A count with no membership rule is not
evidence**, which this project has now paid for repeatedly.

---

## Criterion 2 — placement, and the topology decides it

**19 writers, 1 reader.** That asymmetry is the whole argument.

`backend/services/recommendation_vocab.py` already exists and, per 86.58's gate, guards
the **read** side. But read-side guarding covers **one** site and leaves **nineteen**
producers free to write anything. It can only ever **detect** corruption at the end of the
pipe, never **prevent** it entering.

**So ONE guard belongs on the WRITE side**, and the external evidence agrees for an
independent reason: arXiv:2607.01711v1 (75 incidents 2014–2025; interpretation failures
59/75 = **78.7%**) prescribes controls *"at the boundary where gaps originate, not merely
where they become visible."* Origin is `paper_trader.py:452`; visibility is
`portfolio_manager.py:264`. **We have been guarding visibility for six steps.**

**Against N site fixes:** five prior instances were each fixed at their own site and a
sixth appeared — and `recommendation_vocab.py:95-105` predicted exactly that
(*"A caller that unwraps them back into a literal set has undone the point"*), which
`portfolio_manager` then did by importing only `canonical_recommendation` at `:16` and
hand-writing `_BUY_RECS` at `:60-64`.

**But a single boundary does NOT close the class, and the contract must not imply it
does.** 86.63's gate established the root is in the **prompts** —
`synthesis_agent.md:19` teaches the spaced dialect, `moderator_agent.md:18` the
underscored one, and `:7` says the latter feeds the former. **No Python guard reaches a
`.md` prompt.** A write-side guard contains the split; it does not fix its origin.

---

## Criterion 5 — the open members

**Criterion 5's text is STALE**: it names *"the three open members (86.40, 86.52,
86.58)"*, but **86.58 closed PASS on 2026-08-13**. The open set is **86.40 and 86.52** —
**two of six**, since the class also omits **86.20**, its founding instance.

- **86.40** (P3) — a comment blessing the defect 86.25 removed, one file over. **Not
  subsumed**: a write-side guard does not edit a comment. Explicitly excluded.
- **86.52** (P2) — did 86.25's fix actually land? Two tests say not fully. **Possibly
  subsumed** if the guard covers the same seam, but that cannot be asserted without
  reading those two tests, which I did not do. **Recorded as undetermined, not claimed.**

---

## Criteria 3, 4, 6 — NOT DONE, and why

Each requires a guard to exist:

3. fail loudly on an unknown value, **proven by driving one through it**
4. flag-OFF parity **against an oracle**, not two passing examples
6. mutation-test: revert it, show the check goes red, control GREEN first

**Writing that guard changes SELL/BUY behaviour on the live trade path.** It is the right
work; it is not right at the end of a session in which eight Q/A verdicts found asserted
claims and guards that could not fail. **Deferred deliberately to a fresh session**, with
the traps recorded in the contract so the next executor hits them before the code.

---

## Scope honesty

**No file under `backend/` modified — and here is what that check can and cannot see.**

```
git status --porcelain -- backend/    ->  0 lines
git status --porcelain -- handoff/    -> 17 lines   [POSITIVE CONTROL: probe is live]
```

The control matters: an empty result from a probe that cannot return anything is worth
nothing. This one returns 17 on a path that has changed, so the 0 is a real 0.

**Disclosed blind spot** — `git status` is blind to gitignored files, and
`git check-ignore -v backend/.env` → `.gitignore:5:.env`. **This is precisely the guard
that failed 86.9 last night**, so it does not get to stand alone here. Measured directly
instead:

```
backend/.env  mtime  2026-08-13T20:33:27Z   bytes=6121
```

That timestamp is **yesterday** — it is 86.9's mutation, unchanged since. **This session
did not write `.env`**, established by the mtime, not by the blind check.
- The **50/30** figures are **rejected by their own author** as the wrong population for
  this criterion; **19 writers / 1 reader** is the defensible measurement.
- **86.52's subsumption is undetermined**, not "excluded" — I did not read the two tests.
- The immutable command proves only that `portfolio_manager.py` parses; it is evidence for
  no criterion here.
