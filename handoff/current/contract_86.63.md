# Contract — step 86.63

**Step:** 86.63 — put ONE guard at the recommendation-vocabulary boundary instead of a sixth patch at a sixth site
**Priority:** P2  |  **Status at contract time:** pending
**Date:** 2026-08-14 (authored ~00:25 CEST)

---

## Research gate — PASSED

Run `wf_667c754e-635`, tier `moderate`, **audit-class**. `gate_passed: true`,
recomputed by the script, **no rail drop**. 10 sources read in full (floor 5), 39 URLs
(floor 10), `urls_collected` corroborated 39 <= 39 distinct in the brief, all 10 claimed
sources present, recency section present, `brief_status: COMPLETE`. Completeness critic
ran **7 rounds, dry at 6–7** (K=2 satisfied).
Brief: `handoff/current/research_brief_86.63.md` (53,166 chars).

---

## THE GATE CHANGED THE STEP. Read this before planning any guard.

**The vocabulary split is AUTHORED IN THE PROMPTS, not in Python.** Verified by me
directly, not adopted from the researcher's summary:

```
backend/agents/skills/synthesis_agent.md:19
  "Recommendation values: Strong Buy / Buy / Hold / Sell / Strong Sell"     <- SPACED
backend/agents/skills/moderator_agent.md:18
  "Consensus values: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL"          <- UNDERSCORED
backend/agents/skills/moderator_agent.md:7
  that output "feeds directly into Synthesis Agent (Step 11) as the backbone
   recommendation"
```

Two dialects, written in prose, in two prompt files, **and one feeds the other**.
**No Python guard reaches a `.md` prompt.**

**Consequence for this step, and it must be stated in the artifacts rather than
implied away:** a boundary guard in Python can *detect* the split and *stop it
propagating*. It cannot *fix the origin*. Shipping a guard and calling the class
closed would be the seventh costume. The honest deliverable is a guard that fails
loudly **plus** an explicit statement that the prompt-level divergence is the root and
is not fixed by this step.

---

## Corrections to the step's own audit_basis, both derived

1. **The class is SIX, not five.** The basis lists 86.22 / 86.25 / 86.40 / 86.52 /
   86.58 and **omits 86.20**, the founding instance (verified: `status: done`, P1).
2. **Criterion 5 is STALE.** It names *"the three open members (86.40, 86.52,
   86.58)"*. **86.58 closed PASS on 2026-08-13**, so the open set is **86.40 and
   86.52 — two of six**. Criteria are immutable and will not be edited; the drift is
   disclosed here and must be disclosed to the Q/A.

---

## Seam census (from the brief; to be re-derived in GENERATE per criterion 1)

**~25 WRITE seams vs ~11 READ seams. Only TWO writes are guarded** — both
`resolve_outcome_recommendation` (86.25). Every prior fix landed on the read side or a
single site, which is the mechanism by which a sixth instance kept appearing.

---

## Hypothesis

The defect is a **write-side** absence: values enter recommendation-typed fields with
no parse step. The correct placement is **one guard at the origin seam**
(`paper_trader.py:452`, where an order reason becomes a stored recommendation), not at
the visibility seam (`portfolio_manager.py:264`, where it is finally noticed). External
evidence supports this directly: arXiv:2607.01711v1 (2026; 75 incidents 2014–2025;
Interpretation failures **59/75 = 78.7%**) prescribes controls *"at the boundary where
gaps originate, not merely where they become visible."*

The guard must **fail loudly and never coerce** — RFC 9413's entrenchment argument, and
this project's own history: the 86.20 UNRECOGNISED line is the only reason this class
was ever visible.

---

## Immutable success criteria — copied verbatim from `.claude/masterplan.json`

1. the class membership is RE-DERIVED, not copied from this basis: enumerate every field that carries a recommendation-ish string across a module boundary, state the enumeration command, and report whether the count is five or something else
2. ONE boundary guard is proposed and its placement justified against the alternative of N site fixes; if a single boundary genuinely does not exist, say so and explain what the correct decomposition is
3. the guard FAILS LOUDLY on an unknown value and never silently coerces -- proven by driving an unknown token through it and observing the failure, with 86.20's UNRECOGNISED log line preserved or strengthened, never quieted
4. flag-OFF or pre-change parity is proven: with the guard inert, every existing call site produces byte-identical output, demonstrated against an oracle rather than two passing examples
5. the three open members (86.40, 86.52, 86.58) are each either subsumed by this guard or explicitly excluded with a reason -- a class fix that leaves members unaddressed must say which and why
6. mutation-test the guard: revert it and show the check goes red, with the control observed GREEN first and a byte-identical restore

Immutable verification command:
```
bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/portfolio_manager.py\").read()); print(\"parses\")"'
```
Required live_check: `live_check_86.63.md with the derived class enumeration and a driven demonstration that an unknown vocabulary value fails loudly at the boundary`

---

## Plan

1. **Criterion 1 — re-derive the class.** Enumerate every field carrying a
   recommendation-ish string across a module boundary, state the enumeration command,
   pin `/usr/bin/grep`, pair every zero with a positive control. Report the count and
   whether it is five, six, or something else. **Do not copy the brief's ~25/~11
   figures — reproduce them.**
2. **Criterion 2 — placement.** Justify ONE write-side guard against N site fixes, and
   state plainly that it does **not** reach the prompt-level root.
3. **Criterion 3 — fail loudly.** Drive an unknown token through the guard and observe
   the failure. Preserve or strengthen the 86.20 UNRECOGNISED line; **never quiet it**.
4. **Criterion 4 — inert parity.** With the guard disabled, prove byte-identical output
   at every call site **against an oracle**, not two passing examples.
5. **Criterion 5 — members.** Address 86.40 and 86.52 (the actual open set), and record
   86.20/86.22/86.25/86.58 as closed with whether each is subsumed.
6. **Criterion 6 — mutation-test**, control observed GREEN first, byte-identical restore.

---

## Traps the brief flags — do not trip these

- `paper_positions.recommendation` is **NULLABLE** while `outcome_tracking` is
  **REQUIRED**, so **86.25's sentinel argument does NOT transfer.**
- A `save_paper_position` guard **misses `paper_trader.py:676`**.
- `bigquery_client.py:663` **str()-coerces**.
- The frontend (4 `.tsx`, string-typed) is **structurally unreachable** — out of scope,
  say so rather than claiming coverage.
- The `autonomous_loop.py:2514` contradiction is **UNRESOLVED, not zero.**

---

## Constraints

Paper trading only. **No flag promotions, no `.env` writes, no production behaviour
change without the flag-inert proof of criterion 4.** Do **not** add `new_buy_signal`
to the recommendation vocabulary — that is the symptom. Do **not** quiet the 86.20 log
line. This step is a **P2**; 86.69 (P0) and the harness-correctness steps outrank it on
the operator's stated priority.

---

## References

- `handoff/current/research_brief_86.63.md` — gate PASSED, audit-class, dry
- `backend/agents/skills/synthesis_agent.md:19` / `moderator_agent.md:18,:7` — the prompt-authored root
- `backend/services/recommendation_vocab.py:95-105` — the module that predicted its own next failure
- `backend/services/paper_trader.py:452,:676` — the origin seam and the one a naive guard misses
- `backend/db/bigquery_client.py:626,:638-639,:663` — persistence boundary, precondition precedent, str() coercion
- `handoff/current/evaluator_critique_86.58.md` — the closed sibling; its PASS is what makes criterion 5 stale
