---
name: vocab-boundary-86-63
description: Step 86.63 recommendation-vocabulary boundary -- the count is 6 filed steps not 5, only 2 of ~25 write seams are guarded, and the ROOT is authored in the LLM .md prompts where no Python guard can reach
metadata:
  type: project
---

Step 86.63 ("ONE guard at the recommendation-vocabulary boundary"). Research gate PASSED
2026-08-13, audit-class, 7 rounds, dry at 6-7. Brief:
`handoff/current/research_brief_86.63.md`.

**Why:** the class ("a field receives a value from a DIFFERENT vocabulary than its readers
assume") had been patched five times at five sites and a sixth kept appearing. The gate was
asked to derive the real denominator rather than accept the claimed one.

**How to apply:** these five findings are the ones a future executor will otherwise re-derive
or get wrong.

1. **The claimed count of five is wrong in BOTH directions.** Filed masterplan steps in the
   class are **SIX** -- the caller's list (86.22/86.25/86.40/86.52/86.58) omits **86.20**, the
   founding instance that created `recommendation_vocab.py`. But the step count is the wrong
   unit anyway: the CODE surface is **~25 write seams and ~11 read seams**, of which only
   **2 writes are guarded** (both `resolve_outcome_recommendation`, both from 86.25). Count the
   seams, not the steps.

2. **The ROOT is in the LLM prompt files, not in Python, and no Python guard can reach it.**
   `backend/agents/skills/synthesis_agent.md:19,:82,:163` instructs the SPACED dialect
   (`Strong Buy`) *in a field literally named `action`*; `moderator_agent.md:18,:101` instructs
   the UNDERSCORED dialect (`STRONG_BUY`). That is why so many Python sites write
   `"recommendation": analysis["action"]` and look locally sane. A write-side guard DETECTS
   this drift loudly; it cannot prevent it. Queue the prompt fix separately.

3. **`paper_positions.recommendation` is NULLABLE; `outcome_tracking.recommendation` is
   REQUIRED.** (`scripts/migrations/migrate_paper_trading.py:51` vs
   `scripts/migrations/migrate_bq_schema.py:126`.) So **86.25's core argument -- "SQL NULL is
   unavailable, therefore the missing-data marker must be a string" -- DOES NOT TRANSFER to
   paper_positions.** Copying the `"UNKNOWN"` string sentinel across by analogy is reasoning
   from the wrong table.

4. **A `save_paper_position` guard leaves a laundering path open.** The seam is
   `paper_trader.py:452` (`_pos_rec = reason`) -> `:488`/`:512` -> `save_paper_position`, but
   `:676` re-emits `position.get("recommendation","")` onto a **trade** row, which a
   position-write precondition never sees. Also `bigquery_client.py:663` `str()`-coerces every
   non-numeric value, so a guard above it that only checks `isinstance(v, str)` is undone by it.

5. **There is genuinely no rival canonicaliser** (swept with a positive control:
   `def canonical_recommendation` count = 1). The work is COVERAGE, not a new module. But the
   frontend re-implements the vocabulary in 4 `.tsx` files with `string`-typed fields
   (`frontend/src/lib/types.ts:126,:152,:283,:653`) and two SUBSTRING tests -- structurally
   unreachable from Python. Say so rather than claiming the class is closed.

**Unresolved, do not record as a zero:** `autonomous_loop.py:2514` gates
`action in ("BUY","SELL","HOLD")` while `synthesis_agent.md` instructs five values none of
which are members. I could not find the enclosing function's call site and had NO positive
control for that search -- so it is UNRESOLVED, not absent. Settle it by DRIVING the code.

See [[research-gate-discipline]], [[dead-sell-rule-86-58]], [[rec-vocabulary-86-20]].
