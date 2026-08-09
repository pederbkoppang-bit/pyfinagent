# Contract -- phase-86.22

**Step:** 86.22 (P1) -- the recommendation-vocabulary split is cross-module, and
on the learn-loop side it mislabels rather than drops.
**Date:** 2026-08-10. **Cycle:** 200.
**Research gate:** PASSED -- `handoff/current/research_brief_86.22.md`
(run `wf_e6a9d91d-dda`: 13 sources read in full >= floor 5, 50 URLs >= floor 10,
recency scan performed, audit-class `coverage.dry` after 10 rounds with 2 dry,
18 internal files inspected; 40,357 chars independently re-read, all 13 claimed
URLs verified present).

> **Gate-run disclosure.** The FIRST attempt (`wf_81c3d321-7b8`) researched for 68
> tool calls / 212K tokens and then **dropped its return without calling
> StructuredOutput** -- a FAILED gate, never a pass. That is the SECOND such drop
> tonight (86.6 was the first); both were audit-class runs with long prompts.
> Write-first meant an 832-line brief was already on disk both times. The re-run
> was lean. **It did not merely verify -- it re-researched and produced a
> SHORTER, partly DIFFERENT brief** (13 sources / 40,357 chars, against the first
> run's 15 / 62,165). The gate passed on what is on disk NOW, which is what the
> independent verifier read.

---

## 1. Research-gate summary

**The premise holds, and the mechanism is confirmed at source.**
`outcome_tracker.py:57-58` does
`is_buy = recommendation in ("Strong Buy", "Buy")` with **no case folding**, and
it reads `report["recommendation"]` from `get_recent_reports`, whose FROM clause
resolves through settings to the same `financial_reports.analysis_results` that
86.20 measured. So the 91 literal `BUY` rows match neither `is_buy` nor
`is_sell`, and `directionally_correct` is False for all of them regardless of
return. `memory.py:228-251` carries the identical expression and writes
"Directionally correct: YES/NO" into a reflection.

**FOUR findings that change the design, and the first two change the SEVERITY.**

**(1) There is nothing to backfill today. MEASURED BY ME, not inherited:**

```
financial_reports.agent_memories   : rows=0
financial_reports.outcome_tracking : rows=3
grep directionally_correct -> outcome_tracker.py:59,70 ONLY (never in bigquery_client.py)
```

`directionally_correct` is **never persisted** -- `save_outcome` does not carry
it and the table has no such column, so `outcome_tracking` is CLEAN. The wrong
label's only durable path is LLM-generated `lesson` prose in `agent_memories`,
and that table is **empty**.

**The two gate runs disagreed here and the reconciliation matters.** The first
said the reflection writer "has never fired"; the second said the path "is LIVE".
Both are partly right and I verified the pieces myself: the writer **is wired**
(`autonomous_loop.py:3392` passes a model, with a fail-open on construction
failure) and fires only `if self._model:` (`outcome_tracker.py:147`) -- so the
path is **live-capable** -- while the table is **measured empty**, so nothing has
actually been written. **The correct deliverable is therefore to fix the label
BEFORE the writer produces its first row**, plus a re-runnable proof of the zero
rather than an assertion of it.

**(2) Do NOT reuse 86.20's flag.** `recommendation_vocab.py` exists but its only
consumer reads it behind `paper_recommendation_vocab_fix_enabled`, a flag
justified on MONEY-path grounds. **None of 86.22's consumers place orders.**
Coupling a corpus/metric repair to a trading toggle means the repair cannot ship
until an unrelated trading decision is made -- Fowler's explicit warning against
one toggle for unrelated behaviours. **Reuse the CANONICALISER; do not reuse the
FLAG.**

**(3) Similarity cannot find wrong lessons** if any are ever written: AUROC 0.59,
and contradictions score as *more* similar than duplicates (arXiv 2606.26511).
The only deterministic handle is the fallback lesson literal at
`memory.py:250-253`. Supersede/invalidate beats purge and beats leave-in-place.
Relevant only if the corpus stops being empty.

**(4) Derivation controls, both directions.** `skill_optimizer.py:244` must NOT
be flagged -- it reads the schema-enforced `Literal` at `schemas.py:95` and is
correct by construction; flagging it is a false positive that would train a
reader to ignore the method. And `conflict_detector.py:131` is the worst TRUE
positive: a missed `STRONG BUY` falls through to the substring `elif "BUY"` and
is graded at the **weaker 5.5 threshold** rather than skipped.

**86.20 chose read-side normalisation over constraining the producer at
`schemas.py:40`. Do not reverse that here** -- two steps making opposite
decisions about the same field is the defect this family keeps producing.

## 2. Hypothesis

One canonicaliser, applied at every consumer of the recommendation field, makes
every spelling of an intent reach the same classification -- correcting the
learn-loop label before the corpus it would poison has any rows, and correcting
the analytics and bias consumers that currently drop the producer's own spelling
-- without widening any set and without coupling the repair to a money-path flag.

## 3. Immutable success criteria (copied VERBATIM from `.claude/masterplan.json`)

1. "The defect is REPRODUCED FIRST and recorded verbatim for BOTH directions of the split: a title-case consumer (outcome_tracker.evaluate_recommendation) is driven with the literal 'BUY' and shown to yield directionally_correct=False even when the return is positive, AND an UPPER_SNAKE consumer is driven with the literal 'Strong Buy' and shown to miss"
2. "The population of affected call sites is DERIVED by a stated, re-runnable method rather than hand-listed, and the method is PROVEN not to produce a known false negative -- run it and show it flags backend/services/outcome_tracker.py:57 and backend/agents/bias_detector.py:119; a method that reports either of those clean is rejected regardless of what else it finds"
3. "The recommendation population is RE-DERIVED at fix time from the analysis_results column that outcome_tracker actually reads (prove the table resolution, do not assume it) and recorded: every distinct value, its count, its genuine count, and for EACH consumer whether that value is matched, missed, or correctly excluded"
4. "ONE shared normalisation is used by every fixed consumer. If 86.20 has landed a normaliser this step REUSES it -- assert in a test that no second, independently-defined recommendation vocabulary or normaliser exists in backend/ outside the shared one"
5. "NO STRING THAT IS NOT THAT INTENT IS ADMITTED. Assert per consumer that 'HOLD', 'Hold', 'Sell', 'N/A', '' and None still produce the same non-buy outcome they produce today, and that a non-directional value never becomes directionally_correct=True"
6. "The corrected directionally_correct is measured as a BEFORE/AFTER DELTA on the real column population -- state how many rows change verdict and in which direction, rather than asserting the fix works"
7. "Whether any WRONG reflection has already been persisted to agent memory is MEASURED and reported either way; if any exist, say plainly whether this step repairs them or defers that, and do not silently leave the question open"
8. "MUTATION-TEST every new guard, including reverting the normalisation at each fixed site individually, and confirm each mutant is killed by the assertion that names it -- a guard whose mutant survives does not count"

**Verification command (immutable):**
`bash -c 'source .venv/bin/activate && python -m pytest backend/tests/ -q -k "outcome_tracker or bias_detector or conflict_detector or portfolio_manager"'`
(baseline measured 2026-08-09 BEFORE any code: **23 passed, exit 0**)

**live_check (immutable):** "live_check_86.22.md with: verbatim reproduce-then-fix output for BOTH directions (a title-case consumer missing 'BUY' and an UPPER_SNAKE consumer missing 'Strong Buy'); the derived call-site population with the outcome_tracker:57 and bias_detector:119 false-negative check shown; the re-derived per-value matched/missed table per consumer; and the before/after directionally_correct delta."

## 4. Design

- **Reuse `backend/services/recommendation_vocab.py`.** Criterion 4 requires
  asserting no second vocabulary survives; that assertion must be a test, not a
  claim.
- **A SEPARATE flag, or none.** Argue it in the artifacts. The consumers are
  evaluation, analytics and bias-detection -- none places an order.
- **The derivation's recall is validated in BOTH directions** (criterion 2):
  `outcome_tracker.py:57` and `bias_detector.py:119` must be FLAGGED; a method
  reporting either clean is rejected. And `skill_optimizer.py:244` must NOT be
  flagged -- a false positive is a failure too.
- **The zero is PROVEN, not asserted** -- a count query that anyone can re-run,
  because "there is nothing to backfill" is exactly the kind of convenient claim
  that must carry its command.
- **`conflict_detector`'s substring matching is its own hazard**: `STRONG_SELL`
  contains `SELL`. Fixing it means exact membership on the canonical token, not
  a longer chain of `in` tests.

## 5. Plan

1. **[done]** Research gate PASSED.
2. **[this file]** Contract, BEFORE any code.
3. Reproduce FIRST, BOTH directions (criterion 1): a title-case consumer missing
   `'BUY'`, and an UPPER_SNAKE consumer missing `'Strong Buy'`.
4. Derive the consumer population; validate recall on the two known positives AND
   the known negative before building on it.
5. Re-derive the per-value/per-consumer matched/missed table (criterion 3).
6. Migrate the consumers to the shared canonicaliser; assert no second vocabulary.
7. Negative cases (criterion 5) and the before/after `directionally_correct`
   delta (criterion 6).
8. Measure and report what is already persisted (criterion 7) with a re-runnable
   count.
9. Mutation-test every new guard (criterion 8), in-memory, digests asserted.
10. Q/A; transcribe verbatim; log; flip.

## 6. Traps

- **Do not reuse 86.20's money-path flag** for a non-money-path repair.
- **Do not flag `skill_optimizer.py:244`** -- schema-enforced and correct.
- **Do not reverse 86.20's read-side decision.**
- **Do not claim a lost trade or lost P&L** -- this is an evaluation path.
- **Do not assert the corpus is empty** -- prove it with a query, every time.
- **Do not fix `conflict_detector` by adding another substring test.**
- **Guard the CLASS.** Tonight's repeated lesson: when a call site is found
  guilty once, enumerate every position at it, not the one that was found.

## 7. References

- `handoff/current/research_brief_86.22.md` (13 read in full, 50 URLs).
- CWE-180 -- https://cwe.mitre.org/data/definitions/180.html
- Anti-corruption layer -- https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer
- Fowler, Feature Toggles -- https://martinfowler.com/articles/feature-toggles.html
- Parse, don't validate -- https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/
- OWASP Input Validation -- https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- Refinitiv I/B/E/S -- https://research2.fidelity.com/fidelity/research/reports/release2/Research/RefinitivIBES.asp
- arXiv 2606.26511 (similarity cannot separate contradictions) -- https://arxiv.org/html/2606.26511v1
- Internal: `outcome_tracker.py`, `memory.py`, `bias_detector.py`,
  `api/portfolio.py`, `conflict_detector.py`, `recommendation_vocab.py`,
  `agents/schemas.py`, `skill_optimizer.py`.
