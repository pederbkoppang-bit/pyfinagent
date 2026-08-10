---
name: rec-vocab-class-86-22
description: phase-86.22 recon - 86.20's canonicaliser exists but is behind a MONEY-PATH dark flag; the poisoned artefact is agent_memories not outcome_tracking; similarity CANNOT find a wrong stored lesson (AUROC 0.59)
metadata:
  type: project
---

Four measured facts that a 86.22 executor will otherwise get wrong.

**1. The shared normaliser already exists BUT IS DARK, and its flag is a money-path flag.**
`backend/services/recommendation_vocab.py` (92 lines) is exactly the canonicaliser 86.22 asks
for -- but its only production consumer, `portfolio_manager.py:116`, reads it behind
`paper_recommendation_vocab_fix_enabled`, declared at `settings.py:214` as *"DARK until operator
promotion"*. That flag's own description justifies darkness on TRADING grounds ("ARMING THIS
CHANGES BEHAVIOUR ON BOTH SIDES"). None of 86.22's consumers place orders. **Reusing that flag
would block a learning-corpus repair on an unrelated trading decision** (Fowler: never tie
unrelated behaviours to one toggle).

**2. `directionally_correct` IS NEVER PERSISTED.** `outcome_tracker.py:70` puts it in the returned
dict, but the save call at `:74-83` passes 8 fields and `bigquery_client.py:400-414 save_outcome`
builds its row from exactly those 8. **`outcome_tracking` is CLEAN.** The poison is in
`agent_memories`, via `memory.py:238` (prompt) and `memory.py:250-253` (fallback lesson), persisted
by `bigquery_client.py:503-516`. Aim any remediation query at the right table.

**3. Reflections only exist where a MODEL was passed.** `outcome_tracker.py:147` gates on
`if self._model:`. `outcome_tracker.py:213` (`evaluate_recent`) passes none -> generates nothing.
`autonomous_loop.py:3392` passes `model=model_client` -> **the live daily cycle DOES poison.**

**4. You cannot find the bad lessons by semantic search.** MemStrata (arXiv:2606.26511) measured
cosine AUROC **0.59** separating contradictions from duplicates -- contradictions are *more*
similar (0.812) than real duplicates (0.800) because a value-flip is a minimal edit; max precision
at any threshold 0.67. The handle must be structural. One EXISTS: `memory.py:250-253` writes the
literal `"Incorrect call on {ticker}. Recommended {rec}, actual return {x}%"`, so LLM-FALLBACK
lessons are deterministically greppable. LLM-GENERATED lessons are not -- `save_agent_memory`
stores no recommendation and `build_situation_description` (`memory.py:170-210`) omits it, so those
need an approximate `(ticker, created_at)` join to `analysis_results`. Say which half you measured.

**Bonus false-positive control for the criterion-2 derivation method:**
`skill_optimizer.py:244` looks like the same defect but reads `debate_consensus`, which descends
from the schema-ENFORCED `Literal` at `schemas.py:95` -- underscore is CORRECT there. A method that
flags it is over-broad. `slack_bot/formatters.py:169` already handles both dialects.
`conflict_detector.py:131` is the nastiest site: a missed `STRONG BUY` falls through to
`elif "BUY" in rec_label` and is graded against the WEAKER 5.5 threshold.

**Why:** 86.22's contract has to decide flag scope and corpus remediation, and both decisions are
easy to get wrong from the step text alone (which does not mention the dark flag at all).
**How to apply:** before proposing "gate it like 86.20 did", check whether the consumer moves
money; before proposing "search the corpus for wrong lessons", remember AUROC 0.59.

Related: [[project_rec_vocabulary_86_20]], [[project_decision_input_integrity_61_2]].
Brief: `handoff/current/research_brief_86.22.md`.
