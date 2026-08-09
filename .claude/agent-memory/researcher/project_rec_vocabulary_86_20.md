---
name: rec-vocabulary-86-20
description: Two disjoint recommendation dialects in-repo; .upper() folds case not separator, so STRONG_* dies at the buy AND sell gates; phase-61.2's fix is defeated by the same mismatch
metadata:
  type: project
---

Step 86.20 research. The repo carries **two disjoint recommendation dialects**
and the money gate only speaks one of them.

- Producer (full pipeline): `backend/agents/schemas.py:40` --
  `action: str = Field(description="Strong Buy, Buy, Hold, Sell, or Strong Sell")`.
  A plain `str`; the vocabulary lives only in the **prompt description**, so
  nothing can reject it. Emits SPACED TITLE CASE.
- Consumer (money gate): `backend/services/portfolio_manager.py:63`
  `_BUY_RECS = {"BUY","STRONG_BUY"}` (UNDERSCORE), tested after `.upper()` at
  `:140/:141/:182`.

**`.upper()` folds CASE but never the SEPARATOR.** `"Strong Buy"` -> `"STRONG BUY"`
-> fails `:188` `if rec not in _BUY_RECS: continue`, **with zero logging** (its
neighbours at `:355/:375/:397/:415/:433` all log their skip reason). Plain
`"Buy"` passes by accident, so the bug **selectively destroys the highest-
conviction signal and inverts the conviction ordering reaching the book.**

Three things I nearly missed and would miss again:

1. **The sell side is broken too, with opposite polarity.**
   `"STRONG SELL"` is in neither `_SELL_RECS` (`:59`) nor `_DOWNGRADE_RECS`
   (`:61`), so a full-path Strong Sell on a held position exits only via
   stop-loss. Arming the buy side is fail-DANGEROUS (more entries on a live
   book); arming the sell side is fail-SAFE. They need separate flags.
2. **phase-61.2's fix is defeated by the same mismatch.**
   `backend/services/paper_trader.py:447-457/:488/:512` persists
   `analysis_recommendation` **verbatim, unnormalised**, so `old_rec` at
   `portfolio_manager.py:141` becomes `"STRONG BUY"` and `signal_downgrade`
   (`:154`) stays dead even with `paper_position_recommendation_fix_enabled` ON.
   Fixing the read boundary without the WRITE path leaves 61.2 dead.
3. **The class is >=8 sites across 4 conventions**, not just portfolio_manager:
   underscore-equality (`api/portfolio.py:138-142`, `bias_detector.py:119/128/153`),
   spaced-equality with NO fold at all (`outcome_tracker.py:57`,
   `agents/memory.py:229`), **substring** (`conflict_detector.py:121/131` -- where
   `"STRONG BUY"` fails the `STRONG_BUY` test but PASSES `"BUY" in rec_label`,
   and `"STRONG_SELL"` contains `"SELL"`), and both-spellings
   (`slack_bot/formatters.py:169`, the only site that ever handled both).

`backend/agents/schemas.py:95` already has
`consensus: Literal["STRONG_BUY","BUY","HOLD","SELL","STRONG_SELL"]` -- proof the
underscore dialect is schema-enforceable here; `action` just never got it.
`backend/api/models.py:21-26` is the trap generator: the enum MEMBER is
`STRONG_BUY` but its VALUE is `"Strong Buy"`.

No recommendation canonicaliser exists anywhere in the repo (URL/text/ticker/
date/model-name ones do). No dedicated `decide_trades` test module exists, and
`backend/tests/test_dod4_tier1_coverage_investment.py:884` feeds `"STRONG_BUY"`
-- **the fixtures use a dialect production never emits**, so the suite is green
against an impossible input. See [[reference_vacuous_type_guards_on_bq_string_columns]].

**Why:** normalising MORE aggressively is exactly how an over-permissive money
gate is born (IETF draft-thomson-postel-was-wrong-02); the fold must be a total
function over a finite table, never a substring predicate. CWE-180: canonicalise
BEFORE validating, **once**, and don't decode the same input twice -- 11
scattered `.upper()` calls ARE that anti-pattern.

**How to apply:** before touching this class, MEASURE whether the full path is
even producing rows today -- the lite analyzers (`autonomous_loop.py:2467` gate,
prompts at `:2835`/`:3159`) emit only canonical `BUY/SELL/HOLD`, so the defect is
LATENT unless full-path/orchestrator rows exist. Then fix producer (Literal on
`schemas.py:40`) AND boundary AND write path, default unknown to NOT-a-buy, and
make the skip a counted, logged rejected-reason. Related:
[[project_decision_input_integrity_61_2]], [[feedback_measure_dont_assert_claims]].
