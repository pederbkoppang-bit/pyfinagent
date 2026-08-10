---
name: unknown-direction-86-25
description: 86.25 -- the live degenerate is 100% HOLD not an APPROVE/BUY mismatch; the 3 SELL rows are attributable by NULL-constant columns + microsecond timestamp identity; is_directional was already built and uncalled
metadata:
  type: project
---

Phase-86.25 (`evaluate_recommendation` fed a risk-approval token). Measured 2026-08-10.

**The step's framing is right but the LIVE symptom is different from the stated one.**
`autonomous_loop.py:3394-3399` only ever reads `action=='SELL'` rows, and **32/32 SELL rows carry
`risk_judge_decision = ''`** (BQ-measured). So `.get("risk_judge_decision","HOLD")` at `:3412`
returns `''` -- the default never fires, because `.get`'s default only fires on a MISSING KEY, not a
present-but-empty one -- and the coercion at **`:3416-3417`** (not `:3414-3415`, which is the
comment) rewrites it to `HOLD`. `HOLD` is in neither `BUY_INTENT` nor `SELL_INTENT`, so
`directionally_correct` is `False` for 100% of closed trades. The APPROVE/BUY vocabulary mismatch is
LATENT: it arms the moment a SELL row carries a populated `risk_judge_decision` (19 BUY rows
already do). **Never describe this as "APPROVE_REDUCED reaches the canonicaliser" without saying
that today it does not.**

**The canonicaliser is NOT the bug.** `recommendation_vocab.is_buy_intent` returning False on an
unrecognised token is documented, deliberate behaviour. The bug is recording that `False` as if it
were a measured judgement. **`recommendation_vocab.is_directional` (`:133-141`) already exists,
its docstring names exactly this defect, and it has ZERO production callers** -- the seam is
pre-built; do not add a second helper (the module's own docstring warns that two canonicalisers that
disagree is the same defect wearing a hat).

**Exactly ONE production call site mismatches.** `evaluate_all_pending`
(`outcome_tracker.py:144-149`) passes `analysis_results.recommendation` -- the CORRECT vocabulary.
The class does not generalise across callers of the function; it DOES generalise across writers of
the column (`nightly_outcome_rebuild.py:67` makes the same move).

**PROVENANCE METHOD worth reusing -- attributing rows with no writer column.** The 3
`outcome_tracking` rows spelled `SELL` came from `nightly_outcome_rebuild._compute_outcomes:67`
(`t.get("risk_judge_decision") or t.get("action")` -- the `or` fires because the value is `''`).
Four independent evidence lines, and the two generalisable ones are:
- **Columns whose NULL-ness is a writer's CONSTANT.** `build_outcome_row` hardcodes
  `price_at_recommendation: None` and `beat_benchmark: None`; the rival writer
  (`BigQueryClient.save_outcome`) can produce neither. Two independently-NULL columns is a signature.
- **Timestamp GRANULARITY reveals write SHAPE.** All 3 rows share an identical *microsecond*
  `evaluated_at`. A per-row writer stamps `now()` inside each row dict; a batch writer computes
  `now_iso` once and fans it out. Identical microseconds = batch writer, decisively.
Inadmissible and I said so: row ordering, `git log` alone, "the value is an action and actions
exist". Also: `outcome_tracking` has **no** `directionally_correct` column at all (9 columns), so a
tri-state is an ADDITIVE nullable migration, not a rewrite.

Related: [[project_rec_vocabulary_86_20]], [[project_rec_vocab_class_86_22]],
[[project_outcome_write_82_48]].
