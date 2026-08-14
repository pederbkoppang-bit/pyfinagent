# ADR 0003 -- Attempt-count vs consecutive-CONDITIONAL bounds

Status: accepted
Date: 2026-08-14
Owners: Peder (operator) / Layer-3 harness (Main + fresh executor)
Phase: phase-86.21 (the correction) / phase-86.78 (this relocation)

## Where this came from, and why it moved

The block quoted below lived in `.claude/agents/qa.md`, inside the
prior-attempt/prior-verdict evidence section, immediately after the rule it
corrects. It was written in phase-86.21 and it is accurate.

It moved here in phase-86.78 for one reason: **`qa.md` is read by the Q/A judge
at grading time, and this ADR is not.**

Phase-86.78 removed consequence framing from the judge's instructions, on the
basis of arXiv 2604.15224 -- content held strictly constant, a single
consequence sentence varied across 18,240 judgments, judges measurably LENIENT
in 58 of 72 cells (p<0.001, peak -9.8pp), with reward framing as lenient as
punishment framing. The effect is invisible in chain-of-thought (ERRJ = 0.000),
so a judge cannot detect it acting on itself and cannot compensate for it.

The block below states the superseded rule in full imperative form and walks
through which attempts each competing bound would have force-failed. Framing it
as history does not stop it functioning as instruction to a judge reading the
section top to bottom -- the mechanism 2604.15224 measured is the *presence* of
the consequence at grading time, not its grammatical mood. Deleting it was not
an option either: it is a dated record and the measurement in it is load-bearing
for why the tightest of three disagreeing bounds is no longer live.

So it is preserved unaltered, in a file whose readers -- operators and Main --
are the parties that DECIDE rather than the party that GRADES. The board
recommends; the sponsor decides.

`qa.md` now carries only a pointer to this file, with no summary of the
threshold: a one-line summary of a consequence is still a consequence.

## The record, moved verbatim

Reproduced **byte-for-byte** from `.claude/agents/qa.md` as it stood at the time
of the move (13 lines, 980 bytes,
sha256 `a3596e8ede60a2c7242de5804647de7fb06da4dfcd0667d2ffe4d1b7851f6ebb`). The
leading two-space indent and `>` blockquote markers are qa.md's own list-item
scaffolding and are **deliberately retained** so the bytes match exactly and the
move is provably lossless. Nothing below has been rewritten, condensed, or
tidied.

  > **CORRECTION (phase-86.21, 2026-08-14) — this paragraph previously said
  > "if this would be the third attempt or later, return FAIL", which is a
  > DIFFERENT AND STRICTER RULE than the one CLAUDE.md:371-376 defines, and it
  > was introduced by phase-86.75's counter repoint without anyone deciding it.**
  > Measured against step 36.17's real history `C, F, F, C, C, PASS`: the
  > consecutive rule never fires (longest run = 2), while the attempt-count rule
  > forces FAIL at attempts **4 and 5** — so 36.17 would have been failed twice
  > and never reached the PASS it legitimately earned at attempt 6. It was also
  > stricter than the cumulative budget CLAUDE.md F1b documents (**5** attempts,
  > and that one **escalates to the operator**, it does not auto-FAIL). Two of
  > the three bounds disagreed and the tightest one was live by accident.
  > *Superseded, not annotated: the attempt-count trigger is gone from the rule
  > above, not sitting beside it.*

## Consequences

- The three-way disagreement the block describes is resolved and stays
  resolved: the attempt-count trigger is gone, the consecutive-CONDITIONAL
  threshold and the F1b cumulative budget are both computed **caller-side** in
  `.claude/workflows/qa-verdict.js::enforceEscalation`, after the verdict
  returns.
- `qa.md` retains the *evidence* half in full -- how to obtain the sequence, the
  reset-on-PASS/FAIL semantics, `NO_VERDICT` handling, and the fail-closed
  `None`-never-`0` rule. Only the *consequence* half left.
- Judge-facing invocations of `scripts/qa/verdict_history_86_21.py` now pass
  `--evidence-only`, which suppresses the `auto-FAIL armed` line. That closed a
  channel this ADR's relocation does not touch: the leak was in a tool's stdout,
  not in prose.
- A future editor who wants the reasoning, the 36.17 comparison, or the
  provenance of the correction reads this file. The judge does not need any of
  it to grade.

## References

- `.claude/agents/qa.md` -- the pointer that replaced this block.
- `.claude/workflows/qa-verdict.js::enforceEscalation` -- where the thresholds
  are computed now.
- `scripts/qa/verify_escalation_86_78.mjs` -- the re-runnable checker.
- `CLAUDE.md` F1 / F1b -- the two bounds the block reconciles.
- arXiv 2604.15224 -- consequence framing and LLM-judge leniency.
