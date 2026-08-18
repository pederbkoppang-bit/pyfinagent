---
name: palindromic-fixture-cannot-test-order
description: An ordering assertion whose fixture is all-identical elements cannot fail under reversal; check the fixture's SHAPE against the property the guard names, then drive the differential on the real consumer
metadata:
  type: feedback
---

An assertion that NAMES an ordering property is vacuous when its fixture is
**palindromic** — all elements identical, or symmetric under the transform the
guard claims to catch. Read the fixture, not the assertion's name.

**Why:** 86.85 cycle 1. `verdict_ledger_write.py::_self_test` carried
`check("sequence is oldest->newest", seq == ["CONDITIONAL","CONDITIONAL","CONDITIONAL"])`.
Three identical elements, so `return out[::-1]` in `emit_sequence` **SURVIVED** with
all 11 checks green, including that one. The author's own 5-cell matrix was 5/5
killed and never touched ordering. Ordering was load-bearing: the consumer
`enforceEscalation` counts consecutive CONDITIONALs by scanning **backwards from the
end**, so the differential I drove on the byte-exact real function was
`[PASS,C,C]+C -> n=2 armed` vs reversed `[C,C,PASS]+C -> n=0 SUPPRESSED` — an
escalation silently disarmed. Sibling shapes: sorted-ness asserted on an already-sorted
fixture, "filters by X" asserted on a set with one X, "dedup" asserted with no duplicate.

**How to apply:** for every guard naming an ORDER / SORT / DIRECTION / POSITION
property, ask what the fixture would look like under the inverse transform. If the
expected value is unchanged, the guard cannot fail — build the mutant and run it. Then
do the second half: a survivor is only a finding once you show a **behavioural
differential** on the REAL consumer (see [[survivor-needs-behavioural-differential]]),
because some reversals are genuinely equivalent — [NV,NV,C,C,P,C,C] happens to yield
n=2 either way. Pick the input that discriminates. Related:
[[mutation-probe-must-discriminate]], [[a-control-built-from-your-own-pattern-tests-nothing]].

Same cycle, a second reusable shape: a module can be **loud in one read path and
silent in another**. `read_rows` raised on a corrupt LINE ("a partial read would
under-count"), while `emit_sequence` silently dropped an out-of-vocabulary VALUE — and
that silent filter bypassed the downstream consumer's `sequence_status=unparseable`
fail-closed branch, manufacturing a confident number where the design promised `null`.
Grep every read path in a module for its failure disposition before crediting
"fail-loud" to the module.
