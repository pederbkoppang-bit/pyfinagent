---
name: class-guard-bound-to-the-helper-not-the-call-site
description: A parametrised "pins the CLASS" test that imports and drives the helper cannot see the production call site swapping the helper out, or pre-normalising its argument -- two mutants survived 78 tests + a 10-check AST command (86.88 c4)
metadata:
  type: feedback
---

When an author fixes "guard from instance, not class" by parametrising a test
over every key/member, **check WHICH SUBJECT the parametrisation drives.** If the
test does `from prod import _helper` and asserts on `_helper(x)`, it pins the
class *for the helper* and says nothing about the expression that actually lands
in the record.

**Why:** phase-86.88 cycle 4. `test_the_equality_is_EXACT_over_EVERY_key_not_just_one`
parametrised over every key of `_LITE_RISK_DEFAULT` and killed both cycle-3
survivors. Two of my mutants at the PRODUCTION site survived all 78 tests and the
shipped 10-check AST command:

- **bypass** -- `"judge_verdict_absent": risk_dict.get("reasoning") == _DEFAULT["reasoning"]`
  (stops calling the helper entirely)
- **intermediate alias** -- `_helper({**risk_dict, "reasoning": _DEFAULT["reasoning"]})`
  (still calls the helper; pre-normalises the ARGUMENT)

The author had explicitly flagged the alias case as "covered by the runtime
value-equality guard". Measured false: that guard passes its own arguments and
never observes what the call site passes.

**How to apply:** for every "now pins the class" claim, run two cells --
(1) replace the production expression with a weaker rule that agrees on the
point inputs the behavioural tests use; (2) keep the call but wrap the argument.
Then prove non-equivalence with a differential table over inputs the tests do
NOT use (here: a judge emitting all-default values plus its own `reasoning` --
an ordinary output for the shipped prompt schema -- flipped False -> True).
Severity is WARN not BLOCK when genuine behavioural tests cover the production
site at point inputs; the gap is class-level, not sole-coverage.

Related: [[feedback-mutate-each-duplicated-site-individually]],
[[feedback-slice-and-exec-with-the-collaborator-stubbed]].
