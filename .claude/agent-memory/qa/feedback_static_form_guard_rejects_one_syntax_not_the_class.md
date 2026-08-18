---
name: static-form-guard-rejects-one-syntax-not-the-class
description: An AST guard that rejects a LITERAL at an argument position is defeated by any non-literal expression; 86.108's sole coverage for 3 call sites survived model_name=_client_model_name(None)
metadata:
  type: feedback
---

An AST/source guard written as "reject shape S at position P" catches exactly
shape S, never the semantic class S was an instance of. Attack it by writing the
same defect in a DIFFERENT syntactic form.

**Why:** phase-86.108 cycle 3 (2026-08-17). Cycle 2's Q/A killed a mutant that
hardcoded `model_name="claude-opus-4-8"` at a call site. The fix added
`test_every_parse_call_site_forwards_a_model_name`, which walks the AST of three
modules and fails if the `model_name=` kwarg is missing OR
`isinstance(kw["model_name"], ast.Constant)`. It genuinely kills the literal
form (I reproduced M16, M17 and my own Q11). But it was the SOLE coverage for
`orchestrator.py`'s 3 call sites -- the behavioural drivers only reach
`debate`/`risk_debate` -- and I executed
`model_name=_client_model_name(None)` at the Synthesis-Final site: an
`ast.Call` node, so the guard passes, and it **SURVIVED 37/37**. The
differential is real: the record's rail drops from a measured value to
`unknown` with the basis `no_model_in_scope_at_emit_site`, which is false
because a model WAS in scope. `... or "claude-opus-4-8"` (a `BoolOp`) survives
too and actively misattributes.

**How to apply:** when a guard is a static scan, ask "what is the node type it
tests, and what other node types express the same defect?" Then build one and
run it. Concretely: `Constant` -> try `Call`, `BoolOp`, `Name`, `str("x")`,
`Attribute`. The repair is to whitelist the ACCEPTED form (the kwarg must be a
`Call` to one of the named resolver helpers with a non-None argument), not to
blacklist one rejected form -- a blacklist over an open syntax space cannot be
completed.

**Also from this cycle -- check the survivor for equivalence before filing.** I
tested a third mutant (pass `self.deep_think_client` instead of
`self.synthesis_client`) and it survived too, but `orchestrator.py:684-685`
constructs BOTH clients from the same `deep_model_name`, so it is an EQUIVALENT
mutant and not a finding. Two of my three orchestrator-site survivors were
weak or equivalent; only one carried a real differential. See
[[survivor-needs-behavioural-differential]] and
[[class-guard-bound-to-the-helper-not-the-call-site]].
