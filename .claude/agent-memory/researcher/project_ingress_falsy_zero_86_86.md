---
name: ingress-falsy-zero-86-86
description: Step 86.86 -- 86.74 hardened the CONSUMER but the falsy-zero collapse is at the PRODUCER (autonomous_loop.py lite judges), so the three-state resolver is blind, not wrong; paper_risk_judge_shape_fix_enabled has ZERO prod readers; a mutation matrix is complete only over the SUBJECT it names
metadata:
  type: project
---

Research for 86.86 (falsy-zero at data-INGRESS). Three findings that were not
visible from the step statement, plus the durable method.

**1. The defect is at the PRODUCER; 86.74 fixed the CONSUMER.**
`autonomous_loop.py:3092` (Claude lite) and `:3338` (Gemini lite) do
`float(risk_dict.get("recommended_position_pct") or _LITE_RISK_DEFAULT[...])`,
rewriting a judge's explicit `0.0` to **3.0** at construction. By the time
`portfolio_manager.py:1102` (`_resolve_position_pct`) reads it, the zero is gone,
so `PositionVerdict(SIZE, 3.0)` is *correct about a value that was already
falsified*. **The 86.74 three-state machinery is not wrong -- it is BLIND.** A
resolver can only distinguish states that still reach it. The same 0.0 SURVIVES
on the paths that build the dict as a plain literal
(`_data_integrity_blocked_analysis:2358`, `risk_debate.py:152`,
`orchestrator.py:2415`) -- so the fix binds everywhere EXCEPT the two LLM lite
paths.

**Why:** whenever a step says "we already fixed the falsy-zero", ask *at which
seam* -- producer or consumer. A downstream guard cannot recover information an
upstream `or` destroyed. Protobuf documents exactly this as round-trip
information loss under implicit presence.

**2. `paper_risk_judge_shape_fix_enabled` is a DEAD FLAG.**
`grep -rn --include="*.py" <flag> backend/ scripts/ | grep -v settings.py`
returns **only tests**. 86.74 made the fix unconditional, but `settings.py:352`
still describes it as governing behaviour. Do not hang a new fix on it. (Sibling
flags DO have readers: `reject_binding` -> `portfolio_manager.py:385`,
`parse_fail_reject` -> `risk_debate.py:138`.)

**3. A mutation matrix is complete only over the SUBJECT it names.**
`scripts/qa/mutation_matrix_86_74.py` is excellent -- control-green-first,
sha256 byte-identical restore, NOT_APPLIED on a no-match replace, and a
`selected()` probe because pytest exit 5 on an empty `-k` scores as a KILL. But
`SUBJECTS` is `portfolio_manager.py` only, and
`test_phase_66_2_risk_judge_shape.py:53-60` hand-builds its lite fixture and
**never calls the producer**. That is why the defect survived a passing matrix.
A producer cell added today would score UNSCORABLE/NOT_APPLIED -- **the test must
drive the real construction first, or the cell proves nothing.**

**How to apply:** enumerate the class from the AST, not by hand -- `ast.walk` for
`ast.BoolOp(op=ast.Or)` found **184** `or` nodes across the two files, 10 in the
lite risk constructions. Then classify per site by FALSY TRIGGER, because the
remedy is not uniform: `0.0` (harmful, the defect), `""` on `decision` (harmful,
fail-OPEN to `APPROVE_REDUCED`), `""` on `risk_level` (fabricates "MODERATE"),
`""` on `reasoning` (fabricates a false "parse failed" audit string), `{}` on
`risk_limits` (benign -- installs a stop). Note the serialisation asymmetry: the
falsy test runs BEFORE `float()`, so the string `"0"` is truthy and survives
while the number `0` dies.

**External anchors that settle the design.** Google protobuf field presence is
the closest industrial precedent AND the fix: implicit presence cannot
distinguish "set to 0" from "unset", and Google restored `optional` (default
since v3.15.0) rather than tuning defaults -- *"We recommend always adding the
`optional` label for proto3 basic types."* PEP 661 is **Final (resolved
2026-04-23, Python 3.15)** and its sentinels are **truthy**, so an `or` cannot
destroy one. `dataclasses.MISSING` states the rationale verbatim: *"This sentinel
is used because `None` is a valid value for some parameters with a distinct
meaning."* Codd's SECOND relational model has TWO nulls (value-unknown vs
attribute-missing) = exactly `UNPARSEABLE` vs `ABSENT`. RFC 7396 shows the tax of
overloading one channel: you can no longer SET a value to null. King's
"parse, don't validate" gives the one-resolver rule -- *"Push the burden of proof
upward as far as possible, but no further."*

**ADVERSARIAL, both worth carrying forward:** (a) **Ruff FURB110 pushes code
TOWARD `x or y`**, asserting it provides *"the same functionality"* as
`x if x else y` -- false for falsy-valid values, and there is **no** Python
linter that detects this class (AWS CodeGuru's `dict-get-method` and the
quantifiedcode entry are about `KeyError`, not falsiness -- reading them in full
REFUTED their search snippets). (b) arXiv:2604.01483 calls Pydantic-style shape
validation *"logically shallow"*, but ships no experiments, no limitations
section, and **no proof-timeout semantics** -- so a heavyweight verification
framework does not itself buy fail-closed. Same shape as the AgentSpec note in
[[project_risk_gate_veto_86_74]].

See also [[project_dead_sell_rule_86_58]] (canonicaliser guarded the READ only --
same producer/consumer asymmetry) and
[[feedback_guards_stop_one_seam_short]]-class lessons.
