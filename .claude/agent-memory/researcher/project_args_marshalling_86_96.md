---
name: args-marshalling-86-96
description: Workflow args JSON failures -- "Expecting ',' delimiter" at a '}' is a bracket-TYPE mismatch not a missing comma; insertion-vs-substitution discriminates transit-dropout from mis-composition; the class was 4 events not 2
metadata:
  type: project
---

Step 86.96 research (2026-08-17). Four durable facts about the Layer-3 Workflow
`args` boundary; see `handoff/current/research_brief_86.96.md` for the full derivation.

**1. The parser names the wrong token.** Python's C array parser emits
`Expecting ',' delimiter` for ANY char after a value that is not `,` or `]` --
including `}` and including EOF. Three of four real failures report that message; the
causes are a bracket, a bracket, and a truncation. **Never go looking for a missing
comma.** Recover the container stack with a string/escape-aware scanner reporting the
stack BEFORE the offending char is consumed (a naive scan pops it first and shows the
wrong container).

**2. Insertion vs substitution is the root-cause discriminator.** A one-character
defect is either dropped-in-transit or mis-emitted-at-composition, and they demand
different remedies. Test both repairs: if SUBSTITUTION parses and INSERTION yields
`Extra data`, the character is WRONG, not MISSING -- which exonerates the runtime
(claude-code #69085 / #67765 delta-boundary shear) and pins the caller. Measured both
ways on both payloads.

**Why:** the error string alone cannot separate a caller defect from a genuine
client-side truncation bug -- Anthropic's own #69085 produces the identical message.

**How to apply:** any future one-char JSON failure at this boundary -- run the
insertion/substitution pair before assigning blame.

**3. The mechanism is idiom priming, and it is shape-triggered.** Both bracket failures
sat at `extra.judge_these_specifically`, a LIST field immediately following a DICT
field. The model reused the dict's `"},` close idiom for the list. Identical key,
identical direction, two independent spawns, same day. Fixtures must reproduce the
SHAPE at production scale (p50 3,942 / p90 5,473 chars) -- a 40-char stub cannot
exercise a defect whose mechanism is "the opener is 1,000+ chars out of view".

**4. Silent repair is the expensive failure, not the loud one.** Pre-86.17,
`wf_b098cab6-87b` fell back to `{}` on unparseable args, **completed**, and returned a
gate result with `coverage.audit_class: False` when the caller passed `true`. A
rejected argument costs a retry; a defaulted one produces a confident wrong answer
nothing flags. RFC 9413 s5.1 and the Anthropic tool-use doc both back reject-and-
preserve; arXiv 2605.02363 is the credible opposing case and its cost argument applies
to RESPONSE payloads, not argument payloads.

Census (580 run records, 2026-08-17): string args 394 (390 parse, **4 fail**), plain
objects 90 (**0 fail**), absent 96. The string path is 81.4% of parameterised launches
and carries 100% of the failures. See [[research-gate-depth-86-73]].

Stale-premise warning: step 86.96's description says
`scripts/qa/verify_workflow_args_boundary.mjs` is "RED 84/3"; it is **GREEN 96/0** --
86.92 closed at commit `e45c1bf6`. Re-run a checker before budgeting work to fix it.
