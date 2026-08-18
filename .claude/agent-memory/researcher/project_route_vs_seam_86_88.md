---
name: route-vs-seam-86-88
description: 86.88 -- the step's "dead branch" premise does NOT reproduce (fires on the bare-Name form); every 86.86 guard anchors at/below the producer so ZERO tests execute the 4 caller routes; the 86.86 Q/A already named the fix
metadata:
  type: project
---

Step 86.88 = "guard the ROUTES INTO a seam, not just the seam". Researched 2026-08-16.

**The premise in the step's own filing is WRONG, and I only caught it by reading the
prior step's critique.** 86.88 was filed partly on "`verify_lite_risk_seam_86_86.py:65-66`
is a DEAD branch / zero-assertion guard". Measured by importing the shipped
`or_default_sites` and handing it both shapes:

```
`x or _LITE_RISK_DEFAULT`   -> [(4, '<whole-dict>')]   # FIRES
`dict(_LITE_RISK_DEFAULT)`  -> []                      # blind
```

It is **shape-blind, not dead**. A `dict(...)` is an `ast.Call`; the rule walks `BoolOp`
operands. So the remedy is **widening accepted node shapes**, not "resurrecting" or
deleting a branch. The 86.86 Q/A had already recorded this as NOTE-2 ("the 86.88 filing
rests partly on a premise that does not reproduce") -- I reproduced it rather than
accepting it, which is the right order, but I would have written a wrong brief had I not
read the critique at all.

**Why:** a step's framing is written by whoever filed it, often mid-cycle and from the
self-critical direction. `feedback_read_the_steps_prior_artifacts_first` + `a_labelled
_inference_still_argues`. An over-stated WEAKNESS is still a wrong premise.

**How to apply:** before any 86.8x follow-up, grep the PRIOR step's
`evaluator_critique_*.md` for the new step id -- the Q/A frequently files the follow-up
AND names its fix. Here N1 already said it: *"one end-to-end test that drives the lite
risk-judge block with a stubbed LLM returning {"recommended_position_pct": 0}, or a
checker assertion that risk_dict is not written between parse and producer call."*

## The mechanism (measured, autonomous_loop.py)

`_lite_position_pct:2313` resolves SIZE/ABSENT/UNPARSEABLE and is CORRECT. Four callers --
`:3177`, `:3182` (Claude lite), `:3411`, `:3416` (Gemini lite), all in the no-JSON and
`except` handlers -- do `risk_dict = dict(_LITE_RISK_DEFAULT)` BEFORE the producer, so the
seam resolves **SIZE(3.0)** instead of ABSENT. **Same number, destroyed provenance**:
"judge failed" and "judge said 3.0" become indistinguishable. Not a wrong-value bug --
which is exactly why no test that asserts the NUMBER can see it.

Census: 12 `ast.Name` refs to `_LITE_RISK_DEFAULT` = 1 Assign + 7 Subscript + 4 Call.
`risk_dict` has 6 rebinds, **0 subscript writes** -- corruption is whole-object only.
`dict(X)` is a SHALLOW copy, so nested `risk_limits` is the module-level object itself.

## Coverage is anchored one layer too low

Suite, AST checker and the 6-cell matrix ALL anchor at or below
`_build_lite_risk_assessment`. `test_phase_66_2_risk_judge_shape.py:776-788` drives the
real `decide_trades` but starts from a hand-built `_judge()` dict, i.e. **enters downstream
of the pre-mangle**; `:818-829` asserts on `inspect.getsource` of ONE function. **No test
executes `_run_claude_analysis` / `_run_gemini_analysis`.** The checker still prints
`RESULT: OK, 8 PASS / 0 FAIL` with the defect live.

## Prior art worth not re-finding

- **Delamaro Interface Mutation** = two operator groups; the *second* mutates *"the
  declaration and invocations of the interfaces"*. Primary text is paywalled (IEEE TSE
  27(3)) but described verbatim in a FREE ICSM'07 PDF: `taoxie.cs.illinois.edu/publications/icsm07.pdf`
  (extract with pypdf, not WebFetch -- see [[reference_webfetch_pdf_summaries_fabricate_quotes]]).
- **PyTation, ICSE '26** (`arxiv.org/html/2601.19088v2`) -- `RemFuncArg` mutates *call
  arguments*; **69% of its mutants are not subsumed by Cosmic Ray's**. The answer to "is a
  call-site mutation cell a real technique or something we invented".
- **PIT's** call-acting operators are **OFF by default** and *"fairly unstable"* -- so
  "just run a mutation tool" does not get you call-site coverage.
- **Cling / Coupled Branches** (`ar5iv.labs.arxiv.org/html/2001.04221`) -- the coverage
  unit is caller-branch x callee-branch; found 25 integration faults unit suites missed.
- **muSE** (`ar5iv.labs.arxiv.org/html/2102.06829`) -- non-detection by a checker IS a
  tool flaw; "soundy is fine only if the unsound choices are documented".
- **CERT OBJ06-J** -- copy FIRST, then *"validation must be performed on the copy"*.
- Stubbing: stub the **LLM transport** (`client.messages.create` -> `.content[0].text`;
  `client.generate_content` -> `.text`), never the dict construction -- that returns a
  string and carries no copy of production logic.

Full brief: `handoff/current/research_brief_86.88.md` (15 sources read in full, 30 URLs).
Related: [[project_ingress_falsy_zero_86_86]], [[project_risk_gate_veto_86_74]].
