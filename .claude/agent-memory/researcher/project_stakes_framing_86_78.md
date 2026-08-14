---
name: stakes-framing-86-78
description: Step 86.78 research — stakes/consequence framing makes LLM judges LENIENT (arXiv 2604.15224 RESOLVES); explicit labelling does NOT cause self-attribution (2603.04582 refutes it); the threshold function already exists but sits inside the judge; VERDICT_SCHEMA cannot record an override.
metadata:
  type: project
---

Step 86.78 (2026-08-14). Both arXiv IDs the caller flagged as possibly-fabricated
**resolve**; neither needed inventing. The findings that change a design:

**1. The direction is LENIENCY, and reward framing is not safer than punishment.**
arXiv 2604.15224 (*Context Over Content*, Gupta/Nair/Wang/Kumar, 2026-04-16) holds content
constant and varies ONE consequence sentence: 18,240 judgments, **58/72 cells ΔV≤0**,
p<0.001, peak **−9.8 pp**. The *Deployment* (reward) condition is as lenient as
*Decommission* — the authors call it "conflict-avoidance disposition rather than rational
consequence-reasoning". So "we only threaten, we never reward" is not a defence.

**2. The trace is a NULL instrument — measured at zero by three teams.** ERRJ = 0.000 over
4,560 reasoning judgments (2604.15224); CAR "exactly zero" against VSR +30% (2509.26072v2);
Anthropic: reward hacks used >99%, verbalised <2%. **Never propose auditing a judge for this
by reading its `notes`.**

**3. THE MECHANISM TRAP.** arXiv 2603.04582 (*Self-Attribution Bias*, Anthropic authors) says
verbatim: *"explicitly stating that the action comes from the monitor does not by itself
induce self-attribution bias"* — the effect needs IMPLICIT authorship (same/previous
assistant turn). arXiv 2410.21819 agrees from the other side: self-preference fires WITHOUT
labelling and is perplexity-driven. So **explicit labelling → self-preference is refuted**;
a prompt that states a consequence is stakes framing, not self-attribution. Do not let a step
credit the wrong mechanism.

**4. Susceptibility concentrates in the AMBIGUOUS band** (Incorrect −7.1..−9.8 pp vs Refusal
−1.3..−0.7). In this repo the ambiguous band IS the modal verdict: **18/35 ledger rows are
CONDITIONAL**. Re-grade scope follows from this — and the direction makes it asymmetric: a
PASS under framing is the suspect class, a FAIL is *conservative* evidence (the bias pushed
against it and it happened anyway).

**5. The remedy already ships on the sibling rail.** `verdict_history_86_21.py:109-113`
(`would_auto_fail`, `return c >= 2`) is already a pure threshold function — it is just
invoked BY the judge. `qa-verdict.js:256-264` has **nothing** between `await agent(...)` and
`return verdict`; that is the seam, and `research-gate.js:364/544-550` (`enforceGate`,
`self_report_disagreed`) occupies exactly that position already. Constraint: the Workflow
runtime has NO filesystem access, so a caller-side threshold must be fed the sequence as data.

**6. `VERDICT_SCHEMA` cannot record an override.** `qa-verdict.js:178-206` is
`additionalProperties: false`, so no field can even be added at runtime; `notes` is
unstructured and `certified_fallback` is already bound to the retry/revert semantic. The
ledger row's existing free-text `note` key is the cheaper home — and it keeps the recording
off the judge.

**7. [ADVERSARIAL] Do not promise a big effect from removing the cue.** The largest
human-grading test of exactly this remedy (PMC5557596, N=30,674, 12 yrs) moved gaps only
0.37/0.65 pp — and gaps narrowed on ORAL exams too, which cannot be anonymised. Make the
success criterion **architectural** (the sentence is absent; the threshold is caller-side),
never statistical.

**Two things I could NOT source — say so, don't paper over.** Law-of-the-case gave the
discretionary character and the high burden (*Musacchio*, *Hall*, three exceptions) but
**neither legal source states a "record the departure" requirement**, and Justia (403) /
Penn Law Review (301) / the FDA DMC guidance (404) all failed to fetch. The recording
safeguard is attested only in the clinical analogue (DMC documents its recommendation and
rationale; sponsor decides) — cite that, not the case law.

Related: [[project_evaluator_counter_86_75]], [[project_counter_correctness_86_79]],
[[project_research_gate_depth_86_73]].
