---
name: project-re-research-leg-86-72
description: 86.72 — the QA->RESEARCH edge is an EXTENSION of Anthropic's design not a restoration; the repo's own stress test recommended PRUNING the mechanism; SEMA-RAG is the prior art
metadata:
  type: project
---

Step 86.72 adds a leg where a Q/A verdict can demand MORE DOCUMENTATION instead of
another fix attempt. Four findings that are not derivable from the code:

**1. It is an EXTENSION, not a restoration.** CLAUDE.md's F2 reads as if Anthropic
specifies this. It does not. `harness-design-long-running-apps` documents only a
**QA→Generator** backward edge (*"the QA still added value in catching those last
mile issues for the generator to fix"*) and **no explicit state machine**. Do not let
a contract claim the doc already prescribes the leg.

**2. [ADVERSARIAL, INTERNAL] The repo's own prior evaluation said DELETE it.**
`docs/stress-tests/2026-Q2-opus-4.7.md:88,:109` rates `research_needed` a **PRUNE
candidate** — *"Rarely emitted (researcher fires unconditionally now). Dead-weight in
JSON envelope."* The reconciliation: that premise is true at the RESEARCH phase and
false at the EVALUATE→GENERATE boundary, which is the leg 86.72 targets. Any contract
must argue against this, not around it.

**3. The prior art is SEMA-RAG (arXiv 2605.17101), not anything Anthropic published.**
An E-Agent emits `(s_t, g_t, Q_t+1)` — a sufficiency flag about the EVIDENCE, distinct
from correctness of the ANSWER. Bound the loop **three** ways: `s_t=1`, `t=Tmax`, **and
stagnation `Q_t+1 = empty`** — the third is the one implementations forget. `Tmax=2`.
Ablating the agent costs −6.37 to −8.40 points, the largest drop in their ablation.
Critique-revision literature independently lands on 2-3 rounds.

**4. The literature argues AGAINST self-assessed difficulty tiers.** Anthropic:
*"Agents struggle to judge appropriate effort for different tasks."* Triage
(arXiv 2605.13414, 20 models): self-allocation is **worse than random when binding**
(η_E negative), and models honour their own declared budgets only **6.0-36.6%** of the
time. This is the evidence 86.73 Q2 asks for. Caveat: it does NOT directly measure
*de-escalation to finish sooner*; budget non-compliance is the nearest proxy.

**Why:** 86.72 is easy to build wrong in two ways — as a schema change (a REQUIRED
field breaks BACKWARD compatibility; optional is FULL-compatible) or as a judge-side
branch (telling a judge what its verdict triggers causes leniency in 58/72 cells, per
the arXiv 2604.15224 note already quoted at `qa-verdict.js:501-546`). Score inside,
route outside.

**How to apply:** 86.72 must NOT decide who assesses difficulty — [[project_research_gate_depth_86_73]]
owns that; 86.72 only justifies the caller-supplied tier in writing. It must NOT enable
the `deep` tier (`research-gate.js` deep-tier note: *"Report the gap; do not close it
unilaterally"*). Coordinate with 86.70 (also edits `research-gate.js`) and share ONE
counter with 86.71 — SEMA-RAG's `Tmax` and 86.71's attempt ceiling are the same object.
Re-derive every `research-gate.js` anchor by grep: the audit_basis cites `:201/:202/:190-200`
and the real locations are `:394/:395/:348-393`. Corpus method:
[[reference-wf-run-record-corpus-parsing]].
