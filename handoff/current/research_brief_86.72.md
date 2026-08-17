# Research Brief — step 86.72

**Topic:** Evaluator-driven re-research loops in agent harnesses — research-on-demand
patterns where a verdict can demand MORE DOCUMENTATION rather than another fix attempt;
verdict-envelope schema evolution without weakening existing gates; deterministic
phase-ordering enforcement between an evaluation and the next GENERATE; and
difficulty-tier assessment for research depth (self-assessed vs caller-supplied).

**Tier:** moderate (caller-supplied). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-17.

---

## ENVELOPE (born inert — phase-86.37; updated in place as sources land)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 20,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 2,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "summary": "The QA->RESEARCH edge is an EXTENSION of Anthropic's harness design, not a restoration: harness-design documents only a QA->Generator backward edge and no state machine. The nearest prior art is SEMA-RAG's E-Agent, which emits a sufficiency flag s_t, a gap description g_t and a follow-up query set separate from answer-correctness; ablating it costs -6.37 to -8.40 points. Bound the loop three ways (s_t=1, t=Tmax, stagnation Q=empty) at Tmax=2; critique-revision literature independently converges on 2-3 rounds. Add research_needed as an OPTIONAL VERDICT_SCHEMA field (FULL-compatible per Confluent; a required field would break BACKWARD), and route on it OUTSIDE the judge in the shape of enforceEscalation -- telling a judge what its verdict triggers causes leniency in 58/72 cells. On criterion 4 the literature argues AGAINST self-assessed tiers (Triage: self-allocation worse than random when binding; 6.0-36.6% self-budget compliance), so justify the caller-supplied tier and leave the decision to 86.73. Four disagreements with the audit_basis are reported, including stale line anchors (:202 -> :394) and a population rule under which a naive parse recovers only 92/582 and inverts the conclusion.",
  "brief_path": "handoff/current/research_brief_86.72.md",
  "gate_passed": true
}
```

**Gate arithmetic:** `external_sources_read_in_full` = 8 >= 5 AND `recency_scan_performed`
= true AND all five hard-blocker items checked AND `coverage.audit_class` = false
(so `coverage.dry` is not required) => `gate_passed: true`.

`coverage` is INFORMATIONAL for this step. Two search/fetch rounds were run; the
second round was NOT dry (it added S7 and S8, the two most design-deciding sources),
so `dry` is honestly reported as `false`. For a non-audit step this does not gate.

---

## Search queries run (three-variant discipline, `.claude/rules/research-gate.md` §Search-query composition)

| # | Variant | Query |
|---|---------|-------|
| Q1 | current-year frontier (2026) | `evaluator agent verdict triggers additional research instead of retry LLM harness 2026` |
| Q2 | year-less canonical | `JSON schema evolution backward compatible adding optional field without weakening validation` |
| Q3 | year-less canonical | `LLM agent self-assessed task difficulty vs externally assigned difficulty calibration` |
| Q4 | last-2-year window (2025) | `agentic workflow deterministic phase ordering state machine enforcement 2025` |
| Q5 | year-less canonical | `LLM judge feedback taxonomy insufficient evidence verdict vs incorrect answer` |

## Read in full (>=5 required; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| S1 | https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-08-17 | Official vendor engineering doc (tier 2) | WebFetch (HTML, full) | **The canonical citation for criterion 3** AND the adversarial finding for criterion 4. Verbatim: *"The LeadResearcher synthesizes these results and decides whether more research is needed—if so, it can create additional subagents or refine its strategy."* And on difficulty assessment, verbatim: *"Scale effort to query complexity. **Agents struggle to judge appropriate effort for different tasks**, so we embedded scaling rules in the prompts."* Plus the explicit tier table: *"Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents with clearly divided responsibilities."* Handoff: *"Subagent output to a filesystem to minimize the 'game of telephone.' ... implement artifact systems where specialized agents can create outputs that persist independently."* |
| S2 | https://arxiv.org/html/2606.06324v2 | 2026-08-17 | Preprint, arXiv (tier 1) | WebFetch (arXiv native HTML per §PDF chain) | *"From Failed Trajectories to Reliable LLM Agents: Diagnosing and Repairing Harness Flaws"* (HarnessFix). Treats a failed trajectory as **structured evidence about the harness**, not just a retry signal: *"HarnessFix treats failed trajectories not only as feedback signals, but also as structured evidence for diagnosing and repairing the harness mechanisms behind agent failures."* Seven-layer ETCLOVG taxonomy (Execution / Tool / Context-and-memory / Lifecycle-and-orchestration / Observability / Verification-and-evaluation / Governance). **29 of 30** agents in their dataset carried Lifecycle/Tooling/Observability flaws. Repair beats retry: **+18.4% GAIA, +12.0% SWE-Bench Verified, +8.9% Terminal-Bench 2.0, +6.3% AppWorld**. Diagnosis accuracy: step id 85.0%, harness-layer id 86.2% macro-F1, repair-operator 82.5%. **HONEST NEGATIVE for this step:** it does **not** describe a diagnostic pathway that routes to *more documentation* as an alternative to retry — its "Context and Memory" layer repairs the harness's own retrieval, not the agent's evidence base. So the specific leg 86.72 wants is *not* off-the-shelf prior art. |
| S3 | https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html | 2026-08-17 | Official docs (tier 2) | WebFetch (HTML, full) | The compatibility-type reference for criterion 6 (no gate weakened). Default is **`BACKWARD`** (non-transitive). BACKWARD allows exactly *"Add optional field"* + *"Remove optional field"*, and **forbids** *"Add required field"*. FULL = *"Schemas are both backward and forward compatible"*, allows *"Add field with default value"*, and uniquely permits *"You can upgrade the producers and consumers independently."* **The design rule this yields:** adding an OPTIONAL `research_needed`-style field to `VERDICT_SCHEMA` is FULL-compatible and cannot invalidate any verdict a today's-Q/A emits; making it REQUIRED would break BACKWARD and is exactly the "weakening/changing an existing gate" move criteria 6+7 forbid. |
| S4 | https://arxiv.org/html/2606.27416 | 2026-08-17 | Preprint, arXiv (tier 1) | WebFetch (arXiv native HTML per §PDF chain) | *"Glite ARF: Verifier-Driven Research with Parallel LLM Coding Agents"*. **The canonical prior art for criterion 3's "deterministic phase-ordering enforcement"**: *"the rules of the research process live in scripts that fail loudly when violated, not in prose that agents are merely asked to follow"*; *"Every artefact has a versioned specification and a corresponding verifier that checks it before commit"*; *"a verifier refuses to mark a step complete if an expected command log is missing."* Nine-step lifecycle, *"each step runs in its own subagent and writes to a known place inside the task folder"* — file-based handoff, same shape as pyfinagent's five-file protocol. Errors **block** advancement, warnings surface without blocking. Overhead measured: structural machinery = **~1% wall-clock despite ~25% of commands**; 273 tasks, 12 parallel agents, $498.31 total. **ADVERSARIAL on the verdict-demands-research design:** their verifiers explicitly refuse evidentiary judgement — *"What the verifiers deliberately do not judge is semantic validity ... That judgement stays with the human researcher."* i.e. the strongest published verifier-driven harness deliberately keeps "is this well-enough evidenced?" OUT of the deterministic gate. |

| S5 | https://arxiv.org/html/2605.13414 | 2026-08-17 | Preprint, arXiv (tier 1) | WebFetch (arXiv native HTML per §PDF chain) | **[DECISIVE FOR CRITERION 4]** *"Triage: Evaluating Prospective Metacognitive Control in LLMs under Resource Constraints"*. Tests exactly the operator's proposed design — can a model judge task difficulty and allocate its own budget BEFORE attempting the task. Definition: *"metacognitive control: regulating effort allocation based on judgments of one's own knowledge state"*; *"all decisions are committed before any task is attempted, distinguishing TRIAGE from retrospective measures."* Result across **20 models / 4 benchmarks** (AIME, GPQA, LiveCodeBench, HLE): self-allocation is **worse than random** when binding — *"η_E is negative for the large majority of configurations, meaning the model captures essentially no value once its own per-item allocations are enforced."* And the self-declared budget is not even self-honoured: *"models rarely honor the budgets they themselves declare even when explicitly instructed to"* — compliance **6.0%–36.6%**. Reasoning models are **worse**, not better, at spotting infeasible items (**<8% detection** at full injection). Authors leave self-assessment open: *"whether LLMs can themselves perform the joint optimization...remains untested."* |
| S6 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-17 | Official vendor engineering doc (tier 2) | WebFetch (HTML, full) | The project's own canonical harness reference. Hard-threshold-or-fail: *"Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback on what went wrong."* File handoff: *"Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or with a new file that the previous agent would read in turn."* Contract-first: *"The generator then built against the agreed-upon contract before handing the work off to QA."* **HONEST NEGATIVE — and it is the central finding of this brief:** the documented backward edge is **QA → Generator only** (*"the QA still added value in catching those last mile issues for the generator to fix"*), with **no** path that rejects back to the planner/research phase, and **no explicit state machine** — the harness relies on prompt-encoded personas rather than deterministic control flow. So pyfinagent's proposed QA→RESEARCH edge is a genuine EXTENSION beyond Anthropic's published design, not a restoration of something the doc already specifies. |

| S7 | https://arxiv.org/html/2605.17101 | 2026-08-17 | Preprint, arXiv (tier 1) | WebFetch (arXiv native HTML per §PDF chain) | **[THE MECHANISM PRIOR ART — closest published analogue to what 86.72 wants]** *"SEMA-RAG: A Self-Evolving Multi-Agent RAG Framework for Medical Reasoning"*. An **E-Agent** emits a signal that is explicitly about the EVIDENCE, not the answer: *"Conditioned on clinical anchors Q′, the current textual query set 𝒬ₜ, and the evidence set Cₜ, E-Agent predicts a sufficiency flag sₜ, a gap description gₜ, and the next query set 𝒬ₜ₊₁."* When `sₜ=0`, *"the generated 𝒬ₜ₊₁ is issued in the next round to retrieve additional evidence."* **Three-way bound**: *"Iteration terminates when sₜ=1, t=Tₘₐₓ, or stagnation occurs (i.e., 𝒬ₜ₊₁=∅)"*, default **Tₘₐₓ=2**, performance peaking at Tₘₐₓ∈{2,3}. **Ablation proves the leg is load-bearing**: removing E-Agent is the single largest drop — MedQA-US 89.95%→83.58% (−6.37pt), PubMedQA 59.20%→50.80% (−8.40pt); *"the core gain comes from a self-evolving, sufficiency-driven closed-loop retrieval rather than static retrieval."* |
| S8 | https://www.emergentmind.com/topics/multi-agent-critique-and-revision-326a2d61-fb41-400d-a710-1cbf54133f20 | 2026-08-17 | Secondary/aggregator over peer-reviewed work (tier 3) | WebFetch (HTML, full) | Bounds the loop from a **different domain** than S7, and converges on the same number. Ueda et al. (11 Jul 2025): *"Modest depths (2–3 critique–revision cycles) and parallel critic instantiation (N ≈ 3) yield optimal novelty/feasibility trade-offs"*, with deeper chains or higher N causing performance to *"rapidly saturate or regress."* Named failure modes of critique-revision loops: **sycophancy/conformity, over-correction, mis-calibrated voting weights, accuracy degradation from excessive rounds**. *"Adversarial roles (rebut, question) help mitigate sycophancy and echo chamber effects."* **HONEST NEGATIVE:** it does **not** compare evidence-requesting critics against revision-only critics — that specific comparison is unanswered in this source. |

## Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://aclanthology.org/2025.acl-long.179.pdf | Peer-reviewed ACL 2025 ("RAG-Critic: critic-guided agentic workflow") | Highest-value unfetched item. Binary PDF at ACL Anthology (not arXiv, so the html/ar5iv chain does not apply); the ≥5 floor was already cleared 3× over. **Recommend Main queue this for 86.73.** |
| https://arxiv.org/html/2607.11881v1 | Preprint — "Metacognition in LLMs: Foundations, Progress, and Opportunities" | Survey; S5 (Triage) gives the primary measurement directly. Queue for 86.73 Q2. |
| https://arxiv.org/pdf/2601.07264 | Preprint — "The Confidence Dichotomy: Miscalibration in Tool-Use Agents" | Corroborates S5; budget. |
| https://arxiv.org/pdf/2505.20120 | Preprint — "Agents Require Metacognitive and Strategic Reasoning..." | Corroborates S5; budget. |
| https://openreview.net/forum?id=y9UdO5cmHs | OpenReview — "Mitigating Overconfidence in LLMs" | Confidence calibration, adjacent not central. |
| https://arxiv.org/html/2506.00582 | Preprint — "Do Language Models Mirror Human Confidence?" | Adjacent. |
| https://arxiv.org/html/2510.23458 | Preprint — "BrowseConf: Confidence-Guided Test-Time Scaling" | Adjacent (confidence→compute). |
| https://arxiv.org/pdf/2604.16385 | Preprint — "StressWeb" | Miscalibration under perturbation; adjacent. |
| https://arxiv.org/html/2604.20801v1 | Preprint — "Synthesizing Multi-Agent Harnesses for Vulnerability Discovery" | Harness synthesis, different problem. |
| https://arxiv.org/pdf/2605.03042 | Preprint — "ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration" | Relevant to 86.73 Q1 (fan-out); out of scope here. |
| https://arxiv.org/pdf/2503.24047 | Preprint — Survey of LLM-based Scientific Agents | Survey. |
| https://arxiv.org/pdf/2606.11199 | Preprint — NightFeats @ MMU-RAGent NeurIPS 2025 | Task-specific RAG. |
| https://www.creekservice.org/articles/2024/01/08/json-schema-evolution-part-1.html | Practitioner article — Evolving JSON Schemas I | S3 (Confluent, official docs, tier 2) is the authoritative version. |
| https://www.creekservice.org/articles/2024/01/09/json-schema-evolution-part-2.html | Practitioner article — Evolving JSON Schemas II | Same. |
| https://www.zerodatatools.com/blog/json-schema-versioning-guide/ | Blog — JSON Schema Versioning Guide (2026) | Tier 5. |
| https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide | Vendor blog — LLM agent eval metrics 2026 | Tier 4/5. |
| https://deepeval.com/blog/what-is-an-eval-harness | Vendor blog — eval harness | Tier 4/5. |
| https://techcommunity.microsoft.com/blog/educatordeveloperblog/evaluating-the-evaluator-how-to-test-an-llm-judge-with-microsoft-agent-framework/4516639 | Vendor blog — testing an LLM judge | Tier 4. |
| https://www.judgmentlabs.ai/blogs/agent-judge-solving-long-context-evaluations | Vendor blog — Agent Judge | Tier 4. |
| https://towardsdatascience.com/why-your-multi-agent-system-is-failing-escaping-the-17x-error-trap-of-the-bag-of-agents/ | Community | Tier 5. |
| https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html | (counted in read-in-full as S3) | — |

**URL total: 8 read in full + 20 distinct snippet-only = 28 unique URLs** (floor is 10).

## Recency scan (2024-2026) — PERFORMED

Searched explicitly in the 2024-2026 window (Q1 scoped to `2026`, Q4 scoped to `2025`, plus three year-less canonical queries per `.claude/rules/research-gate.md` §Search-query composition).

**Result: 5 new findings in the window, and they are the load-bearing ones — the canonical 2025 Anthropic sources are SUPERSEDED on the specific question this step asks.**

1. **S7 SEMA-RAG (2026)** supplies a sufficiency-flag mechanism that neither Anthropic source describes. This is new material, not a restatement.
2. **S5 Triage (2026)** measures prospective self-assessed effort allocation across 20 models — evidence that did not exist when the tier system was designed, and it points the opposite way to the operator's stated design.
3. **S2 HarnessFix (2026)** reframes a failed trajectory as evidence about the *harness* rather than the *agent* — with +6.3% to +18.4% from repairing rather than retrying.
4. **S4 Glite ARF (2026)** supplies the measured overhead of deterministic phase gating (**~1% wall-clock for ~25% of commands**) — the cost number the contract needs.
5. **S8 / Ueda et al. (Jul 2025)** bounds critique-revision depth at 2-3 cycles, independently converging on S7's Tₘₐₓ=2.

**Nothing in the window contradicts the ≥5-source / ≥10-URL floors**; no source argues for lowering an evidence floor.

## Key findings

1. **The QA→RESEARCH edge is a genuine EXTENSION of Anthropic's harness design, not a restoration of it.** The harness-design post's only documented backward edge is QA→Generator — *"the QA still added value in catching those last mile issues for the generator to fix"* — with no path rejecting back to planning, and *"no explicit state machine"* (S6, https://www.anthropic.com/engineering/harness-design-long-running-apps). Main must not write the contract as if CLAUDE.md's F2 merely restores something Anthropic specifies.
2. **The nearest published mechanism is a SUFFICIENCY FLAG that is structurally separate from the verdict.** S7's E-Agent emits `(sₜ, gₜ, 𝒬ₜ₊₁)` — a boolean, a gap description, and the follow-up queries — where `sₜ` says *the evidence is insufficient*, distinct from *the answer is wrong* (https://arxiv.org/html/2605.17101). That triple maps 1:1 onto CLAUDE.md's F2 4-key brief (objective / output_format / tool_scope / task_boundaries).
3. **The loop must be bounded three ways, and the bound is small.** S7 terminates on `sₜ=1`, `t=Tₘₐₓ`, **or stagnation (`𝒬ₜ₊₁=∅`)** — the third is the one a naive implementation forgets — with Tₘₐₓ=2 default. S8 independently reports 2-3 cycles optimal, *"rapidly saturate or regress"* beyond. Two different domains, same number.
4. **The mechanism is measurably load-bearing, so building it is justified.** Ablating S7's sufficiency agent costs −6.37 to −8.40 points, the largest single drop in their ablation. S2 reports +6.3% to +18.4% from repairing the harness rather than retrying it.
5. **[ADVERSARIAL, criterion 4] The literature argues AGAINST the researcher self-assessing its own tier.** Anthropic: *"Agents struggle to judge appropriate effort for different tasks, so we embedded scaling rules in the prompts"* (S1). Triage measures it: self-allocation is **worse than random when binding** — *"η_E is negative for the large majority of configurations"* — and models honour their own declared budgets only **6.0%–36.6%** of the time; reasoning models detect infeasible items **<8%** of the time (S5). Anthropic's design has *"The lead agent decompose[] queries into subtasks"* — i.e. caller-supplied, exactly what ships today.
6. **[ADVERSARIAL, criterion 3] The strongest published verifier-driven harness deliberately keeps evidentiary judgement OUT of the deterministic gate.** Glite ARF: *"What the verifiers deliberately do not judge is semantic validity — whether an experiment is well-designed or a baseline appropriate. That judgement stays with the human researcher"* (S4). So "is this fix under-researched?" belongs in the **LLM-judgment leg** of the Q/A, while the **deterministic** leg only enforces that the flag, once emitted, is *obeyed* — a split the contract should make explicit.
7. **Schema evolution has a bright line that criterion 6/7 can be pinned to.** Adding an **optional** field is BACKWARD- and FULL-compatible; adding a **required** field is forbidden under BACKWARD, whose default posture Confluent states as *"The Confluent Schema Registry default compatibility type is `BACKWARD`"* (S3). FULL uniquely allows *"upgrade the producers and consumers independently"* — which is precisely the property needed here, since old Q/A returns and new consumers coexist.
8. **Deterministic phase gating is cheap.** Glite ARF measured the structural machinery at **~1% wall-clock despite ~25% of commands**, over 273 tasks / 12 parallel agents / $498.31 (S4). The cost objection to a hard ordering gate is not supported.

## Internal code inventory

| File | Anchor | Role | Status |
|------|--------|------|--------|
| `scripts/harness/run_harness.py` | 1230 lines; `research_needed` at **:258, :278, :292, :304, :1150** | The ONLY implementation of F2. `:258` sets `research_needed=True`, `:259-276` builds the 4-key brief (`objective` / `output_format` / `tool_scope` / `task_boundaries`), `:278` sets it False otherwise. `run_planner_with_research()` at `:~285` loops `MAX_RESEARCH_ITER` times via injected `spawn_researcher`. `:1150` logs the trigger. | **ALIVE but on the WRONG RAIL.** Confirmed: `grep -c research_needed scripts/harness/run_harness.py` = **5**. This is the optimizer harness, not the Layer-3 per-step loop. |
| `.claude/workflows/qa-verdict.js` | 746 lines; `VERDICT_SCHEMA` at **:414-440** | The rail that actually grades every masterplan step. Schema is `additionalProperties: false`, `required` = `[ok, verdict, reason, violated_criteria, violation_details, certified_fallback, checks_run, harness_compliance_ok, notes]`. `KNOWN_ARG_KEYS` at **:310-312**. | **`grep -c research_needed` = 0.** No field, no arg, no path. This is the dead leg. |
| `.claude/workflows/research-gate.js` | 1101 lines | The researcher rail. | **`grep -c research_needed` = 0.** |
| `.claude/workflows/qa-verdict.js` | **:491-500** | phase-86.31 precedent: a proposed new schema field (`wip_path`) was **weighed and REJECTED** with written reasons — *"it would buy nothing on the failure it targets, because a DROPPED run produces no return object for the field to live in."* | **BINDING PRECEDENT.** Main must argue against this comment, not around it: a `research_needed` field is different (it is consumed on a *successful* return, not a dropped one), and the contract should say so explicitly. |
| `.claude/workflows/qa-verdict.js` | **:501-546** (`enforceEscalation` docblock) | The exact architectural template 86.72 should copy: score inside the judge, **threshold computed outside** it. *"Telling a judge what its verdict will TRIGGER shifts the verdict"* — arXiv 2604.15224, leniency in **58 of 72 cells** (p<0.001, peak −9.8pp), invisible in CoT (ERRJ=0.000). | **DIRECTLY LOAD-BEARING.** Implies: the Q/A must **not** be told that emitting `research_needed` triggers a researcher spawn. Emit the signal; let the CALLER route on it. |
| `.claude/workflows/qa-verdict.js` | **:546-560** | `enforceEscalation` fails closed: *"An absent or unusable sequence yields `null`, never `0`"*; *"There is no branch here that writes `verdict`, and in particular no path from any input to turning a FAIL into a PASS."* | **The pattern criterion 7 demands.** Copy it verbatim in shape. |
| `.claude/workflows/research-gate.js` | **:394-399** | `VALID_TIERS = ['simple','moderate','complex']` (**:394**), `tierRequested = a.tier \|\| null` (**:395**), `tierAbsent` (**:396**), `tierSupported` (**:397**), `tierUnsupported` (**:398**), `tier = tierSupported ? tierRequested : 'moderate'` (**:399**). | **LINE-NUMBER DRIFT — see Disagreements §D1.** Tier is caller-supplied; researcher never assesses difficulty. |
| `.claude/workflows/research-gate.js` | **:348-393** | The deep-tier operator-decision note. Verbatim: *"Enabling the tier would ship producer fan-out onto this N=1 artifact rail -- one brief path, one stage-2 verifier, no cross-branch de-dup -- and pre-empt an open operator decision. Report the gap; do not close it unilaterally."* Cost: *"~1 Claude Max 5-hour rolling window per subagent"*, *"2-3 parallel deep-tier researcher subagents"*. | **CRITERION 5: DO NOT CLOSE.** Note it self-documents an earlier failed operationalisation (*"`grep -c deep` on this file returns 0"* — now returns 8 because the comment itself contains the word). Durable check: `VALID_TIERS` lacks `'deep'` and every occurrence is a comment. |
| `.claude/workflows/research-gate.js` | **:822-845** | `if (tierUnsupported)` → **REFUSES TO SPAWN**, returns `gate_passed:false`, `envelope:null`, zero agents. | ABSENT→`moderate` (proceeds); UNSUPPORTED→hard refusal. The audit_basis's "defaults to moderate when absent" is true only for the ABSENT arm. |
| `docs/stress-tests/2026-Q2-opus-4.7.md` | **:88, :109** | The project's own Opus-4.7 stress test rates `research_needed` a **PRUNE candidate**: *"Rarely emitted (researcher fires unconditionally now). Dead-weight in JSON envelope."* Action item B: *"Remove research_needed flag from planner output (if no current consumer reads it)"* — P3, LOW risk. | **[ADVERSARIAL — INTERNAL]** The repo's own prior evaluation recommended **deleting** the mechanism 86.72 proposes to build. The contract MUST address this head-on. The reconciliation is available and should be stated: the stress test's premise was *"researcher fires unconditionally"* — true at the RESEARCH phase, false at the EVALUATE→GENERATE boundary, which is the leg 86.72 targets. |
| `.claude/masterplan.json` | step **86.73** (pending, P1) | Owns *"who assesses difficulty"* and *"how depth scales"*; explicitly operator-directed and evidence-blocked. | **86.72 MUST NOT PREEMPT.** 86.72 criterion 4 is satisfied by *justifying in writing* the caller-supplied tier; the *decision to change it* is 86.73's. S5 (Triage) is exactly the evidence 86.73's criterion 2 asks for. |
| `.claude/masterplan.json` | step **86.70** (pending, P1) | Also edits `research-gate.js`. | **COORDINATE BEFORE EDITING** (86.72 notes say so). `git status .claude/workflows/` is currently clean. |
| `.claude/masterplan.json` | step **86.71** (pending, P1) | The missing attempt CEILING. | Complementary: ceiling bounds the grinding, re-research improves the fix. S7's `t=Tₘₐₓ` bound is the same object 86.71 is building — **the two steps should share one counter, not build two.** |

## Disagreements with the audit_basis (criterion 1 requires these be REPORTED, not silently adopted)

- **D1 — All research-gate.js line anchors in BOTH 86.72's and 86.73's audit_basis are STALE.** They cite `:202` (tierRequested), `:201` (VALID_TIERS) and `:190-200` (deep-tier note). Measured 2026-08-17: **`VALID_TIERS` is at :394, `tierRequested` at :395, and the deep-tier note spans :348-393.** `:180-215` today holds `jsonLosslessViolation()`, an unrelated phase-86.90 function. The *substance* of every cited claim is confirmed; only the anchors moved (~192 lines). Main must re-derive by grep, not by line number — the file itself warns about exactly this.
- **D2 — The population rule in the audit_basis does not match the artifacts on disk.** It states *"all 527 wf_* run directories ... that have a journal.jsonl"*, step identified from *"its FIRST agent transcript"*. Measured: there are **no `wf_*` directories** and no `journal.jsonl`; the records are **JSON files** at `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/<session-uuid>/workflows/wf_*.json`, across **44 sessions**.
- **D3 — The caller's population figure is CONFIRMED (481/580 → 483/582), but only under a rule the caller did not state, and the naive rule inverts the conclusion.** `args` is a **JSON string on 394 of 582 records**, a dict on only 92, and `null` on 96. A naive `args.step_id` dict access recovers **92 of 582 (15.8%)** and yields a **recency-biased subsample** whose top steps are 86.94 (6 qa / **2** researcher) and 86.97 (5 qa / **2** researcher) — i.e. it makes the researcher look *re-engaged*, the opposite of the finding. Parsing the string form (`json.loads`, plus a regex fallback that recovered 4 more) gives **483 of 582**. **Main must state this parsing rule explicitly in the live_check** or criterion 2's re-derivation is not reproducible.
- **D4 — Positive control is slightly understated but VALID.** The audit_basis says the `consecutive_fails` control *"returns run_harness.py, attempt_budget.py and one drill script"*. Measured: **7 files** (`attempt_budget.py`, `run_harness.py`, `smoke_test_4_17_12.py`, `test_phase_86_32_attempt_budget.py`, `test_phase_76_9_2_max_bridge.py`, `test_phase_75_sre_ops.py`, `autoresearch_health.py`). The control still does its job — the search demonstrably reaches the harness tree — so the absence of `research_needed` from the workflows is genuine, not a search artefact.

## Independent re-derivation (criterion 2)

Population rule: all `wf_*.json` under `~/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/*/workflows/`; `step_id` recovered from `args.step_id`/`args.stepId` where `args` is a dict, **else by `json.loads(args)` where it is a string**, else by regex. **483 of 582 recovered; 99 unrecoverable and counted separately, not dropped.** Role assigned from the script name (`research*`→researcher, `qa*`→qa).

| Step | Total runs | Q/A | Researcher | audit_basis said | Agrees? |
|------|-----------|-----|-----------|------------------|---------|
| 36.8 | 9 | 9 | **0** | 9 / 0 | YES |
| 86.28 | 9 | 8 | 1 | 8 / 1 | YES (total now 9; corpus grew) |
| 36.17 | 8 | 6 | 2 | 6 / 2 | YES |
| 75.5 | 7 | 7 | **0** | 7 / 0 | YES |
| 36.12 | 6 | 6 | **0** | 6 / 0 | YES |
| 78.2 | 6 | 6 | **0** | 6 / 0 | YES |

**All six confirmed.** New since filing: **86.94 = 8 runs (6 qa / 2 researcher)**, **86.97 = 7 (5/2)**, **86.21 = 7 (5/2)**. Script-name distribution: `qa-verdict` 307, `research-gate` 79. **The claim stands: the four highest-churn steps had ZERO researcher re-engagement across 28 combined Q/A runs.**

## Consensus vs debate (external)

**Consensus.** (a) Deterministic code, not prose, must enforce phase ordering — *"the rules of the research process live in scripts that fail loudly when violated, not in prose that agents are merely asked to follow"* (S4); (b) file-based artifact handoff between phases (S1, S4, S6); (c) an evidence-sufficiency signal distinct from an answer-correctness signal materially improves outcomes (S7); (d) revision loops must be shallow, 2-3 rounds (S7, S8).

**Debate.** *Should the evidentiary judgement itself be deterministic?* Glite ARF says no — keep semantic sufficiency with the human (S4). SEMA-RAG says yes, delegate it to an LLM agent, and measures a large gain (S7). **Resolution for pyfinagent:** S7's domain has a *retrievable* ground truth (medical corpora), S4's does not. pyfinagent's case resembles S7 (documentation exists and is fetchable), so an LLM-emitted sufficiency flag is defensible — but per S4 and the `enforceEscalation` precedent, the **routing** on that flag must be deterministic and outside the judge.

**Unsettled.** Whether evidence-requesting critics beat revision-only critics is **not answered** by any source read (S8 explicitly lacks the comparison). This is the core empirical question 86.72 poses, and the honest position is that the literature does not settle it — the ablation in S7 is the nearest proxy and is from a different domain.

## Pitfalls (from literature)

1. **Telling the judge what its verdict triggers makes it lenient** — 58/72 cells, p<0.001, peak −9.8pp, invisible in CoT (arXiv 2604.15224, quoted at `qa-verdict.js:501-546`). **The Q/A prompt must not say "emitting this spawns a researcher."**
2. **Sycophancy, over-correction and degradation from too many rounds** (S8). Bound at 2-3.
3. **Forgetting the stagnation exit.** S7 terminates on `𝒬ₜ₊₁=∅` as well as `sₜ=1` and `t=Tₘₐₓ`; without it a researcher that keeps returning nothing new loops to the ceiling every time.
4. **Self-assessed effort is worse than random when binding** (S5). Do not let the researcher (or the Q/A) set its own depth.
5. **Adding a REQUIRED schema field breaks BACKWARD compatibility** (S3). Optional-only.
6. **A field added to a return that a dropped run never produces buys nothing** — the repo's own 86.31 rejection (`qa-verdict.js:491-500`).

## Application to pyfinagent

- **Criterion 3 (drive the mechanism end to end).** Model it on S7: add an **optional** `research_needed: boolean` + an optional gap object to `qa-verdict.js` `VERDICT_SCHEMA` (:414-440) — optional keeps it FULL-compatible per S3, so no existing verdict becomes invalid and criterion 6 is satisfied by construction. Route on it **outside** the judge, in the shape of `enforceEscalation` (:501-560): a pure function that reads the flag and returns a routing decision, with **no branch that writes `verdict`** — that is how criterion 7 is *demonstrated* rather than asserted. The positive/negative demonstration criterion 3 demands maps directly onto S7's `sₜ=0` vs `sₜ=1` arms.
- **Criterion 4 (difficulty).** Keep the caller-supplied tier and justify it in writing with S1 (*"Agents struggle to judge appropriate effort"*) + S5 (η_E negative; 6.0-36.6% self-budget compliance). Do **not** change it — 86.73 owns that decision. Note for 86.73: S5 does *not* directly measure the specific failure mode 86.73 asks about (an agent **de-escalating** its own difficulty rating to finish sooner); the 6.0-36.6% budget-compliance figure is the nearest proxy. State that gap.
- **Criterion 5 (deep tier).** Raise as a numbered operator ask carrying the note's own measured cost (~1 Max 5-hour window per subagent, 2-3 subagents) plus the N=1-artifact-rail de-dup gap. Do not touch `VALID_TIERS` at `:394`.
- **Criterion 6 (no floor weakened).** Nothing in this brief touches `.claude/rules/research-gate.md`. Demonstrate by diffing that file to HEAD and showing `FLOOR_URLS`/the ≥5 assertion in `verify_research_gate_workflow.mjs` are byte-identical.
- **Criterion 8 (mutation-test).** Per repo doctrine, observe the control GREEN first, then revert each new guard and show RED, then byte-identical restore. The obvious cells: (M1) delete the routing function → the "flag causes a spawn" check goes red; (M2) make the flag default `true` → the negative arm (verdict without the flag does **not** spawn) goes red; (M3) attempt FAIL→PASS through the routing function → must be unreachable by construction.
- **Coordinate with 86.71**: S7's `Tₘₐₓ` and 86.71's attempt ceiling are the same counter. Two independent counters will disagree.

## Research Gate Checklist

Hard blockers:
- [x] **>=5 authoritative external sources READ IN FULL via WebFetch** — 8 (S1-S8). Hierarchy respected: 4 arXiv preprints (tier 1), 3 official docs / vendor engineering (tier 2), 1 aggregator over peer-reviewed work (tier 3). Zero community-tier sources in the read-in-full set.
- [x] **10+ unique URLs total** — 28 (8 full + 20 snippet-only).
- [x] **Recency scan (last 2 years) performed + reported** — dedicated section above; 5 findings in the 2024-2026 window, all load-bearing.
- [x] **Full pages read (not abstracts)** — every arXiv source fetched via `arxiv.org/html/<id>` per `.claude/rules/research-gate.md` §PDF chain. **No `arxiv.org/pdf/` URL was WebFetched.** The one high-value PDF-only source (RAG-Critic, ACL Anthology) is recorded snippet-only and NOT counted.
- [x] **file:line anchors for every internal claim** — see Internal code inventory + Disagreements D1-D4; all anchors re-derived by execution on 2026-08-17, not carried forward.

Soft checks:
- [x] Internal exploration covered every module named in the caller's INTERNAL SCOPE (`qa-verdict.js` VERDICT_SCHEMA, `research-gate.js` args + deep-tier note, `run_harness.py` F2 shape, steps 86.70/86.71/86.73, `.claude/rules/research-gate.md`) — plus one the scope did not name: `docs/stress-tests/2026-Q2-opus-4.7.md`, which contradicts the step.
- [x] Contradictions / consensus noted — §Consensus vs debate, plus 2 external `[ADVERSARIAL]` findings (S4, S5) and 1 internal adversarial finding (the PRUNE recommendation).
- [x] All claims cited per-claim (URL + access date inline, or file:line).

### What could NOT be verified
- Whether an **evidence-requesting** critic outperforms a **revision-only** critic — no source read settles it (S8 explicitly lacks the comparison). S7's ablation is a cross-domain proxy only.
- The specific de-escalation failure mode (an agent lowering its own difficulty rating to finish sooner) is not directly measured by S5; only self-budget non-compliance (6.0-36.6%) is.
- RAG-Critic (ACL 2025) was not read in full and may change the picture on the first bullet. Queued for 86.73.
