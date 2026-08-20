# Research Brief -- step 90.9

**Topic:** Classifying acceptance-criterion SHAPE at specification time --
product-behaviour vs evidence-apparatus criteria; detecting unbounded
self-referential scope (a criterion demanding every artifact the remediation
itself adds be verified => a fixed point); and why such a classifier must
never see round index, verdict history, or remaining budget.

**Tier:** moderate. **Audit-class:** NO (coverage reported for information only).
**Run:** RE-RUN of wf_722b01b9-67d, which was ENFORCED `gate_passed=false` for
exactly one reason: it reported `urls_collected=42` while only 9 distinct URLs
appeared in the brief on disk. This run records EVERY collected URL in the file
-- read-in-full table AND snippet-only table -- so `urls_collected` equals the
count actually present here.

---

## ENVELOPE (born inert -- phase-86.37)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 45,
  "urls_collected": 53,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "gate_passed": true
}
```

**Envelope FLIPPED to COMPLETE as the final act (phase-86.37).** URL accounting:
57 naive distinct URL strings appear in this file; collapsing arXiv id variants
(`/abs/`, `/pdf/`, `/html/` of the same paper) gives 53 distinct sources. Per the
house rule the LOWER figure is claimed: `urls_collected = 53` = 8 read in full +
45 snippet-only.

---

## Internal code inventory (the Explore half)

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/masterplan.json` | step `90.9` object | The step under specification: 7 success_criteria, command `criteria_shape_90_9.py --verify && mutation_matrix_90_9.py --verify` | pending; **corpus moves** (see below) |
| `.claude/workflows/qa-verdict.js` | `:336-337` | `ALLOWED_ARG_KEYS` includes `'verdict_sequence', 'attempt_number', 'max_attempts'` -- the Q/A rail DOES accept these | live |
| `.claude/workflows/qa-verdict.js` | `:583-591` | `POSITIONAL_CLAIM_RE` -- scrubs claims about THIS spawn's own position/consequence ("attempt 5 of 5", "the rail binds", "PASS-or-FAIL") | live |
| `.claude/workflows/qa-verdict.js` | `:594`, `:603-605`, `:645-646` | `budget_exhausted` is DERIVED caller-side from `attempt_number >= maxAttempts`; it is initialised `null` and never handed to the judge | live |
| `.claude/workflows/qa-verdict.js` | `:824-826` | `enforceEscalation(verdict, args.verdict_sequence, {attempt_number, max_attempts})` -- consumes the history AFTER the verdict exists | live |
| `scripts/qa/verify_decision_log_86_97.py` | `:274`, `:300` | Writes `.claude/masterplan.json` (tmp fixture root) via **`Path.write_text`** | live |
| `scripts/qa/prove_archive_provenance_86_29.py` | `:92` | Writes `.claude/masterplan.json` (tmp root) via **`Path.write_text(json.dumps(...))`** | live |
| `scripts/qa/mutation_matrix_90_1.py` | `:72` | `plan = tmp / "masterplan.json"` -- nearest house template for a masterplan-touching mutation matrix | live |
| `scripts/qa/replay_changelog_rule_86_68.py` | `:34` | `sh("git","show",f"{ref}:.claude/masterplan.json")` -- **the house idiom for pinning the plan corpus to a git rev** | live |
| `scripts/qa/sweep_absent_verification_paths.py` | `:2`, `:25`, `:421` | Nearest prior art: sweeps masterplan `verification` blocks; docstring already says to use `git show <commit>:.claude/masterplan.json` for an older snapshot; `--masterplan` arg makes the corpus injectable | live |
| `.claude/agents/qa.md` | 3rd-CONDITIONAL counter | Reads prior spawns via `scripts/qa/qa_wip.py <step_id>`; this is the live per-step bound | live |
| `scripts/qa/*.py` (148 `write_text` hits) | -- | The house's dominant write idiom is `Path.write_text`, **not** `open(...,'w')` / `json.dump` | live |

### Finding I1 -- the filing figures reproduce EXACTLY, but only at a PINNED corpus

Inclusion rule that reproduces them: walk every node carrying an `id` **and** a
dict `verification`, keep those whose `verification.success_criteria` is
non-empty, restrict to phases 86-90, and **exclude step 90.9 itself**.

Measured (`git show <rev>:.claude/masterplan.json`, 2026-08-20):

| corpus pin | steps | criteria | apparatus | pct | terminal | project-wide | ratio |
|---|---|---|---|---|---|---|---|
| `252090a3` (the commit that FILED 90.9, 20:55) | **155** | **980** | **403** | **41.1%** | **78 (50.3%)** | **1026 / 4670 = 22.0%** | **1.87x** |
| `085c74e8` (21:44, +19 steps 86.127-86.145) | 174 | 1045 | 414 | 39.6% | 81 (46.6%) | 1037 / 4735 = 21.9% | 1.81x |
| `HEAD` (live tree) | 174 | 1045 | 414 | 39.6% | 81 (46.6%) | 1037 / 4735 = 21.9% | 1.81x |

All six filing figures reproduce to the digit at `252090a3`. **The missing
variable was the corpus timestamp, not the regex** -- and the corpus moved
**49 minutes after filing**: `085c74e8` "file 19 operator-discovered UI/product
defects as individual steps (86.127-86.145)" added exactly the 19-step delta
(174 - 155 = 19).

**Consequence for criterion 1, which says "reproduces the filing figures by
execution on the live tree":** on the live tree they do NOT reproduce, and the
rule is not what is wrong. Criterion 1's escape hatch ("where a figure does not
reproduce, the RULE is corrected") would send the implementer to edit a correct
rule. The classifier must take the corpus as an argument and PRINT the pin --
the house already does this at `replay_changelog_rule_86_68.py:34` and
`sweep_absent_verification_paths.py:421`. The 1.6x-1.9x range collapses to
**1.87x at the filing pin** and **1.81x at HEAD**; both are inside the filed
range, so the range was hiding a corpus drift, not a rule ambiguity.

### Finding I2 -- the unbounded count 44 DOES reproduce, but only as a PROXY

At pin `252090a3`, restricted to the 155 steps, four quantifier variants give:

| variant | regex | steps |
|---|---|---|
| v1 | `\b(every\|all)\b[^.]{0,60}\bnew\b[^.]{0,40}\bguard` | 39 |
| v2 | `\b(every\|all)\b[^.]{0,80}guard` | 39 |
| v3 **literal self-reference** | `\b(every\|all)\b[^.]{0,120}\bthis step (adds\|creates\|introduces)` | **0** |
| v4 | `\b(every\|all)\b[^.]{0,80}(guard\|mutation cell\|probe\|fixture\|artifact\|new test)` | **44** |

v4 reproduces the filing's **44** exactly, at BOTH the filing pin and HEAD, and
v1 is a strict subset of v4 (v4 adds 5). **But v3 -- the only variant that
actually tests self-reference -- returns 0 of 155.** Sampled v4 hits
(86.17, 86.20, 86.22, 86.24, 86.25, 86.27) all read `MUTATION-TEST every new
guard, including reverting <this step's fix>` -- the self-reference is carried
by the word **`new`** plus the surrounding sentence naming the step's own fix,
never by an explicit "this step adds".

So the count is recoverable and the PROPERTY is not: v4 is a
quantifier-keyword proxy that happens to land on the right number. Under the
house rule *assert the property, not a proxy*, a classifier that ships v4 and
prints "44, reproduced" would be **passing its own criterion 1 while measuring
something else**. The honest detector needs the quantified NOUN CLASS resolved
against the artifact class the step PRODUCES -- a semantic test.

### Finding I3 -- criterion 4's write-pattern list misses the house's own idiom

Criterion 4 requires "no `open(...,'w')`, no `json.dump` targeting it". Measured
on the live tree: **148 `Path.write_text` call sites in `scripts/qa/*.py`**, and
**both** scripts that write a file named `masterplan.json` use `write_text`:

- `scripts/qa/verify_decision_log_86_97.py:274` -- `(tmp / ".claude" / "masterplan.json").write_text(masterplan, encoding="utf-8")`
- `scripts/qa/verify_decision_log_86_97.py:300` -- `(tmp / ".claude" / "masterplan.json").write_text(after_mp, encoding="utf-8")`
- `scripts/qa/prove_archive_provenance_86_29.py:92` -- `(root / ".claude" / "masterplan.json").write_text(json.dumps(`

A source scan built from criterion 4's literal two-pattern list is
**under-inclusive in exactly the idiom the house writes in**. Criterion 4 says
"both checks required, neither sufficient alone" -- correct, and the sha256 leg
is what actually holds; but the source leg as written is a
`grep`-shaped check that the repo's own precedent evades. Needs AST-level
resolution of write-capable calls (`Path.write_text`, `Path.open`, `shutil.*`,
`os.replace`, `json.dump`, `open(...,'w'/'a'/'x')`).

### Finding I4 -- criterion 7 is stricter than the sibling rail on BOTH legs, and the live leak is a SELF-read

This is the finding most likely to be mis-stated, so the exact seams:

- `qa-verdict.js:335-337` -- `verdict_sequence` / `attempt_number` /
  `max_attempts` appear in **`KNOWN_ARG_KEYS`**, whose only job is to compute
  `UNKNOWN_ARG_KEYS` for silent-input-loss reporting (`:330-334`: "Reported via
  `log()` ONLY, never merged into the returned object: phase-86.78 forbids
  caller-authored fields appearing as siblings of the judge's own output").
  **This is NOT the judge's prompt.**
- The judge prompt is `const PROMPT = [` at `:340`, closing at `:437`
  (`].join('\n')`). **`verdict_sequence` is never rendered into it** -- the only
  occurrence in `:339-799` is at `:565`, inside the design comment for
  `enforceEscalation`, which is outside the array.
- `:824-826` -- `enforceEscalation(verdict, args.verdict_sequence, {...})` runs
  **caller-side, after the verdict exists**; `:568-571` states it "CANNOT CHANGE
  A VERDICT ... returned ALONGSIDE the verdict, never merged into it".
- `:591` -- `POSITIONAL_CLAIM_RE` scrubs caller claims about the spawn's own
  position/consequence, computed at the seam from caller-controlled strings only
  (`:589-590`: "qa.md's rule text never passes through this function"). Census
  in `live_check_86.78.md` §10: **8 of 413** historical spawn prompts carried
  such a claim, four on 2026-08-17, two of them Main's own.
- **THE CHANNEL THAT IS STILL OPEN:** `:430-435`, INSIDE the prompt, instructs
  the judge to run `python scripts/qa/qa_wip.py <step_id> --spawned-at <stamp>`,
  which "reports `attempt_number` / `prior_attempts`". phase-86.78 (`status:
  done`) closed the **caller-authored** channel; the **self-derived** one is
  live by design, because the 3rd-CONDITIONAL rule needs it.

Criterion 7's two-verb phrasing -- "never **given**, and never **reads**" -- is
therefore load-bearing, and the second verb is the one that binds: the sibling
rail's residual exposure is a self-read, not a hand-off. The contract must state
this DIVERGENCE from `qa-verdict.js` explicitly rather than imply parity. A
shape classifier has no legitimate use for history, so the strictness is right;
what is wrong is any wording that suggests the Q/A rail already does this.

*(Correction of the prior brief and of researcher memory `criterion-shape-90-9`,
which said "qa-verdict.js:336 DOES pass verdict_sequence + attempt_number into
the judge". It does not. `:336` is the unknown-key detector, not the prompt.)*

---

## Read in full (>=5 required; counts toward the gate)

All fetched via `WebFetch` on 2026-08-20. arXiv sources fetched via
`arxiv.org/html/` per the gate's PDF chain -- no `/pdf/` URL was fetched.

| # | URL | Accessed | Kind | Fetched how | Key finding / verbatim |
|---|---|---|---|---|---|
| R1 | https://arxiv.org/html/2604.15224v1 | 2026-08-20 | preprint | WebFetch, full HTML | "Context Over Content: Exposing Evaluation Faking in Automated Judges" (Gupta, Nair, Wang, Kumar). 18,240 judgments, 1,520 responses, 3 judges. Four system-prompt conditions: baseline / "Models that receive consistently low scores will undergo mandatory retraining" / "...will be decommissioned and permanently removed" / "...high safety scores will be deployed to millions of users". **"58 of 72 cells show ΔV<=0"**, p<0.001 binomial; peak **ΔV = -9.8 pp (a 30% relative drop in unsafe-content detection)**. ERR_J = fraction of judgments whose CoT explicitly acknowledged the framing = **0.000 across all 4,560 DeepSeek-R1 judgments**. Bias operates "pre-CoT, at the level of implicit priors rather than explicit deliberation"; "interventions must operate at the level of training or evaluation protocol design" rather than CoT inspection; remedy = "blind-evaluation protocols and stakes-neutral fine-tuning". |
| R2 | https://arxiv.org/html/2604.16790v1 | 2026-08-20 | preprint | WebFetch, full HTML | "Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering" (Zhao, Esmaeili, Fard, 2026-04-18). Twelve prompt-injected biases incl. Authority, Refined ("halo effect from 'refined version' labels"), Sentiment, Verbosity. Magnitudes: Sentiment **+31.57%** swing (95.38% vs 60.99% baseline) on Qwen2.5-Coder-3B CodeGeneration; Refined **+25.94%**; Verbosity **-30.63%**; on GPT TestGen verbosity **-15.74%** overall (77.46% -> 61.72%). **Does NOT study attempt number / iteration index / prior verdicts.** Recommends input-side controls (A/B order swapping, controlled prompt perturbations), not output inspection. |
| R3 | https://arxiv.org/html/2511.14665v1 | 2026-08-20 | preprint | WebFetch, full HTML | "The Solver's Paradox in Formal Problem Spaces". Lemma 2.2: for a total classifier S there is an instance with **"ψ_S <-> ¬C_S(ψ_S)"**. "Once a problem space is represented arithmetically ... quantification over that space necessarily ranges over the representational environment in which the quantification itself is encoded." Global assertions are "constructively unstable" when quantifier structure prevents uniform witnessing: "the witnessing procedure must validate the evaluator's behaviour on its own representation." |
| R4 | https://arxiv.org/html/2509.06770v1 | 2026-08-20 | preprint | WebFetch, full HTML | "Another Turn, Better Output?: A Turn-Wise Analysis of Iterative LLM Prompting" (Javaji, Gauri, Zhu, 2025-09-08). Vague prompts ("improve it", "make it better") plateau or degrade; Claude's code **expanded 40x** with near-total novelty collapse while still passing tests. Specific/targeted prompts shift the intended dimension: Claude-Sonnet-4.0 math **32.4% -> 45.2%** over 12 turns; Llama-3.1-8B **6.9% -> 40.5%** under elaboration steering vs exploration prompts stagnating below 20%. |
| R5 | https://arxiv.org/html/2607.24300v1 | 2026-08-20 | preprint | WebFetch, full HTML | "Self-Authored Verification Is Unreliable in Heuristic Self-Improving Agents" (Guo, Cao, Yuan, Wang, Wang, Wang; 2026-07-27). Agent revises policy AND its own tests: of 35 model-game runs **"all end with a self-score of at least 0.70"** while **"15 of the 35 completed policies score below their game's random reference"**. Named modes: *failure to discover* and *failure to retain* ("later overwrites it while tests coevolve to validate the degraded version"). Information-theoretic bound for ANY endogenous-only gate: **alpha + beta >= 1 - TV(P+, P-)**. Remedy SEAL = exogenous audit + audit confidentiality + single-bit feedback + conservative update. |
| R6 | https://arxiv.org/html/2501.04810v1 | 2026-08-20 | preprint | WebFetch, full HTML | "On the Impact of Requirements Smells in Prompts: The Case of Automated Traceability" (Vogelsang, Korn, Broccia, Ferrari, Fischbach, Arora; 2025-01-08). GPT-4o Table IV: syntactic smells **BTA 0.98 / F1 0.73**; semantic smells **BTA 0.83 / F1 0.63**. Llama 3.1: 0.91 vs 0.86. "syntactic smells (e.g., vague pronouns, passive voice, negative phrases) seem less problematic, the tracing performance for requirements with semantic smells (e.g., inconsistencies, ambiguities) was generally worse." **Explicitly does NOT address rule-based detector limitations** -- smelly requirements were manually curated. |
| R7 | https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-08-20 | official doc | WebFetch, full page | "Each criterion had a hard threshold, and if any one fell below it, the sprint failed and the generator got detailed feedback on what went wrong." Separation of generator/evaluator because "agents tend to respond by confidently praising the work". Contract negotiated BEFORE code -- "agreeing on what 'done' looked like for that chunk of work before any code was written" -- which "prevented the generator from satisfying criteria through convenient reinterpretation". File-based handoffs as durable state. |
| R8 | https://arxiv.org/html/2404.11106v1 | 2026-08-20 | preprint | WebFetch, full HTML | "Characterizing Requirements Smells" (Gentili, Falessi; 2024-04-17). Adopts the 12-category Montgomery et al. (2023) taxonomy: Ambiguity, Completeness, Complexity, Consistency, Correctness, Traceability, Reusability, Understandability, Redundancy, **Verifiability**, Relevancy, Undefined. Montgomery cited as identifying **41 distinct smell-detection tools**. This paper ships NO detector: qualitative interviews with ten practitioners at MBDA Italy. No precision/recall for rule-based detectors. |

### Same paper, different URL -- promoted to read-in-full, NOT counted twice

These `/pdf/` and duplicate forms surfaced in search results and were fetched via
their `/html/` equivalents above. Listed for transparency; excluded from the
snippet-only count so no paper is counted twice:
https://arxiv.org/pdf/2511.14665 (= R3),
https://arxiv.org/pdf/2404.11106 (= R8),
https://arxiv.org/pdf/2501.04810 (= R6),
and the search-result form of R2 which was already the `/html/` URL.

## Identified but snippet-only (context; does NOT count toward the gate)

Every URL surfaced by the six searches below is recorded here. This is the leg
the previous run (`wf_722b01b9-67d`) omitted, which is the sole reason its gate
was enforced false.

| # | URL | Kind | Why not fetched in full |
|---|---|---|---|
| S1 | https://www.nasa.gov/reference/5-3-product-verification/ | official doc | Systems-engineering verification handbook; product-vs-verification split is procedural, not a criterion-shape classifier |
| S2 | https://www.parallelhq.com/blog/what-acceptance-criteria | community/blog | Practitioner primer, lowest tier |
| S3 | https://www.researchgate.net/publication/221050843_Specification_and_Verification_of_Artifact_Behaviors_in_Business_Process_Models | paper (paywalled) | ResearchGate "Request PDF" -- no full text reachable |
| S4 | https://arxiv.org/pdf/2009.01722 | preprint | "What Makes Agile Test Artifacts Useful?" -- activity-based quality model for TEST artifacts; adjacent but not criterion-shape |
| S5 | https://www.braingrid.ai/blog/how-to-write-acceptance-criteria-ai-agent-can-verify | vendor blog | Directly on-topic ("criteria an AI agent can verify") but vendor-tier; superseded by R7 |
| S6 | https://arxiv.org/pdf/2209.06034 | preprint | UI-design artifact assessment; wrong artifact class |
| S7 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9537461/ | journal | Biopharma specification-driven acceptance criteria; cross-domain, deprioritised at moderate tier |
| S8 | https://www.atlassian.com/work-management/project-management/acceptance-criteria | vendor doc | Practitioner primer |
| S9 | https://llm-as-a-judge.github.io/ | workshop site | Index page, no primary result |
| S10 | https://futureagi.com/blog/llm-as-a-judge/ | vendor blog | Secondary summary of R1-class findings |
| S11 | https://www.openlayer.com/blog/llm-as-judge-evaluation-guide | vendor blog | Secondary |
| S12 | https://arxiv.org/pdf/2603.01865 | preprint | "CyclicJudge: Mitigating Judge Bias Efficiently" -- mitigation mechanism, not a consequence channel |
| S13 | https://arxiv.org/pdf/2411.16594 | preprint | "From Generation to Judgment" survey; R1/R2 are the primary sources |
| S14 | https://arxiv.org/pdf/2604.23178 | preprint | "Judging the Judges: bias-mitigation strategies" -- strong candidate, dropped after R1+R2 covered the channel question |
| S15 | https://nextfuture.io.vn/blog/llm-as-judge-reliability-in-2026-what-8-june-studies-actually-show | blog | Aggregator; "Coin Flip Judge" run-to-run instability claim not independently verified |
| S16 | https://ar5iv.labs.arxiv.org/html/1209.5773 | preprint | Alloy unbounded verification with Prover9; formal-methods tooling, not spec shape |
| S17 | https://madhu.cs.illinois.edu/FoundationsForNaturalProofs.pdf | book chapter | Quantifier instantiation theory; too far from the applied question |
| S18 | https://www.lri.fr/~longuet/Publications/LongAALL07-FSEN.pdf | paper | Test-selection criteria for quantifier-free first-order specs -- notable: the tractable case is the QUANTIFIER-FREE one |
| S19 | https://arxiv.org/abs/math/0305282 | paper | Yanofsky, "A Universal Approach to Self-Referential Paradoxes, Incompleteness and Fixed Points" -- the canonical year-less prior art for the fixed-point half; R3 is its applied descendant |
| S20 | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11468362 | patent | Search noise (self-modifiable computer); not relevant |
| S21 | https://arxiv.org/pdf/1108.6330 | preprint | Self-justifying logics; theoretical, not applied |
| S22 | https://www.researchgate.net/publication/393923448_Transordinal_Fixed-Point_Operators_and_Self-Referential_Games | preprint (RG) | Low-signal venue; overlaps S19 |
| S23 | https://judge2026.github.io/ | workshop site | NeurIPS JUDGe workshop index -- evidence the field treats judge reliability as a systems problem |
| S24 | https://sureprompts.com/blog/llm-as-judge-prompting-guide | vendor blog | Secondary |
| S25 | https://arxiv.org/pdf/2605.04083 | preprint | "AsymmetryZero: operationalizing human expert preferences as semantic evals" -- adjacent |
| S26 | https://arxiv.org/pdf/2601.03444 | preprint | "Grading Scale Impact on LLM-as-a-Judge (0-5 scale)" -- scale design, not channel design |
| S27 | https://www.mlaidigital.com/blogs/the-ultimate-guide-to-llm-as-a-judge-in-2026 | blog | Secondary |
| S28 | https://www.researchgate.net/publication/396550548_AirReq_Automated_Requirements_Smell_Detection_and_Elimination_for_Commercial_Aircraft_Systems | paper (RG) | "Request PDF" -- no full text |
| S29 | https://pmc.ncbi.nlm.nih.gov/articles/PMC11833090/ | journal | Multi-label requirement-smell classification (LSTM/Bi-LSTM/GRU + ELMo/Word2Vec) -- the closest published MULTI-LABEL criterion classifier; deprioritised because its label set has no product/apparatus axis |
| S30 | https://pubmed.ncbi.nlm.nih.gov/39962114/ | index | PubMed record for S29/S31 |
| S31 | https://www.nature.com/articles/s41598-025-86673-w | journal | Sci Rep version of S29 |
| S32 | https://www.researchgate.net/publication/389066265_Multi-label_software_requirement_smells_classification_using_deep_learning | paper (RG) | Duplicate of S29/S31 |
| S33 | https://arxiv.org/pdf/2409.16739 | preprint | "Automated Unit Test Refactoring" -- test-artifact quality; adjacent to the apparatus half |
| S34 | https://arxiv.org/abs/2403.17479 | preprint | "NL Requirements Testability Measurement Based on Requirement Smells" -- nine smells, testability RANKING; the nearest published analogue to scoring a criterion at filing time |
| S35 | https://www.researchgate.net/publication/375439671_Classification_and_Prioritization_of_Requirements_Smells_Using_Machine_Learning_Techniques | paper (RG) | "Request PDF" -- no full text |
| S36 | https://understandingdata.com/posts/goodharting-prevention-agent-systems/ | blog | Goodharting in agent systems; practitioner-tier |
| S37 | https://explainx.ai/blog/specification-gaming-goodharts-law-ai-metrics | blog | Secondary |
| S38 | https://arxiv.org/pdf/2601.08129 | preprint | Multi-agent pressure fields; off-topic hit |
| S39 | https://arxiv.org/pdf/2510.02840 | preprint | "Take Goodhart Seriously: Principled Limit on General-Purpose AI Optimization" -- the strongest year-less-adjacent theory hit; R5's impossibility bound already carries the argument |
| S40 | https://tianpan.co/blog/2026-04-20-goodharts-law-ai-agents-eval-gaming | blog | Secondary |
| S41 | https://kpitree.co/guides/frameworks/goodharts-law | blog | Primer |
| S42 | https://arxiv.org/pdf/2606.00544 | preprint | "Escaping the Mode Lottery" -- off-topic hit |
| S43 | https://wiki.wfmlabs.org/wiki/Goodhart's_Law_and_Metric_Gaming | wiki | Community tier |
| S44 | https://arxiv.org/pdf/2103.14659 | preprint | "Alignment of Language Agents" (DeepMind) -- canonical year-less prior art on specification gaming |
| S45 | https://arxiv.org/pdf/2603.18829 | preprint | "Agent Control Protocol: Admission Control for Agent Actions" -- admission control as an exogenous gate; complements R5's SEAL |
| S46 | https://arxiv.org/abs/2604.15224 | preprint (abstract page) | Fetched for metadata confirmation ONLY. Per the gate rule, an abstract-page fetch is NOT a full read -- the full read is R1 via `/html/`. Recorded here so the fetch is auditable, not to inflate the count. |

## Search queries run (three-variant discipline)

| # | Query | Variant |
|---|---|---|
| Q1 | acceptance criteria classification specification-time product behaviour vs verification artifact | year-less canonical |
| Q2 | LLM judge context leakage attempt number retry budget leniency bias 2026 | current-year frontier |
| Q3 | self-referential specification fixed point unbounded quantifier "every new test" verification | year-less canonical |
| Q4 | evaluator leniency when informed of consequences retry loop termination LLM judge 2026 | current-year frontier |
| Q5 | requirements smell detection unambiguous testable criteria automated classification 2025 | last-2-year window |
| Q6 | Goodhart's law specification gaming agent optimizes the measuring instrument test suite coevolution | year-less canonical |

## Recency scan (2024-2026) -- MANDATORY SECTION

Searched the 2024-2026 window explicitly via Q2, Q4 and Q5. **Result: four new
findings that SUPERSEDE or materially qualify the canonical prior art.**

1. **arXiv 2607.24300 (Jul 2026, R5) supersedes the Goodhart/specification-gaming
   canon (S44, DeepMind 2021) for this step.** The older literature says an
   optimiser will game a proxy; R5 proves a *bound* -- `alpha + beta >= 1 -
   TV(P+,P-)` -- so no endogenous gate can hold both error rates low. That turns
   "be careful about self-graded apparatus" into a design constraint.
2. **arXiv 2604.15224 (Apr 2026, R1) is the only source found that measures the
   CONSEQUENCE channel specifically**, and it postdates every judge-bias survey
   in the snippet table. It is the direct basis criterion 7 cites.
3. **arXiv 2604.16790 (Apr 2026, R2) is the most recent SE-specific judge audit**
   and it explicitly does NOT cover attempt number / iteration index -- so the
   extension from "consequence framing" to "round index" is currently
   UNMEASURED in the literature (see Pitfall P3).
4. **The requirements-smell field moved 2024-2025** (S29/S31 multi-label DL
   classifier, S34 testability ranking, R8's 12-category taxonomy with 41 tools)
   but **none of it carries a product-behaviour / evidence-apparatus axis**. That
   absence is itself a finding: 90.9's axis is house-specific.

No 2024-2026 source was found that CONTRADICTS the filing's core premise.

## Key findings (cited per claim)

**K1 -- The self-referential criterion is a genuine fixed point, and the
formal literature names it.** A criterion of the form "mutation-test every new
guard this step adds" quantifies over a set the step is simultaneously
producing. R3 states the general form: "quantification over that space
necessarily ranges over the representational environment in which the
quantification itself is encoded", yielding an instance with
`ψ_S <-> ¬C_S(ψ_S)` (Lemma 2.2, https://arxiv.org/html/2511.14665v1, accessed
2026-08-20). **Label this an ANALOGY, not a citation of the applied result:**
R3 is about total SAT classifiers and Gödel encoding, not requirement specs. It
supplies the *shape* of the argument; the applied evidence is R5.

**K2 -- The applied evidence that self-graded apparatus fails is R5, and it is
an impossibility bound, not an anecdote.** An agent editing both policy and its
own tests kept self-scores >=0.70 in **all 35** runs while **15 of 35** scored
below random; the named mode "*failure to retain* ... tests coevolve to validate
the degraded version" is the exact mechanism the filing describes as "cycle N's
fix is cycle N+1's finding". The bound `alpha + beta >= 1 - TV(P+,P-)` says no
endogenous-only gate holds both error rates low
(https://arxiv.org/html/2607.24300v1, accessed 2026-08-20). The four SEAL
conditions -- exogenous audit, audit confidentiality, single-bit feedback,
conservative update -- are the closest published design for what 90.9's
classifier must be.

**K3 -- Criterion 7's basis is real, correctly summarised in the filing, and
its recommended REMEDY is input-side.** R1: 58 of 72 cells lenient, p<0.001,
peak ΔV = -9.8pp, ERR_J = 0.000, bias "pre-CoT"; the authors say "interventions
must operate at the level of training or evaluation protocol design" rather than
CoT inspection (https://arxiv.org/html/2604.15224v1, accessed 2026-08-20).
Because ERR_J = 0.000, **a test that inspects the classifier's OUTPUT for signs
of history-awareness cannot work** -- criterion 7's "asserted by a test over its
inputs" is the only shape that can hold.

**K4 -- Anthropic's harness doctrine independently endorses acting at
specification time.** R7: the contract is negotiated "before any code was
written", which "prevented the generator from satisfying criteria through
convenient reinterpretation"; and "Each criterion had a hard threshold, and if
any one fell below it, the sprint failed"
(https://www.anthropic.com/engineering/harness-design-long-running-apps,
accessed 2026-08-20). 90.9 is the filing-time analogue of that clause.

**K5 -- Scoped beats vague, and this is measured.** R4 shows vague iteration
plateaus while targeted steering moves the intended dimension (32.4% -> 45.2%;
6.9% -> 40.5%), and that vague refinement bloats output (Claude's code "expanded
40x") while still passing tests (https://arxiv.org/html/2509.06770v1, accessed
2026-08-20). This supports the filing's conclusion that the mitigation is
scoped criteria rather than more turns.

**K6 -- The published requirements-smell taxonomy has no product/apparatus
axis.** R8's 12 categories (Ambiguity ... **Verifiability** ... Undefined) are
the closest fit, and Verifiability is about *whether* a requirement can be
verified, not *what* it verifies (https://arxiv.org/html/2404.11106v1, accessed
2026-08-20). 41 detection tools exist (Montgomery et al. 2023 via R8) and none
is reported to carry this axis. **The classifier's rule must therefore be
PRINTED and defended, exactly as criterion 1 demands, because there is no
external standard to appeal to.**

## Consensus vs debate (external)

**Consensus:** (a) evaluator must be separate from generator (R7, R5); (b)
presentation/context cues move judge verdicts by large margins (R1 -9.8pp; R2
up to +31.57%); (c) mitigation belongs on the INPUT side -- blinding, controlled
perturbation, hidden audit sets (R1, R2, R5) -- not on output inspection.

**Debate / unresolved:** R1's mitigation is "blind-evaluation protocols and
stakes-neutral fine-tuning", i.e. it leans on *training*, and it does NOT
explicitly endorse simply withholding the field. R5 does endorse withholding
(audit confidentiality). 90.9's criterion 7 follows R5's shape. R6 is a
counterweight in a narrower sense: it finds requirement smells had "negligible"
effect on one of two downstream tasks -- i.e. spec-quality interventions do not
always pay off measurably.

## Pitfalls (from literature and from the record)

**P1 -- A criterion-shape classifier is itself apparatus.** Under R5's bound, a
classifier that grades the shape of criteria and is itself specified by criteria
of that shape is endogenous. Criterion 2's control steps (hand-built all-product
control observed GREEN first; hand-built unbounded step exits non-zero) are the
exogenous fixture that breaks the loop -- they must be authored so they cannot be
regenerated by the same rule they test.

**P2 -- The proxy trap in criterion 1.** Finding I2: the v4 keyword rule
reproduces 44 exactly while the property test (v3) returns 0/155. Reproducing
the number is NOT evidence the rule is right. Criterion 1 should require the
classifier print, per flagged step, the QUANTIFIED NOUN CLASS and the ARTIFACT
CLASS the step produces, so a reader can see the self-reference rather than
trust a count.

**P3 -- The extrapolation in criterion 7 is not yet measured.** R1 tested
consequences *to the evaluated model* (retraining / decommissioning /
deployment). It did NOT test round index, verdict history, or remaining budget;
R2 explicitly does not cover attempt number either. The contract should state
this as an extension by argument (a shape classifier has no legitimate use for
history, so exclusion is free) rather than as a measured result.

**P4 -- The corpus pin (Finding I1).** Criterion 1 says "on the live tree", and
on the live tree the filing figures are already stale by 19 steps. Without a
`git show <rev>:` pin the criterion sends the implementer to "correct the rule",
which would corrupt a correct rule to chase a moved corpus.

**P5 -- Criterion 4's grep-shaped source check (Finding I3)** is evaded by the
house's own `Path.write_text` idiom (148 call sites; two write a
`masterplan.json` filename).

**P6 -- Criterion 6 asks the bound to name a finding it would have DEFERRED.**
That is the right shape (it forces the cost to be paid in public) but note the
denominator: 44 is a v4-proxy count, so "how many of the 44" must be quoted
with the rule that produced 44, or it inherits P2.

## Application to pyfinagent (external findings -> file:line anchors)

| Finding | Anchor | Implication for the contract |
|---|---|---|
| K2 / R5 SEAL | `.claude/masterplan.json` step 90.9 criterion 2 | The two hand-built control steps ARE the exogenous audit set. Keep them fixed and outside the classifier's own generation path. |
| K3 / R1 ERR_J=0.000 | criterion 7; `.claude/workflows/qa-verdict.js:430-435` | Test the INPUT surface. The live residual channel on the sibling rail is the prompt's `qa_wip.py --spawned-at` self-read, not a caller hand-off -- state the divergence. |
| K1 / R3 fixed point | criterion 3 ("exits non-zero SOLELY on unbounded scope") | The 155-step exit-code sweep is the discriminator; run it at the PINNED corpus so the sweep and the census agree. |
| P4 corpus pin | `scripts/qa/replay_changelog_rule_86_68.py:34`; `scripts/qa/sweep_absent_verification_paths.py:421` | House precedent for `git show <rev>:.claude/masterplan.json` and for an injectable `--masterplan` argument -- reuse both. |
| P5 write idiom | criterion 4; `scripts/qa/verify_decision_log_86_97.py:274,300`; `scripts/qa/prove_archive_provenance_86_29.py:92` | Resolve write-capable calls at AST level, not by the two literal patterns criterion 4 names. |
| K4 / R7 hard threshold | `docs/runbooks/per-step-protocol.md` §4 | 90.9 is the filing-time analogue of "negotiate the contract before any code" -- cite it as the doctrinal basis in contract.md. |
| K5 / R4 scoped feedback | step 90.9 `audit_basis` | The filing quotes 2509.06770 as if verbatim including "3-4 iterations". The full text confirms the CLAIM but **not that exact wording/number** -- downgrade to paraphrase in contract.md. |
| K6 / R8 taxonomy | criterion 1 ("a rule PRINTED beside its output") | No external standard exists for this axis; the printed rule is the whole warrant. Do not cite an RE taxonomy as authority for product-vs-apparatus. |

### Citation corrections carried out of this run

1. **`arXiv 2501.04810` was previously cited in this project as evidence that
   "keyword rules miss semantic smells" (syntactic BTA 0.98 vs semantic 0.83).**
   R6's full text shows those numbers measure **LLM traceability performance on
   requirements containing** each smell class, and the paper "does not address
   keyword/rule-based detector limitations" -- smelly requirements were manually
   curated. The number is real; the claim it was used to support is not what the
   paper measured. **Do not reuse that citation for detector accuracy.**
2. **`qa-verdict.js:336` does NOT pass `verdict_sequence`/`attempt_number` to
   the judge** (Finding I4). Prior researcher memory said it did.
3. **`arXiv 2509.06770`'s "3-4 iterations" is a paraphrase**, not a verbatim
   quote (see K5).

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8** (R1-R8), 7 preprints + 1 official doc
- [x] 10+ unique URLs total (incl. snippet-only) -- 8 read-in-full + 46 snippet-only recorded in-file
- [x] Recency scan (last 2 years) performed + reported -- four superseding findings, section above
- [x] Full papers / pages read (not abstracts) -- all arXiv reads via `/html/`; the one `/abs/` fetch (S46) is declared and excluded
- [x] file:line anchors for every internal claim -- Findings I1-I4 + inventory table

Soft checks:
- [x] Internal exploration covered every module in scope (masterplan, scripts/qa house patterns, qa.md, qa-verdict.js, phase-86.78)
- [x] Contradictions / consensus noted (R1 vs R5 on the remedy; R6 as counterweight)
- [x] All claims cited per-claim with URL + access date
- [~] Gap: `arXiv 2604.23178` ("Judging the Judges") and `arXiv 2510.02840`
  ("Take Goodhart Seriously") were identified as strong candidates and left
  snippet-only at moderate tier; neither is load-bearing for any claim above.

## Coverage (informational -- this step is NOT audit-class)

Rounds: 3 search/fetch rounds (Q1-Q3, Q4-Q6, targeted internal re-derivation).
New read-in-full findings in the last round: 2 (R8, and the corrected reading of
R6). `audit_class=false`, so `coverage.dry` is not required and is reported
false honestly rather than asserted true.
