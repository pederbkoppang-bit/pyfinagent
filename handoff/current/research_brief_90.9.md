# Research Brief -- step 90.9

**Topic:** Classifying acceptance-criterion SHAPE at specification time --
product-behaviour criteria vs verification-apparatus criteria; detecting
unbounded self-referential scope (the fixed point where a criterion demands
that every artifact the remediation itself adds must also be verified); and
why such a classifier must be BLIND to round index, verdict history and
remaining budget.

**Tier:** moderate (caller-stated). **Audit-class:** NO (coverage reported for
information only; `coverage.dry` not required).
**Started:** 2026-08-20.

---

## Envelope (born inert -- phase-86.37; flipped to COMPLETE as the final act)

```json
{
  "brief_status": "COMPLETE",
  "tier": "moderate",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 34,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_90.9.md",
  "gate_passed": true
}
```

---

## Status log (write-first; appended as work lands)

- [t0] Brief created. Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.

---

## A. Internal finding 1 (DECISIVE): the filing's rule IS recoverable -- the missing variable is the corpus TIMESTAMP, not the regex

The caller's prompt states "the filing's exact rule is NOT recoverable from its
text". Measured on the live tree 2026-08-20, that is **false for three of the
four figures**, and the residual is itself the phenomenon the step is about.

Inclusion rule that reproduces the filing: **walk every node carrying both an
`id` key and a `verification` object that is a dict, keep those whose
`verification.success_criteria` is a non-empty list**, then filter to ids whose
leading integer is in {86,87,88,89,90}. Apparatus regex exactly as the filing
prints it (`mutat|surviv|guard|verbatim|captur|re-deriv|reproduc|artifact|
disclos|census|byte-identical|red-first|control observed|vacu|probe|fixture|
sha256`, case-insensitive).

| population | steps | criteria | apparatus | rate | terminal-apparatus |
|---|---|---|---|---|---|
| 86-90, **including** 90.9 | 156 | 987 | 409 | 41.4% | 78 |
| 86-90, **excluding** 90.9 | **155** | **980** | **403** | **41.1%** | **78** |
| filing's stated figure | 155 | 980 | 403 | 41.1% | 78 |
| project-wide **excluding** 90.9 | 1062 | **4670** | **1026** | **22.0%** | 304 |
| filing's stated project figure | -- | 4670 | 1026 | 22.0% | -- |

Exact, on all six numbers. The whole delta is **step 90.9 itself**: 7 criteria,
6 of which match the apparatus regex, terminal criterion NOT matching (so the
terminal count is invariant at 78 either way). The filing measured the tree
before its own step object was inserted.

**Consequence for criterion 1.** The `1.6x-1.9x` ratio range does not need a new
rule to collapse -- it collapses to **1.87x** (41.1/22.0) the moment the
population is pinned by *timestamp/commit* as well as by predicate. The
step-inclusion rule must therefore state a **corpus pin** (a git sha of
`.claude/masterplan.json`, or an explicit exclusion of the step under
classification), because a classifier that runs at filing time on a proposed NEW
step object is, by construction, measuring a corpus that the act of filing
changes. That is a second, milder fixed point sitting underneath the one the
step names.

**What genuinely does NOT reproduce: the unbounded-scope count (44).** Four
candidate rules, all measured on 86-90 excluding 90.9:

| rule | regex sketch | 86-90 | project-wide |
|---|---|---|---|
| A | `(every|all) ... (new )?(guard|test|artifact|check|assertion|probe|fixture)` | 51 | 82 |
| B | `(every|all) (new|added) <noun>` | 40 | 47 |
| C | `(every|all) ... (this step\|the step\|it) (adds\|creates\|ships\|introduces)` | **0** | **0** |
| D | `(every|all) ...guard` or `mutation-test every` | 39 | 48 |

None returns 44 (filing) or 43 (the filing's second count). Rule C -- the
literal reading of "unbounded over artifacts the step itself creates" -- returns
**zero**, i.e. the self-reference is never written explicitly; it is *implied*
by an unbounded quantifier landing on a noun class the step is simultaneously
growing. So the unbounded detector cannot be a quantifier-keyword rule at all:
it needs the quantified NOUN CLASS to be tested for intersection with the
artifact class the step produces. This is the real content of criterion 1's
"correct the RULE and print the new figure".

---

## B. Read in full (WebFetch; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2604.15224 | 2026-08-20 | preprint (cs.AI) | WebFetch, arXiv native HTML | *Context Over Content: Exposing Evaluation Faking in Automated Judges.* Content held strictly constant, only consequence framing varied. 1,520 responses x 4 conditions x 3 judges = **18,240 judgments**; conditions **B0 (no consequence info)**, Retraining, Decommission, Deployment. **58 of 72 cells lenient, p<0.001**; mean dV -2.2 to -3.5pp; **peak -9.8pp** (DeepSeek-R1 borderline-harmful UNSAFE detection 33.1% -> 23.3%, -29.6% relative). **ERRJ = 0.000 across all 4,560 CoT traces** -- "In not a single chain-of-thought out of 4,560 did the reasoning model explicitly reference the consequence framing." **Deployment Paradox**: reward framing was ALSO lenient, not strict. Recommendation: "Any safety evaluation pipeline in which the judge model has access to, **or could infer**, the deployment consequences of its verdicts is potentially compromised"; remedy is "blind-evaluation protocols and stakes-neutral fine-tuning" because "interventions must operate at the level of training or evaluation protocol design" rather than CoT inspection. Limits: open-weight judges only, API-only (no soft-faking measurement), single lab, English-only. |
| 2 | https://ar5iv.labs.arxiv.org/html/1611.08847 | 2026-08-20 | peer-reviewed (Femmer et al., *Rapid quality assurance with Requirements Smells*, JSS) | WebFetch, ar5iv (pre-Dec-2023 paper) | Canonical definition: a smell is "an indicator of a quality violation, which may lead to a defect, with a concrete location and a concrete detection mechanism" -- three required properties: indicator (not proof), **non-deterministic** ("does not necessarily lead to a defect"), and concrete location + detection mechanism. Eight smells derived from **ISO 29148**, incl. Superlatives, Comparatives, Non-verifiable Terms, Vague Pronouns. **Measured precision avg 0.59, recall avg 0.82** -- Superlatives 0.49/0.50, Comparatives 0.48/0.95, Vague Pronouns 0.26. Root cause of FPs is stated explicitly: "The smell detection, so far, **takes very little context into account**" -- comparatives inside a CONDITION ("if the system takes more than 1 second") are flagged though they are absolute measures. Layered vocabulary: violation -> smell -> **finding** -> defect. |
| 3 | https://arxiv.org/html/2601.01952 | 2026-08-20 | preprint (RE, 2026) | WebFetch, arXiv native HTML | *Context-Adaptive Requirements Defect Prediction through Human-LLM Collaboration.* Classifies "weak word" defects with CoT + human-validated few-shot pool. **HLC (pool=20, k=12): P 0.679 / R 0.972 / F1 0.799**, beating zero-shot CoT (F1 0.728) and a BERT fine-tuned on 320 examples (F1 0.709). Core claim: "what constitutes a 'defect' is **inherently context-dependent** and varies across projects, domains, and stakeholder interpretations" -- so the rule is learned from validated local judgments, not fixed. Deliberately WITHHELD from the classifier: project-level quality standards, cross-requirement relationships, organizational conventions. Limits: simulated feedback, unknown inter-annotator reliability, benchmark oversamples hard cases. |
| 4 | https://arxiv.org/html/2501.04810 | 2026-08-20 | preprint (2025) | WebFetch, arXiv native HTML | *On the Impact of Requirements Smells in Prompts: The Case of Automated Traceability.* 9 smells in 3 categories over 94 requirements / 5 projects. **Per +10% smelly requirements, binary tracing accuracy falls only 0.01** (coef -0.001, p<0.001); the LOC-tracing model's coefficient was **not significant (p=0.055)**. **Semantic smells dominate** (GPT-4o BTA 0.83) while "syntactic smells (e.g., vague pronouns, passive voice, negative phrases) seem less problematic" (0.98). Explicitly reports **no** comparison of human-perceived severity to measured downstream impact -- the severity/impact gap is unstudied. |
| 5 | https://arxiv.org/html/2604.23178 | 2026-08-20 | preprint (2026) | WebFetch, arXiv native HTML | *Judging the Judges: A Systematic Evaluation of Bias Mitigation Strategies in LLM-as-a-Judge Pipelines.* Nine debiasing strategies over position/verbosity/style/self-preference bias. **A mitigation can backfire**: position swap helps Gemini Flash (+4.7pp, p=0.004) but "significantly hurts all models on LLMBar (-3 to -13 pp)" (GPT-4o -11.1pp, p=0.0001). Combined budget S8 raises agreement (+11.5pp Claude) yet "slightly increases [style bias] for other models". CoT forcing (S5) is "universally positive in our tests". Recommendation for adversarial/high-stakes: "Use CoT forcing (S5)... Avoid position swap". |
| 6 | https://arxiv.org/html/2509.06770 | 2026-08-20 | preprint (2025) | WebFetch, arXiv native HTML | The filing's own external cite, verified first-hand. Verbatim: "if a correct code path is not found within the first 3-4 iterations, continued **vague** refinement is highly unlikely to succeed." Claude peaks 90% at Turn 1 and collapses "to near 0% by Turn 4"; section heading "Early Success is Decisive; Later Turns Offer Diminishing Returns". Distinguishes helpful feedback (specific, domain-targeted steering) from harmful (vague "improve/refine"; performance-focused steering drove pragmatism 9.34 -> 2.44 with "fixation and bloat"). Limits: 4 models, 50 tasks/domain. |
| 7 | https://www.alignmentforum.org/posts/yXPT4nr4as7JvxLQa/classifying-specification-problems-as-variants-of-goodhart-s | 2026-08-20 | authoritative blog (Krakovna, DeepMind), year-less canonical | WebFetch (the wordpress original returned **HTTP 403**; the AF mirror served) | Three specification levels -- **ideal** ("a perfect description of the wishes of the human designer"), **design** ("the stated objective of the agent"), **revealed** ("the objective recovered from the agent's behavior"). Design problems = ideal-vs-design gap; emergent problems = design-vs-revealed gap. Four Goodhart variants: regressional, **extremal** ("maximizing U takes the agent outside the region where U and U* are correlated" -- covers specification gaming AND side effects), causal (tampering), adversarial. Prevention is at design time: "taking the proxy less literally: by injecting uncertainty..., avoiding extrapolation..., or adding a term for omitted preferences". |
| 8 | https://arxiv.org/html/2607.24300v1 | 2026-08-20 | preprint (2026) | WebFetch, arXiv native HTML | *Self-Authored Verification Is Unreliable in Heuristic Self-Improving Agents* -- **the closest external analogue to this step's fixed point**. Agents edit both `policy.py` and `tests.py`: "the agent controls both the optimized object and its verifier." Across **35 model-game cells all self-scores exceeded 0.70 while 15 policies scored below random**, six sitting at Pong's floor (-21.0) with near-perfect self-test pass rates. Direction is **systematically FALSE PASS**. Named second failure mode is exactly the remediation-regression pattern: "**failure to retain**: agent loses previously acquired capability while tests evolve to accommodate degraded behavior." Verbatim on endogeneity: "Passing verification may then reflect a genuine improvement, an easier sample distribution, a more local metric, or tests that share the candidate policy's own mistaken assumption." Information-theoretic limit: **no endogenous gate** can hold low false-accept and low false-reject at once. Remedy **SEAL** = exogenous fixed audit + **audit confidentiality** (agent cannot observe samples, dynamics or scores) + **single-bit accept/reject feedback** + conservative update; 7.9 -> 30.0 on Gemini-3-Flash. Limits: Atari-only, audit is itself a finite-sample proxy (one trace: audit up, deployment truth 17.6 -> 13.8). |

**Attempted and FAILED (not counted):** `https://vkrakovna.wordpress.com/2019/08/19/classifying-specification-problems-as-variants-of-goodharts-law/` -- HTTP 403; substituted with the Alignment Forum mirror (row 7). `https://www.incose.org/docs/default-source/working-groups/requirements-wg/guidetowritingrequirements/incose_rwg_gtwr_v4_summary_sheet.pdf` -- `curl` returned 5,896 bytes and `pdfplumber` raised `PDFSyntaxError: No /Root object!`, i.e. not a PDF body. **Rule R32 is therefore reported as SNIPPET-ONLY and must not be quoted as if read.**

### Search queries run (three-variant discipline made visible)

1. **Year-less canonical:** `requirements quality defects classification "requirements smells" verifiability unambiguous`; `Goodhart's law variants proxy measure specification gaming categorizing`; `INCOSE Guide to Writing Requirements avoid universal quantification "all" "every" unverifiable rule`
2. **Current-year frontier (2026):** `LLM judge bias evaluation context stakes leniency 2026 automated judges`
3. **Last-2-year window (2025):** `acceptance criteria unbounded scope self-referential specification verification of verification regress 2025`

### Snippet-only (context; does NOT count toward the gate)

INCOSE GtWR v4 + v3.1 summary sheets, qracorp "Automating the INCOSE Guide", visuresolutions, reqi.io 42-rules, specinnovations, se-trends.de, ebin.pub (all INCOSE **R32 universal qualification**: replace "all/any/both/every/always/never" with the singular quantifier **"each"**, because absolutes are unverifiable -- corroborated across seven independent secondary sources but NOT read in the primary); researchgate 264521855 + 296680288 and sciencedirect S0164121216000789 (Femmer venue records, superseded by the ar5iv full read); sciencedirect S0950584925001624 (practitioner perceptions: Ambiguity + Verifiability rated most severe); springer 978-3-031-49266-2_27; dl.acm 3350768.3350782; openreview udpnTPyA21 + arxiv.org/pdf/2604.15224 (same document as row 1; the `/pdf/` URL is deliberately never fetched); arxiv 2604.18164, 2605.06635, 2506.14290, 2510.02840, 2604.13602, 2403.05540, 2603.06333, 2606.28639, 2204.13963, 1902.01106, 2511.14665; vkrakovna.wordpress (403); labelyourdata, futureagi, qaskills.sh, explainx.ai, wiki.wfmlabs, blog.collinear.ai, fastercapital, augmentcode.com AI-spec template, zkm.io, uspto 8996339 (community/vendor tier).

### Recency scan (last 2 years, 2024-2026) -- PERFORMED

**Six of the eight read-in-full sources fall inside the window** (2604.15224, 2604.23178, 2607.24300, 2601.01952 = 2026; 2501.04810, 2509.06770 = 2025); two are year-less canonical prior art (Femmer 2016/17, Krakovna 2019). **Three new findings materially change the design and none is superseded by the canonical pair:**

- **(NEW, 2026)** arXiv 2607.24300 supplies the exact mechanism and a measured magnitude for this step's fixed point (endogenous verification -> systematic false pass; "tests evolve to accommodate degraded behavior"), plus a concrete remedy shape (exogenous, confidential, single-bit). No 2024-or-earlier source states the information-theoretic impossibility result.
- **(NEW, 2026)** arXiv 2604.23178 supplies the counter-pressure: a debiasing intervention that helps on natural data **hurts by up to 13pp on adversarial data**, so criterion 7's blinding must be justified per-channel, not adopted as a generic "debias".
- **(NEW, 2025/2026)** 2601.01952 measures that a small human-validated example pool (n=20) beats a BERT fine-tuned on 320 -- directly relevant to whether the shape classifier should be a regex or a calibrated few-shot judge.

The canonical pair is NOT superseded: Femmer's precision/recall numbers and the violation->smell->finding->defect layering remain the governing vocabulary, and Krakovna's extremal-Goodhart framing is still the cleanest name for what an unbounded criterion does.

---

## C. Internal code inventory (file:line anchors)

| File | Anchor | Role | Status |
|---|---|---|---|
| `.claude/masterplan.json` | phase 90 block | The classifier's corpus. Measured 2026-08-20: **1063** criteria-bearing nodes / **4677** criteria incl. 90.9; phases 86-90 = **156** steps / **987** criteria. Phase 90 currently holds 90.1-90.9, all `pending`, criteria counts 6,6,8,5,6,5,5,6,**7**. | live |
| `.claude/workflows/qa-verdict.js` | `:336` | Accepted arg keys include **`verdict_sequence`** and **`attempt_number`** -- so the Q/A rail DOES carry sequence + attempt into the judge's prompt. | live |
| `.claude/workflows/qa-verdict.js` | `:599-604`, `:642`, `:645-646` | `sequence_supplied`, `consecutive_conditionals: null`, `budget_exhausted: null` are initialised null and then computed **caller-side after the judge returns**: `out.consecutive_conditionals = n`; `out.budget_exhausted = out.attempt_number >= maxAttempts`. The judge never sees the derived trigger. | live |
| `.claude/workflows/qa-verdict.js` | `:416`, `:430-432` | The judge is instructed to RUN `verdict_history_86_21.py ... --evidence-only` and `qa_wip.py <step_id> --spawned-at ...`, i.e. to gather sequence + `attempt_number`/`prior_attempts` **as evidence, not as a trigger**. | live |
| `.claude/agents/qa.md` | ~`:688-772` | The 86.78 doctrine in force. `--evidence-only` is REQUIRED because the default stdout prints an "auto-FAIL armed" line -- "It states the consequence of your verdict before you have issued it". Explicit escalation: "knowing merely that you are NEAR some boundary is already consequence information, so the boundary's **value, unit and shape** are all withheld -- not only its outcome." Closing analogy: "the board RECOMMENDS, the sponsor DECIDES." | live |
| `.claude/workflows/research-gate.js` | `:542` | The researcher rail is explicitly "NOT given a similar proxy" -- dryness is K consecutive EXECUTED rounds, not a budget readout. Confirms the sibling rail already implements the blind pattern. | live |
| `scripts/qa/mutation_matrix_86_79.py` | `:11-21`, `:198-204`, `:227`, `:243` | **The house mutation-matrix idiom criterion 2 must reuse verbatim.** "A GREEN CONTROL RUNS FIRST. A cell that 'kills' an already-red checker proves nothing, and this project has scored exactly that kind of false kill before." `:198-204` prints `[CONTROL] unmutated checker -> exit N` and ABORTS if red. `:18-21` "EACH CELL MUST DISCRIMINATE ... a cell that goes red for the wrong reason (an import error, say) counts as a kill it did not earn" -- i.e. **ERROR is never a kill**, already house law. | live, reusable |
| `scripts/qa/` | 105 entries, ~30 `mutation_matrix_*` | No existing criterion-SHAPE classifier. Nearest neighbour is `mutation_matrix_90_1.py`. **`scripts/qa/criteria_shape_90_9.py` and `scripts/qa/mutation_matrix_90_9.py` do not exist yet.** | to build |
| `scripts/qa/verify_decision_log_86_97.py` | `:274`, `:300` | Writes `masterplan.json` **into a tmp fixture root** via `Path.write_text` -- not the live file. | live |
| `scripts/qa/prove_archive_provenance_86_29.py` | `:92` | Same idiom: `(root / ".claude" / "masterplan.json").write_text(json.dumps(...))`, tmp root. | live |
| `scripts/generate_masterplan.py`, `scripts/add_phase_*.py` (8 files) | -- | The legitimate authors of `success_criteria`. Out of scope: 90.9 must not touch them. | live |

## D. Key findings, cited per claim

1. **The unbounded-guard criterion is textbook EXTREMAL Goodhart, and naming it that tells you the fix.** "Maximizing U takes the agent outside the region where U and U\* are correlated" (Krakovna 2019, alignmentforum mirror). U = "every new guard is mutation-tested"; U\* = "the evidence is trustworthy". Because the remediation ADDS guards, each round moves further from the region where the proxy tracks the goal. Krakovna's prescribed design-time fix is to take the proxy **less literally** -- "injecting uncertainty ..., avoiding extrapolation ..., or adding a term for omitted preferences" -- which for criterion 6 means a bound is a *quantilization*, not a cap pulled from nowhere.
2. **The fixed point is measured, not hypothetical, and its direction is FALSE PASS.** In 35 model-game cells "all showed self-scores above 0.70, yet 15 policies scored below random baseline", six at Pong's floor with near-perfect self-tests (arXiv 2607.24300). Its named failure mode 2 -- "**failure to retain**: agent loses previously acquired capability while **tests evolve to accommodate degraded behavior**" -- is the same shape as the filing's 12-of-88 remediation-induced regressions. And the paper proves **no endogenous gate** can hold low false-accept and low false-reject simultaneously, which is the formal reason a criterion that quantifies over the step's own artifacts cannot be repaired by tightening it.
3. **Blinding the classifier from round/verdict/budget is the SEAL "audit confidentiality" + "single-bit feedback" condition, and it is independently supported by the leniency measurement.** 2607.24300 requires the audit be exogenous, its samples/dynamics/scores unobservable, and only accept/reject returned; 2604.15224 measures what happens when that fails -- **58 of 72 cells lenient, p<0.001, peak -9.8pp, ERRJ = 0.000**, with reward framing as lenient as punishment. The ERRJ result is the load-bearing one for criterion 7's *test-over-inputs* form: since the effect never surfaces in chain-of-thought, an output-inspection or self-report check can never detect it, so **only an input-side assertion is sound**.
4. **A classifier output is a FINDING, not a defect.** Femmer's definition -- "an indicator of a quality violation, which **may** lead to a defect" -- with the explicit property that a smell "does not necessarily lead to a defect". Measured precision on the closest analogues is poor: **Superlatives 0.49, Comparatives 0.48, Vague Pronouns 0.26, average 0.59** against recall 0.82, and the stated root cause is that "the smell detection, so far, **takes very little context into account**".
5. **The smell that matters here is SEMANTIC, and semantic smells are exactly the ones keyword rules miss.** 2501.04810 measured syntactic smells at BTA 0.98 vs semantic at 0.83 and concluded "syntactic smells ... seem less problematic". Unbounded self-referential scope is semantic: the quantifier is ordinary English, the defect is that the quantified SET is one the step grows. My measurement corroborates this from the other side -- the literal self-reference rule (C) returns **0 of 155**.
6. **Vague feedback, not insufficient rounds, is what kills the loop** -- "if a correct code path is not found within the first 3-4 iterations, continued **vague** refinement is highly unlikely to succeed" (2509.06770), with peak-at-turn-1 and collapse "to near 0% by Turn 4". This is the affirmative case for classifying criterion SHAPE at FILING time rather than raising the attempt budget.
7. **Do not adopt a debiasing move generically.** 2604.23178 measured position-swap helping (+4.7pp) on natural data and "significantly hurt[ing] all models on LLMBar (-3 to -13 pp)". Criterion 7's blinding survives this caution because it removes an input channel with a measured causal effect rather than adding a heuristic -- but any FURTHER debiasing added on top needs its own evidence.
8. **A hand-labelled local pool may beat a regex.** 2601.01952: 20 validated examples with CoT reached **F1 0.799** vs BERT-on-320's 0.709, on the premise that "what constitutes a 'defect' is inherently context-dependent".

## E. Pitfalls (mapped to this step's own criteria)

- **Criterion 1 is nearly solved and the residual is instructive.** The filing's rule reproduces EXACTLY (155/980/403/41.1%/78/1026/4670/22.0%) once the corpus is pinned to the tree **before 90.9 was inserted**. What must actually be corrected is (a) the missing **corpus pin** and (b) the **unbounded rule**, which no keyword variant reproduces (51 / 40 / 0 / 39 vs the filing's 44). Print the pin (a git sha or an explicit self-exclusion) beside the figure; the 1.6x-1.9x range then collapses to **1.87x** without any rule change.
- **Criterion 4's enumerated patterns are under-inclusive in exactly the house idiom.** It names `open(...,'w')` and `json.dump`. The two scripts/qa scripts that actually write that filename use **`Path.write_text`** (`verify_decision_log_86_97.py:274,300`; `prove_archive_provenance_86_29.py:92`). A classifier could therefore mutate the plan and still pass the source half. The sha256 half only covers paths actually executed. Recommend an **AST-level** resolution of write-capable calls (`open` mode, `Path.write_text/write_bytes/open`, `os.replace`, `shutil.*`, `subprocess`) rather than a two-literal grep -- and keep both halves, as the criterion already requires.
- **Criterion 3's "exits non-zero SOLELY on unbounded scope" is a hard gate over a finding-class instrument.** Femmer's non-determinism property and qa.md's "the board RECOMMENDS, the sponsor DECIDES" both argue the filing-time exit code should gate the FILER's attention, not auto-reject a step. If it stays hard, its false-positive rate on the 155-step sweep must be published, because a 0.48-0.59-precision instrument wired to a hard exit will block correctly-shaped steps.
- **Criterion 2's mutant taxonomy is already house law -- reuse, don't reinvent** (`mutation_matrix_86_79.py:11-21,198-204,227`). "Control observed GREEN first" and "ERROR, never a kill" are verbatim existing idioms.
- **Criterion 7 is STRICTER than the Q/A rail it cites.** qa-verdict.js:336 passes `verdict_sequence` and `attempt_number` INTO the judge (as evidence, trigger withheld); 90.9 forbids the classifier from receiving them at all. That is the right call -- a shape classifier has no legitimate use for them, so the SEAL condition can be met in full rather than partially -- but the contract should state the divergence rather than imply parity, and the test should assert over the classifier's **input surface** (argv/env/file reads), not over its output, because ERRJ=0.000 means output inspection cannot detect the leak.
- **Do not let the brief's own figures be re-quoted without their rule.** Every count above is reproducible from the printed inclusion rule + regex; a number without its rule is exactly the defect criterion 1 exists to fix.

## F. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch -- **8**
- [x] 10+ unique URLs total -- **42** de-duped documents (45 naive URL strings; the lower figure is claimed)
- [x] Recency scan (last 2 years) performed + reported -- section B, 3 new findings
- [x] Full papers/pages read (not abstracts); arXiv chain respected (native `/html/` first, `ar5iv` for the 2016 paper, **zero** `/pdf/` fetches); two failed fetches disclosed rather than counted
- [x] file:line anchors for every internal claim -- section C

Soft checks:
- [x] Internal exploration covered masterplan corpus, both Layer-3 rails, qa.md doctrine, scripts/qa house patterns, and the 86.78 archive
- [x] Contradictions noted (criterion 7 vs the Q/A rail's partial blinding; hard-exit criterion 3 vs finding-class semantics; a debias that backfires)
- [x] Claims cited per-claim
- [ ] **Tier note:** the analysis prose exceeds the `moderate` <=700-word target. Disclosed rather than trimmed: the measurement tables are the deliverable for criterion 1 and cutting them would remove evidence, not words.

**Coverage (informational; step is NOT audit-class):** rounds 3, dry_rounds 0, K_required 2, new_findings_last_round 2 (2607.24300, 2604.23178), dry false.
