---
name: criterion-shape-90-9
description: Step 90.9 -- the filing figures reproduce EXACTLY but only at the FILING COMMIT (corpus moved +19 steps 49min later); the unbounded count 44 reproduces as a keyword PROXY while the property test returns 0/155; qa-verdict.js does NOT pass verdict_sequence to the judge
metadata:
  type: project
---

Step 90.9 (classify acceptance-criterion SHAPE at filing time). RE-MEASURED
2026-08-20 (run 2). Brief: handoff/current/research_brief_90.9.md.

**The figures reproduce EXACTLY -- at a PINNED corpus, not the live tree.**
Inclusion rule = walk nodes with an `id` AND a dict `verification`, keep
non-empty `success_criteria`, phases 86-90, EXCLUDE 90.9 itself. At
`git show 252090a3:.claude/masterplan.json` (the commit that filed 90.9, 20:55):
155 / 980 / 403 / 41.1% / 78 (50.3%) / 1026 of 4670 = 22.0% / ratio **1.87x** --
all six to the digit. At HEAD: 174 / 1045 / 414 / 39.6% / 81 / 21.9% / 1.81x.
The delta is commit `085c74e8` at 21:44 (+19 steps, 86.127-86.145) -- **49
minutes after filing**. So criterion 1's "reproduces ... on the live tree" is
already unsatisfiable, and its escape hatch ("correct the RULE") would corrupt a
correct rule to chase a moved corpus. House precedent for pinning:
`replay_changelog_rule_86_68.py:34` (`git show <ref>:`) and
`sweep_absent_verification_paths.py:421` (`--masterplan` arg).

**The unbounded count 44 DOES reproduce -- as a PROXY.** v4 =
`(every|all)` within 80 chars of `(guard|mutation cell|probe|fixture|artifact|
new test)` gives exactly 44 at BOTH pins; v1 (`every new guard`) = 39 and is a
strict subset. But v3, the only variant testing self-reference literally
(`(every|all) ... this step (adds|creates|introduces)`), returns **0 of 155** --
sampled hits (86.17/86.20/86.22/86.24/86.25/86.27) all read "MUTATION-TEST every
new guard, including reverting <this step's fix>": the self-reference rides on
the word `new` plus the surrounding sentence. Reproducing the NUMBER is not
evidence the RULE is right -- [[feedback_assert_the_property_not_a_proxy]].

**CORRECTION to run 1 / to this memory's prior version: qa-verdict.js does NOT
pass `verdict_sequence`/`attempt_number` into the judge.** `:335-337` is
`KNOWN_ARG_KEYS`, whose only job is computing `UNKNOWN_ARG_KEYS` for
silent-input-loss logging (`:330-334`). The judge prompt is `PROMPT` at `:340`
closing `:437`; `verdict_sequence` never appears in it (the sole `:339-799` hit
is `:565`, a design comment). `enforceEscalation` runs caller-side at `:824`
AFTER the verdict and is "returned ALONGSIDE the verdict, never merged into it"
(`:568-571`). **The channel that IS still open is a SELF-read:** `:430-435`,
inside the prompt, tells the judge to run `scripts/qa/qa_wip.py <step_id>
--spawned-at`, which reports `attempt_number`/`prior_attempts`. phase-86.78
(done) closed the CALLER-authored channel only. So criterion 7's two verbs --
"never given, and never reads" -- and the second one binds.

**Criterion 4's write-pattern list still misses the house idiom.** 148
`Path.write_text` sites in `scripts/qa/*.py`; both scripts writing a
`masterplan.json` filename use it (`verify_decision_log_86_97.py:274,300`,
`prove_archive_provenance_86_29.py:92`). Criterion 4 names only `open(...,'w')`
and `json.dump`. Needs AST-level resolution.

**CITATION CORRECTION -- arXiv 2501.04810.** Its BTA 0.98 (syntactic) vs 0.83
(semantic) measure **LLM traceability performance on requirements containing**
each smell class (Vogelsang et al., GPT-4o Table IV). The paper does NOT address
keyword/rule-based detector limits; smelly requirements were manually curated.
Do NOT reuse it as evidence that "keyword rules miss semantic smells".
Similarly arXiv 2509.06770's "3-4 iterations" is a paraphrase, not verbatim.

**Best external anchors:** arXiv 2607.24300 (self-authored verification: 35/35
self-score >=0.70, 15/35 below random, "tests coevolve to validate the degraded
version", bound `alpha+beta >= 1 - TV(P+,P-)` for ANY endogenous gate, remedy
SEAL = exogenous audit + confidentiality + single-bit feedback); arXiv
2604.15224 (58/72 lenient, p<0.001, -9.8pp, ERR_J=0.000, remedy is INPUT-side);
arXiv 2511.14665 (Solver's Paradox, `psi_S <-> not C_S(psi_S)` -- use as ANALOGY
only, it is about SAT classifiers); arXiv 2404.11106 (12-category smell taxonomy,
41 tools, **no product/apparatus axis exists in the RE literature**).
