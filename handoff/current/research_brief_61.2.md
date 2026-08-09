# Research Brief -- Step 61.2: Decision-Input Integrity (synthetic HOLD sentinel)

Tier: **complex** (caller-specified). Audit-class: **NO** (coverage reported
for information; `coverage.dry` not required). Written incrementally
(write-first). Session date: **2026-08-09**. All internal anchors RE-DERIVED
this session against HEAD; the step's 2026-06-11 anchors are stale.

---

# 1. HEADLINE (MEASURED) -- the step's stated live trigger is REFUTED

The masterplan asserts: *"THE LIVE TRIGGER IS THE CRITIC ... 'Critic returned
unparseable JSON after retry -- proceeding with the UNREVIEWED draft'"*.
**BigQuery refutes this.** Every synthetic row carries
`$.final_synthesis.error = "Failed to parse final report."`, and
`critic_degraded` is ORTHOGONAL -- `true` on rows with REAL scores, `false` on
the very 0.0 row the step cites.

| ticker | ts (UTC) | final_score | recommendation | `$.final_synthesis.error` | `$..critic_degraded` |
|---|---|---|---|---|---|
| CRWD | 2026-08-09 14:12:37 | 5.75 | `'Hold'` | (absent) | **true** |
| DELL | 2026-08-09 14:12:27 | 6.15 | `'Hold'` | (absent) | false |
| PANW | 2026-08-09 14:07:05 | **0.0** | `'HOLD'` | **Failed to parse final report.** | **false** |

Corroborated over 40 days (`financial_reports.analysis_results`, measured
2026-08-09): **185 rows total; 153 are `final_score=0.0` + `recommendation='HOLD'`
(82.7%), and all 153 carry `$.final_synthesis.error`** -- a 153/153 exact match.
`0.0` + `'Hold'` (mixed case) = **0 rows**, so the casing tell is a perfect
discriminator today, and it agrees exactly with the error-presence tell.
`critic_degraded=true` on **45** rows, of which only **3** carry a score > 0.

**Conclusion:** the firing path is the SYNTHESIS draft failing to parse, not the
critic. Emitter re-derived at `backend/agents/orchestrator.py:1681-1688`:
`_parse_json_with_fallback(draft_text, "Synthesis-Final")` returns falsy →
`logger.warning("Failed to parse final report, returning error.")` →
`return {"error": "Failed to parse final report.", "synthesis_iterations": n,
"critic_degraded": bool}`. A fix scoped to "abort instead of proceeding with the
UNREVIEWED draft" would **not close this step**: it would kill the CRWD path that
produced a genuine 5.75, and leave the PANW path open. The step has now
mis-attributed the trigger twice (first the timeout, now the critic).

**But the critic finding is not worthless -- it is a SECOND, distinct defect of
the same class.** 3 rows in 40 days were persisted with a real score while the
quality gate provably did not run, and `critic_degraded` lives only inside the
JSON blob; the `critic_review` STRING column is empty on 179/179 rows since
2026-07-01. An unreviewed report is currently indistinguishable, at the column
level, from a fully-reviewed one.

# 2. THE OTHER HEADLINE -- 61.2 is already BUILT, and DARK

Every criterion except the two manual-path save sites is implemented and
regression-tested; it is gated OFF. Measured live values (`get_settings()`,
2026-08-09):

```
paper_synthesis_integrity_enabled        = False   <-- criteria 1, 4, 6 gate
paper_position_recommendation_fix_enabled= False   <-- criterion 5 gate
claude_code_timeout_s                    = 150     <-- criterion 2 SHIPPED, ungated
```

`handoff/harness_log.md` Cycle 173 (2026-08-07) records **CONDITIONAL #2**, with
the sole blocker being operator promotion of `paper_synthesis_integrity_enabled`
(ask #10). Per CLAUDE.md's 3rd-CONDITIONAL rule, **a third Q/A on unchanged
blocker evidence MUST return FAIL**. `handoff/current/live_check_61.2.md`
already exists (5146 bytes, 2026-08-07), so the verification command's
`test -f` leg passes today.

Flipping the flag WOULD have prevented the PANW row: the guard at
`autonomous_loop.py:2049-2057` fires on `synthesis.get("error")` and raises
`SynthesisDegradedError`, routing to the lite fallback.

---

# 3. Read in full (>=5 required; counts toward the gate) -- 9 sources

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|-------------|---------------------|
| 1 | https://sre.google/workbook/alerting-on-slos/ | 2026-08-09 | Official doc (Google SRE Workbook ch.5) | WebFetch (full) | Multiwindow multi-burn-rate is the recommended pattern: *"you can send a page-level alert when you exceed the 14.4x burn rate over both the previous one hour and the previous five minutes"*; short window ≈ **1/12 the long window**; both windows must independently exceed threshold. Alert-fatigue proof by contradiction: *"you could receive up to 144 alerts per day every day, not act upon any alerts, and still meet the SLO."* Also: *"To avoid multiple alerts from firing if all conditions are true, you need to implement alert suppression."* |
| 2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-08-09 | Official doc (Anthropic) | WebFetch (full) | Constrained decoding gives *"Always valid: No more `JSON.parse()` errors"* / *"Type safe: Guaranteed field types and required fields"* / *"Reliable: No retries needed for schema violations"*. Two INDEPENDENT mechanisms: `output_config.format` (JSON outputs) vs `strict: true` (tool schema); *"You can use these features independently or together in the same request."* Notably the page gives **no** guidance on truncation / refusal / unparseable handling -- that contract is the caller's to define. |
| 3 | https://ar5iv.labs.arxiv.org/html/2107.11277 | 2026-08-09 | Peer-reviewed survey (Hendrickx et al., *ML with a reject option*) | WebFetch (ar5iv, full) | Model is a **pair** `(h, r)`: *"m(x) = ® if the prediction is rejected; h(x) if the prediction is accepted."* The reject symbol is a THIRD outcome, not a value of `h`. Taxonomy: ambiguity vs novelty rejection. Cost ordering is explicit: **`Cc < Cr < Ce`** (correct < reject < error) -- rejecting must be cheaper than being wrong, which is precisely inverted by a synthetic 0.0. Architectures: separated / dependent / integrated rejector. |
| 4 | https://tianpan.co/blog/2026-04-09-structured-output-failures-production-llm | 2026-08-09 | Practitioner (2026-04-09) | WebFetch (full) | Four failure layers: syntax / schema / **semantic** / distribution. *"JSON mode does nothing"* for schema compliance. Compounding: *"A single tool call with 97% schema compliance ... In an agent loop with 10 tool calls, the probability of completing without a single validation failure is 0.97^10 ≈ 74%. With 20 steps, it drops to 54%."* Retry policy: append the serialized validator error to the next request, **one retry maximum**. **[ADVERSARIAL to this step's likely fix]** it endorses *"default-value escalation"*: *"return a typed default value and route the request to a monitoring queue rather than crashing. An agent that returns `{"action": "unknown", "confidence": 0.0}` is more useful than one that throws a runtime exception."* -- note the default is `"unknown"`, an OUT-OF-BAND token, not a valid in-band verdict. |
| 5 | https://arxiv.org/html/2501.10868v3 | 2026-08-09 | Preprint / benchmark (JSONSchemaBench, 10k schemas) | WebFetch (arXiv HTML, full) | Constrained decoding is NOT a guarantee in practice. Empirical coverage on *GitHub Easy*: Guidance 86%, LM-only 65%, OpenAI 29%, Gemini 7%. *GitHub Hard*: Guidance 41%, LM-only 13%, OpenAI 9%. Distinguishes **declared / empirical / true coverage**; XGrammar showed *"38 categories with under-constrained failures ... it allows JSON instances that are invalid according to a given JSON Schema."* Quality is not hurt: *"Constrained decoding, regardless of the framework, achieves higher performance than the unconstrained setting."* |
| 6 | https://sre.google/sre-book/monitoring-distributed-systems/ | 2026-08-09 | Official doc (Google SRE Book ch.6) | WebFetch (full) | *"what's broken, and why?"* -- symptom vs cause. Paging doctrine: *"Every page should be actionable"*, *"Pages should be about a novel problem or an event that hasn't been seen before"*, *"Every time the pager goes off, I should be able to react with a sense of urgency."* Four golden signals incl. **Errors** = *"The rate of requests that fail, either explicitly ..., implicitly, or by policy."* A fabricated 0.0 is an *implicit* failure -- invisible unless made explicit. |
| 7 | https://arxiv.org/html/2607.04430v1 | 2026-08-09 | Preprint 2026 (CIC, uncertainty-aware abstention) | WebFetch (arXiv HTML, full) | Abstention as a **calibrated, certified, binary** decision: *"The system returns the answer only if Utest ≤ t̂; otherwise, it abstains."* Theorem 3.3: *"with probability at least 1−δ, every non-null threshold returned by the algorithm satisfies the desired selection-conditioned risk guarantee."* Measured FDR 0.092±0.007 at α=0.10 (CommonsenseQA/Qwen2.5-7B). **Gap found:** the paper never discusses downstream conflation of abstention with a low-confidence emitted prediction -- exactly 61.2's failure. |
| 8 | https://scikit-learn.org/stable/modules/preprocessing.html | 2026-08-09 | Official doc (scikit-learn) | WebFetch (full) | Rank/quantile transform = `G^{-1}(F(X))`. Failure mode stated verbatim: *"By performing a rank transformation, a quantile transform smooths out unusual distributions and is less influenced by outliers than scaling methods. It does, however, **distort correlations and distances within and across features**."* Boundary handling: *"The normal output is clipped so that the input's minimum and maximum -- corresponding to the 1e-7 and 1 - 1e-7 quantiles respectively -- do not become infinite."* The docs give **no** small-sample guidance -- that caveat must be sourced/derived locally, not cited to sklearn. |
| 9 | https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf | 2026-08-09 | Peer-reviewed (Sculley et al., NeurIPS 2015, *Hidden Technical Debt in ML Systems*) | curl + **pdfplumber** (32,812 chars extracted) | Two directly-applicable debts. **Undeclared Consumers / visibility debt:** *"a prediction from a machine learning model is made widely accessible ... some of these consumers may be undeclared, silently using the output of a given model as an input to another system ... expensive at best and dangerous at worst."* **Plain-Old-Data Type Smell:** *"The rich information used and produced by ML systems is all too often encoded with plain data types like raw floats and integers ... a prediction should know various pieces of information about the model that produced it and how it should be consumed."* This is the canonical statement of exactly why a bare `final_score FLOAT` cannot carry "I failed". |

## Search-query variants run (3-variant discipline, visible)

- **Current-year frontier (2026):** "LLM structured output JSON schema adherence failure constrained decoding 2026 production pipeline abort vs repair"
- **Last-2-year (2025-2026):** "LLM abstention 'I don't know' calibration 2025 2026 selective prediction downstream consumers"
- **Year-less canonical:** "selective prediction reject option abstention machine learning production systems"; "sentinel value anti-pattern null object misuse silent default corrupts downstream decisions data pipeline"

# 4. Identified but snippet-only (context; does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://arxiv.org/abs/2508.07556 | Thesis (Rabanser, selective prediction) | Abs page fetched only; **no arXiv HTML** and no ar5iv (post-snapshot). Per research-gate rules an abstract is NOT a full read. Abstract quote retained: models that *"know when to say 'I do not know'"*. |
| https://arxiv.org/pdf/2508.07556 | PDF | PDF-only; superseded by source 9 for the same argument |
| https://www.researchgate.net/publication/379410216_Machine_learning_with_a_reject_option_a_survey | Mirror | Duplicate of source 3 |
| https://arxiv.org/pdf/2606.23448 | Preprint (selective TS forecasting) | Time-series-specific; tangential |
| https://arxiv.org/pdf/2509.10348 | Preprint (chest X-ray rejection) | Clinical cross-domain; budget |
| https://arxiv.org/pdf/2606.00251 | Preprint (capability self-assessment) | Budget |
| https://arxiv.org/pdf/2503.18826 | Preprint (fair abstaining classifiers) | Fairness angle out of scope |
| https://arxiv.org/pdf/2311.09145 | Preprint (selective regression) | Regression variant; source 3 covers taxonomy |
| https://www.emergentmind.com/topics/selective-abstention | Aggregator | Secondary |
| https://arxiv.org/pdf/2505.18622 | Preprint (CWSA confidence-aware eval) | Budget |
| https://www.geeksforgeeks.org/machine-learning/the-reject-option-pattern-recognition-and-machine-learning/ | Community | Tier-5 |
| https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk | Community | Tier-5 |
| https://letsdatascience.com/blog/structured-outputs-making-llms-return-reliable-json | Blog | Tier-3/5 |
| https://arxiv.org/pdf/2505.04016 | Preprint (SLOT) | Budget |
| https://arxiv.org/pdf/2601.17717 | Preprint (trustworthiness of LLM-generated data survey) | Budget |
| https://arxiv.org/pdf/2603.03305 | Preprint (hidden cost of structured generation) | Budget |
| https://arxiv.org/pdf/2408.02442 | Preprint (*Let Me Speak Freely?*) | Relevant counter-evidence on format restriction cost; snippet retained |
| https://arxiv.org/pdf/2604.14862 | Preprint (schema key wording) | Budget |
| https://arxiv.org/pdf/2606.01926 | Preprint (bias in constrained decoding) | Budget |
| https://devtoollab.com/blog/llm-structured-outputs-guide-2026 | Blog | Tier-3 |
| https://towardsdatascience.com/llm-fallbacks-break-agent-pipelines-i-built-the-missing-recovery-layer/ | Blog | Tier-3 |
| https://medium.com/@adnanmasood/a-field-guide-to-llm-failure-modes-5ffaeeb08e80 | Blog | Tier-3 |
| https://arxiv.org/pdf/2510.25890 | Preprint (ATLAS layered constraints) | Budget |
| https://arxiv.org/pdf/2606.26590 | Preprint (TerraProbe deceptive fixes) | Tangential |
| https://python-patterns.guide/python/sentinel-object/ | Practitioner (Brandon Rhodes) | Python-specific; source 9 is stronger |
| https://dev.to/kalio/cache-aside-and-the-null-sentinel-pattern-5gjc | Community | Tier-5 |
| https://app.studyraid.com/en/read/15149/524746/pitfalls-of-null-pointers-and-sentinel-values | Community | Tier-5 |
| https://stevenstuartm.com/blog/2025/12/18/making-invalid-states-unrepresentable-the-billion-dollar-mistake-that-wasnt.html | Blog | **WebFetch attempted -> HTTP 404** |
| https://dev.to/stevenstuartm/making-invalid-states-unrepresentable-the-billion-dollar-mistake-that-wasnt-2b6m | Blog (mirror) | Tier-5; snippet quote retained: *"Every downstream system that trusts that data inherits the lie."* |
| https://www.emergentmind.com/topics/conformal-abstention | Aggregator | Secondary |
| https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.175682660.02761872/v1 | Preprint (*Learning to Say "I Don't Know"*, 2025-09) | Vision paper; source 7 is stronger |
| https://arxiv.org/pdf/2605.25133 | Preprint (prover-verifier selective prediction) | Budget |
| https://arxiv.org/html/2604.03904v1 | Preprint (I-CALM) | Budget |
| https://arxiv.org/pdf/2604.19444 | Preprint (unsupervised confidence calibration) | Budget |
| https://image-ppubs.uspto.gov/.../12536388 | Patent (self-evaluation for selective prediction) | Patent text; budget |
| https://dev.to/hadi_askari_.../my-llm-keeps-failing-in-production-...-313e | Community | Tier-5 |
| https://blog.gopenai.com/why-your-llm-keeps-breaking-production-and-how-to-fix-it-9cf25d428da8 | Community | Tier-5 |

**URLs collected: 46 unique** (9 read in full + 37 snippet-only).

# 5. Recency scan (2024-2026) -- PERFORMED

Searched the 2024-2026 window explicitly (three query variants above).
**Result: 4 new findings that COMPLEMENT, and 1 that QUALIFIES, the canonical
sources.**

1. **Constrained decoding is GA but not a guarantee** (JSONSchemaBench, 2025;
   Anthropic docs, GA Nov 2025). Empirical coverage as low as 7-41% on hard
   schemas; under-constrained grammars still emit schema-invalid JSON. So
   "just turn on structured outputs" does NOT retire the parse-failure branch
   -- it shrinks it. This supersedes any pre-2024 assumption that schema
   enforcement is either absent or total.
2. **Multi-step compounding is quantified** (tianpan 2026-04): 0.97^10 ≈ 74%,
   0.97^20 ≈ 54%. pyfinagent's 28-agent pipeline sits squarely in that regime;
   a per-call 3% failure rate is a per-report ~50% failure rate.
3. **Certified abstention now has finite-sample guarantees** (CIC, 2026):
   abstention thresholds can carry a provable selection-conditioned risk bound
   rather than a hand-tuned confidence cutoff. Newer than, and stronger than,
   the 2021 reject-option survey's cost-based framing.
4. **Abstention-as-architecture** (TechRxiv 2025-09, I-CALM 2026): the field is
   moving abstention from a post-hoc wrapper into training/inference itself.
   Not actionable for pyfinagent (no model training), recorded for completeness.
5. **QUALIFIER / adversarial:** *Let Me Speak Freely?* (arXiv 2408.02442, 2024)
   reports format restriction degrading LLM reasoning; JSONSchemaBench (2025)
   reports the opposite (*"achieves higher performance than the unconstrained
   setting"*, ~3% gain). The literature is NOT settled on whether tightening
   the synthesis schema costs analytical quality. Do not assert a quality gain
   from schema tightening in the contract.

# 6. Key findings (per-claim cited)

**Q1 -- the sentinel / null-object anti-pattern.**
1. A prediction encoded as a bare float cannot carry its own provenance --
   *"a prediction should know various pieces of information about the model that
   produced it and how it should be consumed"* (Sculley et al. 2015,
   proceedings.neurips.cc/.../86df7dcfd896fcaf2674f757a2463eba-Paper.pdf,
   accessed 2026-08-09). This is the Plain-Old-Data Type Smell, and it is the
   exact mechanism of 61.2.
2. The damage is proportional to the number of **undeclared consumers**:
   *"some of these consumers may be undeclared, silently using the output of a
   given model as an input to another system ... expensive at best and dangerous
   at worst"* (ibid.). pyfinagent has at least 4 (see §8).
3. Rejection must be a THIRD outcome, not a value of the score domain:
   *"m(x) = ® if the prediction is rejected; h(x) if the prediction is
   accepted"* (Hendrickx et al., ar5iv.labs.arxiv.org/html/2107.11277).
4. The cost ordering `Cc < Cr < Ce` (ibid.) is the formal statement of "worse
   than persisting nothing": a synthetic 0.0 pays `Ce` (an error) while the
   system believes it paid `Cr` (a rejection).

**Q2 -- abstention vs a genuine extreme score.**
5. Abstention is binary, calibrated, and OUT-OF-BAND: *"The system returns the
   answer only if `Utest ≤ t̂`; otherwise, it abstains"*
   (arxiv.org/html/2607.04430v1). The abstain signal is never expressed in the
   answer's own value space.
6. Enforcement on consumers is not addressed by the abstention literature --
   both source 3 and source 7 stop at "defer to a human". **This is a genuine
   literature gap**, so the enforcement mechanism must be engineered, not
   cited. The available mechanisms, in decreasing strength: (a) a *typed*
   `Optional`/sum type that will not compile/execute if unhandled; (b) a
   separate NOT-NULL status column that ranking SQL must join on; (c) NULL,
   which propagates but is silently droppable by `or 0.0`; (d) NaN, which
   poisons comparisons loudly but is coerced by `float(x or 0)`; (e) a quality
   flag that code *may* read -- the weakest, and the one that decays.
7. Google SRE's Errors golden signal explicitly includes failures *"implicitly,
   or by policy"* (sre.google/sre-book/monitoring-distributed-systems/) -- a
   fabricated 0.0 is an implicit error and is invisible to symptom monitoring
   until made explicit.

**Q3 -- structured-output robustness / "proceed with the UNREVIEWED draft".**
8. Constrained decoding removes Layer-1 (syntax) and much of Layer-2 (schema)
   but nothing else: *"The model returns schema-valid JSON where the values are
   internally inconsistent or factually wrong ... no current API catches this"*
   (tianpan.co, 2026-04-09).
9. Anthropic's own guarantee is scoped to shape only -- *"Always valid: No more
   `JSON.parse()` errors"* / *"Type safe"* -- and the doc is **silent** on
   truncation, refusal, and unparseable handling
   (platform.claude.com/docs/en/build-with-claude/structured-outputs).
10. Mature practice is **one** repair retry with the validator error fed back,
    then escalate -- not open-ended repair (tianpan.co).
11. **Is "proceed with the UNREVIEWED draft" defensible?** Yes, under exactly
    one explicit contract, and the literature supports it: the fallback must be
    an *out-of-band token* (`{"action": "unknown", "confidence": 0.0}`) plus
    routing *"to a monitoring queue"* (tianpan.co). It is defensible when
    (i) the unreviewed artefact is marked in-band at the consumer's level of
    access -- not buried in a JSON blob; (ii) the marker is on the same row a
    ranking query reads; (iii) an alert is raised; (iv) the consumer's default
    on an unreviewed artefact is *exclusion*, not *neutral inclusion*. It is
    NOT defensible when the marker is invisible to the consumer -- which is
    today's state (`critic_review` empty on 179/179 rows).

**Q4 -- rank normalisation of a fallback score into 1-10.**
12. The standard transform is `G^{-1}(F(X))`; sklearn's stated cost is that it
    *"distort[s] correlations and distances within and across features"*
    (scikit-learn.org/stable/modules/preprocessing.html). For conviction
    ranking this is acceptable (only the ORDER is consumed) but it means the
    fallback conviction is **not comparable in level** to an LLM conviction --
    only in rank.
13. Small-sample failure modes (derived, not cited to sklearn, which is silent):
    (a) with `n <= 1` the percentile is undefined -- the repo already returns a
    mid-scale 5 (`meta_scorer.py:157-159`); (b) with small `n` the transform is
    **scale-destroying** -- 3 uniformly terrible candidates still get 1/5/10, so
    a rank-normalised fallback can manufacture a top-conviction pick out of a
    uniformly bad slate; (c) ties need an explicit convention -- the repo uses
    midpoint ranks (`meta_scorer.py:163-165`); (d) mixing rank-normalised tail
    convictions with LLM head convictions in one sorted list (which the repo
    does at `meta_scorer.py:293-307`) compares two different scales.
    **Mitigation from the literature:** the reject-option cost ordering implies
    the safe fallback is to shrink COVERAGE (drop the tail) rather than
    manufacture a comparable score.

**Q5 -- alerting after N consecutive all-fallback cycles without fatigue.**
14. The canonical pattern is **multiwindow, multi-burn-rate**: require BOTH a
    long and a short window to breach before paging, with short ≈ long/12
    (sre.google/workbook/alerting-on-slos/). Mapped to cycles: "≥N consecutive
    all-fallback cycles" IS the long window; the short window is "the CURRENT
    cycle is still all-fallback", which is what fixes reset time -- the alert
    stops as soon as one healthy cycle lands.
15. Fatigue arithmetic: *"you could receive up to 144 alerts per day ... and
    still meet the SLO"* (ibid.) -- a per-cycle P1 on a persistent condition is
    the documented anti-pattern.
16. Suppression is mandatory when tiers overlap: *"To avoid multiple alerts from
    firing if all conditions are true, you need to implement alert
    suppression."* (ibid.) The repo currently fires the per-cycle P1 AND the
    streak P2 in the same block (`autonomous_loop.py:1064-1108`) with **no
    suppression** -- that is a live fatigue bug in the dark build.
17. *"Every page should be actionable"* / *"Pages should be about a novel
    problem"* (sre.google/sre-book/monitoring-distributed-systems/) -- the
    streak alert must therefore carry the root-cause hint and stop repeating.

# 7. Internal code inventory (all anchors RE-DERIVED 2026-08-09 against HEAD)

| File:line | Role | Status |
|---|---|---|
| `backend/agents/orchestrator.py:1681-1688` | **THE actual emitter.** `_parse_json_with_fallback(draft_text,"Synthesis-Final")` falsy → returns `{"error": "Failed to parse final report.", ...}` | LIVE; the 153/153 source of the fabricated rows |
| `backend/agents/orchestrator.py:1598-1631` | Critic unparseable → 1 retry at 2x `max_output_tokens` (capped 8192) → `critic_degraded=True; break` (the "UNREVIEWED draft" log at `:1624-1626`) | LIVE; phase-75.4. **Orthogonal to the 0.0 rows** (3 rows in 40d proceeded with real scores) |
| `backend/services/autonomous_loop.py:2049-2057` | phase-61.2 criterion-1 guard: `synthesis.get("error") or "scoring_matrix" not in synthesis` → `raise SynthesisDegradedError` | BUILT, **DARK** (`paper_synthesis_integrity_enabled=False`) |
| `backend/services/autonomous_loop.py:2058-2065` | **The fabrication site.** `rec = synthesis.get("recommendation", {})` then `rec.get("action", "HOLD")` -- the uppercase literal here is the ENTIRE casing tell | LIVE (legacy path) |
| `backend/services/autonomous_loop.py:2076-2078` | `synthesis.get("final_weighted_score", synthesis.get("final_score", 0))` -- the `0` default is the score half of the fabrication | LIVE (legacy path) |
| `backend/services/autonomous_loop.py:2122-2141` | Both-paths-failed → honest `_degraded` marker dict (`final_score: None`, `recommendation: None`) | BUILT, DARK |
| `backend/services/autonomous_loop.py:3199-3268` | `_persist_analysis`; NULL-passthrough at **:3247-3248** (`None if _degraded else float(... or 0.0)` / `None if _degraded else (... or "Hold")`); `summary` prefixed `"DEGRADED: "` at :3249-3252 | BUILT, DARK (depends on `_degraded` being set upstream) |
| `backend/services/autonomous_loop.py:2485-2496` | `_fold_degraded_for_trading` -- a `_degraded` analysis returns `None`, never reaching `decide_trades` | BUILT, DARK |
| `backend/services/autonomous_loop.py:2498-2528` | `_degraded_scoring_check` -- **already encodes the casing tell**: `conf_zero_upper = conf==0 and rec.isupper()`. Fires when all degraded or `>=3` | LIVE (alert-only, never gates a trade) |
| `backend/services/autonomous_loop.py:1064-1108` | Meta-scorer degraded P1 + phase-61.2 streak P2 at `>=2` consecutive cycles (`_bump_conviction_fallback_streak`) | P1 LIVE; streak P2 DARK. **No suppression between the two** |
| `backend/services/autonomous_loop.py:1139-1147` | criterion-6 `_rj_portfolio_ctx` built when EITHER `paper_risk_judge_reject_binding` OR the integrity flag is on | Partially live (binding flag was promoted per harness_log; integrity leg DARK) |
| `backend/services/meta_scorer.py:138-143` | `_fallback_conviction` -- the legacy `max(1,min(10,round(cs)))` clamp (the step's "~:138-142" anchor is CORRECT here) | LIVE when flag OFF |
| `backend/services/meta_scorer.py:145-166` | `_rank_normalized_convictions` -- criterion-4 percentile rank, midpoint ties, `n<=1 → 5` | BUILT, DARK |
| `backend/services/meta_scorer.py:168-177` | `_fallback_convictions` dispatcher (flag-gated) | BUILT, DARK |
| `backend/services/meta_scorer.py:293-307` | Tail convictions rank-normalised under the flag; **head (LLM) + tail (fallback) then co-sorted** | BUILT, DARK -- scale-mixing risk per finding 13(d) |
| `backend/services/meta_scorer.py:316-323` | `_fallback_all` -- emits `"fallback (LLM unavailable)"` | LIVE |
| `backend/agents/claude_code_client.py:587-601` | `recommended_step_timeout = 150` (class attr); `__init__(..., timeout_s: int = 150)`; instance `recommended_step_timeout = timeout_s + 30` | **criterion 2 SHIPPED + LIVE + ungated** |
| `backend/config/settings.py:186-192` | `claude_code_timeout_s: int = Field(150, ge=60, le=600)` -- configurable | LIVE, measured **150** |
| `backend/config/settings.py:204-207` | `paper_synthesis_integrity_enabled` | **False** (DARK) |
| `backend/config/settings.py:208-211` | `paper_position_recommendation_fix_enabled` | **False** (DARK) |
| `backend/services/portfolio_manager.py:63` | `_BUY_RECS = {"BUY","STRONG_BUY"}` (step said `:50` -- **stale**) | LIVE |
| `backend/services/portfolio_manager.py:140-141` | `rec = (analysis.get("recommendation") or "HOLD").upper()`; `old_rec = (pos.get("recommendation") or "").upper()` (step said `:114` -- **stale**) | LIVE. **NOTE the `or "HOLD"` default here re-fabricates a NULL recommendation into a HOLD** |
| `backend/services/portfolio_manager.py:154-157` | `if old_rec in _BUY_RECS and rec in _DOWNGRADE_RECS:` → `signal_downgrade` SELL (step said `:127` -- **stale**) | LIVE but structurally dead pre-fix |
| `backend/services/portfolio_manager.py:110-121` | Unsafe-combination WARN when the position-rec fix is ON while integrity is OFF | BUILT |
| `backend/services/paper_trader.py:447-457` | `_pos_rec` selection: analysis recommendation vs trade reason, flag-gated (step said `:305` -- **stale**) | BUILT, DARK |
| `backend/services/paper_trader.py:488, 512` | `"recommendation": _pos_rec` written into `paper_positions` (step said `:329` -- **stale**) | BUILT, DARK |
| `backend/tasks/analysis.py:210-214` | **UNGATED FABRICATION SITE #2**: `final_score=synthesis.get("final_weighted_score", 0)`, `recommendation=rec_obj.get("action","N/A")` | **LIVE, NOT FIXED** |
| `backend/api/analysis.py:210-214` | **UNGATED FABRICATION SITE #3**: byte-identical to the above | **LIVE, NOT FIXED** |
| `backend/tasks/analysis.py:380-407` | The *good* idiom already in-repo: `"final_score": None` on failure, docstring *"check `r["final_score"] is not None` to gate on a successful run"* | LIVE -- copy this shape |
| `backend/tests/test_phase_61_2_decision_integrity.py` | 495 lines, 8 test classes covering criteria 1-6 incl. flag-OFF byte-identity | PRESENT |
| `handoff/current/live_check_61.2.md` | 5146 bytes, 2026-08-07 | PRESENT (the `test -f` leg already passes) |

**Files inspected: 14** (`orchestrator.py`, `autonomous_loop.py`, `meta_scorer.py`,
`claude_code_client.py`, `settings.py`, `portfolio_manager.py`, `paper_trader.py`,
`tasks/analysis.py`, `api/analysis.py`, `bigquery_client.py`,
`test_phase_61_2_decision_integrity.py`, `live_check_61.2.md`,
`harness_log.md`, `.claude/masterplan.json`).

# 8. Undeclared consumers of `final_score` (the blast radius, measured by grep)

`backend/agents/conflict_detector.py:87,115`;
`backend/slack_bot/formatters.py:180`; `backend/slack_bot/scheduler.py:1069` --
all four read `final_weighted_score` with a **`, 0` default**, i.e. each one
independently re-creates the same fabrication. Plus
`portfolio_manager.py:140` (`or "HOLD"`) and `:182`. Any fix that only patches
the persist path leaves six `or 0` / `or "HOLD"` coercions intact.

# 9. BigQuery measurement (verbatim)

Table: `sunny-might-477607-p8.financial_reports.analysis_results` (dataset
location **us-central1**), 540 rows total. Column is `final_score FLOAT
NULLABLE` (confirmed -- not `overall_score`, not `score`).

**All 15 rows since 2026-08-08:**

```
ticker | ts_utc (UTC)          | final_score | recommendation | company_name                         | _path | syn_error                      | critic_degraded
CRWD   | 2026-08-09 14:12:37   | 5.75        | 'Hold'         | CROWDSTRIKE HOLDINGS, INC.           | full  |                                | true
DELL   | 2026-08-09 14:12:27   | 6.15        | 'Hold'         | Dell Technologies Inc.               | full  |                                | false
PANW   | 2026-08-09 14:07:05   | 0.0         | 'HOLD'         | PALO ALTO NETWORKS, INC              | full  | Failed to parse final report.  | false
NTAP   | 2026-08-09 13:08:24   | 0.0         | 'HOLD'         | NetApp, Inc.                         | full  | Failed to parse final report.  | true
HUM    | 2026-08-09 13:07:38   | 0.0         | 'HOLD'         | HUMANA INC                           | full  | Failed to parse final report.  | true
HPE    | 2026-08-09 13:07:38   | 0.0         | 'HOLD'         | HEWLETT PACKARD ENTERPRISE COMPANY   | full  | Failed to parse final report.  | true
CRWD   | 2026-08-09 13:06:58   | 0.0         | 'HOLD'         | CROWDSTRIKE HOLDINGS, INC.           | full  | Failed to parse final report.  | true
DELL   | 2026-08-09 13:06:54   | 0.0         | 'HOLD'         | Dell Technologies Inc.               | full  | Failed to parse final report.  | true
PANW   | 2026-08-09 13:06:19   | 0.0         | 'HOLD'         | PALO ALTO NETWORKS, INC              | full  | Failed to parse final report.  | true
NTAP   | 2026-08-08 21:03:29   | 0.0         | 'HOLD'         | NetApp, Inc.                         | full  | Failed to parse final report.  | true
HUM    | 2026-08-08 21:02:09   | 0.0         | 'HOLD'         | HUMANA INC                           | full  | Failed to parse final report.  | true
HPE    | 2026-08-08 21:02:06   | 0.0         | 'HOLD'         | HEWLETT PACKARD ENTERPRISE COMPANY   | full  | Failed to parse final report.  | true
CRWD   | 2026-08-08 21:01:24   | 0.0         | 'HOLD'         | CROWDSTRIKE HOLDINGS, INC.           | full  | Failed to parse final report.  | true
DELL   | 2026-08-08 21:01:22   | 0.0         | 'HOLD'         | Dell Technologies Inc.               | full  | Failed to parse final report.  | true
PANW   | 2026-08-08 21:01:20   | 0.0         | 'HOLD'         | PALO ALTO NETWORKS, INC              | full  | Failed to parse final report.  | true
```

`recommendation_confidence` and `synthesis_iterations` were **NULL on all 15**;
`total_cost_usd` was the constant fallback `0.1` on all 15;
`standard_model='claude-sonnet-4-6'` on all 15; `company_name` non-NULL on all 15
(criterion 3 confirmed live).

**40-day aggregate:** `total=185`; `final_score=0.0 AND recommendation='HOLD'` =
**153**; `final_score=0.0 AND recommendation='Hold'` = **0**;
`recommendation='N/A'` = **0**; `$.final_synthesis.error IS NOT NULL` = **153**;
`$.final_synthesis.critic_degraded='true'` = **45**, of which **3** have
`final_score > 0`.

**Which columns could carry an explicit degraded flag TODAY (measured over 179
rows since 2026-07-01):**

| Column | Type | Occupancy | Verdict |
|---|---|---|---|
| `data_quality_score` | FLOAT | **NULL on 179/179** | FREE -- natural home for a 0..1 integrity score |
| `critic_review` | STRING | **empty on 179/179** | FREE -- natural home for `"DEGRADED: critic did not run"` |
| `bias_flags` | STRING | **empty on 179/179** | FREE |
| `synthesis_iterations` | INTEGER | **NULL on 179/179** | FREE (autonomous path never passes it) |
| `groupthink_flag` | BOOLEAN | **NULL on 179/179** | FREE -- **the only BOOLEAN column in the table** |
| `recommendation_justification` | STRING | **empty on 179/179** | FREE |
| `recommendation_confidence` | FLOAT | NULL on 165/179 | mostly free but semantically loaded |
| `summary` | STRING | empty on 165/179 | already used by the dark path (`"DEGRADED: ..."` prefix) |
| `overall_reliability` | STRING | **non-NULL on 179/179** | occupied (value distribution not measured) |
| `risk_level` | STRING | **non-NULL on 179/179** | occupied |
| `full_report_json.$._degraded` | JSON | dark-path only | works but is a JSON-blob marker, i.e. weakest enforcement (finding 6e) |

There is **no dedicated degraded/status column**. The strongest available
options without a migration are `data_quality_score` (FLOAT, free) or
`groupthink_flag` (the only free BOOLEAN, but badly named for this). A new
NOT-NULL `analysis_status STRING` column is the option that matches finding 6(b);
note the phase-83.0 lesson that **BigQuery cannot add a REQUIRED column to an
existing table**, so it would be NULLABLE with a backfill + a reader convention.

# 10. Step 61.2 verification -- VERBATIM

**`verification.command`:**

```
cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k 'synthesis or persist or downgrade or meta_scorer or 61_2' -q && test -f handoff/current/live_check_61.2.md
```

Measured: `--collect-only` selects **72 of 3111** tests (3039 deselected) in
6.67s. `handoff/current/live_check_61.2.md` exists (5146 B, 2026-08-07), so the
`test -f` leg passes now.

**All 6 `success_criteria`, verbatim:**

1. "a synthesis result carrying final_synthesis.error (or missing scoring_matrix) is never persisted as a 0.0 final_score with a default HOLD: it is either routed to the existing lite fallback or persisted with NULL score plus an explicit degraded marker; a regression test simulates the timeout and asserts no 0.0/HOLD row is written and the same-cycle trade-decision input is not silently neutralized"
2. "claude_code synthesis/critic-class calls run with timeout >= 150s (per the file's own recommended_step_timeout) and the value is configurable"
3. "_persist_analysis falls back to the quant company_name when market_data.name is absent; live_check shows BQ rows from a post-fix autonomous full-path cycle with non-null company_name"
4. "the meta-scorer fallback no longer emits a constant saturated conviction: composite scores are rank/percentile-normalized into the 1-10 scale, and a WARN-level alert fires after 2 consecutive all-fallback cycles; the root cause of the 06-03..06-10 LLM unavailability is diagnosed and documented in experiment_results.md"
5. "positions persist the analysis recommendation (not the trade reason) so the signal_downgrade rule at portfolio_manager.py:127 can match; covered by a unit test"
6. "RiskJudge receives portfolio sector-breakdown context regardless of paper_risk_judge_reject_binding"

**`verification.live_check`, verbatim:** "live_check_61.2.md containing BQ rows
from at least one post-fix autonomous cycle: non-null company_name on full-path
rows, zero new rows with final_score=0.0 AND final_synthesis.error set, and
non-constant conviction values in paper_trades.signals"

Two notes for PLAN. (a) Criterion 5 names `portfolio_manager.py:127` but the
rule is now at **:154-157**; criteria are immutable, so the contract must record
the drift rather than amend it. (b) Criterion 1 says "simulates the timeout" --
the measured trigger is a synthesis PARSE failure, not a timeout; the criterion
is satisfiable either way because both set `final_synthesis.error`, but a
regression test that only injects a timeout would not cover the live path.

# 11. Where 61.2 ends and 80.14 / 80.26 / 80.30 begin

All four are the same defect class -- *an unavailable value rendered as a
confident one* -- separated by **which side of the API** they live on.

- **61.2 = the WRITE side (backend / data plane).** It owns the fabrication at
  the moment of persistence and the trade-decision consumption of that
  fabricated value. Scope boundary: everything up to and including the row in
  `financial_reports.analysis_results` and the `decide_trades` input. It is the
  only one of the four with **money** in the blast radius. Fixing 61.2 makes a
  degraded state *representable*.
- **80.26 (P2) = the READ side of THIS EXACT ROW.** It measured "SCORE shows
  0.00 for 13 of 16 rows on `/reports`", and its own criterion says: *"establish
  whether 0.00 means 'unscored' or a genuine zero score; rendering a real zero
  and a missing value identically is the underlying defect."* That is 61.2's
  output surfacing to the operator. **Dependency: 80.26 cannot be honestly
  closed before 61.2**, because until the write side distinguishes them, the UI
  has nothing to render differently. 80.26 additionally owns two things 61.2
  does NOT: the empty `30D TREND` column and the missing `COMPANY` on 2 rows.
- **80.14 (P2) = COLOUR semantics for zero-sample metrics**, on `/performance`
  and `/agents`. It is about rendering a metric computed over **zero samples**
  in a success colour. No overlap with `analysis_results`; disjoint from 61.2.
- **80.30 (P1) = FABRICATED FACTS ON FETCH FAILURE**, six frontend sites where a
  swallowed `catch` renders a specific false assertion (e.g. *"BQ
  INFORMATION_SCHEMA.JOBS reports zero billed jobs in range"* on a path where BQ
  was never queried). Same anti-pattern, **frontend-only, no BQ write**. It is
  61.2's mirror image: 61.2 fabricates on the way IN, 80.30 fabricates on the
  way OUT.

Recommended ordering: **61.2 → 80.26 → 80.30 → 80.14**. Nothing in 80.x should
be pulled into 61.2's scope; 61.2 should not touch `frontend/`.

# 12. Consensus vs debate (external)

**Consensus.** (a) Abstention must be out-of-band, never a value in the score's
own domain (sources 3, 7). (b) Rejecting must cost less than being wrong --
`Cc < Cr < Ce` (source 3). (c) Constrained decoding is the 2026 baseline but is
not a guarantee (sources 2, 5). (d) One repair retry, then escalate (source 4).
(e) Alert on persistence with a short-window confirmation, and suppress
overlapping tiers (sources 1, 6).

**Debate.** (a) *Does schema tightening cost reasoning quality?* JSONSchemaBench
says no (~+3%); *Let Me Speak Freely?* says yes. Unsettled -- do not claim a
quality gain. (b) *Abort vs degrade?* Source 4 explicitly prefers a typed
default plus a monitoring queue over throwing; the reject-option literature
prefers a formal abstain. These reconcile only if the "typed default" is an
out-of-band token (`"unknown"`), which is why today's `"HOLD"` fallback fails
BOTH schools -- `HOLD` is a valid in-band verdict. (c) *Where does enforcement
live?* No external source answers this; it must be engineered.

# 13. Pitfalls (from literature + measured)

1. **Fixing the critic instead of the synthesis parse.** Measured: 3 of 45
   `critic_degraded` rows carried real scores. Aborting there would destroy
   working output and miss all 153 fabricated rows.
2. **Patching only the persist path.** Six independent `or 0` / `or "HOLD"`
   coercions exist downstream (§8), including `portfolio_manager.py:140`, which
   would turn a NULL recommendation straight back into `"HOLD"`.
3. **Relying on the casing tell.** It is 153/153 accurate today but it is a
   formatting accident of `rec.get("action", "HOLD")` at
   `autonomous_loop.py:2065`; any prompt or model change breaks it silently.
4. **A JSON-blob marker as the enforcement mechanism.** `$._degraded` is
   invisible to a ranking `SELECT`. Sculley's undeclared-consumer argument
   applies directly.
5. **Alert stacking.** `autonomous_loop.py:1064-1108` fires a per-cycle P1 and a
   streak P2 with no suppression -- the documented fatigue anti-pattern
   (source 1).
6. **Rank-normalisation manufacturing conviction.** On a uniformly bad slate the
   percentile transform still emits a 10. Prefer shrinking coverage.
7. **Scale mixing.** `meta_scorer.py:293-307` co-sorts LLM head convictions with
   rank-normalised tail convictions.
8. **Assuming the flag flip is sufficient.** It is not: `tasks/analysis.py:213`
   and `api/analysis.py:213` fabricate unconditionally, outside the flag.
9. **A third CONDITIONAL.** Two CONDITIONALs are already logged for 61.2
   (Cycle 173). The next Q/A pass on unchanged blocker evidence MUST return
   FAIL per CLAUDE.md.

# 14. Application to pyfinagent (external findings → file:line)

- **Make abstention a third outcome, not a score value** (finding 3) →
  `autonomous_loop.py:2065` (`rec.get("action","HOLD")`) and `:2076-2078`
  (`, 0` default) are the two literals to remove. The correct in-repo idiom
  already exists at `tasks/analysis.py:380-407` (`"final_score": None` +
  the docstring gate `r["final_score"] is not None`).
- **Close the two ungated write sites** (audit_note gap, still open) →
  `tasks/analysis.py:210-214` and `api/analysis.py:210-214`. These are
  byte-identical and are the ONLY part of 61.2 not built.
- **Give the marker a column a ranking query must read** (finding 6b, Sculley
  undeclared consumers) → `data_quality_score` (FLOAT, NULL 179/179) or a new
  `analysis_status STRING`; do NOT rely on `$._degraded` alone.
- **Handle the unreviewed-but-scored case** (findings 8-11) →
  `orchestrator.py:1624-1631` sets `critic_degraded=True` but nothing writes it
  to a column; `critic_review` is empty on 179/179 rows and is FREE.
  Contract for "proceed with the UNREVIEWED draft": mark in-band at the column
  level + alert + consumer default = exclusion, not neutral inclusion.
- **Alerting** (findings 14-17) → keep the streak counter at
  `autonomous_loop.py:1064-1108` but add SUPPRESSION so the per-cycle P1 does
  not co-fire with the streak P2; the "current cycle still degraded" condition
  is the SRE short window that gives a fast reset.
- **Rank normalisation** (findings 12-13) → `meta_scorer.py:145-166` is already
  correct in mechanism; the open risks are scale-mixing at `:293-307` and
  manufactured conviction on a uniformly bad slate.
- **Criterion 2 is DONE** → `settings.py:186-192` (`claude_code_timeout_s=150`,
  `ge=60, le=600`) + `claude_code_client.py:587-601`; measured live at 150.
- **Criterion 3 is DONE** → `autonomous_loop.py:3238-3245`; company_name
  non-NULL on 15/15 rows measured.

# 15. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**9**; 8 via WebFetch, 1 via curl+pdfplumber per the research-gate PDF chain)
- [x] 10+ unique URLs total incl. snippet-only (**46**)
- [x] Recency scan (2024-2026) performed + reported (§5, 5 findings incl. 1 qualifier)
- [x] Full papers / pages read, not abstracts (arXiv 2508.07556 was abstract-only and is therefore listed as SNIPPET-ONLY, not counted)
- [x] file:line anchors for every internal claim (§7, all re-derived 2026-08-09)

Soft checks:
- [x] Internal exploration covered every module in the caller's scope (14 files) plus 4 undeclared consumers found by grep
- [x] Contradictions / consensus noted (§12 -- incl. the unsettled schema-quality debate)
- [x] Claims cited per-claim with URL + access date (§6)
- [ ] **Gap:** the `overall_reliability` and `risk_level` VALUE distributions were not measured (only non-NULL counts), so "occupied" is an inference from non-NULLness.
- [ ] **Gap:** the caller's hard constraints barred running the 72-test selection; only `--collect-only` was executed. Pass/fail status of those 72 tests is UNMEASURED by this brief.

## Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 37,
  "urls_collected": 46,
  "recency_scan_performed": true,
  "internal_files_inspected": 14,
  "coverage": {
    "audit_class": false,
    "rounds": 3,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 2,
    "dry": false
  },
  "brief_path": "handoff/current/research_brief_61.2.md",
  "gate_passed": true
}
```
